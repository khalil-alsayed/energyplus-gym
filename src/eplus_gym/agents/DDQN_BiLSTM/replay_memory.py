# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:35:09 2024

@author: kalsayed
"""

import numpy as np

class ReplayBuffer(object):
    def __init__(self, 
                 sac_max_size,           # capacity for Instantly ring buffer (b)
                 transformer_max_size,   # capacity for Transformer ring buffer (B)
                 input_shape):
        
        # -----------------------------
        # 1) Instantly (single-step) Buffer Setup: b
        # -----------------------------
        self.sac_mem_size = sac_max_size
        self.sac_mem_cntr = 0

        # Each state is shape input_shape, e.g. (18,) if your environment has 18 features
        self.state_memory = np.zeros((self.sac_mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.sac_mem_size, *input_shape),
                                         dtype=np.float32)
        # Example action shape 1 => discrete
        self.action_memory = np.zeros(self.sac_mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.sac_mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.sac_mem_size, dtype=bool)

        # -----------------------------
        # 2) Transformer (sequence) Buffer Setup: B
        # -----------------------------
        # We'll store (S, A, S', r, done) in a separate ring buffer
        self.transformer_mem_size = transformer_max_size
        self.t_mem_cntr = 0
        
        # For storing 96 consecutive states, each with 18 features
        self.sequence_length = 96
        self.state_dim = 18
        
        # S shape = (96, 18)
        self.transformer_state_dim = (self.sequence_length, self.state_dim)
        # S' shape = (96, 18)
        self.transformer_next_state_dim = (self.sequence_length, self.state_dim)
        # A shape = (1,) (the action at the final step of S)
        self.transformer_action_dim = 1

        # Create buffers for sequence-based transitions
        self.transformer_state_memory = np.zeros(
            (self.transformer_mem_size, *self.transformer_state_dim),
            dtype=np.float32
        )
        self.transformer_next_state_memory = np.zeros(
            (self.transformer_mem_size, *self.transformer_next_state_dim),
            dtype=np.float32
        )
        self.transformer_action_memory = np.zeros(
            (self.transformer_mem_size, self.transformer_action_dim),
            dtype=np.float32
        )
        self.transformer_reward_memory = np.zeros(self.transformer_mem_size, dtype=np.float32)
        self.transformer_done_memory   = np.zeros(self.transformer_mem_size, dtype=bool)

        # If you ever want to filter or reorder features, you could do so here,
        # but for simplicity we'll just store all 18 features.
        self.relevant_indices = list(range(self.state_dim))  # [0..17]

    # ----------------------------------------------------
    # 3) Storing a Transition in Buffer b AND building (S, A, S', r, done) for B
    # ----------------------------------------------------
    def store_transition(self, state, action, reward, new_state, done):
        """
        1) Store (s, a, r, s', done) in the single-step "Instantly" buffer b.
        2) If we have >= 96 states in b, automatically build (S, A, S', r, done)
           and store in the Transformer buffer B.
        """
        # -----------------------------
        # 3.1) Store in Buffer b
        # -----------------------------
        sac_index = self.sac_mem_cntr % self.sac_mem_size

        self.state_memory[sac_index] = state
        self.new_state_memory[sac_index] = new_state
        self.action_memory[sac_index] = action
        self.reward_memory[sac_index] = reward
        self.terminal_memory[sac_index] = done

        self.sac_mem_cntr += 1

        # -----------------------------
        # 3.2) Build sequence (S, A, S', r, done) => Buffer B
        # -----------------------------
        if self.sac_mem_cntr >= self.sequence_length:
            # The "current" transition index in b is (sac_mem_cntr - 1)
            # This is the final step where we have a full 96-step window.
            current_idx = (self.sac_mem_cntr - 1) % self.sac_mem_size
            
            # -------------
            # Build S (96 consecutive states) 
            # -------------
            # S covers steps: current_idx - 95, ..., current_idx
            state_seq = []
            for offset in range(self.sequence_length):
                # e.g. offset=0 => idx = current_idx - 95
                # e.g. offset=95 => idx = current_idx
                idx = (current_idx - (self.sequence_length - 1 - offset)) % self.sac_mem_size
                full_state = self.state_memory[idx, self.relevant_indices]
                state_seq.append(full_state)
            state_seq = np.array(state_seq, dtype=np.float32)  # shape (96, 18)

            # -------------
            # Build A (the action at the final step)
            # -------------
            action_t = self.action_memory[current_idx].copy()

            # -------------
            # Build S' (the "next" 96 consecutive states)
            # -------------
            # This is the sequence from current_idx - 95 + 1, ..., current_idx + 1
            # i.e., each index is shifted by +1
            next_state_seq = []
            for offset in range(self.sequence_length):
                idx = (current_idx - (self.sequence_length - 1 - offset)) % self.sac_mem_size
                full_next_state = self.new_state_memory[idx, self.relevant_indices]
                next_state_seq.append(full_next_state)
            next_state_seq = np.array(next_state_seq, dtype=np.float32)  # shape (96, 18)

            # -------------
            # r, done from the "current" single-step transition
            # -------------
            r_t    = self.reward_memory[current_idx]
            done_t = done

            # -------------
            # Store into Transformer buffer B
            # -------------
            t_index = self.t_mem_cntr % self.transformer_mem_size

            self.transformer_state_memory[t_index]      = state_seq
            self.transformer_next_state_memory[t_index] = next_state_seq
            self.transformer_action_memory[t_index]     = action_t
            self.transformer_reward_memory[t_index]     = r_t
            self.transformer_done_memory[t_index]       = done_t

            self.t_mem_cntr += 1


    # ----------------------------------------------------
    # 4) Sampling from b (Instantly buffer)
    # ----------------------------------------------------
    def sample_buffer(self, batch_size):
        """
        Return a batch of (s, a, r, s', done) from the single-step buffer b.
        """
        max_mem = min(self.sac_mem_cntr, self.sac_mem_size)
        batch_indices = np.random.choice(max_mem, batch_size, replace=False)

        states      = self.state_memory[batch_indices]
        actions     = self.action_memory[batch_indices]
        rewards     = self.reward_memory[batch_indices]
        next_states = self.new_state_memory[batch_indices]
        dones       = self.terminal_memory[batch_indices]

        return states, actions, rewards, next_states, dones


    # ----------------------------------------------------
    # 5) Sampling from B (Transformer buffer)
    # ----------------------------------------------------
    def sample_transformer_buffer(self, batch_size):
        """
        Return a batch of (S, A, S', r, done) for Transformer-based training.

         - S: shape (batch_size, 96, 18)
         - A: shape (batch_size, 1)
         - S': shape (batch_size, 96, 18)
         - r: shape (batch_size,)
         - done: shape (batch_size,)
        """
        # How many sequence transitions do we actually have?
        max_t = min(self.t_mem_cntr, self.transformer_mem_size)
        if max_t == 0:
            return None, None, None, None, None  # no data yet

        batch_idx = np.random.choice(max_t, batch_size, replace=False)

        seq_states  = self.transformer_state_memory[batch_idx]      # (batch_size, 96, 18)
        seq_actions = self.transformer_action_memory[batch_idx]     # (batch_size, 1)
        seq_next    = self.transformer_next_state_memory[batch_idx] # (batch_size, 96, 18)
        seq_r       = self.transformer_reward_memory[batch_idx]     # (batch_size,)
        seq_done    = self.transformer_done_memory[batch_idx]       # (batch_size,)

        return seq_states, seq_actions, seq_next, seq_r, seq_done

