# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:32:39 2024

@author: kalsayed
"""

import numpy as np
import torch as T
from eplus_gym.agents.ddqn_mlp.deep_q_network import DeepQNetwork
from eplus_gym.agents.ddqn_mlp.replay_memory import ReplayBuffer
from eplus_gym.envs.energyplus import _find_project_root

from sklearn.preprocessing import MinMaxScaler
import pandas as pd



class DDQN_MLP(object):
    def __init__(self, gamma, epsilon, lr,input_dims, n_actions, 
                 mem_size, eps_min, batch_size, replace, eps_dec,
                 chkpt_dir, algo, env_name):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size=mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        
        

        #Linear decrementation--------------------------------------------------------
        #self.epsilon_decay_episodes = (epsilon - eps_min)/eps_dec
        #----------------------------------------------------------------------------------------
        #new decrementation (exponential)--------------------------------------------------------
        self.epsilon_decay_episodes=(self.eps_min / max(self.eps_min,self.epsilon)) ** (1 /self.eps_dec)
        #----------------------------------------------------------------------------------------


        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    name=self.env_name+'_'+self.algo+'_q_eval',input_dims=self.input_dims,
                                    chkpt_dir=self.chkpt_dir
                                    )

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    name=self.env_name+'_'+self.algo+'_q_next',input_dims=self.input_dims,
                                    chkpt_dir=self.chkpt_dir
                                    )
        
        
        #load all the general knowledge memory
        
        #self.q_eval_explore.transfer_learning_load()
        #self.q_next_explore.transfer_learning_load()
        
       
       
        
        # -------------
        # 3) Normalizer
        # -------------
        proj_root = _find_project_root()
        qtx_dir = (proj_root / "src" / "eplus_gym"  / "envs" / "assets" / "normalization" / "DDQN_MLP.csv")
        self.sample_data = pd.read_csv(qtx_dir).to_numpy()
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.sample_data)
        
                
    
    
    def choose_action_test(self, observation):
        if np.random.random() >= self.epsilon:
            observation_array = np.array(observation).reshape(1, -1)
            # Normalize the observation
            normalized_observation = self.scaler.transform(
                observation_array)
            state = T.tensor([normalized_observation],
                             dtype=T.float).to(self.q_eval.device)
            
            actions = self.q_eval.forward(state)
            
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    
    
   

  

    
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
        
    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)
        # Convert observation to a 2D array
        state_array = np.array(state)
        # Normalize the observation
        normalized_state = self.scaler.transform(state_array)
        
        # Convert observation to a 2D array
        new_state_array=np.array(new_state)
        # Normalize the observation
        normalized_new_state = self.scaler.transform(new_state_array)
        
        states = T.tensor(normalized_state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(normalized_new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones
  
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
    

    def decrement_epsilon(self):
        
        #self.epsilon = max(self.eps_min, self.epsilon - self.epsilon_decay_episodes)  #linear
        self.epsilon = max(self.eps_min, self.epsilon * self.epsilon_decay_episodes)  # exponential

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()     
        
  
        
   

       

    def learn(self):
     
        #DDQN
        
        if self.memory.mem_cntr < self.batch_size:
           return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        #self.decrement_epsilon() 
       
   
