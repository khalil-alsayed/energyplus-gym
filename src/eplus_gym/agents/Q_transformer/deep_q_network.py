# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:16:56 2024

@author: kalsayed
"""
        
import os
import torch as T
import torch.nn as nn
import torch.optim as optim

# Import our Transformer-based Q model
from eplus_gym.agents.Q_transformer.transformer_model import TransformerModel

class DeepQNetwork(nn.Module):
    def __init__(
        self,
        lr,
        n_actions,
        name,
        input_dims,     # e.g. (96, 18) if your sequences are (96,18)
        chkpt_dir,
        d_model=32,
        nhead=4,
        num_layers=2,
        max_len=96
    ):
        """
        A Transformer-based DQN that, given a state sequence, outputs Q-values
        for all actions.

        :param lr: learning rate
        :param n_actions: number of discrete actions
        :param name: checkpoint filename
        :param input_dims: not strictly used in the Transformer directly,
                           but kept for potential checks or reference
        :param chkpt_dir: directory to store checkpoints
        :param d_model, nhead, num_layers, max_len: hyperparams for Transformer
        """
        super(DeepQNetwork, self).__init__()

        self.n_actions = n_actions
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # ----------------------------------------------------
        # 1) Create the Transformer Q-Model
        # ----------------------------------------------------
        self.transformer = TransformerModel(
            d_model=d_model,
            nhead=nhead,
            max_len=max_len,
            n_actions=n_actions  # so the model knows how many Q-values to output
        )
        
        # ----------------------------------------------------
        # 2) Optimizer and Loss
        # ----------------------------------------------------
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state_seq):
        """
        Forward pass to compute Q-values for all actions.

        :param state_seq: Tensor of shape (B, 96, 18)
        :return: Tensor of shape (B, n_actions), the predicted Q-values
        """
        q_values = self.transformer(state_seq)  # (B, n_actions)
        return q_values

    def save_checkpoint(self):
        print('... saving Transformer Q-network checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading Transformer Q-network checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file,weights_only=True))

        

        