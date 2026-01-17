# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:16:56 2024

@author: kalsayed
"""
        
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim





# ----------------------------------------------------
# DeepQNetwork (Single Transformer, multi-feature seq)
# ----------------------------------------------------
class DeepQNetwork(nn.Module):
    def __init__(
        self,
        lr,
        n_actions,
        name,
        input_dims,
        chkpt_dir
    ):
        super(DeepQNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        

        # ----------------------------------------------------
        # Fully-Connected Layers
        # ----------------------------------------------------
        self.fc1 = nn.Linear(*input_dims, 400)  # Increase width of the first layer
        self.fc2 = nn.Linear(400, 300)          # Increase width of the second layer
        self.fc3 = nn.Linear(300, n_actions)    # Output layer
        
        # Dropout layer(s)
        #self.dropout = nn.Dropout(p=0.1)  # you can experiment with p=0.1 or 0.3
        
        # ───── custom weight init  ───────────────────────────────────
        #self._reset_parameters()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def _reset_parameters(self):
       # Xavier-uniform for weights
       nn.init.xavier_uniform_(self.fc1.weight)
       nn.init.xavier_uniform_(self.fc2.weight)
       nn.init.xavier_uniform_(self.fc3.weight)

       # Zero bias
       nn.init.constant_(self.fc1.bias, 0.)
       nn.init.constant_(self.fc2.bias, 0.)
       nn.init.constant_(self.fc3.bias, 0.)
        
    

    def forward(self, state):
        

        # Forward pass through the network
        layer1 = F.relu(self.fc1(state))
        #layer1 = self.dropout(layer1)  # Apply dropout after the first layer
        layer2 = F.relu(self.fc2(layer1))
        #layer2 = self.dropout(layer2)  # Optionally apply dropout again
        actions = self.fc3(layer2)
        return actions
    
       

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
        
  
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
    
 
    
 
        
 
        

        