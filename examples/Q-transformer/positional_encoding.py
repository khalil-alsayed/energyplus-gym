# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:54:42 2025

@author: kalsayed
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        """
        d_model: dimension of each token embedding
        max_len: maximum sequence length (>= 96 for your use-case)
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Fill in sin for even indices, cos for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Reshape to (1, max_len, d_model) so we can broadcast across batch
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: shape (B, seq_len, d_model)
        returns x + positional_encoding
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x