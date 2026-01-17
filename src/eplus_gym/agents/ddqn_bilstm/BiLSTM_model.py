# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:55:34 2025

@author: kalsayed
"""



import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMModel(nn.Module):
    """
    DDQN sequence encoder based on a 2-layer bidirectional LSTM
    with attention pooling.

        • Input  : (B, 96, 18)
        • Output : (B, 65)  – Q-values
    """

    def __init__(self,
                 n_actions      = 65,
                 d_model        = 96,      # 32 per direction  →  param ≈ 55 k
                 lstm_layers    = 2,       # depth chosen to match Transformer
                 dropout        = 0.0):    # NO dropout (fair comparison)
        super().__init__()
        assert d_model % 2 == 0, "`d_model` must be even for Bi-LSTM."

        # ───── Input projection (18 → d_model) ───────────────────────
        self.in_embed = nn.Linear(18, d_model)

        # ───── Bidirectional LSTM encoder (2 layers) ────────────────
        self.encoder = nn.LSTM(input_size    = d_model,
                               hidden_size   = d_model // 2,  # per direction
                               num_layers    = lstm_layers,
                               batch_first   = True,
                               bidirectional = True,
                               dropout       = 0.0)           # keep 0 for parity

        # ───── Attention pooling over time ──────────────────────────
        # α_t = softmax(wᵀ h_t)  → pooled = Σ α_t h_t
        self.attn_score = nn.Linear(d_model, 1, bias=False)

        # Normalise pooled vector (no dropout)
        self.out_norm = nn.LayerNorm(d_model)

        # Final linear: pooled vector → Q for every discrete action
        self.q_proj = nn.Linear(d_model, n_actions)

        #self._reset_parameters()

    # ---------------------------------------------------------------
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_embed.weight)
        nn.init.constant_(self.in_embed.bias, 0.)
        # LSTM weights/biases get PyTorch’s Kaiming-uniform defaults.

    # ---------------------------------------------------------------
    def forward(self, S):                         # S: (B, 96, 18)
        x = self.in_embed(S)                      # (B, 96, 64)
        enc_out, _ = self.encoder(x)              # (B, 96, 64)

        # Attention weights α_t
        α_raw  = self.attn_score(enc_out)         # (B, 96, 1)
        α_norm = F.softmax(α_raw.squeeze(-1), 1)  # (B, 96)
        pooled = (α_norm.unsqueeze(-1) * enc_out).sum(1)  # (B, 64)

        pooled = self.out_norm(pooled)            # (B, 64)
        q_vals = self.q_proj(pooled)              # (B, 65)
        return q_vals

