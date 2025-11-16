# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:55:34 2025

@author: kalsayed
"""



import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding          # as in your current code

class TransformerModel(nn.Module):
    """
    Encoder–decoder DDQN backbone.
        • Encoder:   processes the 96×18 state sequence
        • Decoder:   one query per discrete action (65)
                     returns one scalar Q per action
    Forward:
        S  : (B, 96, 18)
        out: (B, 65)   – Q-values
    """
    def __init__(self,
                 n_actions      = 65,
                 d_model        = 64,
                 nhead          = 4,
                 num_enc_layers = 3,
                 num_dec_layers = 2,
                 dim_ff         = 256,
                 max_len        = 96):
        super().__init__()

        # ───── Encoder ────────────────────────────────────────────────────
        self.in_embed  = nn.Linear(18, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len=max_len)

        enc_layer  = nn.TransformerEncoderLayer(d_model, nhead,
                                                dim_feedforward=dim_ff,
                                                dropout=0,
                                                batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # ───── Decoder ────────────────────────────────────────────────────
        # learnable action-query tokens: (n_actions, d_model)
        self.action_queries = nn.Parameter(torch.randn(n_actions, d_model))

        dec_layer  = nn.TransformerDecoderLayer(d_model, nhead,
                                                dim_feedforward=dim_ff,
                                                dropout=0,
                                                batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        # project each decoded token → scalar Q
        self.q_proj = nn.Linear(d_model, 1, bias=False)     # shared projection

        # (optional) layer-norm on encoder memory to stabilise training
        self.mem_norm = nn.LayerNorm(d_model)

        #self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_embed.weight)
        nn.init.constant_(self.in_embed.bias, 0.)
        nn.init.normal_(self.action_queries, std=0.02)

    def forward(self, S):
        """
        S: (B, 96, 18)
        """
        B = S.size(0)

        # ── Encode sequence ────────────────────────────────────────────
        x   = self.in_embed(S)                  # (B, L, d)
        x   = self.pos_enc(x)
        mem = self.encoder(x)                   # (B, L, d)
        mem = self.mem_norm(mem)                # (B, L, d)

        # ── Prepare B copies of the action-query tokens ────────────────
        #    queries: (B, n_actions, d)
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)

        # ── Decode ─────────────────────────────────────────────────────
        #    tgt_mask not required (no autoregression)
        dec_out = self.decoder(tgt=queries, memory=mem)  # (B, n_actions, d)

        # ── Projection to scalar Qs ────────────────────────────────────
        q_vals = self.q_proj(dec_out).squeeze(-1)         # (B, n_actions)
        return q_vals
