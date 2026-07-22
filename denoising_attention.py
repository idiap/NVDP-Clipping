# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, query, key, value, attn_mask=None, is_causal=False):

       

        B, T, C = query.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        S = key.size(-2)
        if attn_mask is not None:
            attn_mask = (
                attn_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_attention_heads, T, 1)
            )  # [B, nh, T, S]

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        k = k.view(B, T, self.num_attention_heads, C // self.num_attention_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.num_attention_heads, C // self.num_attention_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.num_attention_heads, C // self.num_attention_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)

        ############
        # Attention calculation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Masking
        attn_bias = torch.zeros(B, self.num_attention_heads, T, S, dtype=query.dtype, device =self.device)
        if is_causal:
            # assert attn_mask is None
            temp_mask = torch.ones(T, S, dtype=torch.bool, device = self.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(~attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        att += attn_bias

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        ############
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att


class DenoisingMultiheadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, query, key, value, attn_mask=None, is_causal=False, nvib_tuple=None):
        B, T, C = query.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        S = key.size(-2)
        hs = C // self.num_attention_heads  # Dimension of each head

        # NVIB: In training the key is z and is sampled
        if nvib_tuple is not None: 
            z, pi, mu, logvar, alpha, mask = nvib_tuple
            S = z.size(-2)  # S+1 to include the prior

        # [B, nh, T, S]
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj(query)
        if self.training:
            # NOTE: scaling here
            k = self.k_proj(z) * (1.0 / math.sqrt(hs))
            v = self.v_proj(z)
        else:
            # NOTE: scaling implicit here
            biased_var = torch.exp(logvar) + math.sqrt(hs)
            k = self.k_proj(mu / biased_var)
            v = self.v_proj(mu / biased_var)

        k = k.view(B, S, self.num_attention_heads, hs).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_attention_heads, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, S, self.num_attention_heads, hs).transpose(1, 2)  # (B, nh, T, hs)

        ############
        # Attention calculation
        att = q @ k.transpose(-2, -1)

        # Masking
        attn_bias = torch.zeros(B, self.num_attention_heads, T, S, dtype=query.dtype, device = self.device)
        if is_causal:
            # assert attn_mask is None
            # Notice tril(1) this means offset 1, so the diagonal is not masked
            temp_mask = torch.ones(T, S, dtype=torch.bool, device = self.device).tril(diagonal=1)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_attention_heads, T, 1)
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(~attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        att += attn_bias

        ###### NVIB BIAS TERMS ######
        # Pi term is clamped to avoid log(0)
        pi = torch.clamp(pi.clone(), min=torch.finfo(pi.dtype).tiny)  # [B, S, 1]

        if self.training:
            # L2 norm term
            l2_norm = (1 / (2 * math.sqrt(hs))) * (
                (torch.norm(z, dim=-1, keepdim=True)) ** 2
            )  # [B, S, 1]

            # Include bias terms, copied over heads, broadcasted over T
            att += (torch.log(pi) - l2_norm).unsqueeze(1).permute(0, 1, 3, 2)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        else:
            # L2 norm term
            l2_norm = 0.5 * (
                (torch.norm((mu / torch.sqrt(biased_var)), dim=-1, keepdim=True)) ** 2
            )  # [B, S, 1]

            # Variance penalty term
            var_penalty = torch.sum(
                0.5 * (torch.log(biased_var).masked_fill_(attn_mask[:, 0, 0, :].unsqueeze(-1), 0)),
                dim=-1,
                keepdim=True,
            )  # [B, S, 1]

            # Include bias terms, copied over heads, broadcasted over T
            att += (torch.log(pi) - l2_norm - var_penalty).unsqueeze(1).permute(0, 1, 3, 2)

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            ###### NVIB INTERPOLATION ######
            # Interpolate the pre-projected value "mu" and post-projected query "projected_u"
            # Create projected_u
            w_k = self.k_proj.weight.view(self.num_attention_heads, hs, -1)  # [nh, hs, C]
            w_v = self.v_proj.weight.view(self.num_attention_heads, hs, -1)  # [nh, hs, C]
            if self.v_proj.bias is not None:
                b_v = self.v_proj.bias.view(self.num_attention_heads, hs, 1)  # [nh, hs, C]
            else:
                b_v = torch.zeros(self.num_attention_heads, hs, 1, dtype=query.dtype)
            projected_u = torch.einsum("bhme, hep -> bhmp", q, w_k)
            # out = torch.einsum("bhmp, bnp -> bhmn", projected_u, (mu/biased_var)) # Same as before
            # Calculate the interpolation
            output = (
                torch.einsum("bhmn, bnp -> bhmp", att, (torch.exp(logvar) / biased_var))
                * projected_u
            ) + torch.einsum("bhmn, bnp -> bhmp", att, ((math.sqrt(hs) / biased_var) * mu))
            # Project into the correct space and add the bias. The bias is theoretically
            # multiplied by the attention_probs, but it noramlises so we can just add it.
            y = torch.einsum("bhmp, hep -> bhme", output, w_v) + b_v.unsqueeze(0).permute(
                0, 1, 3, 2
            )

        ############
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att
