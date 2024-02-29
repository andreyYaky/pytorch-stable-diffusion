import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        # embedding is vector that captures info on pixel/token

        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x; (B, Seq_Len, Dim)

        input_shape = x.shape
        B, sequence_length, d_embed = input_shape

        interim_shape = (B, sequence_length, self.n_heads, self.d_head)

        # (B, Seq_Len, Dim) -> (B, Seq_Len, Dim * 3) -> 3 tensors of shape (B, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim = 1)

        # (B, Seq_Len, Dim) -> (B, Seq_Len, H, Dim / H)
        # each head will watch whole sequence but only a part of each pixel/token
        # transpose -> (B, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (B, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        # apply mask (like tril to avoid seeing past)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (B, H, Seq_Len, Seq_Len) @ (B, H, Seq_Len, Dim / H) -> (B, H, Seq_Len, Dim / H)
        output = weight @ v

        # (B, H, Seq_Len, Dim / H) -> (B, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (B, Seq_Len, Dim)
        return output
    
class CrossAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x: (latent): (B, Seq_Len_Q, Dim_Q)
        # y: (context): (B, Seq_Len_KV, Dim_KV) = (B, 77, 768)

        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # multiply query by Wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # no tril mask needed now; want full context
        weight = q @ k.transpose(-1, 2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2).continuous()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output