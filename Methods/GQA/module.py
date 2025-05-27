import math
import torch
import torch.nn as nn


class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_heads, nums_key_value_head, attn_dropout_rate=0.1):
        super(GroupQueryAttention, self).__init__()
        assert hidden_dim % nums_heads == 0  # head_dim
        assert nums_heads % nums_key_value_head == 0  # query.head.num in a group

        self.head_dim = hidden_dim // nums_heads
        self.nums_heads, self.nums_key_value_head = nums_heads, nums_key_value_head
        # initialize qkv: x_dim, qkv_dim/number
        self.q_proj = nn.Linear(hidden_dim, nums_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        # attention dropout
        self.attn_dropout = nn.Dropout(attn_dropout_rate)

    def forward(self, x, attn_mask=None):
        batch, seq_len = x.size(0), x.size(1)  # x.shape: (batch, seq_len, hidden_dim)
        # multi-head qkv cache: -1: nums_heads(query)/nums_key_value_head(key/value)
        q_state = self.q_proj(x).view(batch, seq_len, -1, self.head_dim).permute(0, 2, 1, 3)
        k_state = self.k_proj(x).view(batch, seq_len, -1, self.head_dim).permute(0, 2, 1, 3)
        v_state = self.v_proj(x).view(batch, seq_len, -1, self.head_dim).permute(0, 2, 1, 3)
        # broadcast in num_head dimension for key and value
        k_state = k_state.repeat_interleave(self.nums_heads // self.nums_key_value_head, dim=1)
        v_state = v_state.repeat_interleave(self.nums_heads // self.nums_key_value_head, dim=1)
        # attention weight: [b,n_qh,s,h_d]*[b,n_vh,h_d,s]->[b,?,s,s]
        attention_weight = (q_state @ k_state.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attention_weight = attention_weight.masked_fill(attn_mask == 0, float('-inf'))
        attention_weight = self.attn_dropout(torch.softmax(attention_weight, dim=-1))
        output = (attention_weight @ v_state).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(output)
      
