import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, attn_dropout=0.1, *args, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.attention_dropout = nn.Dropout(attn_dropout)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attn_mask=None):
        batch, seq_len, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # (batch, seq_len, hid_dim)->(batch, head_num, seq_len, head_dim)
        q_state = q.view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        k_state = k.view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        v_state = v.view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        # attn: (batch, head, seq, dim)->(batch, head, seq, seq)
        attn = q_state @ k_state.transpose(-1, -2) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        # attn: (batch, head, seq, seq)->(batch, head, seq, dim)
        attn = self.attention_dropout(torch.softmax(attn, dim=-1))
        output = (attn @ v_state).transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, -1)
        return self.o_proj(output)


class Adapter(nn.Module):
    def __init__(self, dim, adapter_dim, init_scale, act_fn="gelu"):
        super(Adapter, self).__init__()

        self.down_proj = nn.Linear(dim, adapter_dim)
        nn.init.kaiming_normal_(self.down_proj.weight), nn.init.zeros_(self.down_proj.bias)
        self.up_proj = nn.Linear(adapter_dim, dim)
        nn.init.zeros_(self.down_proj.weight), nn.init.zeros_(self.down_proj.bias)
        self.act_func = nn.GELU() if act_fn == "gelu" else nn.ReLU(inplace=True)
        # scale: control the importance of adapter
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x):
        output = self.up_proj(self.act_func(self.down_proj(x)))
        return x + self.scale * output


class AdapterTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, attn_dim, attn_scale, ffn_dim, ffn_scale,
                 attn_fn="gelu", ffn_fn="gelu", attn_dropout_rate=0.1):
        super(AdapterTransformerLayer, self).__init__()
        # multi-head self-attention
        self.MHA = MultiHeadAttention(hidden_dim, num_heads, attn_dropout_rate)
        self.attn_adapter = Adapter(hidden_dim, attn_dim, attn_scale, attn_fn)
        self.attn_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        # FFN: up, down
        self.up_proj = nn.Sequential(  # swishGLU: 8/3
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.ffn_dropout = nn.Dropout(0.1)
        self.ffn_adapter = Adapter(hidden_dim, ffn_dim, ffn_scale, ffn_fn)
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def mha(self, x, attn_mask=None):
        output = self.MHA(x, attn_mask)
        output = self.attn_adapter(output)
        return self.attn_norm(x + output)

    def ffn(self, x):
        output = self.down_proj(self.up_proj(x))
        output = self.ffn_dropout(output)
        output = self.ffn_adapter(output)
        return self.ffn_norm(x + output)

    def forward(self, x, attn_mask=None):
        x = self.mha(x, attn_mask)
        x = self.ffn(x)
        return x
      
