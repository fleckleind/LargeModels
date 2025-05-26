import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 512  # max_seq_len
    batch_size: int = 12
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768  # hidden_dim/size
    dropout: float = 0.1
    head_size: int = n_embed // n_head  # head_dim
    vocab_size: int = 50257  # GPT-2 official tokenizer


class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super(SingleHeadAttention, self).__init__()
        # qkv projection
        self.query = nn.Linear(config.n_embed, config.head_size)
        self.key = nn.Linear(config.n_embed, config.head_size)
        self.value = nn.Linear(config.n_embed, config.head_size)
        # attention mask: w/o gradient computation
        self.register_buffer(
            "attention_mask",
            torch.tril(
                # lower triangular matrix, [max_seq_len, max_seq_len]
                torch.ones(config.block_size, config.block_size)
            )
        )
        self.head_size = config.head_size
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        k, q, v = self.query(x), self.key(x), self.value(x)
        weight = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float('-inf')
        )
        weight = torch.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        output = weight @ v
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        # multi-head self-attention
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_head)
            ]
        )
        # output projection
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # concatenate all heads of single head self-attention
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.dropout(self.proj(output))
        return output


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed * 4),
            nn.GELU(),  # if SWiGLU, hidden as n_embed * 8/3
            nn.Linear(config.n_embed * 4, config.n_embed),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        # sinusoidal->RoPE, LN->RMS, GELU->SWiGLU, MHA->GQA
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # small language model, reduce params with Tie Weight
        self.token_embedding_table.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: input token ids, targets: target token ids
        batch_size, seq_len = idx.size()
        tok_embed = self.token_embedding_table(idx)  # (batch_size, seq_len, n_embed)
        pos_embed = self.position_embedding_table(
            # position index, get position embedding from position embedding table
            torch.arange(seq_len, device=idx.device)
        )
        x = tok_embed + pos_embed
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            batch_size, seq_len, vocab_size = logits.size()
            logits = logits.view(-1, vocab_size)
            targets = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # new token: (batch_size, vocab_size)
            probs = F.softmax(logits. dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            idx = torch.cat([idx, idx_next], dim=1)  # (batch_size, seq_len+1)
        return idx
