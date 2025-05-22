import torch
import torch.nn as nn


class LinearRoLALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, lora_alpha, dropout_rate, merge=False):
        super(LinearRoLALayer, self).__init__()
        self.rank, self.merge = rank, merge
        # pretrained weights setting
        self.linear = nn.Linear(in_features, out_features)  # [d_o, d_i]^T
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False
        # low-rank adaptation
        if rank > 0:
            # random Gaussian initialization
            self.lora_a = nn.Parameter(torch.randn(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.lora_b = nn.Parameter(torch.zeros(rank, in_features))
            self.scale = lora_alpha / rank
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        # merge: combine weight, reduce computation when inference
        if merge:
            self.merge_weights()

    def merge_weights(self, ):
        if self.merge and self.rank > 0:
            self.linear.weight.data += self.scale * (self.lora_a @ self.lora_b)

    def forward(self, x):
        if self.rank > 0:
            output1 = self.linear(x)  # [b,s,d_i]->[b,s,d_o]
            output2 = self.scale * (x @ (self.lora_a @ self.lora_b).T)
            output = output1 + output2
        else:
            output = self.linear(x)
        return self.dropout(output)
      
