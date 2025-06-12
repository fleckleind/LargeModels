# DeepSeek-V3
[DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)

DeepSeek-V3: adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, pioneers an auxiliary-loss-free strategy for load balancing, and sets a multi-token prediction training objective.

## Architecture
Basic Architecture of DeepSeek-V3: based on Transformer, with multi-head latent attention (MLA) for efficient inference, DeepSeekMoE for economical training, and a multi-token Prediction (MTP) training objective to mitigate the performance degradation.

### Implementation Details
DeepSeek LLM-7B: $n_{layer}=30$, $d_{model}=4096$, $n_{heads}=32$, $n_{kv\underline{ }heads}=32$, with $length_{context}=4096$, $batch\underline{ }size_{sequence}=2304$, $learning\underline{ }rate=4.2e-4$ and $2.0T$ tokens.

Compared with LLaMA design, DeepSeek LLM use RMSNorm for LN, SwiGLU for FFN, with an intermediate layer dimension of $8d_{model}/3$, and Rotary Embedding for positional encoding.

Hyperparameters: initialized with a standard deviation of $0.006$ and trained with the AdamW optimizer with $\beta_1=0.9$, $\beta_2=0.95$, and $weight\underline{ }decaay=0.1$, with multi-step learning rate scheduler.

### Multi-Head Latent Attention



### DeepSeekMoE with Auxilary-Loss-Free Load Balancing





## Multi-Token Prediction
Multi-Token Prediction (MTP): extend the prediction scope to multiple future tokens at each position, densifying the training signals and enable the model to pre-plan its representations.

MTP modules: sequentially predict $D$ additional tokens with $D$ sequential modules, and keep the complete causal chain at each prediction depth. MTP consists of a shared embedding layer, a shared output head, a Transformer block, and a projection matrix $M_k\in R^{d\times 2d}$.
```math
{h'}_{i}^{k}=M_k[RMSNorm(h_i^{k-1});RMSNorm(Emb(t_{i+k})]
```
where $h_i^{k-1}\in R^d$ is the representation of the $i$-th token at the $(k-1)$-th depth, $Emb(t_{i+k})\in R^d$ is the embedding of the $(i+k)$-th token, and embedding layer for each MTP is shared with the main model. 
```math
h_{1:T-k}^k=TRM_k({h'}_{1:T-k}^k),\quad
p_{i+k+1}^k=OutHead(h_i^k)
```
where $h_i^k$ is the output representation via the Transformer block $TRM$ at $k$-th depth, $T$ represent the input sequence length, $p_{i+k+1}^k\in R^V$ is the probability distribution for the $k$-th additional prediction token with $V$ vocabulary size. The output head $OutHead(\cdot)$, shared with the main model, consists of linear layer and a softmax function.

MTP Training Objective: cross-entropy loss for each prediction depth,
```math
L_{MTP}^k=CrossEntropy(p_{2+k:T+1}^k, t_{2+k:T+1})=-\frac{1}{T}\sum_{i=2+k}^{T+1} log P_i^k[t_i]
```
with the input sequence length $T$, the ground-truth token $t_i$, and the corresponding prediction probability $P_i^k[t_i]$, and average of the MTP losses across all depths with weighting factor $\lambda$,
```math
L_{MTP}=\frac{\lambda}{D}\sum_{k=1}^DL_{MTP}^k
```

MTP Inference: directly discard the MTP modules and the main model can function independently and normally.








## Pre-Training
FP8 mixed precision training framework, cross-mode MoE training

## Post-Training
distill reaoning capabilities


## Reference
 
[DeepSeek LLM Scaling Open-Source Language Models with Longtermism](https://arxiv.org/pdf/2401.02954)  


