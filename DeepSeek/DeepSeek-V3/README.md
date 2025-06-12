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
Multi-Head Latent Attention (MLA): low-rank joint compression to reduce key-value (KV) cache during inference, with embedding dimension $d$, the number of attention heads $n_h$, the dimension per head $d_h$, and attention input $h_t\in R^d$. The KV calculation is:
```math
c_t^kv=W^{DKV}h_t
```
```math
[k_{t,1}^C;k_{t,2}^C;\ldots;k_{t,n_h}^C=k_t^C=W^{UK}c_t^{KV}
```
```math
[v_{t,1}^C;v_{t,2}^C;\ldots;v_{t,n_h}^C=v_t^C=W^{UK}c_t^{KV}
```
with compressed latent vector $c_t^{KV}\in R^{d_c}$, compression dimension $d_c\ll d_hn_h$, down-projection matrix $W^{DKV}\in R^{d_c\times d}$, up-projection matrices $W^{UK}, W^{UV}\in R^{d_hn_h\times d_c}$, only $c_t^{KV}$ and $k_t^R$ needed to be cache.
```math
\quad
k_t^R=RoPE(W^{KR}h_t),\quad
k_{t,i}=[k_{t,i}^C;k_t^R]
```
and Rotary Positional Embedding is decomposed for positional information. And low-rank compression for the attention queries is shown as follows, with RoPE for each head of query:
```math
c_t^Q=W^{DQ}h_t,\quad
[q_{t,1}^C;q_{t,2}^C;\ldots;q_{t,n_h}^C=q_t^C=W^{UQ}c_t^{Q}
```
```math
[q_{t,1}^R;q_{t,2}^R;\ldots;q_{t,n_h}^R=q_t^R=W^{QR}c_t^{Q},\quad
q_{t,i}=[q_{t,i}^C;q_{t,i}^R]
```

### DeepSeekMoE
DeepSeekMoE: finer-grained experts and isolates some experts as shared ones, with FFN input $u_t$ and output as,
```math
h_t'=u_t+\sum_{i=1}^{N_s}FFN_i^{(s)}(u_t) +\sum_{i=1}^{N_r}g_{i,t}FFN_i^{(r)}(u_t)
```
```math
g_{i,t}=\frac{g_{i,t}'}{\sum_{j=1}^{N_r}g_{j,t}'},\quad
g_{i,t}'=\begin{cases}
s_{i,t},& \text{ } s_{i,t}\in Topk(\{s_{j,t}\vert1\leq j\leq N_r\},K_r)\\
0,& \text{ } \text{otherwise}\end{cases}
```
```math
s_{i,t}=Sigmoid(u_t^\top e_i)
```

Auxiliary-Loss-Free Load Balancing: unbalanced expert load lead to routing collapse and diminish computational efficiency, while too large an auxiliary loss will impair the model performance.
```math
g_{i,t}'=\begin{cases}
s_{i,t},& \text{ } s_{i,t}+b_i\in Topk(\{s_{j,t}+b_j\vert 1\leq j\leq N_r\}, K_r)\\
0,& \text{ otherwise}\end{cases}
```
where the bias term $b_i$ is introduced to determine the top-K routing, if corresponding expert overload the bias term is decreased by bias upedate speed $\gamma$, a hyper-parameter.

Complementary Sequence-Wise Auxiliary Loss: prevent imbalance within any single sequence, with the number $T$ of tokens in sequence:
```math
L_{Bal}=\alpha\sum_{i=1}^{N_r}f_iP_i,\quad
P_i=\frac{1}{T}\sum_{t=1}^Ts_{i,t}',\quad
s_{i,t}'=\frac{s_{i,t}}{\sum_{j=1}^{N_r}s_{j,t}}
```
```math
f_i=\frac{N_r}{K_rT}\sum_{t=1}^T1(s_{i,t}\in Topk(\{s_{j,t}\vert1\leq j\leq N_r\},K_r))
```

Node-Limited Routing: send each token to at most $M$ nodes, selected according to the sum of the highest $K_r/M$ affinity scores of experts distributed on each node, to limit communication costs.

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


