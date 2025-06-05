# DeepSeek-V3
[DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)

DeepSeek-V3: adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, pioneers an auxiliary-loss-free strategy for load balancing, and sets a multi-token prediction training objective.

## Architecture
Basic Architecture of DeepSeek-V3: based on Transformer, with multi-head latent attention (MLA) for efficient inference, DeepSeekMoE for economical training, and a multi-token Prediction (MTP) training objective to mitigate the performance degradation.

### Multi-Head Latent Attention
Multi-Head Latent Attention: low-rank key-value joint compression, with original MHA needs to cache $2n_hd_hl$ elements for each token and kv-cache of GQA is $2n_gd_hl$. The core of MLA is written as follows:
```math
c_t^{KV}=W^{DKV}h_t,\quad k_t^c=W^{UK}c_t^{KV},\quad v_t^c=W^{UV}c_t^{KV}
```
where $c_t^{KV}\in R^{d_c}$ is the compressed latent vector for keys and values with compression dimension $d_c\ll d_hn_h$, $W^{DKV}\in R^{d_c\times d}$ as down-projection matrix, and $W^{UK},W^{UV}\in R^{d_hn_h\times d_c}$ as up-projection matrices, and the kv-cache only has $d_cl$ elements. Also, the queries also can perform the low-rank compression:
```math
c_t^Q=W^{DQ}h_t,\quad q_t^C=W^{UQ}c_t^Q
```

### Decoupled Rotary Position Embedding
RoPE: position-sensitive for both keys and queries, incompatible with low-rank KV compression and generating token between $W^Q$ and $W^{UK}$.


auxiliary-loss-free strategy for load balancing, multi-token prediction objective

### Implementation Details
DeepSeek LLM-7B: $n_{laer}=30$, $d_{model}=4096$, $n_{heads}=32$, $n_{kv\underline{ }heads}=32$, with 4096 context length, 2304 sequence batch size, $4.2e-4$ learning rate and $2.0T$ tokens.

Compared with LLaMA design, DeepSeek LLM use RMSNorm for LN, SwiGLU for FFN, with an intermediate layer dimension of $8d_{model}/3$, and Rotary Embedding for positional encoding.

Hyperparameters: initialized with a standard deviation of $0.006$ and trained with the AdamW optimizer with $\beta_1=0.9$, $\beta_2=0.95$, and $weight\underline{ }decaay=0.1$, with multi-step learning rate scheduler.

## Pre-Training
FP8 mixed precision training framework, cross-mode MoE training

## Post-Training
distill reaoning capabilities


## Reference
[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/pdf/2405.04434)  
[DeepSeek LLM Scaling Open-Source Language Models with Longtermism](https://arxiv.org/pdf/2401.02954)

