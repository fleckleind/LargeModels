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
where $c_t^{KV}\in R^{d_c}$ is the compressed latent vector for keys and values with compression dimension $d_c\ll d_hn_h$, $W^{DKV}\in R^{d_c\times d}$ as down-projection matrix, and $W^{UK},W^{UV}\in R^{d_hn_h\times d_c}$ as up-projection matrices, and the kv-cache only has $d_cl$ elements. Also, the queries also can perform the low-rank compression, with $c_t^Q\in R^{d_c'}$ and $d_c'\ll d_hn_h$:
```math
c_t^Q=W^{DQ}h_t,\quad q_t^C=W^{UQ}c_t^Q
```

Decoupled RoPE Strategy: use additional multi-head queries $q_{t,i}^R\in R^{d_h^R}$ and a shared $k_t^R\in R^{d_h^R}$, as original RoPE is positional-sensitive for both keys and queries and incompatibel with low-rank KV-cache with synthetic token between $W^Q$ and $W^{UK}$, with $q_t^R$ and $k_t^R$ computation:
```math
[q_{t,1}^R;q_{t,2}^R;\ldots;q_{t,n_h}^R]=q_t^R=RoPE(W^{QR}c_t^Q),\quad
q_{t,i}=[q_{t,i}^C;q_{t,i}^R]
```
```math
k_t^R=RoPE(W^{KR}h_t),\quad k_{t,i}=[k_{t,i}^C;k_{t}^R]
```
where $W^{QR}\in R^{d_h^Rn_h\times d_c'}$ and $W^{KR}\in R^{d_h^R\times d}$ produce the decouples queries and key, $RoPE(\cdot)$ is $RoPE$ matrices, and remaining MLA is:
```math
o_{t,i}=\sum_{j=1}^t Softmax(\frac{q_{t,i}^Tk_{j,i}}{\sqrt{d_h+d_h^R}})v_{j,i}^C,\quad
u_t=W^O[o_{t,1};o_{t,2};\ldots;o_{t,n_h}]
```
with total KV-cache containing $(d_c+d_h^R)l$ elements, where $d_c$ is set to $4d_h$ and $d_h^R$ is set to $d_h/2$ in DeepSeek-V2.

### DeepSeekMoE
Fine-Grained Expert Segementation: segment each expert FFN into $m$ smaller experts by reducing the FFN intermediate hidden dimension to $1/m$ times its original size, with more potential combinations while keeping same computational parameters.

Shared Expert Isolation: isolate $K_s$ shared experts to capture and consolidate common knowledge across vayring contexts.
```math
h_t'=u_t+\sum_{i=1}^{N_s}FFN_i^{(s)}(u_t)+\sum_{i=1}^{N_r}g_{i,t} FFN_i^{(r)}（u_t)
```
```math
g_{i,t}=\begin{cases}
s_{i,t},& s_{i,t}\in Top-k(\{s_{j,t}\vert 1\leq j\leq N_r\}, K_r),\\
0,& \text{otherwise}.\end{cases}
```
```math
s_{i,t}=Softmax_i(u_t^\top e_i)
```

### Implementation Details
DeepSeek LLM-7B: $n_{laer}=30$, $d_{model}=4096$, $n_{heads}=32$, $n_{kv\underline{ }heads}=32$, with 4096 context length, 2304 sequence batch size, $4.2e-4$ learning rate and $2.0T$ tokens.

Compared with LLaMA design, DeepSeek LLM use RMSNorm for LN, SwiGLU for FFN, with an intermediate layer dimension of $8d_{model}/3$, and Rotary Embedding for positional encoding.

Hyperparameters: initialized with a standard deviation of $0.006$ and trained with the AdamW optimizer with $\beta_1=0.9$, $\beta_2=0.95$, and $weight\underline{ }decaay=0.1$, with multi-step learning rate scheduler.

### Load Balance Auxiliary Loss
Expert-Level Balance Loss: mitigate the risk of routing collapse,
```math
L_{ExpBal}=\alpha_1 \sum_{i=1}^{N_r}f_iP_i
```
```math
f_i=\frac{N_r}{K_rT}\sum_{t=1}^T1(\text{Token }t\text{ selects Expert } i), \quad
P_i=\frac{1}{T}\sum_{t=1}^Ts_{i,t}
```
where expert-level balance factor $\alpha_1$ is a hyper-parameter, and $T$ denotes the number of tokens in a sequence.

Device-Level Balance Loss: balance computation across different devices, with all routed experts partitioned into $D$ groups ${\epsilon_1,\epsilon_2,\ldots,\epsilon_D}$ with each group on a single device:
```math
L_{DecBal}=\alpha_2\sum_{i=1}^Df_i'P_i'
```
```math
f_i'=\frac{1}{\lvert \epsilon_i\rvert}\sum_{j\in \epsilon_i}f_i,\quad
P_i'=\sum_{j\in \epsilon_i} P_j
```

Communication Balance Loss: balance the token communication of each device, even though the device-limited routing mechanism,
```math
L_{CommBal}=\alpha_3\sum_{i=1}^D f_i''P_i''
```
```math
f_i''=\frac{D}{MT}\sum_{t=1}^T 1(\text{Token }t\text{ is sent to Device }i),\quad
P_i''=\sum_{j\in\epsilon_i}P_j
```
The device-limited routing mechanism on the principle ensures that each device transmits at most $MT$ hidden states to other devices.

## Pre-Training
FP8 mixed precision training framework, cross-mode MoE training

## Post-Training
distill reaoning capabilities


## Reference
[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/pdf/2405.04434)  
[DeepSeek LLM Scaling Open-Source Language Models with Longtermism](https://arxiv.org/pdf/2401.02954)  
[DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)

