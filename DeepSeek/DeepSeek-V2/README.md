# DeepSeek-V2
[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/pdf/2405.04434) 

DeepSeek-V2: adopts multi-head latent attention significantly compressing KV cache and DeepSeekMoE with approved communication balance loss through sparse computation.

## Multi-Head Latent Attention
Multi-Head Latent Attention: low-rank key-value joint compression, with original MHA needs to cache $2n_hd_hl$ elements for each token and kv-cache of GQA is $2n_gd_hl$. The core of MLA is written as follows:
```math
c_t^{KV}=W^{DKV}h_t,\quad k_t^c=W^{UK}c_t^{KV},\quad v_t^c=W^{UV}c_t^{KV}
```
where $c_t^{KV}\in R^{d_c}$ is the compressed latent vector for keys and values with compression dimension $d_c\ll d_hn_h$, $W^{DKV}\in R^{d_c\times d}$ as down-projection matrix, and $W^{UK},W^{UV}\in R^{d_hn_h\times d_c}$ as up-projection matrices, and the kv-cache only has $d_cl$ elements. Also, the queries also can perform the low-rank compression, with $c_t^Q\in R^{d_c'}$ and $d_c'\ll d_hn_h$:
```math
c_t^Q=W^{DQ}h_t,\quad q_t^C=W^{UQ}c_t^Q
```

## Decoupled RoPE Strategy
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

## Auxiliary Loss for Load Balance
DeepSeek-V2 MoE: same architecture with DeepSeekMoE, while communication balance loss is approved to ensure the balanced communication of each device, based on original expert-level (routing collapse) and device-level (device computation) balance loss.

Communication Balance Loss: balance the token communication of each device, even though the device-limited routing mechanism,
```math
L_{CommBal}=\alpha_3\sum_{i=1}^D f_i''P_i''
```
```math
f_i''=\frac{D}{MT}\sum_{t=1}^T 1(\text{Token }t\text{ is sent to Device }i),\quad
P_i''=\sum_{j\in\epsilon_i}P_j
```
where communication balance factor $\alpha_3$ is a hyper-parameter, abd the device-limited routing mechanism on the principle ensures that each device transmits at most $MT$ hidden states to other devices.

