# DeepSeekMoE
[DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)

DeepSeekMoE: involves finely segmenting the experts into $mN$ ones and activating $mK$ of them, and isolating $K_s$ experts as shared ones.

## Architecture
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

## Load Balance Auxiliary Loss
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
