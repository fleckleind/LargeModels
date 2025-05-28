# Switch Transformers
[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Effcient Sparsity](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)

Switch Transformers: simplify the MoE routing algorithm and design a sparsely activated model with reduced communication and computational costs.

## Sparse Routing Simplification
Mixture of Expert Routing: takes a token representation $x$ as input, and routes to the best determined top-$k$ experts selected from ${E_i(x)}$. With the router variable $W_r$, the gate-value for expert $i$ is given by:
```math
p_i(x)=\frac{e^{h(x)_i}}{\sum_{j}^N e^{h(x)_j}},\quad
h(x)=W_r\cdot x
```
Assign $T$ is the set of selected top-$k$ indices from token $x$ routing, the output computation of the layer is:
```math
y = \sum_{i\in T} p_i(x)E_i(x)
```

