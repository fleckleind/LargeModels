# MOE
[Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)

MOE: achieves efficient model expansion through dynamic sparsification, assigning input to multiple expert sub-networks for parallel processing, and only activating some to save computing resources.

## Mathematical Operation
Mathematically, the output $y$ of the MOE model is:
```math
y(x)=\sum_{i=1}^N g_i(x)\cdot f_i(x)
```
where $x$ is the input vector or embedding, $f_i(x)$ is the output of $i$-th expert, and $g_i(x)$ is the gating function output, which is constrained by $\sum_ig_i(x)=1$ and implemented via softmax function.
