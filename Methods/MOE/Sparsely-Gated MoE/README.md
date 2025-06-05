# Sparsely-Gated MoE
[Outrageously Large Neural Networks: the Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/pdf/1701.06538)

Sparsely-Gated Mixture-of-Experts: based on conditional computation, a trainable gating network determines a sparse combination of these experts to use for each example.

## Conditional Computation
Conditional Computation: increase model capacity without a proportional increase in computational costs, where large parts of network are active or inactive on a per-example basis, with the gating decisions binary or sparse and continuous, stochastic or deterministic.

## Sparsely-Gated Mixture-of-Experts Layer
Sparsely-Gated Mixture-of-Experts Layer (MoE): consists of experts, each a simple feed-forward neural network $E_i(x)$, and a trainable gating network $G(x)$ which selects a sparse combination of experts to process each input, with all parts trained jointly by back-propagation.
The output $y$ of the MoE module is written as follows:
```math
y=\sum_{i=1}^n G(x)_iE_i(x)
```
where $n$ experts accept the same sized inputs and produce the same-sized outputs with separate parameters, and the output of gating network $G(x)$ is a sparse $n$-dimensional vector.

## Gating Network
Softmax Gating: non-sparse gating function, multiply the input by a trainable weight matrix $W_g$ and apply $Softmax$ function:
```math
G_\sigma(X)=Softmax(x\cdot W_g)
```

Noisy Top-K Gating: add tunable Gaussian noise and keep only the top-k values, setting the rest to $-\infty$ before taking $Softmax$:
```math
\begin{align}
G(x) &= Softmax(KeepTopK(H(x), k)) \\
H(x)_i &= (x\cdot W_g)_i+StandardNormal()\cdot Softplus((x\cdot W_{noise})_i) \\
KeepTopK(v, k)_i &=\begin{cases}
v_i, & \text{if } v_i \text{ is in the top }k \text{ elements of }v \\
-\infty, & \text{otherwise.}
\end{cases}\end{align}
```
where $StandardNorm()$ is the standard normalization and $softplus$ is a smooth and non-linear activation funciton, as
```math
StandardNorm(z)= \frac{z-\mu}{\sigma},\quad
Softplus(z)= log(1+e^{x})
```

## Balancing Expert Utilization
Self-Reinforcing Imbalance: gating network always produces large weights for the same few experts. Sparsely-Gated MoE take a soft constraint approach on the batch-wose average of each gate.

An additional loss $L_{importance}$ is equal to the square of the coefficient of variation of the set of importance values and multiplied by a hand-tuned scaling factor $w_{importance}$, relative to a batch of training examples to be the batchwise sum of the gate values for that expert, to encourage all experts to have equal importance.
```math
\begin{align}
L_{importance}(X)
&=w_{importance}\cdot CV(Importance(X))^2\\
&=w_{importance}\cdot CV(\sum_{x\in X}G(x))^2\end{align}
```
