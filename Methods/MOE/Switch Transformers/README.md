# Switch Transformers
[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Effcient Sparsity](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)

Switch Transformer: simplify MoE routing algorithm and design intuitive improved models with reduced communication and computational costs, to address complexity, communication costs and training instability.

## Mixture of Expert Routing
MoE: takes a token representation $x$ as input then routes this to the best determined top- $k$ experts selected from set ${E_i(x)}_{i=1}^N$ of $N$ experts. The output computation of the layer is calculated as:
```math
y=\sum_{i\in T}p_i(x)E_i(x), \quad
p_i(x)=\frac{e^{h(x)_i}}{\sum_j^N e^{h(x)_j}}, \quad
h(x)=W_r\cdot x
```
where $p_i(x)$ is the gate-value for expert $i$, router variable $W_r$ produces logits $h(x)$ normalized via $Softmax$ distribution, and $T$ is the set of selected top- $k$ indices for routing the token $x$.

## Switch Routing Layers
Terminologies of switch routing layers:
1. Experts: split across devices and perform standard feed-forward computation with unique parameters.
2. Expert Capacity: batch size of each expert is calculated as $(total\underline{ }tokens\underline{ }per\underline{ }batch/num\underline{ }experts)*capacity\underline{ }factor$.
3. Capacity Factor: allows more buffer to help mitigate token overflow during routing (unevenly dispatch), while increases computation and communication costs.

## Differentiable Load Balancing Loss
Given $N$ experts and a batch $B$ with $T$ tokens, the auxiliary loss is computed between vectors $f$ and $P$, with $\alpha=10^{-2}$:
```math
loss=\alpha\cdot N\cdot \sum_{i=1}^Nf_i\cdot P_i,\quad
```
and the computation of $f$, the fraction of tokens dispatched to expert $i$, and $P$, the router probability allocated for expert $i$, are written as:
```math
f_i=\frac{1}{T}\sum_{x\in B}1\{argmin\text{ }p(x)=i\},\quad
P_i=\frac{1}{T}\sum_{x\in B}p_i(x)
```

## Switch Transformer
Architecture: replace the dense feed forward network (FFN) layer with a sparse Switch FFN layer, operating independently on the tokens in the sequence and returning the selected FFN multiplied by the router gate value, with lower capacity factors setting $(1.0,1.25)$.

Techniques to improve training and fine-tuning:
1. Selective prevision with large sparse methods: cast the local routing operations to $float32$ while preserving $bfloat16$ precision to stabilize model while achieving nearly equal speed to $bfloat16$.
2. Smaller parameter initialization for stability: draw elements from a truncated normal distribution with mean $\mu=0$ and standard deviation $\sigma=\sqrt{s/n}$ where s is a scale hyper-parameter as $0.01$ and $n$ is the number of input units in the weight tensor.
3. Regularize large sparse models: increase the dropout inside the expert, as "expert dropout", to alleviate the overfitting of fine-tuning tasks with very few examples.
