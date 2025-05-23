# GPT-1
[Improving Language Understanding by Generative Pre-Training](https://www.mikecaptain.com/resources/pdf/GPT-1.pdf)

GPT: generative pre-training of a language model on a diverse corpus of unlabeled text, then discriminative fine-tuning on each specific task.

## Unsupervised Pre-Training
Unsupervised Pre-Training: learn a high-capacity language model on a large corpus of text, with standard language modeling objective to maximize following likelihood:
```math
L_1(U)=\sum_i log P(u_i|u_{i-k},\ldots,u_{i-1};\Theta)
```
where $k$ the size of context window, $P$ the conditional probability modeleed by a neural network with learnable parameters $\Theta$. A multi-layer Transformer decoder is used as the language model:
```math
\begin{align}
h_0 &= U W_e + W_p \\
h_l &= transformer\_block (h_{l-1}), \text{ } \forall l\in[1,n]\\
P(u) &= softmax(h_n W_e^\top)
\end{align}
```
where context vector of tokens $U=(u_{-k},\ldots,u_{-1})$, number of layers $n$, token embedding matrix $W_e$, and position embedding matrix $W_p$.

The Transformer architecture is consisted of masked MHA, FFN, and layer normalization, with input as text and position embedding, and ouput as text prediction or classifier.

## Supervised Fine-Tuning
Supervised Fine-Tuning: adapt the model to a discriminative task with labeled data, an added linear output layer $W_y$ with final transformer block's activation $h_l^m$ to predict label $y$:
```math
P(y|x^1,\ldots,x^m)=softmax(h_l^m W_y)
```
and maximize the following objective:
```math
L_2(C)=\sum_{(x,y)} log P(y|x^1,\ldots,x^m)
```
To improve generalization of the supervised model and accelerate convergence, language modeling is added as an auxiliary object to the fine-tuning with weight $\lambda$:
```math
L_3(C)=L_2(C)+\lambda L_1(C)
```

## Task-Specific Input Transformation
Traversal-Style Approach: convert structured inputs into an ordered sequence, to avoid making extensive changes to the architecture across tasks. All transformations include adding randomly initialized start and end tokens $(\langle s\rangle,\langle e\rangle)$, with text entailment, similarity, question Answering and commonsense reasoning tasks concatenating input tokens and label, and using delimiter token (\$) to separate.

## Implementation Details
Model largely follows the original transformer work:
1. Decoder-only Transformer: 12 layers;
2. Masked Self-Attention: 768 dimensional states, 12 heads;
3. Position-wise FFN: 3072 dimensional inner states;
4. Optimizer: Adam optimization scheme, with max learning rate of 2.5e-4, increased linearly from 0 over the first 2000 updates and annealed to 0 using a cosine schedule;
5. Others: 100 epochs, 64 batches, contiguous sequences of 512 tokens, 0.1 attention dropout rate, GELU activation, byte pair encoding with 40000 byte pairs.
