# GPT-2
[Language Models are Unsupervised Multitask Learners](https://storage.prod.researchhub.com/uploads/papers/2020/06/01/language-models.pdf)

Unsupervised Multitask Learning: zero-shot performance, a language model with sufficient capcity will begin to learn to infer and perform tasks demonstrated in natural language sequences.

## Approach
Language Modeling: unsupervised distribution estimation from example $()x_1,x_2,\ldots,x_n$ to variable length sequences of symbols $(s_1,s_2,\ldots,s_n)$. As natural sequential ordering, factorize the joint probabilities over symbols as the product of conditional probabilities:
```math
p(X)=p(s_{n-k},\ldots,s_n|s_1,\ldots,s_{n-k-1})=\prod_{i=1}^n p(s_n|s_1,\ldots,s_{n-1})
```
which predicts unknown suffix $(s_{n-k},\ldots,s_n)$ accroding to known $(s_1,\ldots,s_{n-k-1})$, equals to $p(output|input,task)$

## Implementation Details
The model largely follows the GPT-1 with a few modifications:
1. Layer Normalization: moved to the input of sub-blocks, and added after the final self-attention block;
2. Residual Initialization: scale the weights by a factor of $1/\sqrt{N}$, with $N$ the number of residual layers;
3. Others: byte pair encoding with 50257 vocabularies, context size as 1024 tokens, larger batchsize as 512 batches.
