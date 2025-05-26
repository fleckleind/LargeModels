# GPT-3
[Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)

GPT-3: an autoregressive language model with 175 billion parameters 10x more than any previous non-sparse language model, without any gradient updates or fine-tuning, and performance in few-shot setting.

## In-Context Learning
In-Context Learning: a form of prompt learning, temporarily learns to complete new tasks through examples (contexts) in the prompt, without any parameters tuning through back propagation.

## Task-Specific Data
Four Points on the spectrum of how much task-specific data relied on:
1. Fine Tuning: involve updating the weights of a pre-trained model by training on a supervised dataset specific to the desired task；
2. Few-Shot: model is given a few demonstrations (10-100) of the task at inference time as conditioning, without any weight update;
3. One-Shot: only one demonstration is allowed, in addition to a natural language description of the task;
4. Zero-Shot: no demonstrations are allowed, and the model is only given a natural language instruction describing the task, providing potential for rubstness and avoidance of spurious correlations.

## Model and Implementation
GPT-2: modified initialization, pre-notmalization, reversible tokenization, and locally banded sparse attention patterns, with implementation details shown as follows:
1. Total number of trainable parameters: 175.0B;
2. Others details: 12 layers, 2048 tokens for context window in all models, 12288 units in each bottleneck layer, with 96 attention heads and 128 dimension for each head.
3. Training Params: 3.2M batch size, and $0.6\times10^{-4}$ learning rate.
