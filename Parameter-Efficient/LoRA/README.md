# LoRA
[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685v1/1000)

LoRA: freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer, greatly reducing the number of trainable parameters for downstream tasks.

## Parameter-Efficient
The task-specific parameter increment $\Delta\Phi=\Delta\Phi(\Theta)$ is further encoded by a much smaller-sized set of parameters $\Theta$ with $\vert\Theta\vert\ll\vert\Phi_0\vert$. The task becomes optimizing over $\Theta$:
```math
\mathop{max}_{\Theta} \sum_{(x,y)\in Z} \sum_{t=1}^{\vert y\vert} 
log(p_{\Phi_0+\Delta\Phi(\Theta)} (y_t \vert x, y_{< t}))
```

## Low-Rank Constraint
The weight matrices in dense layers are allowed to have full-rank, and the pre-trained language models can still learn efficiently despite a low-dimensional reparametrization. As large number of parameters in not-full rank $W_0$ is redundant, use full-rank matrices $A, B$ to substitute. For $h=W_0x$, the modified forward pass yields:
```math
h=W_0x+\Delta Wx=W_0x +BAx\rightarrow (W_0+\frac{\alpha}{\gamma}AB)x
```
where $A\in R^{r\times k}$ is random Gaussian initialization and zero for $B\in R^{d\times r}$ with rank $r\ll min(d,k)$. During training, $W_0$ is frozen and $A, B$ contain trainable parameters. The scale $\alpha/\gamma$ represents the importance of low-rank adaptation output.

In the Transformer architecture, the LoRA is limited to only changing the attention weights for downstream tasks and freeze the MLP modules, with 4 times the number of trainable parameters given the same rank $r$ to the latter MLP modules.
