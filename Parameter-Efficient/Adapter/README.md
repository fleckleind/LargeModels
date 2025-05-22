# Adapter
[Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf)

Adapter: add only a few trainable parameters per task, and new tasks can be added without revisiting previous ones, with the parameters of the original network fixed.

## Adapter Tuning
Adapter Tuning Strategy: involve injecting new layers into the original network, with the parameters of the original network frozen, and adapter lauers initialized at random.

Adapter modules have two main features:
1. a small number of parameters, that the total model size grows relatively slowly when more tasks are added,
2. a near-identity initialization, stable training of the adapted model.

## Instantiation for Transformer
The adpater modules is added twice to each Transformer sub-layer, after the projection following multi-headed attention and the two feed-forward layers, but before adding the skip connection back.

The adapter module is a bottleneck architecture with skip connection, projecting original $d$-dimensional features into a smaller dimension $m\ll d$, applying a nonlinearity, then projecting back to $d$ dimension.

Three trainable layers during adapter tuning on the downstream data:
1. adapter, with parameters including biases as $2md+d+m$,
2. layer normalization, with parameters as $2d$,
3. the final classification layer, with parameters as $dN_{class}$.
