# LargeModels

## Encoder-Decoder
Transformer Architecture

## Language Model: Decoder
Casusal LM (Next Token Prediction): includes input layer transferring token to embedding, output layer with softmax activation, and feature layers including multi-head self-attention and feed forward network.

## Preflix LM
Preflix LM: preflix token with self-attention.

## Trainable Params
For $L$ feature layers, the traianble params of MHA include the weight matrics of query, key, value, and output with their corresponding bias:
```math
Params(MHA)=4\times(hid\_dim\times hid\_dim)+4\times hid\_dim
```
the trainable parmas of FFN is:
```math
Params(FFN)=2\times(hid\_dim\times4hid\_dim)+(4+1)hid\_dim
```
the layer normalization include:
```math
Params(LN)=2\times hid\_dim
```
and the input and output layer include trainable params:
```math
Params(I/O)=voc\_size\times hid\_dim
```
Then, the total params of $l$ layers of casual LM is:
```math
\begin{align}
params
&=l(Params(MHA)+Params(FFN)+2Params(LN))+Params(I/O)\\
&=12\times l\times hid\_dim^2+13\times l\times hid\_dim+voc\_size\times hid\_dim\\
&\approx 12\times l\times hid\_dim^2\end{align}
```
