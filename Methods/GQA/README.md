# GQA
[GQA:Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245)

Group-Query Attention: a generalization of multi-query attention, using an intermediate (more than one, less than number of query heads) number of key-value heads.

## Uptraining Multi-Query Attention
Generating multi-query model from multi-head consists two steps:
1. convert multi-head checkpoint into multi-query checkpoint, with the projection matrices for key and value heads as mean pooling,
2. additional pre-training for a small proportion $\alpha$ of its original steps on the same pre-training recipe, to adpat new structure.

## Grouped-Query Attention
Group-Query Attention: divides query heads into $G$ groups, sharing a single key head and value head for each group of query heads, with GQA-G referring $G$ groups, GQA-1 as MQA, and GQA-H with groups equal to number of heads as MHA. 

The propertirs of Larger Model:
1. generally scale the number of heads,
2. suffer relatively less from memoty bandwidth overhead from attention, as KV-cache scales with model dimension while model FLOPs and parameters scale with the square of model dimension,
3. encoder representaitons are computed in parallel, and memoty bandwidth is generally not the primary bottleneck.

According to the properties of larger model, GQA is designed as:
1. compared with MQA, keep the same proportional decrease in bandwidth and capacity as model size increases,
2. remove standard sharding for large models, replicating the single key and value head by the number of model partitions,
3. not apply to the encoder self-attention layers.
