# BERT
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)

BERT: pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers, can be fine-tuned with just one additional output layer for downstream tasks without substantial task-specific architecture modifications.

## Model Architecture
BERT's model architecture: multi-layer bidirectional Transformer encoder based on the original implementation, with the number of layers as $L$, the hidden size as $H$, and the number of self-attention heads as $A$. The base size is $L=12, H=768, A=12$ as 110M parmas, and large size is $L=24, H=1024, A=16$, as total params 340M.

Input/Output Representations: unambiguously represent both a single sentence and a pair of sentences in one token sequence, WordPiece embedding with 30000 token vocabulary is used:
1. \[CLS\]: classification token, the first token of every sequence;
2. \[SEP\]: separator token, to differentiate the sentences in pair.

The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings. Denote input embedding as $E$, the final hidden vector of \[CLS\] token as $C\in R^{H}$, and the final hidden vector for the $i^{th}$ input token as $T_i\in R^{H}$.

## Pre-Training BERT
Pre-Training: the model is trained on unlabeled data over different pre-training tasks.

Masked LM (MLM): simply mask 15\% of all WordPiece tokens in each sequence at random, and then predict those masked tokens rather than reconstructing the entire input. To mitigate the \[MASK\] token not appearing during fine-tuning, 80\% tokens are replaced as \[MASK\], 10\% tokens are randomly replaced, and 10\% tokens are unchanged. $T_i$ will be used to predict the original token with cross entropy loss.

Next Sentence Prediction (NSP): to understand the relationship between sentences, a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Sentences A and B from corpus are inputs, classified label $IsNext$ or $NotNext$ is output.

## Fine-Tuning BERT
Fine-Tuning: the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned end-to-end with labeled data from the downstream tasks. In output layer, token representations are fed for token-level tasks, and $[CLS]$ representation is fed for classification tasks.
