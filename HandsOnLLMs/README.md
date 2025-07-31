# HandsOnLLMs

## Understanding Language Models
Large Language Models:
1. Commonly Useed Models: gpt, llama, Qwen, DeepSeek.
2. LLM Tasks: natural language processing, code-related generation, creative writing.
3. LLM Components: model, tokenizer, prompt template, pipeline (generate, response).

Tokens and Embeddings:
1. Tokens: splited subwords based on corresponding format (BPE), and number of tokens standing for the number of training set.
2. Embedding: sparse one-hot vector, or dense high-dimensional vector, including original word2vec embedding, contexual embedding model (EIMO, BERT), and text vector specially for sentence or document.

Transformer LLMs Architecture
1. Transformer LLMs: input prompt template of chatLM for model representation, output api includes 'generate' (synthetic results), 'model' (hidden output from Transformer decoder), and 'lm_head' (linear layer output with vocab_size dimension).
2. RMSNorm and LayerNorm:
3. KV Cache:


## Using Pretrained Language Models
LLM Text Classification
1. Text Classification Model: representative model (output numeric label), generative model (output text label).
2. Representative Model: map text into vector, and map into label space, with small but fixed class space (sentiment analysis, news classification). Use specific model and private data to train representative model, including supervised learning BERT with logic regression, and transference from classification to matching task via zero-shot.
3. Generative Model: chat format via calling api, or using decoder or encoder-decoder model.

Text Clustering and Topic Models
1. Task: understand the regularity and interpretability of categorical data, and filter outlier and process data unbalance (over-representation of entertainment knowledge, under-representation of knowledge data).
2. Text Clustering: group similar text, compute text embedding (representation) and similarity via KMeans and DBScan, use PCA, UMAP to reduce representation dimension, cluster, and visualization.
3. Topic Model: identify topic distribution of documents (extract keywords) via representation model (LDA, NMF, BERTopic), generative (T5, LLaMA) or OpenAI-api.

Prompt Engineering
1. Prompt Engineering Cores: iterated establishment, evaluation metrics, clear prompt template.
2. [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview): prompt generator (initiate prompt via other LLMs), clear and direct prompt, multi-shot examples, chain of thought, xml tags, system prompts, prefill response (api), chain complex prompts (task-prompt steps), long context tips.
3. Examples:
   ```<text>
   Chain of Thought: <think> xxx(generation), step1, step2, \ldots, </think> + prompt
   Long Context Tips: <doc> xxxxxxx </doc> + instructions/prompts/system
   ```

Agents
1. Agent: LLMs enabling thinking, planing, and tool calling, including prompt engineering (Manus, enhance performance via agent structure), and reinforcement learning (DeepResearch, end-to-end optimization).
2. Agent Workflow: user-request-LLMs, LLMs-(judge)-tools/functions, tools-action(procession)-environment, tools-results-LLMs, LLMs-(judge)-end/recursion-results-user.
3. ReAct (Reason+Act): include think, tool, observation, and results (combining former three components)
4. Function/Tool Call: processing tools, including tools definition and information input.
5. MCP Protocol: use Clients to connect to the server, and format the tools available in the server.
6. Agent Formats: LangChain (inference chain, prompt template, structual otuput), LangGraph, DeepResearch (multi-agents?).

## Training and Fine-Tuning Language Models
LLMs Supervised Fine-Tuning
1. SFT: supervised fine-tuning via prompt (larger prediction token space), traditional fine-tune with domain data of specific task (small prediction space).
2. PEFT: parameter-efficient fine-tuning, save GPU memory, accelerate training, maintain more original information, including LoRA, prefix tuning, adapter tuning, prompt tuning.
3. Data Preparation: process private data with the same LLM input format of public data and train with mixed data.



## Reference
[Hands-On-Large-Language-Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models)  
[Hands-On-Large-Language-Models-CN](https://github.com/bbruceyuan/Hands-On-Large-Language-Models-CN)
