# Notes for beginners: tokenizer, tokens, IDs, embeddings, adapters, model type

## 1) Is GPT-2 encoder/decoder or just decoder?
GPT-2 is a **decoder-only Transformer** (a causal language model).
- It uses **causal self-attention**: token *t* can only attend to tokens ≤ *t*.
- There is no encoder stack (unlike T5/BART).

In Transformers terms, GPT-2 is `AutoModelForCausalLM` / `GPT2LMHeadModel`.

## 2) Tokenizer vs tokens vs token IDs
**Tokenizer** converts raw text into a sequence of integers.

For GPT-2:
- Tokenizer uses **byte-level BPE** (subword units).
- Output is `input_ids`: integers in range `[0, vocab_size-1]`.
- GPT-2 vocab_size is 50257.

Example:
- Text: "hello world"
- Token IDs: [31373, 995]  (example only; exact IDs depend on tokenizer)

Tokenizer's job ends here. It does NOT "make embeddings" by itself.

## 3) Token IDs -> embeddings (inside the model)
Once you have token IDs, the model maps them to vectors:

- Token embedding table: shape `(V, C)`
  - V = vocab size (50257)
  - C = embedding dim / hidden size (768 for GPT-2 124M)

Given `input_ids` of shape `(B, T)`:
- embedding lookup gives `(B, T, C)`.

This happens inside the model forward pass.

## 4) "Embeddings" can mean two different things
### A) Internal embeddings (LLM embeddings)
- The vectors produced by the embedding table and hidden states in the Transformer.
- Used for generation; not stored in a database by default.

### B) Retrieval embeddings (RAG)
- Separate embedding model generates vectors for documents.
- Vectors stored in a vector DB (FAISS/OpenSearch/Pinecone/etc.).
- Not required for GPT-2 training or serving.
- Only needed if you want retrieval-augmented generation.

This project focuses on **LM training/finetuning**, not RAG.

## 5) What does the Transformer do during inference?
Given input IDs `(B, T)`:
1) embeddings -> `(B, T, C)`
2) repeated Transformer blocks (attention + MLP) keep shape `(B, T, C)`
3) final linear projection to vocab logits -> `(B, T, V)`
4) sampling picks next token id from `(B, V)` at the last position
5) append token and repeat

Shapes:
- IDs: `(B, T)` (int64)
- hidden: `(B, T, C)` (float)
- logits: `(B, T, V)` (float)

## 6) Where PyTorch fits
PyTorch is the **tensor engine + autograd**:
- Tensors (CPU memory)
- matrix multiplications / attention / MLP compute
- gradient computation (backprop)
- optimizer updates (AdamW)
- saving/loading checkpoints

## 7) Where Hugging Face Transformers library fits
Transformers (the library) provides:
- GPT-2 model implementation with standard naming (`GPT2LMHeadModel`)
- tokenizer (`AutoTokenizer`)
- weight loading for pretrained GPT-2 (`from_pretrained("gpt2")`)
- config objects, helpers

In this CPU project we use Transformers for correctness and compatibility.

You do NOT have to use Transformers to *understand* Transformers.
But for production and pretrained weight loading, it’s the practical path.

## 8) What are LoRA adapters and what part of training do they change?
During fine-tuning, instead of updating all model weights, LoRA:
- freezes base weights
- injects low-rank trainable matrices into selected Linear layers
- only trains those small matrices

Why it matters:
- far fewer trainable params
- faster/cheaper training
- easy to keep many domain-specific adapters

In this project:
- Full fine-tune: updates all weights (slowest on CPU)
- LoRA: updates adapters only (still slow on CPU, but less memory and easier iteration)

## 9) QLoRA (why we don't run it on CPU)
QLoRA combines:
- 4-bit quantized base model + LoRA
- designed to reduce GPU VRAM

It relies on quantization kernels (typically CUDA via bitsandbytes).
On CPU clusters, it’s generally not supported or not worth the complexity. citeturn0search3turn0search7
