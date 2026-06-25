# 🧠 NanoGPT — Deep Dive: Architecture, Scaling, Dataset, Deployment & API Guide

A comprehensive technical reference for understanding, scaling, and deploying nanoGPT.

---

## 📖 Table of Contents

1. [Complete Project Structure](#1--complete-project-structure)
2. [LLM Architecture — How GPT Works Inside](#2--llm-architecture--how-gpt-works-inside)
3. [Parameter Calculation — The Math Behind Billions](#3--parameter-calculation--the-math-behind-billions)
4. [Scaling Guide — How to Increase Parameters](#4--scaling-guide--how-to-increase-parameters)
5. [Dataset — What It Trains On & What It Can Answer](#5--dataset--what-it-trains-on--what-it-can-answer)
6. [Running Locally — Complete Guide](#6--running-locally--complete-guide)
7. [Deployment with FastAPI — Serve as an API](#7--deployment-with-fastapi--serve-as-an-api)
8. [Frontend Integration](#8--frontend-integration)

---

## 1. 📁 Complete Project Structure

```
nanoGPT/
│
├── model.py                              # 🧠 THE BRAIN — Complete GPT model definition
│   ├── class GPTConfig                   #    Configuration dataclass (n_layer, n_head, n_embd, etc.)
│   ├── class LayerNorm                   #    Custom LayerNorm (optional bias)
│   ├── class CausalSelfAttention         #    Multi-Head Self-Attention with causal mask
│   ├── class MLP                         #    Feed-Forward Network (n_embd → 4×n_embd → n_embd)
│   ├── class Block                       #    Single Transformer Block (Attention + MLP)
│   └── class GPT                         #    Full GPT Model (Embedding + N×Block + LM Head)
│       ├── __init__()                    #      Build model, init weights
│       ├── get_num_params()              #      Count parameters
│       ├── forward()                     #      Forward pass (input → logits + loss)
│       ├── from_pretrained()             #      Load GPT-2 weights from HuggingFace
│       ├── configure_optimizers()        #      Setup AdamW with weight decay
│       ├── estimate_mfu()               #      Estimate Model FLOPs Utilization
│       ├── generate()                    #      Auto-regressive text generation
│       └── crop_block_size()             #      Reduce context window size
│
├── train.py                              # 🏋️ THE TRAINER — Training loop
│   ├── Config defaults (lines 32-74)     #    All hyperparameters with defaults
│   ├── Data loading (lines 114-131)      #    Memory-mapped binary file reader
│   ├── Model init (lines 146-193)        #    Create model (scratch/resume/pretrained)
│   ├── Optimizer setup (lines 196-202)   #    AdamW with gradient scaling
│   ├── Training loop (lines 249-333)     #    The main while loop
│   │   ├── Learning rate scheduling      #      Cosine decay with warmup
│   │   ├── Evaluation & checkpointing    #      Periodic val loss check + save
│   │   ├── Forward/backward pass         #      With gradient accumulation
│   │   └── Logging                       #      Loss, timing, MFU metrics
│   └── DDP support                       #    Multi-GPU distributed training
│
├── sample.py                             # 💬 THE GENERATOR — Text generation / inference
│   ├── Load checkpoint or pretrained     #    Resume from ckpt.pt or load GPT-2
│   ├── Tokenizer setup                   #    Character-level (meta.pkl) or BPE (tiktoken)
│   ├── Prompt encoding                   #    Text or FILE:prompt.txt
│   └── Generation loop                   #    Auto-regressive sampling with temperature
│
├── configurator.py                       # ⚙️ Config override utility
│   └── Parses CLI args like --key=value  #    Overrides global variables in train.py/sample.py
│
├── bench.py                              # 📊 Benchmarking script
│
├── config/                               # 📋 Pre-made configuration files
│   ├── train_shakespeare_char.py         #    Train character-level Shakespeare (10.7M params)
│   ├── train_gpt2.py                     #    Reproduce GPT-2 124M on OpenWebText
│   ├── finetune_shakespeare.py           #    Finetune GPT-2 on Shakespeare
│   ├── eval_gpt2.py                      #    Evaluate GPT-2 base (124M)
│   ├── eval_gpt2_medium.py               #    Evaluate GPT-2 medium (350M)
│   ├── eval_gpt2_large.py                #    Evaluate GPT-2 large (774M)
│   └── eval_gpt2_xl.py                   #    Evaluate GPT-2 XL (1.56B)
│
├── data/                                 # 📚 Datasets
│   ├── shakespeare_char/                 #    Character-level Shakespeare
│   │   ├── prepare.py                    #      Downloads & tokenizes text
│   │   ├── input.txt                     #      Raw Shakespeare text (~1MB)
│   │   ├── train.bin                     #      Binary training tokens
│   │   ├── val.bin                       #      Binary validation tokens
│   │   ├── meta.pkl                      #      Char ↔ Int mapping (vocab=65)
│   │   └── readme.md                     #      Dataset info
│   ├── shakespeare/                      #    BPE-tokenized Shakespeare (for finetuning)
│   │   └── prepare.py                    #      Uses GPT-2 BPE tokenizer
│   └── openwebtext/                      #    OpenWebText (for GPT-2 reproduction)
│       └── prepare.py                    #      Downloads from HuggingFace
│
├── out-shakespeare-char/                 # 💾 Training output (created after training)
│   └── ckpt.pt                           #    Saved model checkpoint (~129 MB)
│
├── .venv/                                # 🐍 Python virtual environment
├── assets/                               # 🖼️ Images for original README
├── LICENSE                               # MIT License
└── README.md                             # Original Karpathy README
```

---

## 2. 🏗 LLM Architecture — How GPT Works Inside

### High-Level Overview

GPT (Generative Pre-trained Transformer) is a **decoder-only transformer** that predicts the next token in a sequence. It learns patterns from text data and generates new text by repeatedly predicting "what comes next."

### Architecture Diagram

```
                        ┌─────────────────────────────────────────────┐
                        │            INPUT TEXT                        │
                        │        "To be or not to"                    │
                        └──────────────┬──────────────────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────────────────────┐
                        │          TOKENIZATION                       │
                        │  "T" "o" " " "b" "e" " " "o" "r" ...      │
                        │   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓          │
                        │  [30, 51, 1, 40, 47, 1, 51, 54, ...]      │
                        └──────────────┬──────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  │
        ┌───────────────────┐ ┌───────────────────┐      │
        │  Token Embedding  │ │Position Embedding │      │
        │  (wte)            │ │(wpe)              │      │
        │                   │ │                   │      │
        │  vocab_size ×     │ │ block_size ×      │      │
        │  n_embd           │ │ n_embd            │      │
        │  (65 × 384)       │ │ (256 × 384)       │      │
        └────────┬──────────┘ └────────┬──────────┘      │
                 │                     │                  │
                 └──────────┬──────────┘                  │
                            │ (add)                       │
                            ▼                             │
                ┌───────────────────────┐                 │
                │     Dropout           │                 │
                └───────────┬───────────┘                 │
                            │                             │
        ╔═══════════════════╧═══════════════════╗         │
        ║     TRANSFORMER BLOCK × n_layer       ║         │
        ║     (repeated 6 times for 10.7M)      ║         │
        ║                                       ║         │
        ║  ┌─────────────────────────────────┐  ║         │
        ║  │         Layer Norm 1            │  ║         │
        ║  └──────────────┬──────────────────┘  ║         │
        ║                 │                     ║         │
        ║  ┌──────────────▼──────────────────┐  ║         │
        ║  │  MULTI-HEAD SELF-ATTENTION      │  ║         │
        ║  │                                 │  ║         │
        ║  │  Input (n_embd=384)             │  ║         │
        ║  │       │                         │  ║         │
        ║  │  ┌────▼────┐                    │  ║         │
        ║  │  │  c_attn │ Linear             │  ║         │
        ║  │  │(384→1152)│ (3 × n_embd)      │  ║         │
        ║  │  └────┬────┘                    │  ║         │
        ║  │       │ split into Q, K, V      │  ║         │
        ║  │  ┌────┼────┐                    │  ║         │
        ║  │  Q    K    V  (each 384-dim)    │  ║         │
        ║  │  │    │    │                    │  ║         │
        ║  │  │ reshape into n_head heads    │  ║         │
        ║  │  │ (6 heads × 64-dim each)      │  ║         │
        ║  │  │    │    │                    │  ║         │
        ║  │  └────┼────┘                    │  ║         │
        ║  │       │                         │  ║         │
        ║  │  Attention = softmax(QK^T/√d)V  │  ║         │
        ║  │  (with causal mask — can only   │  ║         │
        ║  │   attend to previous tokens)    │  ║         │
        ║  │       │                         │  ║         │
        ║  │  ┌────▼────┐                    │  ║         │
        ║  │  │  c_proj │ Linear             │  ║         │
        ║  │  │(384→384) │ Output projection  │  ║         │
        ║  │  └────┬────┘                    │  ║         │
        ║  │       │                         │  ║         │
        ║  └───────┼─────────────────────────┘  ║         │
        ║          │                            ║         │
        ║     ┌────▼────┐                       ║         │
        ║     │ + (add) │ ← Residual Connection ║         │
        ║     └────┬────┘                       ║         │
        ║          │                            ║         │
        ║  ┌───────▼─────────────────────────┐  ║         │
        ║  │         Layer Norm 2            │  ║         │
        ║  └──────────────┬──────────────────┘  ║         │
        ║                 │                     ║         │
        ║  ┌──────────────▼──────────────────┐  ║         │
        ║  │       MLP (Feed-Forward)        │  ║         │
        ║  │                                 │  ║         │
        ║  │  Linear: n_embd → 4×n_embd      │  ║         │
        ║  │  (384 → 1536)                   │  ║         │
        ║  │         │                       │  ║         │
        ║  │      GELU activation            │  ║         │
        ║  │         │                       │  ║         │
        ║  │  Linear: 4×n_embd → n_embd      │  ║         │
        ║  │  (1536 → 384)                   │  ║         │
        ║  │         │                       │  ║         │
        ║  │      Dropout                    │  ║         │
        ║  └─────────┬───────────────────────┘  ║         │
        ║            │                          ║         │
        ║     ┌──────▼──┐                       ║         │
        ║     │ + (add) │ ← Residual Connection ║         │
        ║     └──────┬──┘                       ║         │
        ║            │                          ║         │
        ╚════════════╧══════════════════════════╝         │
                     │                                    │
                     ▼                                    │
        ┌─────────────────────────┐                       │
        │     Final Layer Norm    │                       │
        │     (ln_f)              │                       │
        └────────────┬────────────┘                       │
                     │                                    │
                     ▼                                    │
        ┌─────────────────────────┐                       │
        │   Language Model Head   │                       │
        │   (lm_head)             │                       │
        │                         │                       │
        │   Linear: n_embd →      │                       │
        │   vocab_size             │                       │
        │   (384 → 65)            │  ← Weight Tying ──────┘
        └────────────┬────────────┘    (shares weights
                     │                  with wte)
                     ▼
        ┌─────────────────────────┐
        │   Softmax → Sample      │
        │                         │
        │   [0.01, 0.02, ...,     │
        │    0.15, ..., 0.003]    │
        │   probability over      │
        │   65 characters         │
        │                         │
        │   Selected: "b" (40)    │
        └─────────────────────────┘
                     │
                     ▼
              OUTPUT: "b"
         (next predicted character)
```

### Key Concepts Explained

#### 1. Tokenization
- **Character-level** (Shakespeare): Each character = 1 token. Vocab size = 65.
- **BPE/Subword** (GPT-2): Words split into subwords. Vocab size = 50,257.

#### 2. Embeddings
- **Token Embedding (wte):** Maps each token to a dense vector of size `n_embd`.
- **Position Embedding (wpe):** Tells the model WHERE each token is in the sequence.
- Both are added together → the model knows WHAT token is at WHICH position.

#### 3. Self-Attention ("The Magic")
```
"The cat sat on the mat"
       ↓
Each word asks: "Which other words should I pay attention to?"
       ↓
Q (Query) = "What am I looking for?"
K (Key)   = "What do I contain?"
V (Value) = "What information do I provide?"
       ↓
Attention Score = softmax(Q × K^T / √d) × V
       ↓
"mat" pays high attention to "sat", "cat" → understands context
```

**Causal Mask:** Token at position `t` can ONLY attend to tokens at positions `0, 1, ..., t`. It cannot look at future tokens. This is what makes GPT a **decoder** — it generates left to right.

#### 4. Multi-Head Attention
Instead of one attention, we run `n_head` parallel attention computations (6 heads for our model). Each head has dimension `n_embd / n_head = 384/6 = 64`. Different heads learn different types of relationships (syntax, semantics, etc.).

#### 5. MLP (Feed-Forward Network)
After attention, each token passes through:
```
Input (384) → Linear (384 → 1536) → GELU → Linear (1536 → 384) → Output (384)
```
This is where the model "thinks" — it transforms the attention output into richer representations.

#### 6. Residual Connections
The `+` operations add the input back to the output of each sub-layer. This helps with:
- Gradient flow during training (prevents vanishing gradients)
- Allowing the model to learn "do nothing" when appropriate

#### 7. Weight Tying
The token embedding matrix (`wte`) and the language model head (`lm_head`) share the same weights. This reduces parameters and improves performance.

---

## 3. 🔢 Parameter Calculation — The Math Behind Billions

### Formula

For a GPT model with the configuration `(V=vocab_size, D=n_embd, L=n_layer, T=block_size)`:

```
Parameters ≈ V×D + T×D + L × (12×D² + 13×D) + 2×D + V×D

Where:
├── V×D ............. Token Embedding (shared with lm_head via weight tying)
├── T×D ............. Position Embedding
├── L × ............. Per Transformer Block:
│   ├── 4×D² + D .... c_attn (Q,K,V projection: D → 3D)
│   ├── D² + D ...... c_proj (attention output projection: D → D)
│   ├── 4×D² + 4×D .. c_fc (MLP up-projection: D → 4D)
│   ├── 4×D² + D .... c_proj (MLP down-projection: 4D → D)
│   ├── 2×D ......... LayerNorm 1 (weight + bias)
│   └── 2×D ......... LayerNorm 2 (weight + bias)
├── 2×D ............. Final LayerNorm (ln_f)
└── (V×D) ........... lm_head (SHARED with token embedding — not counted)
```

### Simplified Formula

```
Total ≈ V×D + T×D + L × (12×D² + 13×D) + 2×D
```

### Actual Parameter Counts for Known Configurations

| Model | n_layer (L) | n_head (H) | n_embd (D) | block_size (T) | vocab_size (V) | Total Params |
|---|---|---|---|---|---|---|
| Shakespeare Char (mini) | 4 | 4 | 128 | 64 | 65 | **~0.80M** |
| Shakespeare Char (full) | 6 | 6 | 384 | 256 | 65 | **~10.65M** |
| GPT-2 Base | 12 | 12 | 768 | 1024 | 50,257 | **~124M** |
| GPT-2 Medium | 24 | 16 | 1024 | 1024 | 50,257 | **~350M** |
| GPT-2 Large | 36 | 20 | 1280 | 1024 | 50,257 | **~774M** |
| GPT-2 XL | 48 | 25 | 1600 | 1024 | 50,257 | **~1,558M** |

---

## 4. 🚀 Scaling Guide — How to Increase Parameters

### The 3 Knobs to Scale

To increase model parameters, you modify these 3 values in `GPTConfig` (in `model.py` line 108-116):

```python
@dataclass
class GPTConfig:
    block_size: int = 1024    # Context window (how many tokens the model can "see")
    vocab_size: int = 50304   # Number of unique tokens
    n_layer: int = 12         # 🔧 KNOB 1: Number of transformer blocks (depth)
    n_head: int = 12          # 🔧 KNOB 2: Number of attention heads (width of attention)
    n_embd: int = 768         # 🔧 KNOB 3: Embedding dimension (width of model)
    dropout: float = 0.0
    bias: bool = True
```

### Scaling Rules (Important Constraints!)

```
RULE 1: n_embd MUST be divisible by n_head
        (because each head gets n_embd / n_head dimensions)

RULE 2: n_head should be a power of 2 or a clean divisor of n_embd
        (for GPU memory alignment efficiency)

RULE 3: vocab_size should be a multiple of 64
        (for GPU kernel efficiency)
```

### Scaling Recipes

#### Recipe 1: From 10.7M → 50M Parameters
```bash
# Create a new config file: config/train_50M.py
python train.py \
    --device=mps --compile=False \
    --n_layer=8 --n_head=8 --n_embd=512 \
    --block_size=512 --batch_size=32 \
    --max_iters=10000 --lr_decay_iters=10000
```
**Resources:** ~1 GB RAM, ~1 hour on Mac M4

#### Recipe 2: From 10.7M → 124M Parameters (GPT-2 Base size)
```bash
python train.py \
    --device=mps --compile=False \
    --n_layer=12 --n_head=12 --n_embd=768 \
    --block_size=1024 --batch_size=8 \
    --max_iters=50000 --lr_decay_iters=50000 \
    --learning_rate=6e-4
```
**Resources:** ~4 GB RAM, ~12-24 hours on Mac M4

#### Recipe 3: Custom Scaling Table

| Target Params | n_layer | n_head | n_embd | block_size | RAM Needed | Training Time (M4) |
|---|---|---|---|---|---|---|
| ~10M | 6 | 6 | 384 | 256 | 500 MB | ~30 min |
| ~25M | 8 | 8 | 512 | 256 | 800 MB | ~1 hr |
| ~50M | 8 | 8 | 512 | 512 | 1.5 GB | ~2 hrs |
| ~85M | 10 | 10 | 640 | 512 | 2 GB | ~4 hrs |
| ~124M | 12 | 12 | 768 | 1024 | 4 GB | ~12-24 hrs |
| ~350M | 24 | 16 | 1024 | 1024 | 10 GB | Not feasible on M4 |
| ~774M | 36 | 20 | 1280 | 1024 | 20+ GB | Not feasible on M4 |
| ~1.5B | 48 | 25 | 1600 | 1024 | 40+ GB | Not feasible on M4 |

### How to Create a Custom Config File

Create `config/train_custom_50M.py`:

```python
# 50M parameter model trained on Shakespeare
out_dir = 'out-shakespeare-50M'
eval_interval = 500
eval_iters = 200
log_interval = 10

always_save_checkpoint = False
wandb_log = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 4  # simulate larger batch
batch_size = 32
block_size = 512

# 50M parameter model
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1

learning_rate = 5e-4
max_iters = 10000
lr_decay_iters = 10000
min_lr = 5e-5
beta2 = 0.99
warmup_iters = 200

# Mac M4 settings
device = 'mps'
compile = False
```

Then run:
```bash
python train.py config/train_custom_50M.py
```

### The Pattern for Adding Scale

```
┌──────────────────────────────────────────────────────────────────┐
│                    SCALING PATTERN                                │
│                                                                  │
│   Small Model (10M)          Large Model (1B+)                   │
│   ┌──────────────┐           ┌──────────────────┐                │
│   │ n_layer = 6  │    →      │ n_layer = 48     │  More depth    │
│   │ n_head = 6   │    →      │ n_head = 25      │  More heads    │
│   │ n_embd = 384 │    →      │ n_embd = 1600    │  Wider model   │
│   │ block = 256  │    →      │ block = 2048+    │  More context  │
│   │ batch = 64   │    →      │ batch = 480      │  More data/step│
│   │ data = 1MB   │    →      │ data = 100GB+    │  More data     │
│   │ iters = 5K   │    →      │ iters = 600K     │  More training │
│   └──────────────┘           └──────────────────┘                │
│                                                                  │
│   KEY INSIGHT: Parameters scale roughly as 12 × n_layer × n_embd² │
│   Doubling n_embd → 4× more parameters                          │
│   Doubling n_layer → 2× more parameters                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. 📚 Dataset — What It Trains On & What It Can Answer

### Current Dataset: Tiny Shakespeare

| Property | Value |
|---|---|
| **Source** | Complete works of William Shakespeare |
| **Size** | ~1 MB (1,115,394 characters) |
| **Tokenization** | Character-level (each character = 1 token) |
| **Vocabulary** | 65 unique characters: `!$&',-.3:;?A-Za-z` + space + newline |
| **Train tokens** | 1,003,854 |
| **Validation tokens** | 111,540 |
| **Split ratio** | 90% train / 10% validation |

### What Can This Model Answer?

> **⚠️ Important: This is a TEXT COMPLETION model, NOT a chatbot.**

This model is trained to **complete text** in the style of Shakespeare. It does NOT understand questions or follow instructions.

| What it CAN do ✅ | What it CANNOT do ❌ |
|---|---|
| Generate Shakespeare-style dialogue | Answer factual questions |
| Continue a scene given a character name | Have a conversation with you |
| Produce poetic/dramatic text | Understand or follow instructions |
| Mimic character speech patterns | Reason about math or logic |
| Create new scenes with known characters | Remember previous interactions |

### Example Interactions

```bash
# ✅ WORKS: Complete Shakespeare-style text
python sample.py --out_dir=out-shakespeare-char --device=mps --start="ROMEO:"
# Output: "ROMEO: O, she doth teach the torches to burn bright..."

# ✅ WORKS: Continue a scene
python sample.py --out_dir=out-shakespeare-char --device=mps --start="HAMLET:\nTo be, or not to be"
# Output: Continues in Hamlet's voice

# ❌ DOES NOT WORK: Asking questions
python sample.py --out_dir=out-shakespeare-char --device=mps --start="What is 2+2?"
# Output: Random Shakespeare-like gibberish (model has no concept of math)
```

### Other Available Datasets

| Dataset | Prepare Command | Type | Use Case |
|---|---|---|---|
| Shakespeare (char) | `python data/shakespeare_char/prepare.py` | Character-level | Quick experiments |
| Shakespeare (BPE) | `python data/shakespeare/prepare.py` | Subword (GPT-2 BPE) | Finetuning GPT-2 |
| OpenWebText | `python data/openwebtext/prepare.py` | Subword (GPT-2 BPE) | Reproducing GPT-2 (needs ~50GB disk) |

### Training on Your Own Custom Dataset

You can train on ANY text data. Create a new dataset:

```bash
mkdir data/my_custom_data
```

Create `data/my_custom_data/prepare.py`:

```python
import os
import numpy as np
import tiktoken

# Load your text data
input_file = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_file, 'r') as f:
    data = f.read()

# Use GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(data)

# 90/10 train/val split
n = len(train_ids)
train_data = np.array(train_ids[:int(n*0.9)], dtype=np.uint16)
val_data = np.array(train_ids[int(n*0.9):], dtype=np.uint16)

train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_data.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
print(f"train has {len(train_data)} tokens, val has {len(val_data)} tokens")
```

Place your text in `data/my_custom_data/input.txt`, then:

```bash
python data/my_custom_data/prepare.py
python train.py --dataset=my_custom_data --device=mps --compile=False
```

---

## 6. 💻 Running Locally — Complete Guide

### First-Time Setup

```bash
# Clone and enter the project
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Run this every time you open a new terminal

# Install dependencies
pip install torch numpy transformers datasets tiktoken wandb tqdm

# Prepare dataset
python data/shakespeare_char/prepare.py
```

### Train (One-Time)

```bash
# Activate venv (if not already active)
source .venv/bin/activate

# Train on Mac M4 with MPS
python train.py config/train_shakespeare_char.py --device=mps --compile=False
```

### Generate Text (Anytime After Training)

```bash
# Activate venv
source .venv/bin/activate

# Basic generation
python sample.py --out_dir=out-shakespeare-char --device=mps

# With custom prompt
python sample.py --out_dir=out-shakespeare-char --device=mps --start="KING HENRY:"

# With more control
python sample.py --out_dir=out-shakespeare-char --device=mps \
    --start="JULIET:" --num_samples=5 --max_new_tokens=500 \
    --temperature=0.7 --top_k=100
```

---

## 7. 🌐 Deployment with FastAPI — Serve as an API

### Project Structure for Deployment

```
nanoGPT/
├── ... (existing files)
├── api/
│   ├── server.py              # FastAPI application
│   ├── auth.py                # API key authentication
│   └── requirements.txt       # Additional dependencies
└── out-shakespeare-char/
    └── ckpt.pt                # Trained model checkpoint
```

### Step 1: Install FastAPI

```bash
source .venv/bin/activate
pip install fastapi uvicorn python-dotenv
```

### Step 2: Create the API Server

Create `api/server.py`:

```python
"""
NanoGPT FastAPI Server
Serves the trained Shakespeare model as a REST API.
"""

import os
import sys
import pickle
from contextlib import nullcontext
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Add parent directory to path so we can import model.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import GPTConfig, GPT

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_DIR = os.environ.get("MODEL_DIR", "out-shakespeare-char")
DEVICE = os.environ.get("DEVICE", "mps")  # "mps" for Mac, "cpu" for server
REQUIRE_API_KEY = os.environ.get("REQUIRE_API_KEY", "false").lower() == "true"
API_KEYS = set(os.environ.get("API_KEYS", "").split(",")) - {""}

# ============================================================
# LOAD MODEL (happens once at startup)
# ============================================================
print(f"Loading model from {MODEL_DIR}...")
ckpt_path = os.path.join(os.path.dirname(__file__), '..', MODEL_DIR, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=DEVICE)

gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)

# Load tokenizer
meta_path = os.path.join(
    os.path.dirname(__file__), '..', 'data',
    checkpoint['config']['dataset'], 'meta.pkl'
)
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    print(f"Loaded character-level tokenizer (vocab_size={len(stoi)})")
else:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    print("Using GPT-2 BPE tokenizer")

print(f"Model loaded! Parameters: {gptconf.n_layer}L/{gptconf.n_head}H/{gptconf.n_embd}D")
print(f"Device: {DEVICE}, API Key Required: {REQUIRE_API_KEY}")

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="NanoGPT Shakespeare API",
    description="Generate Shakespeare-style text using a trained GPT model.",
    version="1.0.0",
)

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# AUTHENTICATION (Optional)
# ============================================================
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """Verify API key if authentication is enabled."""
    if not REQUIRE_API_KEY:
        return True  # No auth required
    if api_key is None or api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Pass it in the X-API-Key header.",
        )
    return True

# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================
class GenerateRequest(BaseModel):
    prompt: str = "\n"
    max_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 200
    num_samples: int = 1

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "ROMEO:",
                "max_tokens": 300,
                "temperature": 0.8,
                "top_k": 200,
                "num_samples": 1
            }
        }

class GenerateResponse(BaseModel):
    prompt: str
    generated_texts: list[str]
    model_params: str
    device: str

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model": f"NanoGPT Shakespeare ({gptconf.n_layer}L/{gptconf.n_head}H/{gptconf.n_embd}D)",
        "parameters": f"{model.get_num_params()/1e6:.2f}M",
        "device": DEVICE,
        "auth_required": REQUIRE_API_KEY,
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    authenticated: bool = Depends(verify_api_key),
):
    """Generate Shakespeare-style text from a prompt."""

    # Validate parameters
    if request.max_tokens < 1 or request.max_tokens > 2000:
        raise HTTPException(status_code=400, detail="max_tokens must be between 1 and 2000")
    if request.num_samples < 1 or request.num_samples > 10:
        raise HTTPException(status_code=400, detail="num_samples must be between 1 and 10")
    if request.temperature <= 0 or request.temperature > 2.0:
        raise HTTPException(status_code=400, detail="temperature must be between 0.01 and 2.0")

    try:
        # Encode prompt
        start_ids = encode(request.prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=DEVICE)[None, ...]

        # Generate
        generated_texts = []
        with torch.no_grad():
            ctx = nullcontext() if DEVICE == 'cpu' else torch.amp.autocast(
                device_type='cuda' if 'cuda' in DEVICE else 'cpu',
                dtype=torch.float16
            ) if 'cuda' in DEVICE else nullcontext()

            with ctx:
                for _ in range(request.num_samples):
                    y = model.generate(
                        x,
                        request.max_tokens,
                        temperature=request.temperature,
                        top_k=request.top_k,
                    )
                    text = decode(y[0].tolist())
                    generated_texts.append(text)

        return GenerateResponse(
            prompt=request.prompt,
            generated_texts=generated_texts,
            model_params=f"{model.get_num_params()/1e6:.2f}M",
            device=DEVICE,
        )
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Character not in vocabulary: {e}. This model uses character-level tokenization."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    """Get detailed model information."""
    return {
        "architecture": "GPT (Decoder-only Transformer)",
        "n_layer": gptconf.n_layer,
        "n_head": gptconf.n_head,
        "n_embd": gptconf.n_embd,
        "block_size": gptconf.block_size,
        "vocab_size": gptconf.vocab_size,
        "total_parameters": model.get_num_params(),
        "total_parameters_human": f"{model.get_num_params()/1e6:.2f}M",
        "dropout": gptconf.dropout,
        "bias": gptconf.bias,
        "training_dataset": checkpoint.get('config', {}).get('dataset', 'unknown'),
        "best_val_loss": checkpoint.get('best_val_loss', 'unknown'),
        "training_iterations": checkpoint.get('iter_num', 'unknown'),
    }
```

### Step 3: Create Environment Variables

Create `.env` in the project root:

```env
# Model settings
MODEL_DIR=out-shakespeare-char
DEVICE=mps

# Authentication (set to "true" to require API keys)
REQUIRE_API_KEY=false

# Comma-separated API keys (used when REQUIRE_API_KEY=true)
API_KEYS=sk-nano-abc123,sk-nano-xyz789,sk-nano-user001
```

### Step 4: Run the API Server

```bash
# Without authentication (open access)
source .venv/bin/activate
cd nanoGPT
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# With authentication (API key required)
REQUIRE_API_KEY=true uvicorn api.server:app --host 0.0.0.0 --port 8000
```

The server starts at: `http://localhost:8000`
Swagger docs at: `http://localhost:8000/docs`

### Step 5: Test the API

#### Without API Key (open access):

```bash
# Health check
curl http://localhost:8000/

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "ROMEO:",
    "max_tokens": 300,
    "temperature": 0.8,
    "num_samples": 2
  }'

# Get model info
curl http://localhost:8000/model-info
```

#### With API Key (secured):

```bash
# Generate text with API key
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-nano-abc123" \
  -d '{
    "prompt": "HAMLET:\nTo be, or not to be,",
    "max_tokens": 500,
    "temperature": 0.7,
    "num_samples": 1
  }'

# Without API key → 401 Unauthorized
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ROMEO:"}'
# Response: {"detail": "Invalid or missing API key..."}
```

### Step 6: API Response Format

```json
{
  "prompt": "ROMEO:",
  "generated_texts": [
    "ROMEO:\nO, she doth teach the torches to burn bright!\nIt seems she hangs upon the cheek of night..."
  ],
  "model_params": "10.65M",
  "device": "mps"
}
```

---

## 8. 🖥 Frontend Integration

### Using JavaScript (Fetch API)

```javascript
// Without API key
async function generateText(prompt) {
  const response = await fetch('http://localhost:8000/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: prompt,
      max_tokens: 300,
      temperature: 0.8,
      num_samples: 1
    })
  });
  const data = await response.json();
  return data.generated_texts[0];
}

// With API key
async function generateTextSecured(prompt, apiKey) {
  const response = await fetch('http://localhost:8000/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': apiKey    // Pass the secret key in the header
    },
    body: JSON.stringify({
      prompt: prompt,
      max_tokens: 300,
      temperature: 0.8,
      num_samples: 1
    })
  });

  if (response.status === 401) {
    throw new Error('Invalid API key');
  }

  const data = await response.json();
  return data.generated_texts[0];
}

// Usage
generateText("ROMEO:").then(text => console.log(text));
```

### Using Python (requests)

```python
import requests

# Without API key
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "ROMEO:",
    "max_tokens": 300,
    "temperature": 0.8,
})
print(response.json()["generated_texts"][0])

# With API key
response = requests.post(
    "http://localhost:8000/generate",
    json={"prompt": "HAMLET:", "max_tokens": 500},
    headers={"X-API-Key": "sk-nano-abc123"}
)
print(response.json()["generated_texts"][0])
```

### Deployment Options

```
┌──────────────────────────────────────────────────────────────┐
│                  DEPLOYMENT OPTIONS                          │
│                                                              │
│  LOCAL (Development)                                         │
│  └── uvicorn api.server:app --host 0.0.0.0 --port 8000      │
│                                                              │
│  CLOUD (Production)                                          │
│  ├── AWS EC2 (GPU instance for larger models)                │
│  ├── Google Cloud Run (CPU, for small models)                │
│  ├── Railway / Render (easy deploy, CPU)                     │
│  ├── Hugging Face Spaces (free, CPU/GPU)                     │
│  └── Docker Container (any platform)                         │
│                                                              │
│  DOCKER DEPLOYMENT                                           │
│  ├── Dockerfile                                              │
│  │   FROM python:3.11-slim                                   │
│  │   COPY . /app                                             │
│  │   RUN pip install torch numpy fastapi uvicorn tiktoken     │
│  │   CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0"]  │
│  └── docker run -p 8000:8000 nanogpt-api                     │
│                                                              │
│  AUTHENTICATION MODES                                        │
│  ├── No Auth:     REQUIRE_API_KEY=false                      │
│  │   → Anyone can access the API                             │
│  ├── API Key:     REQUIRE_API_KEY=true + X-API-Key header    │
│  │   → Only users with valid keys can access                 │
│  └── OAuth/JWT:   Add middleware for production auth          │
│      → Full user management, rate limiting, etc.             │
└──────────────────────────────────────────────────────────────┘
```

### Complete End-to-End Flow

```
┌──────────┐     HTTP POST       ┌──────────────┐     Load      ┌──────────┐
│          │  /generate          │              │   ckpt.pt     │          │
│ Frontend │ ──────────────────→ │  FastAPI     │ ────────────→ │  GPT     │
│ (React/  │  { prompt,          │  Server      │               │  Model   │
│  Vue/    │    max_tokens,      │  (server.py) │  Generate     │ (10.7M   │
│  HTML)   │    temperature }    │              │ ←──────────── │  params) │
│          │                     │              │  text tokens  │          │
│          │ ←────────────────── │              │               │          │
│          │  { generated_texts } │              │               │          │
└──────────┘                     └──────────────┘               └──────────┘
     │                                │
     │                                │ (Optional)
     │                           ┌────▼────┐
     │                           │ API Key │
     │                           │ Check   │
     │                           │ (auth)  │
     │                           └─────────┘
     │
     └─── User sees Shakespeare-style text in the browser
```

---

*Built with [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy | Guide by Tapesh Chavle*
