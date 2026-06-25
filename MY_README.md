# 🧠 NanoGPT — Complete Setup & Run Guide (macOS Apple Silicon M4)

A step-by-step guide to set up, train, and interact with a GPT language model on your Mac M4 using [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

---

## 📖 Table of Contents

1. [What is nanoGPT?](#-what-is-nanogpt)
2. [Model Configurations & Resource Requirements](#-model-configurations--resource-requirements)
3. [Prerequisites](#-prerequisites)
4. [Setup (One-Time)](#-setup-one-time)
5. [Training the Model](#-training-the-model)
6. [Generating Text (Inference / Interaction)](#-generating-text-inference--interaction)
7. [Interacting with Pretrained GPT-2 (No Training Needed)](#-interacting-with-pretrained-gpt-2-no-training-needed)
8. [Resume Training from Checkpoint](#-resume-training-from-checkpoint)
9. [How It Works (Architecture)](#-how-it-works-architecture)
10. [Troubleshooting](#-troubleshooting)
11. [Key Files Reference](#-key-files-reference)

---

## 🤖 What is nanoGPT?

nanoGPT is a minimal, clean implementation of the GPT (Generative Pre-trained Transformer) language model by Andrej Karpathy. It can:

- **Train a GPT model from scratch** on any text dataset
- **Fine-tune pretrained GPT-2 models** (124M to 1.5B parameters) from OpenAI
- **Generate text** (inference) from trained or pretrained models

The entire codebase is just two main files:
- `train.py` (~300 lines) — the training loop
- `model.py` (~300 lines) — the GPT model definition

---

## 📊 Model Configurations & Resource Requirements

| Configuration | Parameters | GPU Needed | Memory | Training Time | Mac M4 Compatible |
|---|---|---|---|---|---|
| Shakespeare Char (mini/macbook) | **~0.8M** | CPU / MPS | ~200 MB | ~3 min | ✅ Yes |
| Shakespeare Char (full) | **~10.7M** | MPS | ~500 MB | ~30 min | ✅ Yes |
| GPT-2 (base) | **124M** | 8× A100 40GB | ~320 GB | ~4 days | ❌ No |
| GPT-2 Medium | **350M** | Multi-GPU | ~700 GB | Days | ❌ No |
| GPT-2 Large | **774M** | Multi-GPU | ~1.5 TB | Days | ❌ No |
| GPT-2 XL | **1.56B** | Multi-GPU | ~3 TB+ | Days+ | ❌ No |

> **Note:** On Mac M4, we use `--device=mps` (Metal Performance Shaders) which leverages the Apple Silicon GPU for 2-3× faster training compared to CPU.

---

## 📋 Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.10+** (check with `python3 --version`)
- **Git** (check with `git --version`)
- **~1 GB free disk space**

---

## 🛠 Setup (One-Time)

### Step 1: Clone the Repository

```bash
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv .venv
```

### Step 3: Activate the Virtual Environment

```bash
source .venv/bin/activate
```

> **Note:** You need to run this command every time you open a new terminal. Your prompt should change to show `(.venv)` at the beginning.

### Step 4: Install Dependencies

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

This installs:
| Package | Purpose |
|---|---|
| `torch` (PyTorch) | Deep learning framework with MPS support |
| `numpy` | Numerical computing |
| `transformers` | Load pretrained GPT-2 checkpoints from HuggingFace |
| `datasets` | Download/preprocess datasets from HuggingFace |
| `tiktoken` | OpenAI's fast BPE tokenizer |
| `wandb` | Optional experiment logging |
| `tqdm` | Progress bars |

### Step 5: Prepare the Dataset

```bash
python data/shakespeare_char/prepare.py
```

This downloads the Tiny Shakespeare dataset (~1MB) and creates:
- `data/shakespeare_char/train.bin` — training data (1,003,854 tokens)
- `data/shakespeare_char/val.bin` — validation data (111,540 tokens)
- `data/shakespeare_char/meta.pkl` — character vocabulary (65 unique characters)

---

## 🏋️ Training the Model

### Option A: Full Training (~10.7M params, ~30 min on M4)

This is the **recommended** configuration for Mac. Trains a 6-layer, 6-head Transformer with 384-dim embeddings:

```bash
python train.py config/train_shakespeare_char.py --device=mps --compile=False
```

**What happens during training:**
- The model learns character patterns from Shakespeare's works
- Every 250 iterations, it evaluates on the validation set
- If validation loss improves, a checkpoint is saved to `out-shakespeare-char/ckpt.pt`
- Training runs for 5,000 iterations total
- You'll see output like:
  ```
  step 0: train loss 4.1676, val loss 4.1649
  iter 0: loss 4.1828, time 306.96ms, mfu -100.00%
  iter 10: loss 3.7366, time 8.81ms, mfu 0.14%
  ...
  ```

### Option B: Quick Training (~0.8M params, ~3 min on M4)

A smaller model for quick testing:

```bash
python train.py config/train_shakespeare_char.py \
    --device=mps \
    --compile=False \
    --eval_iters=20 \
    --log_interval=10 \
    --block_size=64 \
    --batch_size=12 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=2000 \
    --lr_decay_iters=2000 \
    --dropout=0.0
```

### Option C: CPU-Only Training (if MPS has issues)

```bash
python train.py config/train_shakespeare_char.py \
    --device=cpu \
    --compile=False \
    --eval_iters=20 \
    --log_interval=1 \
    --block_size=64 \
    --batch_size=12 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=2000 \
    --lr_decay_iters=2000 \
    --dropout=0.0
```

### Training Parameters Explained

| Parameter | Default | Description |
|---|---|---|
| `--device` | `cuda` | Device to train on (`mps` for Mac, `cpu` for CPU) |
| `--compile` | `True` | PyTorch 2.0 compilation (set `False` on Mac) |
| `--batch_size` | `64` | Number of examples per iteration |
| `--block_size` | `256` | Context window size (characters the model can "see") |
| `--n_layer` | `6` | Number of transformer layers |
| `--n_head` | `6` | Number of attention heads per layer |
| `--n_embd` | `384` | Embedding dimension |
| `--max_iters` | `5000` | Total training iterations |
| `--dropout` | `0.2` | Dropout rate for regularization |
| `--learning_rate` | `1e-3` | Learning rate |
| `--eval_interval` | `250` | How often to evaluate and save checkpoints |

---

## 💬 Generating Text (Inference / Interaction)

> **Important:** You only need to train once! The trained model is saved as a checkpoint file (`ckpt.pt`). You can generate text from it as many times as you want without retraining.

### Basic Text Generation

```bash
python sample.py --out_dir=out-shakespeare-char --device=mps
```

This generates 10 samples of 500 characters each. Example output:

```
BUCKINGHAM:
Would this is the little king's good it with all or a such...

Wherefore news? and he should be loved of hand;
And hep the king's sheep consent they like to the heavens...
```

### Custom Prompt (Start with specific text)

```bash
python sample.py \
    --out_dir=out-shakespeare-char \
    --device=mps \
    --start="ROMEO: O, she doth teach the torches to burn bright!"
```

### Control Generation Parameters

```bash
python sample.py \
    --out_dir=out-shakespeare-char \
    --device=mps \
    --start="HAMLET:" \
    --num_samples=5 \
    --max_new_tokens=1000 \
    --temperature=0.8 \
    --top_k=200
```

| Parameter | Default | Description |
|---|---|---|
| `--out_dir` | `out` | Directory containing the trained checkpoint |
| `--start` | `"\n"` | Starting text prompt for generation |
| `--num_samples` | `10` | Number of text samples to generate |
| `--max_new_tokens` | `500` | Maximum characters to generate per sample |
| `--temperature` | `0.8` | Creativity control (lower = more focused, higher = more random) |
| `--top_k` | `200` | Limits sampling to top-k most likely next characters |
| `--device` | `cuda` | Device for inference (`mps` for Mac) |

### Start from a File Prompt

```bash
echo "To be, or not to be, that is the question:" > prompt.txt
python sample.py \
    --out_dir=out-shakespeare-char \
    --device=mps \
    --start=FILE:prompt.txt
```

---

## 🌐 Interacting with Pretrained GPT-2 (No Training Needed)

You can skip training entirely and use OpenAI's pretrained GPT-2 models directly:

```bash
# Use GPT-2 base (124M parameters)
python sample.py \
    --init_from=gpt2 \
    --device=mps \
    --start="What is the meaning of life?" \
    --num_samples=3 \
    --max_new_tokens=200
```

Available pretrained models:
| Model | Parameters | Command |
|---|---|---|
| GPT-2 Base | 124M | `--init_from=gpt2` |
| GPT-2 Medium | 350M | `--init_from=gpt2-medium` |
| GPT-2 Large | 774M | `--init_from=gpt2-large` |
| GPT-2 XL | 1.56B | `--init_from=gpt2-xl` |

> **Warning:** `gpt2-large` and `gpt2-xl` may not fit in your Mac M4's memory. Stick with `gpt2` or `gpt2-medium`.

---

## 🔄 Resume Training from Checkpoint

If you stopped training midway, you can resume from the last saved checkpoint:

```bash
python train.py config/train_shakespeare_char.py \
    --device=mps \
    --compile=False \
    --init_from=resume
```

This loads the checkpoint from `out-shakespeare-char/ckpt.pt` and continues training from where it left off.

---

## 🏗 How It Works (Architecture)

```
┌─────────────────────────────────────────────────────┐
│                    GPT Model                         │
│                                                     │
│  Input: "To be or not"  →  Tokenize  →  [T,o, ,b,...] │
│                                                     │
│  ┌───────────────────────────────────────┐          │
│  │  Token Embedding (vocab_size × n_embd) │          │
│  │  Position Embedding (block_size × n_embd)│        │
│  └───────────────┬───────────────────────┘          │
│                  ↓                                   │
│  ┌───────────────────────────────────────┐          │
│  │  Transformer Block × n_layer           │          │
│  │  ┌─────────────────────────────────┐  │          │
│  │  │  Layer Norm                      │  │          │
│  │  │  Multi-Head Self-Attention       │  │          │
│  │  │  (n_head attention heads)        │  │          │
│  │  │  Layer Norm                      │  │          │
│  │  │  MLP (Feed-Forward Network)      │  │          │
│  │  │  (n_embd → 4×n_embd → n_embd)   │  │          │
│  │  └─────────────────────────────────┘  │          │
│  └───────────────┬───────────────────────┘          │
│                  ↓                                   │
│  ┌───────────────────────────────────────┐          │
│  │  Language Model Head                   │          │
│  │  (n_embd → vocab_size)                 │          │
│  └───────────────┬───────────────────────┘          │
│                  ↓                                   │
│  Output: Probability distribution over next token    │
│  Prediction: "to" (most likely next character)       │
└─────────────────────────────────────────────────────┘
```

### Training Flow

```
Shakespeare Text → Character Tokenization → Train/Val Split
                                                  ↓
                                        Training Loop (5000 iters)
                                                  ↓
                                   Model learns to predict next character
                                                  ↓
                                   Checkpoint saved (ckpt.pt)
                                                  ↓
                                   sample.py loads checkpoint → Generates text
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `externally-managed-environment` error | Use a virtual environment: `python3 -m venv .venv && source .venv/bin/activate` |
| `torch.compile` errors | Add `--compile=False` flag |
| Out of memory on MPS | Reduce `--batch_size` (try 32 or 16) and `--block_size` (try 128) |
| MPS not available | Use `--device=cpu` instead |
| `No such file: ckpt.pt` | You need to train first, or the training didn't save a checkpoint yet |
| Training seems stuck | Output may be buffered on MPS; check `out-shakespeare-char/ckpt.pt` file timestamp |
| Want to retrain from scratch | Delete `out-shakespeare-char/` folder and run training again |

---

## 📁 Key Files Reference

```
nanoGPT/
├── train.py                          # Main training script
├── model.py                          # GPT model definition
├── sample.py                         # Text generation / inference script
├── configurator.py                   # Config override utility
├── config/
│   ├── train_shakespeare_char.py     # Config: Train on Shakespeare (char-level)
│   ├── train_gpt2.py                 # Config: Reproduce GPT-2 124M
│   ├── finetune_shakespeare.py       # Config: Finetune GPT-2 on Shakespeare
│   ├── eval_gpt2.py                  # Config: Evaluate GPT-2 base
│   ├── eval_gpt2_medium.py           # Config: Evaluate GPT-2 medium
│   ├── eval_gpt2_large.py            # Config: Evaluate GPT-2 large
│   └── eval_gpt2_xl.py              # Config: Evaluate GPT-2 XL
├── data/
│   ├── shakespeare_char/
│   │   ├── prepare.py                # Download & tokenize Shakespeare
│   │   ├── input.txt                 # Raw Shakespeare text (~1MB)
│   │   ├── train.bin                 # Tokenized training data
│   │   ├── val.bin                   # Tokenized validation data
│   │   └── meta.pkl                  # Character vocabulary mapping
│   ├── shakespeare/                  # BPE-tokenized Shakespeare (for finetuning)
│   └── openwebtext/                  # OpenWebText dataset (for GPT-2 reproduction)
├── out-shakespeare-char/
│   └── ckpt.pt                       # Trained model checkpoint (~129 MB)
└── .venv/                            # Python virtual environment
```

---

## 🚀 Quick Start (TL;DR)

```bash
# ONE-TIME SETUP
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy transformers datasets tiktoken wandb tqdm
python data/shakespeare_char/prepare.py

# TRAIN (only once, ~30 min on Mac M4)
python train.py config/train_shakespeare_char.py --device=mps --compile=False

# GENERATE TEXT (anytime, as many times as you want!)
python sample.py --out_dir=out-shakespeare-char --device=mps --start="ROMEO:"
```

---

*Built with [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy*
