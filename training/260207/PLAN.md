# Training Plan 260207

**Goal**: Chat model that can solve OpenBookQA without overfitting on it.

## Datasets

| Dataset | Purpose | Size |
|---------|---------|------|
| WikiText-103 | Factual knowledge (Wikipedia) | ~550MB |
| TinyStories | Simple coherent reasoning | ~2GB |
| Alpaca | Instruction following / chat | ~50MB |
| OpenBookQA | **Evaluation only** | - |

## Training Stages

### Stage 1: Pre-training (Mixed)

```bash
mix-datasets --output training/260207/pretrain-data.txt \
    datasets/260207-wikitext/train.txt:0.5 \
    datasets/260207-tinystories/train.txt:0.5
```

**Config**: `training/260207/pretrain.json`

- Data: 50% WikiText + 50% TinyStories
- Effective batch: 64 (4 × 16 accumulation)
- LR: warmup 2000 steps → 6e-4 → cosine decay → 6e-5
- Epochs: 1

**Checkpoint**: `training/260207/checkpoints/pretrain.safetensors`

### Stage 2: Fine-tuning (Alpaca)

**Config**: `training/260207/finetune.json`

- Data: Alpaca instructions
- Resume from: Stage 1 checkpoint
- Effective batch: 32 (4 × 8 accumulation)
- LR: warmup 100 steps → 2e-5 → cosine decay → 2e-6
- Epochs: 3

**Checkpoint**: `training/260207/checkpoints/chat.safetensors`

### Stage 3: Evaluation

```bash
eval-openbookqa --checkpoint training/260207/checkpoints/chat.safetensors
```

**Target**: Beat random baseline (25%)

## Commands

```bash
# 1. Download datasets
cd datasets/260207-wikitext && ./download.sh
cd datasets/260207-tinystories && ./download.sh
cd datasets/260207-alpaca && ./download.sh
cd datasets/260207-openbook-qa && ./download.sh

# 2. Mix pre-training data
mix-datasets --output training/260207/pretrain-data.txt \
    datasets/260207-wikitext/train.txt:0.5 \
    datasets/260207-tinystories/train.txt:0.5

# 3. Pre-train
train --config training/260207/pretrain.json

# 4. Fine-tune
train --config training/260207/finetune.json

# 5. Evaluate
eval-openbookqa --checkpoint training/260207/checkpoints/chat.safetensors

# 6. Chat
infer --checkpoint training/260207/checkpoints/chat.safetensors --prompt "What causes rain?"
```
