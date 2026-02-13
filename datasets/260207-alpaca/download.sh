#!/usr/bin/env bash
# Stanford Alpaca dataset (instruction fine-tuning)
# https://huggingface.co/datasets/tatsu-lab/alpaca
#
# 52K instruction-following examples for chat capability

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ../check.sh; then exit 0; fi

echo "Downloading Alpaca dataset..."

python3 << 'EOF'
from datasets import load_dataset

ds = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
print(f"Dataset size: {len(ds)} examples")

# Format as chat text for continued pre-training
# Using a simple prompt format
with open("train.txt", "w") as f:
    for ex in ds:
        instruction = ex["instruction"]
        input_text = ex["input"]
        output = ex["output"]

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        f.write(prompt)
        f.write("\n\n<|endoftext|>\n\n")

print("Done!")
print(f"  train.txt: {len(ds)} examples")
EOF

sha256sum train.txt > checksums.sha256
