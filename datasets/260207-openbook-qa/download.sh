#!/usr/bin/env bash
# OpenBookQA dataset (EVALUATION ONLY)
# https://huggingface.co/datasets/allenai/openbookqa
#
# Science QA benchmark requiring reasoning over facts + common knowledge
# 4-way multiple choice, ~6000 questions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ../check.sh; then exit 0; fi

echo "Downloading OpenBookQA dataset (for evaluation)..."

python3 << 'EOF'
import json
from datasets import load_dataset

ds = load_dataset("allenai/openbookqa", "main", trust_remote_code=True)

def format_example(ex):
    """Format as evaluation JSON."""
    choices = ex["choices"]
    return {
        "id": ex["id"],
        "question": ex["question_stem"],
        "choices": dict(zip(choices["label"], choices["text"])),
        "answer": ex["answerKey"],
        "fact": ex.get("fact1", ""),
    }

for split in ["train", "validation", "test"]:
    data = [format_example(ex) for ex in ds[split]]
    with open(f"{split}.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"{split}.jsonl: {len(data)} examples")

print("Done!")
EOF

sha256sum train.jsonl validation.jsonl test.jsonl > checksums.sha256
