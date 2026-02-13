#!/usr/bin/env bash
# WikiText-103 dataset
# https://huggingface.co/datasets/Salesforce/wikitext
#
# Clean Wikipedia text from verified Good/Featured articles
# ~550MB, good for factual knowledge

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ../check.sh; then exit 0; fi

echo "Downloading WikiText-103 dataset..."

uv run python << 'EOF'
from datasets import load_dataset

# Use raw version (no <unk> tokens)
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", trust_remote_code=True)

for split in ["train", "validation", "test"]:
    data = ds[split]
    filename = f"{split}.txt"

    with open(filename, "w") as f:
        for example in data:
            text = example["text"].strip()
            if text:  # Skip empty lines
                f.write(text + "\n")

    print(f"{filename}: {len(data)} lines")

print("Done!")
EOF

sha256sum train.txt validation.txt test.txt > checksums.sha256
