#!/usr/bin/env bash
# OpenWebText dataset
# https://huggingface.co/datasets/Skylion007/openwebtext
#
# Open replication of WebText used to train GPT-2
# ~38GB uncompressed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Downloading OpenWebText dataset..."

# Download from HuggingFace using datasets library
python3 << 'EOF'
from datasets import load_dataset
from pathlib import Path

print("Loading dataset (this may take a while)...")
ds = load_dataset("Skylion007/openwebtext", split="train", trust_remote_code=True)

print(f"Dataset size: {len(ds)} documents")

# Save as text file
print("Writing train.txt...")
with open("train.txt", "w") as f:
    for i, example in enumerate(ds):
        f.write(example["text"])
        f.write("\n\n")  # Document separator
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1} documents...")

print("Done!")
EOF

echo "  train.txt: $(du -h train.txt | cut -f1)"
