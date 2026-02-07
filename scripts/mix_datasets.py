"""Mix multiple text datasets with configurable ratios."""

import argparse
import random
from pathlib import Path


def load_chunks(path: Path, chunk_size: int = 10000) -> list[str]:
    """Load text file and split into chunks by character count."""
    text = path.read_text()
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Mix datasets with ratios")
    parser.add_argument("--output", type=Path, required=True, help="Output file path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "sources",
        nargs="+",
        help="path:ratio pairs (e.g., data/wiki.txt:0.5 data/web.txt:0.5)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Parse sources
    sources = []
    for s in args.sources:
        path, ratio = s.rsplit(":", 1)
        sources.append((Path(path), float(ratio)))

    # Normalize ratios
    total = sum(r for _, r in sources)
    sources = [(p, r / total) for p, r in sources]

    print("Mixing datasets:")
    for path, ratio in sources:
        print(f"  {path}: {ratio:.1%}")

    # Load all chunks
    all_chunks = []
    for path, ratio in sources:
        chunks = load_chunks(path)
        # Sample according to ratio
        n_samples = int(len(chunks) * ratio)
        sampled = random.sample(chunks, min(n_samples, len(chunks)))
        all_chunks.extend(sampled)
        print(f"  Loaded {len(sampled)} chunks from {path}")

    # Shuffle
    random.shuffle(all_chunks)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for chunk in all_chunks:
            f.write(chunk)
            f.write("\n\n")

    print(f"\nWrote {len(all_chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
