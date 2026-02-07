"""Evaluate model on OpenBookQA benchmark."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from llm import RTT, tokenizer


def load_questions(path: Path) -> list[dict]:
    """Load JSONL evaluation data."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def score_choice(model: RTT, prompt: str, choice: str) -> float:
    """Score a single choice by computing log probability."""
    full_text = prompt + choice
    tokens = tokenizer.encode(full_text)
    prompt_len = len(tokenizer.encode(prompt))

    if len(tokens) > 512:
        tokens = tokens[:512]

    tokens_arr = jnp.array(tokens)[None, :]
    logits = model.forward(tokens_arr)[0]  # [seq, vocab]

    # Compute log prob of continuation tokens
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    total_log_prob = 0.0
    for i in range(prompt_len - 1, len(tokens) - 1):
        next_token = tokens[i + 1]
        total_log_prob += float(log_probs[i, next_token])

    # Normalize by length
    n_tokens = len(tokens) - prompt_len
    return total_log_prob / max(n_tokens, 1)


def evaluate_question(model: RTT, question: dict) -> tuple[str, bool]:
    """Evaluate a single question, return (predicted, correct)."""
    q = question["question"]
    choices = question["choices"]
    answer = question["answer"]

    # Format prompt
    prompt = f"Question: {q}\nAnswer:"

    # Score each choice
    scores = {}
    for label, text in choices.items():
        scores[label] = score_choice(model, prompt, f" {text}")

    # Pick highest scoring
    predicted = max(scores, key=scores.get)
    return predicted, predicted == answer


def main():
    parser = argparse.ArgumentParser(description="Evaluate on OpenBookQA")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("datasets/260207-openbook-qa/test.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = RTT("model", jax.random.key(0))
    model.load_safetensors(args.checkpoint)

    # Load questions
    questions = load_questions(args.data)
    if args.limit:
        questions = questions[:args.limit]
    print(f"Evaluating on {len(questions)} questions")

    # Evaluate
    correct = 0
    for i, q in enumerate(questions):
        pred, is_correct = evaluate_question(model, q)
        correct += int(is_correct)

        if (i + 1) % 100 == 0:
            acc = correct / (i + 1) * 100
            print(f"  [{i + 1}/{len(questions)}] Accuracy: {acc:.1f}%")

    accuracy = correct / len(questions) * 100
    print(f"\nFinal Accuracy: {accuracy:.1f}% ({correct}/{len(questions)})")
    print(f"Random baseline: 25.0%")


if __name__ == "__main__":
    main()
