# Format train/dev JSON for swift-style embedding model finetuning.
#
# Two rows per input:
#   QA:         prompt+question -> answer_paragraph; negatives = other answer docs
#   paraphrase: prompt+question -> generated_paraphrase; negatives = other questions
# With probability --hard_negative_prob, both rows put a hard negative first
# (QA hard negative: question paraphrase; paraphrase hard negative: correct answer), replacing one in-corpus negative.
# Negatives from the same split; prompts on inputs only.

import argparse
import json
import random
from pathlib import Path

QA_PROMPT = (
    "Instruct: Given a question, retrieve Wikipedia passages that answer the question.\nQuery: "
)
PARAPHRASE_PROMPT = (
    "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question.\nQuery: "
)


def to_swift_example(query, positive, negatives):
    return {
        "messages": [{"role": "user", "content": query}],
        "positive_messages": [[{"role": "user", "content": positive}]],
        "negative_messages": [[{"role": "user", "content": n}] for n in negatives],
    }


def sample(candidates, n, rng, kind):
    candidates = list(candidates)
    if len(candidates) < n:
        raise ValueError(f"Need {n} {kind} negatives but only {len(candidates)} available.")
    return rng.sample(candidates, n)


def to_swift_examples(ex, unique_answers, questions, num_negatives, hard_negative_prob, rng):
    question = ex["question"]
    answer = ex["answer_paragraph"]
    paraphrase = ex["generated_paraphrase"]

    qa_negs = sample(
        (a for a in unique_answers if a != answer), num_negatives, rng, "answer"
    )
    # Same-answer questions cannot be negatives of each other.
    para_negs = sample(
        {q for q, a in questions if a != answer}, num_negatives, rng, "question"
    )

    if rng.random() < hard_negative_prob:
        qa_negs = [paraphrase] + qa_negs[1:]
        para_negs = [answer] + para_negs[1:]

    return [
        to_swift_example(QA_PROMPT + question, answer, qa_negs),
        to_swift_example(PARAPHRASE_PROMPT + question, paraphrase, para_negs),
    ]


def convert_file(input_path, output_path, num_negatives, hard_negative_prob, rng):
    with open(input_path) as f:
        examples = json.load(f)
    # shuffle examples to avoid bias
    random.shuffle(examples)

    unique_answers = list(dict.fromkeys(ex["answer_paragraph"] for ex in examples))
    questions = [(ex["question"], ex["answer_paragraph"]) for ex in examples]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(output_path, "w") as f:
        for ex in examples:
            for row in to_swift_examples(
                ex, unique_answers, questions, num_negatives, hard_negative_prob, rng
            ):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1

    print(
        f"Wrote {n_written} examples from {len(examples)} inputs "
        f"({num_negatives} negatives, hard_negative_prob={hard_negative_prob}): "
        f"{input_path} -> {output_path}"
    )


def default_output(path):
    path = Path(path)
    return path.with_name(path.stem + "_swift.jsonl")


def main(args):
    if not 0.0 <= args.hard_negative_prob <= 1.0:
        raise SystemExit("--hard_negative_prob must be in [0, 1]")
    if args.hard_negative_prob > 0 and args.num_negatives < 1:
        raise SystemExit("--num_negatives must be >= 1 when using hard negatives")

    rng = random.Random(args.seed)

    convert_file(
        args.train_file,
        args.train_out or default_output(args.train_file),
        args.num_negatives,
        args.hard_negative_prob,
        rng,
    )
    if args.dev_file:
        convert_file(
            args.dev_file,
            args.dev_out or default_output(args.dev_file),
            args.num_negatives,
            args.hard_negative_prob,
            rng,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert train/dev JSON to swift embedding finetuning JSONL."
    )
    parser.add_argument("--train_file", required=True, help="Path to train.json")
    parser.add_argument("--dev_file", default=None, help="Path to dev.json")
    parser.add_argument("--train_out", default=None, help="Default: <train>_swift.jsonl")
    parser.add_argument("--dev_out", default=None, help="Default: <dev>_swift.jsonl")
    parser.add_argument("--num_negatives", type=int, required=True)
    parser.add_argument(
        "--hard_negative_prob",
        type=float,
        default=0.0,
        help="Prob in [0, 1] to put hard negative first on both rows (default: 0.0, no hard negatives)",
    )
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
