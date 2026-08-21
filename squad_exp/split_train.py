#!/usr/bin/env python3
"""Split train.json into train-train / train-dev / train-test by answer paragraph.

Each unique answer_paragraph is assigned to exactly one split. Dev and test are
filled by greedily adding paragraphs until each has at least the requested
number of questions; all remaining paragraphs go to train-train.
"""

import argparse
import json
import os
import random
from collections import defaultdict


def load_examples(path):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def group_by_paragraph(examples):
    groups = defaultdict(list)
    for ex in examples:
        groups[ex["answer_paragraph"]].append(ex)
    return groups


def take_split(paragraph_keys, groups, target_questions):
    """Greedily take paragraphs until the split has >= target_questions."""
    selected_keys = []
    n_questions = 0
    while paragraph_keys and n_questions < target_questions:
        key = paragraph_keys.pop()
        selected_keys.append(key)
        n_questions += len(groups[key])
    examples = [ex for key in selected_keys for ex in groups[key]]
    return examples, n_questions, len(selected_keys)


def write_split(output_dir, name, examples):
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    n_paras = len({ex["answer_paragraph"] for ex in examples})
    print(f"Wrote {path}: {len(examples)} questions, {n_paras} answer paragraphs")
    return path


def main(args):
    rng = random.Random(args.seed)
    examples = load_examples(args.input)
    groups = group_by_paragraph(examples)

    paragraph_keys = list(groups.keys())
    rng.shuffle(paragraph_keys)

    total_questions = len(examples)
    needed = args.dev_size + args.test_size
    if needed > total_questions:
        raise ValueError(
            f"dev_size ({args.dev_size}) + test_size ({args.test_size}) = {needed} "
            f"exceeds total questions ({total_questions})"
        )

    test_examples, test_q, test_p = take_split(paragraph_keys, groups, args.test_size)
    dev_examples, dev_q, dev_p = take_split(paragraph_keys, groups, args.dev_size)

    train_examples = [ex for key in paragraph_keys for ex in groups[key]]
    train_q = len(train_examples)
    train_p = len(paragraph_keys)

    if test_q < args.test_size:
        raise ValueError(
            f"Could not reach test_size={args.test_size}; only got {test_q} questions"
        )
    if dev_q < args.dev_size:
        raise ValueError(
            f"Could not reach dev_size={args.dev_size}; only got {dev_q} questions"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    write_split(args.output_dir, "train-test", test_examples)
    write_split(args.output_dir, "train-dev", dev_examples)
    write_split(args.output_dir, "train-train", train_examples)

    print(
        f"Summary: train-train={train_q}q/{train_p}p, "
        f"train-dev={dev_q}q/{dev_p}p, "
        f"train-test={test_q}q/{test_p}p "
        f"(total {total_questions}q / {len(groups)}p)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Split train.json into train-train, train-dev, and train-test "
            "with disjoint answer paragraphs."
        )
    )
    parser.add_argument(
        "--input",
        default="squad_v1.1/train.json",
        help="Path to train.json (list of question/answer_paragraph examples)",
    )
    parser.add_argument(
        "--output_dir",
        default="squad_v1.1",
        help="Directory for train-train.json, train-dev.json, train-test.json",
    )
    parser.add_argument(
        "--dev_size",
        type=int,
        required=True,
        help="Minimum number of questions for train-dev",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        required=True,
        help="Minimum number of questions for train-test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for shuffling answer paragraphs before assignment",
    )
    args = parser.parse_args()
    main(args)
