import argparse
import json
import os


PARAPHRASE_PROMPT = """\
Paraphrase the following question while preserving its meaning exactly. Use different wording and avoid producing a near-identical phrasing.

Output only the paraphrased question and nothing else.

Question: {question}"""

# Batch API is 50% of standard gpt-5.4-mini prices.
BATCH_PRICE_PER_1M_TOKENS = {
    "input": 0.375,
    "output": 2.25,
}
CHARS_PER_TOKEN_ESTIMATE = 4.0
BATCH_ENDPOINT = "/v1/responses"


def has_generated_paraphrase(example):
    value = example.get("generated_paraphrase", "")
    return isinstance(value, str) and bool(value.strip())


def estimate_batch_cost(split_name, examples, label="all"):
    n_questions = len(examples)
    question_chars = sum(len(ex["question"]) for ex in examples)
    prompt_chars = len(PARAPHRASE_PROMPT.format(question="")) * n_questions
    input_chars = question_chars + prompt_chars
    output_chars = question_chars

    input_tokens = input_chars / CHARS_PER_TOKEN_ESTIMATE
    output_tokens = output_chars / CHARS_PER_TOKEN_ESTIMATE
    cost = (
        (input_tokens / 1_000_000) * BATCH_PRICE_PER_1M_TOKENS["input"]
        + (output_tokens / 1_000_000) * BATCH_PRICE_PER_1M_TOKENS["output"]
    )
    print(
        f"[{split_name}][{label}] questions={n_questions}, question_chars={question_chars}, "
        f"prompt_chars={prompt_chars}, est_batch_cost_usd=${cost:.4f}"
    )
    return cost


def split_stats(split_name, examples):
    done = sum(1 for ex in examples if has_generated_paraphrase(ex))
    missing = [ex for ex in examples if not has_generated_paraphrase(ex)]
    print(f"[{split_name}] total={len(examples)}, done={done}, missing={len(missing)}")
    return missing


def make_batch_request(example, model):
    return {
        "custom_id": example["id"],
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": model,
            "max_output_tokens": 128,
            "input": [
                {
                    "role": "user",
                    "content": PARAPHRASE_PROMPT.format(question=example["question"]),
                }
            ],
        },
    }


def write_batch_file(path, examples, model, max_generate):
    selected = examples if max_generate is None else examples[:max_generate]
    seen = set()
    with open(path, "w") as f:
        for ex in selected:
            custom_id = ex["id"]
            if custom_id in seen:
                raise ValueError(f"Duplicate id in batch file {path}: {custom_id}")
            seen.add(custom_id)
            print(json.dumps(make_batch_request(ex, model), ensure_ascii=False), file=f)
    print(f"Wrote {len(selected)} requests to {path}")
    return selected


def main(args):
    train_path = os.path.join(args.data_dir, "train.json")
    valid_path = os.path.join(args.data_dir, "valid.json")

    with open(train_path) as f:
        train_examples = json.load(f)
    with open(valid_path) as f:
        valid_examples = json.load(f)

    train_missing = split_stats("train", train_examples)
    valid_missing = split_stats("eval", valid_examples)

    train_cost_all = estimate_batch_cost("train", train_examples, label="all")
    valid_cost_all = estimate_batch_cost("eval", valid_examples, label="all")
    print(f"[total][all] est_batch_cost_usd=${train_cost_all + valid_cost_all:.4f}")

    train_cost_remaining = estimate_batch_cost("train", train_missing, label="remaining")
    valid_cost_remaining = estimate_batch_cost("eval", valid_missing, label="remaining")
    print(
        f"[total][remaining] est_batch_cost_usd=${train_cost_remaining + valid_cost_remaining:.4f}"
    )

    if not args.write_batch:
        print("Dry run only. Pass --write-batch to write eval/train batch JSONL files.")
        return

    eval_batch_path = os.path.join(args.data_dir, "eval_batch.jsonl")
    train_batch_path = os.path.join(args.data_dir, "train_batch.jsonl")
    written_eval = write_batch_file(eval_batch_path, valid_missing, args.model, args.max_generate)
    written_train = write_batch_file(train_batch_path, train_missing, args.model, args.max_generate)
    estimate_batch_cost("eval", written_eval, label="this_file")
    estimate_batch_cost("train", written_train, label="this_file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing train.json and valid.json")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument(
        "--write-batch",
        action="store_true",
        help="Write eval_batch.jsonl and train_batch.jsonl for missing paraphrases",
    )
    parser.add_argument(
        "--max-generate",
        type=int,
        default=None,
        help="Optional cap on requests written per split (for testing)",
    )
    args = parser.parse_args()
    main(args)
