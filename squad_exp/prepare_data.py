import json
import argparse
import random
from collections import Counter
import os

from datasets import load_dataset


def majority_answer(answer_texts, rng):
    """Pick the most common short answer span; break ties randomly."""
    counts = Counter(answer_texts)
    top = max(counts.values())
    majority = [text for text, n in counts.items() if n == top]
    return rng.choice(majority)


def normalize_text(text):
    return " ".join(text.strip().lower().split())


def normalized_answer_signature(answer_texts):
    return tuple(sorted(normalize_text(text) for text in answer_texts))


def iter_squad_json(path):
    with open(path) as f:
        data = json.load(f)
    for article in data["data"]:
        for para in article["paragraphs"]:
            context_norm = normalize_text(para["context"])
            for qa in para["qas"]:
                answer_sig = normalized_answer_signature(ans["text"] for ans in qa["answers"])
                yield context_norm, answer_sig, qa["question"]


def load_gan_paraphrases(orig_path, para_path):
    """Return mapping: (normalized_context, normalized_original_question) -> paraphrase.

    Gan orig and para are aligned by row order; each pair must share identical
    normalized context and sorted normalized answer signature.
    """
    orig_rows = list(iter_squad_json(orig_path))
    para_rows = list(iter_squad_json(para_path))
    if len(orig_rows) != len(para_rows):
        raise ValueError("Gan orig/para sizes differ.")

    mapping = {}
    for idx, (orig_row, para_row) in enumerate(zip(orig_rows, para_rows)):
        orig_context, orig_answers, orig_q = orig_row
        para_context, para_answers, para_q = para_row
        if (orig_context, orig_answers) != (para_context, para_answers):
            raise ValueError(
                f"Gan orig/para alignment mismatch at index {idx} for (context, sorted answers)."
            )

        key = (orig_context, normalize_text(orig_q))
        if key in mapping:
            raise ValueError(f"Duplicate (context, original_question) key at index {idx}.")
        mapping[key] = para_q
    print(f"Loaded {len(mapping)} Gan paraphrases.")
    return mapping


def prepare_split(squad_split, gan_paraphrases, rng):
    """Build examples for one split.

    Args:
        squad_split: HuggingFace SQuAD dataset split.
        gan_paraphrases: dict mapping (context, question) -> paraphrase, or None.
        rng: random.Random instance.
    """
    examples = []
    used_keys = set()
    missing_gan = 0
    for ex in squad_split:
        question = ex["question"]
        context = ex["context"]
        answers = ex["answers"]
        answer_texts = answers["text"]

        row = {
            "id": ex["id"],
            "question": question,
            "gan_paraphrase": "",
            "generated_paraphrase": "",
            "answer_paragraph": context,
            "short_answer": majority_answer(answer_texts, rng),
        }

        if gan_paraphrases is not None:
            key = (normalize_text(context), normalize_text(question))
            if key in gan_paraphrases:
                row["gan_paraphrase"] = gan_paraphrases[key]
                used_keys.add(key)
            else:
                missing_gan += 1

        examples.append(row)
    if gan_paraphrases is not None:
        print(f"Used {len(used_keys)} Gan paraphrases, missing {missing_gan}.")
    return examples

def main(args):

    rng = random.Random(args.seed)

    ds = load_dataset("rajpurkar/squad", cache_dir=args.cache_dir)
    gan = load_gan_paraphrases(args.gan_orig, args.gan_para)

    train = prepare_split(ds["train"], gan_paraphrases=None, rng=rng)
    valid = prepare_split(ds["validation"], gan_paraphrases=gan, rng=rng)

    os.makedirs(args.output_dir, exist_ok=True)
    for name, data in [("train", train), ("valid", valid)]:
        path = os.path.join(args.output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(data)} examples to {path}")


if __name__ == "__main__":
    default_path = "/scratch/project_2000539/jenna/datasets/embedding-prompt-analysis/paraphrasing-squad"

    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_orig", default=f"{default_path}/dev_orig.json",
                        help="Gan & Ng original questions (eval split)")
    parser.add_argument("--gan_para", default=f"{default_path}/dev_para.json",
                        help="Gan & Ng paraphrased questions (eval split)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default="/scratch/project_2000539/jenna/hf-cache")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
