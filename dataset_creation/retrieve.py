"""Evaluate instruction-conditioned retrieval on target_documents.jsonl."""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from dataset_prompts import ACTION_INSTRUCTIONS, format_e5_query_prompt

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"


def load_records(path, dedupe_source_ids=True):
    """Load dataset records, optionally keeping one row per source_id."""
    records = []
    seen_source_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            source_id = record["source_id"]
            if dedupe_source_ids and source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)
            records.append(record)

    return records


def build_candidates_and_queries(records):
    """
    Build a static target index and one retrieval query per non-empty target.

    Each candidate is keyed by (source_id, action). Queries retrieve against the
    full candidate pool, including targets from other sources and other actions.
    """
    candidates = []
    candidate_index = {}
    queries = []

    for record in records:
        source_id = record["source_id"]
        source_text = record["source_text"]

        for target in record["targets"]:
            action = target["action"]
            target_text = target["target_text"].strip()
            if not target_text:
                continue

            key = (source_id, action)
            if key not in candidate_index:
                candidate_index[key] = len(candidates)
                candidates.append(
                    {
                        "source_id": source_id,
                        "action": action,
                        "target_text": target_text,
                    }
                )

            queries.append(
                {
                    "source_id": source_id,
                    "source_text": source_text,
                    "action": action,
                    "gold_index": candidate_index[key],
                    "gold_target_text": target_text,
                }
            )

    return candidates, queries


def embed_texts(model, texts, batch_size, prompt=""):
    return model.encode(
        texts,
        prompt=prompt,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def embed_queries_by_action(model, queries, batch_size):
    """Embed queries in batches grouped by action-specific E5 prompts."""
    grouped = defaultdict(list)
    for index, query in enumerate(queries):
        grouped[query["action"]].append((index, query["source_text"]))

    embeddings = [None] * len(queries)
    for action, items in grouped.items():
        prompt = format_e5_query_prompt(action)
        indices, texts = zip(*items, strict=True)
        action_embeddings = embed_texts(model, list(texts), batch_size, prompt=prompt)
        for query_index, embedding in zip(indices, action_embeddings, strict=True):
            embeddings[query_index] = embedding

    return np.vstack(embeddings)


def evaluate_retrieval(candidates, queries, scores):
    total = len(queries)
    correct_at_1 = 0
    reciprocal_ranks = []
    per_action = defaultdict(
        lambda: {
            "total": 0,
            "correct_at_1": 0,
            "mrr": 0.0,
            "top1_retrieved_actions": defaultdict(int),
            "top1_same_source": 0,
            "top1_different_source": 0,
        }
    )
    failures = []

    for query_index, query in enumerate(queries):
        action = query["action"]
        gold_index = query["gold_index"]
        ranked_indices = np.argsort(-scores[query_index])
        rank = int(np.where(ranked_indices == gold_index)[0][0]) + 1
        top_index = int(ranked_indices[0])
        retrieved_action = candidates[top_index]["action"]
        retrieved_source_id = candidates[top_index]["source_id"]
        same_source = retrieved_source_id == query["source_id"]

        per_action[action]["total"] += 1
        per_action[action]["mrr"] += 1.0 / rank
        per_action[action]["top1_retrieved_actions"][retrieved_action] += 1
        if same_source:
            per_action[action]["top1_same_source"] += 1
        else:
            per_action[action]["top1_different_source"] += 1
        reciprocal_ranks.append(1.0 / rank)

        if rank == 1:
            correct_at_1 += 1
            per_action[action]["correct_at_1"] += 1
        else:
            failures.append(
                {
                    "source_id": query["source_id"],
                    "source_text": query["source_text"],
                    "action": action,
                    "rank": rank,
                    "gold_target_text": query["gold_target_text"],
                    "retrieved_target_text": candidates[top_index]["target_text"],
                    "retrieved_source_id": retrieved_source_id,
                    "retrieved_action": retrieved_action,
                }
            )

    metrics = {
        "num_queries": total,
        "num_candidates": len(candidates),
        "accuracy_at_1": correct_at_1 / total if total else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "per_action": {
            action: {
                "total": stats["total"],
                "accuracy_at_1": stats["correct_at_1"] / stats["total"],
                "mrr": stats["mrr"] / stats["total"],
                "top1_action_distribution": _format_action_distribution(
                    stats["top1_retrieved_actions"],
                    stats["total"],
                ),
                "top1_source_distribution": _format_source_distribution(
                    stats["top1_same_source"],
                    stats["top1_different_source"],
                    stats["total"],
                ),
            }
            for action, stats in sorted(per_action.items())
        },
    }
    return metrics, failures


def _format_source_distribution(same_source, different_source, total):
    return {
        "same_source": {
            "count": same_source,
            "fraction": same_source / total if total else 0.0,
        },
        "different_source": {
            "count": different_source,
            "fraction": different_source / total if total else 0.0,
        },
    }


def _format_action_distribution(retrieved_actions, total):
    distribution = {}
    for retrieved_action, count in sorted(
        retrieved_actions.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        distribution[retrieved_action] = {
            "count": count,
            "fraction": count / total if total else 0.0,
        }
    return distribution


def _print_action_distribution(distribution):
    parts = [
        f"{action} {stats['fraction']:.2f} ({stats['count']})"
        for action, stats in distribution.items()
    ]
    print(f"             top-1 actions: {', '.join(parts)}")


def _print_source_distribution(distribution):
    same = distribution["same_source"]
    different = distribution["different_source"]
    print(
        f"             top-1 source_id: "
        f"same {same['fraction']:.2f} ({same['count']}), "
        f"different {different['fraction']:.2f} ({different['count']})"
    )


def print_metrics(metrics):
    print(f"Queries: {metrics['num_queries']}")
    print(f"Candidates: {metrics['num_candidates']}")
    print(f"Accuracy@1: {metrics['accuracy_at_1']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print("\nPer-action results:")
    for action, stats in metrics["per_action"].items():
        instruction = ACTION_INSTRUCTIONS[action]
        print(
            f"  {action:10s}  acc@1={stats['accuracy_at_1']:.4f}  "
            f"mrr={stats['mrr']:.4f}  n={stats['total']}"
        )
        print(f"             instruction: {instruction}")
        _print_action_distribution(stats["top1_action_distribution"])
        _print_source_distribution(stats["top1_source_distribution"])


def main(args):
    records = load_records(args.dataset, dedupe_source_ids=not args.keep_duplicate_sources)
    candidates, queries = build_candidates_and_queries(records)

    print(f"Loaded {len(records)} source records from {args.dataset}")
    print(f"Built {len(candidates)} target candidates and {len(queries)} retrieval queries")

    print(f"Loading model {args.model_name}...")
    model = SentenceTransformer(args.model_name, trust_remote_code=True)

    candidate_texts = [candidate["target_text"] for candidate in candidates]
    print("Embedding target candidates...")
    candidate_embeddings = embed_texts(
        model,
        candidate_texts,
        args.batch_size,
        prompt=args.passage_prompt,
    )

    print("Embedding source queries with action instructions...")
    query_embeddings = embed_queries_by_action(model, queries, args.batch_size)

    print("Scoring queries against the target index...")
    scores = query_embeddings @ candidate_embeddings.T

    metrics, failures = evaluate_retrieval(candidates, queries, scores)
    print_metrics(metrics)

    if args.show_failures:
        sample_size = min(args.show_failures, len(failures))
        sampled_failures = random.sample(failures, sample_size)
        print(f"\n{sample_size} random failures:")
        for failure in sampled_failures:
            print("-" * 100)
            print(
                f"{failure['source_id']} / {failure['action']} "
                f"(rank {failure['rank']})"
            )
            print(
                f"Query:     [Instruct: {ACTION_INSTRUCTIONS[failure['action']]}] "
                f"{failure['source_text']}"
            )
            print(
                f"Retrieved: {failure['retrieved_source_id']} / "
                f"{failure['retrieved_action']}"
            )
            print(f"Gold:      {failure['gold_target_text']}")
            print(f"Retrieved: {failure['retrieved_target_text']}")
     

    if args.output_json:
        output = {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "metrics": metrics,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved metrics to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate instruction-conditioned retrieval accuracy"
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), "target_documents.jsonl"),
        help="Path to target_documents.jsonl",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="SentenceTransformer model for retrieval",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--passage-prompt",
        default="",
        help="Optional prompt prefix when embedding target passages",
    )
    parser.add_argument(
        "--keep-duplicate-sources",
        action="store_true",
        help="Keep multiple dataset rows with the same source_id",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=5,
        help="Print this many random retrieval failures; use 0 to disable",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for saving evaluation metrics as JSON",
    )
    main(parser.parse_args())
