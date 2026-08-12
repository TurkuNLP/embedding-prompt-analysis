"""Visualize target_documents.jsonl with e5-multilingual-instruct embeddings and t-SNE."""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
EMBED_INSTRUCTION = (
    "Instruct: Given a text, retrieve semantically similar texts\nQuery: "
)


def parse_ids(id):
    match = re.match(r"(batch_\d+)_doc_\d+", id)
    if match:
        return match.group(1), id
    else:
        raise ValueError(f"Invalid id: {id}")


def source_label_sort_key(source_id):
    match = re.match(r"batch_(\d+)_doc_(\d+)", source_id)
    if match:
        return (int(match.group(2)), int(match.group(1)))
    return (source_id,)

def load_sentences(path):
    """Return texts, block labels, and action labels."""
    texts = []
    block_labels = [] # block refers to all texts generated from the same seed text
    source_labels = [] # source refers to all targets generated from the same source text
    action_labels = [] # action refers to the action applied to the source text to generate the target text

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            block_id, source_id = parse_ids(record["source_id"])
            texts.append(record["source_text"])
            block_labels.append(block_id)
            source_labels.append(source_id)
            action_labels.append("source")

            for target in record["targets"]:
                target_text = target["target_text"].strip()
                if not target_text:
                    continue
                texts.append(target_text)
                block_labels.append(block_id)
                source_labels.append(source_id)
                action_labels.append(target["action"])

    return texts, block_labels, source_labels, action_labels


def embed_texts(model, texts, batch_size, prompt=""):
    return model.encode(
        texts,
        prompt=prompt,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def scatter_by_label(coords, labels, title, output_path, sort_key=None):
    if sort_key is None:
        sort_key = lambda label: label

    unique_labels = sorted(set(labels), key=sort_key)
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(12, 9))
    for label in unique_labels:
        mask = np.array([lbl == label for lbl in labels])
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=28,
            alpha=0.75,
            color=label_to_color[label],
            label=label,
            edgecolors="none",
        )

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(
        title="Label",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
        markerscale=1.5,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main(args):
    

    os.makedirs(args.output_dir, exist_ok=True)

    import random
    texts, block_labels, source_labels, action_labels = load_sentences(args.dataset)
    # Shuffle texts and labels together so text[i] matches label[i]
    combined = list(zip(texts, block_labels, source_labels, action_labels))
    random.shuffle(combined)
    texts, block_labels, source_labels, action_labels = map(list, zip(*combined))


    print(f"Loaded {len(texts)} sentences from {args.dataset}")
    print(f"Blocks (unique: {len(set(block_labels))}): {sorted(set(block_labels))}")
    print(f"Sources (unique: {len(set(source_labels))}): {sorted(set(source_labels))}")
    print(f"Actions (unique: {len(set(action_labels))}): {sorted(set(action_labels))}")

    # print exaples
    print("Examples:")
    for text, source_label, action_label in zip(texts[:20], source_labels[:20], action_labels[:20]):
        print(f"Text: {text} // {action_label} {source_label} ")
    print("-"*100)


    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    print("Computing embeddings...")
    embeddings = embed_texts(model, texts, args.batch_size, prompt="")

    perplexity = min(args.perplexity, max(5, len(texts) - 1))
    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=args.random_state,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(embeddings)

    scatter_by_label(
        coords,
        block_labels,
        "Dataset t-SNE colored by block",
        os.path.join(args.output_dir, "tsne_by_block.png"),
    )
    scatter_by_label(
        coords,
        source_labels,
        "Dataset t-SNE colored by source",
        os.path.join(args.output_dir, "tsne_by_source.png"),
        sort_key=source_label_sort_key,
    )
    scatter_by_label(
        coords,
        action_labels,
        "Dataset t-SNE colored by action",
        os.path.join(args.output_dir, "tsne_by_action.png"),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize dataset with t-SNE embeddings")
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), "target_documents.jsonl"),
        help="Path to target_documents.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "figures"),
        help="Directory for output plots",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(args)
