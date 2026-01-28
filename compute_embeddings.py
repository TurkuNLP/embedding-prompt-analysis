from sentence_transformers import SentenceTransformer
import numpy as np
import argparse
from datasets import load_dataset
import sys
import pickle
from tqdm import tqdm
import transformers
import json
import gc

def load_model(args):
    model = SentenceTransformer(args.model_name, trust_remote_code=True)
    if args.half_model:
        model.half()
    return model

def stream_jsonl_dataset(path, args):
    # Streaming load a local JSONL file (or glob) and yield batches of examples
    dataset = load_dataset(
        "json",
        data_files={"all": path},
        split="all",
        streaming=True,
    )

    batch = []
    for example_index, example in enumerate(dataset):
        if example_index % args.shard_total != args.shard_index:
            continue
        example["global_example_index"] = example_index #index of the example in the original dataset
        if args.max_chars_per_example is not None:
            example[args.field_to_encode] = example[args.field_to_encode][:args.max_chars_per_example]
        batch.append(example)
        if len(batch) == args.batch_size:
            yield batch
            batch = []
    else:
        if batch:
            yield batch

def embed_dataset(model, stream_dataset, args):
    field_to_encode = args.field_to_encode
    metadata_list = open(f"{args.output_path_prefix}.{args.shard_index}.examples.jsonl", "wt")
    embeddings_file = open(f"{args.output_path_prefix}.{args.shard_index}.embeddings.pkl", "wb")
    for batch_index, batch in tqdm(enumerate(stream_dataset), desc="Embedding dataset", file=sys.stderr):
        embeddings = model.encode([example[field_to_encode] for example in batch], convert_to_numpy=True, show_progress_bar=True, batch_size=len(batch))
        for embedding, example in zip(embeddings, batch, strict=True):
            pickle.dump(embedding, embeddings_file)
            metadata={"global_example_index": example["global_example_index"]} #this is what to store in the metadata file, adjust as needed
            print(json.dumps(metadata), file=metadata_list)
        if batch_index % 10 == 0:
            print("Batch", batch_index, "completed", file=sys.stderr, flush=True)
            metadata_list.flush()
            embeddings_file.flush()
    metadata_list.close()
    embeddings_file.close()

if __name__ == "__main__":
 
    def parse_args():
        parser = argparse.ArgumentParser(description="Embed a jsonl dataset using a sentence transformer model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            "--model-name",
            type=str,
            default="KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
            help="Model name for SentenceTransformer"
        )
        parser.add_argument(
            "--half-model",
            action="store_true",
            default=False,
            help="Whether to use half precision for the model by calling .half() in SentenceTransformer"
        )
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="Path to the JSONL dataset file to embed"
        )
        parser.add_argument(
            "--output-path-prefix",
            type=str,
            required=True,
            help="Path prefix for the output files. The files will be named [prefix].shard_index.(examples|embeddings).(jsonl|pkl)"
        )
        parser.add_argument(
            "--shard-index",
            type=int,
            default=0,
            help="Index of the shard to process, 0-based."
        )
        parser.add_argument(
            "--shard-total",
            type=int,
            default=1,
            help="Total number of shards running in parallel."
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=50,
            help="Size of the batch to embed at a time."
        )
        parser.add_argument(
            "--field-to-encode",
            type=str,
            default="text",
            help="Field of the examples to encode."
        )
        parser.add_argument(
            "--max-chars-per-example",
            type=int,
            default=None,
            help="Maximum number of characters per example to encode. If None, no limit is applied."
        )
        return parser.parse_args()

    args = parse_args()
    
    print("Loading model...", file=sys.stderr, flush=True)
    model = load_model(args)
    print("Loading dataset...", file=sys.stderr, flush=True)
    stream_dataset = stream_jsonl_dataset(args.dataset, args)
    print("Embedding dataset...", file=sys.stderr, flush=True)
    embed_dataset(model, stream_dataset, args)
