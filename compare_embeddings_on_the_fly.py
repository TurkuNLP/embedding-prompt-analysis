# python3 compare_embeddings_on_the_fly.py --large embeddings/multilingual-e5-large-instruct/emb_mteb_hotpotqa_corpus.*.embeddings.pkl --queries datasets/mteb_hotpotqa_dev_queries.jsonl --batch 50000 --emb-batch 50 --top-k 20 --dataset-jsonl datasets/mteb_hotpotqa_corpus.jsonl --model intfloat/multilingual-e5-large-instruct --preload-pkl-file-to-memory

import argparse
import os
import compare_embeddings as ce
import tqdm
import json
from sentence_transformers import SentenceTransformer
from prompts import get_prompt
import torch

def build_query_embeddings(model_name, query_file_name, args):
    model = SentenceTransformer(model_name)
    with open(query_file_name, "rt") as f:
        if query_file_name.endswith(".jsonl"):
            queries = [json.loads(q)["text"] for q in f.readlines() if q.strip()]
        else:
            queries=[q for q in f.readlines() if q.strip()]
    if args.use_prompt:
        prompt = get_prompt(model_name, args.use_prompt)
        queries = [prompt + q for q in queries]
    embeddings = model.encode(queries, convert_to_numpy=True, show_progress_bar=True, batch_size=args.emb_batch_size)
    return queries,embeddings

def load_texts_from_jsonl(file_name):
    texts = []
    with open(file_name, "rt") as f:
        for line in tqdm.tqdm(f, desc="Loading texts from JSONL"):
            line=line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            texts.append(text)
    return texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare query text against large embeddings on the fly",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="SentenceTransformer model name for encoding queries.",
    )
    parser.add_argument(
        "--large-embeddings",
        type=str,
        required=True,
        nargs="+",
        help="Paths to all large embeddings .embeddings.pkl files (assuming that the corresponding .examples.jsonl files are in the same directory). As many as you need.",
    )
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Text file with one query per line, or a JSONL file with a 'text' field.",
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=str,
        required=True,
        help="JSONL with the original dataset of large embeddings (for lookups).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of nearest neighbors to return. Use -1 for all. NOTE! Blows memory right now. TODO FIX THIS.",
    )
    parser.add_argument(
        "--use-prompt",
        type=str,
        default=None,
        help="Use prompt for query text (options: hotpotqa, nq, quora) Default: None.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch size for comparing embeddings. Multiplies this many embeddings with all queries at a time",
    )
    parser.add_argument(
        "--emb-batch-size",
        type=int,
        default=100,
        help="Batch size for embedding the query texts with the model.")
    parser.add_argument(
        "--preload-pkl-file-to-memory",
        action="store_true",
        default=False,
        help="Preload the .embeddings.pkl files to memory (one at a time). Do this if you are on puhti/mahti/lumi LUSTRE system.")
    args = parser.parse_args()
    if args.top_k == -1:
        args.top_k = None
    
    # Build query embeddings on the fly
    query_texts, query_embeddings = build_query_embeddings(args.model, args.queries, args)
    query_embeddings_torch = torch.from_numpy(query_embeddings).to(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Shape of query_embeddings: {query_embeddings_torch.shape} on device {query_embeddings_torch.device}", file=sys.stderr, flush=True)
    # Load texts from JSONL for lookups
    texts = load_texts_from_jsonl(args.dataset_jsonl)
    # Load large embeddings in batches for comparison
    large_embeddings_generator = tqdm.tqdm(ce.yield_embeddings(args.large_embeddings,\
        args.preload_pkl_file_to_memory), desc=f"Loading large embeddings from disk")
    large_embeddings_batches = ce.batch_embeddings(large_embeddings_generator, args.batch_size)
    # Compare query embeddings against large embeddings
    all_top_indices, all_top_cosine_similarities = ce.compare_embeddings(tqdm.tqdm(large_embeddings_batches, desc=f"Comparing embeddings batches of {args.batch_size} at a time"), query_embeddings_torch, args.top_k)
    
    # Print results
    for query_text, top_indices, top_cosine_similarities in zip(query_texts, all_top_indices, all_top_cosine_similarities):
        print(f"Query: {query_text}")
        for top_index, top_cosine_similarity in zip(top_indices, top_cosine_similarities):
            print(f"    Top index: {top_index}")
            print(f"    Top cosine similarity: {top_cosine_similarity}")
            print(f"    Text: {texts[top_index][:200]}")
            print("-"*100)
            print()
