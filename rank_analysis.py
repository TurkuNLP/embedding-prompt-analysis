from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from prompts import get_prompt
import compare_embeddings as ce
from compare_embeddings_on_the_fly import build_query_embeddings, load_texts_from_jsonl, create_parser, run_comparison
from tqdm import tqdm
import sys


def targets(queries, corpus):
    index_to_docid = {}
    with open(corpus, "r") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            index_to_docid[i] = d["id"]
    docid_to_index = {v: k for k, v in index_to_docid.items()}
    q_targets = []
    with open(queries, "r") as f:
        for line in f:
            d = json.loads(line)
            q_targets.append(d["targets"])
    return index_to_docid, docid_to_index, q_targets

def make_plot(array, file_name):

    # cut for plotting
    max_diff = 200
    arr = []
    for v in array:
        if v is not None and v < -1 * max_diff:
            arr.append(-1 * max_diff)
        elif v is not None and v > max_diff:
            arr.append(max_diff)
        else:
            arr.append(v)
    array = arr

    # create directory if it doesn't exist
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    file_path = os.path.join("figures", file_name)
    dimension_indices = np.arange(len(array))

    # Prepare values for special plotting
    min_val = min([v for v in array if v is not None]) if any(v is not None for v in array) else 0
    max_val = max([v for v in array if v is not None]) if any(v is not None for v in array) else 1

    plt.figure(figsize=(15, 6))
    
    for i, v in enumerate(array):
        if v is None:
            # white line to indicate missing data
            plt.vlines(i, 0, 0, colors='white', lw=0.7, alpha=0.7)
        elif v == 0:
            # very small visual blue line
            plt.vlines(i, -0.1, 0.1, colors='blue', lw=0.7, alpha=0.7)
        else:
            # Blue line from zero to value at this index
            plt.vlines(i, 0, v, colors='blue', lw=0.7, alpha=0.7)
    plt.title('Delta Rank for Each Example')
    plt.xlabel('Example Index')
    plt.ylabel('Delta Rank')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(pad=0)
    plt.savefig(file_path)
    plt.close()
    print("Saved plot to", file_path)
    

def plot_rank_difference(ranks, ranks_prompt, file_name):

    differences = []
    for (a,_), (b,_) in zip(ranks, ranks_prompt):
        if a is None or b is None:
            diff = None
        else:
            diff = a-b
        differences.append(diff)
    print("Ranks w/o  prompt:", [r for r,s in ranks[:100]])
    print(f"Mean rank w/o prompt: {np.mean([r for r,s in ranks if r is not None])}")
    print(f"Mean rank with prompt: {np.mean([r for r,s in ranks_prompt if r is not None])}")
    print("Ranks with prompt:", [r for r,s in ranks_prompt[:100]])
    print("Differences:      ", differences[:100])
    print(f"Mean difference: {np.mean([d for d in differences if d is not None])}")
    print(f"Positive difference: {np.sum([1 for d in differences if d is not None and d > 0])}")
    print(f"Negative difference: {np.sum([1 for d in differences if d is not None and d < 0])}")

    make_plot(differences, file_name)
 

def get_ranks(query_targets, all_top_indices, all_top_cosine_similarities, docid_to_index):
    print(f"Shape of similarities: {all_top_indices.shape, all_top_cosine_similarities.shape}", file=sys.stderr, flush=True)
    rank_results = []
    for q_targets, top_indices, top_cosine_similarities in zip(query_targets, all_top_indices, all_top_cosine_similarities):
        top_indices = top_indices.tolist()
        top_cosine_similarities = top_cosine_similarities.tolist()
        for t in q_targets:
            # t is document id, we need its index
            target_index = docid_to_index[t] # this is the index of the target in the large embeddings
            if target_index not in top_indices:
                rank = None
                sim = None
            else:
                rank = top_indices.index(target_index)+1 # get the rank of the target
                sim = top_cosine_similarities[rank-1]
            rank_results.append((rank, sim))
    return rank_results



def main(args):

    file_name = f"rank_difference_{args.model.split('/')[-1]}_{args.use_prompt.split('/')[-1]}.png"

    # Build query embeddings on the fly (uses prompt from args)
    query_texts, query_embeddings = build_query_embeddings(args.model, args.queries, args)

    # process targets
    index_to_docid, docid_to_index, query_targets = targets(args.queries, args.dataset_jsonl)
    print(f"Queries: {len(query_texts)}, Targets: {len(query_targets)}", file=sys.stderr, flush=True)
    
    all_top_indices, all_top_cosine_similarities = run_comparison(query_embeddings, args)
    rank_results_prompt = get_ranks(query_targets, all_top_indices, all_top_cosine_similarities, docid_to_index)


    # Same without prompt
    args.use_prompt = None # overwrite use_prompt to None
    query_texts, query_embeddings = build_query_embeddings(args.model, args.queries, args)
    print(f"Queries: {len(query_texts)}, Targets: {len(query_targets)}", file=sys.stderr, flush=True)

    all_top_indices, all_top_cosine_similarities = run_comparison(query_embeddings, args)
    rank_results = get_ranks(query_targets, all_top_indices, all_top_cosine_similarities, docid_to_index)

    plot_rank_difference(rank_results, rank_results_prompt, file_name)

    
if __name__ == "__main__":


    args = create_parser()
    if args.use_prompt is None:
        print("Prompt (--use-prompt) must be given. Use either 'hotpotqa', 'nq', or 'quora'.")
        exit(1)

    main(args)

    



    # argparser for model, dataset, and cache_dir
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--model", type=str, required=True)
    #parser.add_argument("--embeddings", type=str, required=True)
    #parser.add_argument("--corpus", type=str, required=True)
    #parser.add_argument("--queries", type=str, required=True)
   # parser.add_argument("--use-prompt", type=str, default=None)
    #parser.add_argument("--cache_dir", type=str, default="/scratch/project_2000539/jenna/hf-cache")
    #args = parser.parse_args()
