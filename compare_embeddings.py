import pickle
import json
import more_itertools
import argparse
import numpy as np
import torch
import sys

def yield_embeddings(file_prefix):
    with open(f"{file_prefix}.embeddings.pkl", "rb") as f_emb,\
        open(f"{file_prefix}.examples.jsonl", "rt") as f_meta:
        while True:
            try:
                embedding = pickle.load(f_emb)
                metadata = json.loads(f_meta.readline())
            except EOFError:
                break
            yield {"metadata": metadata, "embedding": embedding}

def batch_embeddings(embeddings_generator, batch_size=None):
    """
    `embeddings_generator` is a generator that yields dictionaries with `"metadata"` and `"embedding"` keys; 
    it is assumed that the embeddings are already in numpy array format.

    Yield batches of metadata and embeddings from the generator.
    If batch_size is None, yields everything as a single batch.
    """
    if batch_size is None:
        all_meta_and_embeddings = list(embeddings_generator)
        all_metadata = [example["metadata"] for example in all_meta_and_embeddings]
        all_embeddings = [example["embedding"] for example in all_meta_and_embeddings]
        yield all_metadata, np.vstack(all_embeddings)
    else:
        for example_batch in more_itertools.chunked(embeddings_generator, batch_size):
            metadata_batch = [example["metadata"] for example in example_batch]
            embedding_batch = [example["embedding"] for example in example_batch]
            yield metadata_batch, np.vstack(embedding_batch)

def cosine_similarity_normalized(tensor1, tensor2):
    # Compute norms along each row
    norms1 = tensor1.norm(dim=1, keepdim=True)
    norms2 = tensor2.norm(dim=1, keepdim=True)
    # Normalize data
    tensor1_normalized = tensor1 / norms1
    tensor2_normalized = tensor2 / norms2
    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(tensor1_normalized, tensor2_normalized.T)
    return similarity_matrix

def compare_embeddings(large_embeddings_batches, query_embeddings):
    """
    `large_embeddings_batches` is a generator that yields batches of metadata and embeddings from the large set of embeddings;
    `query_embeddings` is a numpy array of query embeddings.
    """
    query_embeddings_torch = torch.from_numpy(query_embeddings).to(device="cuda" if torch.cuda.is_available() else "cpu")
    top_indices_accumulator = [] #accumulates the top 1 indices for each query embedding in each batch
    top_cosine_similarities_accumulator = [] #accumulates the top 1 cosine similarities for each query embedding in each batch
    batch_index_offset_counter = 0
    for large_metadata_batch, large_embeddings_batch in large_embeddings_batches:
        emb_batch_torch = torch.from_numpy(large_embeddings_batch).to(query_embeddings_torch.device)
        cosine_similarities = cosine_similarity_normalized(query_embeddings_torch, emb_batch_torch)
        #cosine_similarities is a matrix of shape (query_embeddings.shape[0], large_embeddings_batch.shape[0])
        #find the indices of the top 1 cosine similarities for each query embedding
        top_cosine_similarities, top_indices = torch.topk(cosine_similarities, k=1, dim=1)
        top_indices_accumulator.append(top_indices + batch_index_offset_counter)
        top_cosine_similarities_accumulator.append(top_cosine_similarities)
        batch_index_offset_counter += len(large_embeddings_batch)
    #Now we take top-1 across the per-batch maxima
    top_indices_accumulator = torch.cat(top_indices_accumulator, dim=1)
    top_cosine_similarities_accumulator = torch.cat(top_cosine_similarities_accumulator, dim=1)
    top_cosine_similarities_global,top_indices_into_accumulator = torch.topk(top_cosine_similarities_accumulator, k=1, dim=1)
    top_indices_global = top_indices_accumulator.gather(1, top_indices_into_accumulator)
    return top_indices_global, top_cosine_similarities_global

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Compare embeddings of two datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--large-embeddings",
        type=str,
        required=True,
        help="Path prefix for the large set of embeddings within which we seek the nearest neighbors. Will be loaded in batches."
    )
    parser.add_argument(
        "--query-embeddings",
        type=str,
        required=True,
        help="Path prefix for the set of embeddings which acts as a query. Will be loaded fully into GPU memory."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch size for comparing embeddings"
    )
    args = parser.parse_args()

    # File-based comparison
    # 1) Load the query embeddings
    query_examples = list(yield_embeddings(args.query_embeddings))
    query_embeddings = np.vstack([example["embedding"] for example in query_examples])
    # 2) Load the large embeddings in batches
    large_embeddings_generator = yield_embeddings(args.large_embeddings)
    large_embeddings_batches = batch_embeddings(large_embeddings_generator, args.batch_size)
    # 3) Compare the embeddings
    top_indices, top_cosine_similarities = compare_embeddings(large_embeddings_batches, query_embeddings)
    
    print("Shape of top_indices:", top_indices.shape, file=sys.stderr, flush=True)
    print("Shape of top_cosine_similarities:", top_cosine_similarities.shape, file=sys.stderr, flush=True)
    print("First 10 top_indices (indices of nearest neighbors in large_embeddings):", file=sys.stderr, flush=True)
    print(top_indices[:10], file=sys.stderr, flush=True)
    print("First 10 top_cosine_similarities (cosine similarities for each nearest neighbor):", file=sys.stderr, flush=True)
    print(top_cosine_similarities[:10], file=sys.stderr, flush=True)