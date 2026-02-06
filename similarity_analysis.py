from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from prompts import get_prompt
import sys


class ModelWrapper(object):
    def __init__(self, model_name, cache_dir):
        self.cache_dir = cache_dir
        self.model_name = model_name
        print("Loading model:", model_name, file=sys.stderr, flush=True)
        self.model = SentenceTransformer(model_name, cache_folder=self.cache_dir)
        self.encode_fn = self.model.encode


    def embed_texts(self, texts, prompt=None):
        if prompt:
            texts = [prompt + text for text in texts]
        return self.encode_fn(texts, normalize_embeddings=True)


class DataWrappper(object):
    def __init__(self, dataset_name, corpus_file, queries_file):
        self.dataset_name = dataset_name
        self.pairs = self.load_data(corpus_file, queries_file)
        self._report() # report what was loaded

    def load_data(self, corpus_file, queries_file):
        with open(corpus_file, "r") as f:
            corpus = {}
            for line in f:
                d = json.loads(line)
                corpus[d["id"]] = d["text"]
        pairs = []
        with open(queries_file, "r") as f:
            for line in f:
                d = json.loads(line)
                for target in d["targets"]:
                    pairs.append((d["text"], corpus[target]))
        return pairs

    def _report(self):
        print(f"Loaded {len(self.pairs)} pairs for {self.dataset_name}")
        print(f"First example: {self.pairs[0]}")


def make_plot(array, file_name):

    # create directory if it doesn't exist
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    file_path = os.path.join("figures", file_name)

    dimension_indices = np.arange(len(array))

    plt.figure(figsize=(15, 6))
    plt.vlines(dimension_indices, 0, array, colors='blue', lw=0.7, alpha=0.7)
    plt.title('Delta Similarity for Each Example', fontsize=18)
    plt.xlabel('Example Index', fontsize=16)
    plt.ylabel('Delta Similarity', fontsize=16)
    plt.ylim(-0.30, 0.30)  # Fix y-axis limits as requested
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=0)
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Saved plot to", file_path)



def get_pairwise_similarities(model, embeddings1, embeddings2):
    similarities = []
    for i in range(embeddings1.shape[0]):
        s = cosine_similarity(embeddings1[i].reshape(1, -1), embeddings2[i].reshape(1, -1))
        similarities.append(s)
    return similarities


def plot_sim_difference(model, data):
    prompt = get_prompt(model.model_name, data.dataset_name)
    print(f"Query prompt:\n{prompt}")
    target_embeddings = model.embed_texts([t for i,t in data.pairs], prompt=None)
    query_embeddings = model.embed_texts([q for q,t in data.pairs], prompt=None)
    query_prompt_embeddings = model.embed_texts([q for q,t in data.pairs], prompt=prompt)

    pairwise_similarities_plain = get_pairwise_similarities(model, target_embeddings, query_embeddings)
    pairwise_similarities_prompt = get_pairwise_similarities(model, target_embeddings, query_prompt_embeddings)
    differences = [b-a for a,b in zip(pairwise_similarities_plain, pairwise_similarities_prompt)]
    print(f"Mean difference: {np.mean(differences)}")
    print(f"Positive difference: {np.sum(np.array(differences) > 0)}")
    print(f"Negative difference: {np.sum(np.array(differences) < 0)}")

    make_plot(differences, f"{model.model_name.split('/')[-1]}_{data.dataset_name.split('/')[-1]}_sim_difference.png")

def main(args):

    model = ModelWrapper(args.model, args.cache_dir)
    data = DataWrappper(args.dataset, args.corpus, args.queries)
    plot_sim_difference(model, data)


if __name__ == "__main__":

    # argparser for model, dataset, and cache_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["intfloat/multilingual-e5-large-instruct", "Qwen/Qwen3-Embedding-0.6B", "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"])
    parser.add_argument("--dataset", type=str, required=True, choices=["hotpotqa", "nq", "quora"])
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="/scratch/project_2000539/jenna/hf-cache")
    args = parser.parse_args()


    main(args)