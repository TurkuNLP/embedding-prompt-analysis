import argparse
import json
import os
import sys
from pathlib import Path
from datasets import Dataset
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from compare_embeddings import cosine_similarity_normalized


K_VALUES = (1, 5)
SETTINGS = ("normal", "random_distractors", "paraphrase_distractors")


def load_examples(path):
    with open(path) as f:
        return json.load(f)


def to_qrels(examples, query_field="query", target_field="target"):
    """Build corpus, queries, and binary qrels from prepared SQuAD JSON rows."""
    
    # collect unique corpus first, make sure these are unique
    corpus_texts = set([e[target_field] for e in examples]) # collect unique target texts
    text_to_id = {text: i for i, text in enumerate(corpus_texts)}  # dictionary, from text to id

    queries = {"_id": [], "text": []}
    qrels = []
    for i, ex in enumerate(examples):
        qid = f"q{i}"
        queries["_id"].append(qid)
        queries["text"].append(ex[query_field])
        qrels.append({"query-id": qid, "corpus-id": text_to_id[ex[target_field]], "score": 1.0}) 

    corpus_texts = list(text_to_id.keys())
    corpus_ids = [text_to_id[t] for t in corpus_texts]
    corpus = {"_id": corpus_ids, "text": corpus_texts}
    return corpus, queries, qrels


def embed(texts, model, prompt=None):
    return model.encode(texts, prompt=prompt, normalize_embeddings=True)


def load_model(model_name_or_path):
    """Load a SentenceTransformer from a HF Hub id or a local checkpoint path."""
    path = os.path.expanduser(model_name_or_path)
    if os.path.isdir(path):
        path = os.path.abspath(path)
        print(f"Loading fine-tuned checkpoint from {path}...")
    else:
        print(f"Loading Hugging Face model {model_name_or_path}...")
        path = model_name_or_path
    return SentenceTransformer(path)


def add_distractors(corpus, corpus_embeddings, distractor_texts, id_prefix, model, label):
    """Append distractor texts to the corpus and return expanded corpus + embeddings."""
    distractor_ids = [f"{id_prefix}{i}" for i in range(len(distractor_texts))]
    expanded_corpus = {
        "_id": corpus["_id"] + distractor_ids,
        "text": corpus["text"] + distractor_texts,
    }
    print(f"Embedding {len(distractor_texts)} {label}...")
    distractor_embeddings = embed(distractor_texts, model)
    print("Shape before distractors:", corpus_embeddings.shape)
    expanded_embeddings = np.vstack([corpus_embeddings, distractor_embeddings])
    print("Shape after distractors:", expanded_embeddings.shape)

    docindex_to_docid = {}
    for index, doc_id in enumerate(expanded_corpus['_id']):
        docindex_to_docid[index] = doc_id


    return expanded_corpus, expanded_embeddings, docindex_to_docid


def sample_random_questions(examples, n, query_field, seed):
    all_queries = [ex[query_field] for ex in examples if ex.get(query_field)]
    rng = np.random.default_rng(seed)
    if len(all_queries) >= n:
        print(f"Sampling {n} random queries.")
        return list(rng.choice(all_queries, size=n, replace=False))
    print(
        f"Warning: only {len(all_queries)} random distractors available "
        f"(need {n}); using all of them."
    )
    return all_queries


def format_score(value):
    return f"{value*100:.2f}" if value is not None else "n/a"


def print_results(results):
    for k in K_VALUES:
        print(f"\nRecall@{k}")
        for setting in SETTINGS:
            name = setting.replace("_", " ").capitalize()
            print(f"{name}: {format_score(results[k][setting])}")
        print()

def compare_embeddings(query_embeddings, corpus_embeddings, top_k=5):
    if query_embeddings.__class__ != torch.Tensor:
        query_embeddings_torch = torch.from_numpy(query_embeddings)
    else:
        query_embeddings_torch = query_embeddings
    query_embeddings_torch = query_embeddings_torch.to(device="cuda" if torch.cuda.is_available() else "cpu")
    corpus_embeddings_torch = torch.from_numpy(corpus_embeddings).to(query_embeddings_torch.device)

    cosine_similarities = cosine_similarity_normalized(query_embeddings_torch, corpus_embeddings_torch)
    if top_k is None:
        top_cosine_similarities, top_indices = torch.topk(cosine_similarities, k=cosine_similarities.shape[1], dim=1, largest=True, sorted=True)
    else:
        top_cosine_similarities, top_indices = torch.topk(cosine_similarities, k=top_k, dim=1, largest=True, sorted=True)
    return top_cosine_similarities, top_indices



def eval_recall(query_embeddings, corpus_embeddings, qrels, queryindex_to_queryid, docindex_to_docid, top_k):

    all_top_cosine_similarities, all_top_indices = compare_embeddings(query_embeddings, corpus_embeddings, top_k=top_k)
    all_top_indices = all_top_indices.cpu()

    tp, fn = 0, 0
    
    for query_index, (top_indices, top_cosine_similarities) in enumerate(zip(all_top_indices, all_top_cosine_similarities)):
        query_id = queryindex_to_queryid[query_index]
        retrieved_documents = [docindex_to_docid[i.item()] for i in top_indices]
        assert qrels[query_index]["query-id"] == query_id
        target_doc = qrels[query_index]["corpus-id"]
        if target_doc in retrieved_documents:
            tp += 1
        else:
            fn += 1

    recall = tp / (tp + fn)

    return recall


def debug_retrieval(query_embeddings, corpus_embeddings, query_texts, corpus_texts, qrels):

    all_top_cosine_similarities, all_top_indices = compare_embeddings(query_embeddings, corpus_embeddings, top_k=5)
    all_top_indices = all_top_indices.cpu()
    
    for query_index, (top_indices, top_cosine_similarities) in enumerate(zip(all_top_indices, all_top_cosine_similarities)):
        query_text = query_texts[query_index]
        correct_doc_index = qrels[query_index]["corpus-id"]
        retrieved_indeces = [i.item() for i in top_indices]
        retrieved_texts = [corpus_texts[i] for i in retrieved_indeces]
        print("Q:", query_text)
        print("Correct target:", correct_doc_index, corpus_texts[correct_doc_index])
        for ri, rt in zip(retrieved_indeces, retrieved_texts):
            print(f"  {ri}: {rt}")
        print()



def main(args):

    ## prepare data
    examples = load_examples(args.eval_file)
    corpus, queries, qrels = to_qrels(examples, query_field=args.query_field, target_field=args.target_field)
    docindex_to_docid = {}
    for index, doc_id in enumerate(corpus['_id']):
        docindex_to_docid[index] = doc_id

    queryindex_to_queryid = {}
    for index, query_id in enumerate(queries['_id']):
        queryindex_to_queryid[index] = query_id

    print(f"Loaded {len(corpus['_id'])} targets and {len(queries['_id'])} queries.")

    ## load model and embed
    model = load_model(args.model)

    print(f"Embedding {len(corpus['text'])} corpus texts without a prompt...")
    corpus_embeddings = embed(corpus["text"], model)

    PROMPT = f"Instruct: {args.prompt}\nQuery: "
    print(f"Embedding {len(queries['text'])} queries with prompt: {PROMPT}")
    query_embeddings = embed(queries['text'], model, prompt=PROMPT)
    
    
    ## EVAL
    results = {k: {setting: None for setting in SETTINGS} for k in K_VALUES}
    for k in K_VALUES:
        recall = eval_recall(query_embeddings, corpus_embeddings, qrels, queryindex_to_queryid, docindex_to_docid, k)
        print(f"R@{k}", recall)
        results[k]["normal"] = recall

    if args.debug:
        debug_retrieval(query_embeddings, corpus_embeddings, queries['text'], corpus['text'], qrels)

    

    if args.random_distractors:
        random_examples = load_examples(args.random_distractors)
        random_distractors = sample_random_questions(random_examples, len(queries['_id']), args.query_field, args.seed)

        random_expanded_corpus, random_expanded_embeddings, random_expanded_docindex_to_docid = add_distractors(
            corpus,
            corpus_embeddings,
            random_distractors,
            id_prefix="r",
            model=model,
            label="random question distractors",
        )

        for k in K_VALUES:
            recall = eval_recall(query_embeddings, random_expanded_embeddings, qrels, queryindex_to_queryid, random_expanded_docindex_to_docid, k)
            print(f"Random distr. R@{k}", recall)
            results[k]["random_distractors"] = recall

        if args.debug:
            debug_retrieval(query_embeddings, random_expanded_embeddings, queries['text'], random_expanded_corpus['text'], qrels)


    # paraphrase distractors
    paraphrases = [ex["generated_paraphrase"] for ex in examples]

    para_expanded_corpus, para_expanded_embeddings, para_expanded_docindex_to_docid = add_distractors(
        corpus,
        corpus_embeddings,
        paraphrases,
        id_prefix="d",
        model=model,
        label="paraphrase distractors",
    )

    for k in K_VALUES:
        recall = eval_recall(query_embeddings, para_expanded_embeddings, qrels, queryindex_to_queryid, para_expanded_docindex_to_docid, k)
        print(f"Paraphrase distr. R@{k}", recall)
        results[k]["paraphrase_distractors"] = recall

    if args.debug:
        debug_retrieval(query_embeddings, para_expanded_embeddings, queries['text'], para_expanded_corpus['text'], qrels)

    ## print results
    print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3 embedding model on SQuAD retrieval.")
    parser.add_argument("--eval_file", required=True, help="Prepared eval JSON file.")
    parser.add_argument("--query_field", default="query", help="Query field, default: query")
    parser.add_argument("--target_field", default="target", help="Target field, default: target")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to use, will be inserted to 'Instruct: [prompt]\nQuery: ' template.")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="HF Hub model id or path to a fine-tuned SentenceTransformer checkpoint.")
    parser.add_argument("--random-distractors", default="", help="JSON file with random-query distractors (same format as eval data, will read query_field).")
    parser.add_argument("--debug", action="store_true", default=False, help="Print debug output.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random distractor sampling (default: 42).")
    main(parser.parse_args())


# PROMPTS

# QA
# "Given a question, retrieve Wikipedia passages that answer the question"

# TATOEBA
# "Retrieve a [LANG] translation."


# EXAMPLES

# Tatoeba
# python eval_retrieve_eval.py --eval_file tatoeba-mteb/tatoeba_test_fin-eng.json --query_field "query" --target_field "target" --prompt "Retrieve a Finnish translation."

# SQuAD
# python eval_retrieve.py --eval_file squad_v1.1/train-splits/train-dev.json --query_field "question" --target_field "answer_paragraph" --prompt "Given a question, retrieve Wikipedia passages that answer the question" --random-distractors squad_v1.1/train-splits/train-test.json