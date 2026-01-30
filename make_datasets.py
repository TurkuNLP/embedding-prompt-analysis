# a script to turn hf datasets into jsonl files
from datasets import load_dataset
import json
import argparse

def process_corpus(dataset_name, output_dir, cache_dir):
    dataset = load_dataset(dataset_name, "corpus", cache_dir=cache_dir)
    with open(f"{output_dir}/{dataset_name.replace('/', '_')}_corpus.jsonl", "w") as f:
        for item in dataset["corpus"]:
            d = {"id": item["_id"], "text": item["text"]}
            if "title" in item:
                d["text"] = item["title"] + " " + d["text"]
            print(json.dumps(d), file=f)

splits = {"mteb/hotpotqa": "dev", "mteb/nq": "test", "mteb/quora": "dev"}

def process_queries(dataset_name, output_dir, cache_dir):
    q_texts = {}
    dataset = load_dataset(dataset_name, "queries", cache_dir=cache_dir)
    for q in dataset["queries"]:
        q_texts[q["_id"]] = q["text"]

    qrels = load_dataset(dataset_name, "default", split=splits[dataset_name], cache_dir=cache_dir)
    final_questions = {}
    for item in qrels:
        if item["score"] != 1:
            print(f"Warning, score {item['score']} for query {item['query-id']} and corpus {item['corpus-id']}, skipping")
            continue
        q_id = item["query-id"]
        c_id = item["corpus-id"]
        if q_id not in final_questions:
            final_questions[q_id] = {"text": q_texts[q_id], "targets": []}
        final_questions[q_id]["targets"].append(c_id)

    with open(f"{output_dir}/{dataset_name.replace('/', '_')}_{splits[dataset_name]}_queries.jsonl", "w") as f:
        for q_id in final_questions.keys():
            d = {"id": q_id, "text": final_questions[q_id]["text"], "targets": final_questions[q_id]["targets"]}
            print(json.dumps(d), file=f)


def main(args):

    datasets = ["mteb/hotpotqa", "mteb/nq", "mteb/quora"]
    print(f"Processing datasets: {datasets}")
    print(f"Output directory: {args.output_dir}")
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        process_corpus(dataset, args.output_dir, args.cache_dir)
        process_queries(dataset, args.output_dir, args.cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="/scratch/project_2000539/jenna/hf-cache")
    args = parser.parse_args()

    main(args)