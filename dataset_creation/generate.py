from openai import OpenAI
import os
from dataset_prompts import source_prompt, target_prompt
import json
import random


API_KEY = open("/home/jmnybl/api_keys/openai", "rt").read().strip()
os.environ["OPENAI_API_KEY"] = API_KEY

def print_cost(usage, model_name):
    PRICE_PER_1M_TOKENS = {
        "gpt-5.4-mini": {
            "input": 0.75/1000000,
            "cache": 0.075/1000000,
            "output": 4.5/1000000,
        },
        "gpt-5.4-nano": {
            "input": 0.2/1000000,
            "cache": 0.02/1000000,
            "output": 1.25/1000000,
        },
        "gpt-5.4": {
            "input": 2.50/1000000,
            "cache": 0.25/1000000,
            "output": 15.00/1000000,
        },
        "gpt-5.5": {
            "input": 5.00/1000000,
            "cache": 0.50/1000000,
            "output": 30.00/1000000,
        }
    }
    cached_tokens = usage.input_tokens_details.cached_tokens
    input_tokens = usage.input_tokens - cached_tokens
    output_tokens = usage.output_tokens
    cost = PRICE_PER_1M_TOKENS[model_name]["input"] * input_tokens + PRICE_PER_1M_TOKENS[model_name]["cache"] * cached_tokens + PRICE_PER_1M_TOKENS[model_name]["output"] * output_tokens
    print(f"Usage: {input_tokens+cached_tokens} input tokens (cached: {cached_tokens}), {output_tokens} output tokens.")
    print(f"Cost: {cost}$")


def read_seed_texts():
    seeds = []
    with open("/home/jmnybl/random_data/seed_texts.txt", "rt") as f:
        for line in f:
            doc = json.loads(line)
            text = doc["text"]
            text = text.strip()
            seeds.append(text)
    return seeds


def generate_source_documents():

    # read seed texts
    seed_texts = read_seed_texts()

    ## prompt
    user_prompt = source_prompt.replace("[SEED TEXT]", random.choice(seed_texts))
    print("USER PROMPT:", user_prompt)

    ## model
    client = OpenAI()
    model_name = "gpt-5.5"

    response = client.responses.create(
    model = model_name,
    input = [{"role": "user", "content": user_prompt}],
    prompt_cache_key="source_doc_generation_prompt",
    )

    print(response.output_text)
    print_cost(response.usage, model_name)

    # save
    data = json.loads(response.output_text)
    with open("source_documents.jsonl", "at") as f:
        print(json.dumps(data), file=f)


def generate_target_documents(doc_id, text):

    # do not generate if already exists
    with open("target_documents.jsonl", "rt") as f:
        existing_target_documents = []
        for line in f:
            d = json.loads(line)
            id = d["source_id"]
            existing_target_documents.append(id)
    if doc_id in existing_target_documents:
        print(f"Skipping {doc_id} since it already exists.")
        return

    ## prompt
    user_prompt = target_prompt.replace("[SOURCE TEXT]", f"source_id: {doc_id}, text: {text}")
    print("USER PROMPT:", user_prompt)

    ## model
    client = OpenAI()
    model_name = "gpt-5.5"
    response = client.responses.create(
    model = model_name,
    input = [{"role": "user", "content": user_prompt}],
    prompt_cache_key="target_doc_generation_prompt",
    )

    print(response.output_text)
    print_cost(response.usage, model_name)

    
    # save
    data = json.loads(response.output_text)

    with open("target_documents.jsonl", "at") as f:
        print(json.dumps(data), file=f)
    return


def yield_source_documents():
    with open("source_documents.jsonl", "rt") as f:
        for i, line in enumerate(f):
            docs = json.loads(line)["source_documents"]
            for j, d in enumerate(docs):
                id = f"batch_{i}_doc_{j}"
                yield id, d

def main():

    for doc_id, text in yield_source_documents():
        print(doc_id, text)
        _ = generate_target_documents(doc_id, text)
        

main()