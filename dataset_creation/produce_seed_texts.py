import random
import json


seed_texts = []

# HPLT v2.0 English sample
with open ("/home/jmnybl/random_data/eng_Latn.shuf", "rt") as f:
    for line in f:
        doc = json.loads(line)
        text = doc["text"]
        text = text.strip().replace("\n", " ")
        length = random.randint(600, 900)
        if len(text) < length:
            continue
        start_index = random.randint(0, len(text) - length)
        chunk = text[start_index:start_index+length]
        seed_texts.append(chunk)

with open("/home/jmnybl/random_data/seed_texts.txt", "wt") as f:
    for text in seed_texts:
        print(json.dumps({"text": f"...{text}..."}), file=f)