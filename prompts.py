# place to store query prompts for model + dataset combinations


def get_prompt(model_name, dataset_name):
    # intfloat/multilingual-e5-large-instruct: https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
    # Qwen/Qwen3-Embedding-0.6B: https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/task_prompts.json
    # KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5: https://arxiv.org/pdf/2506.20923
    specified_prompts = {
        ("intfloat/multilingual-e5-large-instruct", "hotpotqa"): "Instruct: Given a multi-hop question, retrieve documents that can help answer the question\nQuery: ",
        ("intfloat/multilingual-e5-large-instruct", "nq"): "Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery: ",
        ("intfloat/multilingual-e5-large-instruct", "quora"): "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\nQuery: ",
        ("Qwen/Qwen3-Embedding-0.6B", "hotpotqa"): "Instruct: Given a multi-hop question, retrieve documents that can help answer the question\nQuery:",
        ("Qwen/Qwen3-Embedding-0.6B", "nq"): "Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery:",
        ("Qwen/Qwen3-Embedding-0.6B", "quora"): "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\nQuery:",
        ("KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5", "hotpotqa"): "Instruct: Given a query, retrieve documents that answer the query \n Query: ",
        ("KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5", "nq"): "Instruct: Given a query, retrieve documents that answer the query \n Query: ",
        ("KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5", "quora"): "Instruct: Retrieve semantically similar questions \n Query: ",
    }

    if (model_name, dataset_name) in specified_prompts:
        return specified_prompts[(model_name, dataset_name)]
    
    # model defaults
    if model_name == "intfloat/multilingual-e5-large-instruct":
        print("Prompt not found for (intfloat/multilingual-e5-large-instruct, {dataset_name}), using default prompt.")
        return f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    elif model_name == "Qwen/Qwen3-Embedding-0.6B":
        print("Prompt not found for (Qwen/Qwen3-Embedding-0.6B, {dataset_name}), using default prompt.")
        return f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    elif model_name == "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5":
        print("Prompt not found for (KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5, {dataset_name}), using default prompt.")
        return f"Instruct: Given a query, retrieve documents that answer the query \n Query: "
    else:
        raise ValueError(f"Model {model_name} not supported")