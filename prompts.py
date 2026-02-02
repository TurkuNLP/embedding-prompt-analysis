# place to store query prompts for model + dataset combinations


def get_prompt(model_name, dataset_name):
    # E5: https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
    # Qwen3: https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/task_prompts.json
    specified_prompts = {
        ("e5", "hotpotqa"): "Instruct: Given a multi-hop question, retrieve documents that can help answer the question\nQuery: ",
        ("e5", "nq"): "Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery: ",
        ("e5", "quora"): "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\nQuery: ",
        ("qwen3", "hotpotqa"): "Instruct: Given a multi-hop question, retrieve documents that can help answer the question\nQuery:",
        ("qwen3", "nq"): "Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery:",
        ("qwen3", "quora"): "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\nQuery:",

    }

    if (model_name, dataset_name) in specified_prompts:
        return specified_prompts[(model_name, dataset_name)]
    
    # model defaults
    if model_name == "e5":
        print("Prompt not found for (e5, {dataset_name}), using default prompt.")
        return f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    elif model_name == "qwen3":
        print("Prompt not found for (qwen3, {dataset_name}), using default prompt.")
        return f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    else:
        raise ValueError(f"Model {model_name} not supported")