
ACTION_INSTRUCTIONS = {
    "translate": "Retrieve a Spanish translation.",
    "summary": "Retrieve a very concise TL;DR summary.",
    "paraphrase": "Retrieve texts that are semantically equivalent.",
    "style": "Retrieve semantically equivalent text in poetic / lyrical style.",
    "qa": "Given a question, retrieve a direct answer.",
    "clarify": "Given a text, retrieve a clarifying question for it.",
    "example": "Given a text, retrieve an illustrative, easy to understand example for it.",
    "reaction": "Given a text, retrieve a natural reader reaction for it.",
}


def format_e5_query_prompt(action):
    """Return the E5-instruct query prefix for a dataset action label."""
    instruction = ACTION_INSTRUCTIONS[action]
    return f"Instruct: {instruction}\nQuery: "


dataset_description = """
# Embedding dataset
## Dataset description

This dataset evaluates embedding models on instruction-conditioned retrieval / instruction-sensitive similarity matching.

Each data point consists of an original query text S, an action/instruction A, and a transformed target T. For the same \
source text, multiple actions may be applied, and each action produces a distinct target.

f(S, A_1) → T_1
f(S, A_2) → T_2 ...

The dataset tests whether embedding models can distinguish between different intended targets of the same source content \
based on the given task-specific instruction. Given a retrieval query consisting of (source text, instruction), the \
model must retrieve the correct target text from a pooled candidate index. The candidate pool includes both:

1.  targets produced from the same source under different instructions, and 
2.  targets produced under the same instruction for different sources. 

The candidate set is a static index created by embedding f(T) to support efficient retrieval on large-scale candidate sets. \
This tests whether the model uses both the source semantics and the instruction, rather than relying only on either one.

## Actions

Each action is annotated with

    * Applicability: Whether the action applies to all possible source texts or only to a subset.
    * Conflicts to avoid: Whether the output for this action could overlap with the output of another action.

### Basic actions (direct semantic transformations)

* translate: Translate the source text to Spanish.
    * Instruction: Retrieve a Spanish translation.
    * Applicability: Always
    * Conflicts to avoid: None
* summary: Write a very concise summary (TL;DR) while preserving the main meaning.
    * Instruction: Retrieve a very concise TL;DR summary.
    * Applicability: Only when the source text is not super short.
    * Conflicts to avoid: The summary should be short enough to clearly distinguish it from a paraphrase. If the source is a \
question, the output should not answer it, to avoid overlap with QA.
* paraphrase: Write a paraphrase of the source text while preserving full meaning and style.
    * Instruction: Retrieve texts that are semantically equivalent.
    * Applicability: Always
    * Conflicts to avoid: The paraphrase should preserve all content and the general style of the original text.
* style: Write the source text in a strong, distinct poetic / lyrical style while preserving full meaning.
    * Instruction: Retrieve semantically equivalent text in poetic / lyrical style. 
    * Applicability: Always
    * Conflicts to avoid: The style change should be substantial enough to distinguish it from a paraphrase.


### Indirect actions (source-grounded response)

* qa: Write a direct answer for the question in the source text.
    * Instruction: Given a question, retrieve a direct answer.
    * Applicability: Only when the source text is a question.
    * Conflicts to avoid: None
* clarify: Ask a clarifying question of what was meant by the source text.
    * Instruction: Given a text, retrieve a clarifying question for it.
    * Applicability: Always
    * Conflicts to avoid: The output should be phrased as a question and should not resemble a reader reaction.
* example: Turn the source text into an illustrative, easy-to-understand example
    * Instruction: Given a text, retrieve an illustrative, easy to understand example for it.
    * Applicability: Only when the source text has enough content to explain.
    * Conflicts to avoid: None
* reaction: Write a possible reader reaction tightly connected to the source text.
    * Instruction: Given a text, retrieve a natural reader reaction for it.
    * Applicability: Always
    * Conflicts to avoid: The output should not be phrased as a question, to avoid overlap with clarifying questions. It should also tightly \
connect to the source text to make other source texts semantically invalid.
"""





source_prompt = f"""Your task is to generate 10 random texts, each about 1-3 sentences long. \
Each text should be written in English, and they should be diverse in content and style. Ensure that some of the texts are factual, and \
keep the length and style varied, not all should be single sentence. Include also one that is formatted as a question. \
Use only people names which come from the seed text. Generate texts which could appear somewhere, not completely invented ones. \
To enhance diversity, \
I provide you a short seed text which you should use as inspiration and the generated texts should somehow relate to it. \
However, the inspiration should not be too obvious and you do not need to follow the given text explicitly. Rather, \
the inspiration can be general topic, style or just a single word from the seed text. Return generated texts as json list, \
with the key "source_documents", which is a list of generated texts.

Seed text: [SEED TEXT]
"""



target_prompt = f"""I will provide you a detailed description of a dataset I would like to create, and ask you to \
generate transformed target documents for a given source document based on the dataset definition.

{dataset_description}

For each given source document, generate transformed target documents based on all 8 actions defined in the dataset definition. \
For each action, carefully consider the 'applicability' and 'conflicts to avoid' annotations. The transformed targets should not \
conflict with each other. If an action is not applicable to the given source document, leave the target document empty. \
The transformation should be strong enough to clearly distinguish it from the source document, while keeping the semantic relation intact.

Return the data as a dictionary defined below. 

    * source_id: int
    * source_text: string
    * targets: [
        * action: label of the action (string)
        * target_text: string

Source document: [SOURCE TEXT]
"""


