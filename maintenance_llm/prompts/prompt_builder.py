import json

from maintenance_llm.utils.paths import PROMPT_FILE


def build_prompt (rag: bool, text: str, labels: str, retrieved_context:list = None) -> str: 
    # load prompt template from the JSON file
    with open(PROMPT_FILE, "r") as f:
        prompt_template = json.load(f)

    if rag: 
        # load RAG prompt template
        prompt_lines = prompt_template["RAG_prompt"]
        # Insert retrieved context, input text, and labels into the RAG prompt template
        prompt = "\n".join(prompt_lines).format(context=retrieved_context, text=text, labels=labels)

    else: 
        prompt_lines = prompt_template["base"]
        # insert input text and labels into the prompt template
        prompt = "\n".join(prompt_lines).format(text=text, labels=labels)

    return prompt