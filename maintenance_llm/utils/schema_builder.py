def ollama_schema(valid_labels: list[str]) -> dict:
    """
    Build the inference schema for the Ollama model.

    Args:
        valid_labels (list): List of valid labels for classification.
        input_text (str): The input text to be classified.

    Returns:
        dict: A dictionary representing the inference schema.
    """
    
    schema = {
            "type": "object",
            "properties": {
                "predicted_label": {
                    "type": "string",
                    "enum": valid_labels
                }
            },
            "required": ["predicted_label"]
            }
    
    return schema

def gpt_schema(valid_labels: list[str]) -> dict:
    """
    Build the inference schema for the GPT model.

    Args:
        valid_labels (list): List of valid labels for classification.
        input_text (str): The input text to be classified.

    Returns:
        dict: A dictionary representing the inference schema.
    """
    schema={
        "format":{
            "type" : "json_schema",
            "name" : "formatted_prediction",
            "strict" : True, 
            "schema" : {
                    "type": "object",
                    "properties": {
                        "predicted_label": {
                            "type": "string",
                            "description": "The predicted label for the input text.",
                            "enum": valid_labels
                        }
                    },
                    "required": ["predicted_label"],
                    "additionalProperties": False,
                    },
            },
        }

    return schema

