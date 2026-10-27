import os

import ollama

from maintenance_llm.utils.output_parser import output_parser
from maintenance_llm.utils.schema_builder import ollama_schema

endpoint_url = os.getenv("OLLAMA_ENDPOINT_URL")


class ollama_client:
    """Client class for Interacting with Ollama API."""

    def __init__(
        self,
        model: str,
        valid_labels: list[str],
        endpoint_url=endpoint_url,
        temperature: int = 0,
        max_output_token: int = 10,
    ):
        self.client = ollama.Client(host=endpoint_url)
        self.model = model
        self.valid_labels = valid_labels
        self.schema = ollama_schema(valid_labels)
        self.temperature = temperature
        self.max_output_token = max_output_token

    def predict(self, prompt: str, json_format: bool = False):
        """Run inference for a single prompt"""
        _format = self.schema if json_format else None
        _max_token = 30 if json_format else self.max_output_token
        
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat(
            model=self.model,
            messages=messages,
            think=False,
            format=_format,
            options={
                "temperature": self.temperature,
                "num_predict": _max_token,
            },
        )

        raw_output = response["message"]["content"].strip()

        final_label, status = output_parser(
            raw_output, self.valid_labels, json_format=json_format
        )

        # Store the full output for debugging purposes
        prediction_output = {
            "prompt": prompt,
            "raw_output": raw_output,
            "predicted_label": final_label,
            "status": status,
            "input_tokens": response["prompt_eval_count"],
            "output_tokens": response["eval_count"],
            "total_duration": response["total_duration"]
            / 1e9,  # return value in seconds
            "prompt_eval_duration": response["prompt_eval_duration"]
            / 1e9,  # return value in seconds
            "eval_duration": response["eval_duration"] / 1e9,  # return value in seconds
        }

        return prediction_output
