import json

import pandas as pd

"""parser function to validate and extract the predicted label from the raw output of the LLM response"""

def output_parser(raw_output: str, valid_labels: list, json_format: bool): 
    
    if json_format:
        try:
            output_json = json.loads(raw_output)
            predicted_label = output_json.get("predicted_label", "Review_Required").strip() #Code falls back to "Review Required" if the predicted label is not found in the JSON response

        except json.JSONDecodeError:
            return "Review_Required", "Parsing_Error"

    else: 
        predicted_label = raw_output.strip()    # extract raw output from the response 
        
    # handle invalid labels 
    if predicted_label not in valid_labels:
        return "Review_Required", "Invalid"
    
    else:
        return predicted_label, "Valid"


def batch_parser(records:list, dataset: pd.DataFrame, json_format:bool): 
    valid_labels = dataset["Failure Category"].unique().tolist()
    
    extracted_batch_results = []
    for record in records:
        custom_id = record["custom_id"]
        response_body = record["response"]["body"] 
        
        raw_output = None
        
        for output in response_body["output"]:
            if output.get("type") == "message":
                for content in output.get("content", []):
                    if content.get("type") == "output_text":
                        raw_output = content["text"]
                        break
        
                final_label, status = output_parser(
                raw_output, valid_labels, json_format=json_format
            )
        
        usage = response_body["usage"]
        input_token = usage["input_tokens"]
        output_token = usage["output_tokens"]
        reason_token = usage["output_tokens_details"]["reasoning_tokens"]
        
        extracted_batch_results.append({
            "Work Order ID": custom_id,
            "raw_output": raw_output,
            "predicted_label": final_label,
            "status": status,
            "input_tokens" : input_token,
            "output_tokens": output_token,
            "reason_tokens": reason_token
        })
    
    extracted_batch_results_df = pd.DataFrame(extracted_batch_results)

    # Align and insert corresponding True Label and Input text
    compiled_result_df = extracted_batch_results_df.merge(
        dataset[
            ["Work Order ID", "ServiceOrderText", "Failure Category"]
        ],
        on = "Work Order ID",
        how = "left"
    )

    # rearrange column based on desired arrangement
    compiled_result_df = compiled_result_df[
        [
            "Work Order ID",
            "ServiceOrderText",
            "Failure Category",
            "raw_output",
            "predicted_label",
            "status",
            "input_tokens",
            "output_tokens",
            "reason_tokens"
        ]
    ]

    final_df = compiled_result_df.rename(columns = {"Failure Category" : "True_label"})

    return final_df
    