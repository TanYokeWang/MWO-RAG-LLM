import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import APIError, OpenAI

from maintenance_llm.utils.output_parser import output_parser
from maintenance_llm.utils.paths import BATCH_DIR
from maintenance_llm.utils.schema_builder import gpt_schema

load_dotenv()
api_key = os.getenv(
    "OPENAI_API_KEY"
)  # necessary only if api_key variable is not named "OPEN_AI_KEY"


class gpt_client:
    """Client class for Interacting with GPT API, including single prompt inference and batch processing"""

    def __init__(
        self,
        model: str,
        valid_labels: list[str],
        api_key: str | None = None,
        temperature: int = 0,
        max_output_token: int = 10,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.valid_labels = valid_labels
        self.schema = gpt_schema(valid_labels)
        self.temperature = temperature
        self.max_output_token = max_output_token

    def predict(self, prompt: str, json_format:bool = False, max_retries: int = 3, wait_time: int = 30):
        """Run inference for a single prompt""" 
        text_format = self.schema if json_format else None
        
        for i in range(max_retries):
            start_time = time.perf_counter()
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": prompt}],
                    text = text_format
                )
                inference_time = time.perf_counter() - start_time
                break

            except APIError as e:
                if i == max_retries - 1:
                    raise

                print(f"API error type: {type(e).__name__}")
                print(f"API error message: {e}")
                print(f"API error occurred. Retrying in {wait_time}s... \n")
                time.sleep(wait_time)

        raw_output = response.output_text
        final_label, status = output_parser(
            raw_output, self.valid_labels, json_format=json_format
        )

        prediction_output = {
            "prompt": prompt,
            "raw_output": raw_output,
            "predicted_label": final_label,
            "status": status,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_duration": inference_time,  # return value in seconds
        }
        return prediction_output

#------------------------------------
# *** batching functions *** 
#------------------------------------ 
    # input token calculator (optional) 
    def count_input_tokens(self, prompt: str, json_format: bool = False):
        """"execute request to GPT API to count input tokens for batch file splitting based on concurrent OpenAI Batch API token limit"""
        text_format = self.schema if json_format else None
        
        response = self.client.responses.input_tokens.count(
                            model=self.model,
                            input=[{"role": "user", "content": prompt}],
                            text = text_format
                        )
        
        return response.input_tokens

    # individual batch file creation function
    def create_batch_file(
        self,
        prompt_batch: dict,
        batch_number: int, # the count of batch splits generated based on concurrent OpenAI Batch API token limit 
        input_file_path: str | None = None, # optional - Insert specific file path to save batch file
        json_format: bool = False,
    ):
        """Create (default) and save batch file to local, return only file path for better control for created batch file"""

        if input_file_path is None:
            # Create output directory if it doesn't exist
            batch_dir_today = BATCH_DIR / datetime.now().strftime('%d-%m-%Y')
            batch_dir_today.mkdir(parents=True, exist_ok=True)
            
            input_file_path = (
                f"{batch_dir_today}/"
                f"batch_input_{batch_number}_{datetime.now().strftime('%d-%m-%Y_%H-%M')}.jsonl"
            )

        with open(input_file_path, "w") as f:
            for row in prompt_batch:
                wo_id = row["WO_ID"]
                prompt = row["prompt"]
                
                body = {
                    "model": self.model, 
                    "input": prompt
                    }
                
                if json_format:
                    body["text"] = self.schema

                record = {
                    "custom_id": f"{wo_id}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }

                f.write(json.dumps(record) + "\n")

        return input_file_path

    def submit_batch(self, batch_input_file: str, max_retries = 3, wait_time = 30):
        """Create submission batch file and submit to GPT API Batch for processing"""
        
        # Create submission batch file for {max_retries} attempts
        for attempts in range (1, max_retries + 1): 
            try:
                with open(batch_input_file, "rb") as f:
                    batch_file = self.client.files.create(file=f, purpose="batch")
                print("Batch file created: ", batch_file.id)
                break
            
            except Exception as e:
                print(f"Error batch file creation attempt{attempts}/{max_retries}: {e}\n Retrying in {wait_time}s... \n") 
                if attempts == max_retries:
                    raise
                
                time.sleep(wait_time)
            
        # Submit batch file for {max_retries} attempts    
        for attempts in range (1, max_retries +1): 
            try:
                batch = self.client.batches.create(
                    input_file_id=batch_file.id,
                    endpoint="/v1/responses",
                    completion_window="24h",  # Must be 24h at this time
                )
                print("Batch submitted: ", batch.id)
                break
            
            except Exception as e:
                print(f"Error batch file submission attempt{attempts}/{max_retries}: {e}\n Retrying in {wait_time}s... \n") 
                if attempts == max_retries:
                    raise
                
                time.sleep(wait_time)
            
        return batch

    def retrieve_batch_results(self, batch_number:int, batch_id: str):
        """Check batch status, retrieve completed batch file results and error file if available"""
        batch = self.client.batches.retrieve(batch_id)

        if batch.status == "completed":
            if batch.output_file_id:
                output_file = self.client.files.content(batch.output_file_id)

                # Save result to local
                batch_output_dir= BATCH_DIR / "batch_output"
                batch_output_dir.mkdir(parents=True, exist_ok=True)
                result_file_path = f"{batch_output_dir}/batch_output_{batch_number}_{batch_id}.jsonl"
                output_file.write_to_file(result_file_path)

                print(f"{batch_number}_{batch_id} completed. Batch results saved to: ", result_file_path)
            
            if batch.error_file_id:
                print("Failed batching request recorded... extracting failed output requests")
                error_file = self.client.files.content(batch.error_file_id)
                
                # Save result to local
                error_file_path = f"{batch_output_dir}/batch_errors_{batch_number}_{batch_id}.jsonl"
                error_file.write_to_file(error_file_path)

                print("Batch error file saved to: ", error_file_path)
        
        else:
            print("Batching status: ", batch.status)

        return batch
