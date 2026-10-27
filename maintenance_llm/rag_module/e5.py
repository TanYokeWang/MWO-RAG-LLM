from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from maintenance_llm.utils.paths import SAVE_CONTEXT_DIR


class e5_retriever:
    def __init__(self, context_file:list, record_retrieval:bool = False):
        print("Initiating embedding model e5-base-v2 and embed context data..")
        self.save = record_retrieval
        self.e5 = SentenceTransformer("intfloat/e5-base-v2") # Initialise the e5-base-v2 model
        self.context_db = context_file # loaded and compiled context database from json file
        
        # e5-base-v2 requires retrieval documents to be written as passage: context
        prefixed_context = [
            f"passage: {context}" for context in self.context_db
        ] 
        self.context_db_vector = self.e5.encode(prefixed_context)            
        

    def retrieve_context(self, input_text:list, top_k:int=3):
        # Embedding the full list of input text 
        print("Embedding Input texts in batch...")
        
        prefixed_text_list = [
            f"query: {text}" for text in input_text
        ]   

        input_db_vector = self.e5.encode(
            prefixed_text_list,
            batch_size=64,
            show_progress_bar=True,
            )
        
        # Calculate cosine similarities between the input text and the context database
        similarities = cosine_similarity(input_db_vector, self.context_db_vector)
        
        # get top-k context indices for each input text
        top_k_indices = np.argsort(similarities, axis=1)[:,::-1][:,:top_k]
        
        compiled_context_df = [
            {
                "input_text": input_text[input_idx],
                "retrieved_context": [self.context_db[context_idx] for context_idx in context_indices]
            } 
            for input_idx, context_indices in enumerate(top_k_indices)
        ]
        
        if self.save:
            # Map indices back to context
            save_retrieved_contexts = [
                {
                    "input_text": input_text[input_idx],
                    "retrieved_context": [
                        {
                            "context": self.context_db[context_idx],
                            "similarity_score": similarities[input_idx,context_idx]
                        }
                        for context_idx in context_indices
                    ]
                } 
                for input_idx, context_indices in enumerate(top_k_indices)
            ]
            
            # Save the retrieved contexts with similarity scoring to a CSV file
            df = pd.DataFrame(save_retrieved_contexts)
            df.to_csv(SAVE_CONTEXT_DIR / f"e5_retrieved_contexts_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv", index=False)

        return pd.DataFrame(compiled_context_df)
    