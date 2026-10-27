from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from maintenance_llm.utils.paths import SAVE_CONTEXT_DIR


class minilm_retriever:
    """Context embedding using MiniLM model"""
    
    def __init__(self, context_file:list, record_retrieval:bool = False):
        print("Initiating embedding model MiniLM and embed context data..")
        self.save = record_retrieval
        self.minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") # Initialise the MiniLM model
        self.context_db = context_file  # load compiled context file
        self.context_db_vector = self.minilm.encode(self.context_db)   # Pre-embedding the context database to reduce runtime
        print ("Context file embedding complete. \n")

    def retrieve_context(self, input_text:list, top_k:int=3):
        # Embedding the full list of input text 
        print("Compiling WO texts and retrieved context...")
        input_db_vector = self.minilm.encode(
            input_text,
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
            df.to_csv(SAVE_CONTEXT_DIR / f"minilm_retrieved_contexts_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv", index=False)

        print("Context retrieval complete using MiniLM. \n")

        return pd.DataFrame(compiled_context_df)
    