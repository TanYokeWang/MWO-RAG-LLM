from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from maintenance_llm.utils.paths import SAVE_CONTEXT_DIR


class tfidf_retriever:
    """Context embedding using TF-IDF vectorizer model"""
    
    def __init__(self, context_file:list, record_retrieval:bool = False):
        print("Initiating embedding model tfidfVectorizer and embed context data..")
        self.save = record_retrieval
        self.tfidf = TfidfVectorizer()  # Initialise the tfidf vectorizer
        self.context_db = context_file  # load compiled context file
        self.context_db_vector = self.tfidf.fit_transform(self.context_db) # Embedding the context database
        print ("Context file embedding complete. \n")


    def retrieve_context(self, input_text:list, top_k:int=3): 
        print("Embedding Input texts...")
        input_db_vector = self.tfidf.transform(input_text)    # Embedding the input text
            
        # Calculate cosine similarities between the input text and the context database
        similarities = cosine_similarity(input_db_vector, self.context_db_vector)
        top_k_indices = np.argsort(similarities)[:,::-1][:,:top_k]
            
        # Get top_k contexts
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
            df.to_csv(SAVE_CONTEXT_DIR / f"tfidf_retrieved_contexts_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv", index=False)

        print("Context retrieval complete using TF-IDF vectorizer. \n")

        return pd.DataFrame(compiled_context_df)

