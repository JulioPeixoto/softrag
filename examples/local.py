"""
Example of using the softrag library with local models.

This example demonstrates how to initialize softrag with local Transformers models,
without depending on cloud services.
"""

import torch
from softrag import Rag
from transformers import AutoTokenizer, AutoModel
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings


class LocalEmbeddings:
    """Wrapper for local embedding model."""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = HuggingFaceEmbeddings(model_name=model_name)
    
    def embed_query(self, text):
        """Generate embedding for text using local model."""
        return self.model.embed_query(text)


class LocalChat:
    """Wrapper for local chat model."""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        self.model = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 512}
        )
    
    def invoke(self, prompt):
        """Generate response for a prompt using local model."""
        return self.model.invoke(prompt)


def main():
    # Initialize local models
    embed_model = LocalEmbeddings()
    chat_model = LocalChat()
    
    # Create Rag instance
    rag = Rag(embed_model=embed_model, chat_model=chat_model)
    
    # Add content
    try:
        rag.add_file("document.txt")  # Replace with your file
        print("✅ Content successfully added!")
        
        # Perform a query
        question = "What is the main topic of this document?"
        print(f"\nQuestion: {question}")
        
        response = rag.query(question)
        print(f"\nResponse:\n{response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 