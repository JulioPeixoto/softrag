"""
Example of using the softrag library with local models via Ollama.

This example demonstrates how to initialize softrag with local models
using Ollama for efficient local inference.
"""

import ollama
from softrag import Rag

class OllamaEmbeddings:
    """Wrapper for Ollama embedding model."""
    
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
    
    def embed_query(self, text):
        """Generate embedding for text using Ollama model."""
        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response['embedding']


class OllamaChat:
    """Wrapper for Ollama chat model."""
    
    def __init__(self, model_name="mistral"):
        self.model_name = model_name
    
    def invoke(self, prompt):
        """Generate response for a prompt using Ollama model."""
        response = ollama.chat(model=self.model_name, messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ])
        return response['message']['content']


def main():
    embed_model = OllamaEmbeddings()
    chat_model = OllamaChat()
    
    rag = Rag(embed_model=embed_model, chat_model=chat_model)
    
    try:
        rag.add_file("document.txt") 
        print("✅ Content successfully added!")
        
        question = "What is the main topic of this document?"
        print(f"\nQuestion: {question}")
        
        response = rag.query(question)
        print(f"\nResponse:\n{response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 