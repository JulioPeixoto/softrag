"""
Basic usage example of the softrag library.

This example demonstrates how to initialize softrag with OpenAI models,
add content from a web page, and perform a query.
"""

from softrag import Rag  
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def main():
    # Replace with your API key
    api_key = "your-api-key-here"
    
    # Initialize models
    chat = ChatOpenAI(model="gpt-4o", api_key=api_key)
    embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    
    # Create Rag instance
    rag = Rag(embed_model=embed, chat_model=chat)
    
    try:
        # Add content from a webpage
        rag.add_web(url="https://en.wikipedia.org/wiki/Python_(programming_language)")
        print("✅ Content successfully added!")
        
        # Perform a query
        query = "What are the main features of the Python programming language?"
        print(f"\nQuery: {query}")
        
        # Get the response
        response = rag.query(query)
        print(f"\nResponse:\n{response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 