"""
Basic usage example of the softrag library.

This example demonstrates how to initialize softrag with OpenAI models,
add content from a web page, and perform a query.
"""

from softrag import Rag 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def main():
    api_key = "your-api-key-here"
    
    chat = ChatOpenAI(model="gpt-4o", api_key=api_key)
    embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    
    rag = Rag(embed_model=embed, chat_model=chat)
    
    try:
        rag.add_web(url="https://en.wikipedia.org/wiki/Python_(programming_language)")
        print("✅ Content successfully added!")
        
        query = "What are the main features of the Python programming language?"
        print(f"\nQuery: {query}")
        
        response = rag.query(query)
        print(f"\nResponse:\n{response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 