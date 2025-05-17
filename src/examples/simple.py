"""
Basic usage example of the softrag library.

This example demonstrates how to initialize softrag with OpenAI models,
add content from a web page, and perform a query.
"""

from src.softrag.softrag import Rag 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


api_key = "your-api-key-here"

chat = ChatOpenAI(model="gpt-4o", api_key=api_key, streaming=True)
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

rag = Rag(embed_model=embed, chat_model=chat)

rag.add_web(url="https://en.wikipedia.org/wiki/Python_(programming_language)")

for chunk in rag.query("What are the main features of the Python programming language?", stream=True):
    print(chunk, end="", flush=True) 
    