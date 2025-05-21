"""
Basic usage example of the softrag library.

This example demonstrates how to initialize softrag with OpenAI models,
add content from a web page, and perform a query.
"""

from softrag import Rag 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(model="gpt-4o", api_key=api_key, streaming=True)
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

rag = Rag(embed_model=embed, chat_model=chat)
#rag.add_web(url="https://pt.wikipedia.org/wiki/Python")
rag.add_file("softrag_llm.md")

for chunk in rag.query("how can i integrate softrag with chatgpt?", stream=True):
    print(chunk, end="", flush=True) 
    
