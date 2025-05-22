import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from softrag import Rag 


load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(model="gpt-4o", api_key=api_key, streaming=True)
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

rag = Rag(embed_model=embed, chat_model=chat)
rag.add_file("softrag_llm.md")
#rag.add_web(url="https://pt.wikipedia.org/wiki/Python")

for chunk in rag.query("how can i integrate softrag with chatgpt?", stream=True):
    print(chunk, end="", flush=True)    
