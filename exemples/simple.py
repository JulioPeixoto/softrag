from softrag.softrag import Rag
from langchain_openai import ChatOpenAI, OpenAIEmbeddings   # :contentReference[oaicite:1]{index=1}


api_key="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"
chat = ChatOpenAI(model="gpt-4o", api_key=api_key)         # `.invoke()` pronto :contentReference[oaicite:2]{index=2}
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)                 # `.embed_query()` :contentReference[oaicite:3]{index=3}

rag = Rag(embed_model=embed, chat_model=chat)

# rag.add_file("txt.txt")
print(rag.query("Qual é idade de Julio, e seu jogador favorito?"))

