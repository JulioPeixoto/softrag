# Examples

Each script runs on its own. The first two need nothing but softrag.

| Example | What it shows | Needs |
| --- | --- | --- |
| [`quickstart.py`](quickstart.py) | Indexing, hybrid search, generation, management | nothing |
| [`filtering.py`](filtering.py) | The metadata filter DSL, end to end | nothing |
| [`openai_rag.py`](openai_rag.py) | OpenAI embeddings and chat, with streaming | `pip install 'softrag[openai]'`, `OPENAI_API_KEY` |
| [`local_ollama.py`](local_ollama.py) | Fully offline: local embeddings and generation | Ollama, `nomic-embed-text`, `llama3.2` |
| [`local_embeddings.py`](local_embeddings.py) | sentence-transformers plus a cross-encoder reranker | `pip install 'softrag[local]'` |
| [`chat_over_docs.py`](chat_over_docs.py) | Index a folder, then chat over it with citations | nothing (better with a real model) |

```bash
python examples/quickstart.py
python examples/chat_over_docs.py docs --db kb.db
```

`quickstart.py` and `filtering.py` use `HashEmbedder` and `EchoChatModel`, which
need no key, no network and no download. They are honest about what they are:
the hash embedder has no semantic ability, so treat their retrieval quality as a
demonstration of the API rather than of the library's accuracy. `EchoChatModel`
returns the retrieved context instead of an answer, which makes it a good way to
see exactly what a real model would have been given.
