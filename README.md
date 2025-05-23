# softrag [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![PyPI version](https://img.shields.io/pypi/v/softrag.svg)](https://pypi.org/project/softrag/)

<div align="center">
  <img src="piriquito.png" width="150" alt="SoftRAG mascot – periquito"/>
</div>

Minimal **local-first** Retrieval-Augmented Generation (RAG) library powered by **SQLite + sqlite-vec**.  
Everything—documents, embeddings, cache—lives in a single `.db` file.

---

## 🌟 Features

- **Local-first** – All processing happens locally, no external services required for storage
- **SQLite + sqlite-vec** – Documents, embeddings, and cache in a single `.db` file
- **Model-agnostic** – Works with OpenAI, Hugging Face, Ollama, or any compatible models
- **Blazing-fast** – Optimized for minimal overhead and maximum throughput
- **Multi-format support** – PDF, DOCX, Markdown, text files, and web pages
- **Hybrid retrieval** – Combines keyword search (FTS5) and semantic similarity

## 🚀 Quick Start

```bash
pip install softrag
```

```python
from softrag import Rag
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize
rag = Rag(
    embed_model=OpenAIEmbeddings(model="text-embedding-3-small"),
    chat_model=ChatOpenAI(model="gpt-4o")
)

# Add documents
rag.add_file("document.pdf")
rag.add_web("https://example.com/article")

# Query with context
answer = rag.query("What is the main topic discussed?")
print(answer)
```

## 📚 Documentation

For complete documentation, examples, and advanced usage, see: **[docs/softrag.md](docs/softrag.md)**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit Pull Requests.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

Developed with ❤️ for AI community
