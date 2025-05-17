# SoftRAG 🔍

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Minimal local-first Retrieval-Augmented Generation (RAG) library using SQLite with sqlite-vec.

## 🌟 Features

- **Local-first**: All processing happens locally
- **SQLite + sqlite-vec**: All data (documents, embeddings, cache) lives in a single `.db` file
- **No cloud service dependency**: Pluggable architecture allows you to choose your own LLM backend
- **Simple**: Minimalist and easy-to-use API
- **Plug & Play**: Compatible with LangChain, Transformers, and other frameworks

## 📋 Requirements

- Python 3.12+
- Dependencies: sqlite-vec, trafilatura, pymupdf (for PDFs)
- Access to embedding models and LLMs (uses OpenAI by default)

## 🚀 Installation

```bash
pip install softrag
```

## 🔧 Basic Usage

```python
from softrag import Rag
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize models
chat = ChatOpenAI(model="gpt-4o")
embed = OpenAIEmbeddings(model="text-embedding-3-small")

# Create instance
rag = Rag(embed_model=embed, chat_model=chat)

# Add content
rag.add_file("document.pdf") 
rag.add_web("https://example.com/page")

# Make a query
response = rag.query("What is the main information in this content?")
print(response)
```

## 📚 Examples

See the `examples/` folder for more detailed examples:

- `simple.py`: Basic example with OpenAI
- `local.py`: Example using local Transformers models

## 🔄 How It Works

SoftRAG uses a hybrid approach for retrieval:

1. **Extraction**: Content is extracted from documents and web pages
2. **Splitting**: Text is divided into smaller chunks
3. **Indexing**: Each chunk is indexed by text (SQLite FTS5) and vector embedding
4. **Retrieval**: Queries combine keyword search and vector similarity
5. **Generation**: The most relevant chunks are sent to the LLM along with the question

## 🤝 Contributing

Contributions are welcome! Please feel free to submit Pull Requests.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Developed with ❤️ for local-first RAG.