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

We welcome contributions! Here's how to get started:

### Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Make sure you have it installed:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Getting Started

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/yourusername/softrag.git
   cd softrag
   ```

2. **Install dependencies with uv:**
   ```bash
   uv sync --dev
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

### Making Changes

1. Create a new branch for your feature/fix
2. Make your changes
3. Add tests if applicable
4. Ensure all tests pass
5. Submit a pull request

### Project Structure

- `src/softrag/` - Main library code
- `docs/` - Documentation
- `examples/` - Usage examples
- `tests/` - Test suite

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Give to us your star ⭐

Developed with ❤️ for community
