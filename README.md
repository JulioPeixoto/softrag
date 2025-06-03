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
- **Multi-format support** – PDF, DOCX, Markdown, text files, web pages, and **images**
- **Image understanding** – Uses GPT-4 Vision to analyze and describe images for semantic search
- **Hybrid retrieval** – Combines keyword search (FTS5) and semantic similarity
- **Unified search** – Query across text documents and image descriptions seamlessly

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

# Add different types of content
rag.add_file("document.pdf")
rag.add_web("https://example.com/article")
rag.add_image("photo.jpg")  # 🆕 Image support!

# Query across all content types
answer = rag.query("What is shown in the image and how does it relate to the document?")
print(answer)
```

## 📚 Documentation

For complete documentation, examples, and advanced usage, see: **[docs/softrag.md](docs/softrag.md)**

## 🛠️ Next Steps

- Documentation Creation: Develop comprehensive documentation using tools like Sphinx or MkDocs to provide clear guidance on installation, usage, and contribution.
- Image Support in RAG: Integrate capabilities to handle image data, enabling the retrieval and generation of content based on visual inputs. This could involve incorporating models like CLIP for image embeddings.
- Automated Testing: Implement unit and integration tests using frameworks such as pytest to ensure code reliability and facilitate maintenance.
- Support for Multiple LLM Backends: Extend compatibility to include various language model providers, such as OpenAI, Hugging Face Transformers, and local models, offering users flexibility in choosing their preferred backend.
- Enhanced Context Retrieval: Improve the relevance of retrieved documents by integrating reranking techniques or advanced retrieval models, ensuring more accurate and contextually appropriate responses.
- Performance Benchmarking: Conduct performance evaluations to assess Softrag's efficiency and scalability, comparing it with other RAG solutions to identify areas for optimization.
- Monitoring and Logging: Implement logging mechanisms to track system operations and facilitate debugging, as well as monitoring tools to observe performance metrics and system health.

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
