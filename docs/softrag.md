# softrag

**softrag** is a minimalist local-first Retrieval-Augmented Generation (RAG) library that uses SQLite with [sqlite-vec](https://github.com/asg017/sqlite-vec) for efficient storage of documents, embeddings, and cache in a single `.db` file.

## Overview

- **Local storage:** All data is kept in a single SQLite database file
- **Pluggable RAG:** Inject your own embedding and chat models via dependency injection
- **Multi-format support:** Ingests Markdown, DOCX, PDF, plain text files, and web pages
- **Hybrid retrieval:** Combines semantic (vector) search and keyword (FTS5) search
- **Zero external dependencies:** No cloud services required for storage
- **Lightweight:** Minimal overhead with maximum performance

## Installation

```bash
pip install softrag
```

### Dependencies

The library requires the following dependencies:

- **sqlite-vec**: Vector similarity search in SQLite
- **trafilatura**: Web content extraction
- **langchain-text-splitters**: Text chunking (RecursiveCharacterTextSplitter)
- **llama-index**: Document readers for various file formats
- **pymupdf**: PDF processing (via llama-index)

These are automatically installed with the package.

### Requirements

- Python 3.12+
- SQLite with extension loading support
- Access to embedding and chat models (OpenAI, Hugging Face, Ollama, etc.)

## Quick Start

```python
from softrag import Rag
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize models
chat_model = ChatOpenAI(model="gpt-4o")
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Create RAG instance
rag = Rag(embed_model=embed_model, chat_model=chat_model, db_path="my_knowledge.db")

# Add content
rag.add_file("document.pdf")
rag.add_web("https://example.com/article")

# Query
answer = rag.query("What is the main topic discussed?")
print(answer)
```

## Core Methods

### `Rag.__init__(*, embed_model, chat_model, db_path="softrag.db")`

Initialize a new RAG instance.

**Parameters:**
- `embed_model`: Model implementing `.embed_query(text) -> List[float]`
- `chat_model`: Model implementing `.invoke(prompt) -> str`
- `db_path`: Path to SQLite database file (created if doesn't exist)

### `Rag.add_file(data, metadata=None)`

Add file content to the knowledge base.

**Parameters:**
- `data`: File path (str/Path), bytes, or file-like object
- `metadata`: Optional dictionary with additional metadata

**Supported formats:**
- PDF files
- DOCX documents
- Markdown files
- Plain text files
- Any format supported by UnstructuredReader

**Example:**
```python
rag.add_file("research.pdf", metadata={"author": "John Doe", "year": 2024})
rag.add_file(Path("notes.md"))
```

### `Rag.add_web(url, metadata=None)`

Extract and add web page content.

**Parameters:**
- `url`: Web page URL
- `metadata`: Optional dictionary (URL is automatically added)

**Example:**
```python
rag.add_web("https://arxiv.org/abs/2301.00001", metadata={"type": "paper"})
```

### `Rag.query(question, *, top_k=5, stream=False)`

Query the knowledge base with context-augmented generation.

**Parameters:**
- `question`: The question to answer
- `top_k`: Number of most relevant chunks to retrieve (default: 5)
- `stream`: If True, returns generator yielding response chunks

**Returns:**
- String response (if `stream=False`)
- Generator yielding chunks (if `stream=True`)

**Example:**
```python
# Standard query
answer = rag.query("What are the key findings?", top_k=3)

# Streaming query
for chunk in rag.query("Explain the methodology", stream=True):
    print(chunk, end="", flush=True)
```

## Advanced Configuration

### Custom Text Chunking

The default chunking uses `RecursiveCharacterTextSplitter` with 400 character chunks and 100 character overlap. You can customize this:

```python
# Custom delimiter-based chunking
rag._set_splitter("\n\n")  # Split on double newlines

# Custom function
def my_chunker(text):
    return [chunk.strip() for chunk in text.split("---") if chunk.strip()]

rag._set_splitter(my_chunker)

# Back to default
rag._set_splitter(None)
```

### Model Integration Examples

#### OpenAI Models
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

chat_model = ChatOpenAI(model="gpt-4o", temperature=0.1)
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
```

#### Hugging Face Models
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chat_model = HuggingFacePipeline.from_model_id(
    model_id="microsoft/DialoGPT-medium",
    task="text-generation"
)
```

#### Ollama Models
```python
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

chat_model = Ollama(model="llama2")
embed_model = OllamaEmbeddings(model="llama2")
```

## Database Schema

The SQLite database contains three main components:

- **`documents`**: Stores text chunks and metadata (JSON)
- **`docs_fts`**: FTS5 virtual table for keyword search
- **`embeddings`**: Vector embeddings using sqlite-vec (1536 dimensions by default)

## How It Works

1. **Document Processing**: Files are parsed and text is extracted
2. **Chunking**: Text is split into manageable chunks (default: 400 chars with 100 overlap)
3. **Embedding**: Each chunk is converted to a vector embedding
4. **Storage**: Chunks, embeddings, and metadata are stored in SQLite
5. **Retrieval**: Queries use hybrid search (keyword + semantic similarity)
6. **Generation**: Retrieved chunks provide context for the language model

## Performance Notes

- **Deduplication**: Chunks are deduplicated using SHA-256 hashes
- **Hybrid Search**: Combines FTS5 keyword search with vector similarity
- **Optimized Storage**: Uses WAL mode and optimized page size (32KB)
- **Embedding Dimensions**: Optimized for 1536-dimensional embeddings (OpenAI compatible)

## Error Handling

Common issues and solutions:

- **sqlite-vec not found**: Ensure sqlite-vec is properly installed
- **Unsupported file format**: Check if the file type is supported or use UnstructuredReader
- **Empty results**: Verify documents were added successfully and embeddings are working
- **Model compatibility**: Ensure your models implement the required interfaces

## Best Practices

1. **Batch Processing**: Add multiple files before querying for better performance
2. **Metadata Usage**: Include relevant metadata for better document organization
3. **Chunk Size**: Adjust chunk size based on your content type and model context window
4. **Model Selection**: Choose embedding models compatible with your use case
5. **Database Backup**: Regularly backup your `.db` file as it contains all your data

## API Reference

### Type Definitions

```python
EmbedFn = Callable[[str], List[float]]
ChatFn = Callable[[str, Sequence[str]], str]
Chunker = Union[str, Callable[[str], List[str]], None]
FileInput = Union[str, Path, bytes, bytearray, IO[bytes], IO[str]]
```

### Internal Methods

- `_retrieve(query, k)`: Retrieve k most relevant chunks
- `_persist(text, metadata)`: Store text chunks with embeddings
- `_extract_file(data)`: Extract text from various file formats
- `_extract_web(url)`: Extract text from web pages
- `_ensure_db()`: Initialize database and sqlite-vec
- `_create_schema()`: Create database tables

---

## Give to us your star ⭐