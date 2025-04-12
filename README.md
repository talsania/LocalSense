# Local Semantic Search Engine 🔍 
This project explores how to build a simple, local, privacy-focused semantic search engine using Python, built with Streamlit, Sentence Transformers, and ChromaDB.

> It uses modern NLP to understand what you're looking for, not just matching keywords. It works on your own machine, with your own files, so your data stays with you.
(Currently handles PDFs with plans to support more formats soon)

## Getting started 📜

- Python 3.8+
- CUDA drivers (optional)

## Installation 💽

```bash
git clone https://github.com/talsania/local-semantic-search.git
cd local-semantic-search
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage 🖲️

```bash
streamlit run app.py
```

## Features ✨

- Privacy-first: All processing happens locally
- Semantic search using neural embeddings
- PDF document indexing and search
- Fast retrieval with vector database
- Simple upload and search interface

## Technologies 👨‍💻

- Sentence Transformers (all-MiniLM-L6-v2)
- ChromaDB
- Streamlit
- NLTK
- PyPDF2

## Roadmap 🛣️

- Custom web interface
- Integration with more powerful models
- Support for additional document formats
- OCR capabilities
- Automated document ingestion
- Role-based access control
- Usage analytics

> If you have ideas, find bugs, or want to contribute code, please feel free to open an issue or submit a pull request on GitHub. All contributions are appreciated.
