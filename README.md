# LocalSense 🔍 
This project explores how to build a simple, local, privacy-focused semantic search engine using Python, built with Streamlit, Sentence Transformers, and ChromaDB.

> It uses modern NLP to understand what you're looking for, not just matching keywords. It works on your own machine, with your own files, so your data stays with you.
(Currently handles PDFs with plans to support more formats soon)

## 💽 Installation

```bash
git clone https://github.com/talsania/LocalSense.git
cd LocalSense
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🖲️ Usage

```bash
streamlit run app.py
```

## ✨ Features

- Privacy-first: All processing happens locally
- Semantic search using neural embeddings
- PDF document indexing and search
- Fast retrieval with vector database
- Simple upload and search interface

## 👨‍💻 Technologies

- Sentence Transformers (all-MiniLM-L6-v2)
- ChromaDB
- Streamlit
- NLTK
- PyPDF2
