# Local Semantic Search Engine

Welcome! This is a fun project exploring how to build a simple, local, privacy-focused semantic search engine using Python. It's built with Streamlit, Sentence Transformers, and ChromaDB.

The goal is to index local documents (currently PDFs) and search through their content based on meaning, not just keywords, keeping everything on your own machine.

## Features

* **Local First:** All processing and data storage happen on your machine. No data is sent to external servers.
* **Semantic Search:** Uses sentence embeddings (via Sentence Transformers) to understand query intent.
* **Vector Storage:** Uses ChromaDB for efficient local vector storage and retrieval.
* **Simple UI:** Built with Streamlit for easy file uploading and searching.
* **Chunking:** Splits documents into smaller chunks for more relevant results.
* **(Add more features as you develop them)**

## Installation

1.  **Clone the repository:**
    ```bash
    # Replace YOUR_USERNAME/YOUR_REPOSITORY_NAME with the actual path
    git clone [https://github.com/talsania/local-semantic-search](https://github.com/talsania/local-semantic-search.git)
    cd YOUR_REPOSITORY_NAME
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The first time you run the app, the sentence transformer model will be downloaded, which might take a moment).*

## Usage

1.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
2.  **Upload Documents:** Use the file uploader in the web interface to select PDF documents. Click "Process Uploaded Files". Wait for processing to complete (this involves chunking and embedding).
3.  **Search:** Enter your query in the search box and click "Search". Relevant chunks from your documents will be displayed.

If you have ideas, find bugs, or want to contribute code, please feel free to open an issue or submit a pull request on GitHub. All contributions are appreciated.