import streamlit as st
import chromadb
import os
from sentence_transformers import SentenceTransformer
import PyPDF2
import time
import io
import re # Import regular expressions for splitting

# --- Main Application Logic ---
st.set_page_config(layout="wide")
st.title("Local Semantic Search Engine (Chunked)")
# Important to run before Configuration

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db" # Directory to store ChromaDB data
COLLECTION_NAME = "local_documents_chunked" # New collection name
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Or another suitable model
CHUNK_SIZE_CHARS = 1000 # Approximate size of chunks in characters (adjust as needed)
CHUNK_OVERLAP_CHARS = 150 # How much chunks should overlap (adjust as needed)


# --- Load Embedding Model ---
@st.cache_resource
def load_embedding_model(model_name):
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        st.error(f"Failed to load embedding model '{model_name}': {e}")
        return None

embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)

if embedding_model is None:
    st.stop()

st.sidebar.success(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")


# --- Helper Functions ---

@st.cache_resource
def get_chroma_client():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        return client
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB client: {e}")
        return None

# Note: Collection name changed, so cache might need adjusting if name changes often
# Using collection name in the key for cache robustness
@st.cache_resource(show_spinner=False)
def get_or_create_collection(_client, collection_name):
    if _client is None:
        st.error("ChromaDB client not available.")
        return None
    try:
        # Check if collection exists, delete if it does to ensure clean state with new chunking
        # This is aggressive for debugging/dev; remove for production if not needed
        # try:
        #     _client.delete_collection(name=collection_name)
        #     st.warning(f"Existing collection '{collection_name}' deleted for fresh start.")
        # except:
        #      pass # Ignore if collection doesn't exist

        collection = _client.get_or_create_collection(
             name=collection_name,
             metadata={"hnsw:space": "cosine"} # Example: Specify distance metric if needed
        )
        return collection
    except Exception as e:
        st.error(f"Failed to get/create ChromaDB collection '{collection_name}': {e}")
        return None

def extract_text_from_pdf(file_content_bytes):
    try:
        pdf_file = io.BytesIO(file_content_bytes)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def chunk_text(text, chunk_size=CHUNK_SIZE_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS):
    """ Basic text chunking function """
    if not isinstance(text, str):
         return [] # Return empty list if text is not a string

    # Simple splitting by paragraphs first, then by size if needed
    # Using regex to split by one or more newlines
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if not para.strip():
            continue # Skip empty paragraphs

        # If adding the next paragraph exceeds chunk size (roughly), process the current chunk
        if len(current_chunk) + len(para) + 1 > chunk_size and current_chunk:
             # Simple overlap: could be improved
            chunks.append(current_chunk.strip())
            overlap_start = max(0, len(current_chunk) - chunk_overlap)
            current_chunk = current_chunk[overlap_start:] + "\n" # Start new chunk with overlap

        # Add the paragraph (or part of it if it's too long itself)
        # Basic handling for very long paragraphs: split them further
        para_offset = 0
        while para_offset < len(para):
             remaining_space = chunk_size - len(current_chunk)
             para_part = para[para_offset : para_offset + remaining_space]

             if not current_chunk: # If starting a new chunk
                  current_chunk += para_part
             else:
                  current_chunk += "\n" + para_part # Add with newline separator

             para_offset += len(para_part)

             # If the current chunk is now full, add it and handle overlap
             if len(current_chunk) >= chunk_size:
                  chunks.append(current_chunk.strip())
                  overlap_start = max(0, len(current_chunk) - chunk_overlap)
                  current_chunk = current_chunk[overlap_start:]
                  # If the last part added wasn't the end of the paragraph, need newline
                  if para_offset < len(para):
                     current_chunk += "\n"


    # Add the last remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Alternative simple fixed-size chunking (less context-aware):
    # chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

    return chunks

# Initialize ChromaDB
chroma_client = get_chroma_client()

if chroma_client:
    collection = get_or_create_collection(chroma_client, COLLECTION_NAME)
    if collection:
        st.success(f"Connected to ChromaDB. Collection '{COLLECTION_NAME}' loaded. Documents (Chunks): {collection.count()}")
    else:
        st.error("Failed to load ChromaDB collection.")
        st.stop()
else:
    st.error("Failed to initialize ChromaDB client.")
    st.stop()


# --- UI Elements ---

st.header("1. Add Documents (will be chunked)")
uploaded_files = st.file_uploader("Choose PDF files to add", accept_multiple_files=True, type=['pdf'], key="file_uploader")

if uploaded_files:
    process_button_key = f"process_button_{'_'.join(sorted([f.name for f in uploaded_files]))}"
    if st.button("Process Uploaded Files", key=process_button_key):
        with st.spinner("Processing files (extracting, chunking, embedding)..."):
            total_chunks_added = 0
            skipped_files = 0
            error_files = 0
            progress_bar = st.progress(0.0)
            total_files = len(uploaded_files)
            file_status_placeholder = st.empty() # Placeholder for file status updates

            for i, uploaded_file in enumerate(uploaded_files):
                file_status_placeholder.write(f"Processing file {i+1}/{total_files}: {uploaded_file.name}...")
                file_content_bytes = uploaded_file.getvalue()
                extracted_text = extract_text_from_pdf(file_content_bytes)

                if extracted_text:
                    try:
                        # --- Chunking Step ---
                        text_chunks = chunk_text(extracted_text)
                        if not text_chunks:
                             st.warning(f"Could not generate chunks for {uploaded_file.name}. Skipping.")
                             skipped_files += 1
                             continue

                        st.write(f"-> Extracted text, generated {len(text_chunks)} chunks for {uploaded_file.name}.")

                        # --- Embed and Add Chunks ---
                        chunk_ids = []
                        embeddings = []
                        documents = []
                        metadatas = []

                        # Process chunks in batches for potentially better performance (optional)
                        batch_size = 100 # How many chunks to embed and add at once
                        for j in range(0, len(text_chunks), batch_size):
                             batch_chunks = text_chunks[j:j+batch_size]
                             if not batch_chunks: continue

                             batch_embeddings = embedding_model.encode(batch_chunks, show_progress_bar=False).tolist()
                             batch_ids = [f"{os.path.splitext(uploaded_file.name)[0]}_chunk{j+k}_{int(time.time())}" for k in range(len(batch_chunks))]
                             batch_metadatas = [{"source": uploaded_file.name, "chunk_index": j+k} for k in range(len(batch_chunks))]

                             # Add batch to ChromaDB
                             if collection:
                                  collection.add(
                                      ids=batch_ids,
                                      embeddings=batch_embeddings,
                                      documents=batch_chunks, # Store the chunk text itself
                                      metadatas=batch_metadatas
                                  )
                                  total_chunks_added += len(batch_chunks)
                             else:
                                  st.error(f"Collection object not found while processing {uploaded_file.name}. Stopping batch add.")
                                  break # Stop processing this file if collection is lost


                        st.write(f"--> Added {len(text_chunks)} chunks for '{uploaded_file.name}'")

                    except Exception as e:
                         st.error(f"Error processing chunks for {uploaded_file.name}: {e}")
                         error_files += 1
                else:
                    st.warning(f"Could not extract text from {uploaded_file.name} or it was empty. Skipping.")
                    skipped_files += 1

                # Update progress bar
                progress_bar.progress((i + 1) / total_files)
                file_status_placeholder.empty() # Clear status message for the next file

            st.success(f"Finished processing. Total Chunks Added: {total_chunks_added}, Files Skipped/Empty: {skipped_files}, Files with Errors: {error_files}")


# Search Interface
st.header("2. Search Documents (Chunks)")
search_query = st.text_input("Enter your search query:", key="search_query_input")

if st.button("Search", key="search_button"):
    if search_query and collection:
        with st.spinner("Searching chunks..."):
            st.write(f"Searching for: '{search_query}'")
            try:
                # 1. Embed the query
                query_embedding = embedding_model.encode(search_query, show_progress_bar=False).tolist()

                # 2. Query ChromaDB (searching against chunk embeddings)
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5, # Number of relevant chunks to return
                    include=['documents', 'metadatas', 'distances']
                )

                # --- DEBUG LINE (Optional: Keep or remove) ---
                # st.write("Raw ChromaDB Chunk Results:", results)
                # ---------------------------------------------

                # 3. Display results (now showing chunks)
                st.subheader("Results:")
                if results and results.get('ids') and results['ids'][0]:
                    for i, chunk_id in enumerate(results['ids'][0]):
                        distance = results['distances'][0][i]
                        # The 'document' is now the text of the relevant CHUNK
                        document_chunk = results['documents'][0][i]
                        metadata = results['metadatas'][0][i]
                        source_file = metadata.get('source', 'N/A')
                        chunk_index = metadata.get('chunk_index', 'N/A')

                        st.markdown(f"**Result {i+1}** (Source: {source_file}, Chunk: {chunk_index}, Distance: {distance:.4f})")
                        # Display the relevant chunk text
                        st.text_area(f"Content Chunk:", document_chunk, height=150, key=f"result_chunk_{chunk_id}")
                        st.divider()
                else:
                    st.write("No relevant document chunks found.")

            except Exception as e:
                st.error(f"An error occurred during search: {e}")
    elif not search_query:
        st.warning("Please enter a query.")
    elif not collection:
         st.error("ChromaDB collection not available for searching.")


# --- Display some info ---
st.sidebar.header("Info")
st.sidebar.write(f"Vector DB Path: {os.path.abspath(CHROMA_DB_PATH)}")
st.sidebar.write(f"Collection Name: {COLLECTION_NAME}")
if collection:
     try:
         current_count = collection.count()
         # The count now represents the number of chunks, not files
         st.sidebar.write(f"Chunks Indexed: {current_count}")
     except Exception as e:
         st.sidebar.warning(f"Could not retrieve count: {e}")
         st.sidebar.write("Chunks Indexed: (Error)")
else:
     st.sidebar.write("Chunks Indexed: (Collection not loaded)")