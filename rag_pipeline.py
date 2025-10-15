import os
import pandas as pd
import docx2txt
import io
import msoffcrypto
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
DATA_DIR = "./data"
VECTOR_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Helper Functions ---
def load_and_split_documents(data_dir):
    """Loads documents from a directory and splits them into chunks."""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith(".pdf"):
            # PDF handling remains the same
            # (assuming PyPDF2 is installed and working)
            pass
        elif filename.endswith(".docx"):
            print(f"Loading {filename}...")
            # Use docx2txt to handle .docx files
            try:
                text = docx2txt.process(filepath)
                documents.extend(text_splitter.split_text(text))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        elif filename.endswith(".xlsx"):
            print(f"Loading {filename}...")
            # Use msoffcrypto and pandas to handle encrypted .xlsx files
            try:
                with open(filepath, "rb") as f:
                    file_buffer = io.BytesIO()
                    office_file = msoffcrypto.OfficeFile(f)
                    # NOTE: Replace 'your_password' with the actual password if files are encrypted
                    # For this demo, we assume no password is set or the password is an empty string
                    # If your files are encrypted, you will need to handle the password.
                    # For example: office_file.load_key(password="your_password")
                    office_file.decrypt(file_buffer)
                    df = pd.read_excel(file_buffer)
                    text = df.to_string()
                    documents.extend(text_splitter.split_text(text))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        elif filename.endswith(".csv"):
            print(f"Loading {filename}...")
            try:
                df = pd.read_csv(filepath)
                text = df.to_string()
                documents.extend(text_splitter.split_text(text))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return documents

# --- Main Logic ---
if __name__ == "__main__":
    print("Starting RAG pipeline...")
    # 1. Load and split documents
    documents = load_and_split_documents(DATA_DIR)
    
    if documents:
        print(f"Loaded {len(documents)} document chunks.")

        # 2. Create embeddings
        print("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # 3. Create and persist vector store
        print("Creating and persisting vector store...")
        vector_store = Chroma.from_texts(
            texts=documents,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        vector_store.persist()
        print(f"Vector store saved to {VECTOR_DB_DIR}")
    else:
        print("No documents were loaded. Please check the 'data' directory and file formats.")
