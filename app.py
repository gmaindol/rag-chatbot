import streamlit as st
import requests
import json
import ast
import tempfile
import os
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, CSVLoader, TextLoader, UnstructuredFileLoader, JSONLoader
from langchain_core.documents import Document

# --- Configuration ---
VECTOR_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000/v1/chat/completions")
MODEL_CHOICES = ["gpt-4o", "gemma3-27b", "gpt-oss-20b", "phi3-14b"]
ALLOWED_FILE_TYPES = ["pdf", "docx", "xlsx", "csv", "txt", "md", "xml", "json"]
MAX_FILE_SIZE_MB = 10  # Maximum file size in MB

# --- RAG Setup (cached for performance) ---

@st.cache_resource
def get_embeddings():
    """Initializes and caches the embedding model."""
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

@st.cache_resource
def get_static_retriever(_embeddings):
    """Loads the static vector store (from rag_pipeline.py run)."""
    if not _embeddings: return None
    try:
        vector_store = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=_embeddings
        )
        return vector_store
    except Exception as e:
        st.warning(f"Static vector store not found or corrupt: {e}. Running in dynamic-only mode.")
        return None

embeddings = get_embeddings()
static_db = get_static_retriever(embeddings)

# --- Dynamic File Processing ---
def validate_file(uploaded_file, max_size_mb=MAX_FILE_SIZE_MB):
    """Validates uploaded file size and type."""
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File '{uploaded_file.name}' exceeds {max_size_mb}MB limit"
    return True, ""

def process_uploaded_files(uploaded_files, embeddings):
    """
    Handles file upload, parsing, chunking, and creating a new in-memory vector store.
    Uses temporary files to safely handle the uploaded data.
    """
    if not uploaded_files or not embeddings:
        return None

    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    failed_files = []

    for uploaded_file in uploaded_files:
        # Validate file
        is_valid, error_msg = validate_file(uploaded_file)
        if not is_valid:
            failed_files.append(error_msg)
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1].lower()}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            loader = None
            
            if file_extension == "pdf":
                loader = PyPDFLoader(tmp_path)
            elif file_extension == "docx":
                loader = UnstructuredWordDocumentLoader(tmp_path)
            elif file_extension == "csv":
                loader = CSVLoader(tmp_path)
            elif file_extension in ["txt", "md"]:
                loader = TextLoader(tmp_path)
            elif file_extension == "json":
                loader = JSONLoader(tmp_path, jq_schema='.', text_content=False)
            elif file_extension in ["xlsx", "xml"]:
                loader = UnstructuredFileLoader(tmp_path)
            else:
                failed_files.append(f"Unsupported file type: {uploaded_file.name}")
                continue

            data = loader.load()
            chunks = text_splitter.split_documents(data)
            
            for chunk in chunks:
                chunk.metadata['source_file'] = uploaded_file.name
                chunk.metadata['upload_timestamp'] = datetime.now().isoformat()
                
            all_docs.extend(chunks)

        except Exception as e:
            failed_files.append(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            os.unlink(tmp_path)

    # Display warnings for failed files
    if failed_files:
        for error in failed_files:
            st.warning(error)

    if all_docs:
        dynamic_db = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            collection_name="dynamic_session_data"
        )
        return dynamic_db
    return None

# --- Streamlit UI and Logic ---
st.set_page_config(
    page_title="Knowledge Assistant", 
    page_icon="👋", 
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main container styling */
    .reportview-container {
        background: #f8f9fa;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #4a0e4e 0%, #6a1b9a 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    .main-header p {
        color: #e1bee7;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-active {
        color: #4caf50;
        font-weight: 600;
    }
    .status-inactive {
        color: #9e9e9e;
        font-weight: 600;
    }
    
    /* File uploader */
    .uploadedFile {
        border-left: 3px solid #6a1b9a;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'dynamic_db' not in st.session_state:
    st.session_state['dynamic_db'] = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'session_start' not in st.session_state:
    st.session_state['session_start'] = datetime.now()

# Custom header
st.markdown("""
    <div class="main-header">
        <h1>👋 RAGnificent Knowledge Assistant</h1>
        <p>Secure AI-powered information retrieval with enterprise guardrails</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/4a0e4e/FFFFFF?text=SCB", use_container_width=True)
    st.title("⚙️ Configuration")
    
    # Document Upload Section
    st.header("📄 Document Repository")
    uploaded_files = st.file_uploader(
        "Upload supporting documents",
        type=ALLOWED_FILE_TYPES,
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(ALLOWED_FILE_TYPES).upper()}. Max size: {MAX_FILE_SIZE_MB}MB per file."
    )
    
    if uploaded_files:
        uploaded_files_signature = hash(tuple((f.name, f.size) for f in uploaded_files))
        
        if 'uploaded_files_hash' not in st.session_state or uploaded_files_signature != st.session_state['uploaded_files_hash']:
            with st.spinner(f"📄 Processing {len(uploaded_files)} document(s)..."):
                st.session_state['dynamic_db'] = process_uploaded_files(uploaded_files, embeddings)
            st.session_state['uploaded_files_hash'] = uploaded_files_signature
            if st.session_state['dynamic_db']:
                st.success(f"✅ {len(uploaded_files)} document(s) indexed successfully")
        
        with st.expander("📄 Uploaded Documents"):
            for file in uploaded_files:
                file_size = file.size / 1024  # Convert to KB
                st.text(f"• {file.name} ({file_size:.1f} KB)")
    else:
        st.session_state['dynamic_db'] = None
        st.info("No documents uploaded for this session")

    st.divider()
    
    # Model Selection
    st.header("🤖 Model Selection")
    selected_model = st.selectbox(
        "Language Model:",
        MODEL_CHOICES,
        index=0,
        help="Select the AI model for response generation"
    )
    
    st.divider()
    
    # RAG Configuration
    st.header("📚 Knowledge Retrieval")
    rag_available = static_db is not None or st.session_state['dynamic_db'] is not None
    
    enable_rag = st.toggle(
        "Enable Knowledge Base",
        value=rag_available,
        disabled=not rag_available,
        help="Augment responses with institutional knowledge and uploaded documents"
    )
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        retrieval_k = st.slider("Context chunks to retrieve:", 3, 10, 5)
        chunk_size = st.number_input("Chunk size (tokens):", 500, 2000, 1000, step=100)
        show_sources = st.checkbox("Show sources in response", value=True)
    
    st.divider()
    
    # System Status
    st.header("📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", selected_model)
    with col2:
        st.metric("RAG", "Active" if enable_rag else "Disabled")
    
    st.text(f"Static KB: {'🟢 Active' if static_db else '🔴 Inactive'}")
    st.text(f"Dynamic KB: {'🟢 Active' if st.session_state['dynamic_db'] else '🔴 Inactive'}")
    
    st.divider()
    
    # Information Section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Knowledge Assistant** leverages:
        - **RAG Architecture**: Retrieval-Augmented Generation for accurate, contextual responses
        - **Enterprise Guardrails**: LiteLLM security policies for compliance
        - **Multi-Source Knowledge**: Static institutional data + dynamic document uploads
        - **Secure Processing**: All data processed with bank-grade security protocols
        """)
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.markdown("### 💬 Conversation")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Retrieval Function ---
def get_combined_context(query, static_db, dynamic_db, k=5):
    """Retrieves documents from both static and dynamic stores and combines them."""
    all_docs = []
    
    if static_db:
        static_retriever = static_db.as_retriever(search_kwargs={"k": k})
        all_docs.extend(static_retriever.invoke(query))
    
    if dynamic_db:
        dynamic_retriever = dynamic_db.as_retriever(search_kwargs={"k": k})
        all_docs.extend(dynamic_retriever.invoke(query))
    
    unique_contexts = []
    for doc in all_docs:
        source_name = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown Document'))
        context_string = f"[Source: {source_name}]\n{doc.page_content}"
        if context_string not in unique_contexts:
            unique_contexts.append(context_string)
            
    return "\n\n---\n\n".join(unique_contexts)

# Chat input
if prompt := st.chat_input("Ask about policies, procedures, or uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing..."):
            full_response = ""
            sources_used = []
            
            rag_needed = enable_rag and (static_db or st.session_state['dynamic_db'])
            
            if rag_needed:
                context = get_combined_context(prompt, static_db, st.session_state['dynamic_db'], k=retrieval_k if 'retrieval_k' in locals() else 5)
                
                # Extract sources from context
                sources_used = [line.strip() for line in context.split('\n') if line.startswith('[Source:')]
                
                final_prompt = f"""You are a professional knowledge assistant for General Purpose. 
                
Based on the provided context, answer the user's question accurately and professionally. 

IMPORTANT GUIDELINES:
- Provide clear, concise, and accurate information
- If the context doesn't contain the answer, acknowledge this limitation
- Maintain professional terminology
- Cite sources when referencing specific documents
- For compliance matters, remind users to verify with official channels

Context:
{context}

Question: {prompt}

Answer:"""
            else:
                final_prompt = f"""You are a professional assistant for General Purpose. 
                
Answer the following question professionally. If you don't have specific information, provide general guidance and suggest consulting official resources.

Question: {prompt}

Answer:"""

            payload = {
                "model": selected_model,
                "messages": [{"role": "user", "content": final_prompt}],
                "stream": True
            }

            try:
                response = requests.post(LITELLM_PROXY_URL, json=payload, stream=True)
                response.raise_for_status()

                placeholder = st.empty()
                for chunk in response.iter_content(chunk_size=None):
                    try:
                        json_chunk = chunk.decode("utf-8").strip()
                        if json_chunk == "data: [DONE]" or not json_chunk or json_chunk == "{}":
                            continue
                        
                        data = json.loads(json_chunk.replace("data: ", ""))
                        if "content" in data["choices"][0]["delta"]:
                            content = data["choices"][0]["delta"]["content"]
                            full_response += content
                            placeholder.markdown(full_response + "▌")
                    except json.JSONDecodeError:
                        continue
                placeholder.markdown(full_response)
                
                # Display source citations if enabled and sources were used
                if show_sources and sources_used and rag_needed:
                    with st.expander("📚 Sources Referenced", expanded=False):
                        st.markdown("**Documents used to generate this response:**")
                        for idx, source in enumerate(sources_used, 1):
                            st.markdown(f"{idx}. {source}")
                
            except requests.exceptions.HTTPError as http_err:
                status_code = http_err.response.status_code
                if status_code in [400, 403]:
                    raw_response = http_err.response.text
                    print(f"Raw HTTP Error Response: {raw_response}")
                    
                    try:
                        error_data = http_err.response.json()
                        print(f"Parsed error_data: {json.dumps(error_data, indent=2)}")
                        
                        outer_error = error_data.get("error", {})
                        print(f"Outer error: {outer_error}")
                        
                        # Handle the nested JSON string in 'error.message'
                        inner_message = outer_error.get("message", "{}")
                        print(f"Inner message (type: {type(inner_message).__name__}): {inner_message}")
                        
                        guardrail_details = {}
                        
                        # Try multiple parsing strategies
                        if isinstance(inner_message, dict):
                            # Message is already a dictionary
                            guardrail_details = inner_message.get("error", {})
                        elif isinstance(inner_message, str):
                            # Try to parse as JSON string
                            try:
                                # Replace single quotes with double quotes for valid JSON
                                cleaned_message = inner_message.replace("'", '"')
                                inner_error_data = json.loads(cleaned_message)
                                guardrail_details = inner_error_data.get("error", {})
                            except json.JSONDecodeError:
                                # Try using ast.literal_eval for Python dict strings
                                try:
                                    inner_error_data = ast.literal_eval(inner_message)
                                    if isinstance(inner_error_data, dict):
                                        guardrail_details = inner_error_data.get("error", {})
                                except (ValueError, SyntaxError):
                                    print(f"Failed to parse inner message with ast.literal_eval")
                        
                        print(f"Guardrail details: {guardrail_details}")
                        
                        # Extract all the details
                        category = guardrail_details.get("category", "undetermined")
                        guardrail = guardrail_details.get("guardrail", "unknown")
                        type_ = guardrail_details.get("type", "None")
                        code_ = guardrail_details.get("code", "None")
                        scan_id = guardrail_details.get("scan_id", "None")
                        report_id = guardrail_details.get("report_id", "None")
                        
                        # Check if we actually got meaningful data
                        if guardrail == "unknown" and category == "undetermined":
                            raise ValueError("Could not extract guardrail details")
                        
                        full_response = (
                            f"⚠️ **SECURITY ALERT:** Your request was blocked due to a guardrail policy violation.\n\n"
                            f"**Details:**\n"
                            f"- **Guardrail:** `{guardrail}`\n"
                            f"- **Category:** `{category.capitalize()}`\n"
                            f"- **Type:** `{type_}`\n"
                            f"- **Code:** `{code_}`\n"
                            f"- **Scan ID:** `{scan_id}`\n"
                            f"- **Report ID:** `{report_id}`\n\n"
                            f"This incident has been logged for compliance purposes. Please contact your administrator for assistance.\n\n"
                            f"For support, reach out to: **it.security@yourorganization.com**"
                        )
                        
                    except Exception as parse_err:
                        print(f"Error parsing guardrail response: {type(parse_err).__name__}: {parse_err}")
                        full_response = (
                            f"⚠️ **SECURITY ALERT:** Your request was blocked due to a guardrail policy violation.\n\n"
                            f"**Details:**\n"
                            f"- **Reason:** An undetermined issue occurred during API communication.\n"
                            f"- **Code:** `{status_code}`\n\n"
                            f"This incident has been logged. Please contact your administrator for assistance.\n\n"
                            f"*(Raw response details in console.)*"
                        )
                    
                    st.warning(full_response)
                else:
                    full_response = f"⚠️ Error communicating with LiteLLM proxy: {http_err}"
                    st.error(full_response)
            except requests.exceptions.RequestException as e:
                full_response = f"⚠️ Connection error: Unable to reach the AI service. Please try again."
                st.error(full_response)
            
            if not full_response:
                full_response = "⚠️ An unexpected error occurred. Please try again."
        
        # Store the message with metadata
        message_data = {
            "role": "assistant", 
            "content": full_response,
            "sources": sources_used if rag_needed and sources_used else []
        }
        st.session_state.messages.append(message_data)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem;'>
    <p>🔒 Secure AI-Powered Assistant | Internal Use Only</p>
    <p>Session Duration: Active | Questions Processed: """ + str(len([m for m in st.session_state.messages if m['role'] == 'user'])) + """</p>
</div>
""", unsafe_allow_html=True)
