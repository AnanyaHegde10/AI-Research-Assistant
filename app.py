import os
import hashlib
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import create_vector_db
from rag import answer_question


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    /* Status cards */
    .status-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background-color: #f8fafc;
        margin-bottom: 1rem;
    }

    .status-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }

    .status-text {
        color: #6b7280;
        font-size: 0.9rem;
    }

    /* PDF cards */
    .pdf-card {
        padding: 0.7rem 0.8rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.5rem;
        background-color: white;
    }

    /* Source cards */
    .source-card {
        padding: 0.8rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        background-color: #f8fafc;
        margin-bottom: 0.7rem;
    }

    .source-title {
        font-weight: 600;
        margin-bottom: 0.2rem;
    }

    .source-meta {
        color: #6b7280;
        font-size: 0.85rem;
    }

    /* Chat */
    .chat-empty {
        text-align: center;
        padding: 4rem 1rem;
        color: #6b7280;
    }

    .chat-empty-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed" not in st.session_state:
    st.session_state.processed = False

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = ""

if "processing" not in st.session_state:
    st.session_state.processing = False


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def calculate_files_hash(files):
    """
    Create a unique hash for the currently uploaded PDFs.

    This helps us detect whether the user uploaded
    a new set of PDFs.
    """

    hasher = hashlib.md5()

    for file in files:
        hasher.update(file.name.encode("utf-8"))
        hasher.update(file.getbuffer())

    return hasher.hexdigest()


def process_pdfs(uploaded_files):
    """
    Read multiple PDFs, extract text and create chunks.
    """

    all_chunks = []

    os.makedirs("uploads", exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    for uploaded_file in uploaded_files:

        file_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        # -------------------------------------------------
        # Save PDF
        # -------------------------------------------------

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # -------------------------------------------------
        # Read PDF
        # -------------------------------------------------

        reader = PdfReader(file_path)

        for page_number, page in enumerate(reader.pages, start=1):

            page_text = page.extract_text()

            if not page_text:
                continue

            # -------------------------------------------------
            # Split each page separately
            #
            # This helps preserve page information.
            # -------------------------------------------------

            page_chunks = splitter.split_text(page_text)

            for chunk in page_chunks:

                all_chunks.append(
                    {
                        "text": chunk,
                        "source": uploaded_file.name,
                        "page": page_number
                    }
                )

    return all_chunks


def create_documents_for_vector_db(chunks):
    """
    Convert our chunk dictionaries into LangChain Documents.

    Metadata allows the application to display:
    PDF name + page number in the sources section.
    """

    from langchain_core.documents import Document

    documents = []

    for chunk in chunks:

        document = Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source"],
                "page": chunk["page"]
            }
        )

        documents.append(document)

    return documents


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🤖 AI Research Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload multiple PDFs and chat with them using AI-powered RAG</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 PDF Library")

    st.caption(
        "Upload one or more PDF documents and build your knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    # -----------------------------------------------------
    # Number of PDFs
    # -----------------------------------------------------

    if uploaded_files:

        st.info(
            f"📄 {len(uploaded_files)} PDF(s) selected"
        )

    # -----------------------------------------------------
    # Process PDFs button
    # -----------------------------------------------------

    if uploaded_files:

        current_hash = calculate_files_hash(uploaded_files)

        already_processed = (
            st.session_state.pdf_hash == current_hash
            and st.session_state.processed
        )

        if already_processed:

            st.success(
                "🟢 These PDFs are already processed."
            )

        else:

            if st.button(
                "⚡ Process PDFs",
                use_container_width=True,
                type="primary"
            ):

                st.session_state.processing = True

                try:

                    with st.spinner(
                        "📖 Reading PDFs and creating your knowledge base..."
                    ):

                        # ---------------------------------
                        # Extract and split PDF text
                        # ---------------------------------

                        chunks = process_pdfs(
                            uploaded_files
                        )

                        if not chunks:

                            st.error(
                                "❌ No readable text was found in the uploaded PDFs."
                            )

                            st.session_state.processing = False
                            st.stop()

                        # ---------------------------------
                        # Convert to Documents
                        # ---------------------------------

                        documents = create_documents_for_vector_db(
                            chunks
                        )

                        # ---------------------------------
                        # Create FAISS database
                        # ---------------------------------

                        create_vector_db(
                            documents
                        )

                    # -------------------------------------
                    # Update session state
                    # -------------------------------------

                    st.session_state.processed = True

                    st.session_state.pdf_hash = current_hash

                    st.session_state.uploaded_files = [
                        file.name
                        for file in uploaded_files
                    ]

                    # Start a fresh conversation
                    st.session_state.chat_history = []

                    st.session_state.processing = False

                    st.success(
                        f"✅ {len(uploaded_files)} PDF(s) processed successfully!"
                    )

                    st.rerun()

                except Exception as e:

                    st.session_state.processing = False

                    st.error(
                        f"❌ Error while processing PDFs:\n\n{str(e)}"
                    )


    # =====================================================
    # CURRENTLY LOADED PDFs
    # =====================================================

    if st.session_state.uploaded_files:

        st.divider()

        st.subheader("📂 Currently loaded")

        for file_name in st.session_state.uploaded_files:

            st.markdown(
                f"""
                <div class="pdf-card">
                    📄 <b>{file_name}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        # -----------------------------------------------
        # RAG status
        # -----------------------------------------------

        st.success(
            "🟢 RAG System Ready"
        )

        st.caption(
            "Your PDFs are ready for questions."
        )


    # =====================================================
    # CHAT CONTROLS
    # =====================================================

    st.divider()

    st.subheader("⚙️ Chat Controls")

    # -----------------------------------------------------
    # New Chat
    # -----------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


    # -----------------------------------------------------
    # Clear Chat
    # -----------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# MAIN CHAT AREA
# =========================================================

st.subheader("💬 Chat with your PDFs")


# =========================================================
# EMPTY CHAT STATE
# =========================================================

if not st.session_state.chat_history:

    if st.session_state.processed:

        st.markdown(
            """
            <div class="chat-empty">

                <div class="chat-empty-title">
                    👋 Ask something about your PDFs
                </div>

                <div>
                    Try questions such as:
                </div>

                <br>

                <div>
                    • What is PHP?<br>
                    • Explain the while loop.<br>
                    • What is a local variable?<br>
                    • Give me the syntax of lcfirst().<br>
                    • Explain the example of explode().<br>
                    • Which PDF contains information about functions?
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="chat-empty">

                <div class="chat-empty-title">
                    📚 Upload your PDFs to get started
                </div>

                <div>
                    Upload one or more PDF documents from the sidebar,
                    process them, and start asking questions.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.chat_history:

    role = message["role"]

    with st.chat_message(role):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about your PDFs..."
)


# =========================================================
# HANDLE QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # Check PDF processing
    # -----------------------------------------------------

    if not st.session_state.processed:

        st.warning(
            "⚠️ Please upload and process your PDFs first."
        )

        st.stop()


    # -----------------------------------------------------
    # Display user question
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------------------
    # Save user question
    # -----------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )


    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching your PDFs..."
        ):

            try:

                answer, docs = answer_question(
                    question,
                    st.session_state.chat_history
                )

                st.markdown(answer)

            except Exception as e:

                answer = (
                    "❌ Sorry, I encountered an error while "
                    "searching the PDF."
                )

                docs = []

                st.error(
                    f"{answer}\n\n{str(e)}"
                )


        # =================================================
        # SOURCES
        # =================================================

        if docs:

            with st.expander(
                "📚 View Sources"
            ):

                # Avoid displaying duplicate sources
                displayed_sources = set()

                for i, doc in enumerate(
                    docs,
                    start=1
                ):

                    source = doc.metadata.get(
                        "source",
                        "Unknown PDF"
                    )

                    page = doc.metadata.get(
                        "page",
                        "Unknown"
                    )

                    source_key = (
                        source,
                        page
                    )

                    if source_key in displayed_sources:

                        continue

                    displayed_sources.add(
                        source_key
                    )

                    st.markdown(
                        f"""
                        <div class="source-card">

                            <div class="source-title">
                                📄 {source}
                            </div>

                            <div class="source-meta">
                                📖 Page {page}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write(
                        doc.page_content[:700]
                    )

                    st.divider()


    # -----------------------------------------------------
    # Save assistant answer
    # -----------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )