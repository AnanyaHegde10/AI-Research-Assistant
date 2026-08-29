import os
from pathlib import Path

import streamlit as st

from pdf_processor import process_pdf
from vector_store import create_vector_db, load_vector_db
from rag import answer_question


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .source-box {
        padding: 12px;
        border-radius: 8px;
        margin-top: 8px;
        background-color: rgba(128,128,128,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "db" not in st.session_state:
    st.session_state.db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------

UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(
    exist_ok=True
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 AI Research Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a PDF and ask questions using AI-powered RAG.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload a PDF document to analyze."
    )

    if uploaded_file:

        file_path = UPLOAD_DIR / uploaded_file.name

        # Save file
        if not file_path.exists():

            with open(file_path, "wb") as f:

                f.write(
                    uploaded_file.getbuffer()
                )

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        process_button = st.button(
            "⚙️ Process PDF",
            use_container_width=True
        )

        if process_button:

            try:

                with st.spinner(
                    "Reading and indexing PDF..."
                ):

                    chunks = process_pdf(
                        file_path
                    )

                    db = create_vector_db(
                        chunks
                    )

                    st.session_state.db = db

                    st.session_state.processed_file = (
                        uploaded_file.name
                    )

                    st.session_state.messages = []

                st.success(
                    f"PDF processed successfully! "
                    f"{len(chunks)} chunks created."
                )

            except Exception as e:

                st.error(
                    f"Processing failed: {str(e)}"
                )

    st.divider()

    st.header("⚙️ Settings")

    st.write(
        "Retrieval chunks: **6**"
    )

    st.write(
        "Embedding: **MiniLM-L6-v2**"
    )

    st.write(
        "Vector database: **FAISS**"
    )

    st.write(
        "LLM: **Gemini Flash Lite**"
    )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# MAIN CONTENT
# --------------------------------------------------

if st.session_state.processed_file:

    st.info(
        f"📄 Currently analyzing: "
        f"**{st.session_state.processed_file}**"
    )


else:

    st.info(
        "👈 Upload a PDF from the sidebar "
        "and click **Process PDF** to begin."
    )


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.write(
                        f"📄 {source['source']} "
                        f"— Page {source['page']}"
                    )


# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your PDF..."
)


if question:

    # Check PDF
    if st.session_state.db is None:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the PDF..."
        ):

            result = answer_question(
                question,
                st.session_state.db
            )

        st.markdown(
            result["answer"]
        )

        if result["sources"]:

            with st.expander(
                "📚 Sources"
            ):

                for source in result["sources"]:

                    st.write(
                        f"📄 {source['source']} "
                        f"— Page {source['page']}"
                    )

    # Save answer
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        }
    )