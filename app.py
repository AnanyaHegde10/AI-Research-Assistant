import os

import streamlit as st

from pypdf import PdfReader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from vector_store import (
    create_vector_db,
    load_vector_db
)

from rag import answer_question


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(

    page_title="AI Research Assistant",

    page_icon="🤖",

    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🤖 AI Research Assistant")

st.caption(
    "Upload multiple PDFs and chat with them using AI-powered RAG"
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


if "vector_db" not in st.session_state:

    st.session_state.vector_db = None


# =========================================================
# TRY TO LOAD EXISTING DATABASE
# =========================================================

if st.session_state.vector_db is None:

    existing_db = load_vector_db()

    if existing_db is not None:

        st.session_state.vector_db = existing_db

        st.session_state.processed = True


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 PDF Library")

    st.write(
        "Upload one or more PDF documents and "
        "build your knowledge base."
    )


    # -----------------------------------------------------
    # PDF UPLOADER
    # -----------------------------------------------------

    uploaded_files = st.file_uploader(

        "Upload PDF documents",

        type=["pdf"],

        accept_multiple_files=True
    )


    # -----------------------------------------------------
    # SHOW SELECTED FILES
    # -----------------------------------------------------

    if uploaded_files:

        st.write(
            f"📄 **{len(uploaded_files)} PDF(s) selected**"
        )


    # -----------------------------------------------------
    # PROCESS PDFs
    # -----------------------------------------------------

    if uploaded_files:

        if st.button(
            "⚡ Process PDFs",
            use_container_width=True
        ):

            all_chunks = []


            # Create uploads folder

            os.makedirs(
                "uploads",
                exist_ok=True
            )


            # -------------------------------------------------
            # PROCESSING
            # -------------------------------------------------

            with st.spinner(
                "📚 Reading and processing PDFs..."
            ):

                for uploaded_file in uploaded_files:

                    # -----------------------------------------
                    # Save PDF
                    # -----------------------------------------

                    file_path = os.path.join(
                        "uploads",
                        uploaded_file.name
                    )


                    with open(
                        file_path,
                        "wb"
                    ) as f:

                        f.write(
                            uploaded_file.getbuffer()
                        )


                    # -----------------------------------------
                    # Read PDF
                    # -----------------------------------------

                    reader = PdfReader(
                        file_path
                    )


                    text = ""


                    for page in reader.pages:

                        page_text = page.extract_text()


                        if page_text:

                            text += (
                                page_text
                                + "\n\n"
                            )


                    # -----------------------------------------
                    # Check extracted text
                    # -----------------------------------------

                    if not text.strip():

                        st.warning(
                            f"⚠️ No readable text found in "
                            f"{uploaded_file.name}"
                        )

                        continue


                    # -----------------------------------------
                    # Split text
                    # -----------------------------------------

                    splitter = RecursiveCharacterTextSplitter(

                        chunk_size=1500,

                        chunk_overlap=300
                    )


                    chunks = splitter.split_text(
                        text
                    )


                    # -----------------------------------------
                    # Add chunks
                    # -----------------------------------------

                    all_chunks.extend(
                        chunks
                    )


            # -------------------------------------------------
            # CHECK CHUNKS
            # -------------------------------------------------

            if not all_chunks:

                st.error(
                    "❌ No readable text was found in "
                    "the uploaded PDFs."
                )

                st.stop()


            # -------------------------------------------------
            # CREATE FAISS DATABASE
            # -------------------------------------------------

            with st.spinner(
                "🧠 Building AI knowledge base..."
            ):

                vector_db = create_vector_db(
                    all_chunks
                )


            # -------------------------------------------------
            # SAVE DATABASE TO SESSION
            # -------------------------------------------------

            st.session_state.vector_db = vector_db

            st.session_state.processed = True


            # -------------------------------------------------
            # SAVE FILE NAMES
            # -------------------------------------------------

            st.session_state.uploaded_files = [

                file.name

                for file in uploaded_files

            ]


            # -------------------------------------------------
            # CLEAR OLD CHAT
            # -------------------------------------------------

            st.session_state.chat_history = []


            # -------------------------------------------------
            # SUCCESS
            # -------------------------------------------------

            st.success(
                f"✅ {len(uploaded_files)} PDF(s) "
                f"processed successfully!"
            )


    # =====================================================
    # CURRENTLY LOADED FILES
    # =====================================================

    if st.session_state.uploaded_files:

        st.divider()

        st.subheader("📂 Currently loaded")


        for file_name in st.session_state.uploaded_files:

            st.write(
                f"📄 **{file_name}**"
            )


    # =====================================================
    # RAG STATUS
    # =====================================================

    if st.session_state.vector_db is not None:

        st.success(
            "🟢 RAG System Ready\n\n"
            "Your PDFs are ready for questions."
        )


    # =====================================================
    # CHAT CONTROLS
    # =====================================================

    st.divider()

    st.subheader("⚙️ Chat Controls")


    # -----------------------------------------------------
    # NEW CHAT
    # -----------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


    # -----------------------------------------------------
    # CLEAR CHAT
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
# EMPTY STATE
# =========================================================

if not st.session_state.processed:

    st.info(
        "📚 Upload one or more PDFs from the sidebar "
        "and click **Process PDFs** to get started."
    )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

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
# PROCESS QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # CHECK DATABASE
    # -----------------------------------------------------

    if st.session_state.vector_db is None:

        st.warning(
            "⚠️ Please upload and process a PDF first."
        )

        st.stop()


    # -----------------------------------------------------
    # DISPLAY USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    st.session_state.chat_history.append({

        "role": "user",

        "content": question

    })


    # -----------------------------------------------------
    # GENERATE ANSWER
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching your PDFs..."
        ):

            answer, docs = answer_question(

                question,

                st.session_state.vector_db,

                st.session_state.chat_history

            )


        # ---------------------------------------------
        # DISPLAY ANSWER
        # ---------------------------------------------

        st.markdown(
            answer
        )


        # ---------------------------------------------
        # DISPLAY SOURCES
        # ---------------------------------------------

        if docs:

            with st.expander(
                "📚 View Sources"
            ):

                for i, doc in enumerate(
                    docs,
                    1
                ):

                    st.markdown(
                        f"### Source {i}"
                    )


                    st.write(
                        doc.page_content[:700]
                    )


                    st.divider()


    # -----------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # -----------------------------------------------------

    st.session_state.chat_history.append({

        "role": "assistant",

        "content": answer

    })