import os
import shutil

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# FAISS DATABASE PATH
# =========================================================

FAISS_PATH = "faiss_db"


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

def create_vector_db(documents):
    """
    Create a FAISS vector database from LangChain Documents.

    Each Document contains:
        - page_content
        - metadata
    """

    if not documents:
        raise ValueError(
            "No documents were provided to create the vector database."
        )

    # -----------------------------------------------------
    # Remove old FAISS database
    # -----------------------------------------------------

    if os.path.exists(FAISS_PATH):

        shutil.rmtree(FAISS_PATH)


    # -----------------------------------------------------
    # Create new FAISS database
    # -----------------------------------------------------

    vector_db = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model
    )


    # -----------------------------------------------------
    # Save database
    # -----------------------------------------------------

    vector_db.save_local(
        FAISS_PATH
    )


    return vector_db


# =========================================================
# LOAD VECTOR DATABASE
# =========================================================

def load_vector_db():
    """
    Load previously saved FAISS database.

    Returns:
        FAISS database if it exists.
        None if database does not exist.
    """

    if not os.path.exists(FAISS_PATH):

        return None


    db = FAISS.load_local(

        FAISS_PATH,

        embedding_model,

        allow_dangerous_deserialization=True
    )


    return db