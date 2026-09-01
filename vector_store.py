from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

def create_vector_db(chunks):
    """
    Create FAISS vector database from text chunks
    and save it locally.
    """

    if not chunks:
        raise ValueError("No text chunks were found in the PDFs.")

    vector_db = FAISS.from_texts(
        texts=chunks,
        embedding=embedding_model
    )

    vector_db.save_local("faiss_db")

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

    try:

        db = FAISS.load_local(
            "faiss_db",
            embedding_model,
            allow_dangerous_deserialization=True
        )

        return db

    except Exception as e:

        print("FAISS database could not be loaded:", e)

        return None