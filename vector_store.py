from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


FAISS_DIR = "faiss_db"

# Load embedding model once
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_db(documents):
    """
    Create a FAISS vector database from LangChain Documents.
    """

    if not documents:
        raise ValueError("No documents were provided.")

    vector_db = FAISS.from_documents(
        documents,
        embedding_model
    )

    vector_db.save_local(FAISS_DIR)

    return vector_db


def load_vector_db():
    """
    Load the saved FAISS vector database.
    """

    index_path = Path(FAISS_DIR) / "index.faiss"

    if not index_path.exists():
        raise FileNotFoundError(
            "FAISS database not found. Please upload and process a PDF first."
        )

    db = FAISS.load_local(
        FAISS_DIR,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db