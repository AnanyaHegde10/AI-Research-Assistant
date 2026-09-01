import os

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

    if not chunks:

        raise ValueError(
            "No text chunks were created from the PDFs."
        )


    vector_db = FAISS.from_texts(
        texts=chunks,
        embedding=embedding_model
    )


    vector_db.save_local(
        "faiss_db"
    )


    return vector_db


# =========================================================
# LOAD VECTOR DATABASE
# =========================================================

def load_vector_db():

    index_path = os.path.join(
        "faiss_db",
        "index.faiss"
    )

    if not os.path.exists(index_path):

        return None


    db = FAISS.load_local(

        "faiss_db",

        embedding_model,

        allow_dangerous_deserialization=True
    )


    return db