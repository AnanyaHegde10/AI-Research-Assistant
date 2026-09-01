import os
import streamlit as st

from dotenv import load_dotenv
from google import genai

from vector_store import load_vector_db


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# GET GOOGLE API KEY
# =========================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:

    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

    except Exception:
        GOOGLE_API_KEY = None


# =========================================================
# CHECK API KEY
# =========================================================

if not GOOGLE_API_KEY:

    raise ValueError(
        "GOOGLE_API_KEY is not configured. "
        "Add it to your .env file locally or "
        "Streamlit Cloud Secrets when deploying."
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# =========================================================
# ANSWER QUESTION
# =========================================================

def answer_question(question, chat_history=None):

    # -----------------------------------------------------
    # LOAD FAISS DATABASE
    # -----------------------------------------------------

    db = load_vector_db()

    if db is None:

        raise ValueError(
            "The PDF knowledge base has not been created yet. "
            "Please upload and process your PDFs first."
        )


    # -----------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENTS
    # -----------------------------------------------------

    docs = db.similarity_search(
        question,
        k=5
    )


    # -----------------------------------------------------
    # BUILD PDF CONTEXT
    # -----------------------------------------------------

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )


    # -----------------------------------------------------
    # BUILD CONVERSATION HISTORY
    # -----------------------------------------------------

    conversation = ""

    if chat_history:

        for message in chat_history:

            role = message["role"]
            content = message["content"]

            if role == "user":

                conversation += (
                    f"User: {content}\n"
                )

            elif role == "assistant":

                conversation += (
                    f"Assistant: {content}\n"
                )


    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------

    prompt = f"""
You are an AI Research Assistant that answers questions
using the uploaded PDF documents.

Your job is to provide accurate answers based ONLY on
the retrieved PDF context.

CONVERSATION HISTORY:
{conversation}

RETRIEVED PDF CONTEXT:
{context}

CURRENT QUESTION:
{question}


INSTRUCTIONS:

1. Answer the current question using the PDF context.

2. You may use the conversation history only to
   understand references such as:
   "it", "this", "that", "they", etc.

3. Do NOT use outside knowledge.

4. If the answer is present in the PDF context,
   explain it clearly.

5. If the answer is NOT present in the PDF context,
   say exactly:

"I couldn't find that information in the uploaded PDFs."

6. Keep the answer clear and useful for a student.

7. If the question asks for code or syntax and the
   PDF contains an example, include the example.

8. Do not invent information that is not present
   in the retrieved PDF context.

9. When possible, mention the PDF name and page number
   from the retrieved document metadata.

ANSWER:
"""


    # -----------------------------------------------------
    # GEMINI
    # -----------------------------------------------------

    response = client.models.generate_content(

        model="models/gemini-flash-lite-latest",

        contents=prompt
    )


    # -----------------------------------------------------
    # RETURN ANSWER AND SOURCES
    # -----------------------------------------------------

    return response.text, docs