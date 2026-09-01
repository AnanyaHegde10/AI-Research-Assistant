import os

from dotenv import load_dotenv
from google import genai


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# GEMINI CLIENT
# =========================================================

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY is not configured."
    )


client = genai.Client(
    api_key=api_key
)


# =========================================================
# ANSWER QUESTION
# =========================================================

def answer_question(
    question,
    db,
    chat_history=None
):
    """
    Answer a question using the FAISS database
    and Gemini.
    """

    # -----------------------------------------------------
    # Check database
    # -----------------------------------------------------

    if db is None:

        return (
            "❌ The PDF knowledge base is not loaded. "
            "Please upload and process your PDFs first."
        ), []


    # -----------------------------------------------------
    # Retrieve relevant chunks
    # -----------------------------------------------------

    docs = db.similarity_search(
        question,
        k=5
    )


    # -----------------------------------------------------
    # Build PDF context
    # -----------------------------------------------------

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )


    # -----------------------------------------------------
    # Build conversation history
    # -----------------------------------------------------

    conversation = ""

    if chat_history:

        for message in chat_history:

            role = message.get("role")
            content = message.get("content", "")

            if role == "user":

                conversation += (
                    f"User: {content}\n"
                )

            elif role == "assistant":

                conversation += (
                    f"Assistant: {content}\n"
                )


    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are an AI Research Assistant that answers
questions using the uploaded PDF documents.

Your answers must be based ONLY on the retrieved
PDF context.

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
   explain it clearly and accurately.

5. If the answer is not present in the PDF context,
   say exactly:

"I couldn't find that information in the uploaded PDFs."

6. Keep the answer clear and useful for a student.

7. If the question asks for syntax or code and the
   PDF contains an example, include the example.

8. Do not invent examples and do not add information
   from outside the uploaded PDFs.

ANSWER:
"""


    # -----------------------------------------------------
    # Gemini
    # -----------------------------------------------------

    try:

        response = client.models.generate_content(

            model="models/gemini-flash-lite-latest",

            contents=prompt
        )

        answer = response.text

        return answer, docs

    except Exception as e:

        print("Gemini error:", e)

        return (
            "❌ Sorry, I encountered an error while "
            "generating the answer."
        ), docs