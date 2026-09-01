import os

from dotenv import load_dotenv
from google import genai

from vector_store import load_vector_db


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


# =========================================================
# GEMINI MODEL
# =========================================================

MODEL_NAME = "models/gemini-flash-lite-latest"


# =========================================================
# ANSWER QUESTION
# =========================================================

def answer_question(question, chat_history=None):

    # -----------------------------------------------------
    # Load FAISS database
    # -----------------------------------------------------

    db = load_vector_db()


    if db is None:

        return (
            "I couldn't find a processed PDF database. "
            "Please upload and process your PDFs first."
        ), []


    # -----------------------------------------------------
    # Prepare conversation history
    # -----------------------------------------------------

    conversation = ""

    if chat_history:

        # Use recent conversation only
        recent_history = chat_history[-10:]

        for message in recent_history:

            role = message.get("role", "")
            content = message.get("content", "")

            if role == "user":

                conversation += (
                    f"User: {content}\n"
                )

            elif role == "assistant":

                conversation += (
                    f"Assistant: {content}\n"
                )


    # =====================================================
    # STEP 1: CREATE SEARCH QUERY
    # =====================================================

    # The purpose of this step is to understand questions
    # such as:
    #
    # "give me its example"
    #
    # where "its" refers to something from the previous
    # conversation.

    query_prompt = f"""
You are helping an AI Research Assistant search
uploaded PDF documents.

CONVERSATION HISTORY:
{conversation}

CURRENT QUESTION:
{question}

Create a short search query that represents what the
user is asking about.

Resolve references such as:
- it
- its
- this
- that
- they
- them
- above
- previous

Do NOT answer the question.

Return ONLY the search query.
"""


    # -----------------------------------------------------
    # Generate search query
    # -----------------------------------------------------

    try:

        query_response = client.models.generate_content(

            model=MODEL_NAME,

            contents=query_prompt
        )

        search_query = query_response.text.strip()

    except Exception:

        # If query generation fails, simply use the
        # original question.

        search_query = question


    # =====================================================
    # STEP 2: RETRIEVE DOCUMENTS
    # =====================================================

    docs = db.similarity_search(

        search_query,

        k=5
    )


    # =====================================================
    # STEP 3: BUILD PDF CONTEXT
    # =====================================================

    context_parts = []


    for i, doc in enumerate(docs, 1):

        metadata = doc.metadata or {}

        source = metadata.get(
            "source",
            "Unknown PDF"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )


        context_parts.append(

            f"""
SOURCE {i}
PDF: {source}
PAGE: {page}

CONTENT:
{doc.page_content}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # =====================================================
    # STEP 4: BUILD FINAL PROMPT
    # =====================================================

    prompt = f"""
You are an AI Research Assistant that answers questions
using ONLY the uploaded PDF documents.

Your answers are intended for students.

CONVERSATION HISTORY:
{conversation}

RETRIEVED PDF CONTEXT:
{context}

CURRENT QUESTION:
{question}


IMPORTANT RULES:

1. Answer ONLY using the retrieved PDF context.

2. Use the conversation history only to understand
   references such as:
   "it", "its", "this", "that", "they", "them",
   "above", or "previous".

3. Do NOT use outside knowledge.

4. If the requested information is present in the
   retrieved PDF context, explain it clearly.

5. If the information is NOT present in the retrieved
   PDF context, respond EXACTLY with:

"I couldn't find that information in the uploaded PDFs."

6. Do not invent examples, syntax, definitions,
   page numbers, or facts.

7. If the user asks for syntax and the PDF contains
   syntax, provide it.

8. If the user asks for an example and the PDF contains
   an example, provide it.

9. If the PDF contains a code example, preserve the
   code accurately.

10. Keep explanations simple and useful for a student.

11. Use Markdown headings, bullet points and code blocks
    when they improve readability.

12. Do not mention the retrieval process.

13. Do not say "according to my knowledge".

14. Do not answer from general programming knowledge.

ANSWER:
"""


    # =====================================================
    # STEP 5: GEMINI ANSWER
    # =====================================================

    response = client.models.generate_content(

        model=MODEL_NAME,

        contents=prompt
    )


    answer = response.text


    # =====================================================
    # RETURN ANSWER + SOURCE DOCUMENTS
    # =====================================================

    return answer, docs