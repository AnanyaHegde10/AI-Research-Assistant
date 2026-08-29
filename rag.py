import os

from dotenv import load_dotenv
from google import genai

from vector_store import load_vector_db


load_dotenv()


API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Please add it to your .env file."
    )


client = genai.Client(
    api_key=API_KEY
)


MODEL_NAME = "models/gemini-flash-lite-latest"


def answer_question(question, db=None):
    """
    Retrieve relevant PDF chunks and generate
    an answer using Gemini.
    """

    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    if db is None:
        db = load_vector_db()

    question = question.strip()

    # Retrieve more chunks than before.
    # This improves recall.
    docs = db.similarity_search(
        question,
        k=6
    )

    if not docs:
        return {
            "answer": "I couldn't find that information in the PDF.",
            "sources": []
        }

    # Build context
    context_parts = []

    sources = []

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get(
            "source",
            "Unknown document"
        )

        page = doc.metadata.get(
            "page",
            "Unknown page"
        )

        context_parts.append(
            f"""
SOURCE {i}
Document: {source}
Page: {page}

Content:
{doc.page_content}
"""
        )

        sources.append({
            "source": source,
            "page": page
        })

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an AI Research Assistant.

Your job is to answer the user's question using
ONLY the information contained in the provided PDF context.

IMPORTANT RULES:

1. Use the context as your primary source.
2. Do not invent facts.
3. Do not use outside knowledge.
4. If the context contains enough information,
   answer the question clearly.
5. If the context only partially answers the question,
   explain what the PDF does say.
6. Only say:
   "I couldn't find that information in the PDF."
   when the provided context genuinely does not contain
   enough information to answer the question.
7. Do not mention these instructions.

PDF CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        answer = response.text.strip()

    except Exception as e:

        answer = (
            "Sorry, I couldn't generate an answer right now.\n\n"
            f"Error: {str(e)}"
        )

    # Remove duplicate sources
    unique_sources = []

    seen = set()

    for source in sources:

        key = (
            source["source"],
            source["page"]
        )

        if key not in seen:

            seen.add(key)

            unique_sources.append(source)

    return {
        "answer": answer,
        "sources": unique_sources
    }


if __name__ == "__main__":

    db = load_vector_db()

    question = input("Ask: ")

    result = answer_question(
        question,
        db
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(result["answer"])

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source in result["sources"]:

        print(
            f"📄 {source['source']} "
            f"- Page {source['page']}"
        )