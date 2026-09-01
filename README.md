# 🤖 AI Research Assistant

An AI-powered application that allows users to upload multiple PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG).

The application searches the uploaded PDFs, retrieves the most relevant information, and uses Google's Gemini AI to generate clear answers based only on the retrieved document content.

---

## ✨ Features

- 📚 Upload multiple PDF documents
- 🔎 Semantic search using FAISS
- 🧠 Retrieval-Augmented Generation (RAG)
- 🤖 Google Gemini AI
- 💬 Chat with uploaded PDFs
- 🧠 Conversation memory
- 📄 PDF source references
- 📖 Page-level source information
- ⚡ Fast vector similarity search
- 🎨 Streamlit user interface
- 🔐 API key stored securely using environment variables

---

## 🏗️ Project Architecture

```text
User
  │
  ▼
Streamlit UI
  │
  ▼
Upload PDF
  │
  ▼
PyPDF
  │
  ▼
Text Extraction
  │
  ▼
Text Chunking
  │
  ▼
HuggingFace Embeddings
  │
  ▼
FAISS Vector Database
  │
  ▼
User Question
  │
  ▼
Similarity Search
  │
  ▼
Relevant PDF Chunks
  │
  ▼
Gemini AI
  │
  ▼
Answer + Sources