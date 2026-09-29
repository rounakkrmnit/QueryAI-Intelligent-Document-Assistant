# QueryAI – Intelligent Document Assistant

QueryAI is an AI-powered document question-answering application that allows users to upload PDF documents and ask questions about their content using a Retrieval-Augmented Generation (RAG) pipeline.

The application extracts text from documents, splits it into smaller chunks, generates vector embeddings, stores them in a FAISS vector store, retrieves relevant information for a user query, and uses an LLM to generate a context-aware answer.

---

## 🚀 Features

- 📄 Upload PDF documents
- 🔍 Ask natural-language questions about documents
- 🧩 Automatic document text extraction
- ✂️ Text chunking for efficient retrieval
- 🧠 Semantic search using embeddings
- 📚 Retrieval-Augmented Generation (RAG)
- 🤖 LLM-powered question answering
- 📑 Page-level metadata
- 🌐 Interactive Streamlit interface

---

## 🏗️ System Architecture

```text
                         PDF Document
                              │
                              ▼
                     ┌─────────────────┐
                     │ Document Loader │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Text Extraction │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Text Splitter   │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │   Embeddings    │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ FAISS Vector DB │
                     └────────┬────────┘
                              │
                              │
User Query ───────────────────┘
     │
     ▼
┌─────────────────┐
│ Query Embedding │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Similarity      │
│ Search          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Relevant Chunks │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      LLM        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generated Answer│
└─────────────────┘
```
