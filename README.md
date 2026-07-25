# 📄 DocuChat AI: Enterprise RAG Engine

An end-to-end Retrieval-Augmented Generation (RAG) system built with Python, Streamlit, LangChain, and Google Gemini. DocuChat enables users to upload multi-page PDF documents, indexes the content into a local vector database, and allows precise context-aware document querying.

## 🚀 Features
- **Multi-PDF Extraction:** Parses raw text across uploaded documents using `pypdf`.
- **Recursive Chunking:** Splits long-form text into overlapping chunks (1,000 chars with 200 overlap) to preserve semantic context.
- **Vector Embeddings:** Uses Google's `gemini-embedding-001` to convert text into mathematical vectors.
- **Local Similarity Search:** Leverages `FAISS` (Facebook AI Similarity Search) for fast local vector retrieval.
- **Strict RAG QA Chain:** Powered by `gemini-3.5-flash` with strict prompt constraints to eliminate hallucinations.

## 🛠️ Tech Stack
- **Language:** Python
- **Frontend UI:** Streamlit
- **LLM & Embeddings:** Google Gemini API (`gemini-3.5-flash`, `gemini-embedding-001`)
- **Orchestration:** LangChain (`langchain-classic`)
- **Vector Database:** FAISS

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/DocuChat-AI.git](https://github.com/YOUR_USERNAME/DocuChat-AI.git)
   cd DocuChat-AI