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


## 🧠 Challenges Faced & Solutions

Building a production-ready RAG pipeline using live cloud APIs presented several real-world engineering challenges:

1. **API Model Deprecation (404 NOT_FOUND Error)**
   - *Challenge:* During development, Google rapidly deprecated the `models/embedding-001` and `text-embedding-004` endpoints, causing fatal 404 errors during vectorization. Furthermore, `gemini-2.5-flash` was restricted for new API users.
   - *Solution:* Debugged the API error logs, researched Google's latest v1beta documentation, and successfully migrated the architecture to the new 2026 standards: `gemini-embedding-001` for vectors and `gemini-3.5-flash` for the LLM chain.

2. **LangChain 1.0 Ecosystem Restructuring (ModuleNotFoundError)**
   - *Challenge:* LangChain recently upgraded to version 1.0, which completely restructured the library and removed the legacy `chains` module from the main package, breaking the RAG retrieval chain.
   - *Solution:* Identified the breaking package changes and integrated the new `langchain-classic` dependency to restore the `load_qa_chain` functionality without rewriting the entire prompt architecture.

3. **UI State Management (Amnesia Bug)**
   - *Challenge:* Streamlit's architecture reruns the entire script upon every user interaction, causing the chatbot to "forget" previous questions and erase the visual chat history.
   - *Solution:* Engineered a session state memory system using `st.session_state` to capture, append, and re-render the conversation history chronologically on every page load.