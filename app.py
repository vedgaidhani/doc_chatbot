import time
import streamlit as st
from pypdf import PdfReader
from google import genai

from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain & Vector Store Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings



# 1. PAGE SETUP and API CONFIGURATION

st.set_page_config(page_title="DocuChat AI", page_icon="📄")
st.title("📄 DocuChat AI: Enterprise RAG Engine")

# Securely retrieve the Gemini API Key
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
    if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
except Exception:
    st.error("🚨 Secret vault not found. Please check .streamlit/secrets.toml")
    st.stop()

    



# 2.  HELPER FUNCTIONS (THE RAG ENGINE)...


def get_pdf_text(pdf_docs):
    """Extracts raw text from all uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def get_text_chunks(text):
    """Splits raw text into 1,000-character chunks with 200-character overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Converts text chunks into mathematical vectors and saves them locally."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """Creates a strict QA chain that only answers from the provided PDF context."""
    prompt_template = """
    You are an intelligent document assistant. Answer the user's question as detailed as possible using ONLY the provided context. 
    If the answer is not in the provided context, strictly reply with: "The answer is not available in the uploaded document." 
    Do not invent or guess answers.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    
    
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3, google_api_key=API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



# 3. SIDEBAR UI (DOCUMENT INGESTION)

with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    
    if st.button("Submit & Process"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF file first!")
        else:
            with st.spinner("Processing & Indexing Document..."):
                # Step A: Extract raw text from uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                
                # Step B: Divide the long text into 1,000-char chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Step C: Convert chunks into vectors & save to 'faiss_index' folder
                get_vector_store(text_chunks)
                
                st.success("Document successfully indexed into Vector Database!")
                
                # Optional: Show a quick preview of the extracted text
                with st.expander("Preview Extracted Text"):
                    st.write(raw_text[:500] + "...")


# 4. MAIN CHAT INTERFACE (RAG ENABLED)


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask a question about the document...")

if user_input:
    # 1. Display user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 2. Process AI Response
    with st.spinner("Searching document database..."):
        try:
            # Re-initialize embeddings to read the local database
            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001", 
                google_api_key=API_KEY
            )
            
            # Load the local vector database...
            new_db = FAISS.load_local(
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Search the database for chunks matching the users question
            docs = new_db.similarity_search(user_input)
            
            # Load the strict prompt rules...
            chain = get_conversational_chain()  
            
            
            # Feed the matched document chunks and the question into Gemini...
            response = chain(
                {"input_documents": docs, "question": user_input}, 
                return_only_outputs=True
            )

            ai_replay = response["output_text"]


            # Display the AI  final answer...
            st.session_state.chat_history.append({"role": "ai", "content": ai_replay})
            with st.chat_message("ai"):
                st.write(ai_replay)
                
        except Exception as e:
            st.error(f"🚨 RAG Processing Error: {str(e)}")