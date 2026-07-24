import streamlit as st
from google import genai
import time
from pypdf import PdfReader


st.set_page_config(page_title="DocuChat AI", page_icon="📄")
st.title("📄 DocuChat AI: Enterprise RAG Engine")


try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except FileNotFoundError:
    st.error("🚨 Secret vault not found. Please create .streamlit/secrets.toml")
    st.stop()



with st.sidebar:
    st.header("📂 Document Vault")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            # Read the file directly from Streamlit's temporary memory
            pdf_reader = PdfReader(uploaded_file)
            extracted_text = ""
            
        
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + "\n"
            
        
            st.session_state['document_text'] = extracted_text
            
            st.success(f"Successfully loaded {len(pdf_reader.pages)} pages!")
            
            
            with st.expander("Preview Extracted Text"):
                st.write(extracted_text[:500] + "...")


user_input = st.chat_input("Ask a question about the document...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Brain is thinking..."):
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=user_input,
                )
                success = True
                break 
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg and attempt < max_retries - 1:
                    time.sleep(15) 
                    continue
                elif "503" in error_msg and attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1)) 
                    continue
                else:
                    st.error(f"🚨 API Error: {error_msg}")
                    break 
        
        if success:
            with st.chat_message("ai"):
                st.write(response.text)