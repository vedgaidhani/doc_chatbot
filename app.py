import streamlit as st
from google import genai
import time

# 1. Page Setup
st.set_page_config(page_title="DocuChat AI", page_icon="📄")
st.title("📄 DocuChat AI: Enterprise RAG Engine")

# 2. Secure API Key Loading
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except FileNotFoundError:
    st.error("🚨 Secret vault not found. Please create .streamlit/secrets.toml")
    st.stop()

# 3. The Chat Interface
user_input = st.chat_input("Say hello to your new AI brain...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Brain is thinking (managing API quotas)..."):
        max_retries = 3
        success = False
        
        # Exponential Backoff & Rate Limit Loop
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=user_input,
                )
                success = True
                break  # Success! Break out of the loop.
                
            except Exception as e:
                error_msg = str(e)
                
                # Check for Rate Limit (429)
                if "429" in error_msg and attempt < max_retries - 1:
                    time.sleep(15)  # Wait 15 seconds for quota to reset
                    continue
                    
                # Check for Server Overload (503)
                elif "503" in error_msg and attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))  # 2s, 4s backoff
                    continue
                    
                # If it's any other error, or we ran out of retries, stop and show it
                else:
                    st.error(f"🚨 API Error: {error_msg}")
                    break 
        
        # 4. Safe Rendering
        if success:
            with st.chat_message("ai"):
                st.write(response.text)