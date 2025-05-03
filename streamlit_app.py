import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()
# Page configuration
st.set_page_config(page_title="Document Q&A System", layout="wide")

hf_token = os.getenv("HUGGING_FACE_TOKEN") or st.secrets.get("HUGGING_FACE_TOKEN", None)

# Add a header and description
st.title("Document Q&A System")
st.markdown("Upload a document and ask questions about its contents")

# Create two columns - one for upload, one for chat
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Check if document is already processed (you can track this in session state)
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                # Send file to your FastAPI backend
                files = {"file":(uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                params = {"chunk_size":500}
                upload_response = requests.post("http://127.0.0.1:8000/upload_document", files=files,params= params)

                
                if upload_response.status_code == 200:
                    st.session_state.doc_processed = True
                    st.success("Document processed successfully!")
                else:
                    st.error(f"Error processing document: {upload_response.status_code}")

with col2:
    st.subheader("Ask Questions")
    
    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for q, a in st.session_state.chat_history:
        st.markdown(f"**Q: {q}**")
        st.markdown(f"A: {a}")
        st.markdown("---")
    
    # Query input
    user_input = st.text_input("Enter your question")
    
    if st.button("Ask"):
        if user_input:
            with st.spinner("Getting answer..."):
                response = requests.post(
                f"http://127.0.0.1:8000/ask?query={user_input}&top_k=3"
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "No answer provided")
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_input, answer))
                    
                    # Display the newest result
                    st.markdown(f"**Q: {user_input}**")
                    st.markdown(f"A: {answer}")
                else:
                    st.error(f"Error: {response.status_code}")
        else:
            st.warning("Please enter a question")

# Add a sidebar with additional options
with st.sidebar:
    st.header("Options")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()
    
    st.subheader("About")
    st.info("This is a document Q&A system built with Streamlit and FastAPI.")



    st.subheader("get db length")
    if st.button('len_db'):
        with st.spinner("Getting len..."):
                response = requests.get(
                f"http://127.0.0.1:8000/store"
                )
                data = response.json()
                st.json(data)
        st.subheader("get db length")
    if st.button('get_ids'):
        with st.spinner("Getting ids..."):
                response = requests.get(
                f"http://127.0.0.1:8000/get_ids"
                )
                data = response.json()
                st.json(data)
        