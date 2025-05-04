# app.py for Hugging Face Spaces
import streamlit as st
import os
import tempfile
from app.services.RAG_Chroma import DocumentQA
from app.core.config import settings  # Your DocumentQA class
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = DocumentQA(model_id=settings.MODEL_NAME,hf_token=HF_TOKEN)
    st.session_state.documents = []
    st.session_state.uploaded = False

st.title("Document Q&A System")

# Sidebar for settings
with st.sidebar:
    st.header("Options")
    top_k = st.slider("Number of context chunks", 1, 10, 3)
    
    st.header("About")
    st.info("This is a document Q&A system built with Streamlit.")
    
    # Optional: Debug buttons
    if st.button("Get DB Length"):
        st.write(st.session_state.qa_system.get_db_length())

# Two tabs for Upload and Ask
tab1, tab2 = st.tabs(["Upload Document", "Ask Questions"])

with tab1:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Save file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        doc_id = uploaded_file.name
        # Process button
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                # Process document using your DocumentQA class
                doc_id = st.session_state.qa_system.add_document(file_path,doc_id)
                st.session_state.uploaded = True
                st.session_state.documents.append(doc_id)
                st.success(f"Document processed successfully!")

with tab2:
    st.header("Ask Questions")
    
    if not st.session_state.uploaded:
        st.warning("Please upload and process a document first.")
    else:
        question = st.text_input("Enter your question")
        
        if st.button("Ask"):
            if not question:
                st.error("Please enter a question.")
            else:
                with st.spinner("Generating answer..."):
                    # Use your DocumentQA class to answer
                    answer, context = st.session_state.qa_system.answer_question(question, top_k)
                    
                    # Display results
                    st.subheader("Answer")
                    st.write(answer)
                    
                    with st.expander("Context", expanded=False):
                        st.write(context)