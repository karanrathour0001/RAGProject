# app.py
import streamlit as st
from pdf_reader_fun import pdf_reader
from rag import run
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found in .env")
    st.stop()

st.title("📄 RAG PDF Reader with Groq LLM")

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save PDF temporarily
    pdf_path = f"./pdfs/{uploaded_file.name}"
    os.makedirs("./pdfs", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"Uploaded: {uploaded_file.name}")
    
    # Process PDF
    with st.spinner("Processing PDF and creating embeddings..."):
        vectordb = pdf_reader(pdf_path)
    
    st.success("PDF processed successfully!")

    # Ask Query
    query = st.text_input("Enter your question about the document:")
    
    if query:
        prompt_template = "You are a legal expert. Using the provided data: {context}, answer the question: {query}"
        
        with st.spinner("Getting answer from LLM..."):
            answer = run(vectordb, query, prompt_template)
        
        st.subheader("Answer:")
        st.write(answer)
