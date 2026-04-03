# pdf_reader_fun.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
import os
import time

def pdf_reader(pdf_path):
    start_time = time.time()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 📁 Create DB folder
    os.makedirs("chroma_db", exist_ok=True)

    # 📄 Load PDF
    loader = PyPDFLoader(file_path=pdf_path)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages")

    # ✂️ Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )
    all_splits = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(all_splits)} chunks")

    # 🔥 FIX: Use Fake Embeddings (Cloud Safe)
    embeddings = FakeEmbeddings(size=384)

    print("Creating vector database...")

    # 🧠 Create vector DB (simplified)
    vectordb = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    end_time = time.time()
    print(f"Vector DB ready! Time taken: {end_time - start_time:.2f} seconds")

    return vectordb
