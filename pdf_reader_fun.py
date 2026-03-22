# pdf_reader_fun.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os, time

model_name = "sentence-transformers/all-mpnet-base-v2"
batch_size = 166

def pdf_reader(pdf_path):
    start_time = time.time()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    os.makedirs("chroma_db", exist_ok=True)

    loader = PyPDFLoader(file_path=pdf_path)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    all_splits = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(all_splits)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Creating vector database...")

    vectordb = None
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i + batch_size]
        if vectordb is None:
            vectordb = Chroma.from_documents(documents=batch, embedding=embeddings, persist_directory="chroma_db")
        else:
            vectordb.add_documents(documents=batch)

    end_time = time.time()
    print(f"Vector DB ready! Time taken: {end_time - start_time:.2f} seconds")
    return vectordb
