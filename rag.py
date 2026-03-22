# rag.py
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from groq import Groq
from pdf_reader_fun import pdf_reader

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)

def run(vectordb, query, template):
    print("Starting retrieval and generation process...")
    retriever = vectordb.as_retriever()

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "query"]
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)

    rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)
