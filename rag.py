# rag.py
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ✅ Get API key safely
api_key = st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets")
    st.stop()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run(vectordb, query, template):
    try:
        # 🔍 Retriever
        retriever = vectordb.as_retriever()

        # 🧠 Prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "query"]
        )

        # 🤖 LLM
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=api_key
        )

        # 🔗 RAG Chain
        rag_chain = (
            {
                "context": retriever | format_docs,
                "query": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # 🚀 Run
        response = rag_chain.invoke(query)
        return response

    except Exception as e:
        return f"❌ Error: {str(e)}"
