# rag.py
import streamlit as st
from groq import Groq

# ✅ API key from secrets
api_key = st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets")
    st.stop()

client = Groq(api_key=api_key)


def run(vectordb, query, template):
    try:
        # 🔍 Retriever
        retriever = vectordb.as_retriever()

        # ✅ FIX: use invoke instead of get_relevant_documents
        docs = retriever.invoke(query)

        # 🧠 Context build
        context = "\n\n".join([doc.page_content for doc in docs])

        # 📝 Prompt
        final_prompt = template.format(context=context, query=query)

        # 🤖 Groq call
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"
