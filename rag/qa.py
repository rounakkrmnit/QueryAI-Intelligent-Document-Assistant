import os
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv(override=True)


@st.cache_resource
def create_llm():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found")

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key
    )


def answer_question(llm, vector_store, question, k=3):
    """
    Retrieve relevant document chunks and generate an answer
    using the LLM.
    """

    results = vector_store.similarity_search(
        question,
        k=k
    )

    context_parts = []

    for result in results:
        page = result.metadata.get("page_number", "Unknown")

        context_parts.append(
            f"[Page {page}]\n{result.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided document context.

If the answer cannot be found in the context, say:
"I couldn't find this information in the provided document."

Do not invent or assume information.

Document Context:
-----------------
{context}
-----------------

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content

    if isinstance(answer, list):
        answer = "".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict) and item.get("type") == "text"
        )

    return {
        "answer": answer,
        "sources": results
    }