import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings


@st.cache_resource
def create_embedding_model():
    """
    Create and cache the embedding model.

    The model is loaded once and reused across
    Streamlit reruns.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings