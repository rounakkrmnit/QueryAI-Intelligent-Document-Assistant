from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):
    """
    Create a FAISS vector store from document chunks.

    Args:
        chunks: List of dictionaries containing text and page number.
        embedding_model: HuggingFace embedding model.

    Returns:
        FAISS vector store.
    """

    texts = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "page_number": chunk["page_number"]
        }
        for chunk in chunks
    ]

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )

    return vector_store