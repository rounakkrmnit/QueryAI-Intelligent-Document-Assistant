from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(pages):
    """
    Split extracted PDF pages into smaller text chunks.

    Args:
        pages: List of dictionaries containing page_number and text.

    Returns:
        List of dictionaries containing chunk text and page number.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page_number": page["page_number"]
            })

    return chunks