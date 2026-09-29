from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path):
    """
    Extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of dictionaries containing page text and page number.
    """

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "page_number": page_number,
                "text": text.strip()
            })

    return pages