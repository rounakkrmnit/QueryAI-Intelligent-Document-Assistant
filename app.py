import streamlit as st
import tempfile
import os

from rag.loader import load_pdf
from rag.splitter import split_documents
from rag.embeddings import create_embedding_model
from rag.vectorstore import create_vector_store
from rag.qa import create_llm, answer_question


# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="QueryAI",
    page_icon="📄",
    layout="wide"
)


# ---------------------------------------
# Session State
# ---------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------
# Custom Styling
# ---------------------------------------

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 2rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Main title */
    .queryai-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .queryai-subtitle {
        font-size: 1.15rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }

    /* Cards */
    .card {
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid #30343b;
        background-color: #17191f;
        margin-bottom: 1rem;
    }

    /* Answer box */
    .answer-box {
        padding: 1.5rem;
        border-radius: 14px;
        background-color: #17191f;
        border: 1px solid #30343b;
        margin-top: 1rem;
        line-height: 1.7;
    }

    /* Source box */
    .source-box {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background-color: #17191f;
        border: 1px solid #30343b;
        margin-bottom: 0.5rem;
    }

    /* Section labels */
    .section-label {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    /* Hide form submit button */
    div[data-testid="stFormSubmitButton"] {
      display: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------
# Header
# ---------------------------------------

st.markdown(
    '<div class="queryai-title">📄 QueryAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="queryai-subtitle">'
    'Intelligent Document Assistant powered by RAG'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------
# Upload Section
# ---------------------------------------

st.markdown(
    '<div class="section-label">📁 Upload Document</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    label_visibility="collapsed"
)


# ---------------------------------------
# Process PDF
# ---------------------------------------

if uploaded_file is not None:

    file_name = uploaded_file.name

    # Process only when a new file is uploaded
    if (
        "processed_file" not in st.session_state
        or st.session_state.processed_file != file_name
    ):

        with st.spinner("Processing your document..."):

            # Create temporary PDF
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                pdf_path = temp_file.name

            # Extract text
            pages = load_pdf(pdf_path)

            # Split text into chunks
            chunks = split_documents(pages)

            # Create embedding model
            embedding_model = create_embedding_model()

            # Create FAISS vector store
            vector_store = create_vector_store(
                chunks,
                embedding_model
            )

            # Create Gemini LLM
            llm = create_llm()

            # Store objects in session state
            st.session_state.vector_store = vector_store
            st.session_state.llm = llm
            st.session_state.processed_file = file_name
            st.session_state.document_processed = True
            st.session_state.pages = len(pages)
            st.session_state.chunks = len(chunks)

            # Clear previous chat when a new document is uploaded
            st.session_state.chat_history = []

            # Remove temporary PDF
            os.remove(pdf_path)

        st.success(
            f"Document processed successfully! "
            f"{len(pages)} pages • {len(chunks)} chunks"
        )

    else:

        st.success(
            f"Document ready • "
            f"{st.session_state.pages} pages • "
            f"{st.session_state.chunks} chunks"
        )


# ---------------------------------------
# Question Section
# ---------------------------------------

if st.session_state.get("document_processed", False):

    st.divider()

    st.markdown(
        '<div class="section-label">💬 Ask Your Document</div>',
        unsafe_allow_html=True
    )

    # ---------------------------------------
    # Display Chat History
    # ---------------------------------------

    for chat in st.session_state.chat_history:

        # User question
        st.markdown(
            f"**👤 You:** {chat['question']}"
        )

        # QueryAI answer
        st.markdown(
            '<div class="section-label">🤖 QueryAI</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):
            st.markdown(chat["answer"])

        # ---------------------------------------
        # Sources
        # ---------------------------------------

        if (
            "couldn't find this information"
            not in chat["answer"].lower()
        ):

            st.markdown(
                '<div class="section-label">📚 Sources</div>',
                unsafe_allow_html=True
            )

            shown_pages = set()

            for source in chat["sources"]:

                page = source.metadata.get(
                    "page_number",
                    "Unknown"
                )

                if page not in shown_pages:

                    st.markdown(
                        f"""
                        <div class="source-box">
                            📄 <b>Page {page}</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    shown_pages.add(page)

    # ---------------------------------------
    # New Question
    # ---------------------------------------

    with st.form(
        "question_form",
        clear_on_submit=True
    ):

        question = st.text_input(
            "Ask QueryAI about your document",
            placeholder="Type your question and press Enter...",
            label_visibility="collapsed"
        )

        submitted = st.form_submit_button(
            "Ask QueryAI"
        )

    if submitted:

        if question.strip():

            with st.spinner("Searching the document..."):

                result = answer_question(
                    st.session_state.llm,
                    st.session_state.vector_store,
                    question
                )

            # ---------------------------------------
            # Save Conversation
            # ---------------------------------------

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"]
                }
            )

            # Refresh page
            st.rerun()

        else:

            st.warning("Please enter a question.")