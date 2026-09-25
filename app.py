import streamlit as st

from src.config import *
from src.embeddings import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import Generator


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuQA | RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background: #f5f7fb;
    }

    .main {
        padding-top: 1rem;
    }


    /* =========================
       HERO HEADER
       ========================= */

    .hero {
        background: linear-gradient(
            135deg,
            #111827,
            #1e3a8a
        );

        padding: 30px 34px;
        border-radius: 18px;
        margin-bottom: 25px;
        color: white;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.10);
    }

    .hero-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .hero-subtitle {
        font-size: 15px;
        color: #dbeafe;
        margin: 0;
    }


    /* =========================
       SIDEBAR
       ========================= */

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        font-size: 13px;
        color: #9ca3af !important;
        margin-bottom: 25px;
    }

    .document-card {
        background: #1f2937;
        border: 1px solid #374151;
        padding: 12px 14px;
        border-radius: 10px;
        margin-bottom: 9px;
        font-size: 13px;
    }


    /* =========================
       STAT CARDS
       ========================= */

    .stat-card {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 18px;
        border-radius: 14px;
        text-align: center;

        box-shadow:
            0 3px 12px rgba(0, 0, 0, 0.04);
    }

    .stat-number {
        font-size: 26px;
        font-weight: 700;
        color: #1d4ed8;
    }

    .stat-label {
        font-size: 12px;
        color: #6b7280;
        margin-top: 3px;
    }


    /* =========================
       SECTION TITLES
       ========================= */

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .question-hint {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 8px;
    }


    /* =========================
       ANSWER CARD
       ========================= */

    .answer-header {
        background: white;

        border-left: 5px solid #2563eb;
        border-top: 1px solid #e5e7eb;
        border-right: 1px solid #e5e7eb;
        border-bottom: 1px solid #e5e7eb;

        border-radius: 14px 14px 0 0;

        padding: 17px 22px;

        margin-top: 15px;

        box-shadow:
            0 3px 12px rgba(0, 0, 0, 0.04);
    }

    .answer-label {
        font-size: 14px;
        font-weight: 700;
        color: #2563eb;
        letter-spacing: 0.4px;
    }

    .answer-box {
        background: white;

        border-left: 5px solid #2563eb;
        border-right: 1px solid #e5e7eb;
        border-bottom: 1px solid #e5e7eb;

        border-radius: 0 0 14px 14px;

        padding: 20px 25px;

        margin-bottom: 25px;

        box-shadow:
            0 4px 16px rgba(0, 0, 0, 0.05);
    }


    /* =========================
       SOURCE HEADER
       ========================= */

    .source-header {
        background: #eef2ff;
        padding: 10px 14px;
        border-radius: 9px;
        color: #3730a3;
        font-weight: 600;
        font-size: 13px;
    }


    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }


    /* =========================
       TEXT INPUT
       ========================= */

    div[data-testid="stTextInput"] input {
        border-radius: 10px;
        padding: 13px;
        border: 1px solid #d1d5db;
    }


    /* =========================
       FOOTER
       ========================= */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        padding: 30px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHECK GROQ API KEY
# ============================================================

if not GROQ_API_KEY:

    st.error(
        "GROQ_API_KEY is missing. "
        "Add your Groq API key to the .env file."
    )

    st.stop()


# ============================================================
# LOAD VECTOR STORE
# ============================================================

store = VectorStore(
    VECTOR_DIR,
    COLLECTION_NAME
)


if store.count() == 0:

    st.warning(
        "No indexed documents found. "
        "Please run `python ingest.py` first."
    )

    st.stop()


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

@st.cache_resource
def load_components():

    embedder = Embedder(
        EMBEDDING_MODEL
    )

    retriever = Retriever(
        embedder,
        store,
        TOP_K
    )

    generator = Generator(
        GROQ_API_KEY,
        GROQ_MODEL
    )

    return retriever, generator


retriever, generator = load_components()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        '<div class="sidebar-title">📚 DocuQA</div>'
    )

    st.html(
        '<div class="sidebar-subtitle">'
        'Retrieval-Augmented Document Assistant'
        '</div>'
    )

    st.markdown("### 📄 Knowledge Base")

    documents = [
        "artificial_intelligence.pdf",
        "machine_learning.pdf",
        "deep_learning.pdf",
        "generative_ai.pdf",
        "natural_language_processing.pdf"
    ]

    for document in documents:

        st.html(
            f"""
            <div class="document-card">
                📄 {document}
            </div>
            """
        )

    st.markdown("---")

    st.markdown("### ⚙️ System")

    st.write(
        f"**Embedding:** `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**Retrieval:** Top {TOP_K}"
    )

    st.write(
        f"**LLM:** `{GROQ_MODEL}`"
    )

    st.markdown("---")

    st.caption(
        "Documents → Chunks → Embeddings → "
        "Vector Search → LLM"
    )


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-title">
            📚 Document Intelligence Assistant
        </div>

        <p class="hero-subtitle">
            Ask questions about your documents and receive
            grounded answers with source references.
        </p>

    </div>
    """
)


# ============================================================
# STATISTICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.html(
        """
        <div class="stat-card">
            <div class="stat-number">5</div>
            <div class="stat-label">Documents</div>
        </div>
        """
    )


with col2:

    st.html(
        f"""
        <div class="stat-card">
            <div class="stat-number">
                {store.count()}
            </div>

            <div class="stat-label">
                Indexed Chunks
            </div>
        </div>
        """
    )


with col3:

    st.html(
        f"""
        <div class="stat-card">
            <div class="stat-number">
                {TOP_K}
            </div>

            <div class="stat-label">
                Retrieved Results
            </div>
        </div>
        """
    )


with col4:

    st.html(
        """
        <div class="stat-card">
            <div class="stat-number">
                RAG
            </div>

            <div class="stat-label">
                Architecture
            </div>
        </div>
        """
    )


# ============================================================
# QUESTION
# ============================================================

st.html(
    '<div class="section-title">'
    '💬 Ask your documents'
    '</div>'
)

st.html(
    '<div class="question-hint">'
    'Ask a question based on the indexed documents.'
    '</div>'
)


# ============================================================
# QUESTION INPUT + ENTER BUTTON
# ============================================================

with st.form("question_form", clear_on_submit=False):

    question_col, button_col = st.columns([6, 1])

    with question_col:

        question = st.text_input(
            "Question",
            placeholder=(
                "Example: What is the difference between "
                "machine learning and deep learning?"
            ),
            label_visibility="collapsed"
        )

    with button_col:

        submit_question = st.form_submit_button(
            "➤ Enter",
            use_container_width=True
        )


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown("**Try an example:**")

example_col1, example_col2, example_col3 = st.columns(3)


with example_col1:

    example1 = st.button(
        "🧠 What is AI?",
        use_container_width=True
    )


with example_col2:

    example2 = st.button(
        "🤖 What is RAG?",
        use_container_width=True
    )


with example_col3:

    example3 = st.button(
        "📊 What is supervised learning?",
        use_container_width=True
    )


if example1:

    question = "What is artificial intelligence?"

elif example2:

    question = (
        "What is retrieval augmented generation?"
    )

elif example3:

    question = "What is supervised learning?"


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    st.markdown("---")

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "🔎 Searching the knowledge base..."
    ):

        sources = retriever.retrieve(
            question
        )


    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Generating grounded answer..."
    ):

        answer = generator.answer(
            question,
            sources
        )


    # ========================================================
    # QUESTION DISPLAY
    # ========================================================

    st.html(
        '<div class="section-title">'
        '❓ Your Question'
        '</div>'
    )

    st.info(question)


    # ========================================================
    # ANSWER
    # ========================================================

    st.html(
        '<div class="section-title">'
        '✨ Answer'
        '</div>'
    )


    # Header is HTML ONLY.
    # The LLM answer is displayed separately using st.markdown.
    st.html(
        """
        <div class="answer-header">
            <div class="answer-label">
                GROUNDED RESPONSE
            </div>
        </div>
        """
    )


    # IMPORTANT:
    # Do NOT insert the LLM answer inside unsafe HTML.
    # Streamlit renders it safely as Markdown.
    st.markdown(
        answer
    )


    # ========================================================
    # SOURCES
    # ========================================================

    st.html(
        '<div class="section-title">'
        '📑 Retrieved Sources'
        '</div>'
    )

    st.caption(
        f"The answer was generated using "
        f"{len(sources)} retrieved document chunks."
    )


    for i, source in enumerate(
        sources,
        start=1
    ):

        source_name = source["source"]
        page_number = source["page"]
        distance = source["distance"]


        with st.expander(
            f"📄 Source {i}  •  "
            f"{source_name}  •  Page {page_number}"
        ):

            st.html(
                f"""
                <div class="source-header">
                    📄 {source_name}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    Page {page_number}
                </div>
                """
            )

            st.write("")

            st.write(
                source["text"]
            )

            st.caption(
                f"Similarity distance: "
                f"{distance:.4f}"
            )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.html(
        """
        <div style="
            background:white;
            padding:40px;
            border-radius:15px;
            text-align:center;
            margin-top:25px;
            border:1px solid #e5e7eb;
        ">

            <div style="font-size:48px;">
                🔍
            </div>

            <h3>
                Ask a question to get started
            </h3>

            <p style="color:#6b7280;">
                Your question will be matched against
                the indexed document chunks before
                generating an answer.
            </p>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        Built with Streamlit • Sentence Transformers • FAISS • Groq

        <br>

        Retrieval-Augmented Generation (RAG) Document Q&A

    </div>
    """
)