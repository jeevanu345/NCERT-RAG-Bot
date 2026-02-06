import html
from pathlib import Path

import streamlit as st

from rag_pipeline import build_qa_chain

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_SUBDIR = PROJECT_ROOT / "pdfs"
UPLOADED_PDF_SUBDIR = PROJECT_ROOT / "uploaded_pdfs"
TEXT_SOURCE_FILE = PROJECT_ROOT / "ncert_text.txt"


def _inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #212121;
            --panel: #171717;
            --surface: #242424;
            --surface-soft: #2b2b2b;
            --line: #363636;
            --text: #ececec;
            --text-soft: #a8a8a8;
            --accent: #10a37f;
            --radius: 14px;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
            font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", sans-serif;
        }

        #MainMenu, footer, [data-testid="stHeader"] {
            display: none;
        }

        .block-container {
            max-width: 980px;
            padding-top: 0.45rem;
            padding-bottom: 6rem;
        }

        [data-testid="stSidebar"] {
            background: var(--panel);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        .sidebar-brand {
            font-size: 1.05rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            margin-bottom: 0.75rem;
        }

        .sidebar-label {
            margin-top: 0.75rem;
            margin-bottom: 0.35rem;
            font-size: 0.82rem;
            color: var(--text-soft);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .app-topbar {
            max-width: 920px;
            margin: 0 auto 0.75rem auto;
            padding: 0.75rem 0.25rem 0.25rem 0.25rem;
        }

        .app-title {
            font-size: 1.2rem;
            font-weight: 650;
            letter-spacing: 0.01em;
            color: var(--text);
        }

        .app-subtitle {
            margin-top: 0.25rem;
            font-size: 0.88rem;
            color: var(--text-soft);
        }

        [data-testid="stChatMessage"] {
            max-width: 920px;
            margin: 0 auto 0.8rem auto;
            padding-left: 0.25rem;
            padding-right: 0.25rem;
        }

        [data-testid="stChatMessageContent"] {
            border-radius: var(--radius);
            padding: 0.9rem 1rem 0.85rem 1rem;
            border: 1px solid var(--line);
            background: var(--surface);
            line-height: 1.58;
            word-break: normal;
            overflow-wrap: anywhere;
            min-width: min(100%, 240px);
            max-width: 100%;
        }

        [data-testid="chatAvatarIcon-assistant"] {
            background: var(--accent);
            color: #ffffff;
        }

        [data-testid="chatAvatarIcon-user"] {
            background: #3b3b3b;
            color: #ffffff;
        }

        [data-testid="stChatInput"] {
            max-width: 920px;
            margin: 0.3rem auto 0.4rem auto;
            position: sticky;
            bottom: 0;
            padding-top: 0.65rem;
            background: linear-gradient(180deg, rgba(33,33,33,0) 0%, rgba(33,33,33,1) 28%);
            z-index: 50;
        }

        [data-testid="stChatInput"] textarea {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 1.15rem;
            color: var(--text);
            font-size: 0.96rem;
            padding: 0.75rem 0.95rem;
        }

        [data-testid="stChatInput"] textarea:focus {
            border-color: #4a4a4a;
            box-shadow: none;
        }

        [data-testid="stButton"] button {
            border-radius: 0.8rem;
            border: 1px solid var(--line);
            background: #252525;
            color: var(--text);
        }

        [data-testid="stButton"] button:hover {
            border-color: #4b4b4b;
            background: #2d2d2d;
        }

        .starter-text {
            max-width: 920px;
            margin: 0.4rem auto 0.35rem auto;
            padding: 0 0.25rem;
            color: var(--text-soft);
            font-size: 0.9rem;
        }

        .starter-wrap {
            max-width: 920px;
            margin: 0 auto 0.9rem auto;
            padding: 0 0.25rem;
        }

        .source-note {
            max-width: 920px;
            margin: 0.2rem auto 0.65rem auto;
            padding: 0 0.25rem;
            color: var(--text-soft);
            font-size: 0.86rem;
        }

        @media (max-width: 768px) {
            .app-topbar {
                padding-top: 0.25rem;
            }
            [data-testid="stChatInput"] {
                margin-bottom: 0.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _list_pdf_files():
    pdf_paths = []
    for directory in (PROJECT_ROOT, PDF_SUBDIR, UPLOADED_PDF_SUBDIR):
        if not directory.exists():
            continue
        pdf_paths.extend(directory.glob("*.pdf"))
        pdf_paths.extend(directory.glob("*.PDF"))
    return sorted({path.resolve() for path in pdf_paths}, key=lambda p: str(p).lower())


def _path_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _source_signature(pdf_path: str | None) -> str:
    if pdf_path:
        path = Path(pdf_path)
        if path.exists():
            stat = path.stat()
            return f"pdf:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        return f"pdf-missing:{pdf_path}"

    if TEXT_SOURCE_FILE.exists():
        stat = TEXT_SOURCE_FILE.stat()
        return f"text:{TEXT_SOURCE_FILE.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return "text:missing"


st.set_page_config(page_title="NCERT AI Tutor", layout="wide", initial_sidebar_state="expanded")
_inject_styles()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown('<div class="sidebar-brand">NCERT Tutor</div>', unsafe_allow_html=True)
    if st.button("New chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown('<div class="sidebar-label">Textbook Source</div>', unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader(
        "Upload NCERT textbook PDF",
        type=["pdf"],
        accept_multiple_files=False,
    )

if uploaded_pdf is not None:
    UPLOADED_PDF_SUBDIR.mkdir(parents=True, exist_ok=True)
    target_path = UPLOADED_PDF_SUBDIR / Path(uploaded_pdf.name).name
    upload_key = f"{target_path}:{uploaded_pdf.size}"
    if st.session_state.get("last_upload_key") != upload_key or not target_path.exists():
        target_path.write_bytes(uploaded_pdf.getbuffer())
        st.session_state["last_upload_key"] = upload_key
        st.session_state["preferred_pdf_label"] = _path_label(target_path)
        st.rerun()

pdf_files = _list_pdf_files()
selected_pdf_path = None
selected_source_label = "ncert_text.txt"
selected_source_key = _source_signature(None)

if pdf_files:
    path_by_label = {_path_label(path): path for path in pdf_files}
    pdf_options = list(path_by_label.keys())
    default_label = st.session_state.get("preferred_pdf_label")
    if default_label not in path_by_label:
        default_label = pdf_options[0]

    with st.sidebar:
        selected_pdf_label = st.selectbox(
            "Select NCERT textbook PDF",
            pdf_options,
            index=pdf_options.index(default_label),
        )
        st.caption(f"Current source: {selected_pdf_label}")

    st.session_state["preferred_pdf_label"] = selected_pdf_label
    selected_pdf_path = str(path_by_label[selected_pdf_label])
    selected_source_label = selected_pdf_label
    selected_source_key = _source_signature(selected_pdf_path)
else:
    if not TEXT_SOURCE_FILE.exists():
        st.error("No source found. Upload a PDF or create ncert_text.txt.")
        st.stop()
    with st.sidebar:
        st.warning("No PDF files found. Using ncert_text.txt.")


@st.cache_resource(show_spinner="Preparing the chat workspace...")
def get_chain(pdf_path: str | None, source_signature: str):
    _ = source_signature
    return build_qa_chain(pdf_path=pdf_path)


if st.session_state.get("selected_source") != selected_source_key:
    st.session_state["selected_source"] = selected_source_key
    st.session_state["messages"] = []

try:
    qa_chain = get_chain(selected_pdf_path, selected_source_key)
except Exception as error:
    st.error("Could not initialize the QA chain.")
    st.info("Check that Ollama is running and required models are installed: phi3-fast and nomic-embed-text.")
    st.code(str(error))
    st.stop()

safe_source_label = html.escape(selected_source_label)
st.markdown(
    f"""
    <div class="app-topbar">
      <div class="app-title">NCERT AI Tutor</div>
      <div class="app-subtitle">Focused chat for class 10 NCERT textbooks.</div>
    </div>
    <div class="source-note">Current source: {safe_source_label}</div>
    """,
    unsafe_allow_html=True,
)

starter_prompts = [
    "Summarize this chapter in 6 bullet points.",
    "Explain this topic in simple class-10 language.",
    "Create 5 exam-style questions from this chapter.",
    "What are the important formulas from this section?",
]

if not st.session_state["messages"]:
    st.markdown('<div class="starter-text">Start with one prompt:</div>', unsafe_allow_html=True)
    for index, prompt in enumerate(starter_prompts):
        if st.button(prompt, key=f"starter_{index}", use_container_width=True):
            st.session_state["pending_prompt"] = prompt
            st.rerun()

for msg in st.session_state.messages:
    role = "user" if msg["role"] == "student" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["text"])

user_input = st.chat_input("Message NCERT Tutor")
pending_prompt = st.session_state.pop("pending_prompt", None)
if not user_input and pending_prompt:
    user_input = pending_prompt

if user_input:
    st.session_state.messages.append({"role": "student", "text": user_input})
    try:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"question": user_input})
            answer = response["answer"]
    except Exception as error:
        answer = f"Response failed: {error}"
    st.session_state.messages.append({"role": "tutor", "text": answer})
    st.rerun()
