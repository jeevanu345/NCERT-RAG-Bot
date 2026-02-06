from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import shutil
import hashlib
import re
from datetime import datetime
from pathlib import Path
from pypdf import PdfReader
from chunking import create_chunks_from_text, chunk_text

BASE_VECTOR_DB_DIR = Path("./ncert_db")
DEFAULT_TEXT_DB_DIR = BASE_VECTOR_DB_DIR / "default_text"
DEFAULT_TEXT_COLLECTION = "ncert_default_text"


def _normalize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return token or "document"


def _load_chunks_from_text_file(text_file: str):
    chunks = create_chunks_from_text(text_file)
    if not chunks:
        raise ValueError("No text chunks were created from ncert_text.txt.")
    return chunks


def _extract_text_from_pdf(pdf_path: Path) -> str:
    full_text = []
    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            full_text.append(page_text)
    return "\n".join(full_text)


def _load_chunks_from_pdf(pdf_path: Path):
    text = _extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise ValueError(f"No text could be extracted from: {pdf_path}")
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"No text chunks were created from: {pdf_path}")
    return chunks


def _pdf_vectorstore_location(pdf_path: Path) -> tuple[Path, str]:
    stat = pdf_path.stat()
    source = f"{pdf_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    slug = _normalize_token(pdf_path.stem)

    db_dir = BASE_VECTOR_DB_DIR / f"{slug}_{digest}"
    base_name = f"ncert_{slug}"[:45]
    collection_name = f"{base_name}_{digest}"
    return db_dir, collection_name


def _is_chroma_compatibility_error(error: Exception) -> bool:
    message = str(error)
    return "PersistentData" in message and "max_seq_id" in message


def _backup_incompatible_db(persist_directory: Path) -> Path | None:
    if not persist_directory.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = persist_directory.parent / f"{persist_directory.name}_backup_{timestamp}"
    shutil.move(str(persist_directory), str(backup_path))
    return backup_path


def _build_or_load_vectorstore(
    chunks,
    embedding_model: OllamaEmbeddings,
    persist_directory: Path,
    collection_name: str,
) -> Chroma:
    persist_directory.parent.mkdir(parents=True, exist_ok=True)
    persist_directory_str = str(persist_directory)

    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory_str,
            embedding_function=embedding_model,
        )
        if vectorstore._collection.count() == 0:
            vectorstore.add_texts(chunks)
        return vectorstore
    except Exception as error:
        if not _is_chroma_compatibility_error(error):
            raise

        backup_path = _backup_incompatible_db(persist_directory)
        if backup_path:
            print(f"Incompatible Chroma DB detected. Backup created at: {backup_path}")

        return Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory_str,
        )


def _prepare_source(pdf_path: str | None):
    if pdf_path:
        selected_pdf = Path(pdf_path).expanduser()
        if not selected_pdf.exists():
            raise FileNotFoundError(f"Selected PDF was not found: {selected_pdf}")
        chunks = _load_chunks_from_pdf(selected_pdf)
        persist_directory, collection_name = _pdf_vectorstore_location(selected_pdf)
        return chunks, persist_directory, collection_name

    text_file = "ncert_text.txt"
    if not os.path.exists(text_file):
        raise FileNotFoundError("Please run data_ingestion.py first to create ncert_text.txt.")

    chunks = _load_chunks_from_text_file(text_file)
    return chunks, DEFAULT_TEXT_DB_DIR, DEFAULT_TEXT_COLLECTION


def build_qa_chain(pdf_path: str | None = None):
    """
    Builds and returns the Conversational RAG QA chain.
    Safe to import into Streamlit or other apps.
    """
    # 1. Load and chunk the selected source
    chunks, persist_directory, collection_name = _prepare_source(pdf_path)

    # 2. Define the embedding model and vector store
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = _build_or_load_vectorstore(chunks, embedding_model, persist_directory, collection_name)

    # 3. Create a retriever to search the vector store
    retriever = vectorstore.as_retriever()

    # 4. Define the LLM and the RAG prompt template
    llm = ChatOllama(model="phi3-fast")

    prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful and personalized AI tutor for a Class 10 student.

    Rules for answering:
    - Always answer in **English only**.
    - Do NOT add phrases like "according to the study material", "the text says", or "from the PDF".
    - Use simple, clear, step-by-step explanations.
    - If the student says "I didn't understand", simplify with everyday examples.
    - If the student says "oh got it", "good job", "thanks", "oh nice", or any similar phrase, reply briefly with encouragement (e.g., "Great! Happy you understood.").
    - If the student gives feedback but NOT a question, DO NOT explain again — just acknowledge warmly.
    - Only give detailed explanations if the student actually asks a question.
    - Be concise and avoid repeating long explanations unless specifically asked.
    - Focus on the student’s latest input, while remembering the conversation history.
    - Base your explanation strictly on the "Study material context" below.
    - Do NOT add outside knowledge unless the student explicitly asks for it.

    Conversation so far:
    {chat_history}

    Study material context:
    {context}

    Student's latest message: {question}

    AI Tutor (in English):
    """)

    # 5. Setup memory to store chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 6. Build the Conversational RAG chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    return qa_chain


# 7. If running directly, start the terminal chatbot
if __name__ == "__main__":
    qa_chain = build_qa_chain()

    print("AI Tutor is ready. Ask a question about your Class 10 NCERT book.")
    print("Type 'exit' to quit.")

    while True:
        user_question = input("\nStudent: ")
        if user_question.lower() == 'exit':
            print("Goodbye!")
            break

        response = qa_chain.invoke({"question": user_question})
        print(f"AI Tutor: {response['answer']}")
