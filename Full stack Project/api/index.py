"""Private local backend. Next.js owns browser sessions and forwards requests here."""
from io import BytesIO
from pathlib import PurePath
from typing import Literal
import hashlib
import os
import uuid

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader
from starlette.concurrency import run_in_threadpool

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"))
from .rag_pipeline import DEFAULT_MODEL, answer_question, check_connection, delete_document, index_pages, list_documents, resolve_key

app = FastAPI(title="NCERT Tutor", version="2.0.0")
MAX_UPLOAD = 20 * 1024 * 1024


def workspace(x_workspace_id: str = Header(), x_workspace_kind: Literal['guest', 'account'] = Header(default='guest')) -> str:
    try:
        identifier = str(uuid.UUID(x_workspace_id))
        # The private proxy supplies kind from a verified session. A guest UUID
        # can never address an account library, even if the UUID is identical.
        return f'account:{identifier}' if x_workspace_kind == 'account' else identifier
    except ValueError:
        raise HTTPException(400, "Invalid workspace session") from None


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

    @field_validator("content")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value


class ConnectionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL, min_length=3, max_length=160, pattern=r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


class ChatRequest(ConnectionRequest):
    messages: list[Message] = Field(min_length=1, max_length=40)
    mode: Literal["explain", "revise", "practice", "flashcards"] = "explain"
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)


@app.get("/api/health")
def health():
    return {"status": "ok", "provider": "NVIDIA NIM", "server_key_configured": bool(os.environ.get("NVIDIA_API_KEY")), "default_model": os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL)}


@app.post("/api/connection")
def connection(body: ConnectionRequest, authorization: str | None = Header(default=None)):
    return check_connection(resolve_key(authorization), body.model)


@app.get("/api/documents")
def documents(session: str = Depends(workspace)):
    return {"documents": list_documents(session)}


@app.delete("/api/documents/{doc_id}")
def remove(doc_id: uuid.UUID, session: str = Depends(workspace)):
    if not delete_document(session, str(doc_id)):
        raise HTTPException(404, "Textbook not found in this library")
    return {"deleted": True}


def process_pdf(data: bytes, name: str, session: str):
    if not data.startswith(b"%PDF-"):
        raise HTTPException(415, "The file is not a valid PDF.")
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise HTTPException(422, "This PDF is password protected. Upload an unlocked copy.")
        if len(reader.pages) > 700:
            raise HTTPException(413, "Upload a PDF with no more than 700 pages.")
        pages = []
        total = 0
        for page in reader.pages:
            # Limit pathological compressed content before extraction.
            contents = page.get_contents()
            if contents and len(contents.get_data()) > 8 * 1024 * 1024:
                raise HTTPException(413, "A page is too complex to process. Upload a smaller PDF.")
            text = page.extract_text() or ""
            total += len(text)
            if total > 8_000_000:
                raise HTTPException(413, "This PDF contains too much text. Upload a chapter instead.")
            pages.append(text)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(422, "This PDF could not be read. Try a different text-based PDF.") from None
    return {"document": index_pages(session, name, pages, hashlib.sha256(data).hexdigest())}


@app.post("/api/upload")
async def upload(request: Request, filename: str, session: str = Depends(workspace)):
    name = PurePath(filename.replace("\\", "/")).name.strip()
    if not name or len(name) > 180 or not name.lower().endswith(".pdf") or any(ord(c) < 32 for c in name):
        raise HTTPException(400, "Use a PDF filename with at most 180 characters.")
    buffer = bytearray()
    async for part in request.stream():
        if len(buffer) + len(part) > MAX_UPLOAD:
            raise HTTPException(413, "PDFs must be 20 MB or smaller.")
        buffer.extend(part)
    if not buffer:
        raise HTTPException(400, "The PDF is empty.")
    return await run_in_threadpool(process_pdf, bytes(buffer), name, session)


@app.post("/api/chat")
def chat(body: ChatRequest, session: str = Depends(workspace), authorization: str | None = Header(default=None)):
    if body.messages[-1].role != "user":
        raise HTTPException(400, "The last message must be a question from the student.")
    if sum(len(m.content) for m in body.messages) > 60000:
        raise HTTPException(413, "This conversation is too long. Start a new chat.")
    ids = [str(i) for i in body.document_ids]
    if ids and not set(ids).issubset({d["id"] for d in list_documents(session)}):
        raise HTTPException(404, "A selected textbook is no longer in your library. Select an available textbook.")
    return answer_question(session, [m.model_dump() for m in body.messages], resolve_key(authorization), body.model, body.mode, ids)
