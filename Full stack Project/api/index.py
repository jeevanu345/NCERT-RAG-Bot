from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import requests
import tempfile
import os
import uuid
import requests
from pypdf import PdfReader
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os
load_dotenv(".env.local")

# Assuming we'll have a rag_pipeline here 
from .rag_pipeline import get_rag_chain
from .chunking import chunk_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

class ProcessPDFRequest(BaseModel):
    file_url: str
    filename: str

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "NCERT AI Tutor API is running."}

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages array cannot be empty")
        
        # User's latest message
        user_input = request.messages[-1].get("content", "")
        
        # We can format the history if needed
        # chat_history = format_history(request.messages[:-1])
        
        # Get our chain
        qa_chain = get_rag_chain()
        
        # Generate response
        response = qa_chain.invoke({
            "question": user_input,
            "chat_history": [(msg.get("role"), msg.get("content")) for msg in request.messages[:-1]]
        })
        
        return {
            "answer": response["answer"],
            "sources": [] # In the future we can attach sources here
        }
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process_pdf")
def process_pdf(request: ProcessPDFRequest):
    try:
        # 1. Download PDF from Vercel Blob
        response = requests.get(request.file_url)
        response.raise_for_status()
        
        # 2. Extract text using pypdf
        text = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(response.content)
            temp_pdf_path = temp_pdf.name
            
        try:
            reader = PdfReader(temp_pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        finally:
            os.remove(temp_pdf_path)
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in PDF")
            
        # 3. Chunk text
        chunks = chunk_text(text)
        
        # 4. Push to Pinecone via REST API
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "ncert-rag")
        
        # Get host for index
        host_resp = requests.get(
            f"https://api.pinecone.io/indexes/{pinecone_index_name}",
            headers={"Api-Key": pinecone_api_key}
        )
        if not host_resp.ok:
            return {"status": "error", "message": f"Failed to find index {pinecone_index_name}: {host_resp.text}"}
        
        host_url = f"https://{host_resp.json().get('host')}"
        
        # Embed
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        embedded_vectors = embeddings.embed_documents(chunks)
        
        vectors_to_upsert = []
        for i, text_chunk in enumerate(chunks):
            # Pinecone requires (id, values, metadata)
            vector_id = str(uuid.uuid4())
            vectors_to_upsert.append({
                "id": vector_id, 
                "values": embedded_vectors[i], 
                "metadata": {"text": text_chunk}
            })
            
        # Batch insert to pinecone via REST
        upsert_resp = requests.post(
            f"{host_url}/vectors/upsert",
            headers={
                "Api-Key": pinecone_api_key,
                "Content-Type": "application/json"
            },
            json={"vectors": vectors_to_upsert}
        )
        
        if not upsert_resp.ok:
            return {"status": "error", "message": f"Pinecone upsert failed: {upsert_resp.text}"}
        
        return {"status": "success", "message": f"Successfully vectorized {request.filename}"}
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

