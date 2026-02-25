import os
import requests
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

try:
    # LangChain v1 split classic chains/memory into a separate package.
    from langchain_classic.chains import ConversationalRetrievalChain
    from langchain_classic.memory import ConversationBufferMemory
except ImportError:
    # Backward compatibility for older LangChain layouts.
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from typing import List

class PineconeRESTRetriever(BaseRetriever):
    api_key: str = Field(description="Pinecone API Key")
    index_name: str = Field(description="Pinecone Index Name")
    embeddings: OpenAIEmbeddings = Field(description="Embeddings client")
    k: int = Field(default=4, description="Number of results to fetch")
    host_url: str = Field(default="", description="Pinecone index host")

    def model_post_init(self, __context) -> None:
        if not self.host_url:
            self.host_url = self._get_host()

    def _get_host(self) -> str:
        resp = requests.get(
            f"https://api.pinecone.io/indexes/{self.index_name}",
            headers={"Api-Key": self.api_key}
        )
        if not resp.ok:
            raise ValueError(f"Failed to find index {self.index_name}: {resp.text}")
        return f"https://{resp.json().get('host')}"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        
        resp = requests.post(
            f"{self.host_url}/query",
            headers={
                "Api-Key": self.api_key,
                "Content-Type": "application/json"
            },
            json={
                "vector": query_vector,
                "topK": self.k,
                "includeMetadata": True
            }
        )
        
        if not resp.ok:
            raise ValueError(f"Pinecone query failed: {resp.text}")
            
        results = resp.json()
        docs = []
        for match in results.get("matches", []):
            text = match.get("metadata", {}).get("text", "")
            if text:
                docs.append(Document(page_content=text))
        return docs

def get_rag_chain():
    """
    Builds the Conversational RAG QA chain for Vercel using OpenAI and Pinecone.
    """
    # 1. Environment Variables (Must be configured in Vercel)
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "ncert-rag")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key or not pinecone_api_key:
        raise ValueError("Missing required API keys: OPENAI_API_KEY or PINECONE_API_KEY")

    # 2. Define the embeddings and Vector Store connected to Pinecone
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 3. Create REST Custom Retriever
    retriever = PineconeRESTRetriever(
        api_key=pinecone_api_key,
        index_name=pinecone_index_name,
        embeddings=embeddings,
        k=4
    )

    # 4. Define the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    # 5. Define Prompt 
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

    # 6. Setup Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 7. Build the Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    return qa_chain
