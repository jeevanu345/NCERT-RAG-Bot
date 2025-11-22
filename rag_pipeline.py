from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from chunking import create_chunks_from_text  # Import your chunking function


def build_qa_chain():
    """
    Builds and returns the Conversational RAG QA chain.
    Safe to import into Streamlit or other apps.
    """
    # 1. Load and chunk the documents
    text_file = "ncert_text.txt"
    if not os.path.exists(text_file):
        raise FileNotFoundError("Please run data_ingestion.py first to create ncert_text.txt.")

    chunks = create_chunks_from_text(text_file)

    # 2. Define the embedding model and vector store
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_texts(chunks, embedding_model, persist_directory="./ncert_db")

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
    - If the student says "oh got it", "good job", "thanks", "oh nice", or any similar phrase, reply briefly with encouragement (e.g., "Great! Happy you understood 🎉").
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
