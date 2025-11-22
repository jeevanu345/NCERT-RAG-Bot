import streamlit as st
from rag_pipeline import build_qa_chain

# 🚨 Page config must be first Streamlit command
st.set_page_config(page_title="AI Tutor - Class 10", page_icon="📘")

# Cache the QA chain so it loads only once
@st.cache_resource
def get_chain():
    return build_qa_chain()

qa_chain = get_chain()

# Page layout
st.title("📘 AI Tutor (NCERT Class 10)")
st.markdown("Ask me anything from your NCERT study material!")

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "student" else "assistant"):
        st.markdown(msg["text"])

# Chat input
user_input = st.chat_input("Ask a question...")

if user_input:
    # Store & display student message
    st.session_state.messages.append({"role": "student", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI Tutor response
    response = qa_chain.invoke({"question": user_input})
    answer = response["answer"]

    # Store & display tutor response
    st.session_state.messages.append({"role": "tutor", "text": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
