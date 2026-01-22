import streamlit as st
from chatbot import answer_query

st.set_page_config(
    page_title="Nutrition Recipe Assistant",
    page_icon="🥗",
    layout="centered"
)

st.title("🥗 Nutrition Recipe Assistant")
st.caption(
    "Supports diabetes and high blood pressure. "
    "Provides general, non-diagnostic dietary guidance."
)

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask a recipe or nutrition question...")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get answer from RAG backend
    result = answer_query(user_input)

    assistant_reply = result["answer"]

    if result["sources"]:
        assistant_reply += "\n\n**Sources:**\n"
        for src in result["sources"]:
            assistant_reply += f"- {src}\n"

    assistant_reply += (
        "\n\n_Disclaimer: This assistant provides general dietary guidance only. "
        "Consult a healthcare professional for medical advice._"
    )

    # Show assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
