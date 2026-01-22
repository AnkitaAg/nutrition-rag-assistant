from dotenv import load_dotenv
load_dotenv()

import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

from faiss_store import build_faiss_index

# ---------------- CONFIG ----------------
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
FAISS_PATH = "faiss_index"
SUPPORTED_CONDITIONS = {"diabetes", "hypertension"}
# ---------------------------------------


# ---------- Condition Detection ----------
def detect_condition(query: str):
    q = query.lower()

    if "diabetes" in q:
        return "diabetes"
    if "blood pressure" in q or "hypertension" in q or "low salt" in q:
        return "hypertension"
    if "kidney" in q:
        return "kidney disease"

    return None


# ---------- Load Vector Store (ONCE) ----------
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

import os
from faiss_store import build_faiss_index  # we’ll add this

if not os.path.exists(FAISS_PATH):
    print("FAISS index not found. Building index...")
    build_faiss_index()

vectorstore = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)



# ---------- Answer Generation ----------
def generate_answer(query: str, docs: List[Document]) -> str:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )

    prompt = f"""
You are a helpful nutrition assistant.
You provide GENERAL food and recipe guidance only.
You do NOT give medical advice, diagnosis, or treatment.

Use ONLY the information in the sources below.
If the information is insufficient, say so clearly.

Question:
{query}

Sources:
{context}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content


# ---------- PUBLIC FUNCTION (USED BY STREAMLIT) ----------
def answer_query(user_query: str):
    condition = detect_condition(user_query)

    # ---- GATING ----
    if condition not in SUPPORTED_CONDITIONS:
        return {
            "answer": (
                "I don’t currently have recipe information for that condition. "
                "I can help with diabetes or high blood pressure–friendly recipes."
            ),
            "sources": []
        }

    # ---- CONDITION-CONSTRAINED RETRIEVAL ----
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"condition": condition}
        }
    )

    docs = retriever.invoke(user_query)

    if not docs:
        return {
            "answer": "I couldn’t find relevant information in my sources.",
            "sources": []
        }

    answer = generate_answer(user_query, docs)

    sources = list(
        {doc.metadata.get("source", "unknown") for doc in docs}
    )

    return {
        "answer": answer,
        "sources": sources
    }


# ---------- OPTIONAL: CLI CHAT (Nice to Keep) ----------
def chat():
    print("🥗 Nutrition RAG Assistant")
    print("Supports: diabetes, high blood pressure")
    print("Type 'exit' to quit\n")

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() == "exit":
            break

        result = answer_query(user_query)

        print("\nAssistant:")
        print(result["answer"])

        if result["sources"]:
            print("\nSources:")
            for src in result["sources"]:
                print("-", src)

        print(
            "\nDisclaimer: This assistant provides general dietary guidance only.\n"
        )


if __name__ == "__main__":
    chat()
