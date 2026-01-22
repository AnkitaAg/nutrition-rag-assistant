from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ---------------- CONFIG ----------------
EMBEDDING_MODEL = "text-embedding-3-small"
FAISS_PATH = "faiss_index"
# ----------------------------------------

# 1. Load embeddings + FAISS index
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vectorstore = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ FAISS index loaded")
#2. Create retriever with filter
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4,
        "filter": {"condition": "diabetes"}
    }
)


# 3. Test queries
queries = [
    "What can I eat for dinner if I have diabetes?",
    "Give me a low salt breakfast idea",
    "Suggest recipes for kidney disease"  # should NOT retrieve much
]

for q in queries:
    print("\n==============================")
    print("QUERY:", q)

    docs = retriever.get_relevant_documents(q)

    print(f"Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:300])
        print("Metadata:", doc.metadata)
