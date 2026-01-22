import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ---------------- CONFIG ----------------
DOCS_PATH = "docs"
FAISS_PATH = "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
# ---------------------------------------


def build_faiss_index():
    """
    Builds a FAISS vector index from markdown documents in DOCS_PATH.
    This function is safe to call in local and cloud environments.
    """

    load_dotenv()

    print("🔹 Starting FAISS index build")

    # ---------- Load markdown files ----------
    md_files = list(Path(DOCS_PATH).rglob("*.md"))

    if not md_files:
        raise RuntimeError("No markdown files found in docs/")

    print(f"📄 Found {len(md_files)} markdown files")

    documents = []
    for file_path in md_files:
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents.extend(loader.load())

    print(f"📚 Loaded {len(documents)} raw documents")

    # ---------- Split by markdown headers ----------
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("##", "section"),
            ("###", "subsection"),
        ]
    )

    header_chunks = []
    for doc in documents:
        header_chunks.extend(header_splitter.split_text(doc.page_content))

    print(f"✂️ Header-level chunks: {len(header_chunks)}")

    # ---------- Further chunking ----------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    final_chunks = text_splitter.split_documents(header_chunks)

    print(f"🧩 Final chunk count: {len(final_chunks)}")

    # ---------- Add metadata from folder structure ----------
    for doc in final_chunks:
        source_path = Path(doc.metadata.get("source", ""))

        parts = [p.lower() for p in source_path.parts]

        # condition
        if "conditions" in parts:
            idx = parts.index("conditions")
            if idx + 1 < len(parts):
                doc.metadata["condition"] = parts[idx + 1]

        # content type
        if "recipes" in parts:
            doc.metadata["content_type"] = "recipe"
        elif "guidance" in parts:
            doc.metadata["content_type"] = "guidance"

        # safety scope
        doc.metadata["medical_scope"] = "non-diagnostic"

    # ---------- Build FAISS ----------
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(final_chunks, embeddings)

    # Ensure directory exists
    os.makedirs(FAISS_PATH, exist_ok=True)

    vectorstore.save_local(FAISS_PATH)

    print("✅ FAISS index built and saved successfully")


# ---------- CLI entrypoint ----------
if __name__ == "__main__":
    build_faiss_index()
