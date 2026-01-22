import os
from pathlib import Path


DOCS_PATH = "docs"
FAISS_PATH = "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"


def build_faiss_index():
    """
    Build FAISS index from markdown documents.
    Safe to call in Streamlit Cloud.
    """

    # ⬇️ IMPORTS MOVED INSIDE FUNCTION (CRITICAL FIX)
    from dotenv import load_dotenv
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    load_dotenv()

    print("🔹 Building FAISS index...")

    md_files = list(Path(DOCS_PATH).rglob("*.md"))
    if not md_files:
        raise RuntimeError("No markdown files found in docs/")

    documents = []
    for file_path in md_files:
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents.extend(loader.load())

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "section"), ("###", "subsection")]
    )

    header_chunks = []
    for doc in documents:
        header_chunks.extend(header_splitter.split_text(doc.page_content))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    final_chunks = splitter.split_documents(header_chunks)

    # -------- Metadata from folder structure --------
    for doc in final_chunks:
        source = Path(doc.metadata.get("source", ""))
        parts = [p.lower() for p in source.parts]

        if "conditions" in parts:
            i = parts.index("conditions")
            if i + 1 < len(parts):
                doc.metadata["condition"] = parts[i + 1]

        if "recipes" in parts:
            doc.metadata["content_type"] = "recipe"
        elif "guidance" in parts:
            doc.metadata["content_type"] = "guidance"

        doc.metadata["medical_scope"] = "non-diagnostic"

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(final_chunks, embeddings)

    os.makedirs(FAISS_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_PATH)

    print("✅ FAISS index built successfully")
