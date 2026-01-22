from dotenv import load_dotenv
load_dotenv()

import os
print("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ------------- CONFIG ----------------
DOCS_PATH = "docs"
EMBEDDING_MODEL = "text-embedding-3-small"
# -------------------------------------

# 1. Load markdown files
loader = DirectoryLoader(
    DOCS_PATH,
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# 2. Header-based chunking
headers_to_split_on = [
    ("##", "section"),
    ("###", "subsection"),
]

md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

header_chunks = []
for doc in documents:
    splits = md_splitter.split_text(doc.page_content)
    for split in splits:
        split.metadata.update(doc.metadata)
        header_chunks.append(split)

# 3. Size-based chunking
size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

final_chunks = size_splitter.split_documents(header_chunks)
print(f"Final chunk count: {len(final_chunks)}")

from pathlib import Path

for doc in final_chunks:
    source_path = Path(doc.metadata.get("source", ""))

    # Normalize path parts
    parts = [p.lower() for p in source_path.parts]

    # Extract condition from folder name
    if "conditions" in parts:
        idx = parts.index("conditions")
        if idx + 1 < len(parts):
            doc.metadata["condition"] = parts[idx + 1]

    # Extract content type
    if "recipes" in parts:
        doc.metadata["content_type"] = "recipe"
    elif "guidance" in parts:
        doc.metadata["content_type"] = "guidance"

    # Enforce safety scope
    doc.metadata["medical_scope"] = "non-diagnostic"



# 4. Create embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# 5. Build FAISS vector store
vectorstore = FAISS.from_documents(final_chunks, embeddings)

# 6. Save locally
vectorstore.save_local("faiss_index")

print("✅ FAISS index created and saved")
