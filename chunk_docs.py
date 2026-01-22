from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 1. Load markdown files
loader = DirectoryLoader(
    "docs",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# 2. Split by markdown headers
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

print(f"After header split: {len(header_chunks)} chunks")

# 3. Size-based splitting
size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

final_chunks = size_splitter.split_documents(header_chunks)
print(f"Final chunk count: {len(final_chunks)}")

# 4. Inspect first few chunks
for i, chunk in enumerate(final_chunks[:3]):
    print("\n--- Chunk", i + 1, "---")
    print(chunk.page_content[:400])
    print("Metadata:", chunk.metadata)
