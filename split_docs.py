import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data/1_A_Azeem_Files"

def load_documents(folder: str):
    docs = []
    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            lower = name.lower()
            try:
                if lower.endswith(".pdf"):
                    docs.extend(PyPDFLoader(path).load())
                elif lower.endswith(".docx"):
                    docs.extend(Docx2txtLoader(path).load())
            except Exception as e:
                print(f"[SKIP] {path} -> {e}")
    return docs

if __name__ == "__main__":
    documents = load_documents(DATA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    MIN_CHARS = 200
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHARS]

    print(f"Original docs (pages/sections): {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    # Quick quality check: show one chunk + its metadata
    sample = chunks[0]
    print("-" * 60)
    print("Sample chunk source:", sample.metadata.get("source"))
    print("Sample chunk length:", len(sample.page_content))
    print(sample.page_content[:600].replace("\n", " "))
