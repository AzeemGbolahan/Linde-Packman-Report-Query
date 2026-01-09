import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

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
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"Missing folder: {DATA_DIR}")

    documents = load_documents(DATA_DIR)

    print(f"Loaded {len(documents)} document pages/sections total.")
    print("First 3 items preview:")
    for d in documents[:3]:
        print("-" * 60)
        print("source:", d.metadata.get("source"))
        print(d.page_content[:400].replace("\n", " "))
