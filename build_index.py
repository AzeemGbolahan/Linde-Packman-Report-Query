import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data/1_A_Azeem_Files"
CHROMA_DIR = "chroma"
COLLECTION_NAME = "linde_packman_funding_reports"

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

def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Filter out tiny chunks (greetings, headers, stray lines)
    MIN_CHARS = 200
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHARS]
    return chunks

if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY in .env")

    docs = load_documents(DATA_DIR)
    chunks = split_docs(docs)

    # #to be changed later when i have higher embedding quota limit 
    # MAX_CHUNKS = 100 #testing chunks
    # chunks = chunks[:MAX_CHUNKS]

    print(f"Docs: {len(docs)} | Chunks: {len(chunks)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(f"✅ Saved Chroma DB to: {CHROMA_DIR} (collection: {COLLECTION_NAME})")
