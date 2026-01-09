from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "chroma"
COLLECTION_NAME = "linde_packman_funding_reports"

def pretty_source(path: str) -> str:
    if not path:
        return "unknown"
    return path.replace("data/1_A_Azeem_Files/", "")

if __name__ == "__main__":
    query = input("Ask a question: ").strip()
    if not query:
        raise SystemExit("No query provided.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    results = db.similarity_search_with_score(query, k=10)

    print("\nTop matches:\n")
    for i, (doc, score) in enumerate(results, start=1):
        src = pretty_source(doc.metadata.get("source", ""))
        print(f"{i}) score={score:.4f} | source={src}")
        print(doc.page_content[:700].replace("\n", " "))
        print("-" * 80)
