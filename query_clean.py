import re
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "chroma"
COLLECTION_NAME = "linde_packman_funding_reports"

def pretty_source(path: str) -> str:
    return (path or "unknown").replace("data/1_A_Azeem_Files/", "")

def student_from_source(src: str) -> str:
    """
    Try to infer student name from filenames like:
    2025_Summer_Reports/Gbolahan_Azeem_28_letter-summer25 Azeem Gbolahan.docx
    2023_Summer_Reports/Pathak_Devesh_Pathak_24-report.docx
    """
    if not src:
        return "Unknown"
    base = src.split("/")[-1]
    base = re.sub(r"\.(pdf|docx)$", "", base, flags=re.I)

    # common patterns: Last_First_... or Last_FirstLast_...
    m = re.match(r"([A-Za-z]+)_([A-Za-z]+)", base)
    if m:
        last, first = m.group(1), m.group(2)
        return f"{first} {last}"

    return "Unknown"

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

    # grab more results, then dedupe by student
    results = db.similarity_search(query, k=25)

    grouped = defaultdict(list)
    for doc in results:
        src = pretty_source(doc.metadata.get("source", ""))
        student = student_from_source(src)
        snippet = doc.page_content.strip().replace("\n", " ")
        grouped[student].append((src, snippet))

    # print top evidence per student
    print("\nStudents (deduped) with evidence:\n")
    for student, items in sorted(grouped.items(), key=lambda x: (-len(x[1]), x[0])):
        src, snippet = items[0]
        print(f"- {student}")
        print(f"  source: {src}")
        print(f"  evidence: {snippet[:260]}...")
        print()
