import re
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma


CHROMA_DIR = "chroma"
COLLECTION_NAME = "linde_packman_funding_reports"

def pretty_source(path: str) -> str:
    return (path or "unknown").replace("data/1_A_Azeem_Files/", "")

def student_from_source(src: str) -> str:
    if not src:
        return "Unknown"
    base = src.split("/")[-1]
    base = re.sub(r"\.(pdf|docx)$", "", base, flags=re.I)

    m = re.match(r"([A-Za-z]+)_([A-Za-z]+)", base)
    if m:
        last, first = m.group(1), m.group(2)
        return f"{first} {last}"
    return "Unknown"

def build_evidence_block(grouped: dict, max_students: int = 12, max_snips_per_student: int = 3) -> str:
    # Sort students by how many retrieved snippets they have (rough relevance proxy)
    students_sorted = sorted(grouped.items(), key=lambda x: -len(x[1]))[:max_students]

    blocks = []
    for student, items in students_sorted:
        blocks.append(f"STUDENT: {student}")
        for (src, txt) in items[:max_snips_per_student]:
            txt = " ".join(txt.split())
            blocks.append(f"- SOURCE: {src}")
            blocks.append(f"  SNIPPET: {txt[:500]}")
        blocks.append("")  # spacer
    return "\n".join(blocks)

if __name__ == "__main__":
    question = input("Ask a question: ").strip()
    if not question:
        raise SystemExit("No question provided.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Retrieve more than we show, then group by student
    docs = db.similarity_search(question, k=35)

    grouped = defaultdict(list)
    for d in docs:
        src = pretty_source(d.metadata.get("source", ""))
        student = student_from_source(src)
        grouped[student].append((src, d.page_content.strip()))

    evidence = build_evidence_block(grouped)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system = (
        "You are helping summarize a set of student internship reports.\n"
        "Only use the provided snippets; do not guess.\n"
        "Return a concise bulleted list of students relevant to the user's question.\n"
        "For each student: write 1 short line describing the work.\n"
        "End each bullet with citations in parentheses using the provided SOURCE filenames.\n"
        "If a student's evidence is vague, label it 'AI-related (details unclear)'.\n"
    )

    user = f"""QUESTION:
{question}

EVIDENCE SNIPPETS (grouped by student):
{evidence}

OUTPUT FORMAT:
- Student Name — 1 short description. (source1; source2)
"""

    resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    print("\n" + resp.content.strip() + "\n")
