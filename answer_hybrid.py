import re
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from rank_bm25 import BM25Okapi
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

def tokenize(text: str):
    # simple, robust tokenization for BM25
    return re.findall(r"[a-z0-9]+", text.lower())

def build_evidence(grouped, max_students=12, max_snips_per_student=3):
    students_sorted = sorted(grouped.items(), key=lambda x: -len(x[1]))[:max_students]
    out = []
    for student, items in students_sorted:
        out.append(f"STUDENT: {student}")
        for src, txt in items[:max_snips_per_student]:
            out.append(f"- SOURCE: {src}")
            out.append(f"  SNIPPET: {' '.join(txt.split())[:500]}")
        out.append("")
    return "\n".join(out)

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

    # 1) Vector retrieval: get a broad candidate set
    candidates = db.similarity_search(question, k=60)

    # 2) Keyword re-ranking over those candidates (BM25)
    corpus_texts = [c.page_content for c in candidates]
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    q_tokens = tokenize(question)
    bm25_scores = bm25.get_scores(q_tokens)

    # Take top keyword matches and merge with top vector matches
    keyword_ranked = [candidates[i] for i in sorted(range(len(candidates)), key=lambda i: bm25_scores[i], reverse=True)[:25]]
    vector_ranked = candidates[:25]

    merged = []
    seen = set()
    for d in keyword_ranked + vector_ranked:
        key = (d.metadata.get("source", ""), d.page_content[:200])
        if key not in seen:
            seen.add(key)
            merged.append(d)

    # Group evidence by student
    grouped = defaultdict(list)
    for d in merged:
        src = pretty_source(d.metadata.get("source", ""))
        student = student_from_source(src)
        grouped[student].append((src, d.page_content.strip()))

    evidence = build_evidence(grouped)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system = (
        "You are summarizing student reports.\n"
        "Only use the provided snippets; do not guess.\n"
        "Return a concise bulleted list answering the user's question.\n"
        "Each bullet: Student Name — 1 short description. End with citations in parentheses.\n"
        "If evidence is weak/vague, label it as 'related (details unclear)'.\n"
    )

    user = f"""QUESTION:
{question}

EVIDENCE SNIPPETS:
{evidence}

OUTPUT FORMAT:
- Student Name — 1 short description. (source1; source2)
"""

    resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    print("\n" + resp.content.strip() + "\n")
