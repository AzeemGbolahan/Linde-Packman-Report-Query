import re
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
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
    return re.findall(r"[a-z0-9]+", text.lower())

@st.cache_resource
def get_db_and_models():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return db, llm

def hybrid_retrieve(db, question: str, k_vector: int = 60, k_take: int = 25):
    candidates = db.similarity_search(question, k=k_vector)
    if not candidates:
        return []

    corpus = [c.page_content for c in candidates]
    bm25 = BM25Okapi([tokenize(t) for t in corpus])
    scores = bm25.get_scores(tokenize(question))

    keyword_ranked = [candidates[i] for i in sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:k_take]]
    vector_ranked = candidates[:k_take]

    merged, seen = [], set()
    for d in keyword_ranked + vector_ranked:
        key = (d.metadata.get("source", ""), d.page_content[:200])
        if key not in seen:
            seen.add(key)
            merged.append(d)
    return merged

def group_evidence(docs):
    grouped = defaultdict(list)
    for d in docs:
        src = pretty_source(d.metadata.get("source", ""))
        student = student_from_source(src)
        grouped[student].append((src, d.page_content.strip()))
    return grouped

def build_evidence_text(grouped, max_students=14, max_snips_per_student=3):
    students_sorted = sorted(grouped.items(), key=lambda x: -len(x[1]))[:max_students]
    out = []
    for student, items in students_sorted:
        out.append(f"STUDENT: {student}")
        for src, txt in items[:max_snips_per_student]:
            out.append(f"- SOURCE: {src}")
            out.append(f"  SNIPPET: {' '.join(txt.split())[:550]}")
        out.append("")
    return "\n".join(out)

def answer_question(llm, question: str, grouped):
    evidence = build_evidence_text(grouped)

    system = (
        "You are answering questions about a set of student reports.\n"
        "Only use the provided snippets; do not guess.\n"
        "Give a concise, plain-English answer.\n"
        "If listing students, write one bullet per student.\n"
        "Every bullet must end with citations in parentheses using the SOURCE filenames.\n"
        "If evidence is weak, label it 'related (details unclear)'.\n"
    )

    user = f"""QUESTION:
{question}

EVIDENCE SNIPPETS:
{evidence}

OUTPUT FORMAT EXAMPLES:
- Student Name — description. (file1; file2)
OR a short paragraph answer with citations where appropriate.
"""
    resp = llm.invoke([{"role": "system", "content": system},
                      {"role": "user", "content": user}])
    return resp.content.strip()

# ---------------- UI ----------------
st.set_page_config(page_title="Linde-Packman Report Q&A", page_icon="📄", layout="wide")
st.title("📄 Linde-Packman Report Q&A")
st.info(
    "Answers are generated from the student reports below. "
    "Each response includes citations so you can verify the source."
)
st.caption("Ask questions about the student reports. Answers cite the report filenames.")

question = st.text_input("Ask a question", placeholder="e.g., Which students worked on machine learning and AI?")
col1, col2 = st.columns([1, 3])
with col1:
    k_vector = 60
with col2:
    show_sources = st.checkbox("Show supporting excerpts from reports", value=True)

ask = st.button("Find answers")

if ask:
    if not question.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Searching reports..."):
            db, llm = get_db_and_models()
            docs = hybrid_retrieve(db, question, k_vector=k_vector, k_take=25)
            if not docs:
                st.error("No matches found. Try a different question.")
            else:
                grouped = group_evidence(docs)
                final = answer_question(llm, question, grouped)

        st.subheader("Answer")
        st.write(final)

        if show_sources:
            st.subheader("Sources")
            # Show grouped evidence in expandable sections
            for student, items in sorted(grouped.items(), key=lambda x: -len(x[1])):
                with st.expander(f"{student} ({len(items)} snippet(s))", expanded=False):
                    for src, txt in items[:4]:
                        st.markdown(f"**{src}**")
                        st.write(" ".join(txt.split())[:900] + "…")
