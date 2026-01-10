# Linde-Packman Student Report Q&A System

A question–answer system for exploring, summarizing, and analyzing Linde-Packman–funded student reports using natural language queries.

This project allows a non-technical user (e.g. a faculty advisor or program administrator) to ask high-level questions about many student reports at once and receive clear answers **with citations back to the original report files**.

---

## Purpose

Linde-Packman student reports span many years, projects, and disciplines. Reading them individually makes it difficult to identify patterns such as:

- Which students worked on machine learning or AI?
- Which projects involved cancer, genomics, or clinical research?
- Which students worked outside the United States?
- What kinds of research areas are most common across years?

This system enables fast, reliable exploration of those reports while preserving traceability to the original documents.

---

## What the system does

- Loads student reports stored locally as PDF and DOCX files  
- Splits long documents into searchable text chunks  
- Builds a local vector database (Chroma) for semantic search  
- Uses hybrid retrieval (semantic + keyword matching)  
- Generates concise answers using an LLM  
- Cites the source filenames for every claim  
- Provides a simple Streamlit web interface for non-technical users  

---

## Key design principles

- **Privacy-first**: reports never leave the local machine
- **Traceability**: every answer is backed by cited report files
- **Non-technical usability**: one text box, one button
- **Modularity**: indexing, querying, and UI are cleanly separated

---

## Who this is for

- Faculty advisors
- Research program directors
- Grant administrators
- Academic staff reviewing student outcomes

No programming experience is required to use the interface once it is set up.

---

## Data and privacy

This repository contains **code only**.

- Student reports are stored locally and ignored by Git
- The `data/` directory is excluded via `.gitignore`
- The vector database (`chroma/`) is also excluded
- API keys are stored in a local `.env` file and never committed

This ensures student data remains private and secure.

---

## Project structure
```bash
.
├── app.py                     # Streamlit web interface
├── load_docs.py               # Load PDF/DOCX files from data/
├── split_docs.py              # Split documents into chunks
├── build_index.py             # Build and persist Chroma vector DB
├── query_reports.py           # Basic semantic retrieval
├── query_clean.py             # Clean query utilities
├── answer_with_citations.py   # Answer generation with citations
├── answer_hybrid.py           # Hybrid (BM25 + vector) retrieval
├── README.md
├── .gitignore
├── data/                      # Local only (ignored)
└── chroma/                    # Local only (ignored)
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/AzeemGbolahan/Linde-Packman-Student-Report-Query.git
cd Linde-Packman-Student-Report-Query
```


### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install streamlit python-dotenv langchain langchain-community \
           langchain-openai langchain-chroma chromadb \
           rank-bm25 pypdf docx2txt
```

### 4. Add your OpenAI API key
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_api_key_here
```



---

## Adding reports (local only)

Place report files under the `data/` directory.  
The folder structure can match existing archival organization.
Example:
```bash
data/
└── 1_A_Azeem_Files/
    ├── 2023_Summer_Reports/
    ├── 2024_Summer_Reports/
    └── 2025_Summer_Reports/
```
Supported formats:
- `.pdf`
- `.docx`

---

## Building the index
Run these steps once, or whenever reports change:
```bash
python load_docs.py
python split_docs.py
python build_index.py
```

This creates a local Chroma vector database inside `chroma/`.
---

## Running the interface

Start the application with:
```bash
python -m streamlit run app.py
```


Open a browser and go to:
```bash
http://localhost:8501
```

---

## Using the system

Type a natural-language question into the text box and click **Find answers**.

Example questions:
- Which students worked on machine learning or AI?
- Which students worked on cancer-related projects?
- Which students worked outside the United States?
- Which students worked with hospitals or clinical labs?
- Which projects involved genomics or oncology?

The system will return:
- A concise answer
- Student names
- Short summaries
- Source filenames for verification

---

## Notes and troubleshooting

- Always launch Streamlit using:



This ensures the correct virtual environment is used.

- If you see an error about `OPENAI_API_KEY`, confirm:
- `.env` exists
- The key is valid
- `load_dotenv()` is called in the scripts

- If answers seem incomplete, rebuild the index.

---

## Intended use

This project is designed for **internal academic use** as a decision-support and exploration tool.  
It is not intended for public deployment without additional access controls.

---

## License

Private / internal use only.


