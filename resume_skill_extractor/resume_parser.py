import os
import fitz  # PyMuPDF
import spacy
import streamlit as st
import docx  # python-docx
import re
# --- Load spaCy ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- PDF Extraction ---
@st.cache_data
def extract_text_from_pdf_cached(file_bytes):
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text()
    return text.strip()

# --- DOCX Extraction ---
@st.cache_data
def extract_text_from_docx_cached(file_bytes):
    doc = docx.Document(file_bytes)
    text = []
    for para in doc.paragraphs:
        line = para.text.strip()
        if line:  # skip empty lines
            text.append(line)
    return "\n".join(text)

# --- Universal Extraction (PDF + DOCX) ---
def extract_text(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf_cached(file.read())
    elif file.name.endswith(".docx"):
        return extract_text_from_docx_cached(file)
    else:
        return ""

# --- Clean Text ---
def clean_text(text):
    return " ".join(text.strip().split())












# --- Words to ignore to avoid false positives ---
IGNORE_WORDS = set([
    "Generative AI", "Python", "Machine Learning", "AI", "Resume",
    "CV", "Curriculum", "Vitae", "Project", "Internship"
])

# --- LLM candidate name extraction ---
def extract_candidate_name_llm(resume_text: str, llm) -> str:
    """
    Extract candidate name using LLM (OpenAI, Groq, Ollama)
    llm: callable or LangChain-like object
    Returns None if LLM fails.
    """
    if not resume_text:
        return None

    prompt = f"""
You are an assistant that extracts **only the full name** of the candidate from a resume.
Ignore headings, project names, skills, or company names.
If you cannot find a proper name, reply exactly: Candidate

Resume text (first 2000 chars):
{resume_text[:2000]}
"""
    try:
        # LangChain-style LLM
        if hasattr(llm, "generate"):
            out = llm.generate([prompt])
            generations = getattr(out, "generations", None)
            if generations and generations[0] and generations[0][0]:
                name = generations[0][0].text.strip().splitlines()[0]
                if name and name.lower() != "candidate":
                    return name

        # Simple callable LLM (e.g., Groq, OpenAI wrapper)
        elif callable(llm):
            res = llm(prompt)
            if isinstance(res, dict) and "text" in res:
                res = res["text"]
            name = str(res).strip().splitlines()[0]
            if name and name.lower() != "candidate":
                return name
    except Exception:
        pass

    return None  # LLM failed

# --- Fallback: regex / spaCy ---
def extract_candidate_name_fallback(resume_text: str) -> str:
    lines = resume_text.splitlines()
    # Step 1: Top 30 lines regex check
    for line in lines[:30]:
        ln = line.strip()
        if not ln or any(skip in ln.lower() for skip in ["@", "www", "linkedin", "resume", "cv", "curriculum vitae", "phone", "email"]):
            continue
        ln_clean = re.sub(r"[^A-Za-z\s\.-]", "", ln).strip()
        words = ln_clean.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words) and ln_clean not in IGNORE_WORDS:
            return " ".join(w.capitalize() for w in words)
    # Step 2: spaCy PERSON fallback
    doc = nlp(resume_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text not in IGNORE_WORDS and len(ent.text.split()) >= 2:
            return ent.text.strip()
    return "Candidate"

# --- Unified function: tries LLM first, then fallback ---
def extract_candidate_name(resume_text: str, llm=None) -> str:
    """
    Extract candidate name from resume text.
    - Tries LLM first (if provided)
    - Falls back to regex + spaCy
    """
    if llm:
        name = extract_candidate_name_llm(resume_text, llm)
        if name:
            return name
    return extract_candidate_name_fallback(resume_text)