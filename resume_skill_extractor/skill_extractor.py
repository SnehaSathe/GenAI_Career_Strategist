import os
import re
import json
import socket
import requests
import fitz  # PyMuPDF
import streamlit as st
from langchain_community.llms import Ollama

# ---------------- CONFIG ----------------
groq_api_key = (
    os.getenv("GROQ_API_KEY")
    or st.secrets.get("GROQ_API_KEY")
)
OLLAMA_MODEL = "mistral:latest"


# ---------------- ENV DETECTION ----------------
def is_local_env():
    """Detect if running locally (for Ollama fallback)."""
    try:
        host = socket.gethostname()
        ip = socket.gethostbyname(host)
        return ip.startswith("127.") or ip == "localhost"
    except:
        return False


# ---------------- OLLAMA ----------------
@st.cache_resource
def get_ollama_client():
    """Initialize Ollama client (cached)."""
    return Ollama(model=OLLAMA_MODEL)


def use_ollama(prompt: str) -> str:
    """Run prompt locally via Ollama."""
    llm = get_ollama_client()
    return llm.invoke(prompt)


# ---------------- GROQ ----------------
def use_groq(prompt, model_choice, api_key):
    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json={
            "model": model_choice,
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]

    st.error(f"Groq API error {response.status_code}: {response.text}")
    return None


# ------------------------------------------------------------------ #
#  RULE-BASED HEADING-BULLET EXTRACTOR  (NEW)                         #
#                                                                      #
#  Handles resume skill-section lines like:                            #
#    "• Python & Backend · Python (primary), FastAPI, Flask, ..."      #
#    "• RAG & Vector DB · FAISS, Sentence Transformers, Pinecone"      #
#    "• LLM Integrations · OpenAI/GPT, Groq LLMs, Ollama, LangChain"  #
#    "• Databases : MySQL, SQLite, FAISS (vector)"                     #
#                                                                      #
#  What it does:                                                        #
#   1. Detects bullet lines that contain a heading + separator + list. #
#   2. Splits heading on & / and  → each group is a candidate term.   #
#   3. Splits skill list on commas → each item is a candidate term.   #
#   4. Light cleanup (strips parentheticals like "(primary)").         #
#   No predefined allow/block lists — pure structural parsing.         #
# ------------------------------------------------------------------ #

# Leading bullet characters produced by PyMuPDF / various resume formats
_BULLET_RE = re.compile(
    r'^[\u2022\u25aa\u25b8\u25ba\u25c6\u25cf\u2043•▪▸►◆●\-\*]\s*'
)

# Separators between the heading label and the comma-separated skill list.
# Covers: middle-dot (·  \u00b7), pipe (|), em-dash (—), colon (:)
_SEP_RE = re.compile(r'\s*[\u00b7·|—:]\s*', re.UNICODE)


def _split_heading(heading: str) -> list[str]:
    """
    Split a heading label on ' & ' or ' and ' so each part becomes
    an individual candidate skill term.

    Examples:
        'Python & Backend'   →  ['Python', 'Backend']
        'RAG & Vector DB'    →  ['RAG', 'Vector DB']
        'LLM Integrations'   →  ['LLM Integrations']
        'Cloud & DevOps'     →  ['Cloud', 'DevOps']
    """
    parts = re.split(r'\s+(?:&|and)\s+', heading, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _split_skills(skills_str: str) -> list[str]:
    """
    Split a comma-separated skill list and clean each item.

    - Strips parenthetical qualifiers: 'Python (primary)' → 'Python'
    - Keeps compound names intact: 'GitHub Actions', 'REST APIs', 'CI/CD'
    - Does NOT split on '/' because many skills use it: 'OpenAI/GPT'

    Example:
        'Python (primary), FastAPI, Flask, REST APIs, Async Programming'
        → ['Python', 'FastAPI', 'Flask', 'REST APIs', 'Async Programming']
    """
    items = [s.strip() for s in skills_str.split(',') if s.strip()]
    cleaned = []
    for item in items:
        # Remove trailing parenthetical notes
        item = re.sub(r'\s*\([^)]*\)', '', item).strip()
        if item:
            cleaned.append(item)
    return cleaned


def extract_skills_from_headings(resume_text: str) -> list[str]:
    """
    Parse every bullet line in the resume and extract candidate skill terms
    from BOTH the heading label and the comma-separated skill list.

    A qualifying line must:
      - Start with a bullet character (•, -, *, ▪, ►, etc.)
      - Contain a separator (·, |, —, :) splitting heading from skills

    Returns a flat, deduplicated list of raw candidate strings.
    These are merged with LLM-extracted skills in extract_skills_cached().
    """
    raw: list[str] = []
    lines = resume_text.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # ── Must start with a bullet character ───────────────────────────
        if not _BULLET_RE.match(stripped):
            continue

        # Remove the leading bullet to get pure content
        content = _BULLET_RE.sub('', stripped).strip()

        # ── Must contain a separator between heading and skill list ───────
        if not _SEP_RE.search(content):
            continue

        # Split into exactly 2 parts: heading | skill list
        parts = _SEP_RE.split(content, maxsplit=1)
        if len(parts) != 2:
            continue

        heading_part = parts[0].strip()   # e.g. "Python & Backend"
        skills_part  = parts[1].strip()   # e.g. "Python (primary), FastAPI, Flask"

        # ── Extract candidate terms from heading ──────────────────────────
        # "Python & Backend" → ["Python", "Backend"]
        heading_terms = _split_heading(heading_part)

        # ── Extract candidate terms from skill list ───────────────────────
        # "Python (primary), FastAPI, Flask" → ["Python", "FastAPI", "Flask"]
        skill_terms = _split_skills(skills_part)

        raw.extend(heading_terms)
        raw.extend(skill_terms)

    # Deduplicate preserving first-seen order; drop single-char noise
    seen: set[str] = set()
    result: list[str] = []
    for term in raw:
        key = term.lower().strip()
        if key and len(key) > 1 and key not in seen:
            seen.add(key)
            result.append(term.strip())

    return result


# ---------------- MAIN EXTRACTION ----------------
@st.cache_data(show_spinner=False)
def extract_skills_cached(
    resume_text: str,
    jd_text: str,
    model_choice: str,
    groq_api_key=None
) -> tuple[list[str], list[str]]:
    """
    Extract skills separately for Resume and JD.

    Layer 1 — Rule-based heading-bullet parser (NEW):
        Captures heading labels ("Python & Backend" → "Python", "Backend")
        AND every comma-separated skill after the separator (·  |  :).
        Guaranteed to catch structured Skills sections regardless of LLM.

    Layer 2 — LLM full-text extraction:
        Catches skills buried in prose, project descriptions, summaries.
        Also extracts JD required skills.

    Layer 3 — Merge:
        Structural terms come first (ground truth from explicit skill section).
        LLM-only terms appended to fill gaps from prose sections.

    Always returns: (resume_skills_list, jd_skills_list)
    """

    # ── Layer 1: Rule-based structural extraction ─────────────────────────
    structural_skills = extract_skills_from_headings(resume_text)

    with st.expander("🔍 Debug: Heading-bullet extracted terms", expanded=False):
        st.write(f"{len(structural_skills)} raw terms from bullet headings:",
                 structural_skills)

    # ── Layer 2: LLM full-text extraction ─────────────────────────────────
    prompt = f"""
You are a strict JSON generator.
Extract **only technical skills** from the given texts.

For the Resume:
- Scan every section: summary, skills bullets, project stacks, experience.
- For bullet lines like "Heading & SubHeading · skill1, skill2":
  extract heading parts (split on &) AND all comma-separated skills.
- Normalize each skill to its canonical short form.
  Examples: "Groq LLMs" → "Groq",  "Docker-based deployments" → "Docker",
            "FAISS (vector)" → "FAISS", "Python (primary)" → "Python"
- Remove soft skills, job titles, company names, locations, degree names.
- Deduplicate case-insensitively.

For the Job Description:
- Extract only the required / preferred technical skills mentioned.

Return JSON only — no explanation, no markdown:
{{
    "resume_skills": ["Skill1", "Skill2"],
    "jd_skills": ["SkillA", "SkillB"]
}}

Resume:
{resume_text}

Job Description:
{jd_text}
"""

    raw_result = None

    if is_local_env():
        try:
            raw_result = use_ollama(prompt)
        except Exception:
            raw_result = use_groq(prompt, model_choice, groq_api_key)
    else:
        raw_result = use_groq(prompt, model_choice, groq_api_key)

    resume_skills_llm, jd_skills = [], []

    try:
        if not raw_result:
            st.warning("⚠️ LLM returned no output — using structural extraction only.")
        else:
            cleaned = raw_result.strip()

            # Strip markdown code fences if present
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                cleaned = cleaned.replace("json", "", 1).strip()

            # Find JSON object boundaries robustly
            if not cleaned.startswith("{"):
                start = cleaned.index("{")
                end   = cleaned.rindex("}") + 1
                cleaned = cleaned[start:end]

            data = json.loads(cleaned)
            resume_skills_llm = [str(s).strip() for s in data.get("resume_skills", []) if s]
            jd_skills         = [str(s).strip() for s in data.get("jd_skills",     []) if s]

    except Exception as e:
        st.error(f"⚠️ JSON parsing failed: {e}")
        st.write("Raw LLM output:", raw_result)

    # ── Layer 3: Merge structural + LLM results ───────────────────────────
    # Structural terms come first (explicit skill section = ground truth).
    # LLM-only terms fill gaps from prose / project sections.
    structural_lower = {s.lower() for s in structural_skills}
    merged_resume = structural_skills + [
        s for s in resume_skills_llm
        if s.lower() not in structural_lower
    ]

    return merged_resume, jd_skills