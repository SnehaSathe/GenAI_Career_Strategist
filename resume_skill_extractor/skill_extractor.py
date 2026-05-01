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
    os.getenv("GROQ_API_KEY")        # first check environment variable
    or st.secrets.get("GROQ_API_KEY")  # fallback to secrets.toml
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
#  RULE-BASED HEADING-BULLET EXTRACTOR                                #
#                                                                     #
#  Handles resume skill-section lines like:                           #
#    "• Python & Backend · Python (primary), FastAPI, Flask, ..."     #
#    "• RAG & Vector DB · FAISS, Sentence Transformers, Pinecone"     #
#    "• LLM Integrations · OpenAI/GPT, Groq LLMs, Ollama, LangChain" #
#    "• Databases : MySQL, SQLite, FAISS (vector)"                    #
# ------------------------------------------------------------------ #

# Leading bullet characters produced by PyMuPDF / various resume formats
_BULLET_RE = re.compile(
    r'^[\u2022\u25aa\u25b8\u25ba\u25c6\u25cf\u2043•▪▸►◆●\-\*]\s*'
)

# Separators between heading label and comma-separated skill list
# Covers: middle-dot (· \u00b7), pipe (|), em-dash (—), colon (:)
_SEP_RE = re.compile(r'\s*[\u00b7·|—:]\s*', re.UNICODE)


def _split_heading(heading: str) -> list[str]:
    """
    Split heading on ' & ' or ' and ' so each part becomes a candidate term.
    'Python & Backend'  →  ['Python', 'Backend']
    'RAG & Vector DB'   →  ['RAG', 'Vector DB']
    'LLM Integrations'  →  ['LLM Integrations']
    """
    parts = re.split(r'\s+(?:&|and)\s+', heading, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _split_skills(skills_str: str) -> list[str]:
    """
    Split comma-separated skill list and clean each item.
    Strips parenthetical qualifiers: 'Python (primary)' → 'Python'
    Does NOT split on '/' to keep names like 'OpenAI/GPT' intact.
    """
    items = [s.strip() for s in skills_str.split(',') if s.strip()]
    cleaned = []
    for item in items:
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
    """
    raw: list[str] = []
    lines = resume_text.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Must start with a bullet character
        if not _BULLET_RE.match(stripped):
            continue

        # Remove the leading bullet to get pure content
        content = _BULLET_RE.sub('', stripped).strip()

        # Must contain a separator between heading and skill list
        if not _SEP_RE.search(content):
            continue

        # Split into exactly 2 parts: heading | skill list
        parts = _SEP_RE.split(content, maxsplit=1)
        if len(parts) != 2:
            continue

        heading_part = parts[0].strip()   # e.g. "Python & Backend"
        skills_part  = parts[1].strip()   # e.g. "Python (primary), FastAPI, Flask"

        # "Python & Backend" → ["Python", "Backend"]
        heading_terms = _split_heading(heading_part)

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


# ---------------- SKILL FILTER ----------------
def filter_skill_terms(skills: list[str]) -> list[str]:
    """
    Remove entries that are clearly sentences / phrases, not skill names.

    Blocks:
      - Strings longer than 40 characters
      - Strings with more than 5 words
      - Strings ending with sentence punctuation (. ! ?)
      - Strings starting with common resume action verbs
      - Strings containing metric patterns like "150x", "15%"

    Examples blocked:
      "Independently researched skill-scoring methods for resume-JD matching"
      "Reduced average skill-mapping turnaround from ~20 min to <8 sec"
      "a 150x speed"   "15% productivity"   "30% efficiency gains."
    """
    _SENTENCE_VERBS = re.compile(
        r'^(independently|built|wrote|designed|developed|created|implemented|'
        r'reduced|improved|increased|achieved|managed|led|collaborated|'
        r'investigated|researched|deployed|integrated|automated|optimised|'
        r'optimized|utilized|leveraged|delivered|established|conducted|'
        r'performed|generated|produced|analysed|analyzed)\b',
        re.IGNORECASE
    )

    _METRIC_RE = re.compile(r'\d+\s*[x%]', re.IGNORECASE)

    filtered = []
    for skill in skills:
        s = skill.strip()

        if len(s) > 40:                   # too long in characters
            continue
        if len(s.split()) > 5:            # too many words
            continue
        if s.endswith(('.', '!', '?')):   # ends like a sentence
            continue
        if _SENTENCE_VERBS.match(s):      # starts with action verb
            continue
        if _METRIC_RE.search(s):          # contains a metric
            continue

        filtered.append(s)

    return filtered


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

    Layer 1 — Rule-based heading-bullet parser:
        Captures heading labels ("Python & Backend" → "Python", "Backend")
        AND every comma-separated skill after the separator (·  |  :).

    Layer 2 — LLM full-text extraction:
        Catches skills buried in prose, project descriptions, summaries.
        Also extracts JD required skills.

    Layer 3 — Merge:
        Structural terms come first (ground truth).
        LLM-only terms appended to fill gaps.

    Layer 4 — Filter:
        Removes sentences, metrics, and achievement phrases.

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
- Do NOT include sentences, achievement phrases, or metrics.
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
    structural_lower = {s.lower() for s in structural_skills}
    merged_resume = structural_skills + [
        s for s in resume_skills_llm
        if s.lower() not in structural_lower
    ]

    # ── Layer 4: Filter out sentences, metrics, achievement phrases ────────
    merged_resume = filter_skill_terms(merged_resume)
    jd_skills     = filter_skill_terms(jd_skills)

    return merged_resume, jd_skills


# ---------------- RENDER AI EXPLANATION ----------------
def render_ai_explanation(explanation: dict, match_score: float):
    st.markdown("## 🤖 AI Explanation")

    # Match Level
    st.markdown(f"**Match Level: {match_score:.1f}%**")
    st.progress(match_score / 100)
    st.markdown("---")

    # Overall Summary
    if explanation.get("overall_summary"):
        st.markdown(explanation["overall_summary"])
        st.markdown("")

    # Strengths
    if explanation.get("strengths"):
        st.markdown("**Strengths:**")
        for s in explanation["strengths"]:
            st.markdown(f"- {s}")
        st.markdown("")

    # Skill Gaps
    if explanation.get("skill_gaps"):
        st.markdown("**Gaps:**")
        for g in explanation["skill_gaps"]:
            st.markdown(f"- {g}")
        st.markdown("")

    # Improvement Suggestions / Recommendation
    if explanation.get("recommendation") or explanation.get("interview_tips"):
        st.markdown("**Improvement Suggestions:**")
        if explanation.get("recommendation"):
            st.markdown(f"- {explanation['recommendation']}")
        for tip in explanation.get("interview_tips", []):
            st.markdown(f"- {tip}")
        st.markdown("")

    # Score Explanation
    if explanation.get("score_explanation"):
        st.markdown("**Score Explanation:**")
        st.markdown(explanation["score_explanation"])

    st.markdown("---")