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
    try:
        host = socket.gethostname()
        ip = socket.gethostbyname(host)
        return ip.startswith("127.") or ip == "localhost"
    except:
        return False


# ---------------- OLLAMA ----------------
@st.cache_resource
def get_ollama_client():
    return Ollama(model=OLLAMA_MODEL)


def use_ollama(prompt: str) -> str:
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


# ---------------- LLM ROUTER ----------------
def run_llm(prompt: str, model_choice: str, api_key: str) -> str | None:
    if is_local_env():
        try:
            return use_ollama(prompt)
        except Exception:
            return use_groq(prompt, model_choice, api_key)
    else:
        return use_groq(prompt, model_choice, api_key)


# ---------------- JSON PARSER ----------------
def parse_json_response(raw: str | None, key: str) -> list[str] | str:
    """
    Safely parse a value from LLM JSON response.
    Returns list[str] for list keys, str for string keys.
    """
    if not raw:
        return [] if key != "text" else ""

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned.replace("json", "", 1).strip()
    if not cleaned.startswith("{"):
        try:
            start = cleaned.index("{")
            end = cleaned.rindex("}") + 1
            cleaned = cleaned[start:end]
        except ValueError:
            return [] if key != "text" else ""
    try:
        data = json.loads(cleaned)
        val = data.get(key, [] if key != "text" else "")
        if isinstance(val, list):
            return [str(s).strip() for s in val if s]
        return str(val).strip()
    except json.JSONDecodeError:
        return [] if key != "text" else ""


# ---------------- RESUME NORMALIZER ----------------
def normalize_resume_text(text: str) -> str:
    """
    Normalize raw PyMuPDF-extracted text.
    Converts all unicode bullet/separator variants to consistent markers.
    No predefined skill lists — purely structural cleanup.
    """
    # All unicode middot/separator variants → | SPLIT |
    text = re.sub(
        r'[\u00b7\u2022\u2027\u2023\u25e6\u2043\u204c\u204d·•‣◦‧⁃]',
        ' | ',
        text
    )
    # Unicode bullet glyphs → newline + dash
    text = re.sub(r'[\u25aa\u25b8\u25ba\u25c6\u25cf▪▸►◆●]', '\n- ', text)
    # Collapse spaces, normalize newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------- RULE-BASED STRUCTURAL PARSER ----------------
def extract_raw_terms_from_structure(resume_text: str) -> list[str]:
    """
    Pure structural extraction — no skill knowledge required.
    Pulls every candidate term from the resume structure:
    middot lines, colon lines, comma lists, tables, bullet points.

    These raw terms then go to the LLM for validation/normalization.
    No predefined lists anywhere in this function.
    """
    raw_terms = []
    text = normalize_resume_text(resume_text)
    lines = text.splitlines()

    # Track if we're inside a section that likely contains skills
    # We detect this dynamically by looking at line patterns,
    # not by matching against a predefined header list
    in_list_section = False

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Remove markdown bold/italic artifacts
        line_clean = re.sub(r'\*+', '', line).strip()

        # ── Detect section context dynamically ───────────────────────────
        # A "list section" is a short line with no comma/pipe/colon that
        # likely introduces a skill/tech section below it
        is_short_header = (
            len(line_clean) < 60
            and ',' not in line_clean
            and '|' not in line_clean
            and ' | ' not in line_clean
            and not line_clean.endswith('.')
        )
        if is_short_header and i < len(lines) - 1:
            # Peek at next non-empty line to see if it's a list
            next_lines = [l.strip() for l in lines[i+1:i+4] if l.strip()]
            if next_lines and (',' in next_lines[0] or next_lines[0].startswith('-')):
                in_list_section = True
                continue

        # ── Pattern 1: Middot/pipe separator ─────────────────────────────
        # "Category & SubCat | skill1, skill2, skill3"
        # Works for: "RAG & Vector DB | FAISS, Sentence Transformers"
        if ' | ' in line_clean:
            parts = line_clean.split(' | ', 1)
            category_part = re.sub(r'^[\-\*•]\s*', '', parts[0]).strip()
            skills_part   = parts[1].strip() if len(parts) > 1 else ''

            # Split category on & — each part is a candidate term
            cat_terms = [
                c.strip() for c in re.split(r'\s*&\s*', category_part)
                if c.strip() and len(c.strip()) > 1
            ]
            # Split skills on comma
            skill_terms = [
                s.strip() for s in skills_part.split(',')
                if s.strip() and len(s.strip()) > 1
            ]
            raw_terms.extend(cat_terms)
            raw_terms.extend(skill_terms)
            in_list_section = True
            continue

        # ── Pattern 2: Colon-separated label → values ─────────────────────
        # "Languages: Python, Java, C++"
        # "Frameworks: FastAPI, Flask, Django"
        colon_match = re.match(r'^(.{2,50}):\s*(.{2,})$', line_clean)
        if colon_match:
            values_part = colon_match.group(2)
            items = [s.strip() for s in values_part.split(',') if s.strip()]
            if len(items) >= 2:
                raw_terms.extend(items)
                in_list_section = True
                continue

        # ── Pattern 3: Comma list (in list section or standalone) ─────────
        # "Python, FastAPI, Docker, PostgreSQL, Redis"
        if ',' in line_clean and in_list_section:
            line_clean = re.sub(r'^[\-\*•]\s*', '', line_clean)
            items = [s.strip() for s in line_clean.split(',') if s.strip()]
            if len(items) >= 2:
                raw_terms.extend(items)
                continue

        # ── Pattern 4: Pipe table row ──────────────────────────────────────
        # "| Python | FastAPI | Docker |"
        if line_clean.startswith('|') and '|' in line_clean[1:]:
            items = [
                s.strip() for s in line_clean.strip('|').split('|')
                if s.strip() and not re.match(r'^[-=\s]+$', s.strip())
            ]
            raw_terms.extend(items)
            continue

        # ── Pattern 5: Single item bullet in list section ──────────────────
        # "- Python" or "• FastAPI"
        if in_list_section:
            single = re.sub(r'^[\-\*•]\s*', '', line_clean).strip()
            # Only if it looks like a skill: short, no sentence punctuation
            if single and len(single) < 60 and not single.endswith('.'):
                raw_terms.append(single)
                continue

        # ── Reset list section if we hit a prose line ─────────────────────
        if line_clean.endswith('.') and len(line_clean) > 80:
            in_list_section = False

    # Deduplicate preserving order
    seen = set()
    result = []
    for s in raw_terms:
        key = s.lower().strip()
        if key and key not in seen and len(key) > 1:
            seen.add(key)
            result.append(s.strip())

    return result


# ---------------- LLM SKILL VALIDATOR ----------------
def validate_and_split_skills(
    raw_terms: list[str],
    model_choice: str,
    api_key: str
) -> list[str]:
    """
    LLM decides:
    1. Which raw terms are valid technical skills
    2. Which compound terms to split (e.g. "RAG & Vector DB" → ["RAG","Vector DB"])
    3. Normalization to canonical forms
    4. Deduplication

    No predefined skill lists — the LLM uses its own knowledge to judge.
    """
    if not raw_terms:
        return []

    terms_json = json.dumps(raw_terms)

    prompt = f"""
You are a strict JSON generator and technical skill expert.

I extracted these raw terms from a resume. Your job:

1. VALIDATE: Keep only genuine technical skills.
   A technical skill is: a programming language, framework, library, tool,
   platform, database, cloud service, ML/AI concept, DevOps practice,
   protocol, API, or CS methodology that a hiring manager would care about.
   Remove: generic words, soft skills, section headers, adjectives,
   non-technical nouns, company names, city names, degree names,
   anything vague or non-technical.

2. SPLIT compounds joined by "&" — but ONLY if BOTH parts are valid skills:
   Look at each term. If it is two technical skills joined by "&", split it.
   Use your own knowledge to judge — do not use any fixed list.
   Example logic:
     "RAG & Vector DB"    → RAG is a valid skill, Vector DB is a valid skill → split
     "Cloud & DevOps"     → Cloud is valid, DevOps is valid → split
     "Research & Methods" → neither is a standalone technical skill → remove both
     "AI & GenAI"         → both valid → split

3. KEEP atomic multi-word skills as single entries.
   Use your knowledge to identify skills that are naturally one concept
   even if they contain multiple words or special characters.
   Examples: "CI/CD", "GitHub Actions", "LangChain", "REST API" are single skills.

4. NORMALIZE each skill to its most commonly used, canonical short form.
   Use your knowledge of the tech industry — no fixed mapping needed.
   Examples: "Docker-based deployments" → "Docker",
             "FAISS (vector)" → "FAISS",
             "Groq LLMs" → "Groq"

5. DEDUPLICATE case-insensitively. Keep the cleaner/shorter form.

Raw terms to process:
{terms_json}

Return ONLY this JSON, no explanation, no markdown:
{{
    "skills": ["skill1", "skill2", "skill3"]
}}
"""

    raw = run_llm(prompt, model_choice, api_key)
    result = parse_json_response(raw, "skills")
    return result if result else raw_terms


# ---------------- MAIN EXTRACTION ----------------
@st.cache_data(show_spinner=False)
def extract_skills_cached(
    resume_text: str,
    jd_text: str,
    model_choice: str,
    groq_api_key=None
) -> tuple[list[str], list[str]]:
    """
    Extract skills from ANY resume format + JD.
    No predefined skill lists anywhere.

    Layer 1 — Structural parser:
        Pure regex on resume structure (bullets, tables, colons, middots).
        Pulls raw candidate terms regardless of domain.
        Guaranteed to capture "RAG & Vector DB" category headers.

    Layer 2 — LLM extraction:
        Reads full resume + JD text.
        Catches skills in prose, summaries, project descriptions.
        Also extracts JD skills.

    Layer 3 — LLM validation:
        Validates, splits compounds, normalizes, deduplicates.
        Uses LLM's own knowledge — no hardcoded skill lists.

    Returns: (resume_skills, jd_skills)
    """

    # ── Layer 1: Structural extraction ────────────────────────────────────
    raw_structural_terms = extract_raw_terms_from_structure(resume_text)

    with st.expander("🔍 Debug: Structural raw terms", expanded=False):
        st.write(f"{len(raw_structural_terms)} raw terms from structure:",
                 raw_structural_terms)

    # ── Layer 2: LLM full-text extraction ─────────────────────────────────
    prompt = f"""
You are a strict JSON generator for technical skill extraction.

This resume may use ANY format — structured bullets, colon lists, prose,
tables, or no structure at all. Extract from everything.

TASK:
Extract all individual technical skills from the Resume.
Extract all required technical skills from the Job Description.

RULES:
- Scan every part: summary, skills section, projects, experience, tools,
  certifications, and any other section.
- For structured bullet lines like "Category & Sub | skill1, skill2":
  extract the category parts (split on &) AND all comma-separated skills.
- For prose: extract every tool/technology/framework mentioned.
- Normalize each skill to its canonical short form using your knowledge.
- Split compound terms joined by "/" where both parts are distinct skills.
  Keep atomic multi-word skills as single entries.
- Remove soft skills, job titles, company names, locations, years,
  degree names, and non-technical terms.
- Deduplicate case-insensitively.

Return ONLY this JSON, no markdown, no explanation:
{{
    "resume_skills": ["skill1", "skill2"],
    "jd_skills": ["skillA", "skillB"]
}}

Resume:
{resume_text}

Job Description:
{jd_text}
"""

    raw_result = run_llm(prompt, model_choice, groq_api_key)

    resume_skills_llm, jd_skills = [], []

    if not raw_result:
        st.error("⚠️ LLM extraction call failed.")
    else:
        try:
            cleaned = raw_result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                cleaned = cleaned.replace("json", "", 1).strip()
            if not cleaned.startswith("{"):
                start = cleaned.index("{")
                end = cleaned.rindex("}") + 1
                cleaned = cleaned[start:end]

            data = json.loads(cleaned)
            resume_skills_llm = [str(s).strip() for s in data.get("resume_skills", []) if s]
            jd_skills         = [str(s).strip() for s in data.get("jd_skills", []) if s]

        except Exception as e:
            st.error(f"⚠️ JSON parsing failed: {e}")
            st.write("Raw output:", raw_result)

    # ── Layer 3: Merge + LLM validation ───────────────────────────────────
    # Structural terms come first (ground truth for structured sections)
    # LLM terms fill in prose/project mentions
    structural_lower = {s.lower() for s in raw_structural_terms}
    merged_resume = raw_structural_terms + [
        s for s in resume_skills_llm
        if s.lower() not in structural_lower
    ]

    # LLM validates, splits compounds, normalizes — no predefined lists
    resume_skills = validate_and_split_skills(merged_resume, model_choice, groq_api_key)
    jd_skills     = validate_and_split_skills(jd_skills, model_choice, groq_api_key)

    return resume_skills, jd_skills


# ---------------- EXPLANATION GENERATOR ----------------
def generate_explanation(
    resume_text: str,
    jd_text: str,
    matched_skills: list[str],
    missing_skills: list[str],
    additional_skills: list[str],
    match_score: float,
    model_choice: str,
    api_key: str
) -> dict:
    """
    Generate a personalized explanation based on the actual resume
    and job description content.

    Returns dict with keys:
        overall_summary   : direct verdict on fit
        strengths         : specific strengths from resume
        skill_gaps        : honest gap assessment
        additional_value  : unique value beyond JD requirements
        recommendation    : actionable next steps
        interview_tips    : tips based on actual resume experience
        score_explanation : why this score, what would improve it
    """
    prompt = f"""
You are an expert technical recruiter and career coach.

Analyze this candidate's resume against the job description.
Be specific — reference actual projects, numbers, and technologies
from the resume. Do NOT give generic advice.

Match Score: {match_score:.1f}%
Matched Skills: {json.dumps(matched_skills)}
Missing Skills: {json.dumps(missing_skills)}
Additional Skills (candidate has, JD didn't ask): {json.dumps(additional_skills)}

Resume:
{resume_text}

Job Description:
{jd_text}

Return ONLY this JSON, no markdown, no extra text:
{{
    "overall_summary": "2-3 sentence direct verdict. Reference the actual role and candidate background.",

    "strengths": [
        "Specific strength referencing actual resume project/metric/tech",
        "Specific strength referencing actual resume project/metric/tech",
        "Specific strength referencing actual resume project/metric/tech"
    ],

    "skill_gaps": [
        "Gap with context: is it critical or nice-to-have for this role?",
        "Gap with context: suggest how they might address it"
    ],

    "additional_value": "What unique value does this candidate bring that the JD didn't ask for? Reference actual resume content.",

    "recommendation": "Should they apply? What should they do or say in their application to stand out? Be specific.",

    "interview_tips": [
        "Which specific project to highlight and why",
        "Which metric or achievement to lead with",
        "How to address the skill gap in an interview"
    ],

    "score_explanation": "Why this score? What 1-2 things would push it significantly higher?"
}}
"""

    raw = run_llm(prompt, model_choice, api_key)

    if not raw:
        return {}

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned.replace("json", "", 1).strip()
    if not cleaned.startswith("{"):
        try:
            start = cleaned.index("{")
            end = cleaned.rindex("}") + 1
            cleaned = cleaned[start:end]
        except ValueError:
            return {}
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}