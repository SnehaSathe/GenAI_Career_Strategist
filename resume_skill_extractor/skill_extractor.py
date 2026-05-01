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
def parse_json_response(raw: str | None, key: str) -> list[str]:
    if not raw:
        return []

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
            return []

    try:
        data = json.loads(cleaned)
        return [str(s).strip() for s in data.get(key, []) if s]
    except json.JSONDecodeError:
        return []


# ---------------- BULLET PREPROCESSOR ----------------
def extract_bullet_skills(resume_text: str) -> list[str]:
    """
    Parses resume bullet lines of the pattern:
        "Category & SubCategory · skill1, skill2, skill3"

    For each such line:
      - Splits the CATEGORY part on ' & ' → individual category skills
      - Splits the SKILLS part on ',' → individual skills
      - Returns ALL of them as a flat list

    Example:
      "RAG & Vector DB · FAISS, Sentence Transformers, Pinecone"
      → ["RAG", "Vector DB", "FAISS", "Sentence Transformers", "Pinecone"]

      "AI/ML & GenAI · AI/ML Model Support, Generative AI"
      → ["AI/ML", "GenAI", "AI/ML Model Support", "Generative AI"]

      "Cloud & DevOps · Azure, Docker, GitHub"
      → ["Cloud", "DevOps", "Azure", "Docker", "GitHub"]
    """
    extracted = []

    # Match lines with the · separator (handles both · and the ASCII middot)
    # Pattern: <anything> · <anything>
    bullet_pattern = re.compile(r'(.+?)\s*[·•]\s*(.+)')

    for line in resume_text.splitlines():
        line = line.strip()
        # Strip leading bullet characters
        line = re.sub(r'^[\-\*•▪▸►]\s*', '', line)

        match = bullet_pattern.match(line)
        if not match:
            continue

        category_part = match.group(1).strip()
        skills_part   = match.group(2).strip()

        # Remove bold markdown if present (**text**)
        category_part = re.sub(r'\*+', '', category_part).strip()

        # Split category on ' & ' to get individual category-level skills
        # e.g. "RAG & Vector DB" → ["RAG", "Vector DB"]
        category_skills = [
            c.strip()
            for c in re.split(r'\s*&\s*', category_part)
            if c.strip()
        ]

        # Split skills part on ',' to get individual skills
        # e.g. "FAISS, Sentence Transformers, Pinecone" → ["FAISS", ...]
        inline_skills = [
            s.strip()
            for s in skills_part.split(',')
            if s.strip()
        ]

        extracted.extend(category_skills)
        extracted.extend(inline_skills)

    # Deduplicate preserving order
    seen = set()
    result = []
    for s in extracted:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            result.append(s)

    return result


# ---------------- POST-PROCESSING ----------------
def post_process_skills(skills: list[str], model_choice: str, api_key: str) -> list[str]:
    """
    Send extracted skills to LLM to:
    1. Keep only valid technical skills.
    2. Split any remaining compound entries (e.g. "RAG & Vector DB") if both parts are valid.
    3. Normalize to canonical short forms.
    4. Deduplicate.
    """
    if not skills:
        return []

    skills_json = json.dumps(skills)

    prompt = f"""
You are a strict JSON generator for technical skill validation and normalization.

Given this list of raw extracted terms, do the following:
1. Keep only valid technical skills (tools, frameworks, languages, platforms,
   databases, cloud services, ML/AI concepts, CS methodologies).
2. Remove non-technical terms (generic words, soft skills, adjectives, vague nouns).
3. Split compound entries joined by "&" ONLY if BOTH parts are valid technical skills:
   - "RAG & Vector DB"  → both valid → ["RAG", "Vector DB"]
   - "Cloud & DevOps"   → both valid → ["Cloud", "DevOps"]
   - "AI & GenAI"       → both valid → ["AI", "GenAI"]
4. Keep atomic skills intact:
   - "CI/CD"           → keep as "CI/CD"
   - "GitHub Actions"  → keep as "GitHub Actions"
   - "REST API"        → keep as "REST API"
   - "LangChain"       → keep as "LangChain"
5. Normalize:
   - "RAG systems"              → "RAG"
   - "LLM Integrations"        → "LLMs"
   - "Docker-based deployments" → "Docker"
   - "FAISS (vector)"           → "FAISS"
   - "Groq LLMs"               → "Groq"
6. Remove duplicates (case-insensitive).

Input terms:
{skills_json}

Return ONLY this JSON, no explanation, no markdown:
{{
    "skills": ["skill1", "skill2", "skill3"]
}}
"""

    raw = run_llm(prompt, model_choice, api_key)
    result = parse_json_response(raw, "skills")
    return result if result else skills


# ---------------- MAIN EXTRACTION ----------------
@st.cache_data(show_spinner=False)
def extract_skills_cached(
    resume_text: str,
    jd_text: str,
    model_choice: str,
    groq_api_key=None
) -> tuple[list[str], list[str]]:
    """
    Extract skills from Resume and JD.
    Always returns: (resume_skills_list, jd_skills_list).
    """

    # Step 1: Rule-based bullet parsing — guarantees "RAG & Vector DB"
    # category headers are always captured, no LLM hallucination risk
    bullet_skills = extract_bullet_skills(resume_text)

    # Step 2: LLM extracts skills from full text (catches skills in
    # summary, project descriptions, experience bullets etc.)
    prompt = f"""
You are a strict JSON generator for technical skill extraction.

EXTRACTION RULES:
- Scan EVERY section: Summary, Core Skills, Projects, Experience, Tools.
- Extract every individual technical skill mentioned anywhere.
- For bullet lines like "Category · skill1, skill2": extract ALL skills
  listed after the · separator.
- Normalize:
    "RAG systems"               → "RAG"
    "LLM Integrations"          → "LLMs"
    "Docker-based deployments"  → "Docker"
    "FAISS (vector)"            → "FAISS"
    "REST APIs"                 → "REST API"
    "Groq LLMs"                 → "Groq"
- Keep atomic compound skills intact:
    "CI/CD", "GitHub Actions", "Prompt Engineering",
    "System Design", "LangChain", "LangGraph" → single entries.
- Remove duplicates (case-insensitive).
- Exclude: soft skills, job titles, company names, city names,
  university names, years of experience.

Return ONLY valid JSON, no markdown, no explanation:
{{
    "resume_skills": ["Skill1", "Skill2"],
    "jd_skills": ["SkillA", "SkillB"]
}}

Resume:
{resume_text}

Job Description:
{jd_text}
"""

    raw_result = run_llm(prompt, model_choice, groq_api_key)

    if not raw_result:
        st.error("⚠️ LLM call failed for skill extraction.")
        return [], []

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
        st.write("Raw output was:", raw_result)
        return [], []

    # Step 3: Merge bullet-parsed skills with LLM-extracted skills
    # bullet_skills guarantees category headers like "RAG", "Vector DB" are present
    merged_resume = bullet_skills + [
        s for s in resume_skills_llm
        if s.lower() not in {b.lower() for b in bullet_skills}
    ]

    # Step 4: Post-process both lists — validate, normalize, deduplicate
    resume_skills = post_process_skills(merged_resume, model_choice, groq_api_key)
    jd_skills     = post_process_skills(jd_skills, model_choice, groq_api_key)

    return resume_skills, jd_skills