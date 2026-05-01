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
    os.getenv("GROQ_API_KEY")   # first check environment variable
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


# ---------------- LLM ROUTER ----------------
def run_llm(prompt: str, model_choice: str, api_key: str) -> str | None:
    """Route prompt to Ollama (local) or Groq (cloud)."""
    if is_local_env():
        try:
            return use_ollama(prompt)
        except Exception:
            return use_groq(prompt, model_choice, api_key)
    else:
        return use_groq(prompt, model_choice, api_key)


# ---------------- JSON PARSER ----------------
def parse_json_response(raw: str | None, key: str) -> list[str]:
    """
    Safely parse a list of skills from an LLM JSON response.
    Handles markdown fences and leading/trailing prose gracefully.
    """
    if not raw:
        return []

    cleaned = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned.replace("json", "", 1).strip()

    # Brace-extraction fallback: handles prose before/after JSON
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


# ---------------- COMPOUND SKILL EXPANDER ----------------
def ask_llm_to_validate_parts(
    parts: list[str],
    model_choice: str,
    api_key: str
) -> list[str]:
    """
    Ask the LLM which parts of a split compound skill are valid
    standalone technical skills. Returns only the valid ones.
    """
    parts_json = json.dumps(parts)

    prompt = f"""
You are a strict JSON generator.

Given this list of terms, return only the ones that are valid, standalone
technical skills (tools, frameworks, languages, platforms, methodologies,
databases, cloud services, or computer science concepts).

Exclude: generic words, soft skills, adjectives, vague nouns, non-technical terms.

Input terms: {parts_json}

Return ONLY this JSON, no explanation, no markdown:
{{
    "valid_skills": ["term1", "term2"]
}}
"""

    raw = run_llm(prompt, model_choice, api_key)
    result = parse_json_response(raw, "valid_skills")

    # Fallback: if LLM returns nothing, keep all parts
    return result if result else parts


def expand_compound_skills(
    skills: list[str],
    model_choice: str,
    api_key: str
) -> list[str]:
    """
    Split compound skill entries joined by ' & ' or ' / ' (with spaces)
    into individual skills, using the LLM to validate each part.

    Why spaces matter:
      "RAG & Vector DB" → spaces around & → split → ["RAG", "Vector DB"]
      "CI/CD"           → no spaces around / → kept as-is (atomic skill)
      "AWS/GCP"         → no spaces → kept as-is (main prompt handles this)

    After splitting, the LLM decides if each part is a real technical skill:
      ["RAG", "Vector DB"] → LLM: both valid → expand to two entries
      ["something", "vague"] → LLM: neither valid → keep original string
    """
    # Only match ' & ' or ' / ' with surrounding spaces
    # This deliberately preserves "CI/CD", "AI/ML" etc. as atomic
    splitter = re.compile(r'\s+&\s+|\s+/\s+')

    expanded = []

    for skill in skills:
        parts = splitter.split(skill)

        if len(parts) == 1:
            # No compound separator found → keep as-is
            expanded.append(skill.strip())
            continue

        # Ask LLM which parts are valid standalone technical skills
        valid_parts = ask_llm_to_validate_parts(
            [p.strip() for p in parts],
            model_choice,
            api_key
        )

        if len(valid_parts) >= 2:
            # All/most parts are valid → split into separate skills
            expanded.extend(valid_parts)
        elif len(valid_parts) == 1:
            # Only one part is valid → keep just that
            expanded.extend(valid_parts)
        else:
            # LLM found nothing valid → keep original unchanged
            expanded.append(skill.strip())

    # Deduplicate while preserving insertion order
    seen = set()
    result = []
    for s in expanded:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            result.append(s)

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
    Extract skills from Resume and JD using a single LLM call.
    Post-processes compound skills (e.g. "RAG & Vector DB") by splitting
    them into individual valid technical skills via a second LLM validation call.

    Always returns: (resume_skills_list, jd_skills_list)
    """

    prompt = f"""
You are a strict JSON generator for technical skill extraction.

Rules:
- Extract technical skills from EVERY section of the resume text:
  Summary, Core Skills, Projects, Experience, Tools — all of it.
- For resume bullet lines formatted as "Category · skill1, skill2, skill3",
  extract the category concepts AND every skill listed after the dot/colon.
  Example: "RAG & Vector DB · FAISS, Sentence Transformers, Pinecone"
           → resume_skills must include: RAG, Vector DB, FAISS, Sentence Transformers, Pinecone
- Normalize skills to their common short form:
    "RAG systems"              → "RAG"
    "LLM Integrations"        → "LLMs"
    "Docker-based deployments" → "Docker"
    "FAISS (vector)"           → "FAISS"
    "REST APIs"                → "REST API"
- Do NOT pre-split entries joined by "&" or "/" — return them as-is.
  They will be handled in post-processing.
  Example: "RAG & Vector DB" → return as "RAG & Vector DB" (do not split here)
- Do NOT split naturally atomic skills:
  "CI/CD", "GitHub Actions", "Prompt Engineering",
  "System Design", "LangChain", "LangGraph" → keep as single entries.
- Remove duplicates (case-insensitive).
- Do NOT include: soft skills, years of experience, job titles,
  company names, university names, or city names.
- Return ONLY valid JSON in this exact format, no extra text, no markdown:

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

    # Parse resume and JD skills from JSON response
    try:
        cleaned = raw_result.strip()

        # Strip ```json ... ``` fences
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            cleaned = cleaned.replace("json", "", 1).strip()

        # Brace-extraction fallback
        if not cleaned.startswith("{"):
            start = cleaned.index("{")
            end = cleaned.rindex("}") + 1
            cleaned = cleaned[start:end]

        data = json.loads(cleaned)

        resume_skills = [str(s).strip() for s in data.get("resume_skills", []) if s]
        jd_skills     = [str(s).strip() for s in data.get("jd_skills", []) if s]

    except Exception as e:
        st.error(f"⚠️ JSON parsing failed: {e}")
        st.write("Raw output was:", raw_result)
        return [], []

    # Post-process: split compound skills like "RAG & Vector DB"
    # into ["RAG", "Vector DB"] using LLM validation (no hardcoded list)
    resume_skills = expand_compound_skills(resume_skills, model_choice, groq_api_key)
    jd_skills     = expand_compound_skills(jd_skills, model_choice, groq_api_key)

    return resume_skills, jd_skills