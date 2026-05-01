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


# ---------------- POST-PROCESSING ----------------
def post_process_skills(skills: list[str], model_choice: str, api_key: str) -> list[str]:
    """
    Send extracted skills back to LLM to:
    1. Split any remaining compound entries joined by & or /
       ONLY if each part is a valid technical skill.
    2. Keep atomic skills like CI/CD, AI/ML intact.
    3. Deduplicate.
    """
    if not skills:
        return []

    skills_json = json.dumps(skills)

    prompt = f"""
You are a strict JSON generator.

I have a list of technical skills. Some entries may be compound, joined by "&" or "/".
Your job:
1. For each skill, check if it is a compound of TWO valid technical skills joined by "&" or "/".
   If YES → split into two separate skills.
   If NO  → keep as-is.

How to decide:
- "RAG & Vector DB" → "RAG" is a valid skill, "Vector DB" is a valid skill → SPLIT → ["RAG", "Vector DB"]
- "AWS & GCP"       → both valid → SPLIT → ["AWS", "GCP"]
- "CI/CD"           → this is ONE atomic skill, not two → KEEP → ["CI/CD"]
- "AI/ML"           → "AI" and "ML" are both valid → SPLIT → ["AI", "ML"]
- "LangChain"       → single skill → KEEP → ["LangChain"]
- "REST API"        → single skill → KEEP → ["REST API"]
- "GitHub Actions"  → single skill → KEEP → ["GitHub Actions"]

After splitting, remove duplicates (case-insensitive).

Input skills list:
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

    prompt = f"""
You are a strict JSON generator for technical skill extraction.

EXTRACTION RULES:
- Scan EVERY section: Summary, Core Skills, Projects, Experience, Tools.
- For bullet lines like "RAG & Vector DB · FAISS, Sentence Transformers, Pinecone":
    Step 1 - Take the category part before "·": "RAG & Vector DB"
    Step 2 - Take every skill after "·": FAISS, Sentence Transformers, Pinecone
    Step 3 - Add ALL of them to the skills list including "RAG & Vector DB" as-is.
    Do NOT split "RAG & Vector DB" here — add it exactly as written.

NORMALIZATION:
- "RAG systems"               → "RAG"
- "LLM Integrations"          → "LLMs"
- "Docker-based deployments"  → "Docker"
- "FAISS (vector)"            → "FAISS"
- "REST APIs"                 → "REST API"

IMPORTANT — do NOT split compound entries here:
- "RAG & Vector DB" → add as "RAG & Vector DB" (splitting happens later)
- "CI/CD"           → add as "CI/CD"
- "AWS/GCP"         → add as "AWS/GCP"

EXCLUSIONS:
- No soft skills, no job titles, no company names, no city names,
  no years of experience, no university names.

Remove duplicates (case-insensitive).

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

        resume_skills = [str(s).strip() for s in data.get("resume_skills", []) if s]
        jd_skills     = [str(s).strip() for s in data.get("jd_skills", []) if s]

    except Exception as e:
        st.error(f"⚠️ JSON parsing failed: {e}")
        st.write("Raw output was:", raw_result)
        return [], []

    # Post-process: LLM decides which compounds to split
    resume_skills = post_process_skills(resume_skills, model_choice, groq_api_key)
    jd_skills     = post_process_skills(jd_skills, model_choice, groq_api_key)

    return resume_skills, jd_skills