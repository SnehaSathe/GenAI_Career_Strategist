import os
import json
import socket
import requests
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


# ---------------- SINGLE SOURCE EXTRACTION ----------------
def build_extraction_prompt(source_label: str, text: str) -> str:
    """
    Build a prompt that extracts atomic, normalized skills from a single source.
    Keeping sources separate prevents the LLM from cross-contaminating phrasing.
    """
    return f"""
You are a strict JSON generator for skill extraction.

Rules:
- Extract ONLY individual technical skills from the text below.
- Split compound entries: "RAG & Vector DB" → ["RAG", "Vector DB"]
- Normalize to their most common short form:
    "RAG systems" → "RAG"
    "LLM Integrations" → "LLMs"
    "Docker-based deployments" → "Docker"
    "FAISS (vector)" → "FAISS"
    "FAISS / Vector DB" → ["FAISS", "Vector DB"]
    "REST APIs" → "REST API"
- Remove duplicates.
- Do NOT include soft skills, years of experience, or job titles.
- Return ONLY this JSON, no extra text:

{{
    "{source_label}_skills": ["Skill1", "Skill2", "Skill3"]
}}

Text ({source_label}):
{text}
"""


def run_llm(prompt: str, model_choice: str, api_key: str) -> str | None:
    if is_local_env():
        try:
            return use_ollama(prompt)
        except Exception:
            return use_groq(prompt, model_choice, api_key)
    else:
        return use_groq(prompt, model_choice, api_key)


def parse_skills_from_response(raw: str, key: str) -> list[str]:
    """Safely parse a list of skills from an LLM JSON response."""
    if not raw:
        return []

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Strip ```json ... ``` fences
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(cleaned)
        return [str(s).strip() for s in data.get(key, []) if s]
    except json.JSONDecodeError:
        # Fallback: try extracting the JSON object with braces
        try:
            start = cleaned.index("{")
            end = cleaned.rindex("}") + 1
            data = json.loads(cleaned[start:end])
            return [str(s).strip() for s in data.get(key, []) if s]
        except Exception:
            return []


# ---------------- MAIN EXTRACTION ----------------
@st.cache_data(show_spinner=False)
def extract_skills_cached(
    resume_text: str,
    jd_text: str,
    model_choice: str,
    groq_api_key=None
) -> tuple[list[str], list[str]]:
    """
    Extract and normalize skills separately for Resume and JD.
    Separate prompts prevent cross-source phrasing inconsistency.
    Returns: (resume_skills_list, jd_skills_list)
    """

    # --- Extract resume skills ---
    resume_prompt = build_extraction_prompt("resume", resume_text)
    raw_resume = run_llm(resume_prompt, model_choice, groq_api_key)
    if raw_resume is None:
        st.error("⚠️ LLM call failed for resume extraction.")
    resume_skills = parse_skills_from_response(raw_resume, "resume_skills")

    # --- Extract JD skills ---
    jd_prompt = build_extraction_prompt("jd", jd_text)
    raw_jd = run_llm(jd_prompt, model_choice, groq_api_key)
    if raw_jd is None:
        st.error("⚠️ LLM call failed for JD extraction.")
    jd_skills = parse_skills_from_response(raw_jd, "jd_skills")

    return resume_skills, jd_skills


# ---------------- SKILL MATCHING ----------------
def normalize(skill: str) -> str:
    """Lowercase and strip for fuzzy-safe comparison."""
    return skill.lower().strip()


def match_skills(
    resume_skills: list[str],
    jd_skills: list[str]
) -> tuple[list[str], list[str]]:
    """
    Match JD skills against resume skills using normalized comparison.
    Returns: (matched, missing)
    """
    resume_normalized = {normalize(s): s for s in resume_skills}

    matched = []
    missing = []

    for jd_skill in jd_skills:
        jd_norm = normalize(jd_skill)

        # Check for exact normalized match OR substring containment
        # e.g. "rag" in "rag & vector db" → match
        found = any(
            jd_norm == r_norm or jd_norm in r_norm or r_norm in jd_norm
            for r_norm in resume_normalized
        )

        if found:
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)

    return matched, missing