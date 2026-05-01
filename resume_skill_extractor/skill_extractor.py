import os
import re
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
def use_groq(prompt: str, model_choice: str, api_key: str):
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


# ---------------- PREPROCESSING ----------------
def preprocess_resume_text(text: str) -> str:
    """
    Clean resume text so the LLM can parse structured bullet sections.

    Handles patterns like:
      "RAG & Vector DB · FAISS, Sentence Transformers, Pinecone"
    Converts the · separator to a colon so the LLM sees:
      "RAG & Vector DB: FAISS, Sentence Transformers, Pinecone"
    """
    # Replace bullet-point dot separator (·) used in Core Skills sections
    text = text.replace("·", ":")
    text = text.replace("•", "\n")   # turn bullet glyphs into newlines

    # Collapse excess blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalize unicode dashes/hyphens
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    return text.strip()


def preprocess_jd_text(text: str) -> str:
    """Light cleanup for job description text."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text.strip()


# ---------------- PROMPT BUILDER ----------------
def build_extraction_prompt(source_label: str, text: str) -> str:
    """
    Build a prompt that extracts atomic, normalized skills from a single source.

    Separate prompts per source prevent the LLM from cross-contaminating
    phrasing between resume and JD (which caused RAG ≠ RAG systems).
    """
    return f"""
You are a strict JSON generator for technical skill extraction.

TASK: Extract every individual technical skill mentioned ANYWHERE in the text below.
This includes: Summary, Core Skills section, Project descriptions, Experience bullets,
Tools mentioned inline — every part of the document.

SPLITTING RULES (very important):
- Lines formatted as "Category · skill1, skill2, skill3" → extract the category concepts
  AND every item after the colon/dot as separate skills.
  Example: "RAG & Vector DB: FAISS, Sentence Transformers, Pinecone"
           → ["RAG", "Vector DB", "FAISS", "Sentence Transformers", "Pinecone"]
- Split compound entries joined by "&" or "/":
  "AWS/GCP" → ["AWS", "GCP"]
  "AI/ML Model Support" → ["AI", "ML"]
  "RAG & Vector DB" → ["RAG", "Vector DB"]
- Keep multi-word skills that belong together as one:
  "GitHub Actions", "CI/CD", "System Design", "Async Programming",
  "Prompt Engineering", "LangChain", "LangGraph", "Top-K Retrieval",
  "Cosine Similarity", "Sentence Transformers", "Ollama Embeddings"

NORMALIZATION RULES:
- "RAG systems" → "RAG"
- "RAG & Vector DB" → ["RAG", "Vector DB"]
- "LLM Integrations" → "LLMs"
- "Docker-based deployments" → "Docker"
- "FAISS (vector)" → "FAISS"
- "FAISS / Vector DB" → ["FAISS", "Vector DB"]
- "REST APIs" → "REST API"
- "OpenAI/GPT" → ["OpenAI", "GPT"]
- "Groq LLMs" → ["Groq", "LLMs"]
- "Azure (AI Services, Cloud Fundamentals)" → ["Azure", "Azure AI Services"]
- "GitHub Copilot" → "GitHub Copilot"
- "n8n Automation" → "n8n"
- "POC Development" → "POC Development"

EXCLUSION RULES:
- Do NOT include: soft skills, years of experience, job titles, company names,
  university names, degree names, city/country names, personal pronouns.

DEDUPLICATION:
- Remove duplicates (case-insensitive). Keep the cleaner/shorter form.

OUTPUT FORMAT — return ONLY this JSON, no explanation, no markdown fences:

{{
    "{source_label}_skills": ["Skill1", "Skill2", "Skill3"]
}}

Text ({source_label}):
{text}
"""


# ---------------- JSON PARSER ----------------
def parse_skills_from_response(raw: str | None, key: str) -> list[str]:
    """
    Safely parse a list of skills from an LLM JSON response.
    Handles markdown fences and partial JSON gracefully.
    """
    if not raw:
        return []

    cleaned = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    # Attempt 1: direct JSON parse
    try:
        data = json.loads(cleaned)
        return [str(s).strip() for s in data.get(key, []) if s]
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract first {...} block (handles leading/trailing prose)
    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}") + 1
        data = json.loads(cleaned[start:end])
        return [str(s).strip() for s in data.get(key, []) if s]
    except Exception:
        pass

    return []


# ---------------- MAIN EXTRACTION ----------------
@st.cache_data(show_spinner=False)
def extract_skills_cached(
    resume_text: str,
    jd_text: str,
    model_choice: str,
    groq_api_key: str | None = None
) -> tuple[list[str], list[str]]:
    """
    Extract and normalize skills separately for Resume and JD.

    Why separate prompts?
    → If both texts go into one prompt, the LLM phrases the same concept
      differently per source (e.g. "RAG systems" vs "RAG & Vector DB"),
      which breaks downstream matching.

    Returns: (resume_skills, jd_skills)
    """

    # --- Resume ---
    cleaned_resume = preprocess_resume_text(resume_text)
    resume_prompt = build_extraction_prompt("resume", cleaned_resume)
    raw_resume = run_llm(resume_prompt, model_choice, groq_api_key)

    if raw_resume is None:
        st.error("⚠️ LLM call failed for resume skill extraction.")
        resume_skills = []
    else:
        resume_skills = parse_skills_from_response(raw_resume, "resume_skills")

    # --- Job Description ---
    cleaned_jd = preprocess_jd_text(jd_text)
    jd_prompt = build_extraction_prompt("jd", cleaned_jd)
    raw_jd = run_llm(jd_prompt, model_choice, groq_api_key)

    if raw_jd is None:
        st.error("⚠️ LLM call failed for JD skill extraction.")
        jd_skills = []
    else:
        jd_skills = parse_skills_from_response(raw_jd, "jd_skills")

    return resume_skills, jd_skills


# ---------------- SKILL MATCHING ----------------
def normalize(skill: str) -> str:
    """Lowercase, strip, collapse internal spaces."""
    return re.sub(r'\s+', ' ', skill.lower().strip())


def match_skills(
    resume_skills: list[str],
    jd_skills: list[str]
) -> tuple[list[str], list[str]]:
    """
    Match JD skills against resume skills.

    Strategy (in priority order):
    1. Exact normalized match         → "rag" == "rag"
    2. Substring containment          → "rag" in "rag pipeline" or vice versa
    3. Token overlap (≥1 shared word) → "vector db" shares "db" with "faiss vector db"

    Returns: (matched_jd_skills, missing_jd_skills)
    """
    resume_normalized = [normalize(s) for s in resume_skills]

    matched = []
    missing = []

    for jd_skill in jd_skills:
        jd_norm = normalize(jd_skill)
        jd_tokens = set(jd_norm.split())

        found = False
        for r_norm in resume_normalized:
            r_tokens = set(r_norm.split())

            # Rule 1: exact match
            if jd_norm == r_norm:
                found = True
                break

            # Rule 2: substring containment (handles "RAG" ↔ "RAG systems")
            if jd_norm in r_norm or r_norm in jd_norm:
                found = True
                break

            # Rule 3: meaningful token overlap
            # Ignore single-letter tokens and very generic words
            ignore = {"and", "or", "the", "a", "an", "with", "for", "of", "in"}
            meaningful_jd = jd_tokens - ignore
            meaningful_r = r_tokens - ignore
            if meaningful_jd and meaningful_r and meaningful_jd & meaningful_r:
                found = True
                break

        if found:
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)

    return matched, missing


# ---------------- SCORE ----------------
def compute_match_score(matched: list[str], jd_skills: list[str]) -> float:
    """Return match percentage rounded to 1 decimal."""
    if not jd_skills:
        return 0.0
    return round(len(matched) / len(jd_skills) * 100, 1)