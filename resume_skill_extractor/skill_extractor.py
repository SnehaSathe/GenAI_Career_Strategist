import os
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

    # Debugging message (optional)
    st.error(f"Groq API error {response.status_code}: {response.text}")
    return None


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
    Always returns: (resume_skills_list, jd_skills_list).
    """

    prompt = f"""
You are an advanced AI Resume & Job Description Skill Extractor.

Your task:
Extract ONLY technical skills from Resume and Job Description separately.

⚠️ IMPORTANT RULES:
- Return ONLY valid JSON. No explanation, no markdown, no extra text.
- Do NOT hallucinate skills.
- Extract real skills only if explicitly or clearly implied.
- Normalize skills (e.g., "python", "Python" → "Python").

📌 CRITICAL SKILL EXTRACTION RULES:
1. Extract skills from:
   - Bullet points (•, -, *)
   - Dot separators (·)
   - Comma-separated lists
   - Brackets and grouped skills
   - Phrases like "Python & Backend · Python (primary), FastAPI, Flask"
   - Sections like Skills, Technologies, Tools, Tech Stack

2. IMPORTANT: SPLIT COMPOUND SKILL LINES INTO ATOMIC SKILLS
   Example:
   "Python & Backend · Python (primary), FastAPI, Flask, REST APIs"
   → ["Python", "FastAPI", "Flask", "REST APIs", "Backend"]

3. Remove duplicates and irrelevant words like:
   - primary, basic, advanced, familiar with, experience in

4. Keep ONLY technical skills:
   (Programming languages, frameworks, tools, libraries, databases, cloud, ML, AI tools)

🚨 OUTPUT FORMAT (STRICT JSON ONLY):
{{
    "resume_skills": ["Skill1", "Skill2", "Skill3"],
    "jd_skills": ["SkillA", "SkillB", "SkillC"]
}}

📄 RESUME TEXT:
{resume_text}

📄 JOB DESCRIPTION:
{jd_text}
"""

    raw_result = None

    if is_local_env():
        # Local → prefer Ollama, fallback to Groq
        try:
            raw_result = use_ollama(prompt)
        except Exception:
            raw_result = use_groq(prompt, model_choice, groq_api_key)
    else:
        # Cloud → always Groq
        raw_result = use_groq(prompt, model_choice, groq_api_key)

    # Defaults
    resume_skills, jd_skills = [], []

    # Parse JSON safely
    try:
        if not raw_result:
            return [], []

        cleaned = raw_result.strip()

        # Remove ```json code blocks if present
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            cleaned = cleaned.replace("json", "", 1).strip()

        data = json.loads(cleaned)

        # Extract lists safely
        resume_skills = [str(s).strip() for s in data.get("resume_skills", []) if s]
        jd_skills = [str(s).strip() for s in data.get("jd_skills", []) if s]

    except Exception as e:
        st.error(f"⚠️ JSON parsing failed: {e}")
        st.write("Raw output was:", raw_result)

    return resume_skills, jd_skills


