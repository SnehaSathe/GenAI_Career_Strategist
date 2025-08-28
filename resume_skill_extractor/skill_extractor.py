import os
import json
import requests
import fitz  # PyMuPDF
import streamlit as st
from langchain_community.llms import Ollama

# ---------------- CONFIG ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_MODEL = "mistral:latest"

# ---------------- HELPERS ----------------
@st.cache_resource
def get_ollama_client():
    """Initialize Ollama client (cached)."""
    return Ollama(model=OLLAMA_MODEL)


def use_ollama(prompt: str) -> str:
    """Run prompt locally via Ollama."""
    llm = get_ollama_client()
    return llm.invoke(prompt)


def use_groq(prompt: str, model_choice: str) -> str | None:
    """Run prompt via Groq API. Returns None if fails."""
    if not GROQ_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_choice,
        "messages": [
            {"role": "system", "content": "Extract only technical skills from text, no soft skills or extra words."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 256
    }

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=20
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.warning(f"⚠️ Groq unavailable, using Ollama instead. ({e})")
        return None


# ---------------- MAIN EXTRACTION ----------------
@st.cache_data(show_spinner=False)
def extract_skills_cached(resume_text: str, jd_text: str, model_choice: str) -> tuple[list[str], list[str]]:
    """
    Extract skills separately for Resume and JD.
    Always returns: (resume_skills_list, jd_skills_list).
    """

    prompt = f"""
You are a strict JSON generator.
Extract **only technical skills** from the given texts.
Do not include explanations or extra text.
Return JSON only in this exact format:

{{
    "resume_skills": ["Skill1", "Skill2"],
    "jd_skills": ["SkillA", "SkillB"]
}}

Resume:
{resume_text}

Job Description:
{jd_text}
"""

    raw_result = use_groq(prompt, model_choice) or use_ollama(prompt)

    # Defaults
    resume_skills, jd_skills = [], []

    # Parse JSON safely
    try:
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
