import os
import json
import socket
import fitz  # PyMuPDF
import streamlit as st
from langchain_community.llms import Ollama

# ---------------- CONFIG ----------------
OLLAMA_MODEL = "mistral:latest"  # local model name

# ---------------- ENV DETECTION ----------------
def is_local_env():
    """Detect if running locally (for Ollama)."""
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

# ---------------- MAIN EXTRACTION ----------------
@st.cache_data(show_spinner=False)
def extract_skills_cached(resume_text: str, jd_text: str, model_choice: str = OLLAMA_MODEL) -> tuple[list[str], list[str]]:
    """
    Extract skills separately for Resume and JD.
    Returns: (resume_skills_list, jd_skills_list)
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

    raw_result = None

    # Always use local Ollama
    try:
        raw_result = use_ollama(prompt)
    except Exception as e:
        st.error(f"⚠️ Ollama invocation failed: {e}")
        return [], []

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
