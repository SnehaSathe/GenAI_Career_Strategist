import os
import json
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

def is_localhost():
    """Detect if running locally (for Ollama fallback)."""
    try:
        host = socket.gethostname()
        ip = socket.gethostbyname(host)
        return ip.startswith("127.") or ip == "localhost"
    except:
        return False


@st.cache_resource
def get_ollama_client():
    """Initialize Ollama client (cached)."""
    return Ollama(model=OLLAMA_MODEL)


def use_ollama(prompt: str) -> str:
    """Run prompt locally via Ollama."""
    llm = get_ollama_client()
    return llm.invoke(prompt)


def use_groq(prompt, model_choice, api_key):
    import requests

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
def extract_skills_cached(resume_text: str, jd_text: str,groq_api_key, model_choice: str) -> tuple[list[str], list[str]]:
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

    raw_result = use_groq(prompt, model_choice, groq_api_key) or use_ollama(prompt)

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
