import fitz
import requests
import json
import streamlit as st
import os 

# ---------------- CONFIG ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_MODEL = "mistral:latest"

# ---------------- HELPERS ----------------
@st.cache_data(show_spinner=False)
def ollama_generate(model, prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt}
    )
    return response.json().get("response", "")

@st.cache_resource
def get_ollama_client():
    from langchain_community.llms import Ollama
    return Ollama(model=OLLAMA_MODEL)

def use_ollama(prompt):
    """Run locally via Ollama."""
    llm = get_ollama_client()
    return llm.invoke(prompt)

def use_groq(prompt, model_choice):
    """Call Groq API if available."""
    try:
        if not GROQ_API_KEY:
            raise ValueError("No Groq API key found")

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
def extract_skills_cached(resume_text, jd_text, model_choice):
    """
    Extract skills separately for Resume and JD.
    Always returns: (resume_skills_list, jd_skills_list)
    Ensures both outputs are lists of strings.
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

    raw_result = use_groq(prompt, model_choice)
    if not raw_result:
        raw_result = use_ollama(prompt)

    # Default empty lists
    resume_skills, jd_skills = [], []

    # Parse JSON safely
    try:
        cleaned = raw_result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            cleaned = cleaned.replace("json", "", 1).strip()

        data = json.loads(cleaned)
        if isinstance(data, dict):
            resume_skills = data.get("resume_skills", [])
            jd_skills = data.get("jd_skills", [])

        # Ensure both are lists of strings
        if not isinstance(resume_skills, list):
            resume_skills = [str(resume_skills)] if resume_skills else []
        else:
            resume_skills = [str(s).strip() for s in resume_skills if s]

        if not isinstance(jd_skills, list):
            jd_skills = [str(jd_skills)] if jd_skills else []
        else:
            jd_skills = [str(s).strip() for s in jd_skills if s]

    except Exception as e:
        st.error(f"⚠️ JSON parsing failed: {e}")
        st.write("Raw output was:", raw_result)

    return resume_skills, jd_skills
