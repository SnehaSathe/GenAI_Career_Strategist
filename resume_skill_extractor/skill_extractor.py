import fitz
import requests
import json
import streamlit as st

@st.cache_data
def extract_text_from_pdf(file):
    pdf_document = fitz.open(stream=file.read(),filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text()
    return text.strip()

@st.cache_data(show_spinner=False)
def ollama_generate(model, prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt}
    )
    return response.json().get("response", "")

@st.cache_data(show_spinner=False)
def extract_skills_cached(resume_text, jd_text, model_choice):
    """
    Extract skills separately for Resume and JD.
    Always returns: (resume_skills_list, jd_skills_list)
    """
    prompt = f"""
    Extract only technical skills separately from the given resume and job description.
    Return output in strict JSON format like this:

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

    # Try parsing JSON
    try:
        import json
        data = json.loads(raw_result.strip())
        if isinstance(data, dict):
            resume_skills = data.get("resume_skills", [])
            jd_skills = data.get("jd_skills", [])
    except Exception:
        pass  # keep them as empty lists if parsing fails

    return resume_skills, jd_skills
