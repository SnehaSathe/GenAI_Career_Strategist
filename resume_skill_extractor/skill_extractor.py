import os
import json
import requests
import streamlit as st
import fitz  # PyMuPDF
from langchain_community.llms import Ollama

# ---------------- CONFIG ----------------
OLLAMA_MODEL = "mistral:latest"

groq_api_key = (
    os.getenv("GROQ_API_KEY")
    or st.secrets.get("GROQ_API_KEY", None)
)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("🧠 AI Resume Analyzer (ATS + Skill Intelligence)")

# ---------------- FILE INPUT ----------------
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description")

# ---------------- EXTRACT TEXT FROM PDF ----------------
resume_text = ""
if resume_file:
    doc = fitz.open(stream=resume_file.read(), filetype="pdf")
    for page in doc:
        resume_text += page.get_text()

# ---------------- LLM PROMPT (CORE INTELLIGENCE) ----------------
prompt = f"""
You are an expert ATS resume analyzer and career coach.

TASK:
Analyze resume and job description and return ONLY valid JSON.

IMPORTANT RULES:
- Extract ONLY technical skills (not soft skills unless explicitly technical like "System Design")
- Split combined skill strings like:
  "Python & Backend · Python (primary), FastAPI, Flask"
  INTO:
  ["Python", "Backend", "FastAPI", "Flask"]

- Normalize skills (e.g. "PyTorch" not "pytorch framework")
- Remove duplicates

- Detect:
  1. match_level (0 to 100 float)
  2. strengths (bullet list)
  3. gaps (missing skills from JD)
  4. improvements (actionable suggestions)
  5. resume_skills (clean list)
  6. jd_skills (clean list)

OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "match_level": 0,
  "strengths": [],
  "gaps": [],
  "improvements": [],
  "resume_skills": [],
  "jd_skills": []
}}

---------------- RESUME ----------------
{resume_text}

---------------- JOB DESCRIPTION ----------------
{jd_text}
"""

# ---------------- LLM CALL ----------------
result = None

if resume_text and jd_text:

    if groq_api_key:
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]

    else:
        llm = Ollama(model=OLLAMA_MODEL)
        result = llm.invoke(prompt)

# ---------------- CLEAN JSON PARSING ----------------
data = {}
if result:
    try:
        cleaned = result.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            cleaned = cleaned.replace("json", "").strip()

        data = json.loads(cleaned)

    except Exception as e:
        st.error("⚠️ Failed to parse LLM output")
        st.write(result)

# ---------------- UI OUTPUT ----------------
if data:

    st.subheader("📊 Match Analysis")

    st.metric("Match Level", f"{data.get('match_level', 0)}%")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💪 Strengths")
        for s in data.get("strengths", []):
            st.write("•", s)

        st.markdown("### 🧠 Resume Skills")
        st.write(", ".join(data.get("resume_skills", [])))

    with col2:
        st.markdown("### ⚠️ Gaps")
        for g in data.get("gaps", []):
            st.write("•", g)

        st.markdown("### 🎯 JD Skills")
        st.write(", ".join(data.get("jd_skills", [])))

    st.markdown("### 🚀 Improvement Suggestions")
    for i in data.get("improvements", []):
        st.write("•", i)

# ---------------- OPTIONAL DEBUG ----------------
else:
    st.info("Upload resume and paste job description to start analysis.")