import os
import sys
import json
import base64
from io import BytesIO
import streamlit as st
from fpdf import FPDF
import logging

logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from .resume_parser import extract_text_from_pdf_cached, extract_text_from_docx_cached
    from .skill_extractor import extract_skills_cached
except ImportError:
    from resume_skill_extractor.resume_parser import extract_text_from_pdf_cached, extract_text_from_docx_cached
    from resume_skill_extractor.skill_extractor import extract_skills_cached

from resume_skill_extractor.resume_parser import extract_candidate_name
from jd_skill_gap_analyzer.helper import embed_skills, find_matches, generate_report
from ai_role_recommender.inference_engine import AIRoleRecommender

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# ------------------ LOGO ------------------
with open("logo.png", "rb") as f:
    logo_encoded = base64.b64encode(f.read()).decode()

st.markdown(f"""
<div style="display:flex;align-items:center;padding:15px;background:#f8f9fa;border-radius:10px;margin-bottom:20px;">
<img src="data:image/png;base64,{logo_encoded}" width="80">
<h2 style="margin-left:15px;">AI Resume Analyzer</h2>
</div>
""", unsafe_allow_html=True)

# ------------------ LLM INIT ------------------
llm = None
groq_api_key = st.secrets.get("GROQ_API_KEY")

if groq_api_key:
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.1-8b-instant")
    except:
        pass
model_choice="llama-3.1-8b-instant"  # default Groq model
# ------------------ FUNCTIONS ------------------

def get_match_label(score):
    if score >= 80:
        return "✅ Strong Match"
    elif score >= 60:
        return "⚠️ Moderate Match"
    else:
        return "❌ Low Match"

def generate_llm_explanation(score, resume_skills, jd_skills, matched_skills, missing_skills):
    if not llm:
        return "Basic analysis only (LLM not available)."

    prompt = f"""
    You are an AI resume analysis assistant.

    Match Score: {score}%

    Resume Skills: {resume_skills}
    Job Description Skills: {jd_skills}
    Matched Skills: {[m[0] for m in matched_skills]}
    Missing Skills: {missing_skills}

    Instructions:
    - Explain the match level
    - Highlight strengths
    - Explain gaps
    - Give improvement suggestions
    - Keep it short (max 120 words)
    """

    try:
        return llm.invoke(prompt).content
    except:
        return "Error generating explanation."

# ------------------ INPUT ------------------

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

with col2:
    jd_text = st.text_area("Paste Job Description")

# ------------------ BUTTON ------------------

if st.button("Analyze"):

    if not resume_file or not jd_text:
        st.warning("Upload resume and JD")
    else:

        if resume_file.name.endswith(".pdf"):
            resume_text = extract_text_from_pdf_cached(resume_file.read())
        else:
            resume_text = extract_text_from_docx_cached(resume_file)

        with st.spinner("Analyzing..."):

            resume_skills, jd_skills = extract_skills_cached(
    resume_text,
    jd_text,
    model_choice,
    groq_api_key
)
            resume_vecs, jd_vecs = embed_skills(resume_skills, jd_skills)
            matches, missing, additional, score = find_matches(
    resume_skills, jd_skills, resume_vecs, jd_vecs
)

            score = round((len(matches)/len(jd_skills))*100,2) if jd_skills else 0
            label = get_match_label(score)

            explanation = generate_llm_explanation(
                score, resume_skills, jd_skills, matches, missing
            )
            candidate_name = extract_candidate_name(
    resume_text
)

            if not candidate_name or candidate_name.strip() == "":
                candidate_name = "Candidate"


            # ------------------ DISPLAY ------------------

            st.success(f"Analysis for {candidate_name}")

            st.markdown(f"## Match Score: {score}% {label}")

            colA, colB, colC = st.columns(3)

            with colA:
                st.subheader("Matched Skills")
                st.write([m[0] for m in matches])

            with colB:
                st.subheader("Missing Skills")
                st.write(missing)

            with colC:
                st.subheader("Additional Skills")
                st.write(additional)

            # -------- AI Explanation --------
            st.subheader("AI Explanation")
            st.info(explanation)

            # ------------------ PDF ------------------

            pdf_buffer = BytesIO()

            pdf = generate_report(
                candidate_name=candidate_name,
                matched_skills=matches,
                missing_skills=missing,
                additional_skills=additional,
                jd_skills=jd_skills,
                resume_skills=resume_skills,
                score=score,
                logo_path="logo.png",
                explanation=explanation   # IMPORTANT
            )

            pdf.output(pdf_buffer)
            pdf_buffer.seek(0)

            st.download_button(
                "Download Report",
                data=pdf_buffer,
                file_name="resume_report.pdf",
                mime="application/pdf"
            )