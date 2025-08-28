import streamlit as st
import fitz  # PyMuPDF
import requests
import os
from skill_extractor import extract_skills_cached
from resume_parser import extract_text_from_pdf_cached


# ----------------- CONFIG -----------------
st.set_page_config(page_title="🚀 Resume Skill Extractor", layout="wide")


# ----------------- UI -----------------
st.title("🧠 Smart Resume Skill Extractor")

model_choice = st.selectbox(
    "Select Groq Model (used when online)",
    ["llama3-8b-8192", "mixtral-8x7b-32768"]
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Upload Your Resume (PDF)")
    resume_file = st.file_uploader(
        "Drop your resume PDF here or browse files",
        type=["pdf"],
        label_visibility="collapsed",
        key="resume_uploader"  # ✅ Unique key
    )

with col2:
    st.markdown("### 📄 Upload Job Description")

    # Session state flag to show textarea only after button click
    if "show_jd_textarea" not in st.session_state:
        st.session_state.show_jd_textarea = False

   
    jd_text_input = None

   
    jd_text_input = st.text_area(
            "Paste your Job Description below:", 
            height=200,
            key="jd_text_area"  # ✅ Unique key
        )

# ----------------- READ FILES -----------------
resume_text = None
jd_text = None

if resume_file:
    resume_text = extract_text_from_pdf_cached(resume_file)



if jd_text_input and jd_text_input.strip():
    jd_text = jd_text_input.strip()

# ----------------- RUN EXTRACTION -----------------
if st.button("🚀 Extract Skills", type="primary"):
    if not resume_text or not jd_text:
        st.warning("⚠️ Please provide both Resume and Job Description.")
    else:
        with st.spinner("⏳ Extracting skills... Please wait."):
            resume_skills, jd_skills = extract_skills_cached(resume_text, jd_text, model_choice)


        st.success("✅ Skills extracted successfully!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📄 Resume Skills")
            st.write(", ".join(resume_skills) if resume_skills else "No skills found.")

        with col2:
            st.subheader("📄 Job Description Skills")
            st.write(", ".join(jd_skills) if jd_skills else "No skills found.")

# ----------------- RESET -----------------
if st.button("♻️ Reset / Clear Data"):
    st.cache_data.clear()
    st.session_state.clear()
    st.experimental_rerun()
