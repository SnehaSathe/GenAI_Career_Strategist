import streamlit as st
import fitz  # PyMuPDF
import os
import streamlit as st
from langchain_groq import ChatGroq
from fpdf import FPDF
import sys 
from io import BytesIO
import base64
from PIL import Image

# Ensure parent directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Try relative import first (works for package/module run)
try:
    from resume_skill_extractor.resume_parser import extract_text_from_pdf_cached, extract_text_from_docx_cached,extract_text
    from resume_skill_extractor.skill_extractor import extract_skills_cached
except ImportError:
    # Fallback to absolute import (works for script run)
    from resume_skill_extractor.resume_parser import extract_text_from_pdf_cached , extract_text_from_docx_cached,extract_text
    from resume_skill_extractor.skill_extractor import extract_skills_cached
    

def get_groq_key():
    return st.secrets.get("GROQ_API_KEY", None)


# ----------------- HEADER -----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(BASE_DIR, "logo.png")  # logo in main folder

# --- Read image in binary and encode as base64 ---
with open(image_path, "rb") as f:
    data = f.read()






# ----------------- INPUT SECTION -----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Resume (PDF/DOCX)")
    resume_file = st.file_uploader(
        "Drop your resume PDF/DOCX here or browse files",
        type=["pdf", "docx"],
        label_visibility="collapsed",
        key="resume_uploader"
    )

with col2:
    st.subheader("📝 Job Description")
    jd_text_input = st.text_area(
        "Paste your Job Description below:",
        height=200,
        key="jd_uploader_main"
    )

# ----------------- READ FILES -----------------
resume_text = None
jd_text = None

if resume_file:
    if resume_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf_cached(resume_file.read())
    elif resume_file.name.endswith(".docx"):
        resume_text = extract_text_from_docx_cached(resume_file)


if jd_text_input and jd_text_input.strip():
    jd_text = jd_text_input.strip()


