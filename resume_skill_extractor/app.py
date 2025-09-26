import streamlit as st
import fitz  # PyMuPDF
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from fpdf import FPDF
import sys 
from io import BytesIO
import base64
from PIL import Image
from transformers import pipeline

# Ensure parent directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Try relative import first (works for package/module run)
try:
    from .resume_parser import extract_text_from_pdf_cached, extract_text_from_docx_cached,extract_text
    from .skill_extractor import extract_skills_cached
except ImportError:
    # Fallback to absolute import (works for script run)
    from resume_skill_extractor.resume_parser import extract_text_from_pdf_cached , extract_text_from_docx_cached,extract_text
    from resume_skill_extractor.skill_extractor import extract_skills_cached
    

# ----------------- CONFIG -----------------
st.set_page_config(page_title="🧠 Smart Resume Skill Extractor", page_icon="🧠", layout="wide")

# ----------------- HEADER -----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(BASE_DIR, "logo.png")  # logo in main folder

# --- Read image in binary and encode as base64 ---
with open(image_path, "rb") as f:
    data = f.read()

encoded = base64.b64encode(data).decode()

# Embed in HTML
st.markdown(
    f"""
    <div style="
        display: flex;
        align-items: center;
        padding: 15px;
        background:#f8f9fa;
        border-radius:10px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom:20px;
    ">
        <img src="data:image/png;base64,{encoded}" style='width:100px; height:auto; margin-right:20px;border-radius:10px'/>
        <div style='text-align:left;'>
            <h1 style='margin:0;'>🧠 Smart Resume Skill Extractor</h1>
            <p style='font-size:16px; margin:0;'>Upload your Resume & Job Description to extract skills instantly.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



#### ---------------------- Custom CSS--------------------------
st.markdown(
    """
    <style>
    /* === Background Gradient === */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        font-family: "Segoe UI", sans-serif;
        color: #212529;
    }

    /* === Title === */
    h1 {
        text-align: center;
        font-size: 2.5rem !important;
        background: -webkit-linear-gradient(#212529, #495057);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        
    }

    /* === Subheader === */
    h3 {
        text-align: center;
        font-weight: 400;
        color: #495057;
    }

    /* === Card Elements === */
    .card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 12px solid #dee2e6;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.15);
        color: #212529;
    }
    

    /* === Buttons === */
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 12px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #764ba2, #667eea);
    }

    /* === Tabs === */
    .stTabs [role="tablist"] button {
        background: #fff !important;
        border-radius: 10px !important;
        margin: 3px !important;
        padding: 8px 16px !important;
        font-weight: 500;
        color: #212529 !important;
        box-shadow: 0px 3px 8px rgba(0,0,0,12);
    }

    /* === Text Area & Inputs === */
    .stTextArea, .stTextInput, .stNumberInput, .stSelectbox,.stFileUploader, .stMultiSelect {
        border: 1px solid #6c757d !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
        padding: 5px !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15) !important;
    }

    textarea {
        border: none !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        background-color: #f8f9fa !important;
        color: #212529 !important;
        resize: none !important;
    }
    textarea:focus {
        outline: none !important;
        box-shadow: 0px 0px 10px rgba(102, 126, 234, 0.6) !important;
    }

    /* === File Uploader === */
    .stFileUploader > div {
        border: 2px dashed #6c757d !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
        padding: 16px !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.12) !important;
    }

    /* === Labels === */
    label {
        font-weight: 600 !important;
        color: #212529 !important;
        font-size: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Hide default collapse icon */
    button[title="Collapse"]::before {
        content: '\\2699';  /* Unicode for gear ⚙ */
        font-size: 20px;
    }

    /* Optional: increase button size */
    button[title="Collapse"] {
        width: 40px;
        height: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# ----------------- MODEL CHOICE -----------------

# --- Load Hugging Face Model ---
@st.cache_resource
def load_model():
    return pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

ner = load_model()

model_choice = st.selectbox(
    "🔍 Select Model",
    ["dslim/bert-base-NER"]
)
# ----------------- CHECK IF MODEL IS INSTALLED -----------------
@st.cache_resource
def check_ollama_model_installed(model_name):
    try:
        client = Ollama(model=model_name)
        # Test a simple prompt to check if model is available
        client.invoke("Hello")
        return True
    except Exception as e:
        st.warning(f"⚠ Model '{model_name}' is not installed locally or failed to load.\nError: {e}")
        return False

model_available = check_ollama_model_installed(model_choice)

if not model_available:
    st.info(f"Please install the model '{model_choice}' locally via Ollama CLI before using it.")


    

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




# ----------------- RUN EXTRACTION -----------------
if st.button("🚀 Extract Skills", type="primary", use_container_width=True):
    if not resume_text or not jd_text:
        st.warning("⚠ Please provide both Resume and Job Description.")
    else:
        with st.spinner("⏳ Extracting skills... Please wait."):
            resume_skills, jd_skills = extract_skills_cached(resume_text, jd_text,  model_choice)


        st.success("✅ Skills extracted successfully!")

        # ----------------- RESULTS -----------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Resume Skills")
            if resume_skills:
                st.markdown(" ".join([
                    f"<span style='background:#d1e7dd; padding:6px 12px; border-radius:15px; margin:3px; display:inline-block;'>{skill}</span>"
                    for skill in resume_skills
                ]), unsafe_allow_html=True)
            else:
                st.write("No skills found.")

        with col2:
            st.subheader("📌 Job Description Skills")
            if jd_skills:
                st.markdown(" ".join([
                    f"<span style='background:#ffe5b4; padding:6px 12px; border-radius:15px; margin:3px; display:inline-block;'>{skill}</span>"
                    for skill in jd_skills
                ]), unsafe_allow_html=True)
            else:
                st.write("No skills found.")



# ----------------- RESET -----------------
if st.button("♻ Reset / Clear Data", use_container_width=True):
    st.cache_data.clear()
    st.session_state.clear()
    st.experimental_rerun()