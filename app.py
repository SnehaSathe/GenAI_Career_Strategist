import streamlit as st
import fitz  # PyMuPDF
import os
import streamlit as st
from resume_skill_extractor.resume_parser import extract_candidate_name 
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from fpdf import FPDF
import sys 
from jd_skill_gap_analyzer.helper import embed_skills,find_matches,generate_report
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
    from .resume_parser import extract_text_from_pdf_cached, extract_text_from_docx_cached,extract_text
    from .skill_extractor import extract_skills_cached
except ImportError:
    # Fallback to absolute import (works for script run)
    from resume_skill_extractor.resume_parser import extract_text_from_pdf_cached , extract_text_from_docx_cached,extract_text
    from resume_skill_extractor.skill_extractor import extract_skills_cached
    

# ----------------- CONFIG -----------------
st.set_page_config(page_title="🧠 Smart Resume Skill Extractor", page_icon="🧠", layout="wide")

# ----------------- HEADER -----------------

# Read image file in binary
with open("logo.png", "rb") as f:
    data = f.read()
encoded = base64.b64encode(data).decode()  # encode as base64 string

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


# Get API key safely
groq_api_key = st.secrets["GROQ_API_KEY"]

# --- Initialize LLM ---
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant"  # or whatever model you prefer
)

# --- Initialize LLM safely (llm will be None if we can't init) ---

if groq_api_key:
    try:
        # Example: try to initialize a Groq/LangChain LLM client if you have the lib.
        # Replace/import with whatever client you use in your environment.
        # If you don't have a client, keep llm = None and the regex will be used.
        from langchain_groq import ChatGroq  # adjust to actual package you use (may be different)
        llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.1-8b-instant")
        
    except Exception as e:
        # If langchain_groq is not installed or initialization fails, llm stays None
        st.warning(f"LLM init failed or client not installed: {e}. Falling back to regex extractor.")
        llm = None
else:
    st.info("No GROQ_API_KEY found — using regex fallback for name extraction.")

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

model_choice = st.selectbox(
        "🔍 Select Groq Model (used when online)",
        ["llama-3.1-8b-instant","mixtral-8x7b-32768"],
    index=0,  key="groq_model_select"
    )
    

    

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




import streamlit as st
from io import BytesIO

# --- Initialize analyzed flag ---
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# --- Button row ---
col1, col2, col3 = st.columns([2, 2, 2])

# ----------------- ANALYZE BUTTON -----------------
with col1:
    analyze_clicked = st.button("🚀 Analyze Skills", use_container_width=True)

# ----------------- RESET BUTTON -----------------
with col2:
    reset_clicked = st.button("♻ Reset / Clear Data", use_container_width=True)

# ----------------- DOWNLOAD BUTTON PLACEHOLDER -----------------
with col3:
    download_placeholder = st.empty()  # placeholder to render download button later

# ================== BUTTON ACTIONS ==================

# ----- ANALYZE SKILLS -----
if analyze_clicked:
    if not resume_text or not jd_text:
        st.warning("⚠ Please provide both Resume and Job Description.")
    else:
        with st.spinner("⏳ Analyzing skills... Please wait."):
            resume_skills, jd_skills = extract_skills_cached(
                resume_text, jd_text,
                model_choice=model_choice,
                groq_api_key=groq_api_key
            )
            resume_vecs, jd_vecs = embed_skills(resume_skills, jd_skills)
            matches, missing, additional, score = find_matches(resume_skills, jd_skills, resume_vecs, jd_vecs)
        
            candidate_name = st.session_state.get("candidate_name") or extract_candidate_name(resume_text, llm=llm)

        # Save results in session_state
        st.session_state["resume_skills"] = resume_skills
        st.session_state["jd_skills"] = jd_skills
        st.session_state["matches"] = matches
        st.session_state["missing"] = missing
        st.session_state["additional"] = additional
        st.session_state["score"] = score
        st.session_state.analyzed = True  # ✅ enables download button

        st.success(f"📊 Analysis for **{candidate_name}**")

# ----- RESET / CLEAR -----
if reset_clicked:
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()

# ----- DISPLAY RESULTS IMMEDIATELY -----
if st.session_state.analyzed:
    col1_res, col2_res, col3_res = st.columns(3)
    
    # ✅ Matched Skills
    with col1_res:
        if st.session_state["matches"]:
            st.success(f"✅ {len(st.session_state['matches'])} Matched Skills")
            for jd_skill, resume_skill, sim in st.session_state["matches"]:
                st.markdown(f"- **{jd_skill}**")
        else:
            st.info("No skills matched.")

    # ❌ Missing Skills
    with col2_res:
        if st.session_state["missing"]:
            st.error(f"❌ {len(st.session_state['missing'])} Missing Skills")
            st.markdown(", ".join([f"`{skill}`" for skill in st.session_state["missing"]]))
        else:
            st.success("No missing skills! 🎉")

    # ➕ Additional Skills
    with col3_res:
        if st.session_state["additional"]:
            st.error(f"➕ {len(st.session_state['additional'])} Additional Skills")
            st.markdown(", ".join([f"`{skill}`" for skill in st.session_state["additional"]]))
        else:
            st.success("No Additional skills! 🎉")

# ----- DOWNLOAD BUTTON -----
if st.session_state.analyzed:
    candidate_name = st.session_state.get("candidate_name") or extract_candidate_name(resume_text, llm=llm)
    pdf = generate_report(
        candidate_name=candidate_name,
        matched_skills=st.session_state["matches"],
        missing_skills=st.session_state["missing"],
        additional_skills=st.session_state["additional"],
        jd_skills=st.session_state["jd_skills"],
        resume_skills=st.session_state["resume_skills"],
        score=st.session_state["score"],
        logo_path="logo.png",
    )
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)

    # Render the download button in the placeholder
    download_placeholder.download_button(
        "📥 Download Report",
        data=buffer,
        file_name=f"{candidate_name}_resume_skill_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
else:
    # Disabled button until analysis
    download_placeholder.button("📥 Download Report", disabled=True, use_container_width=True)
