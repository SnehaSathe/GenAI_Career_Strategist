import streamlit as st
import fitz  # PyMuPDF
import requests
import os

# ----------------- CONFIG -----------------
st.set_page_config(page_title="🚀 Resume Skill Extractor", layout="wide")

# API Keys / Models
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_MODEL = "llama3"

# ----------------- HELPERS -----------------
@st.cache_data
def read_pdf(uploaded_file):
    """Read PDF and return text."""
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text.strip()


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


@st.cache_resource
def get_ollama_client():
    from langchain_community.llms import Ollama
    return Ollama(model=OLLAMA_MODEL)


def use_ollama(prompt):
    """Run locally via Ollama."""
    llm = get_ollama_client()
    return llm.invoke(prompt)


@st.cache_data(show_spinner=False)
def extract_skills_cached(text, model_choice):
    """Extract skills for a single text (resume or JD)."""
    prompt = f"""
    Extract only technical skills from the following text.
    Output as a comma-separated list, without generic terms.

    Text:
    {text}
    """
    skills = use_groq(prompt, model_choice)
    if skills:
        return [s.strip() for s in skills.split(",") if s.strip()]
    return [s.strip() for s in use_ollama(prompt).split(",") if s.strip()]

# ----------------- UI -----------------
st.title("🧠 Smart Resume Skill Extractor (Auto Groq → Ollama)")

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

    # Upload option
    jd_file = st.file_uploader(
        "Upload Job Description (PDF only)",
        type=["pdf"],
        key="jd_uploader"  # ✅ Unique key
    )

    # Paste option button
    if st.button("✍️ Paste Job Description", key="paste_button"):
        st.session_state.show_jd_textarea = True

    jd_text_input = None
    if st.session_state.show_jd_textarea:
        jd_text_input = st.text_area(
            "Paste your Job Description below:", 
            height=200,
            key="jd_text_area"  # ✅ Unique key
        )

# ----------------- READ FILES -----------------
resume_text = None
jd_text = None

if resume_file:
    resume_text = read_pdf(resume_file)


if jd_file:
    jd_text = read_pdf(jd_file)
elif jd_text_input and jd_text_input.strip():
    jd_text = jd_text_input.strip()

# ----------------- RUN EXTRACTION -----------------
if st.button("🚀 Extract Skills", type="primary"):
    if not resume_text or not jd_text:
        st.warning("⚠️ Please provide both Resume and Job Description.")
    else:
        with st.spinner("⏳ Extracting skills... Please wait."):
            resume_skills = extract_skills_cached(resume_text, model_choice)
            jd_skills = extract_skills_cached(jd_text, model_choice)

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
