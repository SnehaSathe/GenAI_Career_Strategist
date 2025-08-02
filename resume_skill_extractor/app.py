import streamlit as st
from skill_extractor import extract_skills_from_resume, is_ollama_model_installed

st.set_page_config(
    page_title="🧠 Smart Skill Extractor",
    page_icon="🧠",
    layout="centered"
)

# ---- HEADER ----
st.markdown("<h1 style='text-align:center;'>🧠 Smart Technical Skill Extractor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:18px;'>Extract technical skills like languages, libraries, frameworks, tools, and platforms from any resume using a local LLM (Ollama).</p>",
    unsafe_allow_html=True
)

# ---- SIDEBAR ----
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Choose Ollama Model",
    ["mistral", "zephyr", "llama2", "codellama"],
    index=0
)

# ---- FILE UPLOAD ----
st.markdown("### 📄 Upload Your Resume (PDF)")
uploaded_file = st.file_uploader(
    "Drop your resume PDF here or browse files",
    type=["pdf"],
    label_visibility="collapsed"
)

# ---- MODEL CHECK ----
if not is_ollama_model_installed(model_choice):
    st.error(f"🚫 The selected model '{model_choice}' is not installed.\n\nRun this command in terminal:\n\n`ollama pull {model_choice}`")
    st.stop()

# ---- PROCESSING ----
if uploaded_file:
    st.markdown(f"✅ **Uploaded:** `{uploaded_file.name}`")

    with st.spinner("🔍 Analyzing your resume... please wait..."):
        skills = extract_skills_from_resume(uploaded_file, model_choice)

    if skills:
        st.success("✅ Skills Extracted Successfully!")
        st.markdown("### 🧩 Extracted Technical Skills")
        st.markdown(
            f"<div style='padding:10px; background-color:#f9f9f9; border-radius:10px;'><b>{', '.join(skills)}</b></div>",
            unsafe_allow_html=True
        )
        st.download_button("📥 Download Skills as TXT", "\n".join(skills), file_name="skills.txt")
    else:
        st.warning("⚠️ No skills extracted. Try a different model or resume.")
