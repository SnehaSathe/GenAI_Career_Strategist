import os
import json
import socket
import requests
import fitz  # PyMuPDF
import streamlit as st
from langchain_community.llms import Ollama

# ---------------- CONFIG ----------------
groq_api_key = (
    os.getenv("GROQ_API_KEY")   # first check environment variable
    or st.secrets.get("GROQ_API_KEY")  # fallback to secrets.toml
)
OLLAMA_MODEL = "mistral:latest"


# ---------------- ENV DETECTION ----------------
def is_local_env():
    """Detect if running locally (for Ollama fallback)."""
    try:
        host = socket.gethostname()
        ip = socket.gethostbyname(host)
        return ip.startswith("127.") or ip == "localhost"
    except:
        return False


# ---------------- OLLAMA ----------------
@st.cache_resource
def get_ollama_client():
    """Initialize Ollama client (cached)."""
    return Ollama(model=OLLAMA_MODEL)


def use_ollama(prompt: str) -> str:
    """Run prompt locally via Ollama."""
    llm = get_ollama_client()
    return llm.invoke(prompt)


# ---------------- GROQ ----------------
def use_groq(prompt, model_choice, api_key):
    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json={
            "model": model_choice,
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]

    # Debugging message (optional)
    st.error(f"Groq API error {response.status_code}: {response.text}")
    return None


# ---------------- MAIN EXTRACTION ----------------
@st.cache_data(show_spinner=False)
def extract_skills_cached(
    resume_text: str,
    jd_text: str,
    model_choice: str,
    groq_api_key=None
) -> tuple[list[str], list[str]]:
    """
    Extract skills separately for Resume and JD.
    Always returns: (resume_skills_list, jd_skills_list).
    """

    prompt = f"""
You are an advanced AI Resume Matching and Skill Analysis system.

Your job is to analyze a Resume and Job Description and return:
1. Extracted skills
2. Match percentage
3. ATS-style explanation (Strengths, Gaps, Suggestions)

⚠️ STRICT RULES:
- Return ONLY valid JSON
- No markdown, no explanation outside JSON
- Do NOT hallucinate skills or experience
- Only use information present in the text

----------------------------------------
📌 OUTPUT FORMAT (STRICT JSON):
{
    "resume_skills": ["Skill1", "Skill2"],
    "jd_skills": ["SkillA", "SkillB"],
    "match_level": "85%",
    "ai_explanation": "Full ATS-style paragraph explanation",
    "strengths": ["Point1", "Point2"],
    "gaps": ["Gap1", "Gap2"],
    "improvement_suggestions": ["Suggestion1", "Suggestion2"]
}

----------------------------------------
📌 MATCH LEVEL RULES:
- Compare overlap between resume_skills and jd_skills
- High overlap → 80–100%
- Medium overlap → 50–79%
- Low overlap → below 50%

----------------------------------------
📌 AI EXPLANATION FORMAT (VERY IMPORTANT):
Write in this style (single paragraph):

Example:
"The candidate shows strong alignment with the job description with solid technical expertise in core Python and data analysis tools. However, there are gaps in domain-specific experience and missing exposure to certain required technologies."

----------------------------------------
📌 STRENGTHS RULES:
- Highlight matched skills strongly
- Focus on technical skills only
- Mention libraries, tools, frameworks

Example:
- Python, Pandas, NumPy, Scikit-learn
- Data visualization tools like Matplotlib and Power BI

----------------------------------------
📌 GAPS RULES:
- Mention missing skills from JD
- Mention missing experience or tools

Example:
- No mention of web development skills like HTML/CSS
- Limited exposure to deployment or cloud tools

----------------------------------------
📌 IMPROVEMENT SUGGESTIONS:
- Actionable advice
- Skill-based improvement only
- No generic advice only

Example:
- Add real-world projects using Python and ML
- Learn deployment tools like Docker or AWS
- Improve frontend basics if required by JD

----------------------------------------
📌 SKILL EXTRACTION RULES:
- Extract from bullet points, dots (·), commas, brackets
- Split grouped skills:
  "Python & Backend · Python, FastAPI, Flask"
  → ["Python", "Backend", "FastAPI", "Flask"]

- Normalize skills (Python = python = PYTHON)

----------------------------------------
📄 RESUME:
{resume_text}

📄 JOB DESCRIPTION:
{jd_text}
"""

    raw_result = None

    if is_local_env():
        # Local → prefer Ollama, fallback to Groq
        try:
            raw_result = use_ollama(prompt)
        except Exception:
            raw_result = use_groq(prompt, model_choice, groq_api_key)
    else:
        # Cloud → always Groq
        raw_result = use_groq(prompt, model_choice, groq_api_key)

    # Defaults
    resume_skills, jd_skills = [], []

    # Parse JSON safely
    try:
        if not raw_result:
            return [], []

        cleaned = raw_result.strip()

        # Remove ```json code blocks if present
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            cleaned = cleaned.replace("json", "", 1).strip()

        data = json.loads(cleaned)

        # Extract lists safely
        resume_skills = [str(s).strip() for s in data.get("resume_skills", []) if s]
        jd_skills = [str(s).strip() for s in data.get("jd_skills", []) if s]

    except Exception as e:
        st.error(f"⚠️ JSON parsing failed: {e}")
        st.write("Raw output was:", raw_result)

    return resume_skills, jd_skills


