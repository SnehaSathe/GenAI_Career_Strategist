# 💼 GenAI Career Strategist

An AI-powered career assistant that extracts skills from resumes, analyzes job fit, and recommends optimal roles and learning paths using GenAI + LangChain + Streamlit + Ollama.

---

## 🔍 Use Case

This app helps job seekers and students in AI/Tech domains:
- Extract technical skills from resumes without predefined lists
- Compare their skills with job descriptions
- Get role recommendations (e.g., Prompt Engineer, GenAI Developer)
- Prepare for interviews using AI-generated Q&A

---

## 🧱 Features (Mini Projects)

Each feature is a standalone mini project with its own logic, prompts, and models:

| Feature | Description | Source Code |
|--------|-------------|-------------|
| ✅ Resume Skill Extractor | Extracts relevant technical skills using LLM + NLP | [View Code](modules/resume_skill_extractor/) |
| ✅ JD Skill Gap Analyzer | Finds missing skills by comparing resume & JD | [In Progress](modules/jd_gap_analyzer/) |
| ✅ AI Role Recommender | Suggests suitable AI career roles | [In Progress](modules/job_role_recommender/) |

---

## 🛠️ Tech Stack

- **Python 3.12+**
- **Streamlit**
- **LangChain 0.2+**
- **Ollama (LLM local inference)**
- **HuggingFace Transformers**
- **spaCy, PyMuPDF, KeyBERT**

---

## 🚀 How to Run

```bash
# Clone
git clone https://github.com/SnehaSathe/GenAI_Career_Strategist.git
cd GenAI_Career_Strategist

# Set up environment
pip install -r requirements.txt

# Run any module
cd modules/resume_skill_extractor
streamlit run app.py
