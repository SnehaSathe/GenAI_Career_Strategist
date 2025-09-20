import sys
import os
from resume_skill_extractor.skill_extractor import extract_skills_cached
from resume_skill_extractor.app import resume_text, jd_text
from langchain_community.embeddings import OllamaEmbeddings
from resume_skill_extractor.resume_parser import extract_candidate_name_llm
import numpy as np
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from fpdf import FPDF
sys.path.append(os.path.abspath("..")) 

# Initialize Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")   # default llama2

# Load secrets
groq_api_key = st.secrets["GROQ_API_KEY"]


def embed_skills(resume_skills, jd_skills):
    """Embed each skill separately (one vector per skill)."""
    if not isinstance(resume_skills, list):
        raise ValueError("resume_skills must be a list of strings")
    if not isinstance(jd_skills, list):
        raise ValueError("jd_skills must be a list of strings")

    resume_vectors = embeddings.embed_documents(resume_skills)
    jd_vectors = embeddings.embed_documents(jd_skills)

    return resume_vectors, jd_vectors


def cosine_sim(vec1, vec2):
    dot = np.dot(vec1, vec2)
    mag_vec1 = np.linalg.norm(vec1)
    mag_vec2 = np.linalg.norm(vec2)
    return dot / (mag_vec1 * mag_vec2)


def find_matches(resume_skills, jd_skills, resume_vectors, jd_vectors, threshold=0.7):
    if not jd_skills:
        return [], [], 0.0
    
    matched_skills = []
    missing_skills = []

    for j, jd_skill in enumerate(jd_skills):
        best_sim = 0
        best_resume_skill = None

        for i, resume_skill in enumerate(resume_skills):
            sim = cosine_sim(resume_vectors[i], jd_vectors[j])
            if sim > best_sim:
                best_sim = sim
                best_resume_skill = resume_skill

        if best_sim >= threshold:
            matched_skills.append((jd_skill, best_resume_skill, round(best_sim, 3)))
        else:
            missing_skills.append(jd_skill)

     # ➕ Additional skills = in resume but not in JD
    additional_skills = [skill for skill in resume_skills if skill not in jd_skills]

    score = round((len(matched_skills) / len(jd_skills)) * 100, 2)
    return matched_skills, missing_skills, additional_skills, score






# ----------------- PDF REPORT -----------------
def generate_report(candidate_name,resume_skills, jd_skills, matched_skills, missing_skills, additional_skills, score, logo_path="logo.png"):
    pdf = FPDF()
    pdf.add_page()

    pdf.image(logo_path, x=10, y=8, w=30)
    pdf.set_font("Times", "B", 16)
    pdf.cell(200, 10, "AI Resume Skill Extractor Report", ln=True, align="C")
    pdf.ln(10)


    pdf.set_font("Times", "I", 16)
    pdf.cell(200, 10, f"{candidate_name}", ln=True, align="C")
    pdf.ln(10)

   # 🔹 Add horizontal line separator
    y = pdf.get_y()
    pdf.line(10, y, 200, y)   # (x1,y) to (x2,y)
    pdf.ln(5)



    pdf.set_font("Times", "B", size=14)
    pdf.cell(200, 10, f"Match Score: {score}%", ln=True)
    pdf.ln(5)

    # Draw progress bar background (gray)
    bar_x = 10
    bar_y = pdf.get_y()
    bar_width = 180
    bar_height = 4
    
    pdf.set_fill_color(220, 220, 220)  # light gray
    pdf.rect(bar_x, bar_y, bar_width, bar_height, "F")

    # Draw filled part (green)
    pdf.set_fill_color(100, 200, 100)  # green
    filled_width = (score / 100) * bar_width
    pdf.rect(bar_x, bar_y, filled_width, bar_height, "F")

    pdf.ln(15)

    pdf.set_font("Times", "B", 13)
    pdf.cell(0, 10, "Resume Skills:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, ", ".join(resume_skills))
    pdf.ln(5)

    pdf.set_font("Times", "B", 13)
    pdf.cell(0, 10, "Job Description Skills:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, ", ".join(jd_skills))
    pdf.ln(5)

    pdf.set_font("Times", "B", 13)
    pdf.cell(0, 10, "Matched Skills:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, ", ".join([m[0] for m in matched_skills]))
    pdf.ln(5)

    pdf.set_font("Times", "B", 13)
    pdf.cell(0, 10, "Missing Skills:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, ", ".join(missing_skills) if missing_skills else "None")
    pdf.ln(5)

    pdf.set_font("Times", "B", 13)
    pdf.cell(0, 10, "Additional Skills in Resume:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, ", ".join(additional_skills) if additional_skills else "None")

    return pdf

  # --- Save to buffer or file ---
    if hasattr(output_path, "write"):  # BytesIO
        pdf.output(output_path)
    else:  # string path
        pdf.output(output_path)