import sys
import os
from resume_skill_extractor.skill_extractor import extract_skills_cached 
from resume_skill_extractor.app import resume_text,jd_text
from langchain_community.embeddings import OllamaEmbeddings
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

##Initialize Embedding model
embeddings=(
    OllamaEmbeddings(model="gemma:2b")        ##by default it use llama2
)


# Initialize embeddings
def embed_skills(resume_skills, jd_skills):
    """
    Embed each skill separately (one vector per skill).
    """
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

    score = round((len(matched_skills) / len(jd_skills)) * 100, 2)

    return matched_skills, missing_skills, score

if __name__ == "__main__":

    resume_skills = extract_skills_cached(resume_text,jd_text, model_choice="mistral:latest")
    jd_skills = extract_skills_cached(resume_text,jd_text, model_choice="mistral:latest")

    resume_vecs, jd_vecs = embed_skills(resume_skills,jd_skills)
    matches, missing, score = find_matches(resume_vecs, jd_vecs)

    print("✅ Matches:", matches)
    print("❌ Missing:", missing)
    print("📊 Score:", score)
