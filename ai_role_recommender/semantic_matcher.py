from sentence_transformers import SentenceTransformer, util
from typing import List, Dict

# Local embedding model (FREE)
model = SentenceTransformer("all-MiniLM-L6-v2")


def rank_roles_by_similarity(
    resume_skills: List[str],
    role_descriptions: Dict[str, str],
    top_k: int = 3
):
    """
    Infers suitable roles using semantic similarity
    """

    # Convert resume skills → single semantic profile
    resume_profile = " ".join(resume_skills)
    resume_embedding = model.encode(resume_profile, convert_to_tensor=True)

    results = []

    for role, description in role_descriptions.items():
        role_embedding = model.encode(description, convert_to_tensor=True)
        score = util.cos_sim(resume_embedding, role_embedding).item()

        results.append({
            "role": role,
            "score": round(score, 4)
        })

    # Sort by similarity
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]
