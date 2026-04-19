import json
from inference_engine import AIRoleRecommender

# Load role descriptions (TEXT, not skills)
with open("role_descriptions.json", "r") as f:
    role_descriptions = json.load(f)


def run_demo():
    resume_skills = [
        "Python", "FastAPI", "LangChain",
        "LLM", "Docker", "Machine Learning"
    ]

    jd_skills = [
        "Kubernetes", "Spark", "Airflow"
    ]

    jd_role = "ML Engineer"
    jd_match_score = 0.45  # From your skill gap analyzer

    recommender = AIRoleRecommender(role_descriptions)

    result = recommender.recommend(
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        jd_role=jd_role,
        jd_match_score=jd_match_score
    )

    print(result)


if __name__ == "__main__":
    run_demo()
