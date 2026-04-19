from typing import List, Dict
from .semantic_matcher import rank_roles_by_similarity

MATCH_THRESHOLD = 0.7  # 70% JD-Resume match


class AIRoleRecommender:
    def __init__(self, role_descriptions: Dict[str, Dict]):
        """
        role_descriptions:
        {
            role_name: {
                "domain": "AI_TECH",
                "required_skills": [...],
                "description": "..."
            }
        }
        """
        self.role_descriptions = role_descriptions

    def recommend(
        self,
        resume_skills: List[str],
        jd_skills: List[str],
        jd_role: str,
        jd_match_score: float
    ) -> Dict:

        resume_skills_lower = set(skill.lower() for skill in resume_skills)

        # -------------------------------
        # CASE 1: JD matches resume
        # -------------------------------
        if jd_match_score >= MATCH_THRESHOLD:
            missing_skills = list(
                set(skill.lower() for skill in jd_skills)
                - resume_skills_lower
            )

            return {
                "recommended_role": jd_role,
                "decision_type": "JD_MATCH",
                "match_score": round(jd_match_score * 100, 2),
                "missing_skills": list(missing_skills),
                "alternative_roles": []
            }

        # -------------------------------
        # CASE 2: JD does NOT match
        # 👉 FILTER AI ROLES ONLY HERE
        # -------------------------------
        allowed_roles = {
            role: data
            for role, data in self.role_descriptions.items()
            if data.get("domain") == "AI_TECH"
        }

        # Safety check (non-tech resume)
        if not allowed_roles:
            return {
                "recommended_role": "No AI Role Detected",
                "decision_type": "NON_TECH_RESUME",
                "match_score": 0.0,
                "missing_skills": [],
                "alternative_roles": []
            }

        # Prepare role → skills mapping for matcher
        role_skill_map = {
            role: data["required_skills"]
            for role, data in allowed_roles.items()
        }

        inferred_roles = rank_roles_by_similarity(
            resume_skills=resume_skills,
            role_descriptions=role_skill_map,
            top_k=3
        )

        return {
            "recommended_role": inferred_roles[0]["role"],
            "decision_type": "ROLE_INFERRED_FROM_RESUME",
            "match_score": round(inferred_roles[0]["score"] * 100, 2),
            "missing_skills": [],
            "alternative_roles": inferred_roles[1:]
        }
