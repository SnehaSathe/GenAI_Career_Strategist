from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from resume_parser import extract_text_from_pdf,clean_text
import ast
import subprocess


def get_llm_skill_extractor(model="mistral"):
    print(f"🔁 Using LLM model: {model}")
    llm = Ollama(model=model)

    template = """You are an AI assistant that extracts only technical skills.

From the following resume, extract all technical skills: programming languages, frameworks, libraries, tools, platforms (e.g., WordPress, Excel), and development environments.

❌ Do not include job titles, roles, sentences, categories, or notes.  
✅ Only return a clean Python list like: ["Python", "Excel", "Scikit-learn", "Power BI", "WordPress"]

Resume:
{text}

Output:
"""
    prompt = PromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

def safe_parse_skill_list(llm_output):
    try:
        line = llm_output.strip().split("\n")[0]
        skills = ast.literal_eval(line)
        return [skill.strip() for skill in skills if isinstance(skill, str)]
    except:
        return []

def extract_skills_from_resume(uploaded_file, model="mistral"):
    try:
        raw_text = extract_text_from_pdf(uploaded_file)
        cleaned = clean_text(raw_text)
        chain = get_llm_skill_extractor(model=model)
        llm_output = chain.invoke({"text": cleaned})
        skills = safe_parse_skill_list(llm_output)

        if not skills:
            print("⚠️ Warning: No skills extracted. LLM Output:\n", llm_output)
        return skills
    except Exception as e:
        print(f"❌ Error during skill extraction: {e}")
        return []

def is_ollama_model_installed(model_name):
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        available_models = [line.split()[0].strip() for line in result.stdout.splitlines()[1:] if line]
        
        # Fix: Allow partial match like "mistral" in "mistral:latest"
        return any(model.startswith(model_name) for model in available_models)

    except Exception as e:
        print(f"❌ Error checking Ollama model: {e}")
        return False
