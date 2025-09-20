# **🧠 JD Skill Gap Analyzer (LLM + Streamlit)**

This module is part of a GenAI app that analyzes the skill gap between a candidate's resume and a job description. It highlights matched, missing, and additional skills, and generates a PDF report.

✅ Supports PDF & DOCX files
✅ Uses LLM or Groq embeddings for accurate skill extraction
✅ Built with LangChain, Streamlit, and PyMuPDF

## 📌 Features

📄 Upload Resume (PDF/DOCX) and Job Description (PDF/DOCX or text)
🧠 Extract skills from both documents
⚡ Identify:
Matched skills
Missing skills
Additional skills
📊 Calculate match score (%)
📥 Generate PDF report with candidate name, logo, and watermark
🖥️ Simple Streamlit UI with interactive buttons

## 📸 Screenshot

<img src="https://github.com/SnehaSathe/GenAI_Career_Strategist/blob/main/resume_skill_extractor/resume_skill_extractor.png" width="700"/>

## 🧰 Tech Stack

### Tool	Purpose
LangChain	LLM chaining + prompt templates
LLM / Groq	Skill extraction & semantic matching
PyMuPDF	Extract text from PDF resumes
Streamlit	Web UI & interactive buttons
FPDF2	Generate PDF reports

## 🛠️ Setup Instructions

**1. 🔃 Clone the repository**
git clone https://github.com/SnehaSathe/GenAI_Career_Strategist.git
cd GenAI_Career_Strategist

**2. 💽 Create virtual environment**
conda create -n skillgap-env python=3.12
conda activate skillgap-env

**3. 📦 Install dependencies**
pip install -r requirements.txt

**4. ▶️ Run the app**
streamlit run app.py

## ⚠️ Troubleshooting

Issue	Fix

Resume or JD not extracting skills	Check file format (PDF/DOCX), ensure text is selectable

"Model not installed"	Make sure your LLM model is available locally or via Groq

PDF report fails to generate	Check logo path or permissions

## 🧪 Example Output

**Input:** Resume + Job Description
**Output:**

***Matched Skills:*** ["Python", "SQL", "Power BI"]
***Missing Skills:*** ["TensorFlow", "AWS"]
***Additional Skills:*** ["Excel", "Tableau"]
***Match Score:*** 65%

## 📌 Future Improvements

✅ Add multi-page resume support

⚡ Integrate job description analysis for multiple roles

💾 Export results to JSON / CSV

☁️ Optional cloud deployment with OpenAI or HuggingFace models

## 📃 License

***This project is proprietary for commercial use if sold as a digital product.
Otherwise, you may use it under MIT License.***

## 🙋‍♀️ Author

***Built with ❤️ by Sneha Sathe***

Inspired by real-world GenAI skill analysis problems.

⭐ Star this repo if it helped you!
