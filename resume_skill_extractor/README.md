# **🧠 Smart Technical Skill Extractor using LLM (Ollama + Streamlit)**


This project is a mini GenAI app that extracts **only technical skills** (like programming languages, libraries, tools, platforms, and frameworks) from any **resume PDF** using a **local LLM** via [Ollama](https://ollama.com).

✅ **No API Keys Needed**  
✅ **Works Offline** (via Ollama)  
✅ Built with **LangChain**, **PyMuPDF**, and **Streamlit**



## 📌 Features

- 📄 Upload any resume PDF  
- 🧠 Extract skills like: `Python`, `SQL`, `Scikit-learn`, `Power BI`, `WordPress`, etc.  
- ⚡ Use your own **LLM** like `mistral`, `zephyr`, `llama2`  
- ⚠️ Auto-check if selected model is installed  
- 📥 Download skills as `.txt`  
- 🖥️ Simple Streamlit UI



## 📸 Screenshot

<img src="https://github.com/SnehaSathe/Smart_Skill_Extractor_using_LLM_Ollama_Streamlit/blob/main/screenshot.png?raw=true" width="700"/>



## 🧰 Tech Stack

| Tool            | Purpose                        |
|-----------------|--------------------------------|
| **LangChain**   | Prompt templates + LLM chaining |
| **Ollama**      | Run open LLMs locally           |
| **PyMuPDF**     | Extract text from PDF resumes   |
| **Streamlit**   | Web UI                          |



## 🛠️ Setup Instructions

### 1. 🔃 Clone the repository
```bash
git clone https://github.com/SnehaSathe/Smart_Technical_Skill_Extractor_using_LLM_Ollama_Streamlit.git
cd Smart_Technical_Skill_Extractor_using_LLM_Ollama_Streamlit
````

### 2. 💽 Create virtual environment

```bash
conda create -n resume-env python=3.10
conda activate resume-env
```

### 3. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

### 4. 🧠 Install Ollama + Pull model

* Download from: [https://ollama.com/download](https://ollama.com/download)
* Start Ollama:

  ```bash
  ollama serve
  ```
* Pull a model (e.g. mistral):

  ```bash
  ollama pull mistral
  ```

### 5. ▶️ Run the app

```bash
streamlit run app.py
```

---

## ⚠️ Troubleshooting

| Issue                        | Fix                                                      |
| ---------------------------- | -------------------------------------------------------- |
| `"Model not installed"`      | Make sure to run `ollama pull mistral`                   |
| `subprocess not defined`     | Add `import subprocess` in `skill_extractor.py`                |
| Resume not extracting skills | Try another model (`zephyr`, `llama2`) or check PDF text |

---

## 🧪 Example Output

**Input:** Resume PDF with experience in ML and web dev
**Output:**

```python
["Python", "MySQL", "Power BI", "Scikit-learn", "HTML", "CSS", "WordPress", "NumPy", "Pandas"]
```

---

## 📁 Project Structure

```
📦 smart-skill-extractor
├── app.py                  # Streamlit app
├── extractor.py            # PDF + LLM skill extraction logic
├── requirements.txt        # Python dependencies
├── assets/
│   └── screenshot.png      # UI screenshot
└── README.md
```

---

## 📌 Future Improvements

* ✅ Job description upload + Skill match
* 📊 Matched vs Gap skill analysis
* 💾 Export to JSON/CSV
* ☁️ Optional: OpenAI version for cloud deployment

---

## 📃 License

This project is licensed under the **MIT License**.
Free to use, modify, and distribute.

---

## 🙋‍♀️ Author

Built with ❤️ by [Sneha Sathe](https://github.com/SnehaSathe)
Inspired by real-world GenAI resume automation problems.


⭐ Star this repo if it helped you!



