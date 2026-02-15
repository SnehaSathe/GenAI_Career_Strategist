# 💼 GenAI Career Strategist

A Modular RAG-Based GenAI System for AI Career Intelligence

An AI-powered career intelligence platform that uses Retrieval-Augmented Generation (RAG), semantic embeddings, and structured LLM prompting to:

Extract technical skills from resumes

Analyze job-description skill gaps

Recommend AI/GenAI career roles

Generate personalized interview preparation content


Built using LangChain + Ollama + FAISS + Streamlit with modular architecture for scalability.


---

## 🧠 Problem Statement

Modern job seekers face three major challenges:

1. Resumes are parsed using static keyword systems


2. Skill gap analysis lacks semantic understanding


3. Career role suggestions are generic and not personalized



This project solves these problems using:

Semantic embedding-based retrieval

LLM-driven structured reasoning

Modular RAG pipelines



---

## 🏗️ System Architecture Overview

                ┌────────────────
                │   Resume (PDF)    │
                └────────┬───────┘
                           │
                  Text Extraction (PyMuPDF)
                           │
                       Chunking
                           │
                    Embedding Model
                           │
                        FAISS DB
                           │
      ┌──────────────────────────────────────┐
      │                    │                        │
  Skill Extraction   JD Gap Analysis      Role Recommendation
      │                    │                        │
      Structured LLM Output (JSON Controlled Prompting)


---

## 🧱 Modular Architecture

Each feature is designed as a standalone GenAI module with:

Independent prompt templates

Dedicated retrieval pipeline

Structured output schema

Clear evaluation logic



---

# ✅ 1️⃣ Resume Skill Extractor

### Architecture Type:

RAG-based Semantic Extraction

### Key Decisions:

Used RAG instead of fine-tuning (no labeled dataset)

Used structured JSON output to reduce hallucination

Top-K retrieval to limit token usage


### Embedding Model:

Ollama local embedding model (e.g., nomic-embed-text)

### Vector Store: FAISS


### Retrieval Strategy:

Cosine similarity search

Top-K chunk retrieval

Context injected into LLM prompt


### Chunking Strategy:

500-token fixed chunking

50-token overlap

Improves recall at slight embedding cost


### Evaluation Metrics:

Precision

Recall

F1-score

Hallucination rate comparison (Direct LLM vs RAG)


### Performance:

Avg Latency: ~1.5–2 seconds

Precision improved to ~88% after prompt refinement



---

# ✅ 2️⃣ JD Skill Gap Analyzer

### Approach:

Extract skills from Resume and JD independently

Convert skills into embeddings

Compute semantic similarity

Calculate match percentage


### Scoring Modes:

Candidate View

Recruiter View

Balanced Mode


### Trade-off:

Semantic similarity improves flexibility but may slightly overestimate match percentage for closely related skills.


---

# ✅ 3️⃣ AI Role Recommender

### Logic:

Skill clustering

Role mapping via prompt-based reasoning

Context-aware recommendation


### Example Output:

GenAI Developer

Prompt Engineer

AI Research Assistant

ML Engineer (Entry Level)


## Future upgrade:

Add role-skill embedding clustering instead of prompt-only mapping



---

## ⚙️ Prompt Engineering Strategy

### Each module uses:

Explicit system role definition

Strict JSON schema

Output constraints

Hallucination suppression instruction

Low-temperature inference


### Benefits:

Reduced output noise

Improved parsing reliability

Deterministic responses



---

## 📊 Evaluation Strategy

### Module	Metric	Goal

Skill Extraction	Precision / Recall	Reduce hallucination
Skill Gap	Match Accuracy	Reflect real JD similarity
Role Recommender	Relevance Score	Career alignment accuracy


### Future:

Automated evaluation pipeline

Synthetic benchmark dataset



---

## 📈 Scalability Approach

### Current:

Streamlit UI (Single Node)

Local FAISS

Local LLM via Ollama


### Production Architecture (Planned):

FastAPI backend

Stateless API containers

Dockerized deployment

Redis caching

Managed vector DB (Pinecone / Weaviate)

Horizontal scaling via load balancer



---

## 🚀 Deployment Strategy

### Current:

Local deployment via Streamlit

Modular execution


### Planned:

Docker containerization

AWS EC2 / GCP deployment

Nginx reverse proxy

CI/CD integration



---

## ⚖️ Engineering Trade-offs

Decision	Benefit	Trade-off

RAG over direct LLM	Reduced hallucination	Slight latency increase
Local Ollama	Low cost	Requires local setup
FAISS	Fast retrieval	Not cloud-native scalable
Chunk overlap	Higher recall	More embedding cost



---

## 💰 Cost Optimization

Local LLM inference via Ollama

No paid vector DB

Limited Top-K retrieval

Controlled token size in prompts

Embedding caching


Estimated ~50% cost reduction compared to cloud LLM + external vector DB architecture.


---

## 🛠️ Tech Stack

Python 3.12+

Streamlit

LangChain 0.2+

Ollama (Local LLM Inference)

FAISS

HuggingFace Transformers

spaCy

PyMuPDF

KeyBERT



---

## 📂 Repository Structure

GenAI_Career_Strategist/
│
├── modules/
│   ├── resume_skill_extractor/
│   ├── jd_gap_analyzer/
│   ├── job_role_recommender/
│
├── shared/
│   ├── embeddings.py
│   ├── prompt_templates.py
│
├── requirements.txt
└── README.md


---

## 🎯 Key Engineering Learnings

RAG significantly reduces hallucination in extraction tasks

Structured output improves reliability

Retrieval strategy impacts precision more than model size

Prompt clarity > model size in many GenAI workflows

Token optimization directly impacts cost & latency



---

## 👩‍💻 Author

Sneha Sathe
GenAI / AI Engineer

GitHub: https://github.com/SnehaSathe
LinkedIn: https://www.linkedin.com/in/snehasathe


---


---

If you want next upgrade, I can:

Make a 1-page Architecture Diagram version

Create a portfolio case study version (for recruiters)

Or rewrite this to optimize GitHub SEO so recruiters find it


Tell me which direction you want.
