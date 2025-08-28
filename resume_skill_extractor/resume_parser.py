import os
import fitz  # PyMuPDF
import re
import spacy
import requests
import json
import streamlit as st


import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

    
# --- PDF Extraction ---
@st.cache_data
def extract_text_from_pdf_cached(file_bytes):
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text()
    return text.strip()

def clean_text(text):
    return " ".join(text.strip().split())