import fitz  # PyMuPDF
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return " ".join(page.get_text() for page in doc)

def clean_text(text, max_chars=3000):
    cleaned = " ".join(text.strip().split())
    return cleaned[:max_chars]