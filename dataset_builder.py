import os
import PyPDF2
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('punkt_tab')

model = SentenceTransformer('all-MiniLM-L6-v2')

data = []

def extract_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    return text

def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

def process_paper(pdf_path):
    text = extract_text(pdf_path)
    text = clean_text(text)

    sentences = sent_tokenize(text)

    for s in sentences:
        if len(s) > 30:
            embedding = model.encode(s).tolist()
            data.append({
                "sentence": s,
                "source": os.path.basename(pdf_path),
                "embedding": embedding
            })

folder = "papers"

for file in os.listdir(folder):
    if file.endswith(".pdf"):
        print(f"Processing {file}...")
        process_paper(os.path.join(folder, file))

df = pd.DataFrame(data)
df.to_json("dataset.json", orient="records", indent=2)

print("✅ Dataset created:", len(df), "sentences")