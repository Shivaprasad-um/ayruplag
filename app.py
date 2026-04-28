from flask import Flask, request, jsonify, render_template
import json
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

print("🔥 App started...")

# -------------------------
# Safe Tokenizer (no startup download)
# -------------------------
def safe_tokenize(text):
    try:
        return sent_tokenize(text)
    except:
        nltk.download('punkt')
        return sent_tokenize(text)

# -------------------------
# Stopwords (Ayurveda specific)
# -------------------------
ayurveda_stopwords = {
    "ayurveda", "panchakarma", "vata", "pitta", "kapha",
    "dosha", "doshas", "agni", "ama", "prakriti", "vikriti",
    "rasayana", "dinacharya", "ritucharya", "herbal",
    "therapy", "treatment", "medicine", "body", "health",
    "disease", "balance", "system", "natural", "healing"
}

# -------------------------
# Clean Text
# -------------------------
def clean_text(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return " ".join([w for w in words if w not in ayurveda_stopwords])

# -------------------------
# Load Dataset
# -------------------------
with open("dataset.json") as f:
    dataset = json.load(f)

dataset_sentences = [d["sentence"] for d in dataset]
dataset_sources = [d["source"] for d in dataset]
dataset_clean = [clean_text(s) for s in dataset_sentences]

# -------------------------
# Corpus
# -------------------------
corpus = [
    "ayurveda is a traditional system of medicine focusing on balance",
    "tridosha includes vata pitta and kapha",
    "panchakarma is used for detoxification",
    "agni controls digestion and metabolism",
    "rasayana improves longevity",
    "herbal medicines are widely used in ayurveda",
    "detoxification helps remove toxins from the body",
    "healthy digestion is essential for wellness",
    "doshas maintain physiological balance in the body",
    "natural healing methods are emphasized in ayurveda"
]

corpus_clean = [clean_text(s) for s in corpus]

# -------------------------
# URL Extractor
# -------------------------
def extract_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)

        if res.status_code != 200:
            return ""

        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text(strip=True) for p in paragraphs])

        print("Extracted length:", len(text))

        return text.strip()

    except Exception as e:
        print("URL error:", e)
        return ""

# -------------------------
# Plagiarism Check (TF-IDF)
# -------------------------
def check_plagiarism(text):

    sentences = safe_tokenize(text)

    best_score = 0
    best_result = None

    for sentence in sentences:

        if len(sentence.strip()) < 25:
            continue

        cleaned = clean_text(sentence)
        if not cleaned:
            continue

        # Combine all text
        all_text = dataset_clean + corpus_clean + [cleaned]

        vectorizer = TfidfVectorizer().fit_transform(all_text)
        vectors = vectorizer.toarray()

        input_vec = vectors[-1]

        scores = cosine_similarity([input_vec], vectors[:-1])[0]

        best_idx = scores.argmax()
        score = scores[best_idx]

        if best_idx < len(dataset_sentences):
            match = dataset_sentences[best_idx]
            source = dataset_sources[best_idx]
            type_ = "Research Paper"
        else:
            match = corpus[best_idx - len(dataset_sentences)]
            source = "Built-in Corpus"
            type_ = "Ayurvedic Knowledge"

        if score > best_score:
            best_score = score
            best_result = {
                "input": sentence,
                "match": match,
                "source": source,
                "type": type_,
                "score": round(score * 100, 2)
            }

    if best_result is None:
        return {
            "input": "No meaningful sentence found",
            "match": "No match",
            "source": "None",
            "type": "None",
            "score": 0
        }

    return best_result

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check():
    data = request.json

    text = data.get("text")
    url = data.get("url")

    if url and url.strip():
        text = extract_from_url(url)

        if not text:
            return jsonify({"error": "❌ Could not extract text from URL"})

        text = text[:3000]

    if not text or not text.strip():
        return jsonify({"error": "❌ Enter text or URL"})

    result = check_plagiarism(text)

    print("RESULT:", result)

    return jsonify(result)

# -------------------------
# Health Check (important for Render)
# -------------------------
@app.route("/ping")
def ping():
    return "pong"

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
