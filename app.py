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
# SAFE TOKENIZER
# -------------------------
def safe_tokenize(text):
    try:
        return sent_tokenize(text)
    except Exception as e:
        print("Tokenizer error:", e)
        return [text]  # fallback

# -------------------------
# STOPWORDS
# -------------------------
ayurveda_stopwords = {
    "ayurveda", "panchakarma", "vata", "pitta", "kapha",
    "dosha", "doshas", "agni", "ama", "prakriti", "vikriti",
    "rasayana", "dinacharya", "ritucharya", "herbal",
    "therapy", "treatment", "medicine", "body", "health",
    "disease", "balance", "system", "natural", "healing"
}

# -------------------------
# CLEAN TEXT
# -------------------------
def clean_text(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return " ".join([w for w in words if w not in ayurveda_stopwords])

# -------------------------
# LOAD DATASET
# -------------------------
try:
    with open("dataset.json") as f:
        dataset = json.load(f)
except Exception as e:
    print("❌ Dataset load error:", e)
    dataset = []

dataset_sentences = [d.get("sentence", "") for d in dataset]
dataset_sources = [d.get("source", "Unknown") for d in dataset]
dataset_clean = [clean_text(s) for s in dataset_sentences]

print("✅ Dataset loaded:", len(dataset_sentences))

# -------------------------
# CORPUS
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
# URL EXTRACTOR
# -------------------------
def extract_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)

        if res.status_code != 200:
            print("❌ URL fetch failed")
            return ""

        soup = BeautifulSoup(res.text, "html.parser")

        # Remove junk tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text(strip=True) for p in paragraphs])

        print("✅ Extracted length:", len(text))

        return text.strip()

    except Exception as e:
        print("❌ URL error:", e)
        return ""

# -------------------------
# PLAGIARISM CHECK
# -------------------------
def check_plagiarism(text):

    sentences = safe_tokenize(text)

    best_score = 0
    best_result = None

    for sentence in sentences:

        if len(sentence.strip()) < 20:
            continue

        cleaned = clean_text(sentence)
        if not cleaned:
            continue

        try:
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

        except Exception as e:
            print("❌ Similarity error:", e)

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
# ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check():
    try:
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

        print("✅ RESULT:", result)

        return jsonify(result)

    except Exception as e:
        print("❌ SERVER ERROR:", e)
        return jsonify({"error": "Server error occurred"})

# -------------------------
# HEALTH CHECK
# -------------------------
@app.route("/ping")
def ping():
    return "pong"

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
