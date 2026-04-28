from flask import Flask, request, jsonify, render_template
import json
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import re
import os

# ML imports (lazy load later)
from sentence_transformers import SentenceTransformer, util

# Download tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

# -------------------------
# GLOBAL LAZY VARIABLES
# -------------------------
model = None
dataset_embeddings = None
corpus_embeddings = None

# -------------------------
# Ayurvedic Stopwords
# -------------------------
ayurveda_stopwords = {
    "ayurveda", "panchakarma", "vata", "pitta", "kapha",
    "dosha", "doshas", "agni", "ama", "prakriti", "vikriti",
    "rasayana", "dinacharya", "ritucharya", "herbal",
    "therapy", "treatment", "medicine", "body", "health",
    "disease", "balance", "system", "natural", "healing"
}

# -------------------------
# Clean text
# -------------------------
def clean_text(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return " ".join([w for w in words if w not in ayurveda_stopwords])

# -------------------------
# Load dataset
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
"ayurveda is a traditional system of medicine that focuses on balance between body mind and spirit",
"the concept of tridosha includes vata pitta and kapha which govern physiological functions",
"vata dosha is responsible for movement and communication within the body",
"pitta dosha regulates metabolism digestion and body temperature",
"kapha dosha provides structure stability and lubrication to the body",
"agni refers to the digestive fire that controls metabolism and transformation of food into energy",
"ama is considered as toxic waste formed due to improper digestion",
"panchakarma is a cleansing therapy used to remove toxins from the body",
"abhyanga is an oil massage therapy that improves circulation and relaxation",
"rasayana therapy is used for rejuvenation and promoting longevity",
"dinacharya refers to daily routine practices that maintain health and prevent disease",
"ritucharya is a seasonal regimen followed to adapt to environmental changes",
"ayurveda emphasizes prevention rather than cure of diseases",
"herbal medicines play an important role in ayurvedic treatment",
"ashwagandha is known for its stress relieving and adaptogenic properties",
"turmeric has anti inflammatory and antioxidant benefits",
"triphala is a combination of three fruits used for digestion and detoxification",
"neem is used for its antibacterial and blood purifying properties",
"ayurvedic diet is based on balancing the doshas through proper food choices",
"lifestyle management is a key aspect of ayurvedic healing",
"yoga and meditation are often integrated with ayurveda for holistic health",
"prakriti refers to an individual's body constitution determined at birth",
"vikriti represents the current imbalance in the body",
"shodhana therapy focuses on purification of the body",
"shamana therapy aims at pacifying the aggravated doshas",
"ayurveda considers mind body and soul as interconnected elements",
"proper sleep is essential for maintaining health according to ayurveda",
"stress is considered a major cause of imbalance in doshas",
"herbal formulations are prepared using natural plant based ingredients",
"ayurveda uses natural methods to restore balance in the body",
"digestion is considered the root of overall health in ayurveda",
"detoxification helps in eliminating accumulated toxins",
"balance between doshas ensures proper functioning of the body",
"ayurvedic therapies aim to strengthen the immune system",
"natural healing methods are emphasized over synthetic drugs",
"body tissues known as dhatus play a role in maintaining structure and function",
"ojas is considered the essence of vitality and immunity",
"proper hydration is important for maintaining balance in the body",
"mental clarity is achieved through balanced lifestyle practices",
"ayurveda promotes harmony with nature for overall well being",
"individualized treatment is a key principle in ayurveda",
"pulse diagnosis is used to assess dosha imbalance",
"detox therapies help in improving metabolic functions",
"herbal oils are used in various therapeutic procedures",
"regular exercise supports healthy digestion and metabolism",
"emotional balance is important for maintaining health",
"natural remedies are preferred in ayurvedic treatments",
"improper diet can lead to accumulation of toxins",
"balance of digestive fire is essential for good health",
"ayurvedic principles aim to prevent chronic diseases",
"holistic healing addresses physical mental and emotional aspects",
"healthy lifestyle habits are emphasized in ayurveda",
"cleansing therapies are used to restore internal balance",
"plant based medicines are widely used in ayurvedic practice",
"seasonal detoxification improves overall wellness",
"balanced nutrition supports proper functioning of the body",
"ayurveda encourages mindful eating practices",
"daily routines help maintain internal harmony",
"natural herbs are used to treat various disorders",
"body detoxification improves immunity and vitality",
"healthy digestion leads to better absorption of nutrients",
"ayurvedic medicine focuses on long term wellness",
"balancing doshas helps in preventing diseases",
"traditional healing practices are integrated with modern approaches",
"ayurveda uses personalized therapies for each individual",
"immune strength is enhanced through proper diet and lifestyle",
"herbal supplements are used for maintaining health",
"internal balance leads to improved mental well being",
"detoxification therapies remove harmful substances from the body",
"natural treatments reduce side effects compared to synthetic drugs",
"proper lifestyle supports overall physical and mental health",
"body cleansing is important for maintaining internal purity",
"healthy routines improve longevity and quality of life" ]
corpus_clean = [clean_text(s) for s in corpus]

# -------------------------
# Lazy Load Model
# -------------------------
def get_model():
    global model
    if model is None:
        print("🔄 Loading model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# -------------------------
# Lazy Dataset Embeddings
# -------------------------
def get_dataset_embeddings():
    global dataset_embeddings
    if dataset_embeddings is None:
        print("🔄 Encoding dataset...")
        dataset_embeddings = get_model().encode(dataset_clean, convert_to_tensor=True)
    return dataset_embeddings

# -------------------------
# Lazy Corpus Embeddings
# -------------------------
def get_corpus_embeddings():
    global corpus_embeddings
    if corpus_embeddings is None:
        print("🔄 Encoding corpus...")
        corpus_embeddings = get_model().encode(corpus_clean, convert_to_tensor=True)
    return corpus_embeddings

# -------------------------
# URL Extractor (Improved)
# -------------------------
def extract_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)

        if res.status_code != 200:
            return ""

        soup = BeautifulSoup(res.text, "html.parser")

        # Remove junk
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
# Plagiarism Check
# -------------------------
def check_plagiarism(text):

    sentences = sent_tokenize(text)

    best_score = 0
    best_result = None

    for sentence in sentences:

        if len(sentence.strip()) < 25:
            continue

        cleaned = clean_text(sentence)
        if not cleaned:
            continue

        input_emb = get_model().encode(cleaned, convert_to_tensor=True)

        # Compare dataset
        d_scores = util.cos_sim(input_emb, get_dataset_embeddings())[0]
        d_idx = d_scores.argmax()
        d_score = float(d_scores[d_idx])

        # Compare corpus
        c_scores = util.cos_sim(input_emb, get_corpus_embeddings())[0]
        c_idx = c_scores.argmax()
        c_score = float(c_scores[c_idx])

        if d_score > c_score:
            score = d_score
            match = dataset_sentences[d_idx]
            source = dataset_sources[d_idx]
            type_ = "Research Paper"
        else:
            score = c_score
            match = corpus[c_idx]
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

    # URL input
    if url and url.strip():
        text = extract_from_url(url)

        if not text:
            return jsonify({"error": "❌ Could not extract text from URL"})

        text = text[:3000]  # limit size

    if not text or not text.strip():
        return jsonify({"error": "❌ Enter text or URL"})

    result = check_plagiarism(text)

    print("✅ RESULT:", result)

    return jsonify(result)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    print("🚀 Starting server...")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
