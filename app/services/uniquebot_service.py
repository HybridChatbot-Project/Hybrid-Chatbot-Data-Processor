import re
import csv
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

FALLBACK_FAQ = [
    {
        "question": "¿Qué es la ansiedad?",
        "answer": "la ansiedad es una respuesta natural de alerta del cuerpo que se vuelve problemática cuando es intensa y persistente.",
        "source": "American Psychological Association",
        "comfort": "Respira profundo, lo que sientes es tu cuerpo intentando cuidarte."
    },
    {
        "question": "¿Cómo manejar un ataque de pánico?",
        "answer": "Respira en ciclos 4-4-6 y enfócate en describir mentalmente tres objetos a tu alrededor; la crisis disminuirá.",
        "source": "Mindfulness-Based Stress Reduction (Kabat-Zinn)",
        "comfort": "Mientras respiras, recuerda que esta sensación pasará y no estás solo."
    },
]

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "faq.csv"
STOPWORDS = set(stopwords.words("spanish"))
SIMILARITY_THRESHOLD = 0.23

def _load_faqs():
    faqs = []
    if DATA_PATH.exists():
        with DATA_PATH.open(encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                source = (row.get("source") or "").strip()
                comfort = (row.get("comfort") or "").strip()
                category = (row.get("category") or "").strip().lower() or None
                if question and answer:
                    faqs.append({
                        "question": question,
                        "answer": answer,
                        "source": source,
                        "comfort": comfort,
                        "category": category
                    })
    if not faqs:
        faqs = list(FALLBACK_FAQ)
    return faqs

FAQ_DB = _load_faqs()

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = word_tokenize(text)
    filtered = [t for t in tokens if t not in STOPWORDS]
    return " ".join(filtered)

def _build_vectorizer():
    questions = [preprocess(item["question"]) for item in FAQ_DB]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(questions)
    return vectorizer, matrix

VECTORIZER, FAQ_MATRIX = _build_vectorizer()

def handle_local_message(message: str) -> str:
    if not message.strip():
        return None

    processed = preprocess(message)
    if not processed:
        return None

    query_vec = VECTORIZER.transform([processed])
    similarities = cosine_similarity(query_vec, FAQ_MATRIX)[0]
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]

    if best_score < SIMILARITY_THRESHOLD:
        return None

    return _compose_answer(FAQ_DB[best_idx])

def _compose_answer(item: dict) -> str:
    comfort = item.get("comfort")
    answer = item.get("answer")

    parts = []
    if comfort:
        parts.append(comfort.strip())
    if answer:
        parts.append(answer.strip())

    return " ".join(parts).strip()
