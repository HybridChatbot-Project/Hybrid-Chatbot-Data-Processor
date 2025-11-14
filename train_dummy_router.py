import csv
import joblib
import os
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/faq.csv")
SIMPLE_CATEGORIES = {"ansiedad", "personal"}

def load_phrases():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró {DATA_PATH}.")

    simple, complex_candidates = [], []
    with DATA_PATH.open(encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = (row.get("question") or "").strip()
            category = (row.get("category") or "").strip().lower()
            if not question:
                continue
            normalized = question.lower()
            if category in SIMPLE_CATEGORIES:
                simple.append(normalized)
            else:
                complex_candidates.append(normalized)

    # Remove duplicates preserving order
    simple = list(dict.fromkeys(simple))
    complex_candidates = list(dict.fromkeys(complex_candidates))
    return simple, complex_candidates

def build_dataset():
    simple_phrases, complex_from_csv = load_phrases()
    if not simple_phrases:
        raise ValueError("No se encontraron preguntas elegibles en data/faq.csv. Revisa las categorías permitidas.")

    complex_phrases = [
        "escribe un ensayo profundo sobre la mente humana",
        "necesito código python para analizar datos financieros",
        "cuéntame una historia original de ciencia ficción",
        "explícame cómo funciona la teoría de cuerdas",
        "puedes resumir un libro completo",
        "dame un plan de marketing detallado"
    ] + complex_from_csv

    X = simple_phrases + complex_phrases
    y = (["simple"] * len(simple_phrases)) + (["complejo"] * len(complex_phrases))
    return X, y, simple_phrases

def main():
    X, y, training_phrases = build_dataset()

    router = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    router.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(router, os.path.join("models", "router_pipeline.joblib"))
    joblib.dump(training_phrases, os.path.join("models", "router_training_phrases.joblib"))
    print(f"✅ Router entrenado con {len(X)} frases ({len(training_phrases)} simples / {len(X)-len(training_phrases)} complejas).")

if __name__ == "__main__":
    main()
