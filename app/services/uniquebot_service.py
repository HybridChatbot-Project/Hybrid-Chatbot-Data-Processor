import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

FAQ_DB = [
    {"question": "¿Cuál es el horario de atención?", "answer": "Nuestro horario es de lunes a viernes de 9:00 a 18:00."},
    {"question": "¿Cómo cambio mi contraseña?", "answer": "Puedes cambiar tu contraseña desde Perfil > Seguridad."},
    {"question": "¿Dónde están ubicados?", "answer": "Estamos en Lima Metropolitana."}
]

STOPWORDS = set(stopwords.words("spanish"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in STOPWORDS]

def handle_local_message(message: str) -> str:
    words = set(preprocess(message))
    best_score, best_answer = 0, "Lo siento, no tengo una respuesta exacta."
    for item in FAQ_DB:
        q_words = set(preprocess(item["question"]))
        score = len(words & q_words)
        if score > best_score:
            best_score = score
            best_answer = item["answer"]
    return best_answer