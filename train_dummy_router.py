from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os

X = [
    "hola",
    "buenos días",
    "como estas",
    "cual es tu horario",
    "donde estan ubicados",
    "como cambio mi contraseña",
    "que puedo hacer aqui"
]
y = [
    "simple",
    "simple",
    "simple",
    "simple",
    "simple",
    "simple",
    "complejo"
]

router = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

router.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(router, os.path.join("models", "router_pipeline.joblib"))
joblib.dump(X, os.path.join("models", "router_training_phrases.joblib"))

print("✅ Modelo de clasificación dummy creado correctamente.")
