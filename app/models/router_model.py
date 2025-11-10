import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = os.path.join("models", "router_pipeline.joblib")
TRAINING_PHRASES_PATH = os.path.join("models", "router_training_phrases.joblib")

if os.path.exists(MODEL_PATH):
    router = joblib.load(MODEL_PATH)
    training_phrases = joblib.load(TRAINING_PHRASES_PATH)
    print("✅ Modelo y frases cargadas correctamente.")
else:
    router = None
    training_phrases = []
    print("⚠️ No se encontró el modelo o frases base. Se usará modo simple.")

def router_predict(message: str) -> str:
    try:
        if not router or not training_phrases:
            print("⚠️ No hay modelo cargado, devolviendo 'complejo'")
            return "complejo"

        vectorizer = router.named_steps['vectorizer']
        X_msg = vectorizer.transform([message])
        X_train = vectorizer.transform(training_phrases)

        if X_msg.nnz == 0:
            print(f"🟠 '{message}' no contiene palabras del vocabulario → COMPLEJO")
            return "complejo"

        sim = cosine_similarity(X_msg, X_train).max()

        # Si la similitud es baja, lo consideramos complejo
        if sim < 0.40:
            print(f"🧠 Mensaje: '{message}' | Pred: complejo | Similitud: {sim:.3f}")
            ##print(f"🔴 Similitud baja ({sim:.3f}) → COMPLEJO")
            return "complejo"
        
        ##pred = router.predict([message])[0]
        print(f"🧠 Mensaje: '{message}' | Pred: simple | Similitud: {sim:.3f}")

        ##print(f"🟢 Clasificado como {pred.upper()}")
        return "simple"

    except Exception as e:
        print(f"⚠️ Error en router_predict: {e}")
        return "complejo"