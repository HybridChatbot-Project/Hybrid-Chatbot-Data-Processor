## Evaluación rápida del router (simple vs. complejo)

- Modelo: `models/router_pipeline.joblib` (CountVectorizer + MultinomialNB).
- Dataset de prueba: preguntas de `data/faq.csv` (categorías `ansiedad` y `personal` → `simple`; otras categorías + 6 frases fuera de dominio → `complejo`).

### Comando
Con el entorno virtual activo:
```bash
python - <<'PY'
import csv, joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = Path('models/router_pipeline.joblib')
FAQ_PATH = Path('data/faq.csv')
simple_cats = {'ansiedad', 'personal'}

router = joblib.load(MODEL_PATH)

simple, complex_ = [], []
with FAQ_PATH.open(encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = (row.get('question') or '').strip()
        cat = (row.get('category') or '').strip().lower()
        if not q:
            continue
        (simple if cat in simple_cats else complex_).append(q)

complex_extra = [
    "escribe un ensayo profundo sobre la mente humana",
    "necesito código python para analizar datos financieros",
    "cuéntame una historia original de ciencia ficción",
    "explícame cómo funciona la teoría de cuerdas",
    "puedes resumir un libro completo",
    "dame un plan de marketing detallado"
]

X = simple + complex_ + complex_extra
y_true = ['simple'] * len(simple) + ['complejo'] * (len(complex_) + len(complex_extra))

pred = router.predict(X)
acc = accuracy_score(y_true, pred)
print(f"Muestras: {len(X)} | Accuracy: {acc:.3f}")
print(classification_report(y_true, pred, digits=3))
PY
```

### Resultado (corrida de referencia)
- Muestras: 219
- Accuracy: 0.945
- Precision/Recall/F1:
  - `complejo`: prec 0.957, rec 0.941, f1 0.949 (support 118)
  - `simple`: prec 0.932, rec 0.950, f1 0.941 (support 101)
  - Macro F1: 0.945

Vuelve a ejecutar el comando si cambias `data/faq.csv` o reentrenas el router.***
