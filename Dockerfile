# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/share/nltk_data

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt \
 && python -m nltk.downloader -q -d /usr/share/nltk_data punkt punkt_tab stopwords

COPY run.py .
COPY app ./app
COPY data ./data
COPY models/*.joblib ./models/

EXPOSE 5001
# One worker: conversation context is kept in process memory.
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "8", "--timeout", "300", "run:app"]
