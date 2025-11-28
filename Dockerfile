FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/nltk_data

RUN python -c "import nltk; \
    import ssl; \
    ssl._create_default_https_context = ssl._create_unverified_context; \
    nltk.download('punkt', download_dir='/root/nltk_data'); \
    nltk.download('punkt_tab', download_dir='/root/nltk_data'); \
    nltk.download('stopwords', download_dir='/root/nltk_data'); \
    nltk.download('averaged_perceptron_tagger', download_dir='/root/nltk_data')"

COPY run.py .
COPY train_dummy_router.py .
COPY app/ ./app/
COPY data/ ./data/

ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=run.py
ENV NLTK_DATA=/root/nltk_data

EXPOSE 5001

# CMD ["python", "run.py"]
CMD ["bash", "-c", "python train_dummy_router.py && python run.py"]