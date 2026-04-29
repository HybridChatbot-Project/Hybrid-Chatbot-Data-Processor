FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-docker.txt

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /install /usr/local

RUN mkdir -p /root/nltk_data
ENV NLTK_DATA=/root/nltk_data

RUN python -c "import nltk; \
    nltk.download('punkt', download_dir='/root/nltk_data'); \
    nltk.download('punkt_tab', download_dir='/root/nltk_data'); \
    nltk.download('stopwords', download_dir='/root/nltk_data')"

COPY run.py .
COPY train_dummy_router.py .
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=run.py

EXPOSE 5000

CMD ["sh", "-c", "if [ ! -f models/router_pipeline.joblib ]; then python train_dummy_router.py; fi && python run.py"]