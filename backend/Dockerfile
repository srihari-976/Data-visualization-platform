FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y gcc && apt-get autoremove -y

COPY backend/ .

RUN mkdir -p uploads results data

EXPOSE 5000

ENV FLASK_ENV=production

CMD gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 main:app
