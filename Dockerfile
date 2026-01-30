FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py pipeline.py ./
COPY templates templates/

RUN mkdir -p /app/data /models

EXPOSE 5001

CMD ["python", "app.py"]
