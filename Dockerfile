FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY src/ src/
COPY configs/ configs/

# Create data directory expected by the app at runtime
RUN mkdir -p data/raw data/processed

# Pass ANTHROPIC_API_KEY at runtime via --env or --env-file
# docker run --env-file .env ag-news-classifier
CMD ["python", "src/app.py"]
