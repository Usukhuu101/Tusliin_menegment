FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    HOST=0.0.0.0 PORT=8000 APP_ENV=production

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY data ./data
COPY evaluation ./evaluation
COPY run.py .

# Fail the image build if the safety gate fails (emergency recall, safety checks)
RUN APP_ENV=development python evaluation/run_eval.py

RUN useradd --create-home appuser && mkdir -p /app/storage && chown -R appuser /app/storage
USER appuser
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1
CMD ["python", "run.py"]
