"""
One-command start:   python run.py
Opens the chatbot at http://127.0.0.1:8000  (API docs: /docs)
"""
import uvicorn

from backend.config import settings

if __name__ == "__main__":
    print(f"\n  Mongolian Health Chatbot -> http://{settings.host}:{settings.port}\n")
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=False)
