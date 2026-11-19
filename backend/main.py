"""
FastAPI application entry point.

Run (from the project folder):
    python run.py
or
    uvicorn backend.main:app --reload
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .config import ROOT_DIR, settings
from .services.orchestrator import ChatOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("health-chatbot")
FRONTEND = ROOT_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    orch = ChatOrchestrator(settings)
    app.state.orchestrator = orch
    log.info("Ready | mode=%s | index=%s | docs=%s | statuses=%s", orch.mode, orch.index.name,
             orch.index.meta["doc_count"], ",".join(settings.kb_allowed_statuses))
    for err in orch.index_info.get("errors", []):
        log.warning("KB problem: %s", err)
    yield


app = FastAPI(title="Mongolian Health Chatbot", version="0.1.0", lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(Path(FRONTEND) / "index.html")
