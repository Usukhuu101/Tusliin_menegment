"""HTTP API. All medical logic lives in the orchestrator/agents, not here."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..models.schemas import ChatRequest, ChatResponse, FeedbackRequest

router = APIRouter(prefix="/api")


def _orch(request: Request):
    orch = getattr(request.app.state, "orchestrator", None)
    if orch is None:
        raise HTTPException(503, "Chatbot is starting, try again in a moment")
    return orch


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    message = body.message.strip()
    if not message:
        raise HTTPException(422, "Empty message")
    return _orch(request).handle(message, body.session_id)


@router.post("/feedback")
def feedback(body: FeedbackRequest, request: Request) -> dict:
    orch = _orch(request)
    if not orch.db.turn_belongs_to(body.turn_id, body.session_id):
        raise HTTPException(404, "Unknown turn")
    # Stored for evaluation only. NEVER written into the knowledge base automatically.
    fid = orch.db.add_feedback(session_id=body.session_id, turn_id=body.turn_id, helpful=body.helpful,
                               reason=body.reason, comment=(body.comment or "").strip()[:500] or None)
    return {"ok": True, "feedback_id": fid}


@router.delete("/session/{session_id}")
def delete_session(session_id: str, request: Request) -> dict:
    deleted = _orch(request).db.delete_session(session_id)
    return {"ok": True, "deleted_turns": deleted}


@router.get("/health")
def health(request: Request) -> dict:
    orch = _orch(request)
    return {
        "status": "ok",
        "mode": orch.mode,
        "app_env": orch.s.app_env,
        "kb_index": orch.index.name,
        "kb_docs": orch.index.meta["doc_count"],
        "kb_statuses": list(orch.s.kb_allowed_statuses),
        "retrieval": orch.index.meta.get("retrieval"),
        "versions": orch.versions,
        "db": orch.db.stats(),
    }
