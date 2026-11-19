"""
Weekly review report for the team (Quality Management):
  - knowledge gaps: questions the bot could NOT answer -> what to write next
  - negative feedback: answers users marked 👎 (needs human review)
  - message-type mix

Run:  python scripts/report.py
Feedback is NEVER added to the knowledge base automatically — a human must
review it, write/fix a document, and run the indexer + evaluation.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import settings  # noqa: E402
from backend.database.db import Database  # noqa: E402

if not settings.db_path.exists():
    print("No database yet — chat with the bot first.")
    sys.exit(0)
db = Database(settings.db_path)
print("== Stats ==")
print(json.dumps(db.stats(), ensure_ascii=False, indent=2))
r = db.report()
print("\n== Message types ==")
for k, v in r["message_types"].items():
    print(f"  {k:<14}{v}")
print("\n== Top knowledge gaps (write these topics next) ==")
for g in r["knowledge_gaps"]:
    print(f"  {g['n']:>3}x  {g['query_redacted']}   concepts={g['concepts_json']}")
print("\n== Negative feedback (review these answers) ==")
for f in r["negative_feedback"]:
    print(f"  [{f['reason']}] {f['user_text_redacted']!r} -> {f['message_type']}/{f['risk_level']}  {f['comment'] or ''}")
