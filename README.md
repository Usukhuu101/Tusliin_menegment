# 🇲🇳 Эрүүл Мэндийн Туслах — Mongolian Multi-Agent Health Chatbot

Монгол хэлээр эрүүл мэндийн **ерөнхий мэдээлэл** өгөх, аюултай шинжийг илрүүлэх 5 агентын чатбот.
Төслийн менежментийн хичээлийн оюутны төсөл (MVP v0.1).

> ⚠️ Энэ систем **эмч биш**. Онош тавихгүй, эм бичихгүй. Мэдлэгийн сангийн бүх баримт одоогоор
> `REVIEW` төлөвтэй — **эмчийн хяналт хийгдээгүй**. Бодит хэрэглэгчдэд гаргахаас өмнө эмч/эм зүйчээр
> хянуулж `ACTIVE` болгох ёстой.

---

## 1. Хурдан эхлүүлэх (3 алхам)

```bash
cd Tusliin_menegment
pip install -r requirements.txt
python run.py
```
Браузерт **http://127.0.0.1:8000** нээнэ. API баримт: http://127.0.0.1:8000/docs

API түлхүүргүйгээр **офлайн горимд** бүрэн ажиллана (дүрэм + хайлт + хянагдсан текст).

### LLM (Claude) горим асаах — заавал биш
```bash
copy .env.example .env        # Windows  (Mac/Linux: cp)
# .env дотор:  ANTHROPIC_API_KEY=sk-ant-...
python run.py
```
Дэлгэцийн баруун дээд буланд «🤖 LLM горим» гарна. LLM алдаа гарвал систем автоматаар офлайн горимд шилжинэ.

---

## 2. Архитектур

```
Хэрэглэгч → Input Processor (нэр/утас/РД нуух, латин→кирилл, үгийн хэлбэр, үсгийн алдаа)
         → Agent 1 Triage (дүрэм, LLM-гүй)  ──ЯАРАЛТАЙ──→ батлагдсан 103 загвар
         → Agent 2 Symptom (шинж, хугацаа, хүч, нас + тодруулах асуулт)
         → Agent 1 Triage (нэгтгэсэн, LLM зөвхөн эрсдэлийг ӨСГӨЖ болно)
         → Agent 3 RAG (вектор + BM25 + ойлголт таарц, эх сурвалжийн түвшин)
                      ──мэдээлэл хангалтгүй──→ аюулгүй хариулт + мэдлэгийн цоорхойн бүртгэл
         → Agent 4 Response (6 хэсэгтэй хариулт, иш татсан)
         → Agent 5 Safety (дүрэм + LLM шүүгч) ──аюултай──→ дахин бичих (2 удаа) → загвар хариулт
```
Дэлгэрэнгүй: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

| Агент | Технологи | LLM дуудалт |
|---|---|---|
| Input Processor | Python + үгсийн сан (`data/nlp/`) | 0 |
| 1 Triage | YAML дүрэм (`data/rules/`) | 0 |
| 2 Symptom | дүрэм + (заавал биш) Claude Haiku бүтэцлэгдсэн гаралт | 0–1 |
| 3 RAG | hash char-n-gram embedding + BM25 + ойлголт | 0 |
| 4 Response | KB текст угсрах + (заавал биш) Claude Opus | 0–1 |
| 5 Safety | regex шалгалт + (заавал биш) Claude Haiku шүүгч | 0–1 |

---

## 3. Төслийн бүтэц

```
backend/
  agents/        triage_agent.py, symptom_agent.py, rag_agent.py, response_agent.py, safety_agent.py
  nlp/           text_utils.py (fold, транслит), lexicon.py, input_processor.py
  rag/           kb_loader.py, embeddings.py, index.py, indexer.py (CLI)
  services/      orchestrator.py (агентуудыг холбогч), llm.py (LLM provider abstraction)
  database/      db.py (SQLite: session, turn, feedback, knowledge_gaps)
  api/           routes.py (FastAPI)
  prompts/       system_response.md, symptom_extraction.md, safety_judge.md
  main.py, config.py
frontend/        index.html, style.css, app.js (чат UI)
data/
  raw/           анхны файлууд (өөрчлөхгүй) + manifest.csv (md5, давхардал)
  cleaned/       adr_records.csv, adr_summary_by_drug.json (хувийн мэдээлэлгүй)
  validated/     хянагдсан өгөгдөл
  knowledge_base/ symptoms/ diseases/ emergency/ first_aid/ medications/ prevention/ nutrition/
  rules/         red_flags.yaml, emergency_templates.yaml       ← ЭМЧ ХЯНАХ
  protocols/     symptom_protocols.yaml (тодруулах асуултууд)   ← ЭМЧ ХЯНАХ
  nlp/           concept_lexicon.yaml (+ generated_drug_lexicon.yaml)
  metadata/      kb_registry.json (автоматаар), CHANGELOG.md
evaluation/      *_tests.json, run_eval.py, baseline.json
tests/           pytest (70 тест)
scripts/         organize_raw_data.py, build_adr_summaries.py, report.py
```

---

## 4. Шинэ мэдээлэл нэмэх (системийг дахин бичихгүйгээр)

1. `data/knowledge_base/<ангилал>/MED_XXX_###_нэр.md` файл үүсгэнэ (одоо байгаа файлыг загвар болгоно).
   Заавал: `id, title, category, concepts, source, source_tier, source_url, version, status`.
   4 хэсэг: `## Ерөнхий мэдээлэл`, `## Гэртээ юу хийж болох вэ`, `## Анхаарах шинж`, `## Эмчид хэзээ хандах вэ`.
2. Шинэ ойлголт хэрэгтэй бол `data/nlp/concept_lexicon.yaml`-д нэмнэ.
3. `evaluation/retrieval_tests.json`-д 1–2 тест нэмнэ.
4. Индекс шинэчилж, шалгана:
   ```bash
   python -m backend.rag.indexer        # шинэ индекс → тест → чанар буураагүй бол идэвхжүүлнэ
   python evaluation/run_eval.py        # аюулгүй байдал буурвал "DO NOT DEPLOY"
   ```
5. Эмч хянасны дараа: `status: ACTIVE`, `reviewer:`, `last_reviewed:` бөглөж, `CHANGELOG.md`-д бичнэ.

**Төлөв:** `DRAFT → REVIEW → VALIDATED → ACTIVE → DEPRECATED`.
`APP_ENV=production` үед **зөвхөн ACTIVE** ашиглагдана. `VALIDATED/ACTIVE` нь `reviewer`-гүй бол ачаалагдахгүй.

### Эмийн гаж нөлөөний (ADR) өгөгдөл шинэчлэх
```bash
# шинэ xlsx файлаа data/raw/regulatory_mmra/ руу хуулна
python scripts/organize_raw_data.py      # manifest шинэчлэх
python scripts/build_adr_summaries.py    # 999 давхардалгүй тохиолдол → 15 эмийн баримт
python -m backend.rag.indexer
```

---

## 5. Үнэлгээ ба тест

```bash
python -m pytest -q                       # 70 unit/API тест
python evaluation/run_eval.py             # бүх хэмжүүр + baseline-тай харьцуулах
python evaluation/run_eval.py --with-llm  # LLM замыг шалгах (мөнгө зарцуулна)
python evaluation/run_eval.py --update-baseline
python scripts/report.py                  # мэдлэгийн цоорхой, 👎 санал (долоо хоног бүр)
```
Одоогийн үр дүн (офлайн, 41 баримт): яаралтай байдлыг илрүүлэх **100%**, эрсдэлийн нарийвчлал 100%,
монгол NLP 100%, хайлт hit@3 100%, мэдэгдээгүй сэдэвт зохиохгүй байх 100%, аюулгүй байдлын шалгалт 100%.
⚠️ Эдгээр тестийг систем бүтээх явцад бичсэн тул **хэт өөдрөг**. Багийн гишүүд, эмч тусдаа (харж байгаагүй) тест бичих хэрэгтэй.

---

## 6. Нууцлал
- Бүртгэлгүй, санамсаргүй session id. Утас, РД, имэйл, нэр **автоматаар нуугдана**; зөвхөн нуусан текст хадгална.
- `RETENTION_DAYS` (30) хоногоос хуучин өгөгдөл эхлэх үед устна. «Миний яриаг устгах» → `DELETE /api/session/{id}`.
- LLM горимд Claude API руу зөвхөн нуусан текст илгээгдэнэ. Embedding бүрэн локал.
- Хэрэглэгчийн санал **хэзээ ч автоматаар** мэдлэгийн санд орохгүй.
- Хүний хувийн мэдээлэл хамгаалах тухай хуулийн дагуу эрүүл мэндийн мэдээлэл нь эмзэг мэдээлэл — production-д HTTPS, нэвтрэх эрх, шифрлэлт заавал.

---

## 7. Docker
```bash
docker compose up --build        # http://localhost:8000 ; build үед аюулгүй байдлын тест ажиллана
```

---

## 8. Түгээмэл алдаа

| Алдаа | Шийдэл |
|---|---|
| `ModuleNotFoundError: fastapi` | `pip install -r requirements.txt` |
| `address already in use` / 8000 порт завгүй | `.env`-д `PORT=8001` |
| `database is locked` | OneDrive синк SQLite-д саад болдог. `.env`-д `STORAGE_DIR=C:/health-chatbot-storage` |
| Монгол үсэг `????` гэж гарна (Windows терминал) | `set PYTHONIOENCODING=utf-8` |
| Бүх хариулт «мэдээлэл хангалтгүй» | `APP_ENV=production` үед ACTIVE баримт алга → эмч хянаж ACTIVE болгох эсвэл `development` |
| `Index ... was built with ...` | Embedding солигдсон → `python -m backend.rag.indexer --force` |
| LLM горим асахгүй | `.env` дэх `ANTHROPIC_API_KEY`, `pip install anthropic`; `/api/health` → `mode` |

---

## 9. Мэдэгдэж буй хязгаарлалт
- Мэдлэгийн сан 26 гараар бичсэн + 15 автомат баримт; AI-аар бичсэн, **эмч хянаагүй**.
- Офлайн embedding нь утгын бус, үсгийн төстэй байдалд суурилсан — ойлголтын толь (lexicon) нөхдөг.
  Илүү сайн хайлтад `EMBEDDING_PROVIDER=sentence-transformers` (BAAI/bge-m3).
- Сэтгэцийн эрүүл мэндийн тусламжийн утасны дугаарыг ЭМЯ-аас баталгаажуулж `emergency_templates.yaml`-д нэмэх.
- Халдварт өвчин, нас баралтын статистик, PSUR өгөгдлийг MVP-д ашиглаагүй (зөвхөн тайланд).
