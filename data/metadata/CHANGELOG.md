# Knowledge-base changelog
Record every content change: date, document id, old -> new version/status, who reviewed it.

| Date | Document | Change | Status | Reviewer |
|---|---|---|---|---|
| 2026-11-19 | MED_SYM_001–012, MED_EMG_001–005, MED_DIS_001–006, MED_FA_001–002, MED_MED_001 | v1.0 created (AI-assisted Mongolian adaptation of WHO/NHS) | REVIEW | — (pending) |
| 2026-11-19 | MED_ADR_001–015 | v1.0 generated from 999 de-duplicated MMRA ADR reports (2018–2026) | REVIEW | — (pending) |
| 2026-11-19 | MED_PRV_001–005 (air pollution, HPV vaccine, violence against children, stress after disasters, tobacco) | v1.0 created from WHO Mongolia news (data/raw/who, cleaned in data/cleaned/who_news.json) + WHO fact sheets | REVIEW | — (pending) |
| 2026-11-19 | MED_DIS_004 Улаанбурхан | v1.0 → v1.1: outbreak transmission interrupted (MOH/WHO 17 Nov 2026), last case 24 Aug 2026, check vaccination status | REVIEW | — (pending) |
| 2026-11-19 | MED_DIS_002 Даралт ихсэлт | v1.0 → v1.1: WHO Western Pacific — 1 in 3 adults have hypertension | REVIEW | — (pending) |
| 2026-11-19 | Lexicon | + MedlinePlus synonyms (generated_medlineplus_lexicon.yaml: 10 concepts enriched, 121 topics recognised); + air_pollution, smoke_exposure, hpv, tobacco, stress, disaster, violence | — | — |
| 2026-11-19 | Rules | + RF_CO_SMOKE_SYMPTOMS (smoke alone no longer = emergency), RF_SCREEN_POSITIVE(_CRISIS) ("тийм" to a screening question raises risk), RF_VIOLENCE | — | needs medical review |
