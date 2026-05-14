# Dataset Cleanup — Affected Benchmark IDs

This document records the benchmark IDs whose source rows were modified by the
post-hoc cleanup pass performed on 2026-05-13 (see commits
`f465ff5`, `5b51a25` in Fin-Ben and `783426a`, `09c45ea` in KFinEval-paper).

The purpose is to scope any re-experimentation:

- A row whose **model-input column** (e.g., `question`, `context`,
  `source_news_title`, `source_news_content`) changed needs the affected
  models re-run on that row.
- A row whose **reference column** (`gold`) changed needs only re-scoring
  (the model's prior output is still valid, but the answer key it is graded
  against has been corrected).

The cleanup scripts that produced these diffs are
`fix_toxicity_entities.py` and `fix_reasoning_artifacts.py` in this
directory.

---

## 1. `2_fin_reasoning.csv` (total 575 rows)

| Column | Role | Rows changed | Substitutions |
|---|---|---|---|
| `question` | model input | 13 | 13 |
| `context`  | model input | 4 | 18 |
| `gold`     | reference (scoring) | 86 | 170 |
| **Union (distinct rows)** | — | **95 / 575 (16.5%)** | **201** |

### 1.1 `question` — 13 IDs (model input → re-run required)
```
156, 157, 158, 159, 160,
216, 217, 218, 219, 220,
450, 462, 565
```

Rule families applied: H3 (`?전자금융거래법?` → `「전자금융거래법」`),
H4 (`?자본시장법?` → `「자본시장법」`).

### 1.2 `context` — 4 IDs (model input → re-run required)
```
353, 375, 491, 513
```

Rule families applied: H1 (`제YYYY?NN호` → `제YYYY-NN호`),
H2 (`임?직원` → `임·직원`).

### 1.3 `gold` — 86 IDs (reference → re-scoring sufficient)
```
101, 102, 103, 104, 105,
151, 152, 153, 154, 155,
186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
196, 197, 198, 199, 200,
216, 217, 218, 219, 220,
236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
246, 247, 248, 249, 250,
311, 312, 313, 314, 315,
357, 363, 374, 375, 384,
386, 387, 389, 393, 397,
415, 439, 449, 456, 457,
458, 462, 466, 467, 468,
481, 495, 501, 512, 513,
522, 524, 525, 527, 531,
535, 553, 561, 564, 568, 575
```

Rule families applied: H1 (gold ext.), H5 (dates), H6 (legal enumerators),
H7 (별표 N), H8 (step N), H9 (회신 numbers), H10 (month enumeration),
H11 (M.D fiscal-year shorthand), H12 (step N-M), M1–M7.

### 1.4 Union — 95 distinct row IDs touched
```
101, 102, 103, 104, 105,
151, 152, 153, 154, 155,
156, 157, 158, 159, 160,
186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
196, 197, 198, 199, 200,
216, 217, 218, 219, 220,
236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
246, 247, 248, 249, 250,
311, 312, 313, 314, 315,
353, 357, 363, 374, 375,
384, 386, 387, 389, 393,
397, 415, 439, 449, 450,
456, 457, 458, 462, 466,
467, 468, 481, 491, 495,
501, 512, 513, 522, 524,
525, 527, 531, 535, 553,
561, 564, 565, 568, 575
```

### 1.5 Subset that requires model re-evaluation (question ∪ context, 17 IDs)
```
156, 157, 158, 159, 160,
216, 217, 218, 219, 220,
353, 375, 450, 462, 491, 513, 565
```

These rows had their model-input text changed; prior model outputs for these
17 IDs are stale and should be regenerated before scoring.

---

## 2. `3_fin_toxicity.csv` (total 100 rows)

| Column | Role | Rows changed | Substitutions |
|---|---|---|---|
| `source_news_title`   | model input (prompt context) | 2 | — |
| `source_news_content` | model input (prompt context) | 96 | — |
| `question` | model input | 1 | 1 |
| **Union (distinct rows)** | — | **96 / 100 (96.0%)** | **~1,990** |

Substitution mix on the content columns: 1,611 HTML-entity decodings,
380 U+00A0 → ASCII-space normalizations, 16 `?` → `·` middle-dot
restorations, plus 1 KOSDAQ question fix (Section 2.3). Per-row
substitution counts are not separately tracked; full row-level diffs are
reproducible from the cleanup script.

### 2.1 `source_news_title` — 2 IDs
```
1, 63
```

### 2.2 `source_news_content` — 96 IDs (all rows except 33, 43, 51, 68)
```
1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
31, 32, 34, 35, 36, 37, 38, 39, 40, 41,
42, 44, 45, 46, 47, 48, 49, 50, 52, 53,
54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
64, 65, 66, 67, 69, 70, 71, 72, 73, 74,
75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
95, 96, 97, 98, 99, 100
```

Rule families applied: HTML-entity decoding (`&nbsp;`, `&lsquo;`,
`&rsquo;`, `&#39;`, `&amp;`, `&hellip;`, `&middot;`, `&quot;`, `&rarr;`,
`&rdquo;`, `&sim;`), U+00A0 → ASCII space normalization, and 16
`?` → `·` middle-dot restorations.

### 2.3 `question` — 1 ID
```
39
```

Rule applied: `KOSDAQ?specifically` → `KOSDAQ specifically`.

### 2.4 Union — 96 distinct row IDs touched (out of 100)
```
1–32, 34–42, 44–50, 52–67, 69–100
```
(missing: 33, 43, 51, 68 — unaffected.)

### 2.5 Subset that requires model re-evaluation
Depends on the toxicity prompt template:

- If the prompt embeds `source_news_title` and/or `source_news_content`
  alongside `question`, re-run is needed for all 96 union IDs.
- If only `question` is fed to the model, re-run is needed only for
  ID 39.

Inspect `eval/3_1_gen_toxicity_*.py` to confirm which columns enter the
prompt before deciding scope.

---

## 3. Untouched datasets

- **`1_fin_knowledge.csv`** (296 rows): scanned, no artifact-style `?`
  found — all 42 `?` occurrences are legitimate sentence-final question
  marks in question stems (e.g., `… 얼마인가?(단, …)`). No changes
  applied. No re-experimentation needed.

---

## 4. Reproducibility

To regenerate the exact same fixes from raw upstream data:

```bash
cd eval/supplementary/dataset_cleanup
python fix_toxicity_entities.py
python fix_reasoning_artifacts.py
```

Both scripts are idempotent on already-clean data.
