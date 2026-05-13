# Supplementary Analyses for KFinEval

Post-hoc statistical analyses and ablation experiments that complement the main
benchmark pipeline in `eval/`. These materials support claims made in the
manuscript (Appendix, Section 4) and were also used to address reviewer
requests during the KDD 2026 peer review (see `_manuscript/rebuttal/` for the
rebuttal record that originally accompanied these experiments).

## Directory Layout

| Path | Purpose | Manuscript Link |
|---|---|---|
| `wilson_ci/` | Wilson Score 95% CI for per-category Financial Knowledge accuracy | Appendix `app:reliability_stats`, Table `tab:app_ci_width_knowledge` |
| `bootstrap_ci/` | Bootstrap 95% CI for per-category scores (Knowledge / Reasoning / Toxicity) | Appendix reference; source for Wilson comparison |
| `ranking_stability/` | Kendall's $\tau$ bootstrap ranking stability (see `mcnemar_think_mode/verify_w7s5_all.py`, `verify_w3`) | Appendix `app:reliability_stats` (Ranking Stability) |
| `mcnemar_think_mode/` | McNemar's test for Standard vs Think mode + Think case studies | Section `sec:think_mode`, Table `tab:mcnemar_think`; Appendix `app:think_case_studies` |
| `llama_judge/` | Llama-3.1-70B third-judge cross-validation | Section 4 (RQ4) supporting evidence |
| `context_analysis/` | Context-configuration ablation for Financial Reasoning | Section 4.2 (reasoning variance) |
| `scoring_sensitivity/` | Weight-scheme sensitivity on the overall-score ranking | Appendix reliability discussion |
| `dataset_cleanup/` | One-off dataset hygiene scripts (e.g. HTML-entity / NBSP normalization on `3_fin_toxicity.csv`) | Dataset release notes |

## Running

Each subdirectory is self-contained. Scripts assume benchmark outputs live
under `eval/_results/` (per-domain CSV files produced by the
`1_*`, `2_*`, `3_*` pipelines in `eval/`).

Example:

```bash
cd eval/supplementary/wilson_ci
python wilson_ci.py    # writes wilson_ci_knowledge.csv + wilson_ci_summary.md
```

## Provenance

The original copies of these experiments, as submitted during the KDD 2026
rebuttal, remain in `_manuscript/rebuttal/exp/`. Changes made after the
rebuttal round should be applied here (`eval/supplementary/`), not to the
rebuttal-side tree, to keep the rebuttal record immutable while letting the
camera-ready / next-submission analyses evolve.
