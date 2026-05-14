# Pre-Cleanup Toxicity Results Archive (2026-05-14)

This directory contains the **toxicity benchmark results as they existed before
the dataset cleanup pass on 2026-05-13**. They are preserved for diff/audit
purposes only — the current paper results in `../3_fin_toxicity/` will be
overwritten by the re-experiment.

## Cleanup commits that motivated the re-experiment

| Commit | What it changed |
|---|---|
| `f465ff5` | HTML entities (`&nbsp;`, `&lsquo;`, `&rsquo;`, `&#39;`, `&amp;`, `&hellip;`, `&middot;`, `&quot;`, `&rarr;`, `&rdquo;`, `&sim;`) decoded in `3_fin_toxicity.csv`; U+00A0 → ASCII space; 16 `?` → `·` middle-dot restorations. |
| `5b51a25` | One residual `?` artifact in `3_fin_toxicity.csv` repaired (in addition to `2_fin_reasoning.csv`). |

## Affected rows

96 of 100 toxicity rows had at least one model-input column
(`source_news_title`, `source_news_content`, or `question`) changed.
The 4 untouched rows are IDs 33, 43, 51, 68. See
`../../supplementary/dataset_cleanup/AFFECTED_IDS.md` § 2 for full lists.

## Source dataset hash at re-experiment time

```
sha256: c68bd469cf36387424cae98a8f0f026508982e5099bf2eb6e908d8d28df26807
file:   _datasets/0_integration/3_fin_toxicity.csv  (100 data rows)
```

## Do not edit

This directory should remain a frozen snapshot. If a comparison utility
is needed (e.g. score-delta plots), the rerun manifest at
`../../3_fin_toxicity_rerun_manifest.csv` is the canonical pointer.
