"""
Fix HTML entity encoding and a small number of '?' replacement-char artifacts
in the Financial Toxicity dataset.

Issue
-----
`source_news_content` (96/100 rows) contains unescaped HTML entities such as
`&nbsp;`, `&lsquo;`, `&rsquo;`, `&#39;`, `&amp;`, `&hellip;`, `&middot;`,
`&quot;`, `&rarr;`, `&rdquo;`, `&sim;` left over from the upstream news
ingestion. A separate upstream artifact also caused some middle-dot
characters to be transcoded into ASCII '?' inside titles/contents (e.g.
`삼성?하나?비씨카드`).

Fix
----
1. `html.unescape(...)` on both `source_news_title` and `source_news_content`.
2. Heuristic '?' -> '·' (middle dot) restoration ONLY where both the
   preceding and following characters are non-whitespace and non-punctuation.
   This preserves legitimate question marks (e.g. `'R'그게 뭔데?`) while
   restoring list-style separators (e.g. `200만?300만원` -> `200만·300만원`).

Targets two synced copies of the dataset:
  - _datasets/0_integration/3_fin_toxicity.csv     (Fin-Ben, root repo)
  - _manuscript/_datasets/3_fin_toxicity.csv       (KFinEval-paper repo)
"""

import html
import re
from pathlib import Path

import pandas as pd

ROOT = Path("/home/work/kftc_model/KFinEval")
TARGETS = [
    ROOT / "_datasets" / "0_integration" / "3_fin_toxicity.csv",
    ROOT / "_manuscript" / "_datasets" / "3_fin_toxicity.csv",
]
TEXT_COLS = ["source_news_title", "source_news_content"]

# Replace '?' with middle dot only when sandwiched between non-whitespace,
# non-punctuation glyphs. Avoids touching sentence-ending '?'.
QMARK_RESTORE = re.compile(r"(?<=[^\s\?\.\!\,\;\:])\?(?=[^\s\?\.\!\,\;\:])")
ENTITY_PAT = re.compile(r"&[a-zA-Z]+;|&#\d+;")


def clean_text(s: str) -> tuple[str, dict]:
    stats = {"entity_subs": 0, "qmark_subs": 0, "nbsp_subs": 0}
    if not isinstance(s, str):
        return s, stats
    stats["entity_subs"] = len(ENTITY_PAT.findall(s))
    out = html.unescape(s)
    new_out, n = QMARK_RESTORE.subn("·", out)
    stats["qmark_subs"] = n
    # Normalize U+00A0 (non-breaking space, mostly from &nbsp;) to ASCII space
    # so downstream tokenization treats it identically to regular whitespace.
    nbsp_count = new_out.count(" ")
    if nbsp_count:
        new_out = new_out.replace(" ", " ")
        stats["nbsp_subs"] = nbsp_count
    return new_out, stats


def process_file(path: Path) -> None:
    print(f"\n[Processing] {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    totals = {"entity_subs": 0, "qmark_subs": 0, "nbsp_subs": 0}
    affected_rows = {c: 0 for c in TEXT_COLS}
    for col in TEXT_COLS:
        new_vals = []
        for v in df[col].astype(str).tolist():
            new_v, st = clean_text(v)
            if st["entity_subs"] or st["qmark_subs"] or st["nbsp_subs"]:
                affected_rows[col] += 1
            for k in totals:
                totals[k] += st[k]
            new_vals.append(new_v)
        df[col] = new_vals

    # Re-verify: no entities should remain
    leftover = 0
    for col in TEXT_COLS:
        leftover += df[col].astype(str).apply(lambda v: bool(ENTITY_PAT.search(v))).sum()

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  entity substitutions : {totals['entity_subs']}")
    print(f"  '?' -> middle-dot    : {totals['qmark_subs']}")
    print(f"  U+00A0 -> ASCII space: {totals['nbsp_subs']}")
    print(f"  affected rows / col  : {affected_rows}")
    print(f"  remaining entities   : {leftover}")


if __name__ == "__main__":
    for p in TARGETS:
        process_file(p)
    print("\nDone.")
