"""
Repair upstream encoding artifacts in `2_fin_reasoning.csv` (and a single
residual artifact in `3_fin_toxicity.csv`) where individual characters had
been transcoded into ASCII '?'.

Categorization
--------------
All fixes are confined to known patterns whose correct character is
determinable from surrounding context. Sentence-final '?' (legitimate
question marks) is preserved.

HIGH-CONFIDENCE rules (auto-derivable from convention):
  H1  context : 제YYYY?NN호               -> 제YYYY-NN호        (regulation no.)
  H2  context : 임?직원                    -> 임·직원             (middle dot)
  H3  question: ?전자금융거래법?          -> 「전자금융거래법」 (corner brackets)
  H4  question: ?자본시장법?              -> 「자본시장법」     (corner brackets)
  H5  gold    : YYYY?MM?DD                 -> YYYY-MM-DD         (ISO date)
  H6  gold    : 가?나?다?라?... (목/호)   -> middle dot          (legal enumerator)
  H7  gold    : 별표?N                     -> 별표 N             (space)
  H8  gold    : step?N                     -> step N             (space)
  H9  gold    : 회신 NN?NNNN, 회신(NN?NNNN) -> 회신 NN-NNNN     (doc reference no.)
  H10 gold    : N월?M월?... (month enumeration) -> middle dot
  H11 gold    : M?D부터 ... M?D까지 / (M?D~M?D) (fiscal-year shorthand) -> M.D (period)
  H12 gold    : "step N?M" (CoT sub-step) -> "step N-M" (hyphen)

MEDIUM-CONFIDENCE rules (best-fit chosen after full-context review):
  M1  gold    : 질의?N                     -> 질의 N            (space, matches step N)
  M2  gold    : N?가)                      -> N-가)             (sub-clause hyphen)
  M3  gold    : (민원 정의?법정민원)        -> (민원 정의: 법정민원)  (definition colon)
  M4  gold    : 제22조?4항                 -> 제22조 4항         (space; informal ref)
  M5  gold    : 제10조?13조                -> 제10조~13조        (range tilde)
  M6  gold    : 자?즉,                     -> 자(즉,            (open paren — pair with M7)
  M7  gold    : 자?를                      -> 자)를             (close paren — pair with M6)
  M8  toxicity.question id=39: KOSDAQ?specifically -> KOSDAQ specifically (space)
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path("/home/work/kftc_model/KFinEval")
REASONING_PATHS = [
    ROOT / "_datasets" / "0_integration" / "2_fin_reasoning.csv",
    ROOT / "_manuscript" / "_datasets" / "2_fin_reasoning.csv",
]
TOXICITY_PATHS = [
    ROOT / "_datasets" / "0_integration" / "3_fin_toxicity.csv",
    ROOT / "_manuscript" / "_datasets" / "3_fin_toxicity.csv",
]


# ---------- Reasoning rules ---------------------------------------------------
def fix_reasoning_question(s: str) -> tuple[str, int]:
    if not isinstance(s, str):
        return s, 0
    n = 0
    out, k = re.subn(r"\?전자금융거래법\?", "「전자금융거래법」", s); n += k
    out, k = re.subn(r"\?자본시장법\?", "「자본시장법」", out); n += k
    return out, n


def fix_reasoning_context(s: str) -> tuple[str, int]:
    if not isinstance(s, str):
        return s, 0
    n = 0
    # H1: 제YYYY?NN호  ->  제YYYY-NN호
    out, k = re.subn(r"제(\s*)(\d{2,4})\?(\d{1,4})호", r"제\1\2-\3호", s); n += k
    # H2: 임?직원  ->  임·직원
    out, k = re.subn(r"임\?직원", "임·직원", out); n += k
    return out, n


def fix_reasoning_gold(s: str) -> tuple[str, int]:
    if not isinstance(s, str):
        return s, 0
    n = 0
    out = s

    # H5: full dates YYYY?MM?DD  (apply first so date '?' is not caught by others)
    # Note: do NOT use \b — '\b' fails between an ASCII digit and a Korean char
    # like '에', leaving '2018?11?15에' untouched.
    def date_sub(m):
        return m.group(0).replace("?", "-")
    out, k = re.subn(r"(?<!\d)(\d{4})\?(\d{1,2})\?(\d{1,2})(?!\d)", date_sub, out); n += k

    # H1 (gold extension): 제YYYY?NN호  (also appears inside gold)
    out, k = re.subn(r"제(\s*)(\d{2,4})\?(\d{1,4})호", r"제\1\2-\3호", out); n += k

    # H11: MM?DD fiscal-year shorthand   (M?D부터 ... M?D까지)  ->  M.D부터 ... M.D까지
    # Also covers the range form  (1?1~12?31) inside parens.
    out, k = re.subn(r"(?<!\d)(\d{1,2})\?(\d{1,2})(?=부터|까지|~|\))", r"\1.\2", out); n += k

    # H12: sub-step numbering inside CoT trace.  "step 4?1" -> "step 4-1"
    out, k = re.subn(r"(step\s+\d)\?(\d)", r"\1-\2", out, flags=re.IGNORECASE); n += k

    # H9: 회신 NN?NNNN  and  회신(NN?NNNN)
    out, k = re.subn(r"(회신\s*)(\d{1,2})\?(\d{1,4})", r"\1\2-\3", out); n += k
    out, k = re.subn(r"(회신\()(\d{1,2})\?(\d{1,4})(\))", r"\1\2-\3\4", out); n += k

    # H7: 별표?N  ->  별표 N
    out, k = re.subn(r"별표\?(\d)", r"별표 \1", out); n += k

    # H8: step?N  ->  step N   (covers step?1, step?2)
    out, k = re.subn(r"step\?(\d)", r"step \1", out, flags=re.IGNORECASE); n += k

    # M1: 질의?N  ->  질의 N
    out, k = re.subn(r"질의\?(\d)", r"질의 \1", out); n += k

    # M2: digit?가)  ->  digit-가)
    out, k = re.subn(r"(\d)\?가\)", r"\1-가)", out); n += k

    # M3: (민원 정의?법정민원)  ->  (민원 정의: 법정민원)
    out, k = re.subn(r"\(민원 정의\?법정민원\)", "(민원 정의: 법정민원)", out); n += k

    # M4: 제22조?4항  ->  제22조 4항   (specific: only this article pair)
    out, k = re.subn(r"제22조\?4항", "제22조 4항", out); n += k

    # M5: 제N조?M조  ->  제N조~M조   (range)
    out, k = re.subn(r"제(\d+)조\?(\d+)조", r"제\1조~\2조", out); n += k

    # M6/M7: ...X?즉, ... Y?(를|에)  ->  ...X(즉, ... Y)<particle>   (paren pair)
    # Open anchors observed: 자?즉, 목적?즉,
    out, k = re.subn(r"(자|목적)\?즉,", r"\1(즉,", out); n += k
    # Close anchors observed: 자?를  and  )?에 (where ) is intact closing of inner quote)
    out, k = re.subn(r"자\?를", "자)를", out); n += k
    out, k = re.subn(r"\)\?에", "))에", out); n += k

    # H6: legal enumerator gates 가?나?다?라?...  ->  middle dot.
    # Apply repeatedly because of overlapping matches.
    enum_pat = re.compile(r"(?<=[가나다라마바사아자차])\?(?=[가나다라마바사아자차])")
    while True:
        out, k = enum_pat.subn("·", out)
        n += k
        if k == 0:
            break

    # H10: month enumeration  N월?M월?...  ->  N월·M월·...   (apply repeatedly)
    month_pat = re.compile(r"(?<=월)\?(?=\d{1,2}월)")
    while True:
        out, k = month_pat.subn("·", out)
        n += k
        if k == 0:
            break

    return out, n


def process_reasoning(path: Path) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    totals = {"question": 0, "context": 0, "gold": 0}

    new_q, new_c, new_g = [], [], []
    for q, c, g in zip(df["question"].astype(str), df["context"].astype(str), df["gold"].astype(str)):
        q2, n1 = fix_reasoning_question(q); totals["question"] += n1; new_q.append(q2)
        c2, n2 = fix_reasoning_context(c); totals["context"] += n2; new_c.append(c2)
        g2, n3 = fix_reasoning_gold(g); totals["gold"] += n3; new_g.append(g2)
    df["question"] = new_q
    df["context"] = new_c
    df["gold"] = new_g

    df.to_csv(path, index=False, encoding="utf-8-sig")
    return totals


# ---------- Toxicity residual fix (M8) ---------------------------------------
def fix_toxicity_question(path: Path) -> int:
    df = pd.read_csv(path, encoding="utf-8-sig")
    new_vals = []
    total = 0
    for v in df["question"].astype(str):
        new_v, k = re.subn(r"KOSDAQ\?specifically", "KOSDAQ specifically", v)
        total += k
        new_vals.append(new_v)
    df["question"] = new_vals
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return total


if __name__ == "__main__":
    for p in REASONING_PATHS:
        print(f"\n[Reasoning] {p}")
        t = process_reasoning(p)
        print(f"  question subs : {t['question']}")
        print(f"  context  subs : {t['context']}")
        print(f"  gold     subs : {t['gold']}")
    for p in TOXICITY_PATHS:
        print(f"\n[Toxicity] {p}")
        n = fix_toxicity_question(p)
        print(f"  question subs (KOSDAQ): {n}")
    print("\nDone.")
