"""
Verify all w7S5 rebuttal numbers:
1. Bootstrap CI for correlation coefficients (W1)
2. gpt-5 McNemar + category-level improvement + reasoning domain metrics (W2)
3. Wilson score intervals + Kendall's tau ranking stability (W3)
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import chi2
from pathlib import Path
import glob

RESULTS_DIR = Path("/home/work/kftc_model/KFinEval/eval/_results")

###############################################
# W1: Bootstrap CI for correlation coefficients
###############################################
def verify_w1():
    print("=" * 60)
    print("W1: Bootstrap CI for Correlation Coefficients")
    print("=" * 60)

    knowledge_dir = RESULTS_DIR / "1_fin_knowledge"
    reasoning_dir = RESULTS_DIR / "2_fin_reasoning"
    toxicity_dir = RESULTS_DIR / "3_fin_toxicity"

    # Collect per-model scores from CSV files directly
    # Knowledge: accuracy
    k_models = {}
    for f in sorted(knowledge_dir.glob("*_response.csv")):
        model = f.stem.replace("1_fin_knowledge_", "").replace("_response", "")
        df = pd.read_csv(f, encoding="utf-8-sig")
        acc = (df["is_correct"].astype(str).str.strip().str.lower() == "true").mean() * 100
        k_models[model] = acc

    # Reasoning: overall_quality mean
    r_models = {}
    for f in sorted(reasoning_dir.glob("*_eval.csv")):
        model = f.stem.replace("2_fin_reasoning_", "").replace("_eval", "")
        df = pd.read_csv(f, encoding="utf-8-sig")
        oq = pd.to_numeric(df["overall_quality"], errors="coerce").mean()
        r_models[model] = oq

    # Toxicity: score mean
    t_models = {}
    for f in sorted(toxicity_dir.glob("*_eval.csv")):
        model = f.stem.replace("3_fin_toxicity_", "").replace("_eval", "")
        df = pd.read_csv(f, encoding="utf-8-sig")
        sc = pd.to_numeric(df["score"], errors="coerce").mean()
        t_models[model] = sc

    # Fuzzy match model names (handle gpt-5_2 vs gpt-5.2 etc.)
    def normalize(name):
        return name.replace("_", "-").replace("..", ".").lower()

    # Build aligned data
    k_norm = {normalize(k): (k, v) for k, v in k_models.items()}
    r_norm = {normalize(k): (k, v) for k, v in r_models.items()}
    t_norm = {normalize(k): (k, v) for k, v in t_models.items()}

    common = set(k_norm.keys()) & set(r_norm.keys()) & set(t_norm.keys())
    rows = []
    for key in sorted(common):
        rows.append({
            "model": k_norm[key][0],
            "knowledge": k_norm[key][1],
            "reasoning": r_norm[key][1],
            "toxicity": t_norm[key][1],
        })
    df = pd.DataFrame(rows)
    print(f"\nModels with all 3 scores: {len(df)}")

    pairs = [
        ("knowledge", "reasoning", "Knowledge vs Reasoning"),
        ("knowledge", "toxicity", "Knowledge vs Toxicity"),
        ("reasoning", "toxicity", "Reasoning vs Toxicity"),
    ]

    for col1, col2, label in pairs:
        x, y = df[col1].values, df[col2].values
        r_p, p_p = sp_stats.pearsonr(x, y)
        r_s, p_s = sp_stats.spearmanr(x, y)

        rng = np.random.RandomState(42)
        n = len(x)
        boot_p, boot_s = [], []
        for _ in range(10000):
            idx = rng.choice(n, size=n, replace=True)
            bp, _ = sp_stats.pearsonr(x[idx], y[idx])
            bs, _ = sp_stats.spearmanr(x[idx], y[idx])
            boot_p.append(bp)
            boot_s.append(bs)

        ci_p = np.percentile(boot_p, [2.5, 97.5])
        ci_s = np.percentile(boot_s, [2.5, 97.5])

        print(f"\n{label}:")
        print(f"  Pearson r={r_p:.3f}, 95% CI [{ci_p[0]:.3f}, {ci_p[1]:.3f}], p={p_p:.4f}")
        print(f"  Spearman ρ={r_s:.3f}, 95% CI [{ci_s[0]:.3f}, {ci_s[1]:.3f}], p={p_s:.4f}")

    return df


###############################################
# W2: gpt-5 McNemar + category improvement + reasoning metrics
###############################################
def verify_w2():
    print("\n" + "=" * 60)
    print("W2: Think Mode Analysis")
    print("=" * 60)

    knowledge_dir = RESULTS_DIR / "1_fin_knowledge"

    # gpt-5 McNemar (not gpt-5.2)
    std_path = knowledge_dir / "1_fin_knowledge_gpt-5_response.csv"
    think_path = knowledge_dir / "1_fin_knowledge_gpt-5_reasoning_response.csv"

    if std_path.exists() and think_path.exists():
        std_df = pd.read_csv(std_path, encoding="utf-8-sig")
        think_df = pd.read_csv(think_path, encoding="utf-8-sig")
        merged = std_df[["id", "is_correct"]].merge(
            think_df[["id", "is_correct"]], on="id", suffixes=("_std", "_think"))
        s = merged["is_correct_std"].astype(str).str.strip().str.lower() == "true"
        t = merged["is_correct_think"].astype(str).str.strip().str.lower() == "true"
        a, b, c, d = (s & t).sum(), (s & ~t).sum(), (~s & t).sum(), (~s & ~t).sum()
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
        p_val = 1 - chi2.cdf(chi2_stat, df=1)
        print(f"\ngpt-5: Std={s.mean()*100:.1f}%, Think={t.mean()*100:.1f}%, Δ={t.mean()*100-s.mean()*100:+.1f}pp")
        print(f"  a={a}, b={b}, c={c}, d={d}, χ²={chi2_stat:.2f}, p={p_val:.6f}")
    else:
        print(f"\ngpt-5: files not found")
        similar = [Path(p).name for p in glob.glob(str(knowledge_dir / "*gpt-5*response*"))]
        print(f"  Available: {similar}")

    # Category-level Think improvement (gpt-5.2)
    print("\n--- Category-level Think improvement (gpt-5.2) ---")
    std_df = pd.read_csv(knowledge_dir / "1_fin_knowledge_gpt-5.2_response.csv", encoding="utf-8-sig")
    think_df = pd.read_csv(knowledge_dir / "1_fin_knowledge_gpt-5.2_reasoning_response.csv", encoding="utf-8-sig")
    std_df["correct"] = std_df["is_correct"].astype(str).str.strip().str.lower() == "true"
    think_df["correct"] = think_df["is_correct"].astype(str).str.strip().str.lower() == "true"

    std_cat = std_df.groupby("category")["correct"].mean() * 100
    think_cat = think_df.groupby("category")["correct"].mean() * 100
    delta = (think_cat - std_cat).sort_values(ascending=False)
    print(f"\nTop categories by Think improvement:")
    for cat, d in delta.items():
        print(f"  {cat}: {std_cat[cat]:.1f}% → {think_cat[cat]:.1f}% (Δ={d:+.1f}pp)")

    # Reasoning domain metrics (Standard vs Think for gpt-5.2)
    print("\n--- Reasoning domain: gpt-5.2 Standard vs Think ---")
    reasoning_dir = RESULTS_DIR / "2_fin_reasoning"
    std_r = reasoning_dir / "2_fin_reasoning_gpt-5_2_eval.csv"
    think_r = reasoning_dir / "2_fin_reasoning_gpt-5.2_reasoning_eval.csv"

    dims = ["coherence", "consistency", "accuracy", "completeness", "reasoning", "overall_quality"]
    if std_r.exists() and think_r.exists():
        std_rdf = pd.read_csv(std_r, encoding="utf-8-sig")
        think_rdf = pd.read_csv(think_r, encoding="utf-8-sig")
        print(f"\n| Metric | Standard | Think | Δ |")
        print(f"|---|---|---|---|")
        for dim in dims:
            s_mean = pd.to_numeric(std_rdf[dim], errors="coerce").mean()
            t_mean = pd.to_numeric(think_rdf[dim], errors="coerce").mean()
            print(f"| {dim} | {s_mean:.2f} | {t_mean:.2f} | {t_mean - s_mean:+.2f} |")


###############################################
# W3: Wilson score intervals + Kendall's tau
###############################################
def verify_w3():
    print("\n" + "=" * 60)
    print("W3: Category Imbalance Analysis")
    print("=" * 60)

    knowledge_dir = RESULTS_DIR / "1_fin_knowledge"

    def wilson_ci(p, n, z=1.96):
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        return max(0, center - spread), min(1, center + spread)

    # gpt-5.2 Wilson CI
    print("\n--- Wilson Score Intervals (gpt-5.2) ---")
    df = pd.read_csv(knowledge_dir / "1_fin_knowledge_gpt-5.2_response.csv", encoding="utf-8-sig")
    df["correct"] = df["is_correct"].astype(str).str.strip().str.lower() == "true"

    cat_stats = df.groupby("category").agg(n=("correct", "count"), acc=("correct", "mean")).sort_values("n")

    print(f"\n| Category | n | Accuracy | Wilson 95% CI | CI Width |")
    print(f"|---|---|---|---|---|")
    for cat, row in cat_stats.iterrows():
        lo, hi = wilson_ci(row["acc"], row["n"])
        width = (hi - lo) * 100
        print(f"| {cat} | {row['n']} | {row['acc']*100:.1f}% | [{lo*100:.1f}%, {hi*100:.1f}%] | {width:.1f}pp |")

    small = cat_stats[cat_stats["n"] <= 5]
    large = cat_stats[cat_stats["n"] > 10]
    print(f"\nCategories n≤5: {len(small)} ({small['n'].sum()} instances)")
    print(f"Categories n>10: {len(large)} ({large['n'].sum()} instances, {large['n'].sum()/cat_stats['n'].sum()*100:.0f}%)")
    print(f"Median category size: {cat_stats['n'].median():.0f}")

    # Kendall's tau ranking stability
    print("\n--- Kendall's τ Ranking Stability ---")
    model_accs = {}
    all_model_dfs = {}
    for f in sorted(knowledge_dir.glob("*_response.csv")):
        model = f.stem.replace("1_fin_knowledge_", "").replace("_response", "")
        mdf = pd.read_csv(f, encoding="utf-8-sig")
        mdf["correct"] = mdf["is_correct"].astype(str).str.strip().str.lower() == "true"
        model_accs[model] = mdf["correct"].mean()
        all_model_dfs[model] = mdf

    acc_series = pd.Series(model_accs)
    original_rank = acc_series.rank(ascending=False)

    # Use question IDs from first model
    ref_df = list(all_model_dfs.values())[0]
    all_ids = ref_df["id"].values

    rng = np.random.RandomState(42)
    taus = []
    for _ in range(5000):
        boot_ids = rng.choice(all_ids, size=len(all_ids), replace=True)
        boot_accs = {}
        for model, mdf in all_model_dfs.items():
            boot_sample = mdf[mdf["id"].isin(boot_ids)]
            if len(boot_sample) > 0:
                boot_accs[model] = boot_sample["correct"].mean()
        boot_series = pd.Series(boot_accs)
        common = original_rank.index.intersection(boot_series.index)
        if len(common) > 5:
            boot_rank = boot_series[common].rank(ascending=False)
            tau, _ = sp_stats.kendalltau(original_rank[common], boot_rank)
            taus.append(tau)

    taus = np.array(taus)
    ci = np.percentile(taus, [2.5, 97.5])
    pct_above_08 = (taus > 0.8).mean() * 100
    print(f"Kendall's τ: mean={taus.mean():.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"Iterations with τ > 0.8: {pct_above_08:.1f}%")


if __name__ == "__main__":
    df = verify_w1()
    verify_w2()
    verify_w3()
    print("\n\nDone!")
