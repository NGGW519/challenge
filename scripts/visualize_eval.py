#!/usr/bin/env python3
"""logs/eval/<TAG>/summary.csv → HTML 리포트 (히스토그램, 박스플롯, 회귀 분석).

사용:
    python3 scripts/visualize_eval.py \
        --csv logs/eval/hybrid_v1_20261005_1430/summary.csv \
        --out logs/eval/hybrid_v1_20261005_1430/report.html

의존성: pandas, matplotlib (pyproject.toml에 포함).
matplotlib 누락 환경에선 텍스트 요약만 출력.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
from pathlib import Path

logger = logging.getLogger("visualize_eval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--baseline_csv", default=None,
                   help="비교용 기준 CSV (회귀 검증)")
    return p.parse_args(argv)


def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def build_report(df, baseline_df=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_total = len(df)
    by_run = df.groupby("run")["total"].agg(["mean", "count"]).reset_index()
    n_runs = len(by_run)

    # 1. 분포 히스토그램
    fig, ax = plt.subplots(figsize=(8, 4))
    df["total"].dropna().hist(ax=ax, bins=30, color="#3a7", edgecolor="white")
    ax.set_xlabel("trial total"); ax.set_ylabel("count")
    ax.set_title(f"trial total distribution (n={n_total})")
    img1 = fig_to_data_uri(fig); plt.close(fig)

    # 2. trial별 박스플롯
    fig, ax = plt.subplots(figsize=(8, 4))
    by_trial = [g["total"].dropna().values for _, g in df.groupby("trial_id")]
    labels = sorted(df["trial_id"].dropna().unique().tolist())
    ax.boxplot(by_trial, labels=labels, showmeans=True)
    ax.set_xlabel("trial_id"); ax.set_ylabel("score")
    ax.set_title("score by trial_id")
    img2 = fig_to_data_uri(fig); plt.close(fig)

    # 3. tier 분해 (스택바)
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = ["tier_1_validity", "smoothness", "duration", "efficiency", "tier_3_insertion"]
    pen_cols = ["force_penalty", "contact_penalty"]
    means = [df[c].mean() if c in df.columns else 0 for c in cols]
    pen_means = [df[c].mean() if c in df.columns else 0 for c in pen_cols]
    ax.bar(cols, means, color=["#888", "#3a7", "#37a", "#7a3", "#a37"])
    ax.bar(pen_cols, pen_means, color=["#a73", "#a33"])
    ax.set_ylabel("avg per trial"); ax.set_title("score component breakdown")
    plt.xticks(rotation=20)
    img3 = fig_to_data_uri(fig); plt.close(fig)

    contact_hits = int((df["contact_penalty"].fillna(0) < 0).sum()) if "contact_penalty" in df else 0
    force_hits   = int((df["force_penalty"].fillna(0)   < 0).sum()) if "force_penalty"   in df else 0

    summary = {
        "rows": n_total,
        "runs": n_runs,
        "mean_total":   round(df["total"].mean(), 2)   if "total" in df else None,
        "median_total": round(df["total"].median(), 2) if "total" in df else None,
        "stdev_total":  round(df["total"].std(), 2)    if "total" in df else None,
        "min_total":    round(df["total"].min(), 2)    if "total" in df else None,
        "max_total":    round(df["total"].max(), 2)    if "total" in df else None,
        "contact_pen_hits": contact_hits,
        "force_pen_hits":   force_hits,
    }
    summary_html = "<ul>" + "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in summary.items()) + "</ul>"

    regression_html = ""
    if baseline_df is not None:
        diff = (df["total"].mean() - baseline_df["total"].mean())
        flag = "🟢" if diff >= -1.0 else "🔴" if diff < -3.0 else "🟡"
        regression_html = f"""
        <h2>vs baseline</h2>
        <p>{flag} delta = {diff:+.2f} (current {df["total"].mean():.2f} vs baseline {baseline_df["total"].mean():.2f})</p>
        """

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>AIC eval report</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2em auto; }}
img  {{ max-width: 100%; }}
table {{ border-collapse: collapse; }} td, th {{ padding: 4px 10px; border: 1px solid #ccc; }}
</style></head><body>
<h1>AIC evaluation report</h1>
<h2>summary</h2>
{summary_html}
{regression_html}
<h2>distribution</h2>
<img src="{img1}">
<h2>by trial</h2>
<img src="{img2}">
<h2>component breakdown</h2>
<img src="{img3}">
</body></html>"""
    return html


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas required (pip install pandas)"); return 1

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("not found: %s", csv_path); return 1

    df = pd.read_csv(csv_path)
    baseline = pd.read_csv(args.baseline_csv) if args.baseline_csv else None

    out = Path(args.out) if args.out else csv_path.with_suffix(".html")
    try:
        html = build_report(df, baseline)
    except ImportError as e:
        logger.error("matplotlib required for HTML report (%s); printing summary only", e)
        print(df["total"].describe() if "total" in df else df.describe())
        return 0

    out.write_text(html)
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
