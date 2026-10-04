"""Plot ROC curves and method comparison from a finished run.

Reads results/<category>/<run_id>/{detection.json, judge.json, dataset.json, generation.json}
and writes PNG figures into vis/<category>/<run_id>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def load(p: Path):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-source", default="auto")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    det = load(run_dir / "detection.json")
    gen = load(run_dir / "generation.json")
    dataset = load(run_dir / "dataset.json")
    ex_by_id = {e["sample_id"]: e for e in dataset["examples"]}

    from reppl2.evaluate_stage import labels_from_gold

    em = labels_from_gold(ex_by_id, gen["generations"])
    labels, ids = {}, []
    if (run_dir / "judge.json").exists():
        jp = load(run_dir / "judge.json")
        for r in jp["per_sample"]:
            if r.get("hard_verdict") is not None:
                labels[r["sample_id"]] = int(r["hard_verdict"])
    if not labels or args.label_source == "em_gold":
        labels = em
    ids = sorted(labels.keys())
    y = np.array([labels[s] for s in ids])
    if len(np.unique(y)) < 2:
        print("not enough class variety for ROC; only score table plotted")
        y = None

    method_scores = {}
    for r in det["per_sample"]:
        for m, v in r["methods"].items():
            if isinstance(v, dict) and v.get("risk") is not None:
                method_scores.setdefault(m, {})[r["sample_id"]] = v["risk"]
    if (run_dir / "judge.json").exists():
        jp = load(run_dir / "judge.json")
        cont = {r["sample_id"]: r.get("continuous_score") for r in jp["per_sample"]
                if r.get("continuous_score") is not None}
        if cont:
            method_scores["judge-continuous"] = cont

    if y is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        for m, smap in sorted(method_scores.items()):
            xs = np.array([smap.get(s, np.nan) for s in ids], dtype=np.float64)
            keep = np.isfinite(xs)
            if keep.sum() < 3 or len(np.unique(y[keep])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y[keep], xs[keep])
            from sklearn.metrics import roc_auc_score

            ax.plot(fpr, tpr, label=f"{m} (AUC={roc_auc_score(y[keep], xs[keep]):.3f}, n={keep.sum()})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"Hallucination detection ROC\n{run_dir.name}")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / "roc.png", dpi=150)
        print(f"wrote {out_dir/'roc.png'}")

    fig, ax = plt.subplots(figsize=(7, max(2.0, 0.5 * len(method_scores))))
    names, aucs = [], []
    if y is not None:
        from sklearn.metrics import roc_auc_score

        for m, smap in sorted(method_scores.items()):
            xs = np.array([smap.get(s, np.nan) for s in ids], dtype=np.float64)
            keep = np.isfinite(xs)
            if keep.sum() < 3 or len(np.unique(y[keep])) < 2:
                continue
            names.append(m)
            aucs.append(roc_auc_score(y[keep], xs[keep]))
    if names:
        order = np.argsort(aucs)
        ax.barh([names[i] for i in order], [aucs[i] for i in order], color="#4472c4")
        ax.set_xlim(0, 1)
        ax.set_xlabel("AUROC")
        ax.set_title("Method comparison (higher = better)")
        for i, a in enumerate(aucs):
            ax.text(a + 0.01, i, f"{a:.3f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "method_auroc.png", dpi=150)
        print(f"wrote {out_dir/'method_auroc.png'}")
    else:
        print("no plottable methods")


if __name__ == "__main__":
    main()
