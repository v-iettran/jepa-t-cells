from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from jepa_poc.config import ensure_dir, load_config


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/poc.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = ensure_dir(cfg.data.run_dir)
    annotation = _read_json(run_dir / "annotation_results.json")
    perturbation = _read_json(run_dir / "perturbation_results.json")

    rows = []
    if annotation:
        jepa = annotation.get("jepa", annotation)
        rows.append({"task": "annotation_jepa", "group": "overall", "metric": "linear_macro_f1", "value": jepa["linear_probe"]["macro_f1"]})
        rows.append({"task": "annotation_jepa", "group": "overall", "metric": "knn_macro_f1", "value": jepa["knn_probe"]["macro_f1"]})
        if "pca_baseline" in annotation:
            pca = annotation["pca_baseline"]
            rows.append({"task": "annotation_pca", "group": "overall", "metric": "linear_macro_f1", "value": pca["linear_probe"]["macro_f1"]})
            rows.append({"task": "annotation_pca", "group": "overall", "metric": "knn_macro_f1", "value": pca["knn_probe"]["macro_f1"]})

    def _emit(view_name: str, view: dict) -> None:
        for metric, value in view.get("overall", {}).items():
            rows.append({"task": view_name, "group": "overall", "metric": metric, "value": value})
        for gene, gm in view.get("per_gene", {}).items():
            for metric, value in gm.items():
                if metric == "n_cells":
                    continue
                rows.append({"task": view_name, "group": f"gene:{gene}", "metric": metric, "value": value})
        for cond, cm in view.get("per_condition", {}).items():
            for metric, value in cm.items():
                if metric == "n_cells":
                    continue
                rows.append({"task": view_name, "group": f"cond:{cond}", "metric": metric, "value": value})

    if perturbation:
        if "representation_quality" in perturbation:
            _emit("repr_quality", perturbation["representation_quality"])
        if "head_prediction" in perturbation:
            _emit("head_pred", perturbation["head_prediction"])
        base = perturbation.get("baseline_mean_train_delta", {})
        for metric, value in base.get("overall", {}).items():
            rows.append({"task": "baseline", "group": "overall", "metric": metric, "value": value})
        for gene, gm in base.get("per_gene", {}).items():
            for metric, value in gm.items():
                rows.append({"task": "baseline", "group": f"gene:{gene}", "metric": metric, "value": value})

    df = pd.DataFrame(rows)
    table_path = run_dir / "results_table.md"
    table_path.write_text(df.to_markdown(index=False) if not df.empty else "No result JSONs found.\n")

    # Headline comparison: per-gene delta_pearson for repr_quality vs head_pred vs baseline.
    dp = df[(df["metric"] == "delta_pearson") & (df["group"].str.startswith("gene:"))]
    if not dp.empty:
        pivot = dp.pivot_table(index="group", columns="task", values="value")
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_ylabel("delta_pearson")
        ax.set_title("Per-gene perturbation delta_pearson (condition-matched)")
        ax.axhline(0, color="k", linewidth=0.8)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(run_dir / "results_summary.png", dpi=200)
    elif not df.empty:
        plt.figure(figsize=(8, 4))
        plt.bar(df["task"] + ":" + df["metric"], df["value"].astype(float))
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(run_dir / "results_summary.png", dpi=200)
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
