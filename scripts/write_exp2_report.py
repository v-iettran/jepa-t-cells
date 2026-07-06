"""Write a compact Experiment 2 results report from completed eval JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _metric(results: dict | None, key: str = "head_prediction") -> str:
    if results is None:
        return "pending"
    value = results.get(key, {}).get("overall", {}).get("delta_pearson")
    return "n/a" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="EXPERIMENT_2_REPORT.md")
    parser.add_argument("--a0", default="runs/exp2_eval_A0/perturbation_results_coexpression.json")
    parser.add_argument("--a1", default="runs/exp2_eval_A1/perturbation_results_coexpression.json")
    parser.add_argument("--h0", default="runs/exp2_head_H0/perturbation_results_coexpression.json")
    parser.add_argument("--h2", default="runs/exp2_head_H2/perturbation_results_jepa.json")
    parser.add_argument("--h3", default="runs/exp2_head_H3/perturbation_results_grn.json")
    parser.add_argument("--h3-concat", default="runs/exp2_head_H3_concat/perturbation_results_grn_jepa.json")
    args = parser.parse_args()

    a0 = _load(Path(args.a0))
    a1 = _load(Path(args.a1))
    h0 = _load(Path(args.h0))
    h2 = _load(Path(args.h2))
    h3 = _load(Path(args.h3))
    h3_concat = _load(Path(args.h3_concat))

    a0_repr = _metric(a0, "representation_quality")
    a1_repr = _metric(a1, "representation_quality")
    best = "pending"
    if a0 and a1:
        best = "A1" if float(a1_repr) > float(a0_repr) else "A0"

    lines = [
        "# Experiment 2 Report",
        "",
        "Auto-generated summary. Fill in interpretation and figures after reviewing the full JSON outputs.",
        "",
        "## Encoder Comparison",
        "",
        "| Encoder | Representation-quality delta Pearson |",
        "|---|---:|",
        f"| A0: VICReg, 4000 HVGs | {a0_repr} |",
        f"| A1: SIGReg, 4000 HVGs | {a1_repr} |",
        "",
        f"Selected encoder: **{best}**",
        "",
        "## Head Comparison",
        "",
        "| Head feature | Head delta Pearson |",
        "|---|---:|",
        f"| H0: co-expression SVD | {_metric(h0)} |",
        f"| H2: JEPA gene-token embedding | {_metric(h2)} |",
        f"| H3: GENIE3 state-matched GRN | {_metric(h3)} |",
        f"| H3+H2: GRN + JEPA concat | {_metric(h3_concat)} |",
        "",
        "## Artifact Paths",
        "",
        "- `configs/exp2.yaml`",
        "- `configs/gene_vocab_4000.tsv`",
        "- `data/processed/*_4000.h5ad`",
        "- `data/grn/`",
        "- `runs/exp2_A0/`, `runs/exp2_A1/`",
        "- `runs/exp2_head_H0/`, `runs/exp2_head_H2/`, `runs/exp2_head_H3/`, `runs/exp2_head_H3_concat/`",
        "",
    ]
    Path(args.output).write_text("\n".join(lines))
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
