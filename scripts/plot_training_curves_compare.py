"""Compare per-term training losses and effective rank: exp4 Arm A vs batch-corrupted arm1.

Usage:
    python scripts/plot_training_curves_compare.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOSS_TERMS = [
    ("train/prediction_loss", "Prediction", "#4C78A8"),
    ("train/recon_loss", "Reconstruction", "#F58518"),
    ("train/stabilization_loss", "VICReg (stabilization)", "#54A24B"),
    ("train/loss", "Total", "#111827"),
]

RUNS = [
    {
        "key": "exp4_armA",
        "label": "exp4 Arm A (within-batch VICReg)",
        "path": "runs/exp4_armA/csv_logs/version_0/metrics.csv",
        "linestyle": "-",
    },
    {
        "key": "exp3_arm1",
        "label": "arm1 batch-corrupted (global VICReg)",
        "path": "runs/exp3_arm1/csv_logs/version_0/metrics.csv",
        "linestyle": "--",
    },
]


def _smooth(y: pd.Series, window: int) -> np.ndarray:
    return y.rolling(window=window, min_periods=1, center=True).median().to_numpy()


def load_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("step").drop_duplicates("step", keep="last")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/benchmark/training_curves")
    ap.add_argument("--smooth", type=int, default=50, help="Rolling-median window (in logged rows).")
    ap.add_argument("--max-steps", type=int, default=200_000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {r["key"]: load_run(Path(r["path"])) for r in RUNS}

    # --- Figure 1: per-term losses (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for ax, (col, title, _color) in zip(axes, LOSS_TERMS):
        for run in RUNS:
            df = data[run["key"]]
            y = df[col]
            ax.plot(
                df["step"],
                _smooth(y, args.smooth),
                label=run["label"],
                linestyle=run["linestyle"],
                linewidth=1.6,
                alpha=0.9,
            )
        ax.set_title(title)
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        if col == "train/loss":
            ax.legend(fontsize=8, frameon=False)
    for ax in axes[2:]:
        ax.set_xlabel("Training step")
    fig.suptitle("Per-term training loss (rolling median)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "per_term_losses.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: all losses overlaid per arm ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, run in zip(axes, RUNS):
        df = data[run["key"]]
        for col, title, color in LOSS_TERMS[:-1]:  # exclude total from overlay
            ax.plot(df["step"], _smooth(df[col], args.smooth), label=title, color=color, linewidth=1.5)
        ax.plot(
            df["step"],
            _smooth(df["train/loss"], args.smooth),
            label="Total",
            color="#111827",
            linewidth=2.0,
            alpha=0.7,
        )
        ax.set_title(run["label"])
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Per-term loss decomposition by arm", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "per_term_losses_by_arm.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: effective rank ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    summary = {}
    for run in RUNS:
        df = data[run["key"]]
        er = df["train/geometry/effective_rank"].dropna()
        steps = df.loc[er.index, "step"]
        sm = _smooth(er, max(5, args.smooth // 5))
        ax.plot(steps, sm, label=run["label"], linestyle=run["linestyle"], linewidth=1.8)
        summary[run["key"]] = {
            "start": float(er.iloc[0]),
            "mid": float(er.iloc[len(er) // 2]),
            "end": float(er.iloc[-1]),
            "min": float(er.min()),
            "max": float(er.max()),
            "n_logged": int(len(er)),
        }
    ax.axhline(2.0, color="#9CA3AF", ls=":", lw=1, label="eff-rank = 2")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Effective rank (CLS embedding)")
    ax.set_title("Effective rank across training")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "effective_rank.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 4: combined dashboard ---
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    ax_loss = fig.add_subplot(gs[0, :])
    for run in RUNS:
        df = data[run["key"]]
        ax_loss.plot(
            df["step"],
            _smooth(df["train/loss"], args.smooth),
            label=f"{run['label']} — total",
            linestyle=run["linestyle"],
            linewidth=2,
        )
        ax_loss.plot(
            df["step"],
            _smooth(df["train/prediction_loss"], args.smooth),
            label=f"{run['label']} — pred",
            linestyle=run["linestyle"],
            linewidth=1.2,
            alpha=0.65,
        )
        ax_loss.plot(
            df["step"],
            _smooth(df["train/recon_loss"], args.smooth),
            label=f"{run['label']} — recon",
            linestyle=run["linestyle"],
            linewidth=1.2,
            alpha=0.65,
        )
        ax_loss.plot(
            df["step"],
            _smooth(df["train/stabilization_loss"], args.smooth),
            label=f"{run['label']} — VICReg",
            linestyle=run["linestyle"],
            linewidth=1.2,
            alpha=0.65,
        )
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Per-term training loss")
    ax_loss.legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
    ax_loss.grid(True, alpha=0.25)

    ax_er = fig.add_subplot(gs[1, 0])
    for run in RUNS:
        df = data[run["key"]]
        er = df["train/geometry/effective_rank"].dropna()
        ax_er.plot(
            df.loc[er.index, "step"],
            _smooth(er, max(5, args.smooth // 5)),
            label=run["label"],
            linestyle=run["linestyle"],
            linewidth=1.8,
        )
    ax_er.axhline(2.0, color="#9CA3AF", ls=":", lw=1)
    ax_er.set_xlabel("Training step")
    ax_er.set_ylabel("Effective rank")
    ax_er.set_title("CLS effective rank")
    ax_er.legend(fontsize=7, frameon=False)
    ax_er.grid(True, alpha=0.25)
    ax_er.set_ylim(bottom=0)

    ax_std = fig.add_subplot(gs[1, 1])
    for run in RUNS:
        df = data[run["key"]]
        ax_std.plot(
            df["step"],
            _smooth(df["train/target_std"], args.smooth),
            label=run["label"],
            linestyle=run["linestyle"],
            linewidth=1.8,
        )
    ax_std.set_xlabel("Training step")
    ax_std.set_ylabel("Target token std")
    ax_std.set_title("Target encoder spread (collapse canary)")
    ax_std.legend(fontsize=7, frameon=False)
    ax_std.grid(True, alpha=0.25)

    fig.suptitle("exp4 Arm A vs arm1 (batch-corrupted) — training dynamics", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "training_dashboard.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "training_curves_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = f"""# Training curves: exp4 Arm A vs arm1 (batch-corrupted)

## Effective rank summary

| Arm | Start | Mid | End | Min | Max |
|-----|-------|-----|-----|-----|-----|
| exp4 Arm A | {summary['exp4_armA']['start']:.2f} | {summary['exp4_armA']['mid']:.2f} | {summary['exp4_armA']['end']:.2f} | {summary['exp4_armA']['min']:.2f} | {summary['exp4_armA']['max']:.2f} |
| arm1 (global VICReg) | {summary['exp3_arm1']['start']:.2f} | {summary['exp3_arm1']['mid']:.2f} | {summary['exp3_arm1']['end']:.2f} | {summary['exp3_arm1']['min']:.2f} | {summary['exp3_arm1']['max']:.2f} |

## Figures
- `per_term_losses.png` — 2×2 panel comparing each loss term
- `per_term_losses_by_arm.png` — decomposition within each arm
- `effective_rank.png` — eff-rank trajectories
- `training_dashboard.png` — combined overview (+ target_std)
"""
    (out_dir / "TRAINING_CURVES_REPORT.md").write_text(md)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
