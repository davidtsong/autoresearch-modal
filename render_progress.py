#!/usr/bin/env python3
"""Render progress.png from results.tsv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = Path("results.tsv")
OUTPUT_PATH = Path("progress.png")


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"Missing {RESULTS_PATH}")

    df = pd.read_csv(RESULTS_PATH, sep="\t")
    if df.empty:
        raise SystemExit("results.tsv is empty")

    df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
    df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
    df["status"] = df["status"].astype(str).str.strip().str.upper()
    return df


def render(df: pd.DataFrame) -> None:
    valid = df[df["status"] != "CRASH"].copy().reset_index(drop=True)
    if valid.empty:
        raise SystemExit("No non-crash rows available for plotting")

    baseline_bpb = float(valid.loc[0, "val_bpb"])
    below = valid[valid["val_bpb"] <= baseline_bpb + 0.0005]
    discarded = below[below["status"] == "DISCARD"]
    kept = below[below["status"] == "KEEP"]
    best = float(valid["val_bpb"].min())

    fig, ax = plt.subplots(figsize=(16, 8))

    if not discarded.empty:
        ax.scatter(
            discarded.index,
            discarded["val_bpb"],
            c="#cccccc",
            s=12,
            alpha=0.5,
            zorder=2,
            label="Discarded",
        )

    if not kept.empty:
        ax.scatter(
            kept.index,
            kept["val_bpb"],
            c="#2ecc71",
            s=50,
            zorder=4,
            label="Kept",
            edgecolors="black",
            linewidths=0.5,
        )
        running_min = kept["val_bpb"].cummin()
        ax.step(
            kept.index,
            running_min,
            where="post",
            color="#27ae60",
            linewidth=2,
            alpha=0.7,
            zorder=3,
            label="Running best",
        )

        for idx, bpb in zip(kept.index, kept["val_bpb"]):
            desc = str(valid.loc[idx, "description"]).strip()
            if len(desc) > 45:
                desc = f"{desc[:42]}..."
            ax.annotate(
                desc,
                (idx, bpb),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8.0,
                color="#1a7a3a",
                alpha=0.9,
                rotation=30,
                ha="left",
                va="bottom",
            )

    n_total = len(df)
    n_kept = int((df["status"] == "KEEP").sum())
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
    ax.set_title(
        f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements",
        fontsize=14,
    )
    if ax.has_data():
        ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    margin = max((baseline_bpb - best) * 0.15, 0.0001)
    ax.set_ylim(best - margin, baseline_bpb + margin)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


def main() -> None:
    render(load_results())


if __name__ == "__main__":
    main()
