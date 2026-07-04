"""The two figures that tell the story (SPEC goal G4).

1. Crossover curve  -- matched_root_error vs batch size, one line per optimizer,
   as a small-multiple across noise levels: shows *where* Adam overtakes.
2. Regime heatmap   -- the (batch x noise) grid of the AdaptiveSGD-minus-Adam gap
   (a diverging map centred at zero), with AdaptiveSGD's revert count beside it to
   connect the "why" to the "what".

The data-shaping is pure and unit-tested (`order_batches`, `pivot_gap`,
`reverts_grid`); matplotlib is imported lazily inside the render functions so the
reshaping logic -- and its tests -- need neither matplotlib nor TensorFlow. This
keeps the "core logic is testable, I/O lives in the edges" split from CLAUDE.md.

Colours: the Okabe-Ito palette (the standard colourblind-safe categorical set) for
the optimizers; a blue<->red diverging map with a neutral midpoint for the signed
gap. No rainbow maps -- diverging data gets a diverging palette.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

# Okabe-Ito, assigned in fixed order (never cycled). AdaptiveSGD and its main
# rival Adam get the two most separable hues.
OPT_COLORS = {
    "adaptive_sgd": "#0072B2",  # blue  -- the method under study
    "adam": "#E69F00",          # orange -- the main rival
    "sgd": "#009E73",           # green
    "rmsprop": "#D55E00",       # vermillion
}
OPT_LABELS = {
    "adaptive_sgd": "AdaptiveSGD",
    "adam": "Adam",
    "sgd": "SGD",
    "rmsprop": "RMSprop",
}
OPT_ORDER = ("adaptive_sgd", "adam", "sgd", "rmsprop")


# --------------------------------------------------------------------------- #
# pure data-shaping (no matplotlib / numpy / tensorflow)                       #
# --------------------------------------------------------------------------- #
def load_results(csv_path) -> list[dict]:
    """Read a regime-study CSV back into a list of row dicts (values as strings).

    The aggregation helpers coerce types on the fly, so this stays type-agnostic
    and works identically on freshly-produced in-memory rows.
    """
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def order_batches(labels) -> list[str]:
    """Canonical batch ordering: 'full' first, then descending numeric.

    So the x-axis reads large-batch -> small-batch, i.e. easy -> hard signal.
    """
    seen = list(dict.fromkeys(str(x) for x in labels))

    def key(label: str):
        return (0, 0) if label == "full" else (1, -int(label))

    return sorted(seen, key=key)


def _agg(rows, metric) -> dict:
    """Mean of ``metric`` per (optimizer, noise_std, batch_label), over seeds."""
    acc = defaultdict(list)
    for r in rows:
        key = (r["optimizer"], float(r["noise_std"]), str(r["batch_size"]))
        acc[key].append(float(r[metric]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def _axes(rows):
    noises = sorted({float(r["noise_std"]) for r in rows})
    batches = order_batches({str(r["batch_size"]) for r in rows})
    return noises, batches


def series_by_batch(rows, optimizer, noise, metric="matched_root_error"):
    """(batch labels, metric values) for one optimizer at one noise level."""
    means = _agg(rows, metric)
    _, batches = _axes(rows)
    ys = [means.get((optimizer, float(noise), b), math.nan) for b in batches]
    return batches, ys


def pivot_gap(rows, a="adaptive_sgd", b="adam", metric="matched_root_error"):
    """Signed gap grid ``a`` minus ``b``, indexed [noise][batch].

    Positive => ``a`` (AdaptiveSGD) has the larger error, i.e. ``b`` (Adam) wins.
    """
    means = _agg(rows, metric)
    noises, batches = _axes(rows)
    grid = [
        [means.get((a, ns, bl), math.nan) - means.get((b, ns, bl), math.nan)
         for bl in batches]
        for ns in noises
    ]
    return grid, noises, batches


def reverts_grid(rows, optimizer="adaptive_sgd"):
    """Mean revert-count grid for ``optimizer``, indexed [noise][batch]."""
    means = _agg(rows, "n_reverts")
    noises, batches = _axes(rows)
    grid = [[means.get((optimizer, ns, bl), math.nan) for bl in batches]
            for ns in noises]
    return grid, noises, batches


# --------------------------------------------------------------------------- #
# rendering (matplotlib imported lazily)                                       #
# --------------------------------------------------------------------------- #
def _present_optimizers(rows) -> list[str]:
    have = {r["optimizer"] for r in rows}
    return [o for o in OPT_ORDER if o in have]


def plot_crossover(rows, out_path) -> Path:
    """Small-multiple line chart: matched root error vs batch size, per noise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noises, batches = _axes(rows)
    opts = _present_optimizers(rows)
    x = list(range(len(batches)))
    ncols = 2 if len(noises) > 1 else 1
    nrows = math.ceil(len(noises) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False, sharey=True
    )
    for k, ns in enumerate(noises):
        ax = axes[k // ncols][k % ncols]
        for o in opts:
            _, ys = series_by_batch(rows, o, ns)
            ax.plot(x, ys, marker="o", ms=6, lw=2, color=OPT_COLORS[o],
                    label=OPT_LABELS[o])
        ax.set_title(f"noise σ = {ns:g}")
        ax.set_xticks(x)
        ax.set_xticklabels(batches)
        ax.set_xlabel("batch size (large → small)")
        if k % ncols == 0:
            ax.set_ylabel("matched root error")
        ax.grid(True, alpha=0.3, lw=0.6)
    for k in range(len(noises), nrows * ncols):  # blank any unused panel
        axes[k // ncols][k % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(opts), frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Where does Adam overtake? Matched root error vs batch size",
                 y=1.05, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def _draw_heatmap(ax, grid, batches, noises, cmap, title, fmt, *, vcenter0=False):
    import matplotlib.patheffects as pe
    import numpy as np

    data = np.array(grid, dtype=float)
    kw = {}
    if vcenter0:
        vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
        kw = dict(vmin=-vmax, vmax=vmax)
    im = ax.imshow(data, cmap=cmap, aspect="auto", **kw)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels(batches)
    ax.set_yticks(range(len(noises)))
    ax.set_yticklabels([f"{n:g}" for n in noises])
    ax.set_xlabel("batch size (large → small)")
    ax.set_ylabel("noise σ")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                    color="#111",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    return im


def plot_regime_heatmap(rows, out_path) -> Path:
    """Two panels: the signed AdaptiveSGD-minus-Adam gap, and the revert count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gap, noises, batches = pivot_gap(rows)
    rev, _, _ = reverts_grid(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    im1 = _draw_heatmap(
        ax1, gap, batches, noises, cmap="RdBu_r", fmt="{:+.3f}", vcenter0=True,
        title="AdaptiveSGD − Adam matched root error\n"
              "red = Adam wins, blue = AdaptiveSGD wins",
    )
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    im2 = _draw_heatmap(
        ax2, rev, batches, noises, cmap="Purples", fmt="{:.0f}",
        title="AdaptiveSGD revert count\n(the mechanism behind the gap)",
    )
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def plot_results(rows, out_dir, prefix="") -> list[Path]:
    """Render both figures into ``out_dir``; returns the written paths."""
    out_dir = Path(out_dir)
    return [
        plot_crossover(rows, out_dir / f"{prefix}crossover.png"),
        plot_regime_heatmap(rows, out_dir / f"{prefix}regime_heatmap.png"),
    ]
