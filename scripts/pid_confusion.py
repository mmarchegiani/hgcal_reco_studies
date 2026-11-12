#!/usr/bin/env python
"""
Standalone plot of the node-assignment leakage matrix between particle classes.

The matrix values are hardcoded (taken from plots/maskformer_eval/metrics.json,
key "node_assignment_leakage"; see the leakage block in evaluate_maskformer.py
for how they are computed).  Edit the CONFIG block below to change labels,
title, colours, etc.

Usage:
    python scripts/pid_confusion.py [--out plots/pid_confusion.png]
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mplhep as hep
hep.style.use("CMS")

# --------------------------------------------------------------------------- #
# CONFIG -- edit here
# --------------------------------------------------------------------------- #

# row = true class, column = assigned class; each row is normalised to 1.
CLASSES = ["hadronic", "EM", "muon"]
MATRIX = np.array([
    [0.97, 0.03, 0.00],
    [0.04, 0.96, 0.00],
    [0.04, 0.00, 0.96],
])

TITLE = None
XLABEL = "Predicted particle class"
YLABEL = "True particle class"

CMAP = "Oranges"
VMIN, VMAX = 0.0, 1.0
FIGSIZE = (9, 8)
FMT = "%.2f"             # format of the numbers written in the cells
MIN_SHOWN = 0.005        # cells with |value| below this get no text
WHITE_TEXT_ABOVE = 0.55  # cell text is white when value > this, else black
TITLE_FONTSIZE = 17
TICK_FONTSIZE = 24
CELL_FONTSIZE = 32
XTICK_ROTATION = 0.0
MINOR_TICKS = False

# --------------------------------------------------------------------------- #




def plot_matrix(M, classes, out):
    n = len(classes)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(M, cmap=CMAP, vmin=VMIN, vmax=VMAX)
    ax.text(
        -0.45, -0.7,               
        "Private work (CMS simulation)",              
        fontsize=24,                
        verticalalignment='top',    
        fontproperties="Tex Gyre Heros:italic"
    )
    for i in range(n):
        for j in range(n):
            if abs(M[i, j]) >= MIN_SHOWN:
                ax.text(j, i, FMT % M[i, j], ha="center", va="center",
                        fontsize=CELL_FONTSIZE,
                        color="white" if M[i, j] > WHITE_TEXT_ABOVE else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ha = "right" if 0 < XTICK_ROTATION < 90 else "center"
    ax.set_xticklabels(classes, rotation=XTICK_ROTATION, ha=ha,
                   rotation_mode="anchor" if ha == "right" else "default",
                   fontsize=TICK_FONTSIZE)
    ax.set_yticklabels(classes, fontsize=TICK_FONTSIZE)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(TITLE, fontsize=TITLE_FONTSIZE)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    if not MINOR_TICKS:
        ax.minorticks_off()
        cbar.minorticks_off()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="plots/pid_confusion.png",
                   help="output image path (extension selects the format)")
    args = p.parse_args()

    print("wrote", plot_matrix(MATRIX, CLASSES, args.out))


if __name__ == "__main__":
    main()
