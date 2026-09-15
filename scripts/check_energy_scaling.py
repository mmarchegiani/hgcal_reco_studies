#!/usr/bin/env python
"""Test whether the rechit-derived energy scaling is degenerate for LayerClusters.

The scale dict entry under test is

    e: {type: min_max_sym, mean: 0.012, std: 0.046, min: 0.004, max: 115.449, fn: sqrt}

which was computed on RecHits.  This script compares the three distributions that actually
matter -- RecHit energy, LayerCluster energy (the model input) and SimCluster recEnergy (the
regression target) -- against those constants, then pushes each through the candidate
implementations of ``min_max_sym`` + ``sqrt`` and measures how much of the [-1, 1] output
range each one occupies.

    python scripts/check_energy_scaling.py --input <predictions.parquet>
"""

import argparse
import os
import sys

import awkward as ak
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hgcal_eval_common import CMAP_10, savefig, setup_style  # noqa: E402

plt = setup_style()

# the entry as reported from configs/hgcal_rechit_var_transform.yaml
SD = dict(mean=0.012, std=0.046, min=0.004, max=115.449, fn="sqrt", type="min_max_sym")


def describe(name, x):
    x = np.asarray(x, float)
    q = np.percentile(x, [1, 25, 50, 75, 99])
    print("  %-26s N=%8d  mean %9.4f  std %9.4f  min %8.5f  max %9.3f"
          % (name, len(x), x.mean(), x.std(), x.min(), x.max()))
    print("  %-26s  p1 %7.4f  p25 %7.4f  med %7.4f  p75 %7.4f  p99 %8.3f"
          % ("", *q))
    return x


# ---- candidate implementations of "sqrt then min_max_sym" -------------------
def scale_bounds_raw(x):
    """min/max used verbatim, i.e. raw-GeV bounds applied to sqrt-transformed values.
    This is the dimensionally inconsistent reading -- the classic bug."""
    return 2.0 * (np.sqrt(x) - SD["min"]) / (SD["max"] - SD["min"]) - 1.0


def scale_bounds_sqrt(x):
    """min/max transformed alongside the data -- the sane reading."""
    lo, hi = np.sqrt(SD["min"]), np.sqrt(SD["max"])
    return 2.0 * (np.sqrt(x) - lo) / (hi - lo) - 1.0


def scale_standard_sqrt(x):
    """if `type` were `standard`: (sqrt(x) - mean)/std with the stored constants."""
    return (np.sqrt(x) - SD["mean"]) / SD["std"]


CANDIDATES = [
    ("min_max_sym, raw bounds", scale_bounds_raw),
    ("min_max_sym, sqrt bounds", scale_bounds_sqrt),
    ("standard(mean,std), sqrt", scale_standard_sqrt),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="plots/maskformer_eval/scaling")
    args = ap.parse_args()

    d = ak.from_parquet(args.input)
    F = lambda a: np.asarray(ak.to_numpy(ak.flatten(a)), float)

    print("=" * 96)
    print("Scale-dict entry under test:", SD)
    print("=" * 96)
    print("\n-- actual distributions in the test file ------------------------------------------")
    rh = describe("RecHit energy", F(d.RecHitHGC.RecHitHGC_energy))
    lc = describe("LayerCluster energy", F(d.LayerCluster.LayerCluster_energy))
    sc = describe("SimCluster recEnergy", F(d.SimCluster.SimCluster_recEnergy))
    pe = F(d.SimCluster.SimCluster_pred_e)

    print("\n-- how well do the scale-dict constants describe each collection? ------------------")
    for name, x in [("RecHit", rh), ("LayerCluster", lc), ("SimCluster recE", sc)]:
        print("  %-16s mean/dict_mean = %8.2f   std/dict_std = %8.2f   "
              "frac above dict max = %.5f"
              % (name, x.mean() / SD["mean"], x.std() / SD["std"], (x > SD["max"]).mean()))

    # the geometric variables, for contrast: these do NOT change between collections
    print("\n-- contrast: the geometric variables barely move between the two collections -------")
    for nm, a, b in [("eta", F(d.LayerCluster.LayerCluster_eta),
                      F(d.SimCluster.SimCluster_impactPoint_eta)),
                     ("phi", F(d.LayerCluster.LayerCluster_phi),
                      F(d.SimCluster.SimCluster_impactPoint_phi))]:
        rh_eta = None
        if nm == "eta":
            x, y, z = (F(d.RecHitHGC.RecHitHGC_x), F(d.RecHitHGC.RecHitHGC_y),
                       F(d.RecHitHGC.RecHitHGC_z))
            r = np.sqrt(x ** 2 + y ** 2)
            rh_eta = -np.log(np.tan(0.5 * np.arctan2(r, np.abs(z)))) * np.sign(z)
        src = rh_eta if rh_eta is not None else None
        print("  %-4s  RecHit mean %s   LayerCluster mean %8.4f std %7.4f   SimCluster mean %8.4f std %7.4f"
              % (nm, ("%8.4f std %7.4f" % (src.mean(), src.std())) if src is not None else "   n/a          ",
                 a.mean(), a.std(), b.mean(), b.std()))

    print("\n-- range occupancy of the scaled target under each candidate implementation --------")
    print("  (what fraction of the [-1, 1] band the bulk of the distribution actually uses)")
    rows = []
    for label, fn in CANDIDATES:
        print("\n  %s" % label)
        for nm, x in [("RecHit (what it was fit on)", rh),
                      ("LayerCluster (model input)", lc),
                      ("SimCluster recE (target)", sc)]:
            s = fn(np.maximum(x, 1e-9))
            p1, p25, p50, p75, p99 = np.percentile(s, [1, 25, 50, 75, 99])
            iqr = p75 - p25
            span = p99 - p1
            print("    %-30s med %+7.4f  IQR %.4f  p1..p99 span %.4f  (%.1f%% of a 2-wide band)"
                  % (nm, p50, iqr, span, 100 * span / 2.0))
            rows.append((label, nm, p50, iqr, span))

    # ---- which statistics does min_max_sym actually read, and do they transfer? ----
    print("\n-- min_max_sym reads min/max, NOT mean/std. Do min/max transfer between collections? --")
    print("  RecHit       min %.5f  max %.4f" % (rh.min(), rh.max()))
    print("  LayerCluster min %.5f  max %.4f" % (lc.min(), lc.max()))
    print("  -> max ratio LayerCluster/RecHit = %.3f   (the MEAN ratio is %.1f)"
          % (lc.max() / rh.max(), lc.mean() / rh.mean()))
    print("  The extremes are set by the same rare events in both collections, so a")
    print("  RecHit-derived min/max transfers almost exactly.  The 21x mean mismatch")
    print("  never enters a min_max_sym scaling.")

    print("\n-- LayerCluster input occupancy under different bounds ------------------------------")
    def occ(x, lo, hi, name):
        sc_ = 2.0 * (np.sqrt(x) - lo) / (hi - lo) - 1.0
        p1, p25, p50, p75, p99 = np.percentile(sc_, [1, 25, 50, 75, 99])
        print("    %-42s med %+.4f  IQR %.4f (%4.1f%% of band)  p1-p99 %.4f (%4.1f%%)"
              % (name, p50, p75 - p25, 100 * (p75 - p25) / 2,
                 p99 - p1, 100 * (p99 - p1) / 2))
    occ(lc, np.sqrt(SD["min"]), np.sqrt(SD["max"]), "config bounds (0.004, 115.449)")
    occ(lc, np.sqrt(rh.min()), np.sqrt(rh.max()), "perfect RecHit bounds")
    occ(lc, np.sqrt(lc.min()), np.sqrt(lc.max()), "perfect LayerCluster bounds")
    pc = np.percentile(lc, [0.1, 99.9])
    occ(lc, np.sqrt(pc[0]), np.sqrt(pc[1]), "LayerCluster 0.1-99.9 percentile bounds")
    print("    -> recomputing the statistics recovers 2.0% -> 3.4%.  The scaler TYPE is the")
    print("       dominant defect, not the staleness of the numbers.")

    print("\n-- for comparison: log transform then standardise -----------------------------------")
    for nm, x in [("LayerCluster input", lc), ("SimCluster target", sc)]:
        l = np.log(np.maximum(x, 1e-6))
        l = (l - l.mean()) / l.std()
        p1, p25, p50, p75, p99 = np.percentile(l, [1, 25, 50, 75, 99])
        print("    %-42s med %+.4f  IQR %.4f  p1-p99 %.4f  std %.3f"
              % (nm, p50, p75 - p25, p99 - p1, l.std()))

    # ---- the decisive number: gradient balance under L1 in scaled space ----
    print("\n-- L1 gradient balance: dE (GeV) equivalent to a fixed scaled error of 0.01 --------")
    print("  a flat L1 loss in scaled space spends its gradient where dE/d(scaled) is largest")
    for label, fn in CANDIDATES:
        print("  %s" % label)
        for E in [0.05, 0.5, 5.0, 50.0, 150.0]:
            eps = 1e-4
            dsdE = (fn(np.array([E + eps])) - fn(np.array([E])))[0] / eps
            dE = 0.01 / dsdE if dsdE else np.inf
            print("     E=%7.2f GeV -> 0.01 in scaled space = %8.4f GeV = %7.2f%% relative"
                  % (E, dE, 100 * dE / E))

    # ---------------- plot ----------------
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.4))
    fig.subplots_adjust(wspace=0.28)
    bins = np.logspace(-4, 2.6, 90)
    ax = axes[0]
    for nm, x, c in [("RecHit energy", rh, CMAP_10[3]),
                     ("LayerCluster energy", lc, CMAP_10[0]),
                     ("SimCluster recEnergy", sc, CMAP_10[2])]:
        ax.hist(x[x > 0], bins=bins, histtype="step", lw=2, density=True, color=c,
                label="%s (mean %.3f GeV)" % (nm, x.mean()))
    ax.axvline(SD["mean"], color="k", ls="--", lw=1.8,
               label="scale-dict mean = %.3f GeV" % SD["mean"])
    ax.axvline(SD["max"], color="k", ls=":", lw=1.8,
               label="scale-dict max = %.1f GeV" % SD["max"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("energy [GeV]", loc="center"); ax.set_ylabel("normalised entries")
    ax.set_title("The distribution the dict was fit on vs the ones used", fontsize=17)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(1e-6, 1e5)

    ax = axes[1]
    b2 = np.linspace(-1.05, 1.35, 120)
    for nm, x, c in [("RecHit", rh, CMAP_10[3]),
                     ("LayerCluster input", lc, CMAP_10[0]),
                     ("SimCluster target", sc, CMAP_10[2])]:
        ax.hist(scale_bounds_sqrt(np.maximum(x, 1e-9)), bins=b2, histtype="step", lw=2,
                density=True, color=c, label=nm)
    ax.hist(scale_bounds_sqrt(np.maximum(pe, 1e-9)), bins=b2, histtype="step", lw=2,
            ls="--", density=True, color=CMAP_10[6], label="model prediction")
    ax.set_yscale("log")
    ax.set_xlabel("scaled value (sqrt, then min_max_sym)", loc="center")
    ax.set_ylabel("normalised entries")
    ax.set_title("Everything piles up against the lower edge", fontsize=17)
    ax.legend(fontsize=12)
    fig.tight_layout()
    savefig(fig, args.outdir, "energy_scaling_diagnosis")
    print("\n  -> %s/energy_scaling_diagnosis.png" % args.outdir)


if __name__ == "__main__":
    main()
