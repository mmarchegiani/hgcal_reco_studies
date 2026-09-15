#!/usr/bin/env python
"""Why is the predicted cluster energy so much lower than the true one?

Compares ``SimCluster_pred_e`` with ``SimCluster_recEnergy`` (the maximum
reconstructable energy) and tries to attribute the deficit.  Everything is done
twice, with and without the ``SimCluster_pred_valid`` mask.

Tests performed
---------------
1. response vs true energy       -> is the bias flat or energy dependent?
2. response vs *predicted* energy -> distinguishes a calibration bias from
                                    regression dilution (noise + steep spectrum)
3. log-log fit                    -> quantifies the power-law compression
4. per-event energy budget        -> where does the missing energy go?
5. truth sanity check             -> recEnergy vs the sum of its LayerClusters
6. self-consistency               -> pred_e vs the sum of the LayerClusters the
                                    model itself assigned to the object
7. response vs pred_n_nodes

Usage
-----
    python scripts/energy_investigation.py \
        --input data/hgcal_v1_20260818-T193435/epoch=199-val_loss=11.54392__test.parquet \
        --outdir plots/maskformer_eval/energy
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hgcal_eval_common import (  # noqa: E402
    CMS_blue, CMS_gray, CMS_orange, CMS_red, EvalData, PHYSICS_CLASSES,
    class_color, class_name, profile, robust_sigma, savefig, setup_style,
)

plt = setup_style()

E_EDGES = np.array([0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0])
NODE_EDGES = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 700])
MASKS = [("all_queries", "all matched queries", CMS_blue),
         ("pred_valid", "pred_valid only", CMS_red)]


def get_masks(data):
    return {"all_queries": np.ones(len(data.p_e), bool), "pred_valid": data.p_valid}


# --------------------------------------------------------------------------- #

def plot_2d(data, outdir, masks):
    for key, name, _ in MASKS:
        m = masks[key] & (data.p_e > 1e-3) & (data.p_pred_e > 1e-3)
        fig, ax = plt.subplots(figsize=(9, 8))
        bins = np.logspace(-3, 2.5, 70)
        h = ax.hist2d(data.p_e[m], data.p_pred_e[m], bins=[bins, bins],
                      norm="log", cmap="viridis")
        lo, hi = 1e-3, 10 ** 2.5
        ax.plot([lo, hi], [lo, hi], "w--", lw=2, label=r"$E_{pred}=E_{true}$")
        # median profile
        ctr, med, sig, n = profile(data.p_e[m], data.p_pred_e[m], bins[::4])
        ok = np.isfinite(med)
        ax.plot(ctr[ok], med[ok], "o-", color="red", lw=2.5, ms=6, label="median")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$E_{\mathrm{true}}$ = SimCluster_recEnergy [GeV]")
        ax.set_ylabel(r"$E_{\mathrm{pred}}$ = SimCluster_pred_e [GeV]")
        ax.set_title("Predicted vs true energy (%s)" % name, fontsize=19)
        ax.legend(fontsize=14, loc="upper left")
        ax.grid(False)
        fig.colorbar(h[3], ax=ax, label="SimClusters")
        savefig(fig, outdir, "e_pred_vs_true_2d_%s" % key)


def plot_response(data, outdir, masks, metrics):
    """Median E_pred/E_true in bins of true energy AND in bins of predicted energy."""
    for xvar, xname, fname in [
        ("true", r"$E_{\mathrm{true}}$ (recEnergy) [GeV]", "response_vs_Etrue"),
        ("pred", r"$E_{\mathrm{pred}}$ (pred_e) [GeV]", "response_vs_Epred"),
    ]:
        fig, ax = plt.subplots()
        for key, name, color in MASKS:
            m = masks[key] & (data.p_e > 1e-3) & (data.p_pred_e > 1e-3)
            ratio = data.p_pred_e[m] / data.p_e[m]
            x = data.p_e[m] if xvar == "true" else data.p_pred_e[m]
            ctr, med, sig, n = profile(x, ratio, E_EDGES)
            ok = np.isfinite(med)
            ax.errorbar(ctr[ok], med[ok], yerr=sig[ok],
                        xerr=[(ctr - E_EDGES[:-1])[ok], (E_EDGES[1:] - ctr)[ok]],
                        fmt="o", color=color, lw=2, ms=7, label=name)
            metrics.setdefault("response_%s" % xvar, {})[key] = dict(
                centre=ctr.tolist(),
                median_ratio=np.where(np.isfinite(med), med, None).tolist(),
                n=n.tolist())
        ax.axhline(1.0, color="k", ls="--", lw=1.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xname)
        ax.set_ylabel(r"median $E_{\mathrm{pred}}/E_{\mathrm{true}}$")
        ax.set_title("Energy response binned in %s energy" % xvar, fontsize=19)
        ax.legend(fontsize=14)
        savefig(fig, outdir, fname)

    # the two together: the smoking gun for regression dilution
    fig, ax = plt.subplots()
    m = (data.p_e > 1e-3) & (data.p_pred_e > 1e-3)
    ratio = data.p_pred_e[m] / data.p_e[m]
    for x, lbl, color, mk in [(data.p_e[m], r"binned in $E_{\mathrm{true}}$", CMS_blue, "o"),
                              (data.p_pred_e[m], r"binned in $E_{\mathrm{pred}}$", CMS_orange, "s")]:
        ctr, med, sig, n = profile(x, ratio, E_EDGES)
        ok = np.isfinite(med)
        ax.plot(ctr[ok], med[ok], mk + "-", color=color, lw=2.5, ms=8, label=lbl)
    ax.axhline(1.0, color="k", ls="--", lw=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("binning energy [GeV]")
    ax.set_ylabel(r"median $E_{\mathrm{pred}}/E_{\mathrm{true}}$")
    ax.set_title("Opposite slopes = regression dilution, not a calibration offset",
                 fontsize=17)
    ax.legend(fontsize=14)
    savefig(fig, outdir, "response_dilution_test")


def loglog_fit(data, outdir, masks, metrics):
    fig, ax = plt.subplots()
    xg = np.logspace(-3, 2.4, 100)
    for key, name, color in MASKS:
        m = masks[key] & (data.p_e > 1e-3) & (data.p_pred_e > 1e-3)
        x, y = np.log(data.p_e[m]), np.log(data.p_pred_e[m])
        b, a = np.polyfit(x, y, 1)
        corr = np.corrcoef(x, y)[0, 1]
        sig = robust_sigma(y - x)
        ax.plot(xg, np.exp(a) * xg ** b, "-", color=color, lw=2.5,
                label="%s\n$E_{pred}=%.2f\\,E_{true}^{%.2f}$  $\\rho_{\\log}=%.2f$"
                      % (name, np.exp(a), b, corr))
        metrics.setdefault("loglog_fit", {})[key] = dict(
            slope=float(b), intercept=float(a), amplitude=float(np.exp(a)),
            log_corr=float(corr), sigma68_log_ratio=float(sig),
            std_log_true=float(x.std()), std_log_pred=float(y.std()),
            pivot_energy_GeV=float(np.exp(x.mean())),
            shrinkage_check_corr_times_stdratio=float(corr * y.std() / x.std()),
            n=int(m.sum()))
    ax.plot(xg, xg, "k--", lw=1.5, label="ideal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$E_{\mathrm{true}}$ [GeV]")
    ax.set_ylabel(r"$E_{\mathrm{pred}}$ [GeV]")
    ax.set_title("Power-law fit of the energy response", fontsize=19)
    ax.legend(fontsize=12, loc="upper left")
    savefig(fig, outdir, "loglog_fit")


def log_ratio_distribution(data, outdir, masks, metrics):
    fig, ax = plt.subplots()
    bins = np.linspace(-6, 6, 121)
    for key, name, color in MASKS:
        m = masks[key] & (data.p_e > 1e-3) & (data.p_pred_e > 1e-3)
        lr = np.log(data.p_pred_e[m] / data.p_e[m])
        ax.hist(lr, bins=bins, histtype="step", lw=2, color=color, density=True,
                label="%s: med %.3f, $\\sigma_{68}$ %.3f (=x%.2f)"
                      % (name, np.median(lr), robust_sigma(lr), np.exp(robust_sigma(lr))))
        metrics.setdefault("log_ratio", {})[key] = dict(
            median=float(np.median(lr)), sigma68=float(robust_sigma(lr)),
            mean=float(lr.mean()), n=int(m.sum()))
    ax.axvline(0, color="k", ls=":", lw=1.2)
    ax.set_xlabel(r"$\ln(E_{\mathrm{pred}}/E_{\mathrm{true}})$")
    ax.set_ylabel("normalised SimClusters / bin")
    ax.set_title("Log energy ratio - the scale-free resolution", fontsize=19)
    ax.legend(fontsize=13)
    savefig(fig, outdir, "log_ratio_distribution")

    # per class
    fig, ax = plt.subplots()
    for c in PHYSICS_CLASSES:
        m = (data.p_class == c) & (data.p_e > 1e-3) & (data.p_pred_e > 1e-3)
        if m.sum() < 30:
            continue
        lr = np.log(data.p_pred_e[m] / data.p_e[m])
        ax.hist(lr, bins=bins, histtype="step", lw=2, color=class_color(c), density=True,
                label="%s: med %.2f $\\sigma_{68}$ %.2f (N=%d)"
                      % (class_name(c), np.median(lr), robust_sigma(lr), m.sum()))
    ax.axvline(0, color="k", ls=":", lw=1.2)
    ax.set_xlabel(r"$\ln(E_{\mathrm{pred}}/E_{\mathrm{true}})$")
    ax.set_ylabel("normalised SimClusters / bin")
    ax.set_title("Log energy ratio by true class", fontsize=19)
    ax.legend(fontsize=12)
    savefig(fig, outdir, "log_ratio_by_class")


def energy_budget(data, outdir, metrics):
    """Per-event sums: how much of the true energy does the model account for?"""
    nev = data.n_events
    sum_true = np.bincount(data.p_event, weights=data.p_e, minlength=nev)
    sum_pred = np.bincount(data.p_event, weights=data.p_pred_e, minlength=nev)
    sum_pred_v = np.bincount(data.p_event[data.p_valid],
                             weights=data.p_pred_e[data.p_valid], minlength=nev)
    sum_true_v = np.bincount(data.p_event[data.p_valid],
                             weights=data.p_e[data.p_valid], minlength=nev)
    sum_lc = np.bincount(data.n_event, weights=data.n_energy, minlength=nev)

    metrics["energy_budget"] = dict(
        total_true_recE=float(sum_true.sum()),
        total_pred_e=float(sum_pred.sum()),
        total_pred_e_valid_only=float(sum_pred_v.sum()),
        total_true_recE_of_valid=float(sum_true_v.sum()),
        total_layercluster_energy=float(sum_lc.sum()),
        ratio_pred_over_true=float(sum_pred.sum() / sum_true.sum()),
        ratio_pred_over_true_valid=float(sum_pred_v.sum() / max(sum_true_v.sum(), 1e-9)),
        ratio_predvalid_over_alltrue=float(sum_pred_v.sum() / sum_true.sum()),
        ratio_true_over_lc=float(sum_true.sum() / sum_lc.sum()),
    )

    fig, ax = plt.subplots()
    lo, hi = 1, 2000
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)
    ax.hist(sum_true, bins=bins, histtype="step", lw=2, color=CMS_gray,
            label=r"$\sum E_{\mathrm{true}}$ (%.0f GeV total)" % sum_true.sum())
    ax.hist(sum_pred, bins=bins, histtype="step", lw=2, color=CMS_blue,
            label=r"$\sum E_{\mathrm{pred}}$, all queries (%.0f)" % sum_pred.sum())
    ax.hist(sum_pred_v, bins=bins, histtype="step", lw=2, color=CMS_red,
            label=r"$\sum E_{\mathrm{pred}}$, pred_valid (%.0f)" % sum_pred_v.sum())
    ax.hist(sum_lc, bins=bins, histtype="step", lw=2, color=CMS_orange, ls="--",
            label=r"$\sum E_{\mathrm{LayerCluster}}$ (%.0f)" % sum_lc.sum())
    ax.set_xscale("log")
    ax.set_xlabel("event energy sum [GeV]")
    ax.set_ylabel("events / bin")
    ax.set_title("Per-event energy budget", fontsize=19)
    ax.legend(fontsize=12)
    savefig(fig, outdir, "event_energy_budget")

    fig, ax = plt.subplots()
    good = sum_true > 0
    ax.hist(sum_pred[good] / sum_true[good], bins=np.linspace(0, 3, 80),
            histtype="step", lw=2, color=CMS_blue,
            label="all queries: med %.3f" % np.median(sum_pred[good] / sum_true[good]))
    ax.hist(sum_pred_v[good] / sum_true[good], bins=np.linspace(0, 3, 80),
            histtype="step", lw=2, color=CMS_red,
            label="pred_valid: med %.3f" % np.median(sum_pred_v[good] / sum_true[good]))
    ax.axvline(1, color="k", ls="--", lw=1.5)
    ax.set_xlabel(r"$\sum E_{\mathrm{pred}} / \sum E_{\mathrm{true}}$ per event")
    ax.set_ylabel("events / bin")
    ax.set_title("Event-level energy closure", fontsize=19)
    ax.legend(fontsize=13)
    savefig(fig, outdir, "event_energy_closure")


def truth_sanity(data, outdir, metrics):
    """recEnergy should be (close to) the sum of the LayerClusters matched to it."""
    fig, ax = plt.subplots()
    m = (data.p_e > 1e-3) & (data.p_sum_e_true_nodes > 0)
    r = data.p_sum_e_true_nodes[m] / data.p_e[m]
    ax.hist(r, bins=np.linspace(0, 2, 100), histtype="step", lw=2, color=CMS_gray,
            label=r"$\sum E_{LC}^{\mathrm{truth\ assoc.}} / E_{\mathrm{true}}$: med %.3f"
                  % np.median(r))
    m2 = (data.p_e > 1e-3) & (data.p_sum_e_pred_nodes > 0)
    r2 = data.p_sum_e_pred_nodes[m2] / data.p_e[m2]
    ax.hist(r2, bins=np.linspace(0, 2, 100), histtype="step", lw=2, color=CMS_blue,
            label=r"$\sum E_{LC}^{\mathrm{model\ mask}} / E_{\mathrm{true}}$: med %.3f"
                  % np.median(r2))
    m3 = (data.p_pred_e > 1e-3) & (data.p_sum_e_pred_nodes > 0)
    r3 = data.p_sum_e_pred_nodes[m3] / data.p_pred_e[m3]
    ax.hist(r3, bins=np.linspace(0, 2, 100), histtype="step", lw=2, color=CMS_red,
            label=r"$\sum E_{LC}^{\mathrm{model\ mask}} / E_{\mathrm{pred}}$: med %.3f"
                  % np.median(r3))
    ax.axvline(1, color="k", ls="--", lw=1.5)
    ax.set_xlabel("energy ratio")
    ax.set_ylabel("SimClusters / bin")
    ax.set_title("Where the energy lives: truth vs model mask", fontsize=19)
    ax.legend(fontsize=11)
    savefig(fig, outdir, "energy_closure_masks")
    metrics["mask_energy_closure"] = dict(
        median_truthmask_over_true=float(np.median(r)),
        median_predmask_over_true=float(np.median(r2)),
        median_predmask_over_pred=float(np.median(r3)),
        corr_true_vs_truthmask=float(np.corrcoef(data.p_e, data.p_sum_e_true_nodes)[0, 1]),
        corr_pred_vs_predmask=float(np.corrcoef(data.p_pred_e, data.p_sum_e_pred_nodes)[0, 1]),
        corr_pred_vs_truthmask=float(np.corrcoef(data.p_pred_e, data.p_sum_e_true_nodes)[0, 1]),
    )


def response_vs_n_nodes(data, outdir, masks, metrics):
    fig, ax = plt.subplots()
    for key, name, color in MASKS:
        m = masks[key] & (data.p_e > 1e-3) & (data.p_pred_e > 1e-3)
        lr = np.log(data.p_pred_e[m] / data.p_e[m])
        ctr, med, sig, n = profile(data.p_pred_n_nodes[m], lr, NODE_EDGES)
        ok = np.isfinite(med)
        ax.errorbar(ctr[ok], med[ok], yerr=sig[ok],
                    xerr=[(ctr - NODE_EDGES[:-1])[ok], (NODE_EDGES[1:] - ctr)[ok]],
                    fmt="o", color=color, lw=2, ms=7, label=name)
        metrics.setdefault("logratio_vs_n_nodes", {})[key] = dict(
            centre=ctr.tolist(),
            median=np.where(np.isfinite(med), med, None).tolist(),
            sigma68=np.where(np.isfinite(sig), sig, None).tolist(), n=n.tolist())
    ax.axhline(0, color="k", ls="--", lw=1.5)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("pred_n_nodes")
    ax.set_ylabel(r"$\ln(E_{\mathrm{pred}}/E_{\mathrm{true}})\ \pm\ \sigma_{68}$")
    ax.set_title("Energy response vs number of predicted nodes", fontsize=19)
    ax.legend(fontsize=13)
    savefig(fig, outdir, "logratio_vs_pred_n_nodes")

    # resolution (sigma68) only
    fig, ax = plt.subplots()
    for key, name, color in MASKS:
        m = masks[key] & (data.p_e > 1e-3) & (data.p_pred_e > 1e-3)
        lr = np.log(data.p_pred_e[m] / data.p_e[m])
        ctr, med, sig, n = profile(data.p_pred_n_nodes[m], lr, NODE_EDGES)
        ok = np.isfinite(sig)
        ax.plot(ctr[ok], sig[ok], "o-", color=color, lw=2, ms=7, label=name)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("pred_n_nodes")
    ax.set_ylabel(r"$\sigma_{68}$ of $\ln(E_{\mathrm{pred}}/E_{\mathrm{true}})$")
    ax.set_title("Energy resolution vs number of predicted nodes", fontsize=19)
    ax.legend(fontsize=13)
    savefig(fig, outdir, "sigma_vs_pred_n_nodes")


def alternative_targets(data, outdir, metrics):
    """Is pred_e maybe closer to a different truth definition?"""
    cands = {
        "recEnergy": data.p_e,
        "boundaryEnergy": data.p_e_boundary,
        "sum(LC) truth-associated": data.p_sum_e_true_nodes,
        "sum(LC) model mask": data.p_sum_e_pred_nodes,
    }
    rows = {}
    for name, t in cands.items():
        m = (t > 1e-3) & (data.p_pred_e > 1e-3)
        lr = np.log(data.p_pred_e[m] / t[m])
        b, a = np.polyfit(np.log(t[m]), np.log(data.p_pred_e[m]), 1)
        rows[name] = dict(median_log_ratio=float(np.median(lr)),
                          sigma68_log_ratio=float(robust_sigma(lr)),
                          corr_log=float(np.corrcoef(np.log(t[m]), np.log(data.p_pred_e[m]))[0, 1]),
                          slope=float(b), n=int(m.sum()),
                          sum_ratio=float(data.p_pred_e[m].sum() / t[m].sum()))
    metrics["alternative_truth_definitions"] = rows

    fig, ax = plt.subplots(figsize=(11, 6))
    names = list(rows)
    ax.bar(range(len(names)), [rows[n]["corr_log"] for n in names], color=CMS_blue,
           alpha=0.85, label=r"$\rho$ of $\log E$")
    ax.bar(range(len(names)), [rows[n]["slope"] for n in names], width=0.4,
           color=CMS_red, alpha=0.85, label="log-log slope")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=11, rotation=15, ha="right")
    ax.axhline(1, color="k", ls="--", lw=1.2)
    ax.set_ylabel("value")
    ax.set_title("Which truth definition does pred_e track best?", fontsize=18)
    ax.legend(fontsize=13)
    savefig(fig, outdir, "alternative_truth_definitions")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="plots/maskformer_eval/energy")
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()

    data = EvalData(args.input, max_events=args.max_events)
    masks = get_masks(data)
    metrics = dict(dataset=data.summary())
    os.makedirs(args.outdir, exist_ok=True)

    print("[energy] 2D correlation ..."); plot_2d(data, args.outdir, masks)
    print("[energy] response profiles ..."); plot_response(data, args.outdir, masks, metrics)
    print("[energy] log-log fit ..."); loglog_fit(data, args.outdir, masks, metrics)
    print("[energy] log-ratio ..."); log_ratio_distribution(data, args.outdir, masks, metrics)
    print("[energy] energy budget ..."); energy_budget(data, args.outdir, metrics)
    print("[energy] mask closure ..."); truth_sanity(data, args.outdir, metrics)
    print("[energy] vs pred_n_nodes ..."); response_vs_n_nodes(data, args.outdir, masks, metrics)
    print("[energy] alternative targets ..."); alternative_targets(data, args.outdir, metrics)

    with open(os.path.join(args.outdir, "energy_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    report(metrics, os.path.join(args.outdir, "energy_summary.txt"))


def report(m, path):
    L = ["=" * 78, "Energy investigation", "=" * 78]
    L.append("")
    L.append("-- log-log fit  E_pred = A * E_true^b --------------------------------")
    for k, v in m["loglog_fit"].items():
        L.append("  %-12s A=%.3f  b=%.3f  rho_log=%.3f  sigma68(ln ratio)=%.3f (x%.2f)"
                 % (k, v["amplitude"], v["slope"], v["log_corr"], v["sigma68_log_ratio"],
                    np.exp(v["sigma68_log_ratio"])))
        L.append("               std ln E: true %.3f pred %.3f  pivot %.3f GeV"
                 % (v["std_log_true"], v["std_log_pred"], v["pivot_energy_GeV"]))
        L.append("               rho * std_pred/std_true = %.3f  (== slope if pure dilution)"
                 % v["shrinkage_check_corr_times_stdratio"])
    L.append("")
    L.append("-- energy budget -----------------------------------------------------")
    for k, v in m["energy_budget"].items():
        L.append("  %-34s %12.4f" % (k, v))
    L.append("")
    L.append("-- median response vs TRUE energy ------------------------------------")
    r = m["response_true"]["all_queries"]
    for c, v, n in zip(r["centre"], r["median_ratio"], r["n"]):
        if v is not None:
            L.append("  E_true ~ %8.2f GeV : E_pred/E_true = %6.3f   (N=%d)" % (c, v, n))
    L.append("")
    L.append("-- median response vs PRED energy ------------------------------------")
    r = m["response_pred"]["all_queries"]
    for c, v, n in zip(r["centre"], r["median_ratio"], r["n"]):
        if v is not None:
            L.append("  E_pred ~ %8.2f GeV : E_pred/E_true = %6.3f   (N=%d)" % (c, v, n))
    L.append("")
    L.append("-- mask energy closure -----------------------------------------------")
    for k, v in m["mask_energy_closure"].items():
        L.append("  %-34s %.4f" % (k, v))
    L.append("")
    L.append("-- alternative truth definitions -------------------------------------")
    for k, v in m["alternative_truth_definitions"].items():
        L.append("  %-28s rho_log %.3f  slope %.3f  med ln ratio %+.3f  sigma68 %.3f  sum ratio %.3f"
                 % (k, v["corr_log"], v["slope"], v["median_log_ratio"],
                    v["sigma68_log_ratio"], v["sum_ratio"]))
    L.append("")
    L.append("-- ln(Epred/Etrue) vs pred_n_nodes -----------------------------------")
    r = m["logratio_vs_n_nodes"]["all_queries"]
    for c, med, s, n in zip(r["centre"], r["median"], r["sigma68"], r["n"]):
        if med is not None:
            L.append("  n_nodes ~ %7.1f : median %+6.3f  sigma68 %6.3f  (N=%d)" % (c, med, s, n))
    txt = "\n".join(L)
    with open(path, "w") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
