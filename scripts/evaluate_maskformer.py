#!/usr/bin/env python
"""Comprehensive evaluation of an HGCAL MaskFormer prediction file.

Produces, under ``--outdir``:

  resolution/          inclusive resolution of e, eta, sinphi, cosphi, r
  resolution_by_class/ the same, one plot per variable with the true particle
                       classes overlaid
  efficiency/          clustering (node-assignment) efficiency, inclusively and
                       vs. particle energy / eta / type
  diagnostics/         supporting distributions

and a machine-readable ``metrics.json`` + human-readable ``summary.txt``.

Usage
-----
    python scripts/evaluate_maskformer.py \
        --input data/hgcal_v1_20260818-T193435/epoch=199-val_loss=11.54392__test.parquet \
        --outdir plots/maskformer_eval
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hgcal_eval_common import (  # noqa: E402
    CLASS_INFO, CMAP_10, CMS_blue, CMS_gray, CMS_orange, CMS_red,
    EvalData, PHYSICS_CLASSES, _f, class_color, class_name, eff_profile,
    profile, robust_sigma, savefig, setup_style, stat_label,
)

plt = setup_style()

# --------------------------------------------------------------------------- #
# variable definitions
# --------------------------------------------------------------------------- #
# kind = "rel" -> (pred - true)/true ; "abs" -> pred - true
VARIABLES = [
    dict(key="e",      label=r"$E$",           unit=" GeV", kind="rel",
         truth="p_e",      pred="p_pred_e",      rng=(-2.0, 3.0),  bins=120),
    dict(key="eta",    label=r"$\eta$",        unit="",     kind="abs",
         truth="p_eta",    pred="p_pred_eta",    rng=(-1.0, 1.0),  bins=120),
    dict(key="sinphi", label=r"$\sin\phi$",    unit="",     kind="abs",
         truth="p_sinphi", pred="p_pred_sinphi", rng=(-1.0, 1.0),  bins=120),
    dict(key="cosphi", label=r"$\cos\phi$",    unit="",     kind="abs",
         truth="p_cosphi", pred="p_pred_cosphi", rng=(-1.0, 1.0),  bins=120),
    dict(key="r",      label=r"$r$",           unit=" cm",  kind="abs",
         truth="p_r",      pred="p_pred_r",      rng=(-40.0, 40.0), bins=120),
]

E_EDGES = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0])
ETA_EDGES = np.linspace(1.4, 3.2, 13)
NODE_EDGES = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 700])


def residual(data, var, mask):
    t = getattr(data, var["truth"])[mask]
    p = getattr(data, var["pred"])[mask]
    if var["kind"] == "rel":
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(t > 0, (p - t) / t, np.nan)
    else:
        res = p - t
    return res


def residual_axis_label(var):
    if var["kind"] == "rel":
        return r"$(%s_{\mathrm{pred}} - %s_{\mathrm{true}})\,/\,%s_{\mathrm{true}}$" % (
            var["label"].strip("$"), var["label"].strip("$"), var["label"].strip("$"))
    return r"$%s_{\mathrm{pred}} - %s_{\mathrm{true}}$%s" % (
        var["label"].strip("$"), var["label"].strip("$"),
        (" [%s]" % var["unit"].strip()) if var["unit"].strip() else "")


# --------------------------------------------------------------------------- #
# 1. inclusive resolutions
# --------------------------------------------------------------------------- #

def plot_inclusive_resolutions(data, outdir, metrics):
    sub = os.path.join(outdir, "resolution")
    all_mask = np.ones(len(data.p_e), bool)
    for var in VARIABLES:
        fig, ax = plt.subplots()
        bins = np.linspace(var["rng"][0], var["rng"][1], var["bins"] + 1)
        for mask, name, color in [
            (all_mask, "all matched queries", CMS_blue),
            (data.p_valid, "pred_valid only", CMS_red),
        ]:
            res = residual(data, var, mask)
            res = res[np.isfinite(res)]
            ax.hist(res, bins=bins, histtype="step", lw=2, color=color,
                    label="%s\n%s" % (name, stat_label(res)))
            metrics.setdefault("resolution_inclusive", {})[
                "%s_%s" % (var["key"], "valid" if name.startswith("pred") else "all")] = dict(
                median=float(np.median(res)), sigma68=float(robust_sigma(res)),
                mean=float(res.mean()), n=int(len(res)),
                overflow_frac=float(np.mean(np.abs(residual(data, var, mask)) > max(abs(var["rng"][0]), var["rng"][1]))),
            )
        ax.axvline(0.0, color="k", ls=":", lw=1.2)
        ax.set_xlabel(residual_axis_label(var))
        ax.set_ylabel("SimClusters / bin")
        ax.set_title("Inclusive %s resolution" % var["label"], fontsize=20)
        ax.legend(fontsize=13, loc="upper right")
        savefig(fig, sub, "resolution_%s" % var["key"])

        # log-y companion: the tails matter for energy
        fig, ax = plt.subplots()
        for mask, name, color in [(all_mask, "all matched queries", CMS_blue),
                                  (data.p_valid, "pred_valid only", CMS_red)]:
            res = residual(data, var, mask)
            ax.hist(res[np.isfinite(res)], bins=bins, histtype="step", lw=2,
                    color=color, label=name)
        ax.set_yscale("log")
        ax.axvline(0.0, color="k", ls=":", lw=1.2)
        ax.set_xlabel(residual_axis_label(var))
        ax.set_ylabel("SimClusters / bin")
        ax.set_title("Inclusive %s resolution (log)" % var["label"], fontsize=20)
        ax.legend(fontsize=14)
        savefig(fig, sub, "resolution_%s_logy" % var["key"])


# --------------------------------------------------------------------------- #
# 2. resolutions split by true particle class
# --------------------------------------------------------------------------- #

def plot_resolutions_by_class(data, outdir, metrics):
    sub = os.path.join(outdir, "resolution_by_class")
    for valid_only in (False, True):
        base = data.p_valid if valid_only else np.ones(len(data.p_e), bool)
        tag = "validmask" if valid_only else "allqueries"
        for var in VARIABLES:
            fig, ax = plt.subplots()
            bins = np.linspace(var["rng"][0], var["rng"][1], var["bins"] + 1)
            for c in PHYSICS_CLASSES:
                mask = base & (data.p_class == c)
                if mask.sum() < 10:
                    continue
                res = residual(data, var, mask)
                res = res[np.isfinite(res)]
                ax.hist(res, bins=bins, histtype="step", lw=2, density=True,
                        color=class_color(c),
                        label="%s: med %.3g, $\\sigma_{68}$ %.3g (N=%d)"
                              % (class_name(c), np.median(res), robust_sigma(res), len(res)))
                if not valid_only:
                    metrics.setdefault("resolution_by_class", {}).setdefault(
                        var["key"], {})[class_name(c)] = dict(
                        median=float(np.median(res)), sigma68=float(robust_sigma(res)),
                        n=int(len(res)))
            ax.axvline(0.0, color="k", ls=":", lw=1.2)
            ax.set_xlabel(residual_axis_label(var))
            ax.set_ylabel("normalised SimClusters / bin")
            ax.set_title("%s resolution by true class (%s)"
                         % (var["label"], "pred_valid" if valid_only else "all queries"),
                         fontsize=19)
            ax.legend(fontsize=12, loc="upper right")
            savefig(fig, sub, "resolution_%s_byclass_%s" % (var["key"], tag))


# --------------------------------------------------------------------------- #
# 2b. energy resolution in log space
# --------------------------------------------------------------------------- #
# (E_pred - E_true)/E_true is bounded below by -1 and unbounded above, so on a
# steeply falling spectrum it is dominated by a spike at -1.  ln(E_pred/E_true)
# is symmetric under pred<->true and is the honest resolution variable here.

def plot_energy_logratio(data, outdir, metrics):
    good = (data.p_e > 1e-3) & (data.p_pred_e > 1e-3)
    bins = np.linspace(-6, 6, 121)

    fig, ax = plt.subplots()
    for mask, name, color in [(good, "all matched queries", CMS_blue),
                              (good & data.p_valid, "pred_valid only", CMS_red)]:
        lr = np.log(data.p_pred_e[mask] / data.p_e[mask])
        ax.hist(lr, bins=bins, histtype="step", lw=2, color=color,
                label="%s\n%s" % (name, stat_label(lr)))
        metrics.setdefault("resolution_inclusive", {})[
            "e_logratio_%s" % ("valid" if "valid" in name else "all")] = dict(
            median=float(np.median(lr)), sigma68=float(robust_sigma(lr)),
            mean=float(lr.mean()), n=int(len(lr)),
            sigma68_as_factor=float(np.exp(robust_sigma(lr))))
    ax.axvline(0, color="k", ls=":", lw=1.2)
    ax.set_xlabel(r"$\ln(E_{\mathrm{pred}}/E_{\mathrm{true}})$")
    ax.set_ylabel("SimClusters / bin")
    ax.set_title("Inclusive energy resolution (log ratio)", fontsize=20)
    ax.legend(fontsize=13)
    savefig(fig, os.path.join(outdir, "resolution"), "resolution_e_logratio")

    fig, ax = plt.subplots()
    for c in PHYSICS_CLASSES:
        mask = good & (data.p_class == c)
        if mask.sum() < 30:
            continue
        lr = np.log(data.p_pred_e[mask] / data.p_e[mask])
        ax.hist(lr, bins=bins, histtype="step", lw=2, density=True, color=class_color(c),
                label="%s: med %.2f, $\\sigma_{68}$ %.2f (N=%d)"
                      % (class_name(c), np.median(lr), robust_sigma(lr), mask.sum()))
    ax.axvline(0, color="k", ls=":", lw=1.2)
    ax.set_xlabel(r"$\ln(E_{\mathrm{pred}}/E_{\mathrm{true}})$")
    ax.set_ylabel("normalised SimClusters / bin")
    ax.set_title("Energy resolution (log ratio) by true class", fontsize=19)
    ax.legend(fontsize=12)
    savefig(fig, os.path.join(outdir, "resolution_by_class"),
            "resolution_e_logratio_byclass_allqueries")


# --------------------------------------------------------------------------- #
# 2c. resolution as a function of the particle kinematics
# --------------------------------------------------------------------------- #

def plot_resolution_vs_kinematics(data, outdir, metrics):
    sub = os.path.join(outdir, "resolution")
    specs = [("e_logratio", r"$\sigma_{68}$ of $\ln(E_{pred}/E_{true})$", None),
             ("eta", r"$\sigma_{68}$ of $\Delta\eta$", VARIABLES[1]),
             ("sinphi", r"$\sigma_{68}$ of $\Delta\sin\phi$", VARIABLES[2]),
             ("cosphi", r"$\sigma_{68}$ of $\Delta\cos\phi$", VARIABLES[3]),
             ("r", r"$\sigma_{68}$ of $\Delta r$ [cm]", VARIABLES[4])]

    for xkey, xarr, edges, xlabel, logx in [
            ("energy", data.p_e, E_EDGES, r"true SimCluster $E$ [GeV]", True),
            ("eta", data.p_eta, ETA_EDGES, r"true SimCluster $\eta$", False)]:
        for name, ylabel, var in specs:
            fig, ax = plt.subplots()
            for mask, lbl, color in [(np.ones(len(data.p_e), bool), "all matched queries", CMS_blue),
                                     (data.p_valid, "pred_valid only", CMS_red)]:
                if var is None:
                    m = mask & (data.p_e > 1e-3) & (data.p_pred_e > 1e-3)
                    res = np.log(data.p_pred_e[m] / data.p_e[m])
                else:
                    m = mask
                    res = residual(data, var, m)
                ctr, med, sig, n = profile(xarr[m], res, edges)
                ok = np.isfinite(sig)
                ax.plot(ctr[ok], sig[ok], "o-", color=color, lw=2, ms=7, label=lbl)
                if xkey == "energy" and lbl.startswith("all"):
                    metrics.setdefault("resolution_vs_energy", {})[name] = dict(
                        centre=ctr.tolist(),
                        sigma68=np.where(np.isfinite(sig), sig, None).tolist(),
                        median=np.where(np.isfinite(med), med, None).tolist(),
                        n=n.tolist())
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.set_title("%s resolution vs true %s" % (name, xkey), fontsize=19)
            ax.legend(fontsize=13)
            savefig(fig, sub, "sigma68_%s_vs_%s" % (name, xkey))


# --------------------------------------------------------------------------- #
# 3. clustering efficiency
# --------------------------------------------------------------------------- #

def clustering_efficiency(data):
    """Node-level assignment bookkeeping.

    ``ok``  : node was assigned to the same particle the truth associator picked
    ``has_truth``: node has at least one truth association (the denominator for
    the "correctly predicted indices / total indices" definition)
    """
    has_truth = data.n_true >= 0
    assigned = data.n_pred >= 0
    ok = has_truth & assigned & (data.n_pred == data.n_true)
    return has_truth, assigned, ok


def plot_efficiency(data, outdir, metrics):
    sub = os.path.join(outdir, "efficiency")
    has_truth, assigned, ok = clustering_efficiency(data)
    w = data.n_energy

    eff_all = ok.sum() / len(ok)
    eff_truth = ok[has_truth].sum() / max(has_truth.sum(), 1)
    eff_both = ok[has_truth & assigned].sum() / max((has_truth & assigned).sum(), 1)
    eff_ew = w[ok].sum() / max(w[has_truth].sum(), 1e-12)
    frac_unassigned = 1.0 - assigned.mean()

    metrics["clustering_efficiency"] = dict(
        n_nodes=int(len(ok)),
        n_nodes_with_truth=int(has_truth.sum()),
        n_nodes_assigned=int(assigned.sum()),
        eff_over_all_nodes=float(eff_all),
        eff_over_nodes_with_truth=float(eff_truth),
        eff_over_assigned_and_truth=float(eff_both),
        eff_energy_weighted=float(eff_ew),
        frac_nodes_left_unassigned=float(frac_unassigned),
    )

    # ---- 3a. the inclusive number, as a bar chart ----
    fig, ax = plt.subplots(figsize=(11, 6))
    names = ["correct\n(/ all nodes)", "correct\n(/ nodes with truth)",
             "correct\n(/ assigned & truth)", "correct, E-weighted",
             "nodes left\nunassigned"]
    vals = [eff_all, eff_truth, eff_both, eff_ew, frac_unassigned]
    cols = [CMS_blue, CMS_blue, CMS_blue, CMS_orange, CMS_gray]
    bars = ax.bar(range(len(vals)), vals, color=cols, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, "%.3f" % v,
                ha="center", fontsize=15)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel("fraction of LayerClusters")
    ax.set_title("Clustering (node-assignment) efficiency", fontsize=20)
    savefig(fig, sub, "clustering_efficiency_summary")

    # ---- 3b. vs particle energy / eta ----
    # each node inherits the kinematics of its TRUE particle
    nt = data.n_true[has_truth]
    ok_t = ok[has_truth]
    node_true_e = data.p_e[nt]
    node_true_eta = data.p_eta[nt]
    node_true_class = data.p_class[nt]
    node_w = w[has_truth]

    for xvals, edges, xlabel, name, logx in [
        (node_true_e, E_EDGES, r"true SimCluster $E$ (recEnergy) [GeV]", "vs_energy", True),
        (node_true_eta, ETA_EDGES, r"true SimCluster $\eta$", "vs_eta", False),
    ]:
        ctr, p, e, n = eff_profile(xvals, ok_t, edges)
        # energy weighted version
        which = np.digitize(xvals, edges) - 1
        pw = np.full(len(ctr), np.nan)
        for i in range(len(ctr)):
            sel = which == i
            den = node_w[sel].sum()
            if den > 0:
                pw[i] = node_w[sel & ok_t].sum() / den
        fig, ax = plt.subplots()
        xerr = [ctr - edges[:-1], edges[1:] - ctr]
        ax.errorbar(ctr, p, yerr=e, xerr=xerr, fmt="o", color=CMS_blue, lw=2,
                    ms=7, label="node counting")
        ax.errorbar(ctr, pw, xerr=xerr, fmt="s", color=CMS_orange, lw=2, ms=6,
                    label="energy weighted")
        if logx:
            ax.set_xscale("log")
        ax.set_ylim(0, max(0.05, np.nanmax([np.nanmax(p), np.nanmax(pw)]) * 1.35))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("correct-assignment fraction")
        ax.set_title("Clustering efficiency %s" % name.replace("_", " "), fontsize=20)
        ax.legend(fontsize=14)
        savefig(fig, sub, "clustering_efficiency_%s" % name)
        metrics.setdefault("clustering_efficiency_profiles", {})[name] = dict(
            edges=edges.tolist(), centre=ctr.tolist(),
            eff=np.where(np.isfinite(p), p, None).tolist(),
            eff_err=np.where(np.isfinite(e), e, None).tolist(),
            eff_energy_weighted=np.where(np.isfinite(pw), pw, None).tolist(),
            n=n.tolist())

    # ---- 3c. vs particle type ----
    classes = [c for c in PHYSICS_CLASSES if (node_true_class == c).sum() > 0]
    vals, errs, ns, wvals = [], [], [], []
    for c in classes:
        sel = node_true_class == c
        k, n = ok_t[sel].sum(), sel.sum()
        vals.append(k / n)
        errs.append(np.sqrt(max(k / n * (1 - k / n), 0) / n))
        ns.append(int(n))
        wvals.append(node_w[sel & ok_t].sum() / max(node_w[sel].sum(), 1e-12))
    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(classes))
    ax.bar(x - 0.2, vals, width=0.4, yerr=errs, color=[class_color(c) for c in classes],
           alpha=0.9, label="node counting", capsize=4)
    ax.bar(x + 0.2, wvals, width=0.4, color=[class_color(c) for c in classes],
           alpha=0.45, hatch="//", label="energy weighted")
    for xi, v, n in zip(x, vals, ns):
        ax.text(xi - 0.2, v + 0.008, "%.3f" % v, ha="center", fontsize=13)
        ax.text(xi - 0.2, -0.035 * max(vals + wvals), "N=%d" % n, ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([class_name(c) for c in classes], fontsize=13, rotation=12)
    ax.set_ylabel("correct-assignment fraction")
    ax.set_ylim(-0.06 * max(vals + wvals), max(vals + wvals) * 1.3)
    ax.set_title("Clustering efficiency vs true particle type", fontsize=20)
    ax.legend(fontsize=14)
    savefig(fig, sub, "clustering_efficiency_vs_type")
    metrics.setdefault("clustering_efficiency_profiles", {})["vs_type"] = {
        class_name(c): dict(eff=float(v), eff_err=float(er), n=int(n),
                            eff_energy_weighted=float(wv))
        for c, v, er, n, wv in zip(classes, vals, errs, ns, wvals)}

    # ---- 3d. per-particle recall / purity ----
    recall = np.divide(
        np.bincount(data.n_true[ok], minlength=len(data.p_e)).astype(float),
        np.maximum(data.p_n_true_nodes, 1),
        out=np.full(len(data.p_e), np.nan), where=data.p_n_true_nodes > 0)
    purity = np.divide(
        np.bincount(data.n_pred[ok], minlength=len(data.p_e)).astype(float),
        np.maximum(data.p_n_pred_nodes, 1),
        out=np.full(len(data.p_e), np.nan), where=data.p_n_pred_nodes > 0)
    fig, ax = plt.subplots()
    ax.hist(recall[np.isfinite(recall)], bins=50, range=(0, 1), histtype="step",
            lw=2, color=CMS_blue,
            label="per-particle recall  %s" % stat_label(recall[np.isfinite(recall)]))
    ax.hist(purity[np.isfinite(purity)], bins=50, range=(0, 1), histtype="step",
            lw=2, color=CMS_red,
            label="per-particle purity  %s" % stat_label(purity[np.isfinite(purity)]))
    ax.set_yscale("log")
    ax.set_xlabel("fraction of nodes")
    ax.set_ylabel("SimClusters / bin")
    ax.set_title("Per-particle clustering recall and purity", fontsize=20)
    ax.legend(fontsize=12)
    savefig(fig, sub, "per_particle_recall_purity")
    metrics["per_particle"] = dict(
        recall_median=float(np.nanmedian(recall)),
        recall_mean=float(np.nanmean(recall)),
        purity_median=float(np.nanmedian(purity)),
        purity_mean=float(np.nanmean(purity)),
        frac_particles_with_zero_recall=float(np.nanmean(recall == 0)),
    )

    # recall vs energy, split by class
    fig, ax = plt.subplots()
    fin = np.isfinite(recall)
    ctr, p, e, n = eff_profile(data.p_e[fin], recall[fin] > 0.5, E_EDGES)
    ax.errorbar(ctr, p, yerr=e, fmt="o-", color="k", lw=2, ms=7, label="all")
    for c in PHYSICS_CLASSES:
        sel = fin & (data.p_class == c)
        if sel.sum() < 40:
            continue
        ctr, p, e, n = eff_profile(data.p_e[sel], recall[sel] > 0.5, E_EDGES)
        ax.errorbar(ctr, p, yerr=e, fmt="o-", color=class_color(c), lw=1.8, ms=5,
                    alpha=0.9, label=class_name(c))
    ax.set_xscale("log")
    ax.set_xlabel(r"true SimCluster $E$ (recEnergy) [GeV]")
    ax.set_ylabel("fraction of SimClusters with recall > 0.5")
    ax.set_ylim(0, 1.05)
    ax.set_title("Particle-level clustering efficiency", fontsize=20)
    ax.legend(fontsize=12, ncol=2)
    savefig(fig, sub, "particle_efficiency_vs_energy_byclass")

    return recall, purity


# --------------------------------------------------------------------------- #
# 4. diagnostics
# --------------------------------------------------------------------------- #

def plot_diagnostics(data, outdir, metrics):
    sub = os.path.join(outdir, "diagnostics")

    # pred_valid rate vs energy / eta
    fig, ax = plt.subplots()
    ctr, p, e, n = eff_profile(data.p_e, data.p_valid, E_EDGES)
    ax.errorbar(ctr, p, yerr=e, fmt="o-", color=CMS_blue, lw=2, ms=7)
    ax.set_xscale("log")
    ax.set_xlabel(r"true SimCluster $E$ (recEnergy) [GeV]")
    ax.set_ylabel("fraction with pred_valid = True")
    ax.set_ylim(0, 1.0)
    ax.set_title("Query-validity rate vs true energy", fontsize=20)
    savefig(fig, sub, "pred_valid_vs_energy")

    # class confusion
    fig, ax = plt.subplots(figsize=(9, 8))
    cls = PHYSICS_CLASSES + [5]
    M = np.zeros((len(cls), len(cls)))
    for i, ct in enumerate(cls):
        for j, cp in enumerate(cls):
            M[i, j] = ((data.p_class == ct) & (data.p_pred_class == cp)).sum()
    Mn = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cls)):
        for j in range(len(cls)):
            if M[i, j]:
                ax.text(j, i, "%.2f" % Mn[i, j], ha="center", va="center",
                        fontsize=11, color="white" if Mn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(cls))); ax.set_yticks(range(len(cls)))
    ax.set_xticklabels([class_name(c) for c in cls], rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels([class_name(c) for c in cls], fontsize=11)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    ax.set_title("Classification confusion (row-normalised)", fontsize=19)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    savefig(fig, sub, "class_confusion")
    metrics["classification"] = dict(
        pred_class_counts={class_name(c): int((data.p_pred_class == c).sum()) for c in cls},
        true_class_counts={class_name(c): int((data.p_class == c).sum()) for c in cls},
        accuracy_valid=float((data.p_pred_class[data.p_valid]
                              == data.p_class[data.p_valid]).mean()),
    )

    # pred_n_nodes vs the actual number of assigned nodes
    fig, ax = plt.subplots()
    ax.hist(data.p_pred_n_nodes, bins=np.logspace(0, 3, 40), histtype="step", lw=2,
            color=CMS_blue, label="pred_n_nodes (mask above threshold)")
    ax.hist(data.p_n_pred_nodes, bins=np.logspace(0, 3, 40), histtype="step", lw=2,
            color=CMS_red, label="nodes with pred_match_idx = i (exclusive)")
    ax.hist(data.p_n_true_nodes, bins=np.logspace(0, 3, 40), histtype="step", lw=2,
            color=CMS_gray, label="nodes truly belonging to i")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of nodes per SimCluster")
    ax.set_ylabel("SimClusters / bin")
    ax.set_title("Node multiplicity: mask vs exclusive assignment", fontsize=19)
    ax.legend(fontsize=12)
    savefig(fig, sub, "n_nodes_comparison")
    metrics["n_nodes"] = dict(
        pred_n_nodes_mean=float(data.p_pred_n_nodes.mean()),
        exclusive_assigned_mean=float(data.p_n_pred_nodes.mean()),
        true_mean=float(data.p_n_true_nodes.mean()),
        frac_pred_n_nodes_zero=float((data.p_pred_n_nodes == 0).mean()),
        frac_exclusive_zero=float((data.p_n_pred_nodes == 0).mean()),
    )

    # ---- node-level assignment confusion: which particle do wrong nodes go to? ----
    has_truth, assigned, ok = clustering_efficiency(data)
    sel = has_truth & assigned
    tc = data.p_class[data.n_true[sel]]
    pc = data.p_class[data.n_pred[sel]]
    cls = PHYSICS_CLASSES
    M = np.zeros((len(cls), len(cls)))
    for i, ct in enumerate(cls):
        for j, cp in enumerate(cls):
            M[i, j] = ((tc == ct) & (pc == cp)).sum()
    Mn = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(Mn, cmap="Oranges", vmin=0, vmax=1)
    for i in range(len(cls)):
        for j in range(len(cls)):
            if M[i, j]:
                ax.text(j, i, "%.2f" % Mn[i, j], ha="center", va="center", fontsize=12,
                        color="white" if Mn[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(cls))); ax.set_yticks(range(len(cls)))
    ax.set_xticklabels([class_name(c) for c in cls], rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels([class_name(c) for c in cls], fontsize=11)
    ax.set_xlabel("assigned SimCluster class")
    ax.set_ylabel("true SimCluster class")
    ax.set_title("Node-assignment leakage between particle types", fontsize=17)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    savefig(fig, sub, "node_assignment_class_leakage")
    metrics["node_assignment_leakage"] = {
        class_name(ct): {class_name(cp): float(Mn[i, j]) for j, cp in enumerate(cls)}
        for i, ct in enumerate(cls)}

    # ---- sample composition: how many SimClusters share a CaloParticle? ----
    cp_local = _f(data.df.SimCluster.SimCluster_CaloPartIdx).astype(np.int64)
    cp_glob = data.p_event.astype(np.int64) * 100000 + cp_local
    uniq, counts = np.unique(cp_glob, return_counts=True)
    same_cp = cp_glob[data.n_pred[sel]] == cp_glob[data.n_true[sel]]
    metrics["sample_structure"] = dict(
        n_caloparticles=int(len(uniq)),
        simclusters_per_caloparticle_mean=float(counts.mean()),
        simclusters_per_caloparticle_max=int(counts.max()),
        node_assignment_same_caloparticle=float(same_cp.mean()),
        node_assignment_same_simcluster=float(ok[sel].mean()),
    )
    fig, ax = plt.subplots()
    ax.hist(counts, bins=np.arange(0.5, counts.max() + 1.5), histtype="step", lw=2,
            color=CMS_blue, label="SimClusters per CaloParticle\n(mean %.1f)" % counts.mean())
    ax.hist(data.p_count, bins=np.arange(0.5, data.p_count.max() + 1.5), histtype="step",
            lw=2, color=CMS_orange, label="SimClusters per event\n(mean %.1f)" % data.p_count.mean())
    ax.set_yscale("log")
    ax.set_xlabel("multiplicity")
    ax.set_ylabel("entries / bin")
    ax.set_title("Sample structure: the task is shower sub-structure", fontsize=18)
    ax.legend(fontsize=13)
    savefig(fig, sub, "sample_structure")


# --------------------------------------------------------------------------- #
# 5. resolution vs pred_n_nodes
# --------------------------------------------------------------------------- #

def plot_resolution_vs_n_nodes(data, outdir, metrics):
    sub = os.path.join(outdir, "resolution")
    for valid_only in (False, True):
        base = data.p_valid if valid_only else np.ones(len(data.p_e), bool)
        tag = "validmask" if valid_only else "allqueries"
        res = residual(data, VARIABLES[0], base)
        nn = data.p_pred_n_nodes[base]
        e_true = data.p_e[base]
        good = np.isfinite(res) & (e_true > 0)

        fig, ax = plt.subplots()
        ctr, med, sig, n = profile(nn[good], res[good], NODE_EDGES)
        ax.errorbar(ctr, med, yerr=sig, xerr=[ctr - NODE_EDGES[:-1], NODE_EDGES[1:] - ctr],
                    fmt="o", color=CMS_blue, lw=2, ms=7,
                    label=r"median $\pm\ \sigma_{68}$")
        ax.axhline(0, color="k", ls=":", lw=1.2)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("pred_n_nodes")
        ax.set_ylabel(r"$(E_{\mathrm{pred}}-E_{\mathrm{true}})/E_{\mathrm{true}}$")
        ax.set_title("Energy response vs pred_n_nodes (%s)"
                     % ("pred_valid" if valid_only else "all queries"), fontsize=19)
        ax.legend(fontsize=13)
        for c, v, nn_ in zip(ctr, med, n):
            if np.isfinite(v):
                ax.annotate("%d" % nn_, (c, v), textcoords="offset points",
                            xytext=(0, 12), fontsize=9, ha="center", color=CMS_gray)
        savefig(fig, sub, "energy_response_vs_n_nodes_%s" % tag)

        # sigma68 of the log-ratio, a scale-free resolution measure
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(np.where((data.p_pred_e[base] > 0) & (e_true > 0),
                                 data.p_pred_e[base] / np.where(e_true > 0, e_true, np.nan),
                                 np.nan))
        fig, ax = plt.subplots()
        ctr, med, sig, n = profile(nn[np.isfinite(lr)], lr[np.isfinite(lr)], NODE_EDGES)
        ax.errorbar(ctr, sig, xerr=[ctr - NODE_EDGES[:-1], NODE_EDGES[1:] - ctr],
                    fmt="s", color=CMS_red, lw=2, ms=7, label=r"$\sigma_{68}$")
        ax.errorbar(ctr, med, xerr=[ctr - NODE_EDGES[:-1], NODE_EDGES[1:] - ctr],
                    fmt="o", color=CMS_blue, lw=2, ms=7, label="median")
        ax.axhline(0, color="k", ls=":", lw=1.2)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("pred_n_nodes")
        ax.set_ylabel(r"$\ln(E_{\mathrm{pred}}/E_{\mathrm{true}})$")
        ax.set_title("Log energy response vs pred_n_nodes (%s)"
                     % ("pred_valid" if valid_only else "all queries"), fontsize=19)
        ax.legend(fontsize=13)
        savefig(fig, sub, "energy_logresponse_vs_n_nodes_%s" % tag)

        if not valid_only:
            metrics.setdefault("energy_vs_n_nodes", {})["all"] = dict(
                edges=NODE_EDGES.tolist(), centre=ctr.tolist(),
                median_log_ratio=np.where(np.isfinite(med), med, None).tolist(),
                sigma68_log_ratio=np.where(np.isfinite(sig), sig, None).tolist(),
                n=n.tolist())


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="plots/maskformer_eval")
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()

    print("[eval] loading %s" % args.input)
    data = EvalData(args.input, max_events=args.max_events)
    print("[eval] %s" % data.summary())

    metrics = dict(dataset=data.summary())
    os.makedirs(args.outdir, exist_ok=True)

    print("[eval] inclusive resolutions ...")
    plot_inclusive_resolutions(data, args.outdir, metrics)
    print("[eval] resolutions by class ...")
    plot_resolutions_by_class(data, args.outdir, metrics)
    print("[eval] energy log-ratio resolution ...")
    plot_energy_logratio(data, args.outdir, metrics)
    print("[eval] resolution vs kinematics ...")
    plot_resolution_vs_kinematics(data, args.outdir, metrics)
    print("[eval] clustering efficiency ...")
    plot_efficiency(data, args.outdir, metrics)
    print("[eval] resolution vs pred_n_nodes ...")
    plot_resolution_vs_n_nodes(data, args.outdir, metrics)
    print("[eval] diagnostics ...")
    plot_diagnostics(data, args.outdir, metrics)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    write_summary(metrics, os.path.join(args.outdir, "summary.txt"))
    print("[eval] done -> %s" % args.outdir)


def write_summary(m, path):
    L = []
    d = m["dataset"]
    L.append("=" * 78)
    L.append("MaskFormer evaluation summary")
    L.append("=" * 78)
    L.append("file            : %s" % d["file"])
    L.append("events          : %d" % d["n_events"])
    L.append("SimClusters     : %d" % d["n_particles"])
    L.append("LayerClusters   : %d" % d["n_nodes"])
    L.append("pred_valid rate : %.4f" % d["valid_fraction"])
    L.append("")
    L.append("-- inclusive resolution (median / sigma68) ------------------------")
    for k, v in m.get("resolution_inclusive", {}).items():
        extra = ("  |res|>range %.3f" % v["overflow_frac"]) if "overflow_frac" in v else ""
        L.append("  %-16s median %+8.4f  sigma68 %8.4f  N=%d%s"
                 % (k, v["median"], v["sigma68"], v["n"], extra))
    L.append("")
    L.append("-- clustering efficiency -------------------------------------------")
    ce = m["clustering_efficiency"]
    for k, v in ce.items():
        L.append("  %-32s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))
    L.append("")
    L.append("-- per-particle ----------------------------------------------------")
    for k, v in m["per_particle"].items():
        L.append("  %-34s %.4f" % (k, v))
    L.append("")
    L.append("-- efficiency vs particle type -------------------------------------")
    for k, v in m.get("clustering_efficiency_profiles", {}).get("vs_type", {}).items():
        L.append("  %-18s eff %.4f +- %.4f  (E-weighted %.4f)  N=%d"
                 % (k, v["eff"], v["eff_err"], v["eff_energy_weighted"], v["n"]))
    L.append("")
    L.append("-- node multiplicity -----------------------------------------------")
    for k, v in m["n_nodes"].items():
        L.append("  %-34s %.4f" % (k, v))
    L.append("")
    L.append("-- sample structure ------------------------------------------------")
    for k, v in m["sample_structure"].items():
        L.append("  %-38s %.4f" % (k, v))
    L.append("")
    L.append("-- classification --------------------------------------------------")
    L.append("  accuracy (pred_valid only) %.4f" % m["classification"]["accuracy_valid"])
    L.append("  predicted counts: %s" % m["classification"]["pred_class_counts"])
    L.append("  true      counts: %s" % m["classification"]["true_class_counts"])
    txt = "\n".join(L)
    with open(path, "w") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
