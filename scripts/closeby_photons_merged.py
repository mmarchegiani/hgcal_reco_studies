#!/usr/bin/env python
"""CloseBy di-photon events that TICL reconstructs as a single photon: event displays and energy resolution.

Input: the ``eval`` directory of a layer-cluster / TruthCB MaskFormer run, holding one folder per photon distance
(``..._closeBy_n2_d<dist>cm_...``) with one prediction parquet each.  For the distances up to ``--max-distance``
(default 2 cm) the events with exactly one ``ticlTrackstersCLUEstering`` trackster are selected (the definition of
notebooks/evaluate_closeBy_photons.ipynb; ``--min-trackster-energy`` counts only the tracksters above a threshold).

Outputs under ``--outdir``:

  event_display/d<dist>cm_event<N>_etaphi.png   layer clusters in (eta, phi), three panels: Truth (TruthCB),
                                                MaskFormer, TICL.  One colour per true photon; a MaskFormer object
                                                has the colour of the photon its query is matched to; TICL
                                                tracksters have their own colours; clusters whose best query is
                                                matched to no photon are purple; unassigned clusters are gray.
  event_display/d<dist>cm_event<N>_3d.png       the same three panels in 3D (z, x, y)
  maskformer_photon_response.png                per-photon E_pred / E_true of the MaskFormer in the selected
                                                events (vs the other events of the same samples)
  maskformer_photon_resolution_vs_energy.png    its median and sigma_68 / median in bins of the true energy
  total_energy_response.png                     sum E_reco / sum E_true(TruthCB) for the MaskFormer, the TICL
                                                tracksters (raw) and the TICL candidates (regressed)
  closeby_merged_metrics.json                   all the numbers

usage: closeby_photons_merged.py --input <eval dir> --outdir plots/closeby_photons [--n-display 6]
"""

import argparse
import glob
import json
import os
import re
import sys

import awkward as ak
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hgcal_eval_common import CMAP_10, CMS_blue, CMS_gray, CMS_orange, CMS_red, robust_sigma, savefig, setup_style  # noqa: E402

plt = setup_style()

DEFAULT_INPUT = "data/hgcal_v2_layercluster_truthCaloBoundary_20261117-T234054/eval"
TS = "ticlTrackstersCLUEstering"
TRUTH_COLORS = [CMS_blue, CMS_orange, "#92dadd", "#b9ac70"]      # one per true photon (and its matched query)
TICL_COLORS = [CMS_red, "#a96b59", "#e76300", "#717581", "#b9ac70"]   # one per TICL trackster
UNMATCHED_QUERY, UNMATCHED_COLOR = -2, "#832db6"   # pred_match_idx of a cluster claimed by an extra query slot
COLUMNS = [
    "LayerCluster", "TruthCB", TS, "TICLCand", "LayerClusterToTruthCB", "event_number",
]


# --------------------------------------------------------------------------- #
# reading and selection
# --------------------------------------------------------------------------- #

def find_files(base):
    """{distance in cm: prediction parquet} of the closeBy folders under ``base``."""
    files = {}
    for folder in sorted(glob.glob(os.path.join(base, "*closeBy*"))):
        m = re.search(r"_d(\d+(?:p\d+)?)cm", os.path.basename(folder))
        pq = sorted(glob.glob(os.path.join(folder, "*.parquet")))
        if m and pq:
            files[float(m.group(1).replace("p", "."))] = pq[0]
    return dict(sorted(files.items()))


def dist_tag(d):
    return ("%g" % d).replace(".", "p")


def n_tracksters(df, min_energy=0.0):
    return ak.to_numpy(ak.sum(df[TS][TS + "_raw_energy"] > min_energy, axis=1))


def energy_table(df):
    """Per-event and per-photon energies as flat numpy arrays."""
    t = df.TruthCB
    return dict(
        e_true=ak.to_numpy(ak.flatten(t.TruthCB_energy)),
        e_pred=ak.to_numpy(ak.flatten(t.TruthCB_pred_e)),
        sum_true=ak.to_numpy(ak.sum(t.TruthCB_energy, axis=1)),
        sum_mf=ak.to_numpy(ak.sum(t.TruthCB_pred_e, axis=1)),
        sum_ts_raw=ak.to_numpy(ak.sum(df[TS][TS + "_raw_energy"], axis=1)),
        sum_cand=ak.to_numpy(ak.sum(df.TICLCand.TICLCand_energy, axis=1)),
    )


# --------------------------------------------------------------------------- #
# event display
# --------------------------------------------------------------------------- #

def dphi(a, b):
    return (np.asarray(a) - b + np.pi) % (2 * np.pi) - np.pi


def lc_truth_ids(ev, n_lc):
    """Best TruthCB particle of each layer cluster (smallest association score), -1 without link."""
    a = ev.LayerClusterToTruthCB
    counts = ak.to_numpy(a.LayerClusterToTruthCB_nLayerClusterToTruthCBLinks)
    idx = ak.unflatten(a.LayerClusterToTruthCBLinks_index, counts)
    score = ak.unflatten(a.LayerClusterToTruthCBLinks_score, counts)
    best = ak.argmin(score, axis=1, keepdims=True)
    out = ak.to_numpy(ak.fill_none(ak.firsts(idx[best]), -1)).astype(int)
    assert len(out) == n_lc
    return out


def lc_ticl_ids(ev, n_lc, min_energy=0.0):
    """Index of the trackster holding each layer cluster (the highest-energy one if shared), -1 if none."""
    t = ev[TS]
    e = ak.to_numpy(t[TS + "_raw_energy"])
    n = ak.to_numpy(t[TS + "_n" + TS + "vertices"])
    o = ak.to_numpy(t[TS + "_o" + TS + "vertices"])
    v = ak.to_numpy(t[TS + "vertices_vertices"])
    out = np.full(n_lc, -1)
    for k in np.argsort(e):          # ascending: the most energetic trackster wins
        if e[k] > min_energy:
            out[v[o[k]:o[k] + n[k]]] = k
    return out


def event_view(ev, min_ts_energy, half_width):
    """Everything the displays need for one event, restricted to an (eta, phi) window around the photons."""
    lc, t, ts = ev.LayerCluster, ev.TruthCB, ev[TS]
    f = lambda arr: ak.to_numpy(arr).astype(float)  # noqa: E731
    n_lc = len(lc.LayerCluster_energy)
    truth = dict(e=f(t.TruthCB_energy), eta=f(t.TruthCB_boundaryEta), phi=f(t.TruthCB_boundaryPhi),
                 x=f(t.TruthCB_boundaryX), y=f(t.TruthCB_boundaryY), z=f(t.TruthCB_boundaryZ))
    eta0 = float(np.mean(truth["eta"]))
    phi0 = float(np.arctan2(np.mean(np.sin(truth["phi"])), np.mean(np.cos(truth["phi"]))))

    r, eta, phi = f(t.TruthCB_pred_r), f(t.TruthCB_pred_eta), f(t.TruthCB_pred_phi)
    theta = 2 * np.arctan(np.exp(-eta))
    pred = dict(e=f(t.TruthCB_pred_e), eta=eta, phi=phi, x=r * np.sin(theta) * np.cos(phi),
                y=r * np.sin(theta) * np.sin(phi), z=r * np.cos(theta))
    keep = f(ts[TS + "_raw_energy"]) > min_ts_energy
    ticl = dict(e=f(ts[TS + "_raw_energy"])[keep], eta=f(ts[TS + "_barycenter_eta"])[keep],
                phi=f(ts[TS + "_barycenter_phi"])[keep], x=f(ts[TS + "_barycenter_x"])[keep],
                y=f(ts[TS + "_barycenter_y"])[keep], z=f(ts[TS + "_barycenter_z"])[keep],
                index=np.flatnonzero(keep))

    nodes = dict(x=f(lc.LayerCluster_x), y=f(lc.LayerCluster_y), z=f(lc.LayerCluster_z),
                 eta=f(lc.LayerCluster_eta), phi=f(lc.LayerCluster_phi), e=f(lc.LayerCluster_energy))
    ids = dict(truth=lc_truth_ids(ev, n_lc),
               maskformer=ak.to_numpy(lc.LayerCluster_pred_match_idx).astype(int),
               ticl=lc_ticl_ids(ev, n_lc, min_ts_energy))
    win = (np.abs(nodes["eta"] - eta0) < half_width) & (np.abs(dphi(nodes["phi"], phi0)) < half_width)
    # phi is drawn unwrapped around the photons, so a shower across +-pi stays in one piece
    for d in (nodes, truth, pred, ticl):
        d["phi"] = phi0 + dphi(d["phi"], phi0)
    n_total, e_total = n_lc, float(nodes["e"].sum())
    nodes = {k: v[win] for k, v in nodes.items()}
    ids = {k: v[win] for k, v in ids.items()}
    return dict(nodes=nodes, ids=ids, truth=truth, pred=pred, ticl=ticl, center=(eta0, phi0),
                frac_e_shown=float(nodes["e"].sum()) / e_total if e_total > 0 else np.nan, n_lc=n_total)


def node_sizes(e, e_ref, s_min=4.0, s_max=260.0):
    """Marker area ~ sqrt(E): a linear scale would hide all but the few clusters of the shower core."""
    return s_min + (s_max - s_min) * np.sqrt(np.clip(e / e_ref, 0.0, 1.0))


PANELS = [("truth", "Truth (TruthCB)"), ("maskformer", "MaskFormer"), ("ticl", "TICL (CLUEstering tracksters)")]


def panel_objects(view, kind):
    """[(id, colour, legend label)] of the objects of one panel."""
    tr, pr, ti = view["truth"], view["pred"], view["ticl"]
    if kind == "truth":
        return [(i, TRUTH_COLORS[i % 4], r"$\gamma_{%d}$: $E_{true}$ = %.1f GeV" % (i, tr["e"][i]))
                for i in range(len(tr["e"]))]
    if kind == "maskformer":
        return [(i, TRUTH_COLORS[i % 4], r"query $\to\gamma_{%d}$: $E_{pred}$ = %.1f GeV" % (i, pr["e"][i]))
                for i in range(len(pr["e"]))] + [(UNMATCHED_QUERY, UNMATCHED_COLOR, "query matched to no photon")]
    return [(int(k), TICL_COLORS[j % 5], r"trackster %d: $E_{raw}$ = %.1f GeV" % (k, ti["e"][j]))
            for j, k in enumerate(ti["index"])]


def draw_panel(ax, view, kind, coords, three_d=False):
    nodes, ids = view["nodes"], view["ids"][kind]
    e_ref = float(nodes["e"].max()) if len(nodes["e"]) else 1.0
    size = node_sizes(nodes["e"], e_ref)
    kw = dict(linewidths=0, depthshade=False) if three_d else dict(linewidths=0)
    pts = lambda d, m=slice(None): [d[c][m] for c in coords]  # noqa: E731
    objects = panel_objects(view, kind)
    known = np.isin(ids, [o[0] for o in objects])
    handles = []
    if (~known).any():
        ax.scatter(*pts(nodes, ~known), s=size[~known], c=CMS_gray, alpha=0.45, **kw)
    for oid, color, label in objects:
        m = ids == oid
        ax.scatter(*pts(nodes, m), s=size[m], c=color, alpha=0.75, **kw)
        share = nodes["e"][m].sum()
        handles.append(Line2D([], [], marker="o", color=color, ls="", ms=9,
                              label="%s  (%d LCs, %.1f GeV)" % (label, m.sum(), share)))
    handles.append(Line2D([], [], marker="o", color=CMS_gray, ls="", ms=9, alpha=0.6,
                          label="unassigned  (%d LCs, %.1f GeV)" % ((~known).sum(), nodes["e"][~known].sum())))
    # object positions: true photons at the calorimeter boundary in every panel, plus the panel's own objects
    mk = dict(depthshade=False) if three_d else dict(zorder=5)
    tr = view["truth"]
    for i in range(len(tr["e"])):
        ax.scatter(*pts(tr, [i]), marker="*", s=420, c=TRUTH_COLORS[i % 4] if kind == "truth" else "none",
                   edgecolors="k", linewidths=1.4, **mk)
    handles.append(Line2D([], [], marker="*", mfc="none", mec="k", ls="", ms=16, label="true photon at the boundary"))
    if kind == "maskformer":
        pr = view["pred"]
        for i in range(len(pr["e"])):
            ax.scatter(*pts(pr, [i]), marker="X", s=230, c=TRUTH_COLORS[i % 4], edgecolors="k", linewidths=1.2, **mk)
        handles.append(Line2D([], [], marker="X", mfc="w", mec="k", ls="", ms=12, label="predicted position"))
    if kind == "ticl":
        ti = view["ticl"]
        for j in range(len(ti["e"])):
            ax.scatter(*pts(ti, [j]), marker="P", s=230, c=TICL_COLORS[j % 5], edgecolors="k", linewidths=1.2, **mk)
        handles.append(Line2D([], [], marker="P", mfc="w", mec="k", ls="", ms=12, label="trackster barycenter"))
    return handles


def display_title(view, dist, event_number):
    return ("closeBy photons, d = %g cm, event %d: %d layer clusters, %.0f%% of their energy in the window"
            % (dist, event_number, view["n_lc"], 100 * view["frac_e_shown"]))


def plot_etaphi(view, dist, event_number, outdir, half_width):
    fig, axes = plt.subplots(1, 3, figsize=(27, 10.5), sharex=True, sharey=True)
    eta0, phi0 = view["center"]
    for ax, (kind, title) in zip(axes, PANELS):
        handles = draw_panel(ax, view, kind, ("eta", "phi"))
        ax.set_title(title, fontsize=22)
        ax.set_xlabel(r"$\eta$")
        ax.set_xlim(eta0 - half_width, eta0 + half_width)
        ax.set_ylim(phi0 - half_width, phi0 + half_width)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.legend(handles=handles, fontsize=13, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    axes[0].set_ylabel(r"$\phi$")
    fig.suptitle(display_title(view, dist, event_number) + "   (dot area ~ $\\sqrt{E}$)", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return savefig(fig, outdir, "d%scm_event%d_etaphi" % (dist_tag(dist), event_number), log_pdf=False)


def plot_3d(view, dist, event_number, outdir):
    fig = plt.figure(figsize=(27, 10))
    nodes = view["nodes"]
    lims = {}
    for c in ("x", "y", "z"):
        vals = np.concatenate([nodes[c], view["truth"][c]])
        pad = 0.04 * (vals.max() - vals.min() + 1e-6)
        lims[c] = (vals.min() - pad, vals.max() + pad)
    ext = np.array([lims[c][1] - lims[c][0] for c in ("z", "x", "y")])
    ext = np.maximum(ext, 0.5 * ext.max())   # equal scale, but no axis collapses
    for k, (kind, title) in enumerate(PANELS):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.grid(False)
        # z along the horizontal axis: the shower develops along it
        handles = draw_panel(ax, view, kind, ("z", "x", "y"), three_d=True)
        ax.set_xlim(*lims["z"])
        ax.set_ylim(*lims["x"])
        ax.set_zlim(*lims["y"])
        ax.set_box_aspect(tuple(ext), zoom=1.05)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel("z [cm]", labelpad=16, fontsize=15)
        ax.set_ylabel("x [cm]", labelpad=16, fontsize=15)
        ax.set_zlabel("y [cm]", labelpad=16, fontsize=15)
        ax.tick_params(labelsize=11)
        ax.view_init(elev=22, azim=-62)
        ax.set_title(title, fontsize=22, y=0.97)
        ax.legend(handles=handles, fontsize=13, loc="upper center", bbox_to_anchor=(0.5, 0.08), frameon=False)
    fig.suptitle(display_title(view, dist, event_number) + "   (dot area ~ $\\sqrt{E}$)", fontsize=20)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.12, wspace=0.02)
    return savefig(fig, outdir, "d%scm_event%d_3d" % (dist_tag(dist), event_number), log_pdf=False)


def assignment_fractions(df):
    """Per event, the fraction of the layer-cluster energy whose best query is matched to a photon, is an extra
    (unmatched) query slot, or that no query claims."""
    e, m = df.LayerCluster.LayerCluster_energy, df.LayerCluster.LayerCluster_pred_match_idx
    tot = ak.sum(e, axis=1)
    return {k: ak.to_numpy(ak.sum(e[c], axis=1) / tot)
            for k, c in (("matched_query", m >= 0), ("unmatched_query", m == UNMATCHED_QUERY), ("no_query", m == -1))}


# --------------------------------------------------------------------------- #
# energy resolution
# --------------------------------------------------------------------------- #

def response_stats(x, n_boot=300, seed=0):
    """Median and sigma_68 (half the central 68% interval) of a response, with bootstrap uncertainties."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return dict(n=len(x), median=np.nan, sigma68=np.nan, resolution=np.nan,
                    median_err=np.nan, sigma68_err=np.nan, resolution_err=np.nan)
    rng = np.random.default_rng(seed)
    boot = rng.choice(x, (n_boot, len(x)))
    med_b = np.median(boot, axis=1)
    sig_b = np.array([robust_sigma(b) for b in boot])
    med, sig = float(np.median(x)), float(robust_sigma(x))
    return dict(n=len(x), median=med, sigma68=sig, resolution=sig / med, median_err=float(med_b.std()),
                sigma68_err=float(sig_b.std()), resolution_err=float((sig_b / med_b).std()))


def stat_text(s):
    return (r"med = %.3f, $\sigma_{68}$/med = %.1f $\pm$ %.1f%%, N = %d"
            % (s["median"], 100 * s["resolution"], 100 * s["resolution_err"], s["n"]))


def step_hist(ax, x, edges, color, label, ls="-", lw=2.2, fill=False):
    x = np.clip(np.asarray(x, float), edges[0], edges[-1] - 1e-9)   # overflow goes into the edge bins
    if len(x) == 0:
        return
    w = np.full(len(x), 1.0 / len(x))
    if fill:
        ax.hist(x, edges, weights=w, color=color, alpha=0.18)
    ax.hist(x, edges, weights=w, histtype="step", color=color, lw=lw, ls=ls, label=label)


def panel_keys(tables):
    return list(tables) + ["all"]


def merged(tables, sel_key, name):
    return np.concatenate([t[sel_key][name] for t in tables.values()]) if tables else np.array([])


def get(tables, key, sel_key, name):
    return merged(tables, sel_key, name) if key == "all" else tables[key][sel_key][name]


def panel_title(key, tables):
    return ("d = %s cm combined" % ", ".join("%g" % d for d in tables)) if key == "all" else "d = %g cm" % key


def plot_photon_response(tables, outdir, metrics):
    keys = panel_keys(tables)
    fig, axes = plt.subplots(1, len(keys), figsize=(8.5 * len(keys), 8), sharey=True)
    edges = np.linspace(0.5, 1.5, 41)
    for ax, key in zip(np.atleast_1d(axes), keys):
        out = {}
        for sel_key, color, name, ls, fill in (("merged", CMS_blue, "TICL: 1 trackster", "-", True),
                                               ("other", CMS_gray, "TICL: >1 tracksters", "--", False)):
            r = get(tables, key, sel_key, "e_pred") / get(tables, key, sel_key, "e_true")
            s = out[sel_key] = response_stats(r)
            if s["n"]:
                step_hist(ax, r, edges, color, "%s\n%s" % (name, stat_text(s)), ls=ls, fill=fill)
        metrics[str(key)] = out
        ax.axvline(1.0, color="k", lw=1, ls=":")
        ax.set_xlabel(r"MaskFormer $E_{pred}$ / $E_{true}$ per photon")
        ax.set_title(panel_title(key, tables), fontsize=20)
        ax.legend(fontsize=13, loc="upper left")
    np.atleast_1d(axes)[0].set_ylim(0, np.atleast_1d(axes)[0].get_ylim()[1] * 1.45)
    np.atleast_1d(axes)[0].set_ylabel("fraction of photons / bin")
    fig.suptitle("MaskFormer per-photon energy response in closeBy di-photon events", fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return savefig(fig, outdir, "maskformer_photon_response")


def plot_resolution_vs_energy(tables, outdir, metrics, edges):
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.3]))
    ctr, half = 0.5 * (edges[:-1] + edges[1:]), 0.5 * np.diff(edges)
    for sel_key, color, name, marker in (("merged", CMS_blue, "TICL: 1 trackster", "o"),
                                         ("other", CMS_gray, "TICL: >1 tracksters", "s")):
        e_true, e_pred = merged(tables, sel_key, "e_true"), merged(tables, sel_key, "e_pred")
        stats = [response_stats((e_pred / e_true)[(e_true >= lo) & (e_true < hi)])
                 for lo, hi in zip(edges[:-1], edges[1:])]
        metrics[sel_key] = [dict(e_lo=float(lo), e_hi=float(hi), **s) for lo, hi, s in zip(edges[:-1], edges[1:], stats)]
        kw = dict(xerr=half, fmt=marker, color=color, ms=8, lw=2, capsize=3, label=name)
        a0.errorbar(ctr, [s["median"] for s in stats], yerr=[s["median_err"] for s in stats], **kw)
        a1.errorbar(ctr, [100 * s["resolution"] for s in stats], yerr=[100 * s["resolution_err"] for s in stats], **kw)
    a0.axhline(1.0, color="k", lw=1, ls=":")
    a0.set_ylabel(r"median $E_{pred}/E_{true}$")
    a0.legend(fontsize=14)
    a0.set_title("MaskFormer per-photon response, d = %s cm" % ", ".join("%g" % d for d in tables), fontsize=18)
    a1.set_ylabel(r"$\sigma_{68}$ / median  [%]")
    a1.set_xlabel(r"true photon energy $E_{true}$ (TruthCB) [GeV]")
    a1.set_ylim(bottom=0)
    fig.tight_layout()
    return savefig(fig, outdir, "maskformer_photon_resolution_vs_energy")


TOTALS = [("sum_mf", CMS_blue, r"MaskFormer $\sum E_{pred}$", "-"),
          ("sum_ts_raw", CMS_red, r"TICL trackster $E_{raw}$", "-"),
          ("sum_cand", CMS_orange, r"TICL candidates $\sum E$ (regressed)", "--")]


def plot_total_response(tables, outdir, metrics):
    keys = panel_keys(tables)
    fig, axes = plt.subplots(1, len(keys), figsize=(8.5 * len(keys), 8), sharey=True)
    edges = np.linspace(0.6, 1.3, 43)
    for ax, key in zip(np.atleast_1d(axes), keys):
        out = {}
        sum_true = get(tables, key, "merged", "sum_true")
        for name, color, label, ls in TOTALS:
            r = get(tables, key, "merged", name) / sum_true
            s = out[name] = response_stats(r)
            step_hist(ax, r, edges, color, "%s\n%s" % (label, stat_text(s)), ls=ls, fill=name == "sum_mf")
        metrics[str(key)] = out
        ax.axvline(1.0, color="k", lw=1, ls=":")
        ax.set_xlabel(r"$\sum E_{reco}$ / $\sum E_{true}$ (TruthCB)")
        ax.set_title(panel_title(key, tables), fontsize=20)
        ax.legend(fontsize=13, loc="upper left")
    np.atleast_1d(axes)[0].set_ylim(0, np.atleast_1d(axes)[0].get_ylim()[1] * 1.6)
    np.atleast_1d(axes)[0].set_ylabel("fraction of events / bin")
    fig.suptitle("Total reconstructed energy in di-photon events with a single TICL trackster", fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return savefig(fig, outdir, "total_energy_response")


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT, help="eval directory with one closeBy folder per distance")
    ap.add_argument("--outdir", default="plots/closeby_photons")
    ap.add_argument("--max-distance", type=float, default=2.0, help="largest photon distance to use [cm]")
    ap.add_argument("--min-trackster-energy", type=float, default=0.0,
                    help="count only the tracksters above this raw energy [GeV] (0 = all, as in the notebook)")
    ap.add_argument("--n-display", type=int, default=6, help="event displays per distance")
    ap.add_argument("--half-width", type=float, default=0.06, help="half width of the display window in eta and phi")
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args(argv)

    files = {d: f for d, f in find_files(args.input).items() if d <= args.max_distance}
    if not files:
        sys.exit("no closeBy folder with d <= %g cm under %s" % (args.max_distance, args.input))
    tables, selection, assignment = {}, {}, {}
    disp_dir = os.path.join(args.outdir, "event_display")
    for dist, path in files.items():
        df = ak.from_parquet(path, columns=COLUMNS)
        if args.max_events:
            df = df[:args.max_events]
        sel = n_tracksters(df, args.min_trackster_energy) == 1
        selection[str(dist)] = dict(file=path, n_events=len(df), n_selected=int(sel.sum()),
                                    fraction=float(sel.mean()), displayed_events=[])
        print("[select] d = %g cm: %d / %d events with one TICL trackster (%.1f%%)"
              % (dist, sel.sum(), len(df), 100 * sel.mean()))
        tables[dist] = dict(merged=energy_table(df[sel]), other=energy_table(df[~sel]))
        assignment[str(dist)] = {name: {k: float(np.median(v)) for k, v in assignment_fractions(df[m]).items()}
                                 for name, m in (("merged", sel), ("other", ~sel)) if m.any()}
        print("[assign] d = %g cm: median LC-energy fraction by best query (1 trackster | >1): %s"
              % (dist, "  ".join("%s %.2f | %.2f" % (k, v, assignment[str(dist)].get("other", {}).get(k, np.nan))
                                 for k, v in assignment[str(dist)].get("merged", {}).items())))
        for i in np.flatnonzero(sel)[:args.n_display]:
            ev = df[i]
            view = event_view(ev, args.min_trackster_energy, args.half_width)
            number = int(ev.event_number)
            plot_etaphi(view, dist, number, disp_dir, args.half_width)
            plot_3d(view, dist, number, disp_dir)
            selection[str(dist)]["displayed_events"].append(number)
        print("[display] d = %g cm: events %s" % (dist, selection[str(dist)]["displayed_events"]))

    metrics = dict(selection=selection, maskformer_node_assignment=assignment, min_trackster_energy=args.min_trackster_energy,
                   maskformer_photon_response={}, maskformer_photon_response_vs_energy={}, total_energy_response={})
    plot_photon_response(tables, args.outdir, metrics["maskformer_photon_response"])
    plot_resolution_vs_energy(tables, args.outdir, metrics["maskformer_photon_response_vs_energy"],
                              np.array([20.0, 50.0, 80.0, 110.0, 140.0, 170.0, 200.0]))
    plot_total_response(tables, args.outdir, metrics["total_energy_response"])
    with open(os.path.join(args.outdir, "closeby_merged_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n%-10s %-28s %8s %10s %6s" % ("d [cm]", "quantity", "median", "s68/med", "N"))
    for key in metrics["total_energy_response"]:
        s = metrics["maskformer_photon_response"][key]["merged"]
        print("%-10s %-28s %8.3f %9.1f%% %6d" % (key, "MaskFormer per photon", s["median"], 100 * s["resolution"], s["n"]))
        for name, _, _, _ in TOTALS:
            s = metrics["total_energy_response"][key][name]
            print("%-10s %-28s %8.3f %9.1f%% %6d" % (key, "total: " + name, s["median"], 100 * s["resolution"], s["n"]))
    print("\noutputs under %s" % args.outdir)


if __name__ == "__main__":
    main()
