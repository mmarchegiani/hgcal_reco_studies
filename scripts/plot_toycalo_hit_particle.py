"""Plot the ``hit_particle_*`` collection of a toy-calorimeter stage-1 file.

    python scripts/plot_toycalo_hit_particle.py --input data/photons_100GeV_eta20.parquet

``hit_particle_*`` is the (cell, particle) association table that gives the
truth labels for clustering: which particle put how much energy into which
sensor.  These plots check the raw variables, the closure against
``rechit_energy`` and the structure of the table (sharing between particles,
energy fraction carried by the leading contributor).
"""

import argparse
import json
import os

import numpy as np

from toycalo_common import (
    CMS_blue, CMS_gray, CMS_orange, CMS_purple, CMS_red, N_EE_LAYERS,
    ToyCaloData, describe, exp_label, hist_stat_label, logbins, savefig,
    setup_style,
)

plt = setup_style()


# --------------------------------------------------------------------------- #
# 1. raw variables
# --------------------------------------------------------------------------- #

def plot_raw(d, outdir, metrics):
    sub = os.path.join(outdir, "raw")

    fig, ax = plt.subplots()
    ax.hist(d.a_deposit, bins=logbins(1e-8, 1e-1, 90), histtype="stepfilled",
            color=CMS_blue, alpha=0.35)
    ax.hist(d.a_deposit, bins=logbins(1e-8, 1e-1, 90), histtype="step", lw=2,
            color=CMS_blue, label=hist_stat_label(d.a_deposit, " GeV"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("hit_particle_deposit [GeV]")
    ax.set_ylabel("associations / bin")
    ax.legend(fontsize=13, loc="upper left")
    exp_label(ax)
    savefig(fig, sub, "hit_particle_deposit")
    metrics.setdefault("hit_particle", {})["deposit"] = describe(d.a_deposit)

    fig, ax = plt.subplots()
    bins = np.arange(-0.5, max(d.a_pid_local.max() + 2, 3))
    ax.hist(d.a_pid_local, bins=bins, histtype="stepfilled", color=CMS_orange, alpha=0.35)
    ax.hist(d.a_pid_local, bins=bins, histtype="step", lw=2, color=CMS_orange,
            label=hist_stat_label(d.a_pid_local))
    ax.set_yscale("log")
    ax.set_xlabel("hit_particle_id (index into particles_*)")
    ax.set_ylabel("associations / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "hit_particle_id")
    metrics["hit_particle"]["particle_id"] = describe(d.a_pid_local)

    fig, ax = plt.subplots()
    bins = np.linspace(0, d.a_sensor.max() * 1.02, 121)
    ax.hist(d.a_sensor, bins=bins, histtype="stepfilled", color=CMS_purple, alpha=0.35)
    ax.hist(d.a_sensor, bins=bins, histtype="step", lw=2, color=CMS_purple,
            label=hist_stat_label(d.a_sensor))
    ax.set_xlabel("hit_particle_sensor_idx")
    ax.set_ylabel("associations / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "hit_particle_sensor_idx")

    # the table is written particle-major, so the sensor index is NOT monotonic
    fig, ax = plt.subplots()
    ev0 = d.a_event == 0
    ax.plot(np.arange(ev0.sum()), d.a_sensor[ev0], ".", ms=4, color=CMS_purple,
            label="hit_particle_sensor_idx")
    h0 = d.h_event == 0
    ax.plot(np.arange(h0.sum()), d.h_idx[h0], "-", lw=2, color=CMS_blue,
            label="rechit_idx (sorted)")
    ax.set_xlabel("position in the collection (event 0)")
    ax.set_ylabel("global cell index")
    ax.legend(fontsize=14)
    exp_label(ax)
    savefig(fig, sub, "collection_ordering")


# --------------------------------------------------------------------------- #
# 2. closure against the rechit collection
# --------------------------------------------------------------------------- #

def plot_closure(d, outdir, metrics):
    sub = os.path.join(outdir, "closure")

    unmatched = int((d.a_hit < 0).sum())
    resid = d.h_sum_deposit - d.h_e
    rel = np.divide(resid, d.h_e, out=np.zeros_like(resid), where=d.h_e > 0)

    fig, ax = plt.subplots()
    ax.hist(rel, bins=np.linspace(-1e-12, 1e-12, 81), histtype="step", lw=2,
            color=CMS_blue,
            label=(r"$(\sum_p \mathrm{deposit} - E_{\mathrm{rechit}})/E_{\mathrm{rechit}}$"
                   "\n" "max |abs. resid.| = %.3g GeV\nunmatched associations: %d"
                   % (np.abs(resid).max(), unmatched)))
    ax.set_xlabel("relative closure residual")
    ax.set_ylabel("rechits / bin")
    ax.legend(fontsize=12, loc="upper right")
    exp_label(ax)
    savefig(fig, sub, "per_cell_closure")
    metrics.setdefault("closure", {}).update(
        max_abs_residual_gev=float(np.abs(resid).max()),
        max_rel_residual=float(np.abs(rel).max()),
        unmatched_associations=unmatched,
        n_assoc=int(len(d.a_deposit)), n_rechits=int(len(d.h_e)))

    fig, ax = plt.subplots()
    ax.scatter(d.ev_sum_e, d.ev_sum_deposit, s=70, color=CMS_blue, alpha=0.85,
               edgecolor="k", lw=0.5)
    lim = [0.95 * d.ev_sum_e.min(), 1.05 * d.ev_sum_e.max()]
    ax.plot(lim, lim, "--", color=CMS_red, lw=2, label="y = x")
    ax.set_xlabel(r"$\sum$ rechit_energy per event [GeV]")
    ax.set_ylabel(r"$\sum$ hit_particle_deposit per event [GeV]")
    ax.legend(fontsize=14)
    exp_label(ax)
    savefig(fig, sub, "event_sum_closure")

    fig, ax = plt.subplots()
    ax.scatter(d.h_count, d.a_count, s=70, color=CMS_orange, alpha=0.85,
               edgecolor="k", lw=0.5)
    lim = [0.95 * d.h_count.min(), 1.05 * d.h_count.max()]
    ax.plot(lim, lim, "--", color=CMS_red, lw=2, label="y = x")
    ax.set_xlabel("rechits per event")
    ax.set_ylabel("hit_particle associations per event")
    ax.legend(fontsize=14)
    exp_label(ax)
    savefig(fig, sub, "multiplicity_closure")
    metrics["closure"]["assoc_per_rechit_mean"] = float(d.a_count.sum() / d.h_count.sum())


# --------------------------------------------------------------------------- #
# 3. structure of the association table
# --------------------------------------------------------------------------- #

def plot_structure(d, outdir, metrics):
    sub = os.path.join(outdir, "structure")

    # how many particles share a cell: this is what makes clustering ambiguous
    fig, ax = plt.subplots()
    nmax = max(d.h_n_contrib.max(), 2)
    bins = np.arange(-0.5, nmax + 1.5)
    ax.hist(d.h_n_contrib, bins=bins, histtype="stepfilled", color=CMS_blue, alpha=0.35)
    ax.hist(d.h_n_contrib, bins=bins, histtype="step", lw=2, color=CMS_blue,
            label=hist_stat_label(d.h_n_contrib))
    ax.set_yscale("log")
    ax.set_xlabel("particles contributing to a cell")
    ax.set_ylabel("rechits / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "contributors_per_cell")
    metrics.setdefault("structure", {})["contributors_per_cell"] = describe(d.h_n_contrib)
    metrics["structure"]["shared_cell_fraction"] = float((d.h_n_contrib > 1).mean())

    # leading-contributor purity: 1.0 everywhere means labels are unambiguous
    lead = np.zeros(len(d.h_e))
    np.maximum.at(lead, d.a_hit[d.a_hit >= 0], d.a_deposit[d.a_hit >= 0])
    purity = np.divide(lead, d.h_e, out=np.full(len(d.h_e), np.nan), where=d.h_e > 0)
    fig, ax = plt.subplots()
    ax.hist(purity, bins=np.linspace(0, 1.0001, 51), histtype="stepfilled",
            color=CMS_red, alpha=0.35)
    ax.hist(purity, bins=np.linspace(0, 1.0001, 51), histtype="step", lw=2,
            color=CMS_red, label=hist_stat_label(purity))
    ax.set_yscale("log")
    ax.set_xlabel("leading-particle energy fraction in a cell")
    ax.set_ylabel("rechits / bin")
    ax.legend(fontsize=13, loc="upper left")
    exp_label(ax)
    savefig(fig, sub, "leading_contributor_purity")
    metrics["structure"]["leading_fraction"] = describe(purity)

    # per-particle view of the same table
    fig, ax = plt.subplots()
    ax.hist(d.p_n_cells, bins=25, histtype="stepfilled", color=CMS_orange, alpha=0.35)
    ax.hist(d.p_n_cells, bins=25, histtype="step", lw=2, color=CMS_orange,
            label=hist_stat_label(d.p_n_cells))
    ax.set_xlabel("cells hit per particle")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "cells_per_particle")
    metrics["structure"]["cells_per_particle"] = describe(d.p_n_cells)

    fig, ax = plt.subplots()
    ax.scatter(d.p_n_cells, d.p_sum_deposit, s=70, color=CMS_blue, alpha=0.85,
               edgecolor="k", lw=0.5)
    ax.set_xlabel("cells hit per particle")
    ax.set_ylabel(r"$\sum$ hit_particle_deposit per particle [GeV]")
    exp_label(ax)
    savefig(fig, sub, "cells_vs_deposit_per_particle")

    # does the table reproduce particles_total_energy_deposited_active?
    ref = d.p_edep_active
    rel = np.divide(d.p_sum_deposit - ref, ref, out=np.full(len(ref), np.nan), where=ref > 0)
    fig, ax = plt.subplots()
    ax.hist(rel, bins=np.linspace(-1e-12, 1e-12, 61), histtype="step", lw=2,
            color=CMS_purple,
            label=(r"$(\sum_{\mathrm{cells}} \mathrm{deposit} - E^{\mathrm{active}}_{\mathrm{dep}})"
                   r"/E^{\mathrm{active}}_{\mathrm{dep}}$" "\n" "max |rel.| = %.3g"
                   % np.nanmax(np.abs(rel))))
    ax.set_xlabel("relative residual")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=12)
    exp_label(ax)
    savefig(fig, sub, "particle_deposit_closure")
    metrics["structure"]["particle_deposit_max_rel_residual"] = float(np.nanmax(np.abs(rel)))


# --------------------------------------------------------------------------- #
# 4. where in the detector the associations sit
# --------------------------------------------------------------------------- #

def plot_geometry(d, outdir, metrics):
    sub = os.path.join(outdir, "geometry")
    import matplotlib.colors as mcolors

    ok = d.a_hit >= 0
    lay = d.a_layer[ok]
    dep = d.a_deposit[ok]

    fig, ax = plt.subplots()
    bins = np.arange(-0.5, d.n_layers + 0.5)
    ax.hist(lay, bins=bins, weights=dep / d.n_events, histtype="stepfilled",
            color=CMS_blue, alpha=0.35)
    ax.hist(lay, bins=bins, weights=dep / d.n_events, histtype="step", lw=2,
            color=CMS_blue, label="deposit")
    ax.axvline(N_EE_LAYERS - 0.5, color="k", ls="--", lw=1.5)
    ax.set_xlabel("layer index of the associated cell")
    ax.set_ylabel(r"$\langle \sum$ deposit $\rangle$ per event [GeV]")
    ax.legend(fontsize=14)
    exp_label(ax)
    savefig(fig, sub, "deposit_per_layer")

    fig, ax = plt.subplots(figsize=(11, 7))
    h = ax.hist2d(lay, np.log10(np.clip(dep, 1e-9, None)),
                  bins=[bins, np.linspace(-8, -1, 90)], cmin=1, cmap="viridis",
                  norm=mcolors.LogNorm())
    fig.colorbar(h[3], ax=ax, label="associations")
    ax.set_xlabel("layer index")
    ax.set_ylabel(r"$\log_{10}(\mathrm{deposit}\,/\,\mathrm{GeV})$")
    exp_label(ax)
    savefig(fig, sub, "deposit_vs_layer")

    # deposit vs distance to the incident axis, straight from the truth table
    dr = d.h_dr[d.a_hit[ok]]
    m = dr > 0
    fig, ax = plt.subplots(figsize=(11, 7))
    h = ax.hist2d(np.log10(dr[m]), np.log10(np.clip(dep[m], 1e-9, None)),
                  bins=[np.linspace(-1, 3, 100), np.linspace(-8, -1, 90)],
                  cmin=1, cmap="viridis", norm=mcolors.LogNorm())
    fig.colorbar(h[3], ax=ax, label="associations")
    ax.set_xlabel(r"$\log_{10}(\Delta r$ to shower axis $/\,\mathrm{mm})$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{deposit}\,/\,\mathrm{GeV})$")
    exp_label(ax)
    savefig(fig, sub, "deposit_vs_dr")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/photons_100GeV_eta20.parquet")
    ap.add_argument("--outdir", default="plots/toycalo/hit_particle")
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()

    print("[hit_particle] loading %s" % args.input)
    d = ToyCaloData(args.input, max_events=args.max_events)
    metrics = dict(dataset=d.summary())

    print("[hit_particle] raw ...");        plot_raw(d, args.outdir, metrics)
    print("[hit_particle] closure ...");    plot_closure(d, args.outdir, metrics)
    print("[hit_particle] structure ...");  plot_structure(d, args.outdir, metrics)
    print("[hit_particle] geometry ...");   plot_geometry(d, args.outdir, metrics)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("[hit_particle] done -> %s" % args.outdir)


if __name__ == "__main__":
    main()
