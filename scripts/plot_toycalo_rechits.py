"""Plot the ``rechit_*`` collection of a toy-calorimeter stage-1 file.

    python scripts/plot_toycalo_rechits.py --input data/photons_100GeV_eta20.parquet

Covers the raw cell variables (energy, x, y, z, cell index, pid flag), the
derived shower observables (layer, eta, phi, radius, distance to the incident
axis) and a couple of event displays.
"""

import argparse
import json
import os

import numpy as np

from toycalo_common import (
    CMAP_10, CMS_blue, CMS_gray, CMS_orange, CMS_red, N_EE_LAYERS, REGIONS,
    ToyCaloData, describe, exp_label, hist_stat_label, note, logbins, savefig,
    setup_style,
)

plt = setup_style()


# --------------------------------------------------------------------------- #
# 1. raw single-variable spectra
# --------------------------------------------------------------------------- #

def plot_raw_spectra(d, outdir, metrics):
    sub = os.path.join(outdir, "raw")
    specs = [
        ("rechit_energy", d.h_e, " GeV", np.linspace(0, 0.02, 81), "cell energy"),
        ("rechit_x", d.h_x, " mm", np.linspace(-1500, 1500, 121), "cell x"),
        ("rechit_y", d.h_y, " mm", np.linspace(-1500, 1500, 121), "cell y"),
        ("rechit_z", d.h_z, " mm", np.linspace(3190, 3890, 141), "cell z"),
        ("rechit_idx", d.h_idx.astype(float), "", np.linspace(0, d.h_idx.max() * 1.02, 121),
         "global cell index"),
        ("rechit_stupid_pid", d.h_pid.astype(float), "",
         np.arange(d.h_pid.min() - 0.5, d.h_pid.max() + 1.5), "cell pid flag"),
    ]
    for key, val, unit, bins, label in specs:
        fig, ax = plt.subplots()
        ax.hist(val, bins=bins, histtype="stepfilled", color=CMS_blue, alpha=0.35)
        ax.hist(val, bins=bins, histtype="step", lw=2, color=CMS_blue,
                label=hist_stat_label(val, unit))
        ax.set_xlabel("%s [%s]" % (label, unit.strip()) if unit.strip() else label)
        ax.set_ylabel("rechits / bin")
        ax.legend(fontsize=13, loc="upper right")
        exp_label(ax)
        savefig(fig, sub, key)
        metrics.setdefault("rechit", {})[key] = describe(val)

    # energy needs a log-log view: the spectrum spans 7 decades
    fig, ax = plt.subplots()
    bins = logbins(1e-8, 1e-1, 90)
    ax.hist(d.h_e, bins=bins, histtype="step", lw=2, color=CMS_blue,
            label="all cells")
    for reg, lab, col in REGIONS:
        m = d.h_is_ee if reg == "EE" else ~d.h_is_ee
        ax.hist(d.h_e[m], bins=bins, histtype="step", lw=2, ls="--", color=col,
                label="%s  (%.0f%% of cells)" % (lab, 100 * m.mean()))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rechit energy [GeV]")
    ax.set_ylabel("rechits / bin")
    ax.legend(fontsize=13, loc="upper left")
    exp_label(ax)
    savefig(fig, sub, "rechit_energy_loglog")

    # cumulative energy fraction vs a cell-energy threshold: where to cut noise
    fig, ax = plt.subplots()
    order = np.argsort(d.h_e)
    cum_e = np.cumsum(d.h_e[order]) / d.h_e.sum()
    cum_n = np.arange(1, len(order) + 1) / len(order)
    ax.plot(d.h_e[order], 1 - cum_e, lw=2, color=CMS_blue,
            label="energy kept above threshold")
    ax.plot(d.h_e[order], 1 - cum_n, lw=2, color=CMS_red,
            label="cells kept above threshold")
    ax.set_xscale("log")
    ax.set_xlabel("rechit energy threshold [GeV]")
    ax.set_ylabel("fraction kept")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=14, loc="lower left")
    exp_label(ax)
    savefig(fig, sub, "rechit_energy_threshold_scan")

    for thr in (1e-6, 1e-5, 1e-4, 1e-3):
        m = d.h_e > thr
        metrics.setdefault("rechit_thresholds", {})["E>%g" % thr] = dict(
            cell_fraction=float(m.mean()),
            energy_fraction=float(d.h_e[m].sum() / d.h_e.sum()))


# --------------------------------------------------------------------------- #
# 2. per-event multiplicity and energy sums
# --------------------------------------------------------------------------- #

def plot_event_level(d, outdir, metrics):
    sub = os.path.join(outdir, "event")

    fig, ax = plt.subplots()
    ax.hist(d.h_count, bins=np.linspace(200, 1000, 21), histtype="stepfilled",
            color=CMS_blue, alpha=0.35)
    ax.hist(d.h_count, bins=np.linspace(200, 1000, 21), histtype="step", lw=2,
            color=CMS_blue, label=hist_stat_label(d.h_count))
    ax.set_xlabel("rechits per event")
    ax.set_ylabel("events / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "n_rechits_per_event")

    fig, ax = plt.subplots()
    ax.hist(d.ev_sum_e, bins=20, histtype="stepfilled", color=CMS_orange, alpha=0.35)
    ax.hist(d.ev_sum_e, bins=20, histtype="step", lw=2, color=CMS_orange,
            label=hist_stat_label(d.ev_sum_e, " GeV"))
    ax.set_xlabel(r"$\sum$ rechit energy per event [GeV]")
    ax.set_ylabel("events / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "sum_rechit_energy_per_event")

    # the observable that actually matters: raw deposit / generated energy
    fig, ax = plt.subplots()
    f = 100 * d.ev_sampling
    ax.hist(f, bins=20, histtype="stepfilled", color=CMS_red, alpha=0.35)
    ax.hist(f, bins=20, histtype="step", lw=2, color=CMS_red,
            label=hist_stat_label(f, " %"))
    ax.set_xlabel(r"$\sum E_{\mathrm{rechit}}\,/\,E_{\mathrm{gun}}$ [%]")
    ax.set_ylabel("events / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "sampling_fraction")
    metrics.setdefault("event", {})["sampling_fraction_percent"] = describe(f)
    metrics["event"]["n_rechits_per_event"] = describe(d.h_count)
    metrics["event"]["sum_rechit_energy"] = describe(d.ev_sum_e)

    # multiplicity vs energy: at fixed gun energy the correlation is weak but real
    fig, ax = plt.subplots()
    ax.scatter(d.h_count, d.ev_sum_e, s=60, color=CMS_blue, alpha=0.8,
               edgecolor="k", lw=0.5)
    c = np.corrcoef(d.h_count, d.ev_sum_e)[0, 1]
    ax.set_xlabel("rechits per event")
    ax.set_ylabel(r"$\sum$ rechit energy [GeV]")
    note(ax, r"$\rho$ = %.3f" % c)
    exp_label(ax)
    savefig(fig, sub, "n_rechits_vs_sum_energy")
    metrics["event"]["corr_nhits_sumE"] = float(c)


# --------------------------------------------------------------------------- #
# 3. longitudinal structure
# --------------------------------------------------------------------------- #

def plot_longitudinal(d, outdir, metrics):
    sub = os.path.join(outdir, "longitudinal")
    lay = np.arange(d.n_layers)
    e_per_layer = np.bincount(d.h_layer, weights=d.h_e, minlength=d.n_layers) / d.n_events
    n_per_layer = np.bincount(d.h_layer, minlength=d.n_layers) / d.n_events

    for vals, ylab, name, col in [
        (e_per_layer, r"$\langle \sum E \rangle$ per event [GeV]", "energy_per_layer", CMS_blue),
        (n_per_layer, r"$\langle N_{\mathrm{rechit}} \rangle$ per event", "nhits_per_layer", CMS_orange),
    ]:
        fig, ax = plt.subplots()
        ax.step(lay, vals, where="mid", lw=2, color=col)
        ax.fill_between(lay, 0, vals, step="mid", color=col, alpha=0.25)
        ax.axvline(N_EE_LAYERS - 0.5, color="k", ls="--", lw=1.5)
        ax.text(N_EE_LAYERS - 1, ax.get_ylim()[1] * 0.9, "CE-E like ", ha="right", fontsize=14)
        ax.text(N_EE_LAYERS, ax.get_ylim()[1] * 0.9, " CE-H like", ha="left", fontsize=14)
        ax.set_xlabel("layer index")
        ax.set_ylabel(ylab)
        exp_label(ax)
        savefig(fig, sub, name)

    # shower maximum and 90% containment depth, per event
    ev_layer_e = np.zeros((d.n_events, d.n_layers))
    np.add.at(ev_layer_e, (d.h_event, d.h_layer), d.h_e)
    shower_max = ev_layer_e.argmax(axis=1)
    cum = np.cumsum(ev_layer_e, axis=1) / ev_layer_e.sum(axis=1, keepdims=True)
    depth90 = (cum < 0.9).sum(axis=1)

    fig, ax = plt.subplots()
    for v, lab, col in [(shower_max, "shower max layer", CMS_blue),
                        (depth90, "90% containment layer", CMS_red)]:
        ax.hist(v, bins=np.arange(-0.5, d.n_layers + 0.5), histtype="step", lw=2,
                color=col, label="%s\n%s" % (lab, hist_stat_label(v)))
    ax.set_xlabel("layer index")
    ax.set_ylabel("events / bin")
    ax.set_xlim(0, N_EE_LAYERS + 6)
    ax.legend(fontsize=12)
    exp_label(ax)
    savefig(fig, sub, "shower_depth")
    metrics.setdefault("longitudinal", {})["shower_max_layer"] = describe(shower_max)
    metrics["longitudinal"]["layer90_containment"] = describe(depth90)
    metrics["longitudinal"]["energy_fraction_EE"] = float(
        e_per_layer[:N_EE_LAYERS].sum() / e_per_layer.sum())

    # the full per-event profile, so the event-to-event spread is visible
    fig, ax = plt.subplots()
    for i in range(d.n_events):
        ax.plot(lay, ev_layer_e[i] / ev_layer_e[i].sum(), lw=1, color=CMS_gray, alpha=0.5)
    mean = (ev_layer_e / ev_layer_e.sum(axis=1, keepdims=True)).mean(axis=0)
    ax.plot(lay, mean, lw=3, color=CMS_red, label="mean normalised profile")
    ax.axvline(N_EE_LAYERS - 0.5, color="k", ls="--", lw=1.5)
    ax.set_xlabel("layer index")
    ax.set_ylabel("energy fraction / layer")
    ax.legend(fontsize=14)
    exp_label(ax)
    savefig(fig, sub, "longitudinal_profiles_per_event")

    # z is not equally spaced -> show the actual plane positions
    fig, ax = plt.subplots()
    ax.plot(lay, d.layer_z, "o-", color=CMS_blue, ms=6)
    ax.set_xlabel("layer index")
    ax.set_ylabel("active plane z [mm]")
    ax2 = ax.twinx()
    ax2.step(lay[1:], np.diff(d.layer_z), where="mid", color=CMS_red, lw=2)
    ax2.set_ylabel(r"$\Delta z$ to previous layer [mm]", color=CMS_red)
    ax2.grid(False)
    exp_label(ax)
    savefig(fig, sub, "layer_z_positions")
    metrics["longitudinal"]["layer_z_mm"] = [float(v) for v in d.layer_z]


# --------------------------------------------------------------------------- #
# 4. transverse structure
# --------------------------------------------------------------------------- #

def plot_transverse(d, outdir, metrics):
    sub = os.path.join(outdir, "transverse")

    for key, val, unit, bins, label in [
        ("rechit_r", d.h_r, " mm", np.linspace(500, 1500, 101), r"cell $r=\sqrt{x^2+y^2}$"),
        ("rechit_eta", d.h_eta, "", np.linspace(1.5, 2.6, 111), r"cell $\eta$"),
        ("rechit_phi", d.h_phi, " rad", np.linspace(0, 2 * np.pi, 73), r"cell $\phi$"),
    ]:
        fig, ax = plt.subplots()
        ax.hist(val, bins=bins, histtype="step", lw=2, color=CMS_blue,
                label="unweighted\n" + hist_stat_label(val, unit))
        w = d.h_e / d.h_e.sum() * len(val)
        ax.hist(val, bins=bins, weights=w, histtype="step", lw=2, color=CMS_red,
                label="energy weighted")
        ax.set_xlabel("%s [%s]" % (label, unit.strip()) if unit.strip() else label)
        ax.set_ylabel("rechits / bin (norm.)")
        ax.legend(fontsize=12)
        exp_label(ax)
        savefig(fig, sub, key)
        metrics.setdefault("rechit", {})[key] = describe(val)

    # distance to the incident particle axis = the lateral shower profile
    bins = logbins(0.1, 1000, 80)
    fig, ax = plt.subplots()
    ax.hist(d.h_dr, bins=bins, weights=d.h_e, histtype="step", lw=2, color=CMS_red,
            label="energy weighted")
    ax.hist(d.h_dr, bins=bins, weights=np.full(len(d.h_dr), d.h_e.sum() / len(d.h_dr)),
            histtype="step", lw=2, color=CMS_blue, label="cell counting (scaled)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"distance to shower axis $\Delta r$ [mm]")
    ax.set_ylabel("energy / bin [GeV]")
    ax.legend(fontsize=14)
    exp_label(ax)
    savefig(fig, sub, "lateral_profile")

    # containment: the Moliere-radius equivalent of the toy
    order = np.argsort(d.h_dr)
    cum = np.cumsum(d.h_e[order]) / d.h_e.sum()
    fig, ax = plt.subplots()
    ax.plot(d.h_dr[order], cum, lw=2.5, color=CMS_blue)
    for frac, col in [(0.68, CMS_orange), (0.90, CMS_red), (0.95, CMS_gray)]:
        rr = d.h_dr[order][np.searchsorted(cum, frac)]
        ax.axhline(frac, color=col, ls=":", lw=1.5)
        ax.axvline(rr, color=col, ls=":", lw=1.5)
        ax.plot([rr], [frac], "o", color=col, ms=9,
                label="%.0f%% containment: %.1f mm" % (100 * frac, rr))
        metrics.setdefault("transverse", {})["r%d_mm" % int(100 * frac)] = float(rr)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta r$ to shower axis [mm]")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=14, loc="upper left")
    exp_label(ax)
    savefig(fig, sub, "lateral_containment")

    # how the shower widens with depth
    edges = np.arange(-0.5, d.n_layers + 0.5)
    ctr = np.arange(d.n_layers)
    rms = np.full(d.n_layers, np.nan)
    r68 = np.full(d.n_layers, np.nan)
    for L in range(d.n_layers):
        m = d.h_layer == L
        if m.sum() < 5:
            continue
        w = d.h_e[m]
        rms[L] = np.sqrt(np.average(d.h_dr[m] ** 2, weights=w))
        o = np.argsort(d.h_dr[m])
        c = np.cumsum(w[o]) / w.sum()
        r68[L] = d.h_dr[m][o][np.searchsorted(c, 0.68)]
    fig, ax = plt.subplots()
    ax.plot(ctr, rms, "o-", color=CMS_blue, label=r"energy-weighted RMS of $\Delta r$")
    ax.plot(ctr, r68, "s-", color=CMS_red, label="68% containment radius")
    ax.axvline(N_EE_LAYERS - 0.5, color="k", ls="--", lw=1.5)
    ax.set_xlabel("layer index")
    ax.set_ylabel(r"shower width [mm]")
    ax.legend(fontsize=14)
    exp_label(ax)
    savefig(fig, sub, "shower_width_vs_layer")

    # 2D occupancy in the plane transverse to the beam
    fig, ax = plt.subplots(figsize=(9, 8))
    h = ax.hist2d(d.h_x, d.h_y, bins=[np.linspace(-1500, 1500, 150)] * 2,
                  weights=d.h_e, cmin=1e-9, cmap="viridis",
                  norm=__import__("matplotlib").colors.LogNorm())
    fig.colorbar(h[3], ax=ax, label="energy [GeV]")
    ax.set_xlabel("rechit x [mm]")
    ax.set_ylabel("rechit y [mm]")
    ax.set_aspect("equal")
    exp_label(ax)
    savefig(fig, sub, "xy_energy_map")

    # r-z: shows the two sampling sections and the eta=2 cone
    fig, ax = plt.subplots(figsize=(11, 7))
    h = ax.hist2d(d.h_z, d.h_r, bins=[np.linspace(3190, 3890, 180), np.linspace(500, 1500, 150)],
                  weights=d.h_e, cmin=1e-9, cmap="viridis",
                  norm=__import__("matplotlib").colors.LogNorm())
    fig.colorbar(h[3], ax=ax, label="energy [GeV]")
    ax.set_xlabel("rechit z [mm]")
    ax.set_ylabel(r"rechit $r$ [mm]")
    exp_label(ax)
    savefig(fig, sub, "rz_energy_map")


# --------------------------------------------------------------------------- #
# 5. cell-energy correlations
# --------------------------------------------------------------------------- #

def plot_correlations(d, outdir, metrics):
    sub = os.path.join(outdir, "correlations")
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(11, 7))
    h = ax.hist2d(d.h_layer, np.log10(np.clip(d.h_e, 1e-9, None)),
                  bins=[np.arange(-0.5, d.n_layers + 0.5), np.linspace(-8, -1, 90)],
                  cmin=1, cmap="viridis", norm=mcolors.LogNorm())
    fig.colorbar(h[3], ax=ax, label="rechits")
    ax.set_xlabel("layer index")
    ax.set_ylabel(r"$\log_{10}(E_{\mathrm{rechit}}\,/\,\mathrm{GeV})$")
    exp_label(ax)
    savefig(fig, sub, "energy_vs_layer")

    fig, ax = plt.subplots(figsize=(11, 7))
    m = d.h_dr > 0
    h = ax.hist2d(np.log10(d.h_dr[m]), np.log10(np.clip(d.h_e[m], 1e-9, None)),
                  bins=[np.linspace(-1, 3, 100), np.linspace(-8, -1, 90)],
                  cmin=1, cmap="viridis", norm=mcolors.LogNorm())
    fig.colorbar(h[3], ax=ax, label="rechits")
    ax.set_xlabel(r"$\log_{10}(\Delta r\,/\,\mathrm{mm})$")
    ax.set_ylabel(r"$\log_{10}(E_{\mathrm{rechit}}\,/\,\mathrm{GeV})$")
    exp_label(ax)
    savefig(fig, sub, "energy_vs_dr")

    # the cell index is a proxy for the geometry ordering; check it tracks z
    fig, ax = plt.subplots()
    ax.scatter(d.h_idx, d.h_z, s=1, color=CMS_blue, alpha=0.2)
    ax.set_xlabel("rechit_idx (global cell index)")
    ax.set_ylabel("rechit z [mm]")
    exp_label(ax)
    savefig(fig, sub, "cellidx_vs_z")

    # fraction of the event energy in the N hardest cells
    fracs = []
    for ev in range(d.n_events):
        s, e = d.h_off[ev], d.h_off[ev + 1]
        v = np.sort(d.h_e[s:e])[::-1]
        c = np.cumsum(v) / v.sum()
        fracs.append(np.interp(np.arange(1, 201), np.arange(1, len(c) + 1), c))
    fracs = np.array(fracs)
    fig, ax = plt.subplots()
    n = np.arange(1, 201)
    ax.plot(n, fracs.mean(axis=0), lw=2.5, color=CMS_blue, label="mean")
    ax.fill_between(n, np.percentile(fracs, 16, axis=0), np.percentile(fracs, 84, axis=0),
                    color=CMS_blue, alpha=0.25, label="68% band")
    ax.set_xscale("log")
    ax.set_xlabel("N hardest cells")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=14, loc="lower right")
    exp_label(ax)
    savefig(fig, sub, "energy_fraction_vs_ncells")
    metrics.setdefault("rechit", {})["energy_frac_in_top10_cells"] = float(fracs[:, 9].mean())
    metrics["rechit"]["energy_frac_in_top100_cells"] = float(fracs[:, 99].mean())


# --------------------------------------------------------------------------- #
# 6. event displays
# --------------------------------------------------------------------------- #

def plot_event_displays(d, outdir, n_display=3):
    """x-y and z-dr views of single showers, both zoomed on the incident axis."""
    import matplotlib.colors as mcolors
    sub = os.path.join(outdir, "displays")
    for ev in range(min(n_display, d.n_events)):
        s, e = d.h_off[ev], d.h_off[ev + 1]
        x, y, z, en = d.h_x[s:e], d.h_y[s:e], d.h_z[s:e], d.h_e[s:e]
        ax_x, ax_y = d.h_axis_x[s:e], d.h_axis_y[s:e]
        size = 4 + 500 * (en / en.max()) ** 0.5
        norm = mcolors.LogNorm(vmin=max(en.min(), 1e-7), vmax=en.max())

        fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)

        ax = axes[0]
        ax.scatter(x, y, s=size, c=en, cmap="plasma", norm=norm, alpha=0.85)
        ax.plot(d.ev_axis[0, ev], d.ev_axis[1, ev], "x", color="k", ms=16, mew=3)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_xlim(d.ev_axis[0, ev] - 220, d.ev_axis[0, ev] + 220)
        ax.set_ylim(d.ev_axis[1, ev] - 220, d.ev_axis[1, ev] + 220)
        ax.set_aspect("equal")
        ax.set_title("transverse view (x marks the gun impact point)", fontsize=16)

        ax = axes[1]
        dr = np.hypot(x - ax_x, y - ax_y)
        signed = np.where(x - ax_x >= 0, dr, -dr)
        sc = ax.scatter(z, signed, s=size, c=en, cmap="plasma", norm=norm, alpha=0.85)
        ax.axhline(0, color="k", ls="--", lw=1.5)
        ax.axvline(0.5 * (d.layer_z[N_EE_LAYERS - 1] + d.layer_z[N_EE_LAYERS]),
                   color=CMS_gray, ls=":", lw=2)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel(r"signed $\Delta r$ to shower axis [mm]")
        ax.set_ylim(-150, 150)
        ax.set_title("longitudinal view", fontsize=16)
        fig.colorbar(sc, ax=axes, label="cell energy [GeV]", pad=0.01)

        fig.suptitle(r"event %d    %d rechits    $\sum E_{\mathrm{rechit}}$ = %.3f GeV"
                     r"    $E_{\mathrm{gun}}$ = %.1f GeV"
                     % (d.event_id[ev], len(en), en.sum(), d.g_energy[ev]), fontsize=19)
        savefig(fig, sub, "display_event%02d" % ev)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/photons_100GeV_eta20.parquet")
    ap.add_argument("--outdir", default="plots/toycalo/rechits")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--n-display", type=int, default=3)
    args = ap.parse_args()

    print("[rechits] loading %s" % args.input)
    d = ToyCaloData(args.input, max_events=args.max_events)
    print("[rechits] %s" % d.summary())
    metrics = dict(dataset=d.summary())

    print("[rechits] raw spectra ...");     plot_raw_spectra(d, args.outdir, metrics)
    print("[rechits] event level ...");     plot_event_level(d, args.outdir, metrics)
    print("[rechits] longitudinal ...");    plot_longitudinal(d, args.outdir, metrics)
    print("[rechits] transverse ...");      plot_transverse(d, args.outdir, metrics)
    print("[rechits] correlations ...");    plot_correlations(d, args.outdir, metrics)
    print("[rechits] displays ...");        plot_event_displays(d, args.outdir, args.n_display)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("[rechits] done -> %s" % args.outdir)


if __name__ == "__main__":
    main()
