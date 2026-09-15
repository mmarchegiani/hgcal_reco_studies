"""Plot the ``particles_*`` collection of a toy-calorimeter stage-1 file.

    python scripts/plot_toycalo_particles.py --input data/photons_100GeV_eta20.parquet

One entry per simulated particle: production vertex, momentum direction,
the first impact on an active plane, and the energy it deposited in the active
silicon and in the whole detector.  In the single-particle samples only the
primary survives, so most of these are 1-entry-per-event distributions.
"""

import argparse
import json
import os

import numpy as np

from toycalo_common import (
    CMS_blue, CMS_gray, CMS_orange, CMS_purple, CMS_red, N_EE_LAYERS,
    ToyCaloData, describe, exp_label, hist_stat_label, note, savefig, setup_style,
)

plt = setup_style()

PDG_NAMES = {22: r"$\gamma$", 11: r"$e^-$", -11: r"$e^+$", 13: r"$\mu^-$",
             -13: r"$\mu^+$", 211: r"$\pi^+$", -211: r"$\pi^-$", 111: r"$\pi^0$",
             2112: "n", 2212: "p", 130: r"$K^0_L$"}


def _hist(ax, v, bins, color, unit="", label=""):
    ax.hist(v, bins=bins, histtype="stepfilled", color=color, alpha=0.35)
    ax.hist(v, bins=bins, histtype="step", lw=2, color=color,
            label=(label + "\n" if label else "") + hist_stat_label(v, unit))


# --------------------------------------------------------------------------- #
# 1. identity and multiplicity
# --------------------------------------------------------------------------- #

def plot_identity(d, outdir, metrics):
    sub = os.path.join(outdir, "identity")

    pdg, cnt = np.unique(d.p_pdgid, return_counts=True)
    fig, ax = plt.subplots()
    pos = np.arange(len(pdg))
    ax.bar(pos, cnt, width=0.5, color=CMS_blue, edgecolor="k")
    ax.set_xlim(-0.8, len(pos) - 0.2)
    for p, c in zip(pos, cnt):
        ax.text(p, c, " %d" % c, ha="center", va="bottom", fontsize=14)
    ax.set_xticks(pos)
    ax.set_xticklabels(["%s\n(%d)" % (PDG_NAMES.get(int(v), ""), v) for v in pdg])
    ax.set_xlabel("particles_pdgid")
    ax.set_ylabel("particles")
    ax.set_ylim(0, cnt.max() * 1.18)
    exp_label(ax)
    savefig(fig, sub, "particles_pdgid")
    metrics.setdefault("particles", {})["pdgid_counts"] = {
        str(int(p)): int(c) for p, c in zip(pdg, cnt)}

    fig, ax = plt.subplots()
    _hist(ax, d.p_count, np.arange(-0.5, d.p_count.max() + 1.5), CMS_orange)
    ax.set_xlabel("particles per event")
    ax.set_ylabel("events / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "n_particles_per_event")
    metrics["particles"]["n_per_event"] = describe(d.p_count)

    fig, ax = plt.subplots()
    _hist(ax, d.p_parent, np.arange(d.p_parent.min() - 0.5, d.p_parent.max() + 1.5),
          CMS_purple)
    ax.set_xlabel("particles_parent_idx  (-1 = primary)")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "particles_parent_idx")
    metrics["particles"]["primary_fraction"] = float(d.p_is_primary.mean())


# --------------------------------------------------------------------------- #
# 2. kinematics
# --------------------------------------------------------------------------- #

def plot_kinematics(d, outdir, metrics):
    sub = os.path.join(outdir, "kinematics")

    fig, ax = plt.subplots()
    _hist(ax, d.p_ekin, 20, CMS_blue, " GeV")
    ax.set_xlabel("particles_kinetic_energy [GeV]")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "kinetic_energy")
    metrics.setdefault("particles", {})["kinetic_energy"] = describe(d.p_ekin)

    for key, val, unit, label, col in [
        ("momentum_direction_x", d.p_dx, "", r"$\hat{p}_x$", CMS_blue),
        ("momentum_direction_y", d.p_dy, "", r"$\hat{p}_y$", CMS_orange),
        ("momentum_direction_z", d.p_dz, "", r"$\hat{p}_z$", CMS_red),
    ]:
        fig, ax = plt.subplots()
        _hist(ax, val, 20, col, unit)
        ax.set_xlabel("particles_%s" % key)
        ax.set_ylabel("particles / bin")
        ax.legend(fontsize=13)
        exp_label(ax)
        savefig(fig, sub, key)
        metrics["particles"][key] = describe(val)

    norm = np.sqrt(d.p_dx ** 2 + d.p_dy ** 2 + d.p_dz ** 2)
    fig, ax = plt.subplots()
    _hist(ax, norm, np.linspace(0.999, 1.001, 41), CMS_gray)
    ax.set_xlabel(r"$|\hat{p}|$ of particles_momentum_direction")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "momentum_direction_norm")
    metrics["particles"]["momentum_direction_norm"] = describe(norm)

    for key, val, unit, label, col, bins in [
        ("eta", d.p_eta, "", r"$\eta$ from the momentum direction", CMS_blue,
         np.linspace(1.95, 2.05, 41)),
        ("phi", d.p_phi, " rad", r"$\phi$ from the momentum direction", CMS_orange,
         np.linspace(0, 2 * np.pi, 25)),
    ]:
        fig, ax = plt.subplots()
        _hist(ax, val, bins, col, unit)
        ax.set_xlabel(label + (" [%s]" % unit.strip() if unit.strip() else ""))
        ax.set_ylabel("particles / bin")
        ax.legend(fontsize=13)
        exp_label(ax)
        savefig(fig, sub, "derived_%s" % key)
        metrics["particles"]["derived_%s" % key] = describe(val)


# --------------------------------------------------------------------------- #
# 3. vertex and first active impact
# --------------------------------------------------------------------------- #

def plot_positions(d, outdir, metrics):
    sub = os.path.join(outdir, "positions")

    specs = [
        ("vertex_position_x", d.p_vx, CMS_blue), ("vertex_position_y", d.p_vy, CMS_blue),
        ("vertex_position_z", d.p_vz, CMS_blue),
        ("first_active_impact_position_x", d.p_ix, CMS_red),
        ("first_active_impact_position_y", d.p_iy, CMS_red),
        ("first_active_impact_position_z", d.p_iz, CMS_red),
    ]
    for key, val, col in specs:
        fig, ax = plt.subplots()
        bins = 20 if val.ptp() > 1e-9 else np.linspace(val[0] - 1, val[0] + 1, 21)
        _hist(ax, val, bins, col, " mm")
        ax.set_xlabel("particles_%s [mm]" % key)
        ax.set_ylabel("particles / bin")
        ax.legend(fontsize=13)
        exp_label(ax)
        savefig(fig, sub, key)
        metrics.setdefault("particles", {})[key] = describe(val)

    # the gun scatters over the front face; the impact point follows the track
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(d.p_vx, d.p_vy, s=90, color=CMS_blue, alpha=0.85, edgecolor="k",
               lw=0.5, label="production vertex")
    ax.scatter(d.p_ix, d.p_iy, s=90, marker="^", color=CMS_red, alpha=0.85,
               edgecolor="k", lw=0.5, label="first active impact")
    for i in range(len(d.p_vx)):
        ax.plot([d.p_vx[i], d.p_ix[i]], [d.p_vy[i], d.p_iy[i]], "-", color=CMS_gray,
                lw=1, alpha=0.7)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.legend(fontsize=14)
    exp_label(ax)
    savefig(fig, sub, "vertex_vs_impact_xy")

    fig, ax = plt.subplots()
    _hist(ax, d.p_vr, 20, CMS_blue, " mm", "vertex")
    _hist(ax, d.p_ir, 20, CMS_red, " mm", "first active impact")
    ax.set_xlabel(r"$r = \sqrt{x^2+y^2}$ [mm]")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=11)
    exp_label(ax)
    savefig(fig, sub, "radius_vertex_vs_impact")

    fig, ax = plt.subplots()
    _hist(ax, d.p_flight, 20, CMS_purple, " mm")
    ax.set_xlabel("flight distance vertex -> first active impact [mm]")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "flight_distance")
    metrics["particles"]["flight_distance_mm"] = describe(d.p_flight)

    # which layer the particle first touches: the conversion depth for photons.
    # The layer comes from the cell index, not from z - see below.
    fig, ax = plt.subplots()
    _hist(ax, d.p_ilayer, np.arange(-0.5, d.n_layers + 0.5), CMS_orange)
    ax.set_xlabel("layer of the first active impact (from the cell index)")
    ax.set_ylabel("particles / bin")
    ax.set_xlim(-0.5, N_EE_LAYERS)
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "first_impact_layer")
    metrics["particles"]["first_impact_layer"] = describe(d.p_ilayer)

    # the impact point sits UPSTREAM of the plane reference z that the rechits
    # carry: it is the entry into the active volume, not the cell centre
    fig, ax = plt.subplots()
    _hist(ax, d.p_impact_dz, 20, CMS_red, " mm")
    ax.axvline(0, color="k", ls=":", lw=1.5)
    ax.set_xlabel(r"$z_{\mathrm{impact}} - z_{\mathrm{layer}}$ [mm]")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=12, loc="upper right")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.75)
    note(ax, "always $\\leq 0$: the impact point is the\nentry into the active volume, while\nrechits carry the plane reference z",
         loc="upper left", fontsize=12, color=CMS_red)
    exp_label(ax)
    savefig(fig, sub, "first_impact_z_offset")
    metrics["particles"]["first_impact_dz_mm"] = describe(d.p_impact_dz)

    fig, ax = plt.subplots()
    _hist(ax, d.p_isensor.astype(float), 20, CMS_gray)
    ax.set_xlabel("particles_first_active_impact_sensor_idx")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "first_active_impact_sensor_idx")

    # direction at the impact point vs at the vertex: how much it bends
    dang = np.degrees(np.arccos(np.clip(
        d.p_dx * d.p_idx_ + d.p_dy * d.p_idy + d.p_dz * d.p_idz, -1, 1)))
    fig, ax = plt.subplots()
    _hist(ax, dang, 20, CMS_red, " deg")
    ax.set_xlabel("angle between the vertex and first-impact directions [deg]")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "direction_change_vertex_to_impact")
    metrics["particles"]["direction_change_deg"] = describe(dang)


# --------------------------------------------------------------------------- #
# 4. energy deposition
# --------------------------------------------------------------------------- #

def plot_energy(d, outdir, metrics):
    sub = os.path.join(outdir, "energy")

    fig, ax = plt.subplots()
    _hist(ax, d.p_edep_active, 20, CMS_blue, " GeV")
    ax.set_xlabel("particles_total_energy_deposited_active [GeV]")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "total_energy_deposited_active")

    fig, ax = plt.subplots()
    _hist(ax, d.p_edep_all, 20, CMS_red, " GeV")
    ax.set_xlabel("particles_total_energy_deposited_all [GeV]")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "total_energy_deposited_all")

    metrics.setdefault("particles", {})["edep_active"] = describe(d.p_edep_active)
    metrics["particles"]["edep_all"] = describe(d.p_edep_all)

    # the sampling fraction of the toy, read straight off the truth record
    samp = 100 * d.p_edep_active / d.p_edep_all
    fig, ax = plt.subplots()
    _hist(ax, samp, 20, CMS_purple, " %")
    ax.set_xlabel(r"$E^{\mathrm{active}}_{\mathrm{dep}}\,/\,E^{\mathrm{all}}_{\mathrm{dep}}$ [%]")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "sampling_fraction_active_over_all")
    metrics["particles"]["sampling_fraction_percent"] = describe(samp)

    # containment: how much of the kinetic energy the detector actually absorbs
    cont = 100 * d.p_edep_all / d.p_ekin
    fig, ax = plt.subplots()
    _hist(ax, cont, 20, CMS_orange, " %")
    ax.set_xlabel(r"$E^{\mathrm{all}}_{\mathrm{dep}}\,/\,E_{\mathrm{kin}}$ [%]")
    ax.set_ylabel("particles / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "energy_containment")
    metrics["particles"]["containment_percent"] = describe(cont)

    fig, ax = plt.subplots()
    ax.scatter(d.p_ekin, d.p_edep_active, s=80, color=CMS_blue, alpha=0.85,
               edgecolor="k", lw=0.5)
    c = np.corrcoef(d.p_ekin, d.p_edep_active)[0, 1]
    ax.set_xlabel("particles_kinetic_energy [GeV]")
    ax.set_ylabel(r"$E^{\mathrm{active}}_{\mathrm{dep}}$ [GeV]")
    note(ax, r"$\rho$ = %.3f" % c)
    exp_label(ax)
    savefig(fig, sub, "edep_active_vs_ekin")
    metrics["particles"]["corr_ekin_edep_active"] = float(c)

    # a deeper first impact means less material in front -> less active deposit
    fig, ax = plt.subplots()
    ax.scatter(d.p_ilayer, d.p_edep_active, s=80, color=CMS_red, alpha=0.85,
               edgecolor="k", lw=0.5)
    c = np.corrcoef(d.p_ilayer, d.p_edep_active)[0, 1]
    ax.set_xlabel("layer of the first active impact")
    ax.set_ylabel(r"$E^{\mathrm{active}}_{\mathrm{dep}}$ [GeV]")
    note(ax, r"$\rho$ = %.3f" % c)
    exp_label(ax)
    savefig(fig, sub, "edep_active_vs_first_layer")
    metrics["particles"]["corr_firstlayer_edep_active"] = float(c)

    # this field is a placeholder in the current files: flag it explicitly
    fig, ax = plt.subplots()
    v = d.p_ie
    unset = float(np.mean(v < 0))
    _hist(ax, v, 21, CMS_gray, " GeV")
    ax.set_xlabel("particles_first_impact_active_energy [GeV]")
    ax.set_ylabel("particles / bin")
    note(ax, "%.0f%% of entries are $-1$:\nnot filled by the simulation" % (100 * unset),
         loc="upper right", color=CMS_red)
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "first_impact_active_energy")
    metrics["particles"]["first_impact_active_energy_unset_fraction"] = unset


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/photons_100GeV_eta20.parquet")
    ap.add_argument("--outdir", default="plots/toycalo/particles")
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()

    print("[particles] loading %s" % args.input)
    d = ToyCaloData(args.input, max_events=args.max_events)
    metrics = dict(dataset=d.summary())

    print("[particles] identity ...");   plot_identity(d, args.outdir, metrics)
    print("[particles] kinematics ...")
    plot_kinematics(d, args.outdir, metrics)
    print("[particles] positions ...");  plot_positions(d, args.outdir, metrics)
    print("[particles] energy ...");     plot_energy(d, args.outdir, metrics)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("[particles] done -> %s" % args.outdir)


if __name__ == "__main__":
    main()
