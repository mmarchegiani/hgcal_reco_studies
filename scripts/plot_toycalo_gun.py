"""Plot the ``gun_*`` generator parameters of a toy-calorimeter stage-1 file.

    python scripts/plot_toycalo_gun.py --input data/photons_100GeV_eta20.parquet

These are per-event scalars, so every histogram here has exactly ``n_events``
entries.  Besides the raw distributions the script checks the two conventions
that are easy to get wrong - the gun position is in **m** while the rest of the
file is in **mm**, and the gun direction is a unit vector divided by 1000 - and
closes the gun record against the ``particles_*`` truth.
"""

import argparse
import json
import os

import numpy as np

from toycalo_common import (
    GUN_DIR_SCALE, MM_PER_M, CMS_blue, CMS_gray, CMS_orange, CMS_purple, CMS_red,
    ToyCaloData, describe, exp_label, hist_stat_label, note, savefig, setup_style,
)

plt = setup_style()


def _hist(ax, v, bins, color, unit="", label=""):
    ax.hist(v, bins=bins, histtype="stepfilled", color=color, alpha=0.35)
    ax.hist(v, bins=bins, histtype="step", lw=2, color=color,
            label=(label + "\n" if label else "") + hist_stat_label(v, unit))


# --------------------------------------------------------------------------- #
# 1. the categorical part of the generator configuration
# --------------------------------------------------------------------------- #

def plot_configuration(d, outdir, metrics):
    sub = os.path.join(outdir, "configuration")
    cats = [("gun_id", d.gun_id), ("gun_pdgid", d.g_pdgid.astype(str)),
            ("simtype", d.simtype), ("simulator", d.simulator)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, (name, vals) in zip(axes.ravel(), cats):
        u, c = np.unique(vals, return_counts=True)
        ax.bar(np.arange(len(u)), c, width=0.5, color=CMS_blue, edgecolor="k")
        ax.set_xlim(-0.8, len(u) - 0.2)
        ax.set_xticks(np.arange(len(u)))
        ax.set_xticklabels([str(v) for v in u], fontsize=15)
        ax.set_ylabel("events")
        ax.set_title(name, fontsize=18)
        ax.set_ylim(0, c.max() * 1.2)
        for i, cc in enumerate(c):
            ax.text(i, cc, " %d" % cc, ha="center", va="bottom", fontsize=14)
        metrics.setdefault("gun", {})[name] = {str(k): int(v) for k, v in zip(u, c)}
    fig.suptitle("generator configuration (%d events)" % d.n_events, fontsize=20)
    savefig(fig, sub, "categorical_fields")

    fig, ax = plt.subplots()
    _hist(ax, d.event_id.astype(float), np.arange(-0.5, d.event_id.max() + 1.5), CMS_gray)
    ax.set_xlabel("event_id")
    ax.set_ylabel("events / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "event_id")


# --------------------------------------------------------------------------- #
# 2. energy
# --------------------------------------------------------------------------- #

def plot_energy(d, outdir, metrics):
    sub = os.path.join(outdir, "energy")

    fig, ax = plt.subplots()
    _hist(ax, d.g_energy, 20, CMS_blue, " GeV")
    ax.set_xlabel("gun_energy [GeV]")
    ax.set_ylabel("events / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "gun_energy")
    metrics.setdefault("gun", {})["gun_energy"] = describe(d.g_energy)

    # the nominal 100 GeV is smeared by a fraction of a percent
    rel = 100 * (d.g_energy / 100.0 - 1)
    fig, ax = plt.subplots()
    _hist(ax, rel, 20, CMS_red, " %")
    ax.axvline(0, color="k", ls=":", lw=1.5)
    ax.set_xlabel(r"$E_{\mathrm{gun}}/100\,\mathrm{GeV} - 1$ [%]")
    ax.set_ylabel("events / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "gun_energy_spread")
    metrics["gun"]["gun_energy_spread_percent"] = describe(rel)

    # the gun record must agree with the particle the simulation actually shot
    if len(d.p_ekin) == d.n_events:
        resid = d.p_ekin - d.g_energy
        fig, ax = plt.subplots()
        _hist(ax, resid, np.linspace(-1e-9, 1e-9, 41), CMS_purple, " GeV")
        ax.set_xlabel(r"particles_kinetic_energy $-$ gun_energy [GeV]")
        ax.set_ylabel("events / bin")
        ax.legend(fontsize=12)
        exp_label(ax)
        savefig(fig, sub, "gun_vs_particle_energy")
        metrics["gun"]["max_abs_energy_residual"] = float(np.abs(resid).max())

    # the raw calorimeter response to a fixed-energy gun
    fig, ax = plt.subplots()
    ax.scatter(d.g_energy, d.ev_sum_e, s=80, color=CMS_blue, alpha=0.85,
               edgecolor="k", lw=0.5)
    c = np.corrcoef(d.g_energy, d.ev_sum_e)[0, 1]
    ax.set_xlabel("gun_energy [GeV]")
    ax.set_ylabel(r"$\sum E_{\mathrm{rechit}}$ [GeV]")
    note(ax, r"$\rho$ = %.3f" % c + "\n(stochastic response dominates)")
    exp_label(ax)
    savefig(fig, sub, "gun_energy_vs_rechit_sum")
    metrics["gun"]["corr_gun_energy_rechit_sum"] = float(c)


# --------------------------------------------------------------------------- #
# 3. position
# --------------------------------------------------------------------------- #

def plot_position(d, outdir, metrics):
    sub = os.path.join(outdir, "position")
    for key, val, col in [("gun_position_x", d.g_x, CMS_blue),
                          ("gun_position_y", d.g_y, CMS_orange),
                          ("gun_position_z", d.g_z, CMS_red)]:
        fig, ax = plt.subplots()
        bins = 20 if val.ptp() > 1e-9 else np.linspace(val[0] - 1, val[0] + 1, 21)
        _hist(ax, val, bins, col, " mm")
        ax.set_xlabel("%s [mm]  (stored in m, x1000 here)" % key)
        ax.set_ylabel("events / bin")
        ax.legend(fontsize=13)
        exp_label(ax)
        savefig(fig, sub, key)
        metrics.setdefault("gun", {})[key + "_mm"] = describe(val)

    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(d.g_x, d.g_y, s=110, c=d.g_energy, cmap="viridis",
                    edgecolor="k", lw=0.5)
    fig.colorbar(sc, ax=ax, label="gun_energy [GeV]")
    for rr, col in [(d.g_r.min(), CMS_gray), (d.g_r.max(), CMS_gray)]:
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(rr * np.cos(t), rr * np.sin(t), ":", color=col, lw=1.5)
    ax.set_xlabel("gun_position_x [mm]")
    ax.set_ylabel("gun_position_y [mm]")
    ax.set_aspect("equal")
    ax.set_title(r"vertices lie on the $z$ = %.0f mm face, $r\in$[%.0f, %.0f] mm"
                 % (d.g_z[0], d.g_r.min(), d.g_r.max()), fontsize=16)
    savefig(fig, sub, "gun_position_xy")

    fig, ax = plt.subplots()
    _hist(ax, d.g_r, 20, CMS_purple, " mm")
    ax.set_xlabel(r"gun position $r=\sqrt{x^2+y^2}$ [mm]")
    ax.set_ylabel("events / bin")
    ax.legend(fontsize=13)
    exp_label(ax)
    savefig(fig, sub, "gun_position_radius")
    metrics["gun"]["gun_position_r_mm"] = describe(d.g_r)


# --------------------------------------------------------------------------- #
# 4. direction
# --------------------------------------------------------------------------- #

def plot_direction(d, outdir, metrics):
    sub = os.path.join(outdir, "direction")

    for key, val, col in [("gun_direction_x", d.g_dx, CMS_blue),
                          ("gun_direction_y", d.g_dy, CMS_orange),
                          ("gun_direction_z", d.g_dz, CMS_red)]:
        fig, ax = plt.subplots()
        _hist(ax, val, 20, col)
        ax.set_xlabel(r"%s $\times$ %d  (unit vector)" % (key, GUN_DIR_SCALE))
        ax.set_ylabel("events / bin")
        ax.legend(fontsize=13)
        exp_label(ax)
        savefig(fig, sub, key)
        metrics.setdefault("gun", {})[key + "_unit"] = describe(val)

    fig, ax = plt.subplots()
    _hist(ax, d.g_dir_norm, np.linspace(0.999, 1.001, 41), CMS_gray)
    ax.set_xlabel(r"$|\vec{d}_{\mathrm{gun}}| \times %d$" % GUN_DIR_SCALE)
    ax.set_ylabel("events / bin")
    ax.legend(fontsize=13)
    note(ax, "confirms the /1000 convention\nin the file metadata", loc="upper right")
    exp_label(ax)
    savefig(fig, sub, "gun_direction_norm")
    metrics["gun"]["gun_direction_norm"] = describe(d.g_dir_norm)

    for key, val, bins, col, lab in [
        ("gun_direction_eta", d.g_eta, np.linspace(1.95, 2.05, 41), CMS_blue, r"$\eta$"),
        ("gun_direction_phi", d.g_phi, np.linspace(0, 2 * np.pi, 25), CMS_orange, r"$\phi$"),
    ]:
        fig, ax = plt.subplots()
        _hist(ax, val, bins, col, " rad" if "phi" in key else "")
        ax.set_xlabel("%s  (%s)" % (key, lab))
        ax.set_ylabel("events / bin")
        ax.legend(fontsize=13)
        exp_label(ax)
        savefig(fig, sub, key)
        metrics["gun"][key] = describe(val)

    # closure: the stored eta/phi vs the ones recomputed from the direction vector
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for ax, stored, calc, lab in [
        (axes[0], d.g_eta, d.g_eta_calc, r"$\eta$"),
        (axes[1], d.g_phi, d.g_phi_calc, r"$\phi$"),
    ]:
        resid = calc - stored
        ax.hist(resid, bins=41, histtype="stepfilled", color=CMS_purple, alpha=0.35)
        ax.hist(resid, bins=41, histtype="step", lw=2, color=CMS_purple)
        ax.set_xlabel("recomputed %s $-$ stored %s" % (lab, lab))
        ax.set_ylabel("events / bin")
        ax.set_title("max |residual| = %.2g" % np.abs(resid).max(), fontsize=17)
    fig.suptitle("gun direction angle closure", fontsize=20)
    savefig(fig, sub, "angle_closure")
    metrics["gun"]["max_abs_eta_residual"] = float(np.abs(d.g_eta_calc - d.g_eta).max())
    metrics["gun"]["max_abs_phi_residual"] = float(np.abs(d.g_phi_calc - d.g_phi).max())

    # KNOWN BUG in the toy writer: gun_direction_phi is computed as
    # arctan(dy/dx) with a quadrant fix driven by the sign of dy alone, so it is
    # off by exactly pi whenever dx and dy have opposite signs.  The direction
    # *vector* is fine - recompute phi from it instead of reading the field.
    true_phi = d.g_phi_calc
    dphi = np.mod(true_phi - d.g_phi + np.pi, 2 * np.pi) - np.pi
    broken = np.abs(np.abs(dphi) - np.pi) < 1e-9
    opposite_sign = (d.g_dx * d.g_dy) < 0

    fig, ax = plt.subplots()
    ax.scatter(d.g_phi[~broken], true_phi[~broken], s=110, color=CMS_blue, alpha=0.9,
               edgecolor="k", lw=0.5, label="consistent (%d events)" % (~broken).sum())
    ax.scatter(d.g_phi[broken], true_phi[broken], s=130, marker="X", color=CMS_red,
               alpha=0.9, edgecolor="k", lw=0.5,
               label=r"off by $\pi$ (%d events)" % broken.sum())
    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], "--", color=CMS_gray, lw=2, label="y = x")
    ax.set_xlabel("gun_direction_phi (as stored) [rad]")
    ax.set_ylabel(r"$\mathrm{atan2}(d_y, d_x)$ [rad]")
    ax.legend(fontsize=12, loc="upper left")
    note(ax, r"stored $\phi$ is wrong by $\pi$ iff"
             "\n" r"$\mathrm{sign}(d_x) \neq \mathrm{sign}(d_y)$: %s"
             % ("confirmed" if np.array_equal(broken, opposite_sign) else "NOT confirmed"),
         loc="lower right", color=CMS_red, fontsize=14)
    exp_label(ax)
    savefig(fig, sub, "gun_direction_phi_bug")
    metrics["gun"]["phi_off_by_pi_fraction"] = float(broken.mean())
    metrics["gun"]["phi_bug_matches_opposite_sign_dx_dy"] = bool(
        np.array_equal(broken, opposite_sign))

    # the gun fires radially outward: the position phi equals the *recomputed* phi
    pos_phi = np.mod(np.arctan2(d.g_y, d.g_x), 2 * np.pi)
    dpos = np.mod(pos_phi - true_phi + np.pi, 2 * np.pi) - np.pi
    fig, ax = plt.subplots()
    ax.scatter(true_phi, pos_phi, s=100, color=CMS_blue, alpha=0.85, edgecolor="k", lw=0.5)
    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], "--", color=CMS_red, lw=2, label="y = x")
    ax.set_xlabel(r"$\phi$ recomputed from the gun direction [rad]")
    ax.set_ylabel(r"$\phi$ of the gun position [rad]")
    ax.legend(fontsize=14, loc="upper left")
    note(ax, r"max $|\Delta\phi|$ = %.1e rad" % np.abs(dpos).max() +
             "\nthe gun points away from the beamline", loc="lower right", fontsize=14)
    exp_label(ax)
    savefig(fig, sub, "position_phi_vs_direction_phi")
    metrics["gun"]["max_abs_dphi_position_direction"] = float(np.abs(dpos).max())

    # same check for eta: the vertex sits on the eta of the direction
    eta_pos = -np.log(np.tan(0.5 * np.arctan2(d.g_r, d.g_z)))
    fig, ax = plt.subplots()
    ax.scatter(d.g_eta, eta_pos, s=90, color=CMS_orange, alpha=0.85, edgecolor="k", lw=0.5)
    lim = [min(d.g_eta.min(), eta_pos.min()) - 0.005, max(d.g_eta.max(), eta_pos.max()) + 0.005]
    ax.plot(lim, lim, "--", color=CMS_red, lw=2, label="y = x")
    ax.set_xlabel(r"gun_direction_eta")
    ax.set_ylabel(r"$\eta$ of the gun position w.r.t. the origin")
    ax.legend(fontsize=14)
    exp_label(ax)
    savefig(fig, sub, "position_eta_vs_direction_eta")
    metrics["gun"]["max_abs_deta_position_direction"] = float(np.abs(eta_pos - d.g_eta).max())

    # closure against the particle record
    if len(d.p_dx) == d.n_events:
        dang = np.degrees(np.arccos(np.clip(
            d.g_dx * d.p_dx + d.g_dy * d.p_dy + d.g_dz * d.p_dz, -1, 1)))
        fig, ax = plt.subplots()
        _hist(ax, dang, 41, CMS_purple, " deg")
        ax.set_xlabel("angle(gun direction, particle momentum direction) [deg]")
        ax.set_ylabel("events / bin")
        ax.legend(fontsize=12)
        exp_label(ax)
        savefig(fig, sub, "gun_vs_particle_direction")
        metrics["gun"]["max_angle_gun_particle_deg"] = float(dang.max())

    # and against the shower the gun actually produced
    e_eta = np.bincount(d.h_event, weights=d.h_e * d.h_eta, minlength=d.n_events) / d.ev_sum_e
    fig, ax = plt.subplots()
    ax.scatter(d.g_eta, e_eta, s=90, color=CMS_blue, alpha=0.85, edgecolor="k", lw=0.5)
    lim = [1.94, 2.06]
    ax.plot(lim, lim, "--", color=CMS_red, lw=2, label="y = x")
    ax.set_xlabel(r"gun_direction_eta")
    ax.set_ylabel(r"energy-weighted rechit $\langle\eta\rangle$")
    ax.legend(fontsize=14)
    exp_label(ax)
    savefig(fig, sub, "gun_eta_vs_shower_eta")
    metrics["gun"]["mean_shower_eta_bias"] = float(np.mean(e_eta - d.g_eta))


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/photons_100GeV_eta20.parquet")
    ap.add_argument("--outdir", default="plots/toycalo/gun_parameters")
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()

    print("[gun] loading %s" % args.input)
    d = ToyCaloData(args.input, max_events=args.max_events)
    metrics = dict(dataset=d.summary(), units=d.units)

    print("[gun] configuration ..."); plot_configuration(d, args.outdir, metrics)
    print("[gun] energy ...");        plot_energy(d, args.outdir, metrics)
    print("[gun] position ...");      plot_position(d, args.outdir, metrics)
    print("[gun] direction ...");     plot_direction(d, args.outdir, metrics)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("[gun] done -> %s" % args.outdir)


if __name__ == "__main__":
    main()
