"""Shower profile (HGCAL layer vs. y) of one particle of each type, in a single event.

    python scripts/plot_shower_profiles.py \
        --input data/taus_FineCalo_with_tracksters/tays_fineCalo.parquet \
        --outdir plots/shower_profiles

The point of the picture is the *shape* difference: an electromagnetic shower
(photon, electron) is a narrow, compact blob that starts early and dies out in
the CE-E, while a hadronic one (charged/neutral hadron) is wide, irregular and
leaks deep into the CE-H.

One event is used for all panels.  Inside it, one particle (SimCluster) is
picked per requested class -- the most energetic one that has at least
``--min-clusters`` LayerClusters associated to it by the CMS truth association
(the best match is the one with the *smallest* score, see
``hgcal_eval_common.EvalData._best_truth_match``).  The event itself is chosen,
among the scanned ones, as the one where the *weakest* of those per-class
particles is as energetic as possible, so that every panel shows a real shower.

``sum_energy_lc`` adds up the LayerClusters whose *best* truth match is the
shown particle, so it is not the particle energy: a LayerCluster shared with an
overlapping shower is counted in full here and dropped entirely from the other
particle.

The files carry no layer number, and ``LayerCluster_seedDetId`` is not filled,
so the layer is rebuilt from |z|: the distinct |z| values of the LayerClusters
are grouped with a ``--layer-tol`` tolerance and numbered from 1 at the front of
the CE-E.  The 2 mm Si/scintillator offset inside a CE-H layer is absorbed by
the grouping, the ~1 cm spacing of the CE-E layers is not.
"""

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hgcal_eval_common import (
    CMS_gray, EvalData, _f, _offsets, class_color, class_name, pdg_to_class,
    savefig, setup_style,
)

plt = setup_style()

#: default panels, in the order the user reads them: EM first, hadronic second
DEFAULT_CLASSES = [4, 1, 0, 3]

PDG_NAMES = {22: r"$\gamma$", 11: r"$e^-$", -11: r"$e^+$", 13: r"$\mu^-$",
             -13: r"$\mu^+$", 211: r"$\pi^+$", -211: r"$\pi^-$", 111: r"$\pi^0$",
             2112: "n", -2112: r"$\bar{n}$", 2212: "p", -2212: r"$\bar{p}$",
             130: r"$K^0_L$", 310: r"$K^0_S$", 321: r"$K^+$", -321: r"$K^-$",
             3122: r"$\Lambda$", -3122: r"$\bar{\Lambda}$"}


def pdg_label(pdg):
    return "%s (%d)" % (PDG_NAMES.get(int(pdg), "pdgId"), int(pdg))


# --------------------------------------------------------------------------- #
# layer numbering from |z|
# --------------------------------------------------------------------------- #

def layer_centres(abs_z, tol):
    """Group the distinct |z| of the LayerClusters into layers, front to back."""
    u = np.unique(np.round(np.asarray(abs_z, float), 3))
    if len(u) == 0:
        raise ValueError("no LayerCluster to build the layer numbering from")
    cut = np.nonzero(np.diff(u) > tol)[0]
    return np.array([g.mean() for g in np.split(u, cut + 1)])


def assign_layer(abs_z, centres):
    """Layer number (1 = first CE-E layer) of each LayerCluster."""
    edges = 0.5 * (centres[1:] + centres[:-1])
    return np.searchsorted(edges, np.asarray(abs_z, float)) + 1


def n_ee_layers(centres, min_gap=3.0):
    """Number of CE-E layers = layers before the first big jump in z (cm)."""
    big = np.nonzero(np.diff(centres) > min_gap)[0]
    return int(big[0]) + 1 if len(big) else len(centres)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

class Event3D:
    """Flat view of the LayerCluster <-> SimCluster truth of the first events."""

    def __init__(self, path, max_events, layer_tol):
        df = ak.from_parquet(path, columns=["SimCluster", "LayerCluster"])
        if max_events is not None:
            df = df[:max_events]
        sc, lc = df.SimCluster, df.LayerCluster
        self.path = path
        self.n_events = len(df)

        p_count = np.asarray(ak.to_numpy(ak.num(sc.SimCluster_pdgId, axis=1)))
        n_count = np.asarray(ak.to_numpy(ak.num(lc.LayerCluster_energy, axis=1)))
        p_off = _offsets(p_count)
        self.p_event = np.repeat(np.arange(self.n_events), p_count)
        self.n_event = np.repeat(np.arange(self.n_events), n_count)

        # ---- particles
        self.p_pdg = _f(sc.SimCluster_pdgId)
        self.p_class = pdg_to_class(self.p_pdg).astype(np.int64)
        self.p_e = _f(sc.SimCluster_recEnergy).astype(np.float64)
        self.p_eta = _f(sc.SimCluster_impactPoint_eta).astype(np.float64)
        self.p_y = _f(sc.SimCluster_impactPoint_y).astype(np.float64)
        self.p_z = _f(sc.SimCluster_impactPoint_z).astype(np.float64)

        # ---- LayerClusters
        self.n_e = _f(lc.LayerCluster_energy).astype(np.float64)
        self.n_y = _f(lc.LayerCluster_y).astype(np.float64)
        self.n_z = _f(lc.LayerCluster_z).astype(np.float64)
        self.centres = layer_centres(np.abs(self.n_z), layer_tol)
        self.n_layer = assign_layer(np.abs(self.n_z), self.centres)
        self.n_layers = len(self.centres)
        self.n_ee = n_ee_layers(self.centres)

        # ---- truth association, as a global particle index per LayerCluster
        best, _ = EvalData._best_truth_match(lc)
        self.n_particle = np.where(best >= 0, p_off[self.n_event] + best, -1)
        self.p_n_lc = np.bincount(self.n_particle[self.n_particle >= 0],
                                  minlength=len(self.p_e))

    def candidates(self, cls, min_clusters, min_energy):
        return ((self.p_class == cls) & (self.p_n_lc >= min_clusters)
                & (self.p_e >= min_energy))

    def pick_particle(self, event, cls, min_clusters, min_energy):
        """Most energetic particle of ``cls`` in ``event`` passing the cuts."""
        ok = np.nonzero(self.candidates(cls, min_clusters, min_energy)
                        & (self.p_event == event))[0]
        return int(ok[np.argmax(self.p_e[ok])]) if len(ok) else None

    def pick_event(self, classes, min_clusters, min_energy):
        """Event whose weakest per-class particle is the most energetic."""
        best_e = np.full((len(classes), self.n_events), -np.inf)
        for i, c in enumerate(classes):
            ok = np.nonzero(self.candidates(c, min_clusters, min_energy))[0]
            np.maximum.at(best_e[i], self.p_event[ok], self.p_e[ok])
        score = best_e.min(axis=0)
        if not np.isfinite(score).any():
            missing = [class_name(c) for i, c in enumerate(classes)
                       if not np.isfinite(best_e[i]).any()]
            raise SystemExit(
                "no event in the first %d has a particle of every requested class "
                "with >= %d LayerClusters and E >= %g GeV (never found: %s); "
                "loosen --min-clusters/--min-energy, raise --max-events or drop "
                "classes with --classes"
                % (self.n_events, min_clusters, min_energy,
                   ", ".join(missing) or "none -- no single event has them all"))
        return int(np.argmax(score))


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

#: marker area (pt^2) for the smallest / largest LayerCluster of a panel
S_MIN, S_MAX = 18.0, 420.0


def marker_size(e, e_ref):
    """Area ~ sqrt(E): energy spans decades, the eye reads area, not radius."""
    return S_MIN + (S_MAX - S_MIN) * np.sqrt(np.clip(e, 0, None) / e_ref)


def size_legend(ax, e_ref, color):
    """Three reference energies, since area is the only energy encoding."""
    steps = [e for e in (0.1, 1.0, 10.0, 100.0) if e <= e_ref * 1.05][-3:]
    handles = [plt.scatter([], [], s=marker_size(e, e_ref), facecolor=color,
                           edgecolor="k", linewidth=0.4, alpha=0.75,
                           label="%g GeV" % e) for e in steps]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=13,
                    labelspacing=1.1, borderpad=0.8, handletextpad=1.0,
                    frameon=True, framealpha=0.85, title="LC energy",
                    title_fontsize=13)
    ax.add_artist(leg)


def draw_panel(ax, d, ip, y_span=None, show_size_legend=True):
    """Scatter of (layer, y) for the LayerClusters of particle ``ip``."""
    sel = d.n_particle == ip
    lay, y, e = d.n_layer[sel], d.n_y[sel], d.n_e[sel]
    cls = int(d.p_class[ip])
    color = class_color(cls)

    # CE-H shading: the depth at which a hadronic shower is still alive
    ax.axvspan(d.n_ee + 0.5, d.n_layers + 0.5, color=CMS_gray, alpha=0.10, lw=0)
    ax.axvline(d.n_ee + 0.5, color=CMS_gray, lw=1.2, ls="--")
    ax.text(d.n_ee - 0.7, 0.015, "CE-E", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=13, color=CMS_gray)
    ax.text(d.n_ee + 1.5, 0.015, "CE-H", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=13, color=CMS_gray)

    e_ref = max(e.max(), 1e-3)
    ax.scatter(lay, y, s=marker_size(e, e_ref), facecolor=color, edgecolor="k",
               linewidth=0.4, alpha=0.75, zorder=3)
    ax.axhline(d.p_y[ip], color=color, lw=1.5, ls=":", alpha=0.9, zorder=2)

    ax.set_xlim(0.5, d.n_layers + 0.5)
    if y_span is not None:
        c = 0.5 * (y.min() + y.max())
        ax.set_ylim(c - 0.5 * y_span, c + 0.5 * y_span)
    ax.set_xlabel("HGCAL layer")
    ax.set_ylabel("LayerCluster $y$ [cm]")

    info = "\n".join([
        "%s, %s" % (class_name(cls), pdg_label(d.p_pdg[ip])),
        r"$E$ = %.1f GeV,  $\eta$ = %+.2f" % (d.p_e[ip], d.p_eta[ip]),
        r"%d LC,  $\sum E_{\mathrm{LC}}$ = %.1f GeV" % (len(e), e.sum()),
        r"layers %d-%d,  RMS$(y)$ = %.1f cm" % (lay.min(), lay.max(), y.std()),
    ])
    ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=14, va="top",
            ha="left", bbox=dict(boxstyle="round,pad=0.35", fc="w", ec=color,
                                 lw=1.5, alpha=0.85))
    if show_size_legend:
        size_legend(ax, e_ref, color)
    return dict(n_clusters=int(len(e)), sum_energy_lc=float(e.sum()),
                first_layer=int(lay.min()), last_layer=int(lay.max()),
                mean_layer_eweighted=float(np.average(lay, weights=e)),
                rms_y=float(y.std()),
                rms_y_eweighted=float(np.sqrt(np.average(
                    (y - np.average(y, weights=e)) ** 2, weights=e))))


def panel_span(d, particles):
    """Common y span so the panel widths can be compared by eye."""
    spans = []
    for ip in particles.values():
        y = d.n_y[d.n_particle == ip]
        spans.append(y.max() - y.min())
    return 1.25 * max(spans) + 2.0


def plot_all(d, event, particles, outdir, metrics):
    span = panel_span(d, particles)
    per_particle = {}

    for cls, ip in particles.items():
        fig, ax = plt.subplots()
        per_particle[class_name(cls)] = draw_panel(ax, d, ip, y_span=span)
        ax.set_title("Shower profile of a %s  (event %d)" % (class_name(cls), event),
                     fontsize=19)
        savefig(fig, outdir, "profile_%s" % class_name(cls).replace(" ", "_"))

    ncol = 2 if len(particles) > 1 else 1
    nrow = int(np.ceil(len(particles) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5 * ncol, 6.2 * nrow),
                             sharex=True, squeeze=False)
    for ax, (cls, ip) in zip(axes.ravel(), particles.items()):
        draw_panel(ax, d, ip, y_span=span, show_size_legend=False)
    for ax in axes.ravel()[len(particles):]:
        ax.set_visible(False)
    # sharex hides the tick labels of every row but the last one: keep the axis
    # label only where the numbers are actually drawn
    for col in range(ncol):
        vis = [ax for ax in axes[:, col] if ax.get_visible()]
        for ax in vis[:-1]:
            ax.set_xlabel("")
    fig.suptitle("HGCAL shower shapes by particle type: event %d "
                 "(same $y$ span in every panel)" % event, fontsize=21)
    fig.tight_layout()
    savefig(fig, outdir, "profile_comparison")

    for cls, ip in particles.items():
        per_particle[class_name(cls)].update(
            pdgId=int(d.p_pdg[ip]), energy=float(d.p_e[ip]),
            eta=float(d.p_eta[ip]), impact_y=float(d.p_y[ip]),
            particle_index_in_event=int(ip - np.searchsorted(d.p_event, event)))
    metrics["particles"] = per_particle


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/taus_FineCalo_with_tracksters/"
                                       "taus_fineCalo.parquet")
    ap.add_argument("--outdir", default="plots/shower_profiles")
    ap.add_argument("--event", type=int, default=None,
                    help="event to display (default: auto-selected)")
    ap.add_argument("--max-events", type=int, default=200,
                    help="events read from the file, i.e. the auto-selection pool")
    ap.add_argument("--classes", default=",".join(str(c) for c in DEFAULT_CLASSES),
                    help="comma-separated class indices: charged hadron 0, "
                         "electron 1, muon 2, neutral hadron 3, photon 4")
    ap.add_argument("--min-clusters", type=int, default=10,
                    help="minimum number of LayerClusters of the shown particle")
    ap.add_argument("--min-energy", type=float, default=1.0,
                    help="minimum SimCluster_recEnergy [GeV] of the shown particle")
    ap.add_argument("--layer-tol", type=float, default=0.5,
                    help="|z| tolerance [cm] when grouping LayerClusters into layers")
    args = ap.parse_args()

    classes = [int(c) for c in args.classes.split(",") if c.strip() != ""]
    d = Event3D(args.input, args.max_events, args.layer_tol)
    print("read %d events, %d LayerClusters, %d layers (%d CE-E + %d CE-H)"
          % (d.n_events, len(d.n_e), d.n_layers, d.n_ee, d.n_layers - d.n_ee))

    event = args.event
    if event is None:
        event = d.pick_event(classes, args.min_clusters, args.min_energy)
        print("auto-selected event %d" % event)

    particles = {}
    for c in classes:
        ip = d.pick_particle(event, c, args.min_clusters, args.min_energy)
        if ip is None:
            print("  no %s in event %d with >= %d LC and E >= %g GeV -- skipped"
                  % (class_name(c), event, args.min_clusters, args.min_energy))
            continue
        particles[c] = ip
        print("  %-16s pdgId %6d  E = %7.2f GeV  %4d LC"
              % (class_name(c), d.p_pdg[ip], d.p_e[ip], d.p_n_lc[ip]))
    if not particles:
        raise SystemExit("event %d has no particle passing the cuts" % event)

    metrics = dict(file=os.path.basename(args.input), event=int(event),
                   n_events_scanned=int(d.n_events),
                   n_layers=int(d.n_layers), n_ee_layers=int(d.n_ee),
                   layer_z_cm=[float(z) for z in d.centres],
                   selection=dict(min_clusters=args.min_clusters,
                                  min_energy=args.min_energy,
                                  layer_tol=args.layer_tol))
    plot_all(d, event, particles, args.outdir, metrics)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "shower_profiles.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("wrote %s" % args.outdir)


if __name__ == "__main__":
    main()
