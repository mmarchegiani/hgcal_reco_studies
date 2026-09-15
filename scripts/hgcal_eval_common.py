"""Shared helpers for evaluating the HGCAL MaskFormer predictions.

The prediction files are awkward/parquet dumps with one record per event and two
collections that matter here:

``SimCluster``
    One entry per *true* particle.  The ``SimCluster_pred_*`` fields hold the
    model output for the query that the Hungarian matcher assigned to that
    particle, so index ``i`` of a ``pred`` field and index ``i`` of a truth field
    describe the same object.

``LayerCluster``
    One entry per input node.  ``LayerCluster_pred_match_idx`` is the SimCluster
    index the model assigned the node to (``-2``/``-1`` = unassigned).  The truth
    side is stored ragged-inside-ragged: ``LayerCluster_SimCluster_MatchIdx`` and
    ``LayerCluster_SimCluster_MatchQual`` are the concatenation over nodes of
    ``LayerCluster_SimClusterNumMatch`` entries each.  ``MatchQual`` is the CMS
    association *score*, so the best match is the one with the SMALLEST value.
"""

import os

import awkward as ak
import numpy as np

# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #

CMAP_10 = [
    "#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
    "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd",
]
CMS_blue, CMS_orange, CMS_red, CMS_gray, CMS_purple = CMAP_10[:5]


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    return plt


# --------------------------------------------------------------------------- #
# PDG -> class
# --------------------------------------------------------------------------- #

class_labels = {
    -211: 0, 211: 0, -213: 0, 213: 0, -221: 0, 221: 0, -223: 0, 223: 0,
    -321: 0, 321: 0, -323: 0, 323: 0, -331: 3, 331: 3, -333: 3, 333: 3,
    311: 3, -311: 3, -411: 0, 411: 0, -413: 0, 413: 0, -423: 3, 423: 3,
    -431: 0, 431: 0, -433: 0, 433: 0, -511: 3, 511: 3, -521: 0, 521: 0,
    -523: 0, 523: 0, -531: 3, 531: 3, -541: 0, 541: 0, -1114: 0, 1114: 0,
    -2114: 0, 2114: 0, -2212: 0, 2212: 0, -3112: 0, 3112: 0, -3312: 0,
    3312: 0, -3222: 0, 3222: 0, -3334: 0, 3334: 0, -4122: 0, 4122: 0,
    -4132: 3, 4132: 3, -4232: 0, 4232: 0, -4312: 0, 4312: 0, -4322: 0,
    4322: 0, -4324: 0, 4324: 0, -4332: 3, 4332: 3, -4334: 3, 4334: 3,
    -5112: 0, 5112: 0, -5122: 3, 5122: 3, -5132: 0, 5132: 0, -5232: 3,
    5232: 3, -5332: 0, 5332: 0, -11: 1, 11: 1, -13: 2, 13: 2, -15: 0,
    15: 0, -111: 3, 111: 3, 113: 3, 130: 3, 310: 3, -313: 3, 313: 3,
    -421: 3, 421: 3, -2112: 3, 2112: 3, -3122: 3, 3122: 3, -3322: 3,
    3322: 3, 22: 4,
    1000010020: 0, 1000010030: 0, 1000010040: 0, 1000020030: 0,
    1000020040: 0, 1000030040: 0, 1000030050: 0, 1000020060: 0,
    1000020070: 0, 1000020080: 0, 1000010048: 0, 1000020032: 0,
    -999: 5, -12: -1, 12: -1, -14: -1, 14: -1, -16: -1, 16: -1, 0: -2,
}

_keys = np.array(sorted(class_labels), dtype=np.int64)
_vals = np.array([class_labels[k] for k in _keys], dtype=np.int8)
UNMAPPED = np.int8(-2)

#: class index -> (name, colour).  5 is the model's "null / no object" class.
CLASS_INFO = {
    0: ("charged hadron", CMAP_10[0]),
    1: ("electron", CMAP_10[1]),
    2: ("muon", CMAP_10[2]),
    3: ("neutral hadron", CMAP_10[4]),
    4: ("photon", CMAP_10[6]),
    5: ("residual / null", CMAP_10[3]),
}
PHYSICS_CLASSES = [0, 1, 2, 3, 4]


def pdg_to_class(flat):
    flat = np.asarray(flat, dtype=np.int64)
    i = np.searchsorted(_keys, flat)
    i_clip = np.clip(i, 0, len(_keys) - 1)
    hit = _keys[i_clip] == flat
    return np.where(hit, _vals[i_clip], UNMAPPED)


def class_name(c):
    return CLASS_INFO.get(int(c), (f"class {int(c)}", CMS_gray))[0]


def class_color(c):
    return CLASS_INFO.get(int(c), (f"class {int(c)}", CMS_gray))[1]


# --------------------------------------------------------------------------- #
# loading / flattening
# --------------------------------------------------------------------------- #

def _f(arr):
    """Flatten a jagged array to a contiguous numpy array."""
    return np.asarray(ak.to_numpy(ak.flatten(arr)))


def _offsets(counts):
    off = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    return off


class EvalData:
    """Flat (event-concatenated) view of one prediction file.

    Attributes with the ``p_`` prefix are per *particle* (SimCluster), those with
    the ``n_`` prefix are per *node* (LayerCluster).  Node->particle pointers are
    stored as **global** indices into the particle arrays so that everything can
    be done with plain numpy.
    """

    def __init__(self, path, max_events=None):
        self.path = path
        df = ak.from_parquet(path)
        if max_events is not None:
            df = df[:max_events]
        self.df = df
        sc, lc = df.SimCluster, df.LayerCluster

        self.n_events = len(df)
        self.p_count = np.asarray(ak.to_numpy(ak.num(sc.SimCluster_pdgId, axis=1)))
        self.n_count = np.asarray(ak.to_numpy(ak.num(lc.LayerCluster_energy, axis=1)))
        self.p_off = _offsets(self.p_count)
        self.n_off = _offsets(self.n_count)
        self.p_event = np.repeat(np.arange(self.n_events), self.p_count)
        self.n_event = np.repeat(np.arange(self.n_events), self.n_count)

        # ---------------- particle level: truth ----------------
        self.p_pdg = _f(sc.SimCluster_pdgId)
        self.p_class = pdg_to_class(self.p_pdg).astype(np.int64)
        self.p_e = _f(sc.SimCluster_recEnergy).astype(np.float64)
        self.p_e_boundary = _f(sc.SimCluster_boundaryEnergy).astype(np.float64)
        self.p_eta = _f(sc.SimCluster_impactPoint_eta).astype(np.float64)
        self.p_phi = _f(sc.SimCluster_impactPoint_phi).astype(np.float64)
        self.p_x = _f(sc.SimCluster_impactPoint_x).astype(np.float64)
        self.p_y = _f(sc.SimCluster_impactPoint_y).astype(np.float64)
        self.p_z = _f(sc.SimCluster_impactPoint_z).astype(np.float64)
        self.p_sinphi = np.sin(self.p_phi)
        self.p_cosphi = np.cos(self.p_phi)
        # pred_r matches the 3D impact-point radius (~300 cm), not the transverse one
        self.p_r = np.sqrt(self.p_x ** 2 + self.p_y ** 2 + self.p_z ** 2)

        # ---------------- particle level: prediction ----------------
        self.p_pred_e = _f(sc.SimCluster_pred_e).astype(np.float64)
        self.p_pred_eta = _f(sc.SimCluster_pred_eta).astype(np.float64)
        self.p_pred_sinphi = _f(sc.SimCluster_pred_sinphi).astype(np.float64)
        self.p_pred_cosphi = _f(sc.SimCluster_pred_cosphi).astype(np.float64)
        self.p_pred_r = _f(sc.SimCluster_pred_r).astype(np.float64)
        self.p_pred_phi = _f(sc.SimCluster_pred_phi).astype(np.float64)
        self.p_pred_class = _f(sc.SimCluster_pred_class).astype(np.int64)
        self.p_valid = _f(sc.SimCluster_pred_valid).astype(bool)
        self.p_pred_n_nodes = _f(sc.SimCluster_pred_n_nodes).astype(np.int64)

        # ---------------- node level ----------------
        self.n_energy = _f(lc.LayerCluster_energy).astype(np.float64)
        self.n_eta = _f(lc.LayerCluster_eta).astype(np.float64)
        self.n_phi = _f(lc.LayerCluster_phi).astype(np.float64)
        self.n_z = _f(lc.LayerCluster_z).astype(np.float64)

        n_pred_local = _f(lc.LayerCluster_pred_match_idx).astype(np.int64)
        n_true_local, n_true_qual = self._best_truth_match(lc)
        self.n_pred_local = n_pred_local
        self.n_true_local = n_true_local
        self.n_true_qual = n_true_qual

        base = self.p_off[self.n_event]
        self.n_pred = np.where(n_pred_local >= 0, base + n_pred_local, -1)
        self.n_true = np.where(n_true_local >= 0, base + n_true_local, -1)

        # per-particle node bookkeeping derived from the assignments
        self.p_n_true_nodes = np.bincount(self.n_true[self.n_true >= 0],
                                          minlength=len(self.p_e))
        self.p_n_pred_nodes = np.bincount(self.n_pred[self.n_pred >= 0],
                                          minlength=len(self.p_e))
        self.p_sum_e_true_nodes = np.bincount(
            self.n_true[self.n_true >= 0],
            weights=self.n_energy[self.n_true >= 0], minlength=len(self.p_e))
        self.p_sum_e_pred_nodes = np.bincount(
            self.n_pred[self.n_pred >= 0],
            weights=self.n_energy[self.n_pred >= 0], minlength=len(self.p_e))

    @staticmethod
    def _best_truth_match(lc):
        """Return, per node, the SimCluster index with the best (smallest) score."""
        nmatch = _f(lc.LayerCluster_SimClusterNumMatch).astype(np.int64)
        idx = _f(lc.LayerCluster_SimCluster_MatchIdx).astype(np.int64)
        qual = _f(lc.LayerCluster_SimCluster_MatchQual).astype(np.float64)
        assert nmatch.sum() == len(idx), "NumMatch does not add up to MatchIdx length"

        off = _offsets(nmatch)
        best = np.full(len(nmatch), -2, dtype=np.int64)
        best_q = np.full(len(nmatch), np.nan)
        has = nmatch > 0
        if not has.any():
            return best, best_q
        # vectorised segment-argmin: sort by (segment, score) and take the first
        seg = np.repeat(np.arange(len(nmatch)), nmatch)
        order = np.lexsort((qual, seg))
        seg_s, idx_s, qual_s = seg[order], idx[order], qual[order]
        first = np.empty(len(seg_s), dtype=bool)
        first[0] = True
        first[1:] = seg_s[1:] != seg_s[:-1]
        best[seg_s[first]] = idx_s[first]
        best_q[seg_s[first]] = qual_s[first]
        return best, best_q

    # -------------------------------------------------------------- #
    def particle_mask(self, valid_only):
        return self.p_valid if valid_only else np.ones(len(self.p_e), dtype=bool)

    def summary(self):
        return dict(
            file=os.path.basename(self.path),
            n_events=self.n_events,
            n_particles=len(self.p_e),
            n_nodes=len(self.n_energy),
            valid_fraction=float(self.p_valid.mean()),
        )


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #

def robust_sigma(x):
    """IQR/1.349 - insensitive to the long tails of a resolution distribution."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    q16, q84 = np.percentile(x, [15.865, 84.135])
    return 0.5 * (q84 - q16)


def stat_label(x, unit=""):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "empty"
    return (r"$\mu_{1/2}$=%.3g%s  $\sigma_{68}$=%.3g%s  N=%d"
            % (np.median(x), unit, robust_sigma(x), unit, len(x)))


def binomial_err(k, n):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    err = np.sqrt(np.clip(p * (1 - p), 0, None) / np.where(n > 0, n, np.nan))
    return p, err


def profile(x, y, edges, stat="median"):
    """Return bin centres, the chosen statistic of y per bin, its spread and N."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    which = np.digitize(x, edges) - 1
    ctr = 0.5 * (edges[:-1] + edges[1:])
    val = np.full(len(ctr), np.nan)
    spread = np.full(len(ctr), np.nan)
    n = np.zeros(len(ctr), dtype=int)
    for i in range(len(ctr)):
        sel = which == i
        n[i] = sel.sum()
        if n[i] < 3:
            continue
        yy = y[sel]
        val[i] = np.median(yy) if stat == "median" else yy.mean()
        spread[i] = robust_sigma(yy) if stat == "median" else yy.std() / np.sqrt(n[i])
    return ctr, val, spread, n


def eff_profile(x, ok, edges):
    """Binomial efficiency of boolean ``ok`` in bins of ``x``."""
    x = np.asarray(x, float)
    ok = np.asarray(ok, bool)
    good = np.isfinite(x)
    x, ok = x[good], ok[good]
    which = np.digitize(x, edges) - 1
    ctr = 0.5 * (edges[:-1] + edges[1:])
    k = np.zeros(len(ctr)); n = np.zeros(len(ctr))
    for i in range(len(ctr)):
        sel = which == i
        n[i] = sel.sum()
        k[i] = ok[sel].sum()
    p, e = binomial_err(k, n)
    return ctr, p, e, n.astype(int)


def savefig(fig, outdir, name, log_pdf=True):
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, name + ".png")
    fig.savefig(png, bbox_inches="tight")
    if log_pdf:
        fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return png
