"""Shared helpers for the ``hg_toy_calorimeter`` (toy HGCAL) stage-1 parquet files.

The files are one record per event, flat (no sub-records) but with several
parallel ragged collections:

``rechit_*``
    One entry per *fired sensor cell*.  ``rechit_idx`` is the global cell index
    of the toy geometry (monotonic in z, so it also orders the layers) and
    ``rechit_energy`` is the RAW energy deposited in the active silicon - NOT a
    calibrated energy.  The sampling fraction of the toy is ~0.8 %.

``hit_particle_*``
    The (cell, particle) association table: ``hit_particle_sensor_idx`` points
    into the geometry, ``hit_particle_id`` into the ``particles_*`` collection
    and ``hit_particle_deposit`` is that particle's share of the cell energy.
    Summing it per cell reproduces ``rechit_energy``.

``particles_*``
    One entry per simulated particle (only the primaries are kept in the
    single-particle samples).  ``particles_parent_idx`` points back into the
    same collection, ``-1`` marking a primary.

``gun_*``
    Scalar (per event) generator parameters.  Note the unit quirks recorded in
    the file metadata: the gun position is in **m** while everything else is in
    **mm**, and the gun direction is a unit vector divided by 1000.

Geometry, as reconstructed from the data: 40 active layers at fixed z, the
first 28 with a fine 4.55/7.73 mm alternating spacing (the CE-E analogue) and
the last 12 with a coarse 42.74 mm spacing (the CE-H analogue).
"""

import json
import os

import awkward as ak
import numpy as np

# the maskformer helpers already carry the CMS style + savefig used everywhere
from hgcal_eval_common import (  # noqa: F401  (re-exported on purpose)
    CMAP_10, CMS_blue, CMS_gray, CMS_orange, CMS_purple, CMS_red,
    profile, robust_sigma, savefig, setup_style, stat_label,
)

MM_PER_M = 1000.0
#: the toy writes the gun direction as (unit vector)/1000
GUN_DIR_SCALE = 1000.0
#: layers below this index belong to the fine-sampling "CE-E like" section
N_EE_LAYERS = 28

REGIONS = [
    ("EE", "CE-E like (fine)", CMS_blue),
    ("FH", "CE-H like (coarse)", CMS_orange),
]


def exp_label(ax, rlabel="toy HGCAL"):
    """CMS-style corner label that does *not* claim to be CMS."""
    import mplhep as hep
    hep.label.exp_label(exp="ToyCalo", llabel="Simulation", rlabel=rlabel,
                        ax=ax, fontsize=17)


def note(ax, text, loc="upper left", fontsize=15, color="k"):
    """In-axes annotation - ``set_title`` would collide with :func:`exp_label`."""
    xy = dict(**{"upper left": dict(x=0.04, y=0.92, ha="left", va="top"),
                 "upper right": dict(x=0.96, y=0.92, ha="right", va="top"),
                 "lower left": dict(x=0.03, y=0.05, ha="left", va="bottom"),
                 "lower right": dict(x=0.97, y=0.05, ha="right", va="bottom")}[loc])
    ax.text(xy.pop("x"), xy.pop("y"), text, transform=ax.transAxes,
            fontsize=fontsize, color=color, **xy)


def _f(arr):
    """Flatten one ragged field to a contiguous numpy array."""
    return np.asarray(ak.to_numpy(ak.flatten(arr)))


def _offsets(counts):
    off = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    return off


def eta_of(x, y, z):
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    # guard the exact-beamline case
    t = np.tan(0.5 * np.clip(theta, 1e-12, np.pi - 1e-12))
    return -np.log(t)


class ToyCaloData:
    """Flat (event-concatenated) view of one toy-calorimeter parquet file.

    Prefixes follow the collection they come from: ``h_`` = rechit,
    ``a_`` = hit/particle association, ``p_`` = particle, ``g_`` = gun (scalar,
    one per event).  Pointers into other collections are stored as **global**
    indices so that everything downstream is plain numpy.
    """

    def __init__(self, path, max_events=None):
        self.path = path
        df = ak.from_parquet(path)
        if max_events is not None:
            df = df[:max_events]
        self.df = df
        self.n_events = len(df)
        self.units = self._read_units(path)

        # ------------------------------------------------ per-event scalars
        self.event_id = np.asarray(ak.to_numpy(df.event_id))
        self.simtype = np.asarray(ak.to_numpy(df.simtype))
        self.simulator = np.asarray(ak.to_numpy(df.simulator))
        self.gun_id = np.asarray(ak.to_numpy(df.gun_id))
        self.g_pdgid = np.asarray(ak.to_numpy(df.gun_pdgid)).astype(np.int64)
        self.g_energy = np.asarray(ak.to_numpy(df.gun_energy)).astype(np.float64)
        # position is stored in m, bring it to mm like every other length
        self.g_x = np.asarray(ak.to_numpy(df.gun_position_x)) * MM_PER_M
        self.g_y = np.asarray(ak.to_numpy(df.gun_position_y)) * MM_PER_M
        self.g_z = np.asarray(ak.to_numpy(df.gun_position_z)) * MM_PER_M
        # direction is a unit vector / 1000
        self.g_dx = np.asarray(ak.to_numpy(df.gun_direction_x)) * GUN_DIR_SCALE
        self.g_dy = np.asarray(ak.to_numpy(df.gun_direction_y)) * GUN_DIR_SCALE
        self.g_dz = np.asarray(ak.to_numpy(df.gun_direction_z)) * GUN_DIR_SCALE
        self.g_eta = np.asarray(ak.to_numpy(df.gun_direction_eta)).astype(np.float64)
        self.g_phi = np.asarray(ak.to_numpy(df.gun_direction_phi)).astype(np.float64)
        self.g_dir_norm = np.sqrt(self.g_dx ** 2 + self.g_dy ** 2 + self.g_dz ** 2)
        self.g_eta_calc = eta_of(self.g_dx, self.g_dy, self.g_dz)
        self.g_phi_calc = np.mod(np.arctan2(self.g_dy, self.g_dx), 2 * np.pi)
        self.g_r = np.hypot(self.g_x, self.g_y)

        # ------------------------------------------------ rechits
        self.h_count = np.asarray(ak.to_numpy(ak.num(df.rechit_energy, axis=1)))
        self.h_off = _offsets(self.h_count)
        self.h_event = np.repeat(np.arange(self.n_events), self.h_count)
        self.h_e = _f(df.rechit_energy).astype(np.float64)
        self.h_x = _f(df.rechit_x).astype(np.float64)
        self.h_y = _f(df.rechit_y).astype(np.float64)
        self.h_z = _f(df.rechit_z).astype(np.float64)
        self.h_idx = _f(df.rechit_idx).astype(np.int64)
        self.h_pid = _f(df.rechit_stupid_pid).astype(np.int64)
        self.h_r = np.hypot(self.h_x, self.h_y)
        self.h_eta = eta_of(self.h_x, self.h_y, self.h_z)
        self.h_phi = np.mod(np.arctan2(self.h_y, self.h_x), 2 * np.pi)

        # layer index: the toy has a fixed set of active-plane z positions
        self.layer_z = np.unique(np.round(self.h_z, 3))
        self.n_layers = len(self.layer_z)
        self.h_layer = np.searchsorted(self.layer_z, np.round(self.h_z, 3))
        self.h_is_ee = self.h_layer < N_EE_LAYERS

        # cell-index -> layer map, needed to place the particle impact points
        self._build_layer_index_map()

        # ------------------------------------------------ particles
        self.p_count = np.asarray(ak.to_numpy(ak.num(df.particles_pdgid, axis=1)))
        self.p_off = _offsets(self.p_count)
        self.p_event = np.repeat(np.arange(self.n_events), self.p_count)
        self.p_pdgid = _f(df.particles_pdgid).astype(np.int64)
        self.p_parent = _f(df.particles_parent_idx).astype(np.int64)
        self.p_ekin = _f(df.particles_kinetic_energy).astype(np.float64)
        self.p_vx = _f(df.particles_vertex_position_x).astype(np.float64)
        self.p_vy = _f(df.particles_vertex_position_y).astype(np.float64)
        self.p_vz = _f(df.particles_vertex_position_z).astype(np.float64)
        self.p_dx = _f(df.particles_momentum_direction_x).astype(np.float64)
        self.p_dy = _f(df.particles_momentum_direction_y).astype(np.float64)
        self.p_dz = _f(df.particles_momentum_direction_z).astype(np.float64)
        self.p_ix = _f(df.particles_first_active_impact_position_x).astype(np.float64)
        self.p_iy = _f(df.particles_first_active_impact_position_y).astype(np.float64)
        self.p_iz = _f(df.particles_first_active_impact_position_z).astype(np.float64)
        self.p_isensor = _f(df.particles_first_active_impact_sensor_idx).astype(np.int64)
        self.p_ie = _f(df.particles_first_impact_active_energy).astype(np.float64)
        self.p_idx_ = _f(df.particles_first_active_impact_momentum_direction_x).astype(np.float64)
        self.p_idy = _f(df.particles_first_active_impact_momentum_direction_y).astype(np.float64)
        self.p_idz = _f(df.particles_first_active_impact_momentum_direction_z).astype(np.float64)
        self.p_edep_active = _f(df.particles_total_energy_deposited_active).astype(np.float64)
        self.p_edep_all = _f(df.particles_total_energy_deposited_all).astype(np.float64)

        self.p_eta = eta_of(self.p_dx, self.p_dy, self.p_dz)
        self.p_phi = np.mod(np.arctan2(self.p_dy, self.p_dx), 2 * np.pi)
        self.p_vr = np.hypot(self.p_vx, self.p_vy)
        self.p_ir = np.hypot(self.p_ix, self.p_iy)
        self.p_is_primary = self.p_parent < 0
        # distance travelled between the production vertex and the first active plane
        self.p_flight = np.sqrt((self.p_ix - self.p_vx) ** 2 +
                                (self.p_iy - self.p_vy) ** 2 +
                                (self.p_iz - self.p_vz) ** 2)
        # Layer of the first active impact.  The cell index is the reliable
        # handle here: ``first_active_impact_position_z`` is where the particle
        # *enters* the active volume and so sits a few mm upstream of the plane
        # reference z that the rechits carry.  ``layer_idx_bounds`` is validated
        # against the rechits in :meth:`_build_layer_index_map`.
        self.p_ilayer = np.searchsorted(self.layer_idx_bounds, self.p_isensor)
        self.p_impact_dz = self.p_iz - self.layer_z[
            np.clip(self.p_ilayer, 0, self.n_layers - 1)]

        # ------------------------------------------------ hit <-> particle
        self.a_count = np.asarray(ak.to_numpy(ak.num(df.hit_particle_id, axis=1)))
        self.a_off = _offsets(self.a_count)
        self.a_event = np.repeat(np.arange(self.n_events), self.a_count)
        self.a_pid_local = _f(df.hit_particle_id).astype(np.int64)
        self.a_deposit = _f(df.hit_particle_deposit).astype(np.float64)
        self.a_sensor = _f(df.hit_particle_sensor_idx).astype(np.int64)
        # local -> global particle index
        self.a_particle = self.p_off[self.a_event] + self.a_pid_local
        # local -> global rechit index (cells are unique within an event)
        self.a_hit = self._match_sensor_to_rechit()
        ok = self.a_hit >= 0
        self.a_layer = np.full(len(self.a_hit), -1, dtype=np.int64)
        self.a_layer[ok] = self.h_layer[self.a_hit[ok]]

        # per-cell and per-particle sums derived from the association table
        self.h_n_contrib = np.bincount(self.a_hit[ok], minlength=len(self.h_e))
        self.h_sum_deposit = np.bincount(self.a_hit[ok], weights=self.a_deposit[ok],
                                         minlength=len(self.h_e))
        self.p_n_cells = np.bincount(self.a_particle, minlength=len(self.p_ekin))
        self.p_sum_deposit = np.bincount(self.a_particle, weights=self.a_deposit,
                                         minlength=len(self.p_ekin))

        # ------------------------------------------------ per-event sums
        self.ev_sum_e = np.bincount(self.h_event, weights=self.h_e,
                                    minlength=self.n_events)
        self.ev_sum_deposit = np.bincount(self.a_event, weights=self.a_deposit,
                                          minlength=self.n_events)
        self.ev_sampling = self.ev_sum_e / self.g_energy

        # ------------------------------------------------ shower axis
        self._build_shower_axis()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_units(path):
        import pyarrow.parquet as pq
        md = pq.ParquetFile(path).schema_arrow.metadata or {}
        raw = md.get(b"units")
        return json.loads(raw.decode()) if raw else {}

    def _build_layer_index_map(self):
        """Split the global cell-index range into per-layer blocks.

        The toy numbers its cells layer by layer, so the observed rechit index
        ranges never overlap and the midpoints between consecutive layers are
        valid separators.  The result is checked against the layer taken from z.
        """
        lo = np.array([self.h_idx[self.h_layer == L].min() for L in range(self.n_layers)])
        hi = np.array([self.h_idx[self.h_layer == L].max() for L in range(self.n_layers)])
        assert (hi[:-1] < lo[1:]).all(), "cell-index blocks overlap between layers"
        self.layer_idx_bounds = 0.5 * (hi[:-1] + lo[1:])
        check = np.searchsorted(self.layer_idx_bounds, self.h_idx)
        assert np.array_equal(check, self.h_layer), "cell-index -> layer map is not exact"

    def _match_sensor_to_rechit(self):
        """Global rechit index for every association entry (-1 if not found)."""
        out = np.full(len(self.a_sensor), -1, dtype=np.int64)
        for ev in range(self.n_events):
            hs, he = self.h_off[ev], self.h_off[ev + 1]
            as_, ae = self.a_off[ev], self.a_off[ev + 1]
            cells = self.h_idx[hs:he]           # already sorted by construction
            order = np.argsort(cells, kind="stable")
            cs = cells[order]
            want = self.a_sensor[as_:ae]
            pos = np.searchsorted(cs, want)
            pos_c = np.clip(pos, 0, len(cs) - 1)
            hit = cs[pos_c] == want
            out[as_:ae] = np.where(hit, hs + order[pos_c], -1)
        return out

    def _build_shower_axis(self):
        """Per-hit distance to the incident-particle trajectory.

        The gun fires from the front face along the line through the origin, so
        extrapolating the primary's momentum direction to each layer gives the
        shower axis.  Falls back to the gun record if an event has no primary.
        """
        first = np.full(self.n_events, -1, dtype=np.int64)
        prim = np.where(self.p_is_primary)[0]
        # keep the first primary of each event
        ev_of_prim = self.p_event[prim]
        uniq, first_pos = np.unique(ev_of_prim, return_index=True)
        first[uniq] = prim[first_pos]

        ax_x = np.where(first >= 0, self.p_vx[np.clip(first, 0, None)], self.g_x)
        ax_y = np.where(first >= 0, self.p_vy[np.clip(first, 0, None)], self.g_y)
        ax_z = np.where(first >= 0, self.p_vz[np.clip(first, 0, None)], self.g_z)
        dx = np.where(first >= 0, self.p_dx[np.clip(first, 0, None)], self.g_dx)
        dy = np.where(first >= 0, self.p_dy[np.clip(first, 0, None)], self.g_dy)
        dz = np.where(first >= 0, self.p_dz[np.clip(first, 0, None)], self.g_dz)
        self.ev_axis = np.stack([ax_x, ax_y, ax_z, dx, dy, dz])

        t = (self.h_z - ax_z[self.h_event]) / dz[self.h_event]
        self.h_axis_x = ax_x[self.h_event] + dx[self.h_event] * t
        self.h_axis_y = ax_y[self.h_event] + dy[self.h_event] * t
        self.h_dr = np.hypot(self.h_x - self.h_axis_x, self.h_y - self.h_axis_y)

    # ------------------------------------------------------------------ #
    def summary(self):
        return dict(
            file=os.path.basename(self.path),
            n_events=int(self.n_events),
            n_rechits=int(len(self.h_e)),
            n_assoc=int(len(self.a_deposit)),
            n_particles=int(len(self.p_ekin)),
            n_layers=int(self.n_layers),
            gun_id=str(self.gun_id[0]) if self.n_events else "",
            simtype=str(self.simtype[0]) if self.n_events else "",
            simulator=str(self.simulator[0]) if self.n_events else "",
            mean_rechits_per_event=float(self.h_count.mean()),
            mean_sum_rechit_energy=float(self.ev_sum_e.mean()),
            mean_sampling_fraction=float(self.ev_sampling.mean()),
        )


# --------------------------------------------------------------------------- #
# small plotting helpers
# --------------------------------------------------------------------------- #

def hist_stat_label(x, unit="", name=""):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (name + " (empty)").strip()
    lab = (r"N=%d  $\mu$=%.4g%s" "\n" r"med=%.4g%s  RMS=%.3g%s"
           % (len(x), x.mean(), unit, np.median(x), unit, x.std(), unit))
    return (name + "\n" + lab) if name else lab


def logbins(lo, hi, n):
    return np.logspace(np.log10(lo), np.log10(hi), n + 1)


def add_metric(metrics, section, key, value):
    metrics.setdefault(section, {})[key] = value


def describe(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), std=float(x.std()),
                median=float(np.median(x)), min=float(x.min()), max=float(x.max()),
                p01=float(np.percentile(x, 1)), p99=float(np.percentile(x, 99)))
