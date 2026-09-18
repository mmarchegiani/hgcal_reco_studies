# hgcal_reco_studies
Study of ML-based reconstruction of the HGCAL

## Scripts

`scripts/` holds the plotting/evaluation scripts used for these studies. They are
plain command-line Python scripts (no package to install): each one reads a
parquet file, writes a tree of PNG plots under `--outdir` and dumps the numbers
it computed to a JSON file next to them.

They all need an environment with `awkward`, `pyarrow`, `numpy`, `matplotlib`
and `mplhep`. On this machine that is the `pocket-coffea` micromamba env:

```bash
PY=~/.local/micromamba/envs/pocket-coffea/bin/python
```

### MaskFormer evaluation

Operate on a MaskFormer prediction file (one record per event, with the
`SimCluster_pred_*` fields filled by the Hungarian matcher). The node collection
carrying the prediction (`LayerCluster` or `RecHitHGC`) is auto-detected, and can
be forced with `--node-collection`.

| Script | What it does |
| --- | --- |
| `evaluate_maskformer.py` | Main evaluation: resolution of `e`/`eta`/`sinphi`/`cosphi`/`r` (inclusive and split by true particle class), clustering efficiency vs. energy/eta/type, and supporting diagnostics. Writes `metrics.json` + `summary.txt`. |
| `energy_investigation.py` | Follow-up on the predicted-vs-true cluster energy deficit: response vs. true and vs. predicted energy, log-log fit of the compression, per-event energy budget, truth closure and self-consistency checks. Writes `energy_metrics.json` + `energy_summary.txt`. |
| `check_energy_scaling.py` | One-off check of whether the rechit-derived `min_max_sym`/`sqrt` energy scaling is degenerate when applied to LayerClusters. |
| `hgcal_eval_common.py` | Shared library (data loading, CMS style, profiles, helpers) — not run directly. |

```bash
$PY scripts/evaluate_maskformer.py \
    --input data/hgcal_v1_20260818-T193435/epoch=199-val_loss=11.54392__test.parquet \
    --outdir plots/maskformer_eval

$PY scripts/energy_investigation.py \
    --input data/hgcal_v1_20260818-T193435/epoch=199-val_loss=11.54392__test.parquet \
    --outdir plots/maskformer_eval/energy
```

Or run both in one go with the wrapper (arguments are optional and default to
the file/outdir above):

```bash
./scripts/run_all.sh [prediction.parquet] [outdir]
```

### Toy-calorimeter (`hg_toy_calorimeter`) exploration

Operate on a stage-1 toy-calorimeter parquet file (flat, one record per event,
with the parallel `rechit_*`, `hit_particle_*`, `particles_*` and `gun_*`
collections). Each script covers one collection and writes a `metrics.json`.

| Script | What it does |
| --- | --- |
| `plot_toycalo_rechits.py` | `rechit_*`: raw cell variables (energy, x, y, z, index, pid flag), derived shower observables (layer, eta, phi, radius, distance to the incident axis) and a few event displays (`--n-display`). |
| `plot_toycalo_hit_particle.py` | `hit_particle_*`: the (cell, particle) association table used as clustering truth — raw variables, closure against `rechit_energy`, sharing between particles. |
| `plot_toycalo_particles.py` | `particles_*`: identity/multiplicity, production vertex, direction, first impact on an active plane and deposited energy. |
| `plot_toycalo_gun.py` | `gun_*`: per-event generator parameters, including the unit conventions (position in m, direction divided by 1000) and closure against `particles_*`. |
| `toycalo_common.py` | Shared library (geometry, data loading, helpers) — not run directly. |

These import `toycalo_common` by name, so `scripts/` has to be on the
`PYTHONPATH`:

```bash
PYTHONPATH=scripts $PY scripts/plot_toycalo_rechits.py \
    --input data/photons_100GeV_eta20.parquet \
    --outdir plots/toycalo/rechits
```

The wrapper runs all four and takes care of that:

```bash
./scripts/run_toycalo.sh [input.parquet] [outdir]
```

### Common options

* `--input` — input parquet file (required for the MaskFormer scripts, defaults
  to `data/photons_100GeV_eta20.parquet` for the toy ones).
* `--outdir` — where the plots, `metrics.json` and summaries are written.
* `--max-events N` — read only the first `N` events, useful for a quick test
  (all scripts except `check_energy_scaling.py`).

Both wrappers honour `PYTHON=<interpreter>` to override the default
`pocket-coffea` python.
