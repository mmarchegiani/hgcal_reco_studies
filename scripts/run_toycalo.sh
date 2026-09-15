#!/usr/bin/env bash
# Produce the full set of toy-calorimeter (hg_toy_calorimeter) exploration plots.
#
#   ./scripts/run_toycalo.sh [input.parquet] [outdir]
#
# Needs an environment with awkward + pyarrow + numpy + matplotlib + mplhep.
# On this machine: ~/.local/micromamba/envs/pocket-coffea/bin/python
set -euo pipefail

INPUT="${1:-data/photons_100GeV_eta20.parquet}"
OUTDIR="${2:-plots/toycalo}"
PY="${PYTHON:-$HOME/.local/micromamba/envs/pocket-coffea/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

echo "### input  : $INPUT"
echo "### outdir : $OUTDIR"
echo "### python : $PY"

mkdir -p "$OUTDIR"
"$PY" "$HERE/plot_toycalo_rechits.py"      --input "$INPUT" --outdir "$OUTDIR/rechits"
"$PY" "$HERE/plot_toycalo_hit_particle.py" --input "$INPUT" --outdir "$OUTDIR/hit_particle"
"$PY" "$HERE/plot_toycalo_particles.py"    --input "$INPUT" --outdir "$OUTDIR/particles"
"$PY" "$HERE/plot_toycalo_gun.py"          --input "$INPUT" --outdir "$OUTDIR/gun_parameters"

echo "### done. $(find "$OUTDIR" -name '*.png' | wc -l) plots written under $OUTDIR"
