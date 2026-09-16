#!/usr/bin/env bash
# Run the full MaskFormer evaluation chain.
#
#   ./scripts/run_all.sh [prediction.parquet] [outdir]
#
# Works for both node granularities: the node collection (LayerCluster / RecHitHGC)
# is auto-detected from whichever carries <collection>_pred_match_idx.
#
# Needs an environment with awkward + numpy + matplotlib + mplhep.
# On this machine: ~/.local/micromamba/envs/pocket-coffea/bin/python
set -euo pipefail

INPUT="${1:-data/hgcal_v1_20260818-T193435/epoch=199-val_loss=11.54392__test.parquet}"
OUTDIR="${2:-plots/maskformer_eval}"
PY="${PYTHON:-$HOME/.local/micromamba/envs/pocket-coffea/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTDIR"

echo "### input  : $INPUT"
echo "### outdir : $OUTDIR"
echo "### python : $PY"

"$PY" "$HERE/evaluate_maskformer.py" --input "$INPUT" --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/../$(basename "$OUTDIR")_eval.log" || true

"$PY" "$HERE/energy_investigation.py" --input "$INPUT" --outdir "$OUTDIR/energy" \
    2>&1 | tee "$OUTDIR/../$(basename "$OUTDIR")_energy.log" || true

echo "### done. $(find "$OUTDIR" -name '*.png' | wc -l) plots written under $OUTDIR"
