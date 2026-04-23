#!/bin/bash
# =============================================================================
# Submit prompt-eval jobs for a seed range.
#
# Usage:
#   bash scripts/run_prompt_seed_sweep.sh surprise adapter 0 9 false 0
#   bash scripts/run_prompt_seed_sweep.sh surprise adapter 0 9 false 0.05
# =============================================================================

set -euo pipefail

TASK="${1:-surprise}"         # direction | surprise
MODE="${2:-adapter}"          # base | adapter
START_SEED="${3:-0}"
END_SEED="${4:-9}"
BALANCE_TEST="${5:-false}"    # true | false
MIN_EPS_MARGIN="${6:-0}"
ADAPTER_OVERRIDE="${7:-}"

echo "Submitting prompt sweep:"
echo "  task:        $TASK"
echo "  mode:        $MODE"
echo "  seeds:       $START_SEED..$END_SEED"
echo "  balance:     $BALANCE_TEST"
echo "  eps margin:  $MIN_EPS_MARGIN"
if [[ -n "$ADAPTER_OVERRIDE" ]]; then
    echo "  adapter:     $ADAPTER_OVERRIDE"
fi

for SEED in $(seq "$START_SEED" "$END_SEED"); do
    sbatch scripts/run_prompt_eval.sh \
        "$TASK" "$MODE" "$SEED" "$BALANCE_TEST" "$MIN_EPS_MARGIN" "$ADAPTER_OVERRIDE"
done
