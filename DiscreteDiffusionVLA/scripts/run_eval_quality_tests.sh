#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${PYTHONPATH:-.}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "Running eval/quality test suite (excluding perf)..."
pytest -q -m "not perf"

# Optional heavy sanity check (requires model + dataset args)
if [[ "${RUN_DFM_SANITY:-0}" == "1" ]]; then
  if [[ -z "${DFM_SANITY_ARGS:-}" ]]; then
    echo "RUN_DFM_SANITY=1 set but DFM_SANITY_ARGS is empty. Provide args for scripts/dfm_sanity_check.py"
    exit 2
  fi
  python scripts/dfm_sanity_check.py ${DFM_SANITY_ARGS}
fi
