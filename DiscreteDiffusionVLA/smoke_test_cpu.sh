#!/bin/bash
set -euo pipefail

# Simple CPU-only smoke test for DFM utilities.
# This avoids GPU requirements and heavy model deps.

cd /Users/ali/dev/VLA-DFM/DiscreteDiffusionVLA

export PYTHONPATH=.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

pytest -q tests/test_dfm_schedule.py tests/test_dfm_decode.py
