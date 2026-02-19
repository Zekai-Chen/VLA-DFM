import os
import sys

import torch

# Ensure repo root is on sys.path for local imports (e.g., prismatic)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def pytest_configure():
    # Keep tests deterministic on CPU.
    torch.manual_seed(0)
