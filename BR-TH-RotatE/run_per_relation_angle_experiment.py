from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be set before CUDA is initialized when deterministic CUDA algorithms are used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from supplementary.per_relation_runner import main

    raise SystemExit(main())
