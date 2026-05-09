"""Kaggle entrypoint for Gemma 4 Trust & Safety research experiments.

Run from a Kaggle notebook cell:

    !python notebooks/kaggle_runner.py \
        --input /kaggle/input/agentic-eval \
        --output /kaggle/working/results.jsonl
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from acpa_gemma.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
