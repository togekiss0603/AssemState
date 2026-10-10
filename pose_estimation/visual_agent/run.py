#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pose_estimation.visual_agent.cli import main


if __name__ == "__main__":
    main()
