"""Render the standalone Isaac Lab 104-taxel force vector field."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if "--visualization" not in sys.argv:
    sys.argv.extend(("--visualization", "vector"))

runpy.run_path(
    str(Path(__file__).with_name("render_tactile_presentation.py")),
    run_name="__main__",
)
