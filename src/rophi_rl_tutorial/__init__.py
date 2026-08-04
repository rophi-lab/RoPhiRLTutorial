"""rophi-rl-tutorial: GPU RL training on mjlab (task families under assets/controllers/tasks/scripts)."""

import os
from pathlib import Path

ROPHI_RL_SRC_PATH = Path(__file__).parent
"""Path to the installed ``rophi_rl_tutorial`` package."""

ROPHI_RL_ROOT_PATH = Path(os.environ.get("ROPHI_RL_ROOT", ROPHI_RL_SRC_PATH.parents[1]))
"""Repo root. Override with ``ROPHI_RL_ROOT`` for non-editable installs.

Robot geometry lives in the ``third_party/rophi-tutorial`` submodule rather than
inside the package, so asset paths resolve relative to the repo root, not the package.
An editable install (``uv sync``) makes ``parents[1]`` the repo root.
"""

ROPHI_TUTORIAL_PATH = ROPHI_RL_ROOT_PATH / "third_party" / "rophi-tutorial"
"""Root of the RoPhiTutorial submodule (MJCF + meshes)."""

__all__ = [
    "ROPHI_TUTORIAL_PATH",
    "ROPHI_RL_ROOT_PATH",
    "ROPHI_RL_SRC_PATH",
]
