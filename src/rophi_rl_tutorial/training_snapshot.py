"""Save a reproducible snapshot of a training run into its log directory.

Writes ``params.json`` next to checkpoints so later play / resume / debugging
know the CLI flags, env MDP, and PPO config that produced the run.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def to_jsonable(obj: Any, *, _depth: int = 0, _max_depth: int = 12) -> Any:
    """Best-effort conversion of nested cfgs / tensors / callables to JSON data."""
    if _depth > _max_depth:
        return repr(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, _depth=_depth + 1, _max_depth=_max_depth) for v in obj]
    if isinstance(obj, dict):
        return {
            str(k): to_jsonable(v, _depth=_depth + 1, _max_depth=_max_depth)
            for k, v in obj.items()
        }
    if is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for f in fields(obj):
            try:
                out[f.name] = to_jsonable(
                    getattr(obj, f.name), _depth=_depth + 1, _max_depth=_max_depth
                )
            except Exception as exc:  # noqa: BLE001 — never fail the whole dump
                out[f.name] = f"<unserializable: {exc}>"
        return out
    # Callables / classes before catching torch (modules can be callable).
    if isinstance(obj, type) or callable(obj):
        mod = getattr(obj, "__module__", "?")
        name = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
        return f"{mod}.{name}"
    try:
        import numpy as np
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:  # noqa: BLE001
        pass
    return repr(obj)


def _runtime_info(env: Any) -> dict[str, Any]:
    """Capture shapes / timing from a live env (wrapper-aware)."""
    base = env
    # Unwrap common rsl_rl / video wrappers to the ManagerBasedRlEnv.
    for _ in range(4):
        inner = getattr(base, "unwrapped", None) or getattr(base, "env", None)
        if inner is None or inner is base:
            break
        base = inner

    info: dict[str, Any] = {}
    for key in ("num_envs", "device", "step_dt", "physics_dt", "max_episode_length"):
        if hasattr(base, key):
            val = getattr(base, key)
            info[key] = str(val) if key == "device" else to_jsonable(val)

    obs_space = getattr(env, "observation_space", None) or getattr(base, "observation_space", None)
    act_space = getattr(env, "action_space", None) or getattr(base, "action_space", None)
    if obs_space is not None:
        try:
            info["observation_space"] = {
                k: list(v.shape) for k, v in obs_space.spaces.items()
            }
        except Exception:  # noqa: BLE001
            info["observation_space"] = repr(obs_space)
    if act_space is not None:
        try:
            info["action_dim"] = int(act_space.shape[-1])
        except Exception:  # noqa: BLE001
            info["action_space"] = repr(act_space)
    return info


def save_training_snapshot(
    log_dir: str | Path,
    *,
    task: str,
    cli_args: Any,
    env_cfg: Any,
    rl_cfg: Any,
    env: Any | None = None,
    extra: dict[str, Any] | None = None,
    filename: str = "params.json",
) -> Path:
    """Write ``params.json`` under ``log_dir`` and return its path."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    cli = vars(cli_args) if hasattr(cli_args, "__dict__") else dict(cli_args)
    snapshot: dict[str, Any] = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "cli": to_jsonable(cli),
        "env": to_jsonable(env_cfg),
        "rl": to_jsonable(rl_cfg),
    }
    if env is not None:
        snapshot["runtime"] = _runtime_info(env)
    if extra:
        snapshot["extra"] = to_jsonable(extra)

    out = log_path / filename
    out.write_text(json.dumps(snapshot, indent=2, sort_keys=False) + "\n")
    print(f"[train] wrote training snapshot → {out}")
    return out
