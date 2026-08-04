# 01 — Quick start (repo organization + simplest RL example)

This page is **family-agnostic** layout guidance, plus a minimal end-to-end RL
example (`finger_reach`). For the grasping pipeline (train / play / deploy), see
[02_grasping.md](02_grasping.md).

## Prerequisites

- NVIDIA GPU + Linux
- [uv](https://docs.astral.sh/uv/)
- Submodule init (robot MJCF/meshes):

```bash
git submodule update --init   # not --recursive
uv sync --extra cu128
```

## Task-family layout

Every RL setup should own the same five directories:

```text
src/rophi_rl_tutorial/
  assets/<name>/         # EntityCfgs (robot, objects, …)
  controllers/<name>/    # Control plants / phase machines / deploy
  tasks/<name>/          # make_task(), register(), rewards, obs helpers
scripts/<name>/          # CLI: train, play, export, …
logs/<name>/             # Run artifacts
```

**Grasping** is the reference family (`*/grasping/`). Shared code that is not task-specific
(e.g. `mjlab_patches.py`) stays at the package root.

## Simplest RL example: finger reach

`finger_reach` is the smallest MDP in this repo: Flexiv arm + RH-5 in **free space**
(no table / object / work-cell enclosure), learn to match a commanded **index-fingertip
linear velocity**. The robot asset lives under `assets/finger_reach/` (does not import
grasping).

| Piece | What |
|---|---|
| Scene | Rizon 4 + RH-5 only |
| Obs | arm `q` (7) + arm `qd` (7) + tip target velocity (3) |
| Action | arm joint velocity (7), raw ∈ `[-1, 1]` → scaled by `max_joint_vel` (default **0.5 rad/s**); tracked with PD + gravity FF |
| Command | index-tip velocity ∈ `[-0.2, 0.2]^3` m/s, resampled on reset |
| Reward | `exp(-‖v_tip − v_cmd‖ / σ)` − self-collision indicator |

Train:

```bash
uv run python scripts/finger_reach/train.py --num-envs 1024 --iterations 500
```

Play a checkpoint:

```bash
uv run python scripts/finger_reach/play.py
# or:  --run-dir logs/finger_reach/reach
```

Pick ``model_*.pt`` from the **Checkpoints** tab in the viser GUI (defaults to the
latest). The viewer draws the **index tip** (cyan), **commanded tip velocity**
(green arrow), and **measured tip velocity** (orange arrow), plus live speed /
error readouts.

Export + standalone MuJoCo deploy (tip **position** target → velocity field → policy):

```bash
uv run python scripts/finger_reach/export_policy.py
uv run python scripts/finger_reach/run.py --policy logs/finger_reach/reach/policy.pt
```

Keys in `run.py`: `v` CONTROL, `d` DEFAULT, `u/o j/l i/k` move tip target, `r` snap
target to tip. Overlays: tip target (green sphere), cmd vel (green arrow), measured
tip vel (orange arrow).

Task id: `RoPhi-Finger-Reach`. Source of truth:
`src/rophi_rl_tutorial/tasks/finger_reach/mjlab_task.py`.

Each train run writes ``params.json`` into ``--log-dir`` (CLI flags, full env MDP,
PPO config, and runtime obs/action shapes) so you can reproduce or inspect a
checkpoint later.

## Adding a new RL family

1. Create the five directories with an `__init__.py` under each package path.
2. Implement `tasks/<name>/…` with a `make_task()` (or equivalent) that builds a
   `ManagerBasedRlEnvCfg` and a `register()` that calls mjlab’s `register_mjlab_task`.
3. Put robots/objects in `assets/<name>/`.
4. Put control / deploy code in `controllers/<name>/`.
5. Add scripts under `scripts/<name>/` that call `register()` before `load_env_cfg`.
6. Default `--log-dir` / checkpoints under `logs/<name>/`.

mjlab wires the MDP through managers on `ManagerBasedRlEnvCfg`:

| Manager | Typical contents |
|---|---|
| `scene` | Robot + objects + sensors |
| `actions` | Action terms (policy → plant) |
| `observations` | Actor / critic groups |
| `rewards` | Weighted reward terms |
| `terminations` | Success / timeout / failure |
| `events` | Resets, disturbances |

PPO / runner config is an `RslRlOnPolicyRunnerCfg` registered with the task.

## Where to look next

| Goal | Doc |
|---|---|
| Run the simplest finger-reach example | this page § Simplest RL example |
| Run grasping apple end-to-end | [02_grasping.md](02_grasping.md) § Quick start |
| Change grasp obs / actions / rewards | [02_grasping.md](02_grasping.md) § Extending the grasp MDP |
| Understand grasp control (impedance, hand gate) | [02_grasping.md](02_grasping.md) § Control stack |
