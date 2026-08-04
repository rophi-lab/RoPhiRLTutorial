# RoPhiRLTutorial

GPU RL training on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp, Isaac Lab–style
managers). Task families are split under `assets/`, `controllers/`, `tasks/`, `scripts/`, and
`logs/` so new RL setups can be added without mixing with existing ones.

Families today: **finger_reach** (simplest free-space reach) and **grasping** (Flexiv Rizon 4 +
Robotis RH-5) — see the tutorials below.

## Setup

NVIDIA GPU + Linux. Uses [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:rophi-lab/RoPhiRLTutorial.git
cd RoPhiRLTutorial
git submodule update --init   # not --recursive
uv sync --extra cu128
```

Skip `--recursive`: RoPhiTutorial’s nested `FoundationPose` / `sam2` gitlinks have no URLs in
`.gitmodules`. This repo only needs the top-level checkout (robot MJCF + meshes).

## Tutorials

| Doc | Contents |
|---|---|
| [tutorial/01_quick_start.md](tutorial/01_quick_start.md) | Repo layout, simplest finger-reach example, how to add a family |
| [tutorial/02_grasping.md](tutorial/02_grasping.md) | Grasping quick start, control stack, extending the grasp MDP |

## Why a separate repo

| | [RoPhiTutorial](https://github.com/rophi-lab/RoPhiTutorial) | **RoPhiRLTutorial** |
|---|---|---|
| Role | Hardware / tutorial stack | Batch GPU training |
| `mujoco` | Hardware-oriented pin | `3.10.x` (`mujoco-warp`) |
| Runtime | CPU, latency-bound | CUDA, throughput-bound |

Pins conflict, so the stacks do not share a Python env. The handoff is a **checkpoint file**:
this repo trains and exports; RoPhiTutorial loads the policy. Geometry is shared via the
`third_party/rophi-tutorial` submodule.

## Layout

Each RL setup is a **named family**:

```text
src/rophi_rl_tutorial/
  assets/<family>/        robot / object entity configs
  controllers/<family>/   plants, phase machines, deploy wrappers
  tasks/<family>/         mjlab env, MDP terms, register()
  mjlab_patches.py        shared mjlab fixes
scripts/<family>/         fetch, train, play, distill, export, run
logs/<family>/            training / deploy artifacts (gitignored except .gitkeep)
third_party/              RoPhiTutorial submodule (MJCF + meshes)
tutorial/                 documentation
```

Start with `*/finger_reach/` for the minimal MDP; grasping is under `*/grasping/`.

## Provenance

Ported from [`reflexive-rl`](https://github.com/Alex-Alexiev/reflexive-rl); reward/control from
the `brl_arm_robotis_5F_hand` extension in
[`IsaacLab.brl`](https://github.com/mit-biomimetics/IsaacLab.brl).
