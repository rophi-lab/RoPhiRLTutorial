# 02 — Grasping

Reflexive grasping on **Flexiv Rizon 4 + Robotis RH-5** (mjlab / MuJoCo Warp). Worked example:
**`RoPhi-Grasp-YCB-All`** (full YCB set via per-world mesh variants). Single-object
(`RoPhi-Grasp-Apple`) and the EASY subset (`RoPhi-Grasp-YCB`) reuse the same stack.

Repo layout and how to add a *non-grasping* family: [01_quick_start.md](01_quick_start.md).

| Concern | Path |
|---|---|
| Task factory + registration | `src/rophi_rl_tutorial/tasks/grasping/mjlab_task.py` |
| Reward / contact helpers | `src/rophi_rl_tutorial/tasks/grasping/mdp/terms.py` |
| Phase + impedance / velocity | `src/rophi_rl_tutorial/controllers/grasping/batched.py` |
| Robot / objects / hand PCA | `src/rophi_rl_tutorial/assets/grasping/` |
| CLI | `scripts/grasping/` |
| Logs | `logs/grasping/` |

The quick-start policy uses **`--hand-mode pca`** (action dim 6: grasp score + 5 PCA latents).
Pass `--arm-residual` on train / distill / play / export together for the 12-D residual variant.

---

## Quick start

```bash
# 1. Download + prepare the full YCB set (all objects in YCB_OBJECT_LIST)
uv run python scripts/grasping/fetch_ycb.py

# 2. Train teacher on all YCB objects (PCA hand; per-world mesh variants)
uv run python scripts/grasping/train_grasp.py \
  --task RoPhi-Grasp-YCB-All \
  --hand-mode pca \
  --arm-residual \
  --log-dir logs/grasping/ycb_all \
  --num-envs 2048

# 3. Visualize teacher (second terminal; prefer another GPU)
#    Defaults to --num-envs 1. GUI: env idx + obj idx (hot-swap mesh without OOM).
uv run python scripts/grasping/play_grasp.py \
  --task RoPhi-Grasp-YCB-All \
  --hand-mode pca \
  --arm-residual \
  --run-dir logs/grasping/ycb_all \
  --device cuda:0 \
  --port 8080

# 4. Distill student (reduced obs → same actions)
uv run python scripts/grasping/distill_grasp.py \
  --task RoPhi-Grasp-YCB-All \
  --teacher-run logs/grasping/ycb_all \
  --hand-mode pca \
  --arm-residual \
  --student-obs fingertip \
  --log-dir logs/grasping/ycb_all_distill \
  --device cuda:0

# 5. Export TorchScript for deployment
uv run python scripts/grasping/export_policy.py \
  --task RoPhi-Grasp-YCB-All \
  --run-dir logs/grasping/ycb_all_distill \
  --hand-mode pca \
  --arm-residual \
  --student-obs fingertip \
  --rnn none \
  --out logs/grasping/ycb_all_deploy/policy.pt

# 6. Run exported policy in MuJoCo GUI (Ctrl+drag to perturb the object)
uv run python scripts/grasping/run.py \
  --task RoPhi-Grasp-YCB-All \
  --policy logs/grasping/ycb_all_deploy/policy.pt
```

Play: open http://localhost:8080.

* **Checkpoints** tab → **refresh** / **latest** as `model_*.pt` appears.
* **Env / object** folder → **env idx** (which parallel world to view) and **obj idx**
  (hot-swap that world's YCB mesh). Default `--num-envs 1` avoids OOM; you can still
  browse the full catalogue via **obj idx**. Optional: `--object-idx N --env-idx M`.
* Mirror train flags on play or the checkpoint will not load.

```bash
uv run tensorboard --logdir logs/grasping/ycb_all --port 6006
```

---

## Assets

Robot MJCF/meshes live in the `third_party/rophi-tutorial` submodule (not vendored in the package):

- `assets/scene/flexiv_arm/Rizon4_hand_robot.xml` — 27 DoF (7 arm + 20 hand)
- Arm / hand mesh trees under `Rizon4/` and `robotis5f/`

The shared MJCF has no actuators or sites (`nu=0`, `nsite=0`). mjlab adds those in
`assets/grasping/rizon4_hand.py` so the hardware model stays untouched.

**YCB objects.** `scripts/grasping/fetch_ycb.py` pulls meshes + URDF from
[kwonathan/ycb_urdfs](https://github.com/kwonathan/ycb_urdfs), renames to `v0xx_*`, then
canonicalises / rescales into `datasets/ycb/`. With no flags it prepares the full
`YCB_OBJECT_LIST` (used by `RoPhi-Grasp-YCB-All`). Examples:

```bash
uv run python scripts/grasping/fetch_ycb.py                 # full set
uv run python scripts/grasping/fetch_ycb.py --easy          # EASY subset (RoPhi-Grasp-YCB)
uv run python scripts/grasping/fetch_ycb.py --objects v013_apple
```

Per object (e.g. apple):

```text
datasets/ycb/v013_apple/google_16k/norm_textured.obj
datasets/ycb/v013_apple/google_16k/norm_textured_vhacd.obj
datasets/ycb/v013_apple.urdf
```

Override the output root with `--out`, or set `ROPHI_RL_YCB_ROOT`.

---

## Policy I/O (quick-start flags)

With `--hand-mode pca` (no `--arm-residual`):

| | Dim | Contents |
|---|---|---|
| Teacher actor obs | **59** | contacts(16) + palm→object(3) + hand_joint_pos(20) + hand_tracking(20) |
| Teacher critic obs | **66** | actor + object_pos(3) + object_lin_vel(3) + object_height(1) |
| Action | **6** | grasp_score(1) + PCA latents(5) |
| Student obs (`fingertip`) | **45** | fingertip contacts(5) + hand_joint_pos(20) + hand_tracking(20) |

Add `--arm-residual` on train / distill / play / export together for action dim **12**
(grasp + 6 palm residual + 5 PCA). Distill/export must mirror the teacher’s flags or weight
loading fails.

---

## Control stack

Train, play, and deploy all drive the robot through one mjlab action term:
`GraspAction` in `tasks/grasping/mjlab_task.py`. The policy never writes torques directly.
It outputs a low-dimensional action; `GraspAction` turns that into an arm Cartesian setpoint
and a hand joint target, then a plant applies effort / position commands at physics rate.

```text
policy action  ──►  GraspAction.process_actions  (control rate, ~25 Hz)
                         │
                         ├─ squash grasp_score → [0,1]
                         ├─ decode hand (PCA or joints) → hand_target
                         ├─ optional palm residual
                         └─ BatchedGraspPhase.update → (target_pos, target_quat)
                              └─ impedance.set_target(...)
                              └─ optional hand_gate → force open pose

                   GraspAction.apply_actions     (physics rate, 500 Hz)
                         │
                         ├─ arm: impedance  OR  velocity field + J⁺ + joint PD
                         └─ hand: VF toward hand_target → joint position targets
```

Defaults: physics `dt = 0.002` s (**500 Hz**), `decimation = 20` → policy / control step
**25 Hz**.

### Action layout

| Mode | Dims | Layout |
|---|---|---|
| PCA (quick start) | 6 | `[grasp_score, z₀…z₄]` |
| PCA + `--arm-residual` | 12 | `[grasp_score, Δxyz(3), Δrpy(3), z₀…z₄]` |
| Joints | 21 | `[grasp_score, q₀…q₁₉]` |

`grasp_score` and hand channels are mapped through a sigmoid (default) into `[0, 1]`.
Residuals are *not* sigmoided: clamped to `[-1, 1]` then scaled
(`arm_residual_scale ≈ 3 cm`, `arm_residual_rot_scale ≈ 5°`).

### Phase machine (`BatchedGraspPhase`)

Two latched phases, driven only by `grasp_score` (Schmitt trigger):

| Phase | Enter when | Arm setpoint |
|---|---|---|
| **REACH** | start; or score ≤ `score_low` (0.45) while LIFT | Object + pre-grasp offset (and attitude); optional residual / DR |
| **LIFT** | score ≥ `score_high` (0.55) while REACH | Fixed home retreat pose (or keyboard override in `run.py`) |

The impedance / velocity plant always tracks a **leashed** setpoint
(`max_setpoint_radius = 0.15` m): the commanded Cartesian target is never more than that
distance ahead of the live palm (“carrot on a stick”), so large REACH→LIFT jumps do not
slam the arm.

REACH pose regimes (selected by `phase.reach_randomize`):

- **Training default (`reach_randomize=True`)**: each episode samples a world-frame offset and
  RPY around the object (domain randomisation; arm-agnostic hand).
- **`--arm-residual`**: randomisation off; fixed tuned `pre_grasp_offset` /
  `reach_rpy_fixed`, with the policy’s bounded residual composed on top.

Configured in `controllers/grasping/batched.py` (`BatchedGraspPhaseCfg`) and wired from
`GraspActionCfg.phase` in `mjlab_task.py`.

### Arm plant: impedance vs `--arm-velocity`

Both plants chase the **same** phase target stored on `BatchedArmImpedance`
(`target_pos`, `target_quat`). Only the tracking law changes (`GraspActionCfg.arm_control`).

**Impedance (default, training)** — operational-space spring/damper:

- Cartesian force/moment from pose error → `τ = Jᵀ [F; M]` + joint damping + gravity
- Soft “pull” toward the target; orientation stiffness is lower than position

**Velocity (`--arm-velocity`)** — sim-to-real style:

1. Cartesian velocity field toward the target (linear **and** angular), with quadratic
   slowdown near the goal (`vf_lin_speed=1.0` m/s, `vf_ang_speed=2.0` rad/s by default)
2. Damped Jacobian pseudoinverse: `qd = Jᵀ (J Jᵀ + λ² I)⁻¹ twist`
3. Joint PD + gravity tracking `q_des = q + qd·dt`

Use impedance for the stock YCB pipeline. Use `--arm-velocity` on train/play/run together
when co-adapting the hand policy to a velocity-field arm.

### Hand plant and `--hand-gate`

The hand is always a **joint velocity field** toward `hand_target`, integrated at physics
rate (`hand_v_max`, `hand_slowdown_distance`), then sent as position targets.

**`--hand-gate` off (default):** `hand_target` is whatever the policy decoded. Fingers follow
the policy even while the palm is still far from the object (they may close during approach).

**`--hand-gate` on:** the arm’s proximity latch gates the hand.

| Flag on `BatchedGraspPhase` | Default | Role |
|---|---|---|
| `engage_radius` | **0.10 m** | Palm–object distance below this → `engaged=True` → policy may close |
| `release_radius` | **0.15 m** | Distance above this → `engaged=False` → force open pose |

Between the two radii the latch holds (Schmitt trigger) so the gate does not chatter.
Engagement typically stays true through LIFT (object rides with the palm) and clears if the
object is dropped.

Where to tune:

| Place | How |
|---|---|
| Code default | `BatchedGraspPhaseCfg` in `controllers/grasping/batched.py` |
| Deploy / play | `--hand-gate --engage-radius … --release-radius …` |
| Viz | MuJoCo debug spheres: green = engage, cyan = release; magenta = arm setpoint |

Keep `release_radius > engage_radius`. Without `--hand-gate`, changing the radii has no
effect on finger motion (they still affect the `engaged` flag / overlay only).

### Deploy (`scripts/grasping/run.py`)

Export (`scripts/grasping/export_policy.py`) writes a TorchScript **hand policy**
(`logs/grasping/apple_deploy/policy.pt`): student obs → same action vector as training.

`GraspController` (`controllers/grasping/deploy.py`) does **not** reimplement the arm. In CONTROL
mode it:

1. Runs `HandPolicy.act(student_obs)` at the env control step
2. Calls the original `GraspAction.process_actions(raw_action)`
3. Calls the original `GraspAction.apply_actions()`

So deploy is the same training plant (impedance by default, or `--arm-velocity`), plus a
MuJoCo native GUI for reset / home / lift-target nudges.

```bash
uv run python scripts/grasping/run.py --policy logs/grasping/apple_deploy/policy.pt

uv run python scripts/grasping/run.py --policy logs/grasping/apple_deploy/policy.pt \
  --arm-velocity --hand-gate --engage-radius 0.12 --release-radius 0.18
```

Keys: `r` apple, `d` home, `v` control, `u/o j/l i/k` lift target, Enter reset, Space pause,
Ctrl+drag perturb. Mirror export flags (`--hand-mode`, `--arm-residual`, `--student-obs`).

---

## Training & monitoring

| Flag | Effect |
|---|---|
| `--hand-mode pca\|joints` | Hand action space (quick start: `pca`) |
| `--arm-residual` | 6-D palm residual (off in quick start; pass on train+distill+play together) |
| `--arm-velocity` | Velocity-field arm plant instead of impedance |
| `--hand-gate` | Open hand until palm within `engage_radius` |
| `--rnn gru\|lstm` | Recurrent actor/critic |
| `--blind` / `--limited-obs` | Reduced actor obs |
| `--video` | MP4s under `logs/grasping/ycb_all/videos/train/` |
| `--logger wandb` | Weights & Biases (`--wandb-project rophi-rl-tutorial`) |

Training does not open a 3-D viewer. Use play (default **one env**; pick objects in the GUI):

```bash
uv run python scripts/grasping/play_grasp.py \
  --task RoPhi-Grasp-YCB-All \
  --hand-mode pca \
  --run-dir logs/grasping/ycb_all \
  --object-idx 0 \
  --env-idx 0 \
  --port 8080

# Optional: several parallel worlds (more VRAM)
#   --num-envs 8
```

In the viser **Env / object** folder: **env idx** selects the viewed world; **obj idx**
hot-swaps that world's mesh so you can browse the full YCB set without spawning one
env per object.

---

## Extending the grasp MDP

### New environment / task

`make_task(...)` in `mjlab_task.py` builds a full `ManagerBasedRlEnvCfg` (scene, managers,
`decimation=20`, `episode_length_s=5.0`). `register()` calls mjlab’s `register_mjlab_task`.

Existing ids: `RoPhi-Grasp-Apple`, `RoPhi-Grasp-YCB`, `RoPhi-Grasp-YCB-All`,
`RoPhi-Grasp-Primitives`, and Objaverse / GSO when assets exist.

Checklist for a new object task:

1. Object cfg under `assets/grasping/` (`ycb.py`, `objaverse.py`, …).
2. Fetch data if needed (`scripts/grasping/fetch_ycb.py …`).
3. Register in `register()`:

```python
register_mjlab_task(
    task_id="RoPhi-Grasp-MyObject",
    env_cfg=make_task("my_object", num_envs=4096, source="ycb"),
    play_env_cfg=make_task("my_object", num_envs=16, source="ycb"),
    rl_cfg=grasp_ppo_cfg(),
)
```

4. Train: `uv run python scripts/grasping/train_grasp.py --task RoPhi-Grasp-MyObject --hand-mode pca`

### Observation space

Defaults in `make_task()`:

| Group | Terms | ≈ dim |
|---|---|---|
| **actor** | contacts, palm_to_object, hand_joint_pos, hand_tracking | 59 |
| **critic** | actor + object_pos, object_lin_vel, object_height | 66 |

Student / reduced actor (scripts, not default `make_task`):

| Mechanism | Effect |
|---|---|
| `--limited-obs` | Actor → fingertip(5) + joints + tracking |
| `--blind` | Drop palm→object from actor |
| `--history N` | Frame-stack joint / tracking |
| distill / run | Explicit `student` group |

Add a term: write `(env, …) -> Tensor`, wrap in `ObservationTermCfg`, put in `actor_terms` /
`critic_terms` in `make_task` (or mutate `env_cfg.observations` in `train_grasp.py`).

### Action space

Sole term `"grasp"`: `GraspActionCfg` / `GraspAction`.

| Config | Dim |
|---|---|
| `hand_mode="pca"` | 6 |
| `hand_mode="joints"` | 21 |
| + `arm_residual` | +6 |

Knobs: `hand_mode`, `arm_residual`, `arm_control`, `hand_gate`, `phase`, `squash`, VF/PD gains.
Mirror flags across train / distill / play / export / run.

### Rewards

Defaults in `make_task()`:

| Name | Weight | Intent |
|---|---|---|
| `contact` | 0.1 | Weighted inside-hand contact improvement |
| `height` | 100 | Lift above rest height |
| `hold` | 50 | Dense hold near LIFT goal |
| `success` | 5000 | One-shot after ~2 s hold |

Optional CLI: `--grasp-reward`, `--pca-manifold-weight`, `--contact-weight` /
`--height-weight`, `--hold-to-end`.

Add a term: `(env, …) -> (num_envs,)` in `mjlab_task.py`, then
`RewardTermCfg(func=…, weight=…)` in `make_task` (or inject from `train_grasp.py`).

### Other RL params

| Area | Defaults / overrides |
|---|---|
| Terminations | `time_out`, `success`, `instability` |
| Events | robot/object/state resets; optional `--disturbance` |
| Physics `dt` | 0.002 s → `--physics-dt` |
| Decimation | 20 (25 Hz) → `--decimation` |
| Episode length | 5.0 s → `--episode-length` |
| `num_envs` | 4096 → `--num-envs` |
| PPO | `grasp_ppo_cfg()`; `--iterations`, `--rnn`, `--hidden-dims`, … |

No curriculum manager. Closest levers: REACH DR (`phase.reach_randomize`), object catalogue,
disturbances.

### End-to-end for a new object

```text
1. assets/grasping/ycb.py (or other)     → object EntityCfg
2. scripts/grasping/fetch_*.py           → meshes / URDF
3. tasks/grasping/mjlab_task.register()  → task_id + make_task(...)
4. make_task rewards/obs/actions         → shape the MDP
5. scripts/grasping/train_grasp.py       → PPO
6. scripts/grasping/play_grasp.py        → inspect
7. scripts/grasping/distill_grasp.py     → optional student
8. scripts/grasping/export_policy.py     → TorchScript
9. scripts/grasping/run.py               → deploy GUI
```
