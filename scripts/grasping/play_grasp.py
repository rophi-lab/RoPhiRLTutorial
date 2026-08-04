"""Visualise trained grasping checkpoints in a viser browser viewer.

Two modes:

* ``--checkpoint model_200.pt``  -- load one checkpoint.
* ``--run-dir logs/grasp_run4``  -- browse every checkpoint in the run from a dropdown in
  the viewer's Checkpoints tab, switching between them live.

Defaults to **one env** (one object) to avoid GPU OOM on multi-object tasks. The
viewer GUI has **env idx** and **obj idx** controls: change object without
respawning every YCB mesh world. Pass ``--num-envs N`` to run several worlds in
parallel. Runs on cuda:0 by default so it does not contend with a training run
on cuda:1.
"""

import argparse
import os
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago
from rsl_rl.runners import DistillationRunner
from scipy.spatial.transform import Rotation

from rophi_rl_tutorial.tasks.grasping.mjlab_task import (
    fingertip_contact_binary,
    grasp_matrix_metrics,
    net_wrench_components,
    net_wrench_reward,
    register,
)

sys.path.insert(0, str(Path(__file__).parent))
from variant_switch import set_world_variant, variant_names as _variant_names  # noqa: E402

_PLOT_WINDOW = 250  # policy steps kept in the scrolling grasp-score plot (~10 s at 25 Hz)


class GraspViewer(ViserPlayViewer):
    """ViserPlayViewer plus a live grasp-score plot, the latched phase, and the arm target.

    The grasp score drives a *Schmitt trigger*, not a 0.5 threshold: >=0.55 enters LIFT,
    <=0.45 leaves it, and the 0.45-0.55 deadband holds the previous phase. Plotting the score
    together with the resulting phase makes that hysteresis visible -- the score can dip below
    0.5 while the phase stays LIFT, which is correct, not a bug. Two triads on the palm body
    (``/palm_orient_target`` long, ``/palm_orient_actual`` short) show the commanded vs achieved
    attitude, and the ``/impedance_setpoint`` (orange) and ``/final_target`` (green) spheres show
    where the controller is driving the palm, so the phase decision is legible in the 3-D view.
    """

    def setup(self) -> None:
        super().setup()
        # Apply --env-idx after base setup creates ``_scene``.
        pending = getattr(self, "_pending_env_idx", None)
        if pending is not None:
            self._scene.env_idx = int(pending)
            self._pending_env_idx = None
        self._score_t: deque[int] = deque(maxlen=_PLOT_WINDOW)
        self._score_v: deque[float] = deque(maxlen=_PLOT_WINDOW)
        self._phase_v: deque[float] = deque(maxlen=_PLOT_WINDOW)
        # Env / object selectors. env_idx picks which parallel world the camera shows;
        # obj_idx hot-swaps that world's mesh variant (so num_envs=1 can still browse YCB).
        names = getattr(self, "_env_object_names", None)
        self._last_env_idx = self._scene.env_idx
        obj_entity = self.env.unwrapped.scene["object"]
        self._object_catalogue = _variant_names(obj_entity) or (
            list(dict.fromkeys(names)) if names else None
        )
        self._object_text = None
        self._env_slider = None
        self._obj_slider = None
        self._suppress_obj_cb = False
        self._current_obj_idx: int | None = None
        w2v = self.env.unwrapped.sim.world_to_variant.get("object")
        if w2v is not None:
            self._current_obj_idx = int(w2v[self._scene.env_idx].item())
        with self._server.gui.add_folder("Env / object"):
            if self.env.num_envs > 1:
                self._env_slider = self._server.gui.add_slider(
                    "env idx",
                    min=0,
                    max=self.env.num_envs - 1,
                    step=1,
                    initial_value=self._scene.env_idx,
                )

                @self._env_slider.on_update
                def _on_env(_ev) -> None:
                    self._scene.env_idx = int(self._env_slider.value)
                    self._scene.request_update()
                    self._sync_object_controls()

            if self._object_catalogue:
                init_name = (
                    names[self._scene.env_idx]
                    if names and self._scene.env_idx < len(names)
                    else self._object_catalogue[0]
                )
                init_obj = (
                    self._object_catalogue.index(init_name)
                    if init_name in self._object_catalogue
                    else 0
                )
                self._current_obj_idx = init_obj
                self._object_text = self._server.gui.add_text(
                    "object", self._object_catalogue[init_obj], disabled=True
                )
                self._obj_slider = self._server.gui.add_slider(
                    "obj idx",
                    min=0,
                    max=len(self._object_catalogue) - 1,
                    step=1,
                    initial_value=init_obj,
                )

                @self._obj_slider.on_update
                def _on_obj(_ev) -> None:
                    if self._suppress_obj_cb:
                        return
                    self._apply_object_idx(int(self._obj_slider.value))

            elif names:
                self._object_text = self._server.gui.add_text(
                    "object", names[self._scene.env_idx], disabled=True
                )
        with self._server.gui.add_folder("Policy"):
            self._phase_text = self._server.gui.add_text("phase", "REACH", disabled=True)
            self._score_plot = self._server.gui.add_uplot(
                data=(np.array([0.0]), np.array([0.0]), np.array([0.0])),
                series=(
                    viser.uplot.Series(label="step"),
                    viser.uplot.Series(label="grasp score", stroke="#e04040", width=2),
                    viser.uplot.Series(label="phase (0 reach / 1 lift)", stroke="#4060e0", width=2),
                ),
                scales={"y": viser.uplot.Scale(range=(0.0, 1.0))},
                # Real thresholds, not 0.5: enter LIFT at 0.55, leave at 0.45, latched between.
                title="grasp score & phase  (LIFT >=0.55, REACH <=0.45, latched between)",
                height=170,
                aspect=2.0,
            )
        # Free-play: drag the LIFT target so the arm carries the grasped object to an x/y/z point.
        # Sets the phase controller's lift_override_pos, which replaces the fixed home retreat pose
        # in LIFT (the leash makes the arm pursue it smoothly). Persists across resets.
        self._lift_sliders: dict = {}
        if getattr(self, "_free_play", False):
            term = self.env.unwrapped.action_manager.get_term("grasp")
            home = term.home_goal_pos.detach().cpu().numpy()
            ranges = {"x": (0.30, 0.75), "y": (-0.40, 0.40), "z": (0.0, 0.55)}
            with self._server.gui.add_folder("LIFT target (m) — move the grasped object"):
                for i, ax in enumerate(("x", "y", "z")):
                    lo, hi = ranges[ax]
                    self._lift_sliders[ax] = self._server.gui.add_slider(
                        ax, min=lo, max=hi, step=0.005, initial_value=float(min(max(home[i], lo), hi))
                    )
                    self._lift_sliders[ax].on_update(lambda _ev: self._apply_lift_target())
            self._apply_lift_target()
        # Orientation is shown as two co-located triads centred on the palm:
        #   /palm_orient_target  LONG axes  -- the target (flat hold) attitude.
        #   /palm_orient_actual  SHORT axes -- the palm's actual attitude.
        # When the triads align, the arm is holding the commanded orientation; a gap is the
        # tracking error. Orientation comes from the `palm` body -- the frame the impedance law
        # actually regulates (identical to grasp_site's). Note: at the OPEN pose every finger
        # phalanx shares this exact orientation, so the triad can *look* like it is on a finger;
        # close the hand and it stays with the palm while the fingers curl 178 deg away. The
        # triads are drawn at `palm_surface` (centre of the palm) rather than the `palm` body
        # origin (which sits back at the wrist) purely so they sit on the visible palm.
        robot = self.env.unwrapped.scene["robot"]
        self._palm_body_id = robot.find_bodies(["palm"])[0][0]
        self._palm_pos_id = robot.find_bodies(["palm_surface"])[0][0]
        self._palm_target_frame = self._server.scene.add_frame(
            "/palm_orient_target",
            axes_length=0.13,
            axes_radius=0.006,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        self._palm_actual_frame = self._server.scene.add_frame(
            "/palm_orient_actual",
            axes_length=0.08,
            axes_radius=0.004,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        # Position markers (orientation-free): the tracked setpoint (orange, the control point
        # the impedance law drives) and the ultimate goal (green: object in REACH, home in LIFT).
        self._setpoint_marker = self._server.scene.add_icosphere(
            "/impedance_setpoint",
            radius=0.02,
            color=(230, 150, 40),
            position=(0.0, 0.0, 0.0),
        )
        self._final_target_marker = self._server.scene.add_icosphere(
            "/final_target",
            radius=0.025,
            color=(60, 200, 90),
            position=(0.0, 0.0, 0.0),
        )
        # Orientation-tracking error [deg] between the target and actual palm attitude, so the
        # "is it holding orientation?" question has a number, not just overlaid axes.
        self._orient_text = self._server.gui.add_text("palm orient err (deg)", "0", disabled=True)

        # Grasp-wrench panel: the net contact wrench the hand + gravity apply to the object (what
        # the net-wrench reward drives to zero) plus the grasp-matrix quality metrics, for the
        # displayed env. The orange arrow at the object COM is the net *force* imbalance -- it
        # shrinks to nothing as the grasp holds the object in balanced equilibrium. All read 0 until
        # the object is lifted and held (the rewards are gated on a real in-air grasp).
        with self._server.gui.add_folder("Grasp wrench"):
            self._nw_reward_text = self._server.gui.add_text("net-wrench reward", "0", disabled=True)
            self._nw_force_text = self._server.gui.add_text("net force / weight", "0", disabled=True)
            self._nw_torque_text = self._server.gui.add_text("net torque / (w*L)", "0", disabled=True)
            self._qmsv_text = self._server.gui.add_text("Q_MSV (grasp matrix)", "0", disabled=True)
            self._qgii_text = self._server.gui.add_text("Q_GII (isotropy)", "0", disabled=True)
        self._wrench_arrow = self._server.scene.add_arrows(
            "/net_wrench_force",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=(240, 120, 40),
            shaft_radius=0.006,
            head_radius=0.015,
            head_length=0.03,
            visible=False,
        )

        # Arm velocity-control tuning (arm_control="velocity"): live sliders for the Jacobian-
        # pseudoinverse velocity controller so the arm's reach/approach can be tuned against the
        # running hand policy. Each change calls term.set_velocity_params(...) immediately.
        _vterm = self.env.unwrapped.action_manager.get_term("grasp")
        if getattr(_vterm.cfg, "arm_control", "impedance") == "velocity":
            base_kp = tuple(_vterm.cfg.arm_kp)
            base_kd = tuple(_vterm.cfg.arm_kd)
            v = _vterm._vel
            with self._server.gui.add_folder("Arm velocity control"):
                s_vlin = self._server.gui.add_slider("vf_lin_speed", 0.05, 1.2, 0.01, v["vf_lin_speed"])
                s_elin = self._server.gui.add_slider("vf_lin_eps", 0.005, 0.25, 0.005, v["vf_lin_eps"])
                s_vang = self._server.gui.add_slider("vf_ang_speed", 0.2, 5.0, 0.1, v["vf_ang_speed"])
                s_eang = self._server.gui.add_slider("vf_ang_eps", 0.02, 0.8, 0.01, v["vf_ang_eps"])
                s_damp = self._server.gui.add_slider("pinv_damping", 0.001, 0.3, 0.001, v["vel_damping"])
                s_qdl = self._server.gui.add_slider("qd_limit", 0.5, 5.0, 0.1, v["qd_limit"])
                s_kp = self._server.gui.add_slider("kp_scale", 0.2, 3.0, 0.05, 1.0)
                s_kd = self._server.gui.add_slider("kd_scale", 0.2, 5.0, 0.05, 1.0)

            def _apply_vel(_ev=None):
                _vterm.set_velocity_params(
                    vf_lin_speed=s_vlin.value, vf_lin_eps=s_elin.value,
                    vf_ang_speed=s_vang.value, vf_ang_eps=s_eang.value,
                    vel_damping=s_damp.value, qd_limit=s_qdl.value,
                    arm_kp=[k * s_kp.value for k in base_kp],
                    arm_kd=[k * s_kd.value for k in base_kd],
                )

            for _s in (s_vlin, s_elin, s_vang, s_eang, s_damp, s_qdl, s_kp, s_kd):
                _s.on_update(_apply_vel)
            _apply_vel()

        # Manual debug mode: a slider per RAW action dim, all 0 by default, driving every env.
        # Raw = what the policy outputs: sigmoid maps grasp/hand to [0,1]; the residual is
        # clamped to [-1,1] then scaled to +-0.05 m. So all-zero is the neutral action (grasp
        # score 0.5 -> REACH, no residual, hand joints mid-range). Replaces the policy with a
        # reader of these sliders so you can pose the arm and hand by hand.
        if getattr(self, "_manual", False):
            term = self.env.unwrapped.action_manager.get_term("grasp")
            specs = [("Grasp", "grasp_score", -6.0, 6.0)]
            if term._use_residual:
                specs += [("Arm residual", f"residual_{ax}", -1.0, 1.0) for ax in ("x", "y", "z")]
            if term._pca is not None:
                specs += [("Hand", f"pca_z{i}", -6.0, 6.0) for i in range(term._num_hand)]
            else:
                from rophi_rl_tutorial.assets.grasping import HAND_JOINT_NAMES

                specs += [("Hand", n, -6.0, 6.0) for n in HAND_JOINT_NAMES]
            self._action_sliders = [None] * len(specs)
            for group in ("Grasp", "Arm residual", "Hand"):
                with self._server.gui.add_folder(f"Actions: {group}"):
                    for i, (g, lbl, lo, hi) in enumerate(specs):
                        if g == group:
                            self._action_sliders[i] = self._server.gui.add_slider(
                                lbl, min=lo, max=hi, step=0.01, initial_value=0.0
                            )
            with self._server.gui.add_folder("Actions"):
                zero_btn = self._server.gui.add_button("zero all actions")

                @zero_btn.on_click
                def _(_ev) -> None:
                    for s in self._action_sliders:
                        s.value = 0.0

            self.policy = self._manual_policy

        # REACH-pose tuning: six sliders set a SINGLE offset (position + orientation) of the palm
        # relative to the object; the real arm is driven there (hand open) so you see the actual
        # hand at that pose. Move a slider to an extreme you'd accept, click "save bound point";
        # the per-axis min/max envelope over saved points is the domain-randomisation range.
        self._sliders_off: dict = {}
        if getattr(self, "_tune_reach", False):
            term = self.env.unwrapped.action_manager.get_term("grasp")
            self._base_quat = term._home_site_quat.detach().cpu().numpy()  # flat hold attitude
            self._reach_target_frame = self._server.scene.add_frame(
                "/reach_target",
                axes_length=0.10,
                axes_radius=0.006,
                position=(0.0, 0.0, 0.0),
                wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            with self._server.gui.add_folder("REACH offset: position (m)"):
                for ax in ("x", "y", "z"):
                    self._sliders_off[ax] = self._server.gui.add_slider(
                        ax, min=-0.12, max=0.12, step=0.005, initial_value=0.0
                    )
            with self._server.gui.add_folder("REACH offset: orientation (deg)"):
                for ax in ("roll", "pitch", "yaw"):
                    self._sliders_off[ax] = self._server.gui.add_slider(
                        ax, min=-90.0, max=90.0, step=1.0, initial_value=0.0
                    )
            # Single fixed pose readout: click to print the current 6-slider offset in a form
            # ready to paste as the default grasp pose (pre_grasp_offset + reach orientation).
            with self._server.gui.add_folder("REACH fixed pose"):
                self._pose_text = self._server.gui.add_text(
                    "offset", "(click to read)", disabled=True
                )
                pose_btn = self._server.gui.add_button("print current offset")
                pose_btn.on_click(lambda _ev: self._print_offset())
            self._saved_bounds: list[list[float]] = []
            with self._server.gui.add_folder("REACH bounds (for DR)"):
                save_btn = self._server.gui.add_button("save bound point")
                clear_btn = self._server.gui.add_button("clear bounds")
                self._bounds_text = self._server.gui.add_text(
                    "bounds", "(none saved)", disabled=True
                )
                save_btn.on_click(lambda _ev: self._save_bound_point())
                clear_btn.on_click(lambda _ev: self._clear_bounds())
            self.policy = self._tune_policy

    def _apply_lift_target(self) -> None:
        """Push the LIFT-target sliders into the controller's lift_override_pos (broadcasts to all envs)."""
        term = self.env.unwrapped.action_manager.get_term("grasp")
        xyz = [self._lift_sliders[a].value for a in ("x", "y", "z")]
        term.phase.lift_override_pos = torch.tensor(
            xyz, device=self.env.unwrapped.device, dtype=torch.float32
        )

    def _print_offset(self) -> None:
        """Print the current single slider offset, ready to paste as the fixed default pose."""
        s = self._sliders_off
        v = {a: s[a].value for a in ("x", "y", "z", "roll", "pitch", "yaw")}
        txt = (
            f"pos=({v['x']:+.3f}, {v['y']:+.3f}, {v['z']:+.3f}) m  "
            f"rpy=({v['roll']:+.1f}, {v['pitch']:+.1f}, {v['yaw']:+.1f}) deg"
        )
        self._pose_text.value = txt
        print("[tune-reach] fixed pose:", txt)

    def _offset_pose(self):
        """Current slider offset as (pos[3], quat_wxyz[4]); orientation = flat-hold x rpy."""
        s = self._sliders_off
        off = np.array([s["x"].value, s["y"].value, s["z"].value])
        rpy = np.radians([s["roll"].value, s["pitch"].value, s["yaw"].value])
        b = self._base_quat  # wxyz -> scipy xyzw
        q = (
            Rotation.from_quat([b[1], b[2], b[3], b[0]]) * Rotation.from_euler("xyz", rpy)
        ).as_quat()
        return off, np.array([q[3], q[0], q[1], q[2]])  # wxyz

    def _save_bound_point(self) -> None:
        s = self._sliders_off
        self._saved_bounds.append([s[a].value for a in ("x", "y", "z", "roll", "pitch", "yaw")])
        self._update_bounds_text()

    def _clear_bounds(self) -> None:
        self._saved_bounds = []
        self._update_bounds_text()

    def _update_bounds_text(self) -> None:
        if not self._saved_bounds:
            self._bounds_text.value = "(none saved)"
            return
        arr = np.array(self._saved_bounds)
        axes = ("x", "y", "z", "roll", "pitch", "yaw")
        units = ("m", "m", "m", "deg", "deg", "deg")
        lines = [
            f"{a}:[{arr[:, i].min():+.3g},{arr[:, i].max():+.3g}]{u}"
            for i, (a, u) in enumerate(zip(axes, units, strict=True))
        ]
        self._bounds_text.value = f"{len(self._saved_bounds)} pts  " + "  ".join(lines)
        print("[tune-reach] bounds:", "  ".join(lines))

    def _tune_policy(self, _obs=None) -> torch.Tensor:
        """Drive the palm to object + slider offset (hand open) so the real hand shows the pose."""
        term = self.env.unwrapped.action_manager.get_term("grasp")
        dev = self.env.unwrapped.device
        off, q_wxyz = self._offset_pose()
        obj = self.env.unwrapped.scene["object"].data.root_link_pos_w  # (N, 3)
        term._reach_override_pos = obj + torch.tensor(off, device=dev, dtype=torch.float32)
        term._reach_override_quat = (
            torch.tensor(q_wxyz, device=dev, dtype=torch.float32)
            .unsqueeze(0)
            .expand(obj.shape[0], 4)
            .contiguous()
        )
        a = torch.zeros(
            obj.shape[0], self.env.unwrapped.action_manager.total_action_dim, device=dev
        )
        a[:, term._n_pre :] = -5.0  # hand open (dims after grasp [+ residual])
        return a

    def _manual_policy(self, _obs=None) -> torch.Tensor:
        """Actions from the manual sliders, broadcast to every env (debug mode)."""
        dev = self.env.unwrapped.device
        vals = torch.tensor(
            [s.value for s in self._action_sliders], device=dev, dtype=torch.float32
        )
        return vals.unsqueeze(0).expand(self.env.num_envs, -1).contiguous()

    def _sync_object_controls(self) -> None:
        """Keep obj-idx / label in sync with the currently viewed env."""
        names = getattr(self, "_env_object_names", None)
        idx = int(self._scene.env_idx)
        if not self._object_catalogue:
            if self._object_text is not None and names and idx < len(names):
                self._object_text.value = names[idx]
            return
        name = (
            names[idx]
            if names and idx < len(names)
            else self._object_catalogue[0]
        )
        if self._object_text is not None:
            self._object_text.value = name
        if self._obj_slider is not None and name in self._object_catalogue:
            want = self._object_catalogue.index(name)
            self._current_obj_idx = want
            if int(self._obj_slider.value) != want:
                # Avoid re-entering _apply_object_idx via the slider callback.
                self._suppress_obj_cb = True
                try:
                    self._obj_slider.value = want
                finally:
                    self._suppress_obj_cb = False

    def _apply_object_idx(self, variant_idx: int) -> None:
        """Hot-swap the mesh variant on the currently viewed env."""
        if not self._object_catalogue:
            return
        variant_idx = int(variant_idx)
        if self._current_obj_idx is not None and variant_idx == self._current_obj_idx:
            return
        env_idx = int(self._scene.env_idx)
        name = set_world_variant(
            self.env,
            entity_name="object",
            env_idx=env_idx,
            variant_idx=variant_idx,
        )
        self._current_obj_idx = variant_idx
        names = getattr(self, "_env_object_names", None)
        if names is not None and env_idx < len(names):
            names[env_idx] = name
        if self._object_text is not None:
            self._object_text.value = name
        # Re-seat the object so the new mesh isn't left mid-grasp / mid-air.
        try:
            from mjlab.envs.mdp.events import reset_root_state_uniform
            from mjlab.managers.scene_entity_config import SceneEntityCfg

            ids = torch.tensor([env_idx], device=self.env.unwrapped.device)
            reset_root_state_uniform(
                self.env.unwrapped,
                ids,
                pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                velocity_range={},
                asset_cfg=SceneEntityCfg("object"),
            )
            self.env.unwrapped.sim.forward()
        except Exception:
            pass
        # mjlab's auto-rebuild fingerprints only rgba, not geom_dataid — so a
        # live variant swap leaves the old mesh baked into Viser handles.
        self._rebuild_object_visuals()

    def _rebuild_object_visuals(self) -> None:
        """Force Viser to re-bake body meshes from the current geom_dataid."""
        scene = self._scene
        sync = getattr(scene, "_sync_model_fields", None)
        if callable(sync):
            sync(int(scene.env_idx))
        rebuild = getattr(scene, "rebuild_visual_handles", None)
        if callable(rebuild):
            rebuild()
        scene.request_update()

    def sync_env_to_viewer(self) -> None:
        super().sync_env_to_viewer()
        term = self.env.unwrapped.action_manager.get_term("grasp")
        idx = self._scene.env_idx  # the env currently on screen
        if idx != self._last_env_idx:
            # Switched env: the plotted history belongs to the old env, so clear it.
            self._last_env_idx = idx
            self._score_t.clear()
            self._score_v.clear()
            self._phase_v.clear()
            self._sync_object_controls()
        phase = float(term.phase.phase[idx].item())  # 0 reach, 1 lift
        self._score_t.append(self._step_count)
        self._score_v.append(float(term.grasp_score[idx]))
        self._phase_v.append(phase)
        t = np.array(self._score_t, dtype=float)
        self._score_plot.data = (t, np.array(self._score_v), np.array(self._phase_v))
        self._phase_text.value = "LIFT" if phase > 0.5 else "REACH"

        # Position markers: the tracked setpoint (orange) and the ultimate goal (green).
        setp = term.impedance.target_pos[idx].detach().cpu().numpy()
        tgt_quat = term.impedance.target_quat[idx].detach().cpu().numpy()
        self._setpoint_marker.position = tuple(float(v) for v in setp)
        final = term.phase.base_target[idx].detach().cpu().numpy()
        self._final_target_marker.position = tuple(float(v) for v in final)

        # Orientation triads centred on the palm: target vs actual attitude, plus the error
        # angle. Orientation is the `palm` body's (== grasp_site's == the impedance control
        # frame); position is palm_surface's, purely to place the triads on the visible palm.
        robot = self.env.unwrapped.scene["robot"]
        palm_pos = robot.data.body_link_pos_w[idx, self._palm_pos_id].detach().cpu().numpy()
        palm_quat = robot.data.body_link_quat_w[idx, self._palm_body_id].detach().cpu().numpy()
        palm_xyz = tuple(float(v) for v in palm_pos)
        self._palm_target_frame.position = palm_xyz
        self._palm_target_frame.wxyz = tuple(float(v) for v in tgt_quat)
        self._palm_actual_frame.position = palm_xyz
        self._palm_actual_frame.wxyz = tuple(float(v) for v in palm_quat)
        dot = abs(float(np.dot(tgt_quat, palm_quat)))
        self._orient_text.value = f"{np.degrees(2 * np.arccos(min(1.0, dot))):.1f}"

        # Grasp-wrench readouts + net-force arrow for the displayed env. Best-effort: the sensor
        # data is only populated after a physics step, and the reward helpers touch the model
        # arrays, so any transient hiccup should not take down the viewer loop.
        try:
            base = self.env.unwrapped
            nf, _, fres, tres = net_wrench_components(base)
            nwr = net_wrench_reward(base)
            qmsv, _, qgii = grasp_matrix_metrics(base)
            self._nw_reward_text.value = f"{float(nwr[idx]):.3f}"
            self._nw_force_text.value = f"{float(fres[idx]):.3f}"
            self._nw_torque_text.value = f"{float(tres[idx]):.3f}"
            self._qmsv_text.value = f"{float(qmsv[idx]):.3f}"
            self._qgii_text.value = f"{float(qgii[idx]):.3f}"
            # Arrow = net contact-force imbalance at the object COM, length ~ residual (capped),
            # pointing the way the object is being net-pushed. Vanishes at a balanced hold.
            com = base.scene["object"].data.root_com_pos_w[idx].detach().cpu().numpy()
            fvec = nf[idx].detach().cpu().numpy()
            mag = float(np.linalg.norm(fvec))
            resid = float(fres[idx])
            if mag > 1e-6 and resid > 1e-3:
                end = com + (fvec / mag) * min(resid, 2.0) * 0.15
                self._wrench_arrow.points = np.array([[com, end]], dtype=np.float32)
                self._wrench_arrow.visible = True
            else:
                self._wrench_arrow.visible = False
        except Exception:
            pass

        # REACH tuning: place the /reach_target frame at the commanded offset pose for the viewed
        # object; the real palm is driven there by the tune policy so you see the actual hand.
        if getattr(self, "_tune_reach", False):
            obj = (
                self.env.unwrapped.scene["object"].data.root_link_pos_w[idx].detach().cpu().numpy()
            )
            off, q_wxyz = self._offset_pose()
            pos = obj + off
            self._reach_target_frame.position = (float(pos[0]), float(pos[1]), float(pos[2]))
            self._reach_target_frame.wxyz = tuple(float(v) for v in q_wxyz)


def _checkpoints(run_dir: Path) -> list[Path]:
    """Checkpoints in a run directory, sorted by training step."""

    def step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    return sorted(run_dir.glob("model_*.pt"), key=step)


def _resolve_object_names(obj_cfg, num_envs: int) -> list[str] | None:
    """Per-env object name, matching the variant entity's world->variant allocation.

    Returns None for a single-object (non-variant) task. Reproduces the same largest-remainder
    allocation ``VariantEntityCfg`` applies at sim init, so the viewer's labels line up with the
    meshes actually spawned in each env.
    """
    from mjlab.entity.variants import allocate_worlds

    variants = getattr(obj_cfg, "variants", None)
    if not variants:
        return None
    vnames = list(variants.keys())
    assign = getattr(obj_cfg, "assignment", None)
    if assign is None:
        alloc = allocate_worlds((1.0,) * len(vnames), num_envs)
    elif callable(assign):
        alloc = list(assign(num_envs))
    else:  # dict of per-variant weights
        alloc = allocate_worlds(tuple(assign.get(n, 1.0) for n in vnames), num_envs)
    return [vnames[i] for i in alloc]


def main() -> None:
    p = argparse.ArgumentParser()
    # Default to the diverse-object task so the viewer spawns the same YCB set training saw;
    # pass --task RoPhi-Grasp-Apple for the single-object runs.
    p.add_argument("--task", default="RoPhi-Grasp-YCB")
    p.add_argument("--checkpoint", default=None, help="Single checkpoint file.")
    p.add_argument("--run-dir", default=None, help="Directory of checkpoints to browse.")
    p.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Envs to spawn (default 1 = one object, avoids OOM). For a multi-object task, "
        "pass the catalogue size to browse objects with the 'environment' slider.",
    )
    p.add_argument(
        "--object-idx",
        type=int,
        default=0,
        help="Initial object variant index for env 0 (mesh-variant tasks). "
        "Also changeable live via the 'obj idx' slider.",
    )
    p.add_argument(
        "--env-idx",
        type=int,
        default=0,
        help="Initial env index shown in the viewer.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--squash", default="sigmoid", help="'sigmoid' for run 2+, 'clamp' for run 1.")
    p.add_argument(
        "--share",
        action="store_true",
        help="Request a public viser share URL so the viewer is reachable from anywhere (no port "
        "forwarding). Prints 'SHARE URL: <url>' once connected.",
    )
    p.add_argument(
        "--manual",
        action="store_true",
        help="Debug mode: drive the raw actions from GUI sliders (all 0 by default) instead of "
        "a policy. No checkpoint needed; the episode never auto-resets.",
    )
    p.add_argument(
        "--tune-reach",
        action="store_true",
        help="REACH-pose tuning: draw a cloud of candidate REACH palm poses around each object, "
        "with sliders for position/orientation bounds, to pick good domain-randomisation ranges. "
        "No checkpoint needed.",
    )
    # These MUST match the flags the checkpoint was trained with, or the obs/action dims (and
    # model class) won't line up and loading fails.
    p.add_argument("--blind", action="store_true", help="Match a --blind run (no palm->object in actor).")
    p.add_argument("--history", type=int, default=0, help="Match a --history N run (frame-stacked proprioception).")
    p.add_argument("--rnn", choices=("lstm", "gru"), default=None, help="Match a --rnn run (recurrent policy).")
    p.add_argument("--hand-mode", choices=("joints", "pca"), default=None, help="Match the run's hand action space.")
    p.add_argument("--decimation", type=int, default=None, help="Match the run's control decimation.")
    p.add_argument("--physics-dt", type=float, default=None, help="Match the run's physics timestep.")
    p.add_argument("--arm-residual", action="store_true", help="Match a --arm-residual run (6-D palm residual, no REACH randomisation).")
    p.add_argument("--arm-velocity", action="store_true", help="Use the sim-to-real deploy's Jacobian-pseudoinverse velocity arm controller instead of the training impedance controller (same policy, same phase target). Adds live tuning sliders for the velocity params in the viewer.")
    p.add_argument("--hand-pca-project", type=float, default=None, metavar="A", help="Match a --hand-pca-project run: blend the commanded hand pose toward its PCA-manifold projection by A (joints mode). REQUIRED to faithfully replay a run fine-tuned with projection.")
    p.add_argument(
        "--student",
        action="store_true",
        help="Load a distilled STUDENT checkpoint from a distill run. Sets up student+teacher obs "
        "groups and DistillationRunner. Match --hand-mode / --arm-residual / --student-obs to the "
        "distill run.",
    )
    p.add_argument("--limited-obs", action="store_true", help="Match a --limited-obs run: reduce the ACTOR obs to fingertip contacts + hand_joint_pos + hand_tracking (critic keeps full). REQUIRED to load a checkpoint trained with --limited-obs.")
    p.add_argument("--free-play", action="store_true", help="Never auto-reset (run until you press 'Reset Environment'), and add LIFT-target x/y/z sliders to drag the grasped object around.")
    p.add_argument("--student-obs", choices=("fingertip", "proprio"), default="fingertip",
                   help="(with --student) student obs group: 'fingertip' (5 contacts + proprioception, "
                   "45-D) or 'proprio' (hand_joint_pos + hand_tracking only, 40-D). Must match the "
                   "distilled student (e.g. 'proprio' for distill_gru_proprio).")
    p.add_argument("--hand-gate", action="store_true",
                   help="Arm-gates-hand: the hand policy drives the fingers only once the palm is "
                   "within engage-radius of the object; otherwise the fingers hold the default "
                   "pre-grasp pose. Fixes 'the hand just closes while standing still'.")
    p.add_argument("--engage-radius", type=float, default=None,
                   help="(with --hand-gate) palm->object distance [m] below which the hand engages "
                   "the policy (Schmitt trigger; releases above --release-radius).")
    p.add_argument("--release-radius", type=float, default=None,
                   help="(with --hand-gate) palm->object distance [m] above which the hand disengages "
                   "back to the default pose.")
    args = p.parse_args()

    if not args.manual and not args.tune_reach and not args.checkpoint and not args.run_dir:
        p.error("pass --checkpoint or --run-dir (or --manual / --tune-reach)")

    register()
    env_cfg = load_env_cfg(args.task, play=True)
    obj_cfg = env_cfg.scene.entities["object"]
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.actions["grasp"].squash = args.squash    # Reproduce the training-time env config so the checkpoint's obs/action dims (and model)
    # match. Mirrors the same flags in train_grasp.py.
    if args.decimation is not None:
        env_cfg.decimation = args.decimation
    if args.physics_dt is not None:
        env_cfg.sim.mujoco.timestep = args.physics_dt
    if args.hand_mode is not None:
        env_cfg.actions["grasp"].hand_mode = args.hand_mode
    if args.arm_residual:
        env_cfg.actions["grasp"].arm_residual = True
        env_cfg.actions["grasp"].phase.reach_randomize = False
    if args.arm_velocity:
        env_cfg.actions["grasp"].arm_control = "velocity"
    if args.hand_pca_project is not None:
        env_cfg.actions["grasp"].hand_pca_project = args.hand_pca_project
    if args.limited_obs and not args.student:
        # Reduce the actor obs to the limited set the checkpoint was trained on (critic keeps full).
        actor = env_cfg.observations["actor"].terms
        env_cfg.observations["actor"].terms = {
            "fingertip_contacts": ObservationTermCfg(func=fingertip_contact_binary, params={}),
            "hand_joint_pos": actor["hand_joint_pos"],
            "hand_tracking": actor["hand_tracking"],
        }
    if args.blind:
        env_cfg.observations["actor"].terms.pop("palm_to_object", None)
    if args.history > 0:
        for group in ("actor", "critic"):
            for t in ("hand_joint_pos", "hand_tracking"):
                if t in env_cfg.observations[group].terms:
                    env_cfg.observations[group].terms[t].history_length = args.history
    if args.student:
        # Distilled student: same action space as the teacher, but obs split into teacher (full) +
        # student (reduced) groups so DistillationRunner can load and run the student policy.
        env_cfg.actions["grasp"].hand_mode = args.hand_mode or "pca"
        if args.arm_residual:
            env_cfg.actions["grasp"].arm_residual = True
            env_cfg.actions["grasp"].phase.reach_randomize = False
        teacher_terms = dict(env_cfg.observations["actor"].terms)
        # Match distill_grasp.build_env exactly (same term order) so the student weights line up.
        student_terms = {
            "hand_joint_pos": teacher_terms["hand_joint_pos"],
            "hand_tracking": teacher_terms["hand_tracking"],
        }
        if args.student_obs == "fingertip":
            student_terms = {
                "fingertip_contacts": ObservationTermCfg(func=fingertip_contact_binary, params={}),
                **student_terms,
            }
        env_cfg.observations = {
            "teacher": ObservationGroupCfg(terms=teacher_terms, concatenate_terms=True),
            "student": ObservationGroupCfg(terms=student_terms, concatenate_terms=True),
        }
    if args.manual or args.tune_reach or args.free_play:
        # Don't auto-reset while hand-driving the sliders / tuning bounds / free-playing: drop the
        # success termination and make the timeout effectively never fire, so the episode runs
        # until the viewer's "Reset Environment" button. The instability guard stays as a safety.
        env_cfg.terminations.pop("success", None)
        env_cfg.episode_length_s = 1.0e6
    env_object_names = _resolve_object_names(obj_cfg, args.num_envs)
    if env_object_names is not None:
        env_object_names = list(env_object_names)
    rl_cfg = load_rl_cfg(args.task)
    if args.rnn is not None:
        for m in (rl_cfg.actor, rl_cfg.critic):
            m.class_name = "RNNModel"
            m.rnn_type = args.rnn
            m.rnn_hidden_dim = 256
            m.rnn_num_layers = 1

    if args.hand_gate:
        env_cfg.actions["grasp"].hand_gate = True
        if args.engage_radius is not None:
            env_cfg.actions["grasp"].phase.engage_radius = args.engage_radius
        if args.release_radius is not None:
            env_cfg.actions["grasp"].phase.release_radius = args.release_radius

    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=args.device))

    # Optional initial object for env 0 (mesh-variant tasks); also changeable in the GUI.
    catalogue = _variant_names(env.unwrapped.scene["object"])
    if catalogue is not None:
        if env_object_names is None:
            # num_envs may be smaller than the catalogue; seed labels from assignment.
            w2v = env.unwrapped.sim.world_to_variant.get("object")
            if w2v is not None:
                env_object_names = [catalogue[int(i)] for i in w2v.tolist()]
            else:
                env_object_names = [catalogue[0]] * args.num_envs
        if not (0 <= args.object_idx < len(catalogue)):
            p.error(f"--object-idx {args.object_idx} out of range [0, {len(catalogue)})")
        # Only rewrite when the requested idx differs — replacing Warp arrays
        # (or even a no-op rewrite via broken paths) destabilizes the sim.
        cur = (
            int(w2v[0].item())
            if (w2v := env.unwrapped.sim.world_to_variant.get("object")) is not None
            else None
        )
        if cur != args.object_idx:
            name0 = set_world_variant(
                env, entity_name="object", env_idx=0, variant_idx=args.object_idx
            )
            env_object_names[0] = name0
        else:
            name0 = catalogue[args.object_idx]
            env_object_names[0] = name0
        print(f"[play] env0 object -> {name0} (idx {args.object_idx})")

    runner = None

    def load(path: str):
        nonlocal runner
        if runner is None:
            if args.student:
                import sys
                from types import SimpleNamespace

                sys.path.insert(0, str(Path(__file__).parent))
                from distill_grasp import build_cfg

                cfg = build_cfg(SimpleNamespace(iterations=1, logger="tensorboard", wandb_project="x", run_name="play", rnn=args.rnn))
                runner = DistillationRunner(env, cfg, None, args.device)
            else:
                runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), None, args.device)
        runner.load(path)  # distill checkpoint carries student+teacher; PPO checkpoint is actor
        return runner.get_inference_policy(device=args.device)

    server = viser.ViserServer(port=args.port, label="rophi grasp - play")
    if args.share:
        share_url = server.request_share_url(verbose=False)
        print(f"[play] SHARE URL: {share_url}", flush=True)
    ckpt_mgr = None

    if args.manual or args.tune_reach:
        # Placeholder policy; GraspViewer.setup replaces self.policy (slider reader / hold action).
        action_dim = env.unwrapped.action_manager.total_action_dim
        policy = lambda obs: torch.zeros(env.num_envs, action_dim, device=args.device)  # noqa: E731
        mode = "TUNE-REACH" if args.tune_reach else "MANUAL"
        print(f"[play] {mode} mode: no checkpoint; use the viewer controls.")
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        ckpts = _checkpoints(run_dir)
        if not ckpts:
            p.error(f"no model_*.pt in {run_dir}")
        latest = ckpts[-1]

        def fetch_available() -> list[tuple[str, str]]:
            found = _checkpoints(run_dir)
            now = int(os.path.getmtime(max(found, key=os.path.getmtime)))
            # Newest first, each labelled with how long ago it was written. Re-scans every
            # call so checkpoints written after launch (a live training run) appear.
            return [
                (c.name, format_time_ago(now - int(os.path.getmtime(c)))) for c in reversed(found)
            ]

        ckpt_mgr = CheckpointManager(
            current_name=latest.name,
            fetch_available=fetch_available,
            load_checkpoint=lambda name: load(str(run_dir / name)),
            run_name=run_dir.name,
        )
        policy = load(str(latest))
        print(f"[play] browsing {len(ckpts)} checkpoints in {run_dir}; latest {latest.name}")
    else:
        policy = load(args.checkpoint)
        print(f"[play] checkpoint {args.checkpoint}")

    if env_object_names is not None:
        shown = ", ".join(dict.fromkeys(env_object_names))
        print(f"[play] {args.num_envs} envs; objects now: {shown}")
        print("[play] GUI: 'env idx' selects the viewed world; 'obj idx' swaps that world's mesh")

    if not (0 <= args.env_idx < args.num_envs):
        p.error(f"--env-idx {args.env_idx} out of range [0, {args.num_envs})")

    print(f"[play] open http://localhost:{args.port}")
    viewer = GraspViewer(env, policy, viser_server=server, checkpoint_manager=ckpt_mgr)
    viewer._env_object_names = env_object_names
    viewer._manual = args.manual
    viewer._tune_reach = args.tune_reach
    viewer._free_play = args.free_play
    viewer._pending_env_idx = int(args.env_idx)
    viewer.run()


if __name__ == "__main__":
    main()
