"""Hot-swap the mesh variant on one parallel world (play-time object browser).

mjlab fixes variant assignment at sim init. For ``play_grasp`` we still want a
single-env view that can browse the full catalogue without spawning one world
per object (OOM). This helper rewrites ``geom_dataid`` / inertias for one
``env_idx`` from the already-merged variant meshes + a cheap per-variant
source compile — same data ``build_variant_model`` would have scattered.

Critical: never replace ``sim.wp_model.*`` arrays. mujoco_warp / CUDA graphs
hold the original pointers; assigning a new ``wp.array`` causes illegal memory
access and continuous instability resets. Write through ``sim.model`` (WarpBridge)
with in-place ``[:]`` / index assignment instead.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import torch
from mjlab.entity.variants import (
    SlotKey,
    _classify_geom_role,
    _iter_body_paths,
    _qualified_material_name,
    _qualified_mesh_name,
    _qualified_slot_geom_name,
)


def variant_names(entity: Any) -> list[str] | None:
    meta = getattr(entity, "variant_metadata", None)
    if meta is None:
        return None
    return list(meta.variant_names)


def _as_dev(x, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def set_world_variant(
    env: Any,
    *,
    entity_name: str = "object",
    env_idx: int = 0,
    variant_idx: int,
) -> str:
    """Assign ``variant_idx`` of ``entity_name`` to world ``env_idx``. Returns variant name.

    No-ops (returns current name) when the world is already on ``variant_idx``.
    """
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    entity = unwrapped.scene[entity_name]
    meta = entity.variant_metadata
    if meta is None:
        raise RuntimeError(f"entity {entity_name!r} has no mesh variants")
    nvar = len(meta.variant_names)
    if not (0 <= variant_idx < nvar):
        raise IndexError(f"variant_idx {variant_idx} out of range [0, {nvar})")

    sim = unwrapped.sim
    nworld = sim.num_envs
    if not (0 <= env_idx < nworld):
        raise IndexError(f"env_idx {env_idx} out of range [0, {nworld})")

    variant_name = meta.variant_names[variant_idx]
    w2v = sim.world_to_variant.get(entity_name)
    if w2v is not None and int(w2v[env_idx].item()) == int(variant_idx):
        return variant_name

    prefix = f"{entity_name}/"
    mj_model = sim.mj_model
    model = sim.model  # WarpBridge — in-place only
    device = sim.device

    # --- geom_dataid / geom_matid for this world ---
    for s_idx, slot in enumerate(meta.slots):
        gspec = meta.variant_slot_specs[variant_idx][s_idx]
        full_geom = _qualified_slot_geom_name(prefix, slot.template_geom_name)
        gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, full_geom)
        if gid < 0:
            continue
        if gspec is None:
            model.geom_dataid[env_idx, gid] = -1
            model.geom_matid[env_idx, gid] = -1
            continue
        mid = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_MESH,
            _qualified_mesh_name(prefix, variant_name, gspec.mesh_name),
        )
        model.geom_dataid[env_idx, gid] = int(mid)
        if gspec.material:
            model.geom_matid[env_idx, gid] = mujoco.mj_name2id(
                mj_model,
                mujoco.mjtObj.mjOBJ_MATERIAL,
                _qualified_material_name(prefix, variant_name, gspec.material),
            )
        else:
            model.geom_matid[env_idx, gid] = -1

    # --- body / geom size fields from an isolated source compile ---
    sample_spec = meta.variant_source_specs[0]
    sample_root = list(sample_spec.worldbody.bodies)[0]
    scene_body_id_by_path: dict[str, int] = {}
    for body_path, body in _iter_body_paths(sample_root):
        scene_name = f"{prefix}{body.name or ''}"
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, scene_name)
        if bid >= 0:
            scene_body_id_by_path[body_path] = bid

    scene_geom_id_by_slot: dict[SlotKey, int] = {}
    for slot in meta.slots:
        full_geom = _qualified_slot_geom_name(prefix, slot.template_geom_name)
        gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, full_geom)
        if gid >= 0:
            scene_geom_id_by_slot[slot.key] = gid

    sorted_paths = sorted(scene_body_id_by_path.keys())
    root_body_path = sorted_paths[0]
    root_scene_bid = scene_body_id_by_path[root_body_path]
    ancestor_ids: list[int] = []
    parent_id = int(mj_model.body_parentid[root_scene_bid])
    while True:
        ancestor_ids.append(parent_id)
        if parent_id == 0:
            break
        parent_id = int(mj_model.body_parentid[parent_id])
    base_root_subtreemass = float(mj_model.body_subtreemass[root_scene_bid])

    source_spec = meta.variant_source_specs[variant_idx].copy()
    v_model = source_spec.compile()
    source_root = list(source_spec.worldbody.bodies)[0]
    source_body_id_by_path = {
        body_path: body.id for body_path, body in _iter_body_paths(source_root)
    }

    defaults = sim._default_model_fields

    for body_path, scene_bid in scene_body_id_by_path.items():
        source_bid = source_body_id_by_path[body_path]
        mass = float(v_model.body_mass[source_bid])
        sub = float(v_model.body_subtreemass[source_bid])
        inertia = _as_dev(v_model.body_inertia[source_bid], device)
        invw = _as_dev(v_model.body_invweight0[source_bid], device)
        ipos = _as_dev(v_model.body_ipos[source_bid], device)
        iquat = _as_dev(v_model.body_iquat[source_bid], device)

        model.body_mass[env_idx, scene_bid] = mass
        model.body_subtreemass[env_idx, scene_bid] = sub
        model.body_inertia[env_idx, scene_bid] = inertia
        model.body_invweight0[env_idx, scene_bid] = invw
        model.body_ipos[env_idx, scene_bid] = ipos
        model.body_iquat[env_idx, scene_bid] = iquat

        if "body_mass" in defaults:
            defaults["body_mass"][env_idx, scene_bid] = mass
            defaults["body_subtreemass"][env_idx, scene_bid] = sub
            defaults["body_inertia"][env_idx, scene_bid] = inertia
            defaults["body_invweight0"][env_idx, scene_bid] = invw
            defaults["body_ipos"][env_idx, scene_bid] = ipos
            defaults["body_iquat"][env_idx, scene_bid] = iquat

    mesh_type = mujoco.mjtGeom.mjGEOM_MESH
    for body_path, body in _iter_body_paths(source_root):
        role_ordinals = {"visual": 0, "collision": 0}
        for g in body.geoms:
            if g.type != mesh_type:
                continue
            role = _classify_geom_role(g)
            ordinal = role_ordinals[role]
            role_ordinals[role] += 1
            scene_gid = scene_geom_id_by_slot.get(
                SlotKey(body_path=body_path, role=role, ordinal=ordinal)
            )
            if scene_gid is None:
                continue
            gsize = _as_dev(v_model.geom_size[g.id], device)
            rbound = float(v_model.geom_rbound[g.id])
            gaabb = _as_dev(v_model.geom_aabb[g.id].reshape(2, 3), device)
            gpos = _as_dev(v_model.geom_pos[g.id], device)
            gquat = _as_dev(v_model.geom_quat[g.id], device)

            model.geom_size[env_idx, scene_gid] = gsize
            model.geom_rbound[env_idx, scene_gid] = rbound
            model.geom_aabb[env_idx, scene_gid] = gaabb
            model.geom_pos[env_idx, scene_gid] = gpos
            model.geom_quat[env_idx, scene_gid] = gquat

            if "geom_size" in defaults:
                defaults["geom_size"][env_idx, scene_gid] = gsize
                defaults["geom_rbound"][env_idx, scene_gid] = rbound
                defaults["geom_aabb"][env_idx, scene_gid] = gaabb
                defaults["geom_pos"][env_idx, scene_gid] = gpos
                defaults["geom_quat"][env_idx, scene_gid] = gquat

    # Absolute ancestor subtreemass = template + (variant - template_root).
    # Absolute set (not +=) so repeated swaps don't accumulate deltas.
    delta = (
        float(v_model.body_subtreemass[source_body_id_by_path[root_body_path]])
        - base_root_subtreemass
    )
    for anc_bid in ancestor_ids:
        val = float(mj_model.body_subtreemass[anc_bid]) + delta
        model.body_subtreemass[env_idx, anc_bid] = val
        if "body_subtreemass" in defaults:
            defaults["body_subtreemass"][env_idx, anc_bid] = val

    if w2v is not None:
        w2v[env_idx] = int(variant_idx)

    return variant_name
