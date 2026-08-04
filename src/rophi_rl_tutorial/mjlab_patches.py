"""Runtime patch for an mjlab per-world-mesh-variant bug (mjlab 1.5.2).

When a scene contains a ``VariantEntityCfg`` (per-world mesh variants / "Heterogeneous
Worlds"), the variant realization path never back-assigns the compiled global IDs onto the
*sibling* entities' specs -- their ``MjsBody/MjsSite/...`` ``.id`` fields stay ``-1``.
``Entity._compute_indexing`` reads those ``.id`` values, so an unlucky sibling (here, the
robot) gets ``body_ids = site_ids = joint_ids = [-1, -1, ...]``. Indexing ``data.xpos[:, -1]``
then returns the *last* row for every body, collapsing all forward-kinematics data to a single
position. Concretely this zeroed the arm's gravity compensation, put the impedance control
point on the wrong link, and fed garbage into every positional observation -- so multi-object
training silently never learned and the arm went unstable.

The compiled scene model *does* name the elements correctly (``robot/palm`` -> global id 12),
and the variant spec's element names are already fully prefixed, so we re-resolve the broken
indices by name. Joint *addresses* (qpos/qvel) and ``ctrl_ids`` are name-resolved upstream and
stay correct, so we leave them alone.

Remove this once mjlab fixes sibling-entity index resolution under variants.
"""

from __future__ import annotations

import mujoco
import torch
from mjlab.entity.entity import Entity, EntityIndexing

_APPLIED = False


def apply_variant_indexing_patch() -> None:
    """Idempotently install the fix. Safe on non-variant scenes (it no-ops when IDs resolved)."""
    global _APPLIED
    if _APPLIED:
        return

    _orig_compute_indexing = Entity._compute_indexing

    def _compute_indexing(self, model: mujoco.MjModel, device: str) -> EntityIndexing:
        idx = _orig_compute_indexing(self, model, device)
        # Healthy entities (including the variant object itself) resolve to >= 0; only the
        # broken siblings have -1. Bail out fast in the common case.
        if idx.body_ids.numel() == 0 or int(idx.body_ids.min()) >= 0:
            return idx

        def by_name(elems, objtype) -> torch.Tensor:
            return torch.tensor(
                [mujoco.mj_name2id(model, objtype, e.name) for e in elems],
                dtype=torch.int,
                device=device,
            )

        idx.body_ids = by_name(idx.bodies, mujoco.mjtObj.mjOBJ_BODY)
        idx.site_ids = by_name(idx.sites, mujoco.mjtObj.mjOBJ_SITE)
        idx.joint_ids = by_name(idx.joints, mujoco.mjtObj.mjOBJ_JOINT)
        if idx.tendons:
            idx.tendon_ids = by_name(idx.tendons, mujoco.mjtObj.mjOBJ_TENDON)
        if idx.materials:
            idx.mat_ids = by_name(idx.materials, mujoco.mjtObj.mjOBJ_MATERIAL)

        # Geoms can be unnamed, so collect them from each (now-correct) body's geom range
        # rather than by name. Bodies are contiguous, geoms are grouped by body in order.
        geom_ids: list[int] = []
        for bid in idx.body_ids.tolist():
            adr = int(model.body_geomadr[bid])
            geom_ids.extend(range(adr, adr + int(model.body_geomnum[bid])))
        idx.geom_ids = torch.tensor(geom_ids, dtype=torch.int, device=device)

        # mocap_id was computed from the (-1) root body id; recompute from the fixed root.
        if idx.mocap_id is not None:
            idx.mocap_id = int(model.body_mocapid[int(idx.body_ids[0])])

        return idx

    Entity._compute_indexing = _compute_indexing

    # root_body_id is a property that reads bodies[0].id (also -1); fall back to body_ids[0].
    def _root_body_id(self):
        rid = self.bodies[0].id
        return rid if (rid is not None and rid >= 0) else int(self.body_ids[0])

    EntityIndexing.root_body_id = property(_root_body_id)

    _APPLIED = True
