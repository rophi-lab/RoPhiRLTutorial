"""Download and prepare the YCB object tree used by ``rophi_rl_tutorial.assets.grasping.ycb``.

Pipeline (mirrors ``Manipulator-Software/datasets/update_YCB_physics_properties.py``):

1. Download per object from `kwonathan/ycb_urdfs`_ (already has Google 16k meshes + VHACD)::

       {berkeley}/google_16k/textured.obj
       {berkeley}/google_16k/textured_vhacd.obj
       {berkeley}.urdf

2. Rename into the ``v0xx_...`` layout this repo expects (``013_apple`` → ``v013_apple``,
   ``063-a_marbles`` → ``v063a_marbles``).

3. Canonicalise + rescale meshes (OBB frame, shortest extent in [0.02, 0.09] m) and write::

       v0xx_.../google_16k/norm_textured.obj
       v0xx_.../google_16k/norm_textured_vhacd.obj
       v0xx_....urdf   # mass / inertia scaled with the mesh, origins at 0

Idempotent: objects that already have both ``norm_*`` meshes and a prepared URDF are skipped
unless ``--force``.

Examples::

    uv run --extra assets python scripts/grasping/fetch_ycb.py --easy          # Apple task subset
    uv run --extra assets python scripts/grasping/fetch_ycb.py --objects v013_apple
    uv run --extra assets python scripts/grasping/fetch_ycb.py                  # full YCB_OBJECT_LIST

.. _kwonathan/ycb_urdfs: https://github.com/kwonathan/ycb_urdfs
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from rophi_rl_tutorial import ROPHI_RL_ROOT_PATH
from rophi_rl_tutorial.assets.grasping.ycb import EASY_YCB_OBJECT_LIST, YCB_OBJECT_LIST

##
# Sources.
##

YCB_URDFS_RAW = "https://raw.githubusercontent.com/kwonathan/ycb_urdfs/main/ycb_assets"
MIN_SCALE = 0.02
MAX_SCALE = 0.09

# Mesh files required as *inputs* to the normalize step (downloaded from ycb_urdfs).
_RAW_MESH_FILES = ("textured.obj", "textured_vhacd.obj")


##
# Curated mass / inertia (from Manipulator-Software update_YCB_physics_properties.py).
# Keyed by the ``v0xx`` / ``v0xxa`` id prefix of our object names.
##


@dataclass(frozen=True)
class YcbPhysics:
    mass: float
    ixx: float
    iyy: float
    izz: float
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0


# Ported from Manipulator-Software; v021 corrected to bleach_cleanser (list had wine_glass).
_PHYSICS_BY_ID: dict[str, YcbPhysics] = {
    "v002": YcbPhysics(0.530, 0.00045, 0.00045, 0.00075),
    "v003": YcbPhysics(0.411, 0.00109, 0.00171, 0.00247),
    "v004": YcbPhysics(0.514, 0.00040, 0.00140, 0.00160),
    "v005": YcbPhysics(0.349, 0.00019, 0.00039, 0.00039),
    "v006": YcbPhysics(0.603, 0.00062, 0.00201, 0.00223),
    "v007": YcbPhysics(0.171, 0.00005, 0.00005, 0.00008),
    "v008": YcbPhysics(0.187, 0.00005, 0.00014, 0.00017),
    "v009": YcbPhysics(0.097, 0.00002, 0.00007, 0.00008),
    "v010": YcbPhysics(0.370, 0.00027, 0.00036, 0.00048),
    "v011": YcbPhysics(0.066, 0.00002, 0.00016, 0.00016),
    "v012": YcbPhysics(0.018, 0.000001, 0.000001, 0.000001),
    "v013": YcbPhysics(0.068, 0.00004, 0.00004, 0.00004),
    "v014": YcbPhysics(0.029, 0.00001, 0.00001, 0.00001),
    "v015": YcbPhysics(0.033, 0.00002, 0.00002, 0.00002),
    "v016": YcbPhysics(0.049, 0.00003, 0.00004, 0.00004),
    "v017": YcbPhysics(0.047, 0.00003, 0.00003, 0.00003),
    "v018": YcbPhysics(0.025, 0.00001, 0.00001, 0.00001),
    "v019": YcbPhysics(0.178, 0.00078, 0.00115, 0.00130),
    "v021": YcbPhysics(1.131, 0.00120, 0.00350, 0.00370),  # bleach_cleanser
    "v022": YcbPhysics(0.942, 0.00115, 0.00312, 0.00331),
    "v024": YcbPhysics(0.147, 0.00042, 0.00042, 0.00078),
    "v025": YcbPhysics(0.118, 0.00015, 0.00015, 0.00021),
    "v026": YcbPhysics(0.018, 0.00001, 0.00002, 0.00003),
    "v028": YcbPhysics(0.520, 0.00150, 0.00150, 0.00280),
    "v029": YcbPhysics(0.279, 0.00120, 0.00120, 0.00235),
    "v030": YcbPhysics(0.031, 0.00001, 0.000001, 0.00001),
    "v031": YcbPhysics(0.035, 0.00001, 0.000001, 0.00001),
    "v032": YcbPhysics(0.030, 0.00001, 0.000001, 0.00001),
    "v033": YcbPhysics(0.052, 0.00001, 0.00005, 0.00005),
    "v035": YcbPhysics(0.895, 0.00120, 0.00300, 0.00330),
    "v036": YcbPhysics(0.210, 0.00012, 0.00041, 0.00049),
    "v037": YcbPhysics(0.082, 0.00002, 0.00011, 0.00012),
    "v038": YcbPhysics(0.262, 0.00006, 0.00015, 0.00018),
    "v040": YcbPhysics(0.015, 0.000001, 0.00002, 0.00002),
    "v042": YcbPhysics(0.272, 0.00004, 0.00085, 0.00088),
    "v043": YcbPhysics(0.095, 0.00001, 0.00028, 0.00028),
    "v044": YcbPhysics(0.098, 0.00001, 0.00031, 0.00031),
    "v048": YcbPhysics(0.665, 0.00031, 0.00242, 0.00242),
    "v050": YcbPhysics(0.059, 0.000009, 0.000046, 0.000046),
    "v051": YcbPhysics(0.121, 0.000033, 0.000181, 0.000181),
    "v052": YcbPhysics(0.202, 0.000080, 0.000454, 0.000454),
    "v053": YcbPhysics(0.123, 0.00014, 0.00014, 0.00014),
    "v054": YcbPhysics(0.190, 0.00028, 0.00028, 0.00028),
    "v055": YcbPhysics(0.145, 0.00016, 0.00016, 0.00016),
    "v056": YcbPhysics(0.058, 0.00006, 0.00006, 0.00006),
    "v057": YcbPhysics(0.041, 0.00003, 0.00003, 0.00003),
    "v058": YcbPhysics(0.046, 0.00001, 0.00001, 0.00001),
    "v059": YcbPhysics(0.200, 0.00020, 0.00020, 0.00020),
    "v061": YcbPhysics(0.028, 0.00001, 0.00004, 0.00005),
    "v062": YcbPhysics(0.005, 0.000001, 0.000001, 0.000001),
    "v063a": YcbPhysics(0.020, 1.2e-6, 1.2e-6, 1.2e-6),
    "v063b": YcbPhysics(0.020, 1.2e-6, 1.2e-6, 1.2e-6),
    "v065a": YcbPhysics(0.045, 2.2e-5, 3.5e-5, 3.5e-5),
    "v065b": YcbPhysics(0.045, 2.2e-5, 3.5e-5, 3.5e-5),
    "v065c": YcbPhysics(0.045, 2.2e-5, 3.5e-5, 3.5e-5),
    "v065d": YcbPhysics(0.045, 2.2e-5, 3.5e-5, 3.5e-5),
    "v065e": YcbPhysics(0.045, 2.2e-5, 3.5e-5, 3.5e-5),
    "v065f": YcbPhysics(0.045, 2.2e-5, 3.5e-5, 3.5e-5),
    "v065g": YcbPhysics(0.045, 2.2e-5, 3.5e-5, 3.5e-5),
    "v065h": YcbPhysics(0.045, 2.2e-5, 3.5e-5, 3.5e-5),
    "v065i": YcbPhysics(0.045, 2.2e-5, 3.5e-5, 3.5e-5),
    "v065j": YcbPhysics(0.045, 2.2e-5, 3.5e-5, 3.5e-5),
    "v070a": YcbPhysics(0.012, 4e-6, 4e-6, 4e-6),
    "v070b": YcbPhysics(0.012, 4e-6, 4e-6, 4e-6),
    "v071": YcbPhysics(0.380, 0.00125, 0.00125, 0.00240),
    "v072a": YcbPhysics(0.035, 4e-5, 6e-5, 8e-5),
    "v072b": YcbPhysics(0.035, 4e-5, 6e-5, 8e-5),
    "v072c": YcbPhysics(0.035, 4e-5, 6e-5, 8e-5),
    "v072d": YcbPhysics(0.035, 4e-5, 6e-5, 8e-5),
    "v072e": YcbPhysics(0.035, 4e-5, 6e-5, 8e-5),
    "v073a": YcbPhysics(0.014, 4e-6, 6e-6, 8e-6),
    "v073b": YcbPhysics(0.014, 4e-6, 6e-6, 8e-6),
    "v073c": YcbPhysics(0.014, 4e-6, 6e-6, 8e-6),
    "v073d": YcbPhysics(0.014, 4e-6, 6e-6, 8e-6),
    "v073e": YcbPhysics(0.014, 4e-6, 6e-6, 8e-6),
    "v073f": YcbPhysics(0.014, 4e-6, 6e-6, 8e-6),
    "v073g": YcbPhysics(0.014, 4e-6, 6e-6, 8e-6),
    "v077": YcbPhysics(0.095, 8e-5, 8e-5, 8e-5),
}


def object_id(name: str) -> str:
    """``v013_apple`` → ``v013``; ``v063a_marbles`` → ``v063a``."""
    return name.split("_", 1)[0]


def our_to_berkeley(name: str) -> str:
    """``v063a_marbles`` → ``063-a_marbles``; ``v013_apple`` → ``013_apple``."""
    if not name.startswith("v"):
        raise ValueError(f"Expected v-prefixed YCB name, got {name!r}")
    rest = name[1:]
    match = re.match(r"^(\d+)([a-z])_(.+)$", rest)
    if match:
        return f"{match.group(1)}-{match.group(2)}_{match.group(3)}"
    return rest


def physics_for(name: str) -> YcbPhysics:
    oid = object_id(name)
    try:
        return _PHYSICS_BY_ID[oid]
    except KeyError as e:
        raise KeyError(f"No curated physics for {name!r} (id={oid})") from e


##
# Download helpers.
##


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def download_object(name: str, ycb_root: Path, force: bool) -> str:
    """Fetch textured + textured_vhacd + raw URDF into the v-prefixed layout."""
    berkeley = our_to_berkeley(name)
    mesh_dir = ycb_root / name / "google_16k"
    urdf_path = ycb_root / f"{name}.urdf"

    needed = [mesh_dir / f for f in _RAW_MESH_FILES] + [urdf_path]
    if not force and all(p.exists() for p in needed):
        return "raw-skip"

    for fname in _RAW_MESH_FILES:
        dst = mesh_dir / fname
        if force or not dst.exists():
            url = f"{YCB_URDFS_RAW}/{berkeley}/google_16k/{fname}"
            _download(url, dst)

    if force or not urdf_path.exists() or not _urdf_points_at(urdf_path, name):
        # Download berkeley URDF and rewrite name / mesh paths to our layout.
        src_url = f"{YCB_URDFS_RAW}/{berkeley}.urdf"
        raw_bytes = urllib.request.urlopen(src_url, timeout=60).read()
        urdf_path.parent.mkdir(parents=True, exist_ok=True)
        urdf_path.write_bytes(_rewrite_urdf(raw_bytes, berkeley, name))

    return "raw-ok"


def _urdf_points_at(urdf_path: Path, name: str) -> bool:
    try:
        text = urdf_path.read_text()
    except OSError:
        return False
    return f"{name}/google_16k/" in text


def _rewrite_urdf(raw: bytes, berkeley: str, name: str) -> bytes:
    """Retarget mesh paths and robot name from berkeley naming to ``v0xx_...``."""
    root = ET.fromstring(raw)
    root.set("name", name)
    for mesh in root.findall(".//mesh"):
        fname = mesh.get("filename")
        if fname and berkeley in fname:
            mesh.set("filename", fname.replace(berkeley, name, 1))
    return ET.tostring(root, encoding="utf-8")


##
# Normalize (canonical OBB + scale + physics) — from update_YCB_physics_properties.py.
##


def _as_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def _canonical_transform(mesh: trimesh.Trimesh) -> np.ndarray:
    """World → OBB frame with column reorder [2, 1, 0] (Manipulator convention)."""
    obb_transform = mesh.bounding_box_oriented.primitive.transform
    canonical = np.eye(4)
    canonical[:3, :3] = np.asarray(obb_transform[:3, :3])[:, [2, 1, 0]]
    canonical[:3, 3] = np.asarray(obb_transform[:3, 3])
    if np.linalg.det(canonical[:3, :3]) < 0:
        canonical[:3, 2] *= -1
    return canonical


def _scale_factor(mesh: trimesh.Trimesh) -> float:
    extents = np.asarray(mesh.bounding_box_oriented.primitive.extents, dtype=np.float64)
    shortest = float(extents.min())
    if shortest < MIN_SCALE:
        return MIN_SCALE / shortest
    if shortest > MAX_SCALE:
        return MAX_SCALE / shortest
    return 1.0


def prepare_object(name: str, ycb_root: Path, force: bool) -> str:
    """Write ``norm_textured*.obj`` and update the URDF inertial / mesh paths."""
    mesh_dir = ycb_root / name / "google_16k"
    visual_raw = mesh_dir / "textured.obj"
    collision_raw = mesh_dir / "textured_vhacd.obj"
    visual_norm = mesh_dir / "norm_textured.obj"
    collision_norm = mesh_dir / "norm_textured_vhacd.obj"
    urdf_path = ycb_root / f"{name}.urdf"

    if (
        not force
        and visual_norm.exists()
        and collision_norm.exists()
        and urdf_path.exists()
        and "norm_textured.obj" in urdf_path.read_text()
    ):
        return "prep-skip"

    if not visual_raw.exists() or not collision_raw.exists():
        raise FileNotFoundError(
            f"Missing raw meshes for {name}: need {visual_raw.name} and {collision_raw.name}"
        )

    mesh_visual = _as_trimesh(visual_raw)
    T_inv = np.linalg.inv(_canonical_transform(mesh_visual))
    mesh_visual.apply_transform(T_inv)
    s = _scale_factor(mesh_visual)
    if s != 1.0:
        scale_T = np.eye(4)
        scale_T[:3, :3] *= s
        mesh_visual.apply_transform(scale_T)
    mesh_visual.export(visual_norm)

    mesh_collision = _as_trimesh(collision_raw)
    mesh_collision.apply_transform(T_inv)
    if s != 1.0:
        scale_T = np.eye(4)
        scale_T[:3, :3] *= s
        mesh_collision.apply_transform(scale_T)
    mesh_collision.export(collision_norm)

    com = np.asarray(mesh_visual.center_mass, dtype=np.float64)
    phys = physics_for(name)
    # Mass ~ s^3; inertia ~ s^5 for a similarity (Manipulator used s^2 — keep their convention
    # for drop-in compatibility with existing prepared URDFs in the lab).
    mass = phys.mass * (s**3)
    ixx = phys.ixx * (s**2)
    iyy = phys.iyy * (s**2)
    izz = phys.izz * (s**2)
    ixy = phys.ixy * (s**2)
    ixz = phys.ixz * (s**2)
    iyz = phys.iyz * (s**2)

    _write_prepared_urdf(
        urdf_path,
        name=name,
        com=com,
        mass=mass,
        ixx=ixx,
        iyy=iyy,
        izz=izz,
        ixy=ixy,
        ixz=ixz,
        iyz=iyz,
    )
    return f"prep-ok  s={s:.4f} mass={mass:.4f}kg"


def _write_prepared_urdf(
    urdf_path: Path,
    *,
    name: str,
    com: np.ndarray,
    mass: float,
    ixx: float,
    iyy: float,
    izz: float,
    ixy: float,
    ixz: float,
    iyz: float,
) -> None:
    """Emit a URDF pointing at the normalised meshes with curated inertial."""
    visual = f"{name}/google_16k/norm_textured.obj"
    collision = f"{name}/google_16k/norm_textured_vhacd.obj"
    com_xyz = f"{com[0]} {com[1]} {com[2]}"
    xml = f"""\
<robot name="{name}">
\t<link name="base">
\t\t<visual>
\t\t\t<origin xyz="0 0 0" rpy="0 0 0" />
\t\t\t<geometry>
\t\t\t\t<mesh filename="{visual}" scale="1 1 1" />
\t\t\t</geometry>
\t\t\t<material name="texture">
\t\t\t\t<color rgba="1.0 1.0 1.0 1.0" />
\t\t\t</material>
\t\t</visual>
\t\t<collision>
\t\t\t<origin xyz="0 0 0" rpy="0 0 0" />
\t\t\t<geometry>
\t\t\t\t<mesh filename="{collision}" scale="1 1 1" />
\t\t\t</geometry>
\t\t</collision>
\t\t<inertial>
\t\t\t<origin xyz="{com_xyz}" rpy="0 0 0" />
\t\t\t<mass value="{mass}" />
\t\t\t<inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}" iyy="{iyy}" iyz="{iyz}" izz="{izz}" />
\t\t</inertial>
\t</link>
</robot>
"""
    urdf_path.write_text(xml)


def _ensure_vhacd_fallback(name: str, ycb_root: Path) -> None:
    """If textured_vhacd is missing, build a CoACD multi-hull OBJ (optional dep)."""
    dst = ycb_root / name / "google_16k" / "textured_vhacd.obj"
    if dst.exists():
        return
    src = ycb_root / name / "google_16k" / "textured.obj"
    if not src.exists():
        return
    try:
        import coacd
    except ImportError as e:
        raise RuntimeError(
            f"Missing {dst.name} for {name} and coacd is not installed. "
            "Run: uv sync --extra assets"
        ) from e

    mesh = _as_trimesh(src)
    imesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(imesh)
    # Export as a single OBJ with disjoint components (what MuJoCo/split() expect).
    combined = trimesh.util.concatenate(
        [trimesh.Trimesh(vertices=v, faces=f, process=False) for v, f in parts]
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    combined.export(dst)
    print(f"  [coacd] wrote {dst.relative_to(ycb_root)} ({len(parts)} hulls)")


##
# CLI.
##


def _select_objects(args: argparse.Namespace) -> list[str]:
    if args.objects:
        names = list(args.objects)
    elif args.easy:
        names = list(EASY_YCB_OBJECT_LIST)
    else:
        names = list(YCB_OBJECT_LIST)
    unknown = [n for n in names if n not in YCB_OBJECT_LIST]
    if unknown:
        raise SystemExit(f"Unknown YCB object(s): {unknown}")
    if args.limit is not None:
        names = names[: args.limit]
    return names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out",
        type=Path,
        default=ROPHI_RL_ROOT_PATH / "datasets" / "ycb",
        help="Output YCB root (default: <repo>/datasets/ycb).",
    )
    p.add_argument("--objects", nargs="+", default=None, help="Specific v0xx_* names.")
    p.add_argument("--easy", action="store_true", help="Only EASY_YCB_OBJECT_LIST (Apple task set).")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N selected objects.")
    p.add_argument("--force", action="store_true", help="Re-download / re-prepare even if outputs exist.")
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Stop after fetching textured / textured_vhacd / raw URDF (no normalize).",
    )
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Skip download; only run normalize on objects already present under --out.",
    )
    args = p.parse_args()

    ycb_root: Path = args.out
    if ycb_root.is_symlink():
        print(f"[fetch] note: {ycb_root} is a symlink -> {ycb_root.resolve()}")
    ycb_root.mkdir(parents=True, exist_ok=True)

    names = _select_objects(args)
    print(f"[fetch] {len(names)} objects -> {ycb_root}")

    n_ok = n_skip = n_fail = 0
    for i, name in enumerate(names, 1):
        prefix = f"[{i:02d}/{len(names)}] {name}"
        try:
            statuses = []
            if not args.prepare_only:
                statuses.append(download_object(name, ycb_root, args.force))
                _ensure_vhacd_fallback(name, ycb_root)
            if not args.download_only:
                statuses.append(prepare_object(name, ycb_root, args.force))
            status = ", ".join(statuses)
            if all(s.endswith("skip") for s in statuses):
                n_skip += 1
            else:
                n_ok += 1
            print(f"{prefix}  {status}")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, RuntimeError, ValueError) as e:
            n_fail += 1
            print(f"{prefix}  FAIL: {type(e).__name__}: {e}")

    print(f"\n[fetch] done: {n_ok} updated, {n_skip} skipped, {n_fail} failed -> {ycb_root}")
    if n_fail:
        sys.exit(1)
    apple = ycb_root / "v013_apple" / "google_16k" / "norm_textured.obj"
    if apple.exists():
        print(f"[fetch] ready for RoPhi-Grasp-Apple ({apple})")
    else:
        print("[fetch] tip: run with --easy or --objects v013_apple for the Apple task")


if __name__ == "__main__":
    main()
