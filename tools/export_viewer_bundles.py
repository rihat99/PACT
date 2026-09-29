"""Export the PACT website's 3D viewer bundles (``meta.json`` + ``data.bin`` per clip).

Runs on robotics3 inside contact_anything's venv; both research repos are imported
read-only (``sys.dont_write_bytecode`` keeps them free of ``__pycache__``)::

    ~/better/contact_anything/.venv/bin/python ~/pact_web_export/export_web.py \\
        --out ~/pact_web_export/viewer

Two sets, one bundle per clip under ``<out>/<set>/<stem>/``:

* ``annotation`` (boulder_1, boulder_3): the annotation pipeline's physics-optimized
  body and forces, BetterVideoReconstruction ``out/<stem>/human_optim/kindyn_1.npz``
  (world ``q``, one ``betas``, contact-frame forces in newtons), the fused wall cloud
  ``geometry/scene.npz`` and the per-frame cameras ``geometry/transform.npz``.
* ``prediction`` (climbing_dyno, backflip, olympics_2, yoga_1): PACT's predictions,
  ``willd_videos/out/<stem>/predictions/joint_frames35/{smplx,forces_sup,contacts}.npz``
  (camera-frame ``q_cam`` + per-row ``betas``, forces in body weights) folded into the
  world with ``geometry/transform.npz`` exactly as ``scripts/render_wild_3d.py`` does.

Both sets use the same SMPL-X body (BetterHuman ``SMPLX_NEUTRAL.npz``, 52 joints,
``viewer.bodies.load_body``) and the same limb fold as ``render_wild_3d.py``: each of the
35 contact slots goes to its 22-body joint (fingers -> wrist), toes -> ankle
(``FOLD_JOINT``); every joint that receives a slot is one limb (arrow). A limb's arrow
starts at the middle-finger base for a wrist, mid-foot (ankle-toe midpoint) for an ankle,
and at the joint itself otherwise (``anchor_points``). Prediction forces sum the slots
whose contact probability is >= 0.5 (``joint_forces_cam``); annotation forces sum all
slots (they are exactly zero off contact) and are divided by ``total_mass * 9.81``.

World frame of a bundle: metres, +Y up, gravity exactly -Y. The source world is first
turned 180 deg about X (OpenCV -> Y up) and then by the smallest rotation that takes the
gravity direction (annotation: ``gravity_world``; prediction: the median of the per-row
``gravity_world``, as ``render_wild_3d.py``) onto -Y. For boulder_1 (gravity [0, 1, 0])
that is exactly the 180 deg turn about X.

The mesh is skinned in the browser by linear blend skinning from a rest mesh at the
display identity (annotation: the clip's betas; prediction: the median of the per-row
betas) and the exact per-row FK world joint poses; the self-check compares that LBS
against the full SMPL-X forward pass (per-row betas, pose correctives) on 5 rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True                          # never write into the read-only repos

HOME = Path("/home/rikhat.akizhanov")
CA = HOME / "better" / "contact_anything"
BVR = HOME / "better" / "BetterVideoReconstruction"
WILD = HOME / "better" / "data" / "willd_videos"
sys.path.insert(0, str(CA))
sys.path.insert(0, str(CA / "scripts"))

import cv2                                                                    # noqa: E402
import numpy as np                                                            # noqa: E402
import torch                                                                  # noqa: E402

from model.contact_frames import body22_parent, contact_set                   # noqa: E402
from render_wild_3d import (ARROW_M_PER_BW_3D, FEET_JOINTS, anchor_points,    # noqa: E402
                            load_dump)
from render_wild_overlays import CONTACT_THRESHOLD, FOLD_JOINT                # noqa: E402
from viewer.bodies import _fk_world, load_body, top4_weights                  # noqa: E402

SETS = {"annotation": ("boulder_1", "boulder_3"),
        "prediction": ("climbing_dyno", "backflip", "olympics_2", "yoga_1")}
PRED_DIR = "joint_frames35"
G = 9.81
MAX_SCENE_POINTS = 60_000
SCENE_CONF_DROP_Q = 0.1        # BVR viewer's _SCENE_CONF_DROP_Q: show the cloud the optimizer used
FLOOR_MARGIN_M = 0.01          # render_wild_3d: floor = lowest foot joint - 1 cm
N_CHECK_ROWS = 5
RX180 = np.diag([1.0, -1.0, -1.0])


# ----------------------------------------------------------------------------- rotations
def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotation matrices -> ``(..., 4)`` unit quaternions, xyzw."""
    m = np.asarray(m, np.float64)
    shp = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    q = np.empty((len(m), 4))
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for i, r in enumerate(m):
        if tr[i] > 0:
            s = np.sqrt(tr[i] + 1.0) * 2
            q[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q[i] = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q[i] = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(*shp, 4)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """``(..., 4)`` xyzw -> ``(..., 3, 3)``."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit ``a`` onto unit ``b`` (Rodrigues)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        raise ValueError("antiparallel gravity after the X turn: cannot happen for a sane clip")
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def world_rotation(down: np.ndarray) -> np.ndarray:
    """Source world -> bundle world: 180 deg about X, then gravity onto -Y."""
    g = RX180 @ (down / np.linalg.norm(down))
    return min_rotation(g, np.array([0.0, -1.0, 0.0])) @ RX180


# ----------------------------------------------------------------------------- helpers
def probe_video(path: Path) -> dict:
    """Source video fps / size, and its frame count by decoding (cv2, like the pipelines)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return {"fps": round(float(fps), 6), "width": w, "height": h, "n_frames": n}


def fill_invalid(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rows outside ``valid`` take the nearest valid row (the viewer hides them anyway)."""
    idx = np.flatnonzero(valid)
    near = idx[np.clip(np.searchsorted(idx, np.arange(len(valid))), 0, len(idx) - 1)]
    prev = idx[np.clip(np.searchsorted(idx, np.arange(len(valid))) - 1, 0, len(idx) - 1)]
    rows = np.arange(len(valid))
    pick = np.where(np.abs(prev - rows) < np.abs(near - rows), prev, near)
    return a[pick]


def slot_fold() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """render_wild_3d's fold: slot -> limb joint, the limb joints, the slot vertices."""
    cs = contact_set("frames35")
    slot_parent = np.asarray([FOLD_JOINT.get(body22_parent(j), body22_parent(j))
                              for j in cs.parent_joint52], np.int64)
    return slot_parent, np.unique(slot_parent), np.asarray(cs.vertex_ids, np.int64)


def pose_body(body, betas_rows: np.ndarray, q_rows: np.ndarray, valid: np.ndarray,
              world_from_cam: np.ndarray | None, R: np.ndarray, device) -> dict:
    """Exact per-row FK joint poses and full SMPL-X vertices in the bundle world.

    ``q_rows`` is world-frame (``world_from_cam`` None) or camera-frame. The rest mesh is
    shaped at the median betas of the valid rows (the display identity).
    """
    idx = np.flatnonzero(valid)
    n = len(valid)
    identity = np.median(betas_rows[idx], axis=0).astype(np.float32)
    quat = np.zeros((n, 52, 4))
    pos = np.zeros((n, 52, 3))
    verts = np.zeros((n, 10475, 3), np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            rows = idx[s:s + 64]
            b = torch.as_tensor(betas_rows[rows], dtype=torch.float32, device=device)
            q = torch.as_tensor(q_rows[rows], dtype=torch.float32, device=device)
            wfc = (None if world_from_cam is None else
                   torch.as_tensor(world_from_cam[rows], dtype=torch.float32, device=device))
            qt, pt = _fk_world(body, b, q, wfc)
            rot = quat_to_mat(qt.cpu().numpy())                              # (B, 52, 3, 3) source world
            quat[rows] = mat_to_quat(R @ rot)
            pos[rows] = pt.cpu().numpy() @ R.T
            shaped = body.with_shape(betas=b)
            v = shaped.vertices_from_data(shaped.fk(q)).cpu().numpy().astype(np.float64)
            if world_from_cam is not None:
                w = world_from_cam[rows]
                v = np.einsum("bij,bvj->bvi", w[:, :3, :3], v) + w[:, None, :3, 3]
            verts[rows] = v @ R.T
        shaped = body.with_shape(betas=torch.as_tensor(identity, device=device)[None])
        v_rest = shaped.values.v_shaped[0].cpu().numpy()
        j_rest = shaped.values.rest_joints[0].cpu().numpy()
        weights = top4_weights(shaped.values.skinning_weight_matrix.cpu().numpy())
    skin_idx = np.argsort(weights, axis=1)[:, -4:][:, ::-1].astype(np.uint16)
    skin_w = np.take_along_axis(weights, skin_idx.astype(np.int64), 1).astype(np.float32)
    skin_w /= skin_w.sum(1, keepdims=True)
    return {"quat": quat, "pos": pos, "verts": verts, "v_rest": v_rest, "j_rest": j_rest,
            "skin_idx": skin_idx, "skin_w": skin_w, "identity": identity}


def lbs(arr: dict, row: int) -> np.ndarray:
    """The browser's skinning formula, on the arrays as written to data.bin."""
    v = arr["verts"].astype(np.float64)
    rj = arr["rest_joints"].astype(np.float64)
    rot = quat_to_mat(arr["joint_quat"][row])                                # (J, 3, 3)
    pos = arr["joint_pos"][row].astype(np.float64)
    out = np.zeros_like(v)
    for k in range(4):
        j = arr["skin_idx"][:, k].astype(np.int64)
        local = v - rj[j]
        out += arr["skin_w"][:, k:k + 1] * (np.einsum("vij,vj->vi", rot[j], local) + pos[j])
    return out


def write_bundle(out_dir: Path, meta: dict, arrays: dict) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    meta["arrays"] = {}
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        assert a.dtype.byteorder in ("<", "=", "|"), name
        while len(blob) % 4:
            blob.append(0)
        meta["arrays"][name] = {"dtype": str(a.dtype), "shape": list(a.shape),
                                "offset": len(blob), "nbytes": a.nbytes}
        blob += a.tobytes()
    (out_dir / "data.bin").write_bytes(bytes(blob))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return len(blob)


def read_back(out_dir: Path) -> tuple[dict, dict]:
    meta = json.loads((out_dir / "meta.json").read_text())
    blob = (out_dir / "data.bin").read_bytes()
    arrays = {n: np.frombuffer(blob, np.dtype(s["dtype"]).newbyteorder("<"), int(np.prod(s["shape"])),
                               s["offset"]).reshape(s["shape"]) for n, s in meta["arrays"].items()}
    return meta, arrays


# ----------------------------------------------------------------------------- sets
def export_annotation(stem: str, body, device) -> tuple[dict, dict, dict]:
    ho = BVR / "out" / stem / "human_optim"
    geo = BVR / "out" / stem / "geometry"
    kd = np.load(ho / "kindyn_1.npz", allow_pickle=True)
    cd = np.load(ho / "contacts_1.npz", allow_pickle=True)
    tf = np.load(geo / "transform.npz", allow_pickle=True)
    sc = np.load(geo / "scene.npz", allow_pickle=True)
    slot_parent, joints, slot_vertex = slot_fold()
    assert [str(x) for x in kd["contact_frame_names"]] == list(contact_set("frames35").slot_names)
    assert np.array_equal(cd["frame_vertex"], slot_vertex)

    n = kd["q"].shape[1]
    valid = kd["valid_mask"][0].astype(bool)
    down = np.asarray(kd["gravity_world"], np.float64)
    R = world_rotation(down)
    betas_rows = np.repeat(np.asarray(kd["betas"], np.float32), n, 0)
    posed = pose_body(body, betas_rows, np.asarray(kd["q"][0], np.float32), valid, None, R, device)
    bw_n = float(kd["total_mass"][0]) * G

    ff = np.nan_to_num(kd["frame_forces"][0].astype(np.float64)) @ R.T / bw_n   # (N, 35, 3) body weights
    fc = kd["frame_contact"][0].astype(bool)
    limb_force = np.stack([ff[:, slot_parent == j].sum(1) for j in joints], 1)
    limb_contact = np.stack([fc[:, slot_parent == j].any(1) for j in joints], 1).astype(np.float32)
    # cross-check: BVR's joint-level forces folded the same way (fingers -> wrist, toes -> ankle)
    cfw = np.nan_to_num(kd["contact_forces_world"][0].astype(np.float64)) @ R.T / bw_n
    jfold = np.asarray([FOLD_JOINT.get(body22_parent(j), body22_parent(j)) for j in range(52)])
    alt = np.stack([cfw[:, jfold == j].sum(1) for j in joints], 1)
    fold_diff = float(np.abs(alt - limb_force).max())

    extr = np.asarray(tf["extrinsics"], np.float64)
    wfc = np.linalg.inv(extr)
    fov = np.asarray(tf["fov_y_deg"], np.float64)
    size = (int(tf["image_width"]), int(tf["image_height"]))

    # scene: the confidence-gated cloud the BVR viewer shows, capped at MAX_SCENE_POINTS
    pts, col = np.asarray(sc["points"], np.float64), np.asarray(sc["colors"], np.uint8)
    conf = np.asarray(sc["confidence"])
    keep = conf >= np.quantile(conf, SCENE_CONF_DROP_Q)
    pts, col = pts[keep], col[keep]
    if len(pts) > MAX_SCENE_POINTS:
        pick = np.sort(np.random.default_rng(0).choice(len(pts), MAX_SCENE_POINTS, replace=False))
        pts, col = pts[pick], col[pick]
    src = {"frame_indices": np.asarray(tf["frame_indices"], np.int64), "valid": valid, "R": R,
           "down": down, "bw_n": bw_n, "fold_diff": fold_diff, "fps_src": float(kd["fps"]),
           "video": BVR / "data" / f"{stem}.mp4", "floor_y": None,
           "fk_vs_file": float(np.abs(posed["pos"][valid] - kd["joints_world"][0][valid] @ R.T).max()),
           "stored_bw_n": float(cd["body_weight_n"][0])}
    extra = {"limb_force": limb_force, "limb_contact": limb_contact, "frame_contact": fc,
             "wfc": wfc, "fov": fov, "size": size, "scene": (pts @ R.T, col), "static": bool(tf["static_camera"])}
    return posed, src, extra


def export_prediction(stem: str, body, device) -> tuple[dict, dict, dict]:
    tree = WILD / "out" / stem
    pdir = tree / "predictions" / PRED_DIR
    pr = np.load(pdir / "smplx.npz", allow_pickle=True)
    fs = np.load(pdir / "forces_sup.npz", allow_pickle=True)
    ct = np.load(pdir / "contacts.npz", allow_pickle=True)
    tf = np.load(tree / "geometry" / "transform.npz", allow_pickle=True)
    slot_parent, joints, slot_vertex = slot_fold()
    assert [str(x) for x in fs["limbs"]] == list(contact_set("frames35").slot_names)

    dump = load_dump(tree, PRED_DIR)                   # render_wild_3d's own world + fold
    up = dump["up"]
    R = world_rotation(-up)
    covered = dump["covered"][0]
    wfc = dump["world_from_cam"]
    posed = pose_body(body, np.asarray(pr["betas"][0], np.float32), np.asarray(pr["q_cam"][0], np.float32),
                      covered, wfc, R, device)
    limb_force = np.nan_to_num(dump["force_world"][0].astype(np.float64)) @ R.T      # (N, L, 3) body weights
    probs = np.nan_to_num(fs["contact_probs"][0].astype(np.float64))
    limb_contact = np.stack([probs[:, slot_parent == j].max(1) for j in joints], 1).astype(np.float32)
    frame_contact = np.asarray(ct["contacts"][0], bool) & covered[:, None]
    assert np.allclose(ct["probs"][0][covered], fs["contact_probs"][0][covered], atol=1e-6)

    # render_wild_3d's floor: the lowest ankle / toe joint over covered rows, minus 1 cm
    floor_y = float(posed["pos"][covered][:, list(FEET_JOINTS), 1].min()) - FLOOR_MARGIN_M
    # render_wild_3d draws from the file's joints_world; ours come from the exact FK — compare
    fk_vs_file = float(np.abs(posed["pos"][covered] - pr["joints_world"][0][covered] @ R.T).max())
    anchor_w = np.einsum("nij,nkj->nki", wfc[:, :3, :3], fs["anchor_cam"][0]) + wfc[:, None, :3, 3]
    anchor_vs_vertex = float(np.nanmax(np.linalg.norm(
        anchor_w[covered] @ R.T - posed["verts"][covered][:, slot_vertex], axis=-1)))
    intr = np.asarray(tf["intrinsics_px_orig"], np.float64)
    size = (int(tf["image_width"]), int(tf["image_height"]))
    fov = np.degrees(2 * np.arctan(size[1] / (2 * intr[:, 1, 1])))
    src = {"frame_indices": np.asarray(pr["frame_indices"], np.int64), "valid": covered, "R": R,
           "down": -up, "bw_n": None, "fps_src": float(pr["fps"]), "floor_y": floor_y,
           "video": WILD / "videos" / f"{stem}.mp4", "fk_vs_file": fk_vs_file,
           "anchor_vs_vertex": anchor_vs_vertex,
           "gravity_spread_deg": float(np.degrees(np.arccos(np.clip(
               (lambda g: (g / np.linalg.norm(g, axis=1, keepdims=True)) @ (-up))(
                   pr["gravity_world"][0][covered].astype(np.float64)), -1, 1))).max())}
    extra = {"limb_force": limb_force, "limb_contact": limb_contact, "frame_contact": frame_contact,
             "wfc": wfc, "fov": fov, "size": size, "scene": None, "static": bool(tf["static_camera"])}
    return posed, src, extra


def export_clip(set_name: str, stem: str, body, device, out_root: Path) -> dict:
    fn = export_annotation if set_name == "annotation" else export_prediction
    posed, src, extra = fn(stem, body, device)
    slot_parent, joints, slot_vertex = slot_fold()
    valid, R = src["valid"], src["R"]
    n = len(valid)
    names = list(body.structure.joint_names)
    parents = [int(p) for p in body.structure.parents]

    joint_pos = fill_invalid(posed["pos"], valid)
    joint_quat = fill_invalid(posed["quat"], valid)
    verts_rows = fill_invalid(posed["verts"], valid)
    limb_anchor = anchor_points(joint_pos[None], joints)[0]                   # (N, L, 3)
    frame_pts = verts_rows[:, slot_vertex]                                     # (N, 35, 3)
    limb_force = np.where(valid[:, None, None], extra["limb_force"], 0.0)
    limb_contact = np.where(valid[:, None], extra["limb_contact"], 0.0)
    frame_contact = extra["frame_contact"] & valid[:, None]

    wfc = extra["wfc"]
    cam_pos = wfc[:, :3, 3] @ R.T
    cam_quat = mat_to_quat(R @ wfc[:, :3, :3])
    pelvis = joint_pos[valid][:, 0]
    center = pelvis.mean(0)
    radius = max(1.0, 1.1 * float(np.linalg.norm(joint_pos[valid] - center, axis=-1).max()))
    video = probe_video(src["video"])
    has_scene = extra["scene"] is not None

    arrays = {
        "verts": posed["v_rest"].astype(np.float32),
        "faces": body.structure.faces.cpu().numpy().astype(np.uint32),
        "rest_joints": posed["j_rest"].astype(np.float32),
        "skin_idx": posed["skin_idx"],
        "skin_w": posed["skin_w"],
        "joint_quat": joint_quat.astype(np.float32),
        "joint_pos": joint_pos.astype(np.float32),
        "valid": valid.astype(np.uint8),
        "limb_anchor": limb_anchor.astype(np.float32),
        "limb_force": limb_force.astype(np.float32),
        "limb_contact": limb_contact.astype(np.float32),
        "frame_pts": frame_pts.astype(np.float32),
        "frame_contact": frame_contact.astype(np.uint8),
    }
    if has_scene:
        arrays["scene_points"] = extra["scene"][0].astype(np.float32)
        arrays["scene_colors"] = extra["scene"][1].astype(np.uint8)
    arrays["cam_pos"] = cam_pos.astype(np.float32)
    arrays["cam_quat"] = cam_quat.astype(np.float32)

    bw_n = src["bw_n"] if src["bw_n"] is not None else None
    meta = {
        "set": set_name, "stem": stem,
        "fps": video["fps"],
        "n_frames": n,
        "frame_indices": [int(i) for i in src["frame_indices"]],
        "video": video,
        "n_verts": int(arrays["verts"].shape[0]), "n_faces": int(arrays["faces"].shape[0]),
        "n_joints": 52, "n_limbs": int(len(joints)), "n_contact_frames": int(len(slot_vertex)),
        "joint_names": names, "parents": parents,
        "limbs": [names[int(j)] for j in joints],
        "limb_joints": [int(j) for j in joints],
        "contact_frame_names": list(contact_set("frames35").slot_names),
        "body_weight_n": None if bw_n is None else round(bw_n, 3),
        "force_units": "body_weight",
        "arrow_m_per_bw": float(ARROW_M_PER_BW_3D),
        "floor_y": None if src["floor_y"] is None else round(src["floor_y"], 5),
        "center": [round(float(x), 5) for x in center], "radius": round(radius, 4),
        "camera": {"fov_y_deg": round(float(np.median(extra["fov"])), 4),
                   "aspect": round(extra["size"][0] / extra["size"][1], 6)},
        "has_scene": has_scene,
        "has_camera_path": not extra["static"],
    }
    out_dir = out_root / set_name / stem
    nbytes = write_bundle(out_dir, meta, arrays)

    # ------------------------------------------------------------------ self-checks
    meta_r, arr = read_back(out_dir)
    vrows = np.flatnonzero(valid)
    check_rows = vrows[np.linspace(0, len(vrows) - 1, N_CHECK_ROWS).round().astype(int)]
    lbs_err = [float(np.linalg.norm(lbs(arr, r) - posed["verts"][r], axis=-1).max()) for r in check_rows]
    lbs_mean = [float(np.linalg.norm(lbs(arr, r) - posed["verts"][r], axis=-1).mean()) for r in check_rows]
    g_new = R @ (src["down"] / np.linalg.norm(src["down"]))
    mags = np.linalg.norm(arr["limb_force"], axis=-1)
    on = mags[arr["valid"].astype(bool)]
    report = {
        "set": set_name, "stem": stem, "N": n, "M": int(arr["scene_points"].shape[0]) if has_scene else 0,
        "bytes": nbytes, "lbs_max_err_m": max(lbs_err), "lbs_mean_err_m": float(np.mean(lbs_mean)),
        "valid_frac": float(valid.mean()),
        "gravity_err": float(np.abs(g_new - np.array([0.0, -1.0, 0.0])).max()),
        "g_new": np.round(g_new, 9).tolist(),
        "fk_vs_file_joints_m": src["fk_vs_file"],
        "limbs": meta["limbs"],
        "force_bw_max": float(on.max()), "force_bw_p99": float(np.percentile(on, 99)),
        "force_bw_nonzero_min": float(on[on > 1e-6].min()) if (on > 1e-6).any() else 0.0,
        "sum_force_bw_median_contact_rows": float(np.median(np.linalg.norm(
            arr["limb_force"][arr["valid"].astype(bool)].sum(1), axis=-1))),
        "video": video, "fps_src": src["fps_src"],
        "frame_indices_identity": bool(np.array_equal(src["frame_indices"], np.arange(n))),
        "tilt_from_rx180_deg": float(np.degrees(np.arccos(np.clip((np.trace(R @ RX180.T) - 1) / 2, -1, 1)))),
        "center": meta["center"], "radius": meta["radius"], "has_camera_path": meta["has_camera_path"],
        "camera": meta["camera"],
    }
    if set_name == "annotation":
        report["fold_vs_joint_forces_bw"] = src["fold_diff"]
        report["body_weight_n"] = bw_n
        report["contacts_1_body_weight_n"] = src["stored_bw_n"]
    else:
        fy = arr["joint_pos"][arr["valid"].astype(bool)][:, list(FEET_JOINTS), 1]
        allv = posed["verts"][valid]
        report["floor_y"] = meta["floor_y"]
        report["min_foot_joint_above_floor_m"] = float(fy.min() - meta["floor_y"])
        report["min_joint_above_floor_m"] = float(arr["joint_pos"][arr["valid"].astype(bool)][..., 1].min()
                                                  - meta["floor_y"])
        report["min_vertex_above_floor_m"] = float(allv[..., 1].min() - meta["floor_y"])
        report["frac_rows_vertex_below_floor"] = float((allv[..., 1].min(1) < meta["floor_y"]).mean())
        report["anchor_cam_vs_frame_vertex_m"] = src["anchor_vs_vertex"]
        report["gravity_row_spread_deg_max"] = src["gravity_spread_deg"]
    assert report["gravity_err"] < 1e-6, report["gravity_err"]
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HOME / "pact_web_export" / "viewer")
    ap.add_argument("--only", nargs="*", default=None, help="stems to export (default: all six)")
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    body = load_body(device)
    reports = []
    for set_name, stems in SETS.items():
        for stem in stems:
            if args.only and stem not in args.only:
                continue
            print(f"[{set_name}] {stem}", flush=True)
            rep = export_clip(set_name, stem, body, device, args.out)
            print(json.dumps(rep), flush=True)
            reports.append(rep)
    (args.out / "export_report.json").write_text(json.dumps(reports, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
