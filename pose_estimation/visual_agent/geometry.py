from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import trimesh


EPS = 1e-8


def make_transform(rotation: np.ndarray | Iterable[Iterable[float]], translation: np.ndarray | Iterable[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = orthonormalize_rotation(np.asarray(rotation, dtype=np.float64).reshape(3, 3))
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def transform_to_json(T: np.ndarray) -> dict[str, Any]:
    T = validate_transform(T)
    return {
        "T": T[:3, 3].round(6).tolist(),
        "R": T[:3, :3].round(6).tolist(),
        "matrix4": T.round(6).tolist(),
    }


def validate_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {T.shape}")
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = orthonormalize_rotation(T[:3, :3])
    out[:3, 3] = T[:3, 3]
    return out


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    u, _, vh = np.linalg.svd(R)
    out = u @ vh
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1.0
        out = u @ vh
    return out


def make_rotate(rx: float, ry: float, rz: float) -> np.ndarray:
    sin_x, sin_y, sin_z = np.sin(rx), np.sin(ry), np.sin(rz)
    cos_x, cos_y, cos_z = np.cos(rx), np.cos(ry), np.cos(rz)

    Rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_x, -sin_x], [0.0, sin_x, cos_x]],
        dtype=np.float64,
    )
    Ry = np.array(
        [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]],
        dtype=np.float64,
    )
    Rz = np.array(
        [[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return Rz @ Ry @ Rx


def rotation_from_axis_angle(axis: str, angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    axis = axis.strip().lower()
    if axis == "x":
        return make_rotate(angle, 0.0, 0.0)
    if axis == "y":
        return make_rotate(0.0, angle, 0.0)
    if axis == "z":
        return make_rotate(0.0, 0.0, angle)
    raise ValueError(f"Unknown rotation axis: {axis}")


def rotation_from_vector_axis_angle(axis: np.ndarray | Iterable[float], angle_deg: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis_arr))
    if norm < EPS:
        return np.eye(3, dtype=np.float64)
    axis_arr = axis_arr / norm
    angle = math.radians(float(angle_deg))
    ax, ay, az = axis_arr
    skew = np.array(
        [
            [0.0, -az, ay],
            [az, 0.0, -ax],
            [-ay, ax, 0.0],
        ],
        dtype=np.float64,
    )
    return orthonormalize_rotation(
        np.eye(3, dtype=np.float64) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def rotation_matrix_to_axis_angle(rotation: np.ndarray | Iterable[Iterable[float]]) -> tuple[np.ndarray, float]:
    R = orthonormalize_rotation(np.asarray(rotation, dtype=np.float64).reshape(3, 3))
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos_theta)
    if angle < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0

    if abs(math.pi - angle) < 1e-5:
        eigvals, eigvecs = np.linalg.eig(R)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        axis = np.real(eigvecs[:, idx]).astype(np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < EPS:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis /= norm
        return axis, math.degrees(angle)

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * math.sin(angle))
    axis /= max(float(np.linalg.norm(axis)), EPS)
    return axis, math.degrees(angle)


def scale_rotation_matrix(
    rotation_matrix: np.ndarray | Iterable[Iterable[float]],
    scale: float,
    max_angle_deg: float,
) -> tuple[np.ndarray, float, float, bool]:
    axis, angle_deg = rotation_matrix_to_axis_angle(rotation_matrix)
    scaled_angle = float(angle_deg) * float(scale)
    clipped_angle = float(np.clip(scaled_angle, -abs(float(max_angle_deg)), abs(float(max_angle_deg))))
    return (
        rotation_from_vector_axis_angle(axis, clipped_angle),
        float(angle_deg),
        clipped_angle,
        abs(clipped_angle - scaled_angle) > 1e-8,
    )


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    T = validate_transform(T)
    pts = np.asarray(points, dtype=np.float64)
    return (T[:3, :3] @ pts.T).T + T[:3, 3][None, :]


def bounds_for_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("Cannot compute bounds for empty points")
    return pts.min(axis=0), pts.max(axis=0)


def transform_bbox_aabb(bmin: np.ndarray, bmax: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array(
        [
            [bmin[0], bmin[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]],
            [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmax[1], bmax[2]],
            [bmax[0], bmin[1], bmin[2]],
            [bmax[0], bmin[1], bmax[2]],
            [bmax[0], bmax[1], bmin[2]],
            [bmax[0], bmax[1], bmax[2]],
        ],
        dtype=np.float64,
    )
    world = transform_points(corners, T)
    return world.min(axis=0), world.max(axis=0)


def visual_arrow_length_from_bounds(bmin: np.ndarray, bmax: np.ndarray) -> float:
    extents = np.asarray(bmax, dtype=np.float64) - np.asarray(bmin, dtype=np.float64)
    bbox_min = float(np.min(extents))
    bbox_max = float(np.max(extents))
    if bbox_min > 0.0:
        return max(bbox_min if 2.0 * bbox_min <= bbox_max else bbox_min / 2.0, 1e-3)
    return bbox_max if bbox_max > 0.0 else 1.0


def translation_scale_from_bounds(bmin: np.ndarray, bmax: np.ndarray) -> float:
    extents = np.asarray(bmax, dtype=np.float64) - np.asarray(bmin, dtype=np.float64)
    bbox_min = float(np.min(extents))
    bbox_max = float(np.max(extents))
    if bbox_min > 0.0:
        # Match VLM-Pose: one normalized translation unit is the target
        # object's coordinate-arrow length derived from its world-axis AABB.
        return max(bbox_min if 2.0 * bbox_min <= bbox_max else bbox_min / 2.0, 1e-3)
    return bbox_max if bbox_max > 0.0 else 1.0


def arrow_length_from_bounds(bmin: np.ndarray, bmax: np.ndarray) -> float:
    return visual_arrow_length_from_bounds(bmin, bmax)


def set_mesh_color(mesh: trimesh.Trimesh, rgb: tuple[int, int, int]) -> None:
    rgba = np.array([rgb[0], rgb[1], rgb[2], 255], dtype=np.uint8)
    mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))


def darken_color(rgb: tuple[int, int, int], ratio: float = 0.35) -> tuple[int, int, int]:
    color = np.asarray(rgb, dtype=np.float64)
    out = np.clip(color * (1.0 - ratio), 0, 255).astype(np.uint8)
    return int(out[0]), int(out[1]), int(out[2])


def build_arrow(start: np.ndarray, direction: np.ndarray, length: float, color: tuple[int, int, int]) -> trimesh.Trimesh:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), EPS)
    length = max(float(length), 1e-5)
    head_len = max(length * 0.22, 1e-5)
    body_len = max(length - head_len, 1e-5)
    radius = max(length * 0.018, 1e-5)
    head_radius = radius * 2.0
    start = np.asarray(start, dtype=np.float64)
    body_end = start + direction * body_len

    body = trimesh.creation.cylinder(radius=radius, segment=np.vstack([start, body_end]), sections=20)
    set_mesh_color(body, color)

    head = trimesh.creation.cone(radius=head_radius, height=head_len, sections=20)
    align = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), direction)
    if align is None:
        align = np.eye(4)
    head.apply_transform(align)
    head.apply_translation(body_end)
    set_mesh_color(head, darken_color(color))
    return trimesh.util.concatenate([body, head])


def _axis_color_rgb255(color: tuple[int, int, int] | tuple[float, float, float]) -> tuple[int, int, int]:
    values = [float(component) for component in color]
    if max(values) <= 1.0:
        values = [component * 255.0 for component in values]
    return tuple(int(np.clip(round(component), 0, 255)) for component in values)  # type: ignore[return-value]


def build_coordinate_arrows(
    bounds: tuple[np.ndarray, np.ndarray],
    arrow_len: float | None = None,
    colors: tuple[
        tuple[int, int, int] | tuple[float, float, float],
        tuple[int, int, int] | tuple[float, float, float],
        tuple[int, int, int] | tuple[float, float, float],
    ]
    | None = None,
) -> list[trimesh.Trimesh]:
    bmin, bmax = bounds
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)
    center = (bmin + bmax) * 0.5
    length = float(arrow_len if arrow_len is not None else visual_arrow_length_from_bounds(bmin, bmax))
    axis_colors = colors or ((255, 0, 0), (0, 255, 0), (0, 0, 255))
    return [
        build_arrow(
            center,
            np.array([1.0, 0.0, 0.0]),
            length,
            _axis_color_rgb255(axis_colors[0]),
        ),
        build_arrow(
            center,
            np.array([0.0, 1.0, 0.0]),
            length,
            _axis_color_rgb255(axis_colors[1]),
        ),
        build_arrow(
            center,
            np.array([0.0, 0.0, 1.0]),
            length,
            _axis_color_rgb255(axis_colors[2]),
        ),
    ]


def apply_incremental_update(
    current_T: np.ndarray,
    local_bounds: tuple[np.ndarray, np.ndarray],
    translation_units: Iterable[float],
    arrow_len: float,
    axis: str,
    angle_deg: float,
    rotation_matrix: np.ndarray | Iterable[Iterable[float]] | None = None,
) -> np.ndarray:
    current_T = validate_transform(current_T)
    translation = np.asarray(translation_units, dtype=np.float64).reshape(3) * float(arrow_len)
    rot = (
        orthonormalize_rotation(np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3))
        if rotation_matrix is not None
        else rotation_from_axis_angle(axis, float(angle_deg))
    )
    bmin_w, bmax_w = transform_bbox_aabb(local_bounds[0], local_bounds[1], current_T)
    center = (bmin_w + bmax_w) * 0.5

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot @ current_T[:3, :3]
    out[:3, 3] = rot @ (current_T[:3, 3] - center) + center + translation
    return validate_transform(out)


def transform_distance(T_a: np.ndarray, T_b: np.ndarray, translation_scale: float = 1.0) -> float:
    T_a = validate_transform(T_a)
    T_b = validate_transform(T_b)
    t = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) / max(float(translation_scale), EPS))
    r_delta = T_a[:3, :3].T @ T_b[:3, :3]
    cos_theta = np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)
    r = float(np.arccos(cos_theta) / np.pi)
    return t + r
