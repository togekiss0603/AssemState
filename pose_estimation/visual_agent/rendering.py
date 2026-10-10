from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PYOPENGL_PLATFORM", os.getenv("PYOPENGL_PLATFORM", "egl"))

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    matplotlib = None
    plt = None
    PolyCollection = None
    Poly3DCollection = None

from .geometry import (
    bounds_for_points,
    build_coordinate_arrows,
    translation_scale_from_bounds,
    transform_points,
    validate_transform,
    visual_arrow_length_from_bounds,
)


@dataclass
class PartMesh:
    part_id: str
    obj_path: Path
    meshes: list[trimesh.Trimesh]
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    centroid: np.ndarray
    extents: np.ndarray

    @property
    def vertices(self) -> np.ndarray:
        return np.concatenate([np.asarray(mesh.vertices, dtype=np.float64) for mesh in self.meshes], axis=0)


@dataclass(frozen=True)
class ViewSpec:
    name: str
    elev: float
    azim: float
    reason: str

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "elev": round(float(self.elev), 4),
            "azim": round(float(self.azim), 4),
            "reason": self.reason,
        }


def load_part_meshes(parts_dir: str | Path) -> dict[str, PartMesh]:
    root = Path(parts_dir)
    out: dict[str, PartMesh] = {}
    for obj_path in sorted(root.glob("*.obj")):
        part_id = normalize_part_id(obj_path.stem)
        meshes = load_obj_as_meshes(obj_path)
        vertices = np.concatenate([np.asarray(mesh.vertices, dtype=np.float64) for mesh in meshes], axis=0)
        bmin, bmax = bounds_for_points(vertices)
        out[part_id] = PartMesh(
            part_id=part_id,
            obj_path=obj_path,
            meshes=meshes,
            bbox_min=bmin,
            bbox_max=bmax,
            centroid=vertices.mean(axis=0),
            extents=bmax - bmin,
        )
    if not out:
        raise FileNotFoundError(f"No OBJ files under {root}")
    return out


def load_obj_as_meshes(obj_path: str | Path) -> list[trimesh.Trimesh]:
    loaded = trimesh.load(obj_path, force="scene", process=False)
    meshes: list[trimesh.Trimesh] = []
    if isinstance(loaded, trimesh.Scene):
        for geom in loaded.dump():
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
                meshes.append(geom.copy())
    elif isinstance(loaded, trimesh.Trimesh) and len(loaded.vertices) > 0:
        meshes.append(loaded.copy())
    if not meshes:
        raise ValueError(f"No mesh geometry in {obj_path}")
    return meshes


def normalize_part_id(value: str) -> str:
    return str(int(value)) if str(value).isdigit() else str(value)


def normalize_degrees(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def angular_distance_deg(a: float, b: float) -> float:
    return abs(normalize_degrees(float(a) - float(b)))


PART_COLOR_PALETTE = [
    (0.62, 0.42, 0.25),
    (0.20, 0.42, 0.82),
    (0.62, 0.20, 0.22),
    (0.20, 0.55, 0.35),
    (0.70, 0.48, 0.18),
    (0.45, 0.30, 0.65),
]
MOVING_AXIS_COLORS = ((0.92, 0.08, 0.06), (0.05, 0.62, 0.20), (0.08, 0.26, 0.92))
FIXED_BASE_AXIS_COLORS = ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0))
MOVING_AXIS_COLOR_NAMES = ("red", "green", "blue")
FIXED_BASE_AXIS_COLOR_NAMES = ("cyan", "magenta", "yellow")
RENDER_RULER_TICK_STEP_PX = 128


def stable_part_color(part_id: str) -> tuple[float, float, float]:
    key = normalize_part_id(str(part_id))
    if key.isdigit():
        return PART_COLOR_PALETTE[int(key) % len(PART_COLOR_PALETTE)]
    return PART_COLOR_PALETTE[stable_string_index(key) % len(PART_COLOR_PALETTE)]


def stable_part_rgba(part_id: str) -> np.ndarray:
    rgb = np.asarray(stable_part_color(part_id), dtype=np.float64)
    rgba = np.concatenate([np.clip(rgb * 255.0, 0, 255), np.array([255.0])])
    return rgba.astype(np.uint8)


def stable_string_index(value: str) -> int:
    total = 0
    for char in value:
        total = (total * 131 + ord(char)) & 0xFFFFFFFF
    return total


class PosedSceneRenderer:
    def __init__(
        self,
        width: int = 768,
        height: int = 768,
        lens: float = 70.0,
        sensor_width_mm: float = 36.0,
        camera_dist_ratio: float = 2.2,
        camera_distance_scale: float = 1.15,
        camera_frame_padding: float = 1.25,
        light_intensity: float = 4.0,
        show_ground_plane: bool = True,
        ground_y: float = -0.05,
        ground_alpha: float = 0.38,
        ground_parts_front: bool = True,
        axis_length_scale: float = 1.0,
        show_workcell: bool = False,
        background_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        ambient_light: tuple[float, float, float] = (0.58, 0.58, 0.58),
    ):
        try:
            import pyrender
        except ImportError:
            pyrender = None

        self.pyrender = pyrender
        self.RenderFlags = pyrender.RenderFlags if pyrender is not None else None
        self.width = int(width)
        self.height = int(height)
        self.lens = float(lens)
        self.sensor_width_mm = float(sensor_width_mm)
        self.camera_dist_ratio = float(camera_dist_ratio)
        self.camera_distance_scale = max(0.1, float(camera_distance_scale))
        self.camera_frame_padding = max(0.55, float(camera_frame_padding))
        self.light_intensity = float(light_intensity)
        self.show_ground_plane = bool(show_ground_plane)
        self.ground_y = float(ground_y)
        self.ground_alpha = float(np.clip(ground_alpha, 0.0, 1.0))
        self.ground_parts_front = bool(ground_parts_front)
        self.axis_length_scale = max(0.1, float(axis_length_scale))
        self.show_workcell = bool(show_workcell)
        self.background_color = [float(x) for x in background_color]
        self.ambient_light = [float(x) for x in ambient_light]
        self.renderer = (
            pyrender.OffscreenRenderer(viewport_width=self.width, viewport_height=self.height)
            if pyrender is not None
            else None
        )

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.delete()

    def render_scene_views(
        self,
        parts: dict[str, PartMesh],
        poses: dict[str, np.ndarray],
        output_dir: str | Path,
        prefix: str,
        num_views: int,
        focus_ids: Iterable[str] | None = None,
        moving_part: str | None = None,
        show_arrows: bool = False,
        fixed_axis_part: str | None = None,
        show_fixed_base_axes: bool = False,
        view_specs: list[ViewSpec] | None = None,
        frame_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[Path]:
        specs = view_specs if view_specs is not None else self.default_view_specs(num_views)
        if self.renderer is None:
            return self._render_scene_views_matplotlib(
                parts=parts,
                poses=poses,
                output_dir=output_dir,
                prefix=prefix,
                num_views=len(specs),
                focus_ids=focus_ids,
                moving_part=moving_part,
                show_arrows=show_arrows,
                fixed_axis_part=fixed_axis_part,
                show_fixed_base_axes=show_fixed_base_axes,
                view_specs=specs,
                frame_bounds=frame_bounds,
            )
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        focus = list(focus_ids) if focus_ids is not None else list(parts.keys())
        bmin, bmax = frame_bounds if frame_bounds is not None else self.compute_scene_bounds(parts, poses, focus)
        center = (bmin + bmax) * 0.5
        paths: list[Path] = []
        for idx, spec in enumerate(specs):
            size = self.view_size_for_bounds(bmin, bmax, spec)
            camera_pose = self.view_pose_from_spec(center, size, spec)
            scene = self._build_scene(
                parts,
                poses,
                moving_part,
                show_arrows,
                fixed_axis_part=fixed_axis_part,
                show_fixed_base_axes=show_fixed_base_axes,
            )
            cam = self._make_camera(size)
            cam_node = scene.add(cam, pose=camera_pose)
            self._add_lights(scene, center, size)
            color, _ = self.renderer.render(
                scene,
                flags=self.RenderFlags.RGBA | self.RenderFlags.SHADOWS_DIRECTIONAL | self.RenderFlags.SKIP_CULL_FACES,
            )
            scene.remove_node(cam_node)
            color = add_image_label(color, view_annotation_title(prefix, idx + 1, spec, self.lens, camera_pose))
            path = output / f"{prefix}_view_{idx + 1:02d}.png"
            write_image(path, color)
            paths.append(path)
        return paths

    def _render_scene_views_matplotlib(
        self,
        parts: dict[str, PartMesh],
        poses: dict[str, np.ndarray],
        output_dir: str | Path,
        prefix: str,
        num_views: int,
        focus_ids: Iterable[str] | None = None,
        moving_part: str | None = None,
        show_arrows: bool = False,
        fixed_axis_part: str | None = None,
        show_fixed_base_axes: bool = False,
        view_specs: list[ViewSpec] | None = None,
        frame_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[Path]:
        if plt is None or Poly3DCollection is None:
            raise ImportError("Rendering requires either pyrender or matplotlib in the active Python environment.")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        focus = list(focus_ids) if focus_ids is not None else list(parts.keys())
        bmin, bmax = frame_bounds if frame_bounds is not None else self.compute_scene_bounds(parts, poses, focus)
        y_up_matplotlib = True
        bmin_vis = self._data_to_matplotlib_vis(bmin.reshape(1, 3))[0]
        bmax_vis = self._data_to_matplotlib_vis(bmax.reshape(1, 3))[0]
        lo = np.minimum(bmin_vis, bmax_vis)
        hi = np.maximum(bmin_vis, bmax_vis)
        if self.show_ground_plane:
            floor_point = np.array([0.0, 0.0, self.ground_y], dtype=np.float64)
            lo = np.minimum(lo, floor_point)
            hi = np.maximum(hi, floor_point)
        center = (lo + hi) * 0.5
        max_range = max(
            float(np.max(hi - lo))
            * (0.78 if self.show_ground_plane else 0.65)
            * self.camera_distance_scale,
            1e-3,
        )
        specs = view_specs if view_specs is not None else self.default_view_specs(num_views)
        world_center = (bmin + bmax) * 0.5
        paths: list[Path] = []
        for idx, spec in enumerate(specs):
            path = output / f"{prefix}_view_{idx + 1:02d}.png"
            world_size = self.view_size_for_bounds(bmin, bmax, spec)
            camera_pose = self.view_pose_from_spec(world_center, world_size, spec)
            if self.show_ground_plane and self.ground_parts_front:
                self._render_matplotlib_ground_parts_front(
                    path=path,
                    parts=parts,
                    poses=poses,
                    moving_part=moving_part,
                    show_arrows=show_arrows,
                    fixed_axis_part=fixed_axis_part,
                    show_fixed_base_axes=show_fixed_base_axes,
                    spec=spec,
                    center=center,
                    max_range=max_range,
                    data_bounds=(bmin, bmax),
                )
            else:
                fig = plt.figure(figsize=(self.width / 100.0, self.height / 100.0), dpi=100)
                ax = fig.add_subplot(111, projection="3d")
                self._setup_matplotlib_axes(ax, spec, center, max_range)
                if self.show_ground_plane:
                    self._draw_matplotlib_ground(ax, bmin, bmax)
                self._draw_matplotlib_scene(
                    ax,
                    parts,
                    poses,
                    moving_part,
                    show_arrows,
                    fixed_axis_part=fixed_axis_part,
                    show_fixed_base_axes=show_fixed_base_axes,
                    y_up_to_z_up=y_up_matplotlib,
                )
                fig.tight_layout(pad=0)
                fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0)
                plt.close(fig)
            img = np.asarray(Image.open(path).convert("RGB"))
            img = add_image_label(img, view_annotation_title(prefix, idx + 1, spec, self.lens, camera_pose))
            write_image(path, img)
            paths.append(path)
        return paths

    def _render_matplotlib_ground_parts_front(
        self,
        path: Path,
        parts: dict[str, PartMesh],
        poses: dict[str, np.ndarray],
        moving_part: str | None,
        show_arrows: bool,
        fixed_axis_part: str | None,
        show_fixed_base_axes: bool,
        spec: ViewSpec,
        center: np.ndarray,
        max_range: float,
        data_bounds: tuple[np.ndarray, np.ndarray],
    ) -> None:
        ground_fig = plt.figure(figsize=(self.width / 100.0, self.height / 100.0), dpi=100)
        ground_ax = ground_fig.add_subplot(111, projection="3d")
        self._setup_matplotlib_axes(ground_ax, spec, center, max_range)
        self._draw_matplotlib_ground(ground_ax, *data_bounds)
        ground_img = self._matplotlib_figure_to_image(ground_fig, transparent=False)
        plt.close(ground_fig)

        parts_fig = plt.figure(figsize=(self.width / 100.0, self.height / 100.0), dpi=100)
        parts_fig.patch.set_alpha(0.0)
        parts_ax = parts_fig.add_subplot(111, projection="3d")
        parts_ax.patch.set_alpha(0.0)
        self._setup_matplotlib_axes(parts_ax, spec, center, max_range)
        self._draw_matplotlib_scene(
            parts_ax,
            parts,
            poses,
            moving_part,
            show_arrows,
            fixed_axis_part=fixed_axis_part,
            show_fixed_base_axes=show_fixed_base_axes,
            y_up_to_z_up=True,
        )
        parts_img = self._matplotlib_figure_to_image(parts_fig, transparent=True)
        plt.close(parts_fig)

        ground = ground_img.convert("RGBA")
        foreground = parts_img.convert("RGBA")
        if foreground.size != ground.size:
            foreground = foreground.resize(ground.size, Image.LANCZOS)
        Image.alpha_composite(ground, foreground).convert("RGB").save(path)

    @staticmethod
    def _matplotlib_figure_to_image(fig, transparent: bool) -> Image.Image:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",
            facecolor=(1, 1, 1, 0) if transparent else "white",
            transparent=transparent,
            pad_inches=0,
        )
        buffer.seek(0)
        return Image.open(buffer).convert("RGBA")

    @staticmethod
    def _setup_matplotlib_axes(ax, spec: ViewSpec, center: np.ndarray, max_range: float) -> None:
        ax.view_init(elev=spec.elev, azim=spec.azim)
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()

    @staticmethod
    def _data_to_matplotlib_vis(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        out = np.empty_like(points)
        out[..., 0] = points[..., 0]
        out[..., 1] = points[..., 2]
        out[..., 2] = points[..., 1]
        return out

    def _draw_matplotlib_ground(self, ax, bmin_data: np.ndarray, bmax_data: np.ndarray) -> None:
        x0, x1 = float(bmin_data[0]), float(bmax_data[0])
        z0, z1 = float(bmin_data[2]), float(bmax_data[2])
        sx, sz = x1 - x0, z1 - z0
        pad = max(max(sx, sz) * 0.22, 0.2)
        x0 -= pad
        x1 += pad
        z0 -= pad
        z1 += pad
        plane = [
            [x0, z0, self.ground_y],
            [x1, z0, self.ground_y],
            [x1, z1, self.ground_y],
            [x0, z1, self.ground_y],
        ]
        ax.add_collection3d(
            Poly3DCollection(
                [plane],
                facecolors="#e3d2ad",
                edgecolors="#a99163",
                linewidths=0.8,
                alpha=self.ground_alpha,
            )
        )
        grid_color = (0.47, 0.39, 0.28, 0.30)
        for x in np.linspace(x0, x1, 9):
            ax.plot(
                [x, x],
                [z0, z1],
                [self.ground_y + 0.001, self.ground_y + 0.001],
                color=grid_color,
                linewidth=0.45,
            )
        for z in np.linspace(z0, z1, 9):
            ax.plot(
                [x0, x1],
                [z, z],
                [self.ground_y + 0.001, self.ground_y + 0.001],
                color=grid_color,
                linewidth=0.45,
            )

    def _draw_matplotlib_scene(
        self,
        ax,
        parts: dict[str, PartMesh],
        poses: dict[str, np.ndarray],
        moving_part: str | None,
        show_arrows: bool,
        fixed_axis_part: str | None = None,
        show_fixed_base_axes: bool = False,
        y_up_to_z_up: bool = False,
    ) -> None:
        if self.show_workcell:
            bmin_raw, bmax_raw = self._raw_scene_bounds(parts, poses, parts.keys())
            for mesh, color, alpha in self._workcell_meshes_from_bounds(bmin_raw, bmax_raw):
                self._draw_matplotlib_mesh(ax, mesh, color, alpha, y_up_to_z_up=y_up_to_z_up)

        for part_id, part in sorted(parts.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
            pose = validate_transform(poses.get(part_id, np.eye(4)))
            color = stable_part_color(part_id)
            for mesh in part.meshes:
                vertices = transform_points(np.asarray(mesh.vertices, dtype=np.float64), pose)
                if y_up_to_z_up:
                    vertices = self._data_to_matplotlib_vis(vertices)
                faces = np.asarray(mesh.faces, dtype=np.int64)
                if len(faces) > 600:
                    stride = max(1, len(faces) // 600)
                    faces = faces[::stride]
                polys = [vertices[face] for face in faces]
                collection = Poly3DCollection(polys, facecolors=[color], edgecolors=(0.2, 0.2, 0.2, 0.18), linewidths=0.08, alpha=1.0)
                ax.add_collection3d(collection)
            center = transform_points(part.centroid.reshape(1, 3), pose)[0]
            if y_up_to_z_up:
                center = self._data_to_matplotlib_vis(center.reshape(1, 3))[0]

        if show_fixed_base_axes and fixed_axis_part is not None and fixed_axis_part in parts:
            self._draw_matplotlib_axes_for_part(
                ax,
                parts[fixed_axis_part],
                validate_transform(poses.get(fixed_axis_part, np.eye(4))),
                colors=FIXED_BASE_AXIS_COLOR_NAMES,
                labels=("BX", "BY", "BZ"),
                y_up_to_z_up=y_up_to_z_up,
                linewidth=1.8,
                fontsize=9,
            )

        if moving_part is not None and show_arrows and moving_part in parts:
            self._draw_matplotlib_axes_for_part(
                ax,
                parts[moving_part],
                validate_transform(poses.get(moving_part, np.eye(4))),
                colors=MOVING_AXIS_COLOR_NAMES,
                labels=("X", "Y", "Z"),
                y_up_to_z_up=y_up_to_z_up,
                linewidth=2.2,
                fontsize=10,
            )

    def _draw_matplotlib_axes_for_part(
        self,
        ax,
        part: PartMesh,
        pose: np.ndarray,
        colors: tuple[str, str, str],
        labels: tuple[str, str, str],
        y_up_to_z_up: bool,
        linewidth: float,
        fontsize: int,
    ) -> None:
        bmin_w, bmax_w = self.part_world_bounds(part, pose)
        center = (bmin_w + bmax_w) * 0.5
        arrow_len = translation_scale_from_bounds(part.bbox_min, part.bbox_max) * self.axis_length_scale
        starts = [center.copy(), center.copy(), center.copy()]
        dirs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        if y_up_to_z_up:
            starts = [self._data_to_matplotlib_vis(start.reshape(1, 3))[0] for start in starts]
            dirs = [self._data_to_matplotlib_vis(direction.reshape(1, 3))[0] for direction in dirs]
        for start, direction, color, label in zip(starts, dirs, colors, labels):
            ax.quiver(
                start[0],
                start[1],
                start[2],
                direction[0],
                direction[1],
                direction[2],
                length=arrow_len,
                color=color,
                linewidth=linewidth,
                arrow_length_ratio=0.18,
            )
            label_pos = start + direction * arrow_len * 1.10
            ax.text(
                label_pos[0],
                label_pos[1],
                label_pos[2],
                label,
                color=color,
                fontsize=fontsize,
                fontweight="bold",
            )

    def _draw_matplotlib_mesh(
        self,
        ax,
        mesh: trimesh.Trimesh,
        color: tuple[float, float, float],
        alpha: float,
        y_up_to_z_up: bool,
    ) -> None:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if y_up_to_z_up:
            vertices = self._data_to_matplotlib_vis(vertices)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(faces) > 900:
            faces = faces[:: max(1, len(faces) // 900)]
        polys = [vertices[face] for face in faces]
        collection = Poly3DCollection(
            polys,
            facecolors=[(*color, float(np.clip(alpha, 0.0, 1.0)))],
            edgecolors=(0.12, 0.12, 0.12, 0.12),
            linewidths=0.05,
            alpha=float(np.clip(alpha, 0.0, 1.0)),
        )
        ax.add_collection3d(collection)

    def _build_scene(
        self,
        parts: dict[str, PartMesh],
        poses: dict[str, np.ndarray],
        moving_part: str | None,
        show_arrows: bool,
        fixed_axis_part: str | None = None,
        show_fixed_base_axes: bool = False,
    ):
        scene = self.pyrender.Scene(bg_color=self.background_color, ambient_light=self.ambient_light)
        if self.show_ground_plane:
            ground = self._build_pyrender_ground_mesh(parts, poses)
            ground_material = self.pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.89, 0.82, 0.68, self.ground_alpha],
                metallicFactor=0.0,
                roughnessFactor=0.82,
                alphaMode="BLEND" if self.ground_alpha < 0.999 else "OPAQUE",
            )
            scene.add(self.pyrender.Mesh.from_trimesh(ground, material=ground_material, smooth=False), pose=np.eye(4))
        if self.show_workcell:
            bmin_raw, bmax_raw = self._raw_scene_bounds(parts, poses, parts.keys())
            for mesh, color, alpha in self._workcell_meshes_from_bounds(bmin_raw, bmax_raw):
                material = self.pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[float(color[0]), float(color[1]), float(color[2]), float(alpha)],
                    metallicFactor=0.0,
                    roughnessFactor=0.74,
                    alphaMode="BLEND" if alpha < 0.999 else "OPAQUE",
                )
                scene.add(self.pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False), pose=np.eye(4))
        for part_id, part in parts.items():
            pose = validate_transform(poses.get(part_id, np.eye(4)))
            rgba = stable_part_rgba(part_id)
            for mesh in part.meshes:
                colored_mesh = mesh.copy()
                colored_mesh.visual.face_colors = np.tile(rgba, (len(colored_mesh.faces), 1))
                pm = self.pyrender.Mesh.from_trimesh(colored_mesh, smooth=False)
                scene.add(pm, pose=pose)
        if show_fixed_base_axes and fixed_axis_part is not None and fixed_axis_part in parts:
            pose = validate_transform(poses.get(fixed_axis_part, np.eye(4)))
            part = parts[fixed_axis_part]
            bmin_w, bmax_w = self.part_world_bounds(part, pose)
            arrow_len = translation_scale_from_bounds(part.bbox_min, part.bbox_max) * self.axis_length_scale
            for arrow in build_coordinate_arrows(
                (bmin_w, bmax_w),
                arrow_len=arrow_len,
                colors=FIXED_BASE_AXIS_COLORS,
            ):
                scene.add(self.pyrender.Mesh.from_trimesh(arrow, smooth=False), pose=np.eye(4))
        if moving_part is not None and show_arrows and moving_part in parts:
            pose = validate_transform(poses.get(moving_part, np.eye(4)))
            part = parts[moving_part]
            bmin_w, bmax_w = self.part_world_bounds(part, pose)
            arrow_len = translation_scale_from_bounds(part.bbox_min, part.bbox_max) * self.axis_length_scale
            for arrow in build_coordinate_arrows(
                (bmin_w, bmax_w),
                arrow_len=arrow_len,
                colors=MOVING_AXIS_COLORS,
            ):
                scene.add(self.pyrender.Mesh.from_trimesh(arrow, smooth=False), pose=np.eye(4))
        return scene

    def _build_pyrender_ground_mesh(self, parts: dict[str, PartMesh], poses: dict[str, np.ndarray]) -> trimesh.Trimesh:
        bmin, bmax = self.compute_scene_bounds(parts, poses, parts.keys())
        x0, x1 = float(bmin[0]), float(bmax[0])
        z0, z1 = float(bmin[2]), float(bmax[2])
        sx, sz = x1 - x0, z1 - z0
        pad = max(max(sx, sz) * 0.22, 0.2)
        x0 -= pad
        x1 += pad
        z0 -= pad
        z1 += pad
        vertices = np.array(
            [
                [x0, self.ground_y, z0],
                [x1, self.ground_y, z0],
                [x1, self.ground_y, z1],
                [x0, self.ground_y, z1],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        alpha = int(round(self.ground_alpha * 255.0))
        mesh.visual.face_colors = np.tile(np.array([227, 210, 173, alpha], dtype=np.uint8), (len(faces), 1))
        return mesh

    def _make_camera(self, size: float):
        fx = (self.width * self.lens) / self.sensor_width_mm
        fy = (self.height * self.lens) / self.sensor_width_mm
        return self.pyrender.IntrinsicsCamera(
            fx=fx,
            fy=fy,
            cx=self.width * 0.5,
            cy=self.height * 0.5,
            znear=0.01,
            zfar=max(10.0, size * 12.0),
        )

    def _add_lights(self, scene, center: np.ndarray, size: float) -> None:
        offsets = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        ]
        for offset in offsets:
            light = self.pyrender.DirectionalLight(color=np.ones(3), intensity=self.light_intensity)
            pos = center + offset * max(size * 2.0, 1.0)
            scene.add(light, pose=self.look_at(pos, center))

    def view_pose(self, center: np.ndarray, size: float, idx: int, num_views: int) -> np.ndarray:
        angle = -90.0 + 360.0 * idx / max(1, num_views)
        az = np.radians(angle)
        dist = max(size * self.camera_dist_ratio, 0.5)
        eye = np.array([center[0] + dist * np.cos(az), center[1] + dist * 0.55, center[2] + dist * np.sin(az)])
        return self.look_at(eye, center, up=np.array([0.0, 1.0, 0.0]))

    def view_pose_from_spec(self, center: np.ndarray, size: float, spec: ViewSpec) -> np.ndarray:
        az = np.radians(float(spec.azim))
        elev = np.radians(float(spec.elev))
        dist = max(size * self.camera_dist_ratio, 0.5)
        eye = np.array(
            [
                center[0] + dist * np.cos(elev) * np.cos(az),
                center[1] + dist * np.sin(elev),
                center[2] + dist * np.cos(elev) * np.sin(az),
            ],
            dtype=np.float64,
        )
        return self.look_at(eye, center, up=np.array([0.0, 1.0, 0.0]))

    def view_size_for_bounds(self, bmin: np.ndarray, bmax: np.ndarray, spec: ViewSpec) -> float:
        """Choose a per-view camera scale that keeps the focus bbox inside frame."""
        bmin = np.asarray(bmin, dtype=np.float64)
        bmax = np.asarray(bmax, dtype=np.float64)
        center = (bmin + bmax) * 0.5
        base_size = max(float(np.linalg.norm(bmax - bmin)), 1.0)

        az = np.radians(float(spec.azim))
        elev = np.radians(float(spec.elev))
        view_dir = np.array(
            [
                np.cos(elev) * np.cos(az),
                np.sin(elev),
                np.cos(elev) * np.sin(az),
            ],
            dtype=np.float64,
        )
        view_dir /= max(float(np.linalg.norm(view_dir)), 1e-8)
        probe_eye = center + view_dir
        pose = self.look_at(probe_eye, center, up=np.array([0.0, 1.0, 0.0]))
        right = pose[:3, 0]
        up = pose[:3, 1]
        back = pose[:3, 2]

        corners = np.array(
            [
                [x, y, z]
                for x in (bmin[0], bmax[0])
                for y in (bmin[1], bmax[1])
                for z in (bmin[2], bmax[2])
            ],
            dtype=np.float64,
        )
        rel = corners - center
        tan_half_x = max((self.width * 0.5) / max((self.width * self.lens) / self.sensor_width_mm, 1e-6), 1e-6)
        tan_half_y = max((self.height * 0.5) / max((self.height * self.lens) / self.sensor_width_mm, 1e-6), 1e-6)
        required_dist = 0.5
        for point in rel:
            x_cam = abs(float(np.dot(point, right)))
            y_cam = abs(float(np.dot(point, up)))
            z_back = float(np.dot(point, back))
            required_dist = max(required_dist, z_back + x_cam / tan_half_x, z_back + y_cam / tan_half_y)
        required_size = required_dist / max(float(self.camera_dist_ratio), 1e-6)
        return max(base_size * self.camera_distance_scale, required_size * self.camera_frame_padding)

    def default_view_specs(self, num_views: int) -> list[ViewSpec]:
        count = max(1, int(num_views))
        return [
            ViewSpec(
                name=f"orbit_{idx + 1:02d}",
                elev=20.0,
                azim=90.0 + 360.0 * idx / count,
                reason="VLM-Pose style y-up circular orbit around the operation AABB center",
            )
            for idx in range(count)
        ]

    def plan_adaptive_view_specs(
        self,
        parts: dict[str, PartMesh],
        poses: dict[str, np.ndarray],
        num_views: int,
        focus_ids: Iterable[str],
        moving_part: str,
        fixed_ids: Iterable[str],
    ) -> list[ViewSpec]:
        count = max(1, int(num_views))
        _ = parts, poses, focus_ids, moving_part, fixed_ids
        return self.default_view_specs(count)

    @staticmethod
    def _dominant_local_axis(part: PartMesh) -> np.ndarray:
        idx = int(np.argmax(part.extents))
        axis = np.zeros(3, dtype=np.float64)
        axis[idx] = 1.0
        return axis

    @staticmethod
    def _azim_from_horizontal_direction(direction: np.ndarray) -> float:
        direction = np.asarray(direction, dtype=np.float64)
        return float(np.degrees(np.arctan2(direction[2], direction[0])))

    @staticmethod
    def _has_near_duplicate_view(existing: list[ViewSpec], candidate: ViewSpec) -> bool:
        for item in existing:
            if abs(float(item.elev) - float(candidate.elev)) > 8.0:
                continue
            if angular_distance_deg(item.azim, candidate.azim) < 18.0:
                return True
        return False

    @staticmethod
    def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up if up is not None else [0.0, 1.0, 0.0], dtype=np.float64)
        z = eye - target
        z /= max(float(np.linalg.norm(z)), 1e-8)
        x = np.cross(up, z)
        if np.linalg.norm(x) < 1e-8:
            x = np.cross(np.array([0.0, 0.0, 1.0]), z)
        x /= max(float(np.linalg.norm(x)), 1e-8)
        y = np.cross(z, x)
        T = np.eye(4, dtype=np.float64)
        T[:3, 0] = x
        T[:3, 1] = y
        T[:3, 2] = z
        T[:3, 3] = eye
        return T

    def compute_scene_bounds(
        self,
        parts: dict[str, PartMesh],
        poses: dict[str, np.ndarray],
        part_ids: Iterable[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        bmin, bmax = self._raw_scene_bounds(parts, poses, part_ids)
        if self.show_workcell:
            points = [bmin.reshape(1, 3), bmax.reshape(1, 3)]
            points.extend(
                mesh.vertices
                for mesh, _color, alpha in self._workcell_meshes_from_bounds(bmin, bmax)
                if alpha >= 0.5
            )
            return bounds_for_points(np.concatenate(points, axis=0))
        return bmin, bmax

    def _raw_scene_bounds(
        self,
        parts: dict[str, PartMesh],
        poses: dict[str, np.ndarray],
        part_ids: Iterable[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        points = []
        for part_id in part_ids:
            if part_id not in parts:
                continue
            points.append(transform_points(parts[part_id].vertices, poses.get(part_id, np.eye(4))))
        if not points:
            for part_id, part in parts.items():
                points.append(transform_points(part.vertices, poses.get(part_id, np.eye(4))))
        return bounds_for_points(np.concatenate(points, axis=0))

    @staticmethod
    def part_world_bounds(part: PartMesh, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return bounds_for_points(transform_points(part.vertices, pose))

    def _workcell_meshes_from_bounds(
        self,
        bmin: np.ndarray,
        bmax: np.ndarray,
    ) -> list[tuple[trimesh.Trimesh, tuple[float, float, float], float]]:
        """Create visual-only benchmark workcell meshes around the current scene."""
        asset_meshes = self._furniture_bench_workcell_meshes_from_bounds(bmin, bmax)
        if asset_meshes:
            return asset_meshes
        asset_meshes = self._bullet_workcell_meshes_from_bounds(bmin, bmax)
        if asset_meshes:
            return asset_meshes
        return self._procedural_workcell_meshes_from_bounds(bmin, bmax)

    def _furniture_bench_workcell_meshes_from_bounds(
        self,
        bmin: np.ndarray,
        bmax: np.ndarray,
    ) -> list[tuple[trimesh.Trimesh, tuple[float, float, float], float]]:
        asset_root = Path(__file__).resolve().parents[2] / "assets" / "workcell" / "furniture_bench"
        mesh_root = asset_root / "mesh"
        panda_urdf = asset_root / "franka_description_ros" / "franka_description" / "robots" / "franka_panda.urdf"
        required = [
            mesh_root / "table.obj",
            mesh_root / "background.obj",
            panda_urdf,
        ]
        if any(not path.exists() for path in required):
            return []

        bmin = np.asarray(bmin, dtype=np.float64)
        bmax = np.asarray(bmax, dtype=np.float64)
        span = np.maximum(bmax - bmin, 1e-6)
        target_center = np.array(
            [(bmin[0] + bmax[0]) * 0.5, 0.0, (bmin[2] + bmax[2]) * 0.5],
            dtype=np.float64,
        )
        surface_y = float(bmin[1] - 0.012)

        # FurnitureBench uses a 1.3m x 0.8m tabletop in a Z-up Isaac Gym scene.
        # Our assembly assets use arbitrary scene units, so scale the whole
        # official workcell uniformly while preserving its relative layout.
        table_extent_x = 1.30
        table_extent_z = 0.80
        margin = max(0.18 * max(float(span[0]), float(span[2])), 0.18)
        workcell_scale = max(
            (float(span[0]) + 2.0 * margin) / table_extent_x,
            (float(span[2]) + 2.0 * margin) / table_extent_z,
            1.0,
        )
        official_surface_z = 0.415
        scene_T = np.eye(4, dtype=np.float64)
        scene_T[:3, 3] = np.array(
            [
                target_center[0],
                surface_y - workcell_scale * official_surface_z,
                target_center[2],
            ],
            dtype=np.float64,
        )
        scene_T = scene_T @ self._scale_matrix([workcell_scale, workcell_scale, workcell_scale]) @ self._zup_to_yup_matrix()

        def actor_pose(xyz: tuple[float, float, float], yaw: float = 0.0) -> np.ndarray:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = self._rotation_z(yaw)[:3, :3]
            T[:3, 3] = np.asarray(xyz, dtype=np.float64)
            return T

        def add_mesh(
            path: Path,
            pose: np.ndarray,
            color: tuple[float, float, float],
            alpha: float = 1.0,
        ) -> list[tuple[trimesh.Trimesh, tuple[float, float, float], float]]:
            out: list[tuple[trimesh.Trimesh, tuple[float, float, float], float]] = []
            if alpha <= 0.0:
                return out
            try:
                pieces = self._load_mesh_pieces(path)
            except Exception:
                return out
            for piece in pieces:
                mesh = piece.copy()
                mesh.apply_transform(scene_T @ pose)
                out.append((mesh, color, alpha))
            return out

        meshes: list[tuple[trimesh.Trimesh, tuple[float, float, float], float]] = []
        meshes.extend(add_mesh(mesh_root / "table.obj", actor_pose((0.0, 0.0, 0.4)), (0.17, 0.18, 0.18)))
        meshes.extend(add_mesh(mesh_root / "background.obj", actor_pose((-0.8, 0.0, 0.75)), (0.78, 0.80, 0.80), alpha=0.0))

        franka_base_official = np.array([-0.3, 0.0, 0.43, 1.0], dtype=np.float64)
        franka_base = (scene_T @ franka_base_official)[:3]
        robot_scale = float(np.clip(workcell_scale * 0.46, 0.68, 0.92))
        try:
            meshes.extend(
                self._load_panda_workcell_meshes(
                    panda_urdf,
                    base=franka_base,
                    yaw=0.0,
                    robot_scale=robot_scale,
                    joint_values={
                        "panda_joint1": -0.02630888,
                        "panda_joint2": 0.3758795,
                        "panda_joint3": 0.12485036,
                        "panda_joint4": -2.1383357,
                        "panda_joint5": -0.09431414,
                        "panda_joint6": 2.49649072,
                        "panda_joint7": 0.01921718,
                        "panda_finger_joint1": 0.035,
                        "panda_finger_joint2": 0.035,
                    },
                )
            )
        except Exception:
            return meshes
        return meshes

    def _bullet_workcell_meshes_from_bounds(
        self,
        bmin: np.ndarray,
        bmax: np.ndarray,
    ) -> list[tuple[trimesh.Trimesh, tuple[float, float, float], float]]:
        asset_root = Path(__file__).resolve().parents[2] / "assets" / "workcell"
        table_obj = asset_root / "table" / "table.obj"
        panda_urdf = asset_root / "franka_panda" / "panda.urdf"
        if not table_obj.exists() or not panda_urdf.exists():
            return []

        bmin = np.asarray(bmin, dtype=np.float64)
        bmax = np.asarray(bmax, dtype=np.float64)
        span = np.maximum(bmax - bmin, 1e-6)
        pad = max(max(span[0], span[2]) * 0.25, 0.42)
        x0, x1 = float(bmin[0] - pad), float(bmax[0] + pad)
        z0, z1 = float(bmin[2] - pad), float(bmax[2] + pad)
        surface_y = float(bmin[1] - 0.012)
        table_x = x1 - x0
        table_z = z1 - z0
        table_thickness = max(0.055, 0.045 * max(table_x, table_z))
        table_height = max(0.58, 0.28 * max(table_x, table_z))
        meshes: list[tuple[trimesh.Trimesh, tuple[float, float, float], float]] = []

        cube = trimesh.load(table_obj, force="mesh", process=False)
        if not isinstance(cube, trimesh.Trimesh) or len(cube.vertices) == 0:
            return []
        table_color = (0.54, 0.49, 0.40)
        leg_color = (0.30, 0.30, 0.30)
        top = cube.copy()
        top.apply_transform(self._scale_matrix([table_x, table_z, table_thickness]))
        top.apply_transform(self._zup_to_yup_matrix())
        top.apply_translation([(x0 + x1) * 0.5, surface_y - table_thickness * 0.5, (z0 + z1) * 0.5])
        meshes.append((top, table_color, 1.0))

        leg_w = max(0.07, 0.035 * max(table_x, table_z))
        inset = max(0.14, 0.08 * max(table_x, table_z))
        for lx in (x0 + inset, x1 - inset):
            for lz in (z0 + inset, z1 - inset):
                leg = cube.copy()
                leg.apply_transform(self._scale_matrix([leg_w, leg_w, table_height]))
                leg.apply_transform(self._zup_to_yup_matrix())
                leg.apply_translation([lx, surface_y - table_thickness - table_height * 0.5, lz])
                meshes.append((leg, leg_color, 1.0))

        target = np.array([(bmin[0] + bmax[0]) * 0.5, surface_y, (bmin[2] + bmax[2]) * 0.5], dtype=np.float64)
        base = np.array([x0 + 0.18 * table_x, surface_y + 0.012, z0 + 0.50 * table_z], dtype=np.float64)
        yaw = float(np.arctan2(target[0] - base[0], target[2] - base[2]))
        try:
            meshes.extend(self._load_panda_workcell_meshes(panda_urdf, base=base, yaw=yaw, robot_scale=1.7))
        except Exception:
            return []
        return meshes

    def _load_panda_workcell_meshes(
        self,
        urdf_path: Path,
        base: np.ndarray,
        yaw: float,
        robot_scale: float = 1.0,
        joint_values: dict[str, float] | None = None,
    ) -> list[tuple[trimesh.Trimesh, tuple[float, float, float], float]]:
        root = ET.parse(urdf_path).getroot()
        model_root = urdf_path.parent
        visuals: dict[str, list[tuple[Path, np.ndarray, np.ndarray]]] = {}
        for link in root.findall("link"):
            link_name = link.attrib.get("name", "")
            items: list[tuple[Path, np.ndarray, np.ndarray]] = []
            for visual in link.findall("visual"):
                origin = visual.find("origin")
                T_origin = self._urdf_origin_matrix(origin)
                mesh_elem = visual.find("geometry/mesh")
                if mesh_elem is None:
                    continue
                filename = mesh_elem.attrib.get("filename", "")
                mesh_path = self._resolve_urdf_mesh_path(model_root, filename)
                scale = np.fromstring(mesh_elem.attrib.get("scale", "1 1 1"), sep=" ", dtype=np.float64)
                if scale.size != 3:
                    scale = np.ones(3, dtype=np.float64)
                if mesh_path.exists():
                    items.append((mesh_path, T_origin, scale))
            if items:
                visuals[link_name] = items

        joints = []
        children_by_parent: dict[str, list[dict[str, object]]] = {}
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            item = {
                "name": joint.attrib.get("name", ""),
                "type": joint.attrib.get("type", "fixed"),
                "parent": parent.attrib.get("link", ""),
                "child": child.attrib.get("link", ""),
                "origin": self._urdf_origin_matrix(joint.find("origin")),
                "axis": self._urdf_axis(joint.find("axis")),
            }
            joints.append(item)
            children_by_parent.setdefault(str(item["parent"]), []).append(item)

        default_joint_values = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.72,
            "panda_joint3": 0.0,
            "panda_joint4": -2.32,
            "panda_joint5": 0.0,
            "panda_joint6": 1.68,
            "panda_joint7": 0.78,
            "panda_finger_joint1": 0.035,
            "panda_finger_joint2": 0.035,
        }
        if joint_values:
            default_joint_values.update(joint_values)
        link_pose = {"panda_link0": np.eye(4, dtype=np.float64)}
        stack = ["panda_link0"]
        while stack:
            parent = stack.pop()
            parent_T = link_pose[parent]
            for joint in children_by_parent.get(parent, []):
                name = str(joint["name"])
                joint_T = np.asarray(joint["origin"], dtype=np.float64).copy()
                axis = np.asarray(joint["axis"], dtype=np.float64)
                value = float(default_joint_values.get(name, 0.0))
                if joint["type"] == "revolute":
                    joint_T = joint_T @ self._axis_angle_matrix(axis, value)
                elif joint["type"] == "prismatic":
                    slide = np.eye(4, dtype=np.float64)
                    slide[:3, 3] = axis * value
                    joint_T = joint_T @ slide
                child = str(joint["child"])
                link_pose[child] = parent_T @ joint_T
                stack.append(child)

        place_T = np.eye(4, dtype=np.float64)
        place_T[:3, :3] = self._rotation_y(yaw)[:3, :3]
        place_T[:3, 3] = np.asarray(base, dtype=np.float64)
        robot_scale_T = self._scale_matrix([robot_scale, robot_scale, robot_scale])
        coord_T = self._zup_to_yup_matrix()
        out: list[tuple[trimesh.Trimesh, tuple[float, float, float], float]] = []
        for link_name, link_visuals in visuals.items():
            color = self._panda_link_color(link_name)
            for mesh_path, visual_T, scale in link_visuals:
                pieces = self._load_mesh_pieces(mesh_path)
                for piece in pieces:
                    if not isinstance(piece, trimesh.Trimesh) or len(piece.vertices) == 0:
                        continue
                    mesh = piece.copy()
                    mesh.apply_transform(self._scale_matrix(scale))
                    mesh.apply_transform(place_T @ robot_scale_T @ coord_T @ link_pose.get(link_name, np.eye(4)) @ visual_T)
                    out.append((mesh, color, 1.0))
        return out

    def _procedural_workcell_meshes_from_bounds(
        self,
        bmin: np.ndarray,
        bmax: np.ndarray,
    ) -> list[tuple[trimesh.Trimesh, tuple[float, float, float], float]]:
        bmin = np.asarray(bmin, dtype=np.float64)
        bmax = np.asarray(bmax, dtype=np.float64)
        span = np.maximum(bmax - bmin, 1e-6)
        scene_scale = float(np.clip(max(span[0], span[2]), 1.0, 3.0))
        pad = max(max(span[0], span[2]) * 0.24, 0.35)
        x0, x1 = float(bmin[0] - pad), float(bmax[0] + pad)
        z0, z1 = float(bmin[2] - pad), float(bmax[2] + pad)
        zc = (z0 + z1) * 0.5
        surface_y = float(bmin[1] - 0.018)

        meshes: list[tuple[trimesh.Trimesh, tuple[float, float, float], float]] = []

        table_thickness = 0.055 * scene_scale
        table_top = self._box_mesh(
            center=np.array([(x0 + x1) * 0.5, surface_y - table_thickness * 0.5, zc]),
            extents=np.array([x1 - x0, table_thickness, z1 - z0]),
        )
        meshes.append((table_top, (0.63, 0.55, 0.43), 1.0))

        leg_h = 0.55 * scene_scale
        leg_w = 0.055 * scene_scale
        inset = max(0.12 * scene_scale, 0.12)
        for lx in (x0 + inset, x1 - inset):
            for lz in (z0 + inset, z1 - inset):
                leg = self._box_mesh(
                    center=np.array([lx, surface_y - table_thickness - leg_h * 0.5, lz]),
                    extents=np.array([leg_w, leg_h, leg_w]),
                )
                meshes.append((leg, (0.34, 0.32, 0.30), 1.0))

        base_x = x0 - 0.22 * scene_scale
        base_z = z0 + 0.28 * (z1 - z0)
        base_y = surface_y
        robot = self._robot_arm_meshes(
            base=np.array([base_x, base_y, base_z], dtype=np.float64),
            target=np.array([bmin[0] + 0.42 * span[0], surface_y + 0.30 * scene_scale, zc], dtype=np.float64),
            scale=scene_scale,
        )
        meshes.extend(robot)
        return meshes

    @staticmethod
    def _resolve_urdf_mesh_path(model_root: Path, filename: str) -> Path:
        if filename.startswith("package://"):
            package_path = filename[len("package://") :]
            if package_path.startswith("franka_description/"):
                return model_root.parent / package_path[len("franka_description/") :]
            filename = package_path
        return model_root / filename

    @staticmethod
    def _load_mesh_pieces(mesh_path: Path) -> list[trimesh.Trimesh]:
        try:
            loaded = trimesh.load(mesh_path, force="scene", process=False)
        except ImportError:
            fallback = PosedSceneRenderer._fallback_visual_mesh_path(mesh_path)
            if fallback is None:
                if mesh_path.suffix.lower() == ".dae":
                    return []
                raise
            loaded = trimesh.load(fallback, force="scene", process=False)
        pieces = loaded.dump() if isinstance(loaded, trimesh.Scene) else [loaded]
        return [piece for piece in pieces if isinstance(piece, trimesh.Trimesh) and len(piece.vertices) > 0]

    @staticmethod
    def _fallback_visual_mesh_path(mesh_path: Path) -> Path | None:
        if mesh_path.suffix.lower() != ".dae":
            return None
        parts = list(mesh_path.parts)
        try:
            visual_idx = parts.index("visual")
        except ValueError:
            return None
        name = mesh_path.stem
        parts[visual_idx] = "collision"
        parts[-1] = f"{name}.stl"
        fallback = Path(*parts)
        return fallback if fallback.exists() else None

    @staticmethod
    def _urdf_axis(axis_elem: ET.Element | None) -> np.ndarray:
        if axis_elem is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis = np.fromstring(axis_elem.attrib.get("xyz", "0 0 1"), sep=" ", dtype=np.float64)
        if axis.size != 3 or np.linalg.norm(axis) < 1e-8:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return axis / np.linalg.norm(axis)

    @staticmethod
    def _urdf_origin_matrix(origin_elem: ET.Element | None) -> np.ndarray:
        xyz = np.zeros(3, dtype=np.float64)
        rpy = np.zeros(3, dtype=np.float64)
        if origin_elem is not None:
            raw_xyz = np.fromstring(origin_elem.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
            raw_rpy = np.fromstring(origin_elem.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
            if raw_xyz.size == 3:
                xyz = raw_xyz
            if raw_rpy.size == 3:
                rpy = raw_rpy
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = PosedSceneRenderer._rpy_matrix(rpy)
        T[:3, 3] = xyz
        return T

    @staticmethod
    def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = [float(x) for x in rpy]
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        return Rz @ Ry @ Rx

    @staticmethod
    def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
        x, y, z = axis
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        C = 1.0 - c
        R = np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ],
            dtype=np.float64,
        )
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        return T

    @staticmethod
    def _scale_matrix(scale: Iterable[float]) -> np.ndarray:
        s = np.asarray(list(scale), dtype=np.float64).reshape(3)
        T = np.eye(4, dtype=np.float64)
        T[0, 0], T[1, 1], T[2, 2] = float(s[0]), float(s[1]), float(s[2])
        return T

    @staticmethod
    def _zup_to_yup_matrix() -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        return T

    @staticmethod
    def _rotation_y(angle: float) -> np.ndarray:
        c, s = np.cos(float(angle)), np.sin(float(angle))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
        return T

    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        c, s = np.cos(float(angle)), np.sin(float(angle))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return T

    @staticmethod
    def _panda_link_color(link_name: str) -> tuple[float, float, float]:
        if "finger" in link_name or "hand" in link_name:
            return (0.12, 0.13, 0.15)
        if link_name in {"panda_link0", "panda_link7"}:
            return (0.20, 0.22, 0.25)
        return (0.68, 0.72, 0.74)

    @staticmethod
    def _box_mesh(center: np.ndarray, extents: np.ndarray) -> trimesh.Trimesh:
        mesh = trimesh.creation.box(extents=np.asarray(extents, dtype=np.float64))
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.asarray(center, dtype=np.float64)
        mesh.apply_transform(T)
        return mesh

    def _robot_arm_meshes(
        self,
        base: np.ndarray,
        target: np.ndarray,
        scale: float,
    ) -> list[tuple[trimesh.Trimesh, tuple[float, float, float], float]]:
        base = np.asarray(base, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        scale = float(max(scale, 1.0))
        colors = {
            "dark": (0.12, 0.14, 0.17),
            "joint": (0.20, 0.30, 0.42),
            "link": (0.78, 0.82, 0.86),
            "tool": (0.05, 0.06, 0.07),
        }
        shoulder = base + np.array([0.0, 0.24 * scale, 0.0])
        elbow = base + np.array([0.34 * scale, 0.52 * scale, 0.08 * scale])
        wrist = target + np.array([-0.10 * scale, 0.14 * scale, 0.0])
        tool = target

        meshes: list[tuple[trimesh.Trimesh, tuple[float, float, float], float]] = []
        meshes.append((self._cylinder_between(base, shoulder, 0.085 * scale), colors["dark"], 1.0))
        meshes.append((self._sphere_mesh(shoulder, 0.11 * scale), colors["joint"], 1.0))
        meshes.append((self._cylinder_between(shoulder, elbow, 0.055 * scale), colors["link"], 1.0))
        meshes.append((self._sphere_mesh(elbow, 0.085 * scale), colors["joint"], 1.0))
        meshes.append((self._cylinder_between(elbow, wrist, 0.045 * scale), colors["link"], 1.0))
        meshes.append((self._sphere_mesh(wrist, 0.065 * scale), colors["joint"], 1.0))
        meshes.append((self._cylinder_between(wrist, tool, 0.028 * scale), colors["tool"], 1.0))

        gripper_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        finger_len = 0.13 * scale
        finger_gap = 0.04 * scale
        for sign in (-1.0, 1.0):
            center = tool + gripper_dir * sign * finger_gap + np.array([0.04 * scale, 0.0, 0.0])
            finger = self._box_mesh(
                center=center,
                extents=np.array([finger_len, 0.018 * scale, 0.018 * scale]),
            )
            meshes.append((finger, colors["tool"], 1.0))
        return meshes

    @staticmethod
    def _cylinder_between(start: np.ndarray, end: np.ndarray, radius: float) -> trimesh.Trimesh:
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-8:
            length = 1e-8
            direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        mesh = trimesh.creation.cylinder(radius=float(radius), height=length, sections=32)
        T = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), direction / length)
        T[:3, 3] = (start + end) * 0.5
        mesh.apply_transform(T)
        return mesh

    @staticmethod
    def _sphere_mesh(center: np.ndarray, radius: float) -> trimesh.Trimesh:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(radius))
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.asarray(center, dtype=np.float64)
        mesh.apply_transform(T)
        return mesh


def view_annotation_title(
    prefix: str,
    view_index: int,
    spec: ViewSpec,
    lens: float,
    camera_pose: np.ndarray | None = None,
) -> str:
    _ = lens, camera_pose
    return (
        f"{spec.name} | {prefix} view {view_index:02d} | "
        f"tick={RENDER_RULER_TICK_STEP_PX}px"
    )


def _draw_text_with_halo(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    fill: tuple[int, int, int, int],
    halo: tuple[int, int, int, int],
) -> None:
    draw.text(xy, text, font=font, fill=fill, stroke_width=2, stroke_fill=halo)


def _wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    tokens = text.split()
    if not tokens:
        return [text]
    lines: list[str] = []
    line = tokens[0]
    for token in tokens[1:]:
        candidate = f"{line} {token}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            line = candidate
        else:
            lines.append(line)
            line = token
    lines.append(line)
    return lines


def _draw_panel_rulers(
    draw: ImageDraw.ImageDraw,
    panel_box: tuple[int, int, int, int],
    source_size: tuple[int, int],
    tick_step: int,
) -> None:
    tick_step = max(1, int(tick_step))
    axis_color = (20, 24, 28, 230)
    halo_color = (255, 255, 255, 245)
    text_color = (24, 28, 34, 255)
    x0, y0, x1, y1 = panel_box
    source_w, source_h = source_size
    panel_w = max(1, x1 - x0)
    panel_h = max(1, y1 - y0)
    try:
        tick_font = ImageFont.truetype("DejaVuSans.ttf", size=max(9, min(12, panel_w // 70)))
    except Exception:
        tick_font = ImageFont.load_default()

    def line(points, width_px=1) -> None:
        draw.line(points, fill=halo_color, width=width_px + 2)
        draw.line(points, fill=axis_color, width=width_px)

    line([(x0, y0), (x1, y0)], width_px=1)
    line([(x0, y1), (x1, y1)], width_px=1)
    line([(x0, y0), (x0, y1)], width_px=1)
    line([(x1, y0), (x1, y1)], width_px=1)

    x_ticks = list(range(0, source_w, tick_step))
    if not x_ticks or x_ticks[-1] != source_w:
        x_ticks.append(source_w)
    for source_x in x_ticks:
        px = int(round(x0 + (float(source_x) / max(float(source_w), 1.0)) * panel_w))
        line([(px, y0 - 6), (px, y0 + 6)], width_px=1)
        line([(px, y1 - 6), (px, y1 + 6)], width_px=1)
        label = str(int(source_x))
        _draw_text_with_halo(
            draw,
            (min(max(px + 2, x0), max(x0, x1 - 34)), y0 + 8),
            label,
            tick_font,
            text_color,
            halo_color,
        )

    y_ticks = list(range(0, source_h, tick_step))
    if not y_ticks or y_ticks[-1] != source_h:
        y_ticks.append(source_h)
    for source_y in y_ticks:
        py = int(round(y0 + (float(source_y) / max(float(source_h), 1.0)) * panel_h))
        line([(x0 - 6, py), (x0 + 6, py)], width_px=1)
        line([(x1 - 6, py), (x1 + 6, py)], width_px=1)
        label = str(int(source_y))
        label_y = min(max(py + 2, y0 + 2), max(y0 + 2, y1 - 14))
        _draw_text_with_halo(draw, (max(2, x0 - 40), label_y), label, tick_font, text_color, halo_color)


def add_image_label(color: np.ndarray, label: str, tick_step: int = RENDER_RULER_TICK_STEP_PX) -> np.ndarray:
    if os.getenv("ANNOTATE_RENDERED_VIEWS", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        _ = label, tick_step
        return np.asarray(color)[..., :3].astype(np.uint8)

    scene_img = Image.fromarray(color[..., :3]).convert("RGB")
    source_w, source_h = scene_img.size
    canvas = Image.new("RGBA", scene_img.size, (246, 247, 249, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(13, min(18, canvas.width // 48)))
    except Exception:
        font = ImageFont.load_default()

    max_width = max(80, canvas.width - 24)
    lines = _wrap_text_to_width(draw, label, font, max_width)
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_h = max((box[3] - box[1] for box in line_boxes), default=14)
    title_h = 12 + len(lines) * (line_h + 3)
    draw.rectangle([6, 6, canvas.width - 7, 6 + title_h], fill=(0, 0, 0, 185))
    y = 11
    for line in lines:
        draw.text((12, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_h + 3

    left_margin = max(46, min(68, canvas.width // 10))
    right_margin = max(16, min(28, canvas.width // 32))
    top_margin = max(42, 18 + title_h)
    bottom_margin = max(38, min(54, canvas.height // 11))
    max_panel_w = max(16, canvas.width - left_margin - right_margin)
    max_panel_h = max(16, canvas.height - top_margin - bottom_margin)
    scale = min(max_panel_w / max(source_w, 1), max_panel_h / max(source_h, 1))
    panel_w = max(1, int(round(source_w * scale)))
    panel_h = max(1, int(round(source_h * scale)))
    panel_x0 = left_margin + max(0, (max_panel_w - panel_w) // 2)
    panel_y0 = top_margin + max(0, (max_panel_h - panel_h) // 2)
    panel_x1 = panel_x0 + panel_w
    panel_y1 = panel_y0 + panel_h

    scene_resized = scene_img.resize((panel_w, panel_h), Image.LANCZOS).convert("RGBA")
    draw.rectangle([panel_x0 - 1, panel_y0 - 1, panel_x1 + 1, panel_y1 + 1], fill=(228, 231, 236, 255))
    canvas.alpha_composite(scene_resized, (panel_x0, panel_y0))
    _draw_panel_rulers(
        draw,
        (panel_x0, panel_y0, panel_x1, panel_y1),
        (source_w, source_h),
        tick_step,
    )
    return np.asarray(canvas.convert("RGB"))


def write_image(path: str | Path, color: np.ndarray) -> None:
    if imageio is not None:
        imageio.imwrite(path, color)
        return
    Image.fromarray(np.asarray(color)[..., :3].astype(np.uint8)).save(path)


ORTHOGRAPHIC_MEASUREMENT_VIEWS = (
    {
        "name": "front",
        "title": "Front (eye +Z)",
        "horizontal_axis": 0,
        "vertical_axis": 1,
        "depth_axis": 2,
        "horizontal_label": "+X",
        "vertical_label": "+Y",
    },
    {
        "name": "right",
        "title": "Right (eye +X)",
        "horizontal_axis": 2,
        "vertical_axis": 1,
        "depth_axis": 0,
        "horizontal_label": "+Z",
        "vertical_label": "+Y",
    },
    {
        "name": "top",
        "title": "Top (eye +Y)",
        "horizontal_axis": 0,
        "vertical_axis": 2,
        "depth_axis": 1,
        "horizontal_label": "+X",
        "vertical_label": "+Z",
    },
)


def render_orthographic_measurement_input(
    current_parts: dict[str, PartMesh],
    current_poses: dict[str, np.ndarray],
    target_parts: dict[str, PartMesh],
    target_poses: dict[str, np.ndarray],
    output_path: str | Path,
    moving_part: str,
    fixed_parts: Iterable[str],
    frame_bounds: tuple[np.ndarray, np.ndarray],
    part_color_map: dict[str, tuple[float, float, float]] | None = None,
    dpi: int = 160,
) -> Path:
    if plt is None or PolyCollection is None:
        raise RuntimeError("matplotlib is required for orthographic measurement input rendering")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bmin = np.asarray(frame_bounds[0], dtype=np.float64)
    bmax = np.asarray(frame_bounds[1], dtype=np.float64)
    fixed_parts = list(fixed_parts)
    height_ratios = []
    for view in ORTHOGRAPHIC_MEASUREMENT_VIEWS:
        h_axis = int(view["horizontal_axis"])
        v_axis = int(view["vertical_axis"])
        h_span = max(float(bmax[h_axis] - bmin[h_axis]), 1e-6)
        v_span = max(float(bmax[v_axis] - bmin[v_axis]), 1e-6)
        height_ratios.append(float(np.clip(v_span / h_span, 0.28, 1.75)))
    fig = plt.figure(figsize=(12.0, 7.8), dpi=dpi)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(3, 2, height_ratios=height_ratios)
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(3)])
    column_specs = (
        ("Current", current_parts, current_poses),
        ("Target", target_parts, target_poses),
    )
    for row, view in enumerate(ORTHOGRAPHIC_MEASUREMENT_VIEWS):
        for col, (column_title, parts, poses) in enumerate(column_specs):
            ax = axes[row, col]
            _draw_orthographic_panel(
                ax=ax,
                parts=parts,
                poses=poses,
                view=view,
                bmin=bmin,
                bmax=bmax,
                title=f"{column_title} {view['title']}",
                moving_part=moving_part,
                fixed_parts=fixed_parts,
                part_color_map=part_color_map,
            )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.075, wspace=0.16, hspace=0.36)
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    return output


def render_current_orthographic_input(
    current_parts: dict[str, PartMesh],
    current_poses: dict[str, np.ndarray],
    output_path: str | Path,
    moving_part: str,
    fixed_parts: Iterable[str],
    frame_bounds: tuple[np.ndarray, np.ndarray],
    visible_part_ids: Iterable[str] | None = None,
    part_color_map: dict[str, tuple[float, float, float]] | None = None,
    dpi: int = 160,
) -> Path:
    if plt is None or PolyCollection is None:
        raise RuntimeError("matplotlib is required for orthographic input rendering")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bmin = np.asarray(frame_bounds[0], dtype=np.float64)
    bmax = np.asarray(frame_bounds[1], dtype=np.float64)
    fixed_parts = list(fixed_parts)
    height_ratios = []
    for view in ORTHOGRAPHIC_MEASUREMENT_VIEWS:
        h_axis = int(view["horizontal_axis"])
        v_axis = int(view["vertical_axis"])
        h_span = max(float(bmax[h_axis] - bmin[h_axis]), 1e-6)
        v_span = max(float(bmax[v_axis] - bmin[v_axis]), 1e-6)
        height_ratios.append(float(np.clip(v_span / h_span, 0.28, 1.75)))

    fig = plt.figure(figsize=(7.0, 7.8), dpi=dpi)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(3, 1, height_ratios=height_ratios)
    axes = [fig.add_subplot(grid[row, 0]) for row in range(3)]
    for row, view in enumerate(ORTHOGRAPHIC_MEASUREMENT_VIEWS):
        _draw_orthographic_panel(
            ax=axes[row],
            parts=current_parts,
            poses=current_poses,
            view=view,
            bmin=bmin,
            bmax=bmax,
            title=f"Current {view['title']}",
            moving_part=moving_part,
            fixed_parts=fixed_parts,
            visible_part_ids=visible_part_ids,
            part_color_map=part_color_map,
        )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.075, hspace=0.36)
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    return output


def _draw_orthographic_panel(
    ax,
    parts: dict[str, PartMesh],
    poses: dict[str, np.ndarray],
    view: dict[str, object],
    bmin: np.ndarray,
    bmax: np.ndarray,
    title: str,
    moving_part: str,
    fixed_parts: list[str],
    visible_part_ids: Iterable[str] | None = None,
    part_color_map: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    h_axis = int(view["horizontal_axis"])
    v_axis = int(view["vertical_axis"])
    d_axis = int(view["depth_axis"])
    h0, h1 = float(bmin[h_axis]), float(bmax[h_axis])
    v0, v1 = float(bmin[v_axis]), float(bmax[v_axis])
    h_span = max(h1 - h0, 1e-6)
    v_span = max(v1 - v0, 1e-6)
    h_pad = max((h1 - h0) * 0.08, 0.05)
    v_pad = max((v1 - v0) * 0.08, 0.05)
    ax.set_xlim(h0 - h_pad, h1 + h_pad)
    ax.set_ylim(v0 - v_pad, v1 + v_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor((0.98, 0.985, 0.99))
    grid_step = _nice_measure_step(max(h_span, v_span) / 5.0)
    x_start = np.floor((h0 - h_pad) / grid_step) * grid_step
    x_stop = np.ceil((h1 + h_pad) / grid_step) * grid_step
    y_start = np.floor((v0 - v_pad) / grid_step) * grid_step
    y_stop = np.ceil((v1 + v_pad) / grid_step) * grid_step
    ax.set_xticks(np.arange(x_start, x_stop + grid_step * 0.5, grid_step))
    ax.set_yticks(np.arange(y_start, y_stop + grid_step * 0.5, grid_step))
    ax.grid(True, color=(0.80, 0.83, 0.88), linewidth=0.85)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(str(view["horizontal_label"]), fontsize=11)
    ax.set_ylabel(str(view["vertical_label"]), fontsize=11)
    ax.tick_params(labelsize=10, width=1.0)

    face_items: list[tuple[int, float, np.ndarray, tuple[float, float, float, float], float]] = []
    label_items: list[tuple[int, float, str, float, float]] = []
    if visible_part_ids is None:
        ordered_visible_part_ids = [pid for pid in fixed_parts + [moving_part] if pid in parts]
    else:
        ordered_visible_part_ids = []
        for pid in visible_part_ids:
            if pid in parts and pid not in ordered_visible_part_ids:
                ordered_visible_part_ids.append(pid)
        for pid in [*fixed_parts, moving_part]:
            if pid in parts and pid not in ordered_visible_part_ids:
                ordered_visible_part_ids.append(pid)
    for part_id in ordered_visible_part_ids:
        part = parts[part_id]
        pose = validate_transform(poses.get(part_id, np.eye(4, dtype=np.float64)))
        base_color = stable_part_color(part_id)
        if part_color_map and str(part_id) in part_color_map:
            base_color = tuple(float(value) for value in part_color_map[str(part_id)])
        alpha = 0.84 if part_id == moving_part else 0.76
        edge_alpha = 0.34 if part_id == moving_part else 0.24
        layer = 1 if part_id == moving_part else 0
        projected_points: list[np.ndarray] = []
        projected_depths: list[float] = []
        for mesh in part.meshes:
            vertices = transform_points(np.asarray(mesh.vertices, dtype=np.float64), pose)
            projected_points.append(vertices[:, [h_axis, v_axis]])
            projected_depths.append(float(np.mean(vertices[:, d_axis])))
            for face in np.asarray(mesh.faces, dtype=np.int64):
                face_vertices = vertices[face]
                poly = face_vertices[:, [h_axis, v_axis]]
                depth = float(np.mean(face_vertices[:, d_axis]))
                face_items.append((layer, depth, poly, (*base_color, alpha), edge_alpha))
        if projected_points:
            projected = np.concatenate(projected_points, axis=0)
            center = np.mean(projected, axis=0)
            depth = float(np.mean(projected_depths)) if projected_depths else 0.0
            label_items.append((layer, depth, part_id, float(center[0]), float(center[1])))

    if not face_items:
        return
    face_items.sort(key=lambda item: (item[0], item[1]))
    polys = [item[2] for item in face_items]
    colors = [item[3] for item in face_items]
    edge_colors = [(0.05, 0.07, 0.10, item[4]) for item in face_items]
    collection = PolyCollection(
        polys,
        facecolors=colors,
        edgecolors=edge_colors,
        linewidths=0.08,
        closed=True,
    )
    ax.add_collection(collection)
    for _layer, _depth, part_id, x, y in sorted(label_items, key=lambda item: (item[0], item[1])):
        ax.text(
            x,
            y,
            str(part_id),
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
            zorder=20,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": (0.02, 0.03, 0.05, 0.86),
                "edgecolor": (1.0, 1.0, 1.0, 0.80),
                "linewidth": 0.6,
            },
        )


def _nice_measure_step(raw_step: float) -> float:
    raw = max(float(raw_step), 1e-6)
    exponent = np.floor(np.log10(raw))
    base = raw / (10.0**exponent)
    if base <= 1.0:
        nice = 1.0
    elif base <= 2.0:
        nice = 2.0
    elif base <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * (10.0**exponent))


def _fmt_metric(value: float) -> str:
    value = float(value)
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _draw_metric_scale_bar(
    *,
    ax,
    h0: float,
    h1: float,
    v0: float,
    v1: float,
    h_pad: float,
    v_pad: float,
) -> None:
    span = max(h1 - h0, 1e-6)
    scale_len = _nice_measure_step(span / 4.0)
    if scale_len > span * 0.45:
        scale_len *= 0.5
    x0 = h0 - h_pad * 0.45
    y0 = v0 - v_pad * 0.45
    x1 = x0 + scale_len
    ax.plot([x0, x1], [y0, y0], color=(0.02, 0.03, 0.05), linewidth=2.2, zorder=30)
    tick = max((v1 - v0) * 0.025, 0.015)
    ax.plot([x0, x0], [y0 - tick, y0 + tick], color=(0.02, 0.03, 0.05), linewidth=1.6, zorder=30)
    ax.plot([x1, x1], [y0 - tick, y0 + tick], color=(0.02, 0.03, 0.05), linewidth=1.6, zorder=30)
    ax.text(
        (x0 + x1) * 0.5,
        y0 + tick * 1.5,
        f"{_fmt_metric(scale_len)} scene units",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=(0.02, 0.03, 0.05),
        zorder=31,
        bbox={"facecolor": (1, 1, 1, 0.82), "edgecolor": "none", "pad": 1.5},
    )


def _draw_metric_summary_box(
    *,
    ax,
    bbox_items: list[tuple[int, float, str, float, float, float, float, tuple[float, float, float]]],
    moving_part: str,
    fixed_parts: list[str],
    h_label: str,
    v_label: str,
    grid_step: float,
) -> None:
    if not bbox_items:
        return
    priority = {str(pid): idx for idx, pid in enumerate([*fixed_parts, moving_part])}
    ordered = sorted(
        bbox_items,
        key=lambda item: (priority.get(str(item[2]), 99), item[2]),
    )
    lines = [f"grid={_fmt_metric(grid_step)} units"]
    for _layer, _depth, part_id, x0, y0, x1, y1, _color in ordered[:4]:
        role = "moving" if str(part_id) == str(moving_part) else "fixed"
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        lines.append(
            f"{role} {part_id}: c({h_label},{v_label})=({_fmt_metric(cx)},{_fmt_metric(cy)}), "
            f"size=({_fmt_metric(x1 - x0)}x{_fmt_metric(y1 - y0)})"
        )
    ax.text(
        0.012,
        0.982,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=(0.02, 0.03, 0.05),
        zorder=40,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": (1.0, 1.0, 1.0, 0.86),
            "edgecolor": (0.50, 0.57, 0.68, 0.65),
            "linewidth": 0.7,
        },
    )


def export_posed_scene(parts: dict[str, PartMesh], poses: dict[str, np.ndarray], output_path: str | Path) -> Path:
    meshes: list[trimesh.Trimesh] = []
    for part_id, part in parts.items():
        pose = validate_transform(poses.get(part_id, np.eye(4)))
        for mesh in part.meshes:
            m = mesh.copy()
            m.apply_transform(pose)
            meshes.append(m)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    trimesh.util.concatenate(meshes).export(output)
    return output
