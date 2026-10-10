from __future__ import annotations

import os
import shutil
import tempfile
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .geometry import apply_incremental_update, transform_points, validate_transform, visual_arrow_length_from_bounds
from .rendering import (
    FIXED_BASE_AXIS_COLORS,
    MOVING_AXIS_COLORS,
    PartMesh,
    PosedSceneRenderer,
    ViewSpec,
    add_image_label,
    build_coordinate_arrows,
    export_posed_scene,
    stable_part_color,
    view_annotation_title,
    write_image,
)

ISAACSIM_CAPTURE_BACKENDS = {
    "auto",
    "rgb_annotator",
    "camera",
    "replicator",
    "replicator_writer",
    "viewport",
    "movie_capture",
}


def normalize_isaacsim_capture_backend(value: str | None) -> str:
    backend = str(value or "auto").strip().lower().replace("-", "_")
    if backend == "auto":
        backend = "rgb_annotator"
    if backend == "movie":
        backend = "movie_capture"
    if backend == "replicator":
        backend = "replicator_writer"
    if backend not in ISAACSIM_CAPTURE_BACKENDS:
        raise ValueError(
            "isaacsim renderer must be one of "
            f"{sorted(ISAACSIM_CAPTURE_BACKENDS)}, got {value!r}"
        )
    return backend


@dataclass(frozen=True)
class SceneSimulationConfig:
    backend: str = "offline"
    width: int = 768
    height: int = 768
    camera_distance_scale: float = 1.15
    camera_frame_padding: float = 1.25
    show_ground_plane: bool = True
    ground_y: float = -0.05
    ground_alpha: float = 0.38
    show_workcell: bool = False
    isaacsim_headless: bool = True
    isaacsim_renderer: str = "rgb_annotator"
    isaacsim_capture_updates: int = 8
    scene_obj: Path | None = None

    def __post_init__(self) -> None:
        backend = str(self.backend or "offline").strip().lower()
        renderer = normalize_isaacsim_capture_backend(self.isaacsim_renderer)
        if backend not in {"offline", "isaacsim"}:
            raise ValueError(f"simulation backend must be 'offline' or 'isaacsim', got {self.backend!r}")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "isaacsim_renderer", renderer)
        object.__setattr__(self, "isaacsim_capture_updates", max(1, int(self.isaacsim_capture_updates)))
        object.__setattr__(self, "camera_distance_scale", max(0.1, float(self.camera_distance_scale)))
        object.__setattr__(self, "camera_frame_padding", max(1.0, float(self.camera_frame_padding)))


class SceneSimulationBackend:
    name = "offline"

    def __init__(self, config: SceneSimulationConfig):
        self.config = config
        self.renderer = PosedSceneRenderer(
            width=config.width,
            height=config.height,
            camera_distance_scale=config.camera_distance_scale,
            camera_frame_padding=config.camera_frame_padding,
            show_ground_plane=config.show_ground_plane,
            ground_y=config.ground_y,
            ground_alpha=config.ground_alpha,
            show_workcell=config.show_workcell,
        )

    def apply_pose_update(
        self,
        part: PartMesh,
        current_pose: np.ndarray,
        local_bounds: tuple[np.ndarray, np.ndarray],
        translation_units: Iterable[float],
        arrow_len: float,
        axis: str,
        angle_deg: float,
        rotation_matrix: np.ndarray | None,
    ) -> np.ndarray:
        return apply_incremental_update(
            current_T=current_pose,
            local_bounds=local_bounds,
            translation_units=translation_units,
            arrow_len=arrow_len,
            axis=axis,
            angle_deg=angle_deg,
            rotation_matrix=rotation_matrix,
        )

    def export_posed_scene(
        self,
        parts: dict[str, PartMesh],
        poses: dict[str, np.ndarray],
        output_path: str | Path,
    ) -> Path:
        return export_posed_scene(parts, poses, output_path)

    def to_json(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "renderer": "pyrender_or_matplotlib",
            "camera_distance_scale": self.config.camera_distance_scale,
            "camera_frame_padding": self.config.camera_frame_padding,
            "workcell": self.config.show_workcell,
            "scene_obj": str(self.config.scene_obj) if self.config.scene_obj else None,
        }

    def close(self) -> None:
        self.renderer.close()


class IsaacSimSimulationBackend(SceneSimulationBackend):
    name = "isaacsim"

    def __init__(self, config: SceneSimulationConfig):
        self.config = config
        self.renderer = IsaacSimSceneRenderer(
            width=config.width,
            height=config.height,
            camera_distance_scale=config.camera_distance_scale,
            camera_frame_padding=config.camera_frame_padding,
            show_ground_plane=config.show_ground_plane,
            ground_y=config.ground_y,
            ground_alpha=config.ground_alpha,
            show_workcell=config.show_workcell,
            headless=config.isaacsim_headless,
            capture_backend=config.isaacsim_renderer,
            capture_updates=config.isaacsim_capture_updates,
        )

    def export_posed_scene(
        self,
        parts: dict[str, PartMesh],
        poses: dict[str, np.ndarray],
        output_path: str | Path,
    ) -> Path:
        obj_path = export_posed_scene(parts, poses, output_path)
        usd_path = obj_path.with_suffix(".usd")
        self.renderer.export_usd_stage(parts, poses, usd_path)
        return obj_path

    def to_json(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "renderer": self.config.isaacsim_renderer,
            "headless": self.config.isaacsim_headless,
            "capture_updates": self.config.isaacsim_capture_updates,
            "camera_distance_scale": self.config.camera_distance_scale,
            "camera_frame_padding": self.config.camera_frame_padding,
            "workcell": self.config.show_workcell,
            "scene_obj": str(self.config.scene_obj) if self.config.scene_obj else None,
            "persistent_stage": True,
            "app_lifetime": "one SimulationApp per agent run",
            "pose_update": "kinematic USD prim transform",
            "stage_export_policy": "OBJ exports are preserved; Isaac Sim backend also writes a sidecar .usd stage for exported scenes.",
        }

    def close(self) -> None:
        self.renderer.close()


class IsaacSimSceneRenderer(PosedSceneRenderer):
    """Isaac Sim renderer with the same small interface used by the agent.

    The class intentionally imports Isaac Sim/Omniverse modules lazily so the
    default offline backend remains usable in ordinary Python environments.
    """

    def __init__(
        self,
        width: int = 768,
        height: int = 768,
        lens: float = 70.0,
        sensor_width_mm: float = 36.0,
        camera_dist_ratio: float = 2.2,
        camera_distance_scale: float = 1.15,
        camera_frame_padding: float = 1.25,
        show_ground_plane: bool = True,
        ground_y: float = -0.05,
        ground_alpha: float = 0.38,
        show_workcell: bool = False,
        headless: bool = True,
        capture_backend: str = "auto",
        capture_updates: int = 8,
    ):
        # Do not call PosedSceneRenderer.__init__ here; that would create a
        # pyrender renderer. We only need the inherited view-planning helpers.
        self.width = int(width)
        self.height = int(height)
        self.lens = float(lens)
        self.sensor_width_mm = float(sensor_width_mm)
        self.camera_dist_ratio = float(camera_dist_ratio)
        self.camera_distance_scale = max(0.1, float(camera_distance_scale))
        self.camera_frame_padding = max(1.0, float(camera_frame_padding))
        self.show_ground_plane = bool(show_ground_plane)
        self.ground_y = float(ground_y)
        self.ground_alpha = float(np.clip(ground_alpha, 0.0, 1.0))
        self.show_workcell = bool(show_workcell)
        self.headless = bool(headless)
        self.capture_backend = normalize_isaacsim_capture_backend(capture_backend)
        self.capture_updates = max(1, int(capture_updates))
        self._simulation_app = None
        self._owns_simulation_app = False
        self._stage_initialized = False
        self._part_root_paths: dict[tuple[str, str], str] = {}
        self._workcell_prop_count = 0
        self._rgb_camera = None
        self._rgb_render_product = None
        self._rgb_annotator = None
        self._current_camera_position: np.ndarray | None = None
        self._current_camera_look_at: np.ndarray | None = None
        self._log_callback = None
        self._ensure_isaacsim()

    def set_log_callback(self, callback: Any | None) -> None:
        self._log_callback = callback

    def _log(self, message: str) -> None:
        text = f"[IsaacSimSceneRenderer] {message}"
        if self._log_callback is not None:
            try:
                self._log_callback(text)
                return
            except Exception:
                pass
        print(text, flush=True)

    def _ensure_isaacsim(self) -> None:
        try:
            import omni.kit.app
            import omni.usd

            app = omni.kit.app.get_app()
            context = omni.usd.get_context()
            if app is not None and context is not None:
                self._owns_simulation_app = False
                return
        except Exception:
            pass

        try:
            from isaacsim import SimulationApp
        except Exception:
            try:
                from omni.isaac.kit import SimulationApp
            except Exception as exc:
                raise ImportError(
                    "simulation_backend=isaacsim requires Isaac Sim Python modules. "
                    "Run from the isaaclab/Isaac Sim environment, or use --simulation-backend offline."
                ) from exc

        active_gpu = self._optional_int_env("ISAACSIM_ACTIVE_GPU")
        physics_gpu = self._optional_int_env("ISAACSIM_PHYSICS_GPU")
        extra_args = ["--/renderer/multiGpu/autoEnable=false"]
        if self.capture_backend in {"auto", "movie_capture"}:
            extra_args.extend(["--enable", "omni.kit.capture.viewport"])

        launch_config: dict[str, Any] = {
            "headless": self.headless,
            "width": self.width,
            "height": self.height,
            "multi_gpu": False,
            "max_gpu_count": 1,
            "extra_args": extra_args,
        }
        if active_gpu is not None:
            launch_config["active_gpu"] = active_gpu
        if physics_gpu is not None:
            launch_config["physics_gpu"] = physics_gpu
        elif active_gpu is not None:
            launch_config["physics_gpu"] = active_gpu

        print(f"[IsaacSimSceneRenderer] starting SimulationApp with config: {launch_config}", flush=True)
        self._simulation_app = SimulationApp(launch_config)
        self._owns_simulation_app = True

    @staticmethod
    def _optional_int_env(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return None
        try:
            return int(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer GPU ordinal, got {raw!r}") from exc

    def close(self) -> None:
        if self._rgb_annotator is not None and self._rgb_render_product is not None:
            try:
                self._rgb_annotator.detach([self._rgb_render_product])
            except Exception:
                pass
        if self._rgb_render_product is not None:
            try:
                if hasattr(self._rgb_render_product, "destroy"):
                    self._rgb_render_product.destroy()
            except Exception:
                pass
        if self._rgb_camera is not None:
            try:
                if hasattr(self._rgb_camera, "destroy"):
                    self._rgb_camera.destroy()
            except Exception:
                pass
        self._rgb_camera = None
        self._rgb_annotator = None
        self._rgb_render_product = None
        if self._simulation_app is not None and self._owns_simulation_app:
            self._simulation_app.close()
            self._simulation_app = None

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
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._log(
            "render_scene_views begin "
            f"prefix={prefix} views={len(specs)} parts={sorted(parts.keys())} "
            f"focus={list(focus_ids) if focus_ids is not None else 'all'} "
            f"moving_part={moving_part} arrows={show_arrows} "
            f"fixed_axis_part={fixed_axis_part} fixed_base_axes={show_fixed_base_axes}"
        )
        focus = list(focus_ids) if focus_ids is not None else list(parts.keys())
        bmin, bmax = frame_bounds if frame_bounds is not None else self.compute_scene_bounds(parts, poses, focus)
        center = (bmin + bmax) * 0.5
        base_size = max(float(np.linalg.norm(bmax - bmin)), 1.0)
        self._log(
            "render_scene_views bounds "
            f"center={center.round(4).tolist()} base_size={base_size:.4f} "
            f"camera_distance_scale={self.camera_distance_scale:.3f} "
            f"camera_frame_padding={self.camera_frame_padding:.3f}"
        )
        paths: list[Path] = []
        if self.capture_backend == "rgb_annotator":
            for idx, spec in enumerate(specs):
                size = self.view_size_for_bounds(bmin, bmax, spec)
                self._log(f"view {idx + 1:02d} adaptive camera size={size:.4f} view={spec.name}")
                camera_pose = self.view_pose_from_spec(center, size, spec)
                path = output / f"{prefix}_view_{idx + 1:02d}.png"
                self._log(f"building headless-style RGB stage for {path}")
                self._build_headless_style_stage(
                    parts=parts,
                    poses=poses,
                    moving_part=moving_part,
                    show_arrows=show_arrows,
                    fixed_axis_part=fixed_axis_part,
                    show_fixed_base_axes=show_fixed_base_axes,
                    camera_pose=camera_pose,
                    camera_look_at=center,
                )
                self._log(f"capturing headless-style RGB stage to {path}")
                self._capture_current_stage(path)
                self._log(f"captured {path}")
                self._label_capture(path, view_annotation_title(prefix, idx + 1, spec, self.lens, camera_pose))
                paths.append(path)
            self._log(f"render_scene_views complete prefix={prefix}")
            return paths

        self._log("ensure persistent USD stage")
        self._ensure_stage_initialized()
        for idx, spec in enumerate(specs):
            size = self.view_size_for_bounds(bmin, bmax, spec)
            self._log(f"view {idx + 1:02d} adaptive camera size={size:.4f} view={spec.name}")
            camera_pose = self.view_pose_from_spec(center, size, spec)
            path = output / f"{prefix}_view_{idx + 1:02d}.png"
            self._log(f"updating persistent stage for {path}")
            self._update_stage_for_view(
                parts=parts,
                poses=poses,
                moving_part=moving_part,
                show_arrows=show_arrows,
                fixed_axis_part=fixed_axis_part,
                show_fixed_base_axes=show_fixed_base_axes,
                camera_pose=camera_pose,
                camera_look_at=center,
            )
            self._log(f"capturing stage to {path}")
            self._capture_current_stage(path)
            self._log(f"captured {path}")
            self._label_capture(path, view_annotation_title(prefix, idx + 1, spec, self.lens, camera_pose))
            paths.append(path)
            self._log(f"render_scene_views complete prefix={prefix}")
        return paths

    def _build_headless_style_stage(
        self,
        parts: dict[str, PartMesh],
        poses: dict[str, np.ndarray],
        moving_part: str | None,
        show_arrows: bool,
        fixed_axis_part: str | None,
        show_fixed_base_axes: bool,
        camera_pose: np.ndarray,
        camera_look_at: np.ndarray,
    ) -> None:
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        context = omni.usd.get_context()
        self._log("headless-style context.new_stage begin")
        context.new_stage()
        self._log("headless-style context.new_stage done")
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim did not provide a USD stage")

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/Materials")

        def make_preview_material(path: str, color: tuple[float, float, float], opacity: float = 1.0):
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(np.clip(opacity, 0.0, 1.0)))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return material

        material_by_color: dict[tuple[float, float, float, float], Any] = {}

        def material_for(color: tuple[float, float, float], opacity: float = 1.0):
            alpha = float(np.clip(opacity, 0.0, 1.0))
            key = (*tuple(float(x) for x in color), alpha)
            material = material_by_color.get(key)
            if material is None:
                mat_name = self._usd_identifier(
                    "mat_"
                    + "_".join(str(int(round(c * 255.0))) for c in key[:3])
                    + f"_a{int(round(alpha * 255.0))}"
                )
                material = make_preview_material(f"/World/Materials/{mat_name}", key[:3], alpha)
                material_by_color[key] = material
            return material

        mesh_counter = 0
        for part_id, part in sorted(parts.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
            pose = validate_transform(poses.get(part_id, np.eye(4)))
            color = stable_part_color(part_id)
            material = material_for(color)
            for mesh in part.meshes:
                mesh_vertices = transform_points(np.asarray(mesh.vertices, dtype=np.float64), pose)
                mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
                usd_mesh = UsdGeom.Mesh.Define(stage, f"/World/scene_mesh_{mesh_counter:04d}")
                usd_mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in mesh_vertices])
                usd_mesh.CreateFaceVertexCountsAttr([int(len(face)) for face in mesh_faces])
                usd_mesh.CreateFaceVertexIndicesAttr([int(index) for index in mesh_faces.reshape(-1)])
                bmin = mesh_vertices.min(axis=0)
                bmax = mesh_vertices.max(axis=0)
                usd_mesh.CreateExtentAttr(
                    [
                        Gf.Vec3f(float(bmin[0]), float(bmin[1]), float(bmin[2])),
                        Gf.Vec3f(float(bmax[0]), float(bmax[1]), float(bmax[2])),
                    ]
                )
                usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                usd_mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
                usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                usd_mesh.CreateDoubleSidedAttr(True)
                usd_mesh.GetPrim().SetCustomDataByKey("visualAssembly:partId", str(part_id))
                UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)
                mesh_counter += 1

        if self.show_ground_plane:
            ground_mesh = self._ground_mesh(parts, poses)
            ground_material = material_for((0.82, 0.80, 0.72), self.ground_alpha)
            ground_vertices = np.asarray(ground_mesh.vertices, dtype=np.float64)
            ground_faces = np.asarray(ground_mesh.faces, dtype=np.int64)
            ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
            ground.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in ground_vertices])
            ground.CreateFaceVertexCountsAttr([int(len(face)) for face in ground_faces])
            ground.CreateFaceVertexIndicesAttr([int(index) for index in ground_faces.reshape(-1)])
            ground.CreateDisplayColorAttr([Gf.Vec3f(0.82, 0.80, 0.72)])
            ground.CreateDisplayOpacityAttr([float(self.ground_alpha)])
            ground.CreateDoubleSidedAttr(True)
            UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_material)

        if self.show_workcell:
            bmin_raw, bmax_raw = self._raw_scene_bounds(parts, poses, parts.keys())
            for idx, (mesh, color, opacity) in enumerate(self._workcell_meshes_from_bounds(bmin_raw, bmax_raw)):
                material = material_for(color, opacity)
                vertices = np.asarray(mesh.vertices, dtype=np.float64)
                faces = np.asarray(mesh.faces, dtype=np.int64)
                usd_mesh = UsdGeom.Mesh.Define(stage, f"/World/workcell_{idx:02d}")
                usd_mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in vertices])
                usd_mesh.CreateFaceVertexCountsAttr([int(len(face)) for face in faces])
                usd_mesh.CreateFaceVertexIndicesAttr([int(index) for index in faces.reshape(-1)])
                usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                usd_mesh.CreateDisplayOpacityAttr([float(np.clip(opacity, 0.0, 1.0))])
                usd_mesh.CreateDoubleSidedAttr(True)
                UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)

        if show_fixed_base_axes and fixed_axis_part is not None and fixed_axis_part in parts:
            part = parts[fixed_axis_part]
            pose = validate_transform(poses.get(fixed_axis_part, np.eye(4)))
            bmin_w, bmax_w = self.part_world_bounds(part, pose)
            arrow_len = visual_arrow_length_from_bounds(bmin_w, bmax_w)
            for idx, arrow in enumerate(
                build_coordinate_arrows(
                    (bmin_w, bmax_w),
                    arrow_len=arrow_len,
                    colors=FIXED_BASE_AXIS_COLORS,
                )
            ):
                color = FIXED_BASE_AXIS_COLORS[idx]
                material = material_for(color)
                vertices = np.asarray(arrow.vertices, dtype=np.float64)
                faces = np.asarray(arrow.faces, dtype=np.int64)
                usd_mesh = UsdGeom.Mesh.Define(stage, f"/World/fixed_base_axis_{idx}")
                usd_mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in vertices])
                usd_mesh.CreateFaceVertexCountsAttr([int(len(face)) for face in faces])
                usd_mesh.CreateFaceVertexIndicesAttr([int(index) for index in faces.reshape(-1)])
                usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                usd_mesh.CreateDoubleSidedAttr(True)
                UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)

        if moving_part is not None and show_arrows and moving_part in parts:
            part = parts[moving_part]
            pose = validate_transform(poses.get(moving_part, np.eye(4)))
            bmin_w, bmax_w = self.part_world_bounds(part, pose)
            arrow_len = visual_arrow_length_from_bounds(bmin_w, bmax_w)
            for idx, arrow in enumerate(
                build_coordinate_arrows(
                    (bmin_w, bmax_w),
                    arrow_len=arrow_len,
                    colors=MOVING_AXIS_COLORS,
                )
            ):
                color = MOVING_AXIS_COLORS[idx]
                material = material_for(color)
                vertices = np.asarray(arrow.vertices, dtype=np.float64)
                faces = np.asarray(arrow.faces, dtype=np.int64)
                usd_mesh = UsdGeom.Mesh.Define(stage, f"/World/moving_axis_{idx}")
                usd_mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in vertices])
                usd_mesh.CreateFaceVertexCountsAttr([int(len(face)) for face in faces])
                usd_mesh.CreateFaceVertexIndicesAttr([int(index) for index in faces.reshape(-1)])
                usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                usd_mesh.CreateDoubleSidedAttr(True)
                UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(750.0)
        key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        key.CreateIntensityAttr(650.0)
        key.CreateAngleAttr(0.4)

        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        self._current_camera_position = validate_transform(camera_pose)[:3, 3].copy()
        self._current_camera_look_at = np.asarray(camera_look_at, dtype=np.float64).copy()
        self._log(f"headless-style stage authored meshes={mesh_counter}; updating app")
        for _ in range(max(1, min(3, self.capture_updates))):
            self._update_app()
        self._log("headless-style stage app update complete")

    def export_usd_stage(self, parts: dict[str, PartMesh], poses: dict[str, np.ndarray], output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        bmin, bmax = self.compute_scene_bounds(parts, poses, parts.keys())
        center = (bmin + bmax) * 0.5
        size = max(float(np.linalg.norm(bmax - bmin)), 1.0)
        camera_pose = self.view_pose_from_spec(center, size, ViewSpec("export_preview", 32.0, -45.0, "USD export preview"))
        self._ensure_stage_initialized()
        stage = self._update_stage_for_view(
            parts=parts,
            poses=poses,
            moving_part=None,
            show_arrows=False,
            fixed_axis_part=None,
            show_fixed_base_axes=False,
            camera_pose=camera_pose,
            camera_look_at=center,
        )
        stage.GetRootLayer().Export(str(output))
        return output

    def _ensure_stage_initialized(self):
        if self._stage_initialized:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                return stage

        import omni.usd
        from pxr import UsdGeom, UsdLux

        context = omni.usd.get_context()
        self._log("creating new USD stage")
        context.new_stage()
        self._log("new USD stage created; fetching stage")
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim did not provide a USD stage")

        self._log("authoring persistent /World roots, camera, and lights")
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/Parts")
        UsdGeom.Xform.Define(stage, "/World/Axes")
        UsdGeom.Xform.Define(stage, "/World/Workcell")
        UsdGeom.Xform.Define(stage, "/World/Materials")

        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateFocalLengthAttr(float(self.lens))
        camera.CreateHorizontalApertureAttr(float(self.sensor_width_mm))

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(650.0)
        for idx, offset in enumerate(
            [
                np.array([1.0, 1.5, 1.0]),
                np.array([-1.0, 1.2, -1.0]),
            ]
        ):
            light = UsdLux.DistantLight.Define(stage, f"/World/KeyLight_{idx}")
            light.CreateIntensityAttr(450.0)
            light.CreateAngleAttr(0.35)
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = offset
            self._set_xform_matrix(light, pose)

        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        self._stage_initialized = True
        self._log("persistent USD stage initialized; updating app")
        self._update_app()
        self._log("persistent USD stage update complete")
        return stage

    def _update_stage_for_view(
        self,
        parts: dict[str, PartMesh],
        poses: dict[str, np.ndarray],
        moving_part: str | None,
        show_arrows: bool,
        fixed_axis_part: str | None,
        show_fixed_base_axes: bool,
        camera_pose: np.ndarray,
        camera_look_at: np.ndarray | None = None,
    ):
        from pxr import Gf, UsdGeom, UsdLux

        stage = self._ensure_stage_initialized()

        if self.show_ground_plane:
            self._define_mesh(
                stage,
                "/World/ground",
                self._ground_mesh(parts, poses),
                np.eye(4),
                (0.89, 0.82, 0.68),
                opacity=self.ground_alpha,
            )
            self._set_prim_visible(stage, "/World/ground", True)
        else:
            self._set_prim_visible(stage, "/World/ground", False)

        for idx in range(self._workcell_prop_count):
            self._set_prim_visible(stage, f"/World/Workcell/prop_{idx:02d}", False)
        if self.show_workcell:
            bmin_raw, bmax_raw = self._raw_scene_bounds(parts, poses, parts.keys())
            props = self._workcell_meshes_from_bounds(bmin_raw, bmax_raw)
            self._workcell_prop_count = max(self._workcell_prop_count, len(props))
            for idx, (mesh, color, opacity) in enumerate(props):
                prop_path = f"/World/Workcell/prop_{idx:02d}"
                self._define_mesh(stage, prop_path, mesh, np.eye(4), color, opacity=opacity)
                self._set_prim_visible(stage, prop_path, True)

        visible_roots: set[str] = set()
        for part_id, part in sorted(parts.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
            pose = validate_transform(poses.get(part_id, np.eye(4)))
            root_path = self._ensure_part_prim(stage, part)
            root = UsdGeom.Xform(stage.GetPrimAtPath(root_path))
            root.GetPrim().SetCustomDataByKey("visualAssembly:partId", str(part_id))
            self._set_xform_matrix(root, pose)
            self._set_prim_visible(stage, root_path, True)
            visible_roots.add(root_path)

        for root_path in self._part_root_paths.values():
            if root_path not in visible_roots:
                self._set_prim_visible(stage, root_path, False)

        for idx in range(3):
            self._set_prim_visible(stage, f"/World/Axes/moving_axis_{idx}", False)
            self._set_prim_visible(stage, f"/World/Axes/fixed_base_axis_{idx}", False)

        if show_fixed_base_axes and fixed_axis_part is not None and fixed_axis_part in parts:
            part = parts[fixed_axis_part]
            pose = validate_transform(poses.get(fixed_axis_part, np.eye(4)))
            bmin_w, bmax_w = self.part_world_bounds(part, pose)
            arrow_len = visual_arrow_length_from_bounds(bmin_w, bmax_w)
            for idx, arrow in enumerate(
                build_coordinate_arrows(
                    (bmin_w, bmax_w),
                    arrow_len=arrow_len,
                    colors=FIXED_BASE_AXIS_COLORS,
                )
            ):
                axis_path = f"/World/Axes/fixed_base_axis_{idx}"
                self._define_mesh(stage, axis_path, arrow, np.eye(4), FIXED_BASE_AXIS_COLORS[idx])
                self._set_prim_visible(stage, axis_path, True)

        if moving_part is not None and show_arrows and moving_part in parts:
            part = parts[moving_part]
            pose = validate_transform(poses.get(moving_part, np.eye(4)))
            bmin_w, bmax_w = self.part_world_bounds(part, pose)
            arrow_len = visual_arrow_length_from_bounds(bmin_w, bmax_w)
            for idx, arrow in enumerate(
                build_coordinate_arrows(
                    (bmin_w, bmax_w),
                    arrow_len=arrow_len,
                    colors=MOVING_AXIS_COLORS,
                )
            ):
                color = MOVING_AXIS_COLORS[idx]
                axis_path = f"/World/Axes/moving_axis_{idx}"
                self._define_mesh(stage, axis_path, arrow, np.eye(4), color)
                self._set_prim_visible(stage, axis_path, True)

        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        self._set_xform_matrix(camera, validate_transform(camera_pose))
        camera.CreateFocalLengthAttr(float(self.lens))
        camera.CreateHorizontalApertureAttr(float(self.sensor_width_mm))
        self._current_camera_position = validate_transform(camera_pose)[:3, 3].copy()
        self._current_camera_look_at = (
            np.asarray(camera_look_at, dtype=np.float64).copy()
            if camera_look_at is not None
            else np.zeros(3, dtype=np.float64)
        )
        bmin, bmax = self.compute_scene_bounds(parts, poses, parts.keys())
        scene_size = max(float(np.linalg.norm(bmax - bmin)), 1.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, max(100.0, scene_size * 20.0)))

        self._log("stage updated; updating app")
        self._update_app()
        self._log("app update complete")
        return stage

    def _ensure_part_prim(self, stage: Any, part: PartMesh) -> str:
        key = self._part_stage_key(part)
        cached = self._part_root_paths.get(key)
        if cached is not None and stage.GetPrimAtPath(cached):
            return cached

        from pxr import UsdGeom

        digest = hashlib.sha1(key[1].encode("utf-8")).hexdigest()[:10]
        root_path = f"/World/Parts/{self._usd_identifier(f'part_{part.part_id}_{digest}')}"
        self._log(f"defining persistent part {part.part_id} from {part.obj_path}")
        root = UsdGeom.Xform.Define(stage, root_path)
        root.GetPrim().SetCustomDataByKey("visualAssembly:partId", str(part.part_id))
        root.GetPrim().SetCustomDataByKey("visualAssembly:objPath", key[1])
        self._set_xform_matrix(root, np.eye(4, dtype=np.float64))
        color = stable_part_color(part.part_id)
        for mesh_index, mesh in enumerate(part.meshes):
            mesh_name = self._usd_identifier(f"mesh_{mesh_index:02d}")
            self._define_mesh(stage, f"{root_path}/{mesh_name}", mesh, np.eye(4), color)
        self._set_prim_visible(stage, root_path, False)
        self._part_root_paths[key] = root_path
        return root_path

    @staticmethod
    def _part_stage_key(part: PartMesh) -> tuple[str, str]:
        try:
            obj_key = str(part.obj_path.resolve())
        except Exception:
            obj_key = str(part.obj_path)
        return str(part.part_id), obj_key

    @staticmethod
    def _set_prim_visible(stage: Any, path: str, visible: bool) -> None:
        from pxr import UsdGeom

        prim = stage.GetPrimAtPath(path)
        if not prim:
            return
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _capture_current_stage(self, output_path: Path) -> None:
        if self.capture_backend == "rgb_annotator":
            self._capture_with_rgb_annotator(output_path)
            return
        if self.capture_backend == "camera":
            self._capture_with_camera_sensor(output_path)
            return
        if self.capture_backend == "replicator_writer":
            self._capture_with_replicator_writer(output_path)
            return
        if self.capture_backend == "viewport":
            self._capture_with_viewport(output_path)
            return
        if self.capture_backend == "movie_capture":
            self._capture_with_movie_capture(output_path)
            return
        raise RuntimeError(f"Unsupported Isaac Sim capture backend: {self.capture_backend!r}")

    def _ensure_rgb_annotator(self) -> None:
        if self._rgb_render_product is not None and self._rgb_annotator is not None:
            return
        import omni.replicator.core as rep

        position = self._current_camera_position if self._current_camera_position is not None else np.array([1.0, 1.0, 1.0])
        look_at = self._current_camera_look_at if self._current_camera_look_at is not None else np.zeros(3, dtype=np.float64)
        self._log(
            "creating persistent Replicator RGB annotator with rep.create.camera "
            f"position={np.asarray(position).round(4).tolist()} look_at={np.asarray(look_at).round(4).tolist()}"
        )
        self._rgb_camera = rep.create.camera(
            position=tuple(float(x) for x in np.asarray(position, dtype=np.float64)),
            look_at=tuple(float(x) for x in np.asarray(look_at, dtype=np.float64)),
            focal_length=float(self.lens),
        )
        self._rgb_render_product = rep.create.render_product(self._rgb_camera, (self.width, self.height))
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annotator.attach([self._rgb_render_product])
        for _ in range(max(1, self.capture_updates)):
            self._update_app()

    def _capture_with_rgb_annotator(self, output_path: Path) -> None:
        import omni.replicator.core as rep

        output_path.parent.mkdir(parents=True, exist_ok=True)
        position = (
            self._current_camera_position
            if self._current_camera_position is not None
            else np.array([1.0, 1.0, 1.0], dtype=np.float64)
        )
        look_at = (
            self._current_camera_look_at
            if self._current_camera_look_at is not None
            else np.zeros(3, dtype=np.float64)
        )
        self._log(
            "creating one-shot Replicator camera/render product "
            f"position={np.asarray(position).round(4).tolist()} look_at={np.asarray(look_at).round(4).tolist()}"
        )
        camera = rep.create.camera(
            position=tuple(float(x) for x in np.asarray(position, dtype=np.float64)),
            look_at=tuple(float(x) for x in np.asarray(look_at, dtype=np.float64)),
            focal_length=float(self.lens),
        )
        render_product = rep.create.render_product(camera, (self.width, self.height))
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach([render_product])

        try:
            warmup_frames = max(30, int(self.capture_updates))
            self._log(f"RGB annotator warmup frames={warmup_frames}")
            for _ in range(warmup_frames):
                self._update_app()

            rgb: np.ndarray | None = None
            for attempt in range(1, 4):
                self._log(f"RGB annotator capture attempt {attempt}: orchestrator.step begin")
                rep.orchestrator.step(rt_subframes=max(1, self.capture_updates), pause_timeline=False)
                self._log(f"RGB annotator capture attempt {attempt}: orchestrator.step done")
                for _ in range(5):
                    self._update_app()
                self._log(f"RGB annotator capture attempt {attempt}: get_data begin")
                candidate = np.asarray(rgb_annot.get_data())
                self._log(f"RGB annotator capture attempt {attempt}: get_data done")
                rgb = self._rgb_data_to_uint8(candidate)
                self._log(
                    "RGB annotator stats: "
                    f"shape={rgb.shape}, min={int(rgb.min())}, max={int(rgb.max())}"
                )
                if int(rgb.max()) > 0:
                    break

            if rgb is None or int(rgb.max()) == 0:
                raise RuntimeError("Replicator RGB annotator returned an all-black image after capture attempts")
            write_image(output_path, rgb)
        finally:
            try:
                rgb_annot.detach([render_product])
            except Exception:
                pass
            try:
                if hasattr(render_product, "destroy"):
                    render_product.destroy()
            except Exception:
                pass
            try:
                if hasattr(camera, "destroy"):
                    camera.destroy()
            except Exception:
                pass

    def _update_replicator_camera_pose(self) -> None:
        if self._rgb_camera is None:
            return
        if self._current_camera_position is None or self._current_camera_look_at is None:
            return
        import omni.replicator.core as rep

        position = np.asarray(self._current_camera_position, dtype=np.float64)
        look_at = np.asarray(self._current_camera_look_at, dtype=np.float64)
        self._log(
            "updating Replicator camera pose "
            f"position={position.round(4).tolist()} look_at={look_at.round(4).tolist()}"
        )
        with self._rgb_camera:
            rep.modify.pose(
                position=tuple(float(x) for x in position),
                look_at=tuple(float(x) for x in look_at),
            )
        for _ in range(max(1, min(3, self.capture_updates))):
            self._update_app()

    @staticmethod
    def _rgb_data_to_uint8(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise RuntimeError(f"RGB annotator returned invalid data with shape {arr.shape}")
        rgb = arr[:, :, :3]
        if rgb.dtype.kind == "f":
            if float(np.nanmax(rgb)) <= 1.0:
                rgb = rgb * 255.0
            return np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _capture_with_camera_sensor(self, output_path: Path) -> None:
        self._ensure_extension_enabled("isaacsim.sensors.camera")
        from isaacsim.sensors.camera import Camera

        output_path.parent.mkdir(parents=True, exist_ok=True)
        camera = Camera(
            prim_path="/World/Camera",
            name="visual_assembly_camera",
            resolution=(self.width, self.height),
        )
        try:
            print("[IsaacSimSceneRenderer] initializing Isaac Camera sensor", flush=True)
            camera.initialize(attach_rgb_annotator=True)
            print(
                "[IsaacSimSceneRenderer] Camera render product: "
                f"{camera.get_render_product_path()}",
                flush=True,
            )
            warmup_frames = max(30, self.capture_updates)
            for _ in range(warmup_frames):
                self._update_app()
            rgba = camera.get_rgba(device="cpu")
            if rgba is None:
                raise RuntimeError("Camera.get_rgba() returned None")
            rgba_array = np.asarray(rgba)
            if rgba_array.ndim != 3 or rgba_array.shape[-1] < 3 or rgba_array.size == 0:
                raise RuntimeError(f"Camera.get_rgba() returned invalid data with shape {rgba_array.shape}")
            rgb = rgba_array[:, :, :3]
            if rgb.dtype.kind == "f":
                if float(np.nanmax(rgb)) <= 1.0:
                    rgb = rgb * 255.0
                rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            print(
                "[IsaacSimSceneRenderer] Camera RGBA stats: "
                f"shape={rgba_array.shape}, rgb_min={int(rgb.min())}, rgb_max={int(rgb.max())}",
                flush=True,
            )
            write_image(output_path, rgb)
        finally:
            try:
                camera.destroy()
            except Exception:
                pass

    def _capture_with_replicator_writer(self, output_path: Path) -> None:
        import omni.replicator.core as rep

        with tempfile.TemporaryDirectory(prefix="visual_agent_isaacsim_capture_") as tmp:
            render_product = rep.create.render_product("/World/Camera", (self.width, self.height))
            writer = rep.writers.get("BasicWriter")
            writer.initialize(output_dir=tmp, rgb=True)
            writer.attach(render_product)

            async def capture_once() -> None:
                await rep.orchestrator.step_async(
                    rt_subframes=self.capture_updates,
                    delta_time=0.0,
                    pause_timeline=False,
                )
                writer.detach()
                if hasattr(render_product, "destroy"):
                    render_product.destroy()
                await rep.orchestrator.wait_until_complete_async()

            self._run_isaacsim_coroutine(capture_once(), timeout_s=90.0)
            pngs = sorted(Path(tmp).rglob("*.png"))
            if not pngs:
                raise RuntimeError("Replicator did not produce a PNG")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pngs[0], output_path)

    def _ensure_extension_enabled(self, extension_name: str) -> None:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        if manager.is_extension_enabled(extension_name):
            print(f"[IsaacSimSceneRenderer] Isaac extension already enabled: {extension_name}", flush=True)
            return
        print(f"[IsaacSimSceneRenderer] enabling Isaac extension: {extension_name}", flush=True)
        manager.set_extension_enabled_immediate(extension_name, True)
        for _ in range(3):
            self._update_app()

    def _get_or_create_viewport(self):
        from pxr import Sdf
        from omni.kit.viewport.utility import (
            create_viewport_window,
            get_active_viewport,
        )

        camera_path = Sdf.Path("/World/Camera")
        print("[IsaacSimSceneRenderer] looking up active Isaac viewport", flush=True)
        viewport = get_active_viewport()
        if viewport is None:
            print("[IsaacSimSceneRenderer] creating Isaac viewport window", flush=True)
            viewport_window = create_viewport_window(
                name="VisualAssemblyCapture",
                width=self.width,
                height=self.height,
                camera_path=camera_path,
            )
            viewport = viewport_window.viewport_api if viewport_window is not None else None
        if viewport is None:
            raise RuntimeError("No active Isaac Sim viewport is available")

        print("[IsaacSimSceneRenderer] binding viewport to /World/Camera", flush=True)
        try:
            import omni.kit.commands

            omni.kit.commands.execute("SetViewportCamera", camera_path=str(camera_path), viewport_api=viewport)
        except Exception:
            viewport.camera_path = camera_path
        try:
            viewport.updates_enabled = True
        except Exception:
            pass
        try:
            viewport.set_texture_resolution((self.width, self.height))
        except Exception:
            pass
        try:
            viewport.resolution = (self.width, self.height)
        except Exception:
            pass
        return viewport

    def _capture_with_viewport(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        viewport = self._get_or_create_viewport()
        if output_path.exists():
            output_path.unlink()
        for _ in range(max(1, self.capture_updates)):
            self._update_app()
        print("[IsaacSimSceneRenderer] scheduling viewport file capture", flush=True)
        self._capture_viewport_frame_to_file(viewport, output_path)
        deadline = time.monotonic() + 90.0
        while not output_path.exists() and time.monotonic() < deadline:
            self._update_app()
            time.sleep(0.05)
        if not output_path.exists():
            raise RuntimeError("Viewport capture did not produce a PNG")

    def _capture_viewport_frame_to_file(self, viewport: Any, output_path: Path) -> None:
        from omni.kit.viewport.utility import capture_viewport_to_file

        capture = capture_viewport_to_file(viewport, file_path=str(output_path), is_hdr=False)
        if capture is None:
            raise RuntimeError("Viewport capture could not be scheduled")

        async def wait_for_capture() -> Any:
            return await capture.wait_for_result(completion_frames=max(2, self.capture_updates))

        result = self._run_isaacsim_coroutine(wait_for_capture(), timeout_s=90.0)
        print(f"[IsaacSimSceneRenderer] viewport capture wait result: {result}", flush=True)

    def _capture_with_movie_capture(self, output_path: Path) -> None:
        self._ensure_extension_enabled("omni.kit.capture.viewport")
        from omni.kit.capture.viewport import CaptureExtension, CaptureOptions

        output_path.parent.mkdir(parents=True, exist_ok=True)
        viewport = self._get_or_create_viewport()

        with tempfile.TemporaryDirectory(prefix="visual_agent_isaacsim_movie_capture_") as tmp:
            capture = CaptureExtension.get_instance()
            if capture is None:
                raise RuntimeError("omni.kit.capture.viewport did not provide a CaptureExtension instance")

            try:
                capture.show_default_progress_window = False
            except Exception:
                pass

            options = CaptureOptions()
            options.camera = viewport.camera_path.pathString
            options.output_folder = tmp
            options.file_name = output_path.stem
            options.file_type = ".png"
            options.res_width = self.width
            options.res_height = self.height
            options.spp_per_iteration = 1
            options.path_trace_spp = 1
            options.overwrite_existing_frames = True
            capture.options = options

            print(f"[IsaacSimSceneRenderer] starting Movie Capture current-frame capture in {tmp}", flush=True)
            started = capture.start()
            print(f"[IsaacSimSceneRenderer] Movie Capture start returned {started}", flush=True)
            if not started:
                raise RuntimeError("Movie Capture current-frame capture did not start")

            deadline = time.monotonic() + 120.0
            last_progress_log = 0.0
            while not capture.done and time.monotonic() < deadline:
                now = time.monotonic()
                if now - last_progress_log >= 5.0:
                    print("[IsaacSimSceneRenderer] waiting for Movie Capture to finish", flush=True)
                    last_progress_log = now
                self._update_app()
                time.sleep(0.02)
            if not capture.done:
                raise TimeoutError("Movie Capture current-frame capture timed out")

            reported_outputs = [Path(path) for path in capture.get_outputs()]
            print(
                "[IsaacSimSceneRenderer] Movie Capture reported outputs: "
                f"{[str(path) for path in reported_outputs]}",
                flush=True,
            )
            deadline = time.monotonic() + 30.0
            outputs: list[Path] = []
            while time.monotonic() < deadline:
                discovered = sorted(Path(tmp).rglob("*.png"))
                candidates = [path for path in [*reported_outputs, *discovered] if path.exists()]
                outputs = sorted(set(candidates))
                if outputs:
                    break
                self._update_app()
                time.sleep(0.05)
            print(
                "[IsaacSimSceneRenderer] Movie Capture tmp PNGs: "
                f"{[str(path) for path in sorted(Path(tmp).rglob('*.png'))]}",
                flush=True,
            )
            if not outputs:
                fallback = Path(tmp) / f"{output_path.stem}_viewport_fallback.png"
                print(
                    "[IsaacSimSceneRenderer] Movie Capture reported no readable PNG; "
                    f"waiting on viewport current-frame capture fallback at {fallback}",
                    flush=True,
                )
                self._capture_viewport_frame_to_file(viewport, fallback)
                if fallback.exists():
                    outputs = [fallback]
            if not outputs:
                raise RuntimeError("Movie Capture did not produce a PNG")
            if output_path.exists():
                output_path.unlink()
            shutil.copyfile(outputs[0], output_path)

    def _run_isaacsim_coroutine(self, coroutine: Any, timeout_s: float) -> Any:
        from omni.kit.async_engine import run_coroutine

        task = run_coroutine(coroutine)
        deadline = time.monotonic() + float(timeout_s)
        while not task.done():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Isaac Sim async capture timed out after {timeout_s:.0f}s")
            self._update_app()
        return task.result()

    def _update_app(self) -> None:
        if self._simulation_app is not None:
            self._simulation_app.update()
            return
        import omni.kit.app

        omni.kit.app.get_app().update()

    def _label_capture(self, path: Path, label: str) -> None:
        from PIL import Image

        color = np.asarray(Image.open(path).convert("RGB"))
        write_image(path, add_image_label(color, label))

    @staticmethod
    def _usd_identifier(value: str) -> str:
        from pxr import Tf

        identifier = Tf.MakeValidIdentifier(str(value))
        return identifier or "prim"

    def _define_mesh(
        self,
        stage: Any,
        path: str,
        mesh: Any,
        pose: np.ndarray,
        color: tuple[float, float, float],
        opacity: float = 1.0,
    ) -> None:
        from pxr import Gf, UsdGeom

        usd_mesh = UsdGeom.Mesh.Define(stage, path)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError(f"USD mesh {path} has no valid vertex array")
        if faces.ndim != 2 or faces.shape[1] < 3 or len(faces) == 0:
            raise ValueError(f"USD mesh {path} has no valid polygon face array")
        usd_mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in vertices])
        usd_mesh.CreateFaceVertexCountsAttr([int(len(face)) for face in faces])
        usd_mesh.CreateFaceVertexIndicesAttr([int(index) for index in faces.reshape(-1)])
        bmin = vertices.min(axis=0)
        bmax = vertices.max(axis=0)
        usd_mesh.CreateExtentAttr(
            [
                Gf.Vec3f(float(bmin[0]), float(bmin[1]), float(bmin[2])),
                Gf.Vec3f(float(bmax[0]), float(bmax[1]), float(bmax[2])),
            ]
        )
        usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        usd_mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        usd_mesh.CreateDisplayOpacityAttr([float(np.clip(opacity, 0.0, 1.0))])
        usd_mesh.CreateDoubleSidedAttr(True)
        xformable = UsdGeom.Xformable(usd_mesh.GetPrim())
        self._set_xform_matrix(xformable, validate_transform(pose))

    def _set_xform_matrix(self, xformable: Any, transform: np.ndarray) -> None:
        from pxr import Gf

        transform = validate_transform(transform)
        try:
            xformable.ClearXformOpOrder()
        except Exception:
            pass
        op = xformable.MakeMatrixXform()
        matrix = Gf.Matrix4d(1.0)
        for row in range(4):
            for col in range(4):
                matrix[row, col] = float(transform[col, row])
        op.Set(matrix)

    def _ground_mesh(self, parts: dict[str, PartMesh], poses: dict[str, np.ndarray]):
        import trimesh

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
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def create_simulation_backend(config: SceneSimulationConfig) -> SceneSimulationBackend:
    if config.backend == "isaacsim":
        return IsaacSimSimulationBackend(config)
    return SceneSimulationBackend(config)
