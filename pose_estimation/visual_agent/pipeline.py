from __future__ import annotations

import json
import os
import shutil
import sys
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .clients import (
    BaseVisualVLMClient,
    EvaluatorResult,
    MockVisualVLMClient,
    OpenAIVisualVLMClient,
    ProposerResult,
)
from .data import BenchApplaroInputs, OperationSpec, load_bench_applaro_inputs
from .geometry import (
    bounds_for_points,
    scale_rotation_matrix,
    translation_scale_from_bounds,
    transform_distance,
    transform_points,
    transform_to_json,
    validate_transform,
)
from .prompts import SYSTEM_PROMPT, build_evaluator_prompt, build_proposer_prompt
from .rendering import (
    PartMesh,
    PosedSceneRenderer,
    ViewSpec,
    load_part_meshes,
    render_current_orthographic_input,
    render_orthographic_measurement_input,
)
from .simulation import SceneSimulationConfig, create_simulation_backend, normalize_isaacsim_capture_backend


@dataclass
class VisualAssemblyAgentConfig:
    category: str = "Bench"
    name: str = "applaro"
    data_root: Path = Path(__file__).resolve().parents[2] / "examples/data"
    manual2skill_output_root: Path = Path(__file__).resolve().parents[2] / "examples/stage_outputs"
    output_dir: Path = Path("outputs/visual_vlm_agent")
    model: str | None = None
    base_url: str | None = None
    dry_run: bool = False
    num_views: int = 8
    path_samples: int = 5
    max_iters: int = 5
    render_width: int = 768
    render_height: int = 768
    camera_distance_scale: float = 1.15
    camera_frame_padding: float = 1.25
    render_ground_plane: bool = False
    render_fixed_base_axes: bool = False
    render_workcell: bool = False
    ground_y: float = -0.05
    ground_alpha: float = 0.62
    max_image_side: int = 1200
    temperature: float = 0.35
    api_retries: int = 2
    retry_delay: float = 2.0
    json_repair_retries: int = 1
    vote_threshold: float = 0.35
    accept_confidence: float = 0.70
    accept_ambiguity: float = 0.30
    strict_final_judge: bool = True
    mock_accept_distance: float = 0.05
    show_progress: bool = True
    log_file: Path | None = None
    trust_env_proxy: bool = True
    proxy_url: str | None = None
    fixed_views: bool = False
    commit_failed_paths: bool = False
    rotation_mode: str = "matrix"
    progressive_update_pressure: bool = True
    max_update_scale: float = 2.5
    max_translation_units: float = 4.0
    max_rotation_deg: float = 180.0
    continue_after_failed_operation: bool = False
    simulation_backend: str = "offline"
    isaacsim_headless: bool = True
    isaacsim_renderer: str = "auto"
    isaacsim_capture_updates: int = 8
    proposer_input_mode: str = "current_only_orthographic"
    translation_unit_mode: str = "scene_units"

    def __post_init__(self) -> None:
        self.rotation_mode = str(self.rotation_mode or "matrix").strip().lower()
        if self.rotation_mode not in {"axis_angle", "matrix"}:
            raise ValueError(f"rotation_mode must be 'axis_angle' or 'matrix', got {self.rotation_mode!r}")
        self.simulation_backend = str(self.simulation_backend or "offline").strip().lower()
        if self.simulation_backend not in {"offline", "isaacsim"}:
            raise ValueError(
                f"simulation_backend must be 'offline' or 'isaacsim', got {self.simulation_backend!r}"
            )
        self.proposer_input_mode = str(self.proposer_input_mode or "current_only_orthographic").strip().lower()
        if self.proposer_input_mode not in {"supporting_view", "orthographic", "current_only_orthographic"}:
            raise ValueError(
                "proposer_input_mode must be 'supporting_view', 'orthographic', or "
                "'current_only_orthographic', "
                f"got {self.proposer_input_mode!r}"
            )
        self.translation_unit_mode = str(self.translation_unit_mode or "scene_units").strip().lower()
        if self.translation_unit_mode not in {"vlm_pose", "scene_units"}:
            raise ValueError(
                "translation_unit_mode must be 'vlm_pose' or 'scene_units', "
                f"got {self.translation_unit_mode!r}"
            )
        self.isaacsim_renderer = normalize_isaacsim_capture_backend(self.isaacsim_renderer)
        self.max_update_scale = max(1.0, float(self.max_update_scale))
        self.max_translation_units = max(0.0, float(self.max_translation_units))
        self.max_rotation_deg = max(0.0, float(self.max_rotation_deg))
        self.isaacsim_capture_updates = max(1, int(self.isaacsim_capture_updates))
        self.camera_distance_scale = max(0.1, float(self.camera_distance_scale))
        self.camera_frame_padding = max(1.0, float(self.camera_frame_padding))
        self.ground_y = float(self.ground_y)
        self.ground_alpha = min(max(float(self.ground_alpha), 0.0), 1.0)
        if self.commit_failed_paths:
            self.continue_after_failed_operation = True


class VisualRunLogger:
    def __init__(self, enabled: bool, total: int, log_path: Path):
        self.enabled = enabled
        self.total = max(1, int(total))
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = sys.stderr
        self.completed = 0
        self._file = self.log_path.open("a", encoding="utf-8")
        self._file.write(f"\n=== visual_vlm_assembly_agent run {datetime.now(timezone.utc).isoformat()} ===\n")
        self._file.flush()
        self.pbar = (
            tqdm(
                total=self.total,
                desc="visual_agent",
                unit="iter",
                dynamic_ncols=True,
                leave=True,
                file=self.stream,
            )
            if self.enabled
            else None
        )

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        if self.pbar is not None:
            self.pbar.write(line)
            self.stream.flush()
        else:
            print(line, file=self.stream, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def set_detail(self, detail: str) -> None:
        if self.pbar is not None:
            self.pbar.set_postfix_str(detail, refresh=True)

    def advance(self, steps: int = 1, detail: str = "") -> None:
        steps = max(0, int(steps))
        if self.pbar is not None:
            if detail:
                self.pbar.set_postfix_str(detail, refresh=False)
            remaining = max(0, self.total - self.completed)
            delta = min(steps, remaining)
            if delta > 0:
                self.pbar.update(delta)
                self.completed += delta
            else:
                self.pbar.refresh()

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
        self._file.close()


@dataclass
class CandidatePath:
    sample_id: int
    moving_part: str
    source: str
    valid: bool
    success: bool
    final_pose: np.ndarray
    final_distance_to_mock_target: float | None
    iterations: list[dict[str, Any]]
    final_evaluator: dict[str, Any] | None
    final_views: list[str]
    score: float = 0.0
    vote_detail: dict[str, Any] | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "moving_part": self.moving_part,
            "source": self.source,
            "valid": self.valid,
            "success": self.success,
            "score": round(float(self.score), 6),
            "vote_detail": self.vote_detail or {},
            "final_pose": transform_to_json(self.final_pose),
            "final_distance_to_mock_target": self.final_distance_to_mock_target,
            "iterations": self.iterations,
            "final_evaluator": self.final_evaluator,
            "final_views": self.final_views,
            "error": self.error,
        }


@dataclass(frozen=True)
class ScaledPoseUpdate:
    translation: np.ndarray
    axis: str
    angle: float
    rotation_matrix: np.ndarray | None
    update_scale: float
    residual: str
    remaining_iters: int
    translation_clamped: bool
    rotation_clamped: bool
    matrix_angle_before: float | None = None
    matrix_angle_after: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "Translation": self.translation.round(6).tolist(),
            "Dominant rotation axis": self.axis,
            "Angle": round(float(self.angle), 6),
            "Rotation matrix": (
                self.rotation_matrix.round(6).tolist() if self.rotation_matrix is not None else None
            ),
            "update_scale": round(float(self.update_scale), 6),
            "target_residual": self.residual,
            "remaining_iters": int(self.remaining_iters),
            "translation_clamped": self.translation_clamped,
            "rotation_clamped": self.rotation_clamped,
            "matrix_angle_before": (
                round(float(self.matrix_angle_before), 6) if self.matrix_angle_before is not None else None
            ),
            "matrix_angle_after": (
                round(float(self.matrix_angle_after), 6) if self.matrix_angle_after is not None else None
            ),
        }


class VisualAssemblyAgent:
    def __init__(self, config: VisualAssemblyAgentConfig, client: BaseVisualVLMClient | None = None):
        self.config = config
        self.inputs = load_bench_applaro_inputs(
            data_root=config.data_root,
            manual2skill_output_root=config.manual2skill_output_root,
            category=config.category,
            name=config.name,
        )
        self.initial_parts = load_part_meshes(self.inputs.scene_parts_dir)
        self.target_parts = (
            {}
            if config.proposer_input_mode == "current_only_orthographic"
            else load_part_meshes(self.inputs.target_parts_dir)
        )
        self.client = client or self._make_client()
        self.output_root = Path(config.output_dir) / config.category / config.name
        self.target_view_cache: dict[tuple[int, str, str], list[Path]] = {}
        self.simulation = create_simulation_backend(
            SceneSimulationConfig(
                backend=config.simulation_backend,
                width=config.render_width,
                height=config.render_height,
                camera_distance_scale=config.camera_distance_scale,
                camera_frame_padding=config.camera_frame_padding,
                show_ground_plane=config.render_ground_plane,
                ground_y=config.ground_y,
                ground_alpha=config.ground_alpha,
                show_workcell=config.render_workcell,
                isaacsim_headless=config.isaacsim_headless,
                isaacsim_renderer=config.isaacsim_renderer,
                isaacsim_capture_updates=config.isaacsim_capture_updates,
                scene_obj=self.inputs.scene_obj,
            )
        )

    def _make_client(self) -> BaseVisualVLMClient:
        if self.config.dry_run:
            return MockVisualVLMClient()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY for live inference, or explicitly use --dry-run for a mock smoke test.")
        return OpenAIVisualVLMClient(
            model=self.config.model,
            base_url=self.config.base_url,
            max_image_side=self.config.max_image_side,
            api_retries=self.config.api_retries,
            retry_delay=self.config.retry_delay,
            json_repair_retries=self.config.json_repair_retries,
            trust_env_proxy=self.config.trust_env_proxy,
            proxy_url=self.config.proxy_url,
            rotation_mode=self.config.rotation_mode,
        )

    def run(self) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        total_progress_units = (
            len(self.inputs.operations)
            * max(1, self.config.path_samples)
            * max(1, self.config.max_iters)
        )
        progress = VisualRunLogger(
            enabled=self.config.show_progress,
            total=total_progress_units,
            log_path=self.config.log_file or (self.output_root / "visual_agent_run.log"),
        )
        out_path = self.output_root / "visual_agent_results.json"
        renderer = self.simulation.renderer
        if hasattr(renderer, "set_log_callback"):
            renderer.set_log_callback(progress.log)
        try:
            progress.log(
                "start "
                f"category={self.config.category} name={self.config.name} "
                f"model={self.config.model} base_url={self.config.base_url} "
                f"dry_run={self.config.dry_run} used_vlm={not self.client.is_mock}"
            )
            progress.log(
                "rendering "
                f"ground_plane={self.config.render_ground_plane} "
                f"fixed_base_axes={self.config.render_fixed_base_axes} "
                f"workcell={self.config.render_workcell} "
                f"fixed_base_axis_part={self._fixed_base_axis_part() or 'none'} "
                f"ground_y={self.config.ground_y} "
                f"ground_alpha={self.config.ground_alpha}"
            )
            progress.log(
                "simulation "
                f"backend={self.config.simulation_backend} "
                f"isaacsim_headless={self.config.isaacsim_headless} "
                f"isaacsim_renderer={self.config.isaacsim_renderer} "
                f"isaacsim_capture_updates={self.config.isaacsim_capture_updates} "
                f"camera_distance_scale={self.config.camera_distance_scale} "
                f"camera_frame_padding={self.config.camera_frame_padding}"
            )
            progress.log(
                "network "
                f"trust_env_proxy={self.config.trust_env_proxy} "
                f"proxy_url={'set' if self.config.proxy_url else 'unset'} "
                f"HTTP_PROXY={'set' if os.getenv('HTTP_PROXY') else 'unset'} "
                f"HTTPS_PROXY={'set' if os.getenv('HTTPS_PROXY') else 'unset'} "
                f"ALL_PROXY={'set' if os.getenv('ALL_PROXY') else 'unset'}"
            )
            progress.log(
                f"schedule operations={len(self.inputs.operations)} "
                f"path_samples={self.config.path_samples} max_iters={self.config.max_iters} "
                f"num_views={self.config.num_views} fixed_views={self.config.fixed_views} "
                f"accept_confidence={self.config.accept_confidence} "
                f"accept_ambiguity={self.config.accept_ambiguity} "
                f"rotation_mode={self.config.rotation_mode} "
                f"progressive_update_pressure={self.config.progressive_update_pressure} "
                f"max_update_scale={self.config.max_update_scale} "
                f"continue_after_failed_operation={self.config.continue_after_failed_operation}"
            )
            result = self._run_with_renderer(renderer, progress)
            result["run_log"] = str(progress.log_path)
            result["simulation_backend"] = self.simulation.to_json()
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            progress.log(f"wrote result_json={out_path}")
        finally:
            self.simulation.close()
            self.target_view_cache.clear()
            progress.close()

        return out_path

    def _run_with_renderer(self, renderer: PosedSceneRenderer, progress: VisualRunLogger) -> dict[str, Any]:
        committed_poses = {pid: np.eye(4, dtype=np.float64) for pid in self.initial_parts}
        base_part = self._fixed_base_axis_part()
        assembled_parts = [base_part] if base_part is not None else []
        global_memory: list[dict[str, Any]] = []
        operations_json: list[dict[str, Any]] = []

        initial_view_specs = (
            self._view_specs_for_observation(renderer, self.inputs.operations[0], committed_poses)
            if self.inputs.operations
            else None
        )
        progress.log(f"render initial scene views views={view_spec_names(initial_view_specs)}")
        self._render_initial_assets(renderer, committed_poses, initial_view_specs)
        for operation in self.inputs.operations:
            moving_name = self.inputs.part_names.get(operation.moving_part, "unknown")
            progress.log(
                f"operation {operation.operation_index} start "
                f"moving={operation.moving_part}({moving_name}) fixed={operation.fixed_parts} "
                f"manual_step={operation.manual_step_index}"
            )
            op_dir = self.output_root / f"operation_{operation.operation_index:02d}_part_{operation.moving_part}"
            op_dir.mkdir(parents=True, exist_ok=True)
            initial_view_specs = self._view_specs_for_observation(renderer, operation, committed_poses)
            if self._uses_current_only_visual_loop():
                progress.log(
                    f"operation {operation.operation_index}: current-only visual loop; "
                    "skip target-effect views"
                )
                target_views = []
            else:
                progress.log(
                    f"operation {operation.operation_index}: render target-effect views "
                    f"views={view_spec_names(initial_view_specs)}"
                )
                target_views = self._target_views(renderer, operation, initial_view_specs, op_dir)
            candidates = []
            for sample_id in range(max(1, self.config.path_samples)):
                progress.log(f"operation {operation.operation_index}: candidate {sample_id} start")
                candidate = self._sample_candidate_path(
                    renderer=renderer,
                    operation=operation,
                    sample_id=sample_id,
                    committed_poses=committed_poses,
                    global_memory=global_memory,
                    target_views=target_views,
                    op_dir=op_dir,
                    progress=progress,
                )
                candidates.append(candidate)
                progress.log(
                    f"operation {operation.operation_index}: candidate {sample_id} done "
                    f"valid={candidate.valid} success={candidate.success} "
                    f"iters={len(candidate.iterations)} error={candidate.error or 'none'}"
                )

            selected, vote_info = vote_candidate_paths(
                candidates,
                moving_part=operation.moving_part,
                moving_mesh=self.initial_parts[operation.moving_part],
                vote_threshold=self.config.vote_threshold,
            )
            progress.log(
                f"operation {operation.operation_index}: vote selected_sample={selected.sample_id} "
                f"score={selected.score:.4f} success={selected.success} "
                f"scores={vote_info['scores']}"
            )
            can_commit = selected.success or self.config.continue_after_failed_operation or self.config.commit_failed_paths
            commit_status = (
                "committed_success"
                if selected.success
                else ("committed_provisional_failed_path" if can_commit else "rejected_failed_path")
            )
            selected_preview_poses = {pid: T.copy() for pid, T in committed_poses.items()}
            selected_preview_poses[operation.moving_part] = selected.final_pose.copy()
            if can_commit:
                committed_poses[operation.moving_part] = selected.final_pose.copy()
                if operation.moving_part not in assembled_parts:
                    assembled_parts.append(operation.moving_part)
                self.simulation.export_posed_scene(
                    self.initial_parts,
                    committed_poses,
                    op_dir / "committed_scene_after_operation.obj",
                )
            else:
                self.simulation.export_posed_scene(
                    self._operation_initial_parts(operation),
                    selected_preview_poses,
                    op_dir / "rejected_failed_candidate_subassembly.obj",
                )
            global_memory.append(
                {
                    "operation_index": operation.operation_index,
                    "moving_part": operation.moving_part,
                    "assembled_parts": list(assembled_parts),
                    "selected_sample_id": selected.sample_id,
                    "selected_success": selected.success,
                    "selected_score": selected.score,
                    "commit_status": commit_status,
                }
            )
            operations_json.append(
                {
                    "operation": operation.to_json(),
                    "target_views": self._target_views_record(operation, target_views),
                    "candidate_paths": [candidate.to_json() for candidate in candidates],
                    "vote": vote_info,
                    "selected_path": selected.to_json(),
                    "commit": {
                        "status": commit_status,
                        "committed": can_commit,
                        "stop_after_operation": not can_commit,
                        "reason": (
                            "selected path passed final evaluator"
                            if selected.success
                            else (
                                "selected path failed final evaluator; committed provisionally because continue_after_failed_operation is enabled"
                                if can_commit
                                else "selected path failed final evaluator; not committed to avoid polluting later operations"
                            )
                        ),
                    },
                }
            )
            if can_commit:
                progress.log(
                    f"operation {operation.operation_index}: committed moving_part={operation.moving_part} "
                    f"status={commit_status} assembled_parts={assembled_parts}"
                )
            else:
                progress.log(
                    f"operation {operation.operation_index}: selected path failed and was not committed; "
                    "stop remaining operations to keep global scene clean"
                )
                progress.advance(
                    progress.total - progress.completed,
                    detail=f"op={operation.operation_index} stopped_after_failed_operation",
                )
                break

        progress.log("export final assembled scene and render final views")
        final_scene = self.simulation.export_posed_scene(
            self.initial_parts,
            committed_poses,
            self.output_root / "final_assembled_scene.obj",
        )
        final_views = renderer.render_scene_views(
            self.initial_parts,
            committed_poses,
            self.output_root / "final_views",
            "final_assembled",
            self.config.num_views,
            focus_ids=list(assembled_parts),
            fixed_axis_part=self._fixed_base_axis_part(),
            show_fixed_base_axes=self.config.render_fixed_base_axes,
        )
        return {
            "version": "visual_vlm_assembly_agent.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "used_vlm": not self.client.is_mock,
            "config": {
                "category": self.config.category,
                "name": self.config.name,
                "model": self.config.model,
                "base_url": self.config.base_url,
                "num_views": self.config.num_views,
                "path_samples": self.config.path_samples,
                "max_iters": self.config.max_iters,
                "accept_confidence": self.config.accept_confidence,
                "accept_ambiguity": self.config.accept_ambiguity,
                "dry_run": self.config.dry_run,
                "show_progress": self.config.show_progress,
                "log_file": str(self.config.log_file or (self.output_root / "visual_agent_run.log")),
                "trust_env_proxy": self.config.trust_env_proxy,
                "proxy_url": self.config.proxy_url,
                "fixed_views": self.config.fixed_views,
                "commit_failed_paths": self.config.commit_failed_paths,
                "rotation_mode": self.config.rotation_mode,
                "progressive_update_pressure": self.config.progressive_update_pressure,
                "max_update_scale": self.config.max_update_scale,
                "max_translation_units": self.config.max_translation_units,
                "max_rotation_deg": self.config.max_rotation_deg,
                "continue_after_failed_operation": self.config.continue_after_failed_operation,
                "strict_final_judge": self.config.strict_final_judge,
                "camera_distance_scale": self.config.camera_distance_scale,
                "camera_frame_padding": self.config.camera_frame_padding,
                "render_fixed_base_axes": self.config.render_fixed_base_axes,
                "render_workcell": self.config.render_workcell,
                "fixed_base_axis_part": self._fixed_base_axis_part(),
                "simulation_backend": self.config.simulation_backend,
                "isaacsim_headless": self.config.isaacsim_headless,
                "isaacsim_renderer": self.config.isaacsim_renderer,
                "isaacsim_capture_updates": self.config.isaacsim_capture_updates,
                "proposer_input_mode": self.config.proposer_input_mode,
                "translation_unit_mode": self.config.translation_unit_mode,
            },
            "inputs": self.inputs.to_json(),
            "method": {
                "base_anchor": self._fixed_base_axis_part(),
                "pose_update": "incremental translation plus rotation about moving-part world AABB center; orthographic proposer mode uses scene-unit translation, otherwise VLM-Pose AABB-normalized units; rotation is dominant-axis angle by default or a world incremental 3x3 matrix when rotation_mode=matrix",
                "sampling_unit": "complete per-moving-part closed-loop path",
                "target_pose_matrix_policy": "current_only_orthographic does not load target_parts/GT OBJ and does not render target-effect views; oracle orthographic mode remains available only as a debug baseline",
                "view_policy": "current_only_orthographic renders the active operation subset only: fixed subassembly plus the current moving part; oracle orthographic mode uses shared current/target camera orientations",
                "proposer_input_policy": "current_only_orthographic gives proposer current front/right/top measurement views of the active operation subset plus the manual crop; orthographic remains an oracle/debug mode with current/target front/right/top views",
                "operation_render_scope": "current_only_orthographic hides loose parts unrelated to the current operation so they do not pollute the visual judgment; initial full-scene views remain saved as part-id context",
                "target_view_storage": "target-effect views are skipped in current_only_orthographic; oracle/debug modes write them under each operation's target_effect_views directory",
                "simulation_backend": "offline backend uses pyrender/matplotlib; isaacsim backend builds USD stages from the same part OBJ meshes, captures current/target views in Isaac Sim, applies pose updates through the backend hook, and writes sidecar USD stages beside exported OBJ scenes",
                "workcell_rendering": "when enabled, a visual-only table and simplified robot arm are generated around the current part AABB for more realistic presentation; these props do not affect pose updates, evaluation geometry, or collision checks",
                "fixed_base_axes": "when enabled, cyan/magenta/yellow axes are rendered near the first fixed part from the operation specs as background world-axis references; pose update remains controlled by moving-part red/green/blue axes",
                "success_policy": "accepted only when evaluator Faithfulness is Yes, Confidence reaches accept_confidence, Ambiguity is not above accept_ambiguity, Orientation is Correct, and Contact is Correct; strict_final_judge additionally requires Contact angle Correct and Target residual Close; max_iters is only a safety cap",
                "progressive_update_pressure": "when enabled, evaluator Target residual and remaining iteration budget are used to prompt and numerically scale bounded proposer updates",
                "failed_candidate_commit_policy": "failed selected paths are rejected by default so later operations are not polluted; enable loose ablations with --continue-after-failed-operation",
            },
            "fixed_base_axes": self._fixed_base_axis_metadata(),
            "operations": operations_json,
            "final_pose_by_part": {pid: transform_to_json(T) for pid, T in committed_poses.items()},
            "final_scene_obj": str(final_scene),
            "final_views": [str(path) for path in final_views],
        }

    def _render_initial_assets(
        self,
        renderer: PosedSceneRenderer,
        poses: dict[str, np.ndarray],
        view_specs: list[ViewSpec] | None,
    ) -> None:
        initial_dir = self.output_root / "initial_views"
        renderer.render_scene_views(
            self.initial_parts,
            poses,
            initial_dir,
            "initial_scene",
            self.config.num_views,
            focus_ids=list(self.initial_parts.keys()),
            fixed_axis_part=self._fixed_base_axis_part(),
            show_fixed_base_axes=self.config.render_fixed_base_axes,
            view_specs=view_specs,
        )
        (initial_dir / "view_specs.json").write_text(
            json.dumps(view_specs_to_json(view_specs), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _target_views(
        self,
        renderer: PosedSceneRenderer,
        operation: OperationSpec,
        view_specs: list[ViewSpec] | None,
        op_dir: Path | None = None,
        target_poses: dict[str, np.ndarray] | None = None,
        frame_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[Path]:
        view_key = view_specs_key(view_specs, self.config.num_views)
        frame_key = self._frame_bounds_key(frame_bounds)
        cached = self.target_view_cache.get((operation.operation_index, view_key, frame_key))
        if cached is not None and all(path.exists() for path in cached):
            return cached
        target_subset = self._operation_target_parts(operation)
        poses = target_poses or {pid: np.eye(4, dtype=np.float64) for pid in target_subset}
        suffix = "fixed" if view_specs is None else f"{view_key}_{frame_key}"
        root = op_dir or (self.output_root / f"operation_{operation.operation_index:02d}_part_{operation.moving_part}")
        target_view_dir = root / "target_effect_views" / suffix
        previous_show_ground_plane = getattr(renderer, "show_ground_plane", None)
        if previous_show_ground_plane is not None:
            renderer.show_ground_plane = False
        try:
            paths = renderer.render_scene_views(
                target_subset,
                poses,
                target_view_dir,
                f"target_effect_op_{operation.operation_index:02d}_{suffix}",
                self.config.num_views,
                focus_ids=list(target_subset.keys()),
                view_specs=view_specs,
                frame_bounds=frame_bounds,
            )
        finally:
            if previous_show_ground_plane is not None:
                renderer.show_ground_plane = previous_show_ground_plane
        (target_view_dir / "view_specs.json").write_text(
            json.dumps(view_specs_to_json(view_specs), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.target_view_cache[(operation.operation_index, view_key, frame_key)] = paths
        return paths

    def _aligned_target_poses(self, operation: OperationSpec, reference_poses: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        target_subset = self._operation_target_parts(operation)
        poses = {pid: np.eye(4, dtype=np.float64) for pid in target_subset}
        anchor_id = next((pid for pid in operation.fixed_parts if pid in self.initial_parts and pid in self.target_parts), None)
        if anchor_id is None:
            return poses
        reference_pose = reference_poses.get(anchor_id, np.eye(4, dtype=np.float64))
        current_anchor = self._posed_centroid(self.initial_parts[anchor_id], reference_pose)
        target_anchor = self.target_parts[anchor_id].centroid
        offset = current_anchor - target_anchor
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = offset
        return {pid: T.copy() for pid in target_subset}

    def _shared_operation_frame_bounds(
        self,
        renderer: PosedSceneRenderer,
        operation: OperationSpec,
        current_parts: dict[str, PartMesh],
        current_poses: dict[str, np.ndarray],
        target_parts: dict[str, PartMesh],
        target_poses: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        focus_ids = operation.fixed_parts + [operation.moving_part]
        current_bmin, current_bmax = renderer.compute_scene_bounds(current_parts, current_poses, focus_ids)
        target_bmin, target_bmax = renderer.compute_scene_bounds(target_parts, target_poses, focus_ids)
        return np.minimum(current_bmin, target_bmin), np.maximum(current_bmax, target_bmax)

    @staticmethod
    def _frame_bounds_key(frame_bounds: tuple[np.ndarray, np.ndarray] | None) -> str:
        if frame_bounds is None:
            return "auto"
        arr = np.concatenate([np.asarray(frame_bounds[0], dtype=np.float64), np.asarray(frame_bounds[1], dtype=np.float64)])
        return hashlib.sha1(np.round(arr, 5).tobytes()).hexdigest()[:8]

    def _view_specs_for_observation(
        self,
        renderer: PosedSceneRenderer,
        operation: OperationSpec,
        poses: dict[str, np.ndarray],
    ) -> list[ViewSpec] | None:
        if self.config.fixed_views:
            return None
        focus_ids = operation.fixed_parts + [operation.moving_part]
        return renderer.plan_adaptive_view_specs(
            parts=self.initial_parts,
            poses=poses,
            num_views=self.config.num_views,
            focus_ids=focus_ids,
            moving_part=operation.moving_part,
            fixed_ids=operation.fixed_parts,
        )

    def _sample_candidate_path(
        self,
        renderer: PosedSceneRenderer,
        operation: OperationSpec,
        sample_id: int,
        committed_poses: dict[str, np.ndarray],
        global_memory: list[dict[str, Any]],
        target_views: list[Path],
        op_dir: Path,
        progress: VisualRunLogger,
    ) -> CandidatePath:
        sample_dir = op_dir / f"candidate_{sample_id:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        poses = {pid: T.copy() for pid, T in committed_poses.items()}
        iteration_memory: list[dict[str, Any]] = []
        success = False
        final_evaluator: dict[str, Any] | None = None
        final_views: list[Path] = []
        final_distance: float | None = None
        error: str | None = None
        consumed_progress_units = 0

        try:
            for iteration in range(max(1, self.config.max_iters)):
                progress.set_detail(f"op={operation.operation_index} part={operation.moving_part} sample={sample_id} iter={iteration}")
                view_specs = self._view_specs_for_observation(renderer, operation, poses)
                view_metadata = view_specs_to_json(view_specs)
                if self._uses_current_only_visual_loop():
                    current_parts = self._operation_initial_parts(operation)
                    current_focus_ids = operation.fixed_parts + [operation.moving_part]
                    target_subset = {}
                    target_poses = {}
                    frame_bounds = self._combined_render_bounds([(current_parts, poses, current_focus_ids)])
                    target_views = []
                else:
                    current_parts = self._operation_initial_parts(operation)
                    current_focus_ids = operation.fixed_parts + [operation.moving_part]
                    target_subset = self._operation_target_parts(operation)
                    target_poses = self._aligned_target_poses(operation, poses)
                    frame_bounds = self._shared_operation_frame_bounds(
                        renderer,
                        operation,
                        current_parts,
                        poses,
                        target_subset,
                        target_poses,
                    )
                    target_views = self._target_views(
                        renderer,
                        operation,
                        view_specs,
                        op_dir,
                        target_poses=target_poses,
                        frame_bounds=frame_bounds,
                    )
                progress.log(
                    f"operation {operation.operation_index} candidate {sample_id} iteration {iteration}: "
                    f"render current scene views views={view_spec_names(view_specs)}"
                )
                iter_dir = sample_dir / f"iter_{iteration:02d}"
                current_views = renderer.render_scene_views(
                    current_parts,
                    poses,
                    iter_dir / "current_views",
                    f"op{operation.operation_index}_cand{sample_id}_iter{iteration}",
                    self.config.num_views,
                    focus_ids=current_focus_ids,
                    moving_part=operation.moving_part,
                    show_arrows=False,
                    fixed_axis_part=self._fixed_base_axis_part(),
                    show_fixed_base_axes=False,
                    view_specs=view_specs,
                )
                progress.log(
                    f"operation {operation.operation_index} candidate {sample_id} iteration {iteration}: "
                    f"call evaluator with {len(current_views)} current views and {len(target_views)} target views"
                )
                image_paths = self._image_inputs(operation, current_views, target_views)
                evaluator = self._evaluate(
                    operation,
                    global_memory,
                    iteration_memory,
                    current_views,
                    target_views,
                    image_paths,
                    sample_id,
                    view_metadata,
                )
                final_evaluator = evaluator.to_json()
                accepted = self._accepted_by_evaluator(evaluator)
                progress.log(
                    f"operation {operation.operation_index} candidate {sample_id} iteration {iteration}: "
                    f"evaluator Faithfulness={evaluator.faithfulness} view={evaluator.image_number} "
                    f"confidence={evaluator.confidence:.3f} ambiguity={evaluator.ambiguity:.3f} "
                    f"orientation={evaluator.orientation} contact={evaluator.contact} "
                    f"contact_angle={evaluator.contact_angle} relation={evaluator.required_relation} "
                    f"rotation_penalty={evaluator.rotation_penalty} "
                    f"target_residual={evaluator.target_residual} accepted={accepted}"
                )
                supporting_view = self._save_supporting_view_pair(current_views, target_views, evaluator, iter_dir)
                progress.log(
                    f"operation {operation.operation_index} candidate {sample_id} iteration {iteration}: "
                    f"supporting_view={supporting_view.get('current_copy')}"
                )
                if accepted:
                    success = True
                    final_views = current_views
                    iteration_memory.append(
                        {
                            "iteration": iteration,
                            "adaptive_views": view_metadata,
                            "target_views": self._target_views_record(operation, target_views),
                            "before_views": [str(path) for path in current_views],
                            "after_views": [],
                            "supporting_view": supporting_view,
                            "evaluator": evaluator.to_json(),
                            "accepted_without_update": True,
                        }
                    )
                    consumed_progress_units += 1
                    progress.advance(1, detail=f"op={operation.operation_index} sample={sample_id} accepted")
                    break

                progress.log(
                    f"operation {operation.operation_index} candidate {sample_id} iteration {iteration}: "
                    "call proposer for incremental pose update"
                )
                remaining_iters = max(0, max(1, self.config.max_iters) - iteration - 1)
                update_scale = self._update_scale_for_iteration(evaluator, iteration, iteration_memory)
                progress.log(
                    f"operation {operation.operation_index} candidate {sample_id} iteration {iteration}: "
                    f"target_residual={evaluator.target_residual} update_scale={update_scale:.3f} "
                    f"remaining_iters={remaining_iters}"
                )
                if self.config.proposer_input_mode == "orthographic":
                    proposer_input = self._make_orthographic_proposer_input(
                        operation,
                        current_parts=current_parts,
                        current_poses=poses,
                        target_parts=target_subset,
                        target_poses=target_poses,
                        frame_bounds=frame_bounds,
                        output_dir=iter_dir,
                    )
                elif self.config.proposer_input_mode == "current_only_orthographic":
                    proposer_input = self._make_current_only_orthographic_proposer_input(
                        operation,
                        current_parts=current_parts,
                        current_poses=poses,
                        output_dir=iter_dir,
                    )
                else:
                    proposer_input = Path(supporting_view["current_copy"])
                progress.log(
                    f"operation {operation.operation_index} candidate {sample_id} iteration {iteration}: "
                    f"proposer_input={proposer_input}"
                )
                proposer = self._propose(
                    operation,
                    evaluator,
                    global_memory,
                    iteration_memory,
                    current_views,
                    target_views,
                    image_paths,
                    proposer_input,
                    sample_id,
                    poses,
                    view_metadata,
                    iteration,
                    remaining_iters,
                    update_scale,
                )
                progress.log(
                    f"operation {operation.operation_index} candidate {sample_id} iteration {iteration}: "
                    f"proposer translation={proposer.translation.round(4).tolist()} "
                    f"rotation_mode={self.config.rotation_mode} "
                    f"has_rotation_matrix={proposer.rotation_matrix is not None} "
                    f"rotation_matrix={format_rotation_matrix_for_log(proposer.rotation_matrix)} "
                    f"axis={proposer.axis} angle={proposer.angle:.3f} confidence={proposer.confidence:.3f}"
                )
                scaled_update = self._scaled_pose_update(proposer, evaluator, update_scale, remaining_iters)
                progress.log(
                    f"operation {operation.operation_index} candidate {sample_id} iteration {iteration}: "
                    f"scaled proposer translation={scaled_update.translation.round(4).tolist()} "
                    f"axis={scaled_update.axis} angle_no_scale={scaled_update.angle:.3f} "
                    f"translation_clamped={scaled_update.translation_clamped} "
                    f"rotation_clamped={scaled_update.rotation_clamped}"
                )
                old_pose = poses[operation.moving_part].copy()
                new_pose = self._apply_proposer_update(operation.moving_part, old_pose, scaled_update)
                poses[operation.moving_part] = new_pose
                progress.log(
                    f"operation {operation.operation_index} candidate {sample_id} iteration {iteration}: "
                    f"applied pose update new_translation={new_pose[:3, 3].round(6).tolist()}"
                )
                after_views = renderer.render_scene_views(
                    current_parts,
                    poses,
                    iter_dir / "after_update_views",
                    f"op{operation.operation_index}_cand{sample_id}_after{iteration}",
                    self.config.num_views,
                    focus_ids=current_focus_ids,
                    moving_part=operation.moving_part,
                    show_arrows=False,
                    fixed_axis_part=self._fixed_base_axis_part(),
                    show_fixed_base_axes=False,
                    view_specs=view_specs,
                )
                item = {
                    "iteration": iteration,
                    "adaptive_views": view_metadata,
                    "target_views": self._target_views_record(operation, target_views),
                    "before_views": [str(path) for path in current_views],
                    "after_views": [str(path) for path in after_views],
                    "supporting_view": supporting_view,
                    "proposer_input": str(proposer_input),
                    "proposer_input_mode": self.config.proposer_input_mode,
                    "evaluator": evaluator.to_json(),
                    "proposer": proposer.to_json(),
                    "scaled_proposer": scaled_update.to_json(),
                    "update_pressure": {
                        "enabled": self.config.progressive_update_pressure,
                        "target_residual": evaluator.target_residual,
                        "update_scale": round(float(update_scale), 6),
                        "remaining_iters": remaining_iters,
                        "max_update_scale": self.config.max_update_scale,
                    },
                    "pose_before_update": transform_to_json(old_pose),
                    "pose_after_update": transform_to_json(new_pose),
                }
                if self.client.is_mock:
                    final_distance = self._mock_distance_to_target(operation, poses)
                    item["mock_distance_to_target"] = final_distance
                iteration_memory.append(item)
                final_views = after_views
                consumed_progress_units += 1
                progress.advance(1, detail=f"op={operation.operation_index} sample={sample_id} iter={iteration}")

            if not success and not self.client.is_mock:
                progress.log(
                    f"operation {operation.operation_index} candidate {sample_id}: "
                    "max iterations reached, run final evaluator"
                )
                final_view_specs = self._view_specs_for_observation(renderer, operation, poses)
                final_view_metadata = view_specs_to_json(final_view_specs)
                if self._uses_current_only_visual_loop():
                    final_target_views = []
                    final_parts = self._operation_initial_parts(operation)
                    final_focus_ids = operation.fixed_parts + [operation.moving_part]
                else:
                    final_target_views = self._target_views(renderer, operation, final_view_specs, op_dir)
                    final_parts = self._operation_initial_parts(operation)
                    final_focus_ids = operation.fixed_parts + [operation.moving_part]
                final_views = renderer.render_scene_views(
                    final_parts,
                    poses,
                    sample_dir / "final_candidate_views",
                    f"op{operation.operation_index}_cand{sample_id}_final",
                    self.config.num_views,
                    focus_ids=final_focus_ids,
                    moving_part=operation.moving_part,
                    show_arrows=False,
                    fixed_axis_part=self._fixed_base_axis_part(),
                    show_fixed_base_axes=False,
                    view_specs=final_view_specs,
                )
                image_paths = self._image_inputs(operation, final_views, final_target_views)
                evaluator = self._evaluate(
                    operation,
                    global_memory,
                    iteration_memory,
                    final_views,
                    final_target_views,
                    image_paths,
                    sample_id,
                    final_view_metadata,
                )
                final_evaluator = evaluator.to_json()
                final_evaluator["supporting_view"] = self._save_supporting_view_pair(
                    final_views,
                    final_target_views,
                    evaluator,
                    sample_dir / "final_candidate_views",
                )
                success = self._accepted_by_evaluator(evaluator)
                progress.log(
                    f"operation {operation.operation_index} candidate {sample_id}: "
                    f"final evaluator Faithfulness={evaluator.faithfulness} "
                    f"confidence={evaluator.confidence:.3f} ambiguity={evaluator.ambiguity:.3f} "
                    f"orientation={evaluator.orientation} contact={evaluator.contact} "
                    f"contact_angle={evaluator.contact_angle} relation={evaluator.required_relation} "
                    f"rotation_penalty={evaluator.rotation_penalty} "
                    f"target_residual={evaluator.target_residual} accepted={success}"
                )

            if self.client.is_mock:
                final_distance = self._mock_distance_to_target(operation, poses)
                mock_distance_success = final_distance <= self.config.mock_accept_distance
                if self.config.strict_final_judge:
                    if final_evaluator is None:
                        final_evaluator = {"raw": {}}
                    raw = final_evaluator.setdefault("raw", {})
                    if isinstance(raw, dict):
                        raw.update(
                            {
                                "mock": True,
                                "mock_distance_to_target": final_distance,
                                "mock_distance_success": mock_distance_success,
                                "strict_mock_acceptance": "kept_evaluator_gate",
                            }
                        )
                    final_evaluator["mock_distance_to_target"] = final_distance
                    final_evaluator["mock_distance_success"] = mock_distance_success
                else:
                    success = mock_distance_success
                    final_evaluator = {
                        "Faithfulness": "Yes" if success else "No",
                        "Image number": 1,
                        "Reasoning": (
                            f"mock final distance to fixed subassembly center = {final_distance:.6f}"
                            if self._uses_current_only_visual_loop()
                            else f"mock final distance to aligned target center = {final_distance:.6f}"
                        ),
                        "Confidence": 0.8 if success else 0.3,
                        "Ambiguity": 0.1 if success else 0.8,
                        "Orientation": "Correct" if success else "Incorrect",
                        "Contact": "Correct" if success else "Unclear",
                        "Contact angle": "Correct" if success else "Unclear",
                        "Required relation": "Unknown",
                        "Rotation penalty": "None",
                        "Target residual": "Close" if success else "Far",
                        "raw": {"mock": True},
                    }
        except Exception as exc:
            error = exception_summary(exc)
            progress.log(
                f"operation {operation.operation_index} candidate {sample_id}: "
                f"error={error}"
            )
        finally:
            remaining_units = max(1, self.config.max_iters) - consumed_progress_units
            if remaining_units > 0:
                progress.advance(
                    remaining_units,
                    detail=f"op={operation.operation_index} sample={sample_id} skipped={remaining_units}",
                )

        return CandidatePath(
            sample_id=sample_id,
            moving_part=operation.moving_part,
            source="mock" if self.client.is_mock else "vlm",
            valid=error is None,
            success=success,
            final_pose=poses[operation.moving_part].copy(),
            final_distance_to_mock_target=final_distance,
            iterations=iteration_memory,
            final_evaluator=final_evaluator,
            final_views=[str(path) for path in final_views],
            error=error,
        )

    def _operation_part_ids(self, operation: OperationSpec) -> list[str]:
        part_ids: list[str] = []
        for pid in [*operation.fixed_parts, operation.moving_part]:
            if pid not in part_ids:
                part_ids.append(pid)
        return part_ids

    def _uses_current_only_visual_loop(self) -> bool:
        return self.config.proposer_input_mode == "current_only_orthographic"

    def _fixed_base_axis_part(self) -> str | None:
        for operation in self.inputs.operations:
            if operation.fixed_parts:
                return operation.fixed_parts[0]
        return None

    def _fixed_base_axis_metadata(self) -> dict[str, Any]:
        part_id = self._fixed_base_axis_part()
        disabled_reason = None
        if not self.config.render_fixed_base_axes:
            disabled_reason = "render_fixed_base_axes is disabled"
        elif part_id is None:
            disabled_reason = "no fixed_parts entry found in operation specs"
        elif part_id not in self.initial_parts:
            disabled_reason = f"fixed base axis part {part_id!r} is missing from initial parts"
        return {
            "requested": bool(self.config.render_fixed_base_axes),
            "enabled": bool(self.config.render_fixed_base_axes and part_id is not None and part_id in self.initial_parts),
            "part_id": part_id,
            "part_name": self.inputs.part_names.get(part_id, "unknown") if part_id is not None else None,
            "colors": {"+x": "cyan", "+y": "magenta", "+z": "yellow"},
            "disabled_reason": disabled_reason,
        }

    def _operation_initial_parts(self, operation: OperationSpec) -> dict[str, PartMesh]:
        return {pid: self.initial_parts[pid] for pid in self._operation_part_ids(operation) if pid in self.initial_parts}

    def _operation_target_parts(self, operation: OperationSpec) -> dict[str, PartMesh]:
        return {pid: self.target_parts[pid] for pid in self._operation_part_ids(operation) if pid in self.target_parts}

    def _target_views_record(self, operation: OperationSpec, target_views: list[Path]) -> dict[str, Any]:
        if not target_views:
            return {
                "runtime_only": False,
                "saved_under_output_dir": False,
                "ground_plane": False,
                "camera_framing": None,
                "directory": None,
                "count": 0,
                "paths": [],
                "operation_part_ids": self._operation_part_ids(operation),
                "policy": "current-only visual loop: no target-effect/GT render is used for evaluator or proposer inputs",
            }
        return {
            "runtime_only": False,
            "saved_under_output_dir": True,
            "ground_plane": False,
            "camera_framing": "per-view adaptive distance fits all target-effect operation parts",
            "directory": str(target_views[0].parent) if target_views else None,
            "count": len(target_views),
            "paths": [str(path) for path in target_views],
            "operation_part_ids": self._operation_part_ids(operation),
        }

    def _save_supporting_view_pair(
        self,
        current_views: list[Path],
        target_views: list[Path],
        evaluator: EvaluatorResult,
        output_dir: Path,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not current_views:
            record = {
                "image_number": int(evaluator.image_number),
                "selected_current_index": None,
                "current_source": None,
                "current_copy": None,
                "paired_target_source": None,
                "paired_target_copy": None,
            }
            (output_dir / "supporting_view.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return record

        selected_index = min(max(int(evaluator.image_number), 1), len(current_views)) - 1
        current_source = Path(current_views[selected_index])
        current_copy = output_dir / "supporting_view.png"
        if current_source.resolve() != current_copy.resolve():
            shutil.copy2(current_source, current_copy)

        target_source = None
        target_copy = None
        if target_views:
            target_index = min(selected_index, len(target_views) - 1)
            target_source = Path(target_views[target_index])
            target_copy = output_dir / "supporting_target_effect_view.png"
            if target_source.resolve() != target_copy.resolve():
                shutil.copy2(target_source, target_copy)

        record = {
            "image_number": int(evaluator.image_number),
            "selected_current_index": selected_index,
            "current_source": str(current_source),
            "current_copy": str(current_copy),
            "paired_target_source": str(target_source) if target_source is not None else None,
            "paired_target_copy": str(target_copy) if target_copy is not None else None,
        }
        (output_dir / "supporting_view.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record

    def _make_orthographic_proposer_input(
        self,
        operation: OperationSpec,
        current_parts: dict[str, PartMesh],
        current_poses: dict[str, np.ndarray],
        target_parts: dict[str, PartMesh],
        target_poses: dict[str, np.ndarray],
        frame_bounds: tuple[np.ndarray, np.ndarray],
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "proposer_orthographic_measurement.png"
        target_render_parts, target_render_poses, target_reference_record = self._orthographic_target_reference_scene(
            operation=operation,
            current_parts=current_parts,
            current_poses=current_poses,
            target_parts=target_parts,
        )
        frame_bounds = self._combined_render_bounds(
            [
                (current_parts, current_poses, operation.fixed_parts + [operation.moving_part]),
                (target_render_parts, target_render_poses, operation.fixed_parts + [operation.moving_part]),
            ]
        )
        render_orthographic_measurement_input(
            current_parts=current_parts,
            current_poses=current_poses,
            target_parts=target_render_parts,
            target_poses=target_render_poses,
            output_path=path,
            moving_part=operation.moving_part,
            fixed_parts=operation.fixed_parts,
            frame_bounds=frame_bounds,
        )
        record = {
            "image": str(path),
            "type": "orthographic_current_target_measurement",
            "views": [
                {
                    "name": "front",
                    "eye": "+Z",
                    "horizontal_axis": "+X",
                    "vertical_axis": "+Y",
                    "translation_components_visible": ["Tx", "Ty"],
                },
                {
                    "name": "right",
                    "eye": "+X",
                    "horizontal_axis": "+Z",
                    "vertical_axis": "+Y",
                    "translation_components_visible": ["Tz", "Ty"],
                },
                {
                    "name": "top",
                    "eye": "+Y",
                    "horizontal_axis": "+X",
                    "vertical_axis": "+Z",
                    "translation_components_visible": ["Tx", "Tz"],
                },
            ],
            "frame_bounds_xyz": [
                np.asarray(frame_bounds[0], dtype=np.float64).round(6).tolist(),
                np.asarray(frame_bounds[1], dtype=np.float64).round(6).tolist(),
            ],
            "translation_unit_mode": self.config.translation_unit_mode,
            "translation_unit_scene_units": 1.0 if self.config.translation_unit_mode == "scene_units" else None,
            "moving_part": operation.moving_part,
            "fixed_parts": operation.fixed_parts,
            "target_reference": target_reference_record,
        }
        (output_dir / "proposer_orthographic_measurement.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def _make_current_only_orthographic_proposer_input(
        self,
        operation: OperationSpec,
        current_parts: dict[str, PartMesh],
        current_poses: dict[str, np.ndarray],
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "proposer_current_orthographic.png"
        focus_ids = list(current_parts.keys())
        frame_bounds = self._combined_render_bounds([(current_parts, current_poses, focus_ids)])
        render_current_orthographic_input(
            current_parts=current_parts,
            current_poses=current_poses,
            output_path=path,
            moving_part=operation.moving_part,
            fixed_parts=operation.fixed_parts,
            frame_bounds=frame_bounds,
            visible_part_ids=focus_ids,
        )
        record = {
            "image": str(path),
            "type": "current_only_orthographic_measurement",
            "views": [
                {
                    "name": "front",
                    "eye": "+Z",
                    "horizontal_axis": "+X",
                    "vertical_axis": "+Y",
                    "translation_components_visible": ["Tx", "Ty"],
                },
                {
                    "name": "right",
                    "eye": "+X",
                    "horizontal_axis": "+Z",
                    "vertical_axis": "+Y",
                    "translation_components_visible": ["Tz", "Ty"],
                },
                {
                    "name": "top",
                    "eye": "+Y",
                    "horizontal_axis": "+X",
                    "vertical_axis": "+Z",
                    "translation_components_visible": ["Tx", "Tz"],
                },
            ],
            "frame_bounds_xyz": [
                np.asarray(frame_bounds[0], dtype=np.float64).round(6).tolist(),
                np.asarray(frame_bounds[1], dtype=np.float64).round(6).tolist(),
            ],
            "translation_unit_mode": self.config.translation_unit_mode,
            "translation_unit_scene_units": 1.0 if self.config.translation_unit_mode == "scene_units" else None,
            "moving_part": operation.moving_part,
            "fixed_parts": operation.fixed_parts,
            "visible_parts": focus_ids,
            "target_reference": None,
            "target_reference_policy": "not rendered for proposer; infer the target relation from instruction, manual/tree text, evaluator feedback, and memory",
        }
        (output_dir / "proposer_current_orthographic.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def _orthographic_target_reference_scene(
        self,
        operation: OperationSpec,
        current_parts: dict[str, PartMesh],
        current_poses: dict[str, np.ndarray],
        target_parts: dict[str, PartMesh],
    ) -> tuple[dict[str, PartMesh], dict[str, np.ndarray], dict[str, Any]]:
        render_parts: dict[str, PartMesh] = {}
        render_poses: dict[str, np.ndarray] = {}
        for part_id in operation.fixed_parts:
            if part_id in current_parts:
                render_parts[part_id] = current_parts[part_id]
                render_poses[part_id] = validate_transform(current_poses.get(part_id, np.eye(4, dtype=np.float64)))

        anchor_id = self._target_anchor_part(operation)
        target_to_current = None
        alignment_record: dict[str, Any] = {
            "mode": "fixed_current_plus_gt_moving",
            "fixed_parts_source": "current scene poses; fixed parts do not move in target panels",
            "moving_part_source": "target/GT part transformed into the current fixed-anchor frame",
            "anchor_part": anchor_id,
        }
        if anchor_id is not None and anchor_id in current_parts and anchor_id in self.target_parts:
            target_to_current, sign_choice = self._target_to_current_anchor_transform(
                anchor_id,
                current_poses,
                operation.moving_part,
            )
            alignment_record["target_to_current_anchor_matrix4"] = target_to_current.round(6).tolist()
            alignment_record["axis_sign_choice_xyz"] = list(sign_choice)

        moving_id = operation.moving_part
        if moving_id in target_parts and target_to_current is not None:
            render_parts[moving_id] = target_parts[moving_id]
            render_poses[moving_id] = target_to_current
        elif moving_id in target_parts:
            render_parts[moving_id] = target_parts[moving_id]
            render_poses[moving_id] = validate_transform(np.eye(4, dtype=np.float64))
            alignment_record["fallback"] = "no fixed anchor alignment available; target moving part uses target frame"
        elif moving_id in current_parts:
            render_parts[moving_id] = current_parts[moving_id]
            render_poses[moving_id] = validate_transform(current_poses.get(moving_id, np.eye(4, dtype=np.float64)))
            alignment_record["fallback"] = "target moving part missing; current moving part reused"

        return render_parts, render_poses, alignment_record

    def _target_anchor_part(self, operation: OperationSpec) -> str | None:
        # Operation.fixed_parts is derived from the assembly tree/order: the
        # first available fixed part is the anchor the new moving part attaches to.
        return next((pid for pid in operation.fixed_parts if pid in self.initial_parts and pid in self.target_parts), None)

    def _target_to_current_anchor_transform(
        self,
        anchor_id: str,
        current_poses: dict[str, np.ndarray],
        moving_part: str,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        current_part = self.initial_parts[anchor_id]
        target_part = self.target_parts[anchor_id]
        current_points = transform_points(
            current_part.vertices,
            validate_transform(current_poses.get(anchor_id, np.eye(4, dtype=np.float64))),
        )
        target_points = np.asarray(target_part.vertices, dtype=np.float64)
        current_center, current_axes = self._pca_frame(current_points)
        target_center, target_axes = self._pca_frame(target_points)
        current_moving_center = None
        target_moving_center = None
        if moving_part in self.initial_parts and moving_part in self.target_parts:
            current_moving_center = transform_points(
                self.initial_parts[moving_part].centroid.reshape(1, 3),
                validate_transform(current_poses.get(moving_part, np.eye(4, dtype=np.float64))),
            )[0]
            target_moving_center = np.asarray(self.target_parts[moving_part].centroid, dtype=np.float64)

        best_T: np.ndarray | None = None
        best_signs = (1, 1, 1)
        best_score = float("inf")
        for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            sign_matrix = np.diag(np.asarray(signs, dtype=np.float64))
            rotation = current_axes @ sign_matrix @ target_axes.T
            if np.linalg.det(rotation) < 0:
                continue
            translation = current_center - rotation @ target_center
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rotation
            T[:3, 3] = translation
            if current_moving_center is not None and target_moving_center is not None:
                mapped_moving = rotation @ target_moving_center + translation
                score = float(np.linalg.norm(mapped_moving - current_moving_center))
            else:
                mapped_anchor = transform_points(target_points, T)
                score = float(np.linalg.norm(mapped_anchor.mean(axis=0) - current_center))
            if score < best_score:
                best_score = score
                best_T = T
                best_signs = tuple(int(x) for x in signs)
        if best_T is None:
            best_T = np.eye(4, dtype=np.float64)
            best_T[:3, 3] = current_center - target_center
        return validate_transform(best_T), best_signs

    @staticmethod
    def _pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=np.float64)
        center = pts.mean(axis=0)
        centered = pts - center
        cov = centered.T @ centered / max(len(centered), 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        axes = eigvecs[:, order]
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1.0
        return center, axes

    @staticmethod
    def _combined_render_bounds(
        render_sets: list[tuple[dict[str, PartMesh], dict[str, np.ndarray], list[str]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        points = []
        for parts, poses, ids in render_sets:
            for part_id in ids:
                if part_id not in parts:
                    continue
                points.append(transform_points(parts[part_id].vertices, poses.get(part_id, np.eye(4))))
        if not points:
            return np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)
        return bounds_for_points(np.concatenate(points, axis=0))

    def _translation_scale_reference(self, operation: OperationSpec) -> dict[str, Any] | None:
        part = self.initial_parts.get(operation.moving_part)
        if part is None:
            return None
        if self.config.translation_unit_mode == "scene_units":
            extents = np.asarray(part.extents, dtype=np.float64)
            return {
                "moving_part": operation.moving_part,
                "moving_part_name": self.inputs.part_names.get(operation.moving_part, "unknown"),
                "translation_unit_mode": "scene_units",
                "translation_unit_scene_units": 1.0,
                "moving_part_aabb_extents_scene_units_xyz": extents.round(6).tolist(),
                "translation_unit_fraction_of_aabb_xyz": (1.0 / np.maximum(extents, 1e-8)).round(4).tolist(),
                "source": "orthographic measurement views use world scene units directly",
            }
        unit = float(translation_scale_from_bounds(part.bbox_min, part.bbox_max))
        extents = np.asarray(part.extents, dtype=np.float64)
        ratios = unit / np.maximum(extents, 1e-8)
        return {
            "moving_part": operation.moving_part,
            "moving_part_name": self.inputs.part_names.get(operation.moving_part, "unknown"),
            "translation_unit_mode": "vlm_pose",
            "translation_unit_scene_units": unit,
            "moving_part_aabb_extents_scene_units_xyz": extents.round(6).tolist(),
            "translation_unit_fraction_of_aabb_xyz": ratios.round(4).tolist(),
            "source": "VLM-Pose AABB arrow_len formula: min_extent if 2*min_extent <= max_extent else min_extent/2",
        }

    def _image_inputs(self, operation: OperationSpec, current_views: list[Path], target_views: list[Path]) -> list[Path]:
        if self._uses_current_only_visual_loop():
            return list(current_views) + [self.inputs.manual_step_numbered[operation.manual_step_index]]
        return (
            list(current_views)
            + list(target_views)
            + [
                self.inputs.product_page,
                self.inputs.manual_step_overlay[operation.manual_step_index],
                self.inputs.manual_step_numbered[operation.manual_step_index],
                self.inputs.scene_rot_annotated,
            ]
        )

    def _make_proposer_panel(
        self,
        operation: OperationSpec,
        current_views: list[Path],
        target_views: list[Path],
        evaluator: EvaluatorResult,
        output_dir: Path,
        view_metadata: list[dict[str, Any]] | None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        selected_index = min(max(int(evaluator.image_number), 1), max(1, len(current_views))) - 1
        current_path = Path(current_views[selected_index])
        current_img = Image.open(current_path).convert("RGB")
        crop_box = [0, 0, current_img.width, current_img.height]
        view_name = f"view_{selected_index + 1:02d}"
        view_reason = None
        if view_metadata and selected_index < len(view_metadata):
            view_name = str(view_metadata[selected_index].get("name") or view_name)
            view_reason = view_metadata[selected_index].get("reason")
        moving_part = self.initial_parts.get(operation.moving_part)
        translation_unit = (
            float(translation_scale_from_bounds(moving_part.bbox_min, moving_part.bbox_max))
            if moving_part is not None
            else None
        )
        part_extents = moving_part.extents.astype(float).tolist() if moving_part is not None else None

        panel_width = 1800
        margin = 24
        header_h = 190
        label_h = 34
        footer_h = 120
        compass_gap = 24
        compass_w = 520
        image_w = panel_width - margin * 2 - compass_gap - compass_w
        image_h = 980
        panel_h = header_h + label_h + image_h + footer_h + margin

        panel = Image.new("RGB", (panel_width, panel_h), (246, 247, 249))
        draw = ImageDraw.Draw(panel)
        font_title = _panel_font(28, bold=True)
        font = _panel_font(20)
        font_small = _panel_font(16)

        moving_name = self.inputs.part_names.get(operation.moving_part, "unknown")
        fixed_names = ", ".join(f"{pid} {self.inputs.part_names.get(pid, 'unknown')}" for pid in operation.fixed_parts)
        header_lines = [
            f"Operation {operation.operation_index}: move part {operation.moving_part} ({moving_name})",
            f"Fixed: {fixed_names or 'none'}",
            f"Proposer input: evaluator-selected supporting view {selected_index + 1} ({view_name}).",
            "Left image shows object positions only; no coordinate arrows are drawn on the parts.",
            "Right compass defines the fixed world update frame: red=+X, green=+Y, blue=+Z.",
            (
                f"Scale: 1 translation unit"
                + (f" ({translation_unit:.3f} scene units)" if translation_unit is not None else "")
                + "; compare it with the moving part AABB size on the compass card."
            ),
        ]
        y = 18
        for idx, line in enumerate(header_lines):
            draw.text((margin, y), line, fill=(20, 24, 32), font=font_title if idx == 0 else font)
            y += 36 if idx == 0 else 28

        y0 = header_h
        self._paste_labeled_image(
            panel,
            draw,
            current_img,
            (margin, y0 + label_h),
            (image_w, image_h),
            f"SELECTED SUPPORTING VIEW {selected_index + 1}: {view_name}",
            font,
        )
        self._draw_axis_legend(
            draw,
            (margin + image_w + compass_gap, y0 + label_h),
            translation_unit,
            part_extents,
            _panel_font(30, bold=True),
            _panel_font(22),
        )

        instruction = " ".join(operation.instruction.split())
        instruction = instruction[:500] + ("..." if len(instruction) > 500 else "")
        footer_y = y0 + label_h + image_h + 12
        draw.rectangle([margin, footer_y, panel_width - margin, footer_y + footer_h - 16], fill=(255, 255, 255), outline=(203, 213, 225))
        footer_lines = [
            f"Instruction excerpt: {instruction}",
            "Use the right command compass as the update frame. The left image is full-resolution context for object position, contact, height, and orientation.",
        ]
        fy = footer_y + 8
        for line in footer_lines:
            draw.text((margin + 12, fy), line, fill=(55, 65, 81), font=font_small)
            fy += 24

        path = output_dir / "proposer_input_panel.png"
        panel.save(path)
        record = {
            "panel": str(path),
            "selected_supporting_view": {
                "image_number": selected_index + 1,
                "view_name": view_name,
                "view_reason": view_reason,
                "current_view": str(current_path),
                "proposer_focus_crop_xyxy": crop_box,
            },
            "selected_image_number": int(evaluator.image_number),
            "axis_convention": {
                "red": "+X fixed world axis",
                "green": "+Y fixed world axis",
                "blue": "+Z fixed world axis",
                "origin": "moving part center for update application; compass card is a fixed visual reference",
                "translation_unit": "one normalized translation unit derived from the moving part AABB, matching VLM-Pose arrow length",
                "translation_unit_scene_units": translation_unit,
                "moving_part_aabb_extents_scene_units_xyz": part_extents,
                "translation_unit_relative_to_aabb_xyz": (
                    (float(translation_unit) / np.maximum(np.asarray(part_extents, dtype=np.float64), 1e-8)).round(4).tolist()
                    if part_extents is not None and translation_unit is not None
                    else None
                ),
            },
            "moving_part": operation.moving_part,
            "fixed_parts": operation.fixed_parts,
            "instruction": operation.instruction,
        }
        (output_dir / "proposer_input_panel.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _draw_axis_legend(
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int],
        translation_unit: float | None,
        part_extents: list[float] | None,
        font_bold: ImageFont.ImageFont,
        font: ImageFont.ImageFont,
    ) -> None:
        x, y = xy
        w, h = 520, 980
        draw.rounded_rectangle([x, y, x + w, y + h], radius=6, fill=(255, 255, 255), outline=(148, 163, 184), width=2)
        draw.text((x + 28, y + 30), "Command Compass", fill=(20, 24, 32), font=font_bold)
        cx, cy = x + 245, y + 455
        arrows = [
            ((220, 38, 38), (cx, cy), (cx + 165, cy), "+X", (cx + 184, cy - 18)),
            ((22, 163, 74), (cx, cy), (cx, cy - 165), "+Y", (cx - 28, cy - 220)),
            ((37, 99, 235), (cx, cy), (cx - 122, cy + 122), "+Z", (cx - 185, cy + 132)),
        ]
        for color, start, end, label, label_xy in arrows:
            draw.line([start, end], fill=color, width=12)
            angle = float(np.arctan2(end[1] - start[1], end[0] - start[0]))
            head_len = 28.0
            p1 = (
                end[0] - head_len * np.cos(angle - 0.45),
                end[1] - head_len * np.sin(angle - 0.45),
            )
            p2 = (
                end[0] - head_len * np.cos(angle + 0.45),
                end[1] - head_len * np.sin(angle + 0.45),
            )
            draw.polygon([end, p1, p2], fill=color)
            draw.text(label_xy, label, fill=color, font=font_bold)
        draw.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(15, 23, 42))
        unit_text = "1 translation unit"
        if translation_unit is not None:
            unit_text += f" = {translation_unit:.3f} scene units"
        draw.text((x + 34, y + 715), unit_text, fill=(55, 65, 81), font=font)
        text_y = y + 755
        if part_extents is not None:
            ext = np.asarray(part_extents, dtype=np.float64)
            draw.text((x + 34, text_y), f"Moving AABB X/Y/Z:", fill=(55, 65, 81), font=font)
            draw.text((x + 34, text_y + 34), f"{ext[0]:.3f} / {ext[1]:.3f} / {ext[2]:.3f}", fill=(55, 65, 81), font=font)
            if translation_unit is not None:
                ratios = float(translation_unit) / np.maximum(ext, 1e-8)
                draw.text((x + 34, text_y + 74), f"1 unit = {ratios[0]:.2f}X, {ratios[1]:.2f}Y,", fill=(55, 65, 81), font=font)
                draw.text((x + 34, text_y + 108), f"{ratios[2]:.2f}Z of this AABB.", fill=(55, 65, 81), font=font)
            text_y += 158
        draw.text((x + 34, text_y), "Compass shows direction only,", fill=(55, 65, 81), font=font)
        draw.text((x + 34, text_y + 34), "not a pixel ruler.", fill=(55, 65, 81), font=font)

    @staticmethod
    def _paste_labeled_image(
        panel: Image.Image,
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
        xy: tuple[int, int],
        size: tuple[int, int],
        label: str,
        font: ImageFont.ImageFont,
    ) -> None:
        x, y = xy
        w, h = size
        label_y = y - 34
        draw.rectangle([x, label_y, x + w, y - 4], fill=(17, 24, 39))
        draw.text((x + 10, label_y + 5), label, fill=(255, 255, 255), font=font)
        fitted = _fit_image(image, size)
        panel.paste(fitted, xy)
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(31, 41, 55), width=2)

    def _evaluate(
        self,
        operation: OperationSpec,
        global_memory: list[dict[str, Any]],
        iteration_memory: list[dict[str, Any]],
        current_views: list[Path],
        target_views: list[Path],
        image_paths: list[Path],
        sample_id: int,
        view_metadata: list[dict[str, Any]] | None,
    ) -> EvaluatorResult:
        if self.client.is_mock:
            return MockVisualVLMClient().evaluate(SYSTEM_PROMPT, "", image_paths, 0.0, sample_id)
        prompt = build_evaluator_prompt(
            self.inputs,
            operation,
            global_memory,
            iteration_memory,
            num_current_views=len(current_views),
            num_target_views=len(target_views),
            view_specs=view_metadata,
            fixed_base_axis_part=self._fixed_base_axis_part() if self.config.render_fixed_base_axes else None,
        )
        return self.client.evaluate(SYSTEM_PROMPT, prompt, image_paths, self._sample_temperature(sample_id), sample_id)

    def _propose(
        self,
        operation: OperationSpec,
        evaluator: EvaluatorResult,
        global_memory: list[dict[str, Any]],
        iteration_memory: list[dict[str, Any]],
        current_views: list[Path],
        target_views: list[Path],
        image_paths: list[Path],
        proposer_input: Path,
        sample_id: int,
        poses: dict[str, np.ndarray],
        view_metadata: list[dict[str, Any]] | None,
        iteration_index: int,
        remaining_iters: int,
        update_scale: float,
    ) -> ProposerResult:
        if self.client.is_mock:
            return self._mock_proposer(operation, sample_id, poses)
        prompt = build_proposer_prompt(
            self.inputs,
            operation,
            evaluator.reasoning,
            global_memory,
            iteration_memory,
            view_specs=view_metadata,
            evaluator_feedback=evaluator.to_json(),
            rotation_mode=self.config.rotation_mode,
            iteration_index=iteration_index,
            max_iters=self.config.max_iters,
            remaining_iters=remaining_iters,
            target_residual=evaluator.target_residual,
            update_scale=update_scale,
            progressive_update_pressure=self.config.progressive_update_pressure,
            fixed_base_axis_part=self._fixed_base_axis_part() if self.config.render_fixed_base_axes else None,
            translation_scale_reference=self._translation_scale_reference(operation),
            proposer_input_mode=self.config.proposer_input_mode,
        )
        proposer_images = [proposer_input]
        if self.config.proposer_input_mode == "current_only_orthographic":
            proposer_images.append(self.inputs.manual_step_numbered[operation.manual_step_index])
        return self.client.propose(SYSTEM_PROMPT, prompt, proposer_images, self._sample_temperature(sample_id), sample_id)

    def _update_scale_for_iteration(
        self,
        evaluator: EvaluatorResult,
        iteration: int,
        iteration_memory: list[dict[str, Any]],
    ) -> float:
        if not self.config.progressive_update_pressure:
            return 1.0
        residual = evaluator.target_residual
        max_iters = max(1, int(self.config.max_iters))
        progress = min(1.0, max(0.0, float(iteration + 1) / float(max_iters)))
        streak = self._far_or_unclear_streak(residual, iteration_memory)
        if residual == "Far":
            scale = 1.55 + 0.65 * progress + 0.18 * max(0, streak - 1)
        elif residual == "Medium":
            scale = 1.20 + 0.35 * progress + 0.08 * max(0, streak - 1)
        elif residual == "Unclear":
            scale = 1.30 + 0.45 * progress + 0.12 * max(0, streak - 1)
        else:
            scale = 1.0
        return float(np.clip(scale, 1.0, self.config.max_update_scale))

    def _far_or_unclear_streak(self, current_residual: str, iteration_memory: list[dict[str, Any]]) -> int:
        if current_residual not in {"Far", "Unclear"}:
            return 0
        streak = 1
        for item in reversed(iteration_memory):
            residual = (item.get("evaluator") or {}).get("Target residual")
            if residual in {"Far", "Unclear"}:
                streak += 1
            else:
                break
        return streak

    def _scaled_pose_update(
        self,
        proposer: ProposerResult,
        evaluator: EvaluatorResult,
        update_scale: float,
        remaining_iters: int,
    ) -> ScaledPoseUpdate:
        scale = float(update_scale) if self.config.progressive_update_pressure else 1.0
        scaled_translation = np.asarray(proposer.translation, dtype=np.float64).reshape(3) * scale
        max_units = float(self.config.max_translation_units)
        clipped_translation = np.clip(scaled_translation, -max_units, max_units)
        translation_clamped = not np.allclose(scaled_translation, clipped_translation)

        rotation_scale = 1.0
        raw_angle = float(proposer.angle)
        clipped_angle = float(np.clip(raw_angle, -self.config.max_rotation_deg, self.config.max_rotation_deg))
        rotation_clamped = abs(clipped_angle - raw_angle) > 1e-8
        rotation_matrix = proposer.rotation_matrix
        matrix_angle_before: float | None = None
        matrix_angle_after: float | None = None

        if self.config.rotation_mode == "matrix" and proposer.rotation_matrix is not None:
            rotation_matrix, matrix_angle_before, matrix_angle_after, matrix_clamped = scale_rotation_matrix(
                proposer.rotation_matrix,
                rotation_scale,
                self.config.max_rotation_deg,
            )
            rotation_clamped = rotation_clamped or matrix_clamped

        return ScaledPoseUpdate(
            translation=clipped_translation,
            axis=proposer.axis,
            angle=clipped_angle,
            rotation_matrix=rotation_matrix,
            update_scale=scale,
            residual=evaluator.target_residual,
            remaining_iters=remaining_iters,
            translation_clamped=translation_clamped,
            rotation_clamped=rotation_clamped,
            matrix_angle_before=matrix_angle_before,
            matrix_angle_after=matrix_angle_after,
        )

    def _apply_proposer_update(self, part_id: str, old_pose: np.ndarray, update: ScaledPoseUpdate) -> np.ndarray:
        part = self.initial_parts[part_id]
        arrow_len = 1.0 if self.config.translation_unit_mode == "scene_units" else translation_scale_from_bounds(
            part.bbox_min,
            part.bbox_max,
        )
        return self.simulation.apply_pose_update(
            part=part,
            current_pose=old_pose,
            local_bounds=(part.bbox_min, part.bbox_max),
            translation_units=update.translation,
            arrow_len=arrow_len,
            axis=update.axis,
            angle_deg=update.angle,
            rotation_matrix=update.rotation_matrix if self.config.rotation_mode == "matrix" else None,
        )

    def _accepted_by_evaluator(self, evaluator: EvaluatorResult) -> bool:
        base_accept = (
            evaluator.is_yes
            and evaluator.confidence >= float(self.config.accept_confidence)
            and evaluator.ambiguity <= float(self.config.accept_ambiguity)
            and evaluator.orientation == "Correct"
            and evaluator.contact == "Correct"
        )
        if not base_accept:
            return False
        if not self.config.strict_final_judge:
            return True
        return evaluator.contact_angle == "Correct" and evaluator.target_residual == "Close"

    def _mock_proposer(self, operation: OperationSpec, sample_id: int, poses: dict[str, np.ndarray]) -> ProposerResult:
        part = self.initial_parts[operation.moving_part]
        current_center = self._posed_centroid(part, poses[operation.moving_part])
        target_center = (
            self._mock_current_only_relation_center(operation, poses)
            if self._uses_current_only_visual_loop()
            else self._mock_target_center(operation)
        )
        arrow_len = 1.0 if self.config.translation_unit_mode == "scene_units" else translation_scale_from_bounds(
            part.bbox_min,
            part.bbox_max,
        )
        step_delta = (target_center - current_center) * 0.55
        # Tiny sample-dependent offsets keep the voting code exercised without changing the intended cluster.
        sample_jitter = (sample_id - (self.config.path_samples - 1) * 0.5) * 0.01 * arrow_len
        step_delta = step_delta + np.array([sample_jitter, 0.0, -sample_jitter], dtype=np.float64)
        translation_units = step_delta / max(float(arrow_len), 1e-8)
        return ProposerResult(
            target_object=self.inputs.part_names.get(operation.moving_part, "moving part"),
            instruction_target_relation=operation.instruction[:240],
            current_error=(
                "mock proposer: current moving-part center is offset from fixed subassembly center"
                if self._uses_current_only_visual_loop()
                else "mock proposer: current moving-part center is offset from aligned target-effect center"
            ),
            reasoning=(
                "mock proposer: non-GT dry-run step toward fixed subassembly center"
                if self._uses_current_only_visual_loop()
                else "mock proposer: deterministic geometric step toward aligned target-effect center"
            ),
            translation=translation_units,
            axis="z",
            angle=0.0,
            rotation_matrix=None,
            confidence=0.8,
            raw={"mock": True},
        )

    def _mock_distance_to_target(self, operation: OperationSpec, poses: dict[str, np.ndarray]) -> float:
        current_center = self._posed_centroid(self.initial_parts[operation.moving_part], poses[operation.moving_part])
        target_center = (
            self._mock_current_only_relation_center(operation, poses)
            if self._uses_current_only_visual_loop()
            else self._mock_target_center(operation)
        )
        scale = max(float(np.linalg.norm(self.initial_parts[operation.moving_part].extents)), 1e-6)
        return float(np.linalg.norm(current_center - target_center) / scale)

    def _mock_current_only_relation_center(self, operation: OperationSpec, poses: dict[str, np.ndarray]) -> np.ndarray:
        centers = [
            self._posed_centroid(self.initial_parts[pid], poses.get(pid, np.eye(4, dtype=np.float64)))
            for pid in operation.fixed_parts
            if pid in self.initial_parts
        ]
        if not centers:
            return self._posed_centroid(self.initial_parts[operation.moving_part], poses[operation.moving_part])
        return np.mean(np.stack(centers, axis=0), axis=0)

    def _mock_target_center(self, operation: OperationSpec) -> np.ndarray:
        if not self.target_parts:
            return self._mock_current_only_relation_center(operation, {pid: np.eye(4, dtype=np.float64) for pid in self.initial_parts})
        anchor_id = "1"
        initial_anchor = self._posed_centroid(self.initial_parts[anchor_id], np.eye(4))
        target_anchor = self.target_parts[anchor_id].centroid
        offset = initial_anchor - target_anchor
        return self.target_parts[operation.moving_part].centroid + offset

    def _posed_centroid(self, part: PartMesh, pose: np.ndarray) -> np.ndarray:
        pose = validate_transform(pose)
        return pose[:3, :3] @ part.centroid + pose[:3, 3]

    def _sample_temperature(self, sample_id: int) -> float:
        if self.config.path_samples <= 1:
            return max(0.0, self.config.temperature * 0.5)
        return min(1.0, self.config.temperature + 0.2 * sample_id / max(1, self.config.path_samples - 1))


def view_specs_to_json(view_specs: list[ViewSpec] | None) -> list[dict[str, Any]] | None:
    if view_specs is None:
        return None
    return [spec.to_json() for spec in view_specs]


def view_spec_names(view_specs: list[ViewSpec] | None) -> str:
    if view_specs is None:
        return "fixed_orbit"
    return ",".join(spec.name for spec in view_specs)


def _view_index_by_name(view_metadata: list[dict[str, Any]] | None, name: str) -> int | None:
    if not view_metadata:
        return None
    for idx, spec in enumerate(view_metadata):
        if spec.get("name") == name:
            return idx
    return None


def view_specs_key(view_specs: list[ViewSpec] | None, num_views: int) -> str:
    if view_specs is None:
        return f"fixed_{int(num_views)}"
    payload = json.dumps([spec.to_json() for spec in view_specs], sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def format_rotation_matrix_for_log(rotation_matrix: np.ndarray | None) -> str:
    if rotation_matrix is None:
        return "null"
    return json.dumps(np.asarray(rotation_matrix, dtype=np.float64).round(4).tolist(), separators=(",", ":"))


def _panel_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else []
    ) + [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    fitted = Image.new("RGB", size, (236, 239, 244))
    src = image.copy()
    scale = min(target_w / max(src.width, 1), target_h / max(src.height, 1))
    new_size = (max(1, int(round(src.width * scale))), max(1, int(round(src.height * scale))))
    src = src.resize(new_size, Image.LANCZOS)
    x = (target_w - src.width) // 2
    y = (target_h - src.height) // 2
    fitted.paste(src, (x, y))
    return fitted


def _crop_proposer_focus(image: Image.Image) -> tuple[Image.Image, list[int] | None]:
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    saturation = maxc - minc
    colored = saturation > 24
    dark = maxc < 55
    mask = colored & ~dark
    if int(mask.sum()) < 64:
        return image, None
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)
    pad = int(max(w, h) * 0.35)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image.width - 1, x1 + pad)
    y1 = min(image.height - 1, y1 + pad)
    crop_box = [x0, y0, x1 + 1, y1 + 1]
    return image.crop(tuple(crop_box)), crop_box


def vote_candidate_paths(
    candidates: list[CandidatePath],
    moving_part: str,
    moving_mesh: PartMesh,
    vote_threshold: float,
) -> tuple[CandidatePath, dict[str, Any]]:
    if not candidates:
        raise ValueError("No candidate paths to vote")
    scale = max(float(np.linalg.norm(moving_mesh.extents)), 1.0)
    n = len(candidates)
    pairwise = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dist = transform_distance(candidates[i].final_pose, candidates[j].final_pose, translation_scale=scale)
            pairwise[i, j] = pairwise[j, i] = dist

    scores = []
    for i, candidate in enumerate(candidates):
        neighbors = int(np.sum(pairwise[i] <= vote_threshold))
        final_conf = 0.0
        if candidate.final_evaluator:
            final_conf = float(candidate.final_evaluator.get("Confidence", 0.0) or 0.0)
        success_bonus = 2.0 if candidate.success else 0.0
        valid_bonus = 0.5 if candidate.valid else -2.0
        consistency = neighbors / max(1, n)
        iter_penalty = 0.04 * len(candidate.iterations)
        distance_penalty = 0.0
        if candidate.final_distance_to_mock_target is not None:
            distance_penalty = min(1.5, float(candidate.final_distance_to_mock_target))
        score = success_bonus + valid_bonus + consistency + final_conf - iter_penalty - distance_penalty
        candidate.score = score
        candidate.vote_detail = {
            "neighbor_count": neighbors,
            "consistency": round(float(consistency), 6),
            "success_bonus": success_bonus,
            "valid_bonus": valid_bonus,
            "final_confidence": final_conf,
            "iteration_penalty": round(float(iter_penalty), 6),
            "distance_penalty": round(float(distance_penalty), 6),
        }
        scores.append(score)

    selected_index = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    return candidates[selected_index], {
        "moving_part": moving_part,
        "selected_sample_id": candidates[selected_index].sample_id,
        "selected_index": selected_index,
        "scores": [round(float(score), 6) for score in scores],
        "pairwise_distance": pairwise.round(6).tolist(),
        "vote_threshold": vote_threshold,
    }


def exception_summary(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None:
        parts.append(f"cause={type(cause).__name__}: {cause}")
    return " | ".join(parts)
