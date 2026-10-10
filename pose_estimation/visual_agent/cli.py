from __future__ import annotations

import argparse
import os
from pathlib import Path

from .pipeline import VisualAssemblyAgent, VisualAssemblyAgentConfig


def _default_model() -> str:
    return os.getenv("OPENAI_MODEL") or "gpt-5.2"


def _default_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL")


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Bench/applaro visual VLM closed-loop assembly agent.")
    parser.add_argument("--category", default="Bench")
    parser.add_argument("--name", default="applaro")
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("ASSEMSTATE_DATA_ROOT", str(Path(__file__).resolve().parents[2] / "examples/data"))))
    parser.add_argument(
        "--manual2skill-output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "examples/stage_outputs",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visual_vlm_agent"))
    parser.add_argument("--model", default=_default_model())
    parser.add_argument("--base-url", default=_default_base_url())
    parser.add_argument("--dry-run", action="store_true", help="Explicitly use a mock client; tests software only, not model performance.")
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--path-samples", type=int, default=5)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument(
        "--accept-confidence",
        type=float,
        default=0.70,
        help="Minimum evaluator Confidence required with Faithfulness=Yes before a candidate path is accepted.",
    )
    parser.add_argument(
        "--accept-ambiguity",
        type=float,
        default=0.30,
        help="Maximum evaluator Ambiguity allowed with Faithfulness=Yes before a candidate path is accepted.",
    )
    parser.set_defaults(strict_final_judge=_env_flag("STRICT_FINAL_JUDGE", True))
    parser.add_argument(
        "--strict-final-judge",
        dest="strict_final_judge",
        action="store_true",
        help="Require Contact angle=Correct and Target residual=Close before accepting an operation.",
    )
    parser.add_argument(
        "--no-strict-final-judge",
        dest="strict_final_judge",
        action="store_false",
        help="Use the older acceptance gate that ignores Contact angle and Target residual.",
    )
    parser.add_argument("--render-width", type=int, default=768)
    parser.add_argument("--render-height", type=int, default=768)
    parser.add_argument(
        "--camera-distance-scale",
        type=float,
        default=float(os.getenv("CAMERA_DISTANCE_SCALE", "1.15")),
        help="Minimum bbox scale for camera framing. Larger values move cameras farther away.",
    )
    parser.add_argument(
        "--camera-frame-padding",
        type=float,
        default=float(os.getenv("CAMERA_FRAME_PADDING", "1.25")),
        help="Per-view adaptive framing safety factor used after fitting the current focus bbox.",
    )
    parser.add_argument(
        "--simulation-backend",
        choices=("offline", "isaacsim"),
        default=os.getenv("SIMULATION_BACKEND", "offline"),
        help="3D execution/rendering backend. offline keeps the existing pyrender/matplotlib path; isaacsim builds and captures USD stages inside Isaac Sim.",
    )
    parser.set_defaults(isaacsim_headless=_env_flag("ISAACSIM_HEADLESS", True))
    parser.add_argument(
        "--isaacsim-headless",
        dest="isaacsim_headless",
        action="store_true",
        help="Launch/use Isaac Sim in headless mode when --simulation-backend isaacsim is selected.",
    )
    parser.add_argument(
        "--no-isaacsim-headless",
        dest="isaacsim_headless",
        action="store_false",
        help="Allow a visible Isaac Sim app/window when --simulation-backend isaacsim is selected.",
    )
    parser.add_argument(
        "--isaacsim-renderer",
        choices=(
            "auto",
            "rgb_annotator",
            "rgb-annotator",
            "replicator",
            "replicator_writer",
            "replicator-writer",
            "viewport",
            "camera",
            "movie_capture",
            "movie-capture",
            "movie",
        ),
        default=os.getenv("ISAACSIM_RENDERER", "rgb_annotator"),
        help="Isaac Sim image capture path used by the backend.",
    )
    parser.add_argument(
        "--isaacsim-capture-updates",
        type=int,
        default=int(os.getenv("ISAACSIM_CAPTURE_UPDATES", "8")),
        help="Number of Kit app updates to wait after each Isaac Sim capture request.",
    )
    parser.set_defaults(render_ground_plane=_env_flag("RENDER_GROUND_PLANE", False))
    parser.add_argument(
        "--render-ground-plane",
        dest="render_ground_plane",
        action="store_true",
        help="Render a visual ground plane in scene views.",
    )
    parser.add_argument(
        "--no-render-ground-plane",
        dest="render_ground_plane",
        action="store_false",
        help="Disable the visual ground plane in scene views.",
    )
    parser.set_defaults(render_fixed_base_axes=_env_flag("RENDER_FIXED_BASE_AXES", False))
    parser.add_argument(
        "--render-fixed-base-axes",
        dest="render_fixed_base_axes",
        action="store_true",
        help="Render cyan/magenta/yellow background world axes on the first fixed/base part.",
    )
    parser.add_argument(
        "--no-render-fixed-base-axes",
        dest="render_fixed_base_axes",
        action="store_false",
        help="Disable fixed/base background world axes.",
    )
    parser.set_defaults(render_workcell=_env_flag("RENDER_WORKCELL", False))
    parser.add_argument(
        "--render-workcell",
        dest="render_workcell",
        action="store_true",
        help="Render a visual-only work table and simplified robot arm around the parts.",
    )
    parser.add_argument(
        "--no-render-workcell",
        dest="render_workcell",
        action="store_false",
        help="Disable the visual work table and robot arm.",
    )
    parser.add_argument("--ground-y", type=float, default=float(os.getenv("GROUND_Y", "-0.05")))
    parser.add_argument("--ground-alpha", type=float, default=float(os.getenv("GROUND_ALPHA", "0.38")))
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument(
        "--rotation-mode",
        choices=("axis_angle", "matrix"),
        default=os.getenv("ROTATION_MODE", "matrix"),
        help="Pose update rotation representation. axis_angle keeps the legacy x/y/z+Angle schema; matrix lets the proposer output a 3x3 world incremental rotation matrix.",
    )
    parser.set_defaults(progressive_update_pressure=_env_flag("PROGRESSIVE_UPDATE_PRESSURE", True))
    parser.add_argument(
        "--progressive-update-pressure",
        dest="progressive_update_pressure",
        action="store_true",
        help="Enable residual/iteration-budget based proposer prompting and bounded update scaling.",
    )
    parser.add_argument(
        "--no-progressive-update-pressure",
        dest="progressive_update_pressure",
        action="store_false",
        help="Disable residual/iteration-budget based update pressure.",
    )
    parser.add_argument("--max-update-scale", type=float, default=float(os.getenv("MAX_UPDATE_SCALE", "2.5")))
    parser.add_argument("--max-translation-units", type=float, default=float(os.getenv("MAX_TRANSLATION_UNITS", "4.0")))
    parser.add_argument("--max-rotation-deg", type=float, default=float(os.getenv("MAX_ROTATION_DEG", "180.0")))
    parser.add_argument(
        "--proposer-input-mode",
        choices=("current_only_orthographic", "orthographic", "supporting_view"),
        default=os.getenv("PROPOSER_INPUT_MODE", "current_only_orthographic"),
        help=(
            "Visual input sent to proposer. current_only_orthographic renders active fixed/moving "
            "front/right/top views plus the manual image; orthographic is the oracle/debug current-target mode."
        ),
    )
    parser.add_argument(
        "--translation-unit-mode",
        choices=("scene_units", "vlm_pose"),
        default=os.getenv("TRANSLATION_UNIT_MODE", "scene_units"),
        help="Interpret proposer Translation as direct scene units or VLM-Pose AABB-normalized units.",
    )
    parser.set_defaults(continue_after_failed_operation=_env_flag("CONTINUE_AFTER_FAILED_OPERATION", False))
    parser.add_argument(
        "--continue-after-failed-operation",
        dest="continue_after_failed_operation",
        action="store_true",
        help="Commit the selected failed path provisionally and continue with the next moving part.",
    )
    parser.add_argument(
        "--stop-after-failed-operation",
        dest="continue_after_failed_operation",
        action="store_false",
        help="Restore the old behavior: reject a failed selected path and stop later operations.",
    )
    parser.add_argument("--fixed-views", action="store_true", help="Use the old fixed orbit views instead of adaptive view planning.")
    parser.add_argument(
        "--commit-failed-paths",
        action="store_true",
        help="Backward-compatible alias for --continue-after-failed-operation.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the live tqdm progress bar.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional run log path. Defaults to visual_agent_run.log in the output folder.")
    parser.add_argument(
        "--ignore-env-proxy",
        action="store_true",
        help="Do not let the OpenAI-compatible client use HTTP_PROXY/HTTPS_PROXY/ALL_PROXY from the environment.",
    )
    parser.add_argument(
        "--proxy-url",
        default=None,
        help="Explicit proxy URL for VLM calls, e.g. socks5h://127.0.0.1:7890. Overrides HTTP_PROXY/HTTPS_PROXY.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VisualAssemblyAgentConfig(
        category=args.category,
        name=args.name,
        data_root=args.data_root,
        manual2skill_output_root=args.manual2skill_output,
        output_dir=args.output_dir,
        model=args.model,
        base_url=args.base_url,
        dry_run=args.dry_run,
        num_views=args.num_views,
        path_samples=args.path_samples,
        max_iters=args.max_iters,
        accept_confidence=args.accept_confidence,
        accept_ambiguity=args.accept_ambiguity,
        render_width=args.render_width,
        render_height=args.render_height,
        camera_distance_scale=args.camera_distance_scale,
        camera_frame_padding=args.camera_frame_padding,
        simulation_backend=args.simulation_backend,
        isaacsim_headless=args.isaacsim_headless,
        isaacsim_renderer=args.isaacsim_renderer,
        isaacsim_capture_updates=args.isaacsim_capture_updates,
        render_ground_plane=args.render_ground_plane,
        render_fixed_base_axes=args.render_fixed_base_axes,
        render_workcell=args.render_workcell,
        ground_y=args.ground_y,
        ground_alpha=args.ground_alpha,
        temperature=args.temperature,
        rotation_mode=args.rotation_mode,
        progressive_update_pressure=args.progressive_update_pressure,
        max_update_scale=args.max_update_scale,
        max_translation_units=args.max_translation_units,
        max_rotation_deg=args.max_rotation_deg,
        proposer_input_mode=args.proposer_input_mode,
        translation_unit_mode=args.translation_unit_mode,
        strict_final_judge=args.strict_final_judge,
        continue_after_failed_operation=args.continue_after_failed_operation or args.commit_failed_paths,
        fixed_views=args.fixed_views,
        commit_failed_paths=args.commit_failed_paths,
        show_progress=not args.no_progress,
        log_file=args.log_file,
        trust_env_proxy=not args.ignore_env_proxy,
        proxy_url=args.proxy_url,
    )
    out_path = VisualAssemblyAgent(config).run()
    print(out_path)


if __name__ == "__main__":
    main()
