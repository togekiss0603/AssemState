from __future__ import annotations

import math
from typing import Any

from .data import BenchApplaroInputs, OperationSpec


SYSTEM_PROMPT = """You are a precise visual furniture-assembly reasoning agent.
You reason from rendered 3D views, manual images, optional target-effect/reference renders when provided, part labels, and text instructions.
Use the current scene images as the source of truth. If the scene is unclear or ambiguous, answer No in evaluation.
Orientation errors such as flipped, reversed, upside-down, mirrored, or wrong-side placement must be treated as assembly failures.
When proposing in pose-update mode, output one incremental pose update for the current moving part, not an absolute final pose.
When proposing in candidate-choice mode, choose the best rendered after-state candidate instead of inventing numeric pose values.
Pose updates use fixed scene/world axes, not the moving part's local object axes."""


def format_part_table(part_names: dict[str, str]) -> str:
    return "\n".join(f"- part {pid}: {name}" for pid, name in sorted(part_names.items(), key=lambda kv: int(kv[0])))


def context_memory_text(global_memory: list[dict[str, Any]], iteration_memory: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    if global_memory:
        lines.append("Completed operations:")
        for item in global_memory:
            commit_status = item.get("commit_status", "committed_success")
            lines.append(
                f"- operation {item.get('operation_index')}: part {item.get('moving_part')} "
                f"status={commit_status}. fixed parts after commit: {item.get('assembled_parts')}"
            )
    else:
        lines.append("Completed operations: none.")
    if iteration_memory:
        lines.append("Previous attempts for the current moving part:")
        for item in iteration_memory:
            evaluator = item.get("evaluator", {})
            proposer = item.get("proposer", {})
            pressure = item.get("update_pressure", {})
            lines.append(
                f"- iter {item.get('iteration')}: evaluator={evaluator.get('Faithfulness')} "
                f"view={evaluator.get('Image number')} residual={evaluator.get('Target residual')} "
                f"reason={evaluator.get('Reasoning')}; "
                f"proposer_reason={proposer.get('Reasoning')}; "
                f"update translation={proposer.get('Translation')} axis={proposer.get('Dominant rotation axis')} "
                f"angle={proposer.get('Angle')} scale={pressure.get('update_scale')}"
            )
    else:
        lines.append("Previous attempts for this moving part: none.")
    return "\n".join(lines)


def operation_specific_instruction_text(inputs: BenchApplaroInputs, operation: OperationSpec) -> str:
    processed = str(getattr(operation, "operation_instruction", "") or "").strip()
    if processed:
        return processed
    moving_name = inputs.part_names.get(operation.moving_part, "unknown")
    fixed_names = ", ".join(f"{pid} ({inputs.part_names.get(pid, 'unknown')})" for pid in operation.fixed_parts)
    if inputs.category == "Bench" and inputs.name == "applaro":
        if operation.operation_index == 0 and operation.moving_part == "2":
            return """Current operation: move part 2 (side frame) only.
Fixed subassembly: part 1 (seat panel).
Goal: attach side frame 2 upright to one end of seat panel 1. The frame should stand perpendicular to the panel surface, with its inner attachment faces/holes facing the panel end.
Ignore part 0 for action selection in this operation; it is only context from the same manual step."""
        if operation.operation_index == 1 and operation.moving_part == "0":
            return """Current operation: move part 0 (support bar) only.
Fixed subassembly: parts 1 and 2, already forming one panel-frame corner.
Goal: place support bar 0 horizontally along/between the seat panel and side frame as shown in the manual, with its long direction aligned to the bench length and its end seated at the side-frame connection area.
Do not move parts 1 or 2."""
        if operation.operation_index == 2 and operation.moving_part == "3":
            return """Current operation: move part 3 (remaining side frame) only.
Fixed subassembly: parts 1, 2, and 0.
Goal: attach side frame 3 upright to the open end of the bench, mirrored against side frame 2. Its inner attachment faces should point inward toward the free ends of the seat panel/support bar and the frame should be fully seated, not floating."""
    return f"""Current operation: move part {operation.moving_part} ({moving_name}) only.
Fixed subassembly: {fixed_names}.
Goal: infer the required contact side/end/face, orientation, and seating relation from the manual and text; do not move fixed parts."""


def operation_choice_instruction_text(inputs: BenchApplaroInputs, operation: OperationSpec) -> str:
    return f"""# Candidate Choice Task
Choose the single candidate after-state that best satisfies the current operation and manual.

# Operation-Specific Instruction
{operation_specific_instruction_text(inputs, operation)}

# Choice Rule
- The visual options are labeled A, B, C, ... only. Treat each option as a possible result after applying one candidate action.
- Compare each candidate against the manual and operation-specific instruction.
- Prefer the option with correct contact side/end/face, upright/mirrored orientation, height, seating, and no visible collision or floating.
- Reject options that are close but flipped, mirrored the wrong way, attached to the wrong side/end, floating, colliding, or missing the required contact.
- Do not output translation, rotation_matrix, axis, or angle. Select one option label.

# Full Manual-Step Context
{operation.instruction}

# Output JSON
Return only one JSON object:
{{
  "Selected option": "A",
  "Is acceptable": "Yes or No",
  "Instruction-derived target relation": "state the required contact/placement/orientation",
  "Reasoning": "briefly compare the selected option with the manual and mention why the main alternatives fail if visible",
  "Contact": "Correct or Incorrect or Unclear",
  "Orientation": "Correct or Incorrect or Unclear",
  "Side/end": "Correct or Incorrect or Unclear",
  "Floating/collision": "None or Floating or Collision or Unclear",
  "Confidence": 0.0
}}"""


def adaptive_view_text(view_specs: list[dict[str, Any]] | None) -> str:
    if not view_specs:
        return "Paper-style orbit view planning: views use a fixed equally spaced circular camera ring."
    lines = [
        "Paper-style orbit view planning: current views use equally spaced circular camera orientations; target-effect views share these orientations only when target-effect views are provided.",
        "Each view is a clean object-only render; the x/y/z pose axes are not drawn in the image.",
    ]
    for idx, spec in enumerate(view_specs, start=1):
        hint = _view_axis_projection_hint(spec)
        lines.append(
            f"- view {idx}: {spec.get('name')}: elev={spec.get('elev')}, azim={spec.get('azim')}; "
            f"{hint}; {spec.get('reason')}"
        )
    return "\n".join(lines)


def _view_axis_projection_hint(spec: dict[str, Any]) -> str:
    try:
        elev = math.radians(float(spec.get("elev", 20.0)))
        azim = math.radians(float(spec.get("azim", 90.0)))
    except (TypeError, ValueError):
        return "screen-axis hint unavailable"

    right = (math.sin(azim), 0.0, -math.cos(azim))
    up = (-math.sin(elev) * math.cos(azim), math.cos(elev), -math.sin(elev) * math.sin(azim))
    back = (math.cos(elev) * math.cos(azim), math.sin(elev), math.cos(elev) * math.sin(azim))
    axes = {
        "+x": (1.0, 0.0, 0.0),
        "+y": (0.0, 1.0, 0.0),
        "+z": (0.0, 0.0, 1.0),
    }

    def describe(axis: tuple[float, float, float]) -> str:
        sx = sum(a * b for a, b in zip(axis, right))
        sy = sum(a * b for a, b in zip(axis, up))
        sd = sum(a * b for a, b in zip(axis, back))
        components: list[str] = []
        if abs(sx) >= 0.35:
            components.append("image-right" if sx > 0 else "image-left")
        if abs(sy) >= 0.35:
            components.append("image-up" if sy > 0 else "image-down")
        if abs(sd) >= 0.55:
            components.append("toward camera/depth" if sd > 0 else "away from camera/depth")
        return "+".join(components) if components else "mostly depth"

    return "screen hint " + ", ".join(f"{name}->{describe(axis)}" for name, axis in axes.items())


def assembly_geometry_skills_text(
    inputs: BenchApplaroInputs,
    operation: OperationSpec,
    fixed_base_axis_part: str | None = None,
) -> str:
    return "\n\n".join(
        [
            render_scene_observation_skill_text(),
            world_axis_update_skill_text(inputs, fixed_base_axis_part),
            manual_contact_geometry_skill_text(),
            perpendicular_parallel_skill_text(),
        ]
    )


def render_scene_observation_skill_text() -> str:
    return """## Rendered Scene Observation Skill
- Each rendered 3D view is an object-only camera image: no coordinate axes, no ruler ticks, no title bar, and no instructional text are drawn into the image.
- Judge part pose, contact, height, orientation, and relative position only from the furniture geometry and part colors visible in the scene.
- Use the prompt text and view metadata, not image annotations, to interpret world-axis directions and translation scale."""


def fixed_base_axis_reference_text(inputs: BenchApplaroInputs, fixed_base_axis_part: str | None) -> str:
    if fixed_base_axis_part is None:
        return ""
    part_name = inputs.part_names.get(fixed_base_axis_part, "unknown")
    return f"""
- If cyan/magenta/yellow arrows appear near fixed base part {fixed_base_axis_part} ({part_name}), they are background world-axis references only: cyan = +x, magenta = +y, yellow = +z.
- These fixed-base arrows follow the same scene/world directions as the moving-part arrows, but they are not pose-update axes and do not indicate a movable target.
- The fixed-base arrows are visual annotations only. Do not treat them as furniture geometry, part surfaces, obstacles, connectors, or assembly targets."""


def world_axis_update_skill_text(inputs: BenchApplaroInputs, fixed_base_axis_part: str | None = None) -> str:
    return f"""## World-Axis Update Skill
- Pose updates use a fixed scene/world coordinate system, not the moving part's local coordinate frame and not the camera/image coordinate frame.
- World axes stay fixed across all views and iterations: +x = right, +y = top/up, +z = front.
- Translation [Tx, Ty, Tz] moves the moving part along these fixed world-axis directions, scaled by a stable per-part update unit.
- The scene image is for judging object positions, contact, support, height, overlap, and orientation. The prompt text defines the pose-update axes and scale.
- Rotation also uses these fixed world axes: "Dominant rotation axis" means rotate around the displayed world x, y, or z direction using the right-hand rule.
- Do not assume x/y/z rotates with the object or camera. After the moving part has been rotated, x/y/z still refer to the fixed scene/world directions, not to the object's own long side, face normal, or local mesh axes.
- The scene view itself intentionally has no coordinate arrows on the moving part; use this text convention and the selected orbit-view metadata to interpret update axes.
- Direction words follow the same world-axis convention: behind/back = -z, front = +z, left = -x, right = +x, top/up = +y, and bottom/down = -y.
- Positive rotation signs follow the right-hand rule with these concrete conventions: +Rx rotates +y toward +z, +Ry rotates +z toward +x, and +Rz rotates +x toward +y.
- Negative rotation means the opposite direction: -Rx rotates +z toward +y, -Ry rotates +x toward +z, and -Rz rotates +y toward +x.{fixed_base_axis_reference_text(inputs, fixed_base_axis_part)}"""


def manual_contact_geometry_skill_text() -> str:
    return """## Manual Contact Geometry Skill
- Use the manual page, numbered manual crop, current views, and any optional target-effect/reference views if they are explicitly provided.
- Infer how the moving part should touch the fixed part: perpendicular contact, parallel contact, horizontal support, end-to-end contact, or unknown.
- Judge the visible contact location and contact relation, not only whether the parts are close.
- A part that is close but touching at the wrong face or wrong side is not assembled correctly."""


def perpendicular_parallel_skill_text() -> str:
    return """## Perpendicular vs Parallel Skill
- Explicitly decide whether the target relation appears perpendicular or parallel.
- First infer the intended perpendicular/parallel relation from the manual image and any explicitly provided reference views, then verify whether the current moving part has the same relation to the fixed parts.
- If the manual/reference evidence shows a perpendicular joint but the current view is parallel, skewed, reversed, or unclear, set Contact angle=Incorrect or Unclear and answer Faithfulness=No.
- If the manual/reference evidence shows a parallel joint but the current view is perpendicular, skewed, reversed, or unclear, set Contact angle=Incorrect or Unclear and answer Faithfulness=No.
- Use this relation as visual evidence for orientation and contact reasoning, but do not use a standalone contact-angle penalty.
- If the contact angle cannot be verified from the current views, set Contact angle=Unclear and explain the uncertainty."""


def evaluator_feedback_text(evaluator_reasoning: str, evaluator_feedback: dict[str, Any] | None) -> str:
    if not evaluator_feedback:
        return evaluator_reasoning
    fields = [
        ("Reasoning", evaluator_reasoning),
        ("Contact", evaluator_feedback.get("Contact")),
        ("Contact angle", evaluator_feedback.get("Contact angle")),
        ("Required relation", evaluator_feedback.get("Required relation")),
        ("Target residual", evaluator_feedback.get("Target residual")),
        ("Orientation", evaluator_feedback.get("Orientation")),
        ("Ambiguity", evaluator_feedback.get("Ambiguity")),
    ]
    return "\n".join(f"- {name}: {value}" for name, value in fields if value is not None and value != "")


def build_evaluator_prompt(
    inputs: BenchApplaroInputs,
    operation: OperationSpec,
    global_memory: list[dict[str, Any]],
    iteration_memory: list[dict[str, Any]],
    num_current_views: int,
    num_target_views: int,
    view_specs: list[dict[str, Any]] | None = None,
    fixed_base_axis_part: str | None = None,
) -> str:
    moving_name = inputs.part_names.get(operation.moving_part, "unknown")
    fixed_names = ", ".join(f"{pid} ({inputs.part_names.get(pid, 'unknown')})" for pid in operation.fixed_parts)
    operation_target_ids = [*operation.fixed_parts, operation.moving_part]
    operation_target_names = ", ".join(
        f"{pid} ({inputs.part_names.get(pid, 'unknown')})" for pid in operation_target_ids
    )
    manual_target_names = ", ".join(f"{pid} ({inputs.part_names.get(pid, 'unknown')})" for pid in operation.target_step_parts)
    has_target_views = int(num_target_views) > 0
    operation_visibility_line = (
        f"- Operation target parts visible in the current and target-effect renders: {operation_target_names}"
        if has_target_views
        else f"- Operation relevant parts to judge in the current scene: {operation_target_names}"
    )
    scene_scope_line = (
        "- The rendered 3D current/target views intentionally hide unrelated unassembled parts."
        if has_target_views
        else "- Current scene views may include unrelated loose parts. Ignore them except as clutter/collision context; judge the moving part only relative to the fixed parts."
    )
    orientation_reference = "target-effect views or manual image" if has_target_views else "manual image and text instruction"
    residual_reference = (
        "the text instruction, manual images, and paired target-effect views"
        if has_target_views
        else "the text instruction and manual images"
    )
    supporting_view_text = (
        """- Compare current view i with target-effect view i using the same camera direction.
- Use all view images jointly to decide Faithfulness. Then pick one view image number that best supports your final judgment.
- Prefer the view whose paired target-effect view uses the same camera direction and where the current moving part appears closest to the target arrangement under that same viewpoint.
- The best supporting view should expose the most useful visual evidence for your judgment: contact location, orientation, height, side, floating/collision, or relative-position relation.
- Do not choose a view just because it is aesthetically clear; choose the view that best supports the final Yes/No judgment and is useful for the next pose update.
- If several views are similarly supportive, choose the one where the moving part's relation to fixed parts would be easiest to interpret for a pose update.
- If Faithfulness is Yes, choose the view that most clearly verifies correct contact, orientation, height, and relative position."""
        if has_target_views
        else """- Use all current scene views jointly to decide Faithfulness. Then pick one current view image number that best supports your final judgment.
- Prefer the view where the moving part's relation to the fixed subassembly is easiest to interpret for the next pose update.
- The best supporting view should expose contact location, orientation, height, side/end choice, floating/collision, or relative-position relation.
- Do not choose a view just because it is aesthetically clear; choose the view that best supports the final Yes/No judgment."""
    )
    images_text = (
        f"""- Current scene views: first {num_current_views} images. These show only the fixed parts plus the current moving part after previous updates.
- Target-effect views: next {num_target_views} images. These show the target subassembly for the same operation target parts.
- Pair current view i with target-effect view i when comparing placement, contact, height, and orientation.
- Remaining images: global product manual page, current step manual overlay, numbered manual crop, and initial numbered top-view scene."""
        if has_target_views
        else f"""- Current scene views: first {num_current_views} images. These show the whole current OBJ scene: already placed parts plus loose unassembled parts.
- Remaining image: the numbered manual crop for this operation.
- No target-effect render or GT assembled OBJ image is provided. Judge the intended relation from the manual/instruction only."""
    )
    return f"""# Task
Does the current rendered 3D scene satisfy the text and manual instruction for the current assembly operation?
If the relation is visually unclear, ambiguous, colliding, floating, reversed, or only partially correct, answer "No".

# Current Operation
- Moving part: {operation.moving_part} ({moving_name})
- Fixed parts / current subassembly: {fixed_names}
{operation_visibility_line}
- Full manual-step parts for context only: {manual_target_names}
- Manual step index: {operation.manual_step_index}

# Operation-Specific Instruction
{operation_specific_instruction_text(inputs, operation)}

# Judging Scope
- Judge only whether the moving part has the correct pose, contact, height, and orientation relative to the fixed parts.
{scene_scope_line}
- Do not answer No because a part from the full manual step is absent unless it is listed in the operation target parts above.
- Fixed parts are already committed and must be treated as immovable.
- The standalone contact-angle rotation penalty is disabled; always set "Rotation penalty" to "None".
- Still judge "Contact angle" itself strictly: set it to "Correct" only when the mating face/edge/rail/end is seated at the angle implied by the manual and fixed subassembly. If it is merely close, tilted, floating, crossing through another part, or hard to verify, set "Contact angle" to "Incorrect" or "Unclear" and answer No.

# Assembly Geometry Skills
{assembly_geometry_skills_text(inputs, operation, fixed_base_axis_part)}

# Orientation Hard Constraint
- Orientation is as important as contact and position.
- Answer No if the moving part is flipped, upside-down, mirrored, reversed front/back, attached to the wrong side, or has its long axis/connectors/holes/legs facing a different direction from the {orientation_reference}.
- Answer No if the part touches the fixed subassembly but its orientation is wrong.
- Set "Orientation": "Correct" only when at least two useful views support the same orientation conclusion.
- Set "Orientation": "Unclear" and answer No when the orientation cannot be verified from the available views.
- In Reasoning, explicitly mention the orientation evidence whenever Faithfulness is Yes.

# Confidence and Ambiguity Rubric
- Confidence means how sure you are about your Faithfulness answer, whether it is Yes or No.
- Ambiguity means how unclear the visual evidence is due to occlusion, missing contact evidence, insufficient view angle, or hard-to-judge depth.
- Use Ambiguity <= 0.20 only when multiple views clearly show contact, height, orientation, and relation.
- Use Ambiguity 0.30-0.60 when the relation may be correct but one or more views are occluded or depth/contact is uncertain.
- Use Ambiguity >= 0.70 when the scene is visually unclear or the relevant relation cannot be verified.
- Wrong or unverifiable orientation should increase Ambiguity and force Faithfulness to No.
- If Ambiguity is high, answer No even if the part looks close.

# Target Residual Rubric
- Target residual means how far the moving part still appears from the target pose described jointly by {residual_reference}.
- Use "Far" when the moving part is clearly far from the intended contact area, on the wrong side, has a clearly wrong height, or has a major orientation mismatch.
- Use "Medium" when the part is near the correct area but still needs visible pose correction before contact, height, or orientation matches.
- Use "Close" only when the part is nearly assembled and only needs small final alignment.
- Use "Unclear" when the residual cannot be judged from the current views.
- A Yes answer must have Target residual "Close"; if the residual is Medium, Far, or Unclear, answer No so the proposer gets another chance.

# Supporting View Selection
- "Image number" must be the current scene view that should be used as the supporting view for the next pose update.
{supporting_view_text}

# Full Manual-Step Context
{operation.instruction}

# Part label table
{format_part_table(inputs.part_names)}

# Context Memory Accumulation
{context_memory_text(global_memory, iteration_memory)}

# Adaptive View Selection
{adaptive_view_text(view_specs)}

# Images
{images_text}

# Output JSON
Return only one JSON object:
{{
  "Faithfulness": "Yes" or "No",
  "Image number": 1,
  "Reasoning": "brief visual reason; if Yes, state the visible evidence that contact, height, orientation, and relation are correct",
  "Confidence": 0.0,
  "Ambiguity": 1.0,
  "Orientation": "Correct or Incorrect or Unclear",
  "Contact": "Correct or Incorrect or Unclear",
  "Contact angle": "Correct or Incorrect or Unclear",
  "Required relation": "Perpendicular or Parallel or Horizontal or End-to-end or Unknown",
  "Rotation penalty": "None",
  "Target residual": "Far or Medium or Close or Unclear"
}}"""


def translation_scale_reference_text(reference: dict[str, Any] | None) -> str:
    if not reference:
        return "- 1 translation unit is the stable per-part update unit computed from the moving part AABB."
    unit = reference.get("translation_unit_scene_units")
    extents = reference.get("moving_part_aabb_extents_scene_units_xyz")
    fractions = reference.get("translation_unit_fraction_of_aabb_xyz")
    mode = str(reference.get("translation_unit_mode") or "vlm_pose")
    if mode == "scene_units":
        lines = [
            "- Translation values are expressed directly in world scene units.",
            "- 1 translation unit = 1.000000 scene unit.",
            "- Read approximate dx/dy/dz from the orthographic coordinate ticks when available.",
        ]
    else:
        lines = [
            "- Translation values are normalized by the VLM-Pose AABB arrow-length rule.",
            f"- 1 translation unit = {float(unit):.6f} scene units.",
        ]
    if isinstance(extents, list) and len(extents) == 3:
        lines.append(
            "- Moving part AABB size in scene units [X, Y, Z] = "
            f"[{float(extents[0]):.6f}, {float(extents[1]):.6f}, {float(extents[2]):.6f}]."
        )
    if mode != "scene_units" and isinstance(fractions, list) and len(fractions) == 3:
        lines.append(
            "- So 1 translation unit is about "
            f"{float(fractions[0]):.2f} of the moving part X-size, "
            f"{float(fractions[1]):.2f} of its Y-size, and "
            f"{float(fractions[2]):.2f} of its Z-size."
        )
    lines.append("- Do not infer scale from image pixels; use this numeric scene-unit scale.")
    return "\n".join(lines)


def proposer_input_description_text(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode == "orthographic":
        return """The visual input is an orthographic measurement image, not a perspective camera view.
It contains six clean panels: current front/right/top views and target-effect front/right/top views.
All panels use the same fixed world-coordinate bounds and metric tick units.
- Front panels show X horizontally and Y vertically; they are best for Tx/Ty and Rz.
- Right panels show Z horizontally and Y vertically; they are best for Tz/Ty and Rx.
- Top panels show X horizontally and Z vertically; they are best for Tx/Tz and Ry.
Use current-vs-target coordinate differences directly to estimate translation. Use the moving part's projected long edges and faces across all three views to estimate rotation."""
    if mode == "current_only_orthographic":
        return """The first visual input is a current-only orthographic measurement image, not a perspective camera view.
It contains three clean current-state panels only: front, right, and top. These panels show only the active operation subset: the fixed subassembly plus the moving part. There is no target render in this mode.
The next visual input is the manual step image for the current operation.
- Front shows X horizontally and Y vertically; use it to read current Tx/Ty placement and vertical/upright errors.
- Right shows Z horizontally and Y vertically; use it to read current Tz/Ty depth and height errors.
- Top shows X horizontally and Z vertically; use it to read current plan-view gaps, side/end choice, and horizontal orientation.
- Infer the desired target relation from the text instruction, manual image, assembly tree relation, evaluator feedback, and memory. Do not assume a GT target image exists.
- Unrelated loose parts are intentionally hidden from this measurement image; use the initial full-scene part-id image only as context.
- Use the current-state coordinate ticks to choose an incremental move toward the instructed contact side/end/face."""
    return """The visual input is one clean IsaacSim supporting view selected by the evaluator.
The scene view intentionally has no coordinate arrows or compass panel on the parts."""


def build_proposer_prompt(
    inputs: BenchApplaroInputs,
    operation: OperationSpec,
    evaluator_reasoning: str,
    global_memory: list[dict[str, Any]],
    iteration_memory: list[dict[str, Any]],
    view_specs: list[dict[str, Any]] | None = None,
    evaluator_feedback: dict[str, Any] | None = None,
    rotation_mode: str = "matrix",
    iteration_index: int | None = None,
    max_iters: int | None = None,
    remaining_iters: int | None = None,
    target_residual: str = "Unclear",
    update_scale: float = 1.0,
    progressive_update_pressure: bool = True,
    fixed_base_axis_part: str | None = None,
    translation_scale_reference: dict[str, Any] | None = None,
    proposer_input_mode: str = "current_only_orthographic",
) -> str:
    moving_name = inputs.part_names.get(operation.moving_part, "unknown")
    fixed_names = ", ".join(f"{pid} ({inputs.part_names.get(pid, 'unknown')})" for pid in operation.fixed_parts)
    operation_target_ids = [*operation.fixed_parts, operation.moving_part]
    operation_target_names = ", ".join(
        f"{pid} ({inputs.part_names.get(pid, 'unknown')})" for pid in operation_target_ids
    )
    rotation_policy = proposer_rotation_policy_text(rotation_mode)
    output_json = proposer_output_json_text(moving_name, rotation_mode)
    pressure_text = iteration_budget_pressure_text(
        iteration_index=iteration_index,
        max_iters=max_iters,
        remaining_iters=remaining_iters,
        target_residual=target_residual,
        update_scale=update_scale,
        enabled=progressive_update_pressure,
    )
    scale_text = translation_scale_reference_text(translation_scale_reference)
    input_text = proposer_input_description_text(proposer_input_mode)
    return f"""# Task
The evaluator judged that the current scene does not yet satisfy the text/manual instruction.

Question: What is the target object's incremental pose update?

Predict one incremental update for the moving part only. Do not output an absolute final pose.
{input_text}

First read the instruction and evaluator feedback to infer what relation should be achieved. Then use the visual input for object geometry; use the text world-axis and scale convention below for the incremental movement axes.

Use this fixed scene/world update frame:
- +x = world right
- +y = world top/up
- +z = world front
- These are world axes, not the moving part's local axes.
- The pose update is applied around the moving part center each iteration, but the x/y/z directions remain fixed in the scene/world frame.
- These directions do not rotate with the object and do not rotate with the camera view.
- These fixed world directions are used by the pose-update engine for both translation and rotation.
- Even after the object rotates, x/y/z still mean the fixed scene/world directions.
- Prefer conservative incremental updates when the depth or scale is unclear.
{fixed_base_axis_reference_text(inputs, fixed_base_axis_part)}

Translation is a full continuous 3D vector [Tx, Ty, Tz] in stable per-part update units.
{scale_text}
- Tx, Ty, and Tz may all be non-zero at the same time.
- Do not restrict translation to a single axis or to integer steps.
- Use fractional values for fine corrections, such as [0.25, -0.40, 0.10].
- Move the moving part toward the intended connection area as directly as possible.
- If the fixed and moving parts are clearly separated and the target relation is clear, close most of the visible gap in one update while avoiding obvious collision or overshoot.
- Use smaller fractional values only for final alignment after the parts are already close to the intended contact.
- If the part must move diagonally or vertically while sliding, output combined components, not a sequence of one-axis moves.

{pressure_text}

{rotation_policy}

# Current Operation
- Moving part: {operation.moving_part} ({moving_name})
- Fixed parts that must not move: {fixed_names}
- Operation relevant parts to align: {operation_target_names}
- Manual step index: {operation.manual_step_index}

# Operation-Specific Instruction
{operation_specific_instruction_text(inputs, operation)}

# Proposal Scope
- Propose one update for the moving part only.
- In orthographic mode, the visual input shows current and target-effect panels side by side for the same world-coordinate bounds. Use their coordinate differences to estimate translation.
- In current_only_orthographic mode, the visual input shows only the active operation subset in front/right/top panels with no target render. Use the manual image, instruction text, tree relation, evaluator feedback, and memory to infer the missing target relation.
- In supporting-view mode, the selected 3D render contains only the fixed parts plus the current moving part, so infer the target relation from the text instruction, manual-derived operation description, evaluator feedback, and accumulated memory.
- Ignore unrelated parts from the same manual page unless they are listed as fixed or moving in this operation.
- The scene image has no coordinate arrows or compass panel on the moving part. Use the text world-axis convention and selected orbit-view metadata for pose-update axes.

# Required Reasoning Order
1. Fill "Instruction-derived target relation" from the instruction/manual: which fixed part or side/end/face should the moving part contact, and whether the relation should be parallel, perpendicular, upright, seated, or slid/pushed into place.
2. Fill "Current error" from the selected supporting view plus evaluator feedback: wrong side, too far, wrong height, wrong orientation, floating, collision, not seated, or unclear.
3. Choose Translation and Rotation to reduce that error in one incremental step.

# Assembly Geometry Skills
{assembly_geometry_skills_text(inputs, operation, fixed_base_axis_part)}

# Full Manual-Step Context
{operation.instruction}

# Current Assembly Failure Diagnosis From Evaluator
The following feedback describes why the current scene is not assembled successfully.
Treat it as the concrete failure diagnosis for this pose update.
Your pose update must directly improve these failure points.
In your Reasoning, explain how the Translation and Rotation reduce the evaluator's stated problems, especially contact, contact angle, orientation, wrong side, wrong height, and target residual.
{evaluator_feedback_text(evaluator_reasoning, evaluator_feedback)}

# Context Memory Accumulation
{context_memory_text(global_memory, iteration_memory)}

# Adaptive View Selection
{adaptive_view_text(view_specs)}

# Output JSON
Return only one JSON object with these exact fields and types. Do not nest the update inside another object. Do not write Translation as text.
{output_json}"""


def iteration_budget_pressure_text(
    iteration_index: int | None,
    max_iters: int | None,
    remaining_iters: int | None,
    target_residual: str,
    update_scale: float,
    enabled: bool,
) -> str:
    if not enabled:
        return """## Iteration Budget
Progressive update pressure is disabled. Still propose a direct incremental update that improves the current pose."""
    iter_text = "unknown" if iteration_index is None else str(int(iteration_index))
    max_text = "unknown" if max_iters is None else str(int(max_iters))
    remaining_text = "unknown" if remaining_iters is None else str(int(remaining_iters))
    residual = target_residual if target_residual in {"Far", "Medium", "Close", "Unclear"} else "Unclear"
    return f"""## Iteration-Budget Pressure Skill
- Current iteration index: {iter_text}; max iterations for this candidate: {max_text}; remaining proposer updates after this one: {remaining_text}.
- Evaluator target residual: {residual}.
- Suggested update scale for this turn: {float(update_scale):.3f}.
- You should use the remaining iteration budget to reach the target pose described by the text/manual instruction and evaluator feedback.
- If residual is Far or Unclear, avoid tiny hesitant updates; make a direct, substantial pose correction toward the intended contact area, contact angle, height, and orientation.
- If residual is Medium, close the visible remaining gap while preserving the correct side and contact relation.
- If residual is Close, use a smaller precise correction for final alignment.
- The code will apply a bounded numeric scale to your update, so choose the correct direction and rotation relation first."""


def proposer_rotation_policy_text(rotation_mode: str) -> str:
    if str(rotation_mode).strip().lower() == "matrix":
        return """## World Rotation Matrix Skill
Rotation is parameterized as a world incremental rotation matrix, with axis/angle kept only as a fallback/debug summary.
- Use "Rotation matrix" as a 3x3 world-axis incremental rotation R_delta.
- The matrix is not a local object rotation and not an absolute target orientation.
- The pose engine applies R_new = R_delta @ R_old.
- Prefer the full matrix for coupled 3D corrections: if the part needs simultaneous yaw, pitch, or roll changes to fix contact, height, and orientation, compose those rotations into one R_delta instead of choosing only one x/y/z axis.
- Do not merely convert the fallback axis/Angle into a matrix unless a single world-axis rotation is truly sufficient.
- The "Dominant rotation axis" and "Angle" fields are only a rough debug/fallback summary and do not need to fully describe every component of the matrix.
- Translation and rotation may both be non-zero in the same update.
- If no rotation is needed, output the identity matrix.
- If you are uncertain about a full matrix, output "Rotation matrix": null and use "Dominant rotation axis" plus "Angle" as the fallback.
- The fallback axis still means the fixed displayed world x/y/z axis, with angle in degrees using the right-hand rule.
- Use rotation only as needed to make the visible connection, contact angle, and perpendicular/parallel relation match the instruction-derived target relation.
- Keep the moving part progressing toward the fixed subassembly; do not let orientation reasoning leave the parts far apart when the intended connection location is clear."""
    return """Rotation is parameterized separately from translation, but translation and rotation may both be non-zero in the same update.
- If the part is both misplaced and misoriented, output a non-zero Translation vector and a non-zero Angle together.
- Do not choose only translation or only rotation unless the other component already looks correct.
- Rotation should still use one dominant axis x/y/z and an angle in degrees using the right-hand rule.
- If no rotation is needed, still output "Dominant rotation axis": "z" and "Angle": 0.0. Never output "none".
- Use rotation only as needed to make the visible connection, contact angle, and perpendicular/parallel relation match the instruction-derived target relation.
- Keep the moving part progressing toward the fixed subassembly; do not let orientation reasoning leave the parts far apart when the intended connection location is clear."""


def proposer_output_json_text(moving_name: str, rotation_mode: str) -> str:
    if str(rotation_mode).strip().lower() == "matrix":
        return f"""{{
  "Target object": "{moving_name}",
  "Instruction-derived target relation": "state the required contact/placement/orientation inferred from the instruction",
  "Current error": "state how the left current view differs from the required relation",
  "Reasoning": "explain why the update improves contact, contact angle, orientation, and position",
  "Translation": [0.0, 0.0, 0.0],
  "Rotation matrix": [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ],
  "Dominant rotation axis": "z",
  "Angle": 0.0,
  "Confidence": 0.0
}}"""
    return f"""{{
  "Target object": "{moving_name}",
  "Instruction-derived target relation": "state the required contact/placement/orientation inferred from the instruction",
  "Current error": "state how the left current view differs from the required relation",
  "Reasoning": "explain why the update improves contact, contact angle, orientation, and position",
  "Translation": [0.0, 0.0, 0.0],
  "Dominant rotation axis": "x",
  "Angle": 0.0,
  "Confidence": 0.0
}}"""
