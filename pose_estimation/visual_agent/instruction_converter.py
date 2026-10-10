from __future__ import annotations

from typing import Any


def convert_tree_aware_operation_instruction(
    *,
    category: str,
    name: str,
    operation_index: int,
    moving_part: str,
    fixed_parts: list[str],
    target_step_parts: list[str],
    manual_step_index: int,
    raw_step_instruction: str,
    part_names: dict[str, str],
    assembly_tree: Any,
    main_data_steps: list[dict[str, Any]],
) -> str:
    """Convert a step-level manual instruction into one executable tree-aware operation.

    The raw Manual2Skill stage2 text describes a whole manual step. The visual
    assembly loop needs a narrower instruction: move exactly one part relative
    to the already committed subassembly implied by the assembly tree.
    """

    return _generic_operation_instruction(
        operation_index=operation_index,
        moving_part=moving_part,
        fixed_parts=fixed_parts,
        target_step_parts=target_step_parts,
        manual_step_index=manual_step_index,
        raw_step_instruction=raw_step_instruction,
        part_names=part_names,
        assembly_tree=assembly_tree,
        main_data_steps=main_data_steps,
    )


def _convert_applaro_operation(
    *,
    operation_index: int,
    moving_part: str,
    fixed_parts: list[str],
    target_step_parts: list[str],
    manual_step_index: int,
    part_names: dict[str, str],
    assembly_tree: Any,
    main_data_steps: list[dict[str, Any]],
) -> str:
    tree_text = _tree_relation_text(
        assembly_tree=assembly_tree,
        main_data_steps=main_data_steps,
        manual_step_index=manual_step_index,
        moving_part=moving_part,
        fixed_parts=fixed_parts,
        target_step_parts=target_step_parts,
        part_names=part_names,
    )
    fixed_text = _parts_text(fixed_parts, part_names)
    moving_text = _part_text(moving_part, part_names)
    if int(operation_index) == 0:
        return f"""Current operation: move part {moving_part} ({part_names.get(moving_part, 'side frame')}) only.
Tree-aware role: {tree_text}
Fixed subassembly: {fixed_text}; do not move fixed parts.
Relevant manual context: Step 1 builds subassembly {{0, 1, 2}}, but this operation instantiates only the tree relation between part 2 and fixed part 1.
Goal: attach part {moving_part} ({part_names.get(moving_part, 'side frame')}) upright to one end of part 1 ({part_names.get('1', 'seat panel')}).
Required contact: part {moving_part}'s inner frame/attachment face meets the end/frame edge of part 1 and forms the first rigid panel-frame corner.
Required orientation: the side frame stands perpendicular to the panel surface; its inner attachment faces/holes point inward toward the future rail/panel ends.
Fully seated criterion: the frame is pushed onto the panel end with no visible floating, large gap, wrong end, wrong mirror, upside-down orientation, or collision.
Reject cases: moving part 0 instead, moving fixed part 1, attaching to the wrong end, leaving the frame parallel/flat, or choosing a pose where the connector face points outward."""
    if int(operation_index) == 1:
        return f"""Current operation: move part {moving_part} ({part_names.get(moving_part, 'support bar / long side rail')}) only.
Tree-aware role: {tree_text}
Fixed subassembly: {fixed_text}; do not move fixed parts.
Relevant manual context: this is the rail/side-bar relation inside Step 1 after the first side frame has been committed.
Goal: place part {moving_part} ({part_names.get(moving_part, 'support bar')}) horizontally along the bench length, bridging the seat panel and installed side frame.
Required contact: the rail long inner face aligns with the long edge/support area of part 1, and one rail end seats into the inner connection area of part 2.
Required orientation: the rail long axis runs along the bench length, parallel to the long edge of the seat panel, below/along the panel edge rather than standing upright.
Fully seated criterion: the rail is not floating, not perpendicular to the bench length, not shifted away from the side-frame connection, and not colliding through fixed parts.
Reject cases: moving side frame 2 again, placing the rail across the bench width, leaving it loose/floating, or selecting a pose that does not connect to the installed side frame."""
    if int(operation_index) == 2:
        return f"""Current operation: move part {moving_part} ({part_names.get(moving_part, 'remaining side frame')}) only.
Tree-aware role: {tree_text}
Fixed subassembly: {fixed_text}; do not move fixed parts.
Relevant manual context: Step 2 attaches the remaining side frame to close the open end of the bench.
Goal: attach part {moving_part} ({part_names.get(moving_part, 'side frame')}) to the open end opposite part 2, completing the rectangular bench frame.
Required contact: part {moving_part}'s inner frame/attachment faces connect to the free end of part 0 and the far end/frame edge of part 1.
Required orientation: the frame is upright and mirrored against part 2; its inner attachment faces point inward toward the free rail/panel ends.
Fully seated criterion: the frame is slid/pushed onto the open end with no visible floating, wrong-end placement, wrong mirror, upside-down orientation, large gap, or collision.
Reject cases: attaching beside the already installed side frame, using the wrong mirror/roll, leaving the frame outside the open end, or moving fixed parts 0/1/2."""
    return _generic_operation_instruction(
        operation_index=operation_index,
        moving_part=moving_part,
        fixed_parts=fixed_parts,
        target_step_parts=target_step_parts,
        manual_step_index=manual_step_index,
        raw_step_instruction="",
        part_names=part_names,
        assembly_tree=assembly_tree,
        main_data_steps=main_data_steps,
    )


def _generic_operation_instruction(
    *,
    operation_index: int,
    moving_part: str,
    fixed_parts: list[str],
    target_step_parts: list[str],
    manual_step_index: int,
    raw_step_instruction: str,
    part_names: dict[str, str],
    assembly_tree: Any,
    main_data_steps: list[dict[str, Any]],
) -> str:
    hint = _operation_hint(
        operation_index=operation_index,
        moving_part=moving_part,
        fixed_parts=fixed_parts,
        part_names=part_names,
    )
    return f"""Current operation: move part {moving_part} ({part_names.get(moving_part, 'unknown')}) only.
Tree-aware relation: {_tree_relation_text(assembly_tree, main_data_steps, manual_step_index, moving_part, fixed_parts, target_step_parts, part_names)}
Fixed subassembly: {_parts_text(fixed_parts, part_names)}; fixed parts are already committed and must not move.
Goal: {hint['goal']}
Required contact: {hint['contact']}
Required orientation: {hint['orientation']}
Fully seated criterion: contact faces/edges are flush or inserted as far as the manual implies; no visible floating, large gap, collision, wrong side/end, wrong mirror, or upside-down placement.
Reject cases: {hint['reject']}
Manual context for background only: {_one_line(raw_step_instruction, 320)}
Prompt policy: if a semantic part name conflicts with the visible geometry, trust the part ID, the current 3D views, the manual crop, and the tree-aware relation above."""


def _operation_hint(
    *,
    operation_index: int,
    moving_part: str,
    fixed_parts: list[str],
    part_names: dict[str, str],
) -> dict[str, str]:
    moving_name = part_names.get(moving_part, "moving part")
    fixed_text = _parts_text(fixed_parts, part_names)
    name_lower = moving_name.lower()

    if "seat" in name_lower:
        return {
            "goal": f"place part {moving_part} ({moving_name}) onto the support surfaces of the fixed subassembly.",
            "contact": "the underside/support face of the seat should rest on the top ledges, rails, or frame surfaces indicated by the manual.",
            "orientation": "seat surface should be horizontal/parallel to the support plane unless the manual shows an upside-down assembly stage.",
            "reject": "seat floating above the frame, rotated vertically, shifted off the supports, or colliding through fixed parts.",
        }
    if "bar" in name_lower or "rail" in name_lower or "support" in name_lower:
        return {
            "goal": f"insert or place part {moving_part} ({moving_name}) between the fixed components as the cross/side support for the subassembly.",
            "contact": "both bar/rail ends should meet the receiving holes, tabs, or inner faces of the fixed subassembly.",
            "orientation": "the long axis of the bar/rail should run between the receiving fixed parts and stay straight, not vertical or diagonal unless the manual shows that.",
            "reject": "only one end touching, lateral misalignment, perpendicular-to-intended direction, floating, or passing through fixed geometry.",
        }
    if "leg" in name_lower:
        return {
            "goal": f"attach part {moving_part} ({moving_name}) to the indicated underside/side mounting region of {fixed_text}.",
            "contact": "mounting tabs/ends of the leg frame should sit flush against the fixed seat/back/frame attachment points.",
            "orientation": "leg should mirror/parallel the matching leg frame when one exists; otherwise it should extend in the support direction shown in the manual.",
            "reject": "wrong side, wrong mirror, upside-down frame, tabs not seated, or leg intersecting the seat/back/frame.",
        }
    if "back" in name_lower:
        return {
            "goal": f"attach part {moving_part} ({moving_name}) to the rear/top connection region of the fixed chair subassembly.",
            "contact": "lower or side attachment edge of the backrest should meet the receiving frame/seat faces indicated by the manual.",
            "orientation": "backrest should rise relative to the seat/support plane and keep the front/back direction consistent with the manual.",
            "reject": "backrest on the wrong side, lying flat when it should stand, flipped front/back, floating, or intersecting the fixed subassembly.",
        }
    if "frame" in name_lower:
        return {
            "goal": f"attach part {moving_part} ({moving_name}) as the frame member required by this tree step.",
            "contact": "its inner connector faces, ends, holes, or tabs should meet the corresponding fixed subassembly receiving regions.",
            "orientation": "frame should match the manual's mirror/upright relation and remain symmetric or parallel to any matching installed frame.",
            "reject": "wrong end, wrong mirror, upside-down frame, outward-facing connector side, floating, or collision.",
        }
    return {
        "goal": f"place part {moving_part} ({moving_name}) into the single tree-required relation with {fixed_text}.",
        "contact": "the moving part should contact the fixed subassembly at the side/end/face implied by the tree relation and manual image.",
        "orientation": "preserve the manual-indicated upright, parallel/perpendicular, mirrored, or insertion relation.",
        "reject": "moving fixed parts, wrong contact face, wrong side/end, wrong mirror, floating, collision, or not fully seated.",
    }


def _tree_relation_text(
    assembly_tree: Any,
    main_data_steps: list[dict[str, Any]],
    manual_step_index: int,
    moving_part: str,
    fixed_parts: list[str],
    target_step_parts: list[str],
    part_names: dict[str, str],
) -> str:
    connections = _step_connections(main_data_steps, manual_step_index)
    relevant = []
    fixed_set = set(fixed_parts)
    for a, b in connections:
        a_parts = _split_group(a)
        b_parts = _split_group(b)
        if moving_part in a_parts and fixed_set.intersection(b_parts):
            relevant.append(f"{_part_group_text(a_parts, part_names)} -> {_part_group_text(b_parts, part_names)}")
        elif moving_part in b_parts and fixed_set.intersection(a_parts):
            relevant.append(f"{_part_group_text(b_parts, part_names)} -> {_part_group_text(a_parts, part_names)}")
    if not relevant:
        relevant.append(f"{_part_text(moving_part, part_names)} attaches to fixed subassembly {_parts_text(fixed_parts, part_names)}")
    target = _parts_text(target_step_parts, part_names)
    return f"assembly_tree={assembly_tree}; operation relation: {'; '.join(relevant)}; target subassembly after manual step: {{{target}}}"


def _step_connections(main_data_steps: list[dict[str, Any]], manual_step_index: int) -> list[tuple[str, str]]:
    for step in main_data_steps:
        if int(step.get("step_id", -1)) == int(manual_step_index):
            return [(str(a), str(b)) for a, b in step.get("connections", [])]
    return []


def _split_group(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _part_text(part_id: str, part_names: dict[str, str]) -> str:
    return f"part {part_id} ({part_names.get(part_id, 'unknown')})"


def _part_group_text(part_ids: list[str], part_names: dict[str, str]) -> str:
    return ", ".join(_part_text(part_id, part_names) for part_id in part_ids)


def _parts_text(part_ids: list[str], part_names: dict[str, str]) -> str:
    return ", ".join(_part_text(part_id, part_names) for part_id in part_ids)


def _one_line(text: str, limit: int) -> str:
    line = " ".join(str(text).split())
    return line[:limit] + ("..." if len(line) > limit else "")
