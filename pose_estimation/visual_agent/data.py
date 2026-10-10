from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .instruction_converter import convert_tree_aware_operation_instruction


@dataclass(frozen=True)
class OperationSpec:
    operation_index: int
    moving_part: str
    fixed_parts: list[str]
    manual_step_index: int
    target_step_parts: list[str]
    instruction: str
    operation_instruction: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "operation_index": self.operation_index,
            "moving_part": self.moving_part,
            "fixed_parts": self.fixed_parts,
            "manual_step_index": self.manual_step_index,
            "target_step_parts": self.target_step_parts,
            "instruction": self.instruction,
            "operation_instruction": self.operation_instruction,
        }


@dataclass(frozen=True)
class BenchApplaroInputs:
    category: str
    name: str
    data_root: Path
    manual2skill_output_root: Path
    scene_obj: Path
    scene_parts_dir: Path
    target_parts_dir: Path
    scene_rot_annotated: Path
    product_page: Path
    stage1_json: Path
    stage2_text: Path
    manual_step_overlay: dict[int, Path]
    manual_step_numbered: dict[int, Path]
    part_names: dict[str, str]
    assembly_tree: Any
    main_data_steps: list[dict[str, Any]]
    stage_instructions: dict[int, str]
    operations: list[OperationSpec]

    def to_json(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "data_root": str(self.data_root),
            "manual2skill_output_root": str(self.manual2skill_output_root),
            "scene_obj": str(self.scene_obj),
            "scene_parts_dir": str(self.scene_parts_dir),
            "target_parts_dir": str(self.target_parts_dir),
            "scene_rot_annotated": str(self.scene_rot_annotated),
            "product_page": str(self.product_page),
            "stage1_json": str(self.stage1_json),
            "stage2_text": str(self.stage2_text),
            "manual_step_overlay": {str(k): str(v) for k, v in self.manual_step_overlay.items()},
            "manual_step_numbered": {str(k): str(v) for k, v in self.manual_step_numbered.items()},
            "part_names": self.part_names,
            "assembly_tree": self.assembly_tree,
            "operations": [op.to_json() for op in self.operations],
        }


def load_bench_applaro_inputs(
    data_root: str | Path,
    manual2skill_output_root: str | Path,
    category: str,
    name: str,
) -> BenchApplaroInputs:
    data_root = Path(data_root)
    manual2skill_output_root = Path(manual2skill_output_root)
    scene_dir = data_root / "prescene" / category / name
    mask_dir = data_root / "mask" / category / name
    pdf_dir = data_root / "pdfs" / category / name
    stage_dir = manual2skill_output_root / category / name
    main_data_path = data_root / "main_data.json"
    stage1_path = resolve_stage_output_file(stage_dir, "stage1_output.json")
    stage2_path = resolve_stage_output_file(stage_dir, "stage2_output.txt")

    part_names = load_stage1_part_names(stage1_path)
    record = load_main_data_record(main_data_path, category, name)
    stage_tree = load_stage_tree(stage_dir)
    atomic_contract = load_pose_atomic_contract(stage_dir)
    contract_tree = atomic_contract.get("tree") if atomic_contract else None
    assembly_tree = contract_tree if contract_tree is not None else stage_tree if stage_tree is not None else record.get("assembly_tree")
    stage_instructions = parse_stage2_instructions(stage2_path)
    if atomic_contract:
        operations = build_operations_from_pose_atomic_contract(
            atomic_contract=atomic_contract,
            stage_instructions=stage_instructions,
            part_names=part_names,
            assembly_tree=assembly_tree,
        )
    else:
        operations = build_operations_for_case(
            category=category,
            name=name,
            stage_instructions=stage_instructions,
            part_names=part_names,
            assembly_tree=assembly_tree,
            main_data_steps=record.get("steps", []),
        )
    step_ids = sorted({int(step.get("step_id", idx)) for idx, step in enumerate(record.get("steps", []))})

    inputs = BenchApplaroInputs(
        category=category,
        name=name,
        data_root=data_root,
        manual2skill_output_root=manual2skill_output_root,
        scene_obj=scene_dir / "scene.obj",
        scene_parts_dir=scene_dir / "scene_parts",
        target_parts_dir=data_root / "parts" / category / name,
        scene_rot_annotated=scene_dir / "scene_rot_annotated.png",
        product_page=pdf_dir / "page_1.png",
        stage1_json=stage1_path,
        stage2_text=stage2_path,
        manual_step_overlay={step_id: mask_dir / f"step_{step_id}_mask_overlay.png" for step_id in step_ids},
        manual_step_numbered={step_id: resolve_manual_step_numbered(mask_dir, step_id) for step_id in step_ids},
        part_names=part_names,
        assembly_tree=assembly_tree,
        main_data_steps=record.get("steps", []),
        stage_instructions=stage_instructions,
        operations=operations,
    )
    validate_inputs(inputs)
    return inputs


def resolve_stage_output_file(stage_dir: Path, filename: str) -> Path:
    direct = stage_dir / filename
    if direct.exists():
        return direct
    debug = stage_dir / "debug" / filename
    if debug.exists():
        return debug
    return direct


def resolve_manual_step_numbered(mask_dir: Path, step_id: int) -> Path:
    direct = mask_dir / f"step_{step_id}_mask_overlay_cropped_numbered.png"
    if direct.exists():
        return direct
    # A small number of dataset files use "steo" instead of "step".
    typo = mask_dir / f"steo_{step_id}_mask_overlay_cropped_numbered.png"
    if typo.exists():
        return typo
    return direct


def load_stage1_part_names(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    names: dict[str, str] = {}
    for item in data:
        name = str(item.get("name", "")).strip()
        for number in item.get("number", []):
            names[str(int(number)) if str(number).isdigit() else str(number)] = name
    return names


def load_main_data_record(path: Path, category: str, name: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        if item.get("category") == category and item.get("name") == name:
            return item
    raise KeyError(f"Missing main_data record for {category}/{name}")


def load_stage_tree(stage_dir: Path) -> Any | None:
    """Load the Stage2/materialized tree when available.

    Some materialization scripts store ``tree.json`` as a JSON-encoded string
    containing the actual nested tree, so this helper accepts both the direct
    list form and the encoded-string form.
    """

    path = resolve_stage_output_file(stage_dir, "tree.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, str):
        stripped = data.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return data


def load_pose_atomic_contract(stage_dir: Path) -> dict[str, Any] | None:
    """Load the direct Stage2->pose atomic contract when present."""

    path = resolve_stage_output_file(stage_dir, "pose_atomic_contract.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def build_operations_from_pose_atomic_contract(
    *,
    atomic_contract: dict[str, Any],
    stage_instructions: dict[int, str],
    part_names: dict[str, str],
    assembly_tree: Any,
) -> list[OperationSpec]:
    """Build executable pose operations directly from pose_atomic_contract.json.

    This is the clean Stage2->pose interface: Stage2 decides the atomic
    fixed/moving contract, while pose only grounds it in rendered geometry.
    """

    operations: list[OperationSpec] = []
    for fallback_index, item in enumerate(atomic_contract.get("operations", [])):
        if not isinstance(item, dict):
            continue
        operation_index = int(item.get("operation_index", fallback_index))
        manual_step_index = int(item.get("manual_step_id", item.get("manual_step_index", 0)))
        moving_part = str(item.get("moving_part"))
        fixed_parts = [str(part_id) for part_id in item.get("fixed_parts", [])]
        target_step_parts = [str(part_id) for part_id in item.get("target_subassembly_after_op", [])]
        if not moving_part or not fixed_parts:
            continue
        if not target_step_parts:
            target_step_parts = _unique_keep_order([*fixed_parts, moving_part])
        raw_instruction = (
            str(item.get("source_stage2_instruction", "")).strip()
            or stage_instructions.get(manual_step_index, "")
        )
        operation_instruction = render_pose_atomic_operation_instruction(
            atomic_contract=atomic_contract,
            atomic_operation=item,
            part_names=part_names,
            assembly_tree=assembly_tree,
            raw_step_instruction=raw_instruction,
        )
        operations.append(
            OperationSpec(
                operation_index=operation_index,
                moving_part=moving_part,
                fixed_parts=fixed_parts,
                manual_step_index=manual_step_index,
                target_step_parts=target_step_parts,
                instruction=raw_instruction,
                operation_instruction=operation_instruction,
            )
        )
    return sorted(operations, key=lambda op: int(op.operation_index))


def render_pose_atomic_operation_instruction(
    *,
    atomic_contract: dict[str, Any],
    atomic_operation: dict[str, Any],
    part_names: dict[str, str],
    assembly_tree: Any,
    raw_step_instruction: str,
) -> str:
    moving_part = str(atomic_operation.get("moving_part"))
    fixed_parts = [str(part_id) for part_id in atomic_operation.get("fixed_parts", [])]
    target_parts = [str(part_id) for part_id in atomic_operation.get("target_subassembly_after_op", [])]
    relation = str(atomic_operation.get("relation", "")).strip()
    slot_hint = str(atomic_operation.get("slot_hint", "")).strip()
    reject_cases = str(atomic_operation.get("reject_cases", "")).strip()
    manual_step_id = int(atomic_operation.get("manual_step_id", 0))
    manual_page = atomic_operation.get("manual_page")
    manual_crop = str(atomic_operation.get("manual_crop", "")).strip()
    tree = assembly_tree if assembly_tree is not None else atomic_contract.get("tree")

    return f"""Atomic execution contract from Stage2.
Current operation: move exactly one part, part {moving_part} ({part_names.get(moving_part, 'unknown')}), and do not move fixed parts.
Manual step id: {manual_step_id}; manual page: {manual_page}; manual crop: {manual_crop or 'use the provided manual image for this step'}.
Assembly tree anchor: {tree}.
Fixed parts before this operation: {_parts_text(fixed_parts, part_names)}.
Target subassembly after this operation/manual step: {_parts_text(target_parts, part_names)}.
Relation to satisfy: {relation or 'use the manual-implied relation between the moving part and fixed subassembly'}.
Slot / side / orientation hint: {slot_hint or 'use the visible side, end, height, mirror direction, and connector slot implied by the manual'}.
Fully seated criterion: contact faces/edges are flush or inserted as far as the manual implies; no visible floating, large gap, wrong side/end, wrong mirror, upside-down orientation, or collision.
Reject cases: {reject_cases or 'moving fixed parts, wrong side/end, wrong mirror, floating, collision, or not fully seated'}.
Manual-step context for grounding only: {_contract_one_line(raw_step_instruction, 420)}
Prompt policy: this atomic contract is the primary pose target. Use the manual crop and current rendered views only to ground geometry, scale, side, and orientation."""


def _contract_one_line(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def parse_stage2_instructions(path: Path) -> dict[int, str]:
    text = path.read_text(encoding="utf-8")
    # Stage2 outputs in different runs are not perfectly uniform.  Older prompts
    # used "### Step 1:", while outputs6/default_anchor commonly uses plain
    # "Step 1:" and some experimental prompts use
    # "Step 1 (SEQUENTIAL_ADDITION):" or "Step 1 — SEQUENTIAL_ADDITION:".
    matches = list(
        re.finditer(
            r"(?:^|\n)\s*(?:#{1,6}\s*)?Step\s*(\d+)"
            r"(?:\s*\([^)\n]*\))?\s*(?::|[-\u2013\u2014]\s*[^:\n]{0,80}:)\s*",
            text,
            flags=re.IGNORECASE,
        )
    )
    if not matches:
        return {0: text.strip()}
    out: dict[int, str] = {}
    for idx, match in enumerate(matches):
        step_num = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        out[step_num - 1] = text[start:end].strip()
    return out


CASE_OPERATION_PLANS: dict[tuple[str, str], list[tuple[str, list[str], int, list[str]]]] = {
    ("Bench", "applaro"): [
        ("2", ["1"], 0, ["0", "1", "2"]),
        ("0", ["1", "2"], 0, ["0", "1", "2"]),
        ("3", ["1", "2", "0"], 1, ["0", "1", "2", "3"]),
    ],
    ("Chair", "applaro_3"): [
        ("4", ["5"], 0, ["0", "2", "3", "4", "5"]),
        ("3", ["5", "4"], 0, ["0", "2", "3", "4", "5"]),
        ("2", ["5", "4", "3"], 0, ["0", "2", "3", "4", "5"]),
        ("0", ["5", "4", "3", "2"], 0, ["0", "2", "3", "4", "5"]),
        ("1", ["5", "4", "3", "2", "0"], 1, ["0", "1", "2", "3", "4", "5"]),
    ],
    ("Chair", "glenn"): [
        ("1", ["0"], 0, ["0", "1"]),
        ("2", ["0", "1"], 1, ["0", "1", "2"]),
        ("3", ["0", "1", "2"], 2, ["0", "1", "2", "3"]),
    ],
    ("Chair", "ingolf"): [
        ("0", ["4"], 0, ["0", "1", "4"]),
        ("1", ["4", "0"], 0, ["0", "1", "4"]),
        ("2", ["4", "0", "1"], 1, ["0", "1", "2", "4"]),
        ("3", ["4", "0", "1", "2"], 2, ["0", "1", "2", "3", "4"]),
    ],
    ("Chair", "jules_2"): [
        ("0", ["2"], 0, ["0", "2"]),
        ("1", ["2", "0"], 1, ["0", "1", "2"]),
    ],
}


def build_operations_for_case(
    category: str,
    name: str,
    stage_instructions: dict[int, str],
    part_names: dict[str, str],
    assembly_tree: Any,
    main_data_steps: list[dict[str, Any]],
) -> list[OperationSpec]:
    atomic_details = _parse_atomic_operation_details(
        "\n".join(stage_instructions.values()),
        set(part_names),
    )
    atomic_details_by_key = {
        (str(item.get("moving_part")), int(item.get("manual_step_index", -1))): item
        for item in atomic_details
    }
    raw_specs = operation_plan_for_case(
        category,
        name,
        main_data_steps,
        stage_instructions=stage_instructions,
        part_names=part_names,
    )
    operations: list[OperationSpec] = []
    for operation_index, (moving_part, fixed_parts, manual_step_index, target_step_parts) in enumerate(raw_specs):
        raw_instruction = stage_instructions.get(manual_step_index, "")
        base_instruction = convert_tree_aware_operation_instruction(
            category=category,
            name=name,
            operation_index=operation_index,
            moving_part=moving_part,
            fixed_parts=fixed_parts,
            target_step_parts=target_step_parts,
            manual_step_index=manual_step_index,
            raw_step_instruction=raw_instruction,
            part_names=part_names,
            assembly_tree=assembly_tree,
            main_data_steps=main_data_steps,
        )
        atomic_detail = atomic_details_by_key.get((str(moving_part), int(manual_step_index)))
        operation_instruction = _atomic_operation_instruction_prefix(
            atomic_detail,
            part_names,
        ) + base_instruction if atomic_detail else base_instruction
        operations.append(
            OperationSpec(
                operation_index=operation_index,
                moving_part=moving_part,
                fixed_parts=fixed_parts,
                manual_step_index=manual_step_index,
                target_step_parts=target_step_parts,
                instruction=raw_instruction,
                operation_instruction=operation_instruction,
            )
        )
    return operations


def build_applaro_operations(
    stage_instructions: dict[int, str],
    part_names: dict[str, str],
    assembly_tree: Any,
    main_data_steps: list[dict[str, Any]],
) -> list[OperationSpec]:
    return build_operations_for_case(
        category="Bench",
        name="applaro",
        stage_instructions=stage_instructions,
        part_names=part_names,
        assembly_tree=assembly_tree,
        main_data_steps=main_data_steps,
    )


def operation_plan_for_case(
    category: str,
    name: str,
    main_data_steps: list[dict[str, Any]],
    stage_instructions: dict[int, str] | None = None,
    part_names: dict[str, str] | None = None,
) -> list[tuple[str, list[str], int, list[str]]]:
    if stage_instructions and part_names:
        inferred = infer_operation_plan_from_stage_instructions(stage_instructions, part_names)
        if inferred:
            return inferred
    explicit = CASE_OPERATION_PLANS.get((category, name))
    if explicit:
        return explicit
    return infer_operation_plan_from_steps(main_data_steps)


def infer_operation_plan_from_stage_instructions(
    stage_instructions: dict[int, str],
    part_names: dict[str, str],
) -> list[tuple[str, list[str], int, list[str]]]:
    """Infer atomic operations from generated Stage2 instruction text.

    This is safer than the older main_data fallback because Stage2 text usually
    says which part is first placed as the local base, e.g. "Place the tabletop
    (0) upside down" before attaching legs.
    """

    known_ids = set(part_names)
    atomic_plan = _infer_operation_plan_from_atomic_operations(
        "\n".join(stage_instructions.values()),
        known_ids,
    )
    if atomic_plan:
        return atomic_plan

    assembled: list[str] = []
    operations: list[tuple[str, list[str], int, list[str]]] = []
    for manual_step_index, text in sorted(stage_instructions.items(), key=lambda item: int(item[0])):
        step_ids = _extract_parts_needed_ids(text, known_ids)
        if not step_ids:
            step_ids = _ids_in_text(text, known_ids)
        step_ids = _unique_keep_order(step_ids)
        if not step_ids:
            continue

        target_parts = _unique_keep_order([*assembled, *step_ids])
        if not assembled:
            anchor = _infer_stage_anchor(text, step_ids, known_ids) or step_ids[0]
            assembled = _unique_keep_order([anchor])

        moving_parts = [part_id for part_id in _ids_in_instruction_order(text, step_ids, known_ids) if part_id not in assembled]
        moving_parts.extend(part_id for part_id in step_ids if part_id not in assembled and part_id not in moving_parts)
        for moving_part in moving_parts:
            fixed_parts = assembled.copy()
            if not fixed_parts:
                continue
            operations.append((moving_part, fixed_parts, int(manual_step_index), target_parts))
            assembled = _unique_keep_order([*assembled, moving_part])
    return operations


def _infer_operation_plan_from_atomic_operations(
    text: str,
    known_ids: set[str],
) -> list[tuple[str, list[str], int, list[str]]]:
    details = _parse_atomic_operation_details(text, known_ids)
    if details:
        return [
            (
                str(item["moving_part"]),
                list(item["fixed_parts"]),
                int(item["manual_step_index"]),
                list(item["target_parts"]),
            )
            for item in details
        ]
    return []


def _parse_atomic_operation_details(
    text: str,
    known_ids: set[str],
) -> list[dict[str, Any]]:
    match = re.search(r"<atomic_operations>(.*?)</atomic_operations>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    block = match.group(1)
    op_matches = list(re.finditer(r"^\s*op\s*(\d+)\s*:\s*$", block, flags=re.IGNORECASE | re.MULTILINE))
    operations: list[dict[str, Any]] = []
    for idx, op_match in enumerate(op_matches):
        start = op_match.end()
        end = op_matches[idx + 1].start() if idx + 1 < len(op_matches) else len(block)
        op_text = block[start:end]
        manual_step = _parse_first_int_field(op_text, "manual_step")
        fixed_parts = _parse_id_list_field(op_text, "fixed_parts", known_ids)
        moving_part = _parse_first_int_field(op_text, "moving_part")
        target_parts = _parse_id_list_field(op_text, "target_subassembly_after_op", known_ids)
        if manual_step is None or moving_part is None:
            continue
        moving = str(moving_part)
        if moving not in known_ids or not fixed_parts:
            continue
        if not target_parts:
            target_parts = _unique_keep_order([*fixed_parts, moving])
        operations.append(
            {
                "op_index": int(op_match.group(1)),
                "manual_step": int(manual_step),
                "manual_step_index": int(manual_step) - 1,
                "fixed_parts": fixed_parts,
                "moving_part": moving,
                "target_parts": target_parts,
                "relation": _parse_text_field(op_text, "relation"),
                "slot_hint": _parse_text_field(op_text, "slot_hint"),
            }
        )
    return operations


def _parse_text_field(text: str, field: str) -> str:
    match = re.search(rf"^\s*{re.escape(field)}\s*:\s*(.*?)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    return " ".join(match.group(1).split()) if match else ""


def _atomic_operation_instruction_prefix(
    atomic_detail: dict[str, Any] | None,
    part_names: dict[str, str],
) -> str:
    if not atomic_detail:
        return ""
    moving = str(atomic_detail.get("moving_part", ""))
    fixed_parts = [str(part_id) for part_id in atomic_detail.get("fixed_parts", [])]
    target_parts = [str(part_id) for part_id in atomic_detail.get("target_parts", [])]
    relation = str(atomic_detail.get("relation", "")).strip()
    slot_hint = str(atomic_detail.get("slot_hint", "")).strip()
    fixed_text = _parts_text(fixed_parts, part_names)
    target_text = _parts_text(target_parts, part_names)
    moving_text = _part_text(moving, part_names)
    return f"""Atomic execution contract from Stage2:
- Move exactly one part: {moving_text}.
- Fixed parts before this operation: {fixed_text}.
- Target subassembly after the related manual step: {target_text}.
- Relation to satisfy: {relation or 'use the manual-implied relation between moving and fixed parts'}.
- Slot/side/orientation hint: {slot_hint or 'use the manual-implied side/end/height/mirror slot'}.
Treat this contract as the primary instruction for pose generation; use the manual crop and current renders only to ground geometry.
For the current numeric pose, move only the moving part relative to the fixed parts listed above. Parts that appear in the target subassembly but are not in the moving/fixed lists are future context for the same manual step; do not place or chase those future parts in this operation.

"""


def _part_text(part_id: str, part_names: dict[str, str]) -> str:
    return f"part {part_id} ({part_names.get(part_id, 'unknown')})"


def _parts_text(part_ids: list[str], part_names: dict[str, str]) -> str:
    return ", ".join(_part_text(str(part_id), part_names) for part_id in part_ids)


def _parse_first_int_field(text: str, field: str) -> int | None:
    match = re.search(rf"^\s*{re.escape(field)}\s*:\s*(\d+)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    return int(match.group(1)) if match else None


def _parse_id_list_field(text: str, field: str, known_ids: set[str]) -> list[str]:
    match = re.search(rf"^\s*{re.escape(field)}\s*:\s*\[([^\]]*)\]", text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return []
    out: list[str] = []
    for value in re.findall(r"\d+", match.group(1)):
        part_id = str(int(value))
        if part_id in known_ids:
            out.append(part_id)
    return _unique_keep_order(out)


def _extract_parts_needed_ids(step_text: str, known_ids: set[str]) -> list[str]:
    match = re.search(
        r"Parts\s+Needed\s*:?(.*?)(?:\n\s*[-*]?\s*(?:Instructions|Instruction)\s*:|\Z)",
        step_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    return _ids_in_text(match.group(1), known_ids)


def _infer_stage_anchor(step_text: str, step_ids: list[str], known_ids: set[str]) -> str | None:
    step_set = set(step_ids)
    for line in _instruction_lines(step_text):
        lower = line.lower()
        if not any(verb in lower for verb in ("place", "lay", "stand", "set", "turn", "position", "orient")):
            continue
        ids = [part_id for part_id in _ids_in_text(line, known_ids) if part_id in step_set]
        if ids:
            return ids[0]
    ids = [part_id for part_id in _ids_in_text(step_text, known_ids) if part_id in step_set]
    return ids[0] if ids else None


def _ids_in_instruction_order(step_text: str, step_ids: list[str], known_ids: set[str]) -> list[str]:
    step_set = set(step_ids)
    ids: list[str] = []
    for line in _instruction_lines(step_text):
        ids.extend(part_id for part_id in _ids_in_text(line, known_ids) if part_id in step_set)
    return _unique_keep_order(ids)


def _instruction_lines(step_text: str) -> list[str]:
    parts = re.split(r"\n\s*[-*]?\s*(?:Instructions|Instruction)\s*:?\s*", step_text, flags=re.IGNORECASE, maxsplit=1)
    body = parts[1] if len(parts) > 1 else step_text
    lines = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[-*]\s*", "", line)
        lines.append(line)
    return lines


def _ids_in_text(text: str, known_ids: set[str]) -> list[str]:
    ids: list[str] = []
    patterns = [
        r"\((\d+)\)",
        r"\bpart\s*\(?(\d+)\)?\b",
        r"\b(?:frame|rail|leg|panel|board|bar|bracket|block|shell|seat|shelf|top|stretcher|connector|slat|rod)\s*\(?(\d+)\)?\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            part_id = str(int(match.group(1)))
            if part_id in known_ids:
                ids.append(part_id)
    return _unique_keep_order(ids)


def infer_operation_plan_from_steps(main_data_steps: list[dict[str, Any]]) -> list[tuple[str, list[str], int, list[str]]]:
    """Best-effort fallback for objects without a hand-audited operation order."""
    assembled: list[str] = []
    operations: list[tuple[str, list[str], int, list[str]]] = []
    for step in sorted(main_data_steps, key=lambda item: int(item.get("step_id", 0))):
        step_id = int(step.get("step_id", 0))
        step_parts: list[str] = []
        for item in step.get("parts", []):
            step_parts.extend(_split_part_group(str(item)))
        step_parts = _unique_keep_order(step_parts)
        if not assembled:
            if not step_parts:
                continue
            assembled.append(step_parts[0])
            moving_parts = step_parts[1:]
        else:
            moving_parts = [part for part in step_parts if part not in assembled]
        for moving_part in moving_parts:
            fixed_parts = assembled.copy()
            target_parts = _unique_keep_order([*assembled, moving_part, *step_parts])
            operations.append((moving_part, fixed_parts, step_id, target_parts))
            assembled = _unique_keep_order([*assembled, moving_part])
    return operations


def _split_part_group(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def legacy_applaro_operation_instruction(operation_index: int) -> str:
    """Older hand-written operation prompts kept only as a fallback reference."""
    return [
        """Current operation: move part 2 (side frame) only.
Fixed subassembly: part 1 (seat panel); do not move part 1.
Relevant manual context: Step 1 also mentions part 0, but part 0 is not the moving object in this operation.
Goal: attach side frame 2 upright to one end of seat panel 1.
Required contact: side frame 2 should meet the end/frame edge of seat panel 1 and form the first rigid panel-frame corner.
Required orientation: side frame 2 stands perpendicular to the panel surface; its inner attachment faces/holes face inward toward the panel/rail ends.
Fully seated criterion: the frame is pushed onto the panel end with no visible floating, large gap, wrong-side placement, or upside-down/mirrored orientation.""",
        """Current operation: move part 0 (support bar / long side rail) only.
Fixed subassembly: parts 1 and 2 are already committed; do not move parts 1 or 2.
Relevant manual context: this is the rail placement part of Step 1 after the first side frame is installed.
Goal: place part 0 horizontally along the bench length between/along the seat panel and side frame.
Required contact: part 0's long inner face aligns with the long edge/support area of seat panel 1, and one rail end seats into the inner connection area of side frame 2.
Required orientation: the rail long axis runs along the bench length, parallel to the long edge of the seat panel, below/along the panel edge rather than standing upright.
Fully seated criterion: the rail is not floating, not perpendicular to the bench length, not shifted away from the side frame connection, and not colliding through the fixed parts.""",
        """Current operation: move part 3 (remaining side frame) only.
Fixed subassembly: parts 1, 2, and 0 are already committed; do not move them.
Relevant manual context: Step 2 closes the open end of the bench.
Goal: attach side frame 3 to the open end opposite side frame 2, completing the rectangular bench frame.
Required contact: side frame 3 connects to the free ends of support bar 0 and the far end/frame edge of seat panel 1.
Required orientation: side frame 3 is upright and mirrored against side frame 2; its inner attachment faces point inward toward the free rail/panel ends.
Fully seated criterion: the frame is slid/pushed onto the open end with no visible floating, wrong-end placement, wrong mirror, upside-down orientation, or large gap.""",
    ][int(operation_index)] if int(operation_index) in {0, 1, 2} else ""


def validate_inputs(inputs: BenchApplaroInputs) -> None:
    required_files = [
        inputs.scene_obj,
        inputs.scene_rot_annotated,
        inputs.product_page,
        inputs.stage1_json,
        inputs.stage2_text,
    ]
    required_files.extend(inputs.manual_step_overlay.values())
    required_files.extend(inputs.manual_step_numbered.values())
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)
    for directory in [inputs.scene_parts_dir, inputs.target_parts_dir]:
        if not directory.exists():
            raise FileNotFoundError(directory)
        if not list(directory.glob("*.obj")):
            raise FileNotFoundError(f"No OBJ files found under {directory}")
