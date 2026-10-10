import argparse
import ast
import copy
import json
import os
import time
import openai
from tqdm import tqdm

from config import DATA_DIR, MANUAL_DATA_PATH, OUTPUT_DIR
from utils import load_json, ensure_dir
from stage1_associate import select_materials_for_planning
from stage2_planning_anchor import create_plan
from convert import convert_to_tree

RETRYABLE_OPENAI_ERRORS = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

EVAL_30_NAMES = [
    "Bench/applaro", "Bench/applaro_2", "Bench/applaro_3", "Bench/hemnes", "Bench/hemnes_2",
    "Bench/silveran", "Bench/sjalland", "Bench/tjusig",
    "Chair/agam", "Chair/applaro", "Chair/applaro_2", "Chair/applaro_3", "Chair/bekvam_3017",
    "Chair/bernhard", "Chair/borje", "Chair/dalfred", "Chair/ekenas", "Chair/falholmen",
    "Chair/fanbyn", "Chair/froset", "Chair/glenn", "Chair/grubban", "Chair/herman",
    "Chair/ingolf", "Chair/ingolf_2", "Chair/ivar", "Chair/jokkmokk", "Chair/jules_1",
    "Chair/jules_2", "Chair/kaustby"
]


def are_nested_lists_equal(list1, list2):
    list1_copy = copy.deepcopy(list1)
    list2_copy = copy.deepcopy(list2)

    if not isinstance(list1_copy, list) or not isinstance(list2_copy, list):
        return list1_copy == list2_copy

    if len(list1_copy) != len(list2_copy):
        return False

    for item1 in list1_copy:
        found_match = False
        for item2 in list2_copy:
            if are_nested_lists_equal(item1, item2):
                found_match = True
                list2_copy.remove(item2)
                break
        if not found_match:
            return False

    return True


def find_position(tree, target, path=None):
    if path is None:
        path = []
    if isinstance(tree, list):
        for i, item in enumerate(tree):
            new_path = path + [i]
            if item == target:
                return new_path
            if isinstance(item, list):
                result = find_position(item, target, new_path)
                if result is not None:
                    return result
    return None


def enumerate_symmetric_trees(tree, equiv_parts):
    from itertools import permutations

    holder = []

    def visit(current_tree, remaining_equiv):
        if not remaining_equiv:
            return
        parts = remaining_equiv[0]
        pointer_paths = [find_position(current_tree, part) for part in parts]

        for perm in permutations(parts):
            if any(path is None for path in pointer_paths):
                continue
            candidate = copy.deepcopy(current_tree)
            for path, value in zip(pointer_paths, perm):
                node = candidate
                for idx in path[:-1]:
                    node = node[idx]
                node[path[-1]] = value
            holder.append(copy.deepcopy(candidate))
            visit(candidate, remaining_equiv[1:])

    visit(copy.deepcopy(tree), equiv_parts)
    return holder


def load_predicted_tree(tree_path):
    with open(tree_path) as f:
        tree_str = json.load(f)
    tree_str_fixed = tree_str.replace("None", "null") if isinstance(tree_str, str) else tree_str
    return json.loads(tree_str_fixed) if isinstance(tree_str_fixed, str) else tree_str_fixed


def evaluate_tree_output(item, output_path):
    cat, name = item["category"], item["name"]
    key = f"{cat}/{name}"
    output_folder = os.path.join(output_path, cat, name)
    tree_path = os.path.join(output_folder, "tree.json")
    result_path = os.path.join(output_folder, "eval_result.json")

    result = {
        "sample": key,
        "tree_path": tree_path,
        "success": False,
        "symmetry_checked": False,
        "error": None,
        "predicted_tree": None,
        "ground_truth_tree": item.get("assembly_tree"),
    }

    try:
        predicted_tree = load_predicted_tree(tree_path)
        result["predicted_tree"] = predicted_tree
    except Exception as e:
        result["error"] = f"parse_error: {e}"
    else:
        gt_tree = item["assembly_tree"]
        success = are_nested_lists_equal(predicted_tree, gt_tree)
        if not success and item.get("parts_ct", 0) < 9:
            equiv_path = os.path.join(DATA_DIR, "preassembly_scenes", cat, name, "equiv_parts.txt")
            if os.path.exists(equiv_path):
                try:
                    with open(equiv_path) as f:
                        equiv_parts = ast.literal_eval(f.read())
                    result["symmetry_checked"] = True
                    success = any(
                        are_nested_lists_equal(candidate, gt_tree)
                        for candidate in enumerate_symmetric_trees(predicted_tree, equiv_parts)
                    )
                except Exception as e:
                    result["error"] = f"symmetry_eval_error: {e}"
        result["success"] = success

    ensure_dir(output_folder)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "SUCCESS" if result["success"] else "FAIL"
    extra = f" ({result['error']})" if result["error"] else ""
    print(f"[Eval] {status}: {key} -> {result_path}{extra}", flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       type=str,   default=os.getenv("OPENAI_MODEL", "gpt-5"))
    p.add_argument("--temp",        type=float, default=0.0)
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--prompt_type", type=str, choices=["numbered"], default="numbered")
    p.add_argument("--scene_type",  type=str,   default="original")
    p.add_argument("--convert_prompt", type=str, choices=["tree_ikea_manual"], default="tree_ikea_manual",
                   help="Convert prompt file name (without .txt)")
    p.add_argument("--prompt_version", type=str, choices=["v2m"], default="v2m",
                   help="Released anchor/subassembly prompt version")
    p.add_argument("--output_name", type=str,   default="default_anchor",
                   help="Output directory name under outputs/")
    p.add_argument("--names", type=str, nargs="+", default=None,
                   help="Subset of names to run (e.g. Bench/hemnes Chair/agam). Defaults to all 30.")
    p.add_argument("--all", action="store_true",
                   help="Run all items from main_data.json instead of the default 30-item evaluation subset.")
    return p.parse_args()

def main():
    args = parse_args()
    args.start = 0

    data = load_json(MANUAL_DATA_PATH)
    output_path = os.path.join(OUTPUT_DIR, args.output_name)
    data_dict = {f"{item['category']}/{item['name']}": item for item in data}

    print(f"Output dir : {output_path}")
    print(f"Model      : {args.model}")

    if args.names:
        names_to_run = args.names
    elif args.all:
        names_to_run = [f"{item['category']}/{item['name']}" for item in data]
    else:
        names_to_run = EVAL_30_NAMES
    print(f"Items      : {len(names_to_run)}")

    for key in tqdm(names_to_run, desc="Generating (Anchor) Assembly Graphs"):
        if key not in data_dict:
            print(f"WARNING: {key} not found in data")
            continue
        item = data_dict[key]
        name, cat = item["name"], item["category"]
        print(f"\n[Run] Start {cat}/{name}", flush=True)

        tree_path = os.path.join(output_path, cat, name, "tree.json")
        if os.path.exists(tree_path):
            print(f"[Run] Skip existing tree: {tree_path}", flush=True)
            evaluate_tree_output(item, output_path)
            continue

        stage2_path = os.path.join(output_path, cat, name, "debug", "stage2_output.txt")
        if os.path.exists(stage2_path):
            with open(stage2_path) as f:
                stage2_output = f.read()
            print(f"[Run] Reusing stage2 output: {stage2_path} ({len(stage2_output)} chars)", flush=True)
        else:
            for attempt in range(5):
                try:
                    print(f"[Run] Stage1 begin: {cat}/{name} attempt {attempt + 1}/5", flush=True)
                    stage1_output = select_materials_for_planning(name, cat, output_path, args)
                    print(f"[Run] Stage1 done: {cat}/{name} ({len(stage1_output or '')} chars)", flush=True)
                    print(f"[Run] Stage2 begin: {cat}/{name} attempt {attempt + 1}/5", flush=True)
                    stage2_output = create_plan(name, cat, output_path, stage1_output, args)
                    print(f"[Run] Stage2 done: {cat}/{name} ({len(stage2_output or '')} chars)", flush=True)
                    break
                except RETRYABLE_OPENAI_ERRORS as e:
                    wait = 30 * (attempt + 1)
                    print(
                        f"  stage1/2 error ({e.__class__.__name__}), "
                        f"retry {attempt + 1}/5 in {wait}s",
                        flush=True,
                    )
                    time.sleep(wait)
            else:
                print(f"  FAILED stage1/2 after 5 retries, skip: {cat}/{name}", flush=True)
                continue

        for attempt in range(5):
            try:
                print(f"[Run] Convert begin: {cat}/{name} attempt {attempt + 1}/5", flush=True)
                convert_to_tree(name, cat, output_path, stage2_output, args)
                print(f"[Run] Convert done: {cat}/{name}", flush=True)
                break
            except RETRYABLE_OPENAI_ERRORS as e:
                wait = 30 * (attempt + 1)
                print(f"  convert_to_tree error ({e.__class__.__name__}), retry {attempt+1}/5 in {wait}s")
                time.sleep(wait)
        else:
            print(f"  FAILED convert_to_tree after 5 retries: {cat}/{name}")
            continue

        evaluate_tree_output(item, output_path)
        print(f"Done: {cat}/{name}")

if __name__ == "__main__":
    main()
