from openai import OpenAI
import json
import os
import re
from datetime import datetime

from config import DATA_DIR, DEFAULT_MODEL, DEFAULT_MAX_TOKENS
from utils import load_prompt, ensure_dir
from api_client import make_client, get_request_timeout, resolve_stage_model, build_create_kwargs, format_usage


def parse_merge_sequence(stage2_text):
    """
    Parse <merge_sequence> and <sub_assemblies> from Stage 2 v3 output.
    Returns a nested list, or None if parsing fails.
    """
    # Extract sub_assemblies block
    sa_map = {}
    sa_match = re.search(r'<sub_assemblies>(.*?)</sub_assemblies>', stage2_text, re.DOTALL)
    if sa_match:
        sa_block = sa_match.group(1)
        for line in sa_block.splitlines():
            m = re.match(r'\s*(SA\d+)\s*=\s*\{([^}]+)\}', line)
            if m:
                sa_name = m.group(1)
                parts = [int(x.strip()) for x in m.group(2).split(',') if x.strip().isdigit()]
                sa_map[sa_name] = parts

    # Extract merge_sequence block
    ms_match = re.search(r'<merge_sequence>(.*?)</merge_sequence>', stage2_text, re.DOTALL)
    if not ms_match:
        return None
    ms_block = ms_match.group(1)

    # Parse each line into (lhs, operands)
    merge_lines = {}
    final_operands = None
    for line in ms_block.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match: LHS = merge( ... ) — comment
        m = re.match(r'(M\d+|FINAL)\s*=\s*merge\s*\(([^)]+)\)', line)
        if not m:
            continue
        lhs = m.group(1)
        raw_ops = [x.strip() for x in m.group(2).split(',') if x.strip()]
        if lhs == 'FINAL':
            final_operands = raw_ops
        else:
            merge_lines[lhs] = raw_ops

    if final_operands is None:
        return None

    def resolve(token):
        """Recursively resolve a token to a nested list or int."""
        token = token.strip()
        # part_X or plain integer
        m = re.match(r'^part_(\d+)$', token)
        if m:
            return int(m.group(1))
        if re.match(r'^\d+$', token):
            return int(token)
        # SA reference
        if token in sa_map:
            parts = sa_map[token]
            return parts if len(parts) > 1 else parts[0]
        # M reference
        if token in merge_lines:
            ops = merge_lines[token]
            resolved = [resolve(op) for op in ops]
            # flatten single-element wrapping
            if len(resolved) == 1:
                return resolved[0]
            return resolved
        return None

    # Build final tree from FINAL operands
    resolved_final = [resolve(op) for op in final_operands]
    resolved_final = [r for r in resolved_final if r is not None]

    if not resolved_final:
        return None
    if len(resolved_final) == 1:
        return resolved_final[0]
    return resolved_final


def parse_tree_projection(stage2_text):
    """Parse <tree_projection> emitted by execution-aware Stage 2 prompts."""
    match = re.search(r"<tree_projection>(.*?)</tree_projection>", stage2_text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    block = match.group(1)
    tree_match = re.search(r"Tree\s*:\s*(\[[\d,\s\[\]]+\])", block, re.IGNORECASE)
    if not tree_match:
        return None
    raw = tree_match.group(1)
    try:
        return json.loads(raw)
    except Exception:
        try:
            import ast
            return ast.literal_eval(raw)
        except Exception:
            return None


def convert_to_tree_rule_based(furniture_name, furniture_type, pdf_path, stage2_text, args):
    """Rule-based Stage 3: parse merge_sequence directly, no LLM call."""
    result = parse_tree_projection(stage2_text)
    if result is None:
        result = parse_merge_sequence(stage2_text)

    if result is None:
        print(f"  [rule-based] parse failed for {furniture_type}/{furniture_name}, falling back to LLM")
        return None

    final_output = str(result)

    if args.debug:
        debug_dir = os.path.join(pdf_path, furniture_type, furniture_name, "debug")
        ensure_dir(debug_dir)
        with open(os.path.join(debug_dir, "convert_rule_based.txt"), "w") as f:
            f.write(final_output)

    output_folder = os.path.join(pdf_path, furniture_type, furniture_name)
    ensure_dir(output_folder)
    with open(os.path.join(output_folder, "tree.json"), "w") as f:
        f.write(json.dumps(final_output))
    print(f"Saved assembly graph (rule-based) to {os.path.join(output_folder, 'tree.json')}")
    return final_output

def convert_to_tree(furniture_name, furniture_type, pdf_path, current_plan, args):
    """Convert furniture assembly plan to a tree structure."""
    rule_based = convert_to_tree_rule_based(furniture_name, furniture_type, pdf_path, current_plan, args)
    if rule_based is not None:
        return rule_based

    prompt_text = getattr(args, "convert_prompt", "tree_ikea_manual")

    client = make_client()

    # Load tree conversion prompt
    prompt = load_prompt(prompt_text)
    full_prompt = prompt + "\n" + current_plan + "\n\nYOUR REAL OUTPUT:\n"

    if args.debug:
        prompt_file = os.path.join(pdf_path, furniture_type, furniture_name, "debug", f"convert_prompt.txt")
        with open(prompt_file, "w") as f:
            f.write(full_prompt)
        print(f"Saved convert prompt to {prompt_file}")

    # Generate tree structure
    stage_model = resolve_stage_model("convert", args.model)
    print(
        f"[Convert] request {furniture_type}/{furniture_name}: "
        f"model={stage_model}, prompt={prompt_text}, input_chars={len(current_plan or '')}, "
        f"prompt_chars={len(full_prompt)}",
        flush=True,
    )
    print(f"[Convert] calling client.chat.completions.create for {furniture_type}/{furniture_name}", flush=True)
    response = client.chat.completions.create(
        model=stage_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt}
                ],
            }
        ],
        **build_create_kwargs(stage_model, DEFAULT_MAX_TOKENS, args.temp),
        timeout=get_request_timeout(),
    )
    content = response.choices[0].message.content or ""
    print(
        f"[Convert] response {furniture_type}/{furniture_name}: "
        f"finish_reason={response.choices[0].finish_reason}, chars={len(content)}, "
        f"{format_usage(response.usage)}",
        flush=True,
    )

    # Extract the nested list from the model response
    import re
    raw = content.replace("```python", "").replace("```", "")
    # Find the last line that looks like a nested list: starts with '[' after stripping
    final_output = ""
    for line in reversed(raw.splitlines()):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            final_output = stripped
            break
    # Fallback: find any [...] pattern in the raw output
    if not final_output:
        matches = re.findall(r'\[[\d,\[\] ]+\]', raw)
        if matches:
            final_output = matches[-1]
    if not final_output:
        final_output = raw.strip()

    # Normalize full-width / CJK punctuation that breaks json.loads downstream
    final_output = (final_output
                    .replace('，', ',')   # ，
                    .replace('、', ',')   # 、
                    .replace('［', '[')   # ［
                    .replace('］', ']')   # ］
                    .replace('「', '[')   # 「
                    .replace('」', ']')   # 」
                    .replace('【', '[')   # 【
                    .replace('】', ']'))  # 】

    # Save tree output
    if args.debug:
        tree_file = os.path.join(pdf_path, furniture_type, furniture_name, "debug", f"convert_start_{args.start}.txt")
        with open(tree_file, "a") as f:
            f.write("\n=================================================================\n")
            f.write(f"{furniture_type} {furniture_name}\n\n")
            f.write(final_output)
        raw_file = os.path.join(pdf_path, furniture_type, furniture_name, "debug", "convert_raw.txt")
        with open(raw_file, "w") as f:
            f.write(raw)
        print(f"Appended final output assembly graph to {tree_file}")

    # Save JSON version
    serialized_tree = json.dumps(final_output)
    output_folder = os.path.join(pdf_path, furniture_type, furniture_name)
    ensure_dir(output_folder)

    with open(os.path.join(output_folder, "tree.json"), "w") as f:
        f.write(serialized_tree)
    print(f"Saved assembly graph to {os.path.join(output_folder, 'tree.json')}")
