import os

from config import DATA_DIR, DEFAULT_MAX_TOKENS, SCENE_DIR
from utils import encode_image, load_prompt, ensure_dir, alphanumeric_sort_key
from api_client import make_client, get_request_timeout, resolve_stage_model, build_create_kwargs, format_usage


def create_plan(furniture_name, furniture_type, pdf_path, output_table, args):
    """Anchor + Part Visibility Table Stage 2 (no few-shot, no multi-view)."""
    prompt_version = getattr(args, "prompt_version", "v1")
    if args.prompt_type != "numbered":
        prompt_text = "planning_no_seg_no_num"
    elif prompt_version == "v2":
        prompt_text = "planning_anchor_v2"
    elif prompt_version == "v3":
        prompt_text = "planning_anchor_v3"
    elif prompt_version == "v4":
        prompt_text = "planning_anchor_v4"
    elif prompt_version == "v5a":
        prompt_text = "planning_anchor_v5a"
    elif prompt_version == "v6a":
        prompt_text = "planning_anchor_v5a"  # same Stage 2 as v5a — LAST part already in output
    elif prompt_version == "v7":
        prompt_text = "planning_anchor_v7"
    elif prompt_version == "v8":
        prompt_text = "planning_anchor_v8"
    elif prompt_version == "v2m":
        prompt_text = "planning_anchor_v2m"
    elif prompt_version == "v2m_exec":
        prompt_text = "planning_anchor_v2m_exec"
    else:
        prompt_text = "planning_anchor"
    scene_type_file = "scene_annotated.png" if args.scene_type == "original" else "scene_rot_annotated.png"

    image_path = os.path.join(SCENE_DIR, furniture_type, furniture_name, scene_type_file)
    base64_image = encode_image(image_path)

    client = make_client()
    stage_model = resolve_stage_model("stage2", args.model)
    prompt = load_prompt(prompt_text)
    full_prompt = prompt + "\n\nHere is the JSON parts table:\n" + output_table

    image_messages = [{
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
    }]
    input_paths = [image_path]

    # v2m variants: inject mask step images as hard grouping signal.
    use_mask = (prompt_version in {"v2m", "v2m_exec"})
    mask_dir = os.path.join(DATA_DIR, "mask", furniture_type, furniture_name)
    mask_count = 0
    pdf_count = 0
    if use_mask and os.path.isdir(mask_dir):
        for fn in sorted(os.listdir(mask_dir), key=alphanumeric_sort_key):
            if fn.endswith("mask_overlay_cropped_numbered.png"):
                mask_count += 1
                mask_path = os.path.join(mask_dir, fn)
                input_paths.append(mask_path)
                image_messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(mask_path)}", "detail": "high"}
                })
    else:
        # original behaviour: PDF pages (no mask)
        pdf_dir = os.path.join(DATA_DIR, "pdfs", furniture_type, furniture_name)
        if os.path.isdir(pdf_dir):
            for fn in sorted(os.listdir(pdf_dir), key=alphanumeric_sort_key):
                if fn.endswith(".png"):
                    pdf_count += 1
                    pdf_path = os.path.join(pdf_dir, fn)
                    input_paths.append(pdf_path)
                    image_messages.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(pdf_path)}", "detail": "high"}
                    })

    print(
        f"[Stage2] request {furniture_type}/{furniture_name}: "
        f"model={stage_model}, prompt_version={prompt_version}, prompt={prompt_text}, "
        f"images={len(image_messages)}, masks={mask_count}, pdfs={pdf_count}, "
        f"prompt_chars={len(full_prompt)}",
        flush=True,
    )
    for path in input_paths:
        print(f"[Stage2] input_path {path}", flush=True)
    print(f"[Stage2] calling client.chat.completions.create for {furniture_type}/{furniture_name}", flush=True)
    response = client.chat.completions.create(
        model=stage_model,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": full_prompt}, *image_messages]
        }],
        **build_create_kwargs(stage_model, DEFAULT_MAX_TOKENS, args.temp),
        timeout=get_request_timeout(),
    )

    result = response.choices[0].message.content or ""
    print(
        f"[Stage2] response {furniture_type}/{furniture_name}: "
        f"finish_reason={response.choices[0].finish_reason}, chars={len(result)}, "
        f"{format_usage(response.usage)}",
        flush=True,
    )

    if args.debug:
        output_folder = os.path.join(pdf_path, furniture_type, furniture_name, "debug")
        ensure_dir(output_folder)
        with open(os.path.join(output_folder, "stage2_prompt.txt"), "w") as f:
            f.write(full_prompt)
        with open(os.path.join(output_folder, "stage2_output.txt"), "w") as f:
            f.write(result)
        print(f"Saved stage2 output to {os.path.join(output_folder, 'stage2_output.txt')}")

    return result
