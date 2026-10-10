from openai import OpenAI
import os
import json
from datetime import datetime
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR, SCENE_DIR, MULTIVIEW_SCENE_DIR, DEFAULT_MODEL, DEFAULT_MAX_TOKENS
from utils import encode_image, load_prompt, ensure_dir, alphanumeric_sort_key, save_json
from api_client import make_client, get_request_timeout, resolve_stage_model, build_create_kwargs, format_usage

MULTIVIEW_SUFFIXES = [
    "rot",
    "front_right",
    "front_left",
    "back_left",
    "back_right",
]

def _load_multiview_images(furniture_type, furniture_name, prefix="scene"):
    """Return base64-encoded annotated images for all available multi-view angles."""
    image_list = []
    for suffix in MULTIVIEW_SUFFIXES:
        path = os.path.join(
            MULTIVIEW_SCENE_DIR, furniture_type, furniture_name,
            f"{prefix}_{suffix}_annotated.png"
        )
        if os.path.exists(path):
            image_list.append((path, encode_image(path)))
    return image_list

def generate_json(img_path, manual_path, output_path, model_type=DEFAULT_MODEL, temperature=0, debug=False,
                  furniture_type=None, furniture_name=None):
    """Generate JSON label data from images using the specified model."""
    base64_image = encode_image(img_path)
    base64_image2 = encode_image(manual_path)

    # Build scene image content: single view + optional multi-view
    scene_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"},
        }
    ]
    input_paths = [img_path, manual_path]
    if furniture_type and furniture_name:
        for path, b64 in _load_multiview_images(furniture_type, furniture_name):
            input_paths.append(path)
            scene_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
            })

    prompt = load_prompt("generate_json")

    client = make_client()
    stage_model = resolve_stage_model("stage1", model_type)
    print(
        f"[Stage1A] request {furniture_type}/{furniture_name}: "
        f"model={stage_model}, images={len(scene_content) + 1}, "
        f"scene={img_path}, manual={manual_path}",
        flush=True,
    )
    for path in input_paths:
        print(f"[Stage1A] input_path {path}", flush=True)
    print(f"[Stage1A] calling client.chat.completions.create for {furniture_type}/{furniture_name}", flush=True)
    response = client.chat.completions.create(
        model=stage_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *scene_content,
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image2}",
                            "detail": "high"
                        }
                    },
                ],
            }
        ],
        **build_create_kwargs(stage_model, DEFAULT_MAX_TOKENS, temperature),
        timeout=get_request_timeout(),
    )
    content = response.choices[0].message.content or ""
    print(
        f"[Stage1A] response {furniture_type}/{furniture_name}: "
        f"finish_reason={response.choices[0].finish_reason}, chars={len(content)}, "
        f"{format_usage(response.usage)}",
        flush=True,
    )
    final_output = content.replace("```json", "").replace("```", "")

    # Save the output
    if debug:
        with open(os.path.join(output_path, "stage1_prompt_partial.txt"), "w") as f:
            f.write(prompt)
        print(f"Saved stage1 input prompt A to {os.path.join(output_path, 'stage1_prompt_partial.txt')}")
        with open(os.path.join(output_path, "stage1_output_partial.json"), "w") as f:
            f.write(final_output)
        print(f"Saved stage1 output A to {os.path.join(output_path, 'stage1_output_partial.json')}")
    
    return final_output

def select_materials_for_planning(furniture_name, furniture_type, pdf_path, 
                                  args):
    """Select materials for planning based on furniture specifications."""
    if args.scene_type == "original":
        scene_type = "scene_annotated.png"
    else:
        scene_type = "scene_rot_annotated.png"
    scene_path = os.path.join(SCENE_DIR, furniture_type, furniture_name, scene_type)

    manual_path = os.path.join(DATA_DIR, "pdfs", furniture_type, furniture_name, "page_1.png")
    
    output_folder = os.path.join(pdf_path, furniture_type, furniture_name, "debug")
    ensure_dir(output_folder)
    
    
    raw_table = generate_json(scene_path, manual_path, output_folder, args.model, args.temp, args.debug,
                              furniture_type=furniture_type, furniture_name=furniture_name)
    
    prompt_text = "select_material"
    
    # Get manual pages
    b64_pages = []
    manual_paths = []
    manual_dir = os.path.join(DATA_DIR, "pdfs", furniture_type, furniture_name)
    for page in sorted(os.listdir(manual_dir), key=alphanumeric_sort_key):
        if page.endswith(".png"):
            page_path = os.path.join(manual_dir, page)
            manual_paths.append(page_path)
            b64_pages.append(encode_image(page_path))

    base64_image = encode_image(scene_path)
    
    prompt = load_prompt(prompt_text)
    full_prompt = prompt + "\n\nAnd here is the json file: \n" + raw_table
    
    # Build image messages: single-view scene + multi-view scenes + manual pages
    image_messages = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            },
        }
    ]
    input_paths = [scene_path]
    multiview_items = _load_multiview_images(furniture_type, furniture_name)
    for path, b64 in multiview_items:
        input_paths.append(path)
        image_messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
        })

    # Add each page image to the message
    for page_path, b64_image in zip(manual_paths, b64_pages):
        input_paths.append(page_path)
        image_messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_image}",
                "detail": "high"
            },
        })
    
    client = make_client()
    stage_model = resolve_stage_model("stage1", args.model)
    print(
        f"[Stage1B] request {furniture_type}/{furniture_name}: "
        f"model={stage_model}, images={len(image_messages)}, prompt_chars={len(full_prompt)}",
        flush=True,
    )
    for path in input_paths:
        print(f"[Stage1B] input_path {path}", flush=True)
    print(f"[Stage1B] calling client.chat.completions.create for {furniture_type}/{furniture_name}", flush=True)
    response = client.chat.completions.create(
        model=stage_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    *image_messages
                ],
            }
        ],
        **build_create_kwargs(stage_model, DEFAULT_MAX_TOKENS, args.temp),
        timeout=get_request_timeout(),
    )
    content = response.choices[0].message.content or ""
    print(
        f"[Stage1B] response {furniture_type}/{furniture_name}: "
        f"finish_reason={response.choices[0].finish_reason}, chars={len(content)}, "
        f"{format_usage(response.usage)}",
        flush=True,
    )

    if args.debug:
        with open(os.path.join(output_folder, "stage1_prompt.txt"), "w") as f:
            f.write(full_prompt)
        print(f"Saved stage1 input prompt B to {os.path.join(output_folder, 'stage1_prompt.txt')}")
        with open(os.path.join(output_folder, "stage1_output.json"), "w") as f:
            f.write(content.replace("```json", "").replace("```", ""))
        print(f"Saved stage1 output B to {os.path.join(output_folder, 'stage1_output.json')}")

    return content
