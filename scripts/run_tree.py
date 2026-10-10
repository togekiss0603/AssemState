"""Portable entry point for the released anchor-v2m tree planner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path(os.getenv('ASSEMSTATE_DATA_ROOT', ROOT / 'examples/data')))
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/tree')
    p.add_argument('--model', default=os.getenv('OPENAI_MODEL', 'gpt-5'))
    group = p.add_mutually_exclusive_group()
    group.add_argument('--names', nargs='+')
    group.add_argument('--all', action='store_true')
    args = p.parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    records = json.loads((data_root / 'main_data.json').read_text())
    available = {r['category'] + '/' + r['name'] for r in records}
    names = sorted(available) if args.all else (args.names or ['Bench/applaro'])
    for name in names:
        if name not in available:
            p.error('Unknown product: ' + name)
        required = [data_root / 'pdfs' / name / 'page_1.png', data_root / 'prescene' / name / 'scene_annotated.png']
        for path in required:
            if not path.is_file():
                p.error('Missing input: ' + str(path))
        if not list((data_root / 'mask' / name).glob('*mask_overlay_cropped_numbered.png')):
            p.error('Missing numbered masks: ' + name)
    env = dict(os.environ, ASSEMSTATE_DATA_ROOT=str(data_root), ASSEMSTATE_OUTPUT_ROOT=str(output))
    cmd = [sys.executable, str(ROOT / 'tree/vlm_planner/inference/run_anchor.py'), '--model', args.model,
           '--prompt_version', 'v2m', '--prompt_type', 'numbered', '--convert_prompt', 'tree_ikea_manual',
           '--output_name', 'anchor_v2m', '--debug', '--names', *names]
    result = subprocess.run(cmd, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)
    missing = [n for n in names if not (output / 'anchor_v2m' / n / 'tree.json').is_file()]
    if missing:
        raise SystemExit('Inference incomplete for: ' + ', '.join(missing))
    for name in names:
        result_path = output / 'anchor_v2m' / name / 'eval_result.json'
        if not result_path.is_file() or json.loads(result_path.read_text()).get('error'):
            raise SystemExit('Invalid or unevaluated tree: ' + name)
    print(f'Completed {len(names)} products; outputs: {output / "anchor_v2m"}')


if __name__ == '__main__':
    main()
