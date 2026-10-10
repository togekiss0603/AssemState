"""Check archived appendix prompts, runtime tree prompts, and website coverage.

Optionally validate snapshots and the complete table inventory against a supplied
manuscript ZIP with --paper-zip. No API access or third-party data is required.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    ('M1', 'part_names_print.txt', 'generate_json.txt', True, {}),
    ('M2', 'part_descriptions_print.txt', 'select_material.txt', True, {}),
    ('M3', 'tree_v2m_print.txt', 'planning_anchor_v2m.txt', True, {'—': '--', '✓': '[OK]'}),
    ('M4', 'tree_conversion_print.txt', 'tree_ikea_manual.txt', True, {'’': "'"}),
    ('Pose initial', 'pose_initial_print.txt', 'pose_initial_prompt.txt', False, {}),
    ('Pose feedback', 'pose_feedback_print.txt', 'pose_feedback_prompt.txt', False, {}),
    ('Pose selector', 'pose_selector_print.txt', 'pose_selector_prompt.txt', False, {'≈': '~', '–': '-'}),
]
TABLE_LABELS = {
    'tab:tree_generation', 'tab:pose_sim_result', 'tab:pose_ablation',
    'tab:manual_stage1_example', 'tab:pose_trace', 'tab:pose_trace_config',
}


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.images = []
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                raise ValueError('Duplicate HTML ID: ' + attrs['id'])
            self.ids.add(attrs['id'])
        if tag == 'table':
            self.tables.append(attrs.get('data-paper-label'))
        if tag == 'img':
            self.images.append(attrs['src'])
        if tag in {'a', 'link', 'script', 'img'}:
            self.links.append(attrs.get('href', attrs.get('src', '')))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-zip', type=Path)
    args = parser.parse_args()
    archive = zipfile.ZipFile(args.paper_zip) if args.paper_zip else None
    report = []
    try:
        for label, printed, raw, runtime, replacements in PROMPTS:
            original = (ROOT / 'artifacts/paper_prompts' / raw).read_bytes()
            appendix = (ROOT / 'artifacts/appendix_print' / printed).read_bytes()
            normalized = original.decode('utf-8')
            for before, after in replacements.items():
                normalized = normalized.replace(before, after)
            require(normalized.encode('utf-8') == appendix, label + ': appendix mismatch')
            if runtime:
                require((ROOT / 'tree/vlm_planner/prompts' / raw).read_bytes() == original,
                        label + ': runtime differs from archived original')
            if archive:
                require(archive.read('appendix_materials/raw/' + raw) == original,
                        label + ': manuscript raw mismatch')
                require(archive.read('latex/prompts/' + printed) == appendix,
                        label + ': manuscript print mismatch')
            report.append({'prompt': label, 'raw_sha256': hashlib.sha256(original).hexdigest(),
                           'appendix_sha256': hashlib.sha256(appendix).hexdigest(),
                           'appendix_relation': 'exact' if original == appendix else 'listed display substitutions only',
                           'runtime': 'exact tree template' if runtime else 'historical snapshot, not current runtime'})
        page = Page()
        page.feed((ROOT / 'website/index.html').read_text())
        require(set(page.tables) == TABLE_LABELS and len(page.tables) == 7,
                'Expected all six paper table labels, with pose/release split into two panels')
        for link in page.links:
            if link.startswith('#'):
                require(link[1:] in page.ids, 'Broken fragment: ' + link)
            elif link and '://' not in link and not link.startswith(('mailto:', 'data:')):
                require((ROOT / 'website' / link).is_file(), 'Missing site asset: ' + link)
        provenance = json.loads((ROOT / 'website/assets/paper/PROVENANCE.json').read_text())
        require(len(provenance['files']) == 7, 'Expected seven active manuscript figures')
        for row in provenance['files']:
            require(row['published_file'] in page.images, 'Figure absent from page: ' + row['published_file'])
            require(hashlib.sha256((ROOT / 'website' / row['published_file']).read_bytes()).hexdigest()
                    == row['published_sha256'], 'Figure checksum changed')
            if archive:
                require(hashlib.sha256(archive.read(row['source'])).hexdigest() == row['source_sha256'],
                        'Source figure checksum changed')
        if archive:
            labels = set()
            for source in ['latex/sec/4_exp.tex', 'latex/sec/appendix.tex']:
                tex = re.sub(r'(?m)(?<!\\)%.*$', '', archive.read(source).decode())
                for block in re.findall(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}', tex, re.S):
                    labels.update(re.findall(r'\\label\{([^}]+)\}', block))
            require(labels == TABLE_LABELS, 'Manuscript table inventory has changed')
            for path in (ROOT / 'artifacts/appendix_print').glob('*.txt'):
                require(path.read_bytes() == archive.read('latex/prompts/' + path.name),
                        'Appendix snapshot mismatch: ' + path.name)
        print(json.dumps({'prompts': report, 'website_table_panels': len(page.tables),
                          'manuscript_figures': len(provenance['files']),
                          'supplied_zip_checked': bool(archive),
                          'scope': 'Prompt bytes and document coverage; not an inference rerun or statistical certification.'}, indent=2))
    finally:
        if archive:
            archive.close()


if __name__ == '__main__':
    main()
