"""Copy the runtime subset of an existing assemblydata dataset, with checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import io
import tarfile


def selected_files(source, records):
    files = {Path('main_data.json')}
    for r in records:
        case = Path(r['category']) / r['name']
        if case.is_absolute() or '..' in case.parts or len(case.parts) != 2:
            raise ValueError('Invalid product path: ' + str(case))
        for folder in ['prescene', 'parts', 'pdfs', 'mask']:
            root = source / folder / case
            for path in root.rglob('*'):
                if not path.is_file():
                    continue
                rel = path.relative_to(root)
                keep = ((folder == 'parts' and path.suffix in ['.obj', '.mtl'])
                        or (folder == 'pdfs' and path.suffix == '.png')
                        or (folder == 'mask' and ('mask_overlay.png' in path.name or 'mask_overlay_cropped_numbered.png' in path.name))
                        or (folder == 'prescene' and (path.name in ['scene.obj', 'scene.mtl', 'equiv_parts.txt']
                            or path.name.endswith('_annotated.png') or (rel.parts[0] == 'scene_parts' and path.suffix == '.obj'))))
                if keep:
                    files.add(path.relative_to(source))
        equiv = Path('preassembly_scenes') / case / 'equiv_parts.txt'
        if (source / equiv).is_file():
            files.add(equiv)
    return sorted(files)


def check(source, records):
    missing = []
    coverage_warnings = []
    for r in records:
        case = Path(r['category']) / r['name']
        if case.is_absolute() or '..' in case.parts or len(case.parts) != 2:
            raise ValueError('Invalid product path: ' + str(case))
        for rel in [Path('prescene') / case / 'scene_annotated.png', Path('pdfs') / case / 'page_1.png']:
            if not (source / rel).is_file(): missing.append(str(rel))
        masks = list((source / 'mask' / case).glob('*mask_overlay_cropped_numbered.png'))
        if not masks:
            missing.append('mask/' + str(case) + ' (no numbered step masks)')
        elif len(masks) < len(r.get('steps', [])):
            coverage_warnings.append(f'{case}: {len(masks)} numbered masks for {len(r["steps"])} metadata steps')
    if missing:
        raise ValueError('Incomplete tree input dataset:\n' + '\n'.join(missing))
    return coverage_warnings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path)
    p.add_argument('--archive', type=Path, help='Create a separate dataset release asset, instead of copying a directory.')
    p.add_argument('--check', action='store_true')
    p.add_argument('--strict-step-coverage', action='store_true', help='Fail if any metadata step lacks a numbered mask.')
    args = p.parse_args()
    source = args.source.resolve()
    records = json.loads((source / 'main_data.json').read_text())
    warnings = check(source, records)
    for warning in warnings:
        print('MASK COVERAGE WARNING: ' + warning)
    if warnings and args.strict_step_coverage:
        raise SystemExit('Strict step coverage failed; no masks were fabricated.')
    files = selected_files(source, records)
    if args.check:
        print(f'Validated tree inputs: {len(records)} products; runtime assets {sum((source/f).stat().st_size for f in files):,} bytes.')
        return
    if args.archive:
        if args.destination:
            p.error('--archive and --destination are mutually exclusive')
        archive = args.archive.resolve()
        if archive.exists():
            p.error('Archive already exists; choose a new path.')
        if source in archive.parents:
            p.error('Archive must be outside source.')
        archive.parent.mkdir(parents=True, exist_ok=True)
        manifest = []
        with archive.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1, dereference=True) as tar:
            for rel in files:
                src = source / rel
                manifest.append({'path': rel.as_posix(), 'bytes': src.stat().st_size, 'sha256': hashlib.sha256(src.read_bytes()).hexdigest()})
                tar.add(src, arcname='assemblydata/' + rel.as_posix(), recursive=False)
            raw = (json.dumps({'files': manifest, 'mask_coverage_warnings': warnings}, indent=2) + '\n').encode()
            info = tarfile.TarInfo('assemblydata/MANIFEST.json')
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        archive.with_name(archive.name + '.sha256').write_text(digest.hexdigest() + '  ' + archive.name + '\n')
        print(f'Packed {len(files)} runtime files for {len(records)} products: {archive}')
        return
    if args.destination is None:
        p.error('--destination is required unless --check is used')
    dest = args.destination.resolve()
    if dest.exists():
        p.error('Destination already exists; use a new directory to preserve existing data.')
    if source == dest or source in dest.parents:
        p.error('Destination must be outside source.')
    dest.mkdir(parents=True)
    manifest = []
    for rel in files:
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, dst)
        manifest.append({'path': rel.as_posix(), 'bytes': dst.stat().st_size, 'sha256': hashlib.sha256(dst.read_bytes()).hexdigest()})
    (dest / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Copied {len(files)} files for {len(records)} products to {dest}.')


if __name__ == '__main__':
    main()
