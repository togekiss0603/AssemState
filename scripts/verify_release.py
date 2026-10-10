"""Verify the committed release inventory, asset hashes and obvious credential leaks."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '.venv', '__pycache__', '.pytest_cache'}


def public_files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file()
                  and not any(part in SKIP for part in p.relative_to(ROOT).parts[:-1])
                  and p.relative_to(ROOT).parts[0] not in {'outputs', 'data'}
                  and p.name != 'RELEASE_MANIFEST.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='Maintainer only: refresh inventory after intentional edits.')
    args = parser.parse_args()
    rows = []
    problems = []
    for p in public_files():
        rel = p.relative_to(ROOT).as_posix()
        raw = p.read_bytes()
        if p.is_symlink(): problems.append('Symlink: ' + rel)
        if p.name.startswith('.env') and p.name != '.env.example': problems.append('Environment file: ' + rel)
        if re.search(rb'\bsk-[A-Za-z0-9_-]{20,}', raw): problems.append('Potential API credential: ' + rel)
        if p.suffix == '.py' and re.search(rb'/(?:home|mnt|root)/[A-Za-z0-9_]', raw):
            problems.append('Machine-specific path in runtime code: ' + rel)
        rows.append({'path': rel, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    if problems:
        raise SystemExit('\n'.join(problems))
    manifest = ROOT / 'RELEASE_MANIFEST.json'
    if args.refresh:
        manifest.write_text(json.dumps(rows, indent=2) + '\n')
    elif not manifest.is_file() or json.loads(manifest.read_text()) != rows:
        raise SystemExit('Release file inventory or checksums changed. Review edits before --refresh.')
    print(f'Verified {len(rows)} release files; {sum(r["bytes"] for r in rows):,} bytes.')


if __name__ == '__main__':
    main()
