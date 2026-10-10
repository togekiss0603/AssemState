"""Check release files for accidental credentials and machine-specific paths."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '.venv', '__pycache__', '.pytest_cache'}


def public_files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file()
                  and not any(part in SKIP for part in p.relative_to(ROOT).parts[:-1])
                  and p.relative_to(ROOT).parts[0] not in {'outputs', 'data'})


def main():
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
        rows.append({'path': rel, 'bytes': len(raw)})
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'Checked {len(rows)} release files; {sum(r["bytes"] for r in rows):,} bytes.')


if __name__ == '__main__':
    main()
