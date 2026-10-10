"""Reaggregate archived operation records; no API calls or physics simulation."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
RUNS = ['gpt_text', 'gpt_structured', 'gpt_selfcheck', 'haiku_text',
        'haiku_structured', 'haiku_selfcheck', 'gpt_multiview']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(row):
    return row['case'], row['operation_index'], str(row['moving_part'])


def audit_run(root, label):
    pose = json.loads((root / label / 'pose.json').read_text())
    release = json.loads((root / label / 'release.json').read_text())
    p = {key(r): r for r in pose['operations']}
    s = {key(r): r for r in release['operations']}
    require(len(p) == len(pose['operations']) == 243, label + ': pose operation count')
    require(len(s) == len(release['operations']) == 243, label + ': release operation count')
    require(p.keys() == s.keys(), label + ': pose/release keys differ')
    require(len({k[0] for k in p}) == 55, label + ': product count')
    t = release['thresholds']
    for k, r in p.items():
        b = s[k]
        require(sorted(map(str, r['fixed_parts'])) == sorted(map(str, b['fixed_parts'])), label + ': fixed-set mismatch')
        require(math.isclose(r['CD'], r['CD_pred_to_gt'] + r['CD_gt_to_pred'], rel_tol=1e-5, abs_tol=1e-8), label + ': CD components')
        require(r['PA'] == float(r['CD'] < pose['thresholds']['pa_threshold']), label + ': PA threshold')
        stable = (b['drift'] < t['drift_threshold'] and b['z_drop'] < t['z_drop_threshold']
                  and b['rot_drift_deg'] < t['rot_threshold_deg'])
        require(bool(b['release_stable']) == stable, label + ': release threshold')
    accepted = sum(bool(r['accepted']) for r in p.values())
    stable = sum(bool(r['release_stable']) for r in s.values())
    require(accepted == pose['aggregate']['accepted_ops'], label + ': acceptance aggregate')
    require(stable == release['aggregate']['num_stable'], label + ': stability aggregate')
    for field, agg in [('CD', 'Mean_CD'), ('RMSE_t', 'Mean_RMSE_t'), ('GD_deg', 'Mean_GD_deg'), ('iterations_used', 'avg_iter')]:
        require(math.isclose(mean(r[field] for r in p.values()), pose['aggregate'][agg], rel_tol=1e-5, abs_tol=1e-8), label + ': ' + agg)
    for field, agg in [('drift', 'mean_drift'), ('z_drop', 'mean_z_drop'), ('rot_drift_deg', 'mean_rot_drift_deg')]:
        require(math.isclose(mean(r[field] for r in s.values()), release['aggregate'][agg], rel_tol=1e-5, abs_tol=1e-8), label + ': ' + agg)
    return dict(run=label, products=55, operations=len(p), accepted=accepted,
                judge_percent=100 * accepted / len(p), RMSE_t=mean(r['RMSE_t'] for r in p.values()),
                GD_deg=mean(r['GD_deg'] for r in p.values()), CD=mean(r['CD'] for r in p.values()),
                median_CD=median(r['CD'] for r in p.values()), PA=mean(r['PA'] for r in p.values()),
                stable=stable, stable_percent=100 * stable / len(s),
                drift=mean(r['drift'] for r in s.values()), z_drop=mean(r['z_drop'] for r in s.values())), p


def reproduce(root):
    summaries, rows = [], {}
    for label in RUNS:
        summary, rows[label] = audit_run(root, label)
        summaries.append(summary)
    reference = {k[0] for k in rows[RUNS[0]]}
    require(all({k[0] for k in r} == reference for r in rows.values()), 'product sets differ')
    matched = {}
    for a, b in [('gpt_structured', 'gpt_selfcheck'), ('gpt_multiview', 'gpt_selfcheck'), ('haiku_structured', 'haiku_selfcheck')]:
        keys = lambda label: {k + (tuple(sorted(map(str, r['fixed_parts']))),) for k, r in rows[label].items()}
        matched[a + '__' + b] = len(keys(a) & keys(b))
    return {'provenance': 'Archived records, not fresh inference or a fresh simulator run.',
            'runs': summaries, 'context_matched_operations': matched}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts/pose')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/tables')
    args = p.parse_args()
    report = reproduce(args.artifacts)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(report['runs'][0]))
        writer.writeheader()
        writer.writerows(report['runs'])
    lines = ['# Archived pose and release results', '', report['provenance'], '',
             '| Setting | Judge | RMSE t | GD deg | CD | Median CD | Stable | Drift | Z drop |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['runs']:
        lines.append(f"| {r['run']} | {r['accepted']}/243 | {r['RMSE_t']:.4f} | {r['GD_deg']:.2f} | {r['CD']:.4f} | {r['median_CD']:.4f} | {r['stable']}/243 | {r['drift']:.4f} | {r['z_drop']:.4f} |")
    lines += ['', 'PA@0.01 = 0 for every archived setting.', '',
              'Context matches (case, index, moving part, fixed set): ' + json.dumps(report['context_matched_operations']),
              'GPT structured vs. self-check is not a fully matched ablation.']
    (args.output / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'Validated {len(report["runs"]) * 243} pose/release record pairs. Output: {args.output}')


if __name__ == '__main__':
    main()
