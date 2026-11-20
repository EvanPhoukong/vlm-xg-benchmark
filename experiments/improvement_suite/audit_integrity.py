"""Verify frozen implementations, protected inputs and complete request cohorts."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from vision_trials import ROOT, OUT, REPO


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    code = Path(__file__).resolve().parent
    checks = {}
    for folder, script in [('vision', 'vision_trials.py'), ('pixels', 'pixel_control.py'), ('context', 'context_trial.py'),
                           ('confirmation_bundesliga/inference', 'confirm_frozen.py'),
                           ('confirmation_bundesliga/best_vision', 'confirm_best_vision.py')]:
        protocol = json.loads((OUT/folder/'protocol.json').read_text())
        checks[f'{folder}: frozen source'] = sha(code/script) == protocol['implementation_sha256']
    first = json.loads((OUT.parent/'noise-robustness-v1/protocol.json').read_text())
    checks['noise training: frozen source'] = sha(code/'robust_xg.py') == first['implementation_sha256']
    for path, digest in first['input_sha256'].items():
        checks[f'protected input: {path}'] = sha(path) == digest
    old_integrity = json.loads((ROOT/'rich_vlm_external/integrity.json').read_text())
    for path, digest in old_integrity['sha256'].items():
        checks[f'original study input: {path}'] = sha(path) == digest
    dev_ids = set()
    for name in ['wc2018', 'wc2022', 'euro2024', 'euro2020']:
        dev_ids.update(pd.read_json(ROOT/'data'/f'{name}.jsonl', lines=True).match_id.astype(int))
    fresh = pd.read_json(OUT/'confirmation_bundesliga/selection.jsonl', lines=True)
    checks['confirmation has no earlier match overlap'] = not bool(set(fresh.match_id) & dev_ids)
    checks['confirmation has 200 unique shots'] = len(fresh) == 200 and fresh.shot_id.is_unique
    for relative in ['vision/second_cache.jsonl', 'vision/fewshot_cache.jsonl', 'confirmation_bundesliga/inference/baseline_cache.jsonl']:
        entries = [json.loads(line) for line in (OUT/relative).read_text().splitlines()]
        checks[f'{relative}: complete unique cohort'] = len(entries) == 200 and len({e['shot_id'] for e in entries}) == 200
    incomplete = []
    status_path = OUT/'confirmation_bundesliga/best_vision/partial_status.json'
    if status_path.exists() and not (OUT/'confirmation_bundesliga/best_vision/comparisons.csv').exists():
        status = json.loads(status_path.read_text())
        incomplete.append({'condition': 'Best VLM fresh confirmation', 'completed_shots': status['completed_shots'],
                           'target_shots': status['target_shots'], 'reason': status['reason']})
        checks['incomplete secondary trial is explicitly unscored'] = status['partial_cohort_scored'] is False
    result = {'all_passed': all(checks.values()), 'checks': checks, 'incomplete_conditions': incomplete,
              'note': 'No credentials read or recorded. This checks saved provenance, not whether every planned secondary trial completed or whether models generalize.'}
    (OUT/'integrity.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Integrity checks passed: {sum(checks.values())}/{len(checks)}', flush=True)
    if not result['all_passed']:
        raise SystemExit('A frozen experiment or protected input failed verification.')


if __name__ == '__main__':
    main()
