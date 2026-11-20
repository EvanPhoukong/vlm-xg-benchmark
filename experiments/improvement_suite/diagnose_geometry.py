"""Post-evaluation measurement audit; never used to retune a candidate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import robust_xg as robust
from vision_trials import ROOT, OUT


def audit(frame, answers, method):
    rows = []
    for row, answer in zip(frame.itertuples(index=False), answers):
        s, k, d = robust.scene(row)
        p = answer['shooter']
        se = np.nan if p is None else float(np.linalg.norm(np.array([p['lateral'], p['depth']])-s))
        ke = np.nan
        if len(k) == 1 and answer['goalkeeper'] is not None:
            p = answer['goalkeeper']; ke = float(np.linalg.norm(np.array([p['lateral'], p['depth']])-k[0]))
        estimated = np.array([[v['lateral'], v['depth']] for v in answer['defenders']])
        matched_errors, good = [], 0
        if len(d) and len(estimated):
            costs = np.linalg.norm(np.array(d)[:, None, :]-estimated[None, :, :], axis=2)
            a, b = linear_sum_assignment(costs)
            matched_errors = costs[a, b].tolist(); good = sum(v <= 2 for v in matched_errors)
        rows.append({'method': method, 'shot_id': row.shot_id, 'shooter_error': se, 'keeper_error': ke,
                     'shooter_returned': answer['shooter'] is not None, 'keeper_reference_present': len(k) == 1,
                     'shooter_within_2': bool(np.isfinite(se) and se <= 2),
                     'defender_count_error': len(estimated)-len(d), 'matched_defender_error_mean': float(np.mean(matched_errors)) if matched_errors else np.nan,
                     'defenders_reference': len(d), 'defenders_returned': len(estimated), 'defenders_within_2': good})
    detail = pd.DataFrame(rows)
    summary = {'method': method, 'shots': len(frame), 'shooter_coverage': float(detail.shooter_returned.mean()),
               'shooter_mean_error_returned': float(detail.shooter_error.mean()),
               'shooter_within_2_rate_all': float(detail.shooter_within_2.mean()),
               'keeper_mean_error_returned': float(detail.keeper_error.mean()),
               'defender_count_mae': float(detail.defender_count_error.abs().mean()),
               'defender_recall_within_2': float(detail.defenders_within_2.sum()/detail.defenders_reference.sum()),
               'defender_precision_within_2': float(detail.defenders_within_2.sum()/detail.defenders_returned.sum())}
    return detail, summary


def main():
    frame = pd.read_json(ROOT/'external_vlm_selection.jsonl', lines=True, precise_float=True)
    old = {e['shot_id']: e['answer'] for e in [json.loads(line) for line in (ROOT/'rich_vlm_external/cache.jsonl').read_text().splitlines()]}
    methods = {'original': [old[s] for s in frame.shot_id], 'pixels': json.loads((OUT/'pixels/answers.json').read_text())}
    if (OUT/'vision/multiview_answers.json').exists():
        methods['multiview'] = json.loads((OUT/'vision/multiview_answers.json').read_text())
    if (OUT/'vision/fewshot_scores.csv').exists():
        cache = {e['shot_id']: e['answer'] for e in [json.loads(line) for line in (OUT/'vision/fewshot_cache.jsonl').read_text().splitlines()]}
        methods['fewshot'] = [cache[s] for s in frame.shot_id]
    details, summaries = [], []
    for name, answers in methods.items():
        detail, summary = audit(frame, answers, name)
        details.append(detail); summaries.append(summary)
    pd.concat(details, ignore_index=True).to_csv(OUT/'geometry_audit.csv', index=False)
    pd.DataFrame(summaries).to_csv(OUT/'geometry_summary.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
