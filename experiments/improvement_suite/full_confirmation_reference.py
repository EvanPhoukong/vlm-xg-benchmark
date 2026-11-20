"""Secondary full-pool coordinate controls, after VLM selection was frozen."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import spatial_models as spatial
import robust_xg as robust
from vision_trials import ROOT, OUT, freeze, sha
from context_trial import metadata_for, join_features


def main():
    folder = OUT/'confirmation_bundesliga'
    selected_protocol = folder/'best_vision/protocol.json'
    if not selected_protocol.exists():
        raise ValueError('Select and freeze the VLM candidate before full-pool outcome analysis.')
    paths = sorted((folder/'match_shots').glob('*.json'))
    protocol = {'implementation_sha256': sha(__file__), 'source_sha256': {p.name: sha(p) for p in paths},
                'selected_vlm_protocol_sha256': sha(selected_protocol),
                'purpose': 'Secondary known-coordinate controls across the full 34-match source pool; contains the 200-shot VLM subset, so is not another independent cohort.',
                'no_fitting_or_selection': True}
    freeze(folder/'full_pool_reference', protocol)
    frame = pd.DataFrame([row for p in paths for row in json.loads(p.read_text())['shots']])
    if not frame.shot_id.is_unique:
        raise ValueError('Duplicate full-pool shot IDs.')
    x = spatial.known_features(frame)
    heads = joblib.load(OUT.parent/'noise-robustness-v1/frozen_models.joblib')
    originals = joblib.load(ROOT/'frozen_probability_models.joblib')
    result = frame[['shot_id', 'match_id', 'goal', 'sb_xg']].copy()
    result['rich_known_positions'] = robust.predict(heads['current'], x)
    result['noise_head_known_positions'] = robust.predict(heads['noise_trained'], x)
    baseline = originals['baseline_image']
    result['simple_known_positions'] = baseline['model'].predict_proba(x[baseline['columns']])[:, 1]
    result['training_rate_constant'] = 287/3068
    context = metadata_for(frame, pd.read_csv(folder/'context.csv'))
    models = joblib.load(OUT/'context/frozen_models.joblib')
    result['rich_plus_context'] = models['rich'].predict_proba(join_features(x, context))[:, 1]
    result['simple_plus_context'] = models['simple'].predict_proba(join_features(x, context))[:, 1]
    result.to_csv(folder/'full_pool_reference/predictions.csv', index=False)
    scores = pd.DataFrame([{'method': c, **robust.metrics(result.goal, result[c])} for c in result.columns[3:]])
    scores.to_csv(folder/'full_pool_reference/scores.csv', index=False)
    comparisons = pd.DataFrame([robust.comparison(result, a, b) for a, b in [
        ('rich_known_positions', 'simple_known_positions'),
        ('noise_head_known_positions', 'rich_known_positions'),
        ('rich_plus_context', 'rich_known_positions'), ('rich_known_positions', 'sb_xg')]])
    comparisons.to_csv(folder/'full_pool_reference/comparisons.csv', index=False)
    (folder/'full_pool_reference/cohort.json').write_text(json.dumps({'shots': len(result), 'matches': int(result.match_id.nunique()),
        'goals': int(result.goal.sum()), 'scoring_rate': float(result.goal.mean()), 'contains_vlm_subset': True}, indent=2))
    print(f'Full source pool: {len(result)} shots, {int(result.goal.sum())} goals, {result.match_id.nunique()} matches.', flush=True)
    print(scores.to_string(index=False), flush=True)
    print(comparisons.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
