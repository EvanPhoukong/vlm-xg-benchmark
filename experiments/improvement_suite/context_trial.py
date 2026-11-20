"""Fixed statistical test adding pre-shot context, separately from image-only xG."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
import robust_xg as robust
import spatial_models as spatial
from fetch_context import CATEGORICAL, NUMERIC
from vision_trials import ROOT, OUT, freeze, sha


def metadata_for(frame, metadata):
    rows = metadata.set_index('shot_id').loc[frame.shot_id, CATEGORICAL+NUMERIC].reset_index(drop=True)
    rows[CATEGORICAL] = rows[CATEGORICAL].fillna('Unknown')
    return rows


def model_for(family='spline'):
    if family == 'spline':
        columns, original = spatial.build_model('spline', False)
        geometry = clone(original[:-1])
        penalty = .3
    else:
        columns, geometry, penalty = spatial.BASE, 'passthrough', 1.
    context = ColumnTransformer([
        ('categories', OneHotEncoder(handle_unknown='ignore', min_frequency=20, sparse_output=False), CATEGORICAL),
        ('numbers', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), NUMERIC)])
    transform = ColumnTransformer([('geometry', geometry, columns), ('context', context, CATEGORICAL+NUMERIC)])
    return make_pipeline(transform, LogisticRegression(C=penalty, max_iter=3000))


def join_features(geometry, metadata):
    return pd.concat([geometry.reset_index(drop=True), metadata.reset_index(drop=True)], axis=1)


def main():
    folder = OUT/'context'
    path = folder/'pre_shot_context.csv'
    metadata = pd.read_csv(path)
    if not metadata.shot_id.is_unique:
        raise ValueError('Duplicate event context.')
    protocol = {'implementation_sha256': sha(__file__), 'metadata_sha256': sha(path),
                'features': CATEGORICAL+NUMERIC, 'time_limit_seconds': 15,
                'definition': 'Latest earlier same-team, same-possession, same-period pass within 15 seconds; shot type, play pattern, pressure; never shot outcome or StatsBomb xG.',
                'model': 'Same spline geometry transform + context one-hot/numeric features + L2 logistic C=.3. Simple matched-input baseline uses C=1.',
                'selection': 'One fixed feature set and model; no external tuning.',
                'external_status': 'Extra-input experiment, separate from image-only; previously studied Euro tournaments.'}
    freeze(folder, protocol)
    development = robust.load_development(ROOT)
    x = spatial.known_features(development)
    context = metadata_for(development, metadata)
    combined = join_features(x, context)
    oof = development[['shot_id', 'match_id', 'goal']].copy()
    old_development = pd.read_csv(ROOT/'development_predictions.csv').set_index('shot_id').loc[development.shot_id]
    oof['original_rich_geometry'] = old_development.spline_image.to_numpy()
    audit = []
    for fold, (train, test) in enumerate(GroupKFold(5).split(development, development.goal, development.group_id)):
        model = model_for().fit(combined.iloc[train], development.goal.iloc[train])
        oof.loc[test, 'rich_plus_context'] = model.predict_proba(combined.iloc[test])[:, 1]
        if set(development.iloc[train].match_id) & set(development.iloc[test].match_id):
            raise ValueError('Match overlap.')
        audit.append({'fold': fold, 'train_matches': sorted(int(m) for m in development.iloc[train].match_id.unique()),
                      'test_matches': sorted(int(m) for m in development.iloc[test].match_id.unique())})
        print(f'Context model development fold {fold+1}/5', flush=True)
    oof.to_csv(folder/'development_predictions.csv', index=False)
    pd.DataFrame([{'method': c, **robust.metrics(oof.goal, oof[c])} for c in oof.columns[3:]]).to_csv(folder/'development_scores.csv', index=False)
    (folder/'fold_audit.json').write_text(json.dumps(audit, indent=2))
    rich = model_for().fit(combined, development.goal)
    simple = model_for('baseline').fit(combined, development.goal)
    errors = pd.read_csv(OUT.parent/'noise-robustness-v1/training_error_pool.csv')
    tx, parents = robust.augmented_features(development, errors, robust.SPEC['copies'], robust.SPEC['seed']+100)
    robust_context = model_for().fit(join_features(tx, context.iloc[parents]), development.goal.to_numpy()[parents],
                                    logisticregression__sample_weight=np.full(len(parents), 1/robust.SPEC['copies']))
    joblib.dump({'rich': rich, 'simple': simple, 'robust_context': robust_context}, folder/'frozen_models.joblib')
    selection = pd.read_json(ROOT/'external_vlm_selection.jsonl', lines=True, precise_float=True)
    if set(selection.match_id) & set(development.match_id):
        raise ValueError('External training overlap.')
    ext_context = metadata_for(selection, metadata)
    vx = pd.read_csv(ROOT/'rich_vlm_external/vlm_features.csv')
    known = spatial.known_features(selection)
    result = selection[['shot_id', 'match_id', 'goal']].copy().reset_index(drop=True)
    old = pd.read_csv(ROOT/'rich_vlm_external/predictions.csv').set_index('shot_id').loc[selection.shot_id]
    result['original_hybrid'] = old.hybrid_image.to_numpy()
    result['hybrid_plus_context'] = rich.predict_proba(join_features(vx, ext_context))[:, 1]
    result['noise_training_plus_context'] = robust_context.predict_proba(join_features(vx, ext_context))[:, 1]
    result['noise_training_image_only'] = pd.read_csv(OUT.parent/'noise-robustness-v1/external_predictions.csv').set_index('shot_id').loc[selection.shot_id].noise_trained.to_numpy()
    result['simple_known_positions'] = old.baseline_image.to_numpy()
    result['simple_plus_context_known_positions'] = simple.predict_proba(join_features(known, ext_context))[:, 1]
    result['rich_known_positions'] = old.spatial_image.to_numpy()
    result['rich_plus_context_known_positions'] = rich.predict_proba(join_features(known, ext_context))[:, 1]
    result['statsbomb'] = old.sb_xg.to_numpy()
    result.to_csv(folder/'external_predictions.csv', index=False)
    scores = pd.DataFrame([{'method': c, **robust.metrics(result.goal, result[c])} for c in result.columns[3:]])
    scores.to_csv(folder/'scores.csv', index=False)
    comparisons = pd.DataFrame([robust.comparison(result, a, b) for a, b in [
        ('hybrid_plus_context', 'original_hybrid'), ('noise_training_plus_context', 'noise_training_image_only'),
        ('hybrid_plus_context', 'simple_plus_context_known_positions'),
        ('rich_plus_context_known_positions', 'rich_known_positions')]])
    comparisons.to_csv(folder/'comparisons.csv', index=False)
    # Separate full-tournament controls; never mix their scores with the 200-shot cohort.
    frozen = joblib.load(ROOT/'frozen_probability_models.joblib')
    for name in ['euro2024', 'euro2020']:
        frame = pd.read_json(ROOT/'data'/f'{name}.jsonl', lines=True, precise_float=True)
        sx = spatial.known_features(frame)
        values = join_features(sx, metadata_for(frame, metadata))
        pred = frame[['shot_id', 'match_id', 'goal', 'sb_xg']].copy()
        pred['original_rich'] = frozen['spatial_image']['model'].predict_proba(sx[spatial.RICH])[:, 1]
        pred['rich_plus_context'] = rich.predict_proba(values)[:, 1]
        pred['simple_plus_context'] = simple.predict_proba(values)[:, 1]
        pred.to_csv(folder/f'{name}_known_predictions.csv', index=False)
        pd.DataFrame([{'method': c, **robust.metrics(pred.goal, pred[c])} for c in pred.columns[3:]]).to_csv(folder/f'{name}_known_scores.csv', index=False)
        pd.DataFrame([robust.comparison(pred, 'rich_plus_context', 'original_rich')]).to_csv(folder/f'{name}_known_comparison.csv', index=False)
    print(scores.to_string(index=False), flush=True)
    print(comparisons.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
