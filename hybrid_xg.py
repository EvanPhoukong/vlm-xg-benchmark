"""Evaluate an image-to-geometry-to-logistic-xG hybrid on held-out matches.

The probability model learns from known StatsBomb geometry in training matches.
At evaluation time it receives only geometry computed from VLM-extracted
positions. Its fixed reference-coordinate control uses the identical fitted
probability model. No prompt or hyperparameter search is performed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import geometry_diagnostic as geo
from calibrate_xg import match_bootstrap
import xg_vlm_pilot as pilot


FEATURES = ['distance', 'angle', 'keeper_depth', 'keeper_off_axis', 'keeper_in_cone']
ORIGINAL_FEATURES = ['distance', 'angle', 'header', 'defenders_in_cone']
MODELS = {
    'vlm_xg': 'Luna, direct xG',
    'hybrid_xg': 'Hybrid, Luna positions',
    'reference_geometry_xg': 'Same model, known positions',
    'distance_angle_xg': 'Distance + angle baseline',
    'baseline_xg': 'Original logistic baseline',
    'sb_xg': 'StatsBomb',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def position_features(shooter_lateral, shooter_depth, keeper_lateral, keeper_depth):
    sl, sd, kl, kd = [np.nan if value is None else float(value)
                       for value in [shooter_lateral, shooter_depth, keeper_lateral, keeper_depth]]
    if not np.isfinite([sl, sd]).all():
        return {name: np.nan for name in FEATURES}
    distance, degrees = geo.geometry_from_relative(sl, sd)
    values = {'distance': distance, 'angle': math.radians(degrees),
              'keeper_depth': kd, 'keeper_off_axis': np.nan, 'keeper_in_cone': np.nan}
    if np.isfinite([kl, kd]).all():
        values['keeper_off_axis'] = abs(sl * kd - sd * kl) / max(distance, 1e-6)
        values['keeper_in_cone'] = float(pilot.in_triangle((kl, kd), (sl, sd), (-4, 0), (4, 0)))
    return values


def reference_positions(row):
    sl, sd = geo.relative_coordinates(row.x, row.y)
    keepers = [p for p in row.ff if not p['teammate']
               and (p.get('position') or {}).get('name') == 'Goalkeeper']
    kl, kd = (np.nan, np.nan)
    if len(keepers) == 1:
        kl, kd = geo.relative_coordinates(*keepers[0]['location'][:2])
        if not (-44 <= kl <= 44 and -4 <= kd <= 64):
            kl, kd = (np.nan, np.nan)
    return sl, sd, kl, kd


def reference_features(frame):
    rows = [position_features(*reference_positions(row)) for row in frame.itertuples()]
    return pd.DataFrame(rows, index=frame.index)[FEATURES]


def vlm_features(predictions):
    required = ['shooter_lateral', 'shooter_depth', 'goalkeeper_lateral', 'goalkeeper_depth']
    if predictions.shot_id.duplicated().any():
        raise ValueError('Duplicate VLM shot IDs.')
    numeric = predictions[required].apply(pd.to_numeric, errors='raise')
    rows = [position_features(*row) for row in numeric.itertuples(index=False, name=None)]
    return pd.DataFrame(rows, index=predictions.index)[FEATURES]


def make_model():
    # Both transforms are fitted inside each training fold, never on evaluation rows.
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def validate_frames(all_shots, evaluation):
    for label, frame in [('training pool', all_shots), ('evaluation', evaluation)]:
        if frame.shot_id.duplicated().any() or frame[['shot_id', 'match_id']].isna().any().any():
            raise ValueError(f'Invalid or duplicate IDs in {label}.')
        if not set(frame.goal.unique()).issubset({0, 1}):
            raise ValueError(f'Invalid outcomes in {label}.')
    if not set(evaluation.shot_id).issubset(set(all_shots.shot_id)):
        raise ValueError('An evaluation shot is missing from the reference pool.')
    reference = all_shots.set_index('shot_id').loc[evaluation.shot_id]
    if reference.match_id.tolist() != evaluation.match_id.tolist():
        raise ValueError('Shot and match IDs disagree with the reference pool.')
    for column in ['goal', 'x', 'y', 'sb_xg', 'distance', 'angle', 'header', 'defenders_in_cone']:
        np.testing.assert_allclose(reference[column], evaluation[column], atol=1e-8, rtol=0,
                                   err_msg=f'Source mismatch: {column}')


def fitted_audit(model):
    imputer = model.named_steps['simpleimputer']
    scaler = model.named_steps['standardscaler']
    regression = model.named_steps['logisticregression']
    return {'imputer_medians': imputer.statistics_.tolist(),
            'scaler_means': scaler.mean_.tolist(), 'scaler_scales': scaler.scale_.tolist(),
            'coefficients': regression.coef_[0].tolist(), 'intercept': float(regression.intercept_[0]),
            'feature_names': imputer.get_feature_names_out().tolist()}


def grouped_predictions(all_shots, evaluation, extracted, n_splits=5):
    validate_frames(all_shots, evaluation)
    if set(extracted.shot_id) != set(evaluation.shot_id) or extracted.shot_id.duplicated().any():
        raise ValueError('Exactly one VLM extraction is required for every evaluation shot.')
    extracted = extracted.set_index('shot_id').loc[evaluation.shot_id].reset_index()
    reference_x = reference_features(all_shots).reset_index(drop=True)
    predicted_x = vlm_features(extracted).reset_index(drop=True)
    source = all_shots.reset_index(drop=True)
    result = evaluation.reset_index(drop=True).copy()
    for column in ['hybrid_xg', 'reference_geometry_xg', 'distance_angle_xg', 'replicated_baseline_xg']:
        result[column] = np.nan
    result['hybrid_fold'] = -1
    for name in FEATURES:
        result[f'vlm_{name}'] = predicted_x[name]
    source_indices = pd.Series(np.arange(len(source)), index=source.shot_id).loc[result.shot_id].to_numpy()
    assigned = np.zeros(len(result), dtype=int)
    audit = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (training, testing) in enumerate(splitter.split(source, source.goal, source.match_id)):
        train_matches = set(source.iloc[training].match_id)
        test_matches = set(source.iloc[testing].match_id)
        if train_matches.intersection(test_matches):
            raise ValueError('Match leakage in hybrid fold.')
        target_rows = np.flatnonzero(result.match_id.isin(test_matches).to_numpy())
        if len(target_rows) == 0:
            continue
        assigned[target_rows] += 1
        result.loc[target_rows, 'hybrid_fold'] = fold
        target_source = source_indices[target_rows]
        model = make_model().fit(reference_x.iloc[training], source.iloc[training].goal)
        result.loc[target_rows, 'hybrid_xg'] = model.predict_proba(predicted_x.iloc[target_rows])[:, 1]
        result.loc[target_rows, 'reference_geometry_xg'] = model.predict_proba(reference_x.iloc[target_source])[:, 1]
        small = make_model().fit(reference_x.iloc[training][['distance', 'angle']], source.iloc[training].goal)
        result.loc[target_rows, 'distance_angle_xg'] = small.predict_proba(reference_x.iloc[target_source][['distance', 'angle']])[:, 1]
        original = LogisticRegression(max_iter=1000).fit(source.iloc[training][ORIGINAL_FEATURES], source.iloc[training].goal)
        result.loc[target_rows, 'replicated_baseline_xg'] = original.predict_proba(source.iloc[target_source][ORIGINAL_FEATURES])[:, 1]
        audit.append({'fold': fold, 'training_shots': len(training), 'training_goals': int(source.iloc[training].goal.sum()),
                      'heldout_pool_shots': len(testing), 'evaluation_shots': len(target_rows),
                      'training_match_ids': sorted(int(value) for value in train_matches),
                      'heldout_match_ids': sorted(int(value) for value in test_matches),
                      'evaluation_shot_ids': result.iloc[target_rows].shot_id.tolist(),
                      'fitted_geometry_model': fitted_audit(model)})
    if not np.all(assigned == 1):
        raise ValueError('Each selected shot must be tested exactly once.')
    probabilities = result[['hybrid_xg', 'reference_geometry_xg', 'distance_angle_xg', 'replicated_baseline_xg']]
    if not np.isfinite(probabilities.to_numpy()).all():
        raise ValueError('Missing or nonfinite probability; no selected shots may be dropped.')
    return result, audit


def calculate_metrics(frame):
    rows = []
    for column, label in MODELS.items():
        values = frame[column].to_numpy(dtype=float)
        rows.append({'prediction_column': column, 'model': label,
                     'brier': float(brier_score_loss(frame.goal, values)),
                     'auc': float(roc_auc_score(frame.goal, values)),
                     'logloss': float(log_loss(frame.goal, values, labels=[0, 1])),
                     'mean_xg': float(values.mean())})
    return pd.DataFrame(rows)


def save_plot(frame, metrics, output):
    colors = ['#777f8d', '#bb692e', '#dfad76', '#8a83b1', '#2674b8', '#33835d']
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].barh(metrics.model, metrics.brier, color=colors)
    axes[0].invert_yaxis()
    for index, value in enumerate(metrics.brier):
        axes[0].text(value + .001, index, f'{value:.4f}', va='center', fontsize=9)
    axes[0].set(xlim=(0, metrics.brier.max() * 1.23), xlabel='Brier score — lower is better')
    axes[1].plot([0, 1], [0, 1], '--', color='#777777', linewidth=1)
    for column, color in [('vlm_xg', colors[0]), ('hybrid_xg', colors[1]),
                           ('baseline_xg', colors[4]), ('sb_xg', colors[5])]:
        bins = pd.qcut(frame[column], q=5, duplicates='drop')
        means = frame.groupby(bins, observed=True).agg(predicted=(column, 'mean'), observed=('goal', 'mean'))
        axes[1].plot(means.predicted, means.observed, 'o-', label=MODELS[column], color=color, markersize=4)
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel='Average predicted scoring probability', ylabel='Observed scoring rate')
    axes[1].legend(fontsize=7, loc='lower right')
    figure.suptitle(f'Exploratory hybrid evaluation: {len(frame)} shots, {int(frame.goal.sum())} goals')
    figure.tight_layout()
    figure.savefig(output / 'comparison.png', dpi=160)
    plt.close(figure)


def write_reports(frame, metrics, comparisons, audit, output, protocol, baseline_difference):
    scores = metrics.set_index('prediction_column')
    hybrid = scores.loc['hybrid_xg']
    direct = scores.loc['vlm_xg']
    original = scores.loc['baseline_xg']
    statsbomb = scores.loc['sb_xg']
    lead = (f"The hybrid {'reduced' if hybrid.brier < direct.brier else 'increased'} Brier error compared with direct Luna predictions. "
            f"It {'beat' if hybrid.brier < original.brier else 'did not beat'} the original logistic baseline by point estimate, "
            f"and {'beat' if hybrid.brier < statsbomb.brier else 'did not beat'} StatsBomb by point estimate.")
    lines = ['# Hybrid xG test — simple report', '', '**' + lead + '**', '',
             f'Tested the same {len(frame)} shots ({int(frame.goal.sum())} goals; {frame.match_id.nunique()} matches). '
             f'Observed scoring rate: {frame.goal.mean():.1%}.', '',
             '| Method | Brier — lower is better | AUC — higher is better | Mean predicted xG |',
             '|---|---:|---:|---:|']
    for row in metrics.itertuples():
        lines.append(f'| {row.model} | {row.brier:.4f} | {row.auc:.3f} | {row.mean_xg:.1%} |')
    lines += ['', 'What changed:', '',
              '- Luna supplies shooter and goalkeeper positions from the original images. Code computes the geometry.',
              '- A five-feature logistic model converts geometry into a scoring probability. It uses distance, goal-opening angle, goalkeeper depth, goalkeeper distance from the shot-to-goal-center line, and whether the keeper is inside the shooting cone.',
              '- The model receives no header/body-part feature and no defender-count feature. These were not extracted by the VLM.',
              f"- The statistical model is trained on known coordinates in the {protocol['training_pool_shots']}-shot reference pool. For each prediction, its entire match is excluded from training. The VLM is not fine-tuned.",
              '- The known-positions row uses the exact same fitted probability model with true geometry, to isolate the effect of image extraction.',
              '- The distance-and-angle baseline uses only those two coordinate features, with the same training folds and preprocessing.', '',
              'Uncertainty — hybrid Brier minus comparator; negative favors the hybrid:', '']
    for comparison in comparisons:
        other = comparison['second']
        interval = f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]"
        includes_zero = comparison['ci_low'] <= 0 <= comparison['ci_high']
        lines.append(f"- {MODELS[other]}: {comparison['brier_difference']:+.4f}, approximate 95% match-bootstrap interval {interval}. "
                     + ('This includes no difference.' if includes_zero else ''))
    lines += ['', 'Interpretation limits:', '',
              '- This is supervised statistical training plus VLM feature extraction, rather than zero-shot VLM xG.',
              '- These are exploratory results from one competition and a repeatedly studied evaluation set. A new competition is needed for confirmation.',
              '- The original logistic baseline receives header and opponent-count information unavailable to this hybrid; StatsBomb is a commercial reference with additional information. Those are practical comparisons, not equal-input models.',
              '- Training uses known geometry; evaluation uses VLM measurements. That difference is deliberate and the known-positions control helps measure its cost.',
              '- Confidence intervals resample whole matches 4,000 times, conditional on fitted cross-validation predictions; they do not include refitting uncertainty or correction for multiple comparisons.',
              '- No parameters or feature choices were tuned after scoring this hybrid. The five-feature protocol was saved before extraction and evaluation.', '',
              '![Hybrid comparison](comparison.png)', '']
    summary = '\n'.join(lines)
    (output / 'report.md').write_text(summary, encoding='utf-8')
    detailed = summary + '\n## Validation and reproducibility\n\n'
    detailed += f'- Reference training pool: {protocol["training_pool_shots"]} shots, {protocol["training_pool_matches"]} matches, {protocol["training_pool_goals"]} goals.\n'
    detailed += '- GroupKFold, 5 folds, no shuffling; full matches remain together. The selected shots are scored once.\n'
    detailed += '- Median imputation and standardization are fitted only on each training fold, followed by L2 logistic regression, C=1, max_iter=2000. No class balancing.\n'
    detailed += '- Missing geometry is imputed from training-fold medians and is never replaced with the true evaluation coordinate. All selected shots remain in the scores.\n'
    detailed += f'- Maximum reproduced original-baseline difference: {baseline_difference:.10g}. The saved baseline is retained in the comparisons.\n'
    detailed += '- No goal, body-part, commercial xG, or previous rationale is sent to Luna. Its prompt, schema, low reasoning effort, and high image detail are unchanged from the geometry diagnostic.\n'
    detailed += '- The original 50 saved geometry responses are reused; only the remaining 250 require new requests. Requests run sequentially.\n'
    detailed += '- Positions and distances use the benchmark’s normalized StatsBomb pitch units; angles used by the model are radians.\n\n'
    detailed += 'Sources: [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html), [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).\n'
    (output / 'detailed_report.md').write_text(detailed, encoding='utf-8')
    return summary


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.input)
    selected = geo.select_shots(source, args.n, args.seed)
    all_path = args.output / 'reference_all.jsonl'
    if all_path.exists():
        all_shots = pd.read_json(all_path, lines=True, precise_float=True)
    else:
        all_shots = pilot.load_shots(args.comp, args.season)
        all_shots.to_json(all_path, orient='records', lines=True, double_precision=15)
    validate_frames(all_shots, selected)
    protocol = {
        'input_sha256': sha256(args.input), 'reference_sha256': sha256(all_path),
        'implementation_sha256': sha256(Path(__file__)),
        'training_pool_shots': len(all_shots), 'training_pool_matches': int(all_shots.match_id.nunique()),
        'training_pool_goals': int(all_shots.goal.sum()),
        'competition': args.comp, 'season': args.season, 'evaluation_shots': len(selected),
        'selected_shot_ids': selected.shot_id.tolist(), 'seed': args.seed,
        'features': FEATURES, 'training_features': 'known StatsBomb coordinates',
        'evaluation_features': 'geometry computed from VLM-reported positions only',
        'model': 'median imputer + StandardScaler + L2 LogisticRegression',
        'C': 1.0, 'max_iter': 2000, 'folds': 5, 'splitter': 'GroupKFold without shuffle on full reference pool',
        'vlm_model': args.model, 'geometry_request_signature': geo.request_signature(args.model),
        'bootstrap_resamples': 4000, 'bootstrap_seed': 7, 'tuning': False,
        'comparisons': ['vlm_xg', 'baseline_xg', 'sb_xg', 'reference_geometry_xg', 'distance_angle_xg'],
    }
    path = args.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Protocol or data changed; use a new output folder.')
    path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    selected.to_csv(args.output / 'selection.csv', index=False)
    pool = all_shots.set_index('shot_id').loc[selected.shot_id].reset_index()
    if args.reuse.exists():
        old_metadata = json.loads((args.reuse / 'metadata.json').read_text())
        if old_metadata['input_sha256'] != protocol['input_sha256']:
            raise ValueError('Existing geometry results came from a different source dataset.')
        old_cache = geo.load_cache(args.reuse / 'geometry_cache.jsonl', protocol['geometry_request_signature'])
        target = args.output / 'geometry_cache.jsonl'
        if not target.exists():
            for entry in old_cache.values():
                if entry['shot_id'] not in set(selected.shot_id):
                    continue
                image_path = args.images / f"{entry['shot_id']}.png"
                if sha256(image_path) != entry['image_sha256']:
                    raise ValueError('An existing geometry image changed.')
                with target.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(entry) + '\n')
    cache = geo.load_cache(args.output / 'geometry_cache.jsonl', protocol['geometry_request_signature'])
    for shot_id in selected.shot_id:
        if not (args.images / f'{shot_id}.png').is_file():
            raise FileNotFoundError(f'Missing original image for {shot_id}.')
    print(f'Frozen protocol: {len(selected)} evaluation shots; {len(all_shots)} training-pool shots; '
          f'{len(set(cache).intersection(selected.shot_id))} geometry responses reusable.', flush=True)
    return all_shots, selected, pool, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/gpt_6_luna/shots.csv'))
    parser.add_argument('--images', type=Path, default=Path('results/images'))
    parser.add_argument('--reuse', type=Path, default=Path('results/geometry_gpt_6_luna_50'))
    parser.add_argument('--output', type=Path, default=Path('results/hybrid_gpt_6_luna_300'))
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--comp', type=int, default=43)
    parser.add_argument('--season', type=int, default=106)
    parser.add_argument('--prepare-only', action='store_true', help='Freeze the protocol and load references without model calls')
    parser.add_argument('--offline', action='store_true', help='Require all geometry responses to be cached')
    args = parser.parse_args()
    all_shots, selected, pool, protocol = prepare(args)
    if args.prepare_only:
        return
    if args.offline:
        cached = geo.load_cache(args.output / 'geometry_cache.jsonl', protocol['geometry_request_signature'])
        if not set(selected.shot_id).issubset(cached):
            raise ValueError('Offline evaluation requires all selected geometry responses in the cache.')
    extracted = geo.infer_geometry(pool, args.images, args.output, args.model)
    extracted.to_csv(args.output / 'geometry_predictions.csv', index=False)
    predictions, audit = grouped_predictions(all_shots, selected, extracted)
    baseline_difference = float((predictions.replicated_baseline_xg - predictions.baseline_xg).abs().max())
    if baseline_difference > 1e-5:
        raise ValueError(f'Original baseline did not reproduce: maximum difference {baseline_difference:g}. Inspect data and fold provenance before interpreting results.')
    metrics = calculate_metrics(predictions)
    comparisons = [match_bootstrap(predictions, 'hybrid_xg', other, n_boot=4000, seed=7)
                   for other in protocol['comparisons']]
    predictions.to_csv(args.output / 'predictions.csv', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    pd.DataFrame(comparisons).to_csv(args.output / 'comparisons.csv', index=False)
    (args.output / 'fold_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    save_plot(predictions, metrics, args.output)
    print(write_reports(predictions, metrics, comparisons, audit, args.output, protocol, baseline_difference))


if __name__ == '__main__':
    main()
