"""Fixed offline experiment: train spatial xG using imperfect player positions."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import spatial_models as spatial

SPEC = {
    'seed': 20261022, 'copies': 16, 'stress_copies': 3,
    'defender_jitter_sd': 0.25, 'defender_dropout': 0.05,
    'family': 'spline', 'context': False, 'C': 0.3, 'folds': 5,
    'bootstrap_resamples': 4000,
    'selection': 'One fixed noise specification; no hyperparameter search or external tuning.',
    'primary_comparison': 'noise_trained minus original_hybrid on the same 200 saved Euro 2024 readings',
    'primary_metric': 'Brier score',
    'external_status': 'Previously studied cohort: exploratory reuse, not independent confirmation.',
    'noise_limit': 'Training errors came from an older prompt/rendering. Defender jitter/dropout are assumptions, not measured rates.',
    'weighting': 'One clean and fifteen noisy copies per shot, each with weight 1/16; total shot weight one.'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def scene(row):
    """Read image positions only; ignore outcomes, commercial xG and body part."""
    shooter = np.array([float(row.y)-40, 120-float(row.x)])
    keepers, defenders = [], []
    for player in row.ff:
        if player['teammate']:
            continue
        x, y = player['location'][:2]
        p = np.array([float(y)-40, 120-float(x)])
        if -44 <= p[0] <= 44 and -4 <= p[1] <= 64:
            target = keepers if (player.get('position') or {}).get('name') == 'Goalkeeper' else defenders
            target.append(p)
    return shooter, keepers, defenders


def empirical_errors(development, readings):
    if not development.shot_id.is_unique or not readings.shot_id.is_unique:
        raise ValueError('Duplicate noise-calibration shot IDs.')
    indexed = development.set_index('shot_id', drop=False)
    if not set(readings.shot_id).issubset(indexed.index):
        raise ValueError('Noise calibration includes shots outside development.')
    records = []
    for answer in readings.itertuples():
        row = indexed.loc[answer.shot_id]
        shooter, keepers, _ = scene(row)
        estimate = np.array([answer.shooter_lateral, answer.shooter_depth], dtype=float)
        smissing = not np.isfinite(estimate).all()
        se = np.zeros(2) if smissing else estimate-shooter
        ke = np.array([answer.goalkeeper_lateral, answer.goalkeeper_depth], dtype=float)
        kmissing = not np.isfinite(ke).all() or not bool(answer.goalkeeper_visible)
        ke = ke-keepers[0] if len(keepers) == 1 and not kmissing else np.zeros(2)
        records.append({'shot_id': row.shot_id, 'match_id': int(row.match_id), 'group_id': row.group_id,
                        'shooter_dx': float(se[0]), 'shooter_dy': float(se[1]), 'shooter_missing': smissing,
                        'keeper_dx': float(ke[0]), 'keeper_dy': float(ke[1]),
                        'keeper_missing': bool(kmissing and len(keepers) == 1)})
    return pd.DataFrame(records)


def training_error_pool(errors, training_groups, excluded_groups=()):
    training_groups, excluded_groups = set(training_groups), set(excluded_groups)
    if training_groups & excluded_groups:
        raise ValueError('A match belongs to both training and testing.')
    pool = errors[errors.group_id.isin(training_groups)].reset_index(drop=True)
    if not len(pool) or set(pool.group_id) & excluded_groups:
        raise ValueError('Invalid training-only noise donors.')
    return pool


def clip_point(p):
    return np.clip(p, [-44., -4.], [44., 64.])


def augmented_features(frame, pool, copies, seed, clean_first=True, jitter_sd=None, dropout=None):
    """Corrupt whole scenes, then recompute geometry. Return original row indices."""
    jitter_sd = SPEC['defender_jitter_sd'] if jitter_sd is None else jitter_sd
    dropout = SPEC['defender_dropout'] if dropout is None else dropout
    if copies < 1 or not len(pool) or jitter_sd < 0 or not 0 <= dropout <= 1:
        raise ValueError('Invalid augmentation settings.')
    rng = np.random.default_rng(seed)
    donors = list(pool.itertuples(index=False))
    records, parents = [], []
    for index, row in enumerate(frame.itertuples(index=False)):
        original_s, original_k, original_d = scene(row)
        for copy in range(copies):
            if clean_first and copy == 0:
                shooter, keepers, defenders = original_s, original_k, original_d
            else:
                donor = donors[int(rng.integers(len(donors)))]
                if donor.shooter_missing:
                    records.append({k: np.nan for k in spatial.RICH}); parents.append(index)
                    continue
                shooter = clip_point(original_s+[donor.shooter_dx, donor.shooter_dy])
                keepers = [] if donor.keeper_missing else [clip_point(p+[donor.keeper_dx, donor.keeper_dy]) for p in original_k]
                defenders = [clip_point(p+rng.normal(0, jitter_sd, 2)) for p in original_d if rng.random() >= dropout]
            records.append(spatial.geometry_features(shooter, keepers, defenders))
            parents.append(index)
    return pd.DataFrame(records)[spatial.RICH], np.asarray(parents, dtype=int)


def fit_head(x, y, weights=None):
    columns, model = spatial.build_model('spline', False)
    options = {} if weights is None else {'logisticregression__sample_weight': weights}
    model.fit(x[columns], np.asarray(y), **options)
    return model


def predict(model, x):
    p = model.predict_proba(x[spatial.RICH])[:, 1]
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid probability; no test shots may be removed.')
    return p


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {'brier': float(brier_score_loss(y, p)), 'auc': float(roc_auc_score(y, p)),
            'log_loss': float(log_loss(y, p, labels=[0, 1])), 'mean_xg': float(p.mean())}


def comparison(frame, first, second):
    difference = (frame[first]-frame.goal)**2-(frame[second]-frame.goal)**2
    by_match = pd.DataFrame({'match_id': frame.match_id.to_numpy(), 'difference': difference.to_numpy()})
    grouped = by_match.groupby('match_id', sort=True).difference.agg(['sum', 'count'])
    rng = np.random.default_rng(SPEC['seed'])
    draws = rng.integers(len(grouped), size=(SPEC['bootstrap_resamples'], len(grouped)))
    changes = grouped['sum'].to_numpy()[draws].sum(axis=1)/grouped['count'].to_numpy()[draws].sum(axis=1)
    return {'first': first, 'second': second, 'brier_difference': float(difference.mean()),
            'ci_low': float(np.quantile(changes, .025)), 'ci_high': float(np.quantile(changes, .975))}


def load_development(root):
    frames = []
    for name in ['wc2018', 'wc2022']:
        frame = pd.read_json(root/'data'/f'{name}.jsonl', lines=True, precise_float=True)
        frame['group_id'] = name+'/'+frame.match_id.astype(str)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if not result.shot_id.is_unique:
        raise ValueError('Duplicate development shots.')
    return result


def development_study(development, errors, output):
    clean_x = spatial.known_features(development)[spatial.RICH]
    clean_result = development[['shot_id', 'match_id', 'group_id', 'goal']].copy()
    clean_result['fold'] = -1
    stress_parts, audits = [], []
    folds = GroupKFold(SPEC['folds']).split(development, development.goal, development.group_id)
    for fold, (train, test) in enumerate(folds):
        tr, te = development.iloc[train], development.iloc[test]
        pool = training_error_pool(errors, tr.group_id, te.group_id)
        tx, parent = augmented_features(tr, pool, SPEC['copies'], SPEC['seed']+fold)
        weights = np.full(len(parent), 1/SPEC['copies'])
        plain = fit_head(clean_x.iloc[train], tr.goal)
        robust = fit_head(tx, tr.goal.to_numpy()[parent], weights)
        clean_result.loc[test, 'fold'] = fold
        clean_result.loc[test, 'clean_training'] = predict(plain, clean_x.iloc[test])
        clean_result.loc[test, 'noise_training'] = predict(robust, clean_x.iloc[test])
        sx, parent = augmented_features(te, pool, SPEC['stress_copies'], SPEC['seed']+1000+fold, clean_first=False)
        part = te.iloc[parent][['shot_id', 'match_id', 'group_id', 'goal']].reset_index(drop=True)
        part['fold'] = fold
        part['clean_training'], part['noise_training'] = predict(plain, sx), predict(robust, sx)
        stress_parts.append(part)
        audits.append({'fold': fold, 'training_groups': sorted(set(tr.group_id)), 'testing_groups': sorted(set(te.group_id)),
                       'error_donor_groups': sorted(set(pool.group_id)), 'error_donors': len(pool),
                       'training_original_shots': len(tr), 'augmented_training_rows': len(tx),
                       'training_total_weight': float(weights.sum()), 'match_overlap': False})
        print(f'Completed development fold {fold+1}/{SPEC["folds"]}', flush=True)
    stress = pd.concat(stress_parts, ignore_index=True)
    scores = [{'condition': condition, 'method': method, **metrics(frame.goal, frame[method])}
              for condition, frame in [('clean_positions', clean_result), ('simulated_noisy_positions', stress)]
              for method in ['clean_training', 'noise_training']]
    clean_result.to_csv(output/'development_clean_predictions.csv', index=False)
    stress.to_csv(output/'development_stress_predictions.csv', index=False)
    pd.DataFrame(scores).to_csv(output/'development_scores.csv', index=False)
    save_json(output/'fold_audit.json', audits)
    return pd.DataFrame(scores)


def external_features(answers):
    records = []
    for answer in answers:
        if answer['shooter'] is None:
            records.append({k: np.nan for k in spatial.RICH+spatial.BASE})
            continue
        def point(p):
            values = np.array([p['lateral'], p['depth']], dtype=float)
            if not np.isfinite(values).all() or not (-44 <= values[0] <= 44 and -4 <= values[1] <= 64):
                raise ValueError('Invalid saved VLM measurement.')
            return values
        keepers = [] if answer['goalkeeper'] is None else [point(answer['goalkeeper'])]
        records.append(spatial.geometry_features(point(answer['shooter']), keepers, [point(p) for p in answer['defenders']]))
    return pd.DataFrame(records)


def evaluate_external(root, output, development, plain, robust):
    selection = pd.read_json(root/'external_vlm_selection.jsonl', lines=True, precise_float=True)
    if set(selection.match_id) & set(development.match_id) or set(selection.shot_id) & set(development.shot_id):
        raise ValueError('External data overlap training.')
    entries = [json.loads(line) for line in (root/'rich_vlm_external/cache.jsonl').read_text().splitlines() if line.strip()]
    cache = {entry['shot_id']: entry for entry in entries}
    if len(cache) != len(entries) or len(selection) != 200 or not selection.shot_id.is_unique or set(cache) != set(selection.shot_id):
        raise ValueError('Expected the unchanged complete 200-shot external cache.')
    protocol = json.loads((root/'rich_vlm_external/protocol.json').read_text())
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    for entry in entries:
        if entry['signature'] != signature or sha(root/'rich_vlm_external/images'/f'{entry["shot_id"]}.png') != entry['image_sha256']:
            raise ValueError('External image or protocol changed.')
    x = external_features([cache[s]['answer'] for s in selection.shot_id])
    old = pd.read_csv(root/'rich_vlm_external/predictions.csv').set_index('shot_id').loc[selection.shot_id]
    if not np.array_equal(old.goal.to_numpy(), selection.goal.to_numpy()):
        raise ValueError('External outcomes disagree between files.')
    result = selection[['shot_id', 'match_id', 'goal']].reset_index(drop=True)
    result['original_hybrid'], result['noise_trained'] = predict(plain, x), predict(robust, x)
    reproduction_error = float(np.max(np.abs(result.original_hybrid.to_numpy()-old.hybrid_image.to_numpy())))
    if reproduction_error > 1e-7:
        raise ValueError(f'Current hybrid does not reproduce prior predictions: {reproduction_error}')
    baseline = joblib.load(root/'frozen_probability_models.joblib')['baseline_image']
    result['baseline_from_vlm_positions'] = baseline['model'].predict_proba(x[baseline['columns']])[:, 1]
    for destination, source in [('baseline_known_positions', 'baseline_image'), ('spatial_known_positions', 'spatial_image'), ('statsbomb', 'sb_xg')]:
        result[destination] = old[source].to_numpy()
    result['vlm_direct'] = [cache[s]['answer']['xg'] for s in selection.shot_id]
    scores = pd.DataFrame([{'method': c, **metrics(result.goal, result[c])} for c in result.columns[3:]])
    comparisons = pd.DataFrame([comparison(result, 'noise_trained', other) for other in
                               ['original_hybrid', 'baseline_known_positions', 'baseline_from_vlm_positions', 'statsbomb']])
    result.to_csv(output/'external_predictions.csv', index=False)
    x.to_csv(output/'external_vlm_features.csv', index=False)
    scores.to_csv(output/'scores.csv', index=False)
    comparisons.to_csv(output/'comparisons.csv', index=False)
    changes = result[['shot_id', 'match_id', 'goal', 'original_hybrid', 'noise_trained']].copy()
    changes['brier_change'] = (result.noise_trained-result.goal)**2-(result.original_hybrid-result.goal)**2
    changes.sort_values('brier_change').to_csv(output/'per_shot_changes.csv', index=False)
    return result, scores, comparisons, reproduction_error


LABELS = {'original_hybrid': 'Current hybrid', 'noise_trained': 'Trained with noisy positions',
          'baseline_known_positions': 'Simple baseline (known positions)',
          'baseline_from_vlm_positions': 'Simple baseline (Luna positions)',
          'spatial_known_positions': 'Rich model (known positions)', 'statsbomb': 'StatsBomb', 'vlm_direct': 'Direct Luna'}


def make_report(output, result, scores, comparisons, development):
    indexed, main = scores.set_index('method'), comparisons.iloc[0]
    old, new = indexed.loc['original_hybrid', 'brier'], indexed.loc['noise_trained', 'brier']
    relative = (old-new)/old*100
    verdict = 'The interval includes no improvement.' if main.ci_low <= 0 <= main.ci_high else (
        'The exploratory interval favors the new model.' if main.ci_high < 0 else 'The exploratory interval favors the current model.')
    lines = ['# Experiment 1: training with imperfect positions', '',
             f'**Brier {old:.5f} -> {new:.5f} ({relative:+.2f}% relative reduction).** {verdict}', '',
             f'Paired Brier difference (new minus current): **{main.brier_difference:+.5f}**, approximate 95% match-bootstrap interval **[{main.ci_low:+.5f}, {main.ci_high:+.5f}]**. Negative favors the new model.', '',
             f'Every method uses the same {len(result)} Euro 2024 shots, {int(result.goal.sum())} goals, and {result.match_id.nunique()} matches. Actual scoring rate: {result.goal.mean():.1%}. This previously studied cohort provides an exploratory test, not independent confirmation.', '',
             '| Method | Brier (lower) | AUC (higher) | Mean xG | Log loss (lower) |', '|---|---:|---:|---:|---:|']
    for name in ['original_hybrid', 'noise_trained', 'baseline_known_positions', 'baseline_from_vlm_positions', 'spatial_known_positions', 'statsbomb', 'vlm_direct']:
        r = indexed.loc[name]
        lines.append(f'| {LABELS[name]} | {r.brier:.5f} | {r.auc:.3f} | {r.mean_xg:.1%} | {r.log_loss:.5f} |')
    lines += ['', '![Comparison](comparison.png)', '', '## What changed', '',
              'Only the statistical model training inputs changed. The image-only spline model, settings, 3,068 World Cup training shots, and saved Luna readings are unchanged. Luna was not retrained.', '',
              'Each training shot supplies one clean and fifteen noisy scenes. Shooter/keeper errors are sampled together from 300 World Cup readings, including rare large errors. Defenders receive 0.25-unit Gaussian coordinate noise and a 5% independent omission probability. Defender settings are fixed assumptions. Geometry is recalculated after corruption.', '',
              'Every original shot retains total weight one across its sixteen copies. Augmentation adds neither independent observations nor extra total loss weight. No noise parameter or model was selected using Euro outcomes. Both VLM hybrids use extracted positions only, and all 200 test shots remain included.', '',
              '## Development checks', '',
              'Five folds keep complete World Cup matches together. Noise donors come only from training matches. Simulated test noise uses a separate random stream; scores below assess the simulation rather than real-world accuracy.', '',
              '| Development input | Training | Brier | AUC |', '|---|---|---:|---:|']
    for r in development.itertuples():
        lines.append(f'| {r.condition} | {r.method} | {r.brier:.5f} | {r.auc:.3f} |')
    lines += ['', '## Limits and next decision', '',
              'Training errors came from an older prompt and renderer. No measured training-defender errors were available. Synthetic noise may not reproduce occlusion or location-dependent mistakes. Euro 2024 previously informed our research; these results require fresh matches before claiming reliable improvement.', '',
              'Intervals resample whole matches conditional on fitted models, omit refitting uncertainty, and are not adjusted for multiple comparisons. Known-coordinate methods are practical controls with more accurate measurements, not VLM results. The simple baseline using Luna positions shares the extracted geometry.', '',
              'No new API requests were made. The partial Euro 2020 clearer-image run was not resumed. Review this result before proceeding to another proposed improvement.', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    order = ['original_hybrid', 'noise_trained', 'baseline_known_positions', 'statsbomb']
    values = [indexed.loc[n, 'brier'] for n in order]
    fig, ax = plt.subplots(figsize=(10.5, 5.1))
    ax.barh([LABELS[n] for n in order], values, color=['#7a8698', '#237a8c', '#b68a42', '#57628d'], height=.62)
    ax.invert_yaxis(); ax.set_xlim(0, max(values)*1.17)
    for i, value in enumerate(values):
        ax.text(value+.0008, i, f'{value:.5f}', va='center', fontweight='bold')
    ax.set_xlabel('Brier score - lower is better')
    ax.set_title('Training with imperfect positions: saved Euro 2024 readings', loc='left', pad=17, fontweight='bold')
    ax.spines[['top', 'right', 'left']].set_visible(False); ax.tick_params(axis='y', length=0)
    fig.text(.02, .035, '200 shots / 16 goals / 50 matches. Exploratory reuse of a previously studied cohort.', color='#555555', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig(output/'comparison.png', dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--geometry-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, geometry, output = args.data_root.resolve(), args.geometry_results.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [root/'data/wc2018.jsonl', root/'data/wc2022.jsonl', geometry/'geometry_predictions.csv',
               root/'external_vlm_selection.jsonl', root/'rich_vlm_external/cache.jsonl',
               root/'rich_vlm_external/predictions.csv', root/'rich_vlm_external/protocol.json', root/'frozen_probability_models.joblib']
    hashes = {str(p): sha(p) for p in sources}
    protocol = {**SPEC, 'input_sha256': hashes, 'implementation_sha256': sha(__file__),
                'features_sha256': sha(spatial.__file__), 'python': platform.python_version(), 'sklearn': sklearn.__version__}
    protocol_path = output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Protocol changed. Preserve the experiment and use a new output folder.')
    save_json(protocol_path, protocol)
    print('Fixed one noise-training protocol before evaluation. No API calls.', flush=True)
    development = load_development(root)
    errors = empirical_errors(development, pd.read_csv(geometry/'geometry_predictions.csv'))
    errors.to_csv(output/'training_error_pool.csv', index=False)
    print(f'Development: {len(development)} shots; {len(errors)} training error donors.', flush=True)
    dev_scores = development_study(development, errors, output)
    x = spatial.known_features(development)[spatial.RICH]
    tx, parents = augmented_features(development, errors, SPEC['copies'], SPEC['seed']+100)
    plain = fit_head(x, development.goal)
    robust = fit_head(tx, development.goal.to_numpy()[parents], np.full(len(parents), 1/SPEC['copies']))
    joblib.dump({'current': plain, 'noise_trained': robust, 'columns': spatial.RICH}, output/'frozen_models.joblib')
    print('Models fitted and saved before loading external outcomes.', flush=True)
    result, scores, comparisons, reproduction_error = evaluate_external(root, output, development, plain, robust)
    make_report(output, result, scores, comparisons, dev_scores)
    unchanged = {str(p): sha(p) == hashes[str(p)] for p in sources}
    if not all(unchanged.values()):
        raise ValueError('A protected input changed during the experiment.')
    save_json(output/'integrity.json', {'protected_inputs_unchanged': unchanged,
              'current_hybrid_max_reproduction_difference': reproduction_error,
              'development_shots': len(development), 'development_matches': int(development.group_id.nunique()),
              'development_goals': int(development.goal.sum()), 'external_shots': len(result),
              'external_matches': int(result.match_id.nunique()), 'external_goals': int(result.goal.sum()),
              'training_error_donors': len(errors), 'all_test_shots_retained': True, 'api_requests': 0,
              'models_sha256': sha(output/'frozen_models.joblib')})
    print(scores.to_string(index=False), flush=True)
    print(comparisons.to_string(index=False), flush=True)
    print(f'Report: {output/"report.md"}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
