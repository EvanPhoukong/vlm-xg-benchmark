"""Confirm frozen noise-training and pixel controls on unused Bundesliga matches."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import robust_xg as robust
import spatial_models as spatial
import rich_vlm_eval as base
from vision_trials import ROOT, OUT, freeze, sha, image_part, ask_cached, key_from_authorized_file
from pixel_control import read_pixels
from context_trial import metadata_for, join_features


def adjusted_comparison(frame, first, second):
    row = robust.comparison(frame, first, second)
    difference = (frame[first]-frame.goal)**2-(frame[second]-frame.goal)**2
    grouped = pd.DataFrame({'match': frame.match_id, 'd': difference}).groupby('match').d.agg(['sum', 'count'])
    rng = np.random.default_rng(20261022)
    draws = rng.integers(len(grouped), size=(4000, len(grouped)))
    values = grouped['sum'].to_numpy()[draws].sum(axis=1)/grouped['count'].to_numpy()[draws].sum(axis=1)
    row.update(ci_97_5_low=float(np.quantile(values, .0125)), ci_97_5_high=float(np.quantile(values, .9875)))
    return row


def prepare():
    folder = OUT/'confirmation_bundesliga'
    source_protocol = json.loads((folder/'protocol.json').read_text())
    if sha(OUT.parent/'noise-robustness-v1/frozen_models.joblib') != source_protocol['head_files_sha256']:
        raise ValueError('The noise-trained models changed after cohort reservation.')
    selection = pd.read_json(folder/'selection.jsonl', lines=True, precise_float=True)
    if len(selection) != 200 or not selection.shot_id.is_unique:
        raise ValueError('Invalid fresh cohort.')
    images = folder/'images'; images.mkdir(exist_ok=True)
    for row in selection.itertuples(index=False):
        path = images/f'{row.shot_id}.png'
        if not path.exists():
            base.render_grid(row, path)
    pixel_protocol = json.loads((OUT/'pixels/protocol.json').read_text())
    protocol = {'implementation_sha256': sha(__file__), 'base_prompt': base.PROMPT,
                'model': base.MODEL, 'settings': base.SETTINGS, 'schema': base.SCHEMA,
                'selection_sha256': sha(folder/'selection.jsonl'),
                'context_models_sha256': sha(OUT/'context/frozen_models.joblib'),
                'noise_models_sha256': source_protocol['head_files_sha256'],
                'pixel_protocol_sha256': sha(OUT/'pixels/protocol.json'),
                'image_sha256': {row.shot_id: sha(images/f'{row.shot_id}.png') for row in selection.itertuples()},
                'co_primary': ['noise_head minus original_head', 'pixel_noise_head minus simple_known_positions'],
                'multiple_testing': 'Report 97.5% Bonferroni intervals for the two predeclared primary comparisons, alongside 95% exploratory intervals.',
                'model_selection': 'Frozen before fresh inference; original and noise heads plus fixed pixel detector. Context is secondary extra-input analysis. No fresh-outcome tuning.'}
    signature = freeze(folder/'inference', protocol)
    return folder, selection, images, signature, pixel_protocol['calibration']


def evaluate(folder, selection, answers, pixel_answers):
    vx, px = robust.external_features(answers), robust.external_features(pixel_answers)
    known = spatial.known_features(selection)
    heads = joblib.load(OUT.parent/'noise-robustness-v1/frozen_models.joblib')
    original = joblib.load(ROOT/'frozen_probability_models.joblib')
    result = selection[['shot_id', 'match_id', 'goal', 'sb_xg']].copy().reset_index(drop=True)
    result['original_hybrid'] = robust.predict(heads['current'], vx)
    result['noise_trained'] = robust.predict(heads['noise_trained'], vx)
    result['pixels'] = robust.predict(heads['current'], px)
    result['pixels_noise_head'] = robust.predict(heads['noise_trained'], px)
    result['simple_known_positions'] = original['baseline_image']['model'].predict_proba(known[original['baseline_image']['columns']])[:, 1]
    result['rich_known_positions'] = robust.predict(heads['current'], known)
    result['direct_luna'] = [a['xg'] for a in answers]
    context = metadata_for(selection, pd.read_csv(folder/'context.csv'))
    ctx_models = joblib.load(OUT/'context/frozen_models.joblib')
    result['noise_plus_context'] = ctx_models['robust_context'].predict_proba(join_features(vx, context))[:, 1]
    result['simple_plus_context_known_positions'] = ctx_models['simple'].predict_proba(join_features(known, context))[:, 1]
    result.to_csv(folder/'predictions.csv', index=False)
    vx.to_csv(folder/'vlm_features.csv', index=False)
    px.to_csv(folder/'pixel_features.csv', index=False)
    scores = pd.DataFrame([{'method': c, **robust.metrics(result.goal, result[c])} for c in result.columns[3:]])
    scores.to_csv(folder/'scores.csv', index=False)
    pairs = [('noise_trained', 'original_hybrid'), ('pixels_noise_head', 'simple_known_positions'),
             ('noise_trained', 'simple_known_positions'), ('pixels_noise_head', 'sb_xg'),
             ('noise_plus_context', 'noise_trained'), ('noise_plus_context', 'simple_plus_context_known_positions')]
    comparisons = pd.DataFrame([adjusted_comparison(result, a, b) for a, b in pairs])
    comparisons['co_primary'] = [True, True, False, False, False, False]
    comparisons.to_csv(folder/'comparisons.csv', index=False)
    print(f'Fresh confirmation: {len(result)} shots, {int(result.goal.sum())} goals, {result.match_id.nunique()} matches.', flush=True)
    print(scores.to_string(index=False), flush=True)
    print(comparisons.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true'); parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    folder, selection, images, signature, transform = prepare()
    if args.prepare_only:
        print('Fresh images and inference protocol frozen; no requests or scores.', flush=True); return
    cache_path = folder/'inference/baseline_cache.jsonl'
    if args.offline:
        entries = [json.loads(line) for line in cache_path.read_text().splitlines()]
        cache = {e['shot_id']: e for e in entries}
        if len(cache) != len(entries) or set(cache) != set(selection.shot_id) or any(e['signature'] != signature for e in entries):
            raise ValueError('Offline confirmation requires the full unchanged cache.')
        answers = [base.validate(cache[s]['answer']) for s in selection.shot_id]
    else:
        from openai import OpenAI
        key = key_from_authorized_file(); answers = []
        try:
            with OpenAI(api_key=key, max_retries=5, timeout=120.) as client:
                for i, shot_id in enumerate(selection.shot_id, 1):
                    content = [image_part(images/f'{shot_id}.png'), {'type': 'input_text', 'text': base.PROMPT}]
                    answers.append(ask_cached(client, 'baseline', shot_id, content, folder/'inference', signature))
                    print(f'Fresh confirmation reading {i}/200', flush=True)
        except Exception as error:
            raise SystemExit(f'Confirmation stopped: {type(error).__name__}: {str(error).replace(key, "[REDACTED]")}') from None
    pixels = [read_pixels(images/f'{s}.png', transform) for s in selection.shot_id]
    (folder/'pixel_answers.json').write_text(json.dumps(pixels), encoding='utf-8')
    evaluate(folder, selection, answers, pixels)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
