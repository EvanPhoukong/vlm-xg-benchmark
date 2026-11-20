"""Select the strongest screening VLM candidate, then confirm it prospectively."""
import copy
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits
import robust_xg as robust
import rich_vlm_eval as base
from vision_trials import OUT, freeze, sha, image_part, ask_cached, key_from_authorized_file, disagreement, consensus


def select_candidate():
    candidates = []
    for phase in ['multiview', 'fewshot']:
        scores = pd.read_csv(OUT/'vision'/f'{phase}_scores.csv')
        candidates.append(scores[scores.method.isin([phase, phase+'_noise_head'])])
    chosen = pd.concat(candidates, ignore_index=True).sort_values(['brier', 'method']).iloc[0]
    folder = OUT/'confirmation_bundesliga/best_vision'; folder.mkdir(parents=True, exist_ok=True)
    protocol = {'implementation_sha256': sha(__file__), 'chosen_method': chosen.method,
                'selection': 'Lowest screening Euro Brier among multiple-reading and four-example pipelines, each with current or noise-trained head; no fresh outcomes used.',
                'screening_brier': float(chosen.brier),
                'selection_inputs_sha256': {phase: sha(OUT/'vision'/f'{phase}_scores.csv') for phase in ['multiview', 'fewshot']},
                'selection_sha256': sha(OUT/'confirmation_bundesliga/selection.jsonl'),
                'vision_implementation_sha256': sha(Path(__file__).parent/'vision_trials.py'),
                'noise_models_sha256': sha(OUT.parent/'noise-robustness-v1/frozen_models.joblib'),
                'purpose': 'Secondary prospective confirmation of the strongest screening image-only VLM pipeline. Selected and frozen before fresh baseline scores exist.',
                'comparisons': 'Best VLM vs noise-only, simple known-coordinate baseline and StatsBomb; report 98.333% Bonferroni intervals across these three secondary comparisons.'}
    if (OUT/'confirmation_bundesliga/comparisons.csv').exists() and not (folder/'protocol.json').exists():
        raise ValueError('Cannot select a new candidate after fresh baseline scores have been released.')
    signature = freeze(folder, protocol)
    print(f'Selected and froze {chosen.method} from Euro screening before reading any fresh scores.', flush=True)
    return folder, chosen.method, signature


def compare(frame, first, second):
    row = robust.comparison(frame, first, second)
    differences = (frame[first]-frame.goal)**2-(frame[second]-frame.goal)**2
    grouped = pd.DataFrame({'match': frame.match_id, 'd': differences}).groupby('match').d.agg(['sum', 'count'])
    rng = np.random.default_rng(20261022)
    draws = rng.integers(len(grouped), size=(4000, len(grouped)))
    values = grouped['sum'].to_numpy()[draws].sum(axis=1)/grouped['count'].to_numpy()[draws].sum(axis=1)
    row.update(ci_98_333_low=float(np.quantile(values, .05/6)), ci_98_333_high=float(np.quantile(values, 1-.05/6)))
    return row


def main():
    deadline = time.monotonic()+3*3600
    while not (OUT/'vision/fewshot_comparisons.csv').exists():
        if time.monotonic() > deadline:
            raise SystemExit('Screening did not finish; no additional confirmation requests made.')
        time.sleep(10)
    folder, method, signature = select_candidate()
    # Original/noise/pixel confirmation finishes its API calls before this starts.
    while not (OUT/'confirmation_bundesliga/comparisons.csv').exists():
        if time.monotonic() > deadline:
            raise SystemExit('Baseline confirmation did not finish; no duplicate API job started.')
        time.sleep(10)
    selection = pd.read_json(OUT/'confirmation_bundesliga/selection.jsonl', lines=True, precise_float=True)
    images = OUT/'confirmation_bundesliga/images'
    originals = {e['shot_id']: e['answer'] for e in [json.loads(line) for line in (OUT/'confirmation_bundesliga/inference/baseline_cache.jsonl').read_text().splitlines()]}
    examples = json.loads((OUT/'vision/examples.json').read_text())
    prefix = [{'type': 'input_text', 'text': 'Four worked TRAINING examples follow. They teach reading coordinates only. They contain no scoring labels. The final image is a new situation.'}]
    for i, example in enumerate(examples, 1):
        if sha(example['path']) != example['image_sha256']:
            raise ValueError('Worked-example image changed.')
        prefix += [image_part(example['path']), {'type': 'input_text', 'text': f'Example {i}, correct geometry: '+json.dumps(example['geometry_only'])}]
    answers = []
    from openai import OpenAI
    key = key_from_authorized_file()
    try:
        with OpenAI(api_key=key, max_retries=5, timeout=120.) as client:
            for i, shot_id in enumerate(selection.shot_id, 1):
                path = images/f'{shot_id}.png'
                if method.startswith('fewshot'):
                    content = prefix+[image_part(path), {'type': 'input_text', 'text': 'Now read ONLY the final new image.\n'+base.PROMPT}]
                    answer = ask_cached(client, 'best_fewshot', shot_id, content, folder, signature)
                else:
                    zoom = folder/f'{shot_id}_zoom.png'
                    if not zoom.exists():
                        with Image.open(path) as im:
                            w, h = im.size
                            im.crop((int(w*.25), int(h*.10), int(w*.78), int(h*.58))).resize((1060, 960), Image.Resampling.LANCZOS).save(zoom)
                    content = [image_part(path), image_part(zoom), {'type': 'input_text', 'text': base.PROMPT+'\nThe second image is a fixed enlarged crop of the FIRST image, not another situation. Use full-image axes for all returned coordinates. Read independently and carefully.'}]
                    second = ask_cached(client, 'best_second', shot_id, content, folder, signature)
                    readings = [originals[shot_id], second]
                    if disagreement(*readings) > 2:
                        check = copy.deepcopy(content)
                        check[-1]['text'] += '\nVerify marker centers, each visible defender, and the shooter carefully. Do not assume a point near a line is at zero.'
                        readings.append(ask_cached(client, 'best_third', shot_id, check, folder, signature))
                    answer = consensus(readings)
                answers.append(answer)
                print(f'Best-VLM fresh reading {i}/200 ({method})', flush=True)
    except Exception as error:
        raise SystemExit(f'Best-VLM confirmation stopped: {type(error).__name__}: {str(error).replace(key, "[REDACTED]")}') from None
    heads = joblib.load(OUT.parent/'noise-robustness-v1/frozen_models.joblib')
    x = robust.external_features(answers)
    head = heads['noise_trained' if method.endswith('_noise_head') else 'current']
    result = pd.read_csv(OUT/'confirmation_bundesliga/predictions.csv')
    if result.shot_id.tolist() != selection.shot_id.tolist():
        raise ValueError('Fresh cohort order changed.')
    result['best_vision'] = robust.predict(head, x)
    result.to_csv(folder/'predictions.csv', index=False)
    scores = pd.DataFrame([{'method': 'best_vision', 'screening_method': method, **robust.metrics(result.goal, result.best_vision)}])
    scores.to_csv(folder/'scores.csv', index=False)
    comparisons = pd.DataFrame([compare(result, 'best_vision', other) for other in ['noise_trained', 'simple_known_positions', 'sb_xg']])
    comparisons.to_csv(folder/'comparisons.csv', index=False)
    (folder/'answers.json').write_text(json.dumps(answers), encoding='utf-8')
    print(scores.to_string(index=False), flush=True)
    print(comparisons.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
