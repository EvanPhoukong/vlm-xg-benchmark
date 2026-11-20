"""Fixed sequential trials of repeat visual readings and four worked examples."""
import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
import rich_vlm_eval as base
import robust_xg as robust

ROOT = Path('/mnt/c/Users/ephou/Downloads/xG/output/research-loop')
OUT = Path('/mnt/c/Users/ephou/Downloads/xG/output/improvement-suite')
REPO = Path('/home/evan/xG_Research/vlm-xg-benchmark')
SEED = 20261023


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(folder, protocol):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Protocol changed; use a new output folder.')
    path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def key_from_authorized_file():
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']
    contents = (REPO.parent/'keys.txt').read_text(encoding='utf-8-sig')
    named = re.search(r'(?m)^\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*([^\r\n]+)', contents)
    values = re.findall(r'sk-(?!ant-)[A-Za-z0-9_-]+', named.group(1)) if named else sorted(set(re.findall(r'sk-(?!ant-)[A-Za-z0-9_-]+', contents)))
    if len(values) != 1:
        raise RuntimeError('Could not identify one authorized API key; no credential printed.')
    return values[0]


def image_part(path):
    return {'type': 'input_image', 'image_url': 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode(), 'detail': 'high'}


def disagreement(first, second):
    distances = []
    for role in ['shooter', 'goalkeeper']:
        a, b = first[role], second[role]
        if (a is None) != (b is None):
            return float('inf')
        if a is not None:
            distances.append(float(np.hypot(a['lateral']-b['lateral'], a['depth']-b['depth'])))
    if len(first['defenders']) != len(second['defenders']):
        return float('inf')
    if first['defenders']:
        a = np.array([[p['lateral'], p['depth']] for p in first['defenders']])
        b = np.array([[p['lateral'], p['depth']] for p in second['defenders']])
        cost = np.linalg.norm(a[:, None, :]-b[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(cost)
        distances.extend(cost[rows, columns].tolist())
    return max(distances, default=0.)


def consensus(answers):
    """Robust medoid cardinality; align defenders without using true coordinates."""
    def distance(a, b):
        score = 2*abs(len(a['defenders'])-len(b['defenders']))
        for role in ['shooter', 'goalkeeper']:
            p, q = a[role], b[role]
            score += 4 if (p is None) != (q is None) else (0 if p is None else min(10., np.hypot(p['lateral']-q['lateral'], p['depth']-q['depth'])))
        if a['defenders'] and b['defenders']:
            p = np.array([[v['lateral'], v['depth']] for v in a['defenders']])
            q = np.array([[v['lateral'], v['depth']] for v in b['defenders']])
            cost = np.linalg.norm(p[:, None, :]-q[None, :, :], axis=2)
            rows, columns = linear_sum_assignment(cost)
            score += float(np.minimum(cost[rows, columns], 10).sum())
        return score
    anchor = copy.deepcopy(answers[int(np.argmin([sum(distance(a, b) for b in answers) for a in answers]))])
    for role in ['shooter', 'goalkeeper']:
        points = [[a[role]['lateral'], a[role]['depth']] for a in answers if a[role] is not None]
        if len(points) > len(answers)/2:
            p = np.median(points, axis=0)
            anchor[role] = {'lateral': float(p[0]), 'depth': float(p[1])}
        else:
            anchor[role] = None
    if anchor['defenders']:
        positions = np.array([[p['lateral'], p['depth']] for p in anchor['defenders']])
        gathered = [[] for p in positions]
        for answer in answers:
            if not answer['defenders']:
                continue
            p = np.array([[v['lateral'], v['depth']] for v in answer['defenders']])
            cost = np.linalg.norm(positions[:, None, :]-p[None, :, :], axis=2)
            rows, columns = linear_sum_assignment(cost)
            for r, c in zip(rows, columns):
                if cost[r, c] <= 2:
                    gathered[r].append(p[c].tolist())
        anchor['defenders'] = [{'lateral': float(p[0]), 'depth': float(p[1])} for p in [np.median(group, axis=0) for group in gathered]]
    anchor['xg'] = float(np.median([a['xg'] for a in answers]))
    anchor['reason'] = 'Consensus of independent image readings.'
    return base.validate(anchor)


def worked_examples(folder):
    development = robust.load_development(ROOT)
    candidates = development[development.group_id.str.startswith('wc2018/')].copy()
    # Fixed distance strata; never select examples based on goals, xG or errors.
    order = candidates.sort_values('shot_id')
    examples = []
    for i, (low, high) in enumerate([(0, 8), (8, 16), (16, 24), (24, 70)]):
        eligible = order[(order.distance >= low) & (order.distance < high)]
        row = eligible.sample(n=1, random_state=SEED+i).iloc[0]
        path = folder/f'example_{i+1}.png'
        if not path.exists():
            base.render_grid(row, path)
        s, k, d = robust.scene(row)
        point = lambda p: {'lateral': round(float(p[0]), 1), 'depth': round(float(p[1]), 1)}
        answer = {'shooter': point(s), 'goalkeeper': point(k[0]) if len(k) == 1 else None,
                  'defenders': [point(p) for p in d]}
        examples.append({'shot_id': row.shot_id, 'match_id': int(row.match_id), 'path': str(path),
                         'image_sha256': sha(path), 'geometry_only': answer})
    (folder/'examples.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')
    return examples


def prepare():
    OUT.mkdir(exist_ok=True)
    selection = pd.read_json(ROOT/'external_vlm_selection.jsonl', lines=True, precise_float=True)
    old_cache = {e['shot_id']: e for e in [json.loads(line) for line in (ROOT/'rich_vlm_external/cache.jsonl').read_text().splitlines()]}
    folder = OUT/'vision'
    (folder/'zoom').mkdir(parents=True, exist_ok=True)
    examples = worked_examples(folder)
    if set(e['match_id'] for e in examples) & set(selection.match_id):
        raise ValueError('A worked-example match overlaps the evaluation cohort.')
    for shot_id in selection.shot_id:
        original = ROOT/'rich_vlm_external/images'/f'{shot_id}.png'
        if sha(original) != old_cache[shot_id]['image_sha256']:
            raise ValueError('Original image changed.')
        zoom = folder/'zoom'/f'{shot_id}.png'
        if not zoom.exists():
            # Fixed image-fraction crop, identical for every shot. No query coordinates used.
            im = Image.open(original)
            w, h = im.size
            im.crop((int(w*.25), int(h*.10), int(w*.78), int(h*.58))).resize((1060, 960), Image.Resampling.LANCZOS).save(zoom)
    protocol = {'model': base.MODEL, 'settings': base.SETTINGS, 'schema': base.SCHEMA,
                'base_prompt': base.PROMPT, 'implementation_sha256': sha(__file__),
                'selection_sha256': sha(ROOT/'external_vlm_selection.jsonl'),
                'original_cache_sha256': sha(ROOT/'rich_vlm_external/cache.jsonl'),
                'examples': examples, 'two_readings': 'Original cached reading plus a new independent original+fixed-zoom reading. Third reading only if coordinates differ >2 units, visibility differs, or defender counts disagree.',
                'consensus': 'Majority visibility, coordinate medians, medoid defender list with Hungarian matching within 2 units.',
                'few_shot': 'Four fixed World Cup 2018 images with geometry labels only; no outcomes, xG or query coordinates.',
                'sequential_requests': True, 'no_outcome_based_selection': True,
                'external_status': 'Same previously studied 200 Euro 2024 shots; exploratory.'}
    signature = freeze(folder, protocol)
    return selection, old_cache, folder, examples, signature


def ask_cached(client, condition, shot_id, content, folder, signature):
    path = folder/f'{condition}_cache.jsonl'
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            if entry['signature'] != signature or entry['shot_id'] in cache:
                raise ValueError('Mixed or duplicate new request cache.')
            cache[entry['shot_id']] = entry
    if shot_id in cache:
        return base.validate(cache[shot_id]['answer'])
    start = time.monotonic()
    response = client.responses.create(model=base.MODEL, store=False, reasoning={'effort': 'low'},
        max_output_tokens=4096, input=[{'role': 'user', 'content': content}],
        text={'format': {'type': 'json_schema', 'name': 'shot_geometry', 'strict': True, 'schema': base.SCHEMA}})
    if response.status != 'completed' or not response.output_text:
        raise RuntimeError(f'Incomplete response: {response.status}')
    answer = base.validate(json.loads(response.output_text))
    entry = {'shot_id': shot_id, 'signature': signature, 'condition': condition, 'answer': answer,
             'response_id': response.id, 'returned_model': response.model, 'elapsed_seconds': time.monotonic()-start,
             'usage': response.usage.model_dump() if response.usage else None}
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(entry, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    return answer


def evaluate(selection, answers, label, folder):
    x = robust.external_features(answers)
    old = pd.read_csv(ROOT/'rich_vlm_external/predictions.csv').set_index('shot_id').loc[selection.shot_id]
    result = selection[['shot_id', 'match_id', 'goal']].reset_index(drop=True)
    models = joblib.load(OUT.parent/'noise-robustness-v1/frozen_models.joblib')
    result['original_hybrid'] = old.hybrid_image.to_numpy()
    result['original_reading_noise_head'] = pd.read_csv(OUT.parent/'noise-robustness-v1/external_predictions.csv').set_index('shot_id').loc[selection.shot_id].noise_trained.to_numpy()
    result[label] = robust.predict(models['current'], x)
    result[label+'_noise_head'] = robust.predict(models['noise_trained'], x)
    for name in ['baseline_image', 'sb_xg']:
        result[name] = old[name].to_numpy()
    scores = pd.DataFrame([{'method': c, **robust.metrics(result.goal, result[c])} for c in result.columns[3:]])
    pairs = [(label, 'original_hybrid'), (label+'_noise_head', 'original_reading_noise_head'),
             (label+'_noise_head', 'baseline_image')]
    comparisons = pd.DataFrame([robust.comparison(result, a, b) for a, b in pairs])
    result.to_csv(folder/f'{label}_predictions.csv', index=False)
    x.to_csv(folder/f'{label}_features.csv', index=False)
    scores.to_csv(folder/f'{label}_scores.csv', index=False)
    comparisons.to_csv(folder/f'{label}_comparisons.csv', index=False)
    print(label, scores.to_json(orient='records'), flush=True)
    print(comparisons.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--phase', choices=['multiview', 'fewshot', 'all'], default='all')
    args = parser.parse_args()
    selection, old, folder, examples, signature = prepare()
    if args.prepare_only:
        print('Prepared fixed zooms, training examples and protocol; no requests.', flush=True)
        return
    from openai import OpenAI
    key = key_from_authorized_file()
    try:
        with OpenAI(api_key=key, max_retries=5, timeout=120.) as client:
            if args.phase in ['multiview', 'all']:
                answers, third_reads = [], 0
                for index, shot_id in enumerate(selection.shot_id, 1):
                    image = ROOT/'rich_vlm_external/images'/f'{shot_id}.png'
                    content = [image_part(image), image_part(folder/'zoom'/f'{shot_id}.png'),
                               {'type': 'input_text', 'text': base.PROMPT+'\nThe second image is a fixed enlarged crop of the FIRST image, not another situation. Use full-image axes for all returned coordinates. Read independently and carefully.'}]
                    second = ask_cached(client, 'second', shot_id, content, folder, signature)
                    readings = [old[shot_id]['answer'], second]
                    if disagreement(*readings) > 2:
                        third_reads += 1
                        check = copy.deepcopy(content)
                        check[-1]['text'] += '\nVerify marker centers, each visible defender, and the shooter carefully. Do not assume a point near a line is at zero.'
                        readings.append(ask_cached(client, 'third', shot_id, check, folder, signature))
                    answers.append(consensus(readings))
                    print(f'Multiple-view {index}/{len(selection)}; third readings {third_reads}', flush=True)
                (folder/'multiview_answers.json').write_text(json.dumps(answers), encoding='utf-8')
                evaluate(selection, answers, 'multiview', folder)
            if args.phase in ['fewshot', 'all']:
                prefix = [{'type': 'input_text', 'text': 'Four worked TRAINING examples follow. They teach reading coordinates only. They contain no scoring labels. The final image is a new situation.'}]
                for i, example in enumerate(examples, 1):
                    prefix += [image_part(example['path']), {'type': 'input_text', 'text': f'Example {i}, correct geometry: '+json.dumps(example['geometry_only'])}]
                answers = []
                for index, shot_id in enumerate(selection.shot_id, 1):
                    content = prefix+[image_part(ROOT/'rich_vlm_external/images'/f'{shot_id}.png'),
                                      {'type': 'input_text', 'text': 'Now read ONLY the final new image.\n'+base.PROMPT}]
                    answers.append(ask_cached(client, 'fewshot', shot_id, content, folder, signature))
                    print(f'Worked-example {index}/{len(selection)}', flush=True)
                evaluate(selection, answers, 'fewshot', folder)
    except Exception as error:
        message = str(error).replace(key, '[REDACTED]')
        raise SystemExit(f'Vision trial stopped: {type(error).__name__}: {message}') from None


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
