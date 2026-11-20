"""Prospective direct-image xG search; preparation makes no generation calls."""
import argparse, ast, base64, concurrent.futures, csv, hashlib, json, math, os
from pathlib import Path
import random, re, threading, time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/direct_model_search_v1'
BASE = 'https://raw.githubusercontent.com/hudl/open-data/master/data/'
CAP, MAX_OUTPUT, MAX_ATTEMPTS = 3.0, 4096, 650
RATES = {'gpt-6-luna': (.10, .01, .50), 'gpt-6.1-terra': (1., .05, 5.), 'gpt-6.1-sol': (2., .10, 10.)}
ARMS = {
    'luna_original': {'model': 'gpt-6-luna', 'effort': 'low', 'prompt': 'original'},
    'luna_uncertainty': {'model': 'gpt-6-luna', 'effort': 'low', 'prompt': 'uncertainty'},
    'terra_original': {'model': 'gpt-6.1-terra', 'effort': 'medium', 'prompt': 'original'},
    'terra_uncertainty': {'model': 'gpt-6.1-terra', 'effort': 'medium', 'prompt': 'uncertainty'},
    'sol_original': {'model': 'gpt-6.1-sol', 'effort': 'low', 'prompt': 'original'}}
SCHEMA = {'type': 'object', 'properties': {'xg': {'type': 'number', 'minimum': 0, 'maximum': 1},
    'reason': {'type': 'string'}}, 'required': ['xg', 'reason'], 'additionalProperties': False}
EXTRA = """Before estimating, inspect the shooter's position, the visible opening between the
posts, the goalkeeper's position and reach, and opponents between shooter and goal.
This diagram contains positions only. Body part, ball height, ball speed, balance,
and the preceding action are unobserved. Account for that missing information in
the probability: average over plausible executions compatible with the scene,
instead of assuming an ideal grounded shot or inventing a header or a pass.
Do not infer the outcome from the diagram. Return your own direct estimate.
"""


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8'); temp.replace(path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def freeze(path, value):
    if Path(path).exists():
        if read(path) != value: raise ValueError('Frozen artifact differs: ' + str(path))
    else: save(path, value)


def original_prompt():
    for node in ast.parse((ROOT / 'xg_vlm_pilot.py').read_text(encoding='utf-8-sig')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROMPT' for t in node.targets):
            return ast.literal_eval(node.value).format(context='')
    raise ValueError('Original prompt missing')


def prompts():
    original = original_prompt()
    return {'original': original, 'uncertainty': EXTRA + '\n' + original}


def credential():
    if os.environ.get('OPENAI_API_KEY', '').strip(): return os.environ['OPENAI_API_KEY'].strip()
    text = (ROOT.parent / 'keys.txt').read_text(encoding='utf-8-sig')
    values = set(re.findall(r'sk-(?!ant-)[A-Za-z0-9_-]+', text))
    if len(values) != 1: raise RuntimeError('One OpenAI credential required; no credential displayed')
    return values.pop()


def api(endpoint, key, data=None):
    headers = {'Authorization': 'Bearer ' + key}
    if data is not None: headers['Content-Type'] = 'application/json'
    request = Request('https://api.openai.com/v1/' + endpoint,
        data=None if data is None else json.dumps(data).encode(), headers=headers)
    try:
        with urlopen(request, timeout=150) as response: return json.load(response)
    except HTTPError as error:
        try: code = json.loads(error.read()).get('error', {}).get('code', 'http_error')
        except Exception: code = 'http_error'
        raise RuntimeError(f'HTTP {error.code}: {code}; no automatic retry') from None
    except URLError as error:
        raise RuntimeError('Network failure: ' + type(error.reason).__name__ + '; no automatic retry') from None


def public(relative):
    path = OUT / 'data' / relative.replace('/', '_')
    if not path.exists():
        with urlopen(BASE + relative, timeout=90) as response: content = response.read()
        json.loads(content); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
    return read(path)


def body(arm, prompt, image):
    return {'model': arm['model'], 'store': False, 'service_tier': 'default',
        'reasoning': {'effort': arm['effort']}, 'max_output_tokens': MAX_OUTPUT,
        'input': [{'role': 'user', 'content': [
            {'type': 'input_image', 'detail': 'high', 'image_url': 'data:image/png;base64,' + base64.b64encode(image).decode()},
            {'type': 'input_text', 'text': prompt}]}],
        'text': {'format': {'type': 'json_schema', 'name': 'shot_prediction', 'strict': True, 'schema': SCHEMA}}}


def validate(response, arm):
    model = response.get('model', '')
    if model != arm['model'] and not model.startswith(arm['model'] + '-'): raise ValueError('Unexpected returned model')
    if response.get('status') != 'completed': raise ValueError('Incomplete response')
    if response.get('reasoning', {}).get('effort') != arm['effort']: raise ValueError('Unexpected reasoning setting')
    texts = [p['text'] for item in response.get('output', []) if item.get('type') == 'message'
        for p in item.get('content', []) if p.get('type') == 'output_text']
    answer = json.loads(''.join(texts)); value = answer.get('xg')
    if (set(answer) != {'xg', 'reason'} or isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or not 0 <= value <= 1 or not isinstance(answer['reason'], str)):
        raise ValueError('Invalid raw probability; no clipping or conversion')
    usage = response.get('usage') or {}
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ('input_tokens', 'output_tokens')):
        raise ValueError('Usage missing')
    return answer


def cost(usage, model, conservative=False):
    ir, cr, outr = RATES[model]; inputs, outputs = usage.get('input_tokens', 0), usage.get('output_tokens', 0)
    cached = (usage.get('input_tokens_details') or {}).get('cached_tokens', 0)
    return ((inputs*ir*1.25 if conservative else (inputs-cached)*ir + cached*cr) + outputs*outr) / 1e6


def reserved_cost(entry):
    return cost({'input_tokens': entry['input_bound'], 'output_tokens': MAX_OUTPUT}, entry['model'], True)


def exposure(ledger):
    return sum(cost(a['usage'], a['model'], True) if a['state'] == 'completed' else reserved_cost(a) for a in ledger)


def reserve(ledger, entry):
    if any(a['state'] == 'failed_or_uncertain' for a in ledger): raise ValueError('Unreconciled attempt; stop paid work')
    if len(ledger) >= MAX_ATTEMPTS or exposure(ledger) + reserved_cost(entry) > CAP:
        raise ValueError('Hard campaign cap reached before next request')


def stage_protocol(stage, jobs, selection):
    freeze(OUT / stage / 'protocol.json', {'stage': stage, 'schedule': jobs,
        'selection_sha256': sha(selection), 'campaign_sha256': sha(OUT / 'protocol.json'), 'engine_sha256': sha(__file__)})


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    available = {x['id'] for x in api('models', credential())['data']}
    missing = set(a['model'] for a in ARMS.values()) - available
    if missing: raise ValueError('Requested models unavailable: ' + ', '.join(sorted(missing)))
    protocol = {
        'authorized': 'User requested autonomous direct visual xG experiments, other models allowed, $10 limits; 29 October 2026 America/Los_Angeles.',
        'scope': 'Image + generic fixed instructions -> VLM -> raw xG. No calibration, probability head, baseline fusion, numeric query answers, examples, tools, fine tuning or averaging.',
        'arms': ARMS, 'prompts': prompts(), 'schema': SCHEMA, 'max_output_tokens': MAX_OUTPUT,
        'image_detail': 'high', 'service_tier': 'default', 'new_combined_cap_usd': CAP,
        'attempt_cap': MAX_ATTEMPTS, 'concurrency': 3, 'automatic_retries': 0,
        'prices_per_million': RATES, 'conservative_input_multiplier': 1.25,
        'pricing_sources': ['https://developers.openai.com/api/docs/models/' + m for m in RATES],
        'providers': {'openai': 'saved credential available', 'anthropic': 'not configured; no calls', 'gemini': 'not configured; no calls'},
        'budget': 'New round capped at $3 combined. Prior direct campaign conservative estimate $0.1049749. Other older records/account activity not fully bill-reconciled; no verified account-balance claim.',
        'screen': '50 previously used AFCON development shots, identical saved image bytes. New paired controls. All 5 arms fixed before inference. Reused development, not untouched validation.',
        'novelty': 'New Terra/Sol models and explicit missing-information marginalization, without numeric anchors. Prior effort, marker, grid/zoom, geometry-first and examples remain historical.',
        'selection': 'After complete fixed screen, select minimum raw Brier among all 5 arms; ties in listed order. Original Luna may win. No significance claim from selection data.',
        'confirmation': '200 natural-sample non-penalty freeze-frame shots from 40 outcome-blind randomly selected unused La Liga 2015/16 match groups. Compare winner against newly called original Luna and frozen 4-feature logistic. Separate StatsBomb/no-header logistic references.',
        'primary': 'Winner minus 4-feature logistic and winner minus original Luna Brier. Two 97.5% whole-match bootstrap intervals. Crossing zero does not prove equivalence. No test-set tuning/extensions.',
        'success': 'Independent evidence must support any accuracy claim. Report AUC/log loss, bias, failures and cost even if negative; no commitment to a preferred conclusion.',
        'stop': 'Stop dispatch on uncertain charge, invalid/incomplete output, wrong model, exhausted credit or cap. Save in-flight calls. No replacement shots or partial success claims.',
        'engine_sha256': sha(__file__), 'original_source_sha256': sha(ROOT / 'xg_vlm_pilot.py')}
    freeze(OUT / 'protocol.json', protocol)
    dev = read(ROOT / 'results/direct_scope_suite/reasoning/selection.json'); freeze(OUT / 'screen/selection.json', dev)
    jobs = []; rng = random.Random(2026103001)
    for row in dev:
        image = ROOT / 'results/direct_scope_suite/images/reasoning' / (row['shot_id'] + '.png')
        arms = list(ARMS); rng.shuffle(arms)
        jobs.extend({'shot_id': row['shot_id'], 'arm': a, 'image': str(image.relative_to(ROOT)), 'image_sha256': sha(image)} for a in arms)
    stage_protocol('screen', jobs, OUT / 'screen/selection.json')
    print('Frozen 50-shot, 5-arm screen (250 calls); no paid inference.', flush=True)


def prepare_fresh():
    if (OUT / 'attempts.json').exists(): raise ValueError('Freeze fresh cohort before any paid inference')
    import sys
    sys.path.insert(0, str(ROOT))
    import xg_vlm_pilot as original
    import pandas as pd
    prior = read(ROOT / 'results/direct_scope_suite/data/match_split.json')
    used = set(prior['used_match_ids'] + prior['development_match_ids'] + prior['confirmation_match_ids']); sources = {}
    for root in [ROOT / 'results', Path('/mnt/c/Users/ephou/Downloads/xG/output')]:
        for pattern in ('*selection*.csv', '*predictions*.csv', 'shots*.csv', '*selection*.jsonl'):
            for path in root.rglob(pattern):
                if OUT in path.parents: continue
                try:
                    if path.suffix == '.csv':
                        with path.open(encoding='utf-8-sig', newline='') as handle:
                            values = [int(float(r['match_id'])) for r in csv.DictReader(handle) if r.get('match_id')]
                    else: values = [int(json.loads(line)['match_id']) for line in path.read_text().splitlines() if line.strip()]
                    used.update(values); sources[str(path)] = sha(path)
                except (KeyError, ValueError): continue
    matches = public('matches/11/27.json')
    eligible = sorted([m['match_id'] for m in matches if int(m['match_id']) not in used])
    rng = random.Random(2026103002); selected_matches = sorted(rng.sample(eligible, 40))
    freeze(OUT / 'data/match_split.json', {'competition_id': 11, 'season_id': 27, 'season': '2015/2016',
        'all_matches': len(matches), 'previous_used_match_ids': sorted(used), 'selected_match_ids': selected_matches,
        'prior_table_hashes': sources, 'prior_split_sha256': sha(ROOT / 'results/direct_scope_suite/data/match_split.json'),
        'rule': 'Sample 40 eligible match IDs before events/outcome inspection, then 200 eligible shots without outcome stratification.'})
    rows = []
    for i, mid in enumerate(selected_matches, 1):
        for event in public(f'events/{mid}.json'):
            shot = event.get('shot', {}); ff = shot.get('freeze_frame')
            if event.get('type', {}).get('name') != 'Shot' or shot.get('type', {}).get('name') == 'Penalty' or not isinstance(ff, list): continue
            x, y = event['location'][:2]; distance, angle = original.geometry(x, y)
            cone = sum(original.in_triangle(p['location'][:2], (x, y), original.POST_L, original.POST_R) for p in ff if not p['teammate'])
            row = {'match_id': int(mid), 'shot_id': event['id'], 'x': x, 'y': y, 'ff': ff,
                'goal': int(shot['outcome']['name'] == 'Goal'), 'sb_xg': shot['statsbomb_xg'], 'distance': distance, 'angle': angle,
                'header': int(shot.get('body_part', {}).get('name') == 'Head'), 'defenders_in_cone': cone}
            def prediction(features, coeff, intercept):
                z = intercept[0] + sum(row[k]*v for k,v in zip(features, coeff[0])); return 1/(1+math.exp(-z))
            row['baseline_xg'] = prediction(prior['logistic_features'], prior['logistic_coefficients'], prior['logistic_intercept'])
            row['baseline_image_xg'] = prediction(prior['matched_image_features'], prior['matched_coefficients'], prior['matched_intercept'])
            rows.append(row)
        if i % 10 == 0: print(f'Fresh public match data: {i}/40', flush=True)
    if len({r['shot_id'] for r in rows}) != len(rows) or len(rows) < 200: raise ValueError('Insufficient or duplicate fresh shots')
    selected = rng.sample(sorted(rows, key=lambda r:r['shot_id']), 200); freeze(OUT / 'confirmation/selection.json', selected)
    images = OUT / 'images/confirmation'; images.mkdir(parents=True, exist_ok=True)
    for row in selected:
        path = images / (row['shot_id'] + '.png')
        if not path.exists(): original.render(pd.Series(row), path)
    freeze(OUT / 'data/provenance.json', {'eligible_shots': len(rows), 'selected_shots': 200,
        'event_file_hashes': {p.name:sha(p) for p in (OUT / 'data').glob('events_*.json')},
        'fresh_match_overlap': sorted(set(selected_matches) & used), 'renderer_source_sha256': sha(ROOT / 'xg_vlm_pilot.py'),
        'baseline': 'Exact frozen coefficients fitted on WC2018/2022 3,068 shots; no refitting or new labels.'})
    print('Frozen independent 200-shot cohort; no accuracy scoring or paid inference.', flush=True)


def status(ledger, stage):
    entries = [a for a in ledger if a['stage'] == stage]
    return {'stage': stage, 'completed': sum(a['state'] == 'completed' for a in entries),
        'attempts': len(entries), 'planned': len(read(OUT / stage / 'protocol.json')['schedule']),
        'estimated_usd': sum(cost(a.get('usage', {}), a['model']) for a in ledger),
        'conservative_exposure_usd': exposure(ledger), 'cap_usd': CAP,
        'failures': [a['error'] for a in ledger if a.get('error')]}


def run(stage):
    protocol = read(OUT / 'protocol.json')
    if not (OUT / 'data/provenance.json').exists(): raise ValueError('Freeze independent data before inference')
    if protocol['engine_sha256'] != sha(__file__): raise ValueError('Frozen inference engine changed')
    sp = read(OUT / stage / 'protocol.json')
    if (sp['engine_sha256'] != sha(__file__) or sp['campaign_sha256'] != sha(OUT / 'protocol.json')
        or sp['selection_sha256'] != sha(OUT / stage / 'selection.json')): raise ValueError('Frozen protocol or selection changed')
    if stage == 'confirmation' and read(OUT / stage / 'choice_hash.json')['choice_sha256'] != sha(OUT / 'choice.json'):
        raise ValueError('Frozen selected method changed')
    ledger = read(OUT / 'attempts.json', [])
    if any(a['state'] != 'completed' for a in ledger): raise ValueError('Prior attempt unresolved; no automatic retry')
    done = [a for a in ledger if a['stage'] == stage]
    if [a['job'] for a in done] != sp['schedule'][:len(done)]: raise ValueError('Saved schedule changed')
    key = credential(); counts = read(OUT / 'token_counts.json', {})
    lock = OUT / 'generation.lock'; fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    mutex = threading.Lock()

    def execute(entry, payload):
        began = time.monotonic(); updates = {}
        try:
            response = api('responses', key, payload); save(OUT / entry['response_file'], response)
            updates['usage'] = response.get('usage') or {}
            updates['answer'] = validate(response, ARMS[entry['job']['arm']])
            if updates['usage']['input_tokens'] > entry['input_bound'] or updates['usage']['output_tokens'] > MAX_OUTPUT:
                raise ValueError('Reserved token bound exceeded')
            updates.update(returned_model=response['model'], state='completed')
        except Exception as error: updates.update(state='failed_or_uncertain', error=str(error))
        updates['seconds'] = round(time.monotonic() - began, 3)
        with mutex:
            entry.update(updates); save(OUT / 'attempts.json', ledger); save(OUT / 'status.json', status(ledger, stage))
        return entry['state'] == 'completed'

    queue = iter(enumerate(sp['schedule'][len(done):], len(done)+1)); pending = set(); stopped = False
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            exhausted = False
            while pending or not exhausted:
                while len(pending) < 3 and not exhausted and not stopped:
                    try: index, job = next(queue)
                    except StopIteration:
                        exhausted = True; break
                    image = ROOT / job['image']
                    if sha(image) != job['image_sha256']: raise ValueError('Frozen image changed')
                    arm = ARMS[job['arm']]; payload = body(arm, protocol['prompts'][arm['prompt']], image.read_bytes())
                    countkey = digest({'image': job['image_sha256'], 'model': arm['model'], 'prompt': arm['prompt']})
                    if countkey not in counts:
                        counted = api('responses/input_tokens', key, {k:payload[k] for k in ('model','input','text')})
                        n = counted.get('input_tokens')
                        if type(n) is not int or not 0 < n < 10000: raise ValueError('Input token bound unavailable')
                        counts[countkey] = n; save(OUT / 'token_counts.json', counts)
                    entry = {'stage': stage, 'index': index, 'job': job, 'model': arm['model'], 'state': 'started',
                        'input_bound': counts[countkey] + 1024, 'response_file': f'{stage}/response_{index:04d}.json',
                        'request_sha256': digest(payload), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                    with mutex:
                        reserve(ledger, entry); ledger.append(entry); save(OUT / 'attempts.json', ledger)
                    pending.add(pool.submit(execute, entry, payload))
                if stopped: exhausted = True
                if not pending: break
                finished, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                if not all(f.result() for f in finished): stopped = True
                current = status(ledger, stage)
                if current['completed'] % 10 == 0 or not pending or stopped: print(json.dumps(current), flush=True)
        current = status(ledger, stage); save(OUT / 'status.json', current)
        if current['completed'] != current['planned'] or current['failures']:
            raise ValueError('Stage incomplete; all attempts preserved, no accuracy claim')
        save(OUT / stage / 'complete.json', current)
        print('Fixed stage complete. No automatic extension.', flush=True)
    finally: lock.unlink()


def metrics(y, p):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p); safe = np.clip(p, 1e-15, 1-1e-15)
    return {'brier': float(np.mean((p-y)**2)),
        'log_loss': float(-np.mean(y*np.log(safe)+(1-y)*np.log(1-safe))),
        'auc': float(roc_auc_score(y,p)) if len(set(y)) == 2 else None, 'mean_xg': float(p.mean())}


def comparison(rows, first, second, level=.975):
    import numpy as np
    groups = sorted({r['match_id'] for r in rows})
    losses = {g:np.array([(r[first]-r['goal'])**2-(r[second]-r['goal'])**2 for r in rows if r['match_id']==g]) for g in groups}
    rng = np.random.default_rng(2026103005)
    boots = [np.mean(np.concatenate([losses[g] for g in rng.choice(groups,len(groups))])) for _ in range(10000)]
    return {'first': first, 'second': second, 'difference': float(np.mean(np.concatenate(list(losses.values())))),
        'confidence_level': level, 'interval': np.quantile(boots,[(1-level)/2,1-(1-level)/2]).tolist()}


def scored(stage):
    sp = read(OUT / stage / 'protocol.json'); entries = [a for a in read(OUT / 'attempts.json',[]) if a['stage']==stage]
    if len(entries) != len(sp['schedule']) or any(a['state']!='completed' for a in entries):
        raise ValueError('Only complete frozen stages may be scored')
    answers = {}
    for entry, job in zip(entries,sp['schedule']):
        raw = read(OUT / entry['response_file'])
        if (entry['job'] != job or validate(raw, ARMS[job['arm']]) != entry['answer']
            or raw['usage'] != entry['usage'] or sha(ROOT/job['image']) != job['image_sha256']):
            raise ValueError('Saved inference integrity mismatch')
        answers[job['shot_id'],job['arm']] = entry['answer']
    arms = list(dict.fromkeys(j['arm'] for j in sp['schedule'])); rows = []
    for source in read(OUT / stage / 'selection.json'):
        row = {k:source[k] for k in ('shot_id','match_id','goal','distance','angle','header','defenders_in_cone','baseline_xg','baseline_image_xg','sb_xg')}
        for arm in arms:
            row[arm] = answers[row['shot_id'],arm]['xg']; row[arm+'_reason'] = answers[row['shot_id'],arm]['reason']
        rows.append(row)
    methods = arms+['baseline_xg','baseline_image_xg','sb_xg']
    scores = {m:metrics([r['goal'] for r in rows],[r[m] for r in rows]) for m in methods}
    with (OUT/stage/'predictions.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {'shots':len(rows),'goals':sum(r['goal'] for r in rows),'matches':len({r['match_id'] for r in rows}),'scores':scores}
    if stage == 'screen':
        selected = min(ARMS,key=lambda a:(scores[a]['brier'],list(ARMS).index(a)))
        freeze(OUT/'choice.json',{'arm':selected,'settings':ARMS[selected],
            'rule':'Lowest complete-screen raw Brier; ties in prespecified arm order',
            'development_selection_sha256':sha(OUT/'screen/selection.json'),'screen_predictions_sha256':sha(OUT/'screen/predictions.csv')})
        result['selected'] = selected
    else:
        selected = read(OUT/'choice.json')['arm']; result['primary'] = [comparison(rows,selected,'baseline_xg')]
        if selected != 'luna_original': result['primary'].append(comparison(rows,selected,'luna_original'))
        result['secondary_no_header'] = comparison(rows,selected,'baseline_image_xg',.95)
        result['secondary_statsbomb'] = comparison(rows,selected,'sb_xg',.95)
    save(OUT/stage/'summary.json',result); print(json.dumps(result,indent=2),flush=True)
    return rows,result


def confirm():
    selected = read(OUT/'choice.json')['arm']; arms = ['luna_original']+([] if selected=='luna_original' else [selected])
    rng = random.Random(2026103004); jobs = []
    for row in read(OUT/'confirmation/selection.json'):
        image = OUT/'images/confirmation'/(row['shot_id']+'.png'); order = arms.copy(); rng.shuffle(order)
        jobs.extend({'shot_id':row['shot_id'],'arm':a,'image':str(image.relative_to(ROOT)),'image_sha256':sha(image)} for a in order)
    stage_protocol('confirmation',jobs,OUT/'confirmation/selection.json')
    freeze(OUT/'confirmation/choice_hash.json',{'choice_sha256':sha(OUT/'choice.json')})
    print(f'Frozen independent comparison: {selected}; {len(jobs)} calls. No inference.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','prepare-fresh','run-screen','score-screen','confirm','run-confirmation','score-confirmation'])
    action = parser.parse_args().action
    if action.startswith('run-'): run(action.split('-',1)[1])
    elif action.startswith('score-'): scored(action.split('-',1)[1])
    else: {'prepare':prepare,'prepare-fresh':prepare_fresh,'confirm':confirm}[action]()
