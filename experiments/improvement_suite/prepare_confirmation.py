"""Reserve fresh Bundesliga matches before scoring any candidate on them."""
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd
import requests
from fetch_context import pre_shot_context
from vision_trials import OUT, ROOT, freeze, REPO


def main():
    folder = OUT/'confirmation_bundesliga'; folder.mkdir(parents=True, exist_ok=True)
    protocol = {'competition_id': 9, 'season_id': 281, 'name': 'Bundesliga 2023/24',
                'matches': 34, 'shots': 200, 'seed': 20261023,
                'purpose': 'Fresh matches for frozen noise-trained VLM hybrid and pixel control; never retune models using this cohort.',
                'selection': 'All 34 publicly available matches (Bayer Leverkusen fixtures), then 200 eligible shot IDs without outcome stratification.',
                'coverage_limit': 'This is the public Leverkusen match subset, not a full-league random sample. Initial 40-match preparation stopped before loading events because only 34 matches exist.',
                'image_condition': 'Unchanged single-panel renderer to isolate noise training.',
                'head_files_sha256': hashlib.sha256((OUT.parent/'noise-robustness-v1/frozen_models.joblib').read_bytes()).hexdigest(),
                'context_experiment': 'New input set examined separately, never called image-only.'}
    freeze(folder, protocol)
    if (folder/'selection.jsonl').exists():
        print('Fresh cohort already prepared; no candidate scoring performed.', flush=True); return
    matches_url = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/9/281.json'
    response = requests.get(matches_url, timeout=45); response.raise_for_status()
    matches = pd.DataFrame(response.json())
    if len(matches) != 34:
        raise ValueError('Public match coverage changed; recheck the fixed protocol.')
    chosen = matches.sort_values('match_id').match_id.astype(int).tolist()
    previous = set()
    for name in ['wc2018', 'wc2022', 'euro2024', 'euro2020']:
        previous.update(pd.read_json(ROOT/'data'/f'{name}.jsonl', lines=True).match_id.astype(int))
    if previous & set(chosen):
        raise ValueError('Confirmation matches were already used.')
    (folder/'match_selection.json').write_text(json.dumps(chosen), encoding='utf-8')
    sys.path.insert(0, str(REPO))
    import xg_vlm_pilot as pilot
    (folder/'match_shots').mkdir(exist_ok=True)
    context_rows, source_hashes = [], []
    for i, mid in enumerate(chosen, 1):
        path = folder/'match_shots'/f'{mid}.json'
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            url = f'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{mid}.json'
            response = requests.get(url, timeout=60); response.raise_for_status()
            events = response.json(); rows = []
            for event in events:
                shot = event.get('shot', {})
                ff = shot.get('freeze_frame')
                if (event.get('type') or {}).get('name') != 'Shot' or (shot.get('type') or {}).get('name') == 'Penalty' or not isinstance(ff, list):
                    continue
                x, y = event['location'][:2]
                distance, angle = pilot.geometry(x, y)
                cone = sum(pilot.in_triangle(p['location'][:2], (x, y), pilot.POST_L, pilot.POST_R) for p in ff if not p['teammate'])
                rows.append({'match_id': mid, 'shot_id': event['id'], 'x': x, 'y': y, 'ff': ff,
                             'goal': int((shot.get('outcome') or {}).get('name') == 'Goal'), 'sb_xg': shot['statsbomb_xg'],
                             'body_part': (shot.get('body_part') or {}).get('name'),
                             'header': int((shot.get('body_part') or {}).get('name') == 'Head'),
                             'play_pattern': (event.get('play_pattern') or {}).get('name'),
                             'distance': distance, 'angle': angle, 'defenders_in_cone': cone})
            payload = {'shots': rows, 'context': pre_shot_context(events), 'source': url,
                       'sha256': hashlib.sha256(response.content).hexdigest()}
            path.write_text(json.dumps(payload), encoding='utf-8')
        context_rows.extend(payload['context'])
        source_hashes.append({'match_id': mid, 'url': payload['source'], 'sha256': payload['sha256']})
        print(f'Fresh match preparation {i}/34', flush=True)
    frame = pd.DataFrame([row for mid in chosen for row in json.loads((folder/'match_shots'/f'{mid}.json').read_text())['shots']])
    if not frame.shot_id.is_unique or len(frame) < 200:
        raise ValueError('Invalid fresh shot pool.')
    selected = frame.sort_values('shot_id').sample(n=200, random_state=20261023)
    selected.to_json(folder/'selection.jsonl', orient='records', lines=True, double_precision=15)
    pd.DataFrame(context_rows).to_csv(folder/'context.csv', index=False)
    (folder/'sources.json').write_text(json.dumps(source_hashes, indent=2))
    print(f'Prepared {len(selected)} fresh shot images for subsequent frozen evaluation; no outcomes summarized.', flush=True)


if __name__ == '__main__':
    main()
