"""Save only pre-shot event context from official StatsBomb open event files."""
import hashlib
import json
from pathlib import Path
import time
import pandas as pd
import requests

ROOT = Path('/mnt/c/Users/ephou/Downloads/xG/output/research-loop')
OUT = Path('/mnt/c/Users/ephou/Downloads/xG/output/improvement-suite/context')
CATEGORICAL = ['shot_type', 'play_pattern', 'pass_height']
NUMERIC = ['under_pressure', 'prior_pass', 'pass_cross', 'pass_cutback', 'pass_through_ball', 'pass_seconds_before']


def seconds(event):
    parts = event['timestamp'].split(':')
    return float(parts[0])*3600+float(parts[1])*60+float(parts[2])


def pre_shot_context(events):
    """Only events already observed at the shot can supply context."""
    passes, rows = {}, []
    for event in sorted(events, key=lambda e: e.get('index', 0)):
        key = (event.get('period'), event.get('possession'), (event.get('team') or {}).get('id'))
        kind = (event.get('type') or {}).get('name')
        if kind == 'Shot':
            prior = passes.get(key)
            delta = seconds(event)-seconds(prior) if prior is not None else None
            prior = prior if delta is not None and 0 <= delta <= 15 else None
            p = prior.get('pass', {}) if prior else {}
            rows.append({'shot_id': event['id'], 'shot_type': (event.get('shot', {}).get('type') or {}).get('name', 'Unknown'),
                         'play_pattern': (event.get('play_pattern') or {}).get('name', 'Unknown'),
                         'under_pressure': int(bool(event.get('under_pressure', False))),
                         'prior_pass': int(prior is not None), 'pass_cross': int(bool(p.get('cross', False))),
                         'pass_cutback': int(bool(p.get('cut_back', False))),
                         'pass_through_ball': int(bool(p.get('through_ball', False)) or (p.get('technique') or {}).get('name') == 'Through Ball'),
                         'pass_height': (p.get('height') or {}).get('name', 'No recent pass'),
                         'pass_seconds_before': delta if prior else None})
        elif kind == 'Pass':
            passes[key] = event
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'matches').mkdir(exist_ok=True)
    ids = set()
    for name in ['wc2018', 'wc2022', 'euro2024', 'euro2020']:
        ids.update(pd.read_json(ROOT/'data'/f'{name}.jsonl', lines=True).match_id.astype(int))
    with requests.Session() as session:
        for i, match_id in enumerate(sorted(ids), 1):
            path = OUT/'matches'/f'{match_id}.json'
            if not path.exists():
                url = f'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json'
                for attempt in range(4):
                    try:
                        response = session.get(url, timeout=45); response.raise_for_status()
                        events = response.json()
                        payload = {'match_id': match_id, 'source': url, 'sha256': hashlib.sha256(response.content).hexdigest(),
                                   'shots': pre_shot_context(events)}
                        path.write_text(json.dumps(payload), encoding='utf-8')
                        break
                    except requests.RequestException:
                        if attempt == 3:
                            raise
                        time.sleep(2**attempt)
            print(f'Pre-shot context {i}/{len(ids)} matches', flush=True)
    rows, sources = [], []
    for match_id in sorted(ids):
        payload = json.loads((OUT/'matches'/f'{match_id}.json').read_text())
        rows.extend([{**row, 'match_id': match_id} for row in payload['shots']])
        sources.append({k: payload[k] for k in ['match_id', 'source', 'sha256']})
    frame = pd.DataFrame(rows)
    if not frame.shot_id.is_unique:
        raise ValueError('Duplicate context IDs.')
    frame.to_csv(OUT/'pre_shot_context.csv', index=False)
    (OUT/'sources.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    print(f'Saved pre-shot context for {len(frame)} shots.', flush=True)


if __name__ == '__main__':
    main()
