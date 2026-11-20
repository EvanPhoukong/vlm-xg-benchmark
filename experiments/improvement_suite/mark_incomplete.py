"""Record the quota-stopped secondary trial without scoring a partial cohort."""
import json
from datetime import datetime, timezone
import pandas as pd
from vision_trials import OUT, disagreement, consensus


def cache(path):
    return {e['shot_id']: e for e in [json.loads(line) for line in path.read_text().splitlines()]} if path.exists() else {}


def main():
    root = OUT/'confirmation_bundesliga'; folder = root/'best_vision'
    selected = pd.read_json(root/'selection.jsonl', lines=True)
    original = cache(root/'inference/baseline_cache.jsonl')
    second, third = cache(folder/'best_second_cache.jsonl'), cache(folder/'best_third_cache.jsonl')
    complete = []
    for shot_id in selected.shot_id:
        if shot_id not in second:
            break
        readings = [original[shot_id]['answer'], second[shot_id]['answer']]
        if disagreement(*readings) > 2:
            if shot_id not in third:
                break
            readings.append(third[shot_id]['answer'])
        complete.append({'shot_id': shot_id, 'answer': consensus(readings)})
    status = {'status': 'incomplete', 'reason': 'API credit balance exhausted', 'api_error_code': 'credit_balance_exhausted',
              'completed_shots': len(complete), 'target_shots': len(selected),
              'successful_second_readings': len(second), 'successful_third_readings': len(third),
              'recorded_utc': datetime.now(timezone.utc).isoformat(),
              'partial_cohort_scored': False, 'resume': 'After credits are available, rerun unchanged confirm_best_vision.py to reuse the cache.'}
    (folder/'partial_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    (folder/'partial_answers.json').write_text(json.dumps(complete), encoding='utf-8')
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
