"""Score already-returned direct xG as a secondary prompt diagnostic."""
import json
import pandas as pd
import robust_xg as robust
from vision_trials import ROOT, OUT


def main():
    selection = pd.read_json(ROOT/'external_vlm_selection.jsonl', lines=True, precise_float=True)
    old = pd.read_csv(ROOT/'rich_vlm_external/predictions.csv').set_index('shot_id').loc[selection.shot_id]
    multiple = json.loads((OUT/'vision/multiview_answers.json').read_text())
    cache = {e['shot_id']: e['answer'] for e in [json.loads(line) for line in (OUT/'vision/fewshot_cache.jsonl').read_text().splitlines()]}
    if len(multiple) != 200 or set(cache) != set(selection.shot_id):
        raise ValueError('Direct-prompt diagnostics require complete cohorts.')
    result = selection[['shot_id', 'match_id', 'goal']].copy().reset_index(drop=True)
    result['original_direct'] = old.vlm_direct.to_numpy()
    result['multiview_direct'] = [a['xg'] for a in multiple]
    result['fewshot_direct'] = [cache[s]['xg'] for s in selection.shot_id]
    result.to_csv(OUT/'vision/direct_predictions.csv', index=False)
    scores = pd.DataFrame([{'method': name, **robust.metrics(result.goal, result[name])} for name in result.columns[3:]])
    scores.to_csv(OUT/'vision/direct_scores.csv', index=False)
    pd.DataFrame([robust.comparison(result, name, 'original_direct') for name in ['multiview_direct', 'fewshot_direct']]).to_csv(OUT/'vision/direct_comparisons.csv', index=False)
    print(scores.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
