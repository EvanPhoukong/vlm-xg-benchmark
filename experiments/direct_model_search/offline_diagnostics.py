"""Post-hoc explanatory comparisons; never adjusts VLM probabilities."""
import csv,json
from pathlib import Path
import numpy as np
import run as r


def main():
    source=Path('/mnt/c/Users/ephou/Downloads/xG/output/research-loop/data')
    frozen=r.read(r.ROOT/'results/direct_scope_suite/data/match_split.json')
    training=[]
    for name in ('wc2018','wc2022'):
        file=source/(name+'.jsonl')
        assert r.sha(file)==frozen['training_source_hashes'][name]
        training.extend(json.loads(line) for line in file.read_text().splitlines() if line.strip())
    prior=sum(x['goal'] for x in training)/len(training)
    with (r.OUT/'confirmation/predictions.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    choice=r.read(r.OUT/'choice.json')['arm']
    for row in rows:
        row['goal']=int(row['goal']); row['match_id']=int(row['match_id'])
        row[choice]=float(row[choice]); row['training_prior']=prior
    groups=sorted({x['match_id'] for x in rows}); rng=np.random.default_rng(2026103007)
    bias={g:np.array([x[choice]-x['goal'] for x in rows if x['match_id']==g]) for g in groups}
    samples=[np.mean(np.concatenate([bias[g] for g in rng.choice(groups,len(groups))])) for _ in range(10000)]
    result={'status':'post_hoc_explanatory_only','purpose':'Detect whether apparently good Brier scores also exceed an uninformative training-only prior. This is a comparator, never a VLM correction.',
        'training_shots':len(training),'training_goals':sum(x['goal'] for x in training),'constant_training_prior':prior,
        'constant_scores':r.metrics([x['goal'] for x in rows],[prior]*len(rows)),
        'candidate_minus_constant':r.comparison(rows,choice,'training_prior',.95),
        'candidate_mean_overprediction':float(np.mean([x[choice]-x['goal'] for x in rows])),
        'post_hoc_match_95_bias_interval':np.quantile(samples,[.025,.975]).tolist(),
        'raw_vlm_predictions_unchanged':True}
    r.save(r.OUT/'post_hoc_diagnostics.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
