"""Offline scoring and prespecified development decisions; never invokes a model."""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import common as c

def completed(name):
    protocol=c.read(c.OUT/name/'protocol.json'); ledger=[a for a in c.read(c.OUT/'attempts.json',[]) if a['stage']==name]
    if len(ledger)!=len(protocol['schedule']) or any(a['state']!='completed' for a in ledger): raise ValueError('Full frozen stage required; no partial accuracy scoring')
    for a,j in zip(ledger,protocol['schedule']):
        if any(a[k]!=j[k] for k in j): raise ValueError('Schedule mismatch')
        raw=c.read(c.OUT/a['response_file'])
        if c.prior.validate(raw)!=a['answer'] or raw['usage']!=a['usage'] or raw['reasoning']['effort']!=a['effort']: raise ValueError('Response integrity mismatch')
        if c.sha(c.OUT/j['image'])!=j['image_sha256']: raise ValueError('Image hash changed')
    return protocol,ledger

def comparison(rows,a,b,level=.95,seed=20261029):
    groups=sorted({r['match_id'] for r in rows}); rng=np.random.default_rng(seed)
    losses={g:np.array([(float(r[a])-r['goal'])**2-(float(r[b])-r['goal'])**2 for r in rows if r['match_id']==g]) for g in groups}
    boots=np.array([np.mean(np.concatenate([losses[g] for g in rng.choice(groups,len(groups))])) for _ in range(6000)])
    tail=(1-level)/2
    return {'first':a,'second':b,'brier_difference':float(np.mean(np.concatenate(list(losses.values())))),
            'confidence_level':level,'interval':np.quantile(boots,[tail,1-tail]).tolist(),
            'interpretation':'negative favors first; whole-match resampling; conditions on fixed predictions'}

def score(name):
    protocol,ledger=completed(name); source=c.read(c.OUT/name/'selection.json'); lookup={r['shot_id']:r for r in source}
    conditions=list(dict.fromkeys(j['condition'] for j in protocol['schedule'])); aliases=protocol['metadata'].get('aliases',{}); answers={(a['shot_id'],a['condition']):a for a in ledger}; rows=[]
    for r in source:
        out={k:r[k] for k in ('shot_id','match_id','goal','distance','angle','header','defenders_in_cone','baseline_xg','sb_xg') if k in r}
        if 'baseline_image_xg' in r: out['baseline_image_xg']=r['baseline_image_xg']
        for arm in conditions:
            entry=answers[r['shot_id'],arm]; out[arm]=entry['answer']['xg']; out[arm+'_reason']=entry['answer']['reason']
        for alias,target in aliases.items(): out[alias]=out[target]; out[alias+'_reason']=out[target+'_reason']
        rows.append(out)
    with (c.OUT/name/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    conditions+=list(aliases)
    methods=conditions+['baseline_xg','sb_xg']+(['baseline_image_xg'] if all('baseline_image_xg' in r for r in rows) else [])
    scores={k:c.metrics([r['goal'] for r in rows],[r[k] for r in rows]) for k in methods}
    result={'stage':name,'shots':len(rows),'goals':sum(r['goal'] for r in rows),'matches':len({r['match_id'] for r in rows}),
            'scores':scores,'usage':c.usage(ledger),'conditions':conditions}
    if name=='reasoning':
        result['primary']=comparison(rows,'medium','low')
        result['secondary_high_vs_medium']=comparison(rows,'high','medium')
        effort=min(('low','medium','high'),key=lambda e:(scores[e]['brier'],('low','medium','high').index(e)))
        c.freeze(c.OUT/'reasoning_choice.json',{'effort':effort,'rule':'Lowest 50-shot development Brier; ties favor lower effort','scores_sha256':None,'selection_sha256':c.sha(c.OUT/name/'selection.json')})
        result['selected_effort']=effort
    elif name=='rendering':
        result['primary']=comparison(rows,'outline','standard')
        choice=c.read(c.OUT/'reasoning_choice.json')['effort']; use_outline=scores['outline']['brier']<scores['standard']['brier']
        result['selected_renderer']='outline' if use_outline else 'standard'
        c.freeze(c.OUT/'candidate.json',{'effort':choice,'renderer':result['selected_renderer'],
            'rule':'Effort selected on AFCON development; outline used only if its paired WC development Brier is lower. No confirmation outcomes used.',
            'reasoning_selection_sha256':c.sha(c.OUT/'reasoning/selection.json'),'rendering_selection_sha256':c.sha(c.OUT/'rendering/selection.json')})
    elif name=='confirmation':
        result['primary']=[comparison(rows,'candidate','baseline_xg',.975),comparison(rows,'candidate','original_low',.975)]
        if 'standard_selected' in conditions: result['secondary_renderer']=comparison(rows,'candidate','standard_selected')
        result['secondary_no_header']=comparison(rows,'candidate','baseline_image_xg')
        result['candidate']=c.read(c.OUT/'candidate.json')
    c.save(c.OUT/name/'summary.json',result)
    lines=['# '+name.replace('_',' ').title(),' ',f"{result['shots']} shots, {result['goals']} goals, {result['matches']} matches.",'',
           '| Method | Brier (lower) | AUC (higher) | Log loss (lower) | Mean xG |','|---|---:|---:|---:|---:|']
    for k,s in scores.items():
        auc=f"{s['auc']:.3f}" if s['auc'] is not None else 'undefined'
        lines.append(f"| {k} | {s['brier']:.5f} | {auc} | {s['log_loss']:.5f} | {s['mean_xg']:.1%} |")
    tests=result['primary'] if isinstance(result['primary'],list) else [result['primary']]
    lines+=['','| Primary comparison | Brier difference | Match interval |','|---|---:|---|']
    for t in tests: lines.append(f"| {t['first']} minus {t['second']} | {t['brier_difference']:+.5f} | {100*t['confidence_level']:.1f}%: [{t['interval'][0]:+.5f}, {t['interval'][1]:+.5f}] |")
    lines+=['',f"Estimated stage inference cost: ${result['usage']['estimated_usd']:.6f}. Reasoning tokens are included in output-token charges.",'',
        'This is direct VLM probability prediction. Raw values are unadjusted. The four-feature baseline knows the header flag; the body-part-free comparator is listed separately where available. Both numerical baselines use recorded coordinates. No numerical baseline values or body-part flags enter the image-only model inputs.']
    if name=='confirmation': lines+=['','Two primary comparisons use 97.5% intervals to account for multiplicity. Development choices were frozen before these match-disjoint confirmation outcomes were scored. These intervals condition on the fixed fitted baselines; one tournament does not establish universal performance.']
    else: lines+=['','Development evidence only. A selected winner is not confirmed by the scores used to select it. Secondary comparisons are exploratory.']
    (c.OUT/name/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


def invariance():
    _,ledger=completed('invariance'); original=[a for a in c.read(c.OUT/'attempts.json',[]) if a['stage']=='rendering' and a['condition']=='standard']
    selected=c.read(c.OUT/'invariance/selection.json'); baseline={a['shot_id']:a['answer']['xg'] for a in original}; entries={(a['shot_id'],a['condition']):a for a in ledger}
    rows=[]
    for r in selected:
        p=baseline[r['shot_id']]; repeat=entries[r['shot_id'],'repeat']['answer']['xg']; mirror=entries[r['shot_id'],'mirror']['answer']['xg']
        rows.append({'shot_id':r['shot_id'],'match_id':r['match_id'],'original':p,'repeat':repeat,'mirror':mirror,'repeat_absolute_change':abs(repeat-p),'mirror_absolute_change':abs(mirror-p)})
    groups=sorted({r['match_id'] for r in rows}); rng=np.random.default_rng(20261029)
    changes={g:[r['mirror_absolute_change']-r['repeat_absolute_change'] for r in rows if r['match_id']==g] for g in groups}
    boot=[np.mean([v for g in rng.choice(groups,len(groups)) for v in changes[g]]) for _ in range(6000)]
    result={'shots':len(rows),'mean_repeat_change':float(np.mean([r['repeat_absolute_change'] for r in rows])),
            'mean_mirror_change':float(np.mean([r['mirror_absolute_change'] for r in rows])),
            'mirror_minus_repeat_interval_95':np.quantile(boot,[.025,.975]).tolist(),'usage':c.usage(ledger),
            'interpretation':'Mirroring adds no new outcome labels. Compare its effects to ordinary repeated-call variability; do not treat transformed views as independent test shots.'}
    c.save(c.OUT/'invariance/summary.json',result); c.save(c.OUT/'invariance/observations.json',rows); print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['reasoning','rendering','confirmation','invariance']); args=ap.parse_args()
    invariance() if args.stage=='invariance' else score(args.stage)

