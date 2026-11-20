"""Fixed outcome-free visual counterfactuals; raw direct estimates only."""
import argparse, csv, importlib.util, json, os, random, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('model_search',ROOT/'experiments/direct_model_search/run.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
OUT=ROOT/'results/visual_counterfactuals_v1'
STUDY_CAP=.25
ARM_NAMES=['luna_original','luna_uncertainty']


def player(x,y,keeper=False):
    return {'location':[x,y],'teammate':False,'position':{'name':'Goalkeeper' if keeper else 'Center Back'}}


def scene(x,y,keeper=(118,40),defender=None):
    ff=[player(*keeper,keeper=True)]
    if defender: ff.append(player(*defender))
    return {'x':x,'y':y,'ff':ff}


def cases():
    result=[]
    for i,far in enumerate([16,20,24]):
        result.append({'id':f'distance_{i}','factor':'distance','base':scene(120-far,40),
                       'favorable':scene(120-far/2,40),'expected':'Moving the unblocked central shooter closer should increase xG.'})
    for i,dist in enumerate([16,20,24]):
        # Equal shooter-to-goal-center distance; change the viewing angle.
        result.append({'id':f'angle_{i}','factor':'angle','base':scene(120-dist*.6,40+dist*.8),
                       'favorable':scene(120-dist,40),'expected':'A central view opens more goal than an equal-distance wide view.'})
    for i,depth in enumerate([10,14,18]):
        result.append({'id':f'keeper_{i}','factor':'keeper','base':scene(120-depth,40),
                       'favorable':scene(120-depth,40,keeper=(118,58)),
                       'expected':'Displacing the goalkeeper far from the shooting lane should increase xG.'})
    for i,depth in enumerate([10,14,18]):
        result.append({'id':f'blocking_{i}','factor':'blocking','base':scene(120-depth,40,defender=(124-depth,40)),
                       'favorable':scene(120-depth,40,defender=(124-depth,60)),
                       'expected':'Moving the blocking opponent out of the lane should increase xG.'})
    return result


def prepare():
    if (base.OUT/'confirmation/summary.json').exists(): raise ValueError('Freeze before independent accuracy outcomes are scored')
    import pandas as pd
    sys.path.insert(0,str(ROOT)); import xg_vlm_pilot as original
    OUT.mkdir(parents=True,exist_ok=True); rng=random.Random(2026103006); jobs=[]
    definitions=cases(); base.freeze(OUT/'scenes.json',definitions)
    for case in definitions:
        for condition in ('base','favorable'):
            image=OUT/'images'/(case['id']+'_'+condition+'.png'); image.parent.mkdir(exist_ok=True)
            if not image.exists(): original.render(pd.Series(case[condition]),image)
        arms=ARM_NAMES.copy(); rng.shuffle(arms)
        for arm in arms:
            conditions=['base','repeat','favorable']; rng.shuffle(conditions)
            for condition in conditions:
                image=OUT/'images'/(case['id']+'_'+('base' if condition=='repeat' else condition)+'.png')
                jobs.append({'case':case['id'],'arm':arm,'condition':condition,
                             'image':str(image.relative_to(ROOT)),'image_sha256':base.sha(image)})
    protocol={'scope':'Outcome-free synthetic geometry diagnostic. No new real-shot accuracy claim; no corrections, averages or constraints applied to xG.',
        'rationale':'Separate sensitivity to spatial evidence from generic probability-level changes and ordinary repeat variability.',
        'design':'12 fixed pairs: three each for distance, equal-distance angle, keeper displacement and defender blocking. Original and missing-information Luna; favorable/base/exact-repeat calls randomized. Expected favorable minus base positive. Report every reversal/tie and repeat variability.',
        'limitations':'Simplified synthetic diagrams, not new goal-labelled matches. Expectations are qualitative. Cases share templates and are not independent statistical evidence about football accuracy. One repeat per base scene diagnoses variability, not a complete stochastic characterization.',
        'attempt_cap':72,'study_cap_usd':STUDY_CAP,'shared_new_round_cap_usd':base.CAP,
        'arms':{a:base.ARMS[a] for a in ARM_NAMES},'prompts':base.prompts(),'max_output_tokens':base.MAX_OUTPUT,
        'source_sha256':base.sha(ROOT/'xg_vlm_pilot.py'),'engine_sha256':base.sha(__file__),
        'model_engine_sha256':base.sha(base.__file__),'scenes_sha256':base.sha(OUT/'scenes.json'),
        'schedule':jobs,'no_outcome_input':True,'retries':0,
        'timing':'Prepared after development choice but before independent accuracy outcomes are scored. Paid work starts only after the fixed accuracy campaign finishes, sharing its $3 ceiling.'}
    base.freeze(OUT/'protocol.json',protocol)
    print('Frozen 12 visual counterfactuals with exact-repeat controls (72 calls); no paid inference.',flush=True)


def run():
    protocol=base.read(OUT/'protocol.json')
    if protocol['engine_sha256']!=base.sha(__file__) or protocol['model_engine_sha256']!=base.sha(base.__file__): raise ValueError('Frozen code changed')
    main=base.read(base.OUT/'attempts.json',[])
    if (base.OUT/'generation.lock').exists() or not base.read(base.OUT/'confirmation/complete.json') or any(a['state']!='completed' for a in main):
        raise ValueError('Finish and reconcile main campaign first')
    ledger=base.read(OUT/'attempts.json',[])
    if any(a['state']!='completed' for a in ledger): raise ValueError('Unreconciled prior diagnostic attempt')
    if [a['job'] for a in ledger]!=protocol['schedule'][:len(ledger)]: raise ValueError('Saved schedule changed')
    key=base.credential(); lock=OUT/'generation.lock'; fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd)
    counts=base.read(OUT/'token_counts.json',{})
    try:
        for index,job in enumerate(protocol['schedule'][len(ledger):],len(ledger)+1):
            arm=protocol['arms'][job['arm']]; image=ROOT/job['image']
            if base.sha(image)!=job['image_sha256']: raise ValueError('Frozen image changed')
            payload=base.body(arm,protocol['prompts'][arm['prompt']],image.read_bytes())
            countkey=base.digest({'image':job['image_sha256'],'arm':job['arm']})
            if countkey not in counts:
                n=base.api('responses/input_tokens',key,{k:payload[k] for k in ('model','input','text')}).get('input_tokens')
                if type(n) is not int or not 0<n<10000: raise ValueError('Input count unavailable')
                counts[countkey]=n; base.save(OUT/'token_counts.json',counts)
            entry={'job':job,'model':arm['model'],'state':'started','input_bound':counts[countkey]+1024,
                   'response_file':f'response_{index:03d}.json','request_sha256':base.digest(payload),
                   'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            reservation=base.reserved_cost(entry)
            if (len(ledger)>=72 or base.exposure(ledger)+reservation>STUDY_CAP
                or base.exposure(main)+base.exposure(ledger)+reservation>base.CAP): raise ValueError('Shared spending cap reached')
            ledger.append(entry); base.save(OUT/'attempts.json',ledger)
            try:
                response=base.api('responses',key,payload); base.save(OUT/entry['response_file'],response)
                entry['usage']=response.get('usage') or {}; entry['answer']=base.validate(response,arm)
                if entry['usage']['input_tokens']>entry['input_bound'] or entry['usage']['output_tokens']>base.MAX_OUTPUT: raise ValueError('Usage bound exceeded')
                entry['state']='completed'
            except Exception as error:
                entry.update(state='failed_or_uncertain',error=str(error)); base.save(OUT/'attempts.json',ledger); raise
            base.save(OUT/'attempts.json',ledger)
            current={'complete':index,'planned':72,'estimated_usd':sum(base.cost(a['usage'],a['model']) for a in ledger),
                     'shared_conservative_usd':base.exposure(main)+base.exposure(ledger)}
            base.save(OUT/'status.json',current)
            if index%12==0: print(json.dumps(current),flush=True)
        base.save(OUT/'complete.json',current)
    finally: lock.unlink()


def report():
    protocol=base.read(OUT/'protocol.json'); ledger=base.read(OUT/'attempts.json',[])
    if len(ledger)!=72 or any(a['state']!='completed' for a in ledger): raise ValueError('Complete fixed diagnostic required')
    answers={}; response_ids=set()
    for entry,job in zip(ledger,protocol['schedule']):
        raw=base.read(OUT/entry['response_file']); arm=protocol['arms'][job['arm']]
        payload=base.body(arm,protocol['prompts'][arm['prompt']],(ROOT/job['image']).read_bytes())
        assert entry['job']==job and base.validate(raw,arm)==entry['answer'] and raw['usage']==entry['usage']
        assert base.sha(ROOT/job['image'])==job['image_sha256'] and base.digest(payload)==entry['request_sha256']
        assert raw['id'] not in response_ids; response_ids.add(raw['id'])
        answers[job['case'],job['arm'],job['condition']]=entry['answer']['xg']
    rows=[]
    for case in base.read(OUT/'scenes.json'):
        for arm in ARM_NAMES:
            a,b,repeat=[answers[case['id'],arm,c] for c in ('base','favorable','repeat')]
            rows.append({'case':case['id'],'factor':case['factor'],'arm':arm,'base':a,'favorable':b,'repeat':repeat,
                         'favorable_minus_base':b-a,'repeat_minus_base':repeat-a,'expected_direction':b>a})
    with (OUT/'predictions.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result={}
    for arm in ARM_NAMES:
        selected=[r for r in rows if r['arm']==arm]
        result[arm]={'expected_direction_count':sum(r['expected_direction'] for r in selected),'pairs':12,
            'mean_favorable_minus_base':sum(r['favorable_minus_base'] for r in selected)/12,
            'mean_absolute_repeat_change':sum(abs(r['repeat_minus_base']) for r in selected)/12,
            'ties':sum(r['favorable_minus_base']==0 for r in selected),
            'reversals':[r['case'] for r in selected if r['favorable_minus_base']<0]}
    cost=sum(base.cost(a['usage'],a['model']) for a in ledger)
    key=base.credential().encode(); assert not any(key in p.read_bytes() for p in OUT.rglob('*') if p.is_file())
    main=base.read(base.OUT/'attempts.json'); assert base.exposure(main)+base.exposure(ledger)<=base.CAP
    base.save(OUT/'summary.json',{'results':result,'calls':72,'estimated_usd':cost,
        'shared_conservative_usd':base.exposure(main)+base.exposure(ledger),'request_and_response_audit':'passed'})
    lines=['# Does direct xG respond to controlled visual changes?','',
        'Twelve fixed synthetic pairs cover distance, equal-distance shooting angle, keeper displacement and defender blocking. Every base image has an exact-repeat control. Both original Luna and the selected missing-information prompt see images plus generic instructions only. No coordinates, target probabilities, outcomes or corrections are supplied.','',
        'The protocol was frozen before the independent real-shot accuracy results were scored. This is a separate visual-behavior diagnostic, not extra goal-labelled confirmation or a method for changing the predictions.','',
        '| Method | Expected direction | Mean favorable change | Mean absolute repeat change |','|---|---:|---:|---:|']
    for arm,value in result.items():
        lines.append(f"| {arm} | {value['expected_direction_count']}/12 | {value['mean_favorable_minus_base']:.1%} | {value['mean_absolute_repeat_change']:.1%} |")
    lines+=['','All probabilities and individual ties/reversals are in predictions.csv. The expected direction is qualitative, not an exact xG target. Template cases are correlated, simplify football scenes and cannot establish absolute probability accuracy. One repeat per case is a variability control, not a complete noise estimate.','',
        f'All 72 requests, raw answers, source/image/request hashes and usage were audited. New diagnostic estimate ${cost:.6f}. Its separate $0.25 cap and the combined $3 round cap both include reservations. No retries or output averaging.','']
    for arm,value in result.items(): lines.append(f"{arm}: {value['ties']} ties; reversals: {', '.join(value['reversals']) or 'none'}.")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'results':result,'estimated_usd':cost},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','run','report'])
    {'prepare':prepare,'run':run,'report':report}[parser.parse_args().action]()
