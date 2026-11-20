"""Offline raw-response audit and fixed core-fix comparisons."""
import csv,json,math
from pathlib import Path
import common as c


def complete(name):
    p=c.read(c.OUT/name/'protocol.json'); ledger=c.read(c.OUT/'attempts.json',[])
    rows=[a for a in ledger if a['stage']==name]
    if len(rows)!=len(p['schedule']) or any(a['state']!='completed' for a in rows): raise ValueError('Full fixed '+name+' required')
    return p,rows


def audit():
    p=c.read(c.OUT/'protocol.json'); ledger=c.read(c.OUT/'attempts.json',[]); checks={}
    for field,path in [('source_sha256',c.ROOT/'xg_vlm_pilot.py'),('preparation_sha256',Path(__file__).with_name('prepare.py')),
        ('common_sha256',Path(c.__file__)),('engine_sha256',Path(__file__).with_name('run.py')),
        ('examples_sha256',c.OUT/'training/examples.json'),('pool_sha256',c.OUT/'heldout/pool.json')]:
        checks[field]=p[field]==c.sha(path)
    checks['shared_http_validator_unchanged']=c.sha(c.old.__file__)==c.read(c.ROOT/'results/direct_model_search_v1/protocol.json')['engine_sha256']
    examples=c.read(c.OUT/'training/examples.json'); pool=c.read(c.OUT/'heldout/pool.json')
    checks['examples_disjoint_from_queries']=not({x['match_id'] for x in examples}&{x['match_id'] for x in pool})
    checks['negative_control_same_goal_count']=sum(x['goal'] for x in examples)==sum(x['shuffled_goal'] for x in examples)
    checks['negative_control_changes_labels']=[x['goal'] for x in examples]!=[x['shuffled_goal'] for x in examples]
    split=c.read(c.OUT/'data/match_split.json')
    checks['new_match_groups']=not(set(split['selected_match_ids'])&set(split['previous_match_ids']))
    checks['source_events_unchanged']=all(c.sha(c.OUT/'data'/f'events_{mid}.json')==value for mid,value in c.read(c.OUT/'data/provenance.json')['source_hashes'].items())
    checks['example_images_frozen']=all(c.sha(c.ROOT/x['image'])==x['image_sha256'] for x in examples)
    source=c.read(c.OUT/'training/provenance.json'); originals={}
    for name,value in source['source_hashes'].items():
        path=Path('/mnt/c/Users/ephou/Downloads/xG/output/research-loop/data')/(name+'.jsonl')
        assert c.sha(path)==value
        originals.update({r['shot_id']:r for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())})
    checks['correct_example_outcomes_match_frozen_source']=all(x['goal']==int(originals[x['shot_id']]['goal']) for x in examples)
    motion_source=c.ROOT/'results/dataset_transfer_v1/metrica_v2/provenance.json'
    checks['motion_source_provenance_frozen']=c.sha(motion_source)==c.read(c.OUT/'motion/protocol.json')['metadata']['source_provenance_sha256']
    for record in c.read(motion_source)['games']:
        root=c.ROOT/'results/dataset_transfer_v1/data'
        assert c.sha(root/f"metrica_{record['game']}_events.csv")==record['event_source_sha256']
        for team,value in record['tracking_hashes'].items(): assert c.sha(root/f"metrica_{record['game']}_{team}_tracking.csv")==value
    checks['motion_tracking_source_files_frozen']=True
    choice=c.read(c.OUT/'resource_choice.json')
    checks['usage_only_snapshot']=c.sha(c.OUT/'pilot/attempts_snapshot.json')==choice['pilot_attempts_sha256']
    snapshot=c.read(c.OUT/'pilot/attempts_snapshot.json')
    checks['pilot_records_unchanged']=[a for a in ledger if a['stage']=='pilot']==snapshot
    means={arm:sum(c.cost(a['usage'],a['model'],True) for a in snapshot if a['job']['arm']==arm)/12 for arm in c.ARMS}
    bound=max(a['input_bound'] for a in snapshot if a['job']['arm']=='astra')
    reservation=c.reserve_cost({'model':'gpt-6-astra','input_bound':bound})
    projected={n:c.exposure(snapshot)+1.25*sum(means.values())*(n+36)+180*means['luna']+reservation+.05 for n in (200,150,100)}
    checks['resource_rule_reproduced']=choice['n']==next(n for n in (200,150,100) if projected[n]<=c.CAP)
    checks['main_exact_pool_prefix']=c.read(c.OUT/'main/selection.json')==pool[:choice['n']]
    response_ids=set()
    for name in ('pilot','main','repeats','probes','motion'):
        sp,entries=complete(name); checks[name+'_complete']=True
        checks[name+'_protocol_frozen']=sp['campaign_sha256']==c.sha(c.OUT/'protocol.json')
        for entry,job in zip(entries,sp['schedule']):
            assert entry['job']==job
            for item in job['images']: assert c.sha(c.ROOT/item['path'])==item['sha256']
            payload=c.payload(job); assert c.digest(payload)==entry['request_sha256']
            raw=c.read(c.OUT/entry['response_file']); assert c.old.validate(raw,c.ARMS[job['arm']])==entry['answer']
            assert raw['usage']==entry['usage']
            assert entry['usage']['input_tokens']<=entry['input_bound'] and entry['usage']['output_tokens']<=c.MAX_OUTPUT
            assert raw['id'] not in response_ids; response_ids.add(raw['id'])
    checks['all_raw_answers_requests_models_usage_images']=True
    for row in c.read(c.OUT/'motion/selection.json'):
        frames=c.read(c.OUT/'motion/frames'/(row['shot_id']+'.json'))
        assert [x['offset_frames'] for x in frames]==[-25,-12,0]
        assert all(x['source_frame']<=row['source_start_frame'] for x in frames)
    checks['past_only_motion_frames']=True
    checks['under_dollar_cap']=c.exposure(ledger)<=c.CAP
    checks['under_attempt_cap']=len(ledger)<=c.MAX_ATTEMPTS
    checks['no_generation_lock']=not(c.OUT/'generation.lock').exists()
    key=c.old.credential().encode()
    checks['private_exact_credential_scan']=not any(key in path.read_bytes() for path in c.OUT.rglob('*') if path.is_file())
    result={'passed':all(checks.values()),'checks':checks,'calls':len(ledger),
        'estimated_usd':sum(c.cost(a['usage'],a['model']) for a in ledger),'conservative_usd':c.exposure(ledger),
        'cap_usd':c.CAP,'prior_partial_conservative_usd':1.4552639,
        'by_model':{m:{'calls':sum(a['model']==m for a in ledger),
            'estimated_usd':sum(c.cost(a['usage'],m) for a in ledger if a['model']==m),
            'input_tokens':sum(a['usage']['input_tokens'] for a in ledger if a['model']==m),
            'cached_input_tokens':sum((a['usage'].get('input_tokens_details') or {}).get('cached_tokens',0) for a in ledger if a['model']==m),
            'output_tokens':sum(a['usage']['output_tokens'] for a in ledger if a['model']==m)} for m in c.RATES}}
    c.save(c.OUT/'audit.json',result)
    if not result['passed']: raise ValueError('Audit failed: '+','.join(k for k,v in checks.items() if not v))
    return result


def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main_accuracy():
    sp,entries=complete('main'); answers={(a['job']['shot_id'],a['job']['arm']):a['answer'] for a in entries}; rows=[]
    for source in c.read(c.OUT/'main/selection.json'):
        row={k:source[k] for k in ('shot_id','match_id','goal','distance','angle','header','defenders_in_cone','original_logistic','original_no_header','sb_xg','training_prior')}
        row['example_rate']=sum(x['goal'] for x in c.read(c.OUT/'training/examples.json'))/48
        for arm in c.ARMS:
            row[arm]=answers[row['shot_id'],arm]['xg']; row[arm+'_reason']=answers[row['shot_id'],arm]['reason']
        rows.append(row)
    methods=list(c.ARMS)+['original_logistic','original_no_header','sb_xg','training_prior','example_rate']
    scores={m:c.old.metrics([r['goal'] for r in rows],[r[m] for r in rows]) for m in methods}
    # Import the already verified whole-match comparison helper without altering it.
    import importlib.util
    spec=importlib.util.spec_from_file_location('transfer_comparison',c.ROOT/'experiments/direct_model_search/run.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    pairs=[('astra','luna'),('astra','original_logistic'),('examples','luna'),('examples','original_logistic'),('examples','shuffled')]
    comparisons=[module.comparison(rows,a,b,.99) for a,b in pairs]
    strata={}
    for name,low,high in [('short_6_18',6,18),('under_12',0,12),('18_plus',18,math.inf)]:
        subset=[r for r in rows if low<=r['distance']<high]
        strata[name]={'shots':len(subset),'goals':sum(r['goal'] for r in subset),
            'scores':{m:c.old.metrics([r['goal'] for r in subset],[r[m] for r in subset]) for m in methods} if subset else {}}
    result={'shots':len(rows),'goals':sum(r['goal'] for r in rows),'matches':len({r['match_id'] for r in rows}),
        'scores':scores,'primary':comparisons,'strata':strata,'scope':'Fixed raw VLMs, no method selection. 99% match intervals for five comparisons; subgroup diagnostics descriptive.'}
    save_csv(c.OUT/'main/predictions.csv',rows); c.save(c.OUT/'main/summary.json',result)
    return result


def diagnostic_consistency():
    _,pilot=complete('pilot'); _,repeats=complete('repeats'); _,probes=complete('probes')
    original={(a['job']['shot_id'],a['job']['arm']):a['answer']['xg'] for a in pilot}
    summaries={}; rows=[]
    for arm in c.ARMS:
        values=[abs(a['answer']['xg']-original[a['job']['shot_id'],arm]) for a in repeats if a['job']['arm']==arm]
        indexed={(a['job']['shot_id'],a['job'].get('condition')):a for a in probes if a['job']['arm']==arm}
        keys=sorted({a['job']['shot_id'] for a in probes}); deltas=[]; reversals=[]; ties=[]
        for sid in keys:
            before=indexed[sid,'base']['answer']['xg']; after=indexed[sid,'favorable']['answer']['xg']
            delta=after-before; deltas.append(delta)
            if delta<0: reversals.append(sid)
            if delta==0: ties.append(sid)
            rows.append({'case':sid,'arm':arm,'base':before,'favorable':after,'difference':delta})
        summaries[arm]={'repeat_cases':len(values),'mean_absolute_repeat_change':sum(values)/len(values),
            'probe_cases':len(deltas),'expected_direction_count':sum(d>0 for d in deltas),'ties':ties,'reversals':reversals,
            'mean_favorable_change':sum(deltas)/len(deltas)}
    save_csv(c.OUT/'probes/predictions.csv',rows); c.save(c.OUT/'consistency.json',summaries)
    return summaries


def motion_accuracy():
    _,entries=complete('motion')
    answers={(a['job']['shot_id'],a['job']['condition']):a['answer'] for a in entries}; rows=[]
    for source in c.read(c.OUT/'motion/selection.json'):
        row={k:source[k] for k in ('shot_id','match_id','goal','original_logistic','original_no_header')}
        for arm in ('original','single_ball','history'):
            row[arm]=answers[row['shot_id'],arm]['xg']; row[arm+'_reason']=answers[row['shot_id'],arm]['reason']
        rows.append(row)
    scores={m:c.old.metrics([r['goal'] for r in rows],[r[m] for r in rows]) for m in ('original','single_ball','history','original_logistic','original_no_header')}
    repeat={}
    for arm in ('single_ball','history'):
        delta=[abs(a['xg']-answers[sid,arm]['xg']) for (sid,condition),a in answers.items() if condition==arm+'_repeat']
        repeat[arm]={'cases':len(delta),'mean_absolute_change':sum(delta)/len(delta)}
    result={'shots':len(rows),'goals':sum(r['goal'] for r in rows),'matches':len({r['match_id'] for r in rows}),
        'scores':scores,'exact_repeats':repeat,'history_minus_single_brier':scores['history']['brier']-scores['single_ball']['brier'],
        'limitation':'Previously used anonymized Metrica games, past tracking not raw footage or 3D pose; two groups, descriptive only.'}
    save_csv(c.OUT/'motion/predictions.csv',rows); c.save(c.OUT/'motion/summary.json',result)
    return result


def main():
    integrity=audit(); accuracy=main_accuracy(); consistency=diagnostic_consistency(); motion=motion_accuracy()
    c.save(c.OUT/'summary.json',{'audit':integrity,'main':accuracy,'consistency':consistency,'motion':motion})
    print(json.dumps({'estimated_usd':integrity['estimated_usd'],'calls':integrity['calls'],'main':accuracy,
        'consistency':consistency,'motion':motion},indent=2),flush=True)


if __name__=='__main__': main()
