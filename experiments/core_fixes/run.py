"""Budgeted fixed stages; accounts for all concurrent reservations, no retries."""
import argparse,concurrent.futures,os,random,threading,time
from pathlib import Path
import common as c


def status(ledger,stage):
    group=[a for a in ledger if a['stage']==stage]
    return {'stage':stage,'completed':sum(a['state']=='completed' for a in group),'attempts':len(group),
        'planned':len(c.read(c.OUT/stage/'protocol.json')['schedule']),
        'round_estimated_usd':sum(c.cost(a.get('usage',{}),a['model']) for a in ledger),
        'round_conservative_exposure_usd':c.exposure(ledger),'cap_usd':c.CAP,
        'failures':[a.get('error') for a in ledger if a.get('error')]}


def run(stage):
    protocol=c.read(c.OUT/'protocol.json'); sp=c.read(c.OUT/stage/'protocol.json')
    for field,path in [('source_sha256',c.ROOT/'xg_vlm_pilot.py'),('preparation_sha256',Path(__file__).with_name('prepare.py')),
                       ('common_sha256',Path(c.__file__)),('engine_sha256',Path(__file__))]:
        if protocol[field]!=c.sha(path): raise ValueError('Frozen code changed')
    if sp['campaign_sha256']!=c.sha(c.OUT/'protocol.json'): raise ValueError('Frozen campaign changed')
    if c.sha(c.OUT/'training/examples.json')!=protocol['examples_sha256']: raise ValueError('Training examples changed')
    if c.sha(c.OUT/'heldout/pool.json')!=protocol['pool_sha256']: raise ValueError('Reserved cohort changed')
    ledger=c.read(c.OUT/'attempts.json',[])
    if any(a['state']!='completed' for a in ledger): raise ValueError('Unreconciled prior charge; no retry')
    done=[a for a in ledger if a['stage']==stage]
    if [a['job'] for a in done]!=sp['schedule'][:len(done)]: raise ValueError('Schedule mismatch')
    counts=c.read(c.OUT/'token_counts.json',{}); key=c.old.credential()
    lock=c.OUT/'generation.lock'; fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd)
    mutex=threading.Lock(); current_job=None
    def execute(entry,payload):
        began=time.monotonic(); update={}
        try:
            raw=c.old.api('responses',key,payload); c.save(c.OUT/entry['response_file'],raw)
            update['usage']=raw.get('usage') or {}; update['answer']=c.old.validate(raw,c.ARMS[entry['job']['arm']])
            if update['usage']['input_tokens']>entry['input_bound'] or update['usage']['output_tokens']>c.MAX_OUTPUT:
                raise ValueError('Reserved token bound exceeded')
            update.update(state='completed',returned_model=raw['model'])
        except Exception as error: update.update(state='failed_or_uncertain',error=str(error))
        update['seconds']=round(time.monotonic()-began,3)
        with mutex:
            entry.update(update); c.save(c.OUT/'attempts.json',ledger); c.save(c.OUT/'status.json',status(ledger,stage))
        return update['state']=='completed'
    queue=iter(enumerate(sp['schedule'][len(done):],len(done)+1)); pending=set(); exhausted=False; stopped=False; last_print=0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=c.WORKERS) as pool:
            while pending or not exhausted or current_job:
                while len(pending)<c.WORKERS and not exhausted and not stopped:
                    if current_job is None:
                        try: index,job=next(queue)
                        except StopIteration:
                            exhausted=True; break
                        for item in job['images']:
                            if c.sha(c.ROOT/item['path'])!=item['sha256']: raise ValueError('Frozen query image changed')
                        for example in c.read(c.OUT/'training/examples.json'):
                            if c.sha(c.ROOT/example['image'])!=example['image_sha256']: raise ValueError('Example image changed')
                        payload=c.payload(job); countkey=c.digest(payload)
                        if countkey not in counts:
                            n=c.old.api('responses/input_tokens',key,{k:payload[k] for k in ('model','input','text')}).get('input_tokens')
                            if type(n) is not int or not 0<n<60000: raise ValueError('Input token bound unavailable')
                            counts[countkey]=n; c.save(c.OUT/'token_counts.json',counts)
                        entry={'stage':stage,'index':index,'job':job,'model':c.ARMS[job['arm']]['model'],'state':'started',
                            'input_bound':counts[countkey]+1024,'request_sha256':countkey,
                            'response_file':f'{stage}/response_{index:04d}.json','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
                        current_job=(entry,payload)
                    entry,payload=current_job
                    with mutex:
                        fit=c.can_reserve(ledger,entry)
                        if fit:
                            ledger.append(entry); c.save(c.OUT/'attempts.json',ledger)
                    if not fit:
                        if pending: break  # Await reconciliation of in-flight reservations, not a new budget.
                        raise ValueError('Hard $4 cap reached; preserve incomplete stage')
                    pending.add(pool.submit(execute,entry,payload)); current_job=None
                if stopped: exhausted=True; current_job=None
                if not pending:
                    if exhausted: break
                    continue
                finished,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                if not all(f.result() for f in finished): stopped=True
                progress=status(ledger,stage)
                if progress['completed']>=last_print+12 or stopped:
                    print(__import__('json').dumps(progress),flush=True); last_print=progress['completed']
        progress=status(ledger,stage); c.save(c.OUT/'status.json',progress)
        if progress['completed']!=len(sp['schedule']) or progress['failures']: raise ValueError('Incomplete stage; no replacements')
        c.save(c.OUT/stage/'complete.json',progress); print('Fixed stage complete.',flush=True)
    finally: lock.unlink()


def plan_main():
    ledger=c.read(c.OUT/'attempts.json',[])
    pilot=[a for a in ledger if a['stage']=='pilot']
    if len(pilot)!=48 or any(a['state']!='completed' for a in pilot): raise ValueError('Complete usage pilot required')
    means={arm:sum(c.cost(a['usage'],a['model'],True) for a in pilot if a['job']['arm']==arm)/12 for arm in c.ARMS}
    astra_bound=max(a['input_bound'] for a in pilot if a['job']['arm']=='astra')
    reservation=c.reserve_cost({'model':'gpt-6-astra','input_bound':astra_bound})
    projections={n:c.exposure(ledger)+1.25*sum(means.values())*(n+36)+180*means['luna']+reservation+.05 for n in (200,150,100)}
    choices=[n for n in (200,150,100) if projections[n]<=c.CAP]
    if not choices: raise ValueError('No fixed cohort can be planned within conservative resource rule; no accuracy claim')
    n=choices[0]; c.freeze(c.OUT/'resource_choice.json',{'n':n,'projections_usd':projections,'pilot_mean_conservative_costs':means,
        'rule':'Largest N from 200/150/100 under prespecified usage-only forecast. No pilot accuracy scored.',
        'pilot_attempts_sha256':c.sha(c.OUT/'attempts.json')})
    rows=c.read(c.OUT/'heldout/pool.json')[:n]; c.freeze(c.OUT/'main/selection.json',rows)
    jobs=[]; rng=random.Random(2026103208)
    for row in rows:
        arms=list(c.ARMS); rng.shuffle(arms)
        jobs.extend(c.job(row['shot_id'],arm,[c.OUT/'images/heldout'/(row['shot_id']+'.png')]) for arm in arms)
    c.freeze_stage('main',jobs,{'resource_choice_sha256':c.sha(c.OUT/'resource_choice.json'),
        'selection_sha256':c.sha(c.OUT/'main/selection.json'),'primary_confidence_level':.99})
    print(f'Usage-only rule froze {n} new shots x4 conditions; no accuracy scoring.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['pilot','plan-main','main','repeats','probes','motion'])
    action=parser.parse_args().action
    if action=='plan-main': plan_main()
    else: run(action)
