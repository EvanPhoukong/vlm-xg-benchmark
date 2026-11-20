"""Frozen dataset transfer test. Reuses verified HTTP/validation, no corrections."""
import argparse,concurrent.futures,csv,importlib.util,json,os,random,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('direct_shared',ROOT/'experiments/direct_model_search/run.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
OUT=ROOT/'results/dataset_transfer_v1'
ARM={'model':'gpt-6-luna','effort':'low'}
CAP,ATTEMPTS,WORKERS=1.0,1100,6
COHORTS=('premier_league','womens_world_cup','metrica_v2')


def exposure(ledger): return base.exposure(ledger)


def reserve(ledger,entry):
    if any(a['state']=='failed_or_uncertain' for a in ledger): raise ValueError('Unreconciled prior attempt')
    if len(ledger)>=ATTEMPTS or exposure(ledger)+base.reserved_cost(entry)>CAP: raise ValueError('Hard new-round cap before next call')


def prepare():
    prompt=base.original_prompt(); jobs=[]; selections={}; baseline_hashes={}
    for cohort in COHORTS:
        path=OUT/cohort/'selection.json'; rows=base.read(path)
        if not rows: raise ValueError('Prepared cohort missing')
        selections[cohort]=base.sha(path)
        reference=OUT/cohort/('provenance.json' if cohort=='metrica_v2' else 'baseline.json')
        baseline_hashes[cohort]=base.sha(reference)
        for row in rows:
            image=OUT/'images'/cohort/(row['shot_id']+'.png')
            jobs.append({'cohort':cohort,'shot_id':row['shot_id'],'image':str(image.relative_to(ROOT)),
                         'image_sha256':base.sha(image)})
    random.Random(2026103103).shuffle(jobs)
    protocol={'date':'29 October 2026 America/Los_Angeles','authorization':'User asked to try different datasets; existing no-correction and $10 provider limits persist.',
        'hypothesis':'Does unchanged original direct Luna transfer to new populations and a separate tracking-data provider? No method selection on these cohorts.',
        'arm':ARM,'prompt':prompt,'schema':base.SCHEMA,'image_detail':'high','max_output_tokens':base.MAX_OUTPUT,'service_tier':'default',
        'cap_usd':CAP,'max_attempts':ATTEMPTS,'planned_calls':len(jobs),'workers':WORKERS,'retries':0,
        'rates':base.RATES[ARM['model']],'conservative_input_multiplier':1.25,
        'pricing_source':'https://developers.openai.com/api/docs/models/gpt-6-luna',
        'primary':'Original raw Luna minus original frozen WC2018/22 four-feature logistic Brier in each of Premier League and Womens World Cup. Two prespecified 97.5% whole-match bootstrap intervals. No cross-dataset winner selection or pooled universal superiority claim.',
        'secondary':'Separately fitted same-dataset train-match logistic, no-header versions, training-only constant and StatsBomb references. Metrica is descriptive only (2 games), no population-level match bootstrap claim.',
        'scope':'Identical original image renderer and image-only prompt. No cohort names, body-part flags, coordinates, outcomes, end positions, xG references or numeric probability anchors enter VLM requests. No recalibration, trained VLM head, blending, averaging or fine tuning.',
        'comparability':'New provider Metrica versus two new StatsBomb populations are distinguished. Women cohort identity is not supplied to the unchanged generic VLM. Source games are anonymized for Metrica; its match identity overlap with other providers cannot be established.',
        'selection_hashes':selections,'reference_hashes':baseline_hashes,'data_plan_sha256':base.sha(OUT/'data_plan.json'),
        'data_amendment_sha256':base.sha(OUT/'data_amendment.json'),
        'original_source_sha256':base.sha(ROOT/'xg_vlm_pilot.py'),'preparation_sha256':base.sha(Path(__file__).with_name('prepare.py')),
        'engine_sha256':base.sha(__file__),'shared_engine_sha256':base.sha(base.__file__),
        'prior_new_round_usd':base.read(ROOT/'results/direct_model_search_v1/ROUND_SUMMARY.json')['new_estimated_usd'],
        'account_caveat':'$1 incremental cap; approximately $1.132 prior tracked estimated subtotal across available older records. Complete bills/other account activity unavailable; no live balance claim.',
        'stop':'No partial-cohort scoring or replacements. Stop dispatch on unknown billing, wrong identity, malformed/incomplete response or a cap. Save in-flight results and every failed attempt.',
        'schedule':jobs}
    base.freeze(OUT/'protocol.json',protocol)
    print(f'Frozen {len(jobs)} unchanged-Luna calls across three datasets; $1 cap, 6 concurrent reservations, no generation.',flush=True)


def status(ledger):
    return {'completed':sum(a['state']=='completed' for a in ledger),'attempts':len(ledger),
        'planned':base.read(OUT/'protocol.json')['planned_calls'],
        'estimated_usd':sum(base.cost(a.get('usage',{}),a['model']) for a in ledger),
        'conservative_exposure_usd':exposure(ledger),'cap_usd':CAP,
        'by_cohort':{c:sum(a['state']=='completed' and a['job']['cohort']==c for a in ledger) for c in COHORTS},
        'failures':[a['error'] for a in ledger if a.get('error')]}


def run():
    p=base.read(OUT/'protocol.json')
    for field,path in [('engine_sha256',Path(__file__)),('shared_engine_sha256',Path(base.__file__)),
        ('original_source_sha256',ROOT/'xg_vlm_pilot.py'),('preparation_sha256',Path(__file__).with_name('prepare.py')),
        ('data_plan_sha256',OUT/'data_plan.json')]:
        if p[field]!=base.sha(path): raise ValueError('Frozen code/data plan changed')
    for c in COHORTS:
        if p['selection_hashes'][c]!=base.sha(OUT/c/'selection.json'): raise ValueError('Frozen cohort changed')
    ledger=base.read(OUT/'attempts.json',[])
    if any(a['state']!='completed' for a in ledger): raise ValueError('Prior attempt unreconciled')
    if [a['job'] for a in ledger]!=p['schedule'][:len(ledger)]: raise ValueError('Saved request schedule changed')
    key=base.credential(); counts=base.read(OUT/'token_counts.json',{})
    lock=OUT/'generation.lock'; fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd)
    mutex=threading.Lock()
    def execute(entry,payload):
        began=time.monotonic(); updates={}
        try:
            raw=base.api('responses',key,payload); base.save(OUT/entry['response_file'],raw)
            updates['usage']=raw.get('usage') or {}; updates['answer']=base.validate(raw,ARM)
            if updates['usage']['input_tokens']>entry['input_bound'] or updates['usage']['output_tokens']>base.MAX_OUTPUT:
                raise ValueError('Token reservation exceeded')
            updates.update(state='completed',returned_model=raw['model'])
        except Exception as error: updates.update(state='failed_or_uncertain',error=str(error))
        updates['seconds']=round(time.monotonic()-began,3)
        with mutex:
            entry.update(updates); base.save(OUT/'attempts.json',ledger); base.save(OUT/'status.json',status(ledger))
        return updates['state']=='completed'
    queue=iter(enumerate(p['schedule'][len(ledger):],len(ledger)+1)); pending=set(); exhausted=False; stopped=False; last_print=0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            while pending or not exhausted:
                while len(pending)<WORKERS and not exhausted and not stopped:
                    try: index,job=next(queue)
                    except StopIteration:
                        exhausted=True; break
                    image=ROOT/job['image']
                    if base.sha(image)!=job['image_sha256']: raise ValueError('Image changed')
                    payload=base.body(ARM,p['prompt'],image.read_bytes()); countkey=job['image_sha256']
                    if countkey not in counts:
                        n=base.api('responses/input_tokens',key,{k:payload[k] for k in ('model','input','text')}).get('input_tokens')
                        if type(n) is not int or not 0<n<10000: raise ValueError('Input count unavailable')
                        counts[countkey]=n; base.save(OUT/'token_counts.json',counts)
                    entry={'job':job,'index':index,'model':ARM['model'],'state':'started','input_bound':counts[countkey]+1024,
                        'response_file':f'responses/response_{index:04d}.json','request_sha256':base.digest(payload),
                        'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
                    with mutex:
                        reserve(ledger,entry); ledger.append(entry); base.save(OUT/'attempts.json',ledger)
                    pending.add(pool.submit(execute,entry,payload))
                if stopped: exhausted=True
                if not pending: break
                finished,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                if not all(f.result() for f in finished): stopped=True
                current=status(ledger)
                if current['completed']>=last_print+25 or stopped:
                    print(json.dumps(current),flush=True); last_print=current['completed']
        current=status(ledger); base.save(OUT/'status.json',current)
        if current['completed']!=p['planned_calls'] or current['failures']: raise ValueError('Incomplete fixed study; all records preserved')
        base.save(OUT/'complete.json',current); print('Fixed three-dataset study complete. No automatic extension.',flush=True)
    finally: lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','run'])
    {'prepare':prepare,'run':run}[parser.parse_args().action]()
