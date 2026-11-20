"""Bounded paired direct-image Luna pilot. Default preparation is offline."""
import argparse, ast, base64, csv, hashlib, json, math, os, random, re, time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/reasoning_effort_v1'
MODEL='gpt-6-luna'
LIMIT={'attempts':24,'input':120000,'output':24000,'per_output':2048,'usd':.25}
SCHEMA={'type':'object','properties':{'xg':{'type':'number','minimum':0,'maximum':1},'reason':{'type':'string'}},'required':['xg','reason'],'additionalProperties':False}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8'); t.replace(p)
def read(p,default=None): return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default
def prompt():
    for n in ast.parse((ROOT/'xg_vlm_pilot.py').read_text(encoding='utf-8-sig')).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPT' for t in n.targets): return ast.literal_eval(n.value).format(context='')
    raise ValueError('Original prompt missing')
def body(p,image,effort):
    if effort not in ('low','medium'): raise ValueError('Unapproved effort')
    return {'model':MODEL,'store':False,'reasoning':{'effort':effort},'max_output_tokens':LIMIT['per_output'],'service_tier':'default',
        'input':[{'role':'user','content':[{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(image).decode(),'detail':'high'},{'type':'input_text','text':p}]}],
        'text':{'format':{'type':'json_schema','name':'shot_prediction','strict':True,'schema':SCHEMA}}}
def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/gpt_6_luna/shots.csv'
    with source.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    rng=random.Random(20261029); chosen=[]
    for name,lo,hi in [('close',0,12),('middle',12,18),('far',18,math.inf)]:
        pool=sorted([r for r in rows if lo<=float(r['distance'])<hi],key=lambda r:r['shot_id'])
        chosen.extend({**r,'stratum':name} for r in rng.sample(pool,4))
    rng.shuffle(chosen); schedule=[]
    for r in chosen:
        arms=['low','medium']; rng.shuffle(arms)
        schedule.extend({'shot_id':r['shot_id'],'effort':a} for a in arms)
    protocol={'model':MODEL,'prompt':prompt(),'schema':SCHEMA,'limits':LIMIT,'seed':20261029,'schedule':schedule,
        'selection':'Four shots per distance stratum [0,12),[12,18),[18,infinity); normalized pitch units; seed selection independent of outcomes and prior errors. Previously studied development cohort.',
        'source_sha256':sha(source),'pilot_sha256':sha(ROOT/'xg_vlm_pilot.py'),'runner_sha256':sha(__file__),
        'images':{r['shot_id']:sha(ROOT/'results/images'/(r['shot_id']+'.png')) for r in chosen},
        'rates_per_million':{'input':.10,'cached_input':.01,'output':.50,'conservative_input':.125},
        'price_source':'https://developers.openai.com/api/docs/models/gpt-6-luna','batch_size':6,'retries':0,
        'settings':'Identical high-detail images, prompt, JSON schema, service tier and output allowance; only effort differs. No calibration or downstream model.',
        'preflight':'Official token count plus 1024 input-token safety allowance; reserve full output allowance before generation.',
        'stopping':'Stop on invalid/incomplete response, uncertain charge, unexpected model or budget cap. No replacement shots or automatic expansion.'}
    if (OUT/'protocol.json').exists():
        if read(OUT/'protocol.json')!=protocol or read(OUT/'selection.json')!=chosen: raise ValueError('Frozen protocol changed; stop')
    else: save(OUT/'protocol.json',protocol); save(OUT/'selection.json',chosen)
    return protocol

def credential():
    if os.environ.get('OPENAI_API_KEY','').strip(): return os.environ['OPENAI_API_KEY'].strip()
    text=(ROOT.parent/'keys.txt').read_text(encoding='utf-8-sig')
    named=re.search(r'(?m)^\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*([^\r\n]+)',text)
    values=set(re.findall(r'sk-(?!ant-)[A-Za-z0-9_-]+',named.group(1) if named else text))
    if len(values)!=1: raise ValueError('Cannot identify one OpenAI key; no credential printed')
    return values.pop()
def post(endpoint,data,key):
    request=Request('https://api.openai.com/v1/'+endpoint,data=json.dumps(data).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    try:
        with urlopen(request,timeout=120) as response: return json.load(response)
    except HTTPError as e:
        try: code=json.loads(e.read()).get('error',{}).get('code','http_error')
        except Exception: code='http_error'
        raise RuntimeError(f'HTTP {e.code}: {code}; no automatic retry') from None
    except URLError as e: raise RuntimeError(f'Network failure ({type(e.reason).__name__}); no automatic retry') from None

def totals(ledger): return {k:sum(a.get('usage',{}).get(k,0) for a in ledger) for k in ('input_tokens','output_tokens')}
def reserve(ledger,bound):
    if any(a['state']!='completed' for a in ledger): raise ValueError('Uncertain or failed prior attempt; stop for reconciliation')
    u=totals(ledger); ni=u['input_tokens']+bound; no=u['output_tokens']+LIMIT['per_output']
    if len(ledger)>=LIMIT['attempts'] or ni>LIMIT['input'] or no>LIMIT['output'] or (ni*.125+no*.50)/1e6>LIMIT['usd']: raise ValueError('Hard budget reached before another request')
def validate(r):
    if not (r.get('model')==MODEL or r.get('model','').startswith(MODEL+'-')): raise ValueError('Unexpected returned model')
    if r.get('status')!='completed': raise ValueError('Incomplete response; preserve and stop without raising limits')
    texts=[p['text'] for item in r.get('output',[]) if item.get('type')=='message' for p in item.get('content',[]) if p.get('type')=='output_text']
    a=json.loads(''.join(texts))
    if set(a)!={'xg','reason'} or isinstance(a['xg'],bool) or not isinstance(a['xg'],(int,float)) or not math.isfinite(a['xg']) or not 0<=a['xg']<=1 or not isinstance(a['reason'],str): raise ValueError('Invalid raw prediction')
    u=r.get('usage') or {}
    if any(not isinstance(u.get(k),int) or u[k]<0 for k in ('input_tokens','output_tokens')): raise ValueError('Usage missing; stop for reconciliation')
    return a

def run(protocol,batch):
    lock=OUT/'runner.lock'; fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd)
    try:
        ledger=read(OUT/'attempts.json',[]); counts=read(OUT/'token_counts.json',{}); key=credential()
        for job in protocol['schedule'][len(ledger):len(ledger)+min(batch,6)]:
            img=(ROOT/'results/images'/(job['shot_id']+'.png')).read_bytes(); payload=body(protocol['prompt'],img,job['effort'])
            if job['shot_id'] not in counts:
                counted=post('responses/input_tokens',{k:payload[k] for k in ('model','input','text')},key)
                n=counted.get('input_tokens')
                if not isinstance(n,int) or n<=0: raise ValueError('Cannot preflight image input; no generation attempted')
                counts[job['shot_id']]=n; save(OUT/'token_counts.json',counts)
            bound=counts[job['shot_id']]+1024; reserve(ledger,bound)
            attempt={**job,'state':'started','input_bound':bound,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            ledger.append(attempt); save(OUT/'attempts.json',ledger); started=time.monotonic()
            try:
                response=post('responses',payload,key); name=f"response_{len(ledger):02d}_{job['effort']}.json"; save(OUT/name,response)
                attempt.update(usage=response.get('usage') or {},response_file=name,seconds=round(time.monotonic()-started,3))
                answer=validate(response)
                if attempt['usage']['input_tokens']>bound or attempt['usage']['output_tokens']>LIMIT['per_output']: raise ValueError('Usage exceeded reserved bound; stop')
                attempt.update(state='completed',answer=answer,model=response['model'])
            except Exception as e:
                attempt.update(state='failed_or_uncertain',error=str(e)); save(OUT/'attempts.json',ledger); raise
            save(OUT/'attempts.json',ledger); u=totals(ledger)
            print(f"Saved {len(ledger)}/24 ({job['effort']}); cumulative input={u['input_tokens']}, output={u['output_tokens']}",flush=True)
        print('Batch finished; no partial accuracy scores examined.',flush=True)
    finally: lock.unlink()

def metrics(y,p):
    pos=[i for i,v in enumerate(y) if v]; neg=[i for i,v in enumerate(y) if not v]
    auc=sum(float(p[i]>p[j])+.5*float(p[i]==p[j]) for i in pos for j in neg)/(len(pos)*len(neg)) if pos and neg else None
    q=[max(1e-15,min(1-1e-15,v)) for v in p]
    return {'brier':sum((a-b)**2 for a,b in zip(y,p))/len(y),'log_loss':-sum(a*math.log(b)+(1-a)*math.log(1-b) for a,b in zip(y,q))/len(y),'auc':auc,'mean_xg':sum(p)/len(p)}
def report(protocol):
    ledger=read(OUT/'attempts.json',[])
    if len(ledger)!=24 or any(a['state']!='completed' for a in ledger): raise ValueError('Require full 24-response cohort; no partial accuracy scoring')
    if [{k:a[k] for k in ('shot_id','effort')} for a in ledger]!=protocol['schedule']: raise ValueError('Schedule mismatch')
    answers={(a['shot_id'],a['effort']):a for a in ledger}; rows=[]
    for r in read(OUT/'selection.json'):
        row={k:r[k] for k in ('shot_id','match_id','goal','stratum','distance','baseline_xg','sb_xg')}
        for arm in ('low','medium'):
            a=answers[r['shot_id'],arm]; row[arm]=a['answer']['xg']; row[arm+'_reason']=a['answer']['reason']
        rows.append(row)
    with (OUT/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    y=[int(r['goal']) for r in rows]; scores={k:metrics(y,[float(r[k]) for r in rows]) for k in ('low','medium','baseline_xg','sb_xg')}; usage={}
    for arm in ('low','medium'):
        items=[a for a in ledger if a['effort']==arm]; u=totals(items)
        u['cached_input_tokens']=sum(a['usage'].get('input_tokens_details',{}).get('cached_tokens',0) for a in items)
        u['reasoning_tokens']=sum(a['usage'].get('output_tokens_details',{}).get('reasoning_tokens',0) for a in items)
        u['estimated_usd']=((u['input_tokens']-u['cached_input_tokens'])*.10+u['cached_input_tokens']*.01+u['output_tokens']*.50)/1e6; usage[arm]=u
    import numpy as np
    groups=sorted(set(r['match_id'] for r in rows)); rng=np.random.default_rng(20261029)
    losses={g:[(float(r['medium'])-int(r['goal']))**2-(float(r['low'])-int(r['goal']))**2 for r in rows if r['match_id']==g] for g in groups}
    boots=[float(np.mean([v for g in rng.choice(groups,len(groups)) for v in losses[g]])) for _ in range(4000)]
    interval=np.quantile(boots,[.025,.975]).tolist(); delta=scores['medium']['brier']-scores['low']['brier']
    result={'status':'complete','shots':12,'goals':sum(y),'matches':len(groups),'scores':scores,'usage':usage,'medium_minus_low_brier':delta,'exploratory_match_bootstrap_95':interval,'estimated_total_usd':sum(u['estimated_usd'] for u in usage.values())}
    save(OUT/'summary.json',result)
    lines=['# Direct Luna xG: reasoning-effort pilot','',f"Completed 12 shots ({sum(y)} goals; {len(groups)} matches), 24 successful requests. Only gpt-6-luna was requested. Original image bytes, prompt, schema, high image detail and output allowance are identical; only reasoning effort differs.",'',
        'This reuses the previously studied World Cup development cohort, with four shots per prespecified distance stratum. It is not independent confirmation or a naturally weighted tournament sample. No probability adjustment, hybrid head, new dataset, renderer change or fine-tuning.','',
        '| Method | Brier (lower) | Log loss | AUC | Mean xG |','|---|---:|---:|---:|---:|']
    for k,s in scores.items():
        auc=f"{s['auc']:.3f}" if s['auc'] is not None else 'undefined'
        lines.append(f"| {k} | {s['brier']:.5f} | {s['log_loss']:.5f} | {auc} | {s['mean_xg']:.1%} |")
    lines+=['',f'Medium minus low Brier: {delta:+.5f}; exploratory whole-match 95% interval [{interval[0]:+.5f}, {interval[1]:+.5f}]. Twelve shots cannot establish superiority or equivalence. Intervals condition on this selected small cohort.','',
        '| Effort | Input | Cached input | Output including reasoning | Reasoning subset | Estimated USD |','|---|---:|---:|---:|---:|---:|']
    for k,u in usage.items(): lines.append(f"| {k} | {u['input_tokens']} | {u['cached_input_tokens']} | {u['output_tokens']} | {u['reasoning_tokens']} | ${u['estimated_usd']:.6f} |")
    lines+=['',f"Total usage-derived estimated inference cost: ${result['estimated_total_usd']:.6f}. Published standard rates: $0.10/M input, $0.01/M cached input, $0.50/M output. This is not an account billing statement or balance check. Token-count preflights generate no model responses.",'',
        'The original logistic comparator includes a header flag absent from the image, and uses saved match-grouped out-of-fold predictions. It is a practical reference, not an equal-input ablation. StatsBomb xG is a comparator, not ground truth. Outcomes, source coordinates and comparator predictions were never included in API inputs.','',
        'No hidden retries or automatic expansion. Raw responses, usage, randomized schedule, source/image hashes and predictions are preserved. Rationales are available for factual review; fluent explanations do not prove the internal reasoning process. A larger untouched-match confirmation requires a separate budget and sample-size proposal.','',
        'Sources: [Luna rates/settings](https://developers.openai.com/api/docs/models/gpt-6-luna), [token accounting](https://developers.openai.com/api/docs/guides/token-counting).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--run',action='store_true'); ap.add_argument('--report',action='store_true'); ap.add_argument('--batch',type=int,default=6); args=ap.parse_args(); protocol=prepare()
    if args.run: run(protocol,args.batch)
    elif args.report: report(protocol)
    else: print('Frozen 12-shot, 24-attempt protocol prepared; no API requests.',flush=True)
