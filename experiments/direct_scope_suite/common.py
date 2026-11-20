"""Shared direct-image experiment infrastructure; no API requests on import."""
import ast, base64, csv, hashlib, importlib.util, json, math, os, random, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/direct_scope_suite'
OUT.mkdir(parents=True,exist_ok=True)
MODEL='gpt-6-luna'
PRIOR_COST=.0063094
GLOBAL_CAP=2.00
MAX_OUTPUT=4096
SCHEMA={'type':'object','properties':{'xg':{'type':'number','minimum':0,'maximum':1},'reason':{'type':'string'}},'required':['xg','reason'],'additionalProperties':False}
sp=importlib.util.spec_from_file_location('prior_pilot',ROOT/'experiments/reasoning_effort/run_experiment.py')
prior=importlib.util.module_from_spec(sp); sp.loader.exec_module(prior)
sha=prior.sha
save=prior.save
read=prior.read

def freeze(path,value):
    if path.exists():
        if read(path)!=value: raise ValueError('Frozen artifact changed: '+str(path))
    else: save(path,value)

def raw_prompt(): return prior.prompt()

def request_body(prompt,image,effort):
    if effort not in ('low','medium','high'): raise ValueError('Unapproved setting')
    return {'model':MODEL,'store':False,'reasoning':{'effort':effort},'max_output_tokens':MAX_OUTPUT,'service_tier':'default',
            'input':[{'role':'user','content':[{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(image).decode(),'detail':'high'}, {'type':'input_text','text':prompt}]}],
            'text':{'format':{'type':'json_schema','name':'shot_prediction','strict':True,'schema':SCHEMA}}}

def campaign_protocol():
    value={'model':MODEL,'prompt':raw_prompt(),'schema':SCHEMA,'max_output_tokens':MAX_OUTPUT,'service_tier':'default','image_detail':'high',
           'global_cap_usd':GLOBAL_CAP,'prior_pilot_estimated_usd':PRIOR_COST,'max_generation_attempts':1000,'retries':0,'batch_size':10,
           'prices_per_million':{'input':.1,'cached_input':.01,'conservative_input':.125,'output':.5},
           'stages':{'reasoning':{'n':50,'efforts':['low','medium','high'],'cap_usd':.35},'rendering':{'n':40,'conditions':['standard','outline'],'cap_usd':.20},'confirmation':{'n':150,'cap_usd':.80},'invariance':{'n':20,'cap_usd':.15}},
           'decisions':'Finish fixed reasoning sample before selecting lowest-Brier effort (ties favor lower effort). Then a fixed WC development rendering comparison; use outline only if paired development Brier improves. Freeze candidate before independent confirmation. No confirmation-driven tuning or cohort extension.',
           'scope':'Direct zero-shot image-to-Luna-to-raw-xG. No correction, training, numerical query answers, tools, ensembles or body-part metadata to the VLM.',
           'confirmation':'150 natural-sample shots in unused AFCON match groups. Primary comparisons candidate minus four-feature logistic and candidate minus original low-effort method; Bonferroni 97.5% match intervals. No sampling until significance.',
           'stopping':'Stop before any hard limit; on exhausted credit, unknown charge, malformed/truncated output or unexpected model, preserve everything and stop paid generation. Independent tests may continue offline.',
           'allocation_note':'Expanded autonomy authorized by user after pilot; working ceiling $2, deliberately below the approximately $10 account limit. No reason to spend the remainder once tests are informative.',
           'pricing_source':'https://developers.openai.com/api/docs/models/gpt-6-luna'}
    freeze(OUT/'campaign_protocol.json',value)
    return value

def new_stage(name,jobs,metadata):
    folder=OUT/name; folder.mkdir(exist_ok=True)
    for job in jobs:
        path=OUT/job['image']; job['image_sha256']=sha(path)
    value={'name':name,'prompt':raw_prompt(),'model':MODEL,'schedule':jobs,'metadata':metadata,'common_protocol_sha256':sha(OUT/'campaign_protocol.json'),
           'engine_sha256':sha(Path(__file__)),'runner_sha256':sha(Path(__file__).with_name('run_stage.py'))}
    freeze(folder/'protocol.json',value)
    return value

def usage(ledger):
    u={'input_tokens':0,'output_tokens':0,'cached_tokens':0,'reasoning_tokens':0}
    for a in ledger:
        v=a.get('usage',{})
        u['input_tokens']+=v.get('input_tokens',0); u['output_tokens']+=v.get('output_tokens',0)
        u['cached_tokens']+=(v.get('input_tokens_details') or {}).get('cached_tokens',0)
        u['reasoning_tokens']+=(v.get('output_tokens_details') or {}).get('reasoning_tokens',0)
    u['estimated_usd']=((u['input_tokens']-u['cached_tokens'])*.1+u['cached_tokens']*.01+u['output_tokens']*.5)/1e6
    u['conservative_usd']=(u['input_tokens']*.125+u['output_tokens']*.5)/1e6
    return u

def check_budget(ledger,stage,bound):
    if any(a['state']!='completed' for a in ledger): raise ValueError('Unreconciled attempt: stop all generation')
    if len(ledger)>=1000: raise ValueError('Campaign attempt cap reached')
    group=[a for a in ledger if a['stage']==stage]; extra=(bound*.125+MAX_OUTPUT*.5)/1e6
    if usage(ledger)['conservative_usd']+PRIOR_COST+extra>GLOBAL_CAP: raise ValueError('Campaign dollar cap reached')
    caps={'reasoning':.35,'rendering':.20,'confirmation':.80,'invariance':.15}
    if usage(group)['conservative_usd']+extra>caps[stage]: raise ValueError('Stage dollar cap reached')


def metrics(y,p): return prior.metrics(list(y),list(p))
