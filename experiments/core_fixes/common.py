"""Direct core-fix experiment helpers. No generation on import."""
import base64,importlib.util,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('old_direct',ROOT/'experiments/direct_model_search/run.py')
old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
OUT=ROOT/'results/core_fixes_v1'
read,save,freeze,sha,digest=old.read,old.save,old.freeze,old.sha,old.digest
CAP,MAX_OUTPUT,MAX_ATTEMPTS,WORKERS=4.0,4096,1200,6
RATES={'gpt-6-luna':(.10,.01,.50),'gpt-6-astra':(10.,1.,50.)}
ARMS={'luna':{'model':'gpt-6-luna','effort':'low','examples':None},
      'astra':{'model':'gpt-6-astra','effort':'low','examples':None},
      'examples':{'model':'gpt-6-luna','effort':'low','examples':'goal'},
      'shuffled':{'model':'gpt-6-luna','effort':'low','examples':'shuffled_goal'}}
EXAMPLE_INTRO="""Historical training examples follow. Each image shows a DIFFERENT past shot,
with its single observed goal/no-goal outcome. A single scored shot does not have
true xG 1, and a missed shot does not have true xG 0: these are noisy binary outcomes.
Use the visual situations and outcomes as examples for estimating a new shot's
probability. The final query is separate and its outcome is unknown.
"""
MOTION_PROMPT="""The input contains either one current tracking snapshot or three snapshots
ordered from 1.00 seconds before contact, through 0.48 seconds before contact, to
the shot-contact frame. Estimate xG for the shot in the LAST frame. All frames
precede or equal contact; no future trajectory or outcome is supplied.
Gold star = the eventual shooter in every frame. Magenta diamond = tracked ball.
Blue circles = teammates, red circles = opponents, orange square = opposing keeper.
Goal is at the top. Body part and ball height are unknown. Use the available
positional and motion evidence, assuming an average professional finisher.
Return JSON only: {"xg": <number between 0 and 1>, "reason": "<one short sentence>"}.
"""


def image_part(path):
    return {'type':'input_image','detail':'high','image_url':'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()}


def payload(job):
    arm=ARMS[job['arm']]; parts=[]
    if arm['examples']:
        parts.append({'type':'input_text','text':EXAMPLE_INTRO})
        for i,example in enumerate(read(OUT/'training/examples.json'),1):
            parts.append(image_part(ROOT/example['image']))
            parts.append({'type':'input_text','text':f"Historical example {i}: observed result = "+('goal.' if example[arm['examples']] else 'no goal.')})
        parts.append({'type':'input_text','text':'Historical examples END. The following image is the NEW query shot. Predict its probability only.'})
    parts.extend(image_part(ROOT/item['path']) for item in job['images'])
    parts.append({'type':'input_text','text':MOTION_PROMPT if job.get('motion') else old.original_prompt()})
    return {'model':arm['model'],'store':False,'service_tier':'default','reasoning':{'effort':arm['effort']},
        'max_output_tokens':MAX_OUTPUT,'input':[{'role':'user','content':parts}],
        'text':{'format':{'type':'json_schema','name':'shot_prediction','strict':True,'schema':old.SCHEMA}}}


def cost(usage,model,conservative=False):
    ir,cr,outr=RATES[model]; ni=usage.get('input_tokens',0); no=usage.get('output_tokens',0)
    cached=(usage.get('input_tokens_details') or {}).get('cached_tokens',0)
    return ((ni*ir*1.25 if conservative else (ni-cached)*ir+cached*cr)+no*outr)/1e6


def reserve_cost(entry): return cost({'input_tokens':entry['input_bound'],'output_tokens':MAX_OUTPUT},entry['model'],True)
def exposure(ledger): return sum(cost(a['usage'],a['model'],True) if a['state']=='completed' else reserve_cost(a) for a in ledger)


def can_reserve(ledger,entry):
    if any(a['state']=='failed_or_uncertain' for a in ledger): raise ValueError('Uncertain prior attempt; stop all paid dispatch')
    if len(ledger)>=MAX_ATTEMPTS: raise ValueError('Attempt limit')
    return exposure(ledger)+reserve_cost(entry)<=CAP


def job(sid,arm,paths,**extra):
    return {'shot_id':sid,'arm':arm,'images':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in paths],**extra}


def freeze_stage(name,jobs,metadata=None):
    freeze(OUT/name/'protocol.json',{'name':name,'schedule':jobs,'metadata':metadata or {},
        'campaign_sha256':sha(OUT/'protocol.json'),'engine_sha256':sha(Path(__file__).with_name('run.py')),
        'common_sha256':sha(__file__)})
