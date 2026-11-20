"""Frozen Luna-only task division and image/event late fusion on existing data."""
import argparse, base64, copy, hashlib, json, math, os, sys, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
REPO = Path(__file__).resolve().parents[2]
SUITE = REPO/'experiments/improvement_suite'
sys.path.insert(0, str(SUITE))
import robust_xg as robust
import spatial_models as spatial
from context_trial import join_features, metadata_for
from diagnose_geometry import audit
from fetch_context import CATEGORICAL, NUMERIC
from vision_trials import key_from_authorized_file

MODEL = 'gpt-6-luna'
SEED = 20261023
SETTINGS = {'reasoning_effort': 'low', 'image_detail': 'high', 'max_output_tokens': 4096}
ROOT = Path('/mnt/c/Users/ephou/Downloads/xG/output')
COMMON = """Read this football shot diagram as a visual measurement task.
The attacking goal is the black bar at the TOP. Gold star = shooter; orange
square = opposing goalkeeper; red circles = other opponents; blue circles =
teammates. Use marker CENTERS. Ignore blue players when listing defenders.
Include each visible red player once, including a partly overlapped red marker
if identifiable. Do not invent hidden players.
Axes use normalized pitch units, not meters. LATERAL is 0 at goal center,
negative to image LEFT and positive to image RIGHT. DEPTH is 0 on the goal line
and increases DOWN the image. Posts are (-4,0) and (4,0). The penalty spot is
(0,12); penalty-area depth is 18. Read the plotted star even on a line or near a
goalpost; do not assume it is at the origin.
Return coordinates to approximately 0.1 units if possible using full-image axes.
Use null for an unidentifiable shooter or goalkeeper. Estimate no scoring
probability. Body part, ball height and play context are unknown. Return only
the requested JSON measurements.
"""
ROLES = {
'monolithic': 'Measure the shooter, opposing goalkeeper, and every visible red defender.',
'shooter_keeper': 'Your only task is to measure the gold shooter and orange opposing goalkeeper. Return only those two positions.',
'defenders': 'Your only task is to find and measure every visible red defender. Exclude the orange goalkeeper and all blue teammates. Return only the defender list.'}
POINT = {'type':'object','properties':{'lateral':{'type':'number','minimum':-44,'maximum':44},
'depth':{'type':'number','minimum':-4,'maximum':64}},'required':['lateral','depth'],'additionalProperties':False}
PROPERTIES = {'shooter':{'anyOf':[POINT,{'type':'null'}]},'goalkeeper':{'anyOf':[POINT,{'type':'null'}]},
'defenders':{'type':'array','items':POINT,'maxItems':20}}
SCHEMAS = {r:{'type':'object','properties':{k:copy.deepcopy(PROPERTIES[k]) for k in names},
'required':names,'additionalProperties':False} for r,names in
{'monolithic':['shooter','goalkeeper','defenders'],'shooter_keeper':['shooter','goalkeeper'],'defenders':['defenders']}.items()}
PRIMARY = [('specialists_image','monolithic_image'),('specialists_context','monolithic_context')]
LABELS = {
'monolithic_image':'One Luna reader, image','specialists_image':'Two Luna specialists, image',
'monolithic_context':'One Luna reader + event context','specialists_context':'Two Luna specialists + event context',
'metadata_only':'Event context only','training_rate':'Fixed training scoring rate',
'previous_noise_hybrid':'Previous Luna noise-trained hybrid','pixels_image':'Pixel reader, image',
'pixels_context':'Pixel reader + event context','known_image':'Known positions, image features',
'known_context':'Known positions + event context','simple_known_positions':'Simple baseline, known positions','sb_xg':'StatsBomb'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def save_json(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8'); tmp.replace(path)

def validate(role,answer):
    if role not in SCHEMAS or not isinstance(answer,dict) or set(answer)!=set(SCHEMAS[role]['properties']):
        raise ValueError('Unexpected response fields.')
    points=[answer[k] for k in ['shooter','goalkeeper'] if k in answer and answer[k] is not None]
    if 'defenders' in answer:
        if not isinstance(answer['defenders'],list) or len(answer['defenders'])>20:
            raise ValueError('Invalid defender list.')
        points.extend(answer['defenders'])
    for p in points:
        if not isinstance(p,dict) or set(p)!={'lateral','depth'}:
            raise ValueError('Invalid coordinate fields.')
        for k,low,high in [('lateral',-44,44),('depth',-4,64)]:
            v=p[k]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not low<=v<=high:
                raise ValueError('Invalid coordinate value.')
    return answer

def fuse(positions,defenders):
    validate('shooter_keeper',positions); validate('defenders',defenders)
    return validate('monolithic',{**copy.deepcopy(positions),**copy.deepcopy(defenders)})

def check_model(name):
    if name!=MODEL and not name.startswith(MODEL+'-'):
        raise ValueError('Non-Luna response rejected; no fallback allowed.')

def content_for(role,image):
    return [{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(image).decode(),'detail':'high'},
            {'type':'input_text','text':COMMON+'\n'+ROLES[role]}]

def request_hash(role,image_hash):
    return digest({'image_sha256':image_hash,'role':role,'prompt':COMMON+'\n'+ROLES[role],
                   'schema':SCHEMAS[role],'model':MODEL,'settings':SETTINGS})

def metadata_model():
    transform=ColumnTransformer([
    ('categories',OneHotEncoder(handle_unknown='ignore',min_frequency=20,sparse_output=False),CATEGORICAL),
    ('numbers',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),NUMERIC)])
    return make_pipeline(transform,LogisticRegression(C=.3,max_iter=3000))

def prepare(root,output):
    source=root/'improvement-suite/confirmation_bundesliga'
    selection=pd.read_json(source/'selection.jsonl',lines=True,precise_float=True)
    development=robust.load_development(root/'research-loop')
    if len(selection)!=200 or not selection.shot_id.is_unique:
        raise ValueError('Expected unchanged existing 200-shot cohort.')
    if set(selection.match_id)&set(development.match_id) or set(selection.shot_id)&set(development.shot_id):
        raise ValueError('Training/evaluation overlap.')
    context=pd.read_csv(source/'context.csv')
    train_context_path=root/'improvement-suite/context/pre_shot_context.csv'
    train_context=pd.read_csv(train_context_path)
    if not context.shot_id.is_unique or not set(selection.shot_id).issubset(set(context.shot_id)) or not train_context.shot_id.is_unique:
        raise ValueError('Invalid metadata join.')
    model_path=output/'metadata_only_model.joblib'
    if not model_path.exists():
        joblib.dump(metadata_model().fit(metadata_for(development,train_context),development.goal),model_path)
    protected=[source/p for p in ['selection.jsonl','context.csv','predictions.csv','pixel_answers.json',
                                 'inference/baseline_cache.jsonl','pixel_features.csv']]
    protected += [root/'noise-robustness-v1/frozen_models.joblib',root/'improvement-suite/context/frozen_models.joblib',
                  train_context_path,root/'research-loop/data/wc2018.jsonl',root/'research-loop/data/wc2022.jsonl']
    dependencies=[Path(__file__)]+[SUITE/p for p in ['robust_xg.py','spatial_models.py','context_trial.py',
                                                  'diagnose_geometry.py','vision_trials.py','fetch_context.py']]
    rng=np.random.default_rng(SEED)
    protocol={'study':'Luna task division and image/event late fusion v1','model':MODEL,
    'allowed_openai_models':[MODEL],'fallback':False,'settings':SETTINGS,'common_prompt':COMMON,'roles':ROLES,'schemas':SCHEMAS,
    'seed':SEED,'shots':selection.shot_id.tolist(),'n':200,'source':str(source),
    'schedule':[{'shot_id':s,'roles':rng.permutation(list(ROLES)).tolist()} for s in selection.shot_id],
    'images_sha256':{s:sha(source/'images'/f'{s}.png') for s in selection.shot_id},
    'protected_inputs_sha256':{str(p):sha(p) for p in protected},
    'implementation_sha256':{str(p):sha(p) for p in dependencies},'metadata_model_sha256':sha(model_path),
    'training_shots':len(development),'training_goals':int(development.goal.sum()),'training_rate':float(development.goal.mean()),
    'training_matches':sorted(int(m) for m in development.match_id.unique()),
    'evaluation_matches':sorted(int(m) for m in selection.match_id.unique()),
    'context_features':CATEGORICAL+NUMERIC,
    'context':'Saved shot type, play pattern, pressure and latest earlier same-team pass in same possession/period within 15 seconds. No body part added.',
    'fusion':'Concatenate specialist coordinates; code computes features; unchanged frozen noise-trained image and geometry/context heads. No test fitting or calibration.',
    'metadata_only':'Training-only logistic C=.3, same metadata transform; no tuning.',
    'visual_inputs':'Same unchanged image, shared geometry instructions, role task. No metadata, coordinates, goal outcomes or StatsBomb xG.',
    'missingness':'Retain all shots; frozen training imputation; never substitute true test coordinates.',
    'cohort_status':'Previously studied 200 Bundesliga shots in 34 Leverkusen fixtures; exploratory reuse, not new confirmation.',
    'primary_pairs':PRIMARY,'bootstrap':{'resamples':4000,'primary_confidence':.975,'secondary_confidence':.95,
    'unit':'whole match','multiplicity':'Bonferroni for two primary tests','refit':False},
    'request_limit':600,'sequential':True,'max_sdk_retries':3,
    'cost_limit':'Two specialists cost two calls per shot versus one new geometry-only monolithic call. No equal-cost or different-underlying-model claim.',
    'no_new_datasets':True,
    'documentation':['https://developers.openai.com/api/docs/guides/images-vision?api-mode=responses',
                     'https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses']}
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('Frozen protocol changed; use a new folder.')
    if not path.exists():
        save_json(path,protocol); selection[['shot_id','match_id']].to_csv(output/'selection.csv',index=False)
    return selection,metadata_for(selection,context),protocol,digest(protocol)

def load_cache(output,protocol,signature):
    path=output/'requests.jsonl'; cache={}
    if not path.exists(): return cache
    for line in path.read_text().splitlines():
        e=json.loads(line); s,r=e['shot_id'],e['role']
        if (s,r) in cache or r not in ROLES or s not in protocol['images_sha256']:
            raise ValueError('Duplicate or foreign cache entry.')
        if e['signature']!=signature or e['image_sha256']!=protocol['images_sha256'][s]:
            raise ValueError('Mixed signature or image.')
        if e['request_sha256']!=request_hash(r,e['image_sha256']):
            raise ValueError('Changed request payload.')
        check_model(e['returned_model']); validate(r,e['answer']); cache[(s,r)]=e
    return cache

def ask(client,role,image):
    if MODEL!='gpt-6-luna': raise ValueError('User authorized Luna only.')
    started=time.monotonic()
    response=client.responses.create(model=MODEL,store=False,reasoning={'effort':'low'},max_output_tokens=4096,
    input=[{'role':'user','content':content_for(role,image)}],
    text={'format':{'type':'json_schema','name':'geometry_'+role,'strict':True,'schema':SCHEMAS[role]}})
    check_model(response.model)
    if response.status!='completed' or not response.output_text:
        raise RuntimeError('Incomplete Luna response: '+str(response.status))
    return validate(role,json.loads(response.output_text)),{'response_id':response.id,'returned_model':response.model,
    'elapsed_seconds':time.monotonic()-started,'usage':response.usage.model_dump() if response.usage else {}}

def infer(output,protocol,signature,cache):
    from openai import OpenAI
    key=key_from_authorized_file()
    try:
        with OpenAI(api_key=key,max_retries=3,timeout=120.) as client:
            for item in protocol['schedule']:
                s=item['shot_id']; image=(Path(protocol['source'])/'images'/f'{s}.png').read_bytes()
                if hashlib.sha256(image).hexdigest()!=protocol['images_sha256'][s]:
                    raise ValueError('Image changed.')
                for r in item['roles']:
                    if (s,r) in cache: continue
                    answer,info=ask(client,r,image)
                    e={'shot_id':s,'role':r,'signature':signature,'image_sha256':protocol['images_sha256'][s],
                       'request_sha256':request_hash(r,protocol['images_sha256'][s]),'answer':answer,**info}
                    with (output/'requests.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(e,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
                    cache[(s,r)]=e
                    complete=sum(all((s,r) in cache for r in ROLES) for s in protocol['shots'])
                    save_json(output/'status.json',{'status':'running','complete_shots':complete,'shots':200,
                    'successful_requests':len(cache),'request_limit':600,'model':MODEL,'partial_scores_reported':False})
                    print(f'Luna division: {len(cache)}/600 requests; {complete}/200 shots; role={r}',flush=True)
    except Exception as error:
        message=str(error).replace(key,'[REDACTED]')
        save_json(output/'status.json',{'status':'incomplete','successful_requests':len(cache),'model':MODEL,
        'partial_scores_reported':False,'error_type':type(error).__name__,'error':message[:1200]})
        raise RuntimeError('Luna experiment stopped: '+type(error).__name__+': '+message) from None
    return cache

def paired_interval(frame,a,b,primary=False):
    difference=(frame[a]-frame.goal)**2-(frame[b]-frame.goal)**2
    grouped=pd.DataFrame({'match':frame.match_id,'difference':difference}).groupby('match').difference.agg(['sum','count'])
    rng=np.random.default_rng(SEED); ix=rng.integers(len(grouped),size=(4000,len(grouped)))
    draws=grouped['sum'].to_numpy()[ix].sum(axis=1)/grouped['count'].to_numpy()[ix].sum(axis=1)
    alpha=.025 if primary else .05
    return {'first':a,'second':b,'primary':primary,'brier_difference':float(difference.mean()),
    'confidence_level':1-alpha,'ci_low':float(np.quantile(draws,alpha/2)),'ci_high':float(np.quantile(draws,1-alpha/2))}

def verify_protected(protocol):
    if any(sha(p)!=h for p,h in protocol['protected_inputs_sha256'].items()):
        raise ValueError('Protected input changed.')

def evaluate(selection,context,output,protocol,signature,cache):
    if set(cache)!={(s,r) for s in protocol['shots'] for r in ROLES}:
        raise ValueError('Full 600-response cache required; partial cohorts are not scored.')
    verify_protected(protocol)
    source=Path(protocol['source']); root=source.parents[1]
    heads=joblib.load(root/'noise-robustness-v1/frozen_models.joblib')
    ctx=joblib.load(root/'improvement-suite/context/frozen_models.joblib')['robust_context']
    mono=[cache[(s,'monolithic')]['answer'] for s in selection.shot_id]
    split=[fuse(cache[(s,'shooter_keeper')]['answer'],cache[(s,'defenders')]['answer']) for s in selection.shot_id]
    pixels=json.loads((source/'pixel_answers.json').read_text())
    old={e['shot_id']:e['answer'] for e in map(json.loads,(source/'inference/baseline_cache.jsonl').read_text().splitlines())}
    previous=[old[s] for s in selection.shot_id]
    result=selection[['shot_id','match_id','goal','sb_xg']].reset_index(drop=True).copy()
    reference=pd.read_csv(source/'predictions.csv').set_index('shot_id').loc[selection.shot_id]
    result['previous_noise_hybrid']=reference.noise_trained.to_numpy()
    result['simple_known_positions']=reference.simple_known_positions.to_numpy()
    for label,features in {'monolithic':robust.external_features(mono),'specialists':robust.external_features(split),
    'pixels':robust.external_features(pixels),'known':spatial.known_features(selection)}.items():
        result[label+'_image']=robust.predict(heads['noise_trained'],features)
        result[label+'_context']=ctx.predict_proba(join_features(features,context))[:,1]
        features.to_csv(output/(label+'_features.csv'),index=False)
    result['metadata_only']=joblib.load(output/'metadata_only_model.joblib').predict_proba(context)[:,1]
    result['training_rate']=protocol['training_rate']
    old_features=robust.external_features(previous)
    errors={'previous_geometry':float(np.max(np.abs(robust.predict(heads['noise_trained'],old_features)-result.previous_noise_hybrid))),
    'previous_context':float(np.max(np.abs(ctx.predict_proba(join_features(old_features,context))[:,1]-reference.noise_plus_context.to_numpy()))),
    'pixels':float(np.max(np.abs(result.pixels_image-reference.pixels_noise_head.to_numpy())))}
    if max(errors.values())>1e-10: raise ValueError('Existing controls failed reproduction.')
    scores=pd.DataFrame([{'method':c,'label':LABELS[c],**robust.metrics(result.goal,result[c])} for c in LABELS])
    secondary=[('specialists_context','specialists_image'),('monolithic_context','monolithic_image'),
    ('specialists_context','metadata_only'),('specialists_context','pixels_context'),('specialists_image','previous_noise_hybrid'),
    ('specialists_context','sb_xg'),('specialists_context','simple_known_positions'),('specialists_image','known_image')]
    comparisons=pd.DataFrame([paired_interval(result,a,b,True) for a,b in PRIMARY]+[paired_interval(result,a,b) for a,b in secondary])
    details=[]; summaries=[]
    for label,answers in [('monolithic',mono),('specialists',split),('previous',previous),('pixels',pixels)]:
        detail,summary=audit(selection,answers,label); details.append(detail); summaries.append(summary)
    pd.concat(details,ignore_index=True).to_csv(output/'geometry_audit.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'geometry_summary.csv',index=False)
    result.to_csv(output/'predictions.csv',index=False); scores.to_csv(output/'scores.csv',index=False)
    comparisons.to_csv(output/'comparisons.csv',index=False); save_json(output/'specialist_answers.json',split)
    usage=[]
    for r in ROLES:
        entries=[e for (s,role),e in cache.items() if role==r]
        usage.append({'role':r,'requests':len(entries),
        'input_tokens':sum((e['usage'] or {}).get('input_tokens',0) for e in entries),
        'output_tokens':sum((e['usage'] or {}).get('output_tokens',0) for e in entries),
        'elapsed_seconds':sum(e['elapsed_seconds'] for e in entries)})
    pd.DataFrame(usage).to_csv(output/'usage.csv',index=False)
    save_json(output/'integrity.json',{'protected_inputs_unchanged':True,'control_max_differences':errors,
    'model':MODEL,'returned_models':sorted(set(e['returned_model'] for e in cache.values())),
    'complete_requests':len(cache),'protocol_signature':signature,'train_test_match_overlap':False,'test_fitting':False})
    save_json(output/'status.json',{'status':'complete','complete_shots':200,'successful_requests':len(cache),'model':MODEL})
    print(scores.to_string(index=False),flush=True); print(comparisons.to_string(index=False),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'task-division-v1')
    parser.add_argument('--prepare-only',action='store_true'); parser.add_argument('--offline',action='store_true')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (args.output/'runner.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Another division runner holds this output folder.') from None
        selection,context,protocol,signature=prepare(args.root,args.output)
        cache=load_cache(args.output,protocol,signature)
        print(f'Frozen Luna-only protocol; {len(cache)}/600 responses cached.',flush=True)
        if args.prepare_only: return
        if not args.offline: cache=infer(args.output,protocol,signature,cache)
        evaluate(selection,context,args.output,protocol,signature,cache)
if __name__=='__main__':
    with threadpool_limits(limits=2): main()
