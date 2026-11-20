"""Frozen external VLM experiment with defenders, readable axes, and spline xG."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import spatial_models as spatial
from search_spatial_models import ROOT, metrics

MODEL='gpt-6-luna'
SETTINGS={'reasoning_effort':'low','image_detail':'high','max_output_tokens':4096}
PROMPT="""Read this football shot diagram carefully, then estimate its scoring probability.
The attacking goal is the black bar at the TOP. Gold star = shooter, orange square
= opposing goalkeeper, red circles = other opponents, blue circles = teammates.
Use marker CENTERS. Ignore blue players when listing defenders. Include each
visible red player once, including a partly overlapped red marker if identifiable.
Do not invent hidden players.

The image axes give normalized pitch units, not meters. LATERAL is 0 at goal
center, negative to image LEFT and positive to image RIGHT. DEPTH is 0 on the goal
line and increases DOWN the image. Posts are (-4,0) and (4,0). The penalty spot is
(0,12); the penalty-area depth is 18. Read the plotted star even when it is on a
line or near a goalpost; do not assume it is at the origin.

Return shooter coordinates, goalkeeper coordinates, and a list of red defender
coordinates in this lateral/depth system, to approximately 0.1 units if possible.
Use null for shooter or goalkeeper if their marker is not identifiable.
Then estimate xG between 0 and 1, assuming an average professional finisher.
Body part, ball height, and play context are UNKNOWN: do not invent them.
Give one short sentence explaining xG. Return only the requested JSON.
"""
POINT={'type':'object','properties':{'lateral':{'type':'number','minimum':-44,'maximum':44},
                                    'depth':{'type':'number','minimum':-4,'maximum':64}},
       'required':['lateral','depth'],'additionalProperties':False}
NULL_POINT={'anyOf':[POINT,{'type':'null'}]}
SCHEMA={'type':'object','properties':{'shooter':NULL_POINT,'goalkeeper':NULL_POINT,
        'defenders':{'type':'array','items':POINT,'maxItems':20},
        'xg':{'type':'number','minimum':0,'maximum':1},'reason':{'type':'string'}},
        'required':['shooter','goalkeeper','defenders','xg','reason'],'additionalProperties':False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_grid(row,path):
    figure,axis=plt.subplots(figsize=(7,6))
    axis.set(xlim=(-44,44),ylim=(64,-4),aspect='equal',
             xlabel='Lateral position (pitch units; 0 = goal center)',
             ylabel='Depth from goal line (pitch units)')
    axis.set_xticks(np.arange(-40,41,10)); axis.set_yticks(np.arange(0,61,10))
    axis.grid(color='#dddddd',linewidth=.6,zorder=0)
    for left,right,depth in [(-40,40,60),(-22,22,18),(-10,10,6)]:
        axis.add_patch(Rectangle((left,0),right-left,depth,fill=False,edgecolor='black',linewidth=1.3,zorder=1))
    axis.plot([-4,4],[0,0],color='black',linewidth=4,zorder=3)
    axis.scatter([0,0],[12,60],s=5,c='black',zorder=2)
    axis.add_patch(Circle((0,60),10,fill=False,edgecolor='black',linewidth=1.2))
    keepers=[]
    for player in row.ff:
        x,y=player['location'][:2]
        lateral,depth=float(y)-40,120-float(x)
        keeper=not player['teammate'] and (player.get('position') or {}).get('name')=='Goalkeeper'
        if keeper:
            keepers.append((lateral,depth)); continue
        axis.scatter(lateral,depth,s=85,c='royalblue' if player['teammate'] else 'red',
                     edgecolors='black',linewidths=.7,zorder=4)
    for lateral,depth in keepers:
        axis.scatter(lateral,depth,s=140,c='orange',marker='s',edgecolors='black',linewidths=.8,zorder=5)
    axis.scatter(float(row.y)-40,120-float(row.x),s=290,c='gold',marker='*',edgecolors='black',linewidths=1,zorder=6)
    axis.set_title('Shot situation — goal at top',fontsize=12)
    figure.tight_layout()
    figure.savefig(path,dpi=130)
    plt.close(figure)


def validate(answer):
    if set(answer)!=set(SCHEMA['properties']):
        raise ValueError('Invalid fields in rich geometry response.')
    for point in [answer['shooter'],answer['goalkeeper'],*answer['defenders']]:
        if point is None:
            continue
        if set(point)!={'lateral','depth'}:
            raise ValueError('Invalid point fields.')
        for key,low,high in [('lateral',-44,44),('depth',-4,64)]:
            value=point[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError('Invalid coordinate.')
    if not isinstance(answer['defenders'],list) or len(answer['defenders'])>20 or any(p is None for p in answer['defenders']):
        raise ValueError('Invalid defender list.')
    if isinstance(answer['xg'],bool) or not isinstance(answer['xg'],(int,float)) or not math.isfinite(answer['xg']) or not 0<=answer['xg']<=1:
        raise ValueError('Invalid direct probability.')
    if not isinstance(answer['reason'],str):
        raise ValueError('Invalid reason.')
    return answer


def ask(client,image_bytes):
    response=client.responses.create(model=MODEL,store=False,reasoning={'effort':'low'},
        max_output_tokens=SETTINGS['max_output_tokens'],
        input=[{'role':'user','content':[{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(image_bytes).decode(),'detail':'high'},
                                       {'type':'input_text','text':PROMPT}]}],
        text={'format':{'type':'json_schema','name':'rich_shot_reading','strict':True,'schema':SCHEMA}})
    if response.status!='completed' or not response.output_text:
        raise RuntimeError(f'Incomplete rich geometry response: {response.status}')
    return validate(json.loads(response.output_text)),{'response_id':response.id,'returned_model':response.model,
                'usage':response.usage.model_dump() if response.usage else None}


def extracted_features(answer):
    if answer['shooter'] is None:
        return {key:np.nan for key in set(spatial.RICH+spatial.BASE)}
    def pair(point): return [point['lateral'],point['depth']]
    return spatial.geometry_features(pair(answer['shooter']),
                [pair(answer['goalkeeper'])] if answer['goalkeeper'] is not None else [],
                [pair(p) for p in answer['defenders']])


def prepare():
    selection=pd.read_json(ROOT/'external_vlm_selection.jsonl',lines=True,precise_float=True)
    output=ROOT/'rich_vlm_external'
    images=output/'images'
    images.mkdir(parents=True,exist_ok=True)
    protocol={'model':MODEL,'prompt':PROMPT,'schema':SCHEMA,'settings':SETTINGS,
              'shots':selection.shot_id.tolist(),'n':len(selection),'selection_seed':7,
              'probability_models_sha256':sha(ROOT/'frozen_probability_models.joblib'),
              'feature_code_sha256':sha(Path(spatial.__file__)),
              'implementation_sha256':sha(Path(__file__)),
              'source_sha256':sha(ROOT/'external_vlm_selection.jsonl'),
              'rendering':'Generic labelled pitch axes; keeper and shooter rendered last for visibility; no shot-specific numeric labels',
              'no_external_model_tuning':True,'body_part_only_in_separate_statistical_context_condition':True}
    protocol_path=output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Rich VLM protocol changed; use a fresh output folder.')
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    for row in selection.itertuples():
        path=images/f'{row.shot_id}.png'
        if not path.exists():
            render_grid(row,path)
    print(f'Prepared {len(selection)} external images and froze rich-VLM protocol.',flush=True)
    return selection,output,images,signature


def infer(selection,output,images,signature,offline=False):
    path=output/'cache.jsonl'
    cache={}
    if path.exists():
        for line in path.read_text().splitlines():
            entry=json.loads(line)
            if entry['signature']!=signature or entry['shot_id'] in cache:
                raise ValueError('Mixed or duplicate rich-VLM cache.')
            validate(entry['answer'])
            cache[entry['shot_id']]=entry
    if offline and not set(selection.shot_id).issubset(cache):
        raise ValueError('Offline run requires every response cached.')
    client=None
    answers=[]
    for index,row in enumerate(selection.itertuples(),1):
        image=images/f'{row.shot_id}.png'
        image_hash=sha(image)
        if row.shot_id in cache:
            entry=cache[row.shot_id]
            if entry['image_sha256']!=image_hash:
                raise ValueError('Cached image changed.')
        else:
            if client is None:
                from openai import OpenAI
                client=OpenAI(max_retries=6,timeout=120.)
            answer,info=ask(client,image.read_bytes())
            entry={'shot_id':row.shot_id,'image_sha256':image_hash,'signature':signature,'answer':answer,**info}
            with path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(entry,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
            cache[row.shot_id]=entry
        answers.append(entry['answer'])
        print(f'  External rich VLM {index}/{len(selection)}',flush=True)
    return answers


def evaluate(selection,answers,output):
    reference=pd.read_csv(ROOT/'external_reference_predictions.csv').set_index('shot_id').loc[selection.shot_id].reset_index()
    result=reference.copy()
    features=pd.DataFrame([extracted_features(answer) for answer in answers])
    features['header']=selection.header.to_numpy()
    features.to_csv(output/'vlm_features.csv',index=False)
    result['vlm_direct']=[answer['xg'] for answer in answers]
    models=joblib.load(ROOT/'frozen_probability_models.joblib')
    for condition in ['image','context']:
        value=models['spatial_'+condition]
        result['hybrid_'+condition]=value['model'].predict_proba(features[value['columns']])[:,1]
    scores=[]
    for column in ['vlm_direct','hybrid_image','hybrid_context','spatial_image','spatial_context','baseline_image','baseline_context','sb_xg']:
        scores.append({'method':column,**metrics(result,result[column])})
    repo=Path('/home/evan/xG_Research/vlm-xg-benchmark')
    sys.path.insert(0,str(repo))
    from calibrate_xg import match_bootstrap
    comparisons=[match_bootstrap(result,first,second,4000,7) for first,second in [
        ('hybrid_image','baseline_image'),('hybrid_image','baseline_context'),('hybrid_image','sb_xg'),
        ('hybrid_image','vlm_direct'),('hybrid_image','spatial_image'),
        ('hybrid_context','baseline_context'),('hybrid_context','sb_xg'),
        ('vlm_direct','baseline_image'),('vlm_direct','baseline_context')]]
    result.to_csv(output/'predictions.csv',index=False)
    pd.DataFrame(scores).to_csv(output/'scores.csv',index=False)
    pd.DataFrame(comparisons).to_csv(output/'comparisons.csv',index=False)
    print(f'Fresh VLM cohort: {len(result)} shots; {int(result.goal.sum())} goals; {result.match_id.nunique()} matches.',flush=True)
    print(pd.DataFrame(scores).to_string(index=False),flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    selection,output,images,signature=prepare()
    if args.prepare_only:
        return
    answers=infer(selection,output,images,signature,args.offline)
    evaluate(selection,answers,output)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
