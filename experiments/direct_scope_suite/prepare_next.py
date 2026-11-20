"""Prepare subsequent stages from development-only choices; no model calls."""
import argparse, csv, inspect, random, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import common as c
sys.path.insert(0,str(c.ROOT)); import xg_vlm_pilot as original

def outline(row,path):
    # Only the final shooter scatter style changes; coordinates, size, canvas and other markers remain unchanged.
    code=inspect.getsource(original.render)
    old='s=320, c="gold", marker="*", edgecolors="black"'
    new='s=320, facecolors="none", marker="*", edgecolors="#b8860b", linewidths=2'
    if code.count(old)!=1: raise ValueError('Original rendering source changed')
    namespace={'VerticalPitch':original.VerticalPitch,'plt':plt}; exec(compile(code.replace(old,new),'<outline renderer>','exec'),namespace)
    namespace['render'](row,path)

def rendering():
    effort=c.read(c.OUT/'reasoning_choice.json')['effort']; folder=c.OUT/'rendering'; folder.mkdir(exist_ok=True)
    old=Path('/mnt/c/Users/ephou/Downloads/xG/output/research-loop/data/wc2022.jsonl'); source=pd.read_json(old,lines=True,precise_float=True).set_index('shot_id')
    baseline=pd.read_csv(c.ROOT/'results/gpt_6_luna/shots.csv'); pilot={r['shot_id'] for r in c.read(c.ROOT/'results/reasoning_effort_v1/selection.json')}
    eligible=baseline[~baseline.shot_id.isin(pilot)].sort_values('shot_id'); ids=random.Random(2026102904).sample(eligible.shot_id.tolist(),40)
    indexed=baseline.set_index('shot_id'); rows=[]; jobs=[]; rng=random.Random(2026102905)
    for sid in ids:
        r=source.loc[sid].to_dict(); r['shot_id']=sid; r['baseline_xg']=float(indexed.loc[sid].baseline_xg); r['match_id']=int(r['match_id']); r['goal']=int(r['goal'])
        rows.append(r)
        for label in ('standard','outline'):
            path=c.OUT/'images/rendering'/label/(sid+'.png'); path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                if label=='standard': original.render(pd.Series(r),path)
                else: outline(pd.Series(r),path)
        arms=['standard','outline']; rng.shuffle(arms)
        jobs.extend({'shot_id':sid,'condition':a,'effort':effort,'image':f'images/rendering/{a}/{sid}.png'} for a in arms)
    c.freeze(folder/'selection.json',rows)
    c.new_stage('rendering',jobs,{'effort':effort,'selection_sha256':c.sha(folder/'selection.json'),'selection':'40 original World Cup development shots excluding pilot, selected without outcomes or prior errors',
                               'change':'Hollow dark-gold shooter star; same size and center, same pitch and every other marker. No shift, count annotation or numerical answers.',
                               'primary':'outline minus standard Brier, exploratory whole-match 95% interval; independent AFCON confirmation determines whether any selected gain generalizes.'})
    print('Prepared fixed 40-shot paired marker-visibility study; no API calls.')

def confirmation():
    candidate=c.read(c.OUT/'candidate.json'); rows=c.read(c.OUT/'confirmation/selection.json'); jobs=[]; rng=random.Random(2026102906)
    for r in rows:
        sid=r['shot_id']; standard=f'images/confirmation/{sid}.png'; candidate_image=standard
        if candidate['renderer']=='outline':
            target=c.OUT/'images/confirmation_outline'/(sid+'.png'); target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists(): outline(pd.Series(r),target)
            candidate_image=str(target.relative_to(c.OUT))
        # Aliases avoid paying twice for an identical condition.
        actual=[('original_low','low',standard)]
        if candidate['renderer']=='standard' and candidate['effort']=='low': aliases={'candidate':'original_low'}
        else:
            actual.append(('candidate',candidate['effort'],candidate_image)); aliases={}
        if candidate['renderer']=='outline' and candidate['effort']!='low': actual.append(('standard_selected',candidate['effort'],standard))
        rng.shuffle(actual)
        jobs.extend({'shot_id':sid,'condition':a,'effort':e,'image':i} for a,e,i in actual)
    c.new_stage('confirmation',jobs,{'candidate':candidate,'candidate_sha256':c.sha(c.OUT/'candidate.json'),'selection_sha256':c.sha(c.OUT/'confirmation/selection.json'),
                                    'aliases':aliases,'primary':'candidate minus four-feature logistic and candidate minus original_low, 97.5% intervals for two primary comparisons',
                                    'independence':'Entire confirmation match groups excluded from all previous work, development selections and baseline training; fixed 150-shot natural sample.'})
    print('Candidate frozen and independent confirmation schedule prepared; no API calls.')

def invariance():
    effort=c.read(c.OUT/'reasoning_choice.json')['effort']; rows=c.read(c.OUT/'rendering/selection.json')[:20]; folder=c.OUT/'invariance'; folder.mkdir(exist_ok=True)
    jobs=[]; rng=random.Random(2026102907)
    for r in rows:
        sid=r['shot_id']; standard=f'images/rendering/standard/{sid}.png'; target=c.OUT/'images/mirrored'/(sid+'.png'); target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            with Image.open(c.OUT/standard) as image: ImageOps.mirror(image).save(target)
        arms=['repeat','mirror']; rng.shuffle(arms)
        jobs.extend({'shot_id':sid,'condition':a,'effort':effort,'image':standard if a=='repeat' else str(target.relative_to(c.OUT))} for a in arms)
    c.freeze(folder/'selection.json',rows)
    c.new_stage('invariance',jobs,{'primary':'Mean absolute change for mirror minus exact-repeat, against reused original standard readings; match-bootstrap interval.',
                                 'diagnostic':'20 development scenes; reflection keeps goal at top and all spatial relationships mirrored. Unknown body part; approximate symmetry expected, not strict physical equivalence.',
                                 'selection':'First 20 prespecified rendering shots; no outcome or prediction selection; no probability averaging or adjustment.'})
    print('Prepared mirror versus exact-repeat diagnostic; no API calls.')
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['rendering','confirmation','invariance']); args=ap.parse_args(); globals()[args.stage]()
