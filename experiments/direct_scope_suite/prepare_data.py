"""Fetch public eligible shots; freeze match-disjoint cohorts before paid inference."""
import csv, json, math, random, sys
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import common as c
sys.path.insert(0,str(c.ROOT))
import xg_vlm_pilot as original
BASE='https://raw.githubusercontent.com/hudl/open-data/master/data/'
DATA=c.OUT/'data'; DATA.mkdir(exist_ok=True)

def fetch(relative):
    file=DATA/relative.replace('/','_')
    if not file.exists():
        with urlopen(BASE+relative,timeout=90) as response: content=response.read()
        json.loads(content); file.write_bytes(content)
    return json.loads(file.read_bytes())

def convert(events,mid):
    rows=[]
    for e in events:
        if e.get('type',{}).get('name')!='Shot': continue
        shot=e['shot']; ff=shot.get('freeze_frame')
        if shot.get('type',{}).get('name')=='Penalty' or not isinstance(ff,list): continue
        x,y=e['location'][:2]; distance,angle=original.geometry(x,y)
        opponents=[p for p in ff if not p['teammate']]
        cone=sum(original.in_triangle(p['location'][:2],(x,y),original.POST_L,original.POST_R) for p in opponents)
        rows.append({'match_id':int(mid),'shot_id':e['id'],'x':x,'y':y,'ff':ff,'goal':int(shot['outcome']['name']=='Goal'),'sb_xg':shot['statsbomb_xg'],
                     'body_part':shot.get('body_part',{}).get('name'),'header':int(shot.get('body_part',{}).get('name')=='Head'),'distance':distance,'angle':angle,'defenders_in_cone':cone})
    return rows

def main():
    c.campaign_protocol()
    old=Path('/mnt/c/Users/ephou/Downloads/xG/output/research-loop/data')
    datasets={name:pd.read_json(old/(name+'.jsonl'),lines=True,precise_float=True) for name in ['wc2018','wc2022','euro2020','euro2024']}
    used=set(int(v) for frame in datasets.values() for v in frame.match_id)
    old_bundes=pd.read_json(c.ROOT/'results/improvement_suite/confirmation_bundesliga/selection.jsonl',lines=True)
    used.update(int(v) for v in old_bundes.match_id)
    competitions=fetch('competitions.json')
    found=[x for x in competitions if 'Cup of Nations' in x['competition_name'] and x.get('competition_gender')=='male' and x['season_name']=='2023']
    if len(found)!=1: raise ValueError('Fresh competition cannot be identified uniquely')
    competition=found[0]; matches=fetch(f"matches/{competition['competition_id']}/{competition['season_id']}.json")
    if used.intersection(int(m['match_id']) for m in matches): raise ValueError('Fresh competition overlaps previous match IDs')
    allrows=[]
    for i,m in enumerate(matches,1):
        allrows.extend(convert(fetch(f"events/{m['match_id']}.json"),m['match_id']))
        if i%10==0 or i==len(matches): print(f'Public match files prepared: {i}/{len(matches)}',flush=True)
    if len({r['shot_id'] for r in allrows})!=len(allrows): raise ValueError('Duplicate shot IDs')
    ids=sorted({r['match_id'] for r in allrows}); rng=random.Random(2026102902); rng.shuffle(ids)
    if len(ids)<30: raise ValueError('Too few new match groups for the planned disjoint study')
    development=set(ids[:16]); confirmation=set(ids[16:])
    devpool=sorted([r for r in allrows if r['match_id'] in development],key=lambda r:r['shot_id'])
    testpool=sorted([r for r in allrows if r['match_id'] in confirmation],key=lambda r:r['shot_id'])
    dev=rng.sample(devpool,50); test=rng.sample(testpool,150)
    train=pd.concat([datasets['wc2018'],datasets['wc2022']],ignore_index=True)
    features=['distance','angle','header','defenders_in_cone']; imagefeatures=['distance','angle','defenders_in_cone']
    model=LogisticRegression(max_iter=1000).fit(train[features],train.goal)
    imagemodel=LogisticRegression(max_iter=1000).fit(train[imagefeatures],train.goal)
    for rows in (dev,test):
        frame=pd.DataFrame(rows); full=model.predict_proba(frame[features])[:,1]; image=imagemodel.predict_proba(frame[imagefeatures])[:,1]
        for r,p,q in zip(rows,full,image): r.update(baseline_xg=float(p),baseline_image_xg=float(q))
    c.freeze(DATA/'match_split.json',{'competition':competition,'used_match_ids':sorted(used),'development_match_ids':sorted(development),'confirmation_match_ids':sorted(confirmation),
                                   'source_hashes':{p.name:c.sha(p) for p in DATA.glob('*.json') if p.name not in ('match_split.json',)},
                                   'training_source_hashes':{name:c.sha(old/(name+'.jsonl')) for name in datasets},'training_shots':len(train),
                                   'logistic_features':features,'logistic_coefficients':model.coef_.tolist(),'logistic_intercept':model.intercept_.tolist(),
                                   'matched_image_features':imagefeatures,'matched_coefficients':imagemodel.coef_.tolist(),'matched_intercept':imagemodel.intercept_.tolist(),
                                   'selection':'Seeded whole-match split, then natural random shot samples without outcome stratification. All prior match IDs excluded.'})
    for name,rows in [('reasoning',dev),('confirmation',test)]:
        directory=c.OUT/name; directory.mkdir(exist_ok=True); c.freeze(directory/'selection.json',rows)
        images=c.OUT/'images'/name; images.mkdir(parents=True,exist_ok=True)
        for r in rows:
            path=images/(r['shot_id']+'.png')
            if not path.exists(): original.render(pd.Series(r),path)
    jobs=[]; order=random.Random(2026102903)
    for r in dev:
        efforts=['low','medium','high']; order.shuffle(efforts)
        jobs.extend({'shot_id':r['shot_id'],'condition':e,'effort':e,'image':f"images/reasoning/{r['shot_id']}.png"} for e in efforts)
    c.new_stage('reasoning',jobs,{'selection_sha256':c.sha(c.OUT/'reasoning/selection.json'),'match_split_sha256':c.sha(DATA/'match_split.json'),
                               'primary':'medium minus low Brier, 95% whole-match interval. High is an exploratory effort comparison; winner selected on this development cohort only.',
                               'output_budget_note':'All three efforts receive the same 4096-token allowance, frozen before requests, to reduce truncation risk. Earlier 12-shot pilot used 2048.'})
    print(f'Frozen new cohorts: 50 reasoning shots and 150 confirmation shots in disjoint match groups; {len(used)} prior matches excluded. No paid inference.',flush=True)
if __name__=='__main__': main()
