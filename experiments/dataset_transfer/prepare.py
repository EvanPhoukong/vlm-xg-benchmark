"""Outcome-blind dataset selection and source-to-image preparation; no inference."""
import concurrent.futures,csv,hashlib,importlib.util,json,math,random,sys
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/dataset_transfer_v1'; DATA=OUT/'data'
sys.path.insert(0,str(ROOT)); import xg_vlm_pilot as original
sp=importlib.util.spec_from_file_location('shared_direct',ROOT/'experiments/direct_model_search/run.py')
base=importlib.util.module_from_spec(sp); sp.loader.exec_module(base)
SPECS={'premier_league':{'competition_id':2,'season_id':27,'train_matches':40,'test_matches':40,'shots':500,'seed':2026103101},
       'womens_world_cup':{'competition_id':72,'season_id':30,'train_matches':26,'test_matches':26,'shots':500,'seed':2026103102}}


def fetch(url,name):
    path=DATA/name
    if not path.exists():
        with urlopen(url,timeout=120) as response: content=response.read()
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(path.suffix+'.download'); temp.write_bytes(content); temp.replace(path)
    return path


def geometry_row(mid,sid,x,y,ff,goal,header,sb=None):
    distance,angle=original.geometry(x,y)
    cone=sum(original.in_triangle(p['location'][:2],(x,y),original.POST_L,original.POST_R) for p in ff if not p['teammate'])
    return {'match_id':mid,'shot_id':sid,'x':x,'y':y,'ff':ff,'goal':goal,'header':header,
            'distance':distance,'angle':angle,'defenders_in_cone':cone,'sb_xg':sb}


def sb_rows(path,mid):
    rows=[]; omitted={}
    for event in json.loads(path.read_text()):
        if event.get('type',{}).get('name')!='Shot': continue
        shot=event['shot']; ff=shot.get('freeze_frame'); x,y=event['location'][:2]
        reason=('penalty' if shot.get('type',{}).get('name')=='Penalty' else
                'missing_freeze_frame' if not isinstance(ff,list) else
                'shooter_outside_rendered_half' if not(60<=x<=120 and 0<=y<=80) else None)
        if reason:
            omitted[reason]=omitted.get(reason,0)+1; continue
        rows.append(geometry_row(int(mid),event['id'],x,y,ff,int(shot['outcome']['name']=='Goal'),
                                int(shot.get('body_part',{}).get('name')=='Head'),shot['statsbomb_xg']))
    return rows,omitted


def predict_frozen(rows):
    frozen=base.read(ROOT/'results/direct_scope_suite/data/match_split.json')
    for row in rows:
        for label,features,coeff,intercept in [
            ('original_logistic',frozen['logistic_features'],frozen['logistic_coefficients'],frozen['logistic_intercept']),
            ('original_no_header',frozen['matched_image_features'],frozen['matched_coefficients'],frozen['matched_intercept'])]:
            z=intercept[0]+sum(row[k]*v for k,v in zip(features,coeff[0])); row[label]=1/(1+math.exp(-z))


def source_registry():
    old=base.read(ROOT/'results/direct_model_search_v1/data/match_split.json')
    used=set(old['previous_used_match_ids']+old['selected_match_ids'])
    return used


def freeze_plan():
    DATA.mkdir(parents=True,exist_ok=True); used=source_registry(); cohorts={}
    saved=base.read(OUT/'data_plan.json')
    if saved:
        if saved['prior_used_match_ids']!=sorted(used) or saved['renderer_sha256']!=base.sha(ROOT/'xg_vlm_pilot.py'):
            raise ValueError('Frozen prior-match registry or renderer changed')
        for name,spec in SPECS.items():
            if any(saved['cohorts'][name][key]!=value for key,value in spec.items()):
                raise ValueError('Frozen cohort specification changed')
        return saved['cohorts']
    for name,spec in SPECS.items():
        url=f"https://raw.githubusercontent.com/hudl/open-data/master/data/matches/{spec['competition_id']}/{spec['season_id']}.json"
        path=fetch(url,name+'_matches.json'); matches=json.loads(path.read_text())
        eligible=sorted(m['match_id'] for m in matches if int(m['match_id']) not in used)
        rng=random.Random(spec['seed']); selected=rng.sample(eligible,spec['train_matches']+spec['test_matches'])
        cohorts[name]={**spec,'train_ids':sorted(selected[:spec['train_matches']]),'test_ids':sorted(selected[spec['train_matches']:]),
                       'match_manifest_sha256':base.sha(path),'previous_overlap':sorted(set(selected)&used)}
    base.freeze(OUT/'data_plan.json',{'cohorts':cohorts,'prior_used_match_ids':sorted(used),
        'metrica':'All eligible non-penalty shots from public Sample Games 1 and 2, source-specific match IDs; exploratory 2-match pilot. Anonymization prevents ruling out cross-provider identity overlap.',
        'eligibility':'Non-penalty shots, valid recorded shooter in attacking half. StatsBomb requires shot freeze frame. Metrica requires synchronized tracking and a detectable shooter. Filters do not use outcomes or earlier errors.',
        'selection':'Whole-match train/test split chosen before event downloads, then fixed natural random 500-shot test samples. No goal balancing, result-driven replacements, prompt tuning or sequential cohort extension.',
        'renderer_sha256':base.sha(ROOT/'xg_vlm_pilot.py'),'preparation_sha256':base.sha(__file__)})
    return cohorts


def prepare_statsbomb(cohorts):
    jobs=[(name,mid) for name,spec in cohorts.items() for mid in spec['train_ids']+spec['test_ids']]
    def download(job):
        name,mid=job
        return job,fetch(f'https://raw.githubusercontent.com/hudl/open-data/master/data/events/{mid}.json',f'statsbomb/events_{mid}.json')
    files={}; done=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for job,path in pool.map(download,jobs):
            files[job]=path; done+=1
            if done%20==0 or done==len(jobs): print(f'Public StatsBomb match files: {done}/{len(jobs)}',flush=True)
    for name,spec in cohorts.items():
        train=[]; pool=[]; omitted={}; hashes={}
        for mid in spec['train_ids']+spec['test_ids']:
            path=files[name,mid]; rows,ex=sb_rows(path,mid); hashes[str(mid)]=base.sha(path)
            (train if mid in spec['train_ids'] else pool).extend(rows)
            for reason,count in ex.items(): omitted[reason]=omitted.get(reason,0)+count
        if len(pool)<spec['shots']: raise ValueError(name+': fewer than 500 eligible test shots; do not silently alter cohort')
        selected=random.Random(spec['seed']+100).sample(sorted(pool,key=lambda r:r['shot_id']),spec['shots'])
        features=['distance','angle','header','defenders_in_cone']; image_features=['distance','angle','defenders_in_cone']
        model=LogisticRegression(max_iter=1000).fit(pd.DataFrame(train)[features],[r['goal'] for r in train])
        image_model=LogisticRegression(max_iter=1000).fit(pd.DataFrame(train)[image_features],[r['goal'] for r in train])
        local=model.predict_proba(pd.DataFrame(selected)[features])[:,1]
        local_image=image_model.predict_proba(pd.DataFrame(selected)[image_features])[:,1]
        training_prior=sum(r['goal'] for r in train)/len(train)
        for row,p,q in zip(selected,local,local_image):
            row.update(local_logistic=float(p),local_no_header=float(q),training_prior=training_prior)
        predict_frozen(selected)
        base.freeze(OUT/name/'selection.json',selected)
        base.freeze(OUT/name/'baseline.json',{'features':features,'coefficients':model.coef_.tolist(),'intercept':model.intercept_.tolist(),
            'image_features':image_features,'image_coefficients':image_model.coef_.tolist(),'image_intercept':image_model.intercept_.tolist(),
            'training_matches':spec['train_ids'],'training_shots':len(train),'training_goals':sum(r['goal'] for r in train),
            'test_pool_shots':len(pool),'source_hashes':hashes,'exclusions_all_downloaded_matches':omitted,
            'note':'Separate statistical references only. Local fits and original frozen WC2018/22 fits never supply or correct VLM probabilities.'})
        render(name,selected)
        print(f'{name}: frozen 500-shot cohort and separately fitted references.',flush=True)


def tracking(path):
    with path.open(newline='') as handle:
        reader=csv.reader(handle); first=next(reader); jersey=next(reader); columns=next(reader)
    columns=columns[:3]+[f'Player{jersey[i]}_{axis}' for i in range(3,len(columns)-2,2) for axis in ('x','y')]+['ball_x','ball_y']
    return pd.read_csv(path,skiprows=3,names=columns).set_index('Frame')


def prepare_metrica():
    files={}; download_jobs=[]
    for game in (1,2):
        for team in ('Home','Away'):
            name=f'metrica_{game}_{team}_tracking.csv'
            url=f'https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{game}/Sample_Game_{game}_RawTrackingData_{team}_Team.csv'
            download_jobs.append((game,team,url,name))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for (game,team,url,name),path in zip(download_jobs,executor.map(lambda j:fetch(j[2],j[3]),download_jobs)):
            files[game,team]=path
    rows=[]; audit=[]; omissions=[]
    for game in (1,2):
        eventpath=DATA/f'metrica_{game}_events.csv'
        if not eventpath.exists():
            fetch(f'https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{game}/Sample_Game_{game}_RawEventsData.csv',eventpath.name)
        events=list(csv.DictReader(eventpath.open())); tracks={team:tracking(files[game,team]) for team in ('Home','Away')}
        roles={}; keeper_ids={}
        for period in (1,2):
            for team,table in tracks.items():
                period_table=table[table['Period']==period]
                opening_frame=int(period_table.index[0])
                first_shot=min(int(e['Start Frame']) for e in events if e['Type']=='SHOT' and int(e['Period'])==period)
                # Use only opening tracking, ending before the first shot. Fix
                # identity in period 1; a transient half-time outlier cannot
                # relabel an outfield player as the period-2 goalkeeper.
                end=min(opening_frame+249,first_shot-1)
                start=period_table.loc[opening_frame:end].median(numeric_only=True)
                values={c[:-2]:float(start[c]) for c in table.columns if c.endswith('_x') and c!='ball_x' and pd.notna(start[c])}
                if period==1:
                    keeper_ids[team]=max(values,key=lambda p:abs(values[p]-.5))
                    if any(e['Type']=='SUBSTITUTION' and e['From']==keeper_ids[team] for e in events):
                        raise ValueError('Goalkeeper substitution requires explicit handling; do not infer replacement')
                keeper=keeper_ids[team]; gx=values[keeper]
                if not(gx<.2 or gx>.8): raise ValueError('Ambiguous initial goalkeeper/direction; no guess')
                roles[team,period]={'goalkeeper':keeper,'attack_right':gx<.5,'opening_median_x':gx,
                                    'opening_window_frames':[opening_frame,end]}
            if roles['Home',period]['attack_right']==roles['Away',period]['attack_right']: raise ValueError('Invalid team orientation')
        for index,event in enumerate(events):
            if event['Type']!='SHOT': continue
            sid=f'metrica_game_{game}_event_{index:04d}'; team=event['Team']; period=int(event['Period']); frame=int(event['Start Frame'])
            preceding=[e for e in events[max(0,index-2):index] if 'PENALTY' in e['Subtype'] and e['Team']==team]
            if 'PENALTY' in event['Subtype'] or any(abs(int(e['Start Frame'])-frame)<=25 for e in preceding):
                omissions.append({'shot_id':sid,'reason':'annotated_penalty'}); continue
            try: snapshots={name:table.loc[frame] for name,table in tracks.items()}
            except KeyError:
                omissions.append({'shot_id':sid,'reason':'missing_synchronized_frame'}); continue
            shooter=event['From']; shootercol=shooter+'_x'
            if shootercol not in snapshots[team] or pd.isna(snapshots[team][shootercol]):
                omissions.append({'shot_id':sid,'reason':'shooter_tracking_unavailable'}); continue
            opponent='Away' if team=='Home' else 'Home'
            keeper_col=keeper_ids[opponent]+'_x'
            if keeper_col not in snapshots[opponent] or pd.isna(snapshots[opponent][keeper_col]):
                omissions.append({'shot_id':sid,'reason':'opposing_goalkeeper_tracking_unavailable'}); continue
            right=roles[team,period]['attack_right']
            def transform(x,y): return (x*120,y*80) if right else ((1-x)*120,(1-y)*80)
            x,y=transform(float(event['Start X']),float(event['Start Y']))
            if not(60<=x<=120 and 0<=y<=80):
                omissions.append({'shot_id':sid,'reason':'shooter_outside_rendered_half'}); continue
            ff=[]
            for name,snapshot in snapshots.items():
                for column in tracks[name].columns:
                    if not column.endswith('_x') or column=='ball_x': continue
                    pid=column[:-2]
                    if name==team and pid==shooter: continue
                    px,py=snapshot[column],snapshot[pid+'_y']
                    if pd.isna(px) or pd.isna(py): continue
                    px,py=transform(float(px),float(py))
                    ff.append({'location':[px,py],'teammate':name==team,
                               'position':{'name':'Goalkeeper' if pid==roles[name,period]['goalkeeper'] else 'Outfield'}})
            row=geometry_row(f'metrica_game_{game}',sid,x,y,ff,int('GOAL' in event['Subtype']),int('HEAD' in event['Subtype']))
            row.update(source_start_frame=frame,header_source='Explicit HEAD token in provider subtype; tags never enter VLM input')
            rows.append(row)
        audit.append({'game':game,'roles':[{'team':team,'period':period,**value} for (team,period),value in roles.items()],
            'event_source_sha256':base.sha(eventpath),'tracking_hashes':{name:base.sha(files[game,name]) for name in tracks},
            'penalty_related_event_counts':sum('PENALTY' in e['Subtype'] for e in events),
            'shot_subtypes':sorted({e['Subtype'] for e in events if e['Type']=='SHOT'})})
    predict_frozen(rows); base.freeze(OUT/'metrica_v2/selection.json',rows)
    base.freeze(OUT/'metrica_v2/provenance.json',{'games':audit,'omissions':omissions,
        'coordinate_mapping':'Provider 0..1 positions mapped to normalized StatsBomb 120x80, rotated 180 degrees when team attacks left. Physical source pitch 105x68 m; displayed numbers are normalized units.',
        'role_inference':'Goalkeeper identity inferred from median opening-period-1 tracking (at most 250 frames, always before the first shot), then kept fixed. Direction in each half uses the same keeper in median opening tracking. Stop on goalkeeper substitution. Require x<.2 or >.8 and opposite directions. Roles are inferred, not provider-certified.',
        'frame_rule':'Event Start Frame only; no End Frame, End X/Y, goal outcome, subtype, event/player/team names or clock is rendered. All finite synchronized player positions except the shooter enter the diagram.',
        'limitations':'Two anonymized games only; no population-level confirmation or trustworthy match-bootstrap claim. Cross-provider match identity overlap cannot be excluded from anonymized source.'})
    render('metrica_v2',rows)
    amendment={'stage':'offline preflight, before any generation or model scores',
        'initial_failure':'Single opening-frame extreme position mislabeled game-1 period-2 Away Player22 as goalkeeper; three later frames lacked that player. Offline goalkeeper guard failed.',
        'correction':'Median opening-period identity and identity continuity across halves; stable keeper is Player25. Verify no goalkeeper substitution and available opposing keeper at each shot.',
        'initial_unrun_selection_sha256':base.sha(OUT/'metrica/selection.json') if (OUT/'metrica/selection.json').exists() else None,
        'verified_selection_sha256':base.sha(OUT/'metrica_v2/selection.json'),
        'data_plan_sha256':base.sha(OUT/'data_plan.json'),'revised_preparation_sha256':base.sha(__file__),
        'StatsBomb_cohorts':'unchanged; preparation code revised only for Metrica goalkeeper handling. Initial images and records preserved, never sent to a model.'}
    if (OUT/'data_amendment.json').exists():
        previous=base.read(OUT/'data_amendment.json')
        if previous['verified_selection_sha256']!=amendment['verified_selection_sha256']:
            raise ValueError('Verified Metrica cohort changed')
    else: base.freeze(OUT/'data_amendment.json',amendment)
    print(f'Metrica: {len(rows)} verified eligible shots from two games; initial failed-preflight images retained unused.',flush=True)


def render(name,rows):
    images=OUT/'images'/name; images.mkdir(parents=True,exist_ok=True)
    for row in rows:
        path=images/(row['shot_id']+'.png')
        if not path.exists(): original.render(pd.Series(row),path)


def main():
    plan=freeze_plan(); prepare_statsbomb(plan); prepare_metrica()
    print('All fixed data, original-renderer images and separate baselines prepared. No paid inference.',flush=True)


if __name__=='__main__': main()
