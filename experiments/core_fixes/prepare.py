"""Freeze training-only visual examples, new match groups and past-only motion."""
import concurrent.futures,csv,importlib.util,json,random,sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import common as c
sys.path.insert(0,str(c.ROOT)); import xg_vlm_pilot as original
spec=importlib.util.spec_from_file_location('transfer_preparation',c.ROOT/'experiments/dataset_transfer/prepare.py')
transfer=importlib.util.module_from_spec(spec); spec.loader.exec_module(transfer)


def public(relative):
    path=c.OUT/'data'/relative.replace('/','_')
    if not path.exists():
        from urllib.request import urlopen
        with urlopen('https://raw.githubusercontent.com/hudl/open-data/master/data/'+relative,timeout=90) as response: content=response.read()
        json.loads(content); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(content)
    return c.read(path)


def examples():
    folder=c.OUT/'training'; folder.mkdir(parents=True,exist_ok=True)
    root=Path('/mnt/c/Users/ephou/Downloads/xG/output/research-loop/data')
    prior=c.read(c.ROOT/'results/direct_scope_suite/data/match_split.json'); rows=[]; hashes={}
    for name in ('wc2018','wc2022'):
        file=root/(name+'.jsonl'); assert c.sha(file)==prior['training_source_hashes'][name]
        hashes[name]=c.sha(file); rows.extend(json.loads(line) for line in file.read_text().splitlines() if line.strip())
    chosen=random.Random(2026103201).sample(sorted(rows,key=lambda x:x['shot_id']),48)
    labels=[r['goal'] for r in chosen]; shuffled=labels.copy(); random.Random(2026103202).shuffle(shuffled)
    result=[]
    for row,label in zip(chosen,shuffled):
        image=folder/'images'/(row['shot_id']+'.png'); image.parent.mkdir(exist_ok=True)
        if not image.exists(): original.render(pd.Series(row),image)
        result.append({'shot_id':row['shot_id'],'match_id':int(row['match_id']),'goal':int(row['goal']),
            'shuffled_goal':int(label),'image':str(image.relative_to(c.ROOT)),'image_sha256':c.sha(image)})
    assert sum(labels)==sum(shuffled) and labels!=shuffled
    c.freeze(folder/'examples.json',result)
    c.freeze(folder/'provenance.json',{'source_hashes':hashes,'training_shots':len(rows),'examples':48,
        'selection':'Fixed natural random sample independent of outcomes/errors. No probability labels, geometric answers, baseline predictions, query retrieval or selected goal-rate anchors.',
        'negative_control':'Fixed permutation of binary outcome labels on identical images preserves the exact positive count.',
        'difference_from_prior_work':'Earlier four-example experiment supplied geometry only and explicitly no outcomes; this new condition supplies observed historical binary outcomes to the VLM.'})


def new_cohort():
    previous=c.read(c.ROOT/'results/dataset_transfer_v1/data_plan.json')
    used=set(previous['prior_used_match_ids'])
    for spec in previous['cohorts'].values(): used.update(spec['train_ids']+spec['test_ids'])
    matches=public('matches/11/2.json'); eligible=sorted(m['match_id'] for m in matches if int(m['match_id']) not in used)
    if len(eligible)<30: raise ValueError('Too few genuinely unused La Liga 2016/17 match groups')
    ids=sorted(random.Random(2026103203).sample(eligible,min(40,len(eligible))))
    c.freeze(c.OUT/'data/match_split.json',{'competition_id':11,'season_id':2,'season':'2016/2017',
        'previous_match_ids':sorted(used),'selected_match_ids':ids,'match_manifest_sha256':c.sha(c.OUT/'data/matches_11_2.json'),
        'rule':'Select match IDs before event downloads. Natural 200-shot ordered pool frozen before pilot; resource-only rule picks first 100,150 or200 before any accuracy score.'})
    def get(mid):
        public(f'events/{mid}.json'); return transfer.sb_rows(c.OUT/'data'/f'events_{mid}.json',mid)[0]
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(get,ids): rows.extend(result)
    if len(rows)<200: raise ValueError('Insufficient unused eligible shots')
    selected=random.Random(2026103204).sample(sorted(rows,key=lambda r:r['shot_id']),200)
    transfer.predict_frozen(selected)
    for row in selected: row['training_prior']=287/3068
    c.freeze(c.OUT/'heldout/pool.json',selected)
    folder=c.OUT/'images/heldout'; folder.mkdir(parents=True,exist_ok=True)
    for row in selected:
        image=folder/(row['shot_id']+'.png')
        if not image.exists(): original.render(pd.Series(row),image)
    c.freeze(c.OUT/'data/provenance.json',{'eligible_shots':len(rows),'reserved_pool':200,
        'source_hashes':{str(mid):c.sha(c.OUT/'data'/f'events_{mid}.json') for mid in ids},
        'example_match_overlap':sorted(set(r['match_id'] for r in c.read(c.OUT/'training/examples.json'))&set(ids))})
    print(f'New held-out pool: 200 shots from {len(ids)} unused match groups.',flush=True)


def motion():
    src=c.ROOT/'results/dataset_transfer_v1'; source=c.read(src/'metrica_v2/selection.json')
    provenance=c.read(src/'metrica_v2/provenance.json'); tracks={}; events={}; roles={}
    for record in provenance['games']:
        game=record['game']; events[game]=list(csv.DictReader((src/'data'/f'metrica_{game}_events.csv').open()))
        for team in ('Home','Away'): tracks[game,team]=transfer.tracking(src/'data'/f'metrica_{game}_{team}_tracking.csv')
        for value in record['roles']: roles[game,value['team'],value['period']]=value
    kept=[]; omitted=[]; jobs=[]; rng=random.Random(2026103207)
    for row in source:
        game=int(row['shot_id'].split('_')[2]); idx=int(row['shot_id'].split('_')[-1]); event=events[game][idx]
        team=event['Team']; period=int(event['Period']); frame=row['source_start_frame']; shooter=event['From']
        right=roles[game,team,period]['attack_right']; images=[]; valid=True; records=[]
        for offset in (-25,-12,0):
            try: snapshots={t:tracks[game,t].loc[frame+offset] for t in ('Home','Away')}
            except KeyError: valid=False; break
            if any(int(x['Period'])!=period for x in snapshots.values()): valid=False; break
            def transform(x,y): return (x*120,y*80) if right else ((1-x)*120,(1-y)*80)
            shotpos=snapshots[team]
            if pd.isna(shotpos.get(shooter+'_x')) or pd.isna(shotpos.get('ball_x')) or pd.isna(shotpos.get('ball_y')):
                valid=False; break
            if offset==0: x,y=row['x'],row['y']
            else: x,y=transform(float(shotpos[shooter+'_x']),float(shotpos[shooter+'_y']))
            ff=[]
            for t,snapshot in snapshots.items():
                for col in tracks[game,t].columns:
                    if not col.endswith('_x') or col=='ball_x': continue
                    pid=col[:-2]
                    if t==team and pid==shooter: continue
                    if pd.isna(snapshot[col]) or pd.isna(snapshot[pid+'_y']): continue
                    px,py=transform(float(snapshot[col]),float(snapshot[pid+'_y']))
                    ff.append({'location':[px,py],'teammate':t==team,'position':{'name':'Goalkeeper' if pid==roles[game,t,period]['goalkeeper'] else 'Outfield'}})
            bx,by=transform(float(shotpos['ball_x']),float(shotpos['ball_y']))
            # Match original canvas/markers; only a real tracked ball is added.
            from mplsoccer import VerticalPitch
            pitch=VerticalPitch(half=True,pitch_type='statsbomb',line_color='black'); fig,ax=pitch.draw(figsize=(5,5))
            for player in ff:
                px,py=player['location']; keeper=player['position']['name']=='Goalkeeper'
                pitch.scatter(px,py,ax=ax,s=120 if player['teammate'] or not keeper else 170,
                    c='royalblue' if player['teammate'] else 'orange' if keeper else 'red',
                    marker='s' if keeper and not player['teammate'] else 'o',edgecolors='black')
            pitch.scatter(x,y,ax=ax,s=320,c='gold',marker='*',edgecolors='black')
            pitch.scatter(bx,by,ax=ax,s=65,c='magenta',marker='D',edgecolors='black',zorder=10)
            path=c.OUT/'images/motion'/row['shot_id']/(f'{offset}.png'); path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists(): fig.savefig(path,dpi=110,bbox_inches='tight')
            plt.close(fig); images.append(path); records.append({'source_frame':frame+offset,'offset_frames':offset,'ball':[bx,by]})
        if not valid:
            omitted.append({'shot_id':row['shot_id'],'reason':'Past/current actor or ball tracking unavailable'}); continue
        kept.append(row); records[-1]['contact_frame']=frame
        c.freeze(c.OUT/'motion/frames'/f"{row['shot_id']}.json",records)
        arms=['original','single_ball','history']; rng.shuffle(arms)
        for arm in arms:
            paths=[src/'images/metrica_v2'/(row['shot_id']+'.png')] if arm=='original' else [images[-1]] if arm=='single_ball' else images
            jobs.append(c.job(row['shot_id'],'luna',paths,condition=arm,motion=arm!='original'))
    if len(kept)<30: raise ValueError('Too few past-only motion cases')
    c.freeze(c.OUT/'motion/selection.json',kept); c.freeze(c.OUT/'motion/omissions.json',omitted)
    # Repeat unchanged single/history inputs on first 12 fixed source cases.
    for row in kept[:12]:
        for arm in ('single_ball','history'):
            source_job=next(j for j in jobs if j['shot_id']==row['shot_id'] and j['condition']==arm)
            repeated={**source_job,'condition':arm+'_repeat'}; jobs.append(repeated)
    c.freeze_stage('motion',jobs,{'scope':'Exploratory reused Metrica games, actual past tracking only; no raw footage or pose. Primary descriptive contrast history versus matched single_ball; original is separate reference.',
        'frame_offsets':[-25,-12,0],'fps':25,'post_contact_frames':0,'omitted':len(omitted),'source_provenance_sha256':c.sha(src/'metrica_v2/provenance.json')})


def main():
    c.OUT.mkdir(parents=True,exist_ok=True); examples(); new_cohort()
    protocol={'authorization':'User requested changes that address core errors; only available open data. Keep direct raw xG, no recalibration, $10 provider limits.',
        'cap_usd':c.CAP,'max_attempts':c.MAX_ATTEMPTS,'max_output_tokens':c.MAX_OUTPUT,'workers':c.WORKERS,'retries':0,
        'rates':c.RATES,'arms':c.ARMS,'original_prompt':c.old.original_prompt(),'example_intro':c.EXAMPLE_INTRO,'motion_prompt':c.MOTION_PROMPT,
        'scope':'Astra/Luna zero-shot preserved. Separate 48-example outcome-labelled in-context condition; not represented as zero-shot or weight fine tuning. Shuffled-label negative control preserves example goal count. All VLM probabilities remain raw.',
        'novelty':'Astra not previously evaluated. Earlier examples were four geometry-only demonstrations with no scoring outcomes. New Metrica history uses actual source past frames, not repeated readings/zoom/synthetic motion.',
        'stages':'12 reused AFCON pilot shots x4 conditions for usage/validity only, never accuracy selection. Resource-only fixed rule chooses first100/150/200 from frozen new200-shot pool. Then main, exact repeats, controlled synthetic pairs, and Metrica motion. No method winner selection or test-driven tuning.',
        'resource_rule':'After full pilot, choose largest N from [200,150,100] whose projected conservative remaining cost plus pilot exposure, one maximum Astra response reservation and $0.05 buffer fits $4. Projection=1.25 times pilot mean conservative per-arm cost for N+36 future calls per arm, plus 180 Luna single-call equivalents for motion. If no N fits, stop without accuracy claim.',
        'primary':'Five new-cohort Brier comparisons: Astra-Luna, Astra-original4logistic, examples-Luna, examples-original4logistic, examples-shuffled. 99% whole-match intervals for five tests. No equivalence claim from crossing zero.',
        'core_criteria':'Brier alone does not establish a core fix. Also report AUC, 6-18-unit errors, training-constant reference, correct-vs-shuffled examples, raw exact-repeat variability and controlled directional consistency. No averaging or enforcing probabilities.',
        'no_query_leakage':'Example images/outcomes come only from WC2018/22 training groups. Query labels, coordinates, headers, reference predictions and goal rates never enter requests. Main new groups exclude all488 prior StatsBomb groups.',
        'footage':'User has no authorized local footage; xGHub gate respected. Motion uses available Metrica tracking and cannot replace a real-frame/pose test.',
        'known_prior_conservative_usd':1.4552639,'account_caveat':'Partial older usage only; no full bill or verified balance. New hard ceiling $4 leaves room under $10; stop on exhausted credit/uncertain charge.',
        'source_sha256':c.sha(c.ROOT/'xg_vlm_pilot.py'),'preparation_sha256':c.sha(__file__),
        'common_sha256':c.sha(Path(__file__).with_name('common.py')),'engine_sha256':c.sha(Path(__file__).with_name('run.py')),
        'examples_sha256':c.sha(c.OUT/'training/examples.json'),'pool_sha256':c.sha(c.OUT/'heldout/pool.json')}
    c.freeze(c.OUT/'protocol.json',protocol)
    prior=c.read(c.ROOT/'results/direct_scope_suite/reasoning/selection.json')
    pilot=random.Random(2026103205).sample(prior,12); c.freeze(c.OUT/'pilot/selection.json',pilot)
    jobs=[]; rng=random.Random(2026103206)
    for row in pilot:
        arms=list(c.ARMS); rng.shuffle(arms)
        image=c.ROOT/'results/direct_scope_suite/images/reasoning'/(row['shot_id']+'.png')
        jobs.extend(c.job(row['shot_id'],arm,[image]) for arm in arms)
    c.freeze_stage('pilot',jobs,{'use':'Usage and validity only; no accuracy scoring/selection'})
    motion()
    # Synthetic behavior cases are the previously fixed templates, not new goals.
    probes=c.read(c.ROOT/'results/visual_counterfactuals_v1/scenes.json'); jobs=[]
    for case in probes:
        for arm in c.ARMS:
            conditions=['base','favorable']; rng.shuffle(conditions)
            for condition in conditions:
                image=c.ROOT/'results/visual_counterfactuals_v1/images'/(case['id']+'_'+condition+'.png')
                jobs.append(c.job(case['id'],arm,[image],condition=condition,factor=case['factor']))
    c.freeze_stage('probes',jobs,{'cases_sha256':c.sha(c.ROOT/'results/visual_counterfactuals_v1/scenes.json'),'qualitative_only':True})
    c.freeze_stage('repeats',[{**j,'condition':'repeat'} for j in c.read(c.OUT/'pilot/protocol.json')['schedule']],{'reference':'Corresponding original pilot reading; no averaging'})
    print('Frozen core-fix study: Astra, 48 outcome-labelled examples plus shuffled-label control, past-only Metrica motion. No generation.',flush=True)


if __name__=='__main__': main()
