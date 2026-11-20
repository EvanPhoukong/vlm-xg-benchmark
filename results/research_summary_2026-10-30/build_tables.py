"""Compile saved research metrics offline. No API calls or changed predictions."""
from pathlib import Path
import csv, hashlib, json, math
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/mnt/c/Users/ephou/Downloads/xG/output/research-loop')
groups, sources = {}, {}
def read(path):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8-sig')
def metrics(v):
    return {k:float(v[k]) for k in ('auc','brier','mean_xg')} | {'log_loss':float(v.get('log_loss',v.get('logloss')))}
def from_csv(name,path,allowed=None):
    rows=list(csv.DictReader(read(path).splitlines()))
    key=next(k for k in ('method','prediction_column') if k in rows[0])
    groups[name]={r[key]:metrics(r) for r in rows if allowed is None or r[key] in allowed}
def from_json(name,path):
    data=json.loads(read(path))
    groups[name]={k:metrics(v) for k,v in data['scores'].items()}
def score(rows,col):
    y=[int(r['goal']) for r in rows]; p=[float(r[col]) for r in rows]
    pos=[v for t,v in zip(y,p) if t]; neg=[v for t,v in zip(y,p) if not t]
    return {'auc':sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg)),
        'brier':sum((a-b)**2 for a,b in zip(y,p))/len(y),
        'log_loss':-sum(a*math.log(max(1e-15,b))+(1-a)*math.log(max(1e-15,1-b)) for a,b in zip(y,p))/len(y),
        'mean_xg':sum(p)/len(p)}
originals={}
for label,path in [('Gemma 4 (26B)','shots_gemma_4_26b.csv'),('Claude Sonnet 5.5','shots_sonnet_5_5.csv'),
    ('Claude Opus 5.5','shots_opus5_5.csv'),('Gemini 3.8 Flash','shots_gemini_3_8_flash.csv'),
    ('GPT-6 Luna','results/gpt_6_luna/shots.csv')]:
    rows=list(csv.DictReader(read(path).splitlines()))
    assert len(rows)==len({r['shot_id'] for r in rows})
    originals[label]={r['shot_id']:r for r in rows}
    groups['Original saved cohort: '+label]={'Direct VLM':score(rows,'vlm_xg'),
        'Paired saved logistic':score(rows,'baseline_xg'),'StatsBomb':score(rows,'sb_xg')}
shared=sorted(set.intersection(*(set(v) for v in originals.values())))
assert len(shared)==298
ref=originals['Gemma 4 (26B)']
for records in originals.values():
    for sid in shared:
        for col in ('goal','match_id','x','y','sb_xg'):
            assert records[sid][col]==ref[sid][col],(sid,col)
for label in ('Claude Sonnet 5.5','Claude Opus 5.5','Gemini 3.8 Flash'):
    assert all(originals[label][s]['baseline_xg']==ref[s]['baseline_xg'] for s in shared)
groups['Initial benchmark: common 298 shots']={label:score([r[s] for s in shared],'vlm_xg') for label,r in originals.items()}
for label,col in [('Original logistic','baseline_xg'),('StatsBomb','sb_xg')]:
    groups['Initial benchmark: common 298 shots'][label]=score([ref[s] for s in shared],col)
alignment={'shots':len(shared),'goals':sum(int(ref[s]['goal']) for s in shared),
    'matches':len({ref[s]['match_id'] for s in shared}),'shot_ids':shared,
    'saved_counts':{k:len(v) for k,v in originals.items()},
    'excluded_from_common_table':{k:sorted(set(v)-set(shared)) for k,v in originals.items()},
    'checks':['Unique shot IDs','Identical goals, match IDs, coordinates and StatsBomb probabilities',
    'Identical original logistic across the four original provider files'],
    'note':'Common-shot alignment for this summary only. Two Gemini records are absent; their absence is not explained by this audit. Luna has a different saved logistic fit, retained separately. No saved prediction changed.'}
for name,path in [
    ('Reasoning pilot: 12 reused World Cup shots','results/reasoning_effort_v1/summary.json'),
    ('Reasoning: 50 AFCON development shots','results/direct_scope_suite/reasoning/summary.json'),
    ('Marker: 40 reused World Cup shots','results/direct_scope_suite/rendering/summary.json'),
    ('Independent AFCON: 150 shots','results/direct_scope_suite/confirmation/summary.json'),
    ('Model and prompt screen: 50 reused AFCON shots','results/direct_model_search_v1/screen/summary.json'),
    ('Independent La Liga 2015/16: 200 shots','results/direct_model_search_v1/confirmation/summary.json'),
    ('Independent Premier League: 500 shots','results/dataset_transfer_v1/premier_league/summary.json'),
    ("Independent Women's World Cup: 500 shots",'results/dataset_transfer_v1/womens_world_cup/summary.json'),
    ('Metrica pilot: 46 shots','results/dataset_transfer_v1/metrica_v2/summary.json'),
    ('Independent La Liga 2016/17: 150 shots','results/core_fixes_v1/main/summary.json'),
    ('Past tracking: 43 reused Metrica shots','results/core_fixes_v1/motion/summary.json')]: from_json(name,path)
for name,path in [
    ('First hybrid: 300 World Cup shots','results/hybrid_gpt_6_luna_300/metrics.csv'),
    ('Spatial development: 3068 World Cup shots',ARCHIVE/'development_scores.csv'),
    ('Known coordinates: 1304 Euro 2024 shots',ARCHIVE/'external_reference_scores.csv'),
    ('Known coordinates: 1234 Euro 2020 shots',ARCHIVE/'layered_vlm_external/reference_scores_all.csv'),
    ('Initial richer hybrid: 200 Euro 2024 shots',ARCHIVE/'rich_vlm_external/scores.csv'),
    ('Improvement screen: 200 reused Euro 2024 shots','results/improvement_suite/all_screening_scores.csv'),
    ('Direct position-first outputs: 200 reused Euro shots','results/improvement_suite/vision/direct_scores.csv'),
    ('Pre-shot event context: 200 reused Euro shots','results/improvement_suite/context/scores.csv'),
    ('Independent Bundesliga: 200 shots','results/improvement_suite/confirmation_bundesliga/scores.csv'),
    ('Secondary full Bundesliga pool: 908 shots, includes 200-shot subset','results/improvement_suite/confirmation_bundesliga/full_pool_reference/scores.csv'),
    ('Known positions and context: full Euro 2024','results/improvement_suite/context/euro2024_known_scores.csv'),
    ('Known positions and context: full Euro 2020','results/improvement_suite/context/euro2020_known_scores.csv'),
    ('Specialists: 200 reused Bundesliga shots','results/task_division_v1/scores.csv')]: from_csv(name,path)
for label,path in [('Body part: image-only control, 50 World Cup shots','results/calibration_experiment/image_only_50/metrics.csv'),
    ('Body part supplied: same 50 World Cup shots','results/calibration_experiment/body_part_50/metrics.csv')]:
    from_csv(label,path,{'vlm_xg','baseline_xg','sb_xg'})
assert all(math.isfinite(v) for g in groups.values() for row in g.values() for v in row.values())
def table(rows,label='Method'):
    lines=[f'| {label} | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |','|---|---:|---:|---:|---:|']
    for label,v in rows:
        lines.append(f"| {label} | {v['auc']:.3f} | {v['brier']:.4f} | {v['log_loss']:.4f} | {v['mean_xg']:.1%} |")
    return '\n'.join(lines)
content=['# Complete non-recalibration result tables','',
    'Compiled offline on 30 October 2026. No API calls, retraining or probability correction. Compare methods within the same cohort. Cohorts overlap and must not be pooled as independent replications. Direct VLM outputs, supervised hybrids, known-coordinate statistical models and pixel controls answer different questions. Incomplete trials have no final score. The common-shot initial table is an aligned view of saved outputs; original-size scores are also retained.','']
for name,rows in groups.items(): content+=['## '+name,'',table(rows.items()),'']
(OUT/'FULL_RESULTS.md').write_text('\n'.join(content),encoding='utf-8')
for name,value in [('metrics.json',groups),('sources.json',sources),('initial_cohort_alignment.json',alignment)]:
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'groups':len(groups),'metric_rows':sum(map(len,groups.values())),'sources':len(sources),
    'common_shots':len(shared),'common_goals':alignment['goals'],'common_matches':alignment['matches']},indent=2))
print(table(groups['Initial benchmark: common 298 shots'].items()))

