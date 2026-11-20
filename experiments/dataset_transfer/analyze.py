"""Complete-cohort scoring and provenance audit. Never calls a model."""
import ast,csv,inspect,json,math
from pathlib import Path
import run as r


def audit():
    p=r.base.read(r.OUT/'protocol.json'); ledger=r.base.read(r.OUT/'attempts.json',[])
    if len(ledger)!=p['planned_calls'] or any(a['state']!='completed' for a in ledger):
        raise ValueError('Full fixed three-dataset study required; do not score a partial cohort')
    checks={}
    for field,path in [('engine_sha256',Path(r.__file__)),('shared_engine_sha256',Path(r.base.__file__)),
        ('original_source_sha256',r.ROOT/'xg_vlm_pilot.py'),('preparation_sha256',Path(__file__).with_name('prepare.py')),
        ('data_plan_sha256',r.OUT/'data_plan.json'),('data_amendment_sha256',r.OUT/'data_amendment.json')]:
        checks[field]=p[field]==r.base.sha(path)
    checks['within_dollar_cap']=r.exposure(ledger)<=r.CAP
    checks['within_attempt_cap']=len(ledger)<=r.ATTEMPTS
    checks['all_calls_complete']=all(a['state']=='completed' for a in ledger)
    checks['no_generation_lock']=not (r.OUT/'generation.lock').exists()
    plan=r.base.read(r.OUT/'data_plan.json'); ids=set(); response_ids=set(); source_files={}
    for c in r.COHORTS:
        selection=r.base.read(r.OUT/c/'selection.json')
        checks[c+'_selection_frozen']=p['selection_hashes'][c]==r.base.sha(r.OUT/c/'selection.json')
        ref=r.OUT/c/('provenance.json' if c=='metrica_v2' else 'baseline.json')
        checks[c+'_references_frozen']=p['reference_hashes'][c]==r.base.sha(ref)
        shot_ids={x['shot_id'] for x in selection}
        checks[c+'_unique_shots']=len(shot_ids)==len(selection) and not(ids&shot_ids); ids.update(shot_ids)
        if c!='metrica_v2':
            spec=plan['cohorts'][c]; baseline=r.base.read(ref)
            checks[c+'_no_train_test_overlap']=not(set(spec['train_ids'])&set(spec['test_ids']))
            checks[c+'_outside_prior_matches']=not(set(spec['train_ids']+spec['test_ids'])&set(plan['prior_used_match_ids']))
            checks[c+'_correct_test_groups']=set(x['match_id'] for x in selection)<=set(spec['test_ids'])
            for mid,value in baseline['source_hashes'].items(): source_files[f'statsbomb/events_{mid}.json']=value
            for row in selection:
                for column,features,coeff,intercept in [
                    ('local_logistic',baseline['features'],baseline['coefficients'],baseline['intercept']),
                    ('local_no_header',baseline['image_features'],baseline['image_coefficients'],baseline['image_intercept'])]:
                    z=intercept[0]+sum(row[k]*v for k,v in zip(features,coeff[0]))
                    assert abs(1/(1+math.exp(-z))-row[column])<1e-12
        else:
            provenance=r.base.read(ref)
            for game in provenance['games']:
                source_files[f"metrica_{game['game']}_events.csv"]=game['event_source_sha256']
                for team,value in game['tracking_hashes'].items(): source_files[f"metrica_{game['game']}_{team}_tracking.csv"]=value
            checks['corrected_metrica_only']=all(a['job']['cohort']!='metrica' for a in ledger)
    for entry,job in zip(ledger,p['schedule']):
        assert entry['job']==job
        image=r.ROOT/job['image']; assert r.base.sha(image)==job['image_sha256']
        payload=r.base.body(r.ARM,p['prompt'],image.read_bytes())
        assert r.base.digest(payload)==entry['request_sha256']
        raw=r.base.read(r.OUT/entry['response_file'])
        assert r.base.validate(raw,r.ARM)==entry['answer'] and raw['usage']==entry['usage']
        assert entry['usage']['input_tokens']<=entry['input_bound'] and entry['usage']['output_tokens']<=r.base.MAX_OUTPUT
        assert raw['id'] not in response_ids; response_ids.add(raw['id'])
    checks['exact_raw_answers_usage_identity_requests_images']=True
    checks['source_files_unchanged']=all(r.base.sha(r.OUT/'data'/name)==value for name,value in source_files.items())
    checks['local_reference_predictions_reproduced']=True
    # A private exact credential check avoids false positives on encrypted output.
    credential=r.base.credential().encode()
    checks['no_exact_credential_in_results']=not any(credential in path.read_bytes() for path in r.OUT.rglob('*') if path.is_file())
    result={'passed':all(checks.values()),'checks':checks,'calls':len(ledger),
        'estimated_usd':sum(r.base.cost(a['usage'],a['model']) for a in ledger),
        'conservative_usd':r.exposure(ledger),'new_cap_usd':r.CAP,
        'input_tokens':sum(a['usage']['input_tokens'] for a in ledger),
        'output_tokens':sum(a['usage']['output_tokens'] for a in ledger),
        'reasoning_tokens':sum((a['usage'].get('output_tokens_details') or {}).get('reasoning_tokens',0) for a in ledger),
        'new_gemini_usd':0,'new_claude_usd':0}
    r.base.save(r.OUT/'audit.json',result)
    if not result['passed']: raise ValueError('Audit failed: '+','.join(k for k,v in checks.items() if not v))
    return p,ledger,result


def score():
    p,ledger,integrity=audit(); answers={(a['job']['cohort'],a['job']['shot_id']):a['answer'] for a in ledger}; results={}
    for cohort in r.COHORTS:
        rows=[]
        for source in r.base.read(r.OUT/cohort/'selection.json'):
            row={k:source[k] for k in ('shot_id','match_id','goal','header','distance','angle','defenders_in_cone','original_logistic','original_no_header','sb_xg')}
            for key in ('local_logistic','local_no_header','training_prior'):
                if key in source: row[key]=source[key]
            if cohort=='metrica_v2': row['training_prior']=287/3068
            row.update(vlm_xg=answers[cohort,source['shot_id']]['xg'],reason=answers[cohort,source['shot_id']]['reason'])
            rows.append(row)
        methods=['vlm_xg','original_logistic','original_no_header']
        if cohort!='metrica_v2': methods+=['local_logistic','local_no_header','sb_xg']
        methods+=['training_prior']
        scores={m:r.base.metrics([x['goal'] for x in rows],[x[m] for x in rows]) for m in methods}
        with (r.OUT/cohort/'predictions.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        summary={'shots':len(rows),'goals':sum(x['goal'] for x in rows),'matches':len({x['match_id'] for x in rows}),
            'scores':scores,'goal_rate':sum(x['goal'] for x in rows)/len(rows),
            'mean_overprediction':scores['vlm_xg']['mean_xg']-sum(x['goal'] for x in rows)/len(rows)}
        if cohort!='metrica_v2':
            summary['primary']=r.base.comparison(rows,'vlm_xg','original_logistic',.975)
            summary['secondary_local']=r.base.comparison(rows,'vlm_xg','local_logistic',.95)
            summary['secondary_no_header']=r.base.comparison(rows,'vlm_xg','original_no_header',.95)
        else:
            summary['descriptive_brier_difference']=scores['vlm_xg']['brier']-scores['original_logistic']['brier']
            summary['inference_limit']='Two games; no formal population-level interval or superiority claim.'
        r.base.save(r.OUT/cohort/'summary.json',summary); results[cohort]=summary
    r.base.save(r.OUT/'summary.json',{'cohorts':results,'audit':integrity,'no_probability_changes':True})
    print(json.dumps({'cohorts':results,'estimated_usd':integrity['estimated_usd']},indent=2),flush=True)


if __name__=='__main__': score()
