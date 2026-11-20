"""Final offline provenance, budget and leakage audit. No API calls."""
import json,math,re
from pathlib import Path
import common as c
from analyze import completed

def logistic(row,columns,coef,intercept):
    z=intercept[0]+sum(float(row[k])*v for k,v in zip(columns,coef[0])); return 1/(1+math.exp(-z))
def main():
    stages=['reasoning','rendering','confirmation','invariance']; ledger=c.read(c.OUT/'attempts.json')
    assert not (c.OUT/'generation.lock').exists()
    for stage in stages: completed(stage)
    split=c.read(c.OUT/'data/match_split.json'); development=set(split['development_match_ids']); confirmation=set(split['confirmation_match_ids']); old=set(split['used_match_ids'])
    assert not development&confirmation and not development&old and not confirmation&old
    for name in ['reasoning','confirmation']:
        rows=c.read(c.OUT/name/'selection.json'); allowed=development if name=='reasoning' else confirmation
        for r in rows:
            assert r['match_id'] in allowed
            assert abs(logistic(r,split['logistic_features'],split['logistic_coefficients'],split['logistic_intercept'])-r['baseline_xg'])<1e-10
            assert abs(logistic(r,split['matched_image_features'],split['matched_coefficients'],split['matched_intercept'])-r['baseline_image_xg'])<1e-10
    for name,digest in split['source_hashes'].items(): assert c.sha(c.OUT/'data'/name)==digest
    for name,digest in c.read(c.OUT/'analysis_pre_results_hashes.json').items(): assert c.sha(c.ROOT/name)==digest
    for a in ledger:
        assert a['state']=='completed' and a['usage']['input_tokens']<=a['input_bound'] and a['usage']['output_tokens']<=c.MAX_OUTPUT
        assert a['model']=='gpt-6-luna' or a['model'].startswith('gpt-6-luna-')
        protocol=c.read(c.OUT/a['stage']/'protocol.json')
        assert protocol['prompt']==c.raw_prompt()
        assert c.sha(c.ROOT/'experiments/direct_scope_suite/common.py')==protocol['engine_sha256']
        assert c.sha(c.ROOT/'experiments/direct_scope_suite/run_stage.py')==protocol['runner_sha256']
    # Old source files/images and the earlier pilot's frozen protocol must still reproduce.
    c.prior.prepare()
    candidate=c.read(c.OUT/'candidate.json'); confirmprotocol=c.read(c.OUT/'confirmation/protocol.json')
    assert candidate==confirmprotocol['metadata']['candidate'] and c.sha(c.OUT/'candidate.json')==confirmprotocol['metadata']['candidate_sha256']
    authorized_key=c.prior.credential()
    for p in c.OUT.rglob('*.json'):
        assert authorized_key not in p.read_text(encoding='utf-8'),'Authorized credential exposure in '+str(p)
    u=c.usage(ledger); assert u['conservative_usd']+c.PRIOR_COST<c.GLOBAL_CAP
    result={'status':'passed','generation_calls':len(ledger),'returned_models':sorted({a['model'] for a in ledger}),
            'match_disjoint':True,'recorded_baseline_predictions_reproduced':True,'raw_responses_and_returned_efforts_verified':True,
            'protected_inputs_unchanged':True,'development_choice_frozen_before_confirmation':True,'secrets_scan_passed':True,'secret_scan_method':'Exact authorized-credential check; opaque encrypted response content is not mistaken for a plaintext key',
            'usage':u,'estimated_usd_including_prior_pilot':u['estimated_usd']+c.PRIOR_COST,
            'conservative_usd_including_prior_pilot':u['conservative_usd']+c.PRIOR_COST,
            'response_sha256':{a['response_file']:c.sha(c.OUT/a['response_file']) for a in ledger}}
    c.save(c.OUT/'final_integrity.json',result)
    c.save(c.OUT/'status.json',{'status':'complete','stages':stages,'generation_calls':len(ledger),'usage':u,'estimated_total_including_prior_pilot_usd':u['estimated_usd']+c.PRIOR_COST,'no_active_generation':True})
    print(json.dumps({k:v for k,v in result.items() if k!='response_sha256'},indent=2))
if __name__=='__main__': main()

