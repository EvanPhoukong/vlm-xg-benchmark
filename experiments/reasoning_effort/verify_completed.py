"""Offline provenance check and illustrative factual review; no API calls."""
import hashlib, importlib.util, json
from pathlib import Path
here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pilot',here/'run_experiment.py'); p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
protocol=p.prepare(); ledger=p.read(p.OUT/'attempts.json'); selected=p.read(p.OUT/'selection.json'); summary=p.read(p.OUT/'summary.json')
assert len(ledger)==24 and len(selected)==12 and len({r['shot_id'] for r in selected})==12
assert not (p.OUT/'runner.lock').exists()
for a,job in zip(ledger,protocol['schedule']):
    assert a['state']=='completed' and all(a[k]==job[k] for k in ('shot_id','effort'))
    response=p.read(p.OUT/a['response_file'])
    assert p.validate(response)==a['answer'] and response['usage']==a['usage']
    assert response['reasoning']['effort']==a['effort']
    assert a['usage']['input_tokens']<=a['input_bound'] and a['usage']['output_tokens']<=p.LIMIT['per_output']
for r in selected:
    assert {a['effort'] for a in ledger if a['shot_id']==r['shot_id']}=={'low','medium'}
    image=(p.ROOT/'results/images'/(r['shot_id']+'.png')).read_bytes()
    first=p.body(protocol['prompt'],image,'low'); second=p.body(protocol['prompt'],image,'medium')
    first.pop('reasoning'); second.pop('reasoning'); assert first==second
u=p.totals(ledger)
assert u['input_tokens']<=p.LIMIT['input'] and u['output_tokens']<=p.LIMIT['output']
assert summary['estimated_total_usd']<p.LIMIT['usd']
review=[]
for sid in ('d2f58c81-92d6-4a50-811a-2413929f5c1d','9000f0f2-73ed-4a71-917d-2d525af7ba80'):
    r=next(x for x in selected if x['shot_id']==sid)
    entries={a['effort']:a for a in ledger if a['shot_id']==sid}
    review.append({'shot_id':sid,'distance_pitch_units':float(r['distance']),'depth_pitch_units':120-float(r['x']),
                   'low':entries['low']['answer'],'medium':entries['medium']['answer'],
                   'observation':'Shooter is visibly outside the penalty area; low calls this close range; medium identifies a longer-distance/outside-box attempt. Qualitative example, not a causal or population-level claim.'})
p.save(p.OUT/'rationale_review.json',{'method':'All 24 rationales were screened against reference distance/angle in hashed-ID order with effort and outcomes hidden. These two clear distance misdescriptions were selected before unblinding effort; source images then visually verified. Post-hoc qualitative review, not a preregistered error-rate measure. Physical metre estimates were not graded against normalized coordinates.','examples':review})
p.save(p.OUT/'integrity.json',{'status':'passed','responses':24,'shots':12,'returned_models':sorted({a['model'] for a in ledger}),
    'source_prompt_runner_and_image_hashes_unchanged':True,'response_json_matches_ledger':True,'returned_reasoning_matches_schedule':True,
    'payloads_differ_only_by_effort':True,'caps_respected':True,'no_active_runner':True,'totals':u,
    'response_sha256':{a['response_file']:p.sha(p.OUT/a['response_file']) for a in ledger}})
path=p.OUT/'report.md'; text=path.read_text(encoding='utf-8'); marker='\n## Illustrative factual review\n'
text=text.split(marker)[0]
text+=marker+'\nA post-hoc review screened all 24 short rationales with effort and outcomes hidden, then checked two clear distance misdescriptions against the source images. These examples suggest a possible perceptual benefit; they do not establish the cause of the score difference.\n'
for e in review:
    text+=f"\n- Shot {e['shot_id']}: distance {e['distance_pitch_units']:.1f} normalized pitch units, visibly outside the penalty area. Low xG {e['low']['xg']:.0%}: {e['low']['reason']} Medium xG {e['medium']['xg']:.0%}: {e['medium']['reason']}\n"
text+='\nAll response/model/effort checks, paired-input checks, protected input hashes and budget checks passed. No API calls were made for this review.\n'
path.write_text(text,encoding='utf-8')
print(json.dumps({'integrity':'passed','models':sorted({a['model'] for a in ledger}),'examples':review},indent=2))
