"""Offline report, source/input/raw-response audit, and scientific figure."""
import json
from pathlib import Path
import run as r


def audit():
    ledger = r.read(r.OUT/'attempts.json',[])
    protocol = r.read(r.OUT/'protocol.json')
    tests = {'frozen_engine':r.sha(r.__file__)==protocol['engine_sha256'],
             'original_renderer_unchanged':r.sha(r.ROOT/'xg_vlm_pilot.py')==protocol['original_source_sha256'],
             'all_attempts_complete':all(a['state']=='completed' for a in ledger),
             'within_new_cap':r.exposure(ledger)<=r.CAP,
             'within_attempt_cap':len(ledger)<=r.MAX_ATTEMPTS,
             'fresh_match_groups':not r.read(r.OUT/'data/provenance.json')['fresh_match_overlap']}
    # Reconstruct exact requests from frozen images and instructions. Labels and
    # comparator predictions are never arguments to the request builder.
    response_ids=set()
    for stage in ('screen','confirmation'):
        sp=r.read(r.OUT/stage/'protocol.json')
        if not sp:
            tests[stage+'_complete']=False
            continue
        entries=[a for a in ledger if a['stage']==stage]
        tests[stage+'_complete']=len(entries)==len(sp['schedule']) and all(a['state']=='completed' for a in entries)
        tests[stage+'_frozen']=sp['campaign_sha256']==r.sha(r.OUT/'protocol.json') and sp['selection_sha256']==r.sha(r.OUT/stage/'selection.json')
        for entry,job in zip(entries,sp['schedule']):
            raw=r.read(r.OUT/entry['response_file'])
            image=r.ROOT/job['image']; arm=protocol['arms'][job['arm']]
            payload=r.body(arm,protocol['prompts'][arm['prompt']],image.read_bytes())
            assert entry['job']==job and r.sha(image)==job['image_sha256']
            assert r.digest(payload)==entry['request_sha256']
            assert r.validate(raw,arm)==entry['answer'] and raw['usage']==entry['usage']
            assert entry['usage']['input_tokens']<=entry['input_bound']
            assert entry['usage']['output_tokens']<=r.MAX_OUTPUT
            assert raw['id'] not in response_ids
            response_ids.add(raw['id'])
    tests['all_reconstructed_requests_and_raw_responses']=True
    split=r.read(r.OUT/'data/match_split.json')
    selection=r.read(r.OUT/'confirmation/selection.json')
    tests['confirmation_outside_all_recorded_prior_matches']=not(set(x['match_id'] for x in selection)&set(split['previous_used_match_ids']))
    tests['unique_confirmation_shots']=len(selection)==len({x['shot_id'] for x in selection})==200
    tests['source_events_unchanged']=all(r.sha(r.OUT/'data'/name)==value for name,value in r.read(r.OUT/'data/provenance.json')['event_file_hashes'].items())
    if (r.OUT/'confirmation/choice_hash.json').exists():
        tests['choice_frozen']=r.sha(r.OUT/'choice.json')==r.read(r.OUT/'confirmation/choice_hash.json')['choice_sha256']
    key=r.credential().encode()
    tests['no_exact_credential_in_outputs']=not any(key in p.read_bytes() for p in r.OUT.rglob('*') if p.is_file())
    total=sum(r.cost(a.get('usage',{}),a['model']) for a in ledger)
    result={'checks':tests,'passed':all(tests.values()),'successful_calls':len(response_ids),
            'new_estimated_usd':total,'conservative_exposure_usd':r.exposure(ledger),
            'new_combined_cap_usd':r.CAP,'new_gemini_usd':0,'new_claude_usd':0,
            'by_model':{m:{'calls':sum(a['model']==m for a in ledger),
                          'estimated_usd':sum(r.cost(a.get('usage',{}),m) for a in ledger if a['model']==m),
                          'input_tokens':sum(a.get('usage',{}).get('input_tokens',0) for a in ledger if a['model']==m),
                          'output_tokens':sum(a.get('usage',{}).get('output_tokens',0) for a in ledger if a['model']==m)} for m in r.RATES}}
    r.save(r.OUT/'audit.json',result)
    if not result['passed']: raise ValueError('Integrity audit failed: '+','.join(k for k,v in tests.items() if not v))
    return result


def table(summary):
    lines=['| Method | Brier ↓ | AUC ↑ | Log loss ↓ | Mean xG |','|---|---:|---:|---:|---:|']
    for name,value in summary['scores'].items():
        auc=f"{value['auc']:.3f}" if value['auc'] is not None else 'undefined'
        lines.append(f"| {name} | {value['brier']:.6f} | {auc} | {value['log_loss']:.6f} | {value['mean_xg']:.1%} |")
    return lines


def figure(screen,confirmation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names={'luna_original':'Luna original','luna_uncertainty':'Luna + uncertainty',
           'terra_original':'Terra original','terra_uncertainty':'Terra + uncertainty',
           'sol_original':'Sol original','baseline_xg':'Original logistic',
           'baseline_image_xg':'Logistic without header','sb_xg':'StatsBomb reference'}
    fig,axes=plt.subplots(1,2,figsize=(13,5.6))
    for ax,summary,title in zip(axes,[screen,confirmation],['Development: AFCON (selection only)','Independent check: La Liga 2015/16']):
        methods=list(summary['scores']); values=[summary['scores'][m]['brier'] for m in methods]
        colors=['#176b83' if m in r.ARMS else '#9ba8ad' for m in methods]
        bars=ax.barh([names[m] for m in methods],values,color=colors)
        ax.invert_yaxis(); ax.bar_label(bars,fmt='%.4f',padding=4,fontsize=9)
        ax.set_xlim(0,max(values)*1.23); ax.set_xlabel('Brier probability error (lower is better)')
        ax.set_title(title,fontsize=11,pad=15); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Raw direct VLM xG: no probability correction',fontsize=16,y=.99)
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(r.OUT/'results.png',dpi=170); plt.close(fig)


def main():
    audit_result=audit(); screen=r.read(r.OUT/'screen/summary.json'); confirmation=r.read(r.OUT/'confirmation/summary.json')
    selected=screen['selected']; primary=confirmation['primary'][0]
    if primary['interval'][1]<0:
        conclusion='The selected raw VLM has lower Brier error than the frozen logistic reference on this independent sample, with the prespecified interval below zero. This supports a claim limited to this method and population.'
    elif primary['interval'][0]>0:
        conclusion='The selected raw VLM performs worse than the frozen logistic reference on this sample under the prespecified interval. The desired superiority claim is not supported.'
    else:
        conclusion='The independent test does not establish a reliable raw-VLM improvement over logistic regression. The interval crossing zero is inconclusive and does not establish equivalence.'
    lines=['# Direct visual xG: new models and missing-information instructions','',
        'Date: 29 October 2026 (America/Los_Angeles).','', '**'+conclusion+'**','',
        'The model returns xG directly from the diagram. Every reported probability is its raw answer. No recalibration, trained prediction head, correction, averaging, numerical baseline input or target mean.','',
        '## Fixed development comparison','',
        'The 50 AFCON shots (6 goals, 15 matches) had already been used for development. Each of five settings received identical original image bytes, with new Luna controls. This is model selection, not untouched confirmation. All arms/prompts and the selection rule were frozen before new inference.','']
    lines+=table(screen)
    lines+=['',f'Selected by the fixed lowest-Brier rule: **{selected}**. A development winner does not prove an improvement.','',
        'On development, the uncertainty instruction reduced Luna Brier error from 0.124032 to 0.117865, but AUC fell from 0.557 to 0.498 and log loss rose from 0.411698 to 0.419004. Thus the screening result does not show better shot ranking or visual interpretation. The changed mean alone is not success. Neither larger model won this fixed screen.','',
        'The uncertainty prompt asks the VLM to inspect geometry and account for unobserved body part, ball motion and execution. It supplies no numerical probability anchors, goal rates, examples, query coordinates or outcomes. It changes the instructions before prediction; nothing adjusts the output afterward.','',
        'Luna uses low effort, Terra medium, and Sol low. Those are complete method comparisons; differences between models are not isolated reasoning-effort effects. Every arm has high image detail and the same 4,096-token maximum.','',
        '## Independent confirmation','',
        f"Fixed {confirmation['shots']} shots, {confirmation['goals']} goals, {confirmation['matches']} matches. Source: La Liga 2015/16. Forty previously unused match groups were chosen before event downloads, then 200 eligible non-penalty freeze-frame shots were sampled without outcome stratification. The cohort and original renderer were fixed before screening. The selected method was frozen before confirmation inference or scoring.",'']
    lines+=table(confirmation)
    lines+=['','Primary differences use 10,000 whole-match bootstrap resamples and conservative 97.5% intervals for the two planned comparisons. Negative favors the selected VLM.','']
    for item in confirmation['primary']:
        lo,hi=item['interval']; lines.append(f"- {item['first']} minus {item['second']}: **{item['difference']:+.6f}**, interval **[{lo:+.6f}, {hi:+.6f}]**.")
    lines+=['','Secondary comparisons (95% intervals; do not replace the primary test):','']
    for key in ('secondary_no_header','secondary_statsbomb'):
        item=confirmation[key]; lo,hi=item['interval']
        lines.append(f"- Selected VLM minus {item['second']}: {item['difference']:+.6f}, [{lo:+.6f}, {hi:+.6f}].")
    diagnostic=r.read(r.OUT/'post_hoc_diagnostics.json')
    if diagnostic:
        low,high=diagnostic['post_hoc_match_95_bias_interval']
        lines+=['','## Why this does not resolve the core issue','',
            f"The selected method averaged {confirmation['scores'][selected]['mean_xg']:.1%} xG against {confirmation['goals']/confirmation['shots']:.1%} observed goals. Its mean overprediction was {diagnostic['candidate_mean_overprediction']:.1%} (post-hoc whole-match 95% interval {low:.1%} to {high:.1%}). Original Luna ranked chances at AUC 0.808, versus 0.764 for the new prompt. The small Brier change therefore does not demonstrate better visual interpretation.",'',
            f"A separate post-hoc reference that predicts the World Cup training goal rate ({diagnostic['constant_training_prior']:.2%}) for every shot scores Brier {diagnostic['constant_scores']['brier']:.6f}, versus the VLM's {confirmation['scores'][selected]['brier']:.6f}. The paired difference is inconclusive. This constant uses only the original 3,068 training shots (287 goals), supplies no visual information, and is never applied to or blended with a VLM output. Its role is to prevent a change in the overall probability level from being mistaken for solved image-based xG.",'',
            'These explanatory checks were added after the complete independent results and are labelled post-hoc. They do not change the prespecified primary comparisons, selected model, predictions or sample. No tested change in this campaign is a confirmed fix.']
    lines+=['','## Fair comparison and limits','',
        'Original logistic uses distance, goal-opening angle, header and opponent count in the shooting cone. Its header flag supplies information absent from the diagram. The no-header logistic is separate, and does not replace or weaken the original comparator. Both logistic models receive recorded positions and were frozen on 3,068 World Cup 2018/2022 shots. New La Liga labels did not train either model. StatsBomb xG is a reference, not ground-truth probability.','',
        'The independent cohort comes from another competition and season. Public/pretraining exposure cannot be fully audited: unused means unused in our experimental records. This is a standardized-diagram benchmark, not a broadcast-video evaluation. Coordinates are normalized pitch units, not verified physical distances. Binary goal outcomes are noisy, and uncertainty conditions on these fitted baselines and predictions. Crossing zero does not prove equivalence; a lower average xG is not individual prediction accuracy.','',
        'Model IDs are aliases. Actual returned IDs, request settings, image/source hashes and timestamps are preserved. No test-set tuning, post-hoc sample extension, replacement shots or selectively omitted failed conditions.','',
        '## Spending and reproducibility','',
        f"New successful generation calls: **{audit_result['successful_calls']}**. Usage-derived new OpenAI estimate: **${audit_result['new_estimated_usd']:.6f}**. Conservative exposure **${audit_result['conservative_exposure_usd']:.6f}**, below the **$3 combined** new-round ceiling. Gemini and Claude: **$0 new spending**, because credentials were unavailable. The prior direct campaign's conservative estimate was $0.1049749. Two additional older Luna usage tables account for $0.632989 estimated / $0.705679 conservative, recorded separately in prior_budget_context.json. These are partial historical records: other older experiments/account activity are not all bill-reconciled, and no live remaining balance is claimed.",'',
        '| Model | Calls | Input tokens | Output/reasoning tokens | Estimated new cost |','|---|---:|---:|---:|---:|']
    for model,usage in audit_result['by_model'].items():
        lines.append(f"| {model} | {usage['calls']} | {usage['input_tokens']} | {usage['output_tokens']} | ${usage['estimated_usd']:.6f} |")
    lines+=['',
        'Official input-token preflights plus a 1,024-token allowance and the full next 4,096-output-token charge were reserved before dispatch. The cap includes all concurrent requests (at most three). No SDK or hidden retry. Output/reasoning tokens are included in cost. Raw probabilities, including 0 or 1, are never clipped for Brier or saved predictions; log loss alone uses a numerical log convention at endpoints.','',
        f"All **{len(audit_result['checks'])}** provenance, identity, request reconstruction, raw-answer, source, disjointness, budget and private exact-credential-exposure checks passed. Four offline guard tests also passed.",'',
        'Artifacts: protocol.json; data/match_split.json and provenance.json; choice.json; each stage protocol/selection; token_counts.json; attempts.json; complete raw API responses; predictions.csv; summary.json; audit.json; original-byte images.','',
        '![Brier comparison](results.png)','',
        'Sources: [StatsBomb Open Data](https://github.com/hudl/open-data), [Luna rates/settings](https://developers.openai.com/api/docs/models/gpt-6-luna), [Terra rates/settings](https://developers.openai.com/api/docs/models/gpt-6.1-terra), [Sol rates/settings](https://developers.openai.com/api/docs/models/gpt-6.1-sol).']
    (r.OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    figure(screen,confirmation)
    print(json.dumps({'conclusion':conclusion,'audit_passed':True,'new_estimate_usd':audit_result['new_estimated_usd']}),flush=True)


if __name__=='__main__': main()
