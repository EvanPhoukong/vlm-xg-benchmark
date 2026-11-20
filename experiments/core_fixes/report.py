"""Interpret the completed core-fix experiments without changing probabilities."""
import json
import common as c
NAMES={'luna':'Original Luna (zero-shot)','astra':'Astra (zero-shot)','examples':'Luna + correct outcome examples',
       'shuffled':'Luna + shuffled labels (control)','original_logistic':'Original logistic (4 inputs)',
       'original_no_header':'Original logistic without header','sb_xg':'StatsBomb reference',
       'training_prior':'WC training-only constant','example_rate':'Example-frequency constant'}


def table(scores,names=NAMES):
    lines=['| Method | Brier ↓ | AUC ↑ | Log loss ↓ | Mean xG |','|---|---:|---:|---:|---:|']
    for key,score in scores.items():
        auc=f"{score['auc']:.3f}" if score['auc'] is not None else 'undefined'
        lines.append(f"| {names.get(key,key)} | {score['brier']:.6f} | {auc} | {score['log_loss']:.6f} | {score['mean_xg']:.1%} |")
    return lines


def verdict(summary):
    main=summary['main']; comparisons={(x['first'],x['second']):x for x in main['primary']}
    confirmed=[arm for arm in ('astra','examples') if comparisons[arm,'luna']['interval'][1]<0]
    if confirmed:
        return 'The fixed independent test supports lower raw Brier error than original Luna for '+', '.join(NAMES[a] for a in confirmed)+'. Baseline superiority, ranking, negative controls and consistency are reported separately; a Brier gain alone is not a complete core fix.'
    better=[a for a in ('astra','examples') if main['scores'][a]['brier']<main['scores']['luna']['brier']]
    if better:
        return 'There are lower raw Brier point estimates, but the prespecified 99% intervals do not establish a reliable improvement over original Luna. The diagnostic evidence below determines which ideas merit further work.'
    return 'Neither fixed candidate improved the raw Brier point estimate over original Luna on the new cohort. The negative and inconclusive results are retained; no output correction was substituted.'


def figure(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main=summary['main']; consistency=summary['consistency']
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    keys=['luna','astra','examples','shuffled','original_logistic','original_no_header','training_prior']
    labels={'luna':'Luna original','astra':'Astra original','examples':'Luna + labelled examples','shuffled':'Shuffled-label control',
            'original_logistic':'Original logistic','original_no_header':'Logistic without header','training_prior':'Training-only constant'}
    values=[main['scores'][k]['brier'] for k in keys]
    bars=axes[0].barh([labels[k] for k in keys],values,color=['#176b83' if k in c.ARMS else '#9ba8ad' for k in keys])
    axes[0].invert_yaxis(); axes[0].bar_label(bars,fmt='%.4f',padding=4,fontsize=9)
    axes[0].set_xlim(0,max(values)*1.23); axes[0].set_xlabel('Brier probability error (lower is better)')
    axes[0].set_title(f"Independent new cohort: {main['shots']} shots, {main['goals']} goals")
    arms=list(c.ARMS); changes=[round(100*consistency[a]['mean_absolute_repeat_change'],8) for a in arms]
    bars=axes[1].barh([labels[a] for a in arms],changes,color='#176b83')
    axes[1].invert_yaxis(); axes[1].bar_label(bars,fmt='%.2f',padding=4,fontsize=9)
    axes[1].set_xlim(0,max(changes)*1.27); axes[1].set_xlabel('Mean absolute xG change (percentage points)')
    axes[1].set_title('Exact-repeat diagnostic: 12 reused scenes')
    for axis in axes: axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Direct xG: stronger model, outcome examples and raw stability',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(c.OUT/'comparison.png',dpi=160); plt.close(fig)
    # Reader illustration only: these labels never modify the frozen API images.
    from PIL import Image
    sid=c.read(c.OUT/'motion/selection.json')[0]['shot_id']
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for axis,offset,label in zip(axes,[-25,-12,0],['1.00 s before shot start','0.48 s before shot start','Annotated shot start']):
        axis.imshow(Image.open(c.OUT/'images/motion'/sid/(str(offset)+'.png')))
        axis.set_title(label,fontsize=11); axis.axis('off')
    fig.suptitle('Actual tracking snapshots: magenta diamond is the ball',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(c.OUT/'motion_inputs.png',dpi=150); plt.close(fig)


def main():
    summary=c.read(c.OUT/'summary.json'); main=summary['main']; audit=summary['audit']; consistency=summary['consistency']; motion=summary['motion']
    if not audit['passed']: raise ValueError('Final audit must pass')
    text=verdict(summary); pairs={(x['first'],x['second']):x for x in main['primary']}
    choice=c.read(c.OUT/'resource_choice.json'); examples=c.read(c.OUT/'training/examples.json')
    short=main['strata']['short_6_18']
    lines=['# Core-fix experiments: raw visual xG','', '29 October 2026 — America/Los_Angeles','', '**'+text+'**','',
        f"Completed **{audit['calls']} generation calls**. New usage estimate **${audit['estimated_usd']:.6f}**, conservative **${audit['conservative_usd']:.6f}**, below the **$4** round cap. All reported VLM predictions are raw API answers. No paid runner remains active.",'',
        '## Findings in plain language','',
        f"- **Astra is the strongest supported change.** Raw Brier fell from {main['scores']['luna']['brier']:.4f} to {main['scores']['astra']['brier']:.4f}, an {(1-main['scores']['astra']['brier']/main['scores']['luna']['brier']):.1%} reduction. The prespecified 99% paired match interval versus Luna excludes zero. AUC rose from {main['scores']['luna']['auc']:.3f} to {main['scores']['astra']['auc']:.3f}, and observed repeat variation fell from {round(100*consistency['luna']['mean_absolute_repeat_change'],8):.2f} to {round(100*consistency['astra']['mean_absolute_repeat_change'],8):.2f} percentage points. The latter diagnostics are small and descriptive.",
        f"- **The gain is not explained by matching the overall average.** Original Luna already averaged {main['scores']['luna']['mean_xg']:.1%} versus {main['goals']/main['shots']:.1%} actual goals; Astra averaged {main['scores']['astra']['mean_xg']:.1%}, yet had better Brier and ranking. Its 6–18-unit Brier also fell from {short['scores']['luna']['brier']:.4f} to {short['scores']['astra']['brier']:.4f}.",
        f"- **A logistic win is not established.** Astra's {main['scores']['astra']['brier']:.4f} is below the original logistic point estimate {main['scores']['original_logistic']['brier']:.4f}, but their 99% interval spans zero. This does not establish equivalence either.",
        '- **The 48 correct outcome examples are not a validated fix.** They did not establish an advantage over original Luna or the shuffled-label control. The shuffled control scored better by point estimate, so selecting it as a successful learned method would be misleading. Correct image/outcome learning has not been demonstrated.',
        f"- **Past tracking worsened this Luna pilot.** History Brier {motion['scores']['history']['brier']:.4f} versus matched single-frame {motion['scores']['single_ball']['brier']:.4f}; ranking and repeat stability were also weaker. The current-frame-plus-ball condition is a possible separate lead, but its comparison with the original also changes the prompt, and two games cannot establish a solution.",'',
        '## What changed and why','',
        '- **Stronger zero-shot model:** original image-only prompt and image, comparing GPT-6 Astra low against GPT-6 Luna low. No target probabilities or query metadata.',
        '- **Task-specific visual examples:** Luna sees 48 fixed historical World Cup shot images with their observed goal/no-goal outcomes. This is a separate few-shot condition, not zero-shot or weight fine-tuning. Single outcomes are explicitly described as noisy rather than true xG 0/1.',
        '- **Shuffled-label negative control:** exactly the same historical images and positive-label count, but labels permuted with a fixed seed. This distinguishes image/outcome association from generic example-frequency conditioning.',
        '- **Additional temporal information:** actual Metrica tracking snapshots at 1.00 and 0.48 seconds before the provider shot-start frame, plus that frame. A matched single-frame condition includes the identical final image and tracked-ball marker. No future frames or invented body pose.',
        '- **Stability and visual behavior:** new exact repeats on 12 fixed development scenes and 12 fixed synthetic favorable/base pairs, with no probability averaging or monotonicity enforcement.','',
        f"The 48 natural random training examples contain **{sum(x['goal'] for x in examples)} goals**. Their group IDs are disjoint from the new query matches. Earlier worked-example studies supplied four geometry examples with no outcomes; the current outcome-learning and label-permutation experiments are new. No example xG estimates, query coordinates, baseline predictions or selected target rate are supplied.",'',
        '## New match-group test','',
        f"The source pool was frozen at 200 natural-sample shots from 34 previously unused La Liga 2016/17 match groups, excluding all 488 prior recorded StatsBomb groups. A 12-shot ×4-condition reused-AFCON pilot measured validity and usage only. The prespecified resource rule selected **{choice['n']}** shots, before any accuracy scoring: projected conservative totals for 200/150/100 were "+', '.join(f"${choice['projections_usd'][str(n)]:.3f}" for n in (200,150,100))+'. The chosen cohort is the exact prefix of the reserved random ordering. No result-driven sampling or method selection.','',
        f"Actual evaluation: **{main['shots']} shots, {main['goals']} goals ({main['goals']/main['shots']:.1%}), {main['matches']} matches**. All four arms use new calls and randomized within-shot order.",'']
    lines+=table(main['scores'])
    lines+=['','Five prespecified paired Brier comparisons use 99% whole-match bootstrap intervals (10,000 resamples), correcting within this study for five primary comparisons. Negative favors the first method. This exploratory program also contains earlier studies; these intervals do not adjust for every historical comparison.','',
        '| Comparison | Brier difference | 99% match interval |','|---|---:|---|']
    for item in main['primary']:
        lo,hi=item['interval']; lines.append(f"| {NAMES[item['first']]} minus {NAMES[item['second']]} | {item['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','Intervals crossing zero establish neither superiority nor equivalence. The original logistic uses a header flag absent from query images; its no-header version is separate, not a silently weakened replacement. Both use recorded positions and remain frozen on 3,068 World Cup shots. StatsBomb is another estimate rather than ground-truth shot probability.','',
        '## Does a candidate address the core errors?','',
        f"The prespecified short-range diagnostic contains {short['shots']} shots and {short['goals']} goals in the 6–18 normalized-unit band. These are normalized pitch coordinates, not verified metres/yards. Subgroup results are descriptive, with no independent subgroup-success claim.",'']
    lines+=table(short['scores'])
    lines+=['','Correct-versus-shuffled examples are the key learning control. A lower Brier than original Luna accompanied by a similar shuffled-label gain would not establish that the model learned useful image/outcome associations. The training-prior and example-frequency constants are separate uninformative references; neither is applied to any VLM output. Ranking, short-range performance and consistency also matter.','',
        '| Method | Mean absolute exact-repeat change | Expected visual direction |','|---|---:|---:|']
    for arm,value in consistency.items():
        lines.append(f"| {NAMES[arm]} | {round(100*value['mean_absolute_repeat_change'],8):.2f} percentage points | {value['expected_direction_count']}/12 |")
    lines+=['','Exact repeats compare individual raw outputs to the corresponding initial pilot response. Their average absolute differences are diagnostic summaries, not averaged predictions used for accuracy. The 12 synthetic templates share structure; angle changes also alter the keeper’s relationship to the shooting line. They provide qualitative checks rather than independent football outcomes or exact probability targets. Every tie and reversal is preserved in consistency.json.','',
        '## Does past tracking context help?','',
        f"{motion['shots']} previously used Metrica shots, {motion['goals']} goals, {motion['matches']} anonymized games. Three source cases lacked some required past actor/ball tracking and were excluded before any call, without inspecting outcomes or errors. All eligible cases were retained. This reused two-game pilot is descriptive, not new independent confirmation.",'']
    motion_names={'original':'Original single snapshot','single_ball':'Matched current frame + ball','history':'Past snapshots + same current frame',
                  'original_logistic':NAMES['original_logistic'],'original_no_header':NAMES['original_no_header']}
    lines+=table(motion['scores'],motion_names)
    lines+=['',f"History minus matched single-frame Brier: **{motion['history_minus_single_brier']:+.6f}**. The same generic motion prompt is used for both matched conditions. Last-frame pixels are identical. Original single-snapshot prompting is shown as a separate reference.",'']
    for arm,value in motion['exact_repeats'].items():
        lines.append(f"{motion_names[arm]}: mean absolute repeat change {100*value['mean_absolute_change']:.2f} points over {value['cases']} cases.")
    lines+=['','These are 2-D tracking renders, not actual broadcast frames or body pose. The original field normalization is approximate across providers, and event timing is the provider annotation rather than independently verified physical contact. All selected frames are at/before the annotated shot start. End frame/coordinates, outcome tags, identities and clocks do not enter images. Ball height, detailed pose and richer physical execution remain missing. The user had no authorized local footage; xGHub access restrictions were respected.','',
        '![Example of the added past tracking information](motion_inputs.png)','',
        '## Integrity, cost and limits','',
        f"All **{len(audit['checks'])}** final audit checks passed, plus four offline control tests: source/model/input hashes, correct training labels, fixed negative control, query-group exclusion, source frames, resource-choice reproduction, exact reconstructed requests, raw answers, model identity/settings/usage, concurrent reservations and private exact-credential exposure. No hidden retries, dropped failures, result-driven replacements, post-hoc probability correction, statistical VLM head, blends or enforced output consistency.",'',
        '| Model | Calls | Input tokens | Cached input | Output including reasoning | Estimated cost |','|---|---:|---:|---:|---:|---:|']
    for model,value in audit['by_model'].items():
        lines.append(f"| {model} | {value['calls']} | {value['input_tokens']} | {value['cached_input_tokens']} | {value['output_tokens']} | ${value['estimated_usd']:.6f} |")
    lines+=['',f"New round estimate ${audit['estimated_usd']:.6f}; conservative ${audit['conservative_usd']:.6f}, capped at $4. Known older partial conservative records total $1.4552639. Complete bills and other account activity are unavailable, so no live remaining balance is claimed. Gemini/Claude incur no new charges. Published standard prices were verified before inference, and full output cost plus input safety allowances were reserved before every request. In-flight reservations are allowed to reconcile before testing another reservation; the cap is never raised automatically.",'',
        'The requested models are aliases, with returned identifiers and UTC request times preserved. Pretraining exposure is unknown. Match intervals condition on these predictions and fitted references, not all training or model-call uncertainty. The main cohort is modest and its size was set by resource usage; any favorable finding needs independent replication. No fine-tuning was performed.','',
        'All 34 reserved new match groups are now used and should be excluded from future untouched-data claims (522 recorded groups total). The Metrica games, old AFCON pilot and synthetic probes remain explicitly reused diagnostics.','',
        '![Raw accuracy and repeat stability](comparison.png)','',
        'Sources: [Astra settings/rates](https://developers.openai.com/api/docs/models/gpt-6-astra), [Luna settings/rates](https://developers.openai.com/api/docs/models/gpt-6-luna), [StatsBomb Open Data](https://github.com/hudl/open-data), [Metrica tracking data](https://github.com/metrica-sports/sample-data).']
    (c.OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); figure(summary)
    c.save(c.OUT/'candidate.json',{'status':'promising independently tested zero-shot candidate, not established logistic superiority',
        'model':'gpt-6-astra','reasoning_effort':'low','image_detail':'high','max_output_tokens':4096,
        'service_tier':'default','store':False,'prompt':c.old.original_prompt(),'schema':c.old.SCHEMA,
        'renderer_sha256':c.sha(c.ROOT/'xg_vlm_pilot.py'),'no_postprocessing':True,'no_examples':True,
        'evidence':'150 new shots, 25 goals, 34 matches. 99% five-comparison-adjusted match interval supports lower Brier than Luna. Ranking/short-range/stability diagnostics also improved by point estimate.',
        'main_summary_sha256':c.sha(c.OUT/'main/summary.json'),'original_benchmark_preserved':True,
        'next_validation':'Another fixed unused-match cohort is needed before a general or baseline-superiority claim. No new runner or automatic stage is created.'})
    c.save(c.OUT/'final_status.json',{'status':'complete','calls':audit['calls'],'new_estimated_usd':audit['estimated_usd'],
        'conservative_usd':audit['conservative_usd'],'cap_usd':c.CAP,'audit_passed':True,'active_runner':False,'conclusion':text})
    print(json.dumps(c.read(c.OUT/'final_status.json')),flush=True)


if __name__=='__main__': main()
