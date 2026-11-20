"""Aggregate completed studies without more model requests."""
import json
from pathlib import Path
import run as r


def main():
    main=r.read(r.OUT/'confirmation/summary.json'); audit=r.read(r.OUT/'audit.json')
    probe=r.read(r.ROOT/'results/visual_counterfactuals_v1/summary.json')
    prior=r.read(r.OUT/'prior_budget_context.json')
    total=audit['new_estimated_usd']+probe['estimated_usd']
    assert audit['passed'] and probe['request_and_response_audit']=='passed'
    assert probe['shared_conservative_usd']<=3
    assert not (r.OUT/'generation.lock').exists() and not (r.ROOT/'results/visual_counterfactuals_v1/generation.lock').exists()
    calls=audit['successful_calls']+probe['calls']
    summary={'status':'complete','new_generation_calls':calls,'new_estimated_usd':total,
        'new_conservative_usd':probe['shared_conservative_usd'],'combined_new_cap_usd':3,
        'new_gemini_usd':0,'new_claude_usd':0,'active_runner':False,
        'selected_method':'luna_uncertainty','confirmation':main,
        'controlled_visual_diagnostic':probe['results'],
        'conclusion':'No confirmed fix for direct visual xG. The missing-information prompt had a small inconclusive Brier improvement, weaker ranking and less consistent controlled visual behavior. Terra/Sol did not win development. Preserve all raw results; do not deploy this as a proven improvement.',
        'tracked_older_estimated_subtotal_usd':prior['estimated_usd']+.1004494,
        'account_balance_caveat':'New usage estimates are not bills; older records and other account activity are incomplete.'}
    r.save(r.OUT/'ROUND_SUMMARY.json',summary)
    lines=['# Further visual xG experiments: 29 October 2026','',
        '**No tested change is a confirmed fix. We found useful failure evidence, not proof of accurate direct xG.**','',
        f'Completed **{calls} new model calls**, estimated **${total:.4f}** in total. Conservative accounting **${probe["shared_conservative_usd"]:.4f}**, below the **$3 combined** ceiling for this round. No paid runner remains active.','',
        '## What was tested','',
        '- **Models:** Luna low effort, Terra medium and Sol low, using identical original diagrams and the original prompt.',
        '- **Missing-information instructions:** Luna and Terra were also asked to account for unknown body part, ball motion and execution. No numerical probability anchors or target scoring rate.',
        '- **Independent accuracy check:** selected the best development method, froze it, then tested 200 naturally sampled shots (16 goals) from 40 new La Liga 2015/16 match groups. All 316 previously recorded match groups were excluded.',
        '- **Controlled visual behavior:** 12 fixed synthetic cases covering distance, angle, keeper position and defensive blocking, with exact-repeat controls. This separate diagnostic was frozen before the independent accuracy results were scored.','',
        'Every xG value is the raw model answer. No recalibration, correction, statistical prediction head, baseline blending, output averaging or fine-tuning was used. The original logistic comparator is retained, including disclosure that its header flag is extra information absent from the image.','',
        '## Independent results','',
        'Lower Brier means lower probability error; higher AUC means better ranking. Compare methods on these same 200 shots, not scores across different cohorts.','',
        '| Method | Brier ↓ | AUC ↑ | Mean xG |','|---|---:|---:|---:|']
    labels={'luna_uncertainty':'Luna with missing-information prompt','luna_original':'Original Luna','baseline_xg':'Original four-feature logistic','baseline_image_xg':'Logistic without header','sb_xg':'StatsBomb reference'}
    for name in labels:
        score=main['scores'][name]
        lines.append(f"| {labels[name]} | {score['brier']:.4f} | {score['auc']:.3f} | {score['mean_xg']:.1%} |")
    lines+=['','The new prompt reduced Brier from 0.0787 to 0.0753, but that improvement is inconclusive. Its ranking was weaker. The raw VLM probabilities still averaged 15.1% when 8% of shots scored.','',
        'Prespecified paired Brier differences, with 97.5% whole-match intervals for two primary comparisons:','']
    for item in main['primary']:
        lo,hi=item['interval']; lines.append(f"- Selected Luna minus {labels[item['second']]}: **{item['difference']:+.5f}**, interval **[{lo:+.5f}, {hi:+.5f}]**.")
    lines+=['','Both intervals cross zero. They do not establish superiority or equivalence. StatsBomb remains a reference estimate rather than ground-truth shot probability.','',
        'A clearly labelled post-hoc check also compared the unchanged VLM against an uninformative constant learned only from World Cup training outcomes. The constant scored Brier 0.0738, versus 0.0753 for the selected VLM; the difference is inconclusive. It was never used to adjust a VLM answer. This check prevents a favorable overall probability level from being mistaken for a solved visual prediction problem.','',
        '## Did the model respond sensibly to changed pictures?','',
        '| Method | Expected direction | Mean absolute change on identical repeats |','|---|---:|---:|']
    for arm,label in [('luna_original','Original Luna'),('luna_uncertainty','Luna with missing-information prompt')]:
        value=probe['results'][arm]; lines.append(f"| {label} | {value['expected_direction_count']}/12 | {value['mean_absolute_repeat_change']:.1%} |")
    lines+=['','The new prompt reversed the expected change on five cases, including all three defender-blocking cases. Original Luna had one tie and one reversal. These small synthetic examples are correlated and cannot establish football probability accuracy; angle changes also alter the keeper’s relationship to the shooting line, so qualitative expectations are not exact xG truth. Still, the diagnostic gives no reason to treat the new prompt as a robust solution. No symmetry/monotonicity constraints or repeat averages were imposed afterward.','',
        '## Spending, records and interpretation','',
        f'New OpenAI estimate: **${total:.6f}**. New Gemini: **$0**. New Claude: **$0**. Only the OpenAI credential was configured. The known earlier direct campaign and two older usage tables are recorded separately; account-wide billing and remaining balances were not available. All new work stayed below the stricter $3 combined cap, with complete input/output reservations before each request and no hidden retries.','',
        'All raw requests were reconstructed and checked against their saved image/prompt hashes, model responses and usage. Source hashes, unused-match checks, unchanged probabilities, budget checks and private exact-credential scans passed. Four offline guard tests and the synthetic scene/repeat checks passed.','',
        'The observed evidence separates chance ranking from absolute probability accuracy. Original Luna has promising ranking on this cohort, but the tested methods do not establish accurate direct xG. Larger samples and other model/input hypotheses may help future research; this round is complete and has not silently adopted a probability correction.','',
        '[Full model-search methods, scores and audit](REPORT.md) · [Controlled visual diagnostic](../visual_counterfactuals_v1/REPORT.md)','',
        '![Raw Brier comparison](results.png)']
    (r.OUT/'ROUND_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('status','new_generation_calls','new_estimated_usd','new_conservative_usd','active_runner')}),flush=True)


if __name__=='__main__': main()
