"""Build the plain-English final report from completed frozen experiment summaries."""
from pathlib import Path
import json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as c

def label(name):
    return {'low':'Luna: low reasoning','medium':'Luna: medium reasoning','high':'Luna: high reasoning',
            'baseline_xg':'Logistic baseline (four inputs)','baseline_image_xg':'Logistic without header input','sb_xg':'StatsBomb reference',
            'standard':'Original filled shooter star','outline':'Hollow shooter star','candidate':'Chosen VLM method',
            'original_low':'Original diagram, low reasoning','standard_selected':'Original diagram, chosen reasoning'}.get(name,name)

def table(summary):
    out=['| Method | Brier ↓ | AUC ↑ | Log loss ↓ | Mean predicted chance |','|---|---:|---:|---:|---:|']
    for key,s in summary['scores'].items():
        auc=f"{s['auc']:.3f}" if s['auc'] is not None else 'not defined'
        out.append(f"| {label(key)} | {s['brier']:.5f} | {auc} | {s['log_loss']:.5f} | {s['mean_xg']:.1%} |")
    return '\n'.join(out)

def interval(t):
    return f"{t['brier_difference']:+.5f}, with a {t['confidence_level']:.1%} match-based interval [{t['interval'][0]:+.5f}, {t['interval'][1]:+.5f}]"

def verdict(t):
    lo,hi=t['interval']
    if hi<0: return 'The interval supports lower probability error for the first method on this cohort.'
    if lo>0: return 'The interval supports higher probability error for the first method on this cohort.'
    return 'The interval includes no improvement; the comparison is inconclusive. This does not prove equivalence.'

def main():
    ledger=c.read(c.OUT/'attempts.json',[]); summaries={n:c.read(c.OUT/n/'summary.json') for n in ('reasoning','rendering','confirmation','invariance')}
    candidate=c.read(c.OUT/'candidate.json'); fresh=summaries['confirmation']; total=c.usage(ledger); cost=total['estimated_usd']+c.PRIOR_COST
    stopped=[a for a in ledger if a['state']!='completed']
    lines=['# Can Luna estimate xG directly from a shot image?', '', '## Findings in plain English', '']
    if fresh:
        test=fresh['primary'][0]; lo,hi=test['interval']
        if hi<0: conclusion='The chosen Luna method reduced Brier error below the original four-feature logistic baseline on the independent confirmation sample.'
        elif lo>0: conclusion='The tested improvements did not beat logistic regression: the chosen Luna method had higher Brier error on the independent confirmation sample.'
        else: conclusion='We did not establish that the chosen Luna method beats logistic regression. The independent confirmation result is inconclusive.'
        lines += [f'**{conclusion}**', '', f"The confirmation contains **{fresh['shots']} shots, {fresh['goals']} goals and {fresh['matches']} matches**. The selected method uses **{candidate['effort']} reasoning and the {candidate['renderer']} diagram**. Its probabilities come directly from Luna, with no recalibration or statistical model changing the answers.", '',
                  f"Its Brier score was **{fresh['scores']['candidate']['brier']:.5f}**, versus **{fresh['scores']['baseline_xg']['brier']:.5f}** for logistic regression. Its AUC was **{fresh['scores']['candidate']['auc']:.3f}**, versus **{fresh['scores']['baseline_xg']['auc']:.3f}** for logistic regression.", '',
                  'This conclusion applies to this model, frozen method and eligible AFCON sample. It is not a claim about every VLM or all football competitions.']
    else:
        lines += ['**No independent confirmation result is available.** The report distinguishes completed development evidence from interrupted stages and does not present partial cohorts as final accuracy results.']
    lines += ['',f"**Estimated API spending: ${cost:.4f}**, including the earlier 12-shot pilot. The campaign working cap was **$2**, below the user's approximately **$10** limit. These are usage-based estimates at published Luna rates, not a billing statement or account-balance check.", '',
              '## What was kept fixed', '',
              'The research question stayed the same: give a general-purpose vision-language model an image of a shot and ask it to estimate the probability of a goal. The model received the image, a generic marker legend and the original zero-shot instructions. It did not receive the shot outcome, StatsBomb xG, the logistic prediction, numerical coordinates, body-part metadata or worked examples.', '',
              'No probability correction, supervised prediction head, averaging, baseline blending or fine-tuning was used. Every reported VLM xG is the raw answer returned by GPT-6 Luna. The only candidate changes were reasoning effort and a small change to how the shooter marker was drawn.', '',
              'All three reasoning settings received the same 4,096-token maximum in the new study. The earlier 12-shot pilot used 2,048; its results are kept separate. A larger allowance lets a model finish reasoning; it does not change an answer afterward.', '',
              '## How to read the scores', '',
              '- **Brier** measures probability error; lower is better. It is the primary measure.',
              '- **AUC** measures how well high-quality chances rank above low-quality chances; higher is better. AUC of 0.5 means chance-level ranking.',
              '- **Log loss** also measures probability error and penalizes confident mistakes; lower is better.',
              '- **Mean predicted chance** only checks the overall level. Matching the observed goal rate does not establish good predictions for individual shots.',
              '- **Uncertainty intervals** resample whole matches, keeping shots from a match together. An interval spanning zero cannot establish an improvement.', '',
              '## 1. Did more reasoning help?', '']
    reasoning=summaries['reasoning']
    if reasoning:
        lines += [f"This was a fixed **{reasoning['shots']}-shot development sample**, with **{reasoning['goals']} goals across {reasoning['matches']} matches**, from previously unused AFCON 2023 match groups. Each image was sent once at low, medium and high reasoning effort, in randomized order.", '',table(reasoning), '',
                  'Prespecified medium-minus-low Brier comparison: '+interval(reasoning['primary'])+'. '+verdict(reasoning['primary']), '',
                  f"The lowest development Brier selected **{reasoning['selected_effort']} reasoning**. High-effort comparisons and the selection itself are exploratory; the independent cohort checks whether a chosen benefit survives outside the selection sample.", '',
                  'The earlier 12-shot pilot looked favorable for medium reasoning, but had only two goals and one scored shot contributed about 62% of its net improvement. Its 22% relative Brier reduction was not a reliable baseline-win claim.', '', 'On the new 50-shot cohort, low and high effort had the same mean xG (15.42%) but different Brier scores and ranking. This illustrates why fixing only the average probability would not address all of the observed errors.']
    else: lines+=['Not completed; no partial-cohort accuracy claim is made.']
    lines += ['', '## 2. Did preventing marker overlap help?', '']
    renderer=summaries['rendering']
    if renderer:
        lines += [f"On **{renderer['shots']} World Cup development shots**, the shooter stayed at exactly the same position and marker size. We compared the original filled gold star with a hollow gold star. This can reveal nearby defender markers that an opaque star covers. All other players, pitch markings and instructions stayed unchanged.", '',
                  'This is a specific rendering intervention. It is not a rerun of the earlier numeric-grid, zoom, worked-example or role-separated-panel methods. These World Cup scenes have been studied before, so this comparison is exploratory.', '',table(renderer), '',
                  'Hollow-minus-filled Brier: '+interval(renderer['primary'])+'. '+verdict(renderer['primary']), '',
                  f"The prespecified selection rule chose the **{renderer['selected_renderer']} renderer**. An apparent development improvement was not called confirmed without the separate match-group check below."]
    else: lines+=['Not completed; no partial-cohort accuracy claim is made.']
    lines += ['', '![Filled versus hollow shooter marker](renderer_preview/comparison.png)', '', 'The same development scene is shown twice. Only the shooter-star style changes; this illustration does not by itself demonstrate improved VLM perception.', '', '## 3. Independent confirmation', '']
    if fresh:
        lines += ['The 150-shot cohort and its match groups were reserved before any new model results. The candidate was frozen after development and before confirmation inference. Selection did not enforce a goal rate, cherry-pick difficult cases or balance shot distances. No settings were changed after confirmation scores were viewed.', '',table(fresh), '',
                  'The two primary comparisons use **97.5% intervals**, allowing for testing two questions:', '']
        for t in fresh['primary']:
            if t['second']=='original_low' and candidate['effort']=='low' and candidate['renderer']=='standard':
                lines.append('- The selected candidate is the unchanged original low-effort method. Its response is reused under both labels, avoiding duplicate charges; there is no separate method contrast to test. The baseline interval remains conservatively 97.5%.')
            else: lines.append('- '+label(t['first'])+' minus '+label(t['second'])+': '+interval(t)+'. '+verdict(t))
        t=fresh['secondary_no_header']
        lines += ['', 'A secondary comparison against logistic regression without the header flag gives '+interval(t)+'. This is secondary and does not replace the original four-input baseline.', '',
                  'A model can have lower Brier error but weaker ranking, or the reverse. Both are reported rather than choosing whichever metric makes the model look best.']
    else: lines+=['The reserved cohort was not completed. It remains separate from development; no partial confirmation scores are reported.']
    lines += ['', '## 4. Do irrelevant image changes alter the answer?', '']
    diagnostic=summaries['invariance']
    if diagnostic:
        lo,hi=diagnostic['mirror_minus_repeat_interval_95']
        lines += [f"For **{diagnostic['shots']} development scenes**, an unchanged repeat and a horizontal mirror were compared with the earlier original reading. All football positions are mirrored together and the goal remains at the top. With body part unknown, similar probabilities are expected, but exact equality is not assumed.", '',
                  f"Average absolute xG change: **{100*diagnostic['mean_repeat_change']:.2f} percentage points** for an ordinary repeat; **{100*diagnostic['mean_mirror_change']:.2f} points** for mirroring. The mirror-minus-repeat difference has a 95% match interval of **[{100*lo:+.2f}, {100*hi:+.2f}] percentage points**.", '',
                  'This diagnoses sensitivity to presentation and repeated-call variability. It is not an additional goal-labelled test set or a probability-improvement method. We did not average the answers or enforce symmetry afterward.']
    else: lines+=['Not completed. No diagnostic conclusion is claimed.']
    lines += ['', '## Data, baseline fairness and limits', '',
              'The source remained StatsBomb Open Data. New inference scenes came from the 2023 African Cup of Nations. Preparation excluded all 264 prior recorded match IDs. A seeded split reserved separate match groups for the 50-shot development and 150-shot confirmation cohorts. Penalties and shots lacking the same freeze-frame eligibility as the original study were excluded.', '',
              'The main logistic baseline was fitted once on 3,068 shots from World Cups 2018 and 2022. It uses distance, goal-opening angle, a header flag and opponent count in the shooting cone. The header flag supplies information absent from the image. A body-part-free logistic comparator is therefore shown separately on fresh data. Both numerical baselines receive recorded coordinates, including positions that may overlap in the image; they are practical references rather than identical pixel-input models.', '',
              'Rendering development used saved match-grouped out-of-fold baseline predictions, avoiding an in-sample numerical baseline score. No confirmation matches trained either logistic model. Training coefficients, selected IDs, source-file hashes and image hashes are preserved.', '',
              'Coordinates are normalized StatsBomb pitch units, not verified physical metres or yards. The original Sloan write-up evaluated 298 common shots; later saved Luna data contain 300. Those historical cohorts and the new cohorts are not silently pooled.', '',
              'Fresh means unused in our experimental records and statistical training; VLM pretraining data cannot be audited here. The statistical intervals condition on the fitted baseline models and do not include every source of uncertainty. Snapshots omit body pose, ball height and movement. The model name is an API alias; returned identifiers and request times are saved. Broader conclusions would require other independent competitions.', '',
              '## Cost and reproducibility', '',
              '| Stage | Successful generation calls | Estimated API cost |','|---|---:|---:|',f'| Earlier pilot | 24 | ${c.PRIOR_COST:.6f} |']
    for name in ('reasoning','rendering','confirmation','invariance'):
        entries=[a for a in ledger if a['stage']==name]; count=sum(a['state']=='completed' for a in entries)
        lines.append(f"| {name} | {count} | ${c.usage(entries)['estimated_usd']:.6f} |")
    lines += ['',f"New campaign token totals: **{total['input_tokens']:,} input**, **{total['output_tokens']:,} output including {total['reasoning_tokens']:,} reasoning tokens**. Cached-input tokens: {total['cached_tokens']:,}. Every next request reserved its full allowed output cost before being sent. No hidden retries or automatic replacements were used.", '',
              'The API runner stopped on exhausted credit, uncertain billing, invalid output or a hard limit. An incomplete stage is reported as incomplete rather than quietly dropping shots. Public-data downloads and local statistical resampling are separate from model generation calls.', '',
              'Preserved files include campaign_protocol.json, each stage protocol and selection, match_split.json, attempts.json, token_counts.json, raw response files, predictions.csv, score summaries, image files and the experiment log. No API keys are stored in these artifacts.', '',
              'Sources: [StatsBomb Open Data](https://github.com/hudl/open-data), [official Luna rates/settings](https://developers.openai.com/api/docs/models/gpt-6-luna), [official token-accounting guidance](https://developers.openai.com/api/docs/guides/token-counting).', '',
              '## Why experimentation stopped', '',
              'The fixed sequence tests reasoning, a documented input-visibility problem and presentation stability, followed by a match-disjoint check. Further prompt searching on the same confirmation outcomes would weaken the evidence. The remaining account budget is not a reason to keep sampling until a favorable result appears.']
    if stopped: lines+=['',f'Generation also stopped with {len(stopped)} failed or uncertain attempt(s). Any usage missing from those attempts is not represented as an exact bill.']
    # A compact figure shows paired differences, not incomparable cross-dataset score rankings.
    effects=[]
    if reasoning: effects.append(('Reasoning development: medium − low',reasoning['primary']))
    if renderer: effects.append(('Rendering development: hollow − filled',renderer['primary']))
    if fresh:
        effects.append(('Confirmation: candidate − logistic',fresh['primary'][0]))
        if not (candidate['effort']=='low' and candidate['renderer']=='standard'): effects.append(('Confirmation: candidate − original low',fresh['primary'][1]))
    if effects:
        fig,ax=plt.subplots(figsize=(10,4.5)); ax.axvline(0,color='#666666',linestyle='--',linewidth=1)
        for i,(name,t) in enumerate(effects):
            value=t['brier_difference']; lo,hi=t['interval']; color='#16806a' if hi<0 else ('#bc4b45' if lo>0 else '#66758a')
            ax.errorbar(value,i,xerr=[[max(0,value-lo)],[max(0,hi-value)]],fmt='o',capsize=5,color=color,linewidth=2)
        ax.set_yticks(range(len(effects)),[x[0] for x in effects]); ax.invert_yaxis(); ax.set_xlabel('Brier difference: negative favors the first method'); ax.grid(axis='x',alpha=.2)
        ax.set_title('Paired effects and whole-match uncertainty'); fig.tight_layout(); fig.savefig(c.OUT/'effect_intervals.png',dpi=170); plt.close(fig)
        lines+=['','![Paired effects and uncertainty](effect_intervals.png)','', 'Development intervals are 95%; the two confirmation intervals are 97.5%. These use different cohorts and answer different comparisons.']
    (c.OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    c.save(c.OUT/'ready_for_review.json',{'complete_stages':[n for n,s in summaries.items() if s], 'successful_generation_calls':sum(a['state']=='completed' for a in ledger),'estimated_usd_including_prior_pilot':cost,'working_cap_usd':c.GLOBAL_CAP,'user_budget_approximately_usd':10})
    print('Human-readable final report saved:',c.OUT/'FINAL_REPORT.md')
if __name__=='__main__': main()


