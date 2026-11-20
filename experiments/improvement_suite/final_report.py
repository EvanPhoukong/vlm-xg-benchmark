"""Build a complete report only after all fixed experiments have finished."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vision_trials import ROOT, OUT

LABELS = {
 'original_hybrid': 'Original VLM hybrid', 'noise_trained': 'Noise-trained VLM hybrid',
 'noise_training_image_only': 'Noise-trained VLM hybrid', 'best_vision': 'Selected best VLM pipeline',
 'training_rate_constant': 'Fixed training-rate probability',
 'noise_head_known_positions': 'Noise-trained rich model, known positions',
 'original_direct': 'Original position-first Luna xG', 'multiview_direct': 'Multiple-reading direct Luna xG',
 'fewshot_direct': 'Four-example direct Luna xG',
 'multiview': 'Multiple readings', 'multiview_noise_head': 'Multiple readings + noise training',
 'fewshot': 'Four worked examples', 'fewshot_noise_head': 'Worked examples + noise training',
 'pixels': 'Pixel reader (computer vision)', 'pixels_noise_head': 'Pixel reader + noise training',
 'hybrid_plus_context': 'VLM hybrid + pre-shot context',
 'noise_training_plus_context': 'Noise training + pre-shot context',
 'baseline_image': 'Simple baseline, known positions', 'sb_xg': 'StatsBomb',
 'simple_known_positions': 'Simple baseline, known positions',
 'rich_known_positions': 'Rich model, known positions', 'direct_luna': 'Direct Luna',
 'noise_plus_context': 'Noise training + pre-shot context',
 'simple_plus_context_known_positions': 'Simple baseline + same context',
 'original_rich': 'Rich model, known positions', 'rich_plus_context': 'Rich model + context, known positions',
 'simple_plus_context': 'Simple baseline + context, known positions'}


def table(scores, order):
    indexed = scores.set_index('method')
    lines = ['| Method | Brier ↓ | AUC ↑ | Mean xG | Log loss ↓ |', '|---|---:|---:|---:|---:|']
    for name in order:
        if name == 'best_vision' and name not in indexed.index:
            continue
        row = indexed.loc[name]
        log = row['log_loss'] if 'log_loss' in row else row['logloss']
        lines.append(f'| {LABELS.get(name, name)} | {row.brier:.5f} | {row.auc:.3f} | {row.mean_xg:.1%} | {log:.5f} |')
    return lines


def interval(row, adjusted=False):
    lo, hi = ('ci_97_5_low', 'ci_97_5_high') if adjusted else ('ci_low', 'ci_high')
    return f'{row.brier_difference:+.5f} [{row[lo]:+.5f}, {row[hi]:+.5f}]'


def usage():
    paths = [('Additional full/zoom readings', OUT/'vision/second_cache.jsonl'),
             ('Third disagreement checks', OUT/'vision/third_cache.jsonl'),
             ('Four-example prompts', OUT/'vision/fewshot_cache.jsonl'),
             ('Fresh confirmation readings', OUT/'confirmation_bundesliga/inference/baseline_cache.jsonl'),
             ('Best-VLM fresh example prompts', OUT/'confirmation_bundesliga/best_vision/best_fewshot_cache.jsonl'),
             ('Best-VLM fresh second readings', OUT/'confirmation_bundesliga/best_vision/best_second_cache.jsonl'),
             ('Best-VLM fresh third checks', OUT/'confirmation_bundesliga/best_vision/best_third_cache.jsonl')]
    rows = []
    for name, path in paths:
        entries = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        rows.append({'condition': name, 'successful_requests': len(entries),
                     'input_tokens': sum((e.get('usage') or {}).get('input_tokens', 0) for e in entries),
                     'cached_input_tokens': sum(((e.get('usage') or {}).get('input_tokens_details') or {}).get('cached_tokens', 0) for e in entries),
                     'output_tokens': sum((e.get('usage') or {}).get('output_tokens', 0) for e in entries)})
    return pd.DataFrame(rows)


def score_plot(screen, fresh):
    orders = [['original_hybrid', 'noise_trained', 'multiview', 'multiview_noise_head', 'fewshot', 'fewshot_noise_head', 'pixels_noise_head', 'training_rate_constant'],
              ['original_hybrid', 'noise_trained', 'best_vision', 'pixels_noise_head', 'simple_known_positions', 'sb_xg', 'training_rate_constant']]
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 10.5))
    fig.subplots_adjust(left=.36, right=.97, top=.94, bottom=.13, hspace=.56)
    for ax, frame, order, title in zip(axes, [screen, fresh], orders,
            ['Screening: 200 previously studied Euro shots', 'Confirmation: 200 new Bundesliga shots']):
        indexed = frame.set_index('method')
        order = [name for name in order if name in indexed.index]
        values = [indexed.loc[n, 'brier'] for n in order]
        colors = ['#7c3c8c' if n.startswith('pixels') else '#167f8b' for n in order]
        ax.scatter(values, np.arange(len(order)), s=65, c=colors)
        ax.set_yticks(np.arange(len(order)), [LABELS.get(n, n) for n in order], fontsize=9)
        ax.invert_yaxis(); ax.set_xlim(min(values)-.0015, max(values)+.003)
        for i, value in enumerate(values):
            ax.annotate(f'{value:.5f}', (value, i), xytext=(7, 0), textcoords='offset points', va='center', fontsize=9)
        ax.set_title(title, fontsize=11, loc='left', pad=17, fontweight='bold')
        ax.set_xlabel('Brier score - lower is better', fontsize=10)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0); ax.grid(axis='x', alpha=.18)
    fig.text(.03, .055, 'Each panel compares identical shots. Scores across cohorts are not directly comparable.', fontsize=10, color='#555555')
    fig.text(.03, .03, 'Purple marks separate computer-vision controls.', fontsize=10, color='#555555')
    fig.savefig(OUT/'comparison.png', dpi=170); plt.close(fig)


def forest(screen_comparisons, fresh_comparisons):
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 9.2), gridspec_kw={'height_ratios': [5, 2]})
    fig.subplots_adjust(left=.36, right=.97, top=.93, bottom=.15, hspace=.67)
    for ax, frame, adjusted, title in [(axes[0], screen_comparisons, False, 'Screening effects vs original hybrid (95%)'),
                                      (axes[1], fresh_comparisons.iloc[:2], True, 'Two primary fresh comparisons (97.5%)')]:
        labels = []
        for i, row in frame.reset_index(drop=True).iterrows():
            low, high = (row.ci_97_5_low, row.ci_97_5_high) if adjusted else (row.ci_low, row.ci_high)
            ax.errorbar(row.brier_difference, i, xerr=[[row.brier_difference-low], [high-row.brier_difference]],
                        fmt='o', color='#197c87', capsize=4)
            labels.append(LABELS.get(row['first'], row['first'])+('\nvs simple baseline' if adjusted and i == 1 else ''))
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=9)
        ax.invert_yaxis(); ax.axvline(0, color='#777777', linestyle='--', lw=1)
        ax.set_xlabel('Brier difference: negative favors the new method', fontsize=10)
        ax.set_title(title, fontsize=11, loc='left', pad=15, fontweight='bold')
        ax.spines[['top', 'right', 'left']].set_visible(False); ax.tick_params(axis='y', length=0)
    fig.text(.03, .06, 'Whole-match bootstrap, 4,000 resamples. Intervals condition on fitted models.', fontsize=10, color='#555555')
    fig.text(.03, .035, 'Screening intervals are exploratory and unadjusted; fresh intervals adjust two primary comparisons.', fontsize=10, color='#555555')
    fig.savefig(OUT/'uncertainty.png', dpi=170); plt.close(fig)


def main():
    required = [OUT/'vision/multiview_comparisons.csv', OUT/'vision/fewshot_comparisons.csv', OUT/'confirmation_bundesliga/comparisons.csv']
    if any(not p.exists() for p in required):
        raise SystemExit('All fixed trials and fresh confirmation must finish before the final report.')
    import score_sanity_references
    score_sanity_references.main()
    original = pd.read_csv(OUT.parent/'noise-robustness-v1/scores.csv')
    screen = original[original.method.isin(['original_hybrid', 'noise_trained'])].copy()
    for condition, names in [('multiview', ['multiview', 'multiview_noise_head']), ('fewshot', ['fewshot', 'fewshot_noise_head'])]:
        frame = pd.read_csv(OUT/'vision'/f'{condition}_scores.csv')
        screen = pd.concat([screen, frame[frame.method.isin(names)]], ignore_index=True)
    pixel = pd.read_csv(OUT/'pixels/pixels_scores.csv')
    screen = pd.concat([screen, pixel[pixel.method.isin(['pixels', 'pixels_noise_head', 'baseline_image', 'sb_xg'])]], ignore_index=True)
    context = pd.read_csv(OUT/'context/scores.csv')
    screen = pd.concat([screen, context[context.method.isin(['hybrid_plus_context', 'noise_training_plus_context'])]], ignore_index=True)
    screen = pd.concat([screen, pd.read_csv(OUT/'sanity_baselines/screening_scores.csv')], ignore_index=True)
    screen.to_csv(OUT/'all_screening_scores.csv', index=False)
    best_complete = (OUT/'confirmation_bundesliga/best_vision/comparisons.csv').exists()
    if not best_complete and not (OUT/'confirmation_bundesliga/best_vision/partial_status.json').exists():
        raise ValueError('The secondary confirmation needs a complete result or an explicit incomplete status.')
    fresh_parts = [pd.read_csv(OUT/'confirmation_bundesliga/scores.csv'), pd.read_csv(OUT/'sanity_baselines/fresh_scores.csv')]
    if best_complete:
        fresh_parts.append(pd.read_csv(OUT/'confirmation_bundesliga/best_vision/scores.csv'))
    fresh = pd.concat(fresh_parts, ignore_index=True)
    best_protocol = json.loads((OUT/'confirmation_bundesliga/best_vision/protocol.json').read_text())
    best_fresh_comparisons = pd.read_csv(OUT/'confirmation_bundesliga/best_vision/comparisons.csv') if best_complete else pd.DataFrame()
    partial = {} if best_complete else json.loads((OUT/'confirmation_bundesliga/best_vision/partial_status.json').read_text())
    secondary_status = 'The selected best VLM pipeline completed its secondary prospective check.' if best_complete else f'The extra confirmation of the best screening VLM pipeline stopped at **{partial["completed_shots"]}/200 shots because API credits ran out**. Its partial sample is not scored as a finished result.'
    fcomp = pd.read_csv(OUT/'confirmation_bundesliga/comparisons.csv')
    fdata = pd.read_csv(OUT/'confirmation_bundesliga/predictions.csv')
    comparisons = []
    for path, first in [(OUT.parent/'noise-robustness-v1/comparisons.csv', 'noise_trained'),
                        (OUT/'vision/multiview_comparisons.csv', 'multiview'),
                        (OUT/'vision/fewshot_comparisons.csv', 'fewshot'),
                        (OUT/'pixels/pixels_comparisons.csv', 'pixels'),
                        (OUT/'context/comparisons.csv', 'hybrid_plus_context')]:
        frame = pd.read_csv(path)
        comparisons.append(frame[(frame['first'] == first) & (frame['second'] == 'original_hybrid')].iloc[0])
    scomp = pd.DataFrame(comparisons); scomp.to_csv(OUT/'screening_primary_comparisons.csv', index=False)
    calls = usage(); calls.to_csv(OUT/'api_usage.csv', index=False)
    if not (calls.successful_requests.iloc[[0, 2, 3]] == 200).all():
        raise ValueError('An expected full 200-shot request condition is incomplete.')
    best_vlm = screen[screen.method.isin(['noise_trained', 'multiview', 'multiview_noise_head', 'fewshot', 'fewshot_noise_head'])].sort_values('brier').iloc[0]
    best_pixel = pixel[pixel.method.isin(['pixels', 'pixels_noise_head'])].sort_values('brier').iloc[0]
    score_plot(screen, fresh); forest(scomp, fcomp)
    geometry = pd.read_csv(OUT/'geometry_summary.csv')
    constant_screen = float(screen.set_index('method').loc['training_rate_constant', 'brier'])
    constant_fresh = float(fresh.set_index('method').loc['training_rate_constant', 'brier'])
    nrow = fcomp.iloc[0]; prow = fcomp.iloc[1]
    noise_verdict = 'favored the noise-trained hybrid after the two-comparison adjustment' if nrow.ci_97_5_high < 0 else (
        'favored the original hybrid after adjustment' if nrow.ci_97_5_low > 0 else 'did not establish a difference after the two-comparison adjustment')
    pixel_verdict = 'favored the pixel/noise method over the simple baseline after adjustment' if prow.ci_97_5_high < 0 else (
        'favored the simple baseline after adjustment' if prow.ci_97_5_low > 0 else 'did not establish a pixel/noise advantage over the simple baseline after adjustment')
    lines = ['# xG research: five improvement experiments', '', 'Research date: 22 October 2026 (America/Los_Angeles).', '',
             '## Findings', '',
             f'All five ideas were implemented and screened on 200 shots. The lowest screening Brier among the new image-only VLM pipelines was **{LABELS[best_vlm.method]}: {best_vlm.brier:.5f}**. The separate pixel control reached **{best_pixel.brier:.5f}**. These screening scores reuse the same 200 previously studied Euro 2024 shots and are not independent confirmation.', '',
             f'On 200 newly reserved Bundesliga shots, the primary comparison **{noise_verdict}**. The other primary comparison **{pixel_verdict}**. Exact scores and adjusted intervals appear below. Pixel results are a computer-vision control; extra-context results use more information than the image-only methods.', '',
             secondary_status, '',
             f'A constant probability learned only from the training scoring rate scores **{constant_screen:.5f}** on the Euro screen and **{constant_fresh:.5f}** on the fresh cohort. This is an important reference: improving on the logistic baseline alone does not guarantee useful image-based prediction. Its probability was fixed at 287/3,068 (9.35%) before fresh results, rather than fitted to a test scoring rate.', '',
             '![Screening and fresh-data results](comparison.png)', '',
             'Brier measures probability error and is the primary score (lower is better). AUC measures how well good chances rank above poor ones (higher is better). Mean xG is the average predicted scoring rate, not an accuracy score. Log loss penalizes confident mistakes (lower is better).', '',
             '## 1. Screening: same 200 Euro 2024 shots', '',
             'There are 16 goals across 50 matches (8.0% scoring rate). All methods retain the same shots. World Cups 2018 and 2022 supply the 3,068 training shots, 287 goals and 128 matches. The earlier 300-shot World Cup results are preserved and are not directly compared against this different cohort.', '',
             '### Image-only VLM pipelines', '']
    lines += table(screen, ['original_hybrid', 'noise_trained', 'multiview', 'multiview_noise_head', 'fewshot', 'fewshot_noise_head', 'baseline_image', 'sb_xg', 'training_rate_constant'])
    lines += ['', 'The practical simple baseline uses known player positions. Both VLM hybrids use measured positions only. The same-input simple baseline using Luna positions scored Brier 0.07638; the richer known-coordinate control scored 0.07297. Neither is substituted for a VLM measurement.', '',
              '### Separate pixel controls', '']
    lines += table(screen, ['pixels', 'pixels_noise_head', 'baseline_image', 'sb_xg'])
    lines += ['', 'The pixel methods exploit the fixed diagram colors and layout. They are useful task-specific image baselines and do not demonstrate that a VLM understands football or general broadcast images.', '',
              '### Extra pre-shot context', '']
    lines += table(context, ['original_hybrid', 'hybrid_plus_context', 'noise_training_image_only', 'noise_training_plus_context', 'simple_plus_context_known_positions'])
    lines += ['', 'Context increased the ranking score on the 200-shot screen but worsened Brier. It is not counted as a probability-accuracy win. The comparator with context receives the same event information. No body-part flag is added in this experiment.', '',
              '### Paired screening uncertainty', '', '| New method vs original hybrid | Brier difference [95% interval] |', '|---|---:|']
    for row in scomp.itertuples(index=False):
        lines.append(f'| {LABELS[row.first]} | {interval(pd.Series(row._asdict()))} |')
    lines += ['', '![Paired effects and uncertainty](uncertainty.png)', '',
              'Negative differences favor the new method. These screening intervals resample entire matches and condition on fitted models. They are exploratory and unadjusted for trying several ideas or selecting the best one. Screening controls reuse older API responses; prompt conditions were not randomized in time, so service variability is another possible contributor to prompt differences.', '',
              '## 2. Fresh confirmation: Bundesliga 2023/24', '',
              f'The fixed sample contains **200 shots, {int(fdata.goal.sum())} goals, {fdata.match_id.nunique()} represented matches**, with an actual scoring rate of **{fdata.goal.mean():.1%}**. It comes from all 34 publicly available Leverkusen fixtures, including both teams\' shots. This is not a full-league random sample. None of these matches trained the models or appeared in earlier Euro/World Cup analyses.', '',
              'The noise-trained model, pixel algorithm, selection seed and primary comparisons were frozen before this cohort was scored. The unchanged single-panel rendering isolates model robustness from the unfinished earlier rendering revision. Original and noise-trained VLM methods share the same one-reading-per-image measurements.', '']
    lines += table(fresh, ['original_hybrid', 'noise_trained', 'best_vision', 'pixels', 'pixels_noise_head', 'simple_known_positions', 'rich_known_positions', 'sb_xg', 'direct_luna', 'training_rate_constant'])
    lines += ['', '| Predeclared primary comparison | Brier difference [97.5% interval] |', '|---|---:|']
    for _, row in fcomp.iloc[:2].iterrows():
        lines.append(f'| {LABELS.get(row["first"], row["first"])} vs {LABELS.get(row["second"], row["second"])} | {interval(row, adjusted=True)} |')
    lines += ['', 'The two primary tests use 97.5% intervals (Bonferroni adjustment for two comparisons). Secondary 95% and 97.5% comparisons are saved in confirmation_bundesliga/comparisons.csv. The fresh sample is still small, comes from one club\'s fixtures and conditions intervals on already-fitted models. Results should not be generalized to every competition.', '',
              'Extra-context fresh results, reported separately:', '']
    lines += table(fresh, ['noise_trained', 'noise_plus_context', 'simple_plus_context_known_positions'])
    lines += ['', f'The strongest screening image-only VLM pipeline was also selected and frozen before any fresh baseline scores were released: **{LABELS[best_protocol["chosen_method"]]}**.', '']
    if best_complete:
        lines += ['It received a separate secondary prospective check on the same fresh shots. The table below uses 98.333% intervals for its three secondary comparisons, separately from the two original primary comparisons.', '',
                  '| Selected best VLM vs | Brier difference [98.333% interval] |', '|---|---:|']
        for _, row in best_fresh_comparisons.iterrows():
            lines.append(f'| {LABELS.get(row["second"], row["second"])} | {row.brier_difference:+.5f} [{row.ci_98_333_low:+.5f}, {row.ci_98_333_high:+.5f}] |')
    else:
        lines += [f'**This extra confirmation is incomplete: {partial["completed_shots"]}/200 shots.** The API returned `credit_balance_exhausted`. Its successful readings are preserved, but no partial-cohort Brier/AUC or final intervals are reported. It is not a completed confirmation of the best screening pipeline. Resume the unchanged runner after API credits are available.', '']
    lines += ['', 'Unselected worked-example or multiple-reading variants remain screening results. The best-candidate check uses the same fresh sample; it is secondary and should not be presented as an additional independent dataset.', '',
              '### Full fresh source pool: known-coordinate controls', '',
              'This secondary analysis uses all **908 eligible shots, 102 goals and 34 matches** in the downloaded Leverkusen fixture pool. It includes the 200-shot VLM subset, so it is not a second independent dataset. It was added after the first 200-shot fresh scores were viewed, using already-frozen models and no additional fitting or model selection.', '']
    full_scores = pd.read_csv(OUT/'confirmation_bundesliga/full_pool_reference/scores.csv')
    lines += table(full_scores, ['simple_known_positions', 'rich_known_positions', 'noise_head_known_positions', 'rich_plus_context', 'sb_xg', 'training_rate_constant'])
    full_comp = pd.read_csv(OUT/'confirmation_bundesliga/full_pool_reference/comparisons.csv').iloc[0]
    lines += ['', f'Rich geometry minus the simple known-coordinate baseline: **{interval(full_comp)}**, an unadjusted secondary 95% interval. This supports the statistical value of richer geometry in this pool. These measurements came from the dataset, and do not establish a VLM improvement. Commercial xG is stronger by point estimate on the full pool, despite its weaker point estimate on the small 200-shot subset.', '',
              '## 3. What each experiment actually did', '',
              '### Imperfect-position training', '',
              'The VLM was unchanged. Each training shot became one clean and fifteen corrupted scenes. Shooter and goalkeeper residuals were sampled together from 300 saved training-only World Cup readings, preserving rare large errors. Defender jitter (0.25 pitch units) and omission probability (5%) were fixed assumptions. Geometric features were recalculated from corrupted scenes. Total weight remained one per original shot; augmentation did not create 49,088 independent observations or silently weaken the loss penalty.', '',
              'Whole-match cross-validation excluded test matches from both fitting and noise donors. On simulated noisy development inputs, Brier changed from 0.07303 to 0.07272; on clean inputs it changed from 0.07190 to 0.07210. This supports a robustness tradeoff under the simulation, not a universal improvement.', '',
              '### Multiple readings', '',
              f'Each saved original reading was paired with one new reading of the original image plus a fixed enlarged crop. A third reading was requested for more than two pitch units of disagreement, different visibility, or different defender counts: **{int(calls.iloc[1].successful_requests)} of 200 shots**. Coordinates were merged using majority visibility, medians, and matched defender lists. No true query coordinates or goal outcomes entered these rules.', '',
              '### Four worked examples', '',
              'Four World Cup 2018 examples were selected by fixed distance strata, without using outcomes or earlier errors. Their image geometry labels precede the query image. Neither example outcomes, commercial xG nor the new shot\'s coordinates enter the prompt. This is few-shot prompting, not VLM fine-tuning, and it must be distinguished from the original zero-shot benchmark.', '',
              '### Pixel reader', '',
              'Fixed color thresholds locate gold, orange and red components. Centroids are mapped to pitch coordinates using the same renderer applied to one generic calibration scene. Forty training images checked the mapping before external scoring. Fully hidden markers cannot be recovered. The detector has knowledge of the diagram design but never reads the source coordinates for a query shot.', '',
              '| Measurement condition | Mean shooter error | Shooter within 2 units | Defender count MAE | Defender recall within 2 units |', '|---|---:|---:|---:|---:|']
    for row in geometry.itertuples():
        lines.append(f'| {row.method} | {row.shooter_mean_error_returned:.3f} | {row.shooter_within_2_rate_all:.1%} | {row.defender_count_mae:.3f} | {row.defender_recall_within_2:.1%} |')
    lines += ['', 'Pitch units are normalized StatsBomb coordinates, not measured meters. Coverage and full per-shot measurement errors are saved in geometry_audit.csv. Measurements are diagnostic: they are not goal-prediction scores.', '',
              '### Pre-shot context', '',
              'The model adds shot type, play pattern, pressure and the latest earlier same-team pass in the same possession and period, at most 15 seconds before the shot. Pass features include cross, cutback, through ball, height and elapsed time. Outcomes, commercial xG, player identity and subsequent events are excluded. A one-hot/numeric context transform is fitted on training matches only.', '',
              'Separate full-tournament known-coordinate controls help show the difference between context quality and noisy visual inputs:', '']
    for tournament, count in [('euro2024', 1304), ('euro2020', 1234)]:
        frame = pd.read_csv(OUT/'context'/f'{tournament}_known_scores.csv')
        comp = pd.read_csv(OUT/'context'/f'{tournament}_known_comparison.csv').iloc[0]
        lines += ['', f'**{tournament}, {count} shots; known-coordinate controls:**', '']
        lines += table(frame, ['original_rich', 'rich_plus_context', 'simple_plus_context', 'sb_xg'])
        lines += ['', f'Context minus original rich model: {interval(comp)}. These reused tournament results are secondary, and the intervals include no improvement.', '']
    lines += ['### Secondary direct-probability outputs', '',
              'The same requests also returned Luna\'s own xG. These were not used to fit the statistical heads or select geometry-consensus rules. They are secondary exploratory prompt outputs, separately reported from the image-to-geometry hybrids.', '']
    lines += table(pd.read_csv(OUT/'vision/direct_scores.csv'), ['original_direct', 'multiview_direct', 'fewshot_direct'])
    lines += ['', '## 4. Validation, API usage and reproducibility', '',
              'Sixteen targeted checks passed across the new experiments: split isolation, no outcomes in geometry, correct weighting of duplicate scenes, consensus alignment/visibility, no future-event context, pixel-coordinate mapping, context joins and wider multiplicity-adjusted intervals. The original hybrid reproduced saved predictions within 9.9e-17. Old protected datasets, caches and probability models were preserved.', '',
              '| Request condition | Successful new requests | Input tokens | Cached input tokens | Output tokens |', '|---|---:|---:|---:|---:|']
    for row in calls.itertuples():
        lines.append(f'| {row.condition} | {row.successful_requests:,} | {row.input_tokens:,} | {row.cached_input_tokens:,} | {row.output_tokens:,} |')
    lines += ['', f'Total successful new requests: **{int(calls.successful_requests.sum()):,}**. Requests ran sequentially. Cache counts do not measure unsuccessful network attempts or dollar charges. Noise training, pixel inference and statistical context fitting require no model API calls. Public StatsBomb event downloads are separate from model requests.', '',
              'Artifacts include fixed protocols and hashes, saved successful responses, frozen statistical models, per-shot predictions, score tables, whole-match intervals, fold audits and public-data source hashes. The earlier clearer-image Euro 2020 run remains stopped at 62/200 responses and is not reported as a completed result.', '',
              'One final worker was restarted after a stalled connection. Its successful cached responses were reused with the unchanged protocol. A preserved selection record and restart log document that the selected VLM candidate had been frozen before the first fresh scores were released.', '',
              '## 5. Interpretation and practical next step', '',
              f'The fresh noise comparison {noise_verdict}; the fresh pixel comparison {pixel_verdict}. {secondary_status} A low screening point estimate alone is not an established advantage. Multiple readings plus noise training remains the most promising tested VLM direction, and needs its prospective check completed before claiming a reliable gain.', '',
              'The work separates three questions: whether images can be measured accurately, whether the statistical head tolerates measurement errors, and whether additional event context improves probabilities. AUC increases alone do not override worse Brier. Matching mean xG to the observed scoring rate is also insufficient to establish accurate individual predictions.', '',
              'Further training or prompt choices should use development matches, with a new untouched confirmation set for any revised method. Preserve these negative and positive attempts rather than choosing whichever tournament or metric makes a method appear strongest.', '',
              '## Sources', '',
              '- [StatsBomb open data](https://github.com/statsbomb/open-data): event files and competition metadata; individual file hashes are saved.',
              '- [Training with Noise is Equivalent to Tikhonov Regularization](https://direct.mit.edu/neco/article-pdf/7/1/108/812990/neco.1995.7.1.108.pdf): general motivation for noisy-input training; does not establish success for this xG pipeline.',
              '- [Efficient Test-Time Scaling for Small Vision-Language Models](https://arxiv.org/abs/2510.03574): motivation for multiple augmented views; our coordinate consensus is a separate implementation.',
              '- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198): precedent for image examples in context; not evidence that our four-example prompt must improve.',
              '- [Official OpenAI image-input documentation](https://developers.openai.com/api/docs/guides/images-vision?api-mode=responses): request format used for multiple images. The OpenAI Docs skill was used to check this format.', '']
    (OUT/'final_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Final report: {OUT/"final_report.md"}', flush=True)


if __name__ == '__main__':
    main()
