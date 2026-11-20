"""Write the readable report from the completed, audited saved results."""
import experiment as e

c, OUT = e.c, e.OUT
LABELS = {
    "zero": "Diagram, no examples (fresh)",
    "outcome32": "Diagram + 32 observed-outcome examples",
    "shuffled_outcome32": "Diagram + shuffled outcome labels",
    "teacher32": "Diagram + 32 spatial-teacher examples",
    "shuffled_teacher32": "Diagram + shuffled teacher labels",
    "historical_raw": "Original photograph (previous run, matched shots)",
    "historical_diagram": "Diagram, no examples (previous run, matched shots)",
    "historical_photo_diagram": "Photograph + diagram (previous run, matched shots)",
    "logistic_basic": "Pre-shot distance/angle logistic",
    "logistic_spatial": "Pre-shot spatial logistic",
    "training_constant": "Other matches' training goal rate",
}


def conclusion(primary, arm, shuffled, title):
    direct = primary[arm+"_minus_zero"]["interval"]
    paired = primary[arm+"_minus_"+shuffled]["interval"]
    if direct[1] < 0 and paired[1] < 0:
        text = "supports lower Brier than both no-example diagrams and the shuffled-label control in this cohort."
    elif direct[1] < 0:
        text = "supports lower Brier than no-example diagrams, but does not establish a benefit of correct example/label pairing beyond the shuffled control."
    elif direct[0] > 0:
        text = "supports worse Brier than no-example diagrams in this cohort."
    else:
        text = "does not establish a Brier improvement over no-example diagrams."
    return f"**{title}** {text}"


def main():
    s = c.read(OUT / "summary.json")
    audit = c.read(OUT / "audit.json")
    assert audit["passed"] and not (OUT / "generation.lock").exists()
    table = ["| Method | Brier ↓ | AUC ↑ | Log loss ↓ | Mean xG |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, value in s["metrics"].items():
        table.append(f"| {LABELS[name]} | {value['brier']:.6f} | {value['auc']:.4f} | {value['logloss']:.4f} | {100*value['mean_xg']:.2f}% |")
    comparisons = ["| Prespecified comparison | Brier difference | 98.75% match interval |",
                   "| --- | ---: | --- |"]
    for name, value in s["primary"].items():
        a, b = name.split("_minus_")
        lo, hi = value["interval"]
        comparisons.append(f"| {LABELS[a]} minus {LABELS[b]} | {value['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    conclusion_text = "\n\n".join([
        conclusion(s["primary"], "outcome32", "shuffled_outcome32", "Observed-outcome examples"),
        conclusion(s["primary"], "teacher32", "shuffled_teacher32", "Spatial-teacher examples")])
    goals = ", ".join(f"{mid}: {count}" for mid, count in s["example_goal_counts"].items())
    report = f"""# SoccerTrack v2: player-position diagrams plus in-context learning

**Research summary — 20 November 2026, Pacific time. Completed and audited.**

Tested whether in-context examples add value to the annotation-assisted pitch diagrams that improved the preceding photograph comparison. **In-context learning means providing labelled examples in the model's request while keeping its weights unchanged.** Here, each new diagram was accompanied by 32 other-match diagrams labelled either with observed goal/no-goal outcomes or with probabilities from a spatial logistic teacher. Both mechanisms have matched shuffled-label controls.

{conclusion_text}

## Matched results

All rows use the **same {s['shots']} shots, {s['goals']} goals and {s['matches']} matches**. Actual goal rate: **{100*s['goals']/s['shots']:.2f}%**. Lower Brier/log loss are better; higher AUC indicates better ranking. The model is **Claude Sonnet 5.5, low effort**, for every new visual condition.

{chr(10).join(table)}

The first five rows are the newly generated matched comparison. The three historical image references reuse saved predictions on the identical shots; they are descriptive and were not regenerated or averaged with the new outputs. The fresh no-example diagram requests have exactly the same image bytes, prompt, model and settings as their historical diagram counterparts, so differences also illustrate generation variability. The two logistic models use complete-query-match exclusions, pre-shot timing and training-only scaling. The spatial model receives the same six geometric features used by the teacher.

In this completed run, the fresh no-example diagram condition has the lowest Brier point estimate among the five new visual conditions. Outcome examples worsen both Brier and AUC point estimates. Teacher examples move mean xG closer to the observed rate (10.80% versus 8.92% without examples, against 14.14% observed), but do not lower Brier and reduce AUC from 0.7056 to 0.6842. Every primary interval below includes zero: neither improvement over diagrams alone nor an advantage over the corresponding shuffled control is established. The fresh no-example Brier (0.114746) is close to the preceding diagram score (0.114825); this is a descriptive consistency observation, not an equivalence test.

The teacher is a fitted logistic model, not a source of ground-truth xG. Its lower or higher point Brier cannot by itself establish that a VLM imitates it successfully or beats it reliably. No VLM-versus-logistic significance test was prespecified. No predicted probability was recalibrated, blended, averaged or replaced by a statistical output.

## Four prespecified comparisons

Differences are the first method's Brier minus the second method's Brier; negative favors the first. Each interval resamples complete matches with paired predictions, 20,000 draws and 98.75% confidence. The four intervals use Bonferroni adjustment for a nominal 5% family error rate.

{chr(10).join(comparisons)}

Beating the no-example condition assesses improvement from the complete ICL intervention. Beating the corresponding shuffled-label control additionally tests whether correct diagram/label associations matter beyond the label distribution and example presentation. A point gain or a lower mean prediction alone does not establish either claim. Intervals include only ten match groups and condition on the fitted teacher models and selected example sets; overlapping other-match fits are not refitted within bootstrap draws.

## Inputs and learning mechanisms

- **Query:** the unchanged 640×960 pre-shot player-position diagram from the preceding spatial study. Gold star identifies the shooter; blue circles teammates, red circles opponents and orange square the opposing goalkeeper. The attacking goal is at the top. The source frame is approximately 0.32 seconds before the event.
- **Outcome32:** 32 natural random diagrams from the other nine matches, paired with their single observed GOAL or NO GOAL outcome. The prompt explicitly explains that binary results are noisy observations, not true probabilities.
- **Shuffled outcome32:** the identical images in the identical order, with the outcome labels permuted while preserving the number of goals.
- **Teacher32:** the same 32 diagrams, paired with four-decimal teacher probabilities. The teacher is a ridge-1 logistic model using pre-shot distance, angle, goalkeeper distance from the goal line, keeper lateral error, defenders in the shooting triangle and nearest defender distance. Its coefficients and scaling exclude the query's entire match. Its probabilities on the training examples serve as labels; the query's teacher probability is never supplied.
- **Shuffled teacher32:** the identical example images and order, with the complete rounded teacher-value distribution permuted.
- **No examples:** a fresh prediction using only the query diagram and the preceding spatial study's unchanged prompt.

Each match has a fixed, seeded set of 32 examples, without goal balancing or outcome-driven example selection. Goal counts by excluded query match: **{goals}**. None of these sets contains only one class. A query's entire match is excluded from both its examples and teacher fitting; other matches can appear in different evaluation folds, as in cross-validation.

Example diagrams are resized to 320×480 and placed four per 640×1008 JPEG sheet, with 17-pixel labels. Eight sheets precede the separate full-size query. They are smaller than the query but retain explicit shooter and team/keeper markers. No photographs, post-shot frames, auxiliary interpolated ball track, query outcomes, body-part labels, numeric coordinates, player names, reference xG or query teacher probabilities appear in new requests. Teacher examples are explicitly distinguished from observed-outcome learning.

![Fixed query and first sheets of outcome and teacher examples](input_preview.png)

The illustration uses the first source shot ID in sorted order, independent of prediction quality or outcome. It shows one sheet from each mechanism; actual ICL requests include all eight sheets. The model's weights and low-effort settings remain unchanged, with structured JSON output and a 1,024-token maximum.

## Budget-defined cohort and execution

The maximum pool was the preceding study's 304 source-eligible shots. A seeded natural random order within matches, interleaved across a seeded match order, defined the only allowable prefix. Before any accuracy scoring, the frozen rule reserved the **maximum uncached or five-minute-cache-write input cost plus maximum output cost for every request in a complete five-condition block**. At each step, the largest next block of up to ten complete shots was allowed only if its full reservation fit the remaining $10. No incomplete shot was started just to use the remaining balance. Cache savings could extend the prefix but could not change the conditions, examples or order.

Terminal result: **{s['terminal']['reason']}**, with **{s['shots']} complete paired shots**. The final cohort was determined by usage and the predeclared stopping rule, not by looking at accuracy. This is a resource-selected exploratory cohort rather than a fixed-N independent trial. No method, prompt, example count or later expansion was chosen using these scores.

Explicit five-minute caching applies only to the final historical-example sheet and its prefix. It does not cache query answers as demonstrations or carry previous query responses into later requests. Each request has one independent user message. Caching changes computation reuse and cost, not the intended input content.

**{s['calls']:,} successful generation requests**, zero failed, retried or uncertain requests. New estimated spending: **${s['estimated_usd']:.6f}**; conservative accounting: **${s['conservative_usd']:.6f}**, under the **$10 incremental experiment cap**. Previous completed studies' charges are separate. No paid runner remains active. These are usage-based estimates, not a live provider invoice.

Recorded tokens: {s['usage']['input_tokens']:,} ordinary input; {s['usage']['cache_creation_input_tokens']:,} cache writes; {s['usage']['cache_read_input_tokens']:,} cache reads; {s['usage']['output_tokens']:,} output. Current model rates were verified against [official model documentation](https://platform.claude.com/docs/en/models/sonnet-5-5/overview): $2/M ordinary input, $2.50/M five-minute writes, $0.10/M reads and $10/M output. Conservative completed accounting charges ordinary input at $2.50/M and reads at $0.125/M, retains $2.50/M writes and $10/M output, and reserves pending requests at the full uncached/write bound. Savings are counted only after the provider returns usage. One-hour caching is disabled.

## Verification and limits

The final audit passed **{audit['check_count']} checks**, reconstructing every one of the **{s['calls']:,} request hashes and raw responses**, all **320 example sheets**, exact source-image hashes, complete-match teacher exclusions, fixed label permutations, the paired prefix and block reservations, token bounds, scores, adjusted intervals and spending. Seven financial checks and two full-budget simulations also passed, including a run with no cache hits. The new audit inherits the preceding frozen 21-check source audit; it does not claim to have downloaded or verified complete large annotation files.

The released positions, roles, actor identity, camera resources and pre-shot diagrams remain **annotation-assisted**. The earlier geometry exclusions remain: 118576_2_41455 and 128057_2_46854 had the shooter at or beyond the goal line, leaving 304 shots and 43 goals from the initial 306 shots and 44 goals. Small camera files were fully hash-verified in the source study; the large GSR files were retrieved as validated ranges. Whole large-file SHA256 verification is not claimed. The current ICL requests reuse diagrams rather than re-running camera projection or automatic detection.

All ten university matches were studied previously. This is exploratory match-held-out evaluation, not untouched external confirmation. Supplied annotations may involve offline reconstruction and do not prove a causal real-time tracking pipeline. The professional-finisher assumption remains a mismatch with university footage, and the source event selection can omit blocked/header attempts assigned other event classes. Natural example outcomes are noisy; teacher labels inherit model assumptions and training error. Example and training sets overlap across folds. Conclusions apply to this model, representation, example count and cohort; they do not establish that all forms of ICL help or fail.

## Saved outputs

- [Per-shot probabilities](predictions.csv)
- [Scores, intervals, usage and terminal result](summary.json)
- [Frozen protocol and dependency hashes](protocol.json)
- [Final audit](audit.json)
- [Example identities and labels](examples.json)
- [Complete-block reservation ledger](blocks.json)
- [Code and reproduction notes](../../experiments/soccertrack_spatial_icl/README.md)
- [Preceding spatial-information experiment](../soccertrack_spatial_v1/REPORT.md)

SoccerTrack v2 by Atom Scott, Ikuma Uchida, Kento Kuroda, Yufi Kim and Keisuke Fujii. [Official dataset](https://huggingface.co/datasets/atomscott/soccertrack-v2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Example sheets and diagrams are derived from the dataset's released annotations. Source attribution and retrieval provenance are retained in the preceding experiment.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(str(OUT / "REPORT.md"), flush=True)


if __name__ == "__main__":
    main()
