"""Audit both completed studies and write a readable combined report. Offline only."""
import hashlib
import json
import math

import numpy as np
from PIL import Image,ImageChops

import audit as a
import experiment as e
import teacher as t


def interval(rows,p,q,seed,confidence=.9875):
    y=np.array([r["goal"] for r in rows]);delta=(np.array(p)-y)**2-(np.array(q)-y)**2
    mids=sorted({r["match_id"] for r in rows})
    sums=np.array([delta[[i for i,r in enumerate(rows) if r["match_id"]==m]].sum() for m in mids])
    counts=np.array([sum(r["match_id"]==m for r in rows) for m in mids])
    draws=np.random.default_rng(seed).integers(len(mids),size=(20000,len(mids)))
    boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    tail=(1-confidence)/2
    return {"difference":float(delta.mean()),"confidence":confidence,"interval":np.quantile(boot,[tail,1-tail]).tolist(),"matches":len(mids)}


def audit_teacher():
    p=e.read(t.OUT/"protocol.json");ledger=e.read(t.OUT/"attempts.json");rows=e.read(t.OUT/"evaluation/selection.json")
    pages=e.read(t.OUT/"pages.json");examples=e.read(t.OUT/"examples/manifest.json");checks=[]
    def check(name,valid):
        if not valid:raise ValueError("Teacher audit failed: "+name)
        checks.append(name)
    check("parent ledger unchanged",e.sha(e.OUT/"attempts.json")==p["parent_ledger_sha256"])
    check("parent protocol unchanged",e.sha(e.OUT/"protocol.json")==p["parent_protocol_sha256"])
    for field,rel in [("pages_sha256","pages.json"),("examples_sha256","examples/manifest.json"),("pool_sha256","pool.json"),("declaration_sha256","declaration.json")]:
        check(rel+" frozen",e.sha(t.OUT/rel)==p[field])
    check("teacher Python frozen",e.sha(t.__file__)==p["python_sha256"])
    check("teacher runner frozen",e.sha(e.ROOT/"experiments/statsbomb_icl/teacher_run.mjs")==p["runner_sha256"])
    source={r["shot_id"]:r for r in e.read(e.SOURCE/"selected.json")["train"]}
    check("labels from training StatsBomb reference only",all(r["shot_id"] in source and round(source[r["shot_id"]]["statsbomb_xg"],4)==r["teacher"] for r in examples))
    check("shuffled teacher distribution preserved",sorted(r["teacher"] for r in examples)==sorted(r["shuffled_teacher"] for r in examples))
    check("shuffled teacher associations changed",any(r["teacher"]!=r["shuffled_teacher"] for r in examples))
    check("query and example matches disjoint",not {r["match_id"] for r in rows}&{r["match_id"] for r in examples})
    check("same original held-out prefix",rows==e.read(e.OUT/"confirmation/selection.json")[:len(rows)])
    check("fixed secondary schedule",[r["job"] for r in ledger]==e.schedule(rows,t.ARMS,e.SEED+82))
    check("all secondary calls completed",all(r["state"]=="completed" for r in ledger))
    for correct,shuffled in zip(pages["teacher64"],pages["shuffled_teacher64"]):
        with Image.open(e.ROOT/correct["path"]) as im1,Image.open(e.ROOT/shuffled["path"]) as im2:
            for y in range(0,864,216):assert ImageChops.difference(im1.crop((0,y,768,y+192)),im2.crop((0,y,768,y+192))).getbbox() is None
    check("teacher controls change captions only",True)
    for entry in ledger:
        rebuilt=a.request(entry["job"],p,pages)
        digest=hashlib.sha256(json.dumps(rebuilt,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        assert digest==entry["request_sha256"]
        raw=e.read(t.OUT/entry["response_file"])
        assert raw["status"]=="completed" and raw["model"].startswith(p["model"]) and raw["reasoning"]["effort"]=="low"
        texts=[v["text"] for o in raw["output"] if o["type"]=="message" for v in o["content"] if v["type"]=="output_text"]
        assert json.loads("".join(texts))==entry["answer"] and raw["usage"]==entry["usage"]
        assert 0<=entry["answer"]["xg"]<=1 and math.isfinite(entry["answer"]["xg"])
        assert raw["usage"]["input_tokens"]<=entry["input_bound"] and raw["usage"]["output_tokens"]<=1024
        assert math.isclose(a.token_cost(raw["usage"]),entry["estimated_usd"],abs_tol=1e-12)
        assert math.isclose(a.token_cost(raw["usage"],True),entry["conservative_usd"],abs_tol=1e-12)
        assert entry["conservative_usd"]<=entry["reserved_usd"]
    check("every secondary request hash,response,image and cost verified",True)
    conservative=p["prior_conservative_usd"]+sum(r["conservative_usd"] for r in ledger)
    check("combined spending within $10",conservative<=10)
    check("secondary lock released",not (t.OUT/"generation.lock").exists())
    result={"passed":True,"checks":checks,"check_count":len(checks),"secondary_requests_verified":len(ledger),"combined_conservative_usd":conservative}
    e.save(t.OUT/"audit.json",result)
    return rows,ledger,result


def conclusion(contrasts,teacher=False):
    zero,shuffle=contrasts
    if zero["interval"][1]<0 and shuffle["interval"][1]<0:
        return ("The teacher-labelled examples improved Brier versus both fresh zero-shot predictions and shuffled teacher labels on this cohort. This supports using the image–probability pairing, while remaining an imitation of an existing xG model." if teacher else
                "Correct outcome examples improved Brier versus both zero-shot predictions and shuffled outcomes on this cohort. This supports a benefit from the image–outcome pairing.")
    if zero["interval"][1]<0:
        return ("Teacher-labelled examples improved Brier versus zero-shot predictions, but did not establish an advantage over shuffled teacher labels. The result does not isolate learning from the correct image–probability pairing." if teacher else
                "Outcome examples improved Brier versus zero-shot predictions, but did not establish an advantage over shuffled outcomes. A shift in the overall probability scale can explain improvement without useful image–outcome learning.")
    return ("Teacher-labelled examples did not establish improved Brier over zero-shot predictions. The intervals also determine whether correct pairings beat the shuffled control; a point estimate alone is insufficient." if teacher else
            "Outcome-labelled examples did not establish improved Brier over zero-shot predictions. A lower point estimate alone is insufficient evidence of a reliable benefit.")


def main():
    a.main()
    rows,teacher_ledger,teacher_audit=audit_teacher()
    primary=e.read(e.OUT/"summary.json");secondary=e.read(t.OUT/"summary.json")
    original_rows,original_values,_=e.results("confirmation")
    by_teacher={(r["job"]["shot_id"],r["job"]["arm"]):r["answer"]["xg"] for r in teacher_ledger}
    teacher_values={arm:[by_teacher[(r["shot_id"],arm)] for r in rows] for arm in t.ARMS}
    contrasts={}
    for i,other in enumerate(["zero","shuffled64"]):contrasts["outcome64_minus_"+other]=interval(original_rows,original_values["icl64"],original_values[other],e.SEED+110+i)
    for i,other in enumerate(["zero","shuffled_teacher64"]):contrasts["teacher64_minus_"+other]=interval(rows,teacher_values["teacher64"],teacher_values[other],e.SEED+120+i)
    binary_conclusion=conclusion([contrasts["outcome64_minus_zero"],contrasts["outcome64_minus_shuffled64"]])
    teacher_conclusion=conclusion([contrasts["teacher64_minus_zero"],contrasts["teacher64_minus_shuffled_teacher64"]],True)
    combined={"primary":primary,"teacher":secondary,"combined_primary_contrasts_98_75pct":contrasts,"outcome_conclusion":binary_conclusion,"teacher_conclusion":teacher_conclusion,
        "combined_new_calls":primary["new_calls"]+secondary["new_calls"],"combined_estimated_usd":secondary["combined_estimated_usd"],"combined_conservative_usd":secondary["combined_conservative_usd"],
        "audits":{"primary":e.read(e.OUT/"audit.json"),"teacher":teacher_audit}}
    e.save(e.OUT/"combined_summary.json",combined)
    lines=["# StatsBomb in-context learning investigation","",
        "In-context learning supplies labelled examples inside the inference prompt, without changing model weights. This study tested GPT-6.2 Sol at low reasoning effort on positional shot diagrams.","",
        binary_conclusion,"",teacher_conclusion,"",
        "## Outcome-labelled examples","",
        "A fixed development comparison used 40 shots from 28 separate matches. It tested 0, 8, 32 and 64 naturally sampled training examples, plus shuffled outcomes for 64 examples. There were only two goals in development, so count selection is noisy. The lowest development Brier among correctly labelled conditions selected 64 examples. Those examples contain four goals; the shuffled control preserves that exact count.","",
        f"The frozen confirmation contains **{primary['shots']} shots, {primary['goals']} goals, {primary['matches']} matches**. Lower Brier is better; higher AUC is better.","",
        "| Method | Brier | AUC | Mean predicted xG |","|---|---:|---:|---:|"]
    names={"zero":"No examples","icl64":"64 examples + actual outcomes","shuffled64":"64 examples + shuffled outcomes","logistic_image":"Logistic: image features","logistic_header":"Logistic: plus header information","training_constant":"Training goal-rate constant","statsbomb":"StatsBomb reference"}
    lines += [f"| {names.get(arm,arm)} | {v['brier']:.6f} | {v['auc']:.4f} | {v['mean_xg']:.4f} |" for arm,v in primary["metrics"].items()]
    lines += ["","## Teacher-probability examples","",
        "This is a separate learning target: the same 64 training diagrams are paired with StatsBomb xG predictions, rounded to four decimals. They are teacher-model labels, not observed outcomes or ground-truth probabilities. The control permutes these values across the same images and preserves their complete distribution. It tests use of the correct pairing beyond simply copying a probability scale.","",
        "This secondary comparison was declared before inspecting any confirmation accuracy. It uses a fresh zero-shot control. Its cohort is a usage-selected prefix of the original confirmation pool, so it is not an independent replication. Compare methods within each table.","",
        f"The secondary comparison contains **{secondary['shots']} shots, {secondary['goals']} goals, {secondary['matches']} matches**.","",
        "| Method | Brier | AUC | Mean predicted xG |","|---|---:|---:|---:|"]
    names.update({"teacher64":"64 examples + StatsBomb xG","shuffled_teacher64":"64 examples + shuffled StatsBomb xG"})
    lines += [f"| {names.get(arm,arm)} | {v['brier']:.6f} | {v['auc']:.4f} | {v['mean_xg']:.4f} |" for arm,v in secondary["metrics"].items()]
    lines += ["","## Uncertainty","",
        "For the combined investigation, four comparisons use 98.75% whole-match bootstrap intervals (20,000 resamples), a Bonferroni adjustment across all four contrasts. Negative Brier differences favor correctly labelled ICL. The original per-study 97.5% intervals remain in their separate summaries. Intervals crossing zero do not show equivalence.","",
        "| Brier difference | Estimate | 98.75% match interval |","|---|---:|---:|"]
    for name,v in contrasts.items():lines.append(f"| {name} | {v['difference']:+.6f} | [{v['interval'][0]:+.6f}, {v['interval'][1]:+.6f}] |")
    lines += ["","## Budget and audit","",
        f"Completed **{combined['combined_new_calls']:,} new generation calls**, with no failed or uncertain requests. Estimated new cost: **${combined['combined_estimated_usd']:.4f}**. Conservative accounting: **${combined['combined_conservative_usd']:.4f}**, below the **$10 combined cap**. This is experiment accounting, not a live account balance.","",
        f"The primary audit passed {combined['audits']['primary']['check_count']} checks; the teacher audit passed {teacher_audit['check_count']}. Every request was rebuilt from the frozen images and prompts and matched its saved request hash. Raw responses, model settings, probabilities, token usage and costs reconcile. No paid runner is active.","",
        "## Limits and reproducibility","",
        "Training examples, development matches and query matches are disjoint. All source matches have prior research exposure; this is exploratory evaluation, not untouched external confirmation. The two ICL studies share their query source pool. No query outcomes, reference probabilities, IDs, body parts or coordinates enter an inference request.","",
        "Examples are packed into numbered sheets with 192-pixel diagrams; queries retain their 512-pixel diagrams. Small markers and occlusion remain limitations. Only player positions are visible, without shot body part, ball height or previous motion. One fixed example sample does not settle the value of all ICL approaches.","",
        "Logistic controls train on the separate 2,000-shot training split. The image-feature model uses distance, angle and opponents in the shooting cone. The header model has extra information. StatsBomb is a richer-input reference; the teacher condition explicitly imitates its training labels. All evaluated VLM probabilities are raw: no calibration, averaging, probability head or weight updates.","",
        "Preparation and scoring: `experiments/statsbomb_icl/experiment.py`, `teacher.py`, `audit.py` and `finalize.py`. Native Node inference: `run.mjs` and `teacher_run.mjs`. The teacher runner retains the same reservation engine and adds immutable parent-ledger accounting. Run `test_budget.mjs` for offline cap checks. Protocols, labels, request records and per-shot CSVs are in the two result directories.","",
        "Data attribution: StatsBomb Open Data, https://github.com/hudl/open-data. Retain StatsBomb attribution and follow the source repository license.","",
        "API rate source: https://developers.openai.com/api/docs/models/gpt-6.2-sol (standard input $2, cached input $0.10 and output $10 per million tokens). Reservations cover maximum output, including reasoning, concurrent calls and uncertain charges. No automatic retries."]
    (e.OUT/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"outcome_conclusion":binary_conclusion,"teacher_conclusion":teacher_conclusion,"combined_new_calls":combined["combined_new_calls"],"combined_estimated_usd":combined["combined_estimated_usd"],"combined_conservative_usd":combined["combined_conservative_usd"],"contrasts":contrasts}),flush=True)


if __name__=="__main__":main()
