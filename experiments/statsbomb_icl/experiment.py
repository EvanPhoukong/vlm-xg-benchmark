"""Outcome-labelled visual in-context learning: offline preparation and evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import importlib.util
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/statsbomb_icl_v1"
SOURCE = ROOT / "results/statsbomb_finetune_v1"
spec = importlib.util.spec_from_file_location("icl_data_helpers", ROOT / "experiments/statsbomb_finetune/common.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
read, save, freeze, sha = helpers.read, helpers.save, helpers.freeze, helpers.sha
SEED = 2026112001
ARMS = ["zero", "icl8", "icl32", "icl64", "shuffled64"]
PROMPT = """You are a professional football (soccer) analyst. The query diagram shows a shot
at contact on a half pitch, with the goal at the top. Gold star = shooter; blue circles
= teammates; red circles = opposing outfield players; orange square = opposing keeper.
Estimate the probability of a goal (xG), assuming an average professional finisher.
Body part, ball height, ball speed and previous motion are unknown. Average over the
plausible executions compatible with the positions instead of inventing these details.
Reply with JSON only: {"xg": <number between 0 and 1>, "reason": "<one short sentence>"}."""
INTRO = """Historical examples follow on numbered contact sheets. Each cell is a DIFFERENT
past shot, with its single observed result printed below it: GOAL or NO GOAL.
These are noisy binary outcomes, not xG targets: a scored shot does not have true
xG 1, and a missed shot does not have true xG 0. Use the visual situations and their
outcomes to estimate scoring probabilities for new shots. The unlabelled QUERY
image after the examples is separate; estimate only its probability. Each example
uses the same symbol legend and pitch orientation as the query. Read cells left to
right, then top to bottom. No query outcomes are supplied."""


def query_manifest(row):
    return {"shot_id": row["shot_id"], "image": "results/statsbomb_finetune_v1/" + row["image"],
            "sha256": row["image_sha256"]}


def schedule(rows, arms, seed):
    rng = random.Random(seed)
    jobs = []
    for row in rows:
        order = list(arms)
        rng.shuffle(order)
        jobs.extend({**query_manifest(row), "arm": arm} for arm in order)
    return jobs


def sheets(examples, labels, name):
    """192px diagrams, four by four. Only captions contain the example labels."""
    records = []
    for start in range(0, len(examples), 16):
        chunk = examples[start:start+16]
        canvas = Image.new("RGB", (768, math.ceil(len(chunk)/4)*216), "white")
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(chunk):
            x, y = j % 4*192, j//4*216
            with Image.open(SOURCE/row["image"]) as im:
                canvas.paste(im.convert("RGB").resize((192,192), Image.Resampling.LANCZOS), (x,y))
            label = "GOAL" if labels[start+j] else "NO GOAL"
            draw.text((x+9,y+196), f"Example {start+j+1}: {label}", fill="black")
        path = OUT/"examples"/f"{name}_{start//16+1:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        records.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path),
                        "first_example": start+1, "last_example": start+len(chunk), "size": list(canvas.size)})
    return records


def prepare():
    if (OUT/"protocol.json").exists():
        raise ValueError("Already prepared; preserve frozen artifacts")
    source_protocol = read(SOURCE/"protocol.json")
    if sha(SOURCE/"selected.json") != source_protocol["selected_sha256"]:
        raise ValueError("Source selection changed")
    data = read(SOURCE/"selected.json")
    groups = {k: {r["match_id"] for r in v} for k,v in data.items()}
    assert not any(groups[a] & groups[b] for a,b in [("train","validation"),("train","test"),("validation","test")])
    examples = random.Random(SEED).sample(sorted(data["train"],key=lambda r:r["shot_id"]),64)
    development = random.Random(SEED+1).sample(sorted(data["validation"],key=lambda r:r["shot_id"]),40)
    test_pool = random.Random(SEED+2).sample(sorted(data["test"],key=lambda r:r["shot_id"]),200)
    rows = examples + development + test_pool
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda row: helpers.verify_image(row,SOURCE), rows))
    labels = [r["goal"] for r in examples]
    shuffled = labels.copy()
    random.Random(SEED+3).shuffle(shuffled)
    assert sum(labels) == sum(shuffled) and labels != shuffled
    freeze(OUT/"examples/manifest.json",[{**query_manifest(r),"match_id":r["match_id"],"goal":r["goal"],"shuffled_goal":v}
                                      for r,v in zip(examples,shuffled)])
    pages = {"zero": []}
    for n in (8,32,64):
        pages[f"icl{n}"] = sheets(examples[:n], labels[:n], f"correct{n}")
    pages["shuffled64"] = sheets(examples,shuffled,"shuffled64")
    freeze(OUT/"pages.json",pages)
    freeze(OUT/"development/selection.json",development)
    freeze(OUT/"test_pool.json",test_pool)
    freeze(OUT/"development/jobs.json",schedule(development,ARMS,SEED+4))
    features = ["distance","angle","defenders_in_cone"]
    freeze(OUT/"baselines.json",{"image_features":helpers.fit_logistic(data["train"],features),
        "plus_header":helpers.fit_logistic(data["train"],features+["header"]),
        "training_goal_rate":sum(r["goal"] for r in data["train"])/len(data["train"])})
    protocol = {
        "date_utc":"2026-11-20", "seed":SEED,
        "authorization":"User requested StatsBomb in-context learning investigation, any useful image count, $10 limit.",
        "definition":"Examples in the inference prompt, no model-weight updates.",
        "model":"gpt-6.2-sol", "effort":"low", "service_tier":"default", "image_detail":"high",
        "rates_per_million":{"input":2.,"cached_input":.10,"output":10.},
        "pricing_source":"https://developers.openai.com/api/docs/models/gpt-6.2-sol",
        "cap_usd":10., "max_output_tokens":1024,"max_attempts":800,"workers":6,"retries":0,
        "conservative_input_multiplier":1.25,"input_count_margin":1024,
        "arms":ARMS,"prompt":PROMPT,"intro":INTRO,
        "examples":"Nested natural random training-only sample of 8/32/64 diagrams, binary observed outcomes. Fixed 64-label permutation preserves the goal count.",
        "contact_sheets":"4 columns x up to4 rows, 192px positional diagrams with24px outcome captions; same correct/shuffled positions. Query remains original512px. This format is a study limitation.",
        "development":"40 validation shots, all5 arms fixed. Select minimum development Brier among icl8/icl32/icl64; ties choose fewer. Zero remains comparator even if better.",
        "confirmation":"Separate match groups from examples/development: selected ICL, zero and shuffled labels at selected count. Never select or tune on confirmation outcomes.",
        "resource_rule":"After development, largest N from[200,150,120,100,80] with existing conservative spend +1.5*N*sum(mean conservative cost of zero/selectedICL/selectedshuffle)+one largest request reservation+$0.25 <=$10. If selected count is not64, use shuffled64 mean as conservative forecast. No confirmation accuracy inspected to choose N.",
        "primary":"SelectedICL-minus-zero Brier and selectedICL-minus-matched-shuffled Brier,97.5% whole-match percentile bootstrap intervals,20000 resamples;2 primary contrasts. Negative favors ICL. Logistic comparisons descriptive.",
        "split_history":"Reuses supervised pilot train/validation/test split. Match-disjoint within this experiment, but source matches have prior research exposure. Exploratory confirmation, not untouched external validation. Separate experiment does not score pending fine-tuning test study.",
        "inputs":"Only example image/outcome pairs and query image. No query labels,StatsBombxG,logistic probabilities,body part,IDs,coordinates or goal rates in API inputs. No calibration,averaging or teacher probabilities.",
        "spend":"$10 combined incremental cap. Completed: uncached input+25% and actual output. Concurrent/uncertain: counted input+1024 and maximum output. Not an account balance. Stop on uncertainty; no retries.",
        "source_selected_sha256":sha(SOURCE/"selected.json"),"source_protocol_sha256":sha(SOURCE/"protocol.json"),
        "pages_sha256":sha(OUT/"pages.json"),"examples_sha256":sha(OUT/"examples/manifest.json"),
        "development_sha256":sha(OUT/"development/selection.json"),"test_pool_sha256":sha(OUT/"test_pool.json"),
        "python_sha256":sha(__file__),"runner_sha256":sha(Path(__file__).with_name("run.mjs")),
        "provenance":{"credit":"StatsBomb Open Data","repository":"https://github.com/hudl/open-data","license":"Retain StatsBomb attribution; see source repository LICENSE.pdf."}}
    freeze(OUT/"protocol.json",protocol)
    save(OUT/"preflight.json",{"images_verified":len(rows),"match_overlap":False,
        "development_matches":len({r["match_id"] for r in development}),
        "example_goals":{str(n):sum(labels[:n]) for n in (8,32,64)},"new_generation_calls":0})
    print("Prepared64 examples,40 development shots,200-shot held-out pool; no paid calls.",flush=True)


def results(stage):
    rows,jobs=read(OUT/stage/"selection.json"),read(OUT/stage/"jobs.json")
    entries=[e for e in read(OUT/"attempts.json") if e["stage"]==stage]
    if len(entries)!=len(jobs) or any(e["state"]!="completed" for e in entries):
        raise ValueError("Complete fixed stage required before scoring")
    answers={(e["job"]["shot_id"],e["job"]["arm"]):e["answer"]["xg"] for e in entries}
    values={a:[answers[(r["shot_id"],a)] for r in rows] for a in sorted({j["arm"] for j in jobs})}
    return rows,values,{a:helpers.metrics([r["goal"] for r in rows],p) for a,p in values.items()}


def select():
    rows,values,scores=results("development")
    selected=min(("icl8","icl32","icl64"),key=lambda a:(scores[a]["brier"],int(a[3:])))
    n=int(selected[3:]);examples=read(OUT/"examples/manifest.json")[:n]
    labels=[r["goal"] for r in examples];random.Random(SEED+10+n).shuffle(labels)
    if labels==[r["goal"] for r in examples]:labels=labels[1:]+labels[:1]
    pages=read(OUT/"pages.json");matched="shuffled"+str(n)
    if n!=64:
        source={r["shot_id"]:r for r in read(SOURCE/"selected.json")["train"]}
        pages[matched]=sheets([source[r["shot_id"]] for r in examples],labels,matched)
        freeze(OUT/"matched_shuffle.json",[{"shot_id":r["shot_id"],"goal":r["goal"],"shuffled_goal":v} for r,v in zip(examples,labels)])
    freeze(OUT/"confirmation_pages.json",pages)
    ledger=read(OUT/"attempts.json");spent=sum(e["conservative_usd"] for e in ledger)
    means={a:float(np.mean([e["conservative_usd"] for e in ledger if e["job"]["arm"]==a])) for a in ARMS}
    reserve=max(e["reserved_usd"] for e in ledger)
    forecasts={str(k):spent+1.5*k*(means["zero"]+means[selected]+means["shuffled64"])+reserve+.25 for k in (200,150,120,100,80)}
    fit=[int(k) for k,v in forecasts.items() if v<=10.]
    if not fit:
        save(OUT/"development/metrics.json",scores)
        raise ValueError("No planned confirmation sample fits budget")
    chosen=read(OUT/"test_pool.json")[:max(fit)]
    freeze(OUT/"selection_decision.json",{"selected_arm":selected,"matched_shuffle":matched,"n":len(chosen),
        "development_metrics":scores,"forecast_conservative_usd":forecasts,"spent_conservative_usd":spent,
        "decision":"Development-only Brier selects count;usage-only rule selects confirmation N. No confirmation outcomes scored."})
    freeze(OUT/"confirmation/selection.json",chosen)
    freeze(OUT/"confirmation/jobs.json",schedule(chosen,["zero",selected,matched],SEED+5))
    print(json.dumps(read(OUT/"selection_decision.json")),flush=True)


def interval(rows,p,q,seed):
    mids=sorted({r["match_id"] for r in rows});y=np.array([r["goal"] for r in rows])
    delta=(np.array(p)-y)**2-(np.array(q)-y)**2
    sums=np.array([delta[[i for i,r in enumerate(rows) if r["match_id"]==mid]].sum() for mid in mids])
    counts=np.array([sum(r["match_id"]==mid for r in rows) for mid in mids])
    draws=np.random.default_rng(seed).integers(len(mids),size=(20000,len(mids)))
    boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return {"difference":float(delta.mean()),"confidence":.975,"interval":np.quantile(boot,[.0125,.9875]).tolist(),"matches":len(mids)}


def analyze():
    rows,values,scores=results("confirmation");decision=read(OUT/"selection_decision.json")
    selected,shuffled=decision["selected_arm"],decision["matched_shuffle"];baselines=read(OUT/"baselines.json")
    values["logistic_image"]=helpers.predict(baselines["image_features"],rows)
    values["logistic_header"]=helpers.predict(baselines["plus_header"],rows)
    values["training_constant"]=[baselines["training_goal_rate"]]*len(rows)
    values["statsbomb"]=[r["statsbomb_xg"] for r in rows];y=[r["goal"] for r in rows]
    scores={a:helpers.metrics(y,p) for a,p in values.items()}
    comparisons={f"{selected}_minus_{b}":interval(rows,values[selected],values[b],SEED+20+i) for i,b in enumerate(["zero",shuffled])}
    ledger=read(OUT/"attempts.json")
    summary={"model":"gpt-6.2-sol","selected_examples":int(selected[3:]),"shots":len(rows),"matches":len({r["match_id"] for r in rows}),
        "goals":sum(y),"metrics":scores,"primary":comparisons,"development":decision["development_metrics"],"new_calls":len(ledger),
        "estimated_usd":sum(e["estimated_usd"] for e in ledger),"conservative_usd":sum(e["conservative_usd"] for e in ledger),
        "cap_usd":10.,"failures":sum(e["state"]!="completed" for e in ledger)}
    save(OUT/"summary.json",summary)
    with (OUT/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["shot_id","match_id","goal"]+list(values));writer.writeheader()
        writer.writerows({"shot_id":r["shot_id"],"match_id":r["match_id"],"goal":r["goal"],**{a:p[i] for a,p in values.items()}} for i,r in enumerate(rows))
    lines=["# StatsBomb visual in-context learning","",f"GPT-6.2 Sol, low reasoning. {len(rows)} confirmation shots,{sum(y)} goals,{summary['matches']} matches.",
        f"Selected {summary['selected_examples']} outcome-labelled examples using40 separate development shots. No weight updates.","",
        "| Confirmation method | Brier (lower) | AUC (higher) | Mean xG |","|---|---:|---:|---:|"]
    lines += [f"| {a} | {m['brier']:.6f} | {m['auc']:.4f} | {m['mean_xg']:.4f} |" for a,m in scores.items()]
    lines += ["","Two primary contrasts:97.5% whole-match bootstrap intervals,20,000 draws. Negative Brier difference favors correct examples.",""]
    for name,v in comparisons.items():lines.append(f"- {name}: {v['difference']:+.6f}, interval [{v['interval'][0]:+.6f},{v['interval'][1]:+.6f}].")
    lines += ["","## Development count comparison","","| Examples | Brier | AUC |","|---|---:|---:|"]
    lines += [f"| {a} | {m['brier']:.6f} | {m['auc']:.4f} |" for a,m in decision["development_metrics"].items()]
    lines += ["",f"New API calls:{len(ledger)}. Estimated cost ${summary['estimated_usd']:.4f}; conservative ${summary['conservative_usd']:.4f},within the$10 combined cap.","",
        "Example,development and confirmation match groups are disjoint. Source matches have prior research exposure; this is exploratory, not untouched external confirmation. Query outcomes,IDs,coordinates and reference predictions never enter the prompt.","",
        "Examples are natural random shots with noisy binary outcomes. Diagrams are192px in numbered contact sheets to fit the budget; queries remain512px. This tests this format and one fixed example sample. A null result does not rule out other ICL designs. Only positions are visible; body part,height and motion are absent.","",
        "Logistic controls train only on the2000-shot training split. The image-feature control uses distance,angle and opponents in the cone; the header control has extra information. StatsBomb is a separate reference with richer inputs. No calibration,averaging or teacher probabilities.","",
        "Raw responses,usage,exact request hashes,images,frozen source hashes and per-shot predictions accompany this report. No account-wide balance is claimed. No paid runner remains active after completion.","",
        "StatsBomb Open Data: https://github.com/hudl/open-data. Retain attribution and follow the source license."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps(summary),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["prepare","select","analyze"])
    globals()[parser.parse_args().action]()
