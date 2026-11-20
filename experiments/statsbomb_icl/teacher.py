"""Separate, predeclared ICL study using training-only StatsBomb probability labels."""
import argparse
import csv
import json
import math
import random

from PIL import Image, ImageDraw

import experiment as e

OUT=e.ROOT/"results/statsbomb_icl_teacher_v1"
ARMS=["zero","teacher64","shuffled_teacher64"]
INTRO="""Historical examples follow on numbered contact sheets. Each cell shows a DIFFERENT
past shot paired with the probability predicted by the existing StatsBomb xG model.
These labels are teacher-model probabilities, not observed goal/no-goal outcomes
and not ground-truth probabilities. Use the visual situations and their probability
labels as examples for estimating xG for a new shot. Read cells left to right, then
top to bottom. Each diagram uses the same symbol legend and goal-at-top orientation.
The separate QUERY after the examples has no supplied probability or outcome.
Return your own probability for that query only."""


def declare():
    e.freeze(OUT/"declaration.json",{
        "purpose":"Additional fixed teacher-probability ICL and shuffled-teacher control, declared before scoring any primary confirmation predictions.",
        "model":"gpt-6.2-sol","examples":64,"labels":"StatsBomb xG from training examples,rounded to4 decimals; never query labels.",
        "candidates":[150,100,80],"budget_rule":"After original800 calls, largest N with parent conservative spend+1.15*N*sum(forecast costs of zero and two teacher arms)+largest new reservation+$0.20 <=$10. Forecast input from server count+1024,output from mean original same-format64-example calls plus50%+64 tokens. Freeze N before secondary calls.",
        "scope":"New fresh zero control and paired teacher/shuffled-teacher calls; no weights,calibration or averaging. Same source test pool prefix,so secondary evidence is exploratory and not independent replication of primary.",
        "primary":"Teacher64-minus-fresh-zero and teacher64-minus-shuffledteacher Brier.97.5% whole-match bootstrap intervals within secondary family; combined report also gives98.75% intervals across all4 ICL contrasts.",
        "combined_cap_usd":10.,"intro":INTRO,"no_confirmation_scores_inspected":True})
    print("Declared separate teacher-probability study; no paid calls.",flush=True)


def pages(rows,labels,name):
    result=[]
    for start in range(0,len(rows),16):
        canvas=Image.new("RGB",(768,864),"white");draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(rows[start:start+16]):
            x,y=j%4*192,j//4*216
            with Image.open(e.SOURCE/r["image"]) as im:canvas.paste(im.convert("RGB").resize((192,192),Image.Resampling.LANCZOS),(x,y))
            draw.text((x+8,y+196),f"Example {start+j+1}: xG = {labels[start+j]:.4f}",fill="black")
        p=OUT/"examples"/f"{name}_{start//16+1:02d}.png";p.parent.mkdir(parents=True,exist_ok=True);canvas.save(p)
        result.append({"path":str(p.relative_to(e.ROOT)).replace("\\","/"),"sha256":e.sha(p)})
    return result


def prepare():
    if (OUT/"protocol.json").exists():raise ValueError("Teacher study already frozen")
    original=e.read(e.OUT/"attempts.json")
    if len(original)!=800 or any(r["state"]!="completed" for r in original):raise ValueError("Original fixed800 calls must finish first")
    declaration=e.read(OUT/"declaration.json")
    manifest=e.read(e.OUT/"examples/manifest.json")
    source={r["shot_id"]:r for r in e.read(e.SOURCE/"selected.json")["train"]}
    rows=[source[r["shot_id"]] for r in manifest]
    labels=[round(r["statsbomb_xg"],4) for r in rows];shuffled=labels.copy();random.Random(e.SEED+80).shuffle(shuffled)
    assert sorted(labels)==sorted(shuffled) and labels!=shuffled
    e.freeze(OUT/"examples/manifest.json",[{"shot_id":r["shot_id"],"match_id":r["match_id"],"teacher":v,"shuffled_teacher":w} for r,v,w in zip(rows,labels,shuffled)])
    e.freeze(OUT/"pages.json",{"zero":[],"teacher64":pages(rows,labels,"correct"),"shuffled_teacher64":pages(rows,shuffled,"shuffled")})
    # Freeze the full candidate pool for server token counting before selecting N.
    pool=e.read(e.OUT/"confirmation/selection.json")[:150]
    e.freeze(OUT/"pool.json",pool)
    e.freeze(OUT/"preview_jobs.json",e.schedule(pool[:1],ARMS,e.SEED+81))
    e.freeze(OUT/"protocol.json",{
        "model":"gpt-6.2-sol","intro":INTRO,"prompt":e.PROMPT,"arms":ARMS,"cap_usd":10.,"max_output_tokens":1024,
        "effort":"low","service_tier":"default","rates_per_million":{"input":2.,"cached_input":.1,"output":10.},
        "prior_conservative_usd":sum(r["conservative_usd"] for r in original),"prior_estimated_usd":sum(r["estimated_usd"] for r in original),
        "parent_ledger_sha256":e.sha(e.OUT/"attempts.json"),"parent_protocol_sha256":e.sha(e.OUT/"protocol.json"),
        "pages_sha256":e.sha(OUT/"pages.json"),"examples_sha256":e.sha(OUT/"examples/manifest.json"),"pool_sha256":e.sha(OUT/"pool.json"),
        "declaration_sha256":e.sha(OUT/"declaration.json"),"python_sha256":e.sha(__file__),"runner_sha256":e.sha(e.ROOT/"experiments/statsbomb_icl/teacher_run.mjs"),
        "declaration":declaration,"input_format":"64 training-only teacher probabilities on4 numbered contact sheets; query remains512px. Fresh zero and shuffled-label controls."})
    print("Teacher example images and protocol frozen; no paid calls.",flush=True)


def choose():
    p=e.read(OUT/"protocol.json");preview=e.read(OUT/"token_preview.json");old=e.read(e.OUT/"attempts.json")
    output_forecast={}
    for arm in ARMS:
        observed=[r["usage"]["output_tokens"] for r in old if r["job"]["arm"]==("zero" if arm=="zero" else "icl64")]
        output_forecast[arm]=min(1024,math.ceil(1.5*sum(observed)/len(observed)+64))
    means={a:((preview[a]["input_tokens"]+1024)*2*1.25+output_forecast[a]*10)/1e6 for a in ARMS}
    reserve=max(v["max_reserved_usd"] for v in preview.values())
    forecasts={str(n):p["prior_conservative_usd"]+1.15*n*sum(means.values())+reserve+.20 for n in (150,100,80)}
    choices=[int(n) for n,v in forecasts.items() if v<=10]
    if not choices:raise ValueError("Secondary study cannot fit; no calls")
    selected=e.read(OUT/"pool.json")[:max(choices)]
    e.freeze(OUT/"resource_choice.json",{"n":len(selected),"forecasts":forecasts,"output_token_forecasts":output_forecast,"rule":"Predeclared usage-only choice; no confirmation accuracy scored."})
    e.freeze(OUT/"evaluation/selection.json",selected)
    e.freeze(OUT/"evaluation/jobs.json",e.schedule(selected,ARMS,e.SEED+82))
    print(json.dumps(e.read(OUT/"resource_choice.json")),flush=True)


def analyze():
    rows=e.read(OUT/"evaluation/selection.json");jobs=e.read(OUT/"evaluation/jobs.json");ledger=e.read(OUT/"attempts.json")
    assert len(jobs)==len(ledger) and all(r["state"]=="completed" for r in ledger)
    answers={(r["job"]["shot_id"],r["job"]["arm"]):r["answer"]["xg"] for r in ledger}
    values={a:[answers[(r["shot_id"],a)] for r in rows] for a in ARMS};y=[r["goal"] for r in rows]
    b=e.read(e.OUT/"baselines.json")
    values["logistic_image"]=e.helpers.predict(b["image_features"],rows)
    values["logistic_header"]=e.helpers.predict(b["plus_header"],rows)
    values["statsbomb"]=[r["statsbomb_xg"] for r in rows]
    metrics={a:e.helpers.metrics(y,v) for a,v in values.items()}
    primary={"teacher64_minus_"+a:e.interval(rows,values["teacher64"],values[a],e.SEED+90+i) for i,a in enumerate(["zero","shuffled_teacher64"])}
    p=e.read(OUT/"protocol.json")
    summary={"shots":len(rows),"matches":len({r["match_id"] for r in rows}),"goals":sum(y),"examples":64,"metrics":metrics,"primary":primary,
        "new_calls":len(ledger),"new_estimated_usd":sum(r["estimated_usd"] for r in ledger),"new_conservative_usd":sum(r["conservative_usd"] for r in ledger)}
    summary["combined_estimated_usd"]=p["prior_estimated_usd"]+summary["new_estimated_usd"]
    summary["combined_conservative_usd"]=p["prior_conservative_usd"]+summary["new_conservative_usd"]
    assert summary["combined_conservative_usd"]<=10
    e.save(OUT/"summary.json",summary)
    with (OUT/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["shot_id","match_id","goal"]+list(values));w.writeheader()
        w.writerows({"shot_id":r["shot_id"],"match_id":r["match_id"],"goal":r["goal"],**{a:v[i] for a,v in values.items()}} for i,r in enumerate(rows))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["declare","prepare","choose","analyze"])
    globals()[parser.parse_args().action]()
