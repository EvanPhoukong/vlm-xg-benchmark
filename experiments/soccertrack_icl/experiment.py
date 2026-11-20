"""SoccerTrack v2 visual ICL, with whole-match exclusion and a hard spending cap."""
import argparse
from functools import lru_cache
import csv
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments/soccertrack_comparison"))
import common as c
import prepare as source

OUT=ROOT/"results/soccertrack_icl_v1"
OLD=ROOT/"results/soccertrack_extensive_v1"
SEED=2026112007
ARMS=["zero","outcome8","outcome16","shuffled_outcome16","teacher16","shuffled_teacher16"]
PROMPT=c.PROMPT
OUTCOME_INTRO="""Historical examples follow on numbered contact sheets. Each cell is a DIFFERENT
past shot with its single observed result printed below: GOAL or NO GOAL.
These binary outcomes are noisy, not true probabilities: scoring once does not
mean true xG is 1, and missing once does not mean true xG is 0. Use the situations
and outcomes as examples for estimating a new shot's scoring probability.
All example frames are before contact. No example comes from the query match.
Read left to right, then top to bottom. The separate NEW QUERY after these sheets
has an unknown outcome. Predict only its probability."""
TEACHER_INTRO="""Historical examples follow on numbered contact sheets. Each cell is a DIFFERENT
past shot paired with an xG probability from a logistic model trained on other
matches, excluding the query match. These are teacher-model predictions, not
ground-truth probabilities or observed outcomes. Use the situations and probability
labels as examples for estimating a new shot's scoring probability.
All example frames are before contact. No example comes from the query match.
Read left to right, then top to bottom. The separate NEW QUERY after these sheets
has no supplied probability or outcome. Predict only its probability."""


def query(row,frames):
    item=frames[row["shot_id"]]
    return {"shot_id":row["shot_id"],"match_id":row["match_id"],"image":item["image"],"sha256":item["sha256"]}


def schedule(rows,frames):
    rng=random.Random(SEED+4);jobs=[]
    for row in rows:
        order=ARMS.copy();rng.shuffle(order)
        jobs.extend({**query(row,frames),"arm":arm} for arm in order)
    return jobs


def prepare():
    if (OUT/"protocol.json").exists():raise ValueError("Frozen study already exists")
    rows=c.read(c.OUT/"shots.json");original_audit=c.read(c.OUT/"source_audit.json")
    # Reconcile labels/actors/coordinates against the released local BAS and raw nodes.
    for audit in original_audit:
        regenerated,new_audit=source.extract_match(audit["match_id"])
        if regenerated!=[r for r in rows if r["match_id"]==audit["match_id"]] or json.loads(json.dumps(new_audit))!=audit:
            raise ValueError("Source reconciliation changed")
    frames={f["shot_id"]:f for f in c.read(OLD/"frames.json")}
    if len(rows)!=306 or len(frames)!=306:raise ValueError("Expected complete306-shot audit")
    for frame in frames.values():
        if c.sha(ROOT/frame["image"])!=frame["sha256"] or frame["frame_index"]>=frame["event_frame"]:
            raise ValueError("Frame provenance or past-only check failed")
    source_models=c.read(c.OUT/"lomo_models.json")
    mids=sorted({r["match_id"] for r in rows})

    @lru_cache(maxsize=306)
    def thumbnail(sid):
        with Image.open(ROOT/frames[sid]["image"]) as im:
            im=im.convert("RGB");im.thumbnail((768,384),Image.Resampling.LANCZOS)
            canvas=Image.new("RGB",(768,384),"black")
            canvas.paste(im,((768-im.width)//2,(384-im.height)//2))
            return canvas

    def sheets(examples,labels,mid,arm):
        pages=[]
        for start in range(0,len(examples),8):
            chunk=examples[start:start+8];canvas=Image.new("RGB",(1536,math.ceil(len(chunk)/2)*408),"white");draw=ImageDraw.Draw(canvas)
            for i,r in enumerate(chunk):
                x,y=i%2*768,i//2*408;canvas.paste(thumbnail(r["shot_id"]),(x,y))
                value=labels[start+i]
                label=("GOAL" if value else "NO GOAL") if "outcome" in arm else f"teacher xG = {value:.4f}"
                draw.text((x+12,y+390),f"Example {start+i+1}: {label}",fill="black")
            path=OUT/"examples"/f"{mid}_{arm}_{start//8+1:02d}.jpg"
            canvas.save(path,quality=92,subsampling=0)
            pages.append({"path":str(path.relative_to(ROOT)).replace("\\","/"),"sha256":c.sha(path),"size":list(canvas.size)})
        return pages

    all_pages={};examples_by_match={};models={};predictions=[]
    for i,mid in enumerate(mids):
        training=[r for r in rows if r["match_id"]!=mid]
        image_model=c.fit_logistic(training,["distance_m","angle_rad"])
        header_model=c.fit_logistic(training,["distance_m","angle_rad","header"])
        for name,model in [("logistic_distance_angle",image_model),("logistic_distance_angle_header",header_model)]:
            np.testing.assert_allclose(c.predict(model,rows),c.predict(source_models[mid][name],rows),rtol=0,atol=1e-8)
        models[mid]={"image":image_model,"header":header_model,"constant":sum(r["goal"] for r in training)/len(training)}
        selected=random.Random(SEED+100+i).sample(sorted(training,key=lambda r:r["shot_id"]),16)
        outcomes=[r["goal"] for r in selected];shuffled=outcomes.copy();random.Random(SEED+200+i).shuffle(shuffled)
        if shuffled==outcomes and len(set(outcomes))>1:shuffled=shuffled[1:]+shuffled[:1]
        teacher=[round(p,4) for p in c.predict(image_model,selected)]
        shuffled_teacher=teacher.copy();random.Random(SEED+300+i).shuffle(shuffled_teacher)
        examples_by_match[mid]=[{**query(r,frames),"goal":r["goal"],"shuffled_goal":g,"teacher":p,"shuffled_teacher":q}
                               for r,g,p,q in zip(selected,shuffled,teacher,shuffled_teacher)]
        all_pages[mid]={"zero":[],"outcome8":sheets(selected[:8],outcomes[:8],mid,"outcome8"),
            "outcome16":sheets(selected,outcomes,mid,"outcome16"),"shuffled_outcome16":sheets(selected,shuffled,mid,"shuffled_outcome16"),
            "teacher16":sheets(selected,teacher,mid,"teacher16"),"shuffled_teacher16":sheets(selected,shuffled_teacher,mid,"shuffled_teacher16")}
        tested=[r for r in rows if r["match_id"]==mid]
        for r,p,q in zip(tested,c.predict(image_model,tested),c.predict(header_model,tested)):
            predictions.append({**r,"logistic":p,"logistic_header":q,"training_constant":models[mid]["constant"]})
        print(f"Prepared match {mid}:16 examples from other matches",flush=True)
    pool=random.Random(SEED+1).sample(sorted(predictions,key=lambda r:r["shot_id"]),len(predictions))
    c.freeze(OUT/"pool.json",pool);c.freeze(OUT/"pages.json",all_pages);c.freeze(OUT/"examples.json",examples_by_match);c.freeze(OUT/"baselines.json",models)
    preview=[]
    for mid in mids:
        # Largest decoded query canvas in each match gives a conservative preview.
        candidates=[r for r in rows if r["match_id"]==mid]
        def area(r):
            with Image.open(ROOT/frames[r["shot_id"]]["image"]) as im:return im.width*im.height
        chosen=max(candidates,key=area)
        preview.extend({**query(chosen,frames),"arm":arm} for arm in ARMS)
    c.freeze(OUT/"preview/jobs.json",preview)
    protocol={"dataset":"SoccerTrack v2 v1.2 (called SoccerNet v2 by the user)","seed":SEED,"model":"gpt-6.2-sol","arms":ARMS,
        "prompt":PROMPT,"outcome_intro":OUTCOME_INTRO,"teacher_intro":TEACHER_INTRO,
        "cap_usd":10.,"max_output_tokens":1024,"max_attempts":2000,"workers":6,"retries":0,"effort":"low","service_tier":"default","image_detail":"high",
        "rates_per_million":{"input":2.,"cached_input":.1,"output":10.},"pricing_source":"https://developers.openai.com/api/docs/models/gpt-6.2-sol",
        "selection":"Fixed natural random ordered pool of306 shots,all10 matches. No accuracy-driven arm or example-count selection.",
        "resource_rule":"Before any generation, select largest N from[306,240,200,150,120,100,80,60] for which sum of per-job counted-input conservative costs plus128 output tokens per job, multiplied by1.15, plus largest6-request maximum reservation +$0.20 <=$10. Counts use the largest-canvas query from each match/arm preview plus256 tokens. Hard per-request guards remain in force. No test-driven expansion.",
        "examples":"16 natural random shots from other9 matches for each excluded query match;8 are nested prefix. No goal balancing. Fixed permutations preserve binary counts or complete teacher-value distributions.",
        "teacher":"Distance/angle logistic model trained on other9 matches,including their training example labels but excluding every event of the query match. Its predictions on training examples are in-context teacher labels,rounded4 decimals. No query reference probability supplied.",
        "statistics":"4 primary Brier contrasts:outcome16-zero,outcome16-shuffledoutcome,teacher16-zero,teacher16-shuffledteacher.98.75% whole-match percentile intervals,20000 resamples,conditional on fitted example/model choices.8-example and logistic comparisons descriptive.10 matches limit generalization.",
        "scope":"All source matches were previously studied; exploratory match-held-out cross-validation,not untouched external confirmation. Raw VLM probabilities,no calibration,averaging,head or weight updates.",
        "images":"Original audited query PNGs unchanged. Before-shot example frames letterboxed at most768x384 into2-column sheets,8 per page,JPEG92 with no chroma subsampling. Players remain small and shooters unmarked. No post-shot frames,query outcomes,coordinates,identities or prior query probabilities enter requests.",
        "limitations":["University matches with retained average-professional-finisher prompt.","BAS SHOT/GOAL can omit blocked/header attempts assigned other classes; unidentified penalties may remain.","Baseline uses at-event source coordinates; image is approximately0.32s earlier.","Example sets and logistic fits overlap across folds; bootstrap intervals condition on these fitted inputs.","Some natural16-example samples may contain no goals; a binary shuffled control then cannot change labels,which must be reported.","Compare methods on identical shots;do not compare absolute Brier with StatsBomb cohorts."],
        "budget_scope":"New SoccerTrack ICL run only,$10 combined across all6 conditions; no account-wide balance claim. Other completed experiment ledgers are preserved.",
        "pool_sha256":c.sha(OUT/"pool.json"),"pages_sha256":c.sha(OUT/"pages.json"),"examples_sha256":c.sha(OUT/"examples.json"),"baselines_sha256":c.sha(OUT/"baselines.json"),"preview_sha256":c.sha(OUT/"preview/jobs.json"),
        "source_hashes":{"shots":c.sha(c.OUT/"shots.json"),"audit":c.sha(c.OUT/"source_audit.json"),"frames":c.sha(OLD/"frames.json"),"frame_audit":c.sha(OLD/"frame_audit.json")},
        "python_sha256":c.sha(__file__),"runner_sha256":c.sha(Path(__file__).with_name("run.mjs")),
        "attribution":"SoccerTrack v2 by Atom Scott,Ikuma Uchida,Kento Kuroda,Yufi Kim,Keisuke Fujii. CC BY4.0. https://huggingface.co/datasets/atomscott/soccertrack-v2"}
    c.freeze(OUT/"protocol.json",protocol)
    print("Frozen6-arm study and306-shot pool; no paid calls.",flush=True)


def choose():
    if (OUT/"attempts.json").exists():raise ValueError("Choose cohort before paid predictions")
    p=c.read(OUT/"protocol.json");counts=c.read(OUT/"token_preview.json");pool=c.read(OUT/"pool.json")
    forecasts={};largest=max(v["max_reserved_usd"] for v in counts.values())
    for n in (306,240,200,150,120,100,80,60):
        projected=sum(((counts[r["match_id"]+":"+arm]["input_tokens"]+256)*2*1.25+128*10)/1e6 for r in pool[:n] for arm in ARMS)
        forecasts[str(n)]=projected*1.15+6*largest+.20
    options=[int(n) for n,v in forecasts.items() if v<=10.]
    if not options:raise ValueError("Fixed study cannot fit cap")
    selected=pool[:max(options)]
    if len({r["match_id"] for r in selected})!=10:raise ValueError("Selected cohort does not cover all10 matches")
    c.freeze(OUT/"resource_choice.json",{"n":len(selected),"forecasts_usd":forecasts,"rule":"Usage-only input-count rule;no accuracy inspected.","preview_sha256":c.sha(OUT/"token_preview.json")})
    frames={f["shot_id"]:f for f in c.read(OLD/"frames.json")}
    c.freeze(OUT/"evaluation/selection.json",selected);c.freeze(OUT/"evaluation/jobs.json",schedule(selected,frames))
    print(json.dumps(c.read(OUT/"resource_choice.json")),flush=True)


def interval(rows,p,q,seed):
    y=np.array([r["goal"] for r in rows]);delta=(np.array(p)-y)**2-(np.array(q)-y)**2
    mids=sorted({r["match_id"] for r in rows});sums=np.array([delta[[i for i,r in enumerate(rows) if r["match_id"]==m]].sum() for m in mids]);counts=np.array([sum(r["match_id"]==m for r in rows) for m in mids])
    draws=np.random.default_rng(seed).integers(len(mids),size=(20000,len(mids)));boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return {"difference":float(delta.mean()),"confidence":.9875,"interval":np.quantile(boot,[.00625,.99375]).tolist(),"matches":len(mids)}


def analyze():
    rows=c.read(OUT/"evaluation/selection.json");jobs=c.read(OUT/"evaluation/jobs.json");ledger=c.read(OUT/"attempts.json")
    if [r["job"] for r in ledger]!=jobs or any(r["state"]!="completed" for r in ledger):raise ValueError("Complete fixed schedule required")
    answers={(r["job"]["shot_id"],r["job"]["arm"]):r["answer"]["xg"] for r in ledger};values={a:[answers[(r["shot_id"],a)] for r in rows] for a in ARMS}
    for name in ("logistic","logistic_header","training_constant"):values[name]=[r[name] for r in rows]
    y=[r["goal"] for r in rows];metrics={a:c.metrics(y,v) for a,v in values.items()}
    pairs=[("outcome16","zero"),("outcome16","shuffled_outcome16"),("teacher16","zero"),("teacher16","shuffled_teacher16")]
    primary={a+"_minus_"+b:interval(rows,values[a],values[b],SEED+500+i) for i,(a,b) in enumerate(pairs)}
    summary={"dataset":"SoccerTrack v2","model":"gpt-6.2-sol","shots":len(rows),"goals":sum(y),"matches":len({r["match_id"] for r in rows}),"metrics":metrics,"primary":primary,
        "calls":len(ledger),"estimated_usd":sum(r["estimated_usd"] for r in ledger),"conservative_usd":sum(r["conservative_usd"] for r in ledger),"cap_usd":10.,"failures":0,
        "per_match":{m:{a:c.metrics([r["goal"] for r in rows if r["match_id"]==m],[v[i] for i,r in enumerate(rows) if r["match_id"]==m]) for a,v in values.items()} for m in sorted({r["match_id"] for r in rows})}}
    assert summary["conservative_usd"]<=10;c.save(OUT/"summary.json",summary)
    with (OUT/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["shot_id","match_id","goal"]+list(values));w.writeheader();w.writerows({"shot_id":r["shot_id"],"match_id":r["match_id"],"goal":r["goal"],**{a:v[i] for a,v in values.items()}} for i,r in enumerate(rows))
    print(json.dumps({k:v for k,v in summary.items() if k!="per_match"}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","choose","analyze"]);globals()[p.parse_args().action]()
