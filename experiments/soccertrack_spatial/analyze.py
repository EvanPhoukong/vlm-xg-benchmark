"""Offline scoring of completed spatial-information predictions and matched controls."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

import experiment as e

c,ROOT,OUT=e.c,e.ROOT,e.OUT


def interval(rows,p,q,seed):
    y=np.array([r["goal"] for r in rows]);delta=(np.array(p)-y)**2-(np.array(q)-y)**2
    mids=sorted({r["match_id"] for r in rows})
    sums=np.array([delta[[i for i,r in enumerate(rows) if r["match_id"]==mid]].sum() for mid in mids])
    counts=np.array([sum(r["match_id"]==mid for r in rows) for mid in mids])
    samples=np.random.default_rng(seed).integers(len(mids),size=(20000,len(mids)))
    bootstrap=sums[samples].sum(axis=1)/counts[samples].sum(axis=1)
    return {"difference":float(delta.mean()),"confidence":.9875,"interval":np.quantile(bootstrap,[.00625,.99375]).tolist(),"matches":len(mids)}


def preview(rows):
    # A fixed source ID, not a selection based on prediction accuracy or outcome.
    row=min(rows,key=lambda r:r["shot_id"])
    canvas=Image.new("RGB",(1200,690),"white");draw=ImageDraw.Draw(canvas)
    for i,(kind,title) in enumerate([("raw","Raw panorama"),("diagram","Player positions and roles"),("marked_crop","Marked panorama and shot area")]):
        with Image.open(ROOT/row["images"][kind]["path"]) as im:
            im=im.convert("RGB");im.thumbnail((390,630),Image.Resampling.LANCZOS);canvas.paste(im,(i*400+(400-im.width)//2,40))
        draw.text((i*400+12,12),title,fill="black")
    canvas.save(OUT/"input_preview.png")


def main():
    p=c.read(OUT/"protocol.json");rows=c.read(OUT/"evaluation/selection.json");jobs=c.read(OUT/"evaluation/jobs.json");ledger=c.read(OUT/"attempts.json")
    if len(jobs)!=len(ledger) or [r["job"] for r in ledger]!=jobs or any(r["state"]!="completed" for r in ledger):raise ValueError("Complete fixed schedule required before scoring")
    answers={(r["job"]["shot_id"],r["job"]["arm"]):r["answer"]["xg"] for r in ledger}
    values={a:[answers[(r["shot_id"],a)] for r in rows] for a in e.ARMS}
    baseline={r["shot_id"]:r for r in c.read(OUT/"baseline_predictions.json")}
    for col in ("basic","spatial","constant"):values[col]=[baseline[r["shot_id"]][col] for r in rows]
    # Preserve the established at-event comparator as a descriptive, richer-time reference.
    eligible=c.read(OUT/"eligible.json");event_models={};event_predictions={}
    for mid in sorted({r["match_id"] for r in eligible}):
        training=[r for r in eligible if r["match_id"]!=mid];tested=[r for r in eligible if r["match_id"]==mid]
        event_models[mid]={"event_basic":c.fit_logistic(training,["event_distance_m","event_angle_rad"]),"event_header":c.fit_logistic(training,["event_distance_m","event_angle_rad","header"])}
        for name,model in event_models[mid].items():
            for row,value in zip(tested,c.predict(model,tested)):event_predictions.setdefault(row["shot_id"],{})[name]=value
    c.save(OUT/"event_reference_models.json",event_models)
    for col in ("event_basic","event_header"):values[col]=[event_predictions[r["shot_id"]][col] for r in rows]
    y=[r["goal"] for r in rows];metrics={a:c.metrics(y,v) for a,v in values.items()}
    pairs=[("diagram","raw"),("marked_crop","raw"),("photo_diagram","raw"),("spatial","basic")]
    contrasts={a+"_minus_"+b:interval(rows,values[a],values[b],e.SEED+100+i) for i,(a,b) in enumerate(pairs)}
    source_audit={r["shot_id"]:r for r in c.read(OUT/"source_audit.json")}
    kept=[i for i,r in enumerate(rows) if source_audit[r["shot_id"]]["source_contact_error_m"] is not None and source_audit[r["shot_id"]]["source_contact_error_m"]<=5]
    sensitivity={a:c.metrics([y[i] for i in kept],[v[i] for i in kept]) for a,v in values.items()}
    summary={"dataset":"SoccerTrack v2","model":p["model"],"shots":len(rows),"goals":sum(y),"matches":len({r["match_id"] for r in rows}),"metrics":metrics,"primary":contrasts,
        "sensitivity":{"shots":len(kept),"goals":sum(y[i] for i in kept),"criterion":"Contact-event vs tracked shooter location discrepancy at most5m;score-only subset,models unchanged.","metrics":sensitivity},
        "calls":len(ledger),"estimated_usd":sum(r["estimated_usd"] for r in ledger),"conservative_usd":sum(r["conservative_usd"] for r in ledger),"cap_usd":10.,"exclusions":c.read(OUT/"exclusions.json")}
    assert summary["conservative_usd"]<=10;c.save(OUT/"summary.json",summary)
    with (OUT/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["shot_id","match_id","goal"]+list(values));writer.writeheader()
        writer.writerows({"shot_id":r["shot_id"],"match_id":r["match_id"],"goal":r["goal"],**{a:v[i] for a,v in values.items()}} for i,r in enumerate(rows))
    preview(rows)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
