"""Annotation-assisted spatial xG experiment. No API calls on import."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("spatial_helpers",HERE/"common.py")
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
ROOT,OUT=s.ROOT,s.OUT
spec=importlib.util.spec_from_file_location("stat_helpers",ROOT/"experiments/soccertrack_comparison/common.py")
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
SEED=2026112009
ARMS=["raw","diagram","marked_crop","photo_diagram"]
FEATURES=["distance_m","angle_rad","keeper_line_distance_m","keeper_lateral_error_m","defenders_in_cone","nearest_defender_m"]
PROMPT="""You are a professional football (soccer) analyst. Estimate the probability that
the identified shot results in a goal (expected goals,xG),assuming an average
professional finisher. All inputs show the same situation shortly before contact.
The input may contain a panoramic photograph,a marked full-view/crop photograph,
a positional pitch diagram,or a photograph and diagram together.
In pitch diagrams,the attacking goal is at the top; gold star=shooter,blue circles=
teammates,red circles=opponents,orange square=opposing goalkeeper.
In annotated photos,a yellow S marks the shooter,an orange GK marks the opposing
goalkeeper,and a cyan line marks the attacking goal mouth. Plain photographs have
no added markers. Diagram marker colors refer to the overlay,not jersey colors.
Use only the shown situation. Body part,ball height and previous motion are unknown.
Reply with JSON only:{"xg":<number between0 and1>,"reason":"<one short sentence>"}."""


def star(draw,x,y,r=9):
    points=[]
    for i in range(10):
        a=-math.pi/2+i*math.pi/5;rr=r if i%2==0 else r*.42
        points.append((x+rr*math.cos(a),y+rr*math.sin(a)))
    draw.polygon(points,fill="#ffd43b",outline="#151b16",width=1)


def diagram(row,path):
    im=Image.new("RGB",(640,960),"#fafffa");d=ImageDraw.Draw(im)
    left,right,top,bottom=46,594,58,904
    def point(x,y):return left+(row["goal_sign"]*y+34)/68*(right-left),top+(52.5-row["goal_sign"]*x)/105*(bottom-top)
    def world(lateral,forward):return left+(lateral+34)/68*(right-left),top+forward/105*(bottom-top)
    def line(points,width=2):d.line([world(*p) for p in points],fill="#57645a",width=width)
    line([(-34,0),(34,0),(34,105),(-34,105),(-34,0)])
    line([(-34,52.5),(34,52.5)])
    for goal in (0,105):
        sign=1 if goal==0 else -1
        line([(-20.16,goal),(-20.16,goal+sign*16.5),(20.16,goal+sign*16.5),(20.16,goal)])
        line([(-9.16,goal),(-9.16,goal+sign*5.5),(9.16,goal+sign*5.5),(9.16,goal)])
        line([(-3.66,goal),(-3.66,goal-sign*2.5),(3.66,goal-sign*2.5),(3.66,goal)],3)
        x,y=world(0,goal+sign*11);d.ellipse((x-2,y-2,x+2,y+2),fill="#57645a")
    line([(9.15*math.cos(t),52.5+9.15*math.sin(t)) for t in np.linspace(0,2*math.pi,120)])
    for player in row["players"]:
        x,y=point(player["x"],player["y"])
        if not left-5<=x<=right+5 or not top-5<=y<=bottom+5:continue
        same=player["team"]==row["shooter"]["team"]
        box=(x-4,y-4,x+4,y+4)
        if not same and player["role"]=="goalkeeper":d.rectangle(box,fill="#ef9611",outline="black",width=1)
        else:d.ellipse(box,fill="#4169e1" if same else "#df3038",outline="black",width=1)
    star(d,*point(row["shooter"]["x"],row["shooter"]["y"]))
    im.save(path)


def marked(row,path):
    with Image.open(ROOT/row["raw_image"]) as original:im=original.convert("RGB")
    cam=s.camera(row["match_id"]);sx=im.width/cam["width"];sy=im.height/cam["height"]
    if abs(sx/sy-1)>.015:raise ValueError("Image geometry does not match calibration map shape")
    d=ImageDraw.Draw(im)
    def loc(player):
        box=player["bbox"]
        return (box["x"]+box["w"]/2)*sx,(box["y"]+box["h"]/2)*sy
    points=[]
    for player,color,label in [(row["shooter"],"#ffd400","S"),(row["keeper"],"#ff9618","GK")]:
        x,y=loc(player);points.append((x,y));d.ellipse((x-13,y-13,x+13,y+13),outline=color,width=3);d.text((x+15,y-20),label,fill=color,stroke_width=1,stroke_fill="black")
    goal=[]
    for y in (-3.66,3.66):
        u,v=s.project(row["goal_sign"]*52.5,y,cam);goal.append((u*sx,v*sy))
    d.line(goal,fill="#15c7e8",width=3);points+=goal
    posts=((row["goal_sign"]*52.5,-3.66),(row["goal_sign"]*52.5,3.66))
    for p in row["players"]:
        if p["team"]!=row["shooter"]["team"] and s.triangle((p["x"],p["y"]),(row["shooter"]["x"],row["shooter"]["y"]),*posts):points.append(loc(p))
    margin=60
    box=(max(0,math.floor(min(x for x,y in points)-margin)),max(0,math.floor(min(y for x,y in points)-margin)),min(im.width,math.ceil(max(x for x,y in points)+margin)),min(im.height,math.ceil(max(y for x,y in points)+margin)))
    if box[2]-box[0]<50 or box[3]-box[1]<50:raise ValueError("Invalid assisted crop")
    crop=im.crop(box);crop.thumbnail((1280,720),Image.Resampling.LANCZOS)
    # Upscale the selected crop for a readable presentation; this adds no source detail.
    ratio=min(1280/crop.width,720/crop.height);crop=crop.resize((round(crop.width*ratio),round(crop.height*ratio)),Image.Resampling.LANCZOS)
    full=im.copy();full.thumbnail((1280,640),Image.Resampling.LANCZOS)
    canvas=Image.new("RGB",(1280,full.height+crop.height+48),"white")
    canvas.paste(full,((1280-full.width)//2,24));canvas.paste(crop,((1280-crop.width)//2,full.height+48))
    cd=ImageDraw.Draw(canvas);cd.text((12,6),"Full view",fill="black");cd.text((12,full.height+30),"Shot area",fill="black")
    canvas.save(path,quality=94,subsampling=0)
    return {"crop_box":list(box),"camera_width":cam["width"],"camera_height":cam["height"],"scale_x":sx,"scale_y":sy}


def prepare():
    if (OUT/"protocol.json").exists():raise ValueError("Study already frozen")
    rows=c.read(c.OUT/"shots.json");frames={r["shot_id"]:r for r in c.read(ROOT/"results/soccertrack_extensive_v1/frames.json")}
    selected=[];audits=[];crop_audit={};exclusions=[]
    for old in rows:
        row,audit=s.extract(old,frames[old["shot_id"]]);record={"shot_id":old["shot_id"],"match_id":old["match_id"],**audit}
        if row is None:exclusions.append(record);continue
        if not audit["keeper_direction_agrees"]:raise ValueError("GSR team-side direction conflicts with goalkeeper positions")
        if not row["shooter"]["bbox"] or not row["keeper"]["bbox"]:exclusions.append({**record,"reason":"image_box_missing"});continue
        if c.sha(ROOT/row["raw_image"])!=row["raw_image_sha256"]:raise ValueError("Archived query image changed")
        dp=OUT/"images"/(row["shot_id"]+"_diagram.png");mp=OUT/"images"/(row["shot_id"]+"_marked.jpg")
        diagram(row,dp);crop_audit[row["shot_id"]]=marked(row,mp)
        row["images"]={"raw":{"path":row["raw_image"],"sha256":row["raw_image_sha256"]},"diagram":{"path":str(dp.relative_to(ROOT)).replace("\\","/"),"sha256":c.sha(dp)},"marked_crop":{"path":str(mp.relative_to(ROOT)).replace("\\","/"),"sha256":c.sha(mp)}}
        selected.append(row);audits.append(record)
    selected=random.Random(SEED).sample(sorted(selected,key=lambda r:r["shot_id"]),len(selected))
    c.freeze(OUT/"eligible.json",selected);c.freeze(OUT/"source_audit.json",audits);c.freeze(OUT/"exclusions.json",exclusions);c.freeze(OUT/"crop_audit.json",crop_audit)
    models={};predictions=[]
    for mid in sorted({r["match_id"] for r in selected}):
        training=[r for r in selected if r["match_id"]!=mid];tested=[r for r in selected if r["match_id"]==mid]
        models[mid]={"basic":c.fit_logistic(training,FEATURES[:2]),"spatial":c.fit_logistic(training,FEATURES),"constant":sum(r["goal"] for r in training)/len(training)}
        p=c.predict(models[mid]["basic"],tested);q=c.predict(models[mid]["spatial"],tested)
        predictions.extend({"shot_id":r["shot_id"],"match_id":mid,"goal":r["goal"],"basic":a,"spatial":b,"constant":models[mid]["constant"]} for r,a,b in zip(tested,p,q))
    c.freeze(OUT/"baseline_models.json",models);c.freeze(OUT/"baseline_predictions.json",predictions)
    scores={a:c.metrics([r["goal"] for r in predictions],[r[a] for r in predictions]) for a in ("basic","spatial","constant")}
    c.save(OUT/"baseline_metrics.json",scores)
    protocol={"dataset":"SoccerTrack v2 v1.2 public-author mirror","model":"claude-sonnet-5-5","arms":ARMS,"prompt":PROMPT,"seed":SEED,
        "input":"Single pre-contact snapshot,using GSR player positions/roles and actor ID. Raw PNG unchanged. Diagram and marked photo are annotation-assisted. Calibration H,K,D,Knew reprojects physical goal posts for the photo crop.",
        "timing":"GSR suffix frame=decoded video index+1. Use the exact saved frame0.32 seconds before event. Contact frame retrieved only for source-alignment audit,never model inputs. No interpolated auxiliary ball track.",
        "sample_rule":"Use all source-audit-eligible shots if the pre-generation token forecast fits$10;otherwise choose a deterministic natural prefix from[240,200,150,120,100,80] before accuracy inspection. All four visual conditions fixed;no method selection on scores.",
        "statistics":"4 primary Brier contrasts:diagram-raw,marked_crop-raw,photo_diagram-raw,spatial_logistic-basic_logistic.98.75% whole-match percentile intervals,20000 resamples. Overlapping other-match fits are held fixed in bootstrap.",
        "statistical_features":FEATURES,"baseline":"L2 logistic with ridge1,training-only scaling,leave-one-complete-match-out. Primary basic and spatial baselines share the same pre-contact coordinates and match exclusions.",
        "cap_usd":10.,"rates_per_million":{"input":2.,"cache_write":2.5,"cache_read":.1,"output":10.},"effort":"low","max_output_tokens":1024,"workers":6,"retries":0,"global_inference":True,
        "budget":"New spatial experiment only. Conservative completed exposure charges all input at1.25x uncached input plus actual output;pending/uncertain calls reserve counted input+1024 plus maximum output. Stop on unresolved charges or cap.",
        "scope":"Exploratory reused matches,annotation-assisted information-value test,not end-to-end player detection. No reference xG,query outcomes,body part,identities or numeric coordinates enter visual requests. No calibration of outputs,teacher probabilities,ICL or weight updates.",
        "source":"Only verified byte ranges of public mirror GSR files retrieved;full large-file SHA hashes are not claimed verified. Calibration homography and camera-intrinsic files match local canonical LFS SHA256 exactly. Original pointers and historical results preserved.",
        "source_geometry_caveat":"Raw event locations and tracking at contact can disagree;record errors and report sensitivity excluding contact-location discrepancies over5m. Primary eligibility does not select on this discrepancy or outcomes.",
        "eligible_sha256":c.sha(OUT/"eligible.json"),"source_audit_sha256":c.sha(OUT/"source_audit.json"),"crop_audit_sha256":c.sha(OUT/"crop_audit.json"),"baseline_models_sha256":c.sha(OUT/"baseline_models.json"),
        "python_sha256":c.sha(__file__),"common_sha256":c.sha(HERE/"common.py"),"source_runner_sha256":c.sha(HERE/"source.mjs"),"runner_sha256":c.sha(HERE/"run.mjs"),
        "attribution":"SoccerTrack v2 by Atom Scott,Ikuma Uchida,Kento Kuroda,Yufi Kim,Keisuke Fujii. CC BY4.0. https://huggingface.co/datasets/atomscott/soccertrack-v2"}
    c.freeze(OUT/"protocol.json",protocol)
    print(json.dumps({"eligible":len(selected),"excluded":len(exclusions),"baseline_metrics":scores,"source_contact_error_m":{"median":float(np.median([a['source_contact_error_m'] for a in audits if a['source_contact_error_m'] is not None])),"over5m":sum((a['source_contact_error_m'] or 0)>5 for a in audits)},"calibration_median_error_px":float(np.median([a['calibration_median_error_px'] for a in audits]))}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare"]);globals()[p.parse_args().action]()
