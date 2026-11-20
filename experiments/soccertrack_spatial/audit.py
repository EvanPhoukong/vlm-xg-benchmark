"""Offline provenance, leakage, input-hash and cost audit; no network requests."""
import base64
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np

import experiment as e

c,ROOT,OUT=e.c,e.ROOT,e.OUT


@lru_cache(maxsize=16)
def source_objects(start,end,mid,half,expected):
    filename=OUT/"source/ranges"/f"{mid}_{half}_{start}_{end}.bin"
    data=filename.read_bytes();assert hashlib.sha256(data).hexdigest()==expected
    text=data.decode("utf-8");pattern=re.compile(r'\{\s*"id"\s*:\s*(?:"[^"]+"|\d+)');decoder=json.JSONDecoder();cursor=0;result={}
    while (m:=pattern.search(text,cursor)):
        try:obj,last=decoder.raw_decode(text,m.start())
        except json.JSONDecodeError:break
        cursor=last
        if "image_id" in obj and "supercategory" in obj:result[str(obj["id"])]=obj
    return result


@lru_cache(maxsize=256)
def image_content(relative,expected):
    content=(ROOT/relative).read_bytes();assert hashlib.sha256(content).hexdigest()==expected
    return {"type":"image","source":{"type":"base64","media_type":"image/jpeg" if relative.endswith(".jpg") else "image/png","data":base64.b64encode(content).decode()}}


def request(job,p):
    content=[image_content(item["path"],item["sha256"]) for item in job["images"]]
    content.append({"type":"text","text":p["prompt"]})
    schema={"type":"object","properties":{"xg":{"type":"number"},"reason":{"type":"string"}},"required":["xg","reason"],"additionalProperties":False}
    return {"model":p["model"],"max_tokens":1024,"inference_geo":"global","output_config":{"effort":"low","format":{"type":"json_schema","schema":schema}},"messages":[{"role":"user","content":content}]}


def fingerprint(value):return hashlib.sha256(json.dumps(value,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def cost(u,conservative=False):
    ni,no=u["input_tokens"],u["output_tokens"];nw=u.get("cache_creation_input_tokens",0);nr=u.get("cache_read_input_tokens",0)
    assert all(type(v) is int and v>=0 for v in (ni,no,nw,nr))
    return ((ni+nw+nr)*2.5+no*10 if conservative else ni*2+nw*2.5+nr*.1+no*10)/1e6


def main():
    p=c.read(OUT/"protocol.json");choice=c.read(OUT/"resource_choice.json");eligible=c.read(OUT/"eligible.json");rows=c.read(OUT/"evaluation/selection.json");jobs=c.read(OUT/"evaluation/jobs.json");ledger=c.read(OUT/"attempts.json");checks=[]
    def check(name,ok):
        if not ok:raise ValueError("Audit failed:"+name)
        checks.append(name)
    for field,name in [("eligible_sha256","eligible.json"),("source_audit_sha256","source_audit.json"),("crop_audit_sha256","crop_audit.json"),("baseline_models_sha256","baseline_models.json")]:check(name+" frozen",c.sha(OUT/name)==p[field])
    for field,name in [("python_sha256","experiment.py"),("common_sha256","common.py"),("source_runner_sha256","source.mjs"),("runner_sha256","run.mjs")]:check(name+" frozen",c.sha(e.HERE/name)==p[field])
    check("fixed selected prefix",rows==eligible[:choice["n"]])
    check("selected cohort and schedule hashes",c.sha(OUT/"evaluation/jobs.json")==choice["jobs_sha256"] and c.sha(OUT/"evaluation/selection.json")==choice["selection_sha256"] and c.sha(OUT/"token_preview.json")==choice["preview_sha256"])
    check("complete paired predictions",len(ledger)==len(jobs)==len(rows)*4 and all(r["state"]=="completed" for r in ledger) and [r["job"] for r in ledger]==jobs)
    check("unique shots and ten match groups",len({r["shot_id"] for r in rows})==len(rows) and len({r["match_id"] for r in rows})==10)
    for i,row in enumerate(rows):
        order=e.ARMS[i%4:]+e.ARMS[:i%4]
        assert [j["arm"] for j in jobs[i*4:i*4+4]]==order
        for job in jobs[i*4:i*4+4]:
            expected=[row["images"]["raw"],row["images"]["diagram"]] if job["arm"]=="photo_diagram" else [row["images"][job["arm"]]]
            assert job["images"]==expected and job["shot_id"]==row["shot_id"]
    check("representation schedule reproducible",True)
    files=c.read(OUT/"source/files.json");snapshot_count=0
    for file in (OUT/"snapshots").glob("*.json"):
        snapshot=c.read(file);r=snapshot["source_range"];key=f"{snapshot['match']}_{snapshot['half']}"
        assert r["size"]==files[key]["size"] and snapshot["file_id"]==files[key]["file_id"]
        assert snapshot["full_sha256_verified"] is False
        records=source_objects(r["start"],r["end"],snapshot["match"],snapshot["half"],r["sha256"])
        for obj in snapshot["objects"]:assert records[str(obj["id"])]==obj and int(str(obj["image_id"])[-6:])==snapshot["annotation_frame"]
        assert snapshot["video_frame_index"]==snapshot["annotation_frame"]-1
        snapshot_count+=1
    check("all extracted snapshots reconstruct from hashed raw ranges",snapshot_count==612)
    for mid in sorted({r["match_id"] for r in eligible}):
        for name in ("homography.npy","camera_intrinsics.npz"):
            pointer=(c.DATA/"raw"/mid/(mid+"_"+name)).read_text();expected=re.search(r"oid sha256:([a-f0-9]+)",pointer)[1]
            assert c.sha(OUT/"source/calibration"/(mid+"_"+name))==expected
    check("all twenty small camera resources match canonical LFS SHA256",True)
    original_rows={r["shot_id"]:r for r in c.read(c.OUT/"shots.json")};frames={r["shot_id"]:r for r in c.read(ROOT/"results/soccertrack_extensive_v1/frames.json")}
    for row in eligible:
        original=original_rows[row["shot_id"]];assert row["goal"]==original["goal"] and row["actor_id"]==original["actor_id"]
        fresh,_=e.s.extract(original,frames[row["shot_id"]])
        assert fresh is not None and fresh["snapshot_sha256"]==row["snapshot_sha256"]
        assert row["annotation_frame"]==row["video_frame_index"]+1<row["event_frame"]
        assert row["shooter"]["id"]==row["actor_id"] and row["keeper"]["team"]!=row["shooter"]["team"]
        for name in e.FEATURES:assert abs(fresh[name]-row[name])<1e-10
    check("pre-shot positions,roles,actor identity and spatial features reproduce",True)
    # Directly test that labels/headers/event coordinates cannot influence rendered pixels.
    with tempfile.TemporaryDirectory(prefix="spatial_label_audit_") as directory:
        for row in sorted(eligible,key=lambda r:r["shot_id"])[:3]:
            altered=copy.deepcopy(row);altered.update(goal=1-row["goal"],header=1-row["header"],event_distance_m=999,event_angle_rad=-99,raw_width_fraction=-9,raw_attacking_length_fraction=9)
            dp=Path(directory)/"diagram.png";mp=Path(directory)/"marked.jpg"
            e.diagram(altered,dp);e.marked(altered,mp)
            assert c.sha(dp)==row["images"]["diagram"]["sha256"] and c.sha(mp)==row["images"]["marked_crop"]["sha256"]
    check("outcome/header/event-location mutation leaves input pixels unchanged",True)
    models=c.read(OUT/"baseline_models.json");baseline={r["shot_id"]:r for r in c.read(OUT/"baseline_predictions.json")}
    for mid,group in models.items():
        training=[r for r in eligible if r["match_id"]!=mid];test=[r for r in eligible if r["match_id"]==mid]
        for name,features in [("basic",e.FEATURES[:2]),("spatial",e.FEATURES)]:
            assert mid not in group[name]["train_matches"] and len(group[name]["train_matches"])==9
            refit=c.fit_logistic(training,features)
            np.testing.assert_allclose(c.predict(refit,test),[baseline[r["shot_id"]][name] for r in test],rtol=0,atol=1e-8)
    check("training-only scaling and complete-match exclusions reproduce",True)
    estimated,conservative=0.,0.
    for entry in ledger:
        assert fingerprint(request(entry["job"],p))==entry["request_sha256"]
        raw=c.read(OUT/entry["response_file"])
        assert raw["model"].startswith(p["model"]) and raw["type"]=="message" and raw["stop_reason"]=="end_turn"
        answer=json.loads("".join(item["text"] for item in raw["content"] if item["type"]=="text"))
        assert answer==entry["answer"] and set(answer)=={"xg","reason"} and type(answer["xg"]) in (int,float) and 0<=answer["xg"]<=1
        assert raw["usage"]==entry["usage"]
        ni=raw["usage"]["input_tokens"]+raw["usage"].get("cache_creation_input_tokens",0)+raw["usage"].get("cache_read_input_tokens",0)
        assert ni<=entry["input_bound"] and raw["usage"]["output_tokens"]<=1024
        assert abs(cost(raw["usage"])-entry["estimated_usd"])<1e-12 and abs(cost(raw["usage"],True)-entry["conservative_usd"])<1e-12
        assert entry["conservative_usd"]<=entry["reserved_usd"]
        estimated+=entry["estimated_usd"];conservative+=entry["conservative_usd"]
    check("all model request hashes,image bytes,answers,settings and costs reconcile",True)
    summary=c.read(OUT/"summary.json")
    check("reported costs and cap",abs(summary["estimated_usd"]-estimated)<1e-10 and abs(summary["conservative_usd"]-conservative)<1e-10 and conservative<=10)
    check("no paid runner active",not (OUT/"generation.lock").exists())
    result={"passed":True,"checks":checks,"check_count":len(checks),"snapshots_verified":snapshot_count,"requests_verified":len(ledger),"estimated_usd":estimated,"conservative_usd":conservative,"full_large_file_sha256_verified":False,"small_camera_resource_sha256_verified":True}
    c.save(OUT/"audit.json",result);print(json.dumps({k:v for k,v in result.items() if k!="checks"}),flush=True)


if __name__=="__main__":main()
