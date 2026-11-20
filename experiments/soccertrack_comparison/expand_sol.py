"""Expand the unchanged Sol image-only method to all 306 audited SoccerTrack shots."""
import argparse
from collections import defaultdict
from fractions import Fraction
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import common as c
import run as engine

OLD = c.OUT
OUT = c.ROOT / "results/soccertrack_sol_expansion_v1"
REVISION = "325bd4c57ee8e135ab652f3921d274526ada496b"
MODEL = c.MODELS["sol"]
INPUT_BOUND = 4096


def old_ledger():
    path = OLD/"ledger.json"
    ledger = c.read(path) if path.is_file() else []
    if any(e["state"] != "completed" for e in ledger):
        raise ValueError("An earlier comparison has an unresolved request; reconcile it first")
    return ledger, c.sha(path) if path.is_file() else None


def prepare():
    OUT.mkdir(exist_ok=True)
    (OUT/"frames").mkdir(exist_ok=True)
    rows = c.read(OLD/"shots.json")
    reused = c.read(OLD/"reused_sol.json")
    cached = {r["shot_id"]:r for r in reused}
    pilot = {r["shot_id"]:r for r in c.read(c.PILOT)["shots"]}
    previous, previous_hash = old_ledger()
    previous_protocol = c.read(OLD/"protocol.json")
    prior = previous_protocol["budget"]["prior_conservative_usd"][MODEL["model"]]
    prior += sum(e["charged_bound_usd"] for e in previous if e["model"] == MODEL["model"])
    jobs = []
    for row in sorted(rows,key=lambda r:r["shot_id"]):
        sid = row["shot_id"]
        if sid in cached:
            image = str((c.PILOT.parent/pilot[sid]["frame"]).relative_to(c.ROOT)).replace("\\","/")
            if c.sha(c.ROOT/image) != pilot[sid]["sha256"]:
                raise ValueError("Earlier image changed")
            frame_index = pilot[sid]["frame_index"]
        else:
            image = f"results/soccertrack_sol_expansion_v1/frames/{sid}.png"
            frame_index = row["event_frame"]-9  # event frame is one-based; image index is zero-based.
        if frame_index != row["event_frame"]-9 or frame_index < 0:
            raise ValueError("Invalid past-only frame index")
        jobs.append({"shot_id":sid,"match_id":row["match_id"],"half":row["half"],
            "event_frame":row["event_frame"],"frame_index":frame_index,"image":image,"reuse_original":sid in cached})
    if len(jobs)!=306 or len(cached)!=30:
        raise ValueError("Audited cohort differs; use a new protocol rather than silently changing size")
    worst = (INPUT_BOUND*MODEL["rates"][0]+c.MAX_OUTPUT*MODEL["rates"][2])/1e6
    plan = {"dataset":"SoccerTrack v2 v1.2","dataset_revision":REVISION,"model":MODEL,
        "cohort":jobs,"shots":len(jobs),"goals":sum(r["goal"] for r in rows),"matches":len({r["match_id"] for r in rows}),
        "reused_original":len(cached),"new_predictions":len(jobs)-len(cached),
        "prompt":c.PROMPT,"reasoning_effort":"low","image_detail":"high","max_output_tokens":c.MAX_OUTPUT,
        "primary_comparison":"All 306 Sol probabilities versus existing leave-one-match-out distance/angle logistic predictions; each baseline fold fits the other 9 matches.",
        "secondary":"Official 64-shot test subset versus the original six-training-match baseline; original 30 and new 276 reported separately.",
        "selection":"All audited unique SHOT/GOAL events. No tuning, replacement by score, goal balancing, repeated-reading averaging or output calibration.",
        "interpretation":"More evaluation shots improve precision, not the underlying model. The old 30-shot score may move either way. Previously inspected source matches; exploratory expansion, not untouched confirmation.",
        "budget":{"per_model_cap_usd":10.,"prior_conservative_usd":prior,"per_call_input_bound":INPUT_BOUND,
                  "max_new_reservations_usd":worst*(len(jobs)-len(cached)),"automatic_retries":0,
                  "scope":"This expansion plus logged earlier SoccerTrack Sol usage, not an account-wide balance."},
        "source_hashes":{name:c.sha(OLD/name) for name in ("shots.json","source_audit.json","lomo_predictions.json","lomo_models.json","baseline_models.json","reused_sol.json")},
        "legacy_ledger_sha256":previous_hash,
        "engine_hashes":{name:c.sha(Path(__file__).with_name(name)) for name in ("expand_sol.py","run.py","common.py")},
        "frame_preparation":"Seek from the actual first decoded frame PTS, then decode to the exact 25fps target timestamp, 8 frames before contact. Do not bulk-download all videos (about 77 GiB).",
        "frame_caveat":"Original 30 PNGs are retained byte-for-byte. New PNGs use PyAV RGB plus Pillow LANCZOS at longest side 1536; the original resizing implementation is not established.",
        "limitations":previous_protocol["scope_caveats"]}
    c.freeze(OUT/"plan.json",plan)
    c.freeze(OUT/"reused_sol.json",reused)
    c.save(OUT/"status.json",{"status":"planned","shots":len(jobs),"new_predictions":len(jobs)-len(cached),
        "new_generation_calls":0,"missing_new_frames":sum(not (c.ROOT/j["image"]).is_file() for j in jobs if not j["reuse_original"])})
    print(json.dumps({k:plan[k] for k in ("shots","goals","matches","reused_original","new_predictions","budget")},indent=2))


def check_plan():
    plan = c.read(OUT/"plan.json")
    for name,expected in plan["source_hashes"].items():
        if c.sha(OLD/name)!=expected:
            raise ValueError("Frozen source changed: "+name)
    for name,expected in plan["engine_hashes"].items():
        if c.sha(Path(__file__).with_name(name))!=expected:
            raise ValueError("Frozen expansion implementation changed: "+name)
    _,ledger_hash = old_ledger()
    if ledger_hash != plan["legacy_ledger_sha256"]:
        raise ValueError("Earlier spending changed after planning; reconcile the shared Sol budget")
    return plan


def frame_pts(first_pts,index,time_base):
    delta = Fraction(index,25)/Fraction(time_base.numerator,time_base.denominator)
    if delta.denominator!=1:
        raise ValueError("Frame timing does not land on an exact stream tick")
    return first_pts+int(delta)


def frames():
    plan = check_plan()
    if importlib.util.find_spec("av") is None:
        c.save(OUT/"frame_status.json",{"status":"blocked_missing_pyav","new_generation_calls":0})
        raise SystemExit("PyAV is required for frame-accurate selective video decoding; no videos or paid requests submitted")
    import av
    from PIL import Image
    (OUT/"frames").mkdir(exist_ok=True)
    groups = defaultdict(list)
    for job in plan["cohort"]:
        if not job["reuse_original"] and not (c.ROOT/job["image"]).is_file():
            groups[job["match_id"],job["half"]].append(job)
    audit = c.read(OUT/"frame_audit.json") if (OUT/"frame_audit.json").is_file() else {}
    for (mid,half),jobs in sorted(groups.items()):
        suffix = "1st" if half==1 else "2nd"
        relative = f"videos/{mid}/{mid}_panorama_{suffix}_half.mp4"
        local = c.DATA/relative
        uri = str(local) if local.stat().st_size>200 else f"https://huggingface.co/datasets/atomscott/soccertrack-v2/resolve/{REVISION}/{relative}"
        # HTTPS seek requests avoid saving every complete half-match video locally.
        with av.open(uri,timeout=(30.,60.)) as container:
            stream = container.streams.video[0]
            if not stream.average_rate or Fraction(stream.average_rate.numerator,stream.average_rate.denominator)!=25:
                raise ValueError("Unexpected source frame rate")
            first = next(container.decode(video=0))
            if first.pts is None or not stream.time_base:
                raise ValueError("Cannot establish first decoded frame timing")
            first_pts = first.pts
            for job in sorted(jobs,key=lambda j:j["frame_index"]):
                target = frame_pts(first_pts,job["frame_index"],stream.time_base)
                container.seek(target,stream=stream,backward=True,any_frame=False)
                found = None
                for frame in container.decode(video=0):
                    if frame.pts is None:
                        continue
                    if frame.pts==target:
                        found=frame
                        break
                    if frame.pts>target:
                        break
                if found is None:
                    raise ValueError("Could not locate exact pre-shot frame: "+job["shot_id"])
                image = found.to_image().convert("RGB")
                width,height=image.size
                ratio=min(1.,1536/max(width,height))
                image=image.resize((round(width*ratio),round(height*ratio)),Image.Resampling.LANCZOS)
                image.save(c.ROOT/job["image"])
                audit[job["shot_id"]]={"video":relative,"revision":REVISION,"first_decoded_pts":first_pts,
                    "selected_pts":target,"time_base":str(stream.time_base),"zero_based_index":job["frame_index"],
                    "sha256":c.sha(c.ROOT/job["image"]),"size":list(image.size)}
                c.save(OUT/"frame_audit.json",audit)
                print("Prepared",job["shot_id"],flush=True)
    cohort=[]
    for job in plan["cohort"]:
        path=c.ROOT/job["image"]
        if not job["reuse_original"] and job["shot_id"] not in audit:
            raise ValueError("Unprovenanced new image")
        expected=next(r["image_sha256"] for r in c.read(OUT/"reused_sol.json") if r["shot_id"]==job["shot_id"]) if job["reuse_original"] else audit[job["shot_id"]]["sha256"]
        if c.sha(path)!=expected:
            raise ValueError("Image changed after extraction")
        cohort.append({**job,"sha256":expected})
    c.freeze(OUT/"frames_manifest.json",cohort)
    c.save(OUT/"frame_status.json",{"status":"complete","verified_frames":len(cohort),"new_generation_calls":0})


def runtime_protocol():
    plan=check_plan()
    if not (OUT/"frames_manifest.json").is_file():
        raise ValueError("Extract and audit the additional frames before any paid inference")
    cohort=c.read(OUT/"frames_manifest.json")
    if [{k:r[k] for k in plan["cohort"][0]} for r in cohort]!=plan["cohort"]:
        raise ValueError("Frame manifest does not match the full frozen cohort")
    for item in cohort:
        if c.sha(c.ROOT/item["image"])!=item["sha256"]:
            raise ValueError("Image hash changed")
    protocol={"cohort":cohort,"budget":{"prior_conservative_usd":{MODEL["model"]:plan["budget"]["prior_conservative_usd"]}},
        "plan_sha256":c.sha(OUT/"plan.json"),"frames_sha256":c.sha(OUT/"frames_manifest.json")}
    c.freeze(OUT/"protocol.json",protocol)
    return protocol


def schedule(protocol,arms):
    return [{"arm":"sol","condition":"original",**item} for item in protocol["cohort"] if not item["reuse_original"]]


def paid_run(execute=False):
    # Acquire the existing campaign lock too, preventing simultaneous legacy dispatch.
    lock=OLD/"runner.lock"
    handle=lock.open("x")
    with handle:
        handle.write(json.dumps({"role":"sol_expansion","pid":os.getpid()}));handle.flush();os.fsync(handle.fileno())
        original_api=engine.api
        def bounded_api(endpoint,key,data=None):
            result=original_api(endpoint,key,data)
            if endpoint=="responses/input_tokens":
                count=result["input_tokens"]
                if type(count) is not int or max(count+512,math.ceil(count*1.25))>INPUT_BOUND:
                    raise ValueError("Input exceeds the fixed expansion reservation bound; no generation dispatched")
            return result
        engine.OUT=OUT
        engine.MODELS={"sol":MODEL}
        engine.check_protocol=runtime_protocol
        engine.schedule=schedule
        engine.api=bounded_api
        previous=sys.argv
        sys.argv=[str(Path(engine.__file__)),"--models","sol","--execute" if execute else "--preflight"]
        try:
            engine.main()
        finally:
            sys.argv=previous
            handle.close()
            lock.unlink()


def analyze():
    plan=check_plan()
    cached={r["shot_id"]:r["vlm_xg"] for r in c.read(OUT/"reused_sol.json")}
    ledger=c.read(OUT/"ledger.json") if (OUT/"ledger.json").is_file() else []
    cached.update({e["shot_id"]:e["xg"] for e in ledger if e["state"]=="completed"})
    complete=len(cached)==plan["shots"]
    result={"status":"complete" if complete else "incomplete","completed_predictions":len(cached),
        "planned_predictions":plan["shots"],"new_generation_calls":len(ledger),
        "new_estimated_usd":sum(e.get("estimated_usd",0.) for e in ledger),"scores":None}
    if complete:
        rows=c.read(OLD/"lomo_predictions.json")
        for row in rows:
            mid=row["match_id"]
            models=c.read(OLD/"lomo_models.json")[mid]
            if any(mid in m["train_matches"] for m in models.values()):
                raise ValueError("Baseline contains the evaluated match")
        original_ids={s["shot_id"] for s in c.read(OUT/"reused_sol.json")}
        groups={"all_306":rows,"original_30":[r for r in rows if r["shot_id"] in original_ids],
                "additional_276":[r for r in rows if r["shot_id"] not in original_ids],
                "official_test_64":[r for r in rows if r["match_id"] in c.TEST]}
        result["scores"]={}
        original_models=c.read(OLD/"baseline_models.json")["models"]
        for name,selected in groups.items():
            y=[r["goal"] for r in selected]
            baseline=(c.predict(original_models["logistic_distance_angle"],selected) if name=="official_test_64" else [r["logistic_distance_angle"] for r in selected])
            result["scores"][name]={"sol":c.metrics(y,[cached[r["shot_id"]] for r in selected]),
                "logistic":c.metrics(y,baseline),"baseline_training":"original six training matches" if name=="official_test_64" else "other nine matches in each fold"}
    c.save(OUT/"summary.json",result)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","frames","run","analyze"])
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.stage=="prepare":prepare()
    elif args.stage=="frames":frames()
    elif args.stage=="run":paid_run(args.execute)
    else:analyze()


if __name__=="__main__":
    main()
