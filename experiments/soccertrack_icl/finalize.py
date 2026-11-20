"""Offline reconciliation, full-request audit and report for the completed Sol study."""
import base64
from functools import lru_cache
import hashlib
import json
import math

import numpy as np
from PIL import Image,ImageChops

import experiment as e

c=e.c
PARENT=e.OUT
RESUME=e.ROOT/"results/soccertrack_icl_resume_v1"
SONNET=e.ROOT/"results/soccertrack_icl_sonnet_v1"
OUT=e.ROOT/"results/soccertrack_icl_completed_v1"


@lru_cache(maxsize=256)
def image_part(relative,expected):
    data=(e.ROOT/relative).read_bytes()
    assert hashlib.sha256(data).hexdigest()==expected
    mime="jpeg" if relative.endswith(".jpg") else "png"
    return {"type":"input_image","detail":"high","image_url":"data:image/"+mime+";base64,"+base64.b64encode(data).decode()}


def request(job,p,pages):
    parts=[]
    if job["arm"]!="zero":
        parts.append({"type":"input_text","text":p["teacher_intro"] if "teacher" in job["arm"] else p["outcome_intro"]})
        parts.extend(image_part(page["path"],page["sha256"]) for page in pages[job["match_id"]][job["arm"]])
        parts.append({"type":"input_text","text":"Historical examples END. The following separate image is the NEW QUERY; its outcome is unknown."})
    parts.extend([image_part(job["image"],job["sha256"]),{"type":"input_text","text":p["prompt"]}])
    return {"model":p["model"],"store":False,"service_tier":"default","reasoning":{"effort":"low"},"max_output_tokens":1024,
        "input":[{"role":"user","content":parts}],"text":{"format":{"type":"json_schema","name":"shot_prediction","strict":True,
        "schema":{"type":"object","properties":{"xg":{"type":"number","minimum":0,"maximum":1},"reason":{"type":"string"}},"required":["xg","reason"],"additionalProperties":False}}}}


def fingerprint(body):
    return hashlib.sha256(json.dumps(body,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def openai_cost(u,conservative=False):
    ni,no=u["input_tokens"],u["output_tokens"];nr=(u.get("input_tokens_details") or {}).get("cached_tokens",0)
    assert all(type(v) is int for v in (ni,no,nr)) and 0<=nr<=ni and no>=0
    return (ni*2.5+no*10 if conservative else (ni-nr)*2+nr*.1+no*10)/1e6


def sonnet_cost(u,conservative=False):
    ni,no=u["input_tokens"],u["output_tokens"];nw=u.get("cache_creation_input_tokens",0);nr=u.get("cache_read_input_tokens",0)
    assert all(type(v) is int and v>=0 for v in (ni,no,nw,nr))
    return ((ni+nw+nr)*2.5+no*10 if conservative else ni*2+nw*2.5+nr*.1+no*10)/1e6


def sonnet_request(job,p,pages):
    content=[]
    for part in request(job,p,pages)["input"][0]["content"]:
        if part["type"]=="input_text":content.append({"type":"text","text":part["text"]})
        else:
            header,data=part["image_url"].split(",",1)
            content.append({"type":"image","source":{"type":"base64","media_type":header[5:].split(";")[0],"data":data}})
    schema={"type":"object","properties":{"xg":{"type":"number"},"reason":{"type":"string"}},"required":["xg","reason"],"additionalProperties":False}
    return {"model":"claude-sonnet-5-5","max_tokens":1024,"inference_geo":"global","output_config":{"effort":"low","format":{"type":"json_schema","schema":schema}},"messages":[{"role":"user","content":content}]}


def main():
    p=c.read(PARENT/"protocol.json");pages=c.read(PARENT/"pages.json");examples=c.read(PARENT/"examples.json")
    rows=c.read(PARENT/"evaluation/selection.json");jobs=c.read(PARENT/"evaluation/jobs.json")
    original=c.read(PARENT/"attempts.json");continued=c.read(RESUME/"attempts.json");startup=c.read(SONNET/"attempts.json");probe=c.read(PARENT/"recovery_probe.json")
    successful=[{**x,"response_root":"results/soccertrack_icl_v1"} for x in original if x["state"]=="completed"]
    successful += [{**x,"response_root":"results/soccertrack_icl_resume_v1"} for x in continued]
    checks=[]
    def check(name,value):
        if not value:raise ValueError("Audit failed: "+name)
        checks.append(name)
    check("complete original480-prediction schedule",len(successful)==480 and [x["job"] for x in successful]==jobs and all(x["state"]=="completed" for x in successful))
    check("80 unique queries and all10 matches",len(rows)==80 and len({r["shot_id"] for r in rows})==80 and len({r["match_id"] for r in rows})==10)
    check("source experiment code unchanged",c.sha(e.__file__)==p["python_sha256"])
    check("original runner unchanged",c.sha(e.ROOT/"experiments/soccertrack_icl/run.mjs")==p["runner_sha256"])
    for field,name in [("pages_sha256","pages.json"),("examples_sha256","examples.json"),("pool_sha256","pool.json"),("baselines_sha256","baselines.json"),("preview_sha256","preview/jobs.json")]:check(name+" frozen",c.sha(PARENT/name)==p[field])
    for name,code in [("protocol.json","resume.mjs"),("protocol_fast.json","resume_fast.mjs"),("protocol_robust.json","resume_robust.mjs")]:
        rp=c.read(RESUME/name)
        check(code+" frozen",c.sha(e.ROOT/"experiments/soccertrack_icl"/code)==rp["runner_sha256"])
        check(name+" preserves previous accounting",c.sha(PARENT/"attempts.json")==rp["parent_ledger_sha256"] and c.sha(PARENT/"recovery_probe.json")==rp["recovery_probe_sha256"] and c.sha(SONNET/"attempts.json")==rp["sonnet_ledger_sha256"])
    source_rows=c.read(c.OUT/"shots.json");source_by_id={r["shot_id"]:r for r in source_rows};frames={r["shot_id"]:r for r in c.read(e.OLD/"frames.json")}
    check("source manifests unchanged",all(c.sha(path)==p["source_hashes"][name] for name,path in [("shots",c.OUT/"shots.json"),("audit",c.OUT/"source_audit.json"),("frames",e.OLD/"frames.json"),("frame_audit",e.OLD/"frame_audit.json")]))
    for audit in c.read(c.OUT/"source_audit.json"):
        mid=audit["match_id"]
        assert c.sha(c.DATA/f"bas/{mid}/{mid}_12_class_events.json")==audit["bas_sha256"]
        assert c.sha(c.DATA/f"raw/{mid}/{mid}_player_nodes.csv")==audit["nodes_sha256"]
    check("released BAS and raw annotation hashes unchanged",True)
    check("selected cohort is frozen outcome-blind prefix",rows==c.read(PARENT/"pool.json")[:80])
    check("original randomized schedule reproducible",jobs==e.schedule(rows,frames))
    baselines=c.read(PARENT/"baselines.json");fold_counts={}
    for mid,ex in examples.items():
        assert len(ex)==16 and len({r["shot_id"] for r in ex})==16 and all(r["match_id"]!=mid for r in ex)
        assert all(r["goal"]==source_by_id[r["shot_id"]]["goal"] for r in ex)
        assert sum(r["goal"] for r in ex)==sum(r["shuffled_goal"] for r in ex)
        if len({r["goal"] for r in ex})>1:assert any(r["goal"]!=r["shuffled_goal"] for r in ex)
        assert sorted(r["teacher"] for r in ex)==sorted(r["shuffled_teacher"] for r in ex)
        assert any(r["teacher"]!=r["shuffled_teacher"] for r in ex)
        model=baselines[mid]["image"]
        assert mid not in model["train_matches"] and len(model["train_matches"])==9
        predicted=[round(v,4) for v in c.predict(model,[source_by_id[r["shot_id"]] for r in ex])]
        assert predicted==[r["teacher"] for r in ex]
        fold_counts[mid]={"example_goals8":sum(r["goal"] for r in ex[:8]),"example_goals16":sum(r["goal"] for r in ex),"shuffled_outcomes_changed":any(r["goal"]!=r["shuffled_goal"] for r in ex)}
        assert pages[mid]["outcome8"][0]["sha256"]==pages[mid]["outcome16"][0]["sha256"]
        for correct,shuffled in [("outcome16","shuffled_outcome16"),("teacher16","shuffled_teacher16")]:
            for a,b in zip(pages[mid][correct],pages[mid][shuffled]):
                with Image.open(e.ROOT/a["path"]) as im1,Image.open(e.ROOT/b["path"]) as im2:
                    for y in range(0,im1.height,408):assert ImageChops.difference(im1.crop((0,y,1536,y+384)),im2.crop((0,y,1536,y+384))).getbbox() is None
    check("all example folds exclude their query match",True);check("binary labels and teacher labels reconcile",True);check("shuffle controls preserve distributions and image pixels",True);check("nested8-example condition shares exact first page",True)
    estimated_sol,conservative_sol=0.,0.
    for entry in successful:
        job=entry["job"];assert job["match_id"]==source_by_id[job["shot_id"]]["match_id"]
        assert frames[job["shot_id"]]["frame_index"]<frames[job["shot_id"]]["event_frame"]
        assert fingerprint(request(job,p,pages))==entry["request_sha256"]
        raw=c.read(e.ROOT/entry["response_root"]/entry["response_file"])
        assert raw["status"]=="completed" and raw["model"].startswith("gpt-6.2-sol") and raw["reasoning"]["effort"]=="low"
        answer=json.loads("".join(v["text"] for o in raw["output"] if o["type"]=="message" for v in o["content"] if v["type"]=="output_text"))
        assert answer==entry["answer"] and set(answer)=={"xg","reason"} and type(answer["xg"]) in (int,float) and math.isfinite(answer["xg"]) and 0<=answer["xg"]<=1
        assert raw["usage"]==entry["usage"] and raw["usage"]["input_tokens"]<=entry["input_bound"] and raw["usage"]["output_tokens"]<=1024
        assert math.isclose(openai_cost(raw["usage"]),entry["estimated_usd"],abs_tol=1e-12) and math.isclose(openai_cost(raw["usage"],True),entry["conservative_usd"],abs_tol=1e-12)
        assert entry["conservative_usd"]<=entry["reserved_usd"]
        estimated_sol+=entry["estimated_usd"];conservative_sol+=entry["conservative_usd"]
    check("all 480 Sol request hashes,pixels,raw answers,settings and costs verified",True)
    check("exactly4 separate Sonnet startup results",len(startup)==4 and all(x["state"]=="completed" for x in startup))
    son_est,son_cons=0.,0.
    for entry in startup:
        assert fingerprint(sonnet_request(entry["job"],p,pages))==entry["request_sha256"]
        raw=c.read(SONNET/entry["response_file"]);assert raw["model"].startswith("claude-sonnet-5-5") and raw["stop_reason"]=="end_turn" and raw["usage"]==entry["usage"]
        assert math.isclose(sonnet_cost(raw["usage"]),entry["estimated_usd"],abs_tol=1e-12) and math.isclose(sonnet_cost(raw["usage"],True),entry["conservative_usd"],abs_tol=1e-12)
        son_est+=entry["estimated_usd"];son_cons+=entry["conservative_usd"]
    rejected=[x for x in original if x["state"]!="completed"]
    check("quota rejections retained without inference substitution",len(rejected)==6 and all(x["error"]=="API HTTP 429; no retry" for x in rejected) and probe["http_status"]==429 and probe["error"]["code"]=="credit_balance_exhausted")
    rejected_bound=sum(x["reserved_usd"] for x in rejected)+probe["reserved_usd"]
    combined_est=estimated_sol+son_est;combined_cons=conservative_sol+son_cons+rejected_bound
    check("same$10 combined budget includes all providers and rejection reservations",combined_cons<=10.)
    check("no generation lock remains",all(not (folder/"generation.lock").exists() for folder in (PARENT,RESUME,SONNET)))
    summary=c.read(OUT/"summary.json")
    check("scoring matches all completed Sol calls",math.isclose(summary["estimated_usd"],estimated_sol,abs_tol=1e-10) and summary["calls"]==480)
    summary.update(combined_estimated_usd=combined_est,combined_conservative_usd=combined_cons,sol_estimated_usd=estimated_sol,sonnet_startup_estimated_usd=son_est,rejected_attempts=7,rejected_max_reservation_usd=rejected_bound,total_successful_calls=484,total_api_attempts=491,example_fold_counts=fold_counts)
    c.save(OUT/"summary.json",summary);c.save(OUT/"audit.json",{"passed":True,"checks":checks,"check_count":len(checks),"sol_requests_verified":480,"sonnet_startup_requests_verified":4,"combined_conservative_usd":combined_cons,"rejected_attempts":7,"rejected_max_reservation_usd":rejected_bound})
    conclusion="In-context examples did not establish improved xG predictions on this SoccerTrack v2 sample. Both 16-example Brier comparisons with zero-shot predictions have intervals crossing zero. Logistic regression had the strongest point scores."
    names={"zero":"Sol 6.2: no examples","outcome8":"8 examples + observed outcomes","outcome16":"16 examples + observed outcomes","shuffled_outcome16":"16 examples + shuffled outcomes","teacher16":"16 examples + logistic xG labels","shuffled_teacher16":"16 examples + shuffled logistic xG labels","logistic":"Logistic: distance + angle","logistic_header":"Logistic: also header information","training_constant":"Other-match goal-rate constant"}
    lines=["# SoccerTrack v2 in-context learning","",conclusion,"",
        "In-context learning supplies labelled examples in the inference prompt without updating model weights. This study used GPT-6.2 Sol at low reasoning effort on the existing SoccerTrack v2 panoramic frames. This is the dataset previously referred to as SoccerNet v2 in the project.","",
        f"All methods below use the same **80 shots, 10 goals, 10 matches**. There are 480 complete Sol predictions,one per shot and condition. Lower Brier is better; higher AUC is better.","",
        "| Method | Brier | AUC | Mean predicted xG |","|---|---:|---:|---:|"]
    lines += [f"| {names[a]} | {v['brier']:.6f} | {v['auc']:.4f} | {v['mean_xg']:.4f} |" for a,v in summary["metrics"].items()]
    lines += ["","## Interpretation","",
        "The 16 outcome examples moved mean predicted xG from 9.2% to 12.24%,close to the observed 12.5% goal rate. However,AUC fell from 0.633 to 0.444 on this sample,and Brier rose from 0.1085 to 0.1127. Matching the overall goal rate did not improve shot-specific predictions.","",
        "The logistic-labelled examples are a separate teacher-model condition. These labels come from a distance/angle logistic model fitted on the other nine matches, not from StatsBomb. Their Brier 0.1096 and AUC 0.558 did not improve on no-example predictions. The correctly paired teacher examples had a lower point Brier than shuffled teacher labels,but their adjusted interval also includes zero. Neither label type established an improvement beyond zero-shot or its shuffled control.","",
        "## Primary uncertainty intervals","",
        "Four prespecified Brier contrasts use 98.75% whole-match percentile bootstrap intervals, 20,000 resamples,a Bonferroni adjustment across four comparisons. Negative differences favor correctly labelled ICL. Ten match groups and ten goals limit precision. Intervals condition on the fitted other-match models and example sets,which overlap across folds.","",
        "| Brier difference | Estimate | 98.75% match interval |","|---|---:|---:|"]
    for name,v in summary["primary"].items():lines.append(f"| {name} | {v['difference']:+.6f} | [{v['interval'][0]:+.6f}, {v['interval'][1]:+.6f}] |")
    lines += ["","## Design and limits","",
        "For every query,all 16 examples and all logistic fitting data exclude its entire match. The 8-example condition is a nested prefix. Binary shuffling preserves the number of goals; teacher shuffling preserves every probability value. Image pixels remain identical within each correct/shuffled pair. All conditions receive the unchanged query frame. No query labels,numerical positions,reference probabilities or explicit player/match IDs enter requests.","",
        "The natural ordered 306-shot pool and all six conditions were fixed before inference. A token-count-only rule selected 80 queries spanning all ten matches under the $10 cap. There was no accuracy-driven sample extension or example-count selection. All source matches have prior research exposure: this is exploratory match-held-out evaluation,not untouched external confirmation.","",
        "Examples are panoramic images letterboxed to at most 768×384 in numbered JPEG quality 92 sheets,with eight images per page. Query PNGs remain unchanged. Shooters are unmarked,players and the ball can be tiny,and images precede the event by approximately 0.32 seconds. Missing or ambiguous shot localization is a plausible limitation; this experiment does not isolate its causal effect.","",
        "The distance/angle baseline uses at-event source coordinates. Its header variant has extra body-part information. BAS SHOT/GOAL can omit blocked or headed attempts classified under other actions,and unidentified penalties may remain. The professional-finisher prompt is retained on university matches. All VLM outputs are raw probabilities: no calibration,averaging,probability head or fine-tuning.","",
        "## Completion,recovery and spending","",
        "OpenAI credits ran out after 96 successful Sol calls. Six rejected requests and one bounded diagnostic rejection were preserved. A resource-only alternative-provider attempt completed four Sonnet predictions; after the user added $5 credits,the original Sol plan resumed. Sonnet predictions are excluded from all score tables. The same 80-shot Sol schedule completed without replacing any query or replaying a successful prediction.","",
        "A transient filesystem rename error occurred before a later batch was dispatched. All 372 completed Sol predictions were intact. Six uncommitted reservations were not sent to the API. The recovered runner retries filesystem rename operations only; API requests are not retried automatically. A saved global preflight input bound reduced repeated token-count calls without changing inference payloads.","",
        f"Estimated Sol successful-call spending: **${estimated_sol:.4f}**. Four Sonnet startup calls: **${son_est:.4f}**. Combined successful-call estimate: **${combined_est:.4f}**. Conservative combined accounting: **${combined_cons:.4f}**,including the full **${rejected_bound:.4f}** maximum reservation for all seven rejected attempts. The single combined cap remains **$10**. This is experiment accounting,not a live balance or reconciled invoice.","",
        f"Audit passed{len(checks)} checks,rebuilding all 480 Sol and four Sonnet request hashes from frozen input bytes. Source annotation hashes,match exclusions,label permutations,raw answers,model settings,usage and spending reconcile. No paid runner remains active.","",
        "Raw response journals remain in `../soccertrack_icl_v1/` and `../soccertrack_icl_resume_v1/`. The unused alternative-provider startup remains in `../soccertrack_icl_sonnet_v1/`. This directory contains consolidated per-shot probabilities,summary and audit. Source and protocols are in `experiments/soccertrack_icl/` and the original result folders.","",
        "Data attribution: SoccerTrack v2 by Atom Scott,Ikuma Uchida,Kento Kuroda,Yufi Kim and Keisuke Fujii. [Dataset](https://huggingface.co/datasets/atomscott/soccertrack-v2),[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).",
        "API rates:[OpenAI](https://developers.openai.com/api/docs/models/gpt-6.2-sol),[Anthropic](https://platform.claude.com/docs/en/about-claude/pricing)."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"audit":"passed","checks":len(checks),"sol_predictions":480,"combined_estimated_usd":combined_est,"combined_conservative_usd":combined_cons,"conclusion":conclusion}),flush=True)


if __name__=="__main__":main()
