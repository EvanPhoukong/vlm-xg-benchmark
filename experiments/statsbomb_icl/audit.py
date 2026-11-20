"""Offline audit of the complete frozen ICL study; never calls an API."""
import base64
from functools import lru_cache
import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageChops

import experiment as e


def token_cost(u,conservative=False):
    ni,no=u["input_tokens"],u["output_tokens"]
    nc=(u.get("input_tokens_details") or {}).get("cached_tokens",0)
    assert type(ni) is int and type(no) is int and 0<=nc<=ni and no>=0
    return (ni*2*1.25+no*10 if conservative else (ni-nc)*2+nc*.1+no*10)/1e6


@lru_cache(maxsize=512)
def image_part(relative,expected):
    content=(e.ROOT/relative).read_bytes()
    assert hashlib.sha256(content).hexdigest()==expected
    return {"type":"input_image","detail":"high","image_url":"data:image/png;base64,"+base64.b64encode(content).decode()}


def request(job,protocol,pages):
    parts=[]
    if job["arm"]!="zero":
        parts.append({"type":"input_text","text":protocol["intro"]})
        parts.extend(image_part(p["path"],p["sha256"]) for p in pages[job["arm"]])
        parts.append({"type":"input_text","text":"Historical examples END. The following separate image is the NEW QUERY; its outcome is unknown."})
    parts.extend([image_part(job["image"],job["sha256"]),{"type":"input_text","text":protocol["prompt"]}])
    return {"model":protocol["model"],"store":False,"service_tier":"default","reasoning":{"effort":"low"},"max_output_tokens":1024,
        "input":[{"role":"user","content":parts}],"text":{"format":{"type":"json_schema","name":"shot_prediction","strict":True,
        "schema":{"type":"object","properties":{"xg":{"type":"number","minimum":0,"maximum":1},"reason":{"type":"string"}},"required":["xg","reason"],"additionalProperties":False}}}}


def main():
    protocol=e.read(e.OUT/"protocol.json")
    decision=e.read(e.OUT/"selection_decision.json")
    ledger=e.read(e.OUT/"attempts.json")
    examples=e.read(e.OUT/"examples/manifest.json")
    selections={s:e.read(e.OUT/s/"selection.json") for s in ("development","confirmation")}
    checks=[]

    def checked(name,condition):
        if not condition:raise ValueError("Audit failed: "+name)
        checks.append(name)

    checked("experiment code frozen",e.sha(e.__file__)==protocol["python_sha256"])
    checked("runner code frozen",e.sha(e.ROOT/"experiments/statsbomb_icl/run.mjs")==protocol["runner_sha256"])
    checked("source selection frozen",e.sha(e.SOURCE/"selected.json")==protocol["source_selected_sha256"])
    checked("source protocol frozen",e.sha(e.SOURCE/"protocol.json")==protocol["source_protocol_sha256"])
    for field,rel in [("pages_sha256","pages.json"),("examples_sha256","examples/manifest.json"),("development_sha256","development/selection.json"),("test_pool_sha256","test_pool.json")]:
        checked(rel+" frozen",e.sha(e.OUT/rel)==protocol[field])
    groups={s:{r["match_id"] for r in rs} for s,rs in selections.items()}
    groups["examples"]={r["match_id"] for r in examples}
    checked("all match sets disjoint",not any(groups[a]&groups[b] for a,b in [("examples","development"),("examples","confirmation"),("development","confirmation")]))
    ids=[r["shot_id"] for rs in selections.values() for r in rs]+[r["shot_id"] for r in examples]
    checked("unique shots across example and evaluation groups",len(ids)==len(set(ids)))
    data=e.read(e.SOURCE/"selected.json")
    checked("examples training only",{r["shot_id"] for r in examples}<={r["shot_id"] for r in data["train"]})
    checked("development validation only",{r["shot_id"] for r in selections["development"]}<={r["shot_id"] for r in data["validation"]})
    checked("confirmation test only",{r["shot_id"] for r in selections["confirmation"]}<={r["shot_id"] for r in data["test"]})
    checked("shuffled64 preserves positive count",sum(r["goal"] for r in examples)==sum(r["shuffled_goal"] for r in examples))
    checked("shuffled64 changes associations",any(r["goal"]!=r["shuffled_goal"] for r in examples))
    correct_pages=e.read(e.OUT/"pages.json")
    # Correct and shuffled64 sheets have identical diagram pixels: only captions differ.
    for a,b in zip(correct_pages["icl64"],correct_pages["shuffled64"]):
        with Image.open(e.ROOT/a["path"]) as im1,Image.open(e.ROOT/b["path"]) as im2:
            for y in range(0,im1.height,216):
                assert ImageChops.difference(im1.crop((0,y,768,y+192)),im2.crop((0,y,768,y+192))).getbbox() is None
    checked("correct and shuffled example geometry identical",True)
    checked("confirmation remains frozen pool prefix",selections["confirmation"]==e.read(e.OUT/"test_pool.json")[:decision["n"]])
    estimated,conservative=0.,0.
    for stage in ("development","confirmation"):
        arms=e.ARMS if stage=="development" else ["zero",decision["selected_arm"],decision["matched_shuffle"]]
        seed=e.SEED+4 if stage=="development" else e.SEED+5
        jobs=e.read(e.OUT/stage/"jobs.json")
        checked(stage+" schedule frozen",jobs==e.schedule(selections[stage],arms,seed))
        entries=[r for r in ledger if r["stage"]==stage]
        checked(stage+" complete without replacement",[r["job"] for r in entries]==jobs and all(r["state"]=="completed" for r in entries))
        pages=correct_pages if stage=="development" else e.read(e.OUT/"confirmation_pages.json")
        for entry in entries:
            raw=e.read(e.OUT/entry["response_file"])
            rebuilt=request(entry["job"],protocol,pages)
            digest=hashlib.sha256(json.dumps(rebuilt,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
            assert digest==entry["request_sha256"]
            assert raw["usage"]==entry["usage"]
            assert raw["status"]=="completed" and raw["model"].startswith(protocol["model"])
            assert raw["reasoning"]["effort"]=="low"
            texts=[p["text"] for o in raw["output"] if o["type"]=="message" for p in o["content"] if p["type"]=="output_text"]
            assert json.loads("".join(texts))==entry["answer"]
            assert 0<=entry["answer"]["xg"]<=1 and math.isfinite(entry["answer"]["xg"])
            assert raw["usage"]["input_tokens"]<=entry["input_bound"] and raw["usage"]["output_tokens"]<=1024
            assert math.isclose(token_cost(raw["usage"]),entry["estimated_usd"],abs_tol=1e-12)
            assert math.isclose(token_cost(raw["usage"],True),entry["conservative_usd"],abs_tol=1e-12)
            assert entry["conservative_usd"]<=entry["reserved_usd"]
            estimated+=entry["estimated_usd"];conservative+=entry["conservative_usd"]
        checked(stage+" all request hashes,images,responses,usage,costs verified",True)
    checked("combined cap respected",conservative<=10.)
    checked("no uncertain requests",all(r["state"]=="completed" for r in ledger))
    for model in ("image_features","plus_header"):
        checked(model+" excludes development and confirmation matches",not set(e.read(e.OUT/"baselines.json")[model]["train_matches"])&(groups["development"]|groups["confirmation"]))
    summary=e.read(e.OUT/"summary.json")
    checked("reported costs reconcile",math.isclose(summary["estimated_usd"],estimated,abs_tol=1e-10) and math.isclose(summary["conservative_usd"],conservative,abs_tol=1e-10))
    rows,values,_=e.results("confirmation")
    checked("reported ICL Brier recomputes",all(math.isclose(e.helpers.metrics([r["goal"] for r in rows],p)["brier"],summary["metrics"][a]["brier"],abs_tol=1e-12) for a,p in values.items()))
    checked("generation lock released",not (e.OUT/"generation.lock").exists())
    e.save(e.OUT/"audit.json",{"passed":True,"checks":checks,"check_count":len(checks),"requests_verified":len(ledger),"estimated_usd":estimated,"conservative_usd":conservative})
    print(json.dumps({"audit":"passed","checks":len(checks),"requests_verified":len(ledger)}),flush=True)


if __name__=="__main__":main()
