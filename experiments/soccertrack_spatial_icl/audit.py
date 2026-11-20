"""Offline audit of source reuse, ICL exclusion, exact requests and the $10 cap."""
import base64
from collections import Counter
from functools import lru_cache
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

import experiment as e
import analyze as a

c, ROOT, OUT = e.c, e.ROOT, e.OUT


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=512)
def image_content(relative, expected):
    data = (ROOT / relative).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg" if relative.endswith(".jpg") else "image/png",
                                        "data": base64.b64encode(data).decode()}}


def request(job, p):
    content = []
    if job["arm"] != "zero":
        content.append({"type": "text", "text": p["outcome_intro"] if "outcome" in job["arm"] else p["teacher_intro"]})
        for i, item in enumerate(job["pages"]):
            block = dict(image_content(item["path"], item["sha256"]))
            if i == len(job["pages"]) - 1:
                block["cache_control"] = {"type": "ephemeral"}
            content.append(block)
        content.append({"type": "text", "text": "NEW QUERY (outcome and reference probability unknown):"})
    content.append(image_content(job["query"]["path"], job["query"]["sha256"]))
    content.append({"type": "text", "text": p["prompt"]})
    schema = {"type": "object", "properties": {"xg": {"type": "number"}, "reason": {"type": "string"}},
              "required": ["xg", "reason"], "additionalProperties": False}
    return {"model": p["model"], "max_tokens": 1024, "inference_geo": "global",
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            "messages": [{"role": "user", "content": content}]}


def cost(u, conservative=False):
    ni, no = u["input_tokens"], u["output_tokens"]
    nw, nr = u.get("cache_creation_input_tokens", 0), u.get("cache_read_input_tokens", 0)
    assert all(type(n) is int and n >= 0 for n in (ni, no, nw, nr))
    assert u.get("cache_creation", {}).get("ephemeral_1h_input_tokens", 0) == 0
    return (ni*(2.5 if conservative else 2) + nw*2.5 + nr*(.125 if conservative else .1) + no*10)/1e6


def main():
    checks = []
    def check(name, ok):
        if not ok:
            raise ValueError("Audit failed: " + name)
        checks.append(name)
    p = c.read(OUT / "protocol.json")
    for relative, expected in p["frozen"].items():
        check("frozen:" + relative, c.sha(ROOT / relative) == expected)
    inherited = c.read(e.SOURCE / "audit.json")
    check("inherited21-check source audit passed and remains frozen", inherited["passed"] and inherited["check_count"] == 21)
    terminal, rows, ledger = a.load_complete()
    pool = c.read(OUT / "pool.json")
    source_rows = c.read(e.SOURCE / "eligible.json")
    check("complete deterministic pool order", pool == e.ordered_pool(source_rows) and len(pool) == 304)
    check("unique paired prefix and ten matches", len({r["shot_id"] for r in rows}) == len(rows) and len({r["match_id"] for r in rows}) == 10)
    for row in source_rows:
        assert c.sha(ROOT / row["images"]["diagram"]["path"]) == row["images"]["diagram"]["sha256"]
        assert row["annotation_frame"] == row["video_frame_index"] + 1 < row["event_frame"]
    check("all304 original query diagrams and pre-shot timing preserved", True)
    examples, pages, models = [c.read(OUT / n) for n in ("examples.json", "pages.json", "teacher_models.json")]
    page_count = 0
    with tempfile.TemporaryDirectory(prefix="xg_spatial_icl_audit_") as folder:
        for i, mid in enumerate(sorted({r["match_id"] for r in source_rows})):
            training = [r for r in source_rows if r["match_id"] != mid]
            assert models[mid]["training_ids"] == [r["shot_id"] for r in training]
            assert models[mid]["training_matches"] == sorted({r["match_id"] for r in training})
            fitted = c.fit_logistic(training, e.spatial.FEATURES)
            np.testing.assert_allclose(c.predict(fitted, source_rows), c.predict(models[mid]["model"], source_rows), rtol=0, atol=1e-10)
            expected = e.examples_for(source_rows, mid, i, fitted)
            assert examples[mid] == expected and len(expected) == 32 and len({r["shot_id"] for r in expected}) == 32
            assert all(r["match_id"] != mid for r in expected)
            assert Counter(r["goal"] for r in expected) == Counter(r["shuffled_goal"] for r in expected)
            assert Counter(r["teacher"] for r in expected) == Counter(r["shuffled_teacher"] for r in expected)
            for arm in e.ARMS[1:]:
                rebuilt = e.make_pages(expected, mid, arm, Path(folder))
                assert [r["sha256"] for r in rebuilt] == [r["sha256"] for r in pages[mid][arm]]
                for item in pages[mid][arm]:
                    assert c.sha(ROOT / item["path"]) == item["sha256"]
                    page_count += 1
            print(f"Audited training exclusion and32 example sheets for {mid}", flush=True)
    check("all teacher models exclude the complete query match including scaling", True)
    check("all320 example sheets reproduce with correct labels and fixed permutations", page_count == 320)
    check("training examples,source images and rounded teacher labels reproduce", True)
    preview = c.read(OUT / "token_preview.json")
    choice = c.read(OUT / "resource_choice.json")
    check("token-preview bounds frozen", c.sha(OUT / "token_preview.json") == choice["preview_sha256"] and choice["input_bound"] == max(preview.values()) + 1024)
    blocks = c.read(OUT / "blocks.json")
    per_request = (choice["input_bound"]*2.5 + 1024*10)/1e6
    offset = cursor = 0
    past = 0.
    old_diagram = {r["job"]["shot_id"]: r for r in c.read(e.SOURCE / "attempts.json") if r["job"]["arm"] == "diagram"}
    for b in blocks:
        fits = [n for n in range(1, min(10, len(pool)-offset)+1) if past + n*5*per_request <= 10]
        assert fits and b["shots"] == max(fits) and b["state"] == "completed" and b["offset"] == offset
        assert b["shot_ids"] == [r["shot_id"] for r in pool[offset:offset+b["shots"]]]
        assert abs(b["exposure_before"]-past) < 1e-9
        assert abs(b["maximum_reservation_usd"]-b["shots"]*5*per_request) < 1e-9
        assert b["exposure_before"] + b["maximum_reservation_usd"] <= 10
        entries = ledger[cursor:cursor+b["shots"]*5]
        expected_jobs = []
        indexed = sorted(enumerate(pool[offset:offset+b["shots"]], offset), key=lambda pair: (pair[1]["match_id"], pair[0]))
        for index, row in indexed:
            rotation = index % 5
            for arm in e.ARMS[rotation:] + e.ARMS[:rotation]:
                expected_jobs.append({"shot_id": row["shot_id"], "match_id": row["match_id"], "pool_index": index,
                                      "arm": arm, "query": row["images"]["diagram"], "pages": pages[row["match_id"]][arm]})
        assert [r["job"] for r in entries] == expected_jobs
        for entry in entries:
            assert entry["state"] == "completed" and entry["block"] == b["index"]
            assert entry["input_bound"] == choice["input_bound"] and abs(entry["reserved_usd"]-per_request) < 1e-12
            assert fingerprint(request(entry["job"], p)) == entry["request_sha256"]
            raw = c.read(OUT / entry["response_file"])
            assert raw["type"] == "message" and raw["stop_reason"] == "end_turn"
            assert raw["model"] == p["model"] or raw["model"].startswith(p["model"] + "-")
            answer = json.loads("".join(x["text"] for x in raw["content"] if x["type"] == "text"))
            assert set(answer) == {"xg", "reason"} and type(answer["xg"]) in (int, float) and 0 <= answer["xg"] <= 1 and isinstance(answer["reason"], str)
            assert answer == entry["answer"] and raw["usage"] == entry["usage"]
            u = raw["usage"]
            assert u["input_tokens"] + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0) <= choice["input_bound"]
            assert u["output_tokens"] <= 1024
            assert abs(cost(u)-entry["estimated_usd"]) < 1e-12 and abs(cost(u, True)-entry["conservative_usd"]) < 1e-12
            assert entry["conservative_usd"] <= entry["reserved_usd"]
            if entry["job"]["arm"] == "zero":
                assert entry["request_sha256"] == old_diagram[entry["job"]["shot_id"]]["request_sha256"]
        past += sum(r["conservative_usd"] for r in entries)
        assert abs(b["exposure_after"]-past) < 1e-9 and past <= 10
        offset += b["shots"]
        cursor += len(entries)
    check("maximum uncached full-block reservations respected cap before dispatch", True)
    check("all cyclic schedules,paired query images and exact request hashes reproduce", cursor == len(ledger))
    check("fresh zero requests exactly match previous diagram prompt/model/settings", True)
    check("all raw responses,probability bounds,usage and cache costs reconcile", True)
    check("terminal rule and complete-shot prefix", offset == terminal["selected_n"] and (offset == len(pool) or past+5*per_request > 10))
    values = a.values_for(rows, ledger)
    summary = c.read(OUT / "summary.json")
    for name, probabilities in values.items():
        fresh = c.metrics([r["goal"] for r in rows], probabilities)
        for key, value in fresh.items():
            assert abs(value-summary["metrics"][name][key]) < 1e-12
    for i, (first, second) in enumerate(a.PAIRS):
        assert summary["primary"][first+"_minus_"+second] == a.interval(rows, values[first], values[second], e.SEED+500+i)
    check("reported scores and four adjusted match intervals reproduce", True)
    check("reported spend under10", abs(summary["conservative_usd"]-past) < 1e-9 and past <= 10)
    check("no paid runner active", not (OUT / "generation.lock").exists())
    result = {"passed": True, "checks": checks, "check_count": len(checks), "requests_verified": len(ledger),
              "example_sheets_verified": page_count, "shots": len(rows), "estimated_usd": summary["estimated_usd"],
              "conservative_usd": past, "inherited_source_audit": "../soccertrack_spatial_v1/audit.json",
              "full_large_file_sha256_verified": False}
    c.save(OUT / "audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
