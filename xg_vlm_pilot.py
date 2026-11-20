"""
Pilot: Can a vision-language model (VLM) estimate expected goals (xG) from a shot image?
Data: StatsBomb open data (default: 2022 World Cup) or the xGHub test split
Backends: anthropic | ollama | gemini | openai
"""
import os, json, base64, re, math, argparse, warnings
from hashlib import sha256
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, is_zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsbombpy import sb
from mplsoccer import VerticalPitch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
GOAL, POST_L, POST_R = (120.0, 40.0), (120.0, 36.0), (120.0, 44.0)
OUT = "results"
IMG_DIR = os.path.join("results", "images")
CACHE = None
FABLE = "claude-fable-5-1"
FABLE_MAX_TOKENS = 384
LOCAL_XGHUB_MODEL = "qwen3.5:4b"
SOCCERTRACK_SOL = "gpt-6.1-sol"
SOL_INPUT_LIMIT, SOL_OUTPUT_LIMIT, SOL_SHOT_LIMIT = 4096, 1024, 30

PROMPT = """You are a professional football (soccer) analyst.
The image shows a shot at the moment it is taken, drawn on a half pitch with the goal at the top.
Gold star = shooter. Blue circles = shooter's teammates. Red circles = opponents.
Orange square = opposing goalkeeper.
{context}Estimate the probability that this shot results in a goal (expected goals, xG),
assuming an average professional finisher.
Reply with JSON only: {{"xg": <number between 0 and 1>, "reason": "<one short sentence>"}}"""


def geometry(x, y):
    dist = math.hypot(GOAL[0] - x, GOAL[1] - y)
    a = math.atan2(POST_L[1] - y, POST_L[0] - x)
    b = math.atan2(POST_R[1] - y, POST_R[0] - x)
    return dist, abs(b - a)


def in_triangle(p, a, b, c):
    def s(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def load_shots(comp, season):
    matches = sb.matches(competition_id=comp, season_id=season)
    rows = []
    for i, mid in enumerate(matches.match_id, 1):
        print(f"  loading match {i}/{len(matches)}", end="\r")
        ev = sb.events(match_id=mid)
        for _, r in ev[ev["type"] == "Shot"].iterrows():
            ff = r.get("shot_freeze_frame")
            if r.get("shot_type") == "Penalty" or not isinstance(ff, list):
                continue
            x, y = r["location"][:2]
            dist, angle = geometry(x, y)
            opp = [p for p in ff if not p["teammate"]]
            cone = sum(in_triangle(p["location"][:2], (x, y), POST_L, POST_R) for p in opp)
            rows.append(dict(
                match_id=mid, shot_id=r["id"], x=x, y=y, ff=ff,
                goal=int(r["shot_outcome"] == "Goal"), sb_xg=float(r["shot_statsbomb_xg"]),
                body_part=r.get("shot_body_part"), play_pattern=r.get("play_pattern"),
                distance=dist, angle=angle, header=int(r.get("shot_body_part") == "Head"),
                defenders_in_cone=cone))
    print()
    return pd.DataFrame(rows)


def load_xghub(labels, frames=None):
    """Read published per-shot labels, leaving outcomes out of model inputs."""
    if not is_zipfile(labels):
        raise ValueError("Download the actual xGHub labels.zip. A Git LFS pointer is not an archive.")
    test_ids = set(json.loads(labels.with_name("clip_split.json").read_text())["test"])
    rows = []
    with ZipFile(labels) as archive:
        for member in sorted(archive.namelist()):
            file = PurePosixPath(member)
            if (file.name not in {"labels_tabular.json", "tabular_data.json"} or not file.parent.name.isdecimal()
                    or file.parent.name not in test_ids):
                continue
            shot_id = file.parent.name
            frame = frames / f"{shot_id}.png" if frames else None
            if frame is not None and not frame.is_file():
                continue
            data = json.loads(archive.read(member))
            if type(data["is_goal"]) is not bool:
                raise ValueError(f"Invalid outcome in xGHub shot {shot_id}")
            rows.append(dict(
                shot_id=shot_id, goal=int(data["is_goal"]), frame=frame,
                distance=data["distance_from_goal"], opening=data["angle_to_posts"],
                pressure=data.get("shooter_pressure"),
                occlusion=data.get("goal_face_occlusion_joint"),
            ))
    if not rows:
        raise ValueError("No xGHub shot labels matched the selected input.")
    shots = pd.DataFrame(rows)
    if shots.shot_id.duplicated().any():
        raise ValueError("Duplicate xGHub shot IDs.")
    return shots


def load_soccertrack(manifest):
    rows = json.loads(manifest.read_text(encoding="utf-8"))["shots"]
    for row in rows:
        path = (manifest.parent / row["frame"]).resolve()
        if (not path.is_relative_to(manifest.parent.resolve())
                or row["goal"] not in (0, 1)
                or sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise ValueError(f"Invalid SoccerTrack label or frame for {row['shot_id']}")
        row["frame"] = path
    shots = pd.DataFrame(rows)
    if shots.empty or shots.shot_id.duplicated().any():
        raise ValueError("SoccerTrack needs unique shots with verified frames.")
    return shots


def render_xghub(row, path):
    """Show only the published shot measurements, without recreating missing player positions."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.set_axis_off()
    ax.text(.03, .94, "Shot situation", fontsize=19, weight="bold", va="top")
    features = [
        ("Distance", row.distance, " m"),
        ("Goal opening", row.opening, "°"),
        ("Shooter pressure (0–1)", row.pressure, ""),
        ("Goal face occluded", None if pd.isna(row.occlusion) else row.occlusion * 100, "%"),
    ]
    for i, (label, value, unit) in enumerate(features):
        height = .70 - i * .18
        ax.text(.03, height, label, fontsize=12, va="center")
        ax.text(.97, height, "Unavailable" if pd.isna(value) else f"{value:.2f}{unit}",
                fontsize=12, weight="bold", ha="right", va="center")
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render(row, path):
    pitch = VerticalPitch(half=True, pitch_type="statsbomb", line_color="black")
    fig, ax = pitch.draw(figsize=(5, 5))
    for p in row.ff:
        px, py = p["location"][:2]
        is_gk = (p.get("position") or {}).get("name") == "Goalkeeper"
        if p["teammate"]:
            pitch.scatter(px, py, ax=ax, s=120, c="royalblue", edgecolors="black")
        elif is_gk:
            pitch.scatter(px, py, ax=ax, s=170, c="orange", marker="s", edgecolors="black")
        else:
            pitch.scatter(px, py, ax=ax, s=120, c="red", edgecolors="black")
    pitch.scatter(row.x, row.y, ax=ax, s=320, c="gold", marker="*", edgecolors="black")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


_client = None
_openai_client = None


def read_cache(path):
    with open(path) as f:
        return {entry["shot_id"]: entry for entry in map(json.loads, f)}



def strict_prediction(text):
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    answer = json.loads(fenced[1] if fenced else text)
    xg, reason = answer["xg"], answer["reason"]
    if type(xg) not in (int, float) or not 0 <= xg <= 1 or not isinstance(reason, str) or not reason.strip():
        raise ValueError("Model returned an invalid probability or rationale.")
    return xg, reason

def ask_fable(image, prompt, shot_id, cache_path, budget):
    """Reserve input and maximum output cost before the next Fable request."""
    import anthropic

    request = {
        "model": FABLE,
        "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image}},
            {"type": "text", "text": prompt},
        ]}],
    }
    signature = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    pending = Path(cache_path).with_suffix(".pending")
    dispatched = False
    with pending.open("x") as reservation:
        try:
            cache = read_cache(cache_path) if os.path.exists(cache_path) else {}
            if shot_id in cache:
                entry = cache[shot_id]
                if entry.get("input_sha256") != signature:
                    raise ValueError(f"Cached Fable input changed for shot {shot_id}.")
                return entry
            spent = sum(entry["cost_usd"] for entry in cache.values())
            with anthropic.Anthropic(max_retries=0, timeout=120.0) as client:
                tokens = client.messages.count_tokens(**request).input_tokens
                input_bound = tokens + 256
                reserved = (input_bound * 10 + FABLE_MAX_TOKENS * 50) / 1_000_000
                if spent + reserved > budget:
                    raise ValueError(f"Fable budget of ${budget:.2f} reached after {len(cache)} predictions.")
                reservation.write(json.dumps({"shot_id": shot_id, "reserved_usd": reserved}))
                reservation.flush()
                os.fsync(reservation.fileno())
                dispatched = True
                msg = client.messages.create(**request, max_tokens=FABLE_MAX_TOKENS,
                                             inference_geo="global")
            if msg.model != FABLE or msg.stop_reason != "end_turn":
                raise ValueError("Fable returned an incomplete prediction or a different model served it.")
            text = "".join(block.text for block in msg.content if block.type == "text")
            xg, reason = strict_prediction(text)
            usage = msg.usage
            if usage.input_tokens > input_bound or usage.output_tokens > FABLE_MAX_TOKENS:
                raise ValueError("Fable usage exceeded the preflight estimate.")
            cost = (usage.input_tokens * 10 + usage.output_tokens * 50) / 1_000_000
            entry = dict(shot_id=shot_id, vlm_xg=xg, reason=reason,
                         input_sha256=signature, cost_usd=cost)
            with open(cache_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
            dispatched = False
            return entry
        finally:
            if not dispatched:
                reservation.close()
                pending.unlink()


def ask_anthropic(img_b64, prompt, model):
    global _client
    import anthropic
    _client = _client or anthropic.Anthropic(max_retries=6)
    msg = _client.messages.create(
        model=model, max_tokens=600,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
            {"type": "text", "text": prompt}]}])
    return "".join(b.text for b in msg.content if b.type == "text")


def ask_ollama(img_b64, prompt, model, strict=False):
    import requests

    options = {"temperature": 0, "seed": 7}
    if strict:
        options.update(num_ctx=2048, num_predict=256)
    with requests.Session() as session:
        session.trust_env = False
        response = session.post("http://127.0.0.1:11434/api/generate", timeout=600,
                                allow_redirects=False,
                                json={"model": model, "prompt": prompt, "images": [img_b64],
                                      "stream": False, "format": "json", "think": False,
                                      "options": options})
        response.raise_for_status()
        payload = response.json()
    if strict and (payload.get("done_reason") != "stop" or payload.get("model") != model
                   or payload.get("remote_model")):
        raise ValueError("Ollama did not return a complete prediction from the requested local model.")
    return payload["response"]


def ask_gemini(img_b64, prompt, model, retries=6):
    global _client
    import time
    from google import genai
    from google.genai import types, errors
    _client = _client or genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip())
    for attempt in range(retries):
        try:
            resp = _client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=base64.b64decode(img_b64), mime_type="image/png"), prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json"))
            return resp.text or ""
        except errors.APIError as e:
            if getattr(e, "code", None) in (429, 500, 503) and attempt < retries - 1:
                wait = min(60, 4 * 2 ** attempt)
                print(f"  busy ({e.code}), retrying in {wait}s")
                time.sleep(wait)
                continue
            raise


def ask_openai(img_b64, prompt, model, bounded=False):
    global _openai_client
    from openai import OpenAI

    _openai_client = _openai_client or OpenAI(
        api_key=os.environ["OPENAI_API_KEY"].strip(),
        max_retries=0 if bounded else 6, timeout=120.0)
    request = dict(
        model=model, store=False, reasoning={"effort": "low"},
        max_output_tokens=SOL_OUTPUT_LIMIT if bounded else 2048,
        input=[{"role": "user", "content": [
            {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}", "detail": "high"},
            {"type": "input_text", "text": prompt}]}],
        text={"format": {
            "type": "json_schema", "name": "shot_prediction", "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "xg": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"}},
                "required": ["xg", "reason"], "additionalProperties": False}}})
    client = _openai_client
    if bounded:
        client = client.with_options(max_retries=0)
        request["service_tier"] = "default"
        estimated = client.responses.input_tokens.count(
            **{key: request[key] for key in ("model", "input", "text")}).input_tokens
        if estimated + 512 > SOL_INPUT_LIMIT:
            raise RuntimeError("Sol input exceeded the pilot's per-shot budget.")
    resp = client.responses.create(**request)
    if resp.status != "completed" or not resp.output_text:
        raise RuntimeError(f"OpenAI returned no complete prediction (status={resp.status}).")
    if not bounded:
        return resp.output_text
    if resp.model != model and not resp.model.startswith(model + "-"):
        raise RuntimeError("OpenAI returned an unexpected model.")
    usage = resp.usage
    if usage is None or usage.input_tokens > SOL_INPUT_LIMIT or usage.output_tokens > SOL_OUTPUT_LIMIT:
        raise RuntimeError("Sol returned unexpected token usage.")
    return resp.output_text, usage.input_tokens, usage.output_tokens


def parse(txt):
    try:
        data = json.loads(txt)
        xg, reason = float(data["xg"]), str(data.get("reason") or "")
    except (KeyError, TypeError, ValueError):
        # Retain support for older cached responses with surrounding text.
        m = re.search(r'"xg"\s*:\s*"?([0-9]*\.?[0-9]+)', txt or "")
        if not m:
            return np.nan, ""
        xg = float(m.group(1))
        r = re.search(r'"reason"\s*:\s*"([^"]*)', txt)
        reason = r.group(1) if r else ""
    if not math.isfinite(xg):
        return np.nan, ""
    if xg > 1:
        xg = xg / 100   # answered as a percentage
    return float(np.clip(xg, 0.001, 0.999)), reason


def scores(y, p, clip=True):
    p = np.asarray(p)
    if clip:
        p = np.clip(p, 1e-3, 1 - 1e-3)
    return dict(auc=roc_auc_score(y, p), brier=brier_score_loss(y, p),
                logloss=log_loss(y, p), mean_pred=float(p.mean()))


def bootstrap_brier_diff(y, p1, p2, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(np.mean((p1[i] - y[i]) ** 2) - np.mean((p2[i] - y[i]) ** 2))
    return float(np.mean((p1 - y) ** 2) - np.mean((p2 - y) ** 2)), np.percentile(diffs, [2.5, 97.5])


def calibration_plot(df, cols, path, bins=8):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    for c in cols:
        q = pd.qcut(df[c], q=bins, duplicates="drop")
        g = df.groupby(q, observed=True).agg(pred=(c, "mean"), obs=("goal", "mean"))
        plt.plot(g.pred, g.obs, "o-", label=c)
    plt.xlabel("Mean predicted xG"); plt.ylabel("Observed goal rate")
    plt.legend(); plt.title("Reliability curve"); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", type=int, default=43)
    ap.add_argument("--season", type=int, default=106)
    ap.add_argument("--n", type=int, help="Shot count (default: 300 for StatsBomb, 30 for image pilots)")
    ap.add_argument("--soccertrack", type=Path, metavar="MANIFEST",
                    help="Use a prepared SoccerTrack v2 pre-shot frame manifest with GPT-6.1 Sol")
    ap.add_argument("--xghub", type=Path, metavar="LABELS_ZIP",
                    help="Use xGHub labels.zip with clip_split.json in the same directory")
    ap.add_argument("--xghub-authorized-frames", type=Path, metavar="DIRECTORY",
                    help="Use verified pre-shot PNGs named by shot ID. Hosted use requires permission.")
    ap.add_argument("--fable-budget", type=float, default=1.0, metavar="USD",
                    help="Maximum estimated Fable spend for this run and its cache")
    ap.add_argument("--backend", choices=["anthropic", "ollama", "gemini", "openai"], default="anthropic")
    ap.add_argument("--model", help="Model ID; defaults to the selected backend's standard model")
    ap.add_argument("--context", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.soccertrack and (a.xghub or a.context or a.backend != "openai"):
        ap.error("SoccerTrack requires the OpenAI backend and image-only inputs.")
    image_pilot = bool(a.xghub or a.soccertrack)
    if a.xghub and (a.backend not in {"anthropic", "ollama"} or a.context):
        ap.error("xGHub uses the Anthropic or local Ollama backend with image-only inputs.")
    if a.xghub and a.backend == "ollama" and a.model and a.model.endswith((":cloud", "-cloud")):
        ap.error("The xGHub Ollama backend requires a local model.")
    if a.xghub_authorized_frames and not a.xghub:
        ap.error("--xghub-authorized-frames requires --xghub.")
    if a.n is None:
        a.n = 30 if image_pilot else 300
    if a.n < 1:
        ap.error("--n must be positive.")
    if a.xghub and a.model is None:
        a.model = LOCAL_XGHUB_MODEL if a.backend == "ollama" else FABLE
    if a.soccertrack:
        a.model = a.model or SOCCERTRACK_SOL
        if a.model != SOCCERTRACK_SOL or a.n > SOL_SHOT_LIMIT:
            ap.error("The SoccerTrack pilot is capped at 30 GPT-6.1 Sol requests.")
    if a.model is None:
        a.model = {"anthropic": "claude-sonnet-5-5", "ollama": "gemma4:26b",
                   "gemini": "gemini-3.8-flash", "openai": "gpt-6-luna"}[a.backend]
    fable = a.backend == "anthropic" and a.model == FABLE
    local_xghub = bool(a.xghub and a.backend == "ollama")
    if a.xghub and a.backend == "anthropic" and not fable:
        ap.error("The Anthropic xGHub pilot uses claude-fable-5-1.")
    if fable:
        if not math.isfinite(a.fable_budget) or a.fable_budget <= 0:
            ap.error("--fable-budget must be a positive finite amount in USD.")
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            ap.error("Set ANTHROPIC_API_KEY before using Claude Fable.")
    if a.backend == "openai":
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            ap.error("Set OPENAI_API_KEY before using --backend openai.")
        try:
            import openai
        except ImportError:
            ap.error("Install the OpenAI package first: python -m pip install openai")
    global OUT, CACHE
    slug = re.sub(r"[^A-Za-z0-9.]+", "_", a.model) + ("_context" if a.context else "")
    if a.xghub:
        input_name = "broadcast_frames" if a.xghub_authorized_frames else "annotation_cards"
        slug = f"xghub_{slug}_{input_name}"
    if a.soccertrack:
        slug = f"soccertrack_{slug}"
    OUT = os.path.join("results", slug)
    CACHE = os.path.join(OUT, "vlm_cache.jsonl")
    image_dir = os.path.join(OUT, "images") if a.xghub else IMG_DIR
    os.makedirs(OUT, exist_ok=True); os.makedirs(image_dir, exist_ok=True)

    if a.soccertrack:
        try:
            shots = load_soccertrack(a.soccertrack)
        except (OSError, ValueError, KeyError) as error:
            ap.error(str(error))
        if len(shots) > SOL_SHOT_LIMIT:
            ap.error("Prepare a fixed SoccerTrack manifest of at most 30 shots.")
        data_line = "Data: SoccerTrack v2 v1.2, real panoramic pre-shot images"
        pool_line = f"Prepared official test shots: {len(shots)} ({shots.goal.sum()} goals)"
    elif a.xghub:
        try:
            shots = load_xghub(a.xghub, a.xghub_authorized_frames)
        except (OSError, ValueError, KeyError) as error:
            ap.error(str(error))
        data_line = f"Data: xGHub ({input_name.replace('_', ' ')})"
        pool_line = f"Eligible shots in the official test split: {len(shots)} ({shots.goal.sum()} goals)"
    else:
        print("Loading StatsBomb shots...")
        shots = load_shots(a.comp, a.season)
        data_line = f"Data: StatsBomb open data, competition {a.comp}, season {a.season}"
        pool_line = f"All non-penalty shots with freeze frames: {len(shots)} ({shots.goal.sum()} goals)"
        feats = ["distance", "angle", "header", "defenders_in_cone"]
        shots["baseline_xg"] = cross_val_predict(
            LogisticRegression(max_iter=1000), shots[feats], shots.goal,
            groups=shots.match_id, cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
    print(pool_line)
    sample = shots.sample(n=min(a.n, len(shots)), random_state=a.seed).copy()
    if (fable or image_pilot) and sample.goal.nunique() != 2:
        ap.error("The sampled cohort needs both goals and non-goals for AUC.")
    cache = read_cache(CACHE) if os.path.exists(CACHE) else {}
    ask = {"anthropic": ask_anthropic, "ollama": ask_ollama,
           "gemini": ask_gemini, "openai": ask_openai}[a.backend]

    vlm_xg, reasons = [], []
    for i, row in enumerate(sample.itertuples(), 1):
        if image_pilot or fable or row.shot_id not in cache:
            path = Path(image_dir) / f"{row.shot_id}.png"
            ctx = f"Body part used: {row.body_part}.\n" if a.context else ""
            prompt = PROMPT.format(context=ctx)
            if image_pilot:
                if row.frame:
                    path = row.frame
                    description = ("The image is a panoramic match frame taken shortly before the shot."
                                   if a.soccertrack else
                                   "The image is a broadcast frame taken immediately before the shot.")
                else:
                    render_xghub(row, path)
                    description = "The image shows annotated measurements of the shot situation."
                task = PROMPT.split("{context}", 1)[1].format()
                prompt = f"You are a professional football (soccer) analyst.\n{description}\n{task}"
            elif not path.exists():
                render(row, path)
            b64 = base64.b64encode(path.read_bytes()).decode()
            if local_xghub or a.soccertrack:
                input_hash = sha256((b64 + "\n" + prompt).encode()).hexdigest()
                if row.shot_id in cache and cache[row.shot_id].get("input_sha256") != input_hash:
                    raise SystemExit(f"Cached model input changed for shot {row.shot_id}.")
            if fable:
                try:
                    cache[row.shot_id] = ask_fable(b64, prompt, row.shot_id, CACHE, a.fable_budget)
                except Exception as error:
                    message = str(error) if isinstance(error, ValueError) else type(error).__name__
                    raise SystemExit(f"Fable stopped: {message}. Completed predictions are saved. "
                                     "If a pending request remains, reconcile it before restarting.") from None
        if not fable and row.shot_id not in cache:
            try:
                if a.soccertrack:
                    txt, input_tokens, output_tokens = ask_openai(b64, prompt, a.model, bounded=True)
                    xg, why = strict_prediction(txt)
                elif local_xghub:
                    xg, why = strict_prediction(ask_ollama(b64, prompt, a.model, strict=True))
                else:
                    xg, why = parse(ask(b64, prompt, a.model))
            except Exception as e:
                if local_xghub:
                    raise SystemExit(f"Ollama stopped at shot {i}: {e}\n"
                                     "Successful predictions are cached; rerun to resume.") from e
                if a.backend == "openai":
                    raise SystemExit(f"OpenAI request failed: {e}\n"
                                     "Successful predictions are cached; rerun to resume.") from e
                print(f"  shot {i}: error {e} (not cached; rerun to retry)")
                import time; time.sleep(5)
                vlm_xg.append(np.nan); reasons.append(""); continue
            cache[row.shot_id] = dict(shot_id=row.shot_id, vlm_xg=xg, reason=why)
            if local_xghub or a.soccertrack:
                cache[row.shot_id]["input_sha256"] = input_hash
            if a.soccertrack:
                cache[row.shot_id].update(input_tokens=input_tokens, output_tokens=output_tokens)
            with open(CACHE, "a") as f:
                f.write(json.dumps(cache[row.shot_id]) + "\n")
        vlm_xg.append(cache[row.shot_id]["vlm_xg"]); reasons.append(cache[row.shot_id]["reason"])
        print(f"  VLM {i}/{len(sample)}", end="\r")
    print()
    sample["vlm_xg"], sample["vlm_reason"] = vlm_xg, reasons
    sample = sample.dropna(subset=["vlm_xg"])
    L = [data_line, pool_line,
         f"VLM sample: {len(sample)} shots ({sample.goal.sum()} goals), observed goal rate {sample.goal.mean():.3f}",
         f"VLM: {a.backend}/{a.model}, zero-shot, context={'body part' if a.context else 'image only'}\n",
         f"{'model':<14}{'AUC':>8}{'Brier':>9}{'LogLoss':>9}{'MeanXG':>9}"]
    columns = ["vlm_xg"] if image_pilot else ["vlm_xg", "sb_xg", "baseline_xg"]
    for c in columns:
        s = scores(sample.goal, sample[c], clip=not (fable or image_pilot))
        L.append(f"{c:<14}{s['auc']:>8.3f}{s['brier']:>9.4f}{s['logloss']:>9.4f}{s['mean_pred']:>9.3f}")
    if fable:
        spent = sum(entry["cost_usd"] for entry in cache.values())
        L.append(f"Estimated Fable cost, including cached predictions: ${spent:.4f} (cap ${a.fable_budget:.2f})")
    if a.soccertrack:
        spend = sum((2 * cache[sid]["input_tokens"] + 10 * cache[sid]["output_tokens"]) / 1_000_000
                    for sid in sample.shot_id)
        L.append(f"Estimated Sol cost at uncached rates: ${spend:.4f}")
    if not image_pilot:
        sample["gap"] = sample.vlm_xg - sample.sb_xg
        d, ci = bootstrap_brier_diff(sample.goal, sample.vlm_xg, sample.sb_xg)
        L.append(f"\nBrier(VLM) - Brier(StatsBomb) = {d:+.4f}  (95% bootstrap CI {ci[0]:+.4f} to {ci[1]:+.4f})")
        d2, ci2 = bootstrap_brier_diff(sample.goal, sample.vlm_xg, sample.baseline_xg)
        L.append(f"Brier(VLM) - Brier(baseline)  = {d2:+.4f}  (95% bootstrap CI {ci2[0]:+.4f} to {ci2[1]:+.4f})")
        L.append(f"Spearman rho, VLM vs StatsBomb xG: {spearmanr(sample.vlm_xg, sample.sb_xg).correlation:.3f}")
        L.append(f"Mean gap (VLM - StatsBomb): {sample.gap.mean():+.3f}; mean absolute gap: {sample.gap.abs().mean():.3f}\n")

        sample["cone_bin"] = pd.cut(sample.defenders_in_cone, [-1, 0, 1, 2, 99], labels=["0", "1", "2", "3+"])
        sample["dist_bin"] = pd.cut(sample.distance, [0, 6, 12, 18, 25, 200], labels=["<6", "6-12", "12-18", "18-25", "25+"])
        for g in ["body_part", "play_pattern", "cone_bin", "dist_bin"]:
            t = sample.groupby(g, observed=True).agg(
                n=("goal", "size"), goals=("goal", "sum"), vlm=("vlm_xg", "mean"),
                statsbomb=("sb_xg", "mean"), mean_gap=("gap", "mean")).round(3)
            L.append(f"Disagreement by {g}:\n{t.to_string()}\n")
        cols = ["shot_id", "goal", "sb_xg", "vlm_xg", "gap", "body_part", "distance", "defenders_in_cone", "vlm_reason"]
        L.append("Largest VLM over-estimates:\n" + sample.nlargest(8, "gap")[cols].round(3).to_string(index=False))
        L.append("\nLargest VLM under-estimates:\n" + sample.nsmallest(8, "gap")[cols].round(3).to_string(index=False))

    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(report)
    sample.drop(columns=["frame" if image_pilot else "ff"]).to_csv(os.path.join(OUT, "shots.csv"), index=False)
    calibration_plot(sample, columns, os.path.join(OUT, "calibration.png"))
    print(f"\nSaved results to ./{OUT}/")


if __name__ == "__main__":
    main()
