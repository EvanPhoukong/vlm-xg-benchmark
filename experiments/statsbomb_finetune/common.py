"""StatsBomb supervised image-to-xG pilot. No network or inference on import."""
from pathlib import Path
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/statsbomb_finetune_v1"
SEED = 2026111302
LIMITS = {"train": 2000, "validation": 400, "test": 500}
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
PROMPT = ("The diagram shows a soccer shot at contact. Goal is at the top. "
    "Gold star: shooter. Blue circles: teammates. Red circles: opposing outfield players. "
    "Orange square: opposing goalkeeper. Estimate the probability of a goal from this situation. "
    "Body part, ball height and previous motion are not supplied.")

# Reuse the independently gradient-checked logistic optimizer and metrics.
spec = importlib.util.spec_from_file_location("xg_statistical_helpers", ROOT / "experiments/soccertrack_comparison/common.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
fit_logistic, predict, metrics = helpers.fit_logistic, helpers.predict, helpers.metrics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def freeze(path, value):
    value = json.loads(json.dumps(value, allow_nan=False))
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Frozen artifact differs: {path}")
    else:
        save(path, value)


def geometry(x, y):
    a, b = np.array([120.-x, 36.-y]), np.array([120.-x, 44.-y])
    # atan2(cross magnitude, dot) stays correct when the posts straddle the shooter.
    angle = math.atan2(abs(a[0]*b[1]-a[1]*b[0]), float(a @ b))
    return math.hypot(120.-x, 40.-y), angle


def in_triangle(p, a, b=(120., 36.), c=(120., 44.)):
    def cross(p1, p2, p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1])-(p2[0]-p3[0])*(p1[1]-p3[1])
    values = [cross(p,a,b), cross(p,b,c), cross(p,c,a)]
    return not (min(values) < 0 < max(values))


def finite_location(value):
    return (isinstance(value, list) and len(value) >= 2 and
            all(type(v) in (int, float) and math.isfinite(v) for v in value[:2]))


def extract(event, mid, competition):
    """Inputs use only shooter/player positions; label and StatsBomb reference stay separate."""
    if event.get("type", {}).get("name") != "Shot":
        return None, None
    shot = event.get("shot", {})
    if shot.get("type", {}).get("name") == "Penalty":
        return None, "penalty"
    if not finite_location(event.get("location")):
        return None, "invalid_shooter_location"
    x, y = event["location"][:2]
    if not (60 <= x < 120 and 0 <= y <= 80):
        return None, "shooter_outside_displayed_attacking_half_or_on_goal_line"
    raw = shot.get("freeze_frame")
    if not isinstance(raw, list) or not raw:
        return None, "missing_freeze_frame"
    if any(not finite_location(p.get("location")) or type(p.get("teammate")) is not bool for p in raw):
        return None, "invalid_player_location_or_team"
    outcome = shot.get("outcome", {}).get("name")
    if outcome not in {"Goal", "Blocked", "Off T", "Post", "Saved", "Saved Off Target", "Saved to Post", "Wayward"}:
        return None, "unknown_shot_outcome"
    players = [{"x": float(p["location"][0]), "y": float(p["location"][1]),
                "teammate": p["teammate"], "keeper": p.get("position", {}).get("name") == "Goalkeeper"} for p in raw]
    if not any(p["keeper"] and not p["teammate"] for p in players):
        return None, "opposing_keeper_not_recorded"
    distance, angle = geometry(x, y)
    # Match the legacy definition: includes the keeper in the opponent count.
    cone = sum(in_triangle((p["x"], p["y"]), (x, y)) for p in players if not p["teammate"])
    sb = shot.get("statsbomb_xg")
    if type(sb) not in (int, float) or not 0 <= sb <= 1:
        return None, "missing_statsbomb_reference"
    return {"shot_id": event["id"], "match_id": str(mid), "competition": competition,
            "x": float(x), "y": float(y), "players": players, "goal": int(outcome == "Goal"),
            "distance": distance, "angle": angle, "defenders_in_cone": cone,
            "header": int(shot.get("body_part", {}).get("name") == "Head"),
            "statsbomb_xg": float(sb)}, None


def image_payload(row):
    """The complete whitelist of information allowed to influence image pixels."""
    return {"x": row["x"], "y": row["y"], "players": row["players"]}


def render(row, path):
    """Deterministic Pillow renderer; the only input is the positional whitelist."""
    from PIL import Image, ImageDraw
    payload = image_payload(row)
    scale = 2
    image = Image.new("RGB", (512*scale, 512*scale), "#fcfffc")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 38*scale, 474*scale, 48*scale, 476*scale

    def point(x, y):
        return left + y/80*(right-left), top + (120-x)/60*(bottom-top)

    def line(points, color="#59655b", width=2):
        draw.line([point(x,y) for x,y in points], fill=color, width=width*scale)

    line([(60,0), (120,0), (120,80), (60,80), (60,0)])
    line([(120,18), (102,18), (102,62), (120,62)])
    line([(120,30), (114,30), (114,50), (120,50)])
    line([(120,36), (122.5,36), (122.5,44), (120,44)], width=3)
    spot = point(108,40)
    draw.ellipse([spot[0]-2*scale, spot[1]-2*scale, spot[0]+2*scale, spot[1]+2*scale], fill="#59655b")
    # Centre-circle half arc; no labels, numeric probabilities, filenames, or outcomes.
    line([(60+10*math.cos(t), 40+10*math.sin(t)) for t in np.linspace(-math.pi/2, math.pi/2, 80)])
    for p in payload["players"]:
        if not (60 <= p["x"] <= 120 and 0 <= p["y"] <= 80):
            continue
        px, py = point(p["x"], p["y"])
        radius = 5*scale
        bounds = [px-radius, py-radius, px+radius, py+radius]
        if not p["teammate"] and p["keeper"]:
            draw.rectangle(bounds, fill="#f39c12", outline="#151b16", width=scale)
        else:
            draw.ellipse(bounds, fill="#4169e1" if p["teammate"] else "#df3038", outline="#151b16", width=scale)
    px, py = point(payload["x"], payload["y"])
    star = []
    for i in range(10):
        radius = (10 if i % 2 == 0 else 4.5)*scale
        angle = -math.pi/2 + i*math.pi/5
        star.append((px+radius*math.cos(angle), py+radius*math.sin(angle)))
    draw.polygon(star, fill="#ffd43b", outline="#171c16", width=scale)
    image.resize((512,512), Image.Resampling.LANCZOS).save(path)


@lru_cache(maxsize=8)
def contained_image_directory(root):
    # Resolving each UNC file walks every parent and is extremely slow under Windows/WSL.
    root = Path(root)
    directory = root/"images"
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError("Image directory escapes the experiment")
    return directory


def verify_image(row, root):
    from pathlib import PurePosixPath
    relative = PurePosixPath(row["image"])
    if relative.parts != ("images",row["shot_id"]+".png") or "/" in row["shot_id"] or "\\" in row["shot_id"]:
        raise ValueError("Unexpected image path")
    path = contained_image_directory(str(root))/relative.name
    if path.is_symlink() or sha(path) != row["image_sha256"]:
        raise ValueError("Image hash or symlink integrity failed")
    return path


def check_data():
    protocol = read(OUT / "protocol.json")
    if sha(OUT / "selected.json") != protocol["selected_sha256"]:
        raise ValueError("Selected data changed")
    data = read(OUT / "selected.json")
    match_sets = {s: {r["match_id"] for r in rs} for s,rs in data.items()}
    for a,b in (("train","validation"), ("train","test"), ("validation","test")):
        if match_sets[a] & match_sets[b]:
            raise ValueError("Match leakage")
    ids = [r["shot_id"] for rs in data.values() for r in rs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate shots across splits")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda row:verify_image(row,OUT),[r for rs in data.values() for r in rs]))
    return protocol, data
