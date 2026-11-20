"""Spatial input extraction. Reads only the requested pre-shot tracking snapshot."""
import io
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/soccertrack_spatial_v1"

def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


@lru_cache(maxsize=10)
def camera(mid):
    base=OUT/"source/calibration"
    homography=np.load(base/f"{mid}_homography.npy",allow_pickle=False)
    with np.load(base/f"{mid}_camera_intrinsics.npz",allow_pickle=False) as data:
        matrices={name:data[name].copy() for name in ("K","D","Knew")}
    stream=io.BytesIO((base/f"{mid}_mapx.npy.header").read_bytes())
    version=np.lib.format.read_magic(stream)
    shape,_,_=np.lib.format.read_array_header_1_0(stream) if version==(1,0) else np.lib.format.read_array_header_2_0(stream)
    return {"H":homography,**matrices,"width":shape[1],"height":shape[0]}


def project(x,y,cam):
    """The author pipeline: corner-origin metres -> H -> inverse Knew -> fisheye K,D."""
    v=cam["H"]@np.array([x+52.5,y+34.,1.])
    v=v/v[2];ray=np.linalg.inv(cam["Knew"])@v
    a,b=ray[0]/ray[2],ray[1]/ray[2];r=math.hypot(a,b);theta=math.atan(r)
    d=cam["D"].ravel();td=theta*(1+sum(float(k)*theta**(2*(i+1)) for i,k in enumerate(d)))
    scale=td/r if r>1e-12 else 1.
    result=cam["K"]@np.array([a*scale,b*scale,1.])
    return float(result[0]/result[2]),float(result[1]/result[2])


def entities(snapshot):
    result=[]
    for obj in snapshot["objects"]:
        if obj.get("supercategory")!="object":continue
        attrs=obj.get("attributes") or {};role=attrs.get("role");team=attrs.get("team")
        if role not in ("player","goalkeeper") or team not in ("left","right"):continue
        pitch=obj.get("bbox_pitch") or {};x=pitch.get("x_bottom_middle");y=pitch.get("y_bottom_middle")
        if type(x) not in (float,int) or type(y) not in (float,int) or not math.isfinite(x) or not math.isfinite(y):continue
        result.append({"id":str(attrs.get("player_id")),"role":role,"team":team,"x":float(x),"y":float(y),"bbox":obj.get("bbox_image"),"track_id":obj.get("track_id")})
    return result


def triangle(point,a,b,c):
    def cross(p,q,r):return (p[0]-r[0])*(q[1]-r[1])-(q[0]-r[0])*(p[1]-r[1])
    values=[cross(point,a,b),cross(point,b,c),cross(point,c,a)]
    return not (min(values)<0<max(values))


def extract(row,frame):
    mid=row["match_id"];half=row["half"];number=frame["frame_index"]+1
    snapshot=read(OUT/"snapshots"/f"{mid}_{half}_{number}.json")
    at_contact=read(OUT/"snapshots"/f"{mid}_{half}_{row['event_frame']}.json")
    assert snapshot["video_frame_index"]==frame["frame_index"] and number<row["event_frame"]
    players=entities(snapshot);shooters=[p for p in players if p["id"]==row["actor_id"]]
    if len(shooters)!=1:return None,{"reason":"shooter_missing_or_ambiguous","players":len(players)}
    shooter=shooters[0];keepers=[p for p in players if p["role"]=="goalkeeper" and p["team"]!=shooter["team"]]
    if len(keepers)!=1:return None,{"reason":"opposing_keeper_missing_or_ambiguous","keepers":len(keepers)}
    keeper=keepers[0];goal_sign=1 if shooter["team"]=="left" else -1
    if len(players)<18:return None,{"reason":"too_few_tracked_players","players":len(players)}
    d=52.5-goal_sign*shooter["x"];l=goal_sign*shooter["y"]
    if d<=0 or d>110:return None,{"reason":"shooter_outside_goal_length"}
    posts=((goal_sign*52.5,-3.66),(goal_sign*52.5,3.66))
    a=np.array(posts[0])-np.array([shooter["x"],shooter["y"]]);b=np.array(posts[1])-np.array([shooter["x"],shooter["y"]])
    angle=math.atan2(abs(a[0]*b[1]-a[1]*b[0]),float(a@b))
    defenders=[p for p in players if p["team"]!=shooter["team"] and p["role"]!="goalkeeper"]
    kd=52.5-goal_sign*keeper["x"];kl=goal_sign*keeper["y"]
    features={"distance_m":math.hypot(d,l),"angle_rad":angle,"keeper_line_distance_m":kd,
        "keeper_lateral_error_m":abs(kl-l*kd/d),"defenders_in_cone":sum(triangle((p["x"],p["y"]),(shooter["x"],shooter["y"]),*posts) for p in defenders),
        "nearest_defender_m":min(math.hypot(p["x"]-shooter["x"],p["y"]-shooter["y"]) for p in defenders)}
    cam=camera(mid);errors=[]
    for player in players:
        if player["bbox"]:
            u,v=project(player["x"],player["y"],cam);box=player["bbox"]
            errors.append(math.hypot(u-(box["x"]+box["w"]/2),v-(box["y"]+box["h"])))
    contact=[p for p in entities(at_contact) if p["id"]==row["actor_id"]]
    source_error=None
    if len(contact)==1:
        ex=goal_sign*(row["raw_attacking_length_fraction"]-.5)*105
        ey=goal_sign*(row["raw_width_fraction"]-.5)*68
        source_error=math.hypot(ex-contact[0]["x"],ey-contact[0]["y"])
    result={**row,**features,"event_distance_m":row["distance_m"],"event_angle_rad":row["angle_rad"],"annotation_frame":number,"video_frame_index":frame["frame_index"],
        "players":players,"shooter":shooter,"keeper":keeper,"goal_sign":goal_sign,"raw_image":frame["image"],"raw_image_sha256":frame["sha256"],
        "snapshot_sha256":__import__("hashlib").sha256((OUT/"snapshots"/f"{mid}_{half}_{number}.json").read_bytes()).hexdigest()}
    audit={"players":len(players),"keepers":len(keepers),"source_contact_error_m":source_error,
        "calibration_median_error_px":float(np.median(errors)),"calibration_max_error_px":float(max(errors)),"source_width":cam["width"],"source_height":cam["height"],
        "keeper_direction_agrees":abs(keeper["x"])<20 or (1 if keeper["x"]>=0 else -1)==goal_sign}
    return result,audit
