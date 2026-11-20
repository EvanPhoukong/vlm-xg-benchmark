"""Prepare exact SoccerTrack video frames from its official public mirror."""
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from html.parser import HTMLParser
import re

import av
from PIL import Image
import requests

import study as s


class FolderLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = {}
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            url = dict(attrs).get("href", "")
            match = re.search(r"/(?:drive/folders|file/d)/([A-Za-z0-9_-]+)", url)
            if match:
                self.current = [match[1], []]

    def handle_data(self, value):
        if self.current and value.strip():
            self.current[1].append(value.strip())

    def handle_endtag(self, tag):
        if tag == "a" and self.current:
            identifier, parts = self.current
            self.links[" ".join(parts)] = identifier
            self.current = None


def folder(identifier):
    result = requests.get("https://drive.google.com/embeddedfolderview",
                          params={"id": identifier}, timeout=30)
    result.raise_for_status()
    parser = FolderLinks()
    parser.feed(result.text)
    return parser.links


class RemoteVideo:
    block_size = 4 * 1024 * 1024

    def __init__(self, identifier, size):
        self.session = requests.Session()
        self.identifier, self.size = identifier, size
        self.offset, self.transferred = 0, 0
        self.cache = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        self.offset = {0: 0, 1: self.offset, 2: self.size}[whence] + offset
        if self.offset < 0:
            raise ValueError("Negative video offset")
        return self.offset

    def read(self, count=-1):
        end = min(self.size, self.offset + count) if count >= 0 else self.size
        if end - self.offset > 32 * 1024 * 1024:
            raise ValueError("Unexpected video decoder read")
        output = bytearray()
        while self.offset < end:
            block = self.offset // self.block_size * self.block_size
            stop = min(block + self.block_size, self.size)
            if block not in self.cache:
                params = {"id": self.identifier, "export": "download", "confirm": "t",
                          "range_start": str(block)}
                for attempt in range(3):
                    with self.session.get(
                        "https://drive.usercontent.google.com/download", params=params,
                        headers={"Range": f"bytes={block}-{stop-1}", "Accept-Encoding": "identity"},
                        stream=True, timeout=(15, 60),
                    ) as response:
                        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                            continue
                        if response.status_code != 206 or response.headers.get("Content-Range") != (
                            f"bytes {block}-{stop-1}/{self.size}"
                        ):
                            raise RuntimeError(f"Unexpected video range response ({response.status_code})")
                        chunk = response.raw.read(stop - block + 1)
                        break
                if len(chunk) != stop - block:
                    raise RuntimeError("Incomplete video download range")
                self.cache[block] = chunk
                self.transferred += len(chunk)
                if len(self.cache) > 12:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(block)
            take = min(end, stop)
            output.extend(self.cache[block][self.offset-block:take-block])
            self.offset = take
        return bytes(output)

    def close(self):
        self.session.close()


def prepare_group(mid, half, jobs, video_folders):
    suffix = "1st" if half == 1 else "2nd"
    filename = f"{mid}_panorama_{suffix}_half.mp4"
    local = s.c.DATA / "videos" / mid / filename
    remote = None
    if local.stat().st_size > 200:
        source = str(local)
    else:
        size = int(re.search(r"^size (\d+)$", local.read_text(), re.M)[1])
        remote = RemoteVideo(folder(video_folders[mid])[filename], size)
        source = remote
    records = {}
    try:
        with av.open(source) as container:
            stream = container.streams.video[0]
            stream.thread_type = "FRAME"
            if Fraction(stream.average_rate) != 25:
                raise ValueError("Expected 25 fps SoccerTrack video")
            first_pts = next(container.decode(video=0)).pts
            for job in sorted(jobs, key=lambda j: j["frame_index"]):
                delta = Fraction(job["frame_index"], 25) / stream.time_base
                if delta.denominator != 1:
                    raise ValueError("Frame does not align with an exact decoder timestamp")
                target = first_pts + int(delta)
                container.seek(target, stream=stream, backward=True, any_frame=False)
                found = next(frame for frame in container.decode(video=0)
                             if frame.pts is not None and frame.pts >= target)
                if found.pts != target:
                    raise ValueError("Exact pre-shot video frame was not located")
                image = found.to_image().convert("RGB")
                ratio = min(1., 1536 / max(image.size))
                image = image.resize(tuple(round(value * ratio) for value in image.size),
                                     Image.Resampling.LANCZOS)
                path = s.c.ROOT / job["image"]
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".partial")
                image.save(temporary, format="PNG")
                temporary.replace(path)
                records[job["shot_id"]] = {
                    "sha256": s.c.sha(path), "size": list(image.size),
                    "source_video": str(local.relative_to(s.c.DATA)),
                    "first_pts": first_pts, "selected_pts": target,
                    "time_base": str(stream.time_base), "frame_index": job["frame_index"],
                }
        print(f"Prepared {mid} half {half}, {len(records)} frames", flush=True)
        return records
    finally:
        if remote:
            remote.close()


def main():
    jobs = s.frame_jobs()
    audit_path = s.OUT / "frame_audit.json"
    audit = s.c.read(audit_path) if audit_path.exists() else {}
    groups = defaultdict(list)
    for job in jobs:
        if job["reuse_original"]:
            continue
        path = s.c.ROOT / job["image"]
        record = audit.get(job["shot_id"])
        if record and path.is_file() and s.c.sha(path) == record["sha256"]:
            continue
        groups[job["match_id"], job["half"]].append(job)
    if groups:
        root = folder("1N2Qx2qkFgRtpbHitl2Vh6sLVYGgqkWwn")
        video_folders = folder(root["videos"])
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(prepare_group, mid, half, rows, video_folders)
                       for (mid, half), rows in sorted(groups.items())]
            for future in as_completed(futures):
                audit.update(future.result())
                s.c.save(audit_path, audit)
    cohort = []
    original = {r["shot_id"]: r for r in s.c.read(s.OLD / "reused_sol.json")}
    for job in jobs:
        expected = (original[job["shot_id"]]["image_sha256"] if job["reuse_original"]
                    else audit[job["shot_id"]]["sha256"])
        if s.c.sha(s.c.ROOT / job["image"]) != expected:
            raise ValueError("Frame hash differs")
        cohort.append({**job, "sha256": expected})
    s.c.freeze(s.OUT / "frames.json", cohort)
    print(f"All {len(cohort)} frames verified", flush=True)


if __name__ == "__main__":
    main()

