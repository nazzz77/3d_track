"""Run the pinned MotionCrafter diffusion demo on the shared RGB clips.

Each results/<clip>/<clip>.npz is the untouched output of the official run.py.
The method has no metric scale, saved intrinsics, camera poses or point tracks.
"""
import argparse
import ast
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLIPS = ("pour_lid", "pen", "stir", "shoelace", "paper_fold", "shirt_fold")
SIZE = (512, 384)
COMMIT = "9cb4e9679f5f34e249945544052464ef46324bc2"
MC = "TencentARC/MotionCrafter"
SVD = "stabilityai/stable-video-diffusion-img2vid-xt"
SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CHECKPOINTS = {MC: "fc7b18d5657184607bf4501b02d64ada7540b4e3",
               SVD: "9e43909513c6714f1bc78bcb44d96e733cd242aa",
               SD15: "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"}


def shared_clip(data_dir, name):
    common = str(ROOT / "common")
    if common not in sys.path:
        sys.path.insert(0, common)
    from track3d_data import load_clip

    return load_clip(data_dir / name, fps=10, max_frames=200, size=SIZE)


def verify_cache(hub):
    for repo, revision in CHECKPOINTS.items():
        model = hub / ("models--" + repo.replace("/", "--"))
        actual = (model / "refs/main").read_text().strip()
        if actual != revision or not (model / "snapshots" / revision).is_dir():
            raise ValueError(f"Unexpected offline snapshot for {repo}: {actual}, expected {revision}")


def official_settings(repo):
    """Read simple defaults without importing torch. The pipe call overrides overlap."""
    tree = ast.parse((repo / "run.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    args = main.args
    defaults = {arg.arg: ast.literal_eval(value) for arg, value in
                zip(args.args[-len(args.defaults):], args.defaults)}
    call, = (n for n in ast.walk(main) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "pipe")
    overlap = next(kw.value for kw in call.keywords if kw.arg == "overlap")
    defaults["signature_overlap"] = defaults["overlap"]
    defaults["overlap"] = ast.literal_eval(overlap)
    for provided in ("cache_dir", "save_folder"):
        del defaults[provided]
    return defaults


def window_starts(count, settings):
    width = min(count, settings["window_size"])
    overlap = 0 if count <= width else settings["overlap"]
    if width <= overlap or overlap < 0:
        raise ValueError("Invalid frame count or window stride")
    return list(range(0, count - overlap, width - overlap))


def load_maps(path, count):
    """Reading complete NPY members also checks their ZIP checksums."""
    shape = (count, SIZE[1], SIZE[0])
    maps = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in ("point_map", "valid_mask", "scene_flow", "deform_mask"):
            mask = key.endswith("_mask")
            array = archive[key]
            if array.dtype != (np.bool_ if mask else np.float16) or array.shape != (shape if mask else shape + (3,)):
                raise ValueError(f"{key} has unexpected shape or dtype: {array.shape} {array.dtype}")
            if not mask and any(not np.isfinite(frame).all() for frame in array):
                raise ValueError(f"{key} contains non-finite coordinates")
            maps[key] = array
    return maps


def clip_stats(maps):
    """Exact percentiles over valid depth and jointly valid flow, in model units."""
    valid = maps["valid_mask"]
    deform = maps["deform_mask"]
    depths, magnitudes = [], []
    for t, mask in enumerate(valid):
        depths.append(maps["point_map"][t, :, :, 2][mask].astype(np.float32))
        flow = maps["scene_flow"][t][mask & deform[t]].astype(np.float32)
        magnitudes.append(np.linalg.norm(flow, axis=1))

    def percentiles(parts):
        values = np.concatenate(parts)
        levels = [1, 5, 50, 95, 99]
        if not values.size:
            return {str(level): None for level in levels}
        return dict(zip(map(str, levels), np.percentile(values, levels).tolist()))

    return dict(valid_depth_percentiles=percentiles(depths),
                valid_deform_flow_magnitude_percentiles=percentiles(magnitudes),
                deform_mask_fraction=float(deform.mean()))


def write_video(path, frames, fps):
    """All-intra lossless RGB H.264. Check the actual decoder used by run.py."""
    from decord import VideoReader, cpu

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".part.mp4")
    h, w = frames.shape[1:3]
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{w}x{h}", "-r", str(fps),
           "-i", "pipe:0", "-an", "-c:v", "libx264rgb", "-crf", "0", "-preset", "fast",
           "-g", "1", "-bf", "0", "-pix_fmt", "rgb24", str(temp)]
    try:
        subprocess.run(cmd, input=memoryview(frames).cast("B"), check=True)
        video = VideoReader(str(temp), ctx=cpu(0))
        if len(video) != len(frames):
            raise ValueError(f"Decord counted {len(video)} frames, expected {len(frames)}")
        for start in range(0, len(frames), 8):
            batch = frames[start:start + 8]
            decoded = video.get_batch(list(range(start, start + len(batch)))).asnumpy()
            if not np.array_equal(decoded, batch):
                raise ValueError(f"Decord RGB round trip differs in batch starting at frame {start}")
        del video
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def poll_gpu(stop, log):
    """Poll device-wide memory on the single assigned GPU (also works in PID namespaces)."""
    selector = (os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS")
                or os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")[0]
    cmd = ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"]
    try:
        while not stop.is_set():
            raw = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=5)
            rows = [[part.strip() for part in line.split(",")] for line in raw.splitlines()]
            if len(rows) != 1:
                rows = [row for row in rows if row[0] == selector or row[1].startswith(selector)]
            if len(rows) != 1:
                raise ValueError(f"Cannot identify the assigned GPU for selector {selector!r}")
            _, uuid, mib = rows[0]
            log["gpu_uuid"] = uuid
            log["peak_gpu_mib"] = max(int(mib), log["peak_gpu_mib"] or 0)
            stop.wait(0.5)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        log["warnings"].append(f"GPU memory polling stopped: {exc}")


def official_run(repo, hub, video, output, log):
    cmd = [sys.executable, "run.py", "--video_path", str(video), "--save_folder", str(output),
           "--cache_dir", str(hub)]
    env = dict(os.environ, HF_HOME=str(hub.parent), HF_HUB_CACHE=str(hub),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFUSERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1")
    log["command"] = cmd
    stop = threading.Event()
    monitor = threading.Thread(target=poll_gpu, args=(stop, log), daemon=True)
    start = time.monotonic()
    monitor.start()
    try:
        with tempfile.TemporaryFile() as stderr:
            with subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=stderr,
                                  text=True, encoding="utf-8", errors="replace") as child:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    if "falling back" in line.lower():
                        log["warnings"].append(line.strip())
            stderr.seek(max(0, stderr.tell() - 8192))
            tail = stderr.read().decode("utf-8", errors="replace")
            for line in tail.splitlines():
                if "Warning:" in line and line.strip() not in log["warnings"]:
                    log["warnings"].append(line.strip())
            if child.returncode:
                log["stderr_tail"] = tail
                raise RuntimeError(f"Official run.py exited with code {child.returncode}")
    finally:
        log["subprocess_seconds"] = round(time.monotonic() - start, 3)
        stop.set()
        monitor.join()


def write_json(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temp.unlink()
            raise
    temp.replace(path)


@contextmanager
def directory_lock(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def run_clip(name, data_dir, work, repo, hub, output, settings):
    path = output / f"{name}.npz"
    log_path = output / "log.json"
    warnings = []
    if log_path.exists():
        try:
            old = json.loads(log_path.read_text())
            if (old["status"] == "ok" and old["clip"] == name and old["official_commit"] == COMMIT
                    and old["checkpoint_revisions"] == CHECKPOINTS and old["settings"] == settings
                    and path.stat().st_size == old["archive_bytes"]):
                with zipfile.ZipFile(path) as archive:
                    damaged = archive.testzip()
                    if damaged is not None:
                        raise ValueError(f"Failed checksum for {damaged}")
                print(f"{name}: already done, skipped")
                return old
        except (KeyError, ValueError, TypeError, OSError, zipfile.BadZipFile) as exc:
            warnings.append(f"Retrying an incomplete result: {exc}")

    start = time.monotonic()
    log = dict(status="running", clip=name, official_commit=COMMIT, checkpoint_revisions=CHECKPOINTS,
               settings=settings, size=list(SIZE), T=None, frame_idx=[], timestamps=[], fps=None,
               window_starts=[], metric_scale=False, camera_poses_saved=False, intrinsics_saved=False,
               viewer_poses="identity", warnings=warnings, peak_gpu_mib=None,
               gpu_memory_measurement="Device-wide nvidia-smi samples every 0.5 s during run.py",
               subprocess_seconds=None)
    write_json(log_path, log)
    try:
        data = shared_clip(data_dir, name)
        frames = data.pop("video")
        count = len(frames)
        log.update(T=count, frame_idx=data["frame_idx"].tolist(), timestamps=data["timestamps"].tolist(),
                   fps=data["fps"], window_starts=window_starts(count, settings))
        del data
        video = work / "videos" / f"{name}.mp4"
        write_video(video, frames, log["fps"])
        del frames
        copy = output / video.name
        before = copy.stat().st_mtime_ns if copy.exists() else None
        try:
            official_run(repo, hub, video, output, log)
        finally:
            if copy.exists() and copy.stat().st_mtime_ns != before:
                copy.unlink()
        log["stats"] = clip_stats(load_maps(path, count))
        log.update(status="ok", archive_bytes=path.stat().st_size)
    except Exception as exc:
        log.update(status="error", error=str(exc), traceback=traceback.format_exc())
    log["runtime_seconds"] = round(time.monotonic() - start, 3)
    write_json(log_path, log)
    return log


def update_summary(results):
    # Keep clips from earlier selections when a subset or a concurrent job finishes.
    with directory_lock(results):
        rows = []
        for path in sorted(results.glob("*/log.json")):
            try:
                rows.append(json.loads(path.read_text()))
            except (ValueError, OSError) as exc:
                rows.append(dict(clip=path.parent.name, status="error", error=str(exc)))
        write_json(results / "summary.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", choices=CLIPS, default=CLIPS)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "preprocessed_data")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("MOTIONCRAFTER_WORK", HERE / "work")))
    args = parser.parse_args()
    work, results = args.work_dir.resolve(), args.out_dir.resolve()
    repo, hub = work / "MotionCrafter", work / "hf_cache/hub"
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        parser.exit(1, f"Unexpected MotionCrafter commit: {actual}\n")
    subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", "run.py", "motioncrafter"], check=True)
    verify_cache(hub)
    if (repo / "cache").resolve() != hub.resolve():
        parser.exit(1, "The nested SD 1.5 cache is missing. Run setup_env.sh.\n")
    settings = official_settings(repo)
    rows = []
    for name in dict.fromkeys(args.clips):
        output = results / name
        output.mkdir(parents=True, exist_ok=True)
        with directory_lock(output):
            row = run_clip(name, args.data_dir, work, repo, hub, output, settings)
        rows.append(row)
        update_summary(results)
        print(f"{name}: {row['status']} {row.get('error', '')}", flush=True)
    print(f"\n{'Clip':12} {'Status':8} {'Frames':>6} {'Run (s)':>10} {'GPU (MiB)':>10}")
    for row in rows:
        print(f"{row['clip']:12} {row['status']:8} {str(row['T']):>6} "
              f"{row['runtime_seconds']:10.2f} {str(row['peak_gpu_mib']):>10}")
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    sys.exit(main())
