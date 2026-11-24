"""Run SpatialTrackerV2 on every preprocessed clip and setting.

Made for UMD's Zaratan cluster (see run_spatracker.sbatch), but works on any Linux GPU machine
prepared with setup_env.sh. Clips are read from ../preprocessed_data with common/track3d_data.py,
the loader shared by all methods, so every method sees the same frames.

Results go to results/<clip>/<mode>/ next to this script:
  tracks.npz   tracks for the viser comparison viewer (2D, 3D in camera and world frames, visibility)
  result.npz   official tapip3d format
  overlay.mp4  2D tracks drawn on the video, the quickest check
  viz.html     official 3D viewer (open in a browser; it loads three.js from the internet)
  log.json     settings, runtime, GPU memory and errors
results/summary.json lists every run. Each run is saved as soon as it ends and finished runs are
skipped, so the job can simply be submitted again after a failure or a timeout.

Settings (--modes):
  rgb         video only: depth, intrinsics and camera poses from the SpatialTrackerV2 front model
              (same as the official inference.py --data_type RGB).
  rgbd_fixed  iPhone LiDAR depth + intrinsics, camera fixed at identity (tripod), fixed_cam=True and
              replace_ratio=1.0 as in the official evaluation script with ground-truth depth.
  rgb_static  (optional) front-model depth and intrinsics, camera fixed at identity, fixed_cam=True.
  rgbd_demo   (optional) iPhone LiDAR depth + intrinsics with the official inference.py RGBD settings
              (camera starts at identity and is optimized: fixed_cam=False, replace_ratio=0.2).
rgb and rgbd_fixed are the two groups used to compare methods: video only, and with LiDAR depth.
"""
import pycolmap  # noqa: F401  (imported first, as in the official inference.py)
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPA_COMMIT = "7e12274c52077860cebfe007a6290777db43b63c"  # must match setup_env.sh

# pour_lid: open the lid and pour water, pen: move a pen from a bowl to the table,
# stir: stir vegetables, shoelace: tie a shoelace, paper_fold: fold paper, shirt_fold: fold a shirt
CLIPS = ["pour_lid", "pen", "stir", "shoelace", "paper_fold", "shirt_fold"]
SETTINGS = dict(
    frame_w=518, frame_h=392,       # 4:3, same size as the official RGB preprocessing output
    target_fps=10, max_frames=200,  # sample 30 fps down to 10 fps; larger stride above 200 frames (model window)
    grid=[32, 24],                  # query grid on frame 0 (768 points)
    extra_query_fracs=[1 / 3, 2 / 3],  # hands enter later, so add query points at these times
    extra_grid=[16, 12],
    lidar_min_conf=1,               # LiDAR confidence is 0/1/2; depth below this level is ignored
    vo_points=756,                  # official default number of support points
)
MODES = {
    "rgb": dict(depth="front", camera="front", fixed_cam=False, replace_ratio=0.2),
    "rgbd_fixed": dict(depth="lidar", camera="identity", fixed_cam=True, replace_ratio=1.0),
    "rgb_static": dict(depth="front", camera="identity", fixed_cam=True, replace_ratio=0.2),
    "rgbd_demo": dict(depth="lidar", camera="identity", fixed_cam=False, replace_ratio=0.2),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", nargs="+", default=CLIPS)
    p.add_argument("--modes", nargs="+", default=["rgb", "rgbd_fixed"], choices=list(MODES))
    p.add_argument("--data-dir", default=os.path.join(ROOT, "preprocessed_data"))
    p.add_argument("--common-dir", default=os.path.join(ROOT, "common"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    p.add_argument("--repo-dir", default=os.path.join(HERE, "work", "SpaTrackerV2"),
                   help="SpatialTrackerV2 checkout made by setup_env.sh")
    p.add_argument("--no-overlay", action="store_true", help="skip overlay.mp4")
    p.add_argument("--no-html", action="store_true", help="skip viz.html")
    a = p.parse_args()
    return dict(SETTINGS, clips=a.clips, modes=a.modes, data_dir=os.path.abspath(a.data_dir),
                common_dir=os.path.abspath(a.common_dir), out_dir=os.path.abspath(a.out_dir),
                repo_dir=os.path.abspath(a.repo_dir), make_overlay=not a.no_overlay,
                make_html=not a.no_html)


CFG = parse_args()
sys.path.insert(0, CFG["repo_dir"])
sys.path.insert(0, CFG["common_dir"])
os.chdir(CFG["repo_dir"])  # the official code uses paths relative to the repo

from models.SpaTrackV2.models.predictor import Predictor  # noqa: E402
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track  # noqa: E402
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image  # noqa: E402
from track3d_data import load_clip  # noqa: E402

AMP = dict(device_type="cuda", dtype=torch.bfloat16)  # the official code runs in bf16
RUN_CONFIG = {k: CFG[k] for k in SETTINGS}
RUN_CONFIG["spa_commit"] = SPA_COMMIT


def grid_queries(t, nx, ny, W, H, margin=8):
    xs = np.linspace(margin, W - 1 - margin, nx)
    ys = np.linspace(margin, H - 1 - margin, ny)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([np.full(gx.size, t), gx.ravel(), gy.ravel()], 1).astype(np.float32)


def make_queries(T, W, H):
    q = [grid_queries(0, *CFG["grid"], W, H)]
    # Hands enter after frame 0, so add coarser grids at later frames.
    for frac in CFG["extra_query_fracs"]:
        t = int(round(frac * (T - 1)))
        if 0 < t < T:
            q.append(grid_queries(t, *CFG["extra_grid"], W, H))
    return np.concatenate(q, 0)


def write_overlay(frames, tracks2d, vis, queries, path, fps, trail=8):
    T, H, W, _ = frames.shape
    qx = queries[:, 1]
    hue = (qx - qx.min()) / (np.ptp(qx) + 1e-6) * 170
    hsv = np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], 1).astype(np.uint8)[None]
    colors = [tuple(int(v) for v in c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]]
    qframe = queries[:, 0].astype(int)
    tmp = path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t in range(T):
        img = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        for n in np.where((qframe <= t) & (vis[t] > 0.5))[0]:
            t0 = max(qframe[n], t - trail)
            pts = np.round(tracks2d[t0:t + 1, n]).astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, colors[n], 1, cv2.LINE_AA)
            cv2.circle(img, (int(pts[-1, 0]), int(pts[-1, 1])), 2, colors[n], -1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", path], check=True)
    os.remove(tmp)


def to_np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


class FrontModel:
    """SpatialTrackerV2 front model (VGGT4Track), loaded on first use and kept on the CPU between
    passes. The pass for the current clip is cached, so rgb and rgb_static share it."""

    def __init__(self):
        self.net = None
        self.key = None
        self.out = None

    def __call__(self, clip, video, sub):
        if self.key == (clip, sub):
            return self.out, 0.0
        self.key, self.out = None, None
        if self.net is None:
            self.net = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
            self.net.eval()
        t0 = time.time()
        vt = preprocess_image(video)[None]
        try:
            self.net.to("cuda")
            with torch.no_grad(), torch.amp.autocast(**AMP):
                pred = self.net(vt.cuda() / 255)
            # same post-processing as the official inference.py (RGB branch)
            out = dict(
                video=vt.squeeze(0),
                extrs=to_np(pred["poses_pred"].squeeze()),
                intrs=to_np(pred["intrs"].squeeze()),
                depth=to_np(pred["points_map"][..., 2].squeeze()),
                unc=to_np(pred["unc_metric"].squeeze()) > 0.5,
            )
            del pred
        finally:
            self.net.to("cpu")
            torch.cuda.empty_cache()
        self.key, self.out = (clip, sub), out
        return out, time.time() - t0


def run_one(clip, mode, d, model, front, sub=1):
    m = MODES[mode]
    video = torch.from_numpy(d["video"][::sub]).permute(0, 3, 1, 2).float()  # T,3,H,W in 0..255
    T = video.shape[0]
    front_s = 0.0
    if m["depth"] == "lidar":
        depth = d["depth"][::sub].astype(np.float32)
        intrs = d["intrinsics"][::sub].astype(np.float32)
        unc = (d["conf"][::sub] >= CFG["lidar_min_conf"]).astype(np.float32)
        extrs = None
    else:
        f, front_s = front(clip, video, sub)
        # copies, so the cached front-model pass stays unchanged for the next setting
        video, depth, intrs = f["video"].clone(), f["depth"].copy(), f["intrs"].copy()
        unc, extrs = f["unc"].copy(), f["extrs"].copy()
    if m["camera"] == "identity":
        extrs = np.repeat(np.eye(4, dtype=np.float32)[None], T, 0)
    _, _, H, W = video.shape

    queries = make_queries(T, W, H)
    with torch.amp.autocast(**AMP):
        (c2w, intrs_o, point_map, conf_depth, track3d, track2d, vis, conf,
         video_o) = model.forward(video, depth=depth, intrs=intrs, extrs=extrs,
                                  queries=queries, fps=1, full_point=False, iters_track=4,
                                  query_no_BA=True, fixed_cam=m["fixed_cam"], stage=1,
                                  unc_metric=unc, support_frame=T - 1,
                                  replace_ratio=m["replace_ratio"])
    return dict(queries=queries, c2w=c2w.float(), intrs=intrs_o.float(), point_map=point_map.float(),
                conf_depth=conf_depth.float(), track3d=track3d.float(), track2d=track2d.float(),
                vis=vis.float(), conf=conf.float(), video=video_o.float(), sub=sub, T=T, H=H, W=W,
                front_s=front_s)


def save_outputs(clip, mode, d, r, out_dir):
    c2w = r["c2w"].cpu()
    tr3 = r["track3d"][..., :3].cpu()
    world = (torch.einsum("tij,tnj->tni", c2w[:, :3, :3], tr3) + c2w[:, :3, 3][:, None]).numpy()
    tr2 = to_np(r["track2d"])
    vis = to_np(r["vis"])[..., 0]
    conf = to_np(r["conf"])[..., 0]
    ow, oh = d["orig_size"]
    W, H = r["W"], r["H"]
    tr2_orig = np.stack([(tr2[..., 0] + 0.5) * ow / W - 0.5, (tr2[..., 1] + 0.5) * oh / H - 0.5], -1)
    frame_idx = d["frame_idx"][::r["sub"]]
    fps = d["fps"] / r["sub"]
    np.savez_compressed(
        os.path.join(out_dir, "tracks.npz"),
        queries=r["queries"], tracks2d=tr2[..., :2], tracks2d_orig=tr2_orig, track_depth=tr2[..., 2],
        tracks3d_cam=to_np(tr3), tracks3d_world=world, vis=vis, conf=conf,
        c2w=to_np(c2w), intrinsics=to_np(r["intrs"]), frame_idx=frame_idx,
        timestamps=d["timestamps"][::r["sub"]], fps=np.float32(fps), size=np.array([W, H]),
        orig_size=np.array(d["orig_size"]), mode=mode, settings=json.dumps(MODES[mode]))

    # tapip3d format (same fields and 336px resizing as the official inference.py)
    video = r["video"]
    point_map, conf_depth, intrs = r["point_map"], r["conf_depth"], r["intrs"].clone()
    scale = min(336 / H, 336 / W)
    if scale < 1:
        size = (int(H * scale), int(W * scale))
        video = F.interpolate(video, size=size, mode="bilinear", align_corners=False)
        point_map = F.interpolate(point_map, size=size, mode="nearest")
        conf_depth = F.interpolate(conf_depth[:, None], size=size, mode="nearest")[:, 0]
        intrs[:, :2, :] *= scale
    depth_save = point_map[:, 2].clone()
    depth_save[conf_depth < 0.5] = 0
    res_path = os.path.join(out_dir, "result.npz")
    np.savez_compressed(
        res_path, coords=world.astype(np.float32), extrinsics=np.linalg.inv(to_np(c2w)),
        intrinsics=to_np(intrs), depths=to_np(depth_save), video=(to_np(video) / 255).astype(np.float16),
        visibs=to_np(r["vis"]), unc_metric=to_np(conf_depth))

    warnings = []
    if CFG["make_overlay"]:
        try:
            frames = np.clip(to_np(r["video"]).transpose(0, 2, 3, 1), 0, 255).astype(np.uint8)
            write_overlay(frames, tr2[..., :2], vis, r["queries"],
                          os.path.join(out_dir, "overlay.mp4"), fps)
        except Exception as e:
            warnings.append(f"overlay failed: {e!r}")
    if CFG["make_html"]:
        try:
            subprocess.run([sys.executable, "tapip3d_viz.py", res_path, "--static-html",
                            os.path.join(out_dir, "viz.html"), "--width", "256", "--height", "192",
                            "--fps", str(max(1, round(fps)))], check=True, cwd=CFG["repo_dir"])
        except Exception as e:
            warnings.append(f"viz.html failed: {e!r}")
    return len(frame_idx), fps, warnings


def print_summary(rows):
    print(f"\n{'clip':11s} {'mode':11s} {'status':7s} {'frames':>6s} {'fps':>5s} {'infer_s':>8s} "
          f"{'peak_gb':>7s}  note")
    for r in rows:
        note = r.get("error") or "; ".join(r.get("warnings") or [])
        print(f"{r['clip']:11s} {r['mode']:11s} {r.get('status', ''):7s} {str(r.get('frames', '')):>6s} "
              f"{str(r.get('fps', '')):>5s} {str(r.get('infer_s', '')):>8s} {str(r.get('peak_gb', '')):>7s}  "
              f"{note[:80]}")


def main():
    gpu = torch.cuda.get_device_name(0)
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU: {gpu} ({gpu_gb:.0f} GB)")
    if gpu_gb < 60:
        print("Note: on a GPU with less than about 60 GB, the rgb setting may run out of memory on "
              "long clips and fall back to fewer frames (see the warnings in the summary).")
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    model.spatrack.track_num = CFG["vo_points"]
    model.eval()
    model.to("cuda")
    front = FrontModel()

    jobs = [(clip, mode) for clip in CFG["clips"] for mode in CFG["modes"]]
    os.makedirs(CFG["out_dir"], exist_ok=True)
    summary_path = os.path.join(CFG["out_dir"], "summary.json")
    rows, t_start, n_new = [], time.time(), 0
    d, d_clip = None, None
    front_sub = {}  # frame step that fit in GPU memory for the front-model settings of each clip
    for i, (clip, mode) in enumerate(jobs, 1):
        tag = f"[{i}/{len(jobs)}] {clip}/{mode}"
        out_dir = os.path.join(CFG["out_dir"], clip, mode)
        log_path = os.path.join(out_dir, "log.json")
        if os.path.exists(log_path):
            old = json.load(open(log_path))
            if old.get("status") == "ok":
                print(f"{tag}: already done, skipped")
                rows.append(old)
                continue
        os.makedirs(out_dir, exist_ok=True)
        log = dict(clip=clip, mode=mode, gpu=gpu, settings=MODES[mode], config=RUN_CONFIG,
                   slurm_job_id=os.environ.get("SLURM_JOB_ID"))
        try:
            if d_clip != clip:
                d, d_clip = None, None
                d = load_clip(os.path.join(CFG["data_dir"], clip), fps=CFG["target_fps"],
                              max_frames=CFG["max_frames"], size=(CFG["frame_w"], CFG["frame_h"]))
                d_clip = clip
                print(f"[data] {clip}: {len(d['frame_idx'])} frames at {d['fps']:.1f} fps, "
                      f"{d['size'][0]}x{d['size'][1]}")
        except Exception as e:
            log.update(status="error", error=f"could not load the clip: {e!r}", traceback=traceback.format_exc())
            d, d_clip = None, None

        sub = front_sub.get(clip, 1) if MODES[mode]["depth"] == "front" else 1
        while d is not None:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                r = run_one(clip, mode, d, model, front, sub)
                infer_s = time.time() - t0
                n_frames, fps, warnings = save_outputs(clip, mode, d, r, out_dir)
                if sub > 1:
                    warnings.append(f"out of GPU memory at full length; kept 1 of every {sub} frames ({fps:.1f} fps)")
                log.update(status="ok", frames=n_frames, fps=round(fps, 2), subsample=sub,
                           queries=int(len(r["queries"])), front_s=round(r["front_s"], 1),
                           infer_s=round(infer_s, 1), total_s=round(time.time() - t0, 1),
                           peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), warnings=warnings)
                if MODES[mode]["depth"] == "front":
                    front_sub[clip] = sub
                del r
                break
            except torch.cuda.OutOfMemoryError:
                if sub >= 4:
                    log.update(status="oom", error="out of GPU memory even at 1/4 of the frames")
                    break
                sub *= 2
                print(f"{tag}: out of GPU memory, retrying with 1 of every {sub} frames")
            except Exception as e:  # keep going with the other runs
                log.update(status="error", error=repr(e), traceback=traceback.format_exc())
                break

        json.dump(log, open(log_path, "w"), indent=2, ensure_ascii=False)
        rows.append(log)
        with open(summary_path, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        n_new += 1
        elapsed = time.time() - t_start
        left = elapsed / n_new * (len(jobs) - i)
        msg = {k: log.get(k) for k in ("frames", "fps", "queries", "infer_s", "peak_gb", "error")}
        print(f"{tag}: {log['status']} {msg} | {elapsed / 60:.1f} min so far, about {left / 60:.0f} min left")

    print_summary(rows)
    n_ok = sum(r.get("status") == "ok" for r in rows)
    print(f"\nDone: {n_ok} of {len(jobs)} runs ok. Results: {CFG['out_dir']}")
    sys.exit(0 if n_ok == len(jobs) else 1)


if __name__ == "__main__":
    main()
