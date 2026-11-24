"""Run DELTA (sparse 3D tracking) on every preprocessed clip and setting.

Works on a Runpod pod or on UMD's Zaratan cluster (see run_delta.sbatch); prepare the machine once
with setup_env.sh. Clips are read with common/track3d_data.py, so DELTA sees the same frames as every
other method, and it tracks the same query points as the SpatialTrackerV2 runs.

Results go to results/<clip>/<mode>/ next to this script, in the same format as spatracker_v2, so
viewer/track3d_viewer.py shows them next to the other methods:
  tracks.npz   2D tracks, 3D tracks in camera and world frames, visibility
  result.npz   input depth, intrinsics, camera poses and frames (for the viewer's point cloud)
  overlay.mp4  2D tracks drawn on the video
  log.json     settings, runtime, GPU memory and errors
results/summary.json lists every run. Finished runs are skipped, so the job can be resubmitted.

Settings (--modes):
  rgb         video only: per-frame depth and intrinsics from UniDepth V2 (ViT-L), the official
              DELTA default depth source; tracks are lifted to 3D with UniDepth's intrinsics.
  rgbd_fixed  iPhone LiDAR depth and intrinsics.
DELTA tracks in the camera frame and the camera was on a tripod, so camera and world frames are equal.
"""
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
DELTA_COMMIT = "3367cda1c74d19e73296165f9826b213211678dd"  # must match setup_env.sh

CLIPS = ["pour_lid", "pen", "stir", "shoelace", "paper_fold", "shirt_fold"]
SETTINGS = dict(
    frame_w=512, frame_h=384,       # DELTA's model resolution (4:3, like the recordings)
    target_fps=10, max_frames=200,  # same frame selection as spatracker_v2
    lidar_min_conf=0,               # LiDAR depth is dense; DELTA gets every pixel
    unidepth_chunk=32,              # frames per UniDepth batch (official demo value)
    upsample_factor=4, use_fp16=False,  # official sparse demo defaults
)
# Query points: the exact physical points used by spatracker_v2 (grid at frame 0 plus coarser grids
# at 1/3 and 2/3 of the clip), defined in its 518x392 frames and mapped to DELTA's frames.
REF_SIZE = (518, 392)
REF_GRID, REF_EXTRA_FRACS, REF_EXTRA_GRID, REF_MARGIN = (32, 24), (1 / 3, 2 / 3), (16, 12), 8
MODES = {
    "rgb": dict(depth="unidepth"),
    "rgbd_fixed": dict(depth="lidar"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", nargs="+", default=CLIPS)
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    p.add_argument("--data-dir", default=os.path.join(ROOT, "preprocessed_data"))
    p.add_argument("--common-dir", default=os.path.join(ROOT, "common"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    p.add_argument("--repo-dir", default=os.path.join(HERE, "work", "DELTA"),
                   help="DELTA checkout made by setup_env.sh")
    p.add_argument("--ckpt", default=None, help="default: <repo-dir>/checkpoints/densetrack3d.pth")
    p.add_argument("--no-overlay", action="store_true", help="skip overlay.mp4")
    a = p.parse_args()
    repo = os.path.abspath(a.repo_dir)
    return dict(SETTINGS, clips=a.clips, modes=a.modes, data_dir=os.path.abspath(a.data_dir),
                common_dir=os.path.abspath(a.common_dir), out_dir=os.path.abspath(a.out_dir),
                repo_dir=repo, make_overlay=not a.no_overlay,
                ckpt=os.path.abspath(a.ckpt or os.path.join(repo, "checkpoints", "densetrack3d.pth")))


CFG = parse_args()
sys.path.insert(0, os.path.join(CFG["repo_dir"], "submodules", "UniDepth"))
sys.path.insert(0, CFG["repo_dir"])
sys.path.insert(0, CFG["common_dir"])
os.chdir(CFG["repo_dir"])  # the official code uses paths relative to the repo

from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D  # noqa: E402
from densetrack3d.models.predictor.predictor import Predictor3D  # noqa: E402
from track3d_data import load_clip, nearest_resize, scale_intrinsics  # noqa: E402

RUN_CONFIG = {k: CFG[k] for k in SETTINGS}
RUN_CONFIG["delta_commit"] = DELTA_COMMIT


def to_size(xy, src, dst):
    """Map pixel coordinates between frame sizes (pixel-center convention)."""
    return (np.asarray(xy, np.float64) + 0.5) * np.asarray(dst) / np.asarray(src) - 0.5


def grid_queries(t, nx, ny, W, H, margin):
    xs = np.linspace(margin, W - 1 - margin, nx)
    ys = np.linspace(margin, H - 1 - margin, ny)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([np.full(gx.size, t), gx.ravel(), gy.ravel()], 1)


def make_queries(T, size):
    """Same query points as spatracker_v2, as [t, x, y] in pixels of `size`."""
    q = [grid_queries(0, *REF_GRID, *REF_SIZE, REF_MARGIN)]
    for frac in REF_EXTRA_FRACS:
        t = int(round(frac * (T - 1)))
        if 0 < t < T:
            q.append(grid_queries(t, *REF_EXTRA_GRID, *REF_SIZE, REF_MARGIN))
    q = np.concatenate(q, 0)
    q[:, 1:] = to_size(q[:, 1:], REF_SIZE, size)
    return q.astype(np.float32)


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


class UniDepth:
    """UniDepth V2 (ViT-L), loaded on first use; the pass for the current clip is cached."""

    def __init__(self):
        self.net = None
        self.key = None
        self.out = None

    def __call__(self, clip, video):
        if self.key == clip:
            return self.out, 0.0
        if self.net is None:
            from unidepth.models import UniDepthV2
            self.net = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").eval()
        t0 = time.time()
        self.net.to("cuda")
        depth, intrs = [], []
        try:
            with torch.inference_mode():
                frames = torch.from_numpy(video).permute(0, 3, 1, 2)  # T,3,H,W uint8
                for chunk in torch.split(frames, CFG["unidepth_chunk"], dim=0):
                    pred = self.net.infer(chunk.cuda())  # same call as the official demo
                    depth.append(pred["depth"].squeeze(1).float().cpu().numpy())
                    intrs.append(pred["intrinsics"].float().cpu().numpy())
        finally:
            self.net.to("cpu")
            torch.cuda.empty_cache()
        self.key = clip
        self.out = dict(depth=np.concatenate(depth).astype(np.float32),
                        intrinsics=np.concatenate(intrs).astype(np.float32))
        return self.out, time.time() - t0


def run_one(clip, mode, d, predictor, unidepth):
    video = d["video"]  # T,H,W,3 uint8 at frame_w x frame_h
    T, H, W, _ = video.shape
    depth_s = 0.0
    if MODES[mode]["depth"] == "lidar":
        depth = d["depth"].astype(np.float32)
        intrs = d["intrinsics"].astype(np.float32)
    else:
        u, depth_s = unidepth(clip, video)
        depth, intrs = u["depth"].copy(), u["intrinsics"].copy()
    # one pinhole model per run, as DELTA's 3D conversion takes a single intrinsics matrix
    K = np.median(intrs, axis=0).astype(np.float32)

    queries = make_queries(T, (W, H))
    vid = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().cuda()  # 1,T,3,H,W in 0..255
    vdepth = torch.from_numpy(depth)[None, :, None].float().cuda()           # 1,T,1,H,W
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=CFG["use_fp16"]):
        out = predictor(vid, vdepth, queries=torch.from_numpy(queries)[None].cuda(), segm_mask=None,
                        grid_size=0, grid_query_frame=0, backward_tracking=False,
                        predefined_intrs=torch.from_numpy(K).cuda())
    uv = out["trajs_uv"][0].float().cpu().numpy()            # T,N,2 in frame pixels
    d_e = out["trajs_depth"][0, ..., 0].float().cpu().numpy()  # T,N
    vis = out["vis"][0].float().cpu().numpy()                 # T,N (0/1)
    conf = out["conf"][0].float().cpu().numpy() if out.get("conf") is not None else vis
    xyz = out["trajs_3d_dict"]["coords"][0].float().cpu().numpy()  # T,N,3 camera frame
    return dict(queries=queries, uv=uv, track_depth=d_e, vis=vis, conf=conf.reshape(vis.shape),
                xyz=xyz, depth=depth, intrinsics=intrs, K=K, T=T, H=H, W=W, depth_s=depth_s)


def save_outputs(clip, mode, d, r, out_dir):
    T, H, W = r["T"], r["H"], r["W"]
    ow, oh = d["orig_size"]
    eye = np.repeat(np.eye(4, dtype=np.float32)[None], T, 0)
    np.savez_compressed(
        os.path.join(out_dir, "tracks.npz"),
        queries=r["queries"], tracks2d=r["uv"], tracks2d_orig=to_size(r["uv"], (W, H), (ow, oh)).astype(np.float32),
        track_depth=r["track_depth"], tracks3d_cam=r["xyz"], tracks3d_world=r["xyz"],
        vis=r["vis"], conf=r["conf"], c2w=eye, intrinsics=np.repeat(r["K"][None], T, 0),
        frame_idx=d["frame_idx"], timestamps=d["timestamps"], fps=np.float32(d["fps"]),
        size=np.array([W, H]), orig_size=np.array(d["orig_size"]), mode=mode,
        settings=json.dumps(MODES[mode]))

    # per-frame geometry for the viewer's point cloud, at most 336 px wide like spatracker_v2
    scale = min(1.0, 336 / W, 336 / H)
    h, w = int(H * scale), int(W * scale)
    # Match the pixel centers of the resized RGB when sampling depth and projecting the cloud.
    depth = nearest_resize(r["depth"], w, h)
    video = F.interpolate(torch.from_numpy(d["video"]).permute(0, 3, 1, 2).float() / 255, size=(h, w),
                          mode="bilinear", align_corners=False).numpy()
    K = np.repeat(scale_intrinsics(r["K"], (W, H), (w, h))[None], T, 0)
    np.savez_compressed(
        os.path.join(out_dir, "result.npz"), coords=r["xyz"].astype(np.float32), extrinsics=eye,
        intrinsics=K.astype(np.float32), depths=depth.astype(np.float32), video=video.astype(np.float16),
        visibs=r["vis"][..., None].astype(np.float32))

    warnings = []
    if CFG["make_overlay"]:
        try:
            write_overlay(d["video"], r["uv"], r["vis"], r["queries"], os.path.join(out_dir, "overlay.mp4"),
                          d["fps"])
        except Exception as e:
            warnings.append(f"overlay failed: {e!r}")
    return warnings


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
    print(f"GPU: {gpu} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)")
    model = DenseTrack3D(stride=4, window_len=16, add_space_attn=True, num_virtual_tracks=64,
                         model_resolution=(384, 512), upsample_factor=CFG["upsample_factor"])
    with open(CFG["ckpt"], "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)  # as in the official demo
    predictor = Predictor3D(model=model).eval().cuda()
    unidepth = UniDepth()

    jobs = [(clip, mode) for clip in CFG["clips"] for mode in CFG["modes"]]
    os.makedirs(CFG["out_dir"], exist_ok=True)
    summary_path = os.path.join(CFG["out_dir"], "summary.json")
    rows, t_start, n_new = [], time.time(), 0

    def record_run(row):
        rows.append(row)
        with open(summary_path, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    d, d_clip = None, None
    for i, (clip, mode) in enumerate(jobs, 1):
        tag = f"[{i}/{len(jobs)}] {clip}/{mode}"
        out_dir = os.path.join(CFG["out_dir"], clip, mode)
        log_path = os.path.join(out_dir, "log.json")
        if os.path.exists(log_path):
            old = json.load(open(log_path))
            if old.get("status") == "ok":
                print(f"{tag}: already done, skipped")
                record_run(old)
                continue
        os.makedirs(out_dir, exist_ok=True)
        log = dict(clip=clip, mode=mode, gpu=gpu, settings=MODES[mode], config=RUN_CONFIG)
        t0 = time.time()
        try:
            if d_clip != clip:
                d, d_clip = None, None
                d = load_clip(os.path.join(CFG["data_dir"], clip), fps=CFG["target_fps"],
                              max_frames=CFG["max_frames"], size=(CFG["frame_w"], CFG["frame_h"]))
                d_clip = clip
                print(f"[data] {clip}: {len(d['frame_idx'])} frames at {d['fps']:.1f} fps, "
                      f"{d['size'][0]}x{d['size'][1]}")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            infer_start = time.time()
            r = run_one(clip, mode, d, predictor, unidepth)
            infer_s = time.time() - infer_start
            warnings = save_outputs(clip, mode, d, r, out_dir)
            log.update(status="ok", frames=int(r["T"]), fps=round(float(d["fps"]), 2),
                       queries=int(len(r["queries"])), depth_s=round(r["depth_s"], 1),
                       infer_s=round(infer_s, 1), total_s=round(time.time() - t0, 1),
                       peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                       intrinsics_used=r["K"].round(2).tolist(), warnings=warnings)
            del r
        except Exception as e:  # keep going with the other runs
            log.update(status="error", error=repr(e), traceback=traceback.format_exc())
        json.dump(log, open(log_path, "w"), indent=2, ensure_ascii=False)
        record_run(log)
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
