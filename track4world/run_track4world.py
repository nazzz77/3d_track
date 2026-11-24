"""Run metric, video-only Track4World on the shared Stray clips.

rgb uses demo.load_model's eval/v1.json, seqlen=16, use_3d=True and the DA3
branch of --coordinate world_depthanythingv3. The local track4world_da3.pth
is loaded with the official nested-backbone key remapping. --metric_scale is
on and --use_original_backbone is off. No LiDAR or ARKit data enters the model.

After load_clip(fps=10, max_frames=200), run_demo's --image_size 640 and
INTER_LINEAR yield 640x448 from stored 960x720 RGB. Each distinct query frame
starts a 3d_ff run with --Ts -1 (every remaining frame) and --inference_iters 4.
We call infer directly because forward_video3d_ff discards intrinsics. We keep
run_demo's outer CUDA fp16 autocast, forward_video3d_ff's inner CUDA float32
context and all infer defaults, including force_projection and use_da3_focal.

infer adds the reference point map to the predicted scene flow, then unprojects
the target pixel and depth. Its returned flow_3d[t] is in target camera t, as
used by evaluation/track/eval.py:transform_to_first_frame. Both flow maps are
indexed by pixels of the suffix's first frame. process_geometry inverts DA3
OpenCV world-to-camera extrinsics. All groups use the full run's poses relative
to camera 0. Beyond get_fmaps' 128-frame DA3 batch, the next batch has its own
world origin. The first later suffix sees frames 127 and 128 in one batch and
supplies their relative pose to connect the two full-run coordinate systems.

Results are written under results/<clip>/rgb/ for viewer/track3d_viewer.py.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLIPS = ["pour_lid", "pen", "stir", "shoelace", "paper_fold", "shirt_fold"]
SETTINGS = dict(
    coordinate="world_depthanythingv3", metric_scale=True, mode="3d_ff", Ts=-1,
    image_size=640, inference_iters=4, seqlen=16, use_3d=True,
    force_projection=True, use_da3_focal=True, target_fps=10, max_frames=200,
    config_path="track4world/config/eval/v1.json", visibility_threshold=0.5,
    track4world_commit="1c2a0f814d8352d2750527b70a6c546e056c2c67",
)
REF_SIZE = (518, 392)
REF_GRID, REF_EXTRA_FRACS, REF_EXTRA_GRID, REF_MARGIN = (32, 24), (1 / 3, 2 / 3), (16, 12), 8
DA3_BATCH = 128  # Track4World.get_fmaps, depthanythingv3 branch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", nargs="+", default=CLIPS)
    p.add_argument("--modes", nargs="+", choices=["rgb"], default=["rgb"])
    p.add_argument("--data-dir", type=Path, default=ROOT / "preprocessed_data")
    p.add_argument("--common-dir", type=Path, default=ROOT / "common")
    p.add_argument("--out-dir", type=Path, default=HERE / "results")
    work = Path(os.environ.get("TRACK4WORLD_WORK", HERE / "work"))
    p.add_argument("--repo-dir", type=Path, default=work / "Track4World")
    p.add_argument("--ckpt", type=Path, help="Defaults to <repo-dir>/checkpoints/track4world_da3.pth")
    p.add_argument("--no-overlay", action="store_true", help="Skip overlay.mp4")
    args = p.parse_args()
    args.repo_dir = args.repo_dir.resolve()
    args.ckpt = (args.ckpt or args.repo_dir / "checkpoints/track4world_da3.pth").resolve()
    return args


def configure_repo(repo):
    """Expose the official tree, its absolute DA3 imports and the pinned utils3d checkout."""
    utils = repo / "utils3d"
    paths = [repo, repo / "track4world/nets/external"]
    if (utils / "utils3d/__init__.py").is_file():
        paths.insert(0, utils)
    elif not (utils / "__init__.py").is_file():
        raise ImportError(f"Cannot find the pinned utils3d package under {utils}")
    sys.path[:0] = [str(path) for path in paths]


def build_model(repo, checkpoint):
    """CPU half of demo.load_model, shared by online setup and offline inference."""
    import torch

    configure_repo(repo)
    from track4world.nets.model import Track4World

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    with (repo / SETTINGS["config_path"]).open() as f:
        config = json.load(f)
    model = Track4World(**config["model"], seqlen=SETTINGS["seqlen"], use_3d=SETTINGS["use_3d"],
                        use_model=SETTINGS["coordinate"].split("_")[-1])
    model.load_pretrained_with_remap(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.use_metric_scale = SETTINGS["metric_scale"]
    return model.requires_grad_(False).eval()


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


def frame_size(width, height):
    scale = min(SETTINGS["image_size"] / height, SETTINGS["image_size"] / width)
    return int(width * scale) // 64 * 64, int(height * scale) // 64 * 64


def bilinear_sample(field, xy, product=False):
    """Sample a (T,H,W,C) tensor or array at fixed first-frame pixel centers.

    Gather only the four neighbors on the field's device. For visibility,
    form save_ff's two-channel product at each pixel before interpolation.
    """
    _, height, width, _ = field.shape
    x, y = np.clip(xy, [0, 0], [width - 1, height - 1]).T
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = x - x0, y - y0
    weights = np.stack([(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy], axis=1)
    corners = field[:, np.stack([y0, y0, y1, y1], axis=1), np.stack([x0, x1, x0, x1], axis=1)]
    if product:
        corners = corners[..., :1] * corners[..., 1:2]
    if not isinstance(corners, np.ndarray):
        corners = corners.detach().float().cpu().numpy()
    return (corners * weights.astype(np.float32)[None, :, :, None]).sum(axis=2, dtype=np.float32)


def relative_poses(poses):
    """Convert DA3 camera-to-world estimates to OpenCV camera 0 coordinates."""
    poses = np.asarray(poses, dtype=np.float64)
    if not np.isfinite(poses).all():
        raise ValueError("Nonfinite camera pose")
    return (np.linalg.inv(poses[0]) @ poses).astype(np.float32)


def pose_motion(c2w):
    """Largest translation in meters and rotation in degrees from the reference camera."""
    distance = np.linalg.norm(c2w[:, :3, 3], axis=1)
    rot = c2w[:, :3, :3].astype(np.float64)
    skew = rot - rot.transpose(0, 2, 1)
    sine = np.linalg.norm(skew[:, [2, 0, 1], [1, 2, 0]], axis=1) / 2
    cosine = (np.trace(rot, axis1=1, axis2=2) - 1) / 2
    angle = np.rad2deg(np.arctan2(sine, cosine))
    return dict(max_translation_m=float(distance.max()), max_rotation_deg=float(angle.max()))


def full_run_poses(poses_by_query, log):
    """Express the full-run pose batches in camera 0 using an observed boundary motion."""
    full = poses_by_query[0].copy()
    if len(full) > DA3_BATCH:
        # With the shared 200-frame cap there is only one boundary. The 1/3
        # suffix contains both neighboring images in its first DA3 batch.
        t0 = min(t for t in poses_by_query if t > 0)
        i = DA3_BATCH - t0
        overlap = poses_by_query[t0]
        step = np.linalg.solve(overlap[i - 1].astype(np.float64), overlap[i])
        boundary_pose = full[DA3_BATCH - 1] @ step
        alignment = boundary_pose @ np.linalg.inv(full[DA3_BATCH].astype(np.float64))
        full[DA3_BATCH:] = alignment @ full[DA3_BATCH:]
        log["pose_batch_alignment"] = dict(
            boundary_frame=DA3_BATCH, overlap_query_frame=t0,
            transform_applied_to_second_batch=alignment.tolist())
    return full


def to_world(points, c2w):
    """Apply each target camera's pose to (T,N,3) OpenCV camera-space samples."""
    return points @ c2w[:, :3, :3].transpose(0, 2, 1) + c2w[:, None, :3, 3]


def pixel_intrinsics(normalized, size):
    """utils3d places pixel centers at ((x+0.5)/W, (y+0.5)/H) for depth_to_points."""
    w, h = size
    uv_to_pixels = np.array([[w, 0, -0.5], [0, h, -0.5], [0, 0, 1]], dtype=np.float32)
    return uv_to_pixels @ normalized


def dense_geometry(geometry, frames):
    from track3d_data import nearest_resize, scale_intrinsics
    import cv2

    T, H, W, _ = frames.shape
    scale = min(1., 336 / W, 336 / H)
    w, h = int(W * scale), int(H * scale)
    normalized = geometry["intrinsics"][0].float().cpu().numpy()
    intrinsics = pixel_intrinsics(normalized, (W, H))
    if intrinsics.shape != (T, 3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("Invalid per-frame intrinsics")
    depths = nearest_resize(geometry["points"][0, ..., 2].float().cpu().numpy(), w, h)
    mask = nearest_resize(geometry["mask"][0].cpu().numpy(), w, h)
    depths = np.where(mask & np.isfinite(depths) & (depths > 0), depths, 0)
    video = np.empty((T, 3, h, w), dtype=np.float16)
    for t, frame in enumerate(frames):
        rgb = cv2.resize(frame.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        video[t] = rgb.transpose(2, 0, 1) / 255
    return intrinsics, dict(depths=depths, video=video,
                            intrinsics=scale_intrinsics(intrinsics, (W, H), (w, h)))


def infer_suffix(model, frames, xy, keep_geometry):
    import torch

    T, H, W, _ = frames.shape
    rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].to(device="cuda", dtype=torch.float32)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            (geometry, motion), cache = model.infer(
                rgbs, iters=SETTINGS["inference_iters"], sw=None, is_training=False, tracking3d=True,
                force_projection=SETTINGS["force_projection"], use_da3_focal=SETTINGS["use_da3_focal"])
    del cache, rgbs
    xyz = motion["flow_3d"][0]
    uv = motion["flow_2d"][0].permute(0, 2, 3, 1)
    visconf = motion["visconf_maps_e"][0].permute(0, 2, 3, 1)
    if xyz.shape != (T, H, W, 3) or uv.shape != (T, H, W, 2) or visconf.shape != uv.shape:
        raise ValueError(f"Unexpected 3d_ff maps for {T} frames: {xyz.shape}, {uv.shape}, {visconf.shape}")
    poses = geometry["camera_poses"].float().cpu().numpy()
    if poses.shape != (T, 4, 4):
        raise ValueError(f"Unexpected pose shape {poses.shape}")
    if geometry.get("metric_scale") is None:
        raise ValueError("DA3 did not return the scale needed for metric output")
    return (bilinear_sample(uv, xy), bilinear_sample(xyz, xy),
            bilinear_sample(visconf, xy, product=True)[..., 0], relative_poses(poses),
            dense_geometry(geometry, frames) if keep_geometry else None)


def run_one(frames, model, log):
    T, H, W, _ = frames.shape
    queries = make_queries(T, (W, H))
    times = queries[:, 0].astype(int)
    N = len(queries)
    uv = np.empty((T, N, 2), np.float32)
    cam = np.empty((T, N, 3), np.float32)
    conf = np.zeros((T, N), np.float32)
    poses_by_query = {}
    log.update(queries=N, size=[W, H], suffixes=[])
    for t0 in np.unique(times).tolist():
        selected = times == t0
        entry = dict(query_frame=t0, frames=T - t0, queries=int(selected.sum()))
        log["suffixes"].append(entry)
        print(f"[3d_ff] Query frame {t0}, processing all {T - t0} frames")
        start = time.monotonic()
        xy, xyz, score, poses, dense = infer_suffix(model, frames[t0:], queries[selected, 1:], t0 == 0)
        entry.update(pose_motion(poses), seconds=round(time.monotonic() - start, 2))
        poses_by_query[t0] = poses
        if t0 == 0:
            K, geometry = dense
        uv[t0:, selected] = xy
        cam[t0:, selected] = xyz
        conf[t0:, selected] = score
        uv[:t0, selected] = xy[0]
        cam[:t0, selected] = xyz[0]

    c2w = full_run_poses(poses_by_query, log)
    log["full_run_pose_motion"] = pose_motion(c2w)
    world = to_world(cam, c2w)
    for t0 in poses_by_query:
        selected = times == t0
        world[:t0, selected] = world[t0, selected]

    if not all(np.isfinite(a).all() for a in (uv, cam, world, conf)):
        raise ValueError("Nonfinite sparse trajectory or visibility score")
    vis = (conf > SETTINGS["visibility_threshold"]).astype(np.float32)
    tracks = dict(queries=queries, tracks2d=uv, tracks3d_cam=cam, tracks3d_world=world,
                  track_depth=cam[..., 2], vis=vis, conf=conf, c2w=c2w, intrinsics=K)
    geometry.update(coords=world, visibs=vis[..., None], extrinsics=np.linalg.inv(c2w))
    return tracks, geometry


def write_overlay(frames, tracks, path, fps, trail=8):
    import cv2

    T, H, W, _ = frames.shape
    qframe, qx = tracks["queries"][:, :2].T
    hue = (qx - qx.min()) / (np.ptp(qx) + 1e-6) * 170
    hsv = np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], 1).astype(np.uint8)[None]
    colors = [tuple(int(v) for v in c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]]
    tmp = path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise OSError(f"Could not open video writer for {tmp}")
    try:
        for t in range(T):
            img = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
            for n in np.flatnonzero((qframe <= t) & (tracks["vis"][t] > 0.5)):
                start = max(int(qframe[n]), t - trail)
                pts = tracks["tracks2d"][start:t + 1, n].round().astype(np.int32)
                if len(pts) > 1:
                    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, colors[n], 1, cv2.LINE_AA)
                cv2.circle(img, tuple(map(int, pts[-1])), 2, colors[n], -1, cv2.LINE_AA)
            writer.write(img)
    finally:
        writer.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(path)], check=True)
    tmp.unlink()


def save_outputs(out_dir, data, frames, tracks, geometry, mode, overlay):
    size = (frames.shape[2], frames.shape[1])
    orig = data["orig_size"]
    tracks.update(tracks2d_orig=to_size(tracks["tracks2d"], size, orig).astype(np.float32),
                  frame_idx=data["frame_idx"], timestamps=data["timestamps"], fps=np.float32(data["fps"]),
                  size=np.array(size), orig_size=np.array(orig), mode=mode, settings=json.dumps(SETTINGS))
    with tempfile.TemporaryDirectory(dir=out_dir) as stage:
        stage = Path(stage)
        np.savez_compressed(stage / "tracks.npz", **tracks)
        np.savez_compressed(stage / "result.npz", **geometry)
        if overlay:
            write_overlay(frames, tracks, stage / "overlay.mp4", data["fps"])
        for path in stage.iterdir():
            path.replace(out_dir / path.name)


def write_json(path, data):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(f.name, path)


def print_summary(rows):
    print(f"\n{'clip':11s} {'mode':6s} {'status':7s} {'frames':>6s} {'fps':>6s} "
          f"{'infer_s':>8s} {'peak_gb':>8s}  error")
    for r in rows:
        print(f"{r['clip']:11s} {r['mode']:6s} {r['status']:7s} {r.get('frames', ''):>6} "
              f"{r.get('fps', ''):>6} {r.get('infer_s', ''):>8} {r.get('peak_gb', ''):>8}  "
              f"{r.get('error', '')[:100]}")


def main():
    args = parse_args()
    import cv2
    import torch

    sys.path.insert(0, str(args.common_dir.resolve()))
    from track3d_data import load_clip

    rows, model = [], None
    jobs = [(clip, mode) for clip in args.clips for mode in args.modes]
    for i, (clip, mode) in enumerate(jobs, 1):
        out_dir = args.out_dir / clip / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "log.json"
        tag = f"[{i}/{len(jobs)}] {clip}/{mode}"
        files = ["tracks.npz", "result.npz"] + ([] if args.no_overlay else ["overlay.mp4"])
        old = json.loads(log_path.read_text()) if log_path.exists() else {}
        if old.get("status") == "ok" and all((out_dir / name).is_file() for name in files):
            print(f"{tag}: already done, skipped")
            rows.append(old)
            write_json(args.out_dir / "summary.json", rows)
            continue

        log = dict(clip=clip, mode=mode, settings=SETTINGS, checkpoint=str(args.ckpt),
                   scale="Monocular DA3 metric estimate in meters",
                   intrinsics_used="Full-run infer intrinsics, DA3 mean focal, centered principal point",
                   coordinates="flow_3d[t] is in target OpenCV camera t. All groups use full-run c2w relative to frame 0. "
                               "At the DA3 batch boundary the 1/3 suffix supplies the frame 127 to 128 relative pose.",
                   visibility="Bilinear sample of the visconf channel product. Visible above 0.5. Zero before query.",
                   da3_chunking="get_fmaps concatenates independent DA3 batches of up to 128 frames without pose registration. "
                                "infer restores the last batch's scale and focal across the suffix.")
        start = time.monotonic()
        try:
            log["gpu"] = torch.cuda.get_device_name(0)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if model is None:
                model = build_model(args.repo_dir, args.ckpt).cuda()
            data = load_clip(args.data_dir / clip, fps=SETTINGS["target_fps"], max_frames=SETTINGS["max_frames"],
                             size=None, depth_size="native")
            size = frame_size(*data["size"])
            video = data.pop("video")
            frames = np.stack([cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR) for rgb in video])
            del video
            log.update(frames=len(frames), fps=round(float(data["fps"]), 3))
            infer_start = time.monotonic()
            tracks, geometry = run_one(frames, model, log)
            log["infer_s"] = round(time.monotonic() - infer_start, 2)
            save_outputs(out_dir, data, frames, tracks, geometry, mode, not args.no_overlay)
            del data, frames, tracks, geometry
            log["status"] = "ok"
        except Exception as e:
            log.update(status="error", error=repr(e), traceback=traceback.format_exc())
        log["total_s"] = round(time.monotonic() - start, 2)
        if torch.cuda.is_available():
            log["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(log_path, log)
        rows.append(log)
        write_json(args.out_dir / "summary.json", rows)
        print(f"{tag}: {log['status']} in {log['total_s']} s {log.get('error', '')}")

    print_summary(rows)
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    sys.exit(main())
