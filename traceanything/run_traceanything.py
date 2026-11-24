"""Run the video-only TraceAnything comparison on the shared track3d clips.

load_clip selects up to 200 frames at a target 10 fps at 960x720. As in
scripts/infer.py::_load_images, PIL resizes the long side to 512, crops at the
top left to multiples of 16 (512x384 here), then ToTensor and Normalize map
selected RGB views to [-1, 1]. Use at most 40 evenly spaced views including the
query frames and both endpoints. time_step is k/(T-1) on the full clip.

The official _build_model_from_cfg loads configs/eval.yaml and the checkpoint.
Its flash_attention setting selects PyTorch Flash SDPA with bfloat16 attention.
Call model.forward(views) at the official precision. Its global head returns
ctrl_pts3d and ctrl_conf for ten controls (poly_degree=10). Like scripts/view.py, use
evaluate_bspline_conf with its cubic, clamped knots. Sample controls bilinearly
at each query pixel, then evaluate at every known frame time (0 for a singleton).
For dense geometry evaluate each selected view at its own input time. The model
also predicts timestamps, which the official forward uses for its dense track_pts3d.

The official viewer combines the global heads from all views in one scene.
We interpret this space as the first camera's OpenCV frame, fixed on the tripod.
Fit fx, fy, cx, cy to confident pixels of that first map. Projection uses this
one matrix throughout. Tracks and depths retain the model's nonmetric units.
Visibility starts at the query and carries no occlusion information.

results/<clip>/rgb/ contains tracks.npz, result.npz, overlay.mp4 and log.json.
summary.json includes finished runs skipped when resuming.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLIPS = ["pour_lid", "pen", "stir", "shoelace", "paper_fold", "shirt_fold"]
SETTINGS = dict(
    traceanything_commit="54677b5e7bf11510c2e8c917a509988ad379f8eb",
    hf_revision="51f3ea4398de240d5fe1109b6f8f8f3c7fe506a9",
    model_config="configs/eval.yaml", target_fps=10, max_frames=200,
    input_size=(960, 720), long_side=512, patch_size=16, max_views=40,
    attention="PyTorch SDPA FLASH_ATTENTION", metric=False,
    coordinate_frame="Global ctrl_pts3d interpreted as first-camera OpenCV axes, fixed tripod",
    intrinsics="First view at t=0, pinhole least squares on upper half by confidence",
    depth="Global spline Z at each selected view's input time, in model units, with zero for invalid depth",
    geometry="Nearest selected view by clip frame index (earlier on ties), RGB from the actual frame",
    visibility="One from query frame onward, zero before query, without occlusion prediction",
    confidence="Raw B-spline confidence evaluated at every frame, on the model's unbounded scale",
    time_step="Clip index divided by T-1, including both endpoints, or zero for a singleton",
    before_query="3D position, depth and projection held at query-frame values",
)
REF_SIZE = (518, 392)
REF_GRID, REF_EXTRA_FRACS, REF_EXTRA_GRID, REF_MARGIN = (32, 24), (1 / 3, 2 / 3), (16, 12), 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    work = Path(os.environ.get("TRACEANYTHING_WORK", HERE / "work"))
    p.add_argument("--clips", nargs="+", default=CLIPS)
    p.add_argument("--modes", nargs="+", choices=["rgb"], default=["rgb"])
    p.add_argument("--data-dir", type=Path, default=ROOT / "preprocessed_data")
    p.add_argument("--common-dir", type=Path, default=ROOT / "common")
    p.add_argument("--out-dir", type=Path, default=HERE / "results")
    p.add_argument("--repo-dir", type=Path, default=work / "TraceAnything")
    p.add_argument("--ckpt", type=Path, help="Default: <repo-dir>/checkpoints/trace_anything.pt")
    p.add_argument("--no-overlay", action="store_true", help="Skip overlay.mp4")
    args = p.parse_args()
    args.ckpt = (args.ckpt or args.repo_dir / "checkpoints/trace_anything.pt").resolve()
    return args


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


def select_views(T, query_frames, limit):
    """Start on a uniform grid and snap the closest free slots to required frames."""
    anchors = np.unique([0, T - 1, *query_frames]).astype(int)
    if T < 1 or anchors[0] < 0 or anchors[-1] >= T or len(anchors) > limit:
        raise ValueError("Required frames do not fit the view selection")
    selected = np.rint(np.linspace(0, T - 1, min(T, limit))).astype(int)
    for k in np.setdiff1d(anchors, selected):
        free = np.flatnonzero(~np.isin(selected, anchors))
        selected[free[np.argmin(np.abs(selected[free] - k))]] = k
    return np.sort(selected)


def bilinear_controls(field, xy):
    """Sample a [controls,H,W,channels] field at N pixel centers, giving [controls,N,channels]."""
    _, h, w, _ = field.shape
    x, y = np.clip(xy, [0, 0], [w - 1, h - 1]).T
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = (x - x0).astype(np.float32)[:, None], (y - y0).astype(np.float32)[:, None]
    top = (1 - dx) * field[:, y0, x0] + dx * field[:, y0, x1]
    bottom = (1 - dx) * field[:, y1, x0] + dx * field[:, y1, x1]
    return (1 - dy) * top + dy * bottom


def fit_intrinsics(points, confidence):
    """Fit image pixels to normalized camera rays on the confident half of valid pixels."""
    valid = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0) & np.isfinite(confidence)
    if not np.any(valid):
        raise ValueError("First view has no finite positive-depth calibration pixels")
    threshold = np.median(confidence[valid])
    y, x = np.nonzero(valid & (confidence >= threshold))
    xyz = points[y, x].astype(np.float64)
    rays = xyz[:, :2] / xyz[:, 2:]
    K = np.eye(3, dtype=np.float32)
    squared_error = np.zeros(len(x))
    for axis, pixels in enumerate((x, y)):
        design = np.column_stack((rays[:, axis], np.ones(len(x))))
        fit, _, rank, _ = np.linalg.lstsq(design, pixels, rcond=None)
        if rank < 2 or not np.isfinite(fit).all() or fit[0] <= 0:
            raise ValueError("First view does not support a positive, finite pinhole fit")
        K[axis, axis], K[axis, 2] = fit
        squared_error += (design @ fit - pixels) ** 2
    return K, dict(pixels=len(x), confidence_threshold=float(threshold),
                   reprojection_rmse_px=float(np.sqrt(squared_error.mean())))


def project_tracks(xyz, conf, queries, K):
    """Hold positions before each query, then project with the fixed pinhole camera."""
    qframe = queries[:, 0].astype(int)
    active = np.arange(len(xyz))[:, None] >= qframe
    xyz = np.where(active[..., None], xyz, xyz[qframe, np.arange(len(queries))])
    if not (np.isfinite(xyz).all() and np.isfinite(conf).all()) or np.any(xyz[..., 2] == 0):
        raise ValueError("Nonfinite tracks or confidence, or a track on the camera plane")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        uv = xyz[..., :2] / xyz[..., 2:] * K.diagonal()[:2] + K[:2, 2]
    return xyz, uv, active.astype(np.float32)


def preprocess_frames(video):
    from PIL import Image
    from scripts.infer import _resize_long_side

    frames = []
    for frame in video:
        img = Image.fromarray(frame)
        if img.height > img.width:
            img = img.transpose(Image.Transpose.ROTATE_90)
        img = _resize_long_side(img, SETTINGS["long_side"])
        w, h = img.size
        patch = SETTINGS["patch_size"]
        frames.append(np.asarray(img.crop((0, 0, w - w % patch, h - h % patch))))
    return np.stack(frames)


def run_one(model, frames):
    import torch
    from torchvision import transforms
    from trace_anything.trace_anything import evaluate_bspline_conf

    T, H, W, _ = frames.shape
    queries = make_queries(T, (W, H))
    qframe = queries[:, 0].astype(int)
    selected = select_views(T, np.unique(qframe), SETTINGS["max_views"])
    times = torch.from_numpy(np.arange(T, dtype=np.float32) / max(1, T - 1))
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    views = [dict(img=normalize(frames[k]).unsqueeze(0).cuda(), time_step=float(times[k])) for k in selected]

    xyz = np.empty((T, len(queries), 3), np.float32)
    conf = np.empty((T, len(queries)), np.float32)
    depths = np.empty((len(selected), H, W), np.float32)
    with torch.no_grad():
        # Release the dense all-pairs predictions and local head before evaluating our timeline.
        controls = [(p["ctrl_pts3d"], p["ctrl_conf"]) for p in model.forward(views)]
        for i, k in enumerate(selected):
            pts, scores = (tensor.float().cpu() for tensor in controls[i])
            point_map, score_map = evaluate_bspline_conf(pts, scores, times[k:k + 1])
            point_map = point_map[0].numpy()
            if i == 0:
                K, fit = fit_intrinsics(point_map, score_map[0].numpy())
            z = point_map[..., 2]
            depths[i] = np.where(np.isfinite(point_map).all(axis=-1) & (z > 0), z, 0)

            mask = qframe == k
            if mask.any():
                xy = queries[mask, 1:]
                sampled = bilinear_controls(pts.numpy(), xy)
                sampled_conf = bilinear_controls(scores.numpy()[..., None], xy)[..., 0]
                # The official evaluator accepts a spatial grid. Use a 1 x N grid of query pixels.
                path, score = evaluate_bspline_conf(torch.from_numpy(sampled[:, None]),
                                                    torch.from_numpy(sampled_conf[:, None]), times)
                xyz[:, mask] = path[:, 0].numpy()
                conf[:, mask] = score[:, 0].numpy()
    xyz, uv, vis = project_tracks(xyz, conf, queries, K)
    return dict(queries=queries, xyz=xyz, uv=uv, vis=vis, conf=conf, selected=selected,
                depths=depths, K=K, fit=fit)


def write_overlay(frames, tracks2d, vis, queries, path, fps, trail=8):
    import cv2

    _, h, w, _ = frames.shape
    x = queries[:, 1]
    hue = (x - x.min()) / (np.ptp(x) + 1e-6) * 170
    hsv = np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], axis=1).astype(np.uint8)
    colors = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0].tolist()
    qframe = queries[:, 0].astype(int)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "23", str(path)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for t, frame in enumerate(frames):
            img = frame.copy()
            for n in np.flatnonzero(vis[t] > 0.5):
                start = max(qframe[n], t - trail)
                line = np.rint(tracks2d[start:t + 1, n]).clip(-2**30, 2**30).astype(np.int32)
                if len(line) > 1:
                    cv2.polylines(img, [line], False, colors[n], 1, cv2.LINE_AA)
                cv2.circle(img, tuple(map(int, line[-1])), 2, colors[n], -1, cv2.LINE_AA)
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")


def save_outputs(d, frames, r, out_dir, mode, overlay):
    from PIL import Image
    from track3d_data import nearest_resize, scale_intrinsics

    T, H, W, _ = frames.shape
    size = (W, H)
    eye = np.repeat(np.eye(4, dtype=np.float32)[None], T, axis=0)
    with np.errstate(over="raise", invalid="raise"):
        uv_orig = to_size(r["uv"], size, d["orig_size"]).astype(np.float32)
    np.savez_compressed(
        out_dir / "tracks.npz", queries=r["queries"], tracks2d=r["uv"], tracks2d_orig=uv_orig,
        track_depth=r["xyz"][..., 2], tracks3d_cam=r["xyz"], tracks3d_world=r["xyz"],
        vis=r["vis"], conf=r["conf"], c2w=eye, intrinsics=np.repeat(r["K"][None], T, axis=0),
        frame_idx=d["frame_idx"], timestamps=d["timestamps"], fps=np.float32(d["fps"]),
        size=size, orig_size=d["orig_size"], mode=mode, settings=json.dumps(SETTINGS))

    scale = min(1, 336 / max(W, H))
    w, h = int(W * scale), int(H * scale)
    nearest = np.abs(np.arange(T)[:, None] - r["selected"]).argmin(axis=1)
    depths = nearest_resize(r["depths"], w, h)[nearest]
    K = scale_intrinsics(r["K"], size, (w, h))
    colors = np.empty((T, 3, h, w), np.float16)
    for t, frame in enumerate(frames):
        rgb = Image.fromarray(frame).resize((w, h), Image.Resampling.BILINEAR)
        colors[t] = np.asarray(rgb).transpose(2, 0, 1).astype(np.float32) / 255
    np.savez_compressed(out_dir / "result.npz", depths=depths, video=colors, extrinsics=eye,
                        intrinsics=np.repeat(K[None], T, axis=0), coords=r["xyz"], visibs=r["vis"][..., None])
    if overlay:
        write_overlay(frames, r["uv"], r["vis"], r["queries"], out_dir / "overlay.mp4", d["fps"])


def write_json(path, value):
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(f.name, path)


def print_summary(rows):
    print(f"\n{'clip':11s} {'mode':5s} {'status':7s} {'frames':>6s} {'views':>5s} {'fps':>5s} "
          f"{'infer_s':>8s} {'peak_gb':>7s}  note")
    for r in rows:
        print(f"{r['clip']:11s} {r['mode']:5s} {r['status']:7s} {r.get('frames', ''):>6} "
              f"{r.get('views', ''):>5} {r.get('fps', ''):>5} {r.get('infer_s', ''):>8} "
              f"{r.get('peak_gb', ''):>7}  {r.get('error', '')[:80]}")


def main():
    args = parse_args()
    sys.path[:0] = [str(args.common_dir.resolve()), str(args.repo_dir.resolve())]
    jobs = [(clip, mode) for clip in args.clips for mode in args.modes]
    model, rows = None, []
    for i, (clip, mode) in enumerate(jobs, 1):
        tag = f"[{i}/{len(jobs)}] {clip}/{mode}"
        out_dir = args.out_dir / clip / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "log.json"
        files = ["tracks.npz", "result.npz"] + ([] if args.no_overlay else ["overlay.mp4"])
        if log_path.is_file() and all((out_dir / name).is_file() for name in files):
            with log_path.open() as f:
                old = json.load(f)
            if old.get("status") == "ok":
                print(f"{tag}: already done, skipped")
                rows.append(old)
                write_json(args.out_dir / "summary.json", rows)
                continue

        log = dict(clip=clip, mode=mode, metric=False, settings=SETTINGS, checkpoint=str(args.ckpt),
                   overlay=not args.no_overlay)
        started = time.perf_counter()
        try:
            import torch
            from scripts.infer import _build_model_from_cfg, _load_cfg
            from track3d_data import load_clip

            log["gpu"] = torch.cuda.get_device_name(0)
            if model is None:
                cfg = _load_cfg(str(args.repo_dir / SETTINGS["model_config"]))
                model = _build_model_from_cfg(cfg, str(args.ckpt), torch.device("cuda"))
            d = load_clip(args.data_dir / clip, fps=SETTINGS["target_fps"], max_frames=SETTINGS["max_frames"],
                          size=SETTINGS["input_size"], depth_size="native")
            frames = preprocess_frames(d.pop("video"))
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            infer_start = time.perf_counter()
            r = run_one(model, frames)
            torch.cuda.synchronize()
            log.update(frames=len(frames), fps=round(d["fps"], 2), queries=len(r["queries"]),
                       views=len(r["selected"]), selected_clip_indices=r["selected"].tolist(),
                       selected_frame_idx=d["frame_idx"][r["selected"]].tolist(), size=[frames.shape[2], frames.shape[1]],
                       intrinsics_used=r["K"].tolist(), intrinsics_fit=r["fit"],
                       infer_s=round(time.perf_counter() - infer_start, 2),
                       peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            with TemporaryDirectory(dir=out_dir) as staging:
                stage = Path(staging)
                save_outputs(d, frames, r, stage, mode, not args.no_overlay)
                for name in files:
                    os.replace(stage / name, out_dir / name)
            log["status"] = "ok"
            del d, frames, r
        except Exception as exc:
            log.update(status="error", error=repr(exc), traceback=traceback.format_exc())
        log["total_s"] = round(time.perf_counter() - started, 2)
        write_json(log_path, log)
        rows.append(log)
        write_json(args.out_dir / "summary.json", rows)
        print(f"{tag}: {log['status']} ({log['total_s']} s) {log.get('error', '')}")

    print_summary(rows)
    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"\nDone: {n_ok} of {len(jobs)} runs ok. Results: {args.out_dir}")
    return int(n_ok != len(jobs))


if __name__ == "__main__":
    sys.exit(main())
