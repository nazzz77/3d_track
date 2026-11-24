"""Run TrackCraft3R on the shared iPhone queries and write track3d viewer results.

Both modes use load_clip at 10 fps, at most 200 frames, in 512x384 pixels.
rgb runs scripts/preprocess_da3.py with DA3NESTED-GIANT-LARGE and the default
504-pixel upper_bound_resize. rgbd_fixed uses iPhone LiDAR, iPhone intrinsics
and identity poses. scripts/build_user_npz.py packs both with z depth and w2c
extrinsics. It normalizes poses to camera 0 and stores frame 0's intrinsics.
The predictor uses that four-vector. Projection and depth export use the
original per-frame calibration from the corresponding depth source.

MODEL_SETTINGS are WanSceneFlowPredictor arguments from inference_user_video.py
and eval_interleaved.sh: 832x480 stretch, rank-1024 LoRA on attention and FFN,
last scheduler timestep (-1), 12 track latents, 80 m depth cap and 2/98 depth
percentiles. The released predictor enables diagonal conditioning, pixel
deltas, inlier normalization, the visibility decoder and the pointmap encoder.

Each distinct query frame anchors its own suffix. As in eval_worldtrack.py,
the stride is ceil((suffix_length - 1) / 11), at least one. All offsets share
the anchor. Short passes repeat their last frame up to 12 inputs and discard
padded outputs. The predictor samples its dense anchor-camera track map at
integer pixels after stretching the query coordinates. We sample the sigmoid
visibility map at those same pixels and average repeated real-frame votes.

Results go to results/<clip>/<mode>/{tracks.npz,result.npz,overlay.mp4,log.json}.
summary.json records completed, skipped and failed runs as the job progresses.
"""
import argparse
import gc
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
FRAME_SIZE = (512, 384)
TARGET_FPS, MAX_FRAMES = 10, 200
TRACK_COMMIT = "21e8fcaf4b6375b3044cead210d5808e1d81760b"
CKPT_REV = "e397c1050bd540062cbd6642ba06a344ca80f116"
DA3_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE"
MODEL_SETTINGS = dict(
    model_id="Wan-AI/Wan2.1-T2V-1.3B", height=480, width=832, resize_mode="stretch",
    lora_rank=1024, lora_target_modules="q,k,v,o,ffn.0,ffn.2", regression_timestep=-1,
    track_latent_length=12, diag_max_depth=80.0, pj_norm_percentile_lo=2.0,
    pj_norm_percentile_hi=98.0, diagonal_condition_row=True, pixel_delta=True,
    pj_norm_inlier=True, predict_vis=True, vis_separate_decoder=True, pj_separate_encoder=True,
)
PASS_FRAMES = MODEL_SETTINGS["track_latent_length"]
MODES = {
    "rgb": dict(depth=DA3_MODEL, camera="DA3 per-frame poses and intrinsics",
                scale="DA3 estimated metric scale (meters)", da3_commit=DA3_COMMIT,
                da3_process_res=504),
    "rgbd_fixed": dict(depth="iPhone LiDAR z-depth", camera="iPhone intrinsics, identity poses",
                       scale="LiDAR meters"),
}
REF_SIZE = (518, 392)
REF_GRID, REF_EXTRA_FRACS, REF_EXTRA_GRID, REF_MARGIN = (32, 24), (1 / 3, 2 / 3), (16, 12), 8


def parse_args():
    work = Path(os.environ.get("TRACKCRAFT3R_WORK", HERE / "work")).resolve()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", nargs="+", default=CLIPS)
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    p.add_argument("--data-dir", type=Path, default=ROOT / "preprocessed_data")
    p.add_argument("--common-dir", type=Path, default=ROOT / "common")
    p.add_argument("--out-dir", type=Path, default=HERE / "results")
    p.add_argument("--repo-dir", type=Path, default=work / "TrackCraft3r")
    p.add_argument("--ckpt", type=Path, default=work / "checkpoint/model.safetensors")
    p.add_argument("--da3-python", type=Path, default=work / "da3_env/bin/python")
    p.add_argument("--no-overlay", action="store_true")
    args = p.parse_args()
    args.work = work
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.absolute())
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


def interleaved_runs(anchor, stop):
    """Yield 12 absolute frame indices and the number of real (unpadded) entries."""
    if not 0 <= anchor < stop:
        raise ValueError(f"Empty suffix at frame {anchor} of {stop}")
    stride = max(1, (stop - anchor - 1 + PASS_FRAMES - 2) // (PASS_FRAMES - 1))
    for offset in range(stride):
        # Offset zero already contains the anchor in the official range(0, T, stride).
        first = anchor + (offset or stride)
        real = np.r_[anchor, np.arange(first, stop, stride)]
        yield np.pad(real, (0, PASS_FRAMES - len(real)), mode="edge"), len(real)


def prepare_input(args, mode, clip, tmp):
    """Pack the exact selected frames with the upstream builder in both modes."""
    import cv2
    from PIL import Image

    frames = tmp / "frames"
    frames.mkdir()
    for t, rgb in enumerate(clip["video"]):
        Image.fromarray(rgb).save(frames / f"{t:06d}.png", compress_level=1)

    scripts = args.repo_dir / "scripts"
    if mode == "rgb":
        subprocess.run([args.da3_python, scripts / "preprocess_da3.py", "--frame_dir", frames,
                        "--output_dir", tmp, "--model_name", DA3_MODEL,
                        "--process_res", str(MODES[mode]["da3_process_res"])], check=True)
    else:
        np.save(tmp / "depth.npy", clip["depth"])
        np.save(tmp / "intrinsics.npy", clip["intrinsics"])
        identity = np.repeat(np.eye(4, dtype=np.float32)[None], len(clip["video"]), axis=0)
        np.save(tmp / "extrinsics.npy", identity)

    path = tmp / "input.npz"
    subprocess.run([sys.executable, scripts / "build_user_npz.py", "--video_path", frames,
                    "--depth_npy", tmp / "depth.npy", "--intrinsics_npy", tmp / "intrinsics.npy",
                    "--extrinsics_npy", tmp / "extrinsics.npy", "--depth_convention", "z",
                    "--extrinsics_convention", "w2c", "--output_npz", path], check=True)
    # Only the builder's local JPEG byte array needs pickle. Result archives use plain arrays.
    with np.load(path, allow_pickle=True) as packed:
        images = []
        for jpeg in packed["images_jpeg_bytes"]:
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        depth = packed["depth_map"]
        w2c = packed["extrinsics_w2c"]
        model_k = packed["fx_fy_cx_cy"]
    intrinsics = np.load(tmp / "intrinsics.npy")
    T, H, W = clip["video"].shape[:3]
    if (len(images) != T or depth.shape != (T, H, W) or w2c.shape != (T, 4, 4)
            or intrinsics.shape != (T, 3, 3) or model_k.shape != (4,)):
        raise ValueError("Packed camera, depth or image dimensions disagree with load_clip")
    if (not all(np.isfinite(a).all() for a in (w2c, intrinsics, model_k))
            or np.any(intrinsics[:, (0, 1), (0, 1)] <= 0)):
        raise ValueError("Invalid input camera calibration")
    depth[~np.isfinite(depth) | (depth <= 0)] = 0
    return images, depth, w2c, model_k, intrinsics


def predict_anchor(predictor, images, depth, w2c, model_k, anchor, xy):
    """Average sparse samples in anchor camera space, following eval_worldtrack.py."""
    length, n = len(images) - anchor, len(xy)
    xyz_sum = np.zeros((length, n, 3), dtype=np.float64)
    vis_sum = np.zeros((length, n), dtype=np.float64)
    count = np.zeros(length, dtype=int)
    vis_count = np.zeros(length, dtype=int)
    dummy_vis = np.ones((PASS_FRAMES, n), dtype=bool)  # unused input in predict()
    for ids, real_len in interleaved_runs(anchor, len(images)):
        # predict() itself stretches xy to 832x480 and indexes _last_row_dense at (int(v), int(u)).
        xyz = predictor.predict([images[i] for i in ids], xy, dummy_vis, model_k,
                                depth_map=depth[ids], extrinsics_w2c=w2c[ids])
        if xyz.shape != (PASS_FRAMES, n, 3) or not np.all(predictor._last_oob_mask):
            raise ValueError("Predictor dropped in-bounds reference queries or frames")
        real = ids[:real_len] - anchor
        xyz_sum[real] += xyz[:real_len]
        count[real] += 1
        if predictor._last_vis_dense is not None:
            u, v = predictor._last_query_uv_model.astype(np.int64).T
            vis_sum[real] += predictor._last_vis_dense[:real_len, v, u]
            vis_count[real] += 1
    if np.any(count == 0):
        raise RuntimeError("Interleaved passes missed a frame")
    prob = np.divide(vis_sum, vis_count[:, None], out=np.ones_like(vis_sum), where=vis_count[:, None] > 0)
    fallback_samples = int(np.count_nonzero(vis_count == 0)) * n
    return xyz_sum / count[:, None, None], prob, int(count[0]), fallback_samples


def transform_points(points, pose):
    """Apply one or a batch of OpenCV affine transforms to rows of XYZ points."""
    return points @ pose[..., :3, :3].swapaxes(-1, -2) + pose[..., :3, 3][..., None, :]


def project_camera(points, intrinsics):
    """Pinhole projection in each frame. At exactly zero z use the principal point."""
    z = points[..., 2:3]
    rays = np.divide(points, z, out=np.zeros_like(points), where=z != 0)
    rays[..., 2] = 1
    return (rays @ intrinsics.swapaxes(-1, -2))[..., :2]


def track_clip(predictor, images, depth, w2c, model_k, intrinsics, size):
    T = len(images)
    c2w = np.linalg.inv(w2c)
    queries = make_queries(T, size)
    world = np.empty((T, len(queries), 3), dtype=np.float32)
    camera = np.empty_like(world)
    uv = np.empty((T, len(queries), 2), dtype=np.float32)
    conf = np.zeros((T, len(queries)), dtype=np.float32)
    passes = {}
    fallback_samples = 0
    for anchor in np.unique(queries[:, 0]).astype(int):
        group = queries[:, 0] == anchor
        xyz, prob, num_passes, missing = predict_anchor(predictor, images, depth, w2c, model_k,
                                                        anchor, queries[group, 1:])
        passes[int(anchor)] = num_passes
        fallback_samples += missing
        world_xyz = transform_points(xyz, c2w[anchor])
        camera_xyz = transform_points(world_xyz, w2c[anchor:])
        pixels = project_camera(camera_xyz, intrinsics[anchor:])
        for dest, values in ((world, world_xyz), (camera, camera_xyz), (uv, pixels)):
            dest[anchor:, group] = values
            dest[:anchor, group] = values[0]
        conf[anchor:, group] = prob
    if not all(np.isfinite(a).all() for a in (world, camera, uv, conf, c2w)):
        raise ValueError("Non-finite trajectory, projection, visibility or camera pose")
    if np.any((conf < 0) | (conf > 1)):
        raise ValueError("Visibility decoder returned probabilities outside [0, 1]")
    tracks = dict(queries=queries, tracks2d=uv, tracks3d_world=world, tracks3d_cam=camera,
                  track_depth=camera[..., 2], vis=(conf > 0.5).astype(np.float32),
                  conf=conf, c2w=c2w, intrinsics=intrinsics)
    return tracks, passes, fallback_samples


def write_overlay(frames, tracks, path, fps, trail=8):
    import cv2

    _, H, W, _ = frames.shape
    queries, uv, vis = tracks["queries"], tracks["tracks2d"], tracks["vis"]
    qx = queries[:, 1]
    hue = (qx - qx.min()) / (np.ptp(qx) + 1e-6) * 170
    hsv = np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], 1).astype(np.uint8)[None]
    colors = [tuple(map(int, c)) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]]
    # Keep OpenCV drawing coordinates representable even for tracks close to the camera plane.
    points = np.clip(np.rint(uv), -100000000, 100000000).astype(np.int32)
    start = queries[:, 0].astype(int)
    tmp = path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("Could not open overlay video writer")
    try:
        for t, frame in enumerate(frames):
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for n in np.flatnonzero((start <= t) & (vis[t] > 0.5)):
                pts = points[max(start[n], t - trail):t + 1, n]
                if len(pts) > 1:
                    # The slice is a strided view; OpenCV needs a contiguous int32 point array.
                    cv2.polylines(img, [np.ascontiguousarray(pts.reshape(-1, 1, 2))], False, colors[n], 1,
                                  cv2.LINE_AA)
                cv2.circle(img, tuple(map(int, pts[-1])), 2, colors[n], -1, cv2.LINE_AA)
            writer.write(img)
    finally:
        writer.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", path], check=True)
    tmp.unlink()


def save_outputs(path, clip, mode, settings, tracks, images, depth, w2c, overlay):
    from PIL import Image
    from track3d_data import nearest_resize, scale_intrinsics

    W, H = clip["size"]
    orig_uv = to_size(tracks["tracks2d"], (W, H), clip["orig_size"]).astype(np.float32)
    if not np.isfinite(orig_uv).all():
        raise ValueError("Non-finite projection in the original image")
    np.savez_compressed(path / "tracks.npz", **tracks, tracks2d_orig=orig_uv,
                        frame_idx=clip["frame_idx"], timestamps=clip["timestamps"],
                        fps=np.float32(clip["fps"]), size=clip["size"], orig_size=clip["orig_size"],
                        mode=mode, settings=json.dumps(settings))

    scale = min(1.0, 336 / W, 336 / H)
    small = (int(W * scale), int(H * scale))
    video = np.stack([np.asarray(img.resize(small, Image.Resampling.BILINEAR)) for img in images])
    video = (video.transpose(0, 3, 1, 2).astype(np.float32) / 255).astype(np.float16)
    np.savez_compressed(path / "result.npz", depths=nearest_resize(depth, *small),
                        video=video, extrinsics=w2c, coords=tracks["tracks3d_world"],
                        intrinsics=scale_intrinsics(tracks["intrinsics"], (W, H), small),
                        visibs=tracks["vis"][..., None])
    warnings = []
    if overlay:
        try:  # the overlay is only a preview; do not lose the tracks if drawing fails
            write_overlay(clip["video"], tracks, path / "overlay.mp4", clip["fps"])
        except Exception as e:
            warnings.append(f"overlay failed: {e!r}")
    return warnings


def write_json(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(f.name, path)


def print_summary(rows):
    print(f"\n{'clip':11s} {'mode':11s} {'status':7s} {'frames':>6s} {'fps':>5s} "
          f"{'infer_s':>8s} {'peak_gb':>7s}  error")
    for r in rows:
        print(f"{r['clip']:11s} {r['mode']:11s} {r['status']:7s} {str(r.get('frames', '')):>6s} "
              f"{str(r.get('fps', '')):>5s} {str(r.get('infer_s', '')):>8s} "
              f"{str(r.get('peak_gb', '')):>7s}  {r.get('error', '')[:80]}")


def main():
    args = parse_args()
    sys.path[:0] = [str(args.repo_dir), str(args.common_dir)]
    import torch
    from evaluation.wan_scene_flow_predictor import WanSceneFlowPredictor
    from track3d_data import load_clip

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)")
    jobs = [(name, mode) for name in args.clips for mode in args.modes]
    predictor, clip, loaded = None, None, None
    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, mode) in enumerate(jobs, 1):
        out = args.out_dir / name / mode
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / "log.json"
        tag = f"[{i}/{len(jobs)}] {name}/{mode}"
        outputs = ["result.npz", "tracks.npz"]
        if not args.no_overlay:
            outputs.append("overlay.mp4")
        if log_path.is_file() and all((out / f).is_file() for f in outputs):
            with log_path.open() as f:
                old = json.load(f)
            if old.get("status") == "ok":
                print(f"{tag}: already done, skipped")
                rows.append(old)
                write_json(args.out_dir / "summary.json", rows)
                continue

        settings = dict(**MODES[mode], model=MODEL_SETTINGS, frame_size=FRAME_SIZE,
                        target_fps=TARGET_FPS, max_frames=MAX_FRAMES, trackcraft3r_commit=TRACK_COMMIT,
                        release_ckpt_revision=CKPT_REV, checkpoint_path=str(args.ckpt),
                        model_intrinsics="frame-0 fx, fy, cx, cy from the official user NPZ",
                        output_intrinsics="per-frame calibration of the input depth",
                        coordinates="OpenCV axes, world is frame-0 camera, track_depth is camera z",
                        visibility="conf is the mean decoder probability, or 1 for an active "
                                   "sample without a visibility prediction. vis is conf > 0.5",
                        pre_query="positions held at query frame, visibility and confidence zero",
                        interleaving="strided suffixes sharing the query frame, last input repeated "
                                     "to 12, padded outputs excluded from averages",
                        projection="pinhole, zero camera z projects to the principal point")
        log = dict(clip=name, mode=mode, gpu=gpu, settings=settings)
        started = time.perf_counter()
        try:
            if loaded != name:
                clip = load_clip(args.data_dir / name, fps=TARGET_FPS, max_frames=MAX_FRAMES,
                                 size=FRAME_SIZE)
                loaded = name
            log.update(frames=len(clip["video"]), fps=round(clip["fps"], 3))
            print(f"{tag}: {log['frames']} frames at {log['fps']} fps")
            if mode == "rgb":
                # Give DA3 the whole GPU for its joint depth and pose pass over the clip.
                predictor = None
                gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            prep = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="inputs-", dir=args.work) as tmp:
                images, depth, w2c, model_k, intrinsics = prepare_input(args, mode, clip, Path(tmp))
            log.update(prepare_s=round(time.perf_counter() - prep, 2),
                       model_fx_fy_cx_cy=model_k.tolist())
            if predictor is None:
                predictor = WanSceneFlowPredictor(checkpoint_path=str(args.ckpt), device="cuda",
                                                  **MODEL_SETTINGS)
            start_infer = time.perf_counter()
            tracks, passes, fallback = track_clip(predictor, images, depth, w2c, model_k,
                                                  intrinsics, clip["size"])
            torch.cuda.synchronize()
            log.update(infer_s=round(time.perf_counter() - start_infer, 2),
                       queries=len(tracks["queries"]), passes_per_query_frame=passes,
                       visibility_fallback_samples=fallback,
                       peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       peak_scope="main PyTorch process only; DA3 runs in a separate process")
            with tempfile.TemporaryDirectory(prefix="outputs-", dir=out) as tmp:
                stage = Path(tmp)
                warnings = save_outputs(stage, clip, mode, settings, tracks, images, depth, w2c,
                                        not args.no_overlay)
                for filename in outputs:
                    if (stage / filename).exists():  # overlay.mp4 is missing if drawing failed
                        os.replace(stage / filename, out / filename)
                log["warnings"] = warnings
            log["status"] = "ok"
            del tracks, images, depth
        except Exception as exc:
            log.update(status="error", error=repr(exc), traceback=traceback.format_exc())
            print(log["traceback"], file=sys.stderr)
            predictor = None
        log["total_s"] = round(time.perf_counter() - started, 2)
        write_json(log_path, log)
        rows.append(log)
        write_json(args.out_dir / "summary.json", rows)
        print(f"{tag}: {log['status']} in {log['total_s']} s")
    print_summary(rows)
    ok = sum(row["status"] == "ok" for row in rows)
    print(f"\nDone: {ok} of {len(rows)} runs ok. Results: {args.out_dir}")
    return int(ok != len(rows))


if __name__ == "__main__":
    sys.exit(main())
