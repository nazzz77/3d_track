"""Convert Stray Scanner recordings (raw_data/<clip>.zip) into a method-agnostic format.

Output per clip, in preprocessed_data/<clip>/:
  clip.npz   every recorded frame (30 fps) that has RGB, depth and pose, as 960x720 JPEG bytes, raw LiDAR depth and
             confidence (256x192), per-frame intrinsics at the original resolution,
             raw ARKit poses, timestamps
  meta.json  summary: frame count, fps, sizes, camera-static check, depth confidence stats

Each tracking method reads this with common/track3d_data.py and resizes or subsamples
for its own needs, so every method sees exactly the same frames.

Runs on macOS with the system Python packages numpy, pandas and Pillow. Frames are decoded
with stray_decode.swift (built-in AVFoundation), which this script compiles if needed.

Usage (from the track3d folder):
  python3 common/preprocess_stray.py                 # all clips in raw_data
  python3 common/preprocess_stray.py --clips pen     # selected clips
  python3 common/preprocess_stray.py --overwrite     # redo existing outputs
"""
import argparse
import datetime
import glob
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_W, FRAME_H = 960, 720
JPEG_QUALITY = 0.9
FORMAT_VERSION = 1

TASKS = {
    "pour_lid": "remove the lid of a tumbler and pour water into a bowl",
    "pen": "pick a pen out of a bowl, place it on the table, then put it back",
    "stir": "stir vegetables in a pan with a spatula",
    "shoelace": "tie a shoelace on a shoe",
    "paper_fold": "fold a sheet of paper twice (simple origami)",
    "shirt_fold": "fold a shirt, starting with the sleeves already folded in",
}


def build_decoder(work_dir):
    src = os.path.join(ROOT, "common", "stray_decode.swift")
    exe = os.path.join(work_dir, "stray_decode")
    subprocess.run(["swiftc", "-O", "-suppress-warnings", src, "-o", exe], check=True)
    return exe


def quat_to_c2w(xyz, q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)
    T = np.repeat(np.eye(4)[None], len(q), 0)
    T[:, :3, :3] = R
    T[:, :3, 3] = xyz
    return T.astype(np.float32)


def pixel_shift(a, b):
    """Integer translation between two grayscale images (phase correlation)."""
    A = np.fft.fft2(a - a.mean())
    B = np.fft.fft2(b - b.mean())
    R = A * np.conj(B)
    r = np.fft.ifft2(R / (np.abs(R) + 1e-9)).real
    y, x = np.unravel_index(r.argmax(), r.shape)
    if y > a.shape[0] // 2:
        y -= a.shape[0]
    if x > a.shape[1] // 2:
        x -= a.shape[1]
    return int(x), int(y)


def process_clip(clip, zip_path, out_dir, decoder, tmp_root):
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        raw = os.path.join(tmp, "raw")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw)
        rgbs = glob.glob(os.path.join(raw, "**", "rgb.mp4"), recursive=True)
        if len(rgbs) != 1:
            raise RuntimeError(f"{zip_path}: expected one rgb.mp4, found {len(rgbs)}")
        sdir = os.path.dirname(rgbs[0])

        fdir = os.path.join(tmp, "frames")
        subprocess.run([decoder, rgbs[0], fdir, str(FRAME_W), str(FRAME_H), str(JPEG_QUALITY)], check=True)

        odo = pd.read_csv(os.path.join(sdir, "odometry.csv"), skipinitialspace=True)
        odo.columns = [c.strip() for c in odo.columns]
        odo["frame"] = odo["frame"].astype(int)
        odo = odo.sort_values("frame").reset_index(drop=True)

        def indices(sub):
            return {int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.join(sdir, sub, "*.png"))}

        # Video frame k, depth/k.png, confidence/k.png and odometry row k belong together.
        # Some recordings miss depth for the first frames, so keep only frames present in all streams.
        rgb_idx = set(range(len(glob.glob(os.path.join(fdir, "*.jpg")))))
        depth_idx, conf_idx, odo_idx = indices("depth"), indices("confidence"), set(odo["frame"])
        counts = dict(rgb=len(rgb_idx), depth=len(depth_idx), confidence=len(conf_idx), odometry=len(odo_idx))
        keep = sorted(rgb_idx & depth_idx & conf_idx & odo_idx)
        dropped = sorted((rgb_idx | depth_idx | conf_idx | odo_idx) - set(keep))
        if keep != list(range(keep[0], keep[-1] + 1)):
            raise RuntimeError(f"{clip}: frames present in all streams are not contiguous: {counts}")
        if len(dropped) > 10:
            raise RuntimeError(f"{clip}: too many incomplete frames ({len(dropped)}): {counts}")
        n = len(keep)

        jpegs = [open(os.path.join(fdir, f"{i:06d}.jpg"), "rb").read() for i in keep]
        offsets = np.zeros(n + 1, np.int64)
        offsets[1:] = np.cumsum([len(b) for b in jpegs])
        jpeg_bytes = np.frombuffer(b"".join(jpegs), np.uint8)

        depth_mm = np.stack([np.asarray(Image.open(os.path.join(sdir, "depth", f"{i:06d}.png")))
                             for i in keep]).astype(np.uint16)
        conf = np.stack([np.asarray(Image.open(os.path.join(sdir, "confidence", f"{i:06d}.png")))
                         for i in keep]).astype(np.uint8)

        o = odo.set_index("frame").loc[keep]
        K = np.zeros((n, 3, 3), np.float32)
        K[:, 0, 0], K[:, 1, 1] = o["fx"], o["fy"]
        K[:, 0, 2], K[:, 1, 2] = o["cx"], o["cy"]
        K[:, 2, 2] = 1.0
        arkit_c2w = quat_to_c2w(o[["x", "y", "z"]].to_numpy(), o[["qx", "qy", "qz", "qw"]].to_numpy())
        t = o["timestamp"].to_numpy()
        timestamps = (t - t[0]).astype(np.float64)
        fps = (n - 1) / (t[-1] - t[0])

        first = np.asarray(Image.open(io.BytesIO(jpegs[0])).convert("L"), np.float64)
        last = np.asarray(Image.open(io.BytesIO(jpegs[-1])).convert("L"), np.float64)
        shift = pixel_shift(first, last)
        orig_w, orig_h = [int(v) for v in open(os.path.join(fdir, "source_size.txt")).read().split()]

        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, "clip.npz"),
            jpeg_bytes=jpeg_bytes, jpeg_offsets=offsets,
            depth_mm=depth_mm, depth_conf=conf,
            intrinsics=K, arkit_c2w=arkit_c2w,
            timestamps=timestamps, frame_idx=np.array(keep),
            frame_size=np.array([FRAME_W, FRAME_H]), orig_size=np.array([orig_w, orig_h]),
            depth_size=np.array([depth_mm.shape[2], depth_mm.shape[1]]),
        )

    meta = dict(
        clip=clip,
        task=TASKS.get(clip, ""),
        source_zip=os.path.relpath(zip_path, ROOT),
        format_version=FORMAT_VERSION,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        n_frames=int(n),
        fps=round(float(fps), 3),
        duration_s=round(float(timestamps[-1]), 3),
        frame_counts=counts,
        dropped_frames=dropped,
        frame_size=[FRAME_W, FRAME_H],
        orig_size=[orig_w, orig_h],
        depth_size=[int(depth_mm.shape[2]), int(depth_mm.shape[1])],
        jpeg_quality=JPEG_QUALITY,
        camera_static=bool(max(abs(shift[0]), abs(shift[1])) <= 4),
        first_last_shift_px=list(shift),
        depth_conf_fraction={str(k): round(float((conf == k).mean()), 3) for k in (0, 1, 2)},
        depth_range_m=[round(float(depth_mm[depth_mm > 0].min()) / 1000, 3),
                       round(float(depth_mm.max()) / 1000, 3)],
        notes=[
            "frame_idx holds the original Stray frame numbers; frames missing RGB, depth, confidence "
            "or odometry are listed in dropped_frames; timestamps start at the first kept frame",
            "intrinsics are in pixels of orig_size (1920x1440); rescale for other sizes",
            "depth_mm is raw LiDAR depth in millimeters at depth_size; depth_conf is 0 (low), 1, 2 (high)",
            "arkit_c2w are raw ARKit camera-to-world poses (ARKit camera axes: x right, y up, z backward); "
            "they drift by a few degrees although the phone was on a fixed tripod, so treat the camera as static",
        ],
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default=os.path.join(ROOT, "raw_data"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "preprocessed_data"))
    p.add_argument("--clips", nargs="*", help="clip names (zip names without .zip); default: all")
    p.add_argument("--decoder", help="prebuilt stray_decode binary; compiled automatically if omitted")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    zips = sorted(glob.glob(os.path.join(args.raw_dir, "*.zip")))
    clips = {os.path.splitext(os.path.basename(z))[0]: z for z in zips}
    if args.clips:
        missing = [c for c in args.clips if c not in clips]
        if missing:
            sys.exit(f"not found in {args.raw_dir}: {missing}")
        clips = {c: clips[c] for c in args.clips}

    with tempfile.TemporaryDirectory() as tmp_root:
        decoder = args.decoder or build_decoder(tmp_root)
        for clip, zip_path in clips.items():
            out_dir = os.path.join(args.out_dir, clip)
            if os.path.exists(os.path.join(out_dir, "meta.json")) and not args.overwrite:
                print(f"[skip] {clip}: already preprocessed (use --overwrite to redo)")
                continue
            meta = process_clip(clip, zip_path, out_dir, decoder, tmp_root)
            print(f"[done] {clip}: {meta['n_frames']} frames, {meta['fps']} fps, {meta['duration_s']} s, "
                  f"camera static={meta['camera_static']} (shift {meta['first_last_shift_px']} px)")


if __name__ == "__main__":
    main()
