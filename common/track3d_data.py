"""Load preprocessed track3d clips (preprocessed_data/<clip>/clip.npz) for any tracking method.

Every method should load its frames through load_clip(), so that all methods see the same
frames, the same LiDAR depth and the same camera intrinsics. Only numpy and Pillow are needed.

Example (SpatialTrackerV2 inputs):
    clip = load_clip("preprocessed_data/pen", fps=10, max_frames=200, size=(518, 392))
    clip["video"]       # (T, 392, 518, 3) uint8
    clip["depth"]       # (T, 392, 518) float32 meters, 0 where invalid
    clip["intrinsics"]  # (T, 3, 3) in pixels of the 518x392 frames
"""
import io
import json
import math
import os

import numpy as np
from PIL import Image


def read_meta(clip_dir):
    with open(os.path.join(clip_dir, "meta.json")) as f:
        return json.load(f)


def nearest_resize(a, width, height):
    """Nearest-neighbour resize of an (H, W) or (T, H, W) array using pixel centers."""
    h, w = a.shape[-2:]
    ys = np.minimum(((np.arange(height) + 0.5) * h / height).astype(int), h - 1)
    xs = np.minimum(((np.arange(width) + 0.5) * w / width).astype(int), w - 1)
    return a[..., ys, :][..., xs]


def scale_intrinsics(K, src_size, dst_size):
    """Rescale (..., 3, 3) intrinsics from src_size=(W, H) to dst_size=(W, H) (pixel-center convention)."""
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    K = np.array(K, dtype=np.float32, copy=True)
    K[..., 0, 0] *= sx
    K[..., 1, 1] *= sy
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * sx - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * sy - 0.5
    return K


def select_frames(n, src_fps, fps=None, max_frames=None):
    """Indices into the stored frames for a target fps, capped at max_frames."""
    stride = 1 if fps is None else max(1, round(src_fps / fps))
    if max_frames is not None and math.ceil(n / stride) > max_frames:
        stride = math.ceil(n / max_frames)
    return np.arange(0, n, stride), stride


def load_frames(clip_dir, frame_idx, size=None):
    """Decode original Stray frame numbers in request order as (T, H, W, 3) uint8.

    size is (width, height). By default, keep the stored JPEG dimensions.
    """
    path = os.path.join(clip_dir, "clip.npz")
    requested = np.asarray(frame_idx)
    if requested.ndim != 1 or (requested.size and requested.dtype.kind not in "iu"):
        raise ValueError("frame_idx must be a one-dimensional sequence of integer frame numbers")
    with np.load(path) as data:
        stored = data["frame_idx"]
        indices = np.searchsorted(stored, requested)
        found = indices < len(stored)
        found[found] = stored[indices[found]] == requested[found]
        if not found.all():
            missing = np.unique(requested[~found]).tolist()
            raise ValueError(f"{path}: missing original Stray frame numbers {missing}")
        width, height = map(int, data["frame_size"] if size is None else size)
        frames = np.empty((len(indices), height, width, 3), dtype=np.uint8)
        if len(indices):
            buf, offsets = data["jpeg_bytes"], data["jpeg_offsets"]
            for t, i in enumerate(indices):
                with Image.open(io.BytesIO(buf[offsets[i]:offsets[i + 1]].tobytes())) as jpeg:
                    image = jpeg.convert("RGB")
                    if image.size != (width, height):
                        image = image.resize((width, height), Image.Resampling.BICUBIC)
                    frames[t] = image
    return frames


def load_clip(clip_dir, fps=None, max_frames=None, size=None, depth_size="frame"):
    """Load one clip.

    Args:
        clip_dir: preprocessed_data/<clip> folder.
        fps: target frame rate (stored clips are about 30 fps); None keeps every frame.
        max_frames: upper limit on the number of frames; the stride grows to respect it.
        size: (width, height) of the returned frames; None keeps the stored 960x720.
        depth_size: "frame" resizes depth and confidence to `size` (nearest neighbour);
            "native" keeps the LiDAR resolution (256x192).

    Returns a dict with
        video (T, H, W, 3) uint8, depth (T, h, w) float32 meters (0 = invalid),
        conf (T, h, w) uint8 LiDAR confidence 0/1/2, intrinsics (T, 3, 3) for `size`,
        depth_intrinsics (T, 3, 3) for the returned depth resolution,
        c2w (T, 4, 4) identity camera-to-world (the phone was on a fixed tripod),
        arkit_c2w (T, 4, 4) raw ARKit poses (drift a few degrees; for reference only),
        frame_idx (T,) original Stray frame numbers, timestamps (T,) seconds,
        fps (effective frame rate), stride, size (W, H), orig_size (W, H), meta.
    """
    meta = read_meta(clip_dir)
    data = np.load(os.path.join(clip_dir, "clip.npz"))
    n = int(meta["n_frames"])
    sel, stride = select_frames(n, meta["fps"], fps, max_frames)

    frame_size = tuple(int(v) for v in data["frame_size"])
    orig_size = tuple(int(v) for v in data["orig_size"])
    depth_native = tuple(int(v) for v in data["depth_size"])
    size = frame_size if size is None else (int(size[0]), int(size[1]))

    buf, off = data["jpeg_bytes"], data["jpeg_offsets"]
    frames = []
    for i in sel:
        img = Image.open(io.BytesIO(buf[off[i]:off[i + 1]].tobytes())).convert("RGB")
        if img.size != size:
            img = img.resize(size, Image.BICUBIC)
        frames.append(np.asarray(img))
    video = np.stack(frames)

    depth = data["depth_mm"][sel].astype(np.float32) / 1000.0
    conf = data["depth_conf"][sel]
    if depth_size == "frame":
        depth = nearest_resize(depth, *size)
        conf = nearest_resize(conf, *size)
        d_size = size
    elif depth_size == "native":
        d_size = depth_native
    else:
        raise ValueError(f"depth_size must be 'frame' or 'native', got {depth_size!r}")

    K = data["intrinsics"][sel]
    T = len(sel)
    return dict(
        video=video,
        depth=depth,
        conf=conf,
        intrinsics=scale_intrinsics(K, orig_size, size),
        depth_intrinsics=scale_intrinsics(K, orig_size, d_size),
        c2w=np.repeat(np.eye(4, dtype=np.float32)[None], T, 0),
        arkit_c2w=data["arkit_c2w"][sel],
        frame_idx=data["frame_idx"][sel],
        timestamps=data["timestamps"][sel],
        fps=float(meta["fps"]) / stride,
        stride=int(stride),
        size=size,
        orig_size=orig_size,
        meta=meta,
    )
