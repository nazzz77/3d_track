"""Run the pinned ZeroMSF two-image demo on original Stray frame pairs.

Outputs are the demo arrays and two input PNGs in results/<clip>/<pair>/.
The demo wall time includes its ASCII PLY and NumPy writes. Total wall time
also includes input decoding and output checks, plus model loading on first use.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ANCHORS = dict(pour_lid=488, pen=231, stir=342, shoelace=270, paper_fold=208, shirt_fold=183)
OFFSETS = (3, 9, 30)
SIZE = (512, 384)
COMMIT = "b4ea29f8c7c2c6aeb4c91d27f8fa9311f6786f79"
MODEL = "zmsf_shift+dynamicreplica+vkitti+kubric+pointodyssey+spring_adap_241106"
CONFIG = f"zmsf/configs/{MODEL}.yaml"
WEIGHTS = f"{MODEL}_last.ckpt"
DRIVE_ID = "1F0A1uicevtLvSnZlrSzOW0VREohbsUcT"
ARRAYS = ("points_3d", "scene_flow", "img_colors")
REFERENCE = ("xyz_1 is in camera 1 (x right, y down, z forward). get_sf returns flow_1 on the "
             "image 1 grid. Adding it gives corresponding positions in camera 2 coordinates. "
             "The demo uses identity poses for both views.")
SCALE = ("Monocular model units. The demo removes intrinsics and uses no ground-truth depth "
         "or scale alignment. The 1 cm statistic assumes one model unit is one meter. "
         "The depth-relative statistic is independent of a common scale factor.")


def pair_id(a, b):
    return f"f{a:04d}_f{b:04d}"


def image_paths(folder, a, b):
    return folder / f"0_f{a:04d}.png", folder / f"1_f{b:04d}.png"


def pair_settings(offset):
    return dict(config=CONFIG, size=list(SIZE), frame_offset=offset, nominal_gap_s=offset / 30,
                align_scale=False, camera_space=True, first_camera=False, rope="CroCo PyTorch fallback")


def timestamps_for(clip_dir, frames):
    # load_frames supplies RGB only. Read just the two metadata arrays from the archive.
    with np.load(clip_dir / "clip.npz") as archive:
        numbers = archive["frame_idx"]
        indices = np.searchsorted(numbers, frames)
        if np.any(indices >= len(numbers)) or not np.array_equal(numbers[indices], frames):
            raise ValueError(f"Missing original frames {frames} in {clip_dir}")
        times = archive["timestamps"][indices]
    if not np.isfinite(times).all() or times[1] <= times[0]:
        raise ValueError(f"Invalid timestamps for {frames} in {clip_dir}")
    return times.astype(float).tolist()


def read_array(folder, name, original=False):
    path = folder / (name + ("_original.npy" if original else ".npy"))
    array = np.load(path, allow_pickle=False)
    w, h = SIZE
    shape = (1, h, w, 3) if name == "points_3d" else (1, 3, h, w)
    if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError(f"Invalid {path.name}: expected finite floats of shape {shape}")
    if name == "img_colors" and (array.min() < 0 or array.max() > 1):
        raise ValueError(f"Color values outside [0, 1] in {path.name}")
    return array[0] if name == "points_3d" else np.moveaxis(array[0], 0, -1)


def motion_stats(points, flow):
    """Summarize image 1 pixels used by the official viewer, in model units."""
    depth = points[..., 2]
    valid = depth > 0
    depth = depth[valid]
    if not depth.size:
        raise ValueError("No image 1 pixels with positive depth")
    magnitude = np.linalg.norm(flow[valid], axis=-1)
    levels = (5, 25, 50, 75, 95, 99)

    def percentiles(values):
        return dict(zip(map(str, levels), np.percentile(values, levels).tolist()))

    absolute = magnitude > 0.01
    relative = magnitude > 0.02 * depth
    return dict(valid_pixels=int(depth.size), total_pixels=int(valid.size),
                depth_percentiles=percentiles(depth), flow_magnitude_percentiles=percentiles(magnitude),
                fraction_gt_1cm_assuming_meters=float(absolute.mean()),
                fraction_gt_2pct_depth=float(relative.mean()),
                fraction_gt_1cm_or_2pct_depth=float((absolute | relative).mean()))


def check_ply(path, points, flow, colors):
    """Compare the demo's filtered ASCII vertices with their source pixels."""
    is_flow = path.name == "scene_flow.ply"
    valid = points[..., 2] > 0
    xyz = points[valid]
    with path.open(encoding="ascii") as stream:
        if stream.readline().strip() != "ply" or stream.readline().strip() != "format ascii 1.0":
            raise ValueError(f"Invalid PLY header in {path.name}")
        if stream.readline().strip() != f"element vertex {len(xyz)}":
            raise ValueError(f"Wrong vertex count in {path.name}")
        names = ["x", "y", "z"] + (["flow_x", "flow_y", "flow_z"] if is_flow else [])
        properties = [f"property float {name}" for name in names]
        properties += ["property uchar red", "property uchar green", "property uchar blue", "end_header"]
        if [stream.readline().strip() for _ in properties] != properties:
            raise ValueError(f"Invalid vertex properties in {path.name}")
        rows = np.loadtxt(stream, ndmin=2)
    if rows.shape != (len(xyz), 9 if is_flow else 6):
        raise ValueError(f"Wrong PLY data dimensions in {path.name}")
    # The exporter prints six digits after the decimal point.
    if not np.allclose(rows[:, :3], xyz, rtol=0, atol=0.00000051):
        raise ValueError(f"Point coordinates differ in {path.name}")
    if is_flow:
        vectors = flow[valid]
        if not np.allclose(rows[:, 3:6], vectors, rtol=0, atol=0.00000051):
            raise ValueError(f"Displacements differ in {path.name}")
        # Flow PLY colors are auxiliary grayscale values, independent of image RGB.
        rgb = rows[:, 6:]
        if not ((rgb >= 0) & (rgb <= 255) & (rgb == rgb[:, :1]) & (rgb == np.floor(rgb))).all():
            raise ValueError(f"Invalid grayscale values in {path.name}")
    elif not np.array_equal(rows[:, 3:], (colors[valid] * 255).astype(np.uint8)):
        raise ValueError(f"Point colors differ in {path.name}")


def finish_outputs(folder, warnings):
    arrays = [read_array(folder, name) for name in ARRAYS]
    stats = motion_stats(*arrays[:2])
    redundant = []
    for name, kept in zip(ARRAYS, arrays):
        path = folder / f"{name}_original.npy"
        try:
            original = read_array(folder, name, original=True)
            if not np.allclose(original, kept):
                warnings.append(f"{path.name} differs from {name}.npy at unit resize scale")
        except ValueError as exc:
            warnings.append(str(exc))
        redundant.append(path)
    for name in ("point_cloud.ply", "scene_flow.ply"):
        path = folder / name
        try:
            check_ply(path, *arrays)
        except ValueError as exc:
            warnings.append(str(exc))
        redundant.append(path)
    if warnings:
        warnings.append("Retained both PLY files and all original arrays for inspection")
    else:
        for path in redundant:
            path.unlink()
    return stats


def read_pair(folder):
    """Read a complete pair, checking identity, timing and pixel correspondence."""
    from PIL import Image

    log = json.loads((folder / "log.json").read_text())
    if log["status"] != "ok":
        raise ValueError(log.get("error", f"Run status is {log['status']}"))
    a, b = log["frame_numbers"]
    if (folder.parent.name != log["clip"] or folder.name != pair_id(a, b)
            or folder.name != log["pair"] or a != ANCHORS[log["clip"]] or b - a not in OFFSETS):
        raise ValueError("Frame numbers or clip do not match the pair directory")
    t0, t1 = log["timestamps_s"]
    if not np.isfinite([t0, t1, log["gap_s"]]).all() or t1 <= t0 or not np.isclose(t1 - t0, log["gap_s"]):
        raise ValueError("Inconsistent pair timestamps")
    points, flow, colors = (read_array(folder, name) for name in ARRAYS)
    images = []
    for path in image_paths(folder, a, b):
        with Image.open(path) as img:
            if (img.format, img.size, img.mode) != ("PNG", SIZE, "RGB"):
                raise ValueError(f"Expected a {SIZE} RGB PNG in {path.name}")
            images.append(np.array(img))
    source_colors = colors
    if (folder / "img_colors_original.npy").exists():
        source_colors = read_array(folder, "img_colors", original=True)
    if not np.array_equal(np.rint(source_colors * 255).astype(np.uint8), images[0]):
        raise ValueError("Point colors do not match the first input PNG")
    if not (points[..., 2] > 0).any():
        raise ValueError("Empty positive-depth point cloud")
    return log, points, flow, colors, images


def verify_pair(folder):
    log, points, flow, colors, _ = read_pair(folder)
    a, b = log["frame_numbers"]
    if (log["settings"] != pair_settings(b - a) or not log["reference_frame"] or not log["scale_note"]
            or not log["gpu"] or not isinstance(log["warnings"], list)):
        raise ValueError("Missing or inconsistent pair settings")
    revision = log["checkpoint"]
    if (log["official_commit"] != COMMIT or revision["drive_id"] != DRIVE_ID
            or revision["file"] != WEIGHTS or revision["size_bytes"] <= 0
            or not re.fullmatch("[0-9a-f]{64}", revision["sha256"])):
        raise ValueError("Invalid official code or checkpoint revision")
    resources = [log["demo_s"], log["total_s"], log["peak_allocated_gib"]]
    if not np.isfinite(resources).all() or min(resources) < 0 or log["demo_s"] > log["total_s"]:
        raise ValueError("Invalid timing or memory measurements")
    for key, expected in motion_stats(points, flow).items():
        actual = log["stats"][key]
        if isinstance(expected, dict):
            if actual.keys() != expected.keys() or not all(np.isclose(actual[k], v) for k, v in expected.items()):
                raise ValueError(f"Inconsistent {key}")
        elif not np.isclose(actual, expected):
            raise ValueError(f"Inconsistent {key}")
    # A warning preserves the entire diagnostic set, including deliberate differences.
    originals = [folder / f"{name}_original.npy" for name in ARRAYS]
    plies = [folder / name for name in ("point_cloud.ply", "scene_flow.ply")]
    if log["warnings"] or any(p.exists() for p in originals + plies):
        if not log["warnings"] or not all(p.is_file() for p in originals + plies):
            raise ValueError("Incomplete or unexplained retained demo exports")
        for name, kept in zip(ARRAYS, (points, flow, colors)):
            if not np.allclose(read_array(folder, name, original=True), kept):
                if not any(f"{name}_original.npy" in message for message in log["warnings"]):
                    raise ValueError(f"Unreported difference in {name}_original.npy")
        for ply in plies:
            try:
                check_ply(ply, points, flow, colors)
            except ValueError as exc:
                if str(exc) not in log["warnings"]:
                    raise
    return log


def write_json(path, value):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_model(repo):
    import yaml

    os.chdir(repo)  # The official config's weight_path is relative to the checkout.
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise ValueError("Unexpected ZeroMSF checkout revision")
    ckpt = repo / "checkpoints" / WEIGHTS
    provenance = json.loads(Path(str(ckpt) + ".json").read_text())
    digest = hashlib.sha256()
    with ckpt.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    if provenance != dict(file=WEIGHTS, drive_id=DRIVE_ID, size_bytes=ckpt.stat().st_size,
                          sha256=digest.hexdigest()):
        raise ValueError("Checkpoint differs from its setup manifest")
    sys.path.insert(0, str(repo))
    import demo

    model = demo.instantiate_component(yaml.safe_load((repo / CONFIG).read_text()))
    return demo, model, provenance


def predict_pair(demo, model, load_frames, clip_dir, folder, log):
    import torch
    from PIL import Image

    frames = log["frame_numbers"]
    paths = image_paths(folder, *frames)
    for pixels, path in zip(load_frames(clip_dir, frames, size=SIZE), paths):
        Image.fromarray(pixels).save(path)
    torch.cuda.synchronize()
    start = time.monotonic()
    view1, view2 = demo.prepare_input_data(*paths, target_size=SIZE)
    demo.run_inference(model, view1, view2, folder)
    torch.cuda.synchronize()
    log["demo_s"] = time.monotonic() - start
    log["stats"] = finish_outputs(folder, log["warnings"])


def run_jobs(args):
    sys.path.insert(0, str(HERE.parent / "common"))
    from track3d_data import load_frames

    # Preserve summaries of other clips on selective reruns. Pair logs are authoritative.
    rows = {}
    for path in args.out_dir.glob("*/*/log.json"):
        try:
            row = json.loads(path.read_text())
            if not isinstance(row, dict):
                raise ValueError("Expected a pair log")
            rows[(path.parent.parent.name, path.parent.name)] = row
        except ValueError:
            rows[(path.parent.parent.name, path.parent.name)] = dict(
                clip=path.parent.parent.name, pair=path.parent.name, status="error", error="Invalid log.json")

    model = init_error = None
    manifest = args.repo_dir / "checkpoints" / (WEIGHTS + ".json")
    current_checkpoint = None
    if manifest.is_file():
        try:
            current_checkpoint = json.loads(manifest.read_text())
        except (OSError, ValueError) as exc:
            init_error = f"Cannot read checkpoint manifest: {exc}"
    selected = []
    for clip in dict.fromkeys(args.clips):
        a = ANCHORS[clip]
        for offset in OFFSETS:
            b = a + offset
            pair = pair_id(a, b)
            folder = args.out_dir / clip / pair
            folder.mkdir(parents=True, exist_ok=True)
            old = rows.get((clip, pair), {})
            skip = False
            same_weights = current_checkpoint is None or old.get("checkpoint") == current_checkpoint
            if old.get("status") == "ok" and same_weights:
                try:
                    verify_pair(folder)
                    skip = True
                except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, IndexError) as exc:
                    print(f"Rebuilding {clip}/{pair}: {exc}")
            if skip:
                log = old
            else:
                start = time.monotonic()
                log = dict(clip=clip, pair=pair, status="running", frame_numbers=[a, b],
                           timestamps_s=None, gap_s=None, official_commit=COMMIT,
                           checkpoint=dict(file=WEIGHTS, drive_id=DRIVE_ID),
                           settings=pair_settings(offset),
                           reference_frame=REFERENCE, scale_note=SCALE, warnings=[],
                           demo_s=None, peak_allocated_gib=None)
                write_json(folder / "log.json", log)
                peak_started = False
                try:
                    clip_dir = args.data_dir / clip
                    t0, t1 = timestamps_for(clip_dir, [a, b])
                    log.update(timestamps_s=[t0, t1], gap_s=t1 - t0)
                    if init_error:
                        raise RuntimeError(init_error)
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError("ZeroMSF requires a CUDA GPU for this job")
                    log["gpu"] = torch.cuda.get_device_name(0)
                    torch.cuda.reset_peak_memory_stats()
                    peak_started = True
                    if model is None:
                        try:
                            demo, model, provenance = load_model(args.repo_dir)
                        except Exception as exc:
                            init_error = f"Model initialization failed: {exc!r}"
                            raise
                    log["checkpoint"] = provenance
                    predict_pair(demo, model, load_frames, clip_dir, folder, log)
                    log["status"] = "ok"
                except Exception as exc:
                    log.update(status="error", error=repr(exc), traceback=traceback.format_exc())
                if peak_started:
                    log["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                log["total_s"] = time.monotonic() - start
                write_json(folder / "log.json", log)
            rows[(clip, pair)] = log
            write_json(args.out_dir / "summary.json", [rows[key] for key in sorted(rows)])
            status = "skipped" if skip else log["status"]
            selected.append((log, status))
            print(f"{clip}/{pair} {status}", log.get("error") or " ".join(log["warnings"]), flush=True)

    print(f"\n{'Clip':12} {'Pair':12} {'Status':8} {'Total s':>9} {'Peak GiB':>9}")
    for row, status in selected:
        peak = row["peak_allocated_gib"]
        memory = "-" if peak is None else f"{peak:.3f}"
        print(f"{row['clip']:12} {row['pair']:12} {status:8} {row['total_s']:9.2f} {memory:>9}")
    return int(any(row["status"] != "ok" for row, _ in selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", choices=ANCHORS, default=list(ANCHORS))
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "preprocessed_data")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    work = Path(os.environ.get("ZEROMSF_WORK", HERE / "work"))
    parser.add_argument("--repo-dir", type=Path, default=work / "zero-msf")
    args = parser.parse_args()
    args.data_dir, args.repo_dir, args.out_dir = (p.resolve() for p in (args.data_dir, args.repo_dir, args.out_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Serialize overlapping submissions, including their summary updates, without a result file.
    fd = os.open(args.out_dir, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return run_jobs(args)
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
