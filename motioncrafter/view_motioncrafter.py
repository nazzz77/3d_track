"""View MotionCrafter scene flow following the authors' visualize/visualize.py.

This local NPZ viewer uses the shared Stray RGB frames and identity poses. It
keeps all clip frames available, starts paused behind the recording camera and
uses a nominal 60 degree vertical frustum (intrinsics are absent). Its single
slider supports stepping and scrubbing. Clip selection, display toggles, point
stride and a window readout are local controls. Stride affects only the clouds.
The two colored clouds and textured frustum update in place. Only the current
frame and five previous frames contribute lines, including after a seek. The
hue ramp uses NumPy HSV colors and the capped sample is seeded per frame.
The hidden deformed cloud, remote sharing and HDF5/video inputs are omitted.
"""
import argparse
from functools import lru_cache, partial
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

from run_motioncrafter import (
    CHECKPOINTS, COMMIT, HERE, ROOT, SIZE, clip_stats, load_maps, shared_clip, window_starts,
)


def smooth_pixels(points):
    """Compare right and lower neighbors, extending the last difference at each edge."""
    horizontal = np.linalg.norm(np.diff(points, axis=1), axis=2) <= 0.1
    vertical = np.linalg.norm(np.diff(points, axis=0), axis=2) <= 0.1
    return (np.pad(horizontal, ((0, 0), (0, 1)), mode="edge")
            & np.pad(vertical, ((0, 1), (0, 0)), mode="edge"))


def visible_pixels(points, valid):
    keep = smooth_pixels(points) & valid
    keep[:10] = False
    keep[-40:] = False
    keep[:, :20] = False
    keep[:, -20:] = False
    return keep


def flow_segments(points, ends, moving, center, t):
    pixels = np.flatnonzero(moving)
    count = len(pixels)
    ranks = np.arange(count)
    if count > 6000:
        ranks = np.array(random.Random(42 + t).sample(range(count), 6000))
        pixels = pixels[ranks]
    segments = np.stack((points.reshape(-1, 3)[pixels], ends.reshape(-1, 3)[pixels]), axis=1) - center
    # Evaluate the RGB channels of a full-saturation HSV ramp at original pixel ranks.
    phases = 6.0 * ranks[:, None] / max(1, count - 1) + np.array([0, 4, 2])
    rgb = (255 * np.clip(np.abs(phases % 6 - 3) - 1, 0, 1)).astype(np.uint8)
    return segments, np.repeat(rgb[:, None, :], 2, axis=1)


def frame_geometry(maps, center, t):
    points = maps["point_map"][t].astype(np.float32)
    ends = points + maps["scene_flow"][t]
    visible = visible_pixels(points, maps["valid_mask"][t])
    # The official threshold uses the entire image, before validity and stride.
    magnitude = np.linalg.norm(ends - points, axis=-1)
    moving = (visible & maps["deform_mask"][t] & smooth_pixels(ends)
              & (magnitude > magnitude.mean() + 3 * magnitude.std()))
    lines, colors = flow_segments(points, ends, moving, center, t)
    return points - center, visible & ~moving, moving, lines, colors


class Clip:
    def __init__(self, folder, data_dir):
        name = folder.name
        log = json.loads((folder / "log.json").read_text())
        if log["status"] != "ok":
            raise ValueError(f"run status {log['status']}: {log.get('error', '')}")
        if log["clip"] != name or log["size"] != list(SIZE):
            raise ValueError("Log clip or resolution mismatch")
        if log["official_commit"] != COMMIT or log["checkpoint_revisions"] != CHECKPOINTS:
            raise ValueError("Unexpected MotionCrafter source or checkpoint revisions")
        if log["settings"]["overlap"] != 0 or log["viewer_poses"] != "identity" or log["metric_scale"]:
            raise ValueError("Expected independent windows, identity poses and model-scale geometry")
        data = shared_clip(data_dir, name)
        self.rgb = data["video"]
        if log["T"] != len(self.rgb) or not np.array_equal(log["frame_idx"], data["frame_idx"]):
            raise ValueError("Logged frame count or frame_idx differs from the shared clip")
        if log["fps"] != data["fps"] or not np.array_equal(log["timestamps"], data["timestamps"]):
            raise ValueError("Logged timestamps or fps differ from the shared clip")
        del data
        self.starts = window_starts(len(self.rgb), log["settings"])
        if self.starts != log["window_starts"]:
            raise ValueError("Inconsistent window start indices")
        path = folder / f"{name}.npz"
        if path.stat().st_size != log["archive_bytes"]:
            raise ValueError(f"Unexpected byte count for {path.name}")
        self.maps = load_maps(path, len(self.rgb))
        points = self.maps["point_map"][0].astype(np.float32)
        reference = points[visible_pixels(points, self.maps["valid_mask"][0])]
        if not len(reference):
            raise ValueError("Frame 0 has no valid points after noise removal")
        self.center = reference.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.extent = np.percentile(reference, [2, 98], axis=0) - self.center
        self.log = log
        self.frame = lru_cache(maxsize=6)(partial(frame_geometry, self.maps, self.center))


def check_stats(clip):
    actual = clip_stats(clip.maps)
    stored = clip.log["stats"]
    if not np.isclose(actual["deform_mask_fraction"], stored["deform_mask_fraction"]):
        raise ValueError("deform_mask fraction differs from log")
    for field in ("valid_depth_percentiles", "valid_deform_flow_magnitude_percentiles"):
        for level, value in actual[field].items():
            recorded = stored[field][level]
            if value is None and recorded is None:
                continue
            if value is None or recorded is None or not np.isclose(value, recorded, rtol=1e-5):
                raise ValueError(f"{field} p{level} differs from log")


def check_all(results, names, data_dir):
    # A copied per-clip folder is self-contained. If a summary is present,
    # also verify its rows against the individual logs.
    summary = {}
    summary_path = results / "summary.json"
    failed = False
    if summary_path.exists():
        try:
            rows = json.loads(summary_path.read_text())
            if not isinstance(rows, list):
                raise ValueError("Expected a list of clip summaries")
            indexed = {row["clip"]: row for row in rows}
            if (len(rows) != len(indexed) or any(not isinstance(name, str) or not name
                    or name == ".." or name != Path(name).name for name in indexed)):
                raise ValueError("Expected one summary row per clip")
            summary = indexed
        except (ValueError, KeyError, TypeError, OSError) as exc:
            print(f"summary.json: ERROR {exc}")
            failed = True
    for name in sorted(set(names) | set(summary)):
        try:
            clip = Clip(results / name, data_dir)
            try:
                if summary_path.exists() and clip.log != summary.get(name):
                    raise ValueError("summary.json and log.json disagree")
                check_stats(clip)
                print(f"{name}: OK  T={len(clip.rgb)}  windows={len(clip.starts)}  fps={clip.log['fps']:.3f}")
            finally:
                del clip
        except Exception as exc:
            print(f"{name}: ERROR {exc}")
            failed = True
    if not names:
        print(f"No clip results in {results}", file=sys.stderr)
    return int(failed or not names)


class Viewer:
    def __init__(self, server, results, data_dir, names, initial):
        self.server, self.results, self.data_dir = server, results, data_dir
        self.clip = self.name = None
        scene, gui = server.scene, server.gui
        scene.set_up_direction("-y")
        self.selector = gui.add_dropdown("Clip", names, initial_value=initial)
        self.slider = gui.add_slider("Frame", 0, 1, 1, 0)
        self.play = gui.add_checkbox("Playing", False)
        self.fps = gui.add_slider("Playback fps", 1.0, 60.0, 0.01, 10.0)
        self.stride = gui.add_dropdown("Point stride", ("1", "2", "4"), initial_value="2")
        self.size = gui.add_slider("Point size", 0.001, 0.1, 0.001, 0.01)
        self.width = gui.add_slider("Line width (px)", 0.1, 10.0, 0.1, 1.0)
        self.show_static = gui.add_checkbox("Static points", True)
        self.show_moving = gui.add_checkbox("Moving points", True)
        self.show_flow = gui.add_checkbox("Flow lines", True)
        self.window = gui.add_markdown("")
        empty = np.empty((0, 3), np.float32)
        self.clouds = [scene.add_point_cloud("/" + name, empty, np.empty((0, 3), np.uint8),
                                             point_shape="rounded", point_size=self.size.value)
                       for name in ("static", "moving")]
        self.lines = scene.add_line_segments("/flow", empty.reshape(0, 2, 3),
                                             np.empty((0, 2, 3), np.uint8),
                                             thickness=1.0, thickness_units="screen")
        self.frustum = scene.add_camera_frustum("/camera", fov=np.deg2rad(60), aspect=SIZE[0] / SIZE[1],
                                                scale=0.05)
        scene.add_frame("/camera/axes", axes_length=0.05, axes_radius=0.005)
        self.select(initial)

    def select(self, name):
        self.play.value = False
        self.name, self.clip = name, None
        for node in (*self.clouds, self.lines, self.frustum):
            node.visible = False
        self.frustum.image = None
        self.window.content = f"Loading {name}..."
        try:
            self.clip = Clip(self.results / name, self.data_dir)
        except Exception as exc:
            self.window.content = f"Cannot load {name}: {exc}"
            print(self.window.content, file=sys.stderr)
            self.slider.value = self.slider.max = 0
            return
        self.slider.value = 0
        self.slider.max = len(self.clip.rgb) - 1
        self.fps.value = self.clip.log["fps"]
        self.last_frame = self.last_stride = self.last_trail = None
        self.frustum.position = -self.clip.center
        self.frustum.visible = True
        low, high = self.clip.extent
        span = max(float((high - low)[:2].max()), 0.1)
        eye = -self.clip.center.copy()
        # MotionCrafter's origin lies inside the scene (point-map z spans about -0.6..1.5),
        # so step well back from the nearest points to see the whole frame.
        eye[2] = min(eye[2], low[2]) - 0.8 * span
        self.server.initial_camera.position = eye
        self.server.initial_camera.look_at = (0, 0, 0)
        self.server.initial_camera.up = (0, -1, 0)
        for client in self.server.get_clients().values():
            client.camera.position = eye
            client.camera.look_at = (0, 0, 0)
            client.camera.up_direction = (0, -1, 0)
        self.render()

    def render(self):
        clip = self.clip
        t = min(self.slider.value, len(clip.rgb) - 1)
        stride = int(self.stride.value)
        with self.server.atomic():
            if (t, stride) != (self.last_frame, self.last_stride):
                points, static, moving, _, _ = clip.frame(t)
                for node, mask in zip(self.clouds, (static, moving)):
                    sample = mask[::stride, ::stride]
                    node.points = points[::stride, ::stride][sample]
                    node.colors = clip.rgb[t, ::stride, ::stride][sample]
                self.last_stride = stride
            if self.show_flow.value and self.last_trail != t:
                trail = [clip.frame(i) for i in range(max(0, t - 5), t + 1)]
                self.lines.points = np.concatenate([frame[3] for frame in trail])
                self.lines.colors = np.concatenate([frame[4] for frame in trail])
                self.last_trail = t
            if self.last_frame != t:
                self.frustum.image = clip.rgb[t]
                index = int(np.searchsorted(clip.starts, t, side="right") - 1)
                stop = (clip.starts + [len(clip.rgb)])[index + 1]
                self.window.content = (f"Independent window **{index + 1}/{len(clip.starts)}**, "
                                       f"frames **{clip.starts[index]}-{stop - 1}** (zero-based).")
                self.last_frame = t
            self.clouds[0].visible = self.show_static.value
            self.clouds[1].visible = self.show_moving.value
            for cloud in self.clouds:
                cloud.point_size = self.size.value
            self.lines.visible = self.show_flow.value
            self.lines.thickness = self.width.value

    def serve(self):
        deadline = time.monotonic()
        while True:
            if self.name != self.selector.value:
                self.select(self.selector.value)
            if self.clip is not None:
                now = time.monotonic()
                if self.play.value and now >= deadline:
                    self.slider.value = (self.slider.value + 1) % len(self.clip.rgb)
                    deadline = now + 1 / self.fps.value
                self.render()
            time.sleep(0.015)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", help="Initial clip in the viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--check", action="store_true", help="Validate all runs and exit")
    parser.add_argument("--results-root", type=Path, default=HERE / "results", help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "preprocessed_data", help=argparse.SUPPRESS)
    args = parser.parse_args()
    names = tuple(sorted(p.name for p in args.results_root.glob("*") if p.is_dir()))
    if args.check:
        return check_all(args.results_root, names, args.data_dir)
    if not names or (args.clip and args.clip not in names):
        parser.exit(1, f"No matching clip in {args.results_root}. Available: {', '.join(names)}\n")
    import viser
    server = viser.ViserServer(host=args.host, port=args.port, label="MotionCrafter scene flow")
    try:
        viewer = Viewer(server, args.results_root, args.data_dir, names, args.clip or names[0])
        print(f"Ready: {viewer.name} at http://{args.host}:{server.get_port()}", flush=True)
        viewer.serve()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
