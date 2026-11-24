"""Compare point tracking runs on a shared Stray clip in a local viser scene."""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    import viser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("spatracker_v2", "delta", "trackcraft3r", "track4world", "traceanything")
SETTINGS = ("rgb", "rgbd_fixed")
DEFAULT_MOTION_PX = 60
TRACK_KEYS = (
    "queries", "tracks3d_world", "vis", "tracks2d_orig", "c2w", "intrinsics",
    "frame_idx", "timestamps", "fps", "size", "orig_size",
)
GEOMETRY_KEYS = ("depths", "intrinsics", "extrinsics", "video")
EMPTY_POINTS = np.empty((0, 3), np.float32)
EMPTY_COLORS = np.empty((0, 3), np.uint8)


def run_label(path):
    method, _, _, mode, _ = path.parts[-5:]
    return f"{method} / {mode}"


def discover(root):
    """Read archive directories only. Array data is loaded for one clip at a time."""
    clips = {}
    skipped = 0
    for path in sorted(root.glob("*/results/*/*/tracks.npz")):
        clip = path.parent.parent.name
        archives = [(path, TRACK_KEYS)]
        result = path.with_name("result.npz")
        if result.exists():
            archives.append((result, GEOMETRY_KEYS))
        for archive, required in archives:
            with np.load(archive) as data:
                missing = sorted(set(required).difference(data.files))
            if missing:
                print(f"Warning: skipping {clip}, {run_label(path)}: "
                      f"{archive.name} is missing {', '.join(missing)}", file=sys.stderr)
                skipped += 1
                break
        else:
            clips.setdefault(clip, []).append(path)
    return dict(sorted(clips.items())), skipped


def read_arrays(path, keys):
    with np.load(path) as data:
        return {key: data[key] for key in keys}


def nearest_frame(times, timestamp):
    """Return the closest sample, using the earlier sample for a tie."""
    right = int(np.searchsorted(times, timestamp))
    if right == 0:
        return 0
    if right == len(times) or timestamp - times[right - 1] <= times[right] - timestamp:
        return right - 1
    return right


def query_colors(queries, size, orig_size):
    original = (queries[:, 1:].astype(np.float64) + 0.5) * orig_size / size - 0.5
    xy = np.clip(original / np.maximum(orig_size - 1, 1), 0, 1)
    rgb = [colorsys.hsv_to_rgb(0.88 * x, 0.85, 1.0 - 0.35 * y) for x, y in xy]
    return np.rint(np.asarray(rgb).reshape(-1, 3) * 255).astype(np.uint8)


@dataclass
class Geometry:
    depths: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    video: np.ndarray

    def __post_init__(self):
        n, h, w = self.depths.shape
        if (self.video.shape != (n, 3, h, w) or self.intrinsics.shape != (n, 3, 3)
                or self.extrinsics.shape != (n, 4, 4)):
            raise ValueError("Inconsistent depth, video, or camera shapes in result.npz")
        self.inv_k = np.linalg.inv(self.intrinsics.astype(np.float32))
        self.c2w = np.linalg.inv(self.extrinsics.astype(np.float32))

    def bounding_corners(self):
        """Bound every cloud without unprojecting each pixel or scanning during playback."""
        _, h, w = self.depths.shape
        corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]],
                           dtype=np.float32)
        far = np.max(self.depths, axis=(1, 2), where=np.isfinite(self.depths), initial=0)
        rays = corners @ self.inv_k.transpose(0, 2, 1)
        origins = self.c2w[:, :3, 3]
        extent = (rays * far[:, None, None]) @ self.c2w[:, :3, :3].transpose(0, 2, 1)
        return np.concatenate([origins, (extent + origins[:, None]).reshape(-1, 3)])


@dataclass
class Run:
    method: str
    setting: str
    queries: np.ndarray
    tracks3d_world: np.ndarray
    vis: np.ndarray
    tracks2d_orig: np.ndarray
    c2w: np.ndarray
    intrinsics: np.ndarray
    frame_idx: np.ndarray
    timestamps: np.ndarray
    fps: float
    size: np.ndarray
    orig_size: np.ndarray
    geometry: Geometry | None
    non_metric: bool = False
    lidar_scale: float | None = None

    @property
    def label(self):
        return f"{self.method} / {self.setting}"

    def __post_init__(self):
        nframes, ntracks = self.vis.shape
        if not nframes or (self.frame_idx.shape, self.timestamps.shape) != ((nframes,), (nframes,)):
            raise ValueError(f"{self.label}: expected one frame number and timestamp per frame")
        if not np.all(np.isfinite(self.timestamps)) or np.any(np.diff(self.timestamps) < 0):
            raise ValueError(f"{self.label}: timestamps must be finite and sorted")
        self.fps = float(self.fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"{self.label}: expected a positive fps")
        if (self.tracks3d_world.shape != (nframes, ntracks, 3)
                or self.tracks2d_orig.shape != (nframes, ntracks, 2)
                or self.queries.shape != (ntracks, 3)
                or self.c2w.shape != (nframes, 4, 4) or self.intrinsics.shape != (nframes, 3, 3)
                or (self.geometry is not None and len(self.geometry.depths) != nframes)):
            raise ValueError(f"{self.label}: inconsistent track or camera shapes")
        times = self.queries[:, 0]
        if not np.all((times >= 0) & (times < nframes) & (times == np.floor(times))):
            raise ValueError(f"{self.label}: invalid query frame")
        self.query_frames = times.astype(np.intp)
        active = (self.vis > 0.5) & (np.arange(nframes)[:, None] >= self.query_frames)
        self.valid3d = active & np.isfinite(self.tracks3d_world).all(axis=-1)
        self.valid2d = active & np.isfinite(self.tracks2d_orig).all(axis=-1)
        track_ids = np.arange(ntracks)
        anchor = self.tracks2d_orig[self.query_frames, track_ids]
        valid = self.valid2d & self.valid2d[self.query_frames, track_ids]
        # Measure trajectory motion from its query-frame sample in original pixels.
        offset = np.subtract(self.tracks2d_orig, anchor, where=valid[..., None],
                             out=np.zeros_like(self.tracks2d_orig))
        self.max_motion = np.hypot(offset[..., 0], offset[..., 1]).max(axis=0)
        self.colors = query_colors(self.queries, self.size, self.orig_size)
        self.palette = list(map(tuple, self.colors.tolist()))
        tracks = self.tracks3d_world[self.valid3d]
        pose = self.c2w[0]
        self.center = np.median(tracks, axis=0) if tracks.size else pose[:3, 3] + pose[:3, 2]
        samples = [self.c2w[:, :3, 3], tracks]
        if self.geometry is not None:
            samples.append(self.geometry.bounding_corners())
        points = np.concatenate(samples)
        self.lower, self.upper = points.min(axis=0), points.max(axis=0)

    @classmethod
    def load(cls, path):
        log_path = path.with_name("log.json")
        log = json.loads(log_path.read_text()) if log_path.exists() else {}
        setting = log.get("mode", path.parent.name)
        if setting != path.parent.name:
            raise ValueError(f"{log_path}: mode {setting!r} disagrees with the run directory")
        settings = log.get("settings", {})
        metric = log.get("metric", settings.get("metric", settings.get("metric_scale")))
        result = path.with_name("result.npz")
        geometry = Geometry(**read_arrays(result, GEOMETRY_KEYS)) if result.exists() else None
        run = cls(path.parents[3].name, setting, geometry=geometry, non_metric=metric is False,
                  **read_arrays(path, TRACK_KEYS))
        if run.non_metric:
            clip_path = path.parents[4] / "preprocessed_data" / path.parents[1].name / "clip.npz"
            with np.load(path) as data:
                ids = np.arange(len(run.queries))
                if "track_depth" in data:
                    depth = data["track_depth"][run.query_frames, ids]
                else:
                    pose = run.c2w[run.query_frames]
                    relative = run.tracks3d_world[run.query_frames, ids] - pose[:, :3, 3]
                    depth = (relative * pose[:, :3, 2]).sum(axis=1)
            run.lidar_scale = run.query_lidar_scale(clip_path, depth)
        return run

    def query_lidar_scale(self, clip_path, depth):
        """Median LiDAR z / track z, sampled at visible queries in their source frames."""
        pixels = np.floor(self.queries[:, 1:] + 0.5)
        keep = (self.valid3d[self.query_frames, np.arange(len(depth))]
                & np.isfinite(depth) & (depth > 0)
                & ((pixels >= 0) & (pixels < self.size)).all(axis=1))
        if not keep.any():
            return None
        numbers = self.frame_idx[self.query_frames[keep]]
        with np.load(clip_path) as data:
            stored = data["frame_idx"]
            indices = np.searchsorted(stored, numbers)
            if np.any(indices >= len(stored)) or not np.array_equal(stored[indices], numbers):
                raise ValueError(f"{clip_path}: missing LiDAR query frame")
            lidar = data["depth_mm"]
            # Index native depth exactly as nearest_resize(depth, *self.size) would.
            _, height, width = lidar.shape
            x, y = ((pixels[keep] + 0.5) * (width, height) / self.size).astype(np.intp).T
            meters = lidar[indices, y, x].astype(np.float64) / 1000
        valid = np.isfinite(meters) & (meters > 0)
        if not valid.any():
            return None
        return float(np.median(meters[valid] / depth[keep][valid]))

    def scale_description(self, enabled):
        if self.lidar_scale is not None:
            if enabled:
                return f" (scaled x{self.lidar_scale:.4f})"
            return f" (LiDAR x{self.lidar_scale:.4f}, off)"
        return " (LiDAR scale unavailable)" if self.non_metric else ""


def load_images(root, clip, frame_ids):
    """Decode each requested original RGB frame once, shared across all runs."""
    from track3d_data import load_frames

    clip_dir = root / "preprocessed_data" / clip
    with np.load(clip_dir / "clip.npz") as data:
        width, height = map(int, data["frame_size"])
    scale = min(1.0, 640 / width, 480 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    numbers = np.unique(np.concatenate(frame_ids))
    return dict(zip(numbers, load_frames(clip_dir, numbers, size=size), strict=True))


def point_cloud(geometry, t, stride):
    if geometry is None:
        return EMPTY_POINTS, EMPTY_COLORS
    depth = geometry.depths[t, ::stride, ::stride]
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    pixels = np.column_stack((x * stride, y * stride, np.ones_like(x))).astype(np.float32)
    camera = (pixels @ geometry.inv_k[t].T) * depth[y, x, None]
    pose = geometry.c2w[t]
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    rgb = geometry.video[t].transpose(1, 2, 0)[y * stride, x * stride].astype(np.float32)
    return world, np.rint(255 * np.clip(rgb, 0, 1)).astype(np.uint8)


def track_points(run, t, selected=True):
    mask = run.valid3d[t] & selected
    return run.tracks3d_world[t, mask], run.colors[mask]


def trail_edges(positions, valid, t, length, selected=True):
    """Return adjacent visible endpoint pairs and their track IDs in a K-frame window."""
    start = max(0, t + 1 - length)
    frame, track = np.nonzero(valid[start:t] & valid[start + 1:t + 1] & selected)
    frame += start
    return np.stack((positions[frame, track], positions[frame + 1, track]), axis=1), track


def trail_segments(run, t, length, selected=True):
    points, track = trail_edges(run.tracks3d_world, run.valid3d, t, length, selected)
    return points, np.repeat(run.colors[track, None], 2, axis=1)


def overlay_image(rgb, run, t, length, dots=True, trails=True, selected=True):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    scale = np.asarray(image.size) / run.orig_size
    if trails:
        edges, ids = trail_edges(run.tracks2d_orig, run.valid2d, t, length, selected)
        lines = ((edges + 0.5) * scale - 0.5).reshape(-1, 4)
        for line, i in zip(lines.tolist(), ids):
            draw.line(line, fill=run.palette[i], width=1)
    if dots:
        ids = np.flatnonzero(run.valid2d[t] & selected)
        xy = (run.tracks2d_orig[t, ids] + 0.5) * scale - 0.5
        for (x, y), i in zip(xy.tolist(), ids):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=run.palette[i])
    return np.asarray(image)


def check_run(run, images):
    sizes, counts, samples = [], [], []
    for t in (0, len(run.timestamps) - 1):
        cloud, rgb = point_cloud(run.geometry, t, 2)
        points, colors = track_points(run, t)
        lines, line_colors = trail_segments(run, t, 10)
        frame = images[run.frame_idx[t]]
        overlay = overlay_image(frame, run, t, 10)
        for positions, shades in ((cloud, rgb), (points, colors), (lines, line_colors)):
            if positions.shape != shades.shape or not np.isfinite(positions).all():
                raise ValueError(f"{run.label}: invalid geometry at frame {t}")
            samples.append(positions.reshape(-1, 3))
        if overlay.shape != frame.shape or overlay.dtype != np.uint8:
            raise ValueError(f"{run.label}: invalid RGB overlay at frame {t}")
        sizes.append(len(cloud))
        counts.append(len(points))
    xyz = np.concatenate(samples)
    if run.lidar_scale is not None:
        xyz *= run.lidar_scale
    box = "empty"
    if len(xyz):
        lo, hi = xyz.min(axis=0).round(3), xyz.max(axis=0).round(3)
        box = f"{lo} to {hi}"
    moving = np.count_nonzero(run.max_motion > DEFAULT_MOTION_PX)
    scale = "unavailable" if run.non_metric else "none"
    if run.lidar_scale is not None:
        scale = f"{run.lidar_scale:.4f}"
    unit = "model units" if run.non_metric and run.lidar_scale is None else "m"
    return (f"T={len(run.timestamps)} N={len(run.queries)} cloud={sizes[0]}/{sizes[1]} "
            f"visible={counts[0]}/{counts[1]} moving(>{DEFAULT_MOTION_PX}px)={moving} "
            f"scale={scale} bbox(first+last, {unit})={box}")


def check_all(root, clips):
    failures = 0
    for clip, paths in clips.items():
        for path in paths:
            try:
                run = Run.load(path)
                images = load_images(root, clip, [run.frame_idx[[0, -1]]])
                print(f"{clip} | {run.label} | {check_run(run, images)}", flush=True)
                del run, images
            except (OSError, KeyError, ValueError, np.linalg.LinAlgError) as exc:
                print(f"Failed: {clip} | {run_label(path)} | {exc}", file=sys.stderr)
                failures += 1
    return failures


def grid_axis(names, limits):
    """Pack columns or rows using every run, reserving space for empty cells."""
    spans = [high - low for _, low, high in limits]
    typical = max(0.1, float(np.median(spans)))
    gap = 0.2 * typical
    cursor = 0.0
    cells = {}
    for name in names:
        lower = min((low for key, low, _ in limits if key == name), default=0.0)
        upper = max((high for key, _, high in limits if key == name), default=typical)
        cells[name] = (cursor - lower, lower, upper)
        cursor += upper - lower + gap
    return cells


class RunNodes:
    """Persistent scene nodes in one translated coordinate frame."""

    def __init__(self, scene, name, run):
        self.run = run
        self.scale = 1.0
        self.cloud_at = lru_cache(maxsize=4)(partial(point_cloud, run.geometry))
        self.cloud_frame = None
        self.group = scene.add_frame(name, show_axes=False)
        self.label = scene.add_label(f"{name}/label", run.label, anchor="bottom-center")
        self.cloud = scene.add_point_cloud(f"{name}/cloud", EMPTY_POINTS, EMPTY_COLORS)
        self.tracks = scene.add_point_cloud(
            f"{name}/tracks", EMPTY_POINTS, EMPTY_COLORS, point_shape="circle",
        )
        self.trails = scene.add_line_segments(
            f"{name}/trails", EMPTY_POINTS.reshape(0, 2, 3), EMPTY_COLORS.reshape(0, 2, 3),
            thickness=1.5, thickness_units="screen",
        )
        self.camera = scene.add_camera_frustum(
            f"{name}/camera", fov=1.0, aspect=float(run.size[0] / run.size[1]),
            scale=0.1, thickness=1, thickness_units="screen", format="jpeg",
        )
        self.camera_frame = None

    def update(self, t, rgb, ui, selected):
        from viser.transforms import SO3

        run = self.run
        self.cloud.visible = ui.cloud.value and run.geometry is not None
        self.tracks.visible = ui.tracks.value
        self.trails.visible = ui.trails.value
        self.camera.visible = ui.frustum.value
        self.cloud.point_size = ui.point_size.value
        self.tracks.point_size = ui.track_size.value
        cloud_frame = (t, ui.cloud_stride)
        if self.cloud.visible and self.cloud_frame != cloud_frame:
            self.cloud.points, self.cloud.colors = self.cloud_at(*cloud_frame)
            self.cloud_frame = cloud_frame
        if self.tracks.visible:
            self.tracks.points, self.tracks.colors = track_points(run, t, selected)
        if self.trails.visible:
            self.trails.points, self.trails.colors = trail_segments(run, t, ui.length.value, selected)
        if self.camera.visible:
            self.camera.position = run.c2w[t, :3, 3] * self.scale
        if self.camera.visible and self.camera_frame != t:
            pose = run.c2w[t]
            self.camera.wxyz = SO3.from_matrix(pose[:3, :3]).wxyz
            self.camera.fov = float(2 * np.arctan(run.size[1] / (2 * run.intrinsics[t, 1, 1])))
            self.camera.image = rgb[::2, ::2]
            self.camera_frame = t


class Viewer:
    def __init__(self, server: viser.ViserServer, root, clips, initial_clip):
        self.server, self.root, self.clips = server, root, clips
        self.lock = threading.Condition()
        self.playing = self.closed = False
        self.loaded_clip = self.reference = None
        self.nodes, self.checkboxes = {}, {}
        server.scene.set_up_direction("-y")
        gui = server.gui
        gui.main_panel.dock_left()
        gui.main_panel.set_width(360)
        self.clip = gui.add_dropdown("Clip", tuple(clips), initial_value=initial_clip)
        self.image = gui.add_image(np.zeros((1, 1, 3), np.uint8), label="RGB", format="jpeg")
        self.frame = gui.add_slider("Frame", 0, 1, 1, 0)
        self.play = gui.add_button("Play")
        self.fps = gui.add_number("Playback fps", 10.0, min=0.001, step=0.001)
        self.loop = gui.add_checkbox("Loop", True)
        self.overlay = gui.add_dropdown("2D overlay run", ("None",))
        self.readout = gui.add_markdown("")
        paths = [path for clip_paths in clips.values() for path in clip_paths]
        # Keep the comparison grid fixed and append any newly discovered groups.
        methods = (*METHODS, *sorted({path.parents[3].name for path in paths} - set(METHODS)))
        settings = (*SETTINGS, *sorted({path.parent.name for path in paths} - set(SETTINGS)))
        with gui.add_folder("Methods"):
            self.methods = {name: gui.add_checkbox(name, True) for name in methods}
        with gui.add_folder("Settings"):
            self.settings = {name: gui.add_checkbox(name, True) for name in settings}
        self.run_folder = gui.add_folder("Runs")
        with gui.add_folder("Scene"):
            self.layout = gui.add_dropdown("Layout", ("grid", "overlay"))
            self.align_scale = gui.add_checkbox("Scale non-metric runs to LiDAR", True)
            self.only_moving = gui.add_checkbox("Only moving tracks", False)
            self.threshold = gui.add_slider("Motion threshold (original px)", 0, 2400, 1,
                                            DEFAULT_MOTION_PX)
            self.cloud = gui.add_checkbox("Point cloud", True)
            self.tracks = gui.add_checkbox("Tracks", True)
            self.trails = gui.add_checkbox("Trails", True)
            self.frustum = gui.add_checkbox("Camera frustum", True)
            self.length = gui.add_slider("Trail length (frames)", 1, 60, 1, 10)
            self.cloud_stride = 2
            self.manual_stride = False
            self.stride = gui.add_slider("Point cloud stride", 1, 4, 1, self.cloud_stride,
                                         marks=((1, "1 px"), (2, "2 px"), (4, "4 px")))
            self.point_size = gui.add_slider("Point size (m)", 0.001, 0.04, 0.001, 0.006)
            self.track_size = gui.add_slider("Track point size (m)", 0.001, 0.05, 0.001, 0.012)
        with self.lock:
            self.load_clip(initial_clip)
        for handle in (self.clip, self.frame, self.overlay, self.layout, self.cloud, self.tracks,
                       self.trails, self.frustum, self.length, self.stride, self.point_size,
                       self.track_size, self.align_scale, self.only_moving, self.threshold,
                       *self.methods.values(), *self.settings.values()):
            handle.on_update(self.on_change)
        self.play.on_click(self.on_play)
        self.fps.on_update(self.on_fps)
        server.on_client_connect(self.place_camera)
        self.worker = threading.Thread(target=self.playback, name="track-playback", daemon=True)
        self.worker.start()

    def load_clip(self, clip):
        runs = [Run.load(path) for path in self.clips[clip]]
        methods, settings = list(self.methods), list(self.settings)
        runs.sort(key=lambda r: (settings.index(r.setting), methods.index(r.method)))
        images = load_images(self.root, clip, [run.frame_idx for run in runs])
        with self.server.atomic():
            self.set_playing(False)
            for node in self.nodes.values():
                node.group.remove()
            for checkbox in self.checkboxes.values():
                checkbox.remove()
            self.nodes.clear()
            self.checkboxes.clear()
            self.reference = None
            self.loaded_clip, self.images = clip, images
            with self.run_folder:
                for i, run in enumerate(runs):
                    checkbox = self.server.gui.add_checkbox(run.label, True)
                    checkbox.on_update(self.on_change)
                    self.checkboxes[run.label] = checkbox
                    self.nodes[run.label] = RunNodes(self.server.scene, f"/run_{i}", run)
            self.overlay.options = tuple(self.nodes)
            self.overlay.value = runs[0].label
            diagonal = max(np.linalg.norm(r.orig_size) for r in runs)
            self.threshold.max = max(DEFAULT_MOTION_PX, int(np.ceil(diagonal)))
            self.threshold.value = min(self.threshold.value, self.threshold.max)
            self.update_layout()
            self.refresh()
        for client in self.server.get_clients().values():
            self.place_camera(client)

    def on_change(self, event):
        # Programmatic GUI edits call callbacks synchronously. Their caller renders once.
        if event.client is None:
            return
        with self.lock:
            if self.loaded_clip != self.clip.value:
                try:
                    self.load_clip(self.clip.value)
                except (OSError, ValueError, KeyError) as exc:
                    print(f"Warning: cannot load {self.clip.value}: {exc}", file=sys.stderr)
                    self.clip.value = self.loaded_clip
                    self.readout.content = f"Could not load clip: {exc}"
            else:
                reframe = event.target is self.layout or event.target is self.align_scale
                with self.server.atomic():
                    if reframe:
                        self.update_layout()
                    if event.target is self.stride:
                        self.manual_stride = True
                    self.refresh()
                if reframe:
                    for client in self.server.get_clients().values():
                        self.place_camera(client)

    def update_layout(self):
        x_limits, y_limits = [], []
        align = self.align_scale.value
        for node in self.nodes.values():
            run = node.run
            node.scale = run.lidar_scale if align and run.lidar_scale is not None else 1.0
            # Viser frame scale affects its axes only. Scale each geometry handle.
            node.cloud.scale = node.tracks.scale = node.trails.scale = node.scale
            lower, upper = run.lower * node.scale, run.upper * node.scale
            x_limits.append((run.method, lower[0], upper[0]))
            y_limits.append((run.setting, lower[1], upper[1]))
        columns = grid_axis(self.methods, x_limits)
        rows = grid_axis(self.settings, y_limits)
        grid = self.layout.value == "grid"
        for node in self.nodes.values():
            run = node.run
            x, left, right = columns[run.method]
            y, top, _ = rows[run.setting]
            node.group.position = (x, y, 0) if grid else (0, 0, 0)
            node.label.visible = grid
            node.label.position = ((left + right) / 2, top - 0.05, run.center[2] * node.scale)
            node.label.text = run.label + (run.scale_description(True) if align else "")

    def sync_controls(self, visible):
        # Keep the video available even when the user hides every 3D run.
        ref = self.nodes[visible[0] if visible else next(iter(self.nodes))].run
        t = min(max(0, self.frame.value), len(ref.timestamps) - 1)
        if ref is not self.reference:
            t = 0
            if self.reference is not None:
                old_times = self.reference.timestamps
                old_t = min(max(0, self.frame.value), len(old_times) - 1)
                t = nearest_frame(ref.timestamps, old_times[old_t])
            self.reference = ref
            self.frame.max = len(ref.timestamps) - 1
            self.frame.disabled = len(ref.timestamps) == 1
            self.fps.value = ref.fps
        self.frame.value = t
        options = tuple(visible) or ("None",)
        if self.overlay.options != options:
            self.overlay.options = options
        self.overlay.disabled = not visible
        # The viser slider has a linear step. Snap past 3 in the drag direction.
        stride = self.stride.value if self.manual_stride else (4 if len(visible) > 4 else 2)
        if stride == 3:
            stride = 2 if self.cloud_stride == 4 else 4
        self.cloud_stride = self.stride.value = stride
        return ref, t

    def refresh(self):
        with self.server.atomic():
            visible = [
                key for key, node in self.nodes.items() if self.checkboxes[key].value
                and self.methods[node.run.method].value and self.settings[node.run.setting].value
            ]
            ref, t = self.sync_controls(visible)
            timestamp = ref.timestamps[t]
            rgb = self.images[ref.frame_idx[t]]
            lines = [f"Frame **{ref.frame_idx[t]}** at **{timestamp:.3f} s** ({ref.label})"]
            for key, node in self.nodes.items():
                run = node.run
                i = t if run is ref else nearest_frame(run.timestamps, timestamp)
                count = np.count_nonzero(run.valid3d[i])
                moving = run.max_motion > self.threshold.value
                selected = moving if self.only_moving.value else True
                node.group.visible = key in visible
                suffix = "" if node.group.visible else " (hidden)"
                shown = ""
                if self.only_moving.value:
                    shown = f", {np.count_nonzero(run.valid3d[i] & moving)} after filter"
                lines.append(f"{key}{run.scale_description(self.align_scale.value)}: {count} visible, "
                             f"{np.count_nonzero(moving)} moving{shown}{suffix}")
                if not node.group.visible:
                    continue
                frame = self.images[run.frame_idx[i]]
                node.update(i, frame, self, selected)
                if key == self.overlay.value:
                    rgb = overlay_image(frame, run, i, self.length.value,
                                        self.tracks.value, self.trails.value, selected)
                    if run.frame_idx[i] != ref.frame_idx[t]:
                        lines.append(f"RGB overlay: frame {run.frame_idx[i]} at {run.timestamps[i]:.3f} s")
            self.image.image = rgb
            self.readout.content = "  \n".join(lines)

    def place_camera(self, client):
        with self.lock, client.atomic():
            nodes = [node for node in self.nodes.values() if node.group.visible]
            if not nodes:
                return
            low = np.min([n.group.position + n.scale * n.run.lower for n in nodes], axis=0)
            high = np.max([n.group.position + n.scale * n.run.upper for n in nodes], axis=0)
            if self.layout.value == "grid":
                low[1] = min(low[1], *(n.group.position[1] + n.label.position[1] for n in nodes))
            center = (low + high) / 2
            width, height = high[:2] - low[:2]
            span = max(height, width / max(client.camera.aspect, 0.1))
            distance = max(0.25, 0.6 * span / np.tan(client.camera.fov / 2))
            client.camera.position = (center[0], center[1], low[2] - distance)
            client.camera.look_at = center
            client.camera.up_direction = (0.0, -1.0, 0.0)

    def set_playing(self, playing):
        self.playing = playing
        self.play.label = "Pause" if playing else "Play"
        self.lock.notify_all()

    def on_play(self, _event):
        with self.lock:
            if not self.playing and self.frame.value == self.frame.max:
                self.frame.value = 0
                self.refresh()
            self.set_playing(not self.playing)

    def on_fps(self, event):
        if event.client is not None:
            with self.lock:
                self.fps.value = max(0.001, self.fps.value)
                self.lock.notify_all()

    def playback(self):
        with self.lock:
            while not self.closed:
                self.lock.wait_for(lambda: self.playing or self.closed)
                if self.closed:
                    break
                started = time.monotonic()
                if self.frame.value == self.frame.max and not self.loop.value:
                    self.set_playing(False)
                    continue
                self.frame.value = (self.frame.value + 1) % (self.frame.max + 1)
                self.refresh()
                period = 1 / max(0.001, self.fps.value)
                self.lock.wait(max(0.001, started + period - time.monotonic()))

    def close(self):
        with self.lock:
            self.closed = True
            self.lock.notify_all()
        self.worker.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Track3d project root")
    parser.add_argument("--clip", help="Initial clip (default: first available)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--check", action="store_true", help="Check all run endpoints offline")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "common"), str(PROJECT_ROOT / "common")]
    clips, skipped = discover(root)
    if not clips:
        parser.exit(1, f"No usable runs found under {root}\n")
    if args.check:
        return int(bool(check_all(root, clips) or skipped))
    if args.clip is not None and args.clip not in clips:
        parser.error(f"unknown clip {args.clip!r}; available clips: {', '.join(clips)}")

    import viser

    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="3D track comparison")
    try:
        viewer = Viewer(server, root, clips, args.clip or next(iter(clips)))
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close()
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
