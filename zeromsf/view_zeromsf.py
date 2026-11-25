"""Local viewer for ZeroMSF pairs, independently following visualize_viser.py.

Positive-depth pixels of image 1 keep their RGB while moving along the full
predicted displacement. Playback loops at 10 fps, initially paused at step 0.
This version switches clips and pairs in place, shows both input PNGs and the
recorded gap, and starts just behind the recording camera with image y down.
It updates one cloud in place and reads only the selected pair's arrays.
Optional full-displacement lines on a pixel grid and an absolute motion filter
both start off. Line color goes from blue to red, saturating at the pair's
95th percentile magnitude. Thresholds and point sizes are in model units.
The CLI also supports offline result checks and binds to loopback by default.
"""
import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np

from run_zeromsf import HERE, read_pair, verify_pair


def vector_segments(points, flow, magnitude, mask, stride, color_limit):
    """Sample in image coordinates so holes never shift the vector grid."""
    grid = np.s_[::stride, ::stride]
    valid = mask[grid]
    start, delta = points[grid][valid], flow[grid][valid]
    strength = np.clip(magnitude[grid][valid] / max(color_limit, 1e-12), 0, 1)
    rgb = np.stack((strength, 1 - np.abs(2 * strength - 1), 1 - strength), axis=-1)
    colors = np.rint(rgb * 255).astype(np.uint8)
    return np.stack((start, start + delta), axis=1), np.repeat(colors[:, None, :], 2, axis=1)


def check_run(folder, summary_row):
    log = verify_pair(folder)
    if log != summary_row:
        raise ValueError("summary.json is missing this log or differs from it")
    note = " | " + " ".join(log["warnings"]) if log["warnings"] else ""
    return f"{log['stats']['valid_pixels']} points, gap {log['gap_s']:.3f} s{note}"


def check_all(root, folders):
    failed = False
    index = {}
    try:
        rows = json.loads((root / "summary.json").read_text())
        if not isinstance(rows, list):
            raise ValueError("Expected a list of pair logs")
        for row in rows:
            key = (row["clip"], row["pair"])
            if key in index:
                raise ValueError(f"Duplicate summary entry {key}")
            index[key] = row
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR summary.json: {exc}")
        failed = True
    paths = {(p.parent.name, p.name): p for p in folders}
    if not (paths or index):
        print(f"No pair results found under {root}")
        return 1
    for clip, pair in sorted(paths.keys() | index.keys()):
        try:
            if (clip, pair) not in paths:
                raise ValueError("Missing pair directory")
            detail = check_run(paths[clip, pair], index.get((clip, pair)))
            print(f"OK {clip}/{pair}: {detail}")
        except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, IndexError) as exc:
            print(f"ERROR {clip}/{pair}: {exc}")
            failed = True
    return int(failed)


class Viewer:
    def __init__(self, server, pairs, clip, steps):
        self.server, self.pairs, self.steps = server, pairs, steps
        self.lock = threading.RLock()
        self.playing = False
        server.scene.set_up_direction("-y")
        gui = server.gui
        self.clip = gui.add_dropdown("Clip", tuple(pairs), initial_value=clip)
        self.pair = gui.add_dropdown("Pair", tuple(pairs[clip]))
        self.message = gui.add_markdown("")
        blank = np.zeros((1, 1, 3), np.uint8)
        self.images = [gui.add_image(blank, label=f"Image {i}", format="png") for i in (1, 2)]
        with gui.add_folder("Animation"):
            self.step = gui.add_slider("Step", 0, steps, 1, 0)
            play = gui.add_button("Play")
            pause = gui.add_button("Pause")
            reset = gui.add_button("Reset")
            self.size = gui.add_slider("Point size", 0.001, 0.1, 0.001, 0.008)
        with gui.add_folder("Motion"):
            self.only_moving = gui.add_checkbox("Moving only", False)
            self.threshold = gui.add_slider("Min displacement (model units)", 0.0, 1.0, 0.001, 0.01)
            self.vectors = gui.add_checkbox("Flow vectors", False)
            self.stride = gui.add_slider("Vector stride (pixels)", 2, 64, 1, 16)
        self.cloud = server.scene.add_point_cloud(
            "/image1", np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8), point_size=self.size.value)
        self.lines = server.scene.add_line_segments(
            "/flow", np.empty((0, 2, 3), np.float32), np.empty((0, 2, 3), np.uint8),
            thickness=1.5, thickness_units="screen", visible=False)
        self.load(clip, self.pair.value)
        for handle in (self.clip, self.pair, self.step, self.size, self.only_moving,
                       self.threshold, self.vectors, self.stride):
            handle.on_update(self.change)
        play.on_click(lambda _: self.play(True))
        pause.on_click(lambda _: self.play(False))
        reset.on_click(lambda _: self.reset())
        server.on_client_connect(self.place_camera)

    def load(self, clip, pair):
        log, self.points, self.flow, colors, images = read_pair(self.pairs[clip][pair])
        self.loaded = (clip, pair)
        self.colors = np.rint(255 * colors).astype(np.uint8)
        self.valid = self.points[..., 2] > 0
        self.magnitude = np.linalg.norm(self.flow, axis=-1)
        magnitudes = self.magnitude[self.valid]
        self.color_limit = float(np.percentile(magnitudes, 95))
        self.bounds = np.percentile(self.points[self.valid], [2, 98], axis=0)
        self.playing = False
        with self.server.atomic():
            self.pair.options = tuple(self.pairs[clip])
            self.pair.value = pair
            self.step.value = 0
            self.threshold.max = max(0.01, float(magnitudes.max()))
            self.threshold.value = min(self.threshold.value, self.threshold.max)
            self.message.content = f"Recorded gap **{log['gap_s']:.3f} s**"
            for i, (panel, pixels, frame) in enumerate(zip(self.images, images, log["frame_numbers"]), 1):
                panel.label = f"Image {i} - frame {frame}"
                panel.image = pixels
            self.select_points()
        for client in self.server.get_clients().values():
            self.place_camera(client)

    def place_camera(self, client):
        with self.lock, client.atomic():
            low, high = self.bounds
            center = (low + high) / 2
            span = max(high[1] - low[1], (high[0] - low[0]) / max(client.camera.aspect, 0.1))
            distance = 0.6 * span / np.tan(client.camera.fov / 2)
            client.camera.position = (0.0, 0.0, min(-0.05 * center[2], low[2] - distance))
            client.camera.look_at = center
            client.camera.up_direction = (0.0, -1.0, 0.0)

    def select_points(self):
        self.mask = self.valid
        if self.only_moving.value:
            self.mask = self.valid & (self.magnitude > self.threshold.value)
        self.start, self.delta = self.points[self.mask], self.flow[self.mask]
        self.cloud.colors = self.colors[self.mask]
        self.redraw()
        self.draw_vectors()

    def draw_vectors(self):
        self.lines.visible = self.vectors.value
        if self.vectors.value:
            self.lines.points, self.lines.colors = vector_segments(
                self.points, self.flow, self.magnitude, self.mask, self.stride.value, self.color_limit)

    def redraw(self):
        self.cloud.points = self.start + (self.step.value / self.steps) * self.delta

    def change(self, event):
        if event.client is None:  # Programmatic edits are rendered by their caller.
            return
        with self.lock, self.server.atomic():
            if event.target in (self.clip, self.pair):
                clip = self.clip.value
                pair = self.pair.value if self.pair.value in self.pairs[clip] else next(iter(self.pairs[clip]))
                try:
                    self.load(clip, pair)
                except (ValueError, KeyError, TypeError, OSError, EOFError) as exc:
                    self.clip.value, self.pair.value = self.loaded
                    self.message.content = f"Could not open {clip}/{pair}: {exc}"
            elif event.target is self.step:
                self.redraw()
            elif event.target is self.size:
                self.cloud.point_size = self.size.value
            elif event.target in (self.only_moving, self.threshold):
                self.select_points()
            else:
                self.draw_vectors()

    def play(self, playing):
        with self.lock:
            self.playing = playing

    def reset(self):
        with self.lock, self.server.atomic():
            self.playing = False
            self.step.value = 0
            self.redraw()

    def tick(self):
        with self.lock:
            if self.playing:
                with self.server.atomic():
                    self.step.value = (self.step.value + 1) % (self.steps + 1)
                    self.redraw()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", help="Initial clip")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--check", action="store_true", help="Validate all saved pairs and exit")
    parser.add_argument("--results-root", type=Path, default=HERE / "results", help=argparse.SUPPRESS)
    args = parser.parse_args()
    folders = sorted(p for p in args.results_root.glob("*/*") if p.is_dir())
    if args.check:
        return check_all(args.results_root, folders)
    pairs = {}
    for folder in folders:
        try:
            log = json.loads((folder / "log.json").read_text())
            if log["status"] == "ok":
                pairs.setdefault(folder.parent.name, {})[folder.name] = folder
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Skipping {folder}: {exc}")
    if not pairs:
        parser.exit(1, f"No completed pairs under {args.results_root}\n")
    if args.clip is not None and args.clip not in pairs:
        parser.error(f"Available clips: {', '.join(pairs)}")
    if args.steps < 1:
        parser.error("--steps must be positive")

    import viser

    server = viser.ViserServer(host=args.host, port=args.port, label="ZeroMSF scene flow")
    try:
        viewer = Viewer(server, pairs, args.clip or next(iter(pairs)), args.steps)
        print(f"Ready: {viewer.loaded[0]}/{viewer.loaded[1]}", flush=True)
        while True:
            time.sleep(0.1)
            viewer.tick()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
