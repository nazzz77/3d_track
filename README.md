# 3D point tracking baselines on iPhone manipulation videos

Five 3D point trackers run on six tabletop manipulation clips recorded with an iPhone (Stray Scanner:
RGB, LiDAR depth, intrinsics; fixed tripod), with identical frames and query points, plus a viser
viewer that shows all of them side by side.

| Method | Settings | Folder |
| --- | --- | --- |
| SpatialTrackerV2 | `rgb`, `rgbd_fixed` | `spatracker_v2/` |
| DELTA (sparse 3D) | `rgb` (UniDepth V2 depth), `rgbd_fixed` | `delta/` |
| TrackCraft3R | `rgb` (Depth Anything 3 depth + camera), `rgbd_fixed` | `trackcraft3r/` |
| Track4World | `rgb` | `track4world/` |
| TraceAnything | `rgb` | `traceanything/` |

`rgb` = video only; `rgbd_fixed` = iPhone LiDAR depth and intrinsics with the camera held fixed (only
for methods that accept external depth). Clips: `pour_lid`, `pen`, `stir`, `shoelace`, `paper_fold`,
`shirt_fold`. All methods see the same frames (10 fps, at most 200 frames per clip) and track the same
1,152 query points (a 32x24 grid on frame 0 plus 16x12 grids at 1/3 and 2/3 of the clip).

## Browse the results (no GPU needed)

1. Clone this repository.
2. Download `track3d_data.zip` (about 4.4 GB) from the Google Drive link shared with you, and unzip it
   in the repository root. It adds `preprocessed_data/` and `<method>/results/` next to the code. On a
   machine without a browser:

   ```bash
   pip install gdown && gdown --fuzzy "<Google Drive link>"   # saves track3d_data.zip
   unzip track3d_data.zip
   ```
3. Start the viewer and open http://localhost:8080:

   ```bash
   bash viewer/run_viewer.sh --clip pen
   ```

   The first run creates `.venv/` with the packages in `viewer/requirements.txt` (needs Python 3.12 or
   newer, or `uv`). On a remote machine, forward the port: `ssh -L 8080:localhost:8080 <host>`.

The viewer has a frame slider and playback, the RGB frame with one run's 2D tracks drawn on it, and a 3D
grid (one row per setting, one column per method) with each run's point cloud, tracks, trails and
camera. It can filter by method, setting and run, show only moving tracks, and scale non-metric runs
(TraceAnything) to the LiDAR depth. `bash viewer/run_viewer.sh --check` validates every result file
without starting a server.

## Re-run a method (GPU)

Each method folder has three files:

- `setup_env.sh`: one-time install of Miniforge, a Python env, the official code at a pinned commit, and
  every checkpoint (cached under `<method>/work/`, so jobs can run without internet).
- `run_<method>.py`: the runner; it reads frames with `common/track3d_data.py` and writes
  `results/<clip>/<setting>/`.
- `run_<method>.sbatch`: the job script; submit it with Slurm (`sbatch`) or run it with plain `bash`.

```bash
cd spatracker_v2
bash setup_env.sh                 # needs internet; 5-15 minutes
bash run_spatracker.sbatch        # or: sbatch run_spatracker.sbatch
```

Finished runs are skipped, so a failed or interrupted job can simply be started again. Tested on an
NVIDIA A100 80GB (Linux, CUDA 12.x driver). Peak GPU memory: SpatialTrackerV2 about 60 GB on the longest
clip, Track4World 30 GB, TraceAnything 21 GB, DELTA 18 GB, TrackCraft3R 17 GB. Do not use Blackwell GPUs
(B200, RTX 5090, RTX PRO 6000): the pinned PyTorch versions do not support them. Containers that show
every host core but cap CPU time need `OMP_NUM_THREADS` near the granted vCPU count (the job scripts
default to 8, or `SLURM_CPUS_PER_TASK`).

## Result format

`<method>/results/<clip>/<setting>/`:

- `tracks.npz`: `queries` (N, 3) as [t, x, y]; `tracks2d_orig` (T, N, 2) in 1920x1440 pixels;
  `tracks3d_world` (T, N, 3) in the first camera's frame (OpenCV axes, meters for metric methods);
  `tracks3d_cam`, `track_depth`, `vis` (visible when > 0.5), `conf`, `c2w`, `intrinsics`, `frame_idx`
  (original Stray frame numbers), `timestamps`, `fps`, `size`, `orig_size`, `settings`.
- `result.npz`: per-frame depth, intrinsics, world-to-camera extrinsics and frames (for point clouds).
- `overlay.mp4` (2D tracks on the video) and `log.json` (settings, runtime, GPU memory, notes).

## Preprocessing

`common/preprocess_stray.py` turns the raw Stray Scanner recordings into `preprocessed_data/` (frames as
JPEG at 960x720, raw LiDAR depth and confidence, intrinsics, timestamps). It decodes video with macOS
AVFoundation, so it runs on a Mac only; the archive above already contains its output.
`common/track3d_data.py` (`load_clip`, `load_frames`) is the loader every method and the viewer use.
