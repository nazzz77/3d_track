# Results: first round

Five 3D point trackers on six iPhone manipulation clips, run with their official code and default
settings on the same frames (10 fps, or 7.5 fps for the two longest clips to stay under 200 frames;
113-185 frames per clip) and the same 1,152 grid query points.
`rgb` = video only; `rgbd_fixed` = iPhone LiDAR depth and intrinsics with the camera held fixed.
All 48 runs finished without errors (one NVIDIA A100 80GB per method).

## Depth vs. LiDAR

For every visible track point, the predicted depth is compared with the iPhone LiDAR depth
(high-confidence pixels only) at the pixel where the method places the point, over all frames of all
clips (480k-605k samples per run). Each clip first gets one scale factor (median LiDAR / predicted
depth); the error is the median relative depth error after that scaling.

| Method (`rgb`) | Error after scaling, median over clips | Range over clips | Predicted / LiDAR depth before scaling |
| --- | --- | --- | --- |
| SpatialTrackerV2 | 1.6% | 1.0-3.4% | 1.27 |
| Track4World | 2.4% | 1.6-5.4% | 1.23 |
| TrackCraft3R (DA3 depth) | 2.6% | 1.6-8.8% | 1.19 |
| DELTA (UniDepth V2 depth) | 3.2% | 2.3-5.3% | 1.22 |
| TraceAnything | 7.5% | 5.1-13.3% | 0.12 (not metric) |

- After scaling, four of the five methods recover the scene geometry to within a few percent;
  SpatialTrackerV2 is closest to the LiDAR.
- Before scaling, the methods that predict metric depth all place these close-range scenes 19-27% too
  far, so none of them gives accurate absolute depth here.
- TraceAnything is not metric and stretches the far background, which causes most of its error.
- With LiDAR as input (`rgbd_fixed`), the errors are 0.2-0.8%; this mainly confirms that the protocol
  is consistent.
- The score measures depth at the pixel where a method places a point. It does not show whether the
  point is tracked correctly.

## Motion

The camera is static, so most points should stay still. Share of tracks that move more than 60 px
(in 1920x1440 pixels) from their query position at some point, over all clips:

| Method | `rgb` | `rgbd_fixed` |
| --- | --- | --- |
| TraceAnything | 8% | |
| SpatialTrackerV2 | 12% | 12% |
| DELTA | 20% | 20% |
| Track4World | 21% | |
| TrackCraft3R | 24% | 27% |

![Moving tracks on the busiest frame of each clip (rows) for each video-only run (columns)](docs/moving_tracks.png)

The figure shows only the tracks that move more than 60 px, on the busiest frame of each clip, with a
10-frame trail. Colors encode the query position, so the same point has the same color in every column.

- DELTA and Track4World keep motion on the hands and the manipulated objects.
- TrackCraft3R also moves static objects, such as the bowl in pour_lid and pen.
- TraceAnything misses the paper fold: in paper_fold only 34 of its tracks move, against 391-436 for
  DELTA, Track4World and TrackCraft3R. It does follow the shirt (345 moving tracks in shirt_fold).
- SpatialTrackerV2 drags some points along the arm in pen and covers less of the paper and the shirt
  (287 and 306 moving tracks, against about 400 and 600 for DELTA and Track4World).
