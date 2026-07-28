# Run evidence

Figures drawn from the four competition `report.json` files and the one recorded
pose log. Nothing here is hand-drawn or reconstructed — each figure names the
run it came from, and the scripts read the reports directly.

| File | What it is | Source | Used by |
|---|---|---|---|
| `qualifier1-route.png` | Every metre the robot drove in Qualifier 1, over the 42-point grid, coloured by match time, with the five grasps marked | `frames/index.jsonl` (3 Hz pose log) + `report.json` | `navigation/README.md` |
| `qualifier1-grid-map.png` | The grid map the 12-shot centre scan produced: identity and vote count per cell, collected cells outlined | `grid_map.json` + `report.json` | `mission/README.md` |
| `arena-control-ui.png` | The web UI `arena_control_node` serves on port 18765 | the real page, filled with a Qualifier 1 pose and localisation result | `navigation/README.md` |
| `cycle-time-budget.png` | Where each collection cycle's seconds went, all four matches | `report.json` `cycles[].phases` | `mission/README.md` |

## About the pose log

Qualifier 1 is the only match with a recorded trajectory. It ran with
`--frame-rec-fps 3`; the setting was turned off on the morning of the finals to
save write bandwidth, so the other three matches have event timestamps but no
per-frame pose.

The log is the **localiser's estimate**, not ground truth. Two artefacts of that
are filtered out of `qualifier1-route.png` before plotting, both by
`sim/isaacsim/scripts/replay_match.py:clean_pose_log` in the development repo:

* **Standing still.** During the 12-shot scan the robot holds one point and only
  turns, but the estimate wanders over a 24 cm span. Those samples are collapsed
  to their median.
* **Single-sample yaw spikes.** A 3-point median removes them. Real rotations —
  the 30° scan steps, the ±45° at a mini-goal, the −135° at the storage corner —
  span several samples and survive untouched.

Filtering removes 4.5 m of the 51.1 m raw path length. The robot did not drive
those 4.5 m; the estimator did.
