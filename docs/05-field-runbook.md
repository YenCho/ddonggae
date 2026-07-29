# 05 — Field Runbook

The competition-day procedure, in the order it was executed. This page is written to be
followed while nervous and short of time: numbered steps, one command per step, an explicit
"what a healthy output looks like" for every check, and a symptom → action table at the end.

The reasoning behind each check lives elsewhere and is linked rather than repeated —
[`06-troubleshooting.md`](06-troubleshooting.md) for causes and history,
[`04-getting-started.md`](04-getting-started.md) for installation and one-time setup,
[`../mission/README.md`](../mission/README.md) for the full match-runner CLI, and
[`../mission/docs/match-strategy.md`](../mission/docs/match-strategy.md) for why the match is
played the way it is.

> **The shipped tools print Korean.** `field_autopilot.py`, `match_runner.py` and the perception
> tools log in Korean, because that is what ran. Every log line quoted below is given verbatim with
> an English gloss, so you can match the characters on screen without reading Korean.

---

## 0. The shape of the day

| Phase | When | Where |
|---|---|---|
| Venue setup: power, bringup, health, photometry, stitch | on arrival, before anything is scored | §1–§6 |
| Pose seeding and one `safe` rehearsal cycle | as soon as the arena is free | §7–§8 |
| Target announcement → **1 minute** to change settings | immediately before the match | §9 |
| 180 s autonomous run, team standing away from the computer | the match | §9 |
| Recover the report, re-seat, re-check | between rounds | §10 |
| Teardown, pull `logs/field_ops/` | end of day | §11 |

The one-minute settings window is the reason everything else on this page happens earlier. On the
day, the *only* thing that changes is two command-line arguments (`--target-shape`,
`--target-fruit`). Nothing is rebuilt, reconfigured or retrained inside that minute.

---

## 1. Power-on order

Order matters, and it is the reverse of what people usually do.

0. **Kill switch OFF before the Jetson is booted.** The 12 V rail must be dead while the Jetson
   comes up and enumerates USB. Booting with the switch already on is what produced most of our
   "the OpenRB isn't there" and "the gripper node just died" incidents — see the note below.
1. **Jetson on.** Wait for the desktop/SSH to come up. Nothing else is powered yet.
2. **USB devices seated** — LiDAR, both RealSense cameras, the Arduino, the OpenRB-150. Confirm
   before applying motor power, because a USB reconnect *while running* resets the Arduino and
   loses its RAM-held PID configuration ([`06-troubleshooting.md` §2](06-troubleshooting.md)).
3. **Kill switch / drive battery ON**, now that the boards have already enumerated. The
   OpenRB-150's Dynamixel bus is powered from this rail. If the switch is off, the servos are
   simply absent and every `LIFT_*` and gripper command does nothing — this was the entire cause
   of a "dead board" panic on 2026-07-21.
4. **Robot placed on the floor, mast at the bottom.** Always power-cycle the OpenRB with the mast
   *down*: mast home lives in the board's RAM and is captured on the first lift command after boot.
5. Only now run the bringup (§2).

```bash
lsusb | grep -i -e intel -e arduino -e robotis     # 2x RealSense, Arduino, OpenRB-150
ls -l /dev/rplidar                                  # the udev symlink the launch opens
```

Four devices, four lines. If the OpenRB is missing, go back to step 0: switch the 12 V rail off,
reboot the Jetson with it off, and only then switch it on.

> **Why step 0 exists.** Boot the Jetson with the kill switch already on and the OpenRB-150 either
> fails to enumerate or enumerates and then disappears under load, taking `gripper_bridge_node`
> with it. Boot with the rail dead and the same board is reliable. We never instrumented the cause,
> so treat this as an observed rule with a guess attached: the working theory is that with 12 V
> live at boot the board backfeeds current into the shared ground/USB path, and the Jetson's port
> protection cuts the port rather than powering a device it reads as faulty. What is certain is the
> symptom and that the ordering fixes it. Full write-up in
> [`06-troubleshooting.md` §2](06-troubleshooting.md#the-openrb-is-missing-or-drops-out-after-booting-with-12-v-live).

---

## 2. Bringup — one command

```bash
cd <repo-root>
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 -u mission/field_autopilot.py up
```

`up` (`mission/field_autopilot.py:249`) does five things in this order: operator checklist →
kill duplicate motor bridges → launch the stack (or reuse a running one) → wait for the required
topics → health-check → seed the start pose.

**Answer the checklist honestly.** It asks one combined question
(`perception/fieldlib.py:969`) that the software genuinely cannot see:

> `[체크] 물리 상태 일괄 확인 — ①배터리 ON ②우하단 출발구역 북향 배치 ③마스트 최저(다운) ④그리퍼 주변 클리어 ⑤경기장 안 클리어. 전부 확인했나요? (y/n)`
>
> *Physical state, all at once — ① battery ON ② placed in the bottom-right start zone facing north
> ③ mast fully down ④ nothing near the gripper ⑤ arena clear. All confirmed?*

`--yes` skips it. Use `--yes` only when you have already done the walk-around; the checklist is
30 seconds and it catches the two failure modes (battery off, robot facing the wrong way) that
otherwise cost you a whole rehearsal.

Three things about `up` worth knowing before you rely on it:

- It sets the stack's environment itself (`perception/fieldlib.py:884`): `DRIVE_TYPE=mecanum`,
  `BASE_SCAN_YAW=0.0`, `IMU_TOPIC=/imu/data`, `LAUNCH_CAMERAS=true`, cameras with depth enabled.
  If you instead run `scripts/run_lightweight_arena_control.sh` by hand with no environment, you
  get its own defaults — `DRIVE_TYPE=diff` and `BASE_SCAN_YAW=3.14159265359` — which is a
  diff-drive bridge and a 180°-rotated LiDAR on a mecanum robot. Use `up`.
- It does **not** enable the web UI (`ENABLE_WEB_UI` defaults to `false`). If you want the
  click-to-goal map at `http://<robot-ip>:18765` for a visual sanity check, export
  `ENABLE_WEB_UI=true` before `up`. Never during a scored run.
- It reuses a stack that is already running rather than restarting it, and prints
  `스택 이미 기동돼 있음 — 재사용` (*stack already up — reusing*). That is what makes it safe to
  re-run `up` between rounds (§10).

Output goes to `logs/field_ops/<YYYYmmdd_HHMMSS>_up/`, including `stack_launch.log` and
`verdict.json`.

---

## 3. Health check — what must answer

`up` runs this automatically; run it alone at any time with:

```bash
python3 mission/field_autopilot.py check      # 8 s listen, launches nothing
```

Exit **0** = every required stream present. Exit **2** = something required is missing.

### A healthy report looks exactly like this

```
  [✓] LiDAR /laser_scan
  [✓] arena localization status
  [✓] 모터 브리지 /motor/state                     ← motor bridge
  [✓] 상단캠 RGB                                   ← top camera RGB
  [✓] 근접캠 RGB                                   ← near (bottom) camera RGB
  [✓] 상단캠 depth                                 ← top camera depth
  [✓] 근접캠 depth                                 ← near camera depth
  [✓] IMU /imu/data
  [✓] 그리퍼 브리지                                ← gripper bridge telemetry
  [✓] /chassis/odom
  [✓] 그리퍼/마스트 보드(OpenRB) 실동작 (width 78.4mm)   ← OpenRB really alive
  [✓] wall_range 락 ({'reason': ..., 'latency_ms': 41})   ← localisation lock
== 전체: 정상
```

`== 전체: 정상` means *overall: normal*. The failure form is
`== 전체: 필수 항목 결손 — 위 ✗ 확인` (*required items missing — check the ✗ above*).

Marks (`perception/fieldlib.py:846`):

| Mark | Meaning |
|---|---|
| `✓` | stream is alive |
| `✗필수` | **required** and missing — this is what makes the exit code 2 |
| `–` | missing, but not required — **you must read these yourself** |

The required set (`HEALTH_SPEC`, `perception/fieldlib.py:782`) is: `/laser_scan`, arena status,
`/motor/state`, both cameras' RGB **and** depth, and `/chassis/odom`. IMU, gripper telemetry and
the localisation lock are reported but do not fail the check.

Both camera *depths* are required, not just RGB. Ranging is measured from aligned depth
(`--range-mode depth`), so a stack that came up RGB-only will scan perfectly and then miss every
grasp.

---

## 4. Gripper liveness — the cheapest insurance on this list

**Read this section even if you skip everything else. The gripper is where two of our three
competition-day losses live.** At 05:12 on the morning of the finals the bridge died on an
uncaught `termios.error` and nothing restarted it. In Final 1 the process stayed up but the board
stopped answering mid-match, and an octahedron was carried and released with an empty hand.
The post-mortem is [`07-results-and-lessons.md` §3.2](07-results-and-lessons.md).

Both of those changed the tooling, and the changes are in this tree:

> **`field_autopilot.py up` now treats the gripper board as required.** It checks the board
> directly after the health check, and if it is dead it runs `ensure_gripper_alive()` — clear
> orphan duplicates, wait out the launch respawn, kick the launch child, start a standalone as a
> last resort. If that still fails, `up` fails, with the same missing-requirement verdict the
> runner would give. A green `up` is now a green gripper. The launch also carries
> `respawn=True, respawn_delay=2.0`, and the bridge swallows `termios.error` and reconnects.

Do all three of these before every match:

**4.1 — Read the board line in the health report.** `gripper_board_ok()`
(`perception/fieldlib.py:797`) parses the actual payload rather than trusting the topic's
existence, because a board unplugged from USB keeps publishing `{"width_mm": null, "error": true}`
and a topic-existence check happily passes that. Its failure details are specific:

| Detail printed | Meaning |
|---|---|
| `그리퍼 텔레메트리 없음` | no telemetry at all — the bridge is not running |
| `보드 error=true (USB/전원 확인: lsusb 에 ROBOTIS OpenRB-150)` | board reports an error — check USB and the kill switch |
| `width_mm=null — 서보 미검출` | servo not detected — check the Dynamixel connector |
| `connected=false` | the bridge cannot open the serial port |

**4.2 — Command a real open/close.** Telemetry alive is not the same as motion. On 2026-07-19 the
board reported `READY 0 / FAULT 1 / DXL_POWER 0` and every grasp attempt closed on empty air; that
incident is why `gripper_alive()` exists (`mission/match_runner.py:1964`).

```bash
ros2 topic pub --once /gripper/command std_msgs/String "{data: OPEN}"
ros2 topic pub --once /gripper/command std_msgs/String "{data: CLOSE}"
ros2 topic echo --once /gripper/state      # connected:true, width_mm not null, error:false
```

Watch the fingers. If `/gripper/state` is healthy and the fingers do not move, the servo bus is
the problem, not the node.

**4.3 — Exercise the mast, over the topic, never the tty.**

```bash
ros2 topic pub --once /lift/command std_msgs/String "{data: LIFT_TO_TOP}"     # ~7 s
ros2 topic echo --once /lift/state                                            # home_set: 1
ros2 topic pub --once /lift/command std_msgs/String "{data: LIFT_TO_BOTTOM}"  # ~6 s
```

`home_set` must be `1` before you trust any mast command. **Never open the OpenRB's serial port
from a second process** — it asserts DTR, reboots the board, force-opens the gripper, and loses the
mast home, after which `LIFT_TO_TOP` drives into the hard stop. `gripper_bridge_node` owns the port
exclusively and passes `LIFT_*` through for exactly this reason.

If the gripper is dead and you cannot fix it, `match_runner.py` will still ask whether to continue
without it (approach-only). In a scored match that is worth 0 points; fix the gripper.

---

## 5. Venue photometry re-lock

**This is the only calibration that must be redone at the venue.** The stitch *geometry* describes
the relationship between two camera brackets, not between the robot and the room, and does not
change when you move buildings. Photometry does, because the lighting does.

Auto-exposure must stay locked. Two free-running cameras converge to different exposures and split
the stitch seam in brightness — measured ΔY **+12.4** at the seam with auto-exposure on
(2026-07-20). That is a perception failure, not a cosmetic one.

**Raise the mast first.** Measuring with the mast down inflates every seam metric several-fold:
the down-position overlap band is 29 rows of extreme lens edge, dominated by vignetting and
specular reflection. Measured the same day, same lighting: ΔY **28** down versus **6** up.

```bash
ros2 topic pub --once /lift/command std_msgs/String "{data: LIFT_TO_TOP}"   # ~7 s
python3 perception/tools/photometry_tune.py --check      # ~8 s diagnosis
python3 perception/tools/photometry_tune.py --sweep      # ~1 min, applies + prints launch args
```

Acceptance gates the tool enforces: floor luma Y in **100–140**, |ΔY| ≤ **6** across the seam,
|ΔR/G| and |ΔB/G| ≤ **0.05**, saturation ≤ **8 %**. The exposure it picks is the one whose floor Y
is closest to the target (default 115) among those under the saturation cap — the cap exists
because at exposure 300 in the lab on 2026-07-20 the X-strokes of an arena floor marker burned out
entirely.

Three things to know while you do this:

- **Do not chase the colour delta.** The residual ΔR/G between the top D435 and the bottom D435i is
  a sensor-unit difference that runtime white balance does not correct: sweeping WB from 3200 K to
  6000 K moved ΔR/G by 0.004. Match brightness, accept the chroma offset.
- **The sweep applies values at runtime; it does not edit the launch.** They survive until the
  stack is restarted. If you restart, either re-run `--sweep` or paste the launch argument line the
  tool prints. White balance must be written as a **float-formatted string** (`"4600.0"`, not
  `"4600"`) — the `realsense2_camera` parameter is declared as a double, and an integer-looking
  string is silently rejected, which is how a white-balance lock once appeared to be applied and
  was not.
- **Known path bug.** `perception/tools/photometry_tune.py:52` computes
  `REPO_ROOT = Path(__file__).resolve().parents[3]`, which was correct at the tool's earlier
  location (`scripts/dev/field_ops/`) and is now one level too high. `OUT_DIR`
  (`:62`) therefore resolves to `data/calibration/photometry/` **outside** the repository, and
  `--apply last` will not find a sweep you ran from a different checkout. Either fix the index to
  `parents[2]` or note where the JSON actually landed.

The stored artefact is `perception/calibration/photometry/arena.json`: the 2026-07-20 21:25 sweep,
exposure **200** / gain **64** / WB **4600 K**, `power_line_frequency: 2`, recorded with
`"pass": false` (seam ΔY −6.19 against a ±6 criterion) — an honest record of one lighting
condition. The launch defaults are exposure 320 / gain 64 / WB 6300.0 K at 1920×1080. Both are
measurements of a room, not constants of the robot: re-sweep at your venue and let the tool write
the values it finds.

---

## 6. Stitch verification

You are verifying, not recalibrating. Recalibrate only if a bracket, the mast, or a camera has
physically moved.

```bash
DISPLAY=:0 python3 perception/tools/stitch_live_view.py --mast up
```

The viewer uses the same `Stitcher` and the same `perception/calibration/stitch/*.json` as the
match runner, so what you see on screen is literally what A1 receives. Keys: `u` / `d` switch
between the up and down calibrations as you move the mast, `s` saves the current frame as PNG,
`f` toggles fullscreen, `q` / `Esc` quits.

What to look for at the red seam line:

| Check | Pass | Fail means |
|---|---|---|
| Floor pattern continuous across the line | geometry OK | re-fit needed (§below) |
| Brightness continuous across the line | photometry OK | redo §5 |
| Both left and right edges line up, not just the centre | scale/roll OK | the calibration was fitted from too few points |

That last row is the failure that cost the most time: a calibration fitted from a **single**
correspondence point fixes translation and constrains neither field of view nor roll, giving the
classic "centre lines up, both edges splay apart" look — measured here as 0.53° of roll error.

If you must re-fit at the venue:

```bash
python3 perception/tools/stitch_calibrator.py --mast up     # then open http://<robot>:8099/
```

Spread the correspondence points across the **full width** of the overlap band and use a **flat**
target — the cameras sit at different heights, so anything raised off the floor has parallax that
no 2D transform removes. Do not force a homography: the band is only ~70 rows tall, and fitting one
made the corners diverge by 330 px. Current inlier RMS is 0.78 px (`up`, affine, 6 points) and
1.13 px (`down`, similarity). Full detail:
[`../perception/docs/stitching-and-calibration.md`](../perception/docs/stitching-and-calibration.md).

Then put the mast back down before the next step.

---

## 7. Seed the start pose

The robot does not estimate where it starts; it is **told**. `START_POSE = (1.8, −1.8, +90°)`
(`perception/fieldlib.py:81`) — bottom-right of the map, facing north (+y), which is official
`(380, 20) cm`. Seeding is what removes the square arena's four-fold yaw ambiguity at t=0.

`field_autopilot.py up` seeds automatically. A good seed prints:

```
✓ 포즈 시딩 OK: (+1.80,-1.80,+90deg) — 실물이 우하단 북향인지 눈으로 확인
```
*(pose seeding OK — now confirm with your eyes that the robot really is bottom-right, facing north)*

Do exactly what it says. The software cannot tell the difference between "the pose is right" and
"the pose is confidently wrong", and a confidently-wrong localiser has already caused one object to
be released in the START corner instead of the storage box (2026-07-21 16:18) — which is why the
placement gate now exists.

A bad seed prints `! 시딩 후 pose 불일치/없음: ...` (*pose mismatch or absent after seeding*).
Re-place the robot in the start zone facing north and re-run `up`.

The match runner re-seeds on its own at `[SEED]` and requires a localisation lock with latency
< 150 ms within a 3 s window; if it never locks it exits **4** rather than driving blind
(`mission/match_runner.py:861`).

---

## 8. The dry run and the rehearsal

Escalate. Do not go straight to a full-speed run.

**8.1 — Offline self-test (no ROS, no robot, seconds).** Run it after any edit at all.

```bash
python3 mission/match_runner.py --offline
```

It exercises the GT parser, the corridor/street router (including a regression case on a full
28-object arena that asserts the returned path actually clears every obstacle), the 42-cell snap,
the storage pin layout, and the two-set candidate filter. Exit 1 = something failed.

**8.2 — Dry run with the robot powered but not moving.**

```bash
python3 mission/match_runner.py --dry-run --gt-text "apple:150,200;plain:250,300;octahedron:100,150"
```

Motion, gripper and lift commands are logged instead of sent; scan and inference still run if
frames are available. This is where you confirm the model loads, the stitch is sane, and the
routes are what you expect.

**8.3 — One `safe` cycle, one object, no placement.**

```bash
python3 mission/match_runner.py \
  --gt-text "apple:150,200;plain:250,300;octahedron:100,150" \
  --max-objects 1 --no-place --speed-profile safe
```

Two gates in that run, in order:

1. **The scan-vs-truth table**, printed immediately after the scan and *before the robot drives
   anywhere* — ok / wrong / ghosts / missed against the layout you typed. If the scan is wrong,
   stop here. Do not debug perception with a moving robot; switch to `record` (§12) and analyse
   offline.
2. **The grasp.** If it misses by a consistent centimetre, adjust `GRIP_FORWARD_M = 0.115`
   (`perception/fieldlib.py:78`) in ±0.01 steps. It is the single source of truth for the gripper's
   forward creep; do not scatter local offsets.

`gt_layout_generator.html` (open it in a browser) rolls a rule-legal 28-of-42 layout from a seed
and emits the matching `--gt-text ...` command line, so you are not typing 28 objects by hand.

**8.4 — Escalate the speed profile.**

| Profile | cruise / approach / place (m/s) | Use |
|---|---|---|
| `safe` | 0.5 / 0.4 / 0.3 | first cycle, isolating a problem |
| `normal` | 0.8 / 0.6 / 0.4 | default |
| `fast` | 0.95 / 0.7 / 0.45 | ≈95 % of the measured ~1.0 m/s hardware ceiling; use under time pressure |

Run a full cycle at `normal`, then at `fast`, and adopt the fastest profile that keeps the success
rate **and** finishes inside 180 s. Compare `summary.total_sec` and the per-cycle `route` /
`approach` / `grasp` / `carry` phase times in each `report.json`. If `fast` starts producing
localisation wobble or failed grasps, fall back to `normal`.

**Verify the profile is actually taking effect.** Cruise speed is pushed to the arena node with a
live `ros2 param set`; the runner logs `[SETUP] arena max_linear_mps ← 0.8`. If you instead see
`[SETUP] arena 속도 반영 실패(기본값으로 계속)` (*speed not applied, continuing with the default*),
the profile is only affecting relative moves and your `fast` run is not fast. Before this was added
on 2026-07-20 the profiles silently did nothing on goal-following drives at all.

---

## 9. The match

The target shape is announced in the morning; the target fruit is announced one minute before the
start. Neither requires a rebuild — both are command-line arguments to an already-running stack.

**Before the announcement**, with the stack up and healthy:

- gripper liveness re-checked (§4)
- mast **down**, robot in the start zone facing north, pose seeded and eyeballed (§7)
- previous round's terminal closed, so a stray `match_runner` cannot be publishing goals
- the command line already typed into the terminal, with only the two target arguments missing

**On the announcement**, fill in the two arguments and run:

```bash
bash scripts/run_match_day.sh
```

That is the whole command. It brings the stack up, health-checks it, seeds the pose, asks for the
two announced targets as single keypresses, and then holds at the runner's `(y/n)` prompt until the
starter says go. Every match parameter is fixed inside the script, at the values the 07-23 04:13
practice runs settled on:

| | |
|---|---|
| `--speed-profile fast` | 0.95 cruise / 0.7 approach / 0.45 place |
| `--scan-shots 12 --scan-settle 0.5` | twelve captures at 30°, half a second of settling each |
| `--votes-k 1 --fruit-k 1` | one vote confirms a cell (see `perception/docs/grid-voting.md`) |
| `--release-ticks 600` | ≈52.7° of finger opening at the drop |
| `--order fruit-first` | the 20-point set first |
| `--pair bp` | banana/pineapple verifier on, apple/orange off (finals; the qualifiers ran `on` then `ao`) |
| `--consistency on` | expected-value target ranking (finals only) |
| `--nav street --range-mode depth --scan-mast up` | |

If the stack is already up, add `--no-bringup`; to check the configuration without running,
`--check`. To override anything for one match, pass it after `--`:
`bash scripts/run_match_day.sh -- --consistency off`.

The equivalent manual invocation, if you ever need to drive the runner directly:

```bash
python3 mission/match_runner.py --yes \
  --target-shape icosahedron --target-fruit apple \
  --order fruit-first --speed-profile fast \
  --scan-shots 12 --scan-settle 0.5 --votes-k 1 --fruit-k 1 \
  --release-ticks 600 --pair bp --consistency on --nav street
```

| Argument | Why |
|---|---|
| `--yes` | skips every operator prompt. In a scored match a blocking `(y/n)` prompt costs seconds you do not have. Only use it because you already did the walk-around. |
| `--target-shape` / `--target-fruit` | competition mode. Set 1 = a polyhedron shape at 10 pts (quota 4), Set 2 = a fruit face at 20 pts (quota 3). In this mode there is **no non-target fallback ever** — a mispick costs 2× the object's value, so refusing to pick is a legitimate outcome. |
| `--order fruit-first` | collects the 20-point set while any remain. `nearest` is the default. |
| `--speed-profile fast` | what the practice runs settled on; every match ran it. |

Leave `--range-mode` at its default (`depth`). The alternative, `ground`, over-estimates range by
27–64 mm for any object whose visible face sits 4–7 cm above the floor and produced 0/4 icosahedron
grasps on 2026-07-20.

**The first ~10 s are model load** (import, weights, CUDA warm-up), overlapped with the GT prompt
and the startup checks by a background thread. Start the program, then step away.

### Reading the run while it happens

You are not allowed to touch the computer, but you can watch. In order:

| Log line | English | Means |
|---|---|---|
| `== 토픽 헬스체크 ==` … `== 전체: 정상` | topic health check … overall normal | startup gate passed |
| `[SEED] 출발 포즈 시딩 (+1.80,-1.80, yaw 90°)` | seeding start pose | |
| `  localization 락 OK — reason=… latency=…ms` | localisation locked | latency must be < 150 ms, else exit 4 |
| `[MAST_UP] LIFT_TO_TOP — /lift/command 브리지 경유 (tty 직접 금지)` | mast up via the bridge topic (never the tty) | non-blocking; it rises while the robot drives |
| `[GOTO_CENTER] street_diag → (+0.25,+0.25)` | diagonal to the highway line, then street north to the scan point | `street_L` instead means the pre-shot is off and it fell back to the older right-angle route |
| `[SCAN] 마스트업 step 스캔 시작` | mast-up scan starting | 12 shots at 30°, 0.5 s settle each |
| `[SCAN 결과] 확정 N셀 / presence-only M셀` | scan result: N confirmed cells, M presence-only | |
| the ok / wrong / ghosts / missed table | | printed **before** collection starts |
| `[COLLECT] 수거 루프 시작 (실전 타깃 …)` | collection loop starting, competition targets | |
| `[수거 1] … — 34.2s` | cycle 1 finished, elapsed | one line per object |

If the gripper telemetry is missing at `[COLLECT]`, the runner warns
`⚠ 그리퍼 텔레메트리 없음 (서보 미검출/전원?) — 파지는 전부 실패할 상태`
(*no gripper telemetry (servo undetected / power?) — every grasp will fail*) and, without `--yes`,
asks whether to continue approach-only. With `--yes` it continues. That is the failure §4 exists to
prevent.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | ran to completion (a `PARTIAL` verdict still exits 0 — read the report) |
| 1 | bad input, or an `--offline` self-test failure |
| 2 | required topics missing at the health check (also argparse's code for a bad flag — read the message, not the number) |
| 3 | operator aborted the physical checklist |
| 4 | localisation never locked after seeding |
| 5 | mast movement could not be confirmed (stitch and back-projection are height-dependent, so continuing would corrupt every measurement) |
| 6 | YOLO weights failed to load |

`stage_report()` runs in a `finally`, so a crash or a Ctrl-C still writes a complete
`report.json` up to the point of failure.

---

## 10. Between rounds

Keep the stack running. Restarting it costs a bringup, a health check and a photometry re-apply,
and buys nothing.

1. **Copy the run out, immediately.** `logs/field_ops/<ts>_e2e/` — `report.json`, `grid_map.json`,
   `grid_map.png`, the per-shot raw top/near pairs and overlays. Do it now, not at the end of the
   day; the next run creates a new directory but a crash-and-restart can lose an unsynced one.
2. **Read three numbers from `summary`**: `total_sec` against the 180 s budget, `score_est`, and
   `placed_by_cls` against quota. Then `scan.gt_compare` if you entered a layout.
3. **Re-place the robot** in the start zone, mast **down**, facing north.
4. **Re-run `up`.** It reuses the running stack, re-cleans duplicate motor bridges and re-seeds the
   pose. Confirm the seed with your eyes again (§7).
5. **Re-check the gripper** (§4). It is the check that already cost a final; it takes 20 seconds.
6. **Battery.** If anything felt sluggish, `python3 mission/field_autopilot.py up --battery-probe`
   commands a 3 cm move and calls the battery off if the move fails *while wheel odometry stays
   within 5 mm*. Note that the motor bridge reports "connected" with the drive battery off, because
   the Arduino is powered over USB from the Jetson — that combination is deliberately misleading.
7. **Only re-do photometry if the venue lighting changed** — house lights, a stage light, an
   opened door. Re-run `--check` (8 s) to decide before spending a minute on `--sweep`.
8. **Confirm no stray processes.** A previous `match_runner` still publishing goals will fight the
   new one: `pgrep -af match_runner.py`.

---

## 11. Shutdown and data recovery

```bash
python3 mission/field_autopilot.py down
```

It SIGINTs any rosbag, SIGINTs the stack, waits 6 s and then reports leftovers by name
(`arena launch`, `mecanum 브리지`, `arena 노드`, `RealSense 카메라`, `라이다`, `rosbag`). If it
prints `잔존 프로세스 있음` (*leftover processes present*), wait a moment and run `down` again —
it will print the exact `kill -INT` line for each one. `잔존 프로세스 없음` means clean.

Then take the whole of `logs/field_ops/` with you. Everything needed for offline analysis is in
there, and the one thing you cannot reconstruct afterwards is an image you did not save. **Always
keep the raw top/near pair, never only the stitched frame** — almost every perception question we
could not answer later was a case where only the stitch had been kept.

---

## 12. If X goes wrong, do Y

Under time pressure, in the order you are likely to meet them. Causes and full history are in
[`06-troubleshooting.md`](06-troubleshooting.md), linked per row.

| Symptom | Do this, now | Detail |
|---|---|---|
| **The gripper does not close; the robot reaches the object and stalls.** *This is what cost Final 1 (worth 90 pts).* | `pgrep -af gripper_bridge_node`. If it is gone, relaunch the stack (`down`, then `up`) — the node has `respawn=True, respawn_delay=2.0`, so a dead bridge comes back in about two seconds. If it is alive, `ros2 topic echo --once /gripper/state` and check `connected` / `error` / `width_mm`. | [§6](06-troubleshooting.md#6-gripper-and-mast) |
| **The 3-minute clock ran out mid-grasp.** *This is what cost Final 2.* | Drop to a faster profile only if `report.json` shows the loss was in `route`/`carry`, not in `approach`/`grasp`. There is **no time-aware policy** — the state machine will happily start a cycle it cannot finish. Reduce `--max-objects` if the arena layout is long-haul. | [§7](06-troubleshooting.md#7-match-level) |
| **The model is right in the lab and wrong at the venue** (a fruit reads as the wrong fruit). *This is what cost the qualifiers — the printed apple was a yellowish under-ripe apple, not the announced red one.* | Nothing at runtime recovers a colour prior that was never trained. Photograph the actual objects at the actual venue and fine-tune between rounds if you have hours; that is what we did on 2026-07-24. Within the one-minute window, the only lever is to not target that fruit. | [§5](06-troubleshooting.md#5-perception) |
| Every move times out, odometry never changes | Drive battery / kill switch is off. Switch on, `check`. Confirm with `up --battery-probe`. | [§1](06-troubleshooting.md#1-power-and-motion) |
| Motors stutter at low speed | Two motor bridges on one tty. `pkill -f mecanum_bridge_node`, then `up` (which also does this automatically). | [§1](06-troubleshooting.md#1-power-and-motion) |
| No camera topics, or one camera missing | `down`, then `up` (re-sets the RSUSB environment). If it persists, physically replug. If the *OpenRB* is what vanished, check the kill switch first. | [§4](06-troubleshooting.md#4-cameras-stitch-seam-and-photometry) |
| No localisation, or the pose is 90° off | Re-place bottom-right facing north and re-run `up` to re-seed. Confirm `imu_topic:=/imu/data` is set — without the IMU prior the square arena's four-fold yaw ambiguity reopens. | [§3](06-troubleshooting.md#3-lidar-and-localisation) |
| `LIFT_*` does nothing | Kill switch first, then `check`, then power-cycle the OpenRB **with the mast at the bottom**. Confirm `home_set` in `/lift/state`. Never open the tty. | [§6](06-troubleshooting.md#6-gripper-and-mast) |
| The scan reports ghosts or misses cells | **Stop before the robot drives.** The GT table already told you. Switch to capture: `field_autopilot.py record --mast down --gt-text "..."`, analyse offline. | [§5](06-troubleshooting.md#5-perception) |
| The stitch seam splits in brightness | Auto-exposure is not locked, or the WB string was written without a decimal point. Redo §5 with the mast **up**. | [§4](06-troubleshooting.md#4-cameras-stitch-seam-and-photometry) |
| The robot stops short of or overshoots the object | Check the `[approach]` line: frequent `src=ground` fallback means depth is returning holes (check the depth profile and alignment), not that `ground` is acceptable. Otherwise nudge `GRIP_FORWARD_M` by ±0.01. | [§5](06-troubleshooting.md#5-perception), [§6](06-troubleshooting.md#6-gripper-and-mast) |
| A single relative move takes seconds longer than predicted | A move issued without an explicit `max_v` falls back to the firmware's position mode at 0.233 m/s. Check `move_stats` in `report.json` before blaming the controller. | [§1](06-troubleshooting.md#1-power-and-motion) |
| Nothing above fits | **Capture, do not guess.** `field_autopilot.py record` saves raw top/near RGB+depth pairs, stitched YOLO overlays, per-frame pose and a rosbag. Ctrl-C to stop. | [§8](06-troubleshooting.md#8-when-none-of-this-helps-capture-do-not-guess) |

---

## 13. The checks we did not have, and should have

Stated so that you can add them rather than rediscover them:

- **A pre-match gripper liveness gate that blocks.** §4 is a procedure, executed by a person, under
  time pressure. It should be an automatic gate that `field_autopilot.py up` fails on, not a line in
  a report that prints as `–`. One line: pass `require_gripper=True`.
- **A positive liveness requirement on the grasp verdict.** Restarting a dead bridge is solved
  (`respawn=True`, serial-exception recovery, the four-step self-heal in `up`). What is not solved
  is the case that actually cost Final 1: the bridge alive but the board silent. The runner treats
  "no answer" as permission to continue, and it should instead demand a fresh verdict inside the
  0.85 s the check already takes, and give up the cycle if it does not get one.
- **A time-aware collection policy** — "do not start a cycle that cannot finish inside the
  remaining budget". The runner measures the budget and reports overrun; it never acts on it.

---

## Where to go next

| You want to | Read |
|---|---|
| Understand why a check exists at all | [`06-troubleshooting.md`](06-troubleshooting.md) |
| Every flag `match_runner.py` accepts | [`../mission/README.md`](../mission/README.md) |
| Why the match is played this way | [`../mission/docs/match-strategy.md`](../mission/docs/match-strategy.md) |
| Install and one-time setup | [`04-getting-started.md`](04-getting-started.md) |
| What actually happened, including all three failures | [`07-results-and-lessons.md`](07-results-and-lessons.md) |
