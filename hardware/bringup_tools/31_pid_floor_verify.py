#!/usr/bin/env python3
"""바닥용 PID 검증: 전진/후진 대칭 스텝으로 제자리 복귀하며 트래킹 측정.

30_pid_step_tune.py와 같은 판정 지표(rise/overshoot/steady/ripple)를
전진·후진 각 구간에 대해 계산한다. RR은 B상 단채널(rr1ch) 상태.
"""
import csv
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]   # repo root (hardware/bringup_tools/ -> ../..)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _serial_util import Mecanum, WHEELS, find_port, parse_state  # noqa: E402

CPR = 1320.0
PID = sys.argv[1] if len(sys.argv) > 1 else "10 5 0 8 250 6.9 8"
TARGETS = [float(t) for t in sys.argv[2].split(",")] if len(sys.argv) > 2 else [3.0, 6.0, 9.0]
SEG_DURATION = float(sys.argv[3]) if len(sys.argv) > 3 else 1.6  # s per direction
WHEEL_R = 0.034

out_dir = REPO_ROOT / "logs" / "mecanum_bringup" / f"pid_verify_{time.strftime('%H%M%S')}"
out_dir.mkdir(parents=True, exist_ok=True)
summary = [f"pid_floor_verify  cpr={CPR}  pid={PID}  seg={SEG_DURATION}s bidirectional"]


def collect(dev, target, duration):
    samples = []
    cmd = f"w {target} {target} {target} {target}"
    start = time.time()
    last_cmd = 0.0
    while time.time() - start < duration:
        now = time.time()
        if now - last_cmd > 0.2:
            dev.send(cmd)
            last_cmd = now
        line = dev.readline()
        if line:
            st = parse_state(line)
            if st:
                samples.append(st)
    return samples


def to_series(samples):
    series = []
    for prev, cur in zip(samples, samples[1:]):
        dt = (cur["ms"] - prev["ms"]) / 1000.0
        if dt <= 0:
            continue
        vel = [(cur["ticks"][i] - prev["ticks"][i]) * 2 * math.pi / CPR / dt for i in range(4)]
        series.append(((cur["ms"] - samples[0]["ms"]) / 1000.0, vel, cur["pwm"]))
    return series


def analyze(series, target, duration, label):
    sign = 1.0 if target >= 0 else -1.0
    steady_from = max(duration - 0.6, duration * 0.6)
    lines = [f"\n[{label} {target:+.1f} rad/s]"]
    worst_err = 0.0
    for i, name in enumerate(WHEELS):
        vel = [(t, v[i] * sign) for t, v, _ in series]
        tgt = abs(target)
        if not vel:
            lines.append(f"  {name:12s} 데이터 없음")
            continue
        rise = next((t for t, v in vel if v >= 0.9 * tgt), None)
        peak = max(v for _, v in vel)
        steady = [v for t, v in vel if t >= steady_from]
        mean = sum(steady) / len(steady) if steady else float("nan")
        ripple = (max(steady) - min(steady)) if steady else float("nan")
        err = (mean - tgt) / tgt * 100
        worst_err = max(worst_err, abs(err))
        rise_s = f"{rise:.2f}s" if rise is not None else ">dur"
        lines.append(f"  {name:12s} rise={rise_s:>6s} overshoot={(peak - tgt) / tgt * 100:+6.1f}% "
                     f"steady={mean:5.2f} ({err:+5.1f}%) ripple={ripple:.2f}")
    return lines, worst_err


dev = Mecanum(find_port())
try:
    dev.command("stream 0")
    ok, _ = dev.command("rr1ch 1")
    print("rr1ch:", ok)
    if PID != "-":
        ok, _ = dev.command("pid " + PID)
        print("pid:", ok)
    else:
        print("pid: (펌웨어 기본값 사용)")

    for target in TARGETS:
        print(f"\n== ±{target} rad/s ({SEG_DURATION}s each way, "
              f"편도 ~{target * WHEEL_R * SEG_DURATION:.2f} m) ==")
        dev.command("z")
        dev.send("stream 1")
        fwd = collect(dev, target, SEG_DURATION)
        dev.send("w 0 0 0 0")  # 정지로 적분 리셋 (deadband에서 integral=0)
        time.sleep(0.4)
        rev = collect(dev, -target, SEG_DURATION)
        dev.send("w 0 0 0 0")
        time.sleep(0.25)
        dev.send("stream 0")
        dev.command("stop")
        dev.drain(0.3)

        for label, samples in (("fwd", fwd), ("rev", rev)):
            series = to_series(samples)
            t = target if label == "fwd" else -target
            with open(out_dir / f"step_{target:g}_{label}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s"] + [f"v_{x}" for x in WHEELS] + [f"pwm_{x}" for x in WHEELS])
                for ts, vel, pwm in series:
                    w.writerow([f"{ts:.3f}"] + [f"{v:.3f}" for v in vel] + list(pwm))
            lines, worst = analyze(series, t, SEG_DURATION, label)
            for ln in lines:
                print(ln)
            summary.extend(lines)
        time.sleep(0.6)
finally:
    try:
        dev.send("w 0 0 0 0")
        dev.command("stop")
    except Exception:
        pass
    dev.close()

(out_dir / "summary.txt").write_text("\n".join(summary) + "\n")
print(f"\n결과 저장: {out_dir}")
