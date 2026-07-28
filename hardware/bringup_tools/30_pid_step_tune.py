#!/usr/bin/env python3
"""휠 velocity PID 스텝응답 측정/튜닝 — 로봇을 블록 위에 올리고 실행.

각 목표 속도(rad/s)로 4바퀴 동시 스텝 명령(`w`)을 주고 STATE(10Hz) tick으로
측정 속도를 복원해 상승시간/오버슈트/정상상태 오차를 바퀴별로 리포트한다.
게인을 바꿔가며 반복 실행 → 만족스러우면 real.yaml에 반영.

사용 (반복 루프):
  python3 scripts/dev/mecanum/30_pid_step_tune.py \
      --pid "12 7 0 45 220 14 200" --targets 3,6,9 --cpr 2464

  --pid "kp ki kd min_pwm max_pwm ff_slope int_limit"  # 생략 시 펌웨어 현재값 유지
  --plot  # matplotlib 있으면 png 저장

출력: logs/mecanum_bringup/pid_<HHMMSS>/ 에 CSV + summary.txt
튜닝 가이드는 docs/hardware/mecanum_bringup_day_plan.md 5단계 참조.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _serial_util import Mecanum, WHEELS, find_port, parse_state  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_step(dev, target, duration, cpr):
    """Command all wheels to `target` rad/s, return list of state samples."""
    dev.command("z")
    dev.send("stream 1")
    samples = []
    cmd = f"w {target} {target} {target} {target}"
    start = time.time()
    last_cmd = 0.0
    while time.time() - start < duration:
        now = time.time()
        if now - last_cmd > 0.2:  # velocity mode 500ms watchdog보다 짧게 재전송
            dev.send(cmd)
            last_cmd = now
        line = dev.readline()
        if line:
            state = parse_state(line)
            if state:
                samples.append(state)
    dev.send("w 0 0 0 0")
    time.sleep(0.2)
    dev.send("stream 0")
    dev.command("stop")
    dev.drain(0.3)

    # tick -> rad/s
    series = []  # (t_sec, [v_fl, v_fr, v_rl, v_rr], [pwm x4])
    for prev, cur in zip(samples, samples[1:]):
        dt = (cur["ms"] - prev["ms"]) / 1000.0
        if dt <= 0:
            continue
        vel = [
            (cur["ticks"][i] - prev["ticks"][i]) * 2.0 * 3.141592653589793 / cpr / dt
            for i in range(4)
        ]
        series.append(((cur["ms"] - samples[0]["ms"]) / 1000.0, vel, cur["pwm"]))
    return series


def analyze(series, target, duration):
    """Per-wheel rise time / overshoot / steady-state stats."""
    report = []
    steady_from = max(duration - 0.7, duration * 0.6)
    for i, name in enumerate(WHEELS):
        vel = [(t, v[i]) for t, v, _ in series]
        if not vel:
            report.append((name, None))
            continue
        rise = next((t for t, v in vel if v >= 0.9 * target), None)
        peak = max(v for _, v in vel)
        steady = [v for t, v in vel if t >= steady_from]
        mean = sum(steady) / len(steady) if steady else float("nan")
        ripple = (
            (sum((v - mean) ** 2 for v in steady) / len(steady)) ** 0.5
            if steady else float("nan")
        )
        report.append((name, {
            "rise_s": rise,
            "overshoot_pct": (peak - target) / target * 100.0,
            "steady_mean": mean,
            "steady_err_pct": (mean - target) / target * 100.0,
            "ripple_rad_s": ripple,
        }))
    return report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=None)
    parser.add_argument("--targets", default="3,6,9",
                        help="목표 wheel rad/s 목록 (쉼표 구분)")
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--cpr", type=float, default=2464.0)
    parser.add_argument("--pid", default=None,
                        help='"kp ki kd min_pwm max_pwm ff_slope int_limit"')
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    targets = [float(t) for t in args.targets.split(",") if t.strip()]
    out_dir = REPO_ROOT / "logs" / "mecanum_bringup" / f"pid_{time.strftime('%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    input("로봇이 블록 위(바퀴 공중)인지 확인 후 Enter — 4바퀴가 동시에 돕니다!")

    dev = Mecanum(find_port(args.port))
    summary_lines = [f"pid_step_tune  cpr={args.cpr}  pid={args.pid or '(firmware default)'}"]
    try:
        dev.command("stream 0")
        if args.pid:
            ok, _ = dev.command("pid " + args.pid)
            print(f"pid 설정: {ok}")
            summary_lines.append(f"pid cmd -> {ok}")

        for target in targets:
            print(f"\n== step {target} rad/s ({args.duration}s) ==")
            series = run_step(dev, target, args.duration, args.cpr)
            with open(out_dir / f"step_{target:g}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["t_s"] + [f"v_{w}" for w in WHEELS]
                                + [f"pwm_{w}" for w in WHEELS])
                for t, vel, pwm in series:
                    writer.writerow([f"{t:.3f}"] + [f"{v:.3f}" for v in vel]
                                    + list(pwm))
            summary_lines.append(f"\n[target {target} rad/s]")
            for name, stats in analyze(series, target, args.duration):
                if stats is None:
                    line = f"  {name:12s} 데이터 없음"
                else:
                    rise = f"{stats['rise_s']:.2f}s" if stats["rise_s"] else ">dur"
                    line = (f"  {name:12s} rise={rise:>6s} "
                            f"overshoot={stats['overshoot_pct']:+6.1f}% "
                            f"steady={stats['steady_mean']:5.2f} "
                            f"({stats['steady_err_pct']:+5.1f}%) "
                            f"ripple={stats['ripple_rad_s']:.2f}")
                print(line)
                summary_lines.append(line)

            if args.plot:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 4))
                    for i, w in enumerate(WHEELS):
                        ax.plot([t for t, _, _ in series],
                                [v[i] for _, v, _ in series], label=w)
                    ax.axhline(target, color="k", ls="--", lw=0.8)
                    ax.set_xlabel("t [s]"); ax.set_ylabel("wheel rad/s")
                    ax.legend(); ax.set_title(f"step {target} rad/s")
                    fig.savefig(out_dir / f"step_{target:g}.png", dpi=110)
                    plt.close(fig)
                except ImportError:
                    print("  (matplotlib 없음 — plot 생략)")
            time.sleep(0.8)
    finally:
        dev.close()

    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"\n결과 저장: {out_dir}")
    print("판정 기준: rise<0.4s, overshoot<15%, steady err<±5%, ripple<0.5, 4바퀴 편차 작을 것")
    print("만족 시 real.yaml mecanum_bridge_node의 speed_kp/ki/kd/min_pwm/max_pwm/"
          "feedforward_slope 갱신 (브리지가 접속 시 자동 push)")


if __name__ == "__main__":
    main()
