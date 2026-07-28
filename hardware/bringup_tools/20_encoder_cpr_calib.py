#!/usr/bin/env python3
"""JGB37-520 엔코더 CPR 실측 (출력축 기준, 기어비 포함).

바퀴를 손으로 정확히 N바퀴(기본 5) 돌려 tick 수로 CPR을 계산한다.
로봇은 블록 위(바퀴 공중) 상태여야 하고, ROS 브리지는 꺼둘 것.

사용:
  python3 scripts/dev/mecanum/20_encoder_cpr_calib.py [--port ...] [--revs 5]
      [--wheel front_left]   # 생략 시 4바퀴 순회
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _serial_util import Mecanum, WHEELS, find_port  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=None)
    parser.add_argument("--revs", type=float, default=5.0)
    parser.add_argument("--wheel", choices=WHEELS, default=None)
    args = parser.parse_args()

    wheels = [args.wheel] if args.wheel else WHEELS
    dev = Mecanum(find_port(args.port))
    results = {}
    try:
        dev.command("stream 0")
        for name in wheels:
            idx = WHEELS.index(name)
            input(
                f"\n[{name}] 바퀴에 테이프로 기준 표시를 하고 Enter → "
                f"tick 카운터를 0으로 리셋합니다."
            )
            dev.command("z")
            input(
                f"[{name}] 바퀴를 '전진' 방향으로 정확히 {args.revs:g}바퀴 돌린 뒤 Enter"
            )
            ticks = dev.read_ticks()[idx]
            cpr = abs(ticks) / args.revs
            results[name] = cpr
            print(f"  ticks={ticks}  →  CPR ≈ {cpr:.1f}")
            if ticks < 0:
                print("  (주의: tick 부호가 음수 — 10_wheel_selftest의 encoder_sign 결과 재확인)")
    finally:
        dev.close()

    if results:
        mean_cpr = sum(results.values()) / len(results)
        print("\n== 결과 ==")
        for name, cpr in results.items():
            dev_pct = (cpr - mean_cpr) / mean_cpr * 100 if mean_cpr else 0.0
            print(f"  {name:12s} CPR {cpr:8.1f}  ({dev_pct:+.1f}% vs 평균)")
        print(f"  평균 CPR = {mean_cpr:.1f}")
        print("\n반영할 곳:")
        print(f"  1) real.yaml mecanum_bridge_node: encoder_cpr: {round(mean_cpr)}")
        print("  2) 펌웨어 기본값(ENCODER_CPR)과 다르면 mecanum_encoder_control.ino도 갱신 후 재플래시")
        print("\n다음 단계: python3 scripts/dev/mecanum/30_pid_step_tune.py --cpr "
              f"{round(mean_cpr)}")


if __name__ == "__main__":
    main()
