#!/usr/bin/env python3
"""바퀴/엔코더/부호 셀프테스트 — 조립 직후 1순위 실행.

로봇을 블록 위에 올려 바퀴가 공중에 뜨게 한 뒤 실행한다.
바퀴를 하나씩 +PWM으로 돌리면서 (1) 실제 회전 방향을 사용자에게 묻고
(2) 엔코더 tick 부호/생존을 자동 판정해, 최종적으로

  - 펌웨어 `sign` 명령 한 줄
  - real.yaml `motor_signs` / `encoder_signs` 스니펫

을 출력한다. ROS 브리지는 반드시 꺼둘 것 (시리얼 독점).

사용:
  python3 scripts/dev/mecanum/10_wheel_selftest.py [--port /dev/ttyACM0]
      [--pwm 110] [--duration 1.0] [--apply]

  --apply : 판정한 sign을 즉시 펌웨어에 전송하고 `m 0.1 0 0` 전진 확인까지 수행.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _serial_util import Mecanum, WHEELS, find_port  # noqa: E402

DEAD_TICKS = 20


def spin_one_wheel(dev, index, pwm, duration):
    """Drive only wheel `index` at +pwm for `duration`s, return tick delta."""
    dev.command("z")
    start = dev.read_ticks()
    vec = [0, 0, 0, 0]
    vec[index] = pwm
    dev.command("p {} {} {} {}".format(*vec))
    time.sleep(duration)
    dev.command("p 0 0 0 0")
    time.sleep(0.3)
    end = dev.read_ticks()
    return end[index] - start[index]


def ask_direction(name):
    while True:
        ans = input(
            f"    -> {name}: 바퀴가 로봇 '전진' 방향으로 돌았습니까? [y=전진/n=후진/x=안돎/r=재시도] "
        ).strip().lower()
        if ans in ("y", "n", "x", "r"):
            return ans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=None)
    parser.add_argument("--pwm", type=int, default=110)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dev = Mecanum(find_port(args.port), verbose=args.verbose)
    try:
        dev.command("stream 0")
        print("\n== 핀맵 확인 (docs/hardware/mecanum_wiring.md와 대조) ==")
        _, lines = dev.command("pins", expect_prefixes=("PINS serial",), timeout=2.0)
        for line in lines:
            print("  " + line)

        print(
            "\n== 바퀴별 테스트 ==\n"
            "  로봇이 블록 위에 있고 바퀴 4개가 모두 공중에 떠 있는지 확인하세요.\n"
            f"  각 바퀴를 +PWM {args.pwm}으로 {args.duration}s 돌립니다.\n"
        )
        motor_signs = [1, 1, 1, 1]
        encoder_signs = [1, 1, 1, 1]
        problems = []

        for i, name in enumerate(WHEELS):
            input(f"[{i + 1}/4] {name} — Enter를 누르면 회전 시작 (바퀴를 주시!)")
            while True:
                ticks = spin_one_wheel(dev, i, args.pwm, args.duration)
                ans = ask_direction(name)
                if ans == "r":
                    continue
                break
            print(f"    tick delta = {ticks}")
            if ans == "x":
                problems.append(f"{name}: 모터 무반응 — 드라이버 채널/전원/DIR·PWM 배선 확인")
                continue
            forward = ans == "y"
            motor_signs[i] = 1 if forward else -1
            if abs(ticks) < DEAD_TICKS:
                problems.append(f"{name}: 엔코더 tick {ticks} — A/B 배선 또는 5V/GND 확인")
                continue
            # 물리 전진 회전에서 tick이 +가 되도록 encoder sign 결정.
            physically_forward_ticks = ticks if forward else -ticks
            encoder_signs[i] = 1 if physically_forward_ticks > 0 else -1

        print("\n== 판정 결과 ==")
        for i, name in enumerate(WHEELS):
            print(f"  {name:12s} motor_sign={motor_signs[i]:+d} encoder_sign={encoder_signs[i]:+d}")
        if problems:
            print("\n!! 하드웨어 문제 (sign으로 해결 불가):")
            for p in problems:
                print("   - " + p)

        sign_cmd = "sign {} {} {} {} {} {} {} {}".format(*motor_signs, *encoder_signs)
        print("\n== 적용 방법 ==")
        print(f"  펌웨어 즉시 적용:   {sign_cmd}")
        print("  real.yaml (src/robot_bringup/config/real.yaml → mecanum_bridge_node):")
        print(f"    motor_signs: [{', '.join(str(s) for s in motor_signs)}]")
        print(f"    encoder_signs: [{', '.join(str(s) for s in encoder_signs)}]")

        if args.apply and not problems:
            print("\n== --apply: sign 전송 + 전진 확인 ==")
            dev.command(sign_cmd)
            dev.command("geom 0.040 0.150 0.125")  # 80mm 휠 가정, 실측 후 갱신
            before = dev.read_ticks()
            dev.command("m 0.1 0 0")
            time.sleep(1.5)
            dev.command("stop")
            time.sleep(0.3)
            after = dev.read_ticks()
            deltas = [a - b for a, b in zip(after, before)]
            print(f"  m 0.1 0 0 (1.5s) tick delta = {deltas}")
            ok = all(d > DEAD_TICKS for d in deltas)
            print("  판정: " + ("4바퀴 모두 + → 전진 OK" if ok else "!! 일부 바퀴 tick이 +가 아님 — 위 결과 재확인"))
    finally:
        dev.close()

    print("\n다음 단계: python3 scripts/dev/mecanum/20_encoder_cpr_calib.py")


if __name__ == "__main__":
    main()
