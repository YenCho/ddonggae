#!/usr/bin/env python3
"""아레나 점유격자 맵(PGM + YAML) 생성기.

왜 필요한가
-----------
`stadium_nav2_map.pgm` 은 손으로 만들어졌고 생성기가 트리에 없었다. 그래서
아레나 크기가 바뀌면(2m 데모 등) 맵을 다시 만들 방법이 없다. 맵은 **상수 편집으로
못 바꾸는 유일한 산출물**이라 이게 없으면 축소 작업이 막힌다.

`OccupancyMap.inner_wall_bounds()` 가 PGM 에서 벽 사각형을 역으로 유도하고
`WallRangeLocalizer` 가 그걸 받아 쓰므로, 올바른 PGM/YAML 만 주면 로컬라이저는
따라온다. 로컬라이저에서 아레나 크기를 아는 다른 곳은 없다.

인수 조건
---------
**경기 맵을 바이트 단위로 재현할 것.** 이게 통과해야 데모 맵도 같은 규칙으로
만들어졌다는 증명이 된다. 그래서 `--verify` 를 먼저 돌리고 `--out` 을 쓴다:

    python3 make_arena_map.py --half 2.0 --verify stadium_nav2_map.pgm
    python3 make_arena_map.py --half 1.0 --out demo2m_nav2_map.pgm

기하 (경기 맵에서 역설계)
-------------------------
- 이미지는 벽 바깥으로 `margin_m`(0.5m) 만큼 더 크다. 전체 변 = 2*(half+margin).
  경기 half=2.0 → 5m → 250px @0.02. 데모 half=1.0 → 3m → 150px.
- 벽선은 **3px 두께**이고 벽 좌표를 **중심**으로 놓인다 (±1px).
- 각 벽은 수직인 두 벽의 **중심선까지** 그려진다 — 세로벽은 가로벽 중심행까지,
  가로벽은 세로벽 중심열까지. 그래서 모서리에서 2x2 만 겹친다.
  (경기 맵 occupied 3392 = 세로 1206 + 가로 1206 - 모서리 16 + 테두리 996.
   이 식이 맞아떨어지는 게 위 규칙이 옳다는 근거다.)
- 이미지 테두리 1px 도 occupied 다. 스캔이 맵 밖으로 새지 않게 하는 안전선.
- 값은 {0 = occupied, 254 = free} 두 가지뿐. maxval 255.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FREE = 254
OCCUPIED = 0
MAXVAL = 255

DEFAULT_RESOLUTION = 0.02
DEFAULT_MARGIN_M = 0.5
DEFAULT_WALL_PX = 3
DEFAULT_BORDER_PX = 1

# nav2 map_server 기본값과 동일 — arena 로컬라이저는 임계값을 쓰지 않지만
# 같은 YAML 을 nav2 에 물릴 수 있게 유지한다.
OCCUPIED_THRESH = 0.65
FREE_THRESH = 0.196


def _fmt(value: float) -> str:
    """2.0 -> '2', 0.5 -> '0.5' — 주석 문구를 경기 맵과 같은 모양으로."""
    text = f"{value:g}"
    return text


def build_pgm(half_m: float,
              margin_m: float = DEFAULT_MARGIN_M,
              resolution: float = DEFAULT_RESOLUTION,
              wall_px: int = DEFAULT_WALL_PX,
              border_px: int = DEFAULT_BORDER_PX) -> bytes:
    """벽이 ±half_m 에 있는 정사각 아레나의 P5 PGM 바이트를 만든다."""
    span_m = 2.0 * (half_m + margin_m)
    size = int(round(span_m / resolution))
    if abs(size * resolution - span_m) > 1e-9:
        raise SystemExit(
            f"2*(half+margin) = {span_m} 이 resolution {resolution} 으로 "
            f"정수 픽셀이 안 된다 (= {span_m / resolution})")
    if wall_px % 2 == 0:
        raise SystemExit("wall_px 는 홀수여야 벽 좌표에 중심을 놓을 수 있다")

    origin = -(half_m + margin_m)
    half_w = wall_px // 2

    def col_of(x_m: float) -> int:
        return int(round((x_m - origin) / resolution))

    def row_of(y_m: float) -> int:
        return (size - 1) - int(round((y_m - origin) / resolution))

    c_left, c_right = col_of(-half_m), col_of(+half_m)
    r_top, r_bottom = row_of(+half_m), row_of(-half_m)

    grid = bytearray([FREE]) * (size * size)

    def fill(c0: int, c1: int, r0: int, r1: int):
        for r in range(max(0, r0), min(size - 1, r1) + 1):
            base = r * size
            for c in range(max(0, c0), min(size - 1, c1) + 1):
                grid[base + c] = OCCUPIED

    # 세로벽 — 가로벽 중심행(r_top..r_bottom)까지만
    fill(c_left - half_w, c_left + half_w, r_top, r_bottom)
    fill(c_right - half_w, c_right + half_w, r_top, r_bottom)
    # 가로벽 — 세로벽 중심열(c_left..c_right)까지만
    fill(c_left, c_right, r_top - half_w, r_top + half_w)
    fill(c_left, c_right, r_bottom - half_w, r_bottom + half_w)

    # 이미지 테두리
    for b in range(border_px):
        fill(b, size - 1 - b, b, b)
        fill(b, size - 1 - b, size - 1 - b, size - 1 - b)
        fill(b, b, b, size - 1 - b)
        fill(size - 1 - b, size - 1 - b, b, size - 1 - b)

    comment = (f"# {_fmt(span_m)}m x {_fmt(span_m)}m stadium map, "
               f"{_fmt(resolution)}m/cell, physical walls at +/-{_fmt(half_m)}m")
    header = f"P5\n{comment}\n{size} {size}\n{MAXVAL}\n".encode()
    return header + bytes(grid)


def build_yaml(pgm_name: str, half_m: float,
               margin_m: float = DEFAULT_MARGIN_M,
               resolution: float = DEFAULT_RESOLUTION) -> str:
    origin = -(half_m + margin_m)
    return (f"image: {pgm_name}\n"
            f"resolution: {resolution}\n"
            f"origin: [{origin}, {origin}, 0.0]\n"
            f"negate: 0\n"
            f"occupied_thresh: {OCCUPIED_THRESH}\n"
            f"free_thresh: {FREE_THRESH}\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="아레나 점유격자 맵 생성 (경기 맵 바이트 동일 재현이 인수 조건)")
    ap.add_argument("--half", type=float, required=True,
                    help="벽까지의 반폭 [m] — 경기 2.0, 2m 데모 1.0")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN_M,
                    help=f"벽 바깥 여백 [m] (기본 {DEFAULT_MARGIN_M})")
    ap.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION,
                    help=f"셀 크기 [m] (기본 {DEFAULT_RESOLUTION})")
    ap.add_argument("--wall-px", type=int, default=DEFAULT_WALL_PX,
                    help=f"벽선 두께 [px], 홀수 (기본 {DEFAULT_WALL_PX})")
    ap.add_argument("--out", type=Path,
                    help="PGM 출력 경로")
    # 경기 맵은 stadium.yaml -> stadium_nav2_map.pgm 이라 stem 이 다르다.
    # 그 관례를 따를 수 있게 yaml 경로를 따로 받는다.
    ap.add_argument("--yaml", type=Path,
                    help="YAML 출력 경로 (기본: --out 과 같은 stem 의 .yaml)")
    ap.add_argument("--verify", type=Path,
                    help="기존 PGM 과 바이트 비교만 하고 끝낸다")
    args = ap.parse_args()

    pgm = build_pgm(args.half, args.margin, args.resolution, args.wall_px)

    if args.verify:
        want = args.verify.read_bytes()
        if pgm == want:
            print(f"OK  {args.verify} 와 바이트 동일 ({len(pgm)} bytes)")
            return 0
        print(f"FAIL {args.verify} 와 다르다: 생성 {len(pgm)}B vs 기존 {len(want)}B",
              file=sys.stderr)
        for i, (a, b) in enumerate(zip(pgm, want)):
            if a != b:
                print(f"  첫 불일치 오프셋 {i}: 생성 {a} vs 기존 {b}", file=sys.stderr)
                break
        return 1

    if not args.out:
        ap.error("--out 또는 --verify 중 하나는 필요하다")
    args.out.write_bytes(pgm)
    yaml_path = args.yaml or args.out.with_suffix(".yaml")
    yaml_path.write_text(build_yaml(args.out.name, args.half,
                                    args.margin, args.resolution))
    size = int(round(2.0 * (args.half + args.margin) / args.resolution))
    print(f"{args.out}  ({size}x{size}, 벽 +/-{args.half}m)")
    print(f"{yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
