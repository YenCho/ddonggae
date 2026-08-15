#!/usr/bin/env python3
"""최종 경기 E2E 검증 — 마스트업 센터스캔 → 마스트다운 수거 → 골대 적재 (2026-07-19).

오늘 검증 전략: **마스트를 올린 상태로 중앙 8방향 스캔**(depth 역투영 — 마스트업
ground 역투영 금지 계약)으로 42격자 지도를 만들고, **마스트를 내린 뒤** 확정 셀을
차례로 수거해 보관함(골대)에 적재한다. 클로드 없이 조작자 단독 운용.

상태기계:
  STARTUP → SEED → MAST_UP → GOTO_CENTER → SCAN(8x45° CW) → MAST_DOWN
  → COLLECT { ROUTE → FACE → APPROACH → GRASP → CARRY } xN → REPORT

계약 (자세한 근거는 fieldlib.py):
  - 추론은 항상 스티치 프레임 (A1 imgsz=896 원척도, face imgsz=224 BGR).
  - 3D 위치는 원본 캠 픽셀에서만: 스캔=depth 중앙값 역투영(마스트업 캘리브 무관),
    접근=depth 역투영 우선(7/21 전환 — ground는 라운드 클래스 접점 가정 붕괴로
    +5~10cm 편향, icosahedron 파지 0/4 원인. depth 무효 픽셀만 ground 폴백).
  - 리프트/그리퍼는 /lift/command·/gripper/command 브리지 경유 (tty 직접 금지).
  - [2026-07-23 저녁 조작자 지시] 파지 판정(/gripper/grasp) 게이트가 **기본값**.
    단 게이트가 걸리는 건 `empty`(빈손이라고 적극 판정) 뿐이다:
      held            → 진행
      empty           → OPEN → +0.02m → CLOSE 1회 재시도, 재실패면 셀 skip+후퇴
      unknown/무응답  → **그대로 진행** (재시도·skip 없음, grasp_unconfirmed 기록)
    OpenRB 가 간헐 이상이라 실제로 물었는데 판정만 못 내는 사례가 잦다 — 이때
    재시도하면 문 물체를 스스로 떨어뜨리고 skip 하면 멀쩡한 수거를 버린다.
    `empty` 까지 무시하려면 --place-anyway.
  - YOLO는 프로그램 시작과 동시에 백그라운드 프리로드, 스캔 추론은
    전 샷 배치 (A1 1배치 + face 1배치 + pair 1배치, PNG 저장 백그라운드).
  - 파지 전진 깊이는 fieldlib.GRIP_FORWARD_M 단일 소스.
  - [2026-07-22] 수거 주행 기본값이 street 모드(street_nav.py 통합)로 변경 —
    하이웨이 x 정렬 → street 북진 → mini-goal → ±45° 회전 → 접근/파지 →
    되감기 → 북향 → 후진 남하 → 적재. 규칙: navigation/docs/control-and-routing.md.
    롤백: --nav legacy (종전 코리도 라우팅) 또는 match_runner.py.bak-20260722.

사용:
  # 실전 본경기 — 2세트 타깃 (룰북 §5/§7: 세트1 형상 4개 x10점 + 세트2 과일 3개 x20점,
  # 오픽업 = 기본점수 2배 감점이라 비타깃 수거 폴백 없음. cube 공지 시 --target-shape cube)
  python3 mission/match_runner.py --yes \
      --target-shape icosahedron --target-fruit apple

  # 검증/리허설 (단일 우선 클래스 — 종전 동작)
  python3 mission/match_runner.py \
      --gt-text "apple:150,200;plain:250,300" [--max-objects 3] [--target-class apple]

  # 스택 자동 기동 + 무프롬프트
  python3 mission/match_runner.py --launch-stack --yes --gt-file gt.txt

  # 기존 스캔 재사용 (수거만)
  python3 mission/match_runner.py --skip-scan \
      --map-file logs/field_ops/<ts>_e2e/grid_map.json

  # 이동/그리퍼 없이 리허설 (스캔/추론은 가능하면 수행, 없으면 GT 합성)
  python3 mission/match_runner.py --dry-run --gt-text "..."

  # ROS 없이 자가테스트 (인자·GT 파서·코리도 라우터)
  python3 mission/match_runner.py --offline

산출물: logs/field_ops/<ts>_e2e/ — report.json, grid_map.json, grid_map.png,
gt.json, scan/shotNN_{top,near,stitched,overlay}.png (+depth), stack_launch.log.

종료코드: 0 정상 / 1 인자 오류 / 2 헬스체크 결손 / 3 체크리스트 중단
          4 localization 락 실패 / 5 리프트 이동 미확인 / 6 모델 로드 실패
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[0] / "perception"))
import fieldlib as fl  # noqa: E402
# [2026-07-22] street 주행 통합 (--nav street, 기본값). 규칙 단일 소스:
# navigation/docs/control-and-routing.md. 롤백: --nav legacy (종전 코리도 라우팅) 또는
# 파일 백업 match_runner.py.bak-20260722.
sys.path.insert(0, str(SCRIPT_DIR.parents[0] / "navigation"))
import street_nav as snv  # noqa: E402

REPO_ROOT = fl.REPO_ROOT
# arena 노드가 기본으로 무는 맵 (lightweight_real.launch.py 의 map_yaml 기본값).
# `--offline` 의 P9 가 이 맵에서 유도한 아레나 크기를 fl.ARENA_HALF_M 과 대조해
# "런너는 4m, 노드는 2m 맵" 같은 사고를 로봇 없이 잡는다. 런치 기본값을 바꾸면
# 이 이름도 같이 바꿀 것.
DEFAULT_MAP_YAML_NAME = "stadium.yaml"
from geometry import (  # noqa: E402
    CameraMount, Intrinsics, pixel_to_ground)

# ---- 튜닝 상수 (본 스크립트 로컬 — 공용 계약값은 fieldlib) ----
SCAN_SHOTS = 8
# [데모 2m] 스캔 조준 yaw [deg, map 절대각] — None 이면 회전 없음(경기 동작 그대로).
# 2m 필드에서는 6칸의 방위각 스팬이 45.4°(103.0~148.4°)라 HFOV 69° 한 프레임에
# 전부 들어온다. 그래서 스핀 대신 126° 를 조준하고 1샷만 찍는다. 제자리 회전은
# 각 칸의 *방위각*을 바꾸지 않으므로(어느 칸이 프레임에 들어오는지만 바뀐다)
# 한 프레임에 다 들어오는 배치에서 스핀은 회전 시간만 쓰고 새 정보를 주지 않는다.
SCAN_AIM_YAW_DEG = None
A1_SCAN_CONF = 0.25          # 스캔: 저문턱 (셀 투표 누적이 거름)
A1_APPROACH_CONF = 0.35      # 접근: 60번 계약(0.4) 근처, 스티치라 소폭 완화
# [폐기 2026-07-23 04시 — 1순위를 480 으로 교체. 롤백 시 이 값으로 복귀]
# A1_IMGSZ_NEAR_APPROACH = 640  # [2026-07-23 01시] 접근 near 단독 추론 크기.
#     01:17 실기 (250,200) 재현: 640=conf 0.98 검출, 896=0건 — 640 고정.
A1_IMGSZ_NEAR_APPROACH = 480  # [2026-07-23 04시 — 조작자 지시] 접근 1순위.
# 무검출이면 **재촬영이 아니라 imgsz 를 바꿔** 같은 프레임을 다시 본다.
# 접근은 정지 장면이라 다시 찍어도 같은 이미지 → 같은 무검출이다(011250/250_200
# 은 try1·fail 둘 다 실패). 반면 A1 은 초근접 물체에서 입력 스케일에 대해 사실상
# 카오스라, imgsz 를 32 바꾸는 것만으로 conf 가 0.068 → 0.824 로 뒤집힌다.
# 즉 **독립 시행을 늘리는 것**이 유효하다.
#
# 1순위가 640 이 아니라 480 인 이유 — 근접캠 프레임은 물체가 크므로 낮은 imgsz 가
# 학습 분포(imgsz=640/mosaic=1.0/scale=0.5)에 더 가깝다. 실기 approach near 33장
# 전수 실측 (conf>=0.35, 그리퍼 손가락 오검출 1건 제외):
#   | 사다리           | 실질 커버리지 | 1순위 적중 | 1회 추론 |
#   | 640 -> 480 -> 448 |    28/33     |   25/33    | 640: 81.1ms
#   | **480 -> 448 -> 640** |  28/33   | **27/33**  | 480: 57.3ms / 448: 66.4ms
# 커버리지는 같은데 480 이 1순위 적중이 2장 많고 매 접근이 24ms 빠르다.
# 둘 다 검출한 25장에서 박스 편차 중앙값 2px, conf 0.961 vs 0.949 로 문턱(0.35)
# 대비 무의미한 차이다.
# 640 은 3순위로 남긴다 — 커버리지 기여는 없지만(28/33 동일) 종전 동작을 사다리
# 안에 보존해 회귀 위험을 없앤다. 비용은 480/448 이 둘 다 실패한 5/33 에서만.
# 512 는 뺐다 — 커버리지 동일한데 화면 우하단 그리퍼 손가락을 octahedron 0.920
# 으로 오검출한다 (576 에서는 0.047 — 512 특유 패턴).
# 근거표: docs/07-results-and-lessons.md
A1_IMGSZ_NEAR_LADDER = (A1_IMGSZ_NEAR_APPROACH, 448, 640)

# [2026-07-23 조작자 승인] 스캔 하이브리드 — 스티치 @896 은 원거리만 쓰고,
# 근접(seam 아래)은 near 원본 프레임을 이 크기로 따로 추론해 대체한다. 0=끔.
#
# 왜 접근(480)과 값이 다른가 — 두 프레임의 물체 크기 분포가 2.2배 다르기 때문이다.
#   접근 near 프레임: 원본 물체 높이 중앙값 537px (초근접 물체 하나가 화면을 채움)
#   스캔 near 프레임: 원본 물체 높이 중앙값 241px (근접~원거리 혼재, p10 157/p90 291)
# 접근은 큰 물체 하나만 분포 안에 넣으면 되니 작게(480 → 네트 134px).
# 스캔은 더 낮추면 근접은 좋아지나 **원거리가 같이 작아진다** — 448 이면 p10 이
# 네트 37px 까지 내려가 소물체 검출이 무너지기 시작한다. 512 는 p10 42px 을
# 지키면서 근접 물체도 상한(네트 ~150px) 아래에 둔다.
# 실측(031540/021919 2런, 확정셀): baseline 7/6 → 448 9/8 → 480 9/8 →
#   **512 10/8** → 640 8/8. 유령셀 감소, snap_err 동등, 클래스갈림 0.
# 비용: 샷당 ~89ms → 12샷 스캔 +1.1s. 근거표: docs/07-results-and-lessons.md
# [롤백 취소 2026-07-23 06시] 054834 의 "이상한 곳에 bbox" 는 검출이 아니라
# **오버레이 그리기** 문제였다 — near 원본 검출의 box/u/v 는 near 프레임
# (1920x1080) 좌표인데 그걸 스티치(1908x2044) 위에 그대로 그렸다. 맵 좌표는
# u_src/v_src + 카메라별 intrinsics 라 영향 없었다. 아래 draw_poly/draw_uv 로
# 스티치 좌표를 따로 실어 보내 해결했다.
A1_IMGSZ_SCAN_NEAR = 512

# [2026-07-24 조작자 지시] 스티치/near 원본 표를 가르는 near 프레임 세로 문턱
# (near 높이 대비 비율). 종전은 "seam 아래면 무조건 near 원본"(= 1.0 상당)이라
# seam 바로 아래(=near 프레임 최상단)에 걸친 물체가 near 원본에서 **윗면이
# 프레임 밖으로 잘려** 과일면을 통째로 잃었다. 자세한 근거·실측은 스왑
# 블록(_process_shots_batch) 주석 참조. 근거 문서 docs/07-results-and-lessons.md.
NEAR_STITCH_SPLIT = 0.5
# 유지한 스티치 표와 같은 물체를 가리키는 near 원본 표를 제거할 때의 매칭
# 반경 [near 프레임 px]. 실측 대응쌍 15건 거리 7.0~40.5px.
NEAR_DUP_PX = 120.0
# [2026-07-24 조작자 지시] 스캔 near 원본의 **잘린 조각 검출 기각**.
#
# 근거(012206 실기 shot06): near box [1661,923,1919,1078] / 프레임 1920x1080 —
# 우변·하변 **2변**에 걸친 모서리 파편이라 큐브의 온전한 면이 하나도 안 잡혔고,
# face 모델이 그 무지 조각을 `pineapple 0.904` 로 환각 → (200,200) 이 파인애플로
# 확정됐다. 종횡비(h/w 0.60 vs 진짜 과일 최소 0.62)로는 분리 불가로 확인돼
# **기하(경계 접촉)** 로 가른다.
#
# ⚠ 적용 범위는 **스캔 near 경로 한 곳뿐**이다(_process_shots_batch 스왑 블록).
# 접근/파지 직전 near 추론(collect 경로)에는 절대 걸지 않는다 — 거기선 대상이
# 코앞이라 프레임을 정상적으로 채우며 경계에 닿는 게 당연하고, 기각하면 파지가
# 통째로 죽는다.
#
# ⚠ 미검증 항목: 리포트 det 에 box 가 저장되지 않아 전 런 코퍼스 통계가 없다.
# 양성 사례는 위 1건뿐. MIN_EDGES=2(모서리 파편)로 보수적으로 시작한다 —
# 1 로 낮추면 하변만 걸친 정상 근접 물체까지 날아갈 수 있다. 되돌리려면
# NEAR_CLIP_DROP=False 한 줄. 영향은 shot["near_clip"] 카운트로 추적한다.
NEAR_CLIP_DROP = True
NEAR_CLIP_MARGIN_PX = 3     # 경계에서 이 이내면 '닿음'
NEAR_CLIP_MIN_EDGES = 2     # 몇 변 이상 닿아야 '조각'으로 보고 기각하는가


def clip_edge_count(box, w: int, h: int, margin: int = NEAR_CLIP_MARGIN_PX) -> int:
    """bbox 가 프레임 4변 중 몇 변에 닿았는지. 잘린 조각 판정용 (2026-07-24)."""
    x1, y1, x2, y2 = box
    return (int(x1 <= margin) + int(y1 <= margin)
            + int(x2 >= w - 1 - margin) + int(y2 >= h - 1 - margin))
# [2026-07-23] 디버그 이미지 저장 포맷. 종전 PNG(level6)는 스티치 1장에
# **3167ms** 가 걸려, daemon 저장 스레드가 수거 단계와 겹치며 코어 1~2개를
# 96초간 태웠다(021919 실기: 74장 130MB, 마지막 shot06_near.png 는 프로세스
# 종료에 잘려 truncated PNG 가 됨). JPEG q95 는 62ms(51배) / 0.92MiB(3.7배).
# subsampling=0(4:4:4) 는 필수 — 크로마 서브샘플링은 색을 뭉개서 나중에
# apple/orange 면 판정을 오프라인 분석할 때 방해가 된다.
# 검출 패리티 실측(실기 스티치 3장, A1 imgsz=896): 검출 수·클래스 전부 동일,
# Δconf 최대 0.0013, Δbox 최대 0.1px — 이 저장소의 TensorRT 패리티 허용치
# (CONF_TOL 0.06 / BOTTOM_PX_TOL 4.0px)의 1/40 수준이라 A/B 신뢰도 영향 없음.
# depth(uint16)는 JPEG 불가 → PNG 유지, compress_level 만 1로 (무손실 유지).
JPEG_OPTS = dict(format="JPEG", quality=95, subsampling=0)

# [2026-07-23 조작자 지시] GT 대조 수치를 성능 지표로 쓰지 말 것.
# 현재 운용에서 물체는 아레나에 **무작위로 뿌려두고**, GT 는 사람이 **눈으로 보고**
# 입력한다. 즉 GT 자체의 셀 좌표·클래스가 부정확하다. 그래서 "정답 2/28" 같은
# 숫자는 인식 성능이 아니라 대부분 **GT 입력 오차**를 재고 있다.
# 지도 품질은 GT 대신 수거 결과(적재 수 / 추정 점수 / 접근 실패 사유)로 판단한다.
# 이 배너는 리포트를 나중에 읽는 사람이 숫자를 오해하지 않도록 매 런 출력한다.
GT_UNRELIABLE_REASON = (
    "GT 는 물체를 무작위 배치 후 사람이 육안으로 입력한 값이라 좌표·클래스가 "
    "부정확하다. 이 대조 수치는 인식 성능 지표가 아니며 분석에서 제외한다 "
    "(2026-07-23 조작자 지시). 지도 품질은 수거 결과로 판단할 것."
)
GT_UNRELIABLE_BANNER = (
    "\n" + "!" * 70 + f"\n⚠ GT 대조는 참고용 — 성능 지표 아님\n  {GT_UNRELIABLE_REASON}\n"
    "  자세한 설명과 대체 지표: docs/07-results-and-lessons.md\n" + "!" * 70)

MAX_SNAP_ERR_M = 0.30        # 90번과 동일 격자 스냅 게이트
FRUIT_TIE_CONF_MARGIN = 0.10  # 과일표 동률 시 face_conf 합 우위가 이 이상이면 채택
# ---- AO 동수 갈림표 = apple 확정 [2026-07-24 조작자 지시] --------------------
# 예선 2경기 실측: 과일표가 apple/orange 로 **동수**로 갈린 셀은 전수 apple 이
# 정답이었다. 오독이 **apple→orange 단방향**(orange 를 apple 로 읽은 사례 0)이라
# "동수 갈림 = apple 을 한 번 놓친 것"으로 해석하는 편이 옳다. 그래서 conf 우열을
# 보지 않고(=conf 합 타이브레이크보다 **먼저**) apple 로 확정한다.
# 근거·적용범위·되돌리는 법: perception/docs/grid-voting.md
AO_TIE_PAIR = frozenset(("apple", "orange"))
AO_TIE_PREFER = "apple"      # None 으로 두면 종전 동작(conf 합 → conflict) 복원
# ---- 그리퍼(로봇 자체) 오검 기각 [2026-07-23 조작자 제안 · 15:45 실기 실측] ----
# 실측 근거는 사용처(_process_shots_batch) 주석. 양쪽 마진이 커서 튜닝 불필요.
GRIPPER_DARK_V = 90          # 이 밝기(max RGB) 미만이면 "검은 픽셀"
GRIPPER_DARK_FRAC = 0.40     # 박스 내 검은 픽셀 비율이 이 이상이면 기각
#   실측: 그리퍼 0.75 / 사과 스티커 0.001 / 큐브 0.000 / 바닥 0.000
GRIPPER_NEAR_RANGE_M = 0.40  # near 캠에서 이보다 가까운 검출에만 conf 하한 적용
GRIPPER_NEAR_CONF_MIN = 0.60 # 실측: 유령 0.283/0.369 vs 정상 최소 0.637
# [2026-07-23] pair 검증기 라우트별 on/off. 종전 --pair 는 on/off 뿐이라
# AO/BP 를 한꺼번에 켜고 껐다. 154504 실기 리플레이에서 **BP 가 정답 banana
# (face conf 0.97~0.98)를 pineapple conf 1.0 으로 뒤집어** 2셀을 잃은 것이
# 확인돼(교체 banana→pineapple 6건 vs 역방향 1건) BP 만 끌 수 있게 분리한다.
# AO 는 같은 리플레이에서 약한 사과 셀을 apple 1.0 으로 되돌리는 등 순효과가
# 해롭지 않아 유지. 근거: perception/docs/models.md
PAIR_ROUTES = {"on": {"AO", "BP"}, "ao": {"AO"}, "bp": {"BP"}, "off": set()}
STANDOFF_M = 0.45            # 셀 앞 정지 지점 (셀에서 로봇 쪽으로)
FAST_HOP_THRESHOLD_M = 0.50  # 초과 측정 시 방어 홉 후 재측정
MEASURE_TRIES = 6            # 접근 측정 재시도 상한 (최초/홉 후 공통).
                             # 성공 시 즉시 탈출이라 비용은 전패 셀에서만
                             # ~0.6s/회. ⚠ 6은 검증값이 아니라 초기 추정 —
                             # report cycles[].measure.attempt/hop_attempt
                             # 분포가 쌓이면 재검토할 것 (전부 1~3회차
                             # 성공이면 3으로 축소; 2026-07-21 합의).
HOP_LAND_M = 0.40            # 홉 착지: 물체까지 이 거리를 남기고 전진.
                             # '절반 전진'(구)은 ~0.25m 에 착지 — 근접캠
                             # (전방 0.187/높이 0.314/틸트 53.2°, V-FoV 42°)
                             # 하단 시야 밖이라 재측정 5/5 전패 (7/20).
                             # 0.40m 는 실측 성공 측정이 몰린 0.36~0.51 중앙.
# 속도 프로파일: 경기 시간(180s) vs 안정성 스윕용. 실기 최대 ~1.0m/s 실측,
# 최종 클램프는 real.yaml(0.9/0.6/2.0)+펌웨어. 회전은 position 프로파일이라
# per-move 지정 불가(real.yaml position_* 로 튜닝).
SPEED_PROFILES = {
    #        주행(cruise) 접근(approach) 적재(place)
    "safe":   dict(cruise=0.5, approach=0.4, place=0.3),
    "normal": dict(cruise=0.8, approach=0.6, place=0.4),   # 기존 기본값
    "fast":   dict(cruise=0.95, approach=0.7, place=0.45),  # 실측 한계의 ~95%
}
GRASP_WAIT_S = 2.5
# [2026-07-23 조작자 지시] 파지 직전 face 재검증 추론의 시간 상한 [s].
# 초과하면 결과를 버리고 그대로 파지한다 (보조 게이트 — 지연이 손해).
VERIFY_TIMEOUT_S = 0.5
# [폐기 2026-07-23 저녁 — held 게이트 부활로 미사용. 게이트를 다시 끌 때 복원]
# 게이트 없던 오전 한나절의 CLOSE 관측 대기 상한 [s]. 판정을 안 쓰므로 길게
# 기다릴 이유가 없고, 하한만 손가락 물리 닫힘(~0.6s)의 두 배로 잡았다.
# 게이트 ON 인 지금은 wait_held() 기본값(GRASP_WAIT_S)을 쓴다 — 브리지 판정
# 래치가 CLOSE+0.85s 라 실제로는 그 시점에 조기 반환하고, 2.5s 는 피드백이
# 죽었을 때만 소진되는 상한이다.
GRASP_OBSERVE_S = 1.2
STORAGE_RETREAT_M = 0.35     # 63번/전략 계약
PLACE_PUSH_M = 0.025         # [2026-07-22 04시 조작자 지시] 적재 전진을 슬롯
                             # 목표보다 이만큼 더 밀어 넣는다("갖다 박기") —
                             # 언더슛으로 물체가 보관함 턱에 걸치는 것 방지.
                             # 과진입 스톨은 기존 timeout→후퇴 경로가 흡수.
# [2026-07-23 조작자 지시 +5cm,+5cm] (-1.45,-1.45) → (-1.40,-1.40).
# 적재 진입 전 걸치는 점(운반 드리프트 종점). 종전 값은 보관함 쪽으로 5cm 더
# 깊어, 드리프트로 비스듬히 들어오는 동안 문 물체가 경기장 림에 걸리는 경우가
# 잦았다(조작자 관측). 최종 적재 위치는 슬롯(핀) 좌표가 정하므로 이 값은
# 접근 경로만 바꾼다 — 물체가 놓이는 자리는 불변.
STAGING_XY = fl.official_cm_to_map(60.0, 60.0)   # 보관함 앞 스테이징 (공식 60,60)

# ---- 보관함 적재 슬롯: 볼링핀 배치 (7/20 재설계) ----
# 로봇은 적재 내내 좌하 구석(45도)을 정면으로 보고 진입한다. 그리퍼 길이를
# 감안했을 때 공식좌표 (30,30)cm 가 로봇이 들어갈 수 있는 가장 안쪽 지점 —
# 이건 *로봇 기준점*이지 물체가 놓이는 자리가 아니다. 여기를 볼링핀 1번으로
# 두고 구석 반대쪽(대각)으로 행을 늘리며 핀을 추가한다.
STORAGE_CORNER_YAW = math.radians(-135.0)   # 좌하 구석 정면 (-x,-y)
PIN1_CM = (30.0, 30.0)          # 핀 1번 = 로봇 목표 (물체 위치 아님)
PIN_ROW_STEP_CM = 12.0          # 대각(구석 반대) 행 간격
PIN_LAT_STEP_CM = 13.0          # 대각에 수직인 열 간격
# 룰: 위에서 봤을 때 물체가 보관함 40x40 밖으로 돌출되면 미인정.
# 물체는 8cm → 중심이 [4, 36]cm 안에 있어야 한다(0.5cm 여유).
STORAGE_BOX_CM = 40.0
OBJ_HALF_CM = 4.0

# ---- 경량 HUD 연동 (2026-07-22, 2줄 프로토콜로 확장 2026-07-23) ----
# 구 HUD(scripts/match_hud_display.py)는 매 프레임 렌더 + PNG 재인코딩으로
# 코어 1개를 영구 점유해 sllidar 스케줄링을 밀어냈다(프레임당 374ms 실측).
# 신규 scripts/match_hud_lite.py 는 이 파일 2줄만 저주기(0.1Hz)로 읽고,
# 띄울 것이 바뀔 때만 미리 인코딩된 사진을 디스플레이에 올린다.
#   1줄: 이미지 키 = <클래스><quota 순번> (예 apple2) 또는 start/end
#   2줄: 현재 예상점수 정수 (POINTS 기준). HUD 는 점수가 오른 것을 처음 본
#        틱에서만 points<점수> 를 1회 띄우고 다음 틱에 목표물체로 돌아간다.
# quota 순번을 붙이는 이유: 룰북 quota(과일 3 / 다면체·cube 4)에 맞춰 사진이
# 클래스당 3~4장 준비돼 있고, "지금 그 클래스의 몇 번째를 노리는가"가 곧
# 사진 번호다. 파지 실패로 재시도하면 번호가 그대로라 같은 사진이 유지된다.
# 기본 경로는 match_hud_lite.py 의 DEFAULT_STATE_FILE 과 같아야 한다
# (다르게 쓰려면 양쪽 모두 HUD_STATE_FILE 환경변수로 지정).
# 계약 문서: docs/05-field-runbook.md
HUD_STATE_FILE = Path(os.environ.get("HUD_STATE_FILE", REPO_ROOT / "logs/hud/target"))


def hud_write(key: str, score: int = 0) -> None:
    """HUD 상태 파일 기록. HUD 미사용/실패해도 경기에 영향 없음 (예외 삼킴)."""
    # [2026-07-24 — 조작자 지시] HUD 미사용 → 상태 파일도 쓰지 않는다. HUD 가
    # 안 뜨는 이상 이 기록은 소비자가 없다. HUD_AUTOSTART=True 로 되돌리면
    # 종전대로 매 파지/적재마다 기록한다 (호출부 4곳은 그대로 둔다).
    if not HUD_AUTOSTART:
        return
    try:
        HUD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = HUD_STATE_FILE.with_name(HUD_STATE_FILE.name + ".tmp")
        tmp.write_text(f"{key}\n{int(score)}\n")
        tmp.replace(HUD_STATE_FILE)   # 원자적 교체 — HUD 가 부분 기록을 읽지 않게
    except OSError:
        pass


# [2026-07-24] HUD 자동 브링업 — 종전엔 조작자가 별도 터미널에서 match_hud_lite
# 를 직접 띄워야 했고, 잊으면 경기 중 디스플레이가 죽어 있었다. 러너가 직접
# 자식 프로세스로 띄우고 종료 시 같이 내린다. 실패해도 경기에는 영향 없음
# (HUD 는 표시 전용). 끄려면 --no-hud.
#
# [2026-07-24 — 조작자 지시] HUD 미사용 확정 → 자동 기동 전면 중단.
# False 인 동안 --no-hud 없이 돌려도 match_hud_lite 는 뜨지 않는다. 되살리려면
# 이 값만 True 로 되돌리면 종전 동작(러너가 자식으로 기동)이 그대로 복원된다.
# 코드/인자(--no-hud, start_hud, hud_write)는 폐기하지 않고 그대로 남긴다.
HUD_AUTOSTART = False
HUD_LITE = REPO_ROOT / "scripts" / "match_hud_lite.py"
HUD_CACHE_DIR = REPO_ROOT / "hardware" / "hud" / "cache"
_hud_proc: subprocess.Popen | None = None


def hud_running() -> bool:
    """이미 떠 있는 match_hud_lite 가 있으면 True (중복 기동 → USB 충돌 방지)."""
    try:
        r = subprocess.run(["pgrep", "-f", "match_hud_lite.py"],
                           capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def start_hud(out_dir: Path) -> None:
    """경량 HUD 를 백그라운드로 기동. 캐시가 없으면 --build 를 먼저 돌린다.

    전체가 별도 스레드에서 돌아 러너 기동을 막지 않으며, 어떤 실패도 삼킨다.
    """
    def _worker():
        global _hud_proc
        try:
            if not HUD_LITE.exists() or hud_running():
                return
            log = open(out_dir / "hud_lite.log", "ab", buffering=0)
            if not any(HUD_CACHE_DIR.glob("*.*")):
                print("HUD 캐시 없음 — 사진 캐시 생성 중 (1회)")
                subprocess.run([sys.executable, str(HUD_LITE), "--build"],
                               stdout=log, stderr=subprocess.STDOUT, timeout=300)
            _hud_proc = subprocess.Popen(
                [sys.executable, str(HUD_LITE)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(f"HUD 기동 (pid {_hud_proc.pid}, 로그 {out_dir/'hud_lite.log'})")
        except Exception as e:  # noqa: BLE001
            print(f"HUD 기동 실패 (무시): {e}")

    threading.Thread(target=_worker, daemon=True).start()


# HUD 는 러너 종료 후에도 남긴다 (start_new_session): 경기 종료 화면(`end`)이
# 0.1Hz 틱으로 올라올 시간을 주고, 정상 상태 CPU 가 사실상 0 이라 상주 비용이
# 없다. 다음 런은 hud_running() 으로 이미 떠 있는 것을 재사용한다.
# 수동 종료: pkill -f match_hud_lite.py


def storage_pins():
    """볼링핀 순서의 적재 슬롯. [(로봇목표_맵m, 예상물체_cm)] 와 탈락분.

    물체는 로봇보다 그리퍼 길이(GRIP_FORWARD_M)만큼 구석 쪽에 놓이므로,
    그 예상 위치가 보관함 밖으로 나가는 핀은 애초에 후보에서 뺀다.
    """
    s = math.sqrt(0.5)
    lo, hi = OBJ_HALF_CM + 0.5, STORAGE_BOX_CM - OBJ_HALF_CM - 0.5
    rows = [(r, [(i - r / 2.0) * PIN_LAT_STEP_CM for i in range(r + 1)])
            for r in range(3)]
    # 삼각형이 40cm 상자에 다 안 들어가는 경우를 대비한 깊이 0 좌우 확장
    rows.append((0, [-PIN_LAT_STEP_CM, PIN_LAT_STEP_CM]))
    keep, dropped = [], []
    for r, lats in rows:
        for lat in lats:
            rx = PIN1_CM[0] + s * (PIN_ROW_STEP_CM * r + lat)
            ry = PIN1_CM[1] + s * (PIN_ROW_STEP_CM * r - lat)
            ox = rx - s * fl.GRIP_FORWARD_M * 100.0   # 물체는 구석 쪽으로
            oy = ry - s * fl.GRIP_FORWARD_M * 100.0
            item = (fl.official_cm_to_map(rx, ry), (ox, oy))
            (keep if lo <= ox <= hi and lo <= oy <= hi else dropped).append(item)
    return keep, dropped


STORAGE_PINS, STORAGE_PINS_DROPPED = storage_pins()
TARGET_GATE_M = 0.6          # 접근 중 예상 셀과 이보다 먼 검출은 오타겟으로 배제
# TARGET_GATE_FIRST_M = 0.25  # [폐기 2026-07-23 — 실험 기능 토글 전면 제거]
#                             # 첫 측정 반 칸 게이트. 실기에서 한 번도 켜지 않았고
#                             # (07-23 실런 전량 features=[]), 오타겟 방지는 격자
#                             # 스냅 검증(상시)이 담당한다. 게이트는 항상 0.6m.

# ---- [폐기 2026-07-23 — 조작자 지시 "실험 기능 물어보는 거 다 빼자"] ----
# 07-22 밤의 A/B 토글(EXP_FEATURES / parse_features / prompt_features / --features)
# 전부 제거. 세 기능의 귀착점:
#   snap  (격자 스냅 검증)   → 07-23 00시 상시 승격
#   south (남단 행 예외)     → 07-23 04시 "운반만 즉시 드리프트"로 재정의·상시
#   gate  (첫 측정 0.25m)    → 미채택 (실기 미사용, 위 주석 참조)
# 경기 당일 구성은 고정이며 선택 UI 가 없다. 실행은 scripts/run_match_day.sh.
# 코리도 팽창: 물체 반경 여유 + 로봇 반경 + 투표수 반비례 불확실성
CORRIDOR_BASE_M = 0.10
ROBOT_RADIUS_M = 0.16
VOTE_UNCERT_M = 0.05
DETOUR_CLEAR_M = 0.5
ARENA_CLAMP_M = fl.ARENA_HALF_M - 0.1   # 벽에서 10cm 안쪽까지만 목표 허용
PRESENCE_OBSTACLE_MIN = 2    # far presence 셀은 이 관측수 이상일 때만 장애물


# 배치 forward 청크 크기. 7/20: 스캔 65샷을 1배치로 넣다가 Orin Nano 8GB
# 통합메모리가 고갈(NvMap error 12 → CUDACachingAllocator assert)해서 도입.
# stitched 해상도 activation이 커서 A1은 작게, 224 크롭인 face는 크게 잡는다.
# 적재 투하 시 개방량. 전체 OPEN(real.yaml open_deg=214.37)은 보관함 벽에
# 손가락이 닿아 7/20 실기에서 문제 — 물체를 놓을 만큼만 "살짝" 벌린다.
# 절대각이 아니라 *현재 위치 기준 상대 틱*: 물체를 물고 있으면 손가락이
# closed_deg 가 아니라 물체 폭에서 멈춰 있어 절대각 지정이 의미가 없다.
# XC330 = 4096 tick / 360deg.
DXL_DEG_PER_TICK = 360.0 / 4096.0
PLACE_RELEASE_TICKS = 450.0        # ≈39.6deg 만큼만 벌림 (--release-ticks)
# 마스트 이동 완료 시간 상한(s) — 실측(상승 ~7s / 하강 ~6s). MOVING=0 폴링
# 판정과 OR 로 묶어 둘 중 먼저 성립하는 쪽에서 끊는다. 7/20: 폴링 단독이
# 상승 시간을 13.5s로 과대평가하던 문제.
MAST_UP_ASSUME_S = 7.0
MAST_MID_ASSUME_S = 4.0    # 스트로크 절반 — 실측 전 추정 (풀업 7s 의 ~절반+여유)
MAST_DOWN_ASSUME_S = 6.0
# 적재 전진/후퇴 이동의 응답 대기(s). 기본 40s에서 상향 — 보관함 구석까지
# 밀어넣는 저속(v_place) 이동이 40s 안에 DONE을 못 받는 경우 대비 (7/20).
PLACE_MOVE_TIMEOUT_S = 90.0
SCAN_SETTLE_S = 0.35   # 스캔 회전 후 정착 대기(s). 7/20: 0.6 → 0.35 (--scan-settle)
A1_BATCH_CHUNK = 8
FACE_BATCH_CHUNK = 32


def _chunked_predict(model, items: list, chunk: int, **kw) -> list:
    """items를 chunk 단위로 나눠 predict하고 결과를 이어붙인다."""
    out = []
    for i in range(0, len(items), chunk):
        out.extend(model.predict(items[i:i + chunk], **kw))
    return out


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# =========================================================================
# 모델 프리로드 — 프로그램 시작과 동시에 백그라운드 기동 (main()에서 start).
# ultralytics import(~4s)+로드+CUDA 워밍업(~6s)이 GT 입력·헬스체크(8s)·
# 주행과 완전히 겹치므로 스캔 지점 도착 시점엔 이미 준비돼 있는 게 보통.
# =========================================================================
_PRELOAD = {"thread": None, "models": None, "pair": None,
            "err": None, "pair_err": None, "warmup_warn": None, "sec": None,
            "a1_backend": None}


def _preload_models(pair_on: bool):
    t0 = time.monotonic()
    try:
        from ultralytics import YOLO
        for w in (fl.A1_WEIGHTS, fl.FACE_WEIGHTS):
            if not Path(w).exists():
                raise FileNotFoundError(f"weight 없음: {w}")
        # [2026-07-24] .pt 하드코딩 제거 — 파일명 규약을 만족하는 TensorRT
        # 엔진이 있으면 그걸 쓰고, 조금이라도 조건이 어긋나면 .pt 로 떨어진다
        # (fl.select_infer_weights 주석 참조). A1 은 스티치 896 · near 512 를
        # 모두 쓰므로 둘 다 지원하는 엔진만 채택된다.
        a1_path, a1_why = fl.select_infer_weights(
            fl.A1_WEIGHTS, A1_BATCH_CHUNK,
            (fl.A1_IMGSZ_STITCHED, A1_IMGSZ_SCAN_NEAR or fl.A1_IMGSZ_STITCHED))
        _PRELOAD["a1_backend"] = a1_why
        a1 = YOLO(str(a1_path))
        # face 는 seg 재학습 예정이라 엔진을 만들지 않는다 (2026-07-24 결정)
        face = YOLO(str(fl.FACE_WEIGHTS))
        try:  # 워밍업 (첫 추론의 CUDA 초기화 비용 선지불)
            a1.predict(np.zeros((fl.A1_IMGSZ_STITCHED, fl.A1_IMGSZ_STITCHED, 3),
                                dtype=np.uint8),
                       imgsz=fl.A1_IMGSZ_STITCHED, conf=0.5, verbose=False)
            face.predict(np.zeros((224, 224, 3), dtype=np.uint8),
                         imgsz=fl.FACE_IMGSZ, conf=0.5, verbose=False)
        except Exception as e:  # noqa: BLE001 — 워밍업 실패는 치명 아님
            _PRELOAD["warmup_warn"] = str(e)
        _PRELOAD["models"] = (a1, face)
    except Exception as e:  # noqa: BLE001
        _PRELOAD["err"] = str(e)
    if pair_on:
        try:
            _PRELOAD["pair"] = fl.PairVerifier()
        except Exception as e:  # noqa: BLE001 — 검증기 없으면 face 단독
            _PRELOAD["pair_err"] = str(e)
    _PRELOAD["sec"] = time.monotonic() - t0


def start_model_preload(pair_on: bool):
    if _PRELOAD["thread"] is not None:
        return
    th = threading.Thread(target=_preload_models, args=(pair_on,), daemon=True)
    _PRELOAD["thread"] = th
    th.start()


# =========================================================================
# depth 역투영 수식 (90번 포팅 — 원본 캠 프레임 전용, tilt/forward만 사용)
# =========================================================================
def mask_bottom_uv(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    bottom = ys.max()
    return float(xs[ys >= bottom - 1].mean()), float(bottom)


def _near_to_stitch(st, pts):
    """near 원본 픽셀 → 스티치 픽셀 (오버레이 표시 전용).

    [2026-07-23] 스티처의 A 는 near→top프레임 호모그래피이고, 스티치 좌표는
    top 프레임에서 left 만큼 왼쪽으로 민 것이다. 3D 계산에는 쓰지 않는다
    (그건 원본 캠에서만 — Stitcher 도크스트링 계약)."""
    out = []
    for u, v in pts:
        q = st.A @ np.array([float(u), float(v), 1.0])
        out.append((float(q[0] / q[2] - st.left), float(q[1] / q[2])))
    return out


def _dark_fraction(img: np.ndarray, box) -> float:
    """검출 박스 안에서 "검은" 픽셀이 차지하는 비율. [2026-07-23]

    밝기는 채널 최댓값(=HSV 의 V)으로 잰다 — 채널 순서(BGR/RGB)에 무관하므로
    스티치(BGR)와 near 원본(RGB) 어느 쪽을 넘겨도 같은 값이 나온다.
    게임 물체는 전부 무광 흰색이고 그리퍼만 검정이라 이 한 값으로 갈린다.
    박스가 비었거나 좌표가 이미지 밖이면 0.0 (= 기각 안 함, 보수적).
    """
    try:
        x1, y1, x2, y2 = (int(v) for v in box)
    except (TypeError, ValueError):
        return 0.0
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float((patch.max(axis=2) < GRIPPER_DARK_V).mean())


def robust_depth_median(depth_mm: np.ndarray, u: float, v: float,
                        radius: int | None = None):
    """접점 주변 패치의 depth 중앙값 (mm→m).

    radius 기본값은 해상도 비례(640폭 기준 3px 상당) — aligned depth 는 컬러
    해상도로 업샘플되어 나오므로, FHD 에서 radius 3 이면 각도상 1/3 범위만
    덮고 업샘플 구멍에 취약해진다 (2026-07-21 해상도 전환 대응)."""
    h, w = depth_mm.shape[:2]
    if radius is None:
        radius = max(3, int(round(3 * w / 640.0)))
    u0, v0 = int(round(u)), int(round(v))
    patch = depth_mm[max(0, v0 - radius):min(h, v0 + radius + 1),
                     max(0, u0 - radius):min(w, u0 + radius + 1)].astype(np.float64)
    valid = patch[patch > 0]
    return float(np.median(valid)) / 1000.0 if valid.size else None


def pixel_depth_to_robot_xy(u, v, depth_m, intr: Intrinsics, mount: CameraMount):
    x_cam = (u - intr.cx) / intr.fx * depth_m
    y_cam = (v - intr.cy) / intr.fy * depth_m
    tilt = math.radians(mount.tilt_deg)
    y_fwd = depth_m * math.cos(tilt) - y_cam * math.sin(tilt) + mount.forward_m
    # [2026-07-23] x_cam 은 **광축 기준** 좌우인데 종전엔 이를 그대로 로봇
    # 중심선 기준 x_right 로 반환했다 = 카메라가 중심선 위에 있다는 가정.
    # mount.lateral_m(+가 우측)만큼 더해 로봇 기준으로 옮긴다. 기본값 0.0
    # 이라 실측 주입 전까지 동작은 종전과 동일하다 (CameraMount 주석 참조).
    return x_cam + mount.lateral_m, y_fwd


def to_map(x_right, y_fwd, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (pose[0] + y_fwd * c + x_right * s, pose[1] + y_fwd * s - x_right * c)


# =========================================================================
# 코리도 라우터 (순수 함수 — --offline 유닛 테스트 대상)
# =========================================================================
def inflation_for(votes: int) -> float:
    return CORRIDOR_BASE_M + ROBOT_RADIUS_M + VOTE_UNCERT_M * (1.0 / max(1, votes))


def point_seg_dist(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / l2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def corridor_free(p0, p1, obstacles, clearance=None):
    """obstacles=[{cell,xy,votes}] — 선분과 장애물 셀 중심 거리 < 팽창이면 blocked.

    clearance 를 주면 votes 기반 팽창 대신 그 값을 쓴다 (street 축정렬 주행용 —
    inflation_for 의 0.16m 는 로봇 **외접** 반경이라 임의 방향 주행 기준이고,
    street 는 진행축이 고정이라 반폭 0.095m 가 맞는 값이다).
    반환 (free: bool, blocker: dict|None).
    """
    for ob in obstacles:
        lim = inflation_for(ob["votes"]) if clearance is None else clearance
        if point_seg_dist(ob["xy"], p0, p1) < lim:
            return False, ob
    return True, None


def path_free(pts, obstacles, clearance=None):
    """연속 경유점 전체가 클리어인지. (free, 첫 막힌 blocker) 반환."""
    for a, b in zip(pts[:-1], pts[1:]):
        free, blk = corridor_free(a, b, obstacles, clearance)
        if not free:
            return False, blk
    return True, None


# ---- street(격자 사이 통로) 격자 ----
# 물체는 격자점(공식 50cm 간격)에만 놓인다. 그 사이 25cm 지점을 잇는 선이
# street — 물체 반폭 4cm + 로봇 반폭 9.5cm = 13.5cm 이므로 25cm 통로면 안전.
# 대각선 주행은 격자점 위를 지나므로 금지.
# 공식 중앙선(x: 25+50k, y: 75+50k)에서 유도. 상한은 아레나 한 변을 따라간다.
_ARENA_SIDE_CM = int(round(fl.ARENA_HALF_M * 200.0))
STREET_XS_M = tuple(fl.official_cm_to_map(x, 0.0)[0]
                    for x in range(25, _ARENA_SIDE_CM - 24, fl.GRID_PITCH_CM))
STREET_YS_M = tuple(fl.official_cm_to_map(0.0, y)[1]
                    for y in range(75, _ARENA_SIDE_CM - 24, fl.GRID_PITCH_CM))
# 하단 하이웨이(공식 y<100 = 객체 없음) 중심 = 공식 y=75.
# ⚠ snv.HIGHWAY_Y_M(공식 60)과 **다른 값**이다 — 아래 HIGHWAY_FREE_Y_M 주석 참조.
HIGHWAY_Y_M = fl.official_cm_to_map(0.0, 75.0)[1]
# street 축정렬 주행의 측방 여유. 로봇 반폭 0.095 + 물체 반폭 0.04 = 0.135,
# 여기에 pose 오차 여유 0.065 → 0.20. street 통로 반폭 0.25 안에 들어간다.
# 일반 inflation_for(0.31, 외접반경 기준)를 쓰면 정상 street 도 전부 막힘 판정.
STREET_CLEAR_M = 0.20
# 이 선(공식 y=60cm) 남쪽은 물체가 없어 어떤 yaw로든 주행 가능 — 최남단
# 물체행 y=100 의 실루엣 하단 ~96cm, 로봇 외접반경 16cm → y<80 이면 비접촉.
# 60 은 보수 마진. 이 밴드 안에서 완결되는 레그는 yaw 정렬을 생략한다.
HIGHWAY_FREE_Y_M = fl.official_cm_to_map(0.0, 60.0)[1]
# [2026-07-22] street 주행의 하이웨이 라인(snv.HIGHWAY_Y_M)이 -1.80 → -1.40 으로
# 올라와 이 경계와 일치한다 (조작자 지시 — 사이클당 남하/북진 왕복 0.8m 단축).
# 기하 검증은 street_nav.HIGHWAY_Y_M 주석 참조 (그리퍼 선단 마진 10cm).
# street 레그 축정렬 허용 오차(°): 진행 방향이 로봇 4축(전/후/좌/우) 중 가장
# 가까운 축에서 이 이내면 무회전 (15°의 스윕 반폭 0.125+0.04 ≤ 0.20 안전).
ALIGN_TOL_DEG = 15.0
# ---- 하산 파지 [2026-07-23 신규 · 2026-07-24 전략 재정의 — 조작자 지시] ----
# 스캔점(공식 225,225)은 x=200 행과 x=250 행 사이 street 위다. 그 두 행의 타깃은
# mini-goal(남동 기본 / 남서 미러) 중 하나가 **같은 street 위**에 놓이므로,
# 하이웨이(-1.40)까지 내려갔다 같은 x 로 되올라오는 왕복이 통째로 낭비였다
# (스캔점 y=+0.25 기준 최대 왕복 ~3.3m ≈ 8s). 같은 street 면 한 레그로 붙인다.
#
# [2026-07-24 조작자 지시 — 적용 범위를 x=250 행으로 한정] 예외는 **미러
# mini-goal 이 걸리는 x=250 행에만** 둔다. x=200(기본 대각)은 폐기한다:
#   · 판정은 동쪽 부분맵(촬영 종료 +1s)으로만 할 수 있는데, x=200 열은 스캔
#     스윕의 화각 가장자리(스캔점 기준 방위 101~135°)라 한 샷밖에 안 걸린다.
#     votes_k 기본 3/2 에서 동쪽 맵에 x=200 은 **0셀**이었다 (7/23 실기 4런 전수).
#   · x=250 은 동쪽 스윕 한복판이라 매 런 3~5셀이 동쪽 단계에서 확정된다.
#   · 미러 mini-goal 은 적재함(남서) 쪽 street 라 파지 후 운반 드리프트도 0.5m
#     짧다 (기본 mini-goal x=+0.75 → 미러 x=+0.25).
DESCEND_STREET_TOL_M = 0.15   # 현재 x 와 mini-goal street x 의 허용 오차
# DESCEND_MIN_DY_M = 0.30  # [폐기 2026-07-24 — 롤백 시 이 값과 조건 복원]
#   "내려오면서"라는 표현을 그대로 옮긴 방향 제약이라 안전 근거가 없었는데,
#   ① mini-goal 이 스캔점과 일치하는 y=250 행(dy=0)과 ② 북쪽 두 행(dy<0)을
#   통째로 잘라냈다. 방향은 무의미하다 — drive_to 는 북향 요를 고정한 채
#   홀로노믹으로 움직여 전진=북·후진=남이 같은 코드 경로이고 통로도 동일하다.
#   대신 "로봇이 필드 안에 있을 때만"으로 바꾼다. 적재 후(y≈-1.45)나
#   하이웨이(-1.40)에서 켜지면 go_to_mini_goal 의 **복귀 드리프트** 분기를
#   건너뛰어(그 분기는 descend 조기 return 뒤에 있다) 7/22 에 없앤 제자리
#   135° 회전이 되살아난다. 적재 후퇴 종점 x 는 street x 와 0.20m 차라
#   ①(±0.15) 만으로는 못 막는다 — 이 조건이 그 방어다.
DESCEND_FIELD_MIN_Y_M = 0.30  # snv.HIGHWAY_Y_M 보다 이만큼 북쪽 = 필드 안
# 하산 제외 행 (공식 y) [2026-07-24 조작자 지시]. y=100 은 미러 mini-goal 이
# 공식 (225,75) = 남단 행이라 파지 후 운반이 "즉시 적재 드리프트"(south_row
# 예외)인데, 시작 yaw 가 미러 정면 +45° 라 적재 정면 -135° 까지 **180°** 를
# 이동 중에 돌아야 한다. 실기 검증된 것은 135°(표준)와 90°(남단 기본)뿐이고,
# 180° 는 wrap_angle 부호 경계라 시작 몇 틱의 회전 방향이 진동할 수 있다.
# 순이득도 x=250 중 가장 작다(~2.5s — 적재함에서 가도 어차피 가까운 행이라
# "지금 가면 싸고 나중에 가면 비싼" 정도가 가장 작다).
DESCEND_SKIP_ROWS = (100,)
# 하산 후보 정렬 = **무조건 가장 북쪽(y 최대) 먼저** [2026-07-24 조작자 지시].
# 근거: 첫 파지 시각은 어느 후보든 마스트 하강(~6s)에 걸려 동일하다 — 스캔
# 레그(0~1.3s)는 전부 그 안에 흡수돼 "공짜"고, 가까운 걸 골라도 첫 파지가
# 빨라지지 않는다(그냥 mini-goal 에서 더 기다린다). 순서에 따라 달라지는 것은
# **안 고른 물체를 나중에 적재함(남서 구석)에서 다시 가는 비용**뿐이고, 이는
# 북쪽일수록 비싸다(적재함 레그 y=150 2.9m → y=350 4.9m). 목적함수
#   f(C) = max(2 + s_C, M) − d_C     (s=스캔레그, d=적재레그, M=마스트초)
# 는 마스트가 느리면 −d_C(북쪽 최소), 빠르면 s−d(상수라 동률)라 **어떤 M 에서도
# 북쪽이 최적 또는 동률**이다. 종전의 "거리 우선"은 y=250(레그 0m)을 먼저 집어
# M=6 기준 ~1.3s / 경로 1.0m 손해였다. ⚠ "레그가 긴 것"이 아니라 "북쪽"이다 —
# y=150 도 레그 1.0m 로 길지만 남쪽이라 적재 레그가 짧아 최악의 선택이다.
# 제자리 회전 속도캡 [m/s 바퀴환산] — 미전달 시 position_max_rad_s 6.0에 묶여
# 본체 ~1.2rad/s (회당 2~9s, 7/21 실기 최대 낭비 항목). 0.35 → 본체 ~1.8rad/s.
# ⚠ 고속 회전은 슬립 증가 여지 — do_rotate의 잔차 검증 재회전이 흡수 (현장 확인).
# ROTATE_MAX_V = 0.245  # [폐기 2026-07-22 오후] 0.7배 하향분 — 실기 테스트에서
#                       # 0.35 원복해도 문제 없음 확인 (조작자 지시 롤백).
# ROTATE_MAX_V = 0.35   # [폐기 2026-07-22 저녁] 조작자 지시 "제자리 회전 1.5배"
ROTATE_MAX_V = 0.525    # = 0.35 x 1.5 (본체 ~2.7 rad/s. street_nav TURN_YAW_RATE
                        # 1.2 와 세트 — 과격 시 0.35 롤백)
# ---- 제자리 회전 정착 컷 [2026-07-23 조작자 지시] ----
# 03:15 실기: 적재 진입 잔차 회전(11.9°) 하나가 **5.89s** 를 태우고 실패했다
# ('firmware move timeout'). 펌웨어 데드라인은 프로파일 예상 x2 + 5.0s 라,
# 정착에 실패하면 소각 시간이 통째로 그 값이 된다. 적재 전진/후퇴에는 이미
# 1.0s 정착 컷(do_move_fire)이 있는데 회전에만 없어 생긴 구멍 — 같은 컷을 단다.
# 예상시간은 펌웨어와 같은 사다리꼴 프로파일로 계산한다(아래 상수는 real.yaml
# mecanum_bridge_node 와 동기 — 값이 바뀌면 여기도 갱신).
ROT_WHEEL_RAD_PER_YAW = 0.198 / 0.0388   # 본체 yaw 1rad 당 바퀴 회전 [rad]
ROT_ACCEL_RAD_S2 = 24.0                  # position_accel_rad_s2
ROT_WHEEL_RADIUS_M = 0.0388              # wheel_radius_m
ROT_OVERHEAD_S = 0.5                     # 직렬 왕복 + 명령 소화 실측 여유
ROT_SETTLE_CUT_S = 1.0                   # 예상시간 초과 후 이만큼만 더 기다린다
# 스캔 스핀 전용 컷 [2026-07-23 조작자 지시 "스캔 회전에 정착 컷 1.5초"].
# 근거: 05:43:54 실기의 4번째 스캔 회전이 **3.21s** (나머지 10회는 0.75~0.80s,
# 30° 프로파일 예상 0.67s) — 2.5s 가 통째로 펌웨어 정착 그라인드였다.
# 펌웨어는 네 바퀴 전부 잔차 ≤0.30rad(본체 3.4°)를 50ms 유지해야 DONE 을 내고,
# 실패하면 데드라인(프로파일x2+5.0s = 30°에서 6.3s)까지 갈린다. 튀는 스텝은
# 매번 다르다(03:15 런은 7번째 1.90s) — 각도가 아니라 확률이다.
# ⚠ 스캔 회전만 컷을 따로 두는 이유: 여기서 잘린 잔차는 **무해**하다.
#   각 샷은 회전 후 pose 의 실제 yaw 를 읽어 기록하고(cap["yaw_deg"]) 역투영도
#   그 pose 를 쓰므로, 25°만 돌아도 25°로 기록될 뿐 정확도 손실이 없다.
#   누적 부족분도 HFOV 69° 대비 무시 가능(컷당 ~1°, 전 스텝 잘려도 ~11°).
# SCAN_ROT_SETTLE_CUT_S = 1.5  # [폐기 2026-07-23 저녁 — 롤백 시 이 값 복원]
# SCAN_ROT_SETTLE_CUT_S = 0.5  # [폐기 2026-07-23 저녁 — 같은 지시로 0.3 까지]
# [2026-07-23 저녁 조작자 지시] 1.5 → 0.5 → **0.3**. 30° 기준 실효 timeout
# (= rotate_profile_sec 0.67 + ROT_OVERHEAD_S 0.5 + 컷) 2.67s → 1.67s → **1.47s**.
# 근거(max_v=0.525 적용 후 30° 회전 121건 / 11런 실측):
#   ≤0.85s(정상 정착) 111건 91.7%. 0.85s 초과 스텝 **전량**은
#   [1.22, 1.22, 1.61, 1.90, 1.90, 2.67, 2.67, 2.67, 3.21, 5.57] — 정상군과
#   그라인드군 사이가 비어 있는 이봉분포라 컷 위치를 잡기 쉽다.
#   → 정상 스텝은 0.75~0.80s 에 몰려 있고(p90 0.80s) timeout 1.47s 는 그
#     **1.8배**다. 정상 회전이 잘릴 여지는 없다 (1.5→0.5 때는 2.1배였다).
#   컷 발생률 4.1%(1.5) → 5.8%(0.5) → **6.6%(0.3)**, 회수는 런당
#   0.32s → 0.81s → **0.95s**. 0.5→0.3 의 순증은 런당 0.15s 뿐이다 —
#   긴 꼬리(3.21·5.57s)는 이미 컷 1.5 가 다 잡았고 이번에 새로 잘리는 건
#   1.61s 한 건뿐이라, 이 조정의 실이득이 작다는 점을 명시해 둔다.
# ⚠ 더 낮출 때 먼저 볼 것: 컷마다 비영 트위스트 선점 해제가 뒤늦은
#   move_result('preempted by cmd_vel')를 남긴다. 다음 회전이 그걸 자기 응답으로
#   오인하면 대기 없이 반환한다(= 회전 중 촬영 → 모션블러). 선점 응답 ~20ms 대
#   다음 pop 까지 0.45s+ 라 현재 컷 빈도에서는 안 걸리지만, 빈도가 오를수록
#   이 경로 노출도 함께 오른다.
SCAN_ROT_SETTLE_CUT_S = 0.3
# ---- 접근 단발이동 정착 컷 [2026-07-24 조작자 지시] ----
# 접근(카메라-depth→odom 상대이동)도 회전·적재와 같은 펌웨어 정착 그라인드를
# 탄다: 오늘 실기 접근 35건 중 4건이 [4.28, 4.30, 7.14, 7.27]s 로 튀었고 정상군
# 31건은 1.03~1.39s(move_profile_sec 예측과 ±0.3s). 정상 최대 1.39s와 그라인드
# 최저 4.28s 사이 ~2.9s 빈 골이 있어 컷 위치가 명확하다. do_approach_fire 가
# timeout = move_profile_sec(d, max_v) + APPROACH_SETTLE_CUT_S 로 끊는다.
# ⚠ 접근은 오컷 = 파지 실패(운반+적재 ~12s + 슬롯 손실)라 밀착 컷보다 보수적이어야
#   한다. 1.0s 면 관측 정상 35건 전부에 ≥0.96s 여유(오컷 0). 0.8s 까지 안전하나
#   회수 순증 런당 ~0.2s 뿐이라 컨벤션(ROT_SETTLE_CUT_S/do_move_fire) 통일값 유지.
# 근거·분포·회수: docs/07-results-and-lessons.md §7.5.
APPROACH_SETTLE_CUT_S = 1.0
# 스티치에 들어가는 top/near RGB 의 header.stamp 편차 경고 문턱 [2026-07-23].
# 두 카메라는 같은 순간을 노려 발행되므로 실측 지연차는 수 ms 여야 한다
# (스트림별 실지연 실측: top_rgb 0.107s / near_rgb 0.103s → 차 ~4ms).
# 15fps 한 프레임(67ms)의 1.5배를 넘으면 서로 다른 순간의 두 장을 붙인 것이다.
RGB_STAMP_SPREAD_WARN_S = 0.10


def rotate_profile_sec(dyaw: float, max_v: float) -> float:
    """펌웨어 사다리꼴 프로파일의 회전 소요 예측 [s] (startPositionMove 와 동식).

    실측 대조: 30° · max_v 0.525 → 예측 0.67s, 실기 0.76~0.77s (여유 0.5s 포함).
    """
    d = abs(float(dyaw)) * ROT_WHEEL_RAD_PER_YAW      # 바퀴 회전량 [rad]
    vmax = max(0.1, (max_v or 0.233) / ROT_WHEEL_RADIUS_M)
    t_acc = vmax / ROT_ACCEL_RAD_S2
    d_acc = 0.5 * ROT_ACCEL_RAD_S2 * t_acc * t_acc
    if 2.0 * d_acc >= d:                               # 삼각 프로파일
        return 2.0 * math.sqrt(d / ROT_ACCEL_RAD_S2)
    return 2.0 * t_acc + (d - 2.0 * d_acc) / vmax


# 펌웨어 위치이동(직진)의 순항 각속도 캡 [rad/s] — real.yaml mecanum
# position_max_rad_s. cmd_vel max_v(m/s)를 줘도 위치제어는 이 값에 묶여
# 본체 ~0.233 m/s(=6.0×wheel_radius)로 순항한다(2026-07-24 접근 35건 실측 일치).
POSITION_MAX_RAD_S = 6.0


def move_profile_sec(d_m: float, max_v: float) -> float:
    """펌웨어 사다리꼴 프로파일의 직진 소요 예측 [s] (rotate_profile_sec 의 선형판).

    위치이동은 position_max_rad_s(6.0)=0.233 m/s 에 캡되므로 max_v 는 그 아래로만
    유효하다. 실측 대조(2026-07-24 접근 35건): 예측이 실기와 ±0.3 s 안에서 일치
    (대부분 예측이 근소 과대 — 안전측). 근거·컷 설계는 docs/07-results-and-lessons.md §7.5.
    """
    dw = abs(float(d_m)) / ROT_WHEEL_RADIUS_M               # 바퀴 회전량 [rad]
    vmax = max(0.1, min(POSITION_MAX_RAD_S, (max_v or 0.233) / ROT_WHEEL_RADIUS_M))
    t_acc = vmax / ROT_ACCEL_RAD_S2
    d_acc = 0.5 * ROT_ACCEL_RAD_S2 * t_acc * t_acc
    if 2.0 * d_acc >= dw:                                   # 삼각 프로파일
        return 2.0 * math.sqrt(dw / ROT_ACCEL_RAD_S2)
    return 2.0 * t_acc + (dw - 2.0 * d_acc) / vmax


def descend_candidate(cell, pose) -> tuple:
    """(mirror, descend) — 이 셀의 mini-goal 에 **한 레그로** 붙을 수 있는가.

    [2026-07-24 조작자 지시 — 전략 확정] 순수 기하 판정이라 모듈 함수로 둔다
    (--offline 자가테스트가 로봇 없이 전수 검사할 수 있게 — 7/23 하루 종일
    하산이 0회 발동한 것을 아무도 못 잡은 원인이 이 테스트의 부재였다).

    조건 세 개:
      ① 로봇이 **필드 안**(하이웨이 + DESCEND_FIELD_MIN_Y_M 북쪽)에 있을 것.
         하이웨이 이남에서는 표준 경로(x 정렬 → 북진)가 이미 최단이고,
         복귀 드리프트를 건너뛰면 제자리 135° 회전이 되살아난다.
      ② 제외 행(DESCEND_SKIP_ROWS)이 아닐 것.
      ③ **미러** mini-goal 이 지금 서 있는 street 와 같은 x 일 것.
    ③이 미러 전용이라 스캔점(공식 225,225)에서는 **x=250 행만** 성립한다
    (미러 mini-goal x = 물체 x - 25 = 225 를 만족하는 열이 x=250 뿐).
    남/북은 묻지 않는다 — 상수 블록 DESCEND_MIN_DY_M 폐기 사유 참조.

    >>> descend_candidate((250, 200), (0.25, 0.25, 1.57))   # 스캔점
    (True, True)
    >>> descend_candidate((200, 200), (0.25, 0.25, 1.57))   # x=200 은 폐기
    (False, False)
    >>> descend_candidate((250, 200), (-1.45, -1.45, -2.36))  # 적재 후
    (False, False)
    """
    if pose is None:
        return False, False
    if pose[1] <= snv.HIGHWAY_Y_M + DESCEND_FIELD_MIN_Y_M:
        return False, False
    if int(cell[1]) in DESCEND_SKIP_ROWS:
        return False, False
    # [폐기 2026-07-24] 기본(남동) 대각 후보 — 롤백 시 `for mir in (False, True)`
    # 루프로 되돌리고 아래 판정을 그 안으로 옮긴다 (상수 블록 사유 참조).
    gx = snv.mini_goal_for(cell, mirror=True)[0]
    # 미러 mini-goal 이 최서열 street(공식 x=25)로 가는 경우는 금지. 그 street
    # 남단은 적재함(0~40cm)과 겹쳐 파지 후 남하(street_bail / 운반)가 적재함을
    # 관통한다 — 남서안 전면 폐기(2026-07-21)의 사유 그대로다.
    if gx < STREET_XS_M[1] - 1e-6:
        return False, False
    if abs(pose[0] - gx) <= DESCEND_STREET_TOL_M:
        return True, True
    return False, False


def descend_sort_key(cell, pose) -> tuple:
    """하산 후보 정렬 키 — **작을수록 우선**. 무조건 가장 북쪽(y 최대) 먼저.

    [2026-07-24 조작자 지시] 상수 DESCEND_SKIP_ROWS 위 블록의 목적함수 참조.
    스캔 레그는 마스트 하강에 흡수돼 첫 파지 시각에 영향이 없고, 순서에 따라
    달라지는 것은 안 고른 물체의 나중 적재함 왕복뿐이라 **북쪽이 항상 최적/동률**.
    pose 는 시그니처 호환을 위해 받되 쓰지 않는다 (판정은 셀 y 만으로 결정).
    """
    del pose
    return (-int(cell[1]),)


def _descend_sort_key_legacy(cell, pose) -> tuple:
    """[폐기 2026-07-24 — 롤백 시 위 함수를 이걸로 교체] 거리 버킷 → 북쪽.
    y=250(레그 0m)을 먼저 집어 M=6 기준 ~1.3s / 경로 1.0m 손해였다."""
    gy = snv.mini_goal_for(cell, mirror=True)[1]
    leg = abs(float(pose[1]) - gy)
    return (round(leg / 0.5), -int(cell[1]))   # 0.5 = 폐기된 DESCEND_DIST_BUCKET_M


def _nearest(vals, v):
    return min(vals, key=lambda t: abs(t - v))


def plan_route_street(p0, p1, obstacles) -> dict | None:
    """street 격자 위에서만 축정렬로 이동하는 경로. 전 구간 검사해 통과안만 반환.

    p0/p1 은 street 위가 아닐 수 있으므로 (파지 지점은 격자점 앞) 가장 가까운
    street 로 나온 뒤 → street 축정렬 L → 목표로 진입한다. x먼저/y먼저 두 안 중
    전 구간 클리어인 것을 고른다.
    """
    best = None
    for sx, sy in ((_nearest(STREET_XS_M, p0[0]), _nearest(STREET_YS_M, p0[1])),):
        for tx, ty in ((_nearest(STREET_XS_M, p1[0]), _nearest(STREET_YS_M, p1[1])),):
            for order in ("x", "y"):
                mid = (tx, sy) if order == "x" else (sx, ty)
                pts = [tuple(p0), (sx, sy), mid, (tx, ty), tuple(p1)]
                # 중복 좌표 제거 (동일 지점 연속이면 레그 낭비)
                comp = [pts[0]]
                for q in pts[1:]:
                    if math.hypot(q[0] - comp[-1][0], q[1] - comp[-1][1]) > 0.02:
                        comp.append(q)
                free, _ = path_free(comp, obstacles, clearance=STREET_CLEAR_M)
                if free and (best is None or len(comp) < len(best)):
                    best = comp
    if best is None:
        return None
    return {"mode": "street", "waypoints": [tuple(q) for q in best[1:]],
            "reason": f"street 축정렬 {len(best) - 1}-leg (격자 사이 25cm 통로)"}


def plan_route(p0, p1, obstacles) -> dict:
    """직선 → 축정렬 L 2안 → **street 격자** → 검증된 우회점 순으로 통과안 선택.

    7/20: 기존 detour 분기는 우회점을 계산만 하고 **그 경로가 실제로 뚫렸는지
    한 번도 검사하지 않아** 다른 물체를 그대로 관통했다 (실기 06:23 운반 경로가
    장애물 6셀을 팽창반경 안으로 통과, 최소거리 4.8cm). 이제 모든 후보를
    path_free 로 전 구간 검증하고, 통과안이 없으면 명시적으로 알린다.
    """
    free, blk = corridor_free(p0, p1, obstacles)
    if free:
        return {"mode": "direct", "waypoints": [tuple(p1)],
                "reason": "직선 코리도 클리어 — 단일 대각 질주"}
    reason0 = f"직선 막힘: 셀 {blk.get('cell')} (표 {blk['votes']})"
    for mode, mid in (("L_x_first", (p1[0], p0[1])),
                      ("L_y_first", (p0[0], p1[1]))):
        ok, _ = path_free([tuple(p0), tuple(mid), tuple(p1)], obstacles)
        if ok:
            return {"mode": mode, "waypoints": [tuple(mid), tuple(p1)],
                    "reason": f"{reason0} → 축정렬 {mode} 통과"}
    # L 2안 막힘 → street 격자 위 축정렬 (아레나가 물체로 가득 찼을 때의 주 경로)
    st = plan_route_street(p0, p1, obstacles)
    if st is not None:
        st["reason"] = f"{reason0} → L 2안 막힘 → {st['reason']}"
        return st
    # 마지막: 수직 우회점 — 단, 반드시 전 구간 검증하고 통과할 때만 채택
    ax, ay = blk["xy"]
    dxs, dys = p1[0] - p0[0], p1[1] - p0[1]
    seg_len = math.hypot(dxs, dys) or 1.0
    ux, uy = dxs / seg_len, dys / seg_len
    t = (ax - p0[0]) * ux + (ay - p0[1]) * uy
    qx, qy = p0[0] + t * ux, p0[1] + t * uy
    ox, oy = qx - ax, qy - ay
    n = math.hypot(ox, oy)
    if n < 1e-6:
        ox, oy, n = -uy, ux, 1.0  # 선분이 셀 중심 관통 — 좌수직으로 비킴
    for sgn in (1.0, -1.0):      # 양쪽 다 시도
        wp = (max(-ARENA_CLAMP_M, min(ARENA_CLAMP_M, ax + sgn * ox / n * DETOUR_CLEAR_M)),
              max(-ARENA_CLAMP_M, min(ARENA_CLAMP_M, ay + sgn * oy / n * DETOUR_CLEAR_M)))
        ok, _ = path_free([tuple(p0), wp, tuple(p1)], obstacles)
        if ok:
            return {"mode": "detour", "waypoints": [wp, tuple(p1)],
                    "reason": f"{reason0} → street 실패 → 검증된 수직 "
                              f"{DETOUR_CLEAR_M}m 우회점 3-leg"}
    # 전부 실패: 완전 무충돌 경로가 없다. 그래도 **대각 직선 강행은 최악** —
    # 격자점 위를 그대로 지난다. 검사를 통과 못해도 street 격자를 따라가는 편이
    # 항상 낫다(물체는 격자점에만 있고 street 는 그 사이 통로). 7/20 실기에서
    # 직선 강행이 장애물 6셀을 관통한 뒤 채택.
    forced = plan_route_street(p0, p1, [])   # 장애물 무시 = 순수 street 격자
    if forced is not None:
        return {"mode": "street_forced", "waypoints": forced["waypoints"],
                "reason": f"{reason0} → ⚠ 무충돌 경로 없음 — street 격자 강제 "
                          f"({len(forced['waypoints'])}-leg, 대각 직선 금지)"}
    return {"mode": "blocked", "waypoints": [tuple(p1)],
            "reason": f"{reason0} → ⚠ 통과 경로 없음 — 직선 강행 (충돌 위험)"}


# =========================================================================
# E2E 러너
# =========================================================================
# ---- 실전 2세트 타깃 모드 (룰북 §5/§7) ----
# 세트1 목표 형상은 당일 오전, 세트2 목표 과일은 경기 직전 공지.
# 세트1 정육면체는 과일면이 없으므로 지각 identity 가 "plain" — cube 는 alias.
SHAPE_ALIAS = {"cube": "plain"}
POINTS = {**{c: 20 for c in fl.FRUITS},
          **{c: 10 for c in fl.POLYHEDRA | {"plain"}}}


def build_targets(args) -> dict:
    """--target-shape/--target-fruit → {identity: quota}. 미지정 시 {} (종전 동작)."""
    t = {}
    if getattr(args, "target_shape", None):
        t[SHAPE_ALIAS.get(args.target_shape, args.target_shape)] = args.shape_quota
    if getattr(args, "target_fruit", None):
        t[args.target_fruit] = args.fruit_quota
    return t


def match_candidates(cells, collected, skip_cells, targets, placed_by_cls,
                     order="nearest", probe_p=None):
    """실전 타깃 모드의 수거 후보 셀 목록.

    - 타깃 클래스 확정 셀만 (오픽업 = 기본점수 2배 감점 → 비타깃 폴백 없음).
    - conflict 셀 제외 (identity 불확실 = 감점 리스크).
    - 세트별 quota 소진 클래스 제외 (형상 4 / 과일 3 이상은 경기장에 없다).
    - order=fruit-first/shape-first 는 해당 세트 잔여가 있는 동안 그 세트 우선
      (과일 20점 > 형상 10점 — 시간 부족 리스크 시 fruit-first 권장).

    probe_p: [2026-07-24 신규 — 정합성 인자 모드 전용, 사양 §7-2 "후보 필터 완화"]
      {cell: p} 를 주면 conflict 셀·혼동쌍 파트너 라벨 셀을 **probe 후보**로 추가
      개방한다(p >= CF_P_MIN). 종전 필터로는 결손 가지(파트너 셀 probe)와 conflict
      회수가 **구조적으로 불가능**했다. 파지는 여전히 근접 verify(strict) 로만
      통과하므로 "타깃 외 수거 절대 금지" 불변식은 그대로다.
      None(기본)이면 종전 동작과 100% 동일하다.
    """
    cand = [c for c, info in cells.items()
            if c not in collected and c not in skip_cells
            and info.get("identity") in targets
            and not info.get("conflict")
            and placed_by_cls[info["identity"]] < targets[info["identity"]]]
    if probe_p:
        seen = set(cand)
        cand = cand + [c for c, p in sorted(probe_p.items())
                       if p >= CF_P_MIN and c in cells and c not in seen
                       and c not in collected and c not in skip_cells]
    if order != "nearest" and cand:
        pref = fl.FRUITS if order == "fruit-first" else (fl.POLYHEDRA | {"plain"})
        pri = [c for c in cand if cells[c]["identity"] in pref]
        if pri:
            cand = pri
    return cand


# =========================================================================
# 정합성 인자(Consistency Factor) + conflict 클래스        [2026-07-24 신규]
# 사양: mission/docs/match-strategy.md  §2 / §3 / §6.5
# =========================================================================
# 기본 **OFF**. `--consistency on` 으로만 켠다 — 켜면 pick_target 의 "하산 우선 +
# 최근접"이 시간대비 기댓값 랭킹으로 교체되고, conflict·파트너 셀이 probe 후보로
# 열린다. 실기 검증된 종전 경로를 기본값으로 남기기 위한 선택이다.
#
# ⚠ 선행조건(사양 §7-4 — 하드 블로커): **동일 혼동쌍(apple↔orange) 근접 판별력.**
#   근접이 비결정이면 probe 결과가 오염돼 아래 소진(exhaustion) 갱신 자체가
#   무너진다. banana/pineapple 은 현재도 성립.
RULEBOOK_FRUIT_N = 3       # 룰북 §5: 과일 클래스당 정확히 3개 (개수 사전의 근거)
CF_FLIP_EPS = 0.20         # 라벨 1개가 혼동쌍 파트너로 뒤집혔을 확률 (§1.3 ±1~±2 역산)
CF_TIE_P = 0.5             # conflict(동점) 셀의 무정보 사전확률
CF_P_MIN = 0.05            # 이 미만 확률의 셀은 후보에서 제외 (헛걸음 방지)
CF_MAX_ENUM = 20000        # 배정 열거 상한 — 넘으면 폴백(종전 최근접)

# ---- §4 실측 파라미터 (성공 사이클 182건 중앙값, report.json phases 집계) ----
CF_EFF_SPEED = 0.55        # m/s — route 중앙 6.0s @ ~3.3m 역산 (회전·정착 포함)
CF_APPROACH_S = 2.6        # 정밀 접근·측정
CF_FACE_S = 1.2            # 근접 face + pair 재검증 (= 사양의 T_ver)
CF_BAIL_S = 1.4            # 불일치 abort 복귀 (T_probe 9.8~12.4 상단 흡수)
CF_GRASP_S = 1.0           # 파지
CF_CARRY_FIXED_S = 9.5     # 운반 고정분(드리프트·슬롯·릴리스) — carry 12.8s 역산
CF_STORAGE_XY = fl.official_cm_to_map(20.0, 20.0)   # 보관함 중심 = 공식 (20,20)cm
CF_DESCEND_SAVE_S = 6.0    # 하산 파지가 없애는 하이웨이 왕복(~3.3m / 0.55m/s)


def conflict_pair(info: dict) -> frozenset:
    """conflict 셀의 동점 후보쌍 `{X, Y}` (사양 §6.5 "분류 규칙").

    `_finalize_scan` 이 과일표 동률에서 `conflict="conflicting_fruit"` 를 달고
    `fruit_hits` 에 갈린 표를 남긴다 — 그 상위 2개가 곧 동점쌍이다. 과일 동점이
    아니면(진짜 plain 등) 빈 집합.
    """
    if info.get("conflict") != "conflicting_fruit":
        return frozenset()
    fh = {k: v for k, v in (info.get("fruit_hits") or {}).items()
          if k in fl.FRUITS}
    if len(fh) < 2:
        return frozenset()
    return frozenset(k for k, _ in sorted(fh.items(), key=lambda kv: -kv[1])[:2])


def consistency_probs(cells, target, placed=0, collected=frozenset(),
                      evidence=None, eps=CF_FLIP_EPS):
    """혼동쌍 부분계에서 셀별 "이 셀이 target 일 확률" p(c). → ({cell: p}, 진단문)

    **한 메커니즘이 사양의 케이스를 전부 흡수한다.** 혼동쌍(AO=apple/orange,
    BP=banana/pineapple)은 룰북상 각 3개씩 = 쌍당 6셀로 닫혀 있다(§1.2 + 룰북).
    그 셀들이 target 라벨 / partner 라벨 / conflict 로 어떻게 흩어졌든, **"정확히
    3개가 target"** 이라는 제약을 만족하는 **모든 배정을 열거**하고 라벨 신뢰도로
    가중해 주변확률을 낸다. 그래서 케이스 분기를 하드코딩하지 않는다:

      - §2.1 과다(`p=(3-적재)/n`) / 결손(파트너 셀 `p=hidden/n`) 분기
      - §3 `d=±1` 전개, `|d|>=2`(5/1) 도 같은 식으로 자동
      - §6.5 conflict 배정 플레이북(1개 → 개수 2 클래스 / 2개 → 3·3·3·1·3·3·2·2)
      - §6.5 **구조적 축퇴** — 개수만으로 못 가리는 경우 p 가 1 로 튀지 않는다
        (H_A "conflict=Y" 와 H_B "conflict=X + X라벨 1개가 Y" 가 둘 다 열거된다)
      - §2.2 축차 갱신 — 근접 검증 결과를 `evidence` 로 넣으면 **소진이 자동**.
        상대 클래스가 다 차면 남은 배정이 하나뿐이라 p 가 1 로 수렴한다.

    evidence: {cell: 관측 클래스} — 근접 verify 로 **확정**된 셀. hard 제약이라
      해당 셀은 배정에서 고정된다. 이것이 `p=1` 의 **유일한** 정당한 근거다
      (개수-사전만으로는 위 축퇴 때문에 확정 불가 — 사양 §6.5).
    placed: 이미 적재한 target 개수 (잔여 = 3 - placed).
    """
    evidence = evidence or {}
    if target not in fl.FRUITS:
        return {}, f"{target}: 형상 타깃 — 혼동쌍 없음"
    route = fl.PairVerifier.ROUTE
    pair_id = route.get(target)
    partner = next((f for f in sorted(fl.FRUITS)
                    if f != target and route.get(f) == pair_id), None)

    pool, kind = [], {}
    for c, info in cells.items():
        if c in collected:
            continue
        tie = conflict_pair(info)
        if tie:
            if target in tie:
                pool.append(c)
                kind[c] = "tie"
        elif info.get("identity") == target:
            pool.append(c)
            kind[c] = "t"
        elif partner is not None and info.get("identity") == partner:
            pool.append(c)
            kind[c] = "p"
    need = RULEBOOK_FRUIT_N - placed
    if need <= 0:
        return {}, f"{target} quota 소진 ({placed}/{RULEBOOK_FRUIT_N})"
    if not pool:
        return {}, f"{target} 부분계 후보 없음"

    forced_t = [c for c in pool if evidence.get(c) == target]
    forced_x = [c for c in pool if c in evidence and evidence[c] != target]
    free = [c for c in pool if c not in evidence]
    need_free = need - len(forced_t)
    warn = ""
    if len(pool) + placed != 2 * RULEBOOK_FRUIT_N:
        # 쌍 합계가 6이 아니다 = 과일이 형상으로 새거나 셀을 놓쳤다는 뜻.
        # 사양 §6.5 정합성 게이트 이탈 — 막지는 않고 경고만 (§1.5 와 동일 태도).
        warn = f" ⚠ 쌍 합계 {len(pool) + placed}≠6 (개수 사전 이탈 — 참고용)"
    if need_free < 0:
        return ({c: (1.0 if evidence.get(c) == target else 0.0) for c in pool},
                f"{target}: 확정 {len(forced_t)} > 잔여 {need} — 검증 모순" + warn)
    if need_free > len(free):
        return ({c: (1.0 if c in free or evidence.get(c) == target else 0.0)
                 for c in pool},
                f"{target}: 자유셀 {len(free)} < 필요 {need_free} — 개수 모순" + warn)
    n_comb = math.comb(len(free), need_free)
    if n_comb > CF_MAX_ENUM:
        return {}, f"{target}: 배정 열거 {n_comb} > 상한 {CF_MAX_ENUM} — 폴백" + warn

    def _w(c, is_t):
        k = kind[c]
        if k == "tie":          # 동점 = 무정보 (표가 갈렸다는 사실 자체가 중립)
            return CF_TIE_P
        if k == "t":
            return (1.0 - eps) if is_t else eps
        return eps if is_t else (1.0 - eps)

    tot, acc = 0.0, {c: 0.0 for c in free}
    for S in itertools.combinations(range(len(free)), need_free):
        sset = set(S)
        w = 1.0
        for i, c in enumerate(free):
            w *= _w(c, i in sset)
        tot += w
        for i in sset:
            acc[free[i]] += w
    if tot <= 0.0:
        return {}, f"{target}: 가중합 0 — 폴백" + warn
    p = {c: acc[c] / tot for c in free}
    p.update({c: 1.0 for c in forced_t})
    p.update({c: 0.0 for c in forced_x})
    # [2026-07-24 조작자 지시] 비대칭 분할(1/5·2/4 등) 진단을 명시화 — 조작자가
    # "지금 orange 5셀 중 사과 2개를 뒤지는 중"임을 로그로 이해할 수 있게 한다.
    # 룰북상 잔여 target 은 `need` 개인데 그중 target 라벨(n_t)로 이미 보이는 건
    # 소수뿐이면, 나머지(need-n_t)는 partner 라벨 셀에 숨어 있다는 뜻이다. 이 값이
    # 곧 probe 로 회수할 "숨은 {target}" 추정 개수다. 확률/순서에는 영향 없음(진단만).
    n_t = sum(1 for c in pool if kind[c] == "t")
    n_p = sum(1 for c in pool if kind[c] == "p")
    n_tie = sum(1 for c in pool if kind[c] == "tie")
    hidden = max(0, need - n_t - n_tie)
    split = (f"라벨 {target}×{n_t}/{partner}×{n_p}"
             + (f"/동점×{n_tie}" if n_tie else ""))
    hint = (f" → {partner} {n_p}셀 중 ~{hidden}개가 숨은 {target} 추정(probe 대상)"
            if hidden > 0 else "")
    note = (f"{target}: 부분계 {len(pool)}셀({split}, 자유 {len(free)} / 확정 "
            f"{len(forced_t)}✓ {len(forced_x)}✗), 잔여 {need}, 배정 {n_comb}가지"
            + hint + warn)
    return p, note


def cf_time_exp(cell_xy, pose, p: float) -> tuple:
    """(T_exp, T_full, T_probe) — 사양 §2 의 기대 소요 시간.

    `T_exp = p·T_full + (1-p)·T_probe`. 불일치(probe 실패)면 파지·운반을 안 하므로
    T_probe 만 든다 — T_full 하나로 나누면 불확실 셀이 체계적으로 과소평가된다.
    분모는 반드시 **전체 사이클**(접근+파지+보관함 운반+드랍)이라야 "보관함
    최근접 먼저(SJF)" 순서가 살아난다 (§2, grid_match_strategy §6.2 동일 결론).
    """
    d_go = math.hypot(cell_xy[0] - pose[0], cell_xy[1] - pose[1])
    d_st = math.hypot(cell_xy[0] - CF_STORAGE_XY[0], cell_xy[1] - CF_STORAGE_XY[1])
    t_reach = d_go / CF_EFF_SPEED + CF_APPROACH_S + CF_FACE_S
    t_probe = t_reach + CF_BAIL_S
    t_full = t_reach + CF_GRASP_S + CF_CARRY_FIXED_S + d_st / CF_EFF_SPEED
    return p * t_full + (1.0 - p) * t_probe, t_full, t_probe


# [2026-07-23 조작자 승인] 프리샷 스위치. False = 센터 스캔 1회만.
#
# [폐지 → 복원 2026-07-23 08시] 한 번 껐다가 되살렸다. 경위:
#   끈 이유  — 프리샷은 "초근접 탑뷰 A1 conf 0.05~0.11 < 문턱 0.25" 를 우회하려고
#              만든 보조 장치였고, 그 원인이 하이브리드로 해결됐다(아래 표).
#   되살린 이유 — 하이브리드가 푼 것은 **A1 검출(물체 존재)** 뿐이고, **face
#              과일 판정**은 여전히 근접에서 무너진다. 같은 4셀 과일 판정 실측:
#                프리샷(1.26m) n=51  face_conf 중앙 0.965  게이트(0.5) 탈락 **0%**
#                스핀  (0.45m) n=19  face_conf 중앙 0.894  게이트 탈락 **16%**
#              face 모델은 imgsz=224 / mosaic=0.0 / scale=0.05 로 스케일 증강이
#              거의 없어 근접 대형 크롭에서 conf 가 무너지고 apple↔orange 가
#              뒤집힌다(근접 크롭 16개 중 6개가 imgsz 에 따라 라벨 변동).
#              다중 스케일 투표도 무효 — 틀린 답도 conf 0.96 으로 나온다.
#
# 그래서 **둘 다 쓴다**: 프리샷은 과일 정체(원거리 크롭), 스핀은 물체 존재(12샷).
# 종전의 exclude_cells 기각("초근접 스핀표 불신")은 제거했다 — 그 불신의 근거였던
# 초근접 A1 미검출이 하이브리드로 해결됐기 때문이다.
#
# 프리샷은 "초근접 탑뷰 A1 conf 0.05~0.11 < 문턱 0.25"(preshot_cells 도크스트링,
# 7/21) 를 우회하려고 만든 보조 장치였다. 그 근본 원인(스티치 @896 배율 0.42 →
# 초근접 물체가 네트 250px = 학습 분포 밖)이 하이브리드(A1_IMGSZ_SCAN_NEAR)로
# 해결돼 역할이 끝났다.
#
# 실기 14런 전수 재현 — 프리샷 전담 4셀 중 확정 개수(스핀 샷만, 문턱 3표 정상 적용):
#   스티치 @896 단독 (프리샷 도입 당시)  1.14/4   ← 프리샷을 만든 이유
#   하이브리드 스핀만 (프리샷 없음)      3.14/4
#   실기 실제 (프리샷 포함)             3.14/4   ← 평균·분포 완전 동일
# 14런 중 13런에서 하이브리드(스핀만) >= 실기. 예외 011726 1건(-1셀).
#
# 시간 이득: 프리샷 블록(지점 정차→마스트 대기→북향 회전→캡처) 실측 중앙값 5s
# + 마스트 대기가 중앙 도착 이후로 밀려 주행과 완전히 겹침(아래 stage_goto_center).
#
# True 로 되돌리면 종전 동작(프리샷 캡처 + 전담 4셀 votes_k/fruit_k 완화)이 그대로
# 복원된다 — _infer_phase / _finalize_scan 은 _preshot_cap 유무로 자동 분기한다.
PRESHOT_ENABLED = True


def preshot_cells() -> set:
    """스캔점 대각 인접 4셀 (0.35m) — 스핀 스캔의 구조적 사각.

    7/21 실증: 마스트 up(14:32)·mid(17:12) 공히 인접 셀이 8샷 전체 미검출
    (초근접 탑뷰 A1 conf 0.05~0.11 < 문턱 0.25). 서진 레그1 종료 지점
    (해당 셀들에서 1.8~2.3m, 마스트다운 = 최적 캘리브)의 프리샷이 전담한다."""
    ox, oy = fl.map_to_official_cm(*fl.CENTER_SCAN_XY)
    p = fl.GRID_PITCH_CM
    xs = (int(ox // p) * p, int(ox // p) * p + p)
    ys = (int(oy // p) * p, int(oy // p) * p + p)
    return {(x, y) for x in xs for y in ys}


class FrameRecorder:
    """실기 전 구간 원해상도 프레임 상시 저장 (기본 3 fps/캠). [2026-07-23]

    왜 3 fps 원해상도인가 — Orin Nano 실측(1920x1080, JPEG q95 4:4:4):

      | 방식                    | CPU(코어) | 디스크    | 3분 경기 |
      | 전 프레임(30fps) q95     |  1.04     | 16 MB/s  | 2.9 GB   |
      | 전 프레임 q85 4:2:0      |  0.63     | 6.3 MB/s | 1.1 GB   |
      | 축소 960x540 q85 30fps  |  0.18     | 1.9 MB/s | 0.34 GB  |
      | **3fps/캠 원해상도 q95** |  **0.21** | 3.2 MB/s | 0.58 GB  |

    전 프레임 저장은 코어 1개를 통째로 먹는다 — HUD(구 match_hud_display)에서
    걷어낸 양을 그대로 반납해 sllidar 스케줄링/localization 을 다시 망친다.
    축소 저장은 CPU 는 더 싸지만 나중에 YOLO 재분석에 못 쓴다(해상도 손실).
    그래서 **해상도는 지키고 프레임레이트를 낮춘다.**
    Orin Nano 에는 NVENC 하드웨어 인코더가 없어 인코딩은 전부 CPU 다.

    구독 자체는 러너가 이미 하고 있으므로(fieldlib FieldNode, 최신 1장 보관)
    추가 비용은 인코딩+쓰기뿐이다. 같은 프레임을 두 번 저장하지 않도록
    FieldNode.stamp 로 갱신 여부를 본다.
    """

    def __init__(self, fn, out_dir: Path, fps: float = 3.0, quality: int = 95,
                 min_free_gb: float = 5.0):
        self.fn = fn
        self.dir = out_dir / "frames"
        self.period = 1.0 / max(0.1, fps)
        self.quality = int(quality)
        self.min_free_gb = float(min_free_gb)
        self.label = "init"          # 러너가 갱신하는 현재 단계 (프레임 상관용)
        self.saved = 0
        self.dropped = 0             # 인코딩이 주기를 못 따라간 횟수
        self._stop = threading.Event()
        self._th = None
        self._t0 = time.monotonic()
        self._imu0 = None            # 첫 IMU yaw (상대각 기준점)
        self._loc0 = None            # 첫 localizer yaw (상대각 기준점)

    def start(self):
        if self.fn is None:
            return self
        self.dir.mkdir(parents=True, exist_ok=True)
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=timeout)

    def _loop(self):
        try:
            os.nice(10)      # 리눅스에서 nice 는 스레드 단위 — 이 스레드만 양보
        except OSError:
            pass
        from PIL import Image
        idx = 0
        last_stamp = {}
        index_path = self.dir / "index.jsonl"
        while not self._stop.is_set():
            t_tick = time.monotonic()
            try:
                stamps = {k: self.fn.stamp.get(k) for k in ("top_rgb", "near_rgb")}
                if stamps != last_stamp and all(v is not None for v in stamps.values()):
                    last_stamp = stamps
                    pair = self.fn.rgb_pair()
                    if pair is not None:
                        el = time.monotonic() - self._t0
                        for cam, arr in zip(("top", "near"), pair):
                            Image.fromarray(arr).save(
                                self.dir / f"{idx:05d}_{el:07.2f}_{cam}.jpg",
                                format="JPEG", quality=self.quality, subsampling=0)
                        # 자세: localizer(라이다 정합)와 IMU 를 **둘 다** 남긴다.
                        # 어느 쪽이 틀렸는지는 둘을 나란히 봐야만 갈린다.
                        rec = {"i": idx, "t": round(el, 2), "label": self.label}
                        try:
                            p = self.fn.pose()
                            if p:
                                rec["x"], rec["y"] = round(p[0], 4), round(p[1], 4)
                                rec["yaw_loc"] = round(math.degrees(p[2]), 2)
                        except Exception:      # noqa: BLE001
                            pass
                        yi = imu_yaw_deg(self.fn)
                        if yi is not None:
                            rec["yaw_imu"] = round(yi, 2)
                            if self._imu0 is None:
                                self._imu0 = yi
                            # 시작 대비 상대각 — IMU 는 절대 기준이 없다
                            rec["yaw_imu_rel"] = round(
                                (yi - self._imu0 + 180) % 360 - 180, 2)
                        if "yaw_loc" in rec and "yaw_imu_rel" in rec:
                            if self._loc0 is None:
                                self._loc0 = rec["yaw_loc"]
                            loc_rel = (rec["yaw_loc"] - self._loc0 + 180) % 360 - 180
                            # 이 값이 0 에서 벌어지면 두 소스가 갈라진 것 = yaw 이상
                            rec["yaw_diff"] = round(loc_rel - rec["yaw_imu_rel"], 2)
                        with open(index_path, "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                        idx += 1
                        self.saved += 1
                        if self.saved % 60 == 0 and _free_gb(self.dir) < self.min_free_gb:
                            print(f"[frames] 디스크 여유 {_free_gb(self.dir):.1f}GB "
                                  f"< {self.min_free_gb}GB — 저장 중단", flush=True)
                            return
            except Exception:      # noqa: BLE001 — 기록은 미션을 절대 못 죽인다
                self.dropped += 1
            spent = time.monotonic() - t_tick
            if spent > self.period:
                self.dropped += 1
            self._stop.wait(max(0.0, self.period - spent))


def imu_yaw_deg(fn):
    """IMU 기반 yaw(도). 없으면 None.

    [2026-07-23] localizer yaw 와 **독립적인** 각도 소스다. 둘을 나란히 기록해야
    "yaw 가 틀렸다"가 (a) IMU 드리프트인지 (b) 라이다 정합 실패인지 (c) 둘 다인지
    구분된다. IMU 는 절대 기준이 없어 시작 시점 대비 상대각으로만 의미가 있으므로,
    분석 때는 두 계열의 **차이(diff)** 추이를 봐야 한다 — 각각의 절대값이 아니라.

    [정정 2026-07-23 04시] 처음엔 /imu/data 쿼터니언에서 뽑으려 했는데, 실기의
    D435i IMU 는 **orientation 을 발행하지 않는다** (전부 0, covariance[0]=-1 =
    ROS 관례상 '미제공'). 자이로/가속만 나온다. 그래서 첫 런의 index.jsonl 에
    yaw_imu 필드가 아예 없었다.
    대신 arena_control_node 가 이미 gyro-z 를 적분해 status 에 실어 보낸다
    (`imu_prior.yaw_integrated`, 라디안, 언랩됨 — 절대 기준 없음).
    새 노드/필터를 띄우지 않고 그 값을 그대로 쓴다 (추가 CPU 0).
    """
    st = fn.arena_status() if fn is not None else None
    if not st:
        return None
    p = st.get("imu_prior") or {}
    y = p.get("yaw_integrated")
    if y is None or not p.get("fresh", True):
        return None
    return math.degrees(float(y))


def _free_gb(path: Path) -> float:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 1e9
    except OSError:
        return float("inf")


class E2ERunner:
    def __init__(self, args, gt: dict, out_dir: Path, fn):
        self.args = args
        self.gt = gt
        self.out = out_dir
        self.fn = fn                      # FieldNode | None(dry-run 한정)
        self.scan_dir = out_dir / "scan"
        self.models = None                # (a1, face)
        self.pair = None                  # PairVerifier (구성 A)
        self.pair_routes = PAIR_ROUTES[args.pair]   # 활성 라우트 (2026-07-23)
        self._pending_writes = []   # [(경로, 인코딩 bytes)] — 종료 후 일괄 기록
        self.votes_k = args.votes_k or fl.CELL_VOTES_MIN
        self.fruit_k = args.fruit_k or fl.CELL_FRUIT_K
        self.cells: dict = {}             # (x,y)cm -> {identity,votes,fruit_hits,conflict}
        self.presence: Counter = Counter()
        self.cell_votes = defaultdict(list)
        self.collected: set = set()
        self.skip_cells: set = set()
        self.slot_idx = 0
        self.hold_last = False   # [2026-07-22] 마지막 타깃을 문 채 종료 중
        self.targets = build_targets(args)   # {identity: quota} | {} (실전 2세트)
        self.placed_by_cls = Counter()       # 적재 성공 집계 (quota 대비)
        # [2026-07-24 신규] 정합성 인자 모드 (사양 §2/§6.5). 기본 off.
        self.cf_on = getattr(args, "consistency", "off") == "on"
        # 근접 verify 로 **확정**된 셀만 담는다: {cell: 관측 클래스}. 사양 §6.5 —
        # 개수-사전은 축퇴 때문에 p=1 을 못 만들고, 소진(exhaustion)의 유일한 근거가
        # 이 hard evidence 다. 스캔 라벨은 절대 여기 들어오지 않는다.
        self.cf_evidence: dict = {}
        # probe 셀(스캔 identity ≠ 사냥 대상)에서 "지금 무엇을 노리는가".
        # {cell: 목표 클래스} — collect_one 의 verify 기대값·적재 집계에 쓴다.
        self.cf_intent: dict = {}
        self._cf_last_ident = None    # 직전 verify 가 실제로 본 클래스
        self._preshot_cap = None             # 근접 4셀 전담 프리샷 (7/21)
        self._preshot_cells: set = set()
        self.move_stats = []   # 이동별 소요/예상 계측 (병목 추적)
        prof = SPEED_PROFILES[args.speed_profile]
        self.v_cruise = args.max_v if args.max_v is not None else prof["cruise"]
        self.v_approach = prof["approach"]
        self.v_place = prof["place"]
        # 스캔 회전 정착 대기(s). 7/20 실기: 0.6은 과잉으로 보여 파라미터화.
        self.scan_settle_s = float(getattr(args, "scan_settle", SCAN_SETTLE_S))
        self.release_ticks = float(getattr(args, "release_ticks", PLACE_RELEASE_TICKS))
        # 7/20: 적재 경로 pose 안정화 제거는 롤백 — 기본 True(기존 동작).
        # 재검증 후 필요하면 --no-place-settle 로 다시 끈다.
        self.place_settle = not bool(getattr(args, "no_place_settle", False))
        self.sim_pose = list(fl.START_POSE)
        self._sim_init = False
        # [2026-07-22] street 주행 (navigation/docs/control-and-routing.md).
        # dry-run 은 래퍼(street_* 메서드)가 sim_pose 로 직접 처리하므로
        # StreetNavigator 는 real 모드에서만 호출된다.
        self._pose_skew_warned = False   # [2026-07-23] 클럭 경고 1회만
        self._rgb_spread_warned = False  # [2026-07-23] RGB stamp 편차 경고 1회만
        self.nav_mode = getattr(args, "nav", "street")
        self.snav = (snv.StreetNavigator(fn, log=self.log,
                                         dry_run=bool(args.dry_run))
                     if self.nav_mode == "street" else None)
        # [2026-07-23 신규] 적재 드리프트 회전 튜닝 실행인자 반영. 모듈 전역이
        # 아니라 이 인스턴스에만 쓴다 (StreetNavigator.__init__ 주석 참조).
        if self.snav is not None:
            self.snav.drift_yaw_rate = float(args.drift_yaw_rate)
            self.snav.drift_kp_yaw = float(args.drift_kp_yaw)
        # [2026-07-23] 실험 기능 토글 제거 — 구성은 고정이다 (상수 주석 참조).
        self.report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "args": vars(args), "gt": [[list(c), v] for c, v in sorted(gt.items())],
                       "stages": [], "scan": {}, "cycles": [], "log": []}
        # ---- 관객용 실시간 피드 상태 (/match/state) [데모] ----
        self._phase = "STARTUP"
        self._active_cell = None      # 지금 노리는 셀 (x,y)cm | None
        self._cycle_live = None       # 진행 중 사이클 요약 dict | None
        self._drifting = False        # 운반 드리프트 진행 중 (경기 통틀어 한 곳)
        self._t_mission0 = None       # run() 시작 monotonic
        self._match_state_t = 0.0     # 마지막 발행 시각 (레이트 리밋)

    # ---- 공통 유틸 ----
    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.report["log"].append(line)
        self.publish_match_state()

    def mark(self, stage: str, ok: bool, t0: float, detail=None):
        self.report["stages"].append(
            {"stage": stage, "ok": ok, "sec": round(time.monotonic() - t0, 2),
             "detail": detail})
        self._phase = stage
        self.save_report()
        self.publish_match_state(force=True)

    # ---- 관객용 실시간 피드 (/match/state) [데모] ----
    # 왜 여기인가: 러너는 ROS 로 아무것도 발행하지 않고 stdout + report.json 만
    # 남긴다. 그래서 화면에 띄울 수 있는 "지금 무엇을 왜 하는가"가 존재하지 않았다.
    # 셀의 **판정 결과**(수거 대상 / 거부 / 보류)는 어디에도 저장돼 있지 않고
    # targets·placed_by_cls·collected·skip_cells 에서 파생될 뿐이라, 그 파생을
    # 여기서 한 번 하고 내보낸다.
    # 경기 로직에 대한 영향: 없음. 예외는 전부 삼키고, 0.2s 미만 간격은 건너뛴다.
    MATCH_STATE_MIN_INTERVAL_S = 0.2

    def cell_status(self, cell, info: dict) -> str:
        """셀 하나의 관객용 판정. 우선순위가 곧 의미 순서다."""
        if cell in self.collected:
            return "collected"          # 이미 적재함에 넣었다
        if self._active_cell is not None and tuple(cell) == tuple(self._active_cell):
            return "active"             # 지금 이걸 노리고 있다
        if info.get("conflict"):
            return "conflict"           # 과일 판정 충돌 — 확신 없어 보류
        if cell in self.skip_cells:
            return "skipped"            # 시도했다 실패/거부됨
        ident = info.get("identity")
        if not self.targets:
            return "pending"            # 리허설 모드(실전 타깃 없음)
        quota = self.targets.get(ident)
        if quota is None:
            return "refused"            # 타깃 클래스가 아니다 — 집으면 감점
        if self.placed_by_cls.get(ident, 0) >= quota:
            return "quota_full"         # 맞는 클래스지만 쿼터가 찼다
        return "target"                 # 수거 예정

    # 사이클 하위 단계는 rec["phases"] 에 이미 채워지는 키에서 파생한다 —
    # 11곳을 따로 계측하지 않고도 "지금 어디"가 정확히 나온다.
    CYCLE_PHASES = ("route", "face", "approach", "grasp", "carry")

    def sub_phase(self) -> str | None:
        rec = self._cycle_live
        if rec is None:
            return None
        if self._drifting:
            return "drift"        # CARRY 안의 유일한 드리프트 구간
        done = rec.get("phases", {})
        for p in self.CYCLE_PHASES:
            if p not in done:
                return p          # 앞 단계까지 끝났으면 지금은 이 단계
        return "place"

    def match_state_payload(self) -> dict:
        elapsed = (time.monotonic() - self._t_mission0
                   if self._t_mission0 is not None else 0.0)
        cells = []
        for cell, info in sorted(self.cells.items()):
            mx, my = fl.official_cm_to_map(*cell)
            # 판정 **근거**를 같이 보낸다. 지금까지는 집계된 표 수만 남기고
            # 원시 투표(어느 카메라가, 몇 m 에서, A1/face 를 얼마의 확신으로 봤는지)를
            # 버리고 있었다 — 화면이 "왜 그렇게 판정했는가"를 말하려면 이게 필요하다.
            ev = [{"a1": v.get("a1"), "a1_conf": v.get("a1_conf"),
                   "face": v.get("identity"), "face_conf": v.get("face_conf"),
                   "cam": v.get("cam"), "range_m": v.get("range_m"),
                   "shot": v.get("shot")}
                  for v in self.cell_votes.get(cell, [])[-6:]]
            cells.append({
                "cell": list(cell), "xy": [round(mx, 3), round(my, 3)],
                "identity": info.get("identity"), "votes": info.get("votes"),
                "fruit_hits": info.get("fruit_hits"),
                "conflict": info.get("conflict"), "evidence": ev,
                # 화면이 "집으면 몇 점 손해"를 말할 수 있도록 배점을 같이 보낸다.
                # 미스픽은 자기 배점의 -2배다 (docs/01-competition-and-rules.md §4).
                "points": POINTS.get(info.get("identity"), 0),
                "status": self.cell_status(cell, info)})
        presence = []
        for cell, n in sorted(self.presence.items()):
            if cell in self.cells:
                continue
            mx, my = fl.official_cm_to_map(*cell)
            presence.append({"cell": list(cell), "xy": [round(mx, 3), round(my, 3)],
                             "count": n})
        # 아레나 기하는 페이로드에 실어 보낸다 — 화면이 4m/2m 상수를 또 복제하지
        # 않게 하려는 것이다(이미 저장소에 격자 테이블 사본이 5개 있다).
        # 반이레나는 상수가 있으면 그걸, 없으면 적재함 사각의 좌하단에서 뽑는다
        # (STORAGE_RECT_MAP[0] == -ARENA_HALF_M). 두 방식 다 4m/2m 에서 맞는다.
        half_m = getattr(fl, "ARENA_HALF_M", None)
        if half_m is None:
            half_m = abs(fl.STORAGE_RECT_MAP[0])
        arena = {
            "half_m": half_m,
            "storage": list(fl.STORAGE_RECT_MAP),
            "grid_xs_cm": list(fl.GRID_XS_CM), "grid_ys_cm": list(fl.GRID_YS_CM),
            "start_xy": [fl.START_POSE[0], fl.START_POSE[1]],
            "scan_xy": list(fl.CENTER_SCAN_XY),
        }
        return {
            # phase 는 mark() 가 실제로 찍는 단계만 쓴다 (SEED / GOTO_CENTER /
            # SCAN / COLLECT). 없는 단계 이름을 지어내지 않는다.
            "stamp": time.time(), "phase": self._phase,
            "sub_phase": self.sub_phase(),
            "elapsed_sec": round(elapsed, 1),
            "budget_sec": self.report.get("summary", {}).get("match_budget_sec", 180),
            "arena": arena,
            "targets": dict(self.targets), "placed_by_cls": dict(self.placed_by_cls),
            "score_est": sum(POINTS.get(k, 0) * n
                             for k, n in self.placed_by_cls.items()),
            "cycle": self._cycle_live,
            "cells": cells, "presence": presence,
            "log_tail": self.report["log"][-6:],
        }

    def publish_scan_overlay(self, blob: bytes, shot: int):
        """스캔 오버레이 JPEG 을 관객 화면으로 발행. 시각화가 경기를 막지 않는다."""
        if self.fn is None:
            return
        try:
            pub = self.fn.pub.get("scan_overlay")
            if pub is None:
                return
            msg = self.fn._CompressedImage()
            msg.format = "jpeg"
            msg.data = list(blob) if not isinstance(blob, (bytes, bytearray)) else blob
            pub.publish(msg)
            self.log(f"  [관객화면] 스캔 오버레이 발행 shot{shot:02d} "
                     f"({len(blob)/1048576:.2f} MiB)")
        except Exception:
            pass

    def publish_match_state(self, force: bool = False):
        if self.fn is None:
            return
        now = time.monotonic()
        if not force and now - self._match_state_t < self.MATCH_STATE_MIN_INTERVAL_S:
            return
        self._match_state_t = now
        try:
            pub = self.fn.pub.get("match_state")
            if pub is None:
                return
            from std_msgs.msg import String
            pub.publish(String(data=json.dumps(self.match_state_payload(),
                                               ensure_ascii=False, default=str)))
        except Exception:
            pass   # 시각화는 경기를 절대 방해하지 않는다

    def save_report(self):
        self.report["cells"] = [
            {"cell": list(c), **info} for c, info in sorted(self.cells.items())]
        self.report["presence"] = [
            {"cell": list(c), "count": n} for c, n in sorted(self.presence.items())]
        (self.out / "report.json").write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2, default=str))

    def spin(self, sec: float):
        if self.fn is not None:
            self.fn.spin_for(sec)
        else:
            time.sleep(min(sec, 0.2))  # dry-run: 대기 축약

    def pose(self):
        if self.args.dry_run:
            if not self._sim_init:
                self._sim_init = True
                if self.fn is not None:
                    p = self.fn.pose()
                    if p:
                        self.sim_pose = list(p)
            return tuple(self.sim_pose)
        return self.fn.pose() if self.fn is not None else None

    def _sim_apply(self, dx, dy, dyaw):
        x, y, yaw = self.sim_pose
        self.sim_pose = [x + dx * math.cos(yaw) - dy * math.sin(yaw),
                         y + dx * math.sin(yaw) + dy * math.cos(yaw),
                         wrap_angle(yaw + dyaw)]

    # ---- 명령 래퍼 (dry-run 게이트) ----
    def do_move(self, dx, dy=0.0, dyaw=0.0, max_v=None, timeout=40.0,
                tag: str = "") -> dict:
        if self.args.dry_run:
            self._sim_apply(dx, dy, dyaw)
            self.log(f"  [dry-run] move dx={dx:+.3f} dy={dy:+.3f} "
                     f"dyaw={math.degrees(dyaw):+.1f}° max_v={max_v}")
            return {"ok": True, "dry": True}
        t0 = time.monotonic()
        r = self.fn.move_relative(dx, dy, dyaw, max_v=max_v, timeout=timeout)
        el = time.monotonic() - t0
        # 이동 프로파일 예상시간 대비 과다하면 기록 — 7/20 실기에서 -0.2m 후진
        # 한 번에 8s가 날아갔는데(프로파일 예상 ~1.1s) 원인이 펌웨어 정착 실패인지
        # 응답 유실인지 로그만으로 가려지지 않아 계측을 상시화한다.
        dist = math.hypot(dx, dy)
        v = max_v or 0.233                       # position_max_rad_s 6.0 환산
        expect = dist / max(v, 0.05) + abs(dyaw) / 1.0 + 0.5
        self.move_stats.append({"tag": tag, "d": round(dist, 3),
                                "dyaw": round(dyaw, 3), "max_v": max_v,
                                "sec": round(el, 2), "expect": round(expect, 2),
                                "ok": bool(r.get("ok")),
                                # 브리지 실패는 "reason" 키 (7/21: 종전 error만
                                # 읽어 스톨 사유가 전량 err=None 으로 기록됐다)
                                "err": r.get("error") or r.get("reason")})
        if el > max(2.0, expect * 2.0):
            self.log(f"  ⚠ 이동 지연 [{tag or 'move'}] {el:.1f}s "
                     f"(예상 {expect:.1f}s, d={dist:.2f}m dyaw={math.degrees(dyaw):+.0f}° "
                     f"max_v={max_v}) ok={r.get('ok')} err={r.get('error')}")
        return r

    def do_move_fire(self, dx, dy=0.0, max_v=None, tag="",
                     settle_s: float = 1.0) -> dict:
        """odom 단발 위치이동 + **정착 컷 1.0s** [2026-07-22 오후 조작자 지시].

        move_relative 자체는 펌웨어(엔코더) 위치이동이지만 do_move 는 DONE
        응답을 블로킹 대기한다(적재 구간 한도 90s) — 적재 푸시처럼 밀착이
        의도된 상황에선 정착이 영영 안 와 그라인드로 수 초를 전소한다.
        프로파일 예상시간 + settle_s 만 대기하고, 기한 초과면 비영 트위스트
        1틱(데드존 미만 wz — 실움직임 없음)으로 position move 를 선점
        해제(bridge 'preempted by cmd_vel')한 뒤 진행한다.
        legacy(nav!=street)는 선점 채널(snav)이 없어 종전 블로킹 유지.
        """
        if self.args.dry_run or self.snav is None:
            return self.do_move(dx, dy, max_v=max_v,
                                timeout=PLACE_MOVE_TIMEOUT_S, tag=tag)
        v = max_v or 0.233
        expect = math.hypot(dx, dy) / max(v, 0.05) + 0.7   # 가속램프+직렬 왕복
        r = self.do_move(dx, dy, max_v=max_v, timeout=expect + settle_s, tag=tag)
        if not r.get("ok") and (r.get("error") or r.get("reason")) == "timeout":
            self.snav._publish(0.0, 0.0, 0.05)
            self.spin(0.1)
            self.snav._stop()
            r = {"ok": True, "reason": "settle_cut", "settle_cut": True}
        return r

    def do_rotate_fire(self, dyaw: float, tag: str = "",
                       settle_s: float = ROT_SETTLE_CUT_S) -> dict:
        """제자리 회전 + **정착 컷** [2026-07-23 조작자 지시].

        do_move_fire 의 회전판. 펌웨어가 정착에 실패하면 자체 데드라인
        (프로파일 x2 + 5.0s)까지 갈리므로 — 03:15 실기에서 11.9° 회전 하나가
        5.89s 를 태웠다 — 프로파일 예상시간 + settle_s 만 기다리고
        비영 트위스트 1틱(데드존 미만 wz)으로 선점 해제한다. 남은 잔차는
        do_rotate 의 재회전(1회)과 15° 수용 문턱이 흡수한다.
        settle_s: 기본 ROT_SETTLE_CUT_S(1.0). 스캔 스핀은 잔차가 무해해
        SCAN_ROT_SETTLE_CUT_S(0.3) 를 넘겨 쓴다 (상수 주석 참조).
        legacy(nav!=street) 는 선점 채널(snav)이 없어 종전 블로킹 유지."""
        if self.args.dry_run or self.snav is None:
            return self.do_move(0.0, 0.0, dyaw, max_v=ROTATE_MAX_V, tag=tag)
        expect = rotate_profile_sec(dyaw, ROTATE_MAX_V) + ROT_OVERHEAD_S
        r = self.do_move(0.0, 0.0, dyaw, max_v=ROTATE_MAX_V,
                         timeout=expect + settle_s, tag=tag)
        if not r.get("ok") and (r.get("error") or r.get("reason")) == "timeout":
            self.snav._publish(0.0, 0.0, 0.05)
            self.spin(0.1)
            self.snav._stop()
            self.log(f"  [회전] 정착 컷 {expect + settle_s:.1f}s "
                     f"(예상 {expect:.1f}s) — 잔차는 재회전/후속 단계가 흡수")
            r = {"ok": True, "reason": "settle_cut", "settle_cut": True}
        return r

    def do_approach_fire(self, dx, dy=0.0, max_v=None, tag="접근",
                         settle_s: float = APPROACH_SETTLE_CUT_S) -> dict:
        """접근 단발이동 + **정착 컷** [2026-07-24 조작자 지시].

        do_move_fire(적재 밀착)·do_rotate_fire(회전)의 접근판. 접근은 카메라-depth
        로 목표를 잡아 odom 상대이동으로 실행되는데, 펌웨어가 정착 톨러런스
        (position_tolerance_rad 0.30 = 본체 1.2cm)를 50ms 유지 못 하면 데드라인까지
        갈린다 — 오늘 실기 35건 중 4건이 4.28~7.27s(정상군 1.03~1.39s)로 튀었다.
        정상 접근시간은 firmware 위치이동 프로파일(0.233m/s=position_max_rad_s 6.0)
        로 ±0.3s 안에 설명되므로, move_profile_sec+settle_s 만 기다리고 비영 트위스트
        1틱(데드존 미만 wz)으로 선점 해제한 뒤 진행한다.
        ⚠ 컷 시점엔 명령 변위를 이미 다 주행(+여유; 0.233m/s×timeout ≫ d)한 뒤라
          파지 정밀도에 무해 — 정체는 마지막 mm를 못 죽이는 것뿐이다.
        legacy(nav!=street)는 선점 채널(snav)이 없어 종전 블로킹 do_move 유지.
        근거·분포: docs/07-results-and-lessons.md §7.5.
        """
        if self.args.dry_run or self.snav is None:
            return self.do_move(dx, dy, max_v=max_v, tag=tag)
        expect = move_profile_sec(math.hypot(dx, dy), max_v)
        r = self.do_move(dx, dy, max_v=max_v, timeout=expect + settle_s, tag=tag)
        if not r.get("ok") and (r.get("error") or r.get("reason")) == "timeout":
            self.snav._publish(0.0, 0.0, 0.05)
            self.spin(0.1)
            self.snav._stop()
            self.log(f"  [접근] 정착 컷 {expect + settle_s:.1f}s "
                     f"(예상 {expect:.1f}s) — 도착 확정, 파지 진행")
            r = {"ok": True, "reason": "settle_cut", "settle_cut": True}
        return r

    def do_goto(self, x, y, max_v, settle: bool = True) -> bool:
        if self.args.dry_run:
            self.sim_pose[0], self.sim_pose[1] = float(x), float(y)
            self.log(f"  [dry-run] goto ({x:+.2f},{y:+.2f}) max_v={max_v}")
            return True
        # arena goal = localizer 폐루프 연속 제어. 속도는 real.yaml 캡 소관.
        # 7/21 17:12 실기: arena 접근 감속이 펌웨어 데드존 아래로 떨어지면
        # 목표 0.17~0.35m 앞에 주차 → 고정 25s 전소가 3회(75s+). 대응:
        # ① 거리 기반 적응 타임아웃 ② 정체 조기탈출(goto_xy_goal 내 2.5s/3cm)
        # ③ odom 단발이동 폴백 복원(7/20 제거분 — 위치제어라 데드존 무관).
        pose = self.pose()
        dist = math.hypot(x - pose[0], y - pose[1]) if pose is not None else 2.0
        to = max(6.0, dist / max(max_v or 0.3, 0.2) * 2.0 + 3.0)
        ok = self.fn.goto_xy_goal(x, y, timeout=to,
                                  settle=0.2 if settle else 0.0)
        if not ok:
            pose = self.pose()
            if pose is not None:
                dxw, dyw = x - pose[0], y - pose[1]
                rem = math.hypot(dxw, dyw)
                if rem <= 0.35:
                    # '정착 제거' (7/21 18:18): tol 바로 밖 잔차는 폴백 없이
                    # 수용 — 펌웨어 소거리 이동은 정착 그라인드로 8s를 태운다.
                    self.log(f"  [goto] 잔여 {rem:.2f}m 수용 — 주행하며 보정")
                    return True
                if rem < 0.8:
                    c, s = math.cos(pose[2]), math.sin(pose[2])
                    fwd = dxw * c + dyw * s
                    left = -dxw * s + dyw * c
                    self.log(f"  [goto폴백] 잔여 {rem:.2f}m — odom 단발이동 마무리")
                    r = self.do_move(fwd, left, max_v=min(max_v or 0.4, 0.4),
                                     tag='goto폴백')
                    ok = bool(r.get("ok"))
            if not ok:
                self.log(f"  goto ({x:+.2f},{y:+.2f}) 미도달 (폴백 후에도) — 계속 진행")
        return ok

    def do_rotate(self, yaw, tag: str = "회전", tol: float = 0.10) -> bool:
        """제자리 회전 (7/21 개편) — do_move 경유로 소요/성패를 move_stats에
        계측하고(종전 rotate_to_yaw는 전량 미계측 — 최대 낭비 항목이 병목
        리포트에 안 잡혔다), ROTATE_MAX_V로 회전 바퀴캡을 상향한다.
        완료 후 pose 잔차 >15°면 1회 재회전 — 메카넘 회전 슬립(k_eff 오차
        ~39%)·추정 지연이 남긴 오차를 다음 레그 전에 흡수 (7/21 16:17
        동일 -90° 정렬 3연속 재발의 방지책)."""
        if self.args.dry_run:
            self.sim_pose[2] = wrap_angle(yaw)
            self.log(f"  [dry-run] rotate → {math.degrees(yaw):+.1f}°")
            return True
        for attempt in (1, 2):
            pose = self.pose()
            if pose is None:
                return False
            err = wrap_angle(yaw - pose[2])
            # 미세 회전 생략 문턱 0.05→0.10rad (7/21 18:18 실기: 9° 회전이
            # 펌웨어 정착 그라인드로 2.9s — 소각 회전은 비용 > 이득.
            # 잔차는 접근 시각 측정/적재 게이트가 흡수).
            # [2026-07-22 04시] tol 파라미터화 — 적재 진입은 0.20(11.5°)으로
            # 완화 (드리프트가 이미 구석 정면 ±12° 안으로 도착).
            if abs(err) <= tol:
                return True
            if attempt == 2 and abs(err) <= math.radians(15.0):
                return True   # 잔차 15° 이내 — 재회전 비용 > 이득 (정렬 허용오차)
            # [2026-07-23] 정착 컷 경유 (do_rotate_fire) — 펌웨어 데드라인
            # (프로파일 x2+5.0s) 소각 방지.
            self.do_rotate_fire(err, tag=f"{tag}{attempt}")
            self.spin(0.15)   # pose 갱신 대기 후 잔차 재확인
        pose = self.pose()
        return (pose is not None
                and abs(wrap_angle(yaw - pose[2])) <= math.radians(15.0))

    def align_leg_yaw(self, wp):
        """street 레그 진입 전 **최소 잔차** 축정렬 (7/21 v3 — cardinal 스냅 폐기).

        목적은 통로 스윕폭 최소화: 로봇은 정사각+메카넘이라 전/후/좌/우 어느
        면이 진행 방향이어도 무방하다. 진행 방향과 로봇 4축(yaw+k·90°)의
        오프셋이 ±{ALIGN_TOL}° 이내면 무회전, 초과 시 잔차만 회전(최대 45°).

        v1(진행 방향 cardinal로 몸통 스냅, 최대 180° 회전)은 16:13/16:16
        실기에서 사이클당 90/180° 회전 2~4회를 만들었고 두 런 모두 로컬라이저
        붕괴(미러 오수렴·START 코너 투하)로 이어져 폐기. 대조군 14:32 런
        (동일 FHD 카메라, 회전 로직 없음)은 무사고 — 회전 폭증이 원인 상관.
        제자리 회전은 회전 중 라이다 스캔 스미어 + 메카넘 슬립 odom 오차로
        yaw 추정을 열화시키고, 정사각 아레나 wall_range 는 90° 대칭이라
        yaw 오차가 ±45°를 넘으면 미러 모드로 오수렴할 수 있다.

        15° 오차의 스윕 반폭 = 0.095·cos15°+0.129·sin15° ≈ 0.125m
        (+물체 반폭 0.04 = 0.165) ≤ STREET_CLEAR_M 0.20 — 기하 안전."""
        pose = self.pose()
        if pose is None:
            return
        if pose[1] < HIGHWAY_FREE_Y_M and wp[1] < HIGHWAY_FREE_Y_M:
            return  # 레그 전체가 하이웨이 안전밴드(공식 y<60cm) — 자유 yaw
        dx, dy = wp[0] - pose[0], wp[1] - pose[1]
        if math.hypot(dx, dy) < 0.35:
            return  # 짧은 레그 — 회전 비용이 충돌 위험 감소를 초과
        off = wrap_angle(math.atan2(dy, dx) - pose[2])
        # 4축 대칭 잔차: [-45°, +45°)
        off_axis = (off + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0
        if abs(off_axis) > math.radians(ALIGN_TOL_DEG):
            tgt = wrap_angle(pose[2] + off_axis)
            self.log(f"  [street] 레그 축정렬 잔차 {math.degrees(off_axis):+.0f}° "
                     f"회전 → yaw {math.degrees(tgt):+.0f}°")
            self.do_rotate(tgt)

    def follow_route(self, route, max_v, settle: bool = True):
        """경로 실행 — street 계열 모드는 레그별 cardinal 정렬 후 주행.

        7/21 16:16 실기 완화: ① 마지막 레그는 정렬 생략 — 직후 FACE(수거)/
        구석 정면(적재) 회전이 반드시 따라와 이중 회전 낭비였다 (16:18:02
        -90° 정렬 직후 +48° 재회전 9s 실측). ② 중간 경유점은 settle 없이
        통과 (settle은 최종점만)."""
        wps = route["waypoints"]
        align = route["mode"] in ("street", "street_forced")
        for i, wp in enumerate(wps):
            last = i == len(wps) - 1
            if align and not last:
                self.align_leg_yaw(wp)
            self.do_goto(wp[0], wp[1], max_v, settle=settle and last)

    # ---- street 주행 래퍼 (2026-07-22 통합 — navigation/docs/control-and-routing.md) ----
    # dry-run 게이트를 이 계층에서 처리한다 (do_move/do_goto 와 동일 관례).
    def street_x_snap(self, x: float) -> float:
        """가장 가까운 street 중심 x (map).

        후보와 클램프 범위는 STREET_XS_M 에서 유도한다 — 최서열(공식 x=25)은
        적재함과 겹쳐 쓰지 않으므로 [1] 부터다 (descend_candidate 의 금지와 동일).
        """
        lo, hi = STREET_XS_M[1], STREET_XS_M[-1]
        step = fl.GRID_PITCH_CM / 100.0
        k = round((float(x) - lo) / step)
        return max(lo, min(hi, lo + step * k))

    def descent_plan(self, cell, pose) -> tuple:
        """(mirror, descend) — 판정은 모듈 함수 descend_candidate 가 한다.

        여기서는 현장 롤백 스위치(--no-descend)만 얹는다 [2026-07-24].
        """
        if getattr(self.args, "no_descend", False):
            return False, False
        return descend_candidate(cell, pose)

    def street_face(self, to_object: bool, mirror: bool = False) -> bool:
        """mini-goal 에서의 ±45° 회전 (불변식 I-2 — 회전은 여기서만).

        mirror=True (남서 대각 mini-goal, 2026-07-23 하산 파지)면 물체가
        북동 대각이라 부호가 반대다 — 북향 복귀는 절대각이라 동일."""
        if self.args.dry_run:
            self.sim_pose[2] = math.pi * (0.5 if not to_object
                                          else (0.25 if mirror else 0.75))
            self.log("  [dry-run] street "
                     + (("CW45 → 물체 정면(미러)" if mirror
                         else "CCW45 → 물체 정면") if to_object else "→ 북향"))
            return True
        r = (self.snav.face_object(mirror=mirror) if to_object
             else self.snav.face_north())
        if not r.get("ok"):
            # 실패해도 중단하지 않는다 — 이후 이동의 I-5 게이트(북향 복원)나
            # 접근 측정 실패가 각자 안전하게 처리한다.
            self.log(f"  ⚠ [street] 회전 미수렴 ({r.get('reason')}) — 계속")
        return bool(r.get("ok"))

    def street_south(self, x: float, tag: str) -> bool:
        """x 의 street 를 따라 하이웨이(snv.HIGHWAY_Y_M)까지 남하 (북향=후진).

        [2026-07-22] 실패 시 trace/횡통계를 report.street_failures 에 보존 —
        00:50 실기에서 이탈 원인(드리프트 vs 로컬 튐)을 로그만으론 판별할 수
        없었다. 다음 실기부턴 틱 단위 궤적이 남는다.

        [2026-07-23 — 조작자 승인] 보존 조건을 **실패 → 실패 or 횡복구 발생**
        으로 확대하고 키를 `street_failures` → `street_legs` 로 바꿨다.
        이 레그(스캔후_남하)는 01:10~01:20 실기에서 두 런 모두 13.6/15.5cm 로
        street 여유 11.5cm 를 넘었는데 **`ok=True` 로 끝나** 통계·trace 가 전혀
        남지 않았다 (텍스트 로그의 한 줄이 전부였다). 이탈의 원인을 지연 대
        드리프트로 가르려면 이 레그의 틱 궤적이 반드시 필요하다."""
        if self.args.dry_run:
            self.sim_pose[:] = [float(x), snv.HIGHWAY_Y_M, math.pi / 2.0]
            self.log(f"  [dry-run] street 남하 ({x:+.2f},{snv.HIGHWAY_Y_M:+.2f})")
            return True
        r = self.snav.drive_to((float(x), snv.HIGHWAY_Y_M), tag,
                               v_max=snv.V_STREET_MPS, along="forward",
                               pos_tol=snv.POS_TOL_ROUGH_M,
                               yaw_tol=snv.YAW_ARRIVE_ROUGH_RAD)   # 남하 = 중간 레그 rough
        if not r.get("ok"):
            self.log(f"  ⚠ [street] 남하 실패 ({r.get('reason')})")
        elif r.get("recoveries"):
            self.log(f"  [street] {tag} 도착 — 횡복구 {r['recoveries']}회 "
                     f"(최대 {r.get('max_cross_m')}m)")
        # [2026-07-23] 성공했더라도 횡복구가 한 번이라도 있었으면 보존한다
        # (docstring 참조 — 종전 `if not ok` 는 15.5cm 이탈을 통째로 놓쳤다).
        if not r.get("ok") or r.get("recoveries"):
            self.report.setdefault("street_legs", []).append(
                {"tag": tag, "ok": bool(r.get("ok")), "reason": r.get("reason"),
                 "max_cross_m": r.get("max_cross_m"),
                 "recoveries": r.get("recoveries"),
                 "trace": r.get("trace")})
        return bool(r.get("ok"))

    def street_home(self, tag: str = "필드이탈_남하") -> bool:
        """어디에 있든 street 규칙으로 하이웨이 자유밴드까지 복귀.

        스캔점/mini-goal 등 실전 위치는 항상 street 위라 최근접 street x 로
        스냅해 남하한다 (스냅 잔차 ≤12.5cm 는 drive_to 횡보정이 흡수)."""
        if self.args.dry_run:
            self.sim_pose[1] = snv.HIGHWAY_Y_M
            self.sim_pose[2] = math.pi / 2.0
            return True
        pose = self.pose()
        if pose is None:
            return False
        # [2026-07-22] 하이웨이 라인이 자유밴드 경계(-1.40)와 같아져 도착 잔차
        # (±4cm)가 문턱을 넘나든다 — 5cm 허용해 무의미한 2cm 재남하를 막는다.
        # [2026-07-22 오후] +0.05→+0.15: 얕은 핀(2·3) 적재 후퇴 후 y≈-1.32 가
        # 남하 재트리거되면 복귀 드리프트(go_to_mini_goal 문턱 동일값)가 못
        # 뜬다. y≤-1.25 회랑은 물체 최남단행(연 -1.04)과 여유 확인됨.
        if pose[1] <= HIGHWAY_FREE_Y_M + 0.15:
            return True   # 이미 남쪽 자유밴드 (공식 y<60cm)
        self.snav.release_goal()   # arena goal 추종과 싸우지 않게 (STOP)
        return self.street_south(self.street_x_snap(pose[0]), tag)

    def do_gripper_release(self, wait: float = 1.0):
        """적재 투하 — *현재 손가락 위치 기준* 상대 틱만큼만 벌린다.

        물체를 물고 있으면 손가락은 closed_deg 가 아니라 물체 폭에서 멈춰
        있으므로 절대각(SET_DEG 110 등)은 물체마다 개방량이 달라진다.
        /gripper/state 의 present_deg 를 읽어 +ticks 만큼만 연다.
        """
        ticks = self.release_ticks
        delta = ticks * DXL_DEG_PER_TICK
        if self.args.dry_run:
            self.log(f"  [dry-run] 그리퍼 상대 개방 +{ticks:.0f}tick (+{delta:.1f}°)")
            return
        st = self.fn.gripper_status()
        cur = st.get("present_deg")
        if cur is None:
            cur = st.get("closed_deg")
            self.log("  [투하] present_deg 없음 — closed_deg 기준으로 폴백")
        if cur is None:
            self.log("  [투하] 그리퍼 상태 없음 — OPEN 폴백")
            self.do_gripper("OPEN", wait=wait)
            return
        tgt = float(cur) + delta
        self.log(f"  [투하] {float(cur):.1f}° → {tgt:.1f}° "
                 f"(+{ticks:.0f}tick / +{delta:.1f}°)")
        self.do_gripper(f"SET_DEG {tgt:.2f}", wait=wait)

    def do_gripper(self, cmd: str, wait: float = 0.0):
        if self.args.dry_run:
            self.log(f"  [dry-run] 그리퍼 {cmd}")
            return
        self.fn.gripper(cmd)
        if wait:
            self.fn.spin_for(wait)

    # ---- 모델 ----
    def load_models(self) -> bool:
        """프로그램 시작 시 기동된 프리로드 스레드에서 결과 인수.
        (프리로드 미기동이면 여기서 기동 후 대기 — 동작 동일, 시간만 소모)"""
        if self.models is not None:
            return True
        if _PRELOAD["thread"] is None:
            start_model_preload(self.args.pair != "off")
        th = _PRELOAD["thread"]
        if th.is_alive():
            self.log("  모델 프리로드 완료 대기중...")
            th.join(timeout=120.0)
        if _PRELOAD["models"] is None:
            self.log(f"모델 로드 실패: {_PRELOAD['err'] or '프리로드 미완료'}")
            return False
        self.models = _PRELOAD["models"]
        if _PRELOAD["warmup_warn"]:
            self.log(f"모델 워밍업 경고: {_PRELOAD['warmup_warn']}")
        self.log(f"YOLO 2단계 모델 로드+워밍업 {_PRELOAD['sec']:.1f}s "
                 "(A1 cube_face + unified_face, 시작과 동시 프리로드)")
        if _PRELOAD.get("a1_backend"):
            self.log(f"  A1 백엔드: {_PRELOAD['a1_backend']}")
        # [2026-07-24] face 가중치 A/B 추적 — FACE_WEIGHTS 오버라이드로 돌린
        # 런과 기본 런을 사후에 반드시 구분할 수 있어야 한다 (fl.FACE_WEIGHTS 주석).
        face_rel = str(fl.FACE_WEIGHTS)
        is_default = fl.FACE_WEIGHTS == fl._FACE_DEFAULT
        self.report.setdefault("models", {})["face"] = face_rel
        self.report["models"]["face_default"] = is_default
        self.report["models"]["a1_backend"] = _PRELOAD.get("a1_backend")
        self.log(f"  face 가중치: {Path(face_rel).parent.name}/{Path(face_rel).name}"
                 + ("" if is_default else "   ⚠ 기본값 아님(FACE_WEIGHTS 오버라이드)"))
        if self.args.pair != "off" and self.pair is None:
            if _PRELOAD["pair"] is not None:
                self.pair = _PRELOAD["pair"]
                # [2026-07-24] AO/BP 버전 표기 — AO_WEIGHTS/BP_WEIGHTS 오버라이드로
                # 돌린 런과 기본 런을 사후에 구분해야 한다 (PairVerifier 주석).
                # [2026-07-24 조작자 지시] AO/BP 기본값 모두 실사 재학습본으로 승격됨.
                # [폐기 2026-07-24 — 구 기본값 판정. 롤백 시 복원]
                #     ao_default = (ao_ver == "pair_apple_orange_v2")
                #     bp_default = (bp_ver == "pair_banana_pineapple_v2_1")
                ao_ver = getattr(self.pair, "ao_version", "?")
                bp_ver = getattr(self.pair, "bp_version", "?")
                ao_default = (ao_ver == "pair_apple_orange_real_20260724")
                bp_default = (bp_ver == "pair_banana_pineapple_real_v1")
                self.report.setdefault("models", {})["ao"] = ao_ver
                self.report["models"]["ao_default"] = ao_default
                self.report["models"]["bp"] = bp_ver
                self.report["models"]["bp_default"] = bp_default
                # 활성 라우트에 실제로 쓰이는 검증기만 경고 대상으로 본다.
                warns = []
                if "AO" in self.pair_routes and not ao_default:
                    warns.append("AO_WEIGHTS")
                if "BP" in self.pair_routes and not bp_default:
                    warns.append("BP_WEIGHTS")
                self.log("pair 검증기 ON — 활성 라우트 "
                         f"{'/'.join(sorted(self.pair_routes))} 재검증"
                         f" · AO={ao_ver} · BP={bp_ver}"
                         + ("" if not warns
                            else f"   ⚠ 기본값 아님({'/'.join(warns)} 오버라이드)"))
            else:
                self.log(f"pair 검증기 로드 실패({_PRELOAD['pair_err']}) "
                         "— face 단독 모드")
        return True

    def intr(self, cam: str):
        if self.fn is None:
            return None
        k = self.fn.camera_k(cam)
        return Intrinsics.from_camera_info(k) if k else None

    def make_stitcher(self, mast: str) -> fl.Stitcher:
        """카메라 해상도에 맞는 스티처.

        캘리브(up/down.json)는 파일의 res 좌표계 그대로 로드된다 —
        2026-07-21 최종본은 FHD(1920x1080) 네이티브(마스트 up/down 각각)라
        FHD 라이브에서는 재스케일 없이 그대로 쓰이고, 다른 해상도로 돌면
        라이브 camera_info(없으면 크롭 모델)로 재스케일된다."""
        msg = self.fn.last.get("top_rgb") if self.fn is not None else None
        if msg is None or (int(msg.width), int(msg.height)) == fl.CALIB_REF_RES:
            return fl.Stitcher(mast)
        intr = {}
        for cam in ("top", "near"):
            k = self.fn.camera_k(cam)
            intr[cam] = (float(k[0]), float(k[4]), float(k[2]), float(k[5])) if k else None
        if any(v is None for v in intr.values()):
            intr = None       # camera_info 미수신 → 중앙크롭 모델 폴백
        return fl.Stitcher(mast, w=int(msg.width), h=int(msg.height), intr=intr)

    def _detect_stitched(self, stitched_rgb: np.ndarray, st: fl.Stitcher, conf: float):
        """스티치 RGB → A1 추론(BGR) → 검출별 바닥접점의 원본 캠 픽셀.
        반환 (dets, bgr). det={cls,conf,u,v,cam,u_src,v_src,box}."""
        a1 = self.models[0]
        bgr = np.ascontiguousarray(stitched_rgb[:, :, ::-1])
        r = a1.predict(bgr, imgsz=fl.A1_IMGSZ_STITCHED, conf=conf, verbose=False)[0]
        return self._dets_from_result(r, st), bgr

    @staticmethod
    def _dets_from_result(r, st: fl.Stitcher) -> list:
        """A1 결과 1건 → 검출별 바닥접점의 원본 캠 픽셀 목록.

        st=None 이면 입력이 near 원본 프레임 — 역매핑 없이 그대로 near
        픽셀로 취급한다 (2026-07-23 접근 near 단독 추론 경로)."""
        dets = []
        if r.masks is None:
            return dets
        h, w = r.orig_shape
        for mask_t, box in zip(r.masks.data, r.boxes):
            mask = mask_t.cpu().numpy()
            if mask.shape != (h, w) and h * w > 1_500_000:
                # 고해상도(FHD 스티치 ~1920x2700): 원본 크기 리샘플은 인스턴스당
                # ~20MB 라 OOM 위험(7/20 65샷 사고 전례) — 바닥접점을 모델
                # 스케일에서 구하고 좌표만 확대한다. 모델 마스크 1px ≈ 원본 3px
                # = 640 기준 1px 상당이라 각도 정밀도는 기존과 동등.
                mh, mw = mask.shape
                mb = mask > 0.5
                rows = np.nonzero(mb.any(axis=1))[0]
                if rows.size == 0:
                    continue
                rb = int(rows.max())
                xs = np.nonzero(mb[max(0, rb - 1):rb + 1].any(axis=0))[0]
                u = float(xs.mean() * (w - 1) / (mw - 1))
                v = float(rb * (h - 1) / (mh - 1))
            else:
                if mask.shape != (h, w):
                    yi = np.linspace(0, mask.shape[0] - 1, h).astype(int)
                    xi = np.linspace(0, mask.shape[1] - 1, w).astype(int)
                    mask = mask[yi][:, xi]
                mask = mask > 0.5
                if not mask.any():
                    continue
                u, v = mask_bottom_uv(mask)
            cam, u_src, v_src = (("near", u, v) if st is None
                                 else st.to_source(u, v))  # 3D는 원본 캠에서만
            dets.append({"cls": r.names[int(box.cls)], "conf": float(box.conf),
                         "u": u, "v": v, "cam": cam,
                         "u_src": float(u_src), "v_src": float(v_src),
                         "box": [int(t) for t in box.xyxy[0].tolist()]})
        return dets

    @staticmethod
    def _face_identity(res):
        if getattr(res, "probs", None) is not None:
            return res.names[int(res.probs.top1)], float(res.probs.top1conf)
        faces = [(res.names[int(b.cls)], float(b.conf)) for b in (res.boxes or [])]
        return fl.face_vote(faces)

    # =====================================================================
    # 단계들
    # =====================================================================
    def stage_seed(self):
        t0 = time.monotonic()
        x, y, yaw = fl.START_POSE
        self.log(f"[SEED] 출발 포즈 시딩 ({x:+.2f},{y:+.2f}, yaw {math.degrees(yaw):.0f}°)")
        self.sim_pose = list(fl.START_POSE)
        loc = {}
        ok = False
        if self.fn is not None:
            self.fn.seed_pose(x, y, yaw)
            for _ in range(12):          # 0.25s x12 = 총 3s 창 (7/21 폴링 세분화)
                self.fn.spin_for(0.25)
                loc = self.fn.arena_status().get("localization") or {}
                lat = loc.get("latency_ms")
                if loc and lat is not None and float(lat) < 150.0:
                    ok = True
                    break
        if ok:
            self.log(f"  localization 락 OK — reason={loc.get('reason')} "
                     f"latency={loc.get('latency_ms')}ms")
        else:
            self.log("  localization 락 실패 (status.localization 부재 또는 latency>=150ms)")
            if not self.args.dry_run:
                self.mark("SEED", False, t0, loc)
                print("\n중단: localization 없이는 주행 불가. 스택/LiDAR 확인 후 재시도.")
                sys.exit(4)
            self.log("  [dry-run] 시뮬 포즈로 계속")
        self.mark("SEED", True, t0, loc)

    def stage_mast(self, cmd: str, label: str, wait: bool = True):
        t0 = time.monotonic()
        self._mast_assume = (MAST_DOWN_ASSUME_S if "BOTTOM" in cmd else
                             MAST_MID_ASSUME_S if "MID" in cmd else
                             MAST_UP_ASSUME_S)
        self.log(f"[{label}] {cmd} — /lift/command 브리지 경유 (tty 직접 금지)"
                 + ("" if wait else " (비대기 — 주행과 병렬 진행)"))
        if self.args.dry_run:
            self.log("  [dry-run] 리프트 명령 생략")
            self.mark(label, True, t0, "dry-run")
            return
        # [2026-07-23 계측] 명령 발행 지연 측정. 조작자 관측 "마스트가 명령
        # 몇 초 뒤에야 내려온다"의 원인 후보가 둘인데 로그로 안 갈렸다:
        #   (A) 이 프로세스의 GIL 경합 — 배치추론 스레드(6~7s)가 GIL 을 쥐고
        #       있어 위 로그 다음 줄인 publish 자체가 늦게 실행된다.
        #   (B) 별도 프로세스인 gripper_bridge_node 의 콜백→시리얼 write 지연.
        # publish 앞뒤를 재면 (A)만 여기에 잡힌다. (A)가 아니면 (B)로 좁혀진다.
        # 15:45 실기 근거: MAST_DOWN 명령 직후 카메라 스트림이 2.36초 완전
        # 정지했다(east 배치추론 6.0s 진행 중) — 프로세스 포화는 확실하나
        # 그게 리프트 명령까지 늦췄는지는 이 계측 없이는 알 수 없다.
        t_pub = time.monotonic()
        self.fn.lift(cmd)
        pub_lat = time.monotonic() - t_pub
        if pub_lat > 0.1:
            self.log(f"  ⚠ [마스트] /lift/command 발행에 {pub_lat:.2f}s "
                     f"— 프로세스 포화(GIL) 의심")
        self.report.setdefault("mast_cmd_latency", []).append(
            {"cmd": cmd, "pub_sec": round(pub_lat, 3)})
        if not wait:
            self._mast_pending = (label, t0)
            return
        self._mast_finish(label, t0)

    def stage_mast_wait(self):
        """비대기 stage_mast의 완료 동기화 (스캔 전 필수 — 캘리브가 높이 의존)."""
        pending = getattr(self, "_mast_pending", None)
        if pending is None:
            return
        self._mast_pending = None
        label, t0 = pending
        self.log(f"[{label}] 이동 완료 대기 (주행과 병렬로 올라가던 리프트)")
        self._mast_finish(label, t0)

    def _mast_finish(self, label: str, t0: float):
        # MOVING=0 관측 OR 실측 기반 시간 상한 — 둘 중 먼저 성립하는 쪽 (7/20).
        # 비대기(wait=False) 경로는 명령 시점 t0 이후 이미 흐른 시간을 빼서
        # "명령 후 N초"가 되도록 남은 예산만 넘긴다.
        assume = getattr(self, "_mast_assume", MAST_UP_ASSUME_S)
        assume = max(0.0, assume - (time.monotonic() - t0))
        ok, why = self.fn.lift_wait_idle(timeout=30.0, assume_done_s=assume)
        if ok:
            self.log(f"  {label} 완료 — {why} / 명령 후 {time.monotonic() - t0:.1f}s")
            self.mark(label, True, t0, why)
            return
        self.log(f"  {label} 이동 완료 미확인 (LIFT_STATUS 폴링 타임아웃)")
        if not self.args.yes:
            a = input(f"[체크] 마스트가 실제로 {label} 위치에 도달했나요? (y/n): ").strip().lower()
            if a in ("y", "yes", "ㅛ"):
                self.mark(label, True, t0, "operator-confirmed")
                return
        self.mark(label, False, t0)
        print(f"\n중단: {label} 확인 실패 — 스티치/역투영 캘리브가 마스트 위치에 "
              "의존하므로 진행 불가.")
        sys.exit(5)

    def _take_preshot(self):
        """레그1 종료 지점(마스트업, 스캔지점 x)에서 북향 1샷 캡처 — 인접 4셀 전담.

        추론은 스캔 배치에 합류하므로 모델 로드 대기가 없다 (캡처 ~1.5s).
        캡처 실패 시 프리샷을 비활성화해 4셀은 종전대로 스핀 표를 쓴다."""
        self._preshot_cap, self._preshot_cells = None, set()
        if not PRESHOT_ENABLED:
            return
        if self.args.dry_run or self.fn is None:
            self.log("[프리샷] dry-run/ROS 없음 — 생략 (4셀은 스핀 표 사용)")
            return
        self.do_rotate(math.pi / 2.0, tag="프리샷정면")  # 북향 (출발 yaw라 보통 무회전)
        cap = self._capture_one(99, fresh_timeout=3.0)
        if "frames" not in cap:
            self.log(f"[프리샷] 캡처 실패({cap.get('note')}) — 4셀은 스핀 표 사용")
            return
        self._preshot_cap = cap
        self._preshot_cells = preshot_cells()
        self.log(f"[프리샷] 캡처 완료 (yaw={cap.get('yaw_deg')}) — 전담 4셀 "
                 f"{sorted(self._preshot_cells)}")

    def stage_goto_center(self, preshot: bool = False, mast_cmd: str = None,
                          mast_label: str = None):
        """스캔 지점까지 직각 2레그 street 경로 (대각선 금지).

        레그1: 하단 하이웨이(y<-1.0, 규칙상 객체 없음)를 따라 서쪽으로
               스캔 지점의 x(=공식 225, 격자열 200/250 사이 street)까지.
               마스트 상승은 레그1(서진) 시작과 동시에 걸어 주행과 병렬 (2026-07-21).
        (preshot) 레그1 종료 지점(스캔지점 x 도달)에서 마스트업 북향 1샷 —
               스캔점 인접 4셀 전담 (7/21 3번 방안: 스핀 스캔의 0.35m 구조적
               사각 해소). 캘리브가 높이 의존이라 촬영 직전 상승 완료 동기화.
        레그2: 그 street를 따라 북쪽으로 스캔 지점까지.
        street 중심은 격자점에서 25cm — 객체 반폭 4cm + 로봇 반폭 9.5cm
        여유로 통과 가능. 대각선 주행은 격자점 위를 지나 충돌 위험.
        """
        t0 = time.monotonic()
        cx, cy = fl.CENTER_SCAN_XY
        p = self.pose()
        # [A8] 종전 리터럴 -1.80/-1.25 를 유도로. 4m 에서는 값이 동일하다
        # (START_POSE[1] = -1.80, HIGHWAY_Y_M = 공식 75cm = -1.25).
        start_y = p[1] if p else fl.START_POSE[1]
        # 하이웨이 y밴드로 클램프 (이미 하이웨이 안이면 현재 y 유지)
        hw_y = min(start_y, HIGHWAY_Y_M)
        # [2026-07-22 조작자 지시] first-scan(프리샷) 진입은 서진→북진 L 대신
        # 단일 대각(메카넘 홀로노믹). 대각 종점은 자유밴드 경계(하이웨이 라인
        # y=-1.40) — 프리샷 y(-1.25)까지 곧장 대각을 올리면 물체열 x=0.5 교차
        # 시 북향 그리퍼 선단(-y+0.26)과 최남단 실루엣(-1.04) 간격이 ~3cm 로
        # 붕괴한다. 잔여 15cm 는 street 북진으로 마무리 (사실상 단일 대각).
        # [2026-07-23] 대각 진입은 프리샷과 무관하게 유지한다. 종전엔 street_diag
        # 가 preshot 플래그에 묶여 있어, 프리샷만 끄면 느린 L자 경로로 되돌아가
        # 오히려 손해였다 (대각은 7/22 조작자 지시로 채택된 더 빠른 경로).
        street_diag = (preshot and self.nav_mode == "street"
                       and self.snav is not None and not self.args.dry_run)
        route_name = "street_diag" if street_diag else "street_L"
        self.log(f"[GOTO_CENTER] {route_name} → ({cx:+.2f},{cy:+.2f}) "
                 f"(max_v {self.v_cruise})")
        if mast_cmd:
            self.stage_mast(mast_cmd, mast_label, wait=False)
        if street_diag:
            # 프리샷 지점과 센터 스캔점은 x 가 같다(공식 225 = map 0.25) — 대각
            # 종점을 센터의 x 로 잡으면 프리샷 유무와 무관하게 같은 경로다.
            dx = fl.CENTER_SCAN_XY[0]
            self.log(f"  대각 진입 → ({dx:+.2f},{snv.HIGHWAY_Y_M:+.2f}) "
                     f"[L자 폐지 2026-07-22]")
            self.snav.release_goal()
            r1 = self.snav.drive_free((dx, snv.HIGHWAY_Y_M), "첫스캔_대각",
                                      v_max=min(self.v_cruise, 0.6))
            ok1 = bool(r1.get("ok"))
            if not ok1:
                self.log(f"  ⚠ 대각 미수렴 ({r1.get('reason')}) — goto 폴백")
                ok1 = self.do_goto(dx, snv.HIGHWAY_Y_M, self.v_cruise,
                                   settle=False)
            if PRESHOT_ENABLED:
                px, py = fl.PRESHOT_XY
                self.log(f"  프리샷 북진 ({px:+.2f},{py:+.2f})")
                r1b = self.snav.drive_to((px, py), "프리샷_북진",
                                         v_max=self.v_cruise,
                                         pos_tol=snv.POS_TOL_ROUGH_M,
                                         yaw_tol=snv.YAW_ARRIVE_ROUGH_RAD)
                if not r1b.get("ok"):
                    self.log(f"  ⚠ 북진 미수렴 ({r1b.get('reason')}) — goto 폴백")
                    self.do_goto(px, py, self.v_cruise)
                # 프리샷은 스캔과 동일 캘리브라 상승 완료 동기화가 필수였다.
                self.stage_mast_wait()
                self._take_preshot()
            # [2026-07-23] 프리샷을 끄면 여기서 마스트를 기다리지 않는다.
            # 리프트는 레그1 시작과 동시에 걸려 있고, **스캔 직전까지만** 완료되면
            # 되므로 대기를 run() 의 stage_mast_wait() (센터 도착 후)로 미룬다.
            # 종전엔 여기서 블로킹해 남은 북진 1.65m 를 리프트와 겹치지 못했다.
        else:
            self.log(f"  레그1: 하이웨이 서진 ({cx:+.2f},{hw_y:+.2f})")
            self.align_leg_yaw((cx, hw_y))
            ok1 = self.do_goto(cx, hw_y, self.v_cruise, settle=False)  # 중간점
            # [2026-07-23] 프리샷을 끄면 이 분기의 중간 정차·마스트 대기도 함께
            # 빠진다 (안 그러면 캡처만 생략되고 정차/대기 비용은 그대로 낸다).
            if preshot and PRESHOT_ENABLED:
                # [2026-07-21] 프리샷을 레그1 종료 지점(하이웨이)이 아니라 street 를
                # 조금 올라간 fl.PRESHOT_XY(공식 225,75)에서 찍는다. 전담 셀까지
                # 1.82m→1.27m 로 가까워져 과일면 크롭이 커지고 중간 행 가려짐 여유도
                # 늘어난다 (폐기 사유 상세는 fieldlib.PRESHOT_XY 주석).
                # 레그2 경로 위의 지점이라 정지 횟수는 종전과 동일하다.
                px, py = fl.PRESHOT_XY
                self.log(f"  프리샷 지점 북진 ({px:+.2f},{py:+.2f})")
                self.align_leg_yaw((px, py))
                self.do_goto(px, py, self.v_cruise)
                self.stage_mast_wait()  # 프리샷은 스캔과 동일 캘리브 — 상승 완료 필수
                self._take_preshot()
        self.log(f"  레그2: street 북진 ({cx:+.2f},{cy:+.2f})")
        self.align_leg_yaw((cx, cy))
        ok2 = self.do_goto(cx, cy, self.v_cruise)
        ok = bool(ok1 and ok2)
        p = self.pose()
        if p:
            self.log(f"  도착 pose ({p[0]:+.2f},{p[1]:+.2f}, {math.degrees(p[2]):+.0f}°)")
        self.mark("GOTO_CENTER", ok, t0, {"pose": p, "route": route_name})

    # ---- SCAN ----
    def stage_scan(self):
        t0 = time.monotonic()
        self.log(f"[SCAN] 마스트업 {self.args.scan_mode} 스캔 시작 "
                 f"({self.args.scan_shots}스텝, votes_k={self.votes_k} "
                 f"fruit_k={self.fruit_k}, {self.args.range_mode} 역투영)")
        self.scan_dir.mkdir(parents=True, exist_ok=True)
        if self.fn is not None:
            self.fn.publish("control", "STOP")  # arena goal 해제 (70번 계약)
            self.spin(0.4)

        caps = []
        # [데모] 1샷 허용. 종전 하한 2 는 "항상 한 바퀴 돈다"는 전제였는데,
        # 6칸이 한 프레임에 들어오는 배치에서는 회전 자체가 불필요하다.
        # n==1 이면 아래에서 회전 루프·동서 분할·북향 닫기가 모두 빠진다.
        n = max(1, int(self.args.scan_shots))

        # --- 2~3단계 준비: 모델 대기 → 프리샷/배치 추론 → 투표 확정 ---
        # (촬영 루프보다 먼저 정의 — 동쪽 phase 스레드가 촬영 중에 시작된다)
        # [2026-07-22 조작자 지시] 추론은 GPU 작업이라 주행과 겹칠 수 있다 —
        # 백그라운드 스레드로 돌리고 그 동안 하이웨이로 남하한다. 도착 즈음
        # 투표가 끝나 수거 1사이클을 하이웨이에서 바로 시작 (사이클당 남하
        # 중복 제거 + 추론 대기 정지시간 제거, 합계 ~5-8s 절약).
        # 스레드 안전: _process_shots_batch 는 캡처된 프레임·모델·dict 읽기만
        # 하고 ROS spin 을 하지 않는다 (camera_k 는 fn.last dict 읽기 전용).
        infer_state = {}

        def _infer_phase(sub_caps, first: bool, final: bool):
            """부분 배치 추론+투표. first=모델/스티처/프리샷, final=최종 확정.
            오류문자열 반환(정상 None). [2026-07-22 오후 동서 2단계 분할]"""
            if first:
                have_models = self.load_models()
                if not have_models and not self.args.dry_run:
                    return "YOLO 모델 없이는 스캔 불가."
                mkey = getattr(self.args, "scan_mast", "up")
                infer_state["st"] = self.make_stitcher(mkey)
                infer_state["mounts"] = (
                    {"top": CameraMount(**fl.TOP_MOUNT_UP),
                     "near": CameraMount(**fl.NEAR_MOUNT_UP)}
                    if mkey == "up" else
                    {"top": CameraMount(**fl.TOP_MOUNT_MID),
                     "near": CameraMount(**fl.NEAR_MOUNT_MID)})
                pre_cells = (self._preshot_cells
                             if self._preshot_cap is not None else set())
                infer_state["pre_cells"] = pre_cells
                if pre_cells:
                    pre = self._process_shots_batch(
                        [self._preshot_cap], infer_state["st"],
                        infer_state["mounts"], only_cells=pre_cells)
                    self.report["scan"]["preshot"] = pre
                    for shot in pre:
                        self.log(f"  [프리샷 추론] 검출 {shot.get('detections', 0)}건 "
                                 f"→ 전담 4셀 투표 {shot.get('voted', 0)}")
            # [2026-07-23] 스핀 표 기각(exclude_cells) 제거 — 프리샷 전담 4셀도
            # 스핀 표를 정상 투표시킨다. 종전 기각 사유였던 "초근접 스핀표 불신"
            # (A1 이 초근접 물체를 아예 못 봄)은 하이브리드로 해결됐다. 이제 두
            # 소스가 역할을 나눈다: 프리샷=과일 정체(원거리 크롭), 스핀=물체 존재.
            # 실측 근거는 PRESHOT_ENABLED 주석.
            shots = self._process_shots_batch(
                sub_caps, infer_state["st"], infer_state["mounts"])
            for shot in shots:
                self.log(f"  [추론 {shot['shot'] + 1}/{n}] "
                         f"yaw={shot.get('yaw_deg')} 검출 "
                         f"{shot.get('detections', 0)}건 "
                         f"투표 {shot.get('voted', 0)} ({shot.get('note', 'ok')})")
            self.report["scan"].setdefault("shots", []).extend(shots)

            # 합성 폴백: dry-run + 프레임/모델 없음 + GT 존재 → GT로 합성
            if final:
                if not self.cell_votes and self.args.dry_run and self.gt:
                    self.log("  [dry-run] 실프레임 없음 — GT로 합성 스캔 "
                             "(수거 로직 리허설용)")
                    for cell, cls in self.gt.items():
                        for _ in range(self.votes_k):
                            self.cell_votes[cell].append(
                                {"identity": cls, "a1": cls, "a1_conf": 0.9,
                                 "face_conf": 0.9 if cls in fl.FRUITS else None,
                                 "cam": "sim", "range_m": 1.0, "shot": -1})
                self._finalize_scan()
            else:
                self._finalize_scan(partial=True)
            return None

        # [2026-07-22 오후] 동서 2단계: 촬영 루프가 동쪽 샷 완료 시점에
        # _scan_bg_start 로 phase A(동쪽) 추론을 조기 시작한다. phase B(서쪽)는
        # 촬영 완료 신호 후 이어서 돌고, 수거 루프는 동쪽 부분맵으로 먼저 출발
        # (pick_target_gated 가 우선순위 충돌 시 완성 대기).
        box = {}
        cap_done = threading.Event()
        self._scan_east_ready = threading.Event()
        self._scan_full_done = threading.Event()
        th_box = {"th": None}

        def _bg_two_phase(east_caps):
            try:
                e = _infer_phase(east_caps, first=True, final=False)
                if e:
                    box["err"] = e
                    self._scan_full_done.set()
                    return
                self._scan_east_ready.set()
                cap_done.wait(timeout=120.0)
                e = _infer_phase(caps[len(east_caps):], first=False, final=True)
                if e:
                    box["err2"] = e          # 서쪽 실패 — 동쪽 맵으로 계속
                    self.log(f"  ⚠ [스캔] 서쪽 추론 실패 — 동쪽 부분맵으로 진행: "
                             f"{e.splitlines()[0]}")
            except Exception:
                box["err"] = "추론 예외:\n" + traceback.format_exc()
            finally:
                self._scan_full_done.set()

        use_bg = (self.nav_mode == "street" and self.snav is not None
                  and not self.args.dry_run and self.fn is not None
                  and self.args.scan_mode != "continuous"
                  # 동서 분할은 샷이 2장 이상일 때만 의미가 있다. 1샷이면
                  # phase B 가 빈 리스트를 받게 되므로 단일 phase 로 간다.
                  and n >= 2)
        if use_bg:
            def _start(east_caps):
                self.log(f"  [스캔] 동쪽 {len(east_caps)}샷 추론 백그라운드 시작 "
                         "— 서쪽 촬영과 병렬")
                th_box["th"] = threading.Thread(
                    target=_bg_two_phase, args=(list(east_caps),), daemon=True)
                th_box["th"].start()
            self._scan_bg_start = _start
        else:
            self._scan_bg_start = None

        # --- 1단계: 촬영 (모델 로드와 무관 — 회전이 먼저 시작됨) ---
        err = None
        try:
            if self.args.scan_mode == "continuous" and not self.args.dry_run \
                    and self.fn is not None:
                self.log(f"  연속 회전 촬영 (추론은 회전 완료 후 일괄)")
                self.fn.last.pop("move_result", None)
                self.fn.publish("move",
                                {"dx": 0.0, "dy": 0.0, "dyaw": -2 * math.pi})
                k = 0
                deadline = time.monotonic() + 40.0
                while ("move_result" not in self.fn.last
                       and time.monotonic() < deadline):
                    cap = self._capture_one(k, fresh_timeout=1.0)
                    caps.append(cap)
                    self.log(f"  [촬영 {k + 1}] yaw={cap.get('yaw_deg')} "
                             f"({cap.get('note', 'ok')})")
                    k += 1
            else:
                step = -2 * math.pi / n  # CW (CCW 정착 타임아웃 회피)
                # [2026-07-22 오후] 동서 분할: CW 스캔은 앞 절반+1 샷이
                # 동반구(북→동→남)를 먼저 훑는다 — 그 시점에 동쪽 배치추론을
                # 백그라운드로 시작해 서쪽 촬영과 겹친다.
                east_n = n // 2 + 1
                # [데모] 조준 회전 — 촬영 전 1회, 절대 yaw. 경기 인자(None)면
                # 이 블록 자체가 건너뛰어져 종전 동작과 완전히 동일하다.
                aim = getattr(self.args, "scan_aim_yaw", None)
                if aim is not None and not self.args.dry_run and self.fn is not None:
                    self.log(f"  [스캔] 조준 회전 → yaw {aim:.0f}°")
                    self.do_rotate(math.radians(float(aim)), tol=0.10)
                # [폐기 2026-07-22 오후 — 롤백 시 t_gate 대신 복원]
                # self.spin(self.scan_settle_s)  # (루프 첫 줄) 정지 안정화
                # cap = self._capture_one(k)     # 카운트 기반 +2프레임 대기
                t_gate = time.monotonic() + self.scan_settle_s
                for k in range(n):
                    # 정착 대기와 신선프레임 대기 병렬 (stamp > 회전종료+정착)
                    cap = self._capture_one(k, min_stamp=t_gate)
                    caps.append(cap)
                    self.log(f"  [촬영 {k + 1}/{n}] yaw={cap.get('yaw_deg')} "
                             f"({cap.get('note', 'ok')})")
                    if k + 1 == east_n and self._scan_bg_start is not None:
                        self._scan_bg_start(caps[:east_n])  # 동쪽 추론 조기 시작
                    if k < n - 1:
                        # [2026-07-23 버그픽스] max_v 미전달 → 펌웨어 기본
                        # position_max_rad_s(6.0 = 본체 ~1.2rad/s)로만 돌았다.
                        # 07-22 저녁의 "제자리 회전 1.5배"(ROTATE_MAX_V 0.525)가
                        # do_rotate 경로에만 걸려 스캔 회전은 전혀 안 빨라진
                        # 원인. 로그 실측: 이 경로 회전 340건이 max_v=None,
                        # 45° 중앙값 1.00s (do_rotate 0.35 경로는 0.90s).
                        # [2026-07-23 조작자 지시] 정착 컷 경유 (1.5→0.5→0.3s,
                        # 30° 실효 timeout 1.47s — 값 근거는 상수 주석 참조).
                        # 05:43:54 실기 4번째 회전 3.21s(정상 0.77s) — 펌웨어
                        # 정착 그라인드가 스캔의 유일한 가변 병목이었다.
                        # 잘린 잔차는 무해하다: 이 샷의 yaw 는 회전 명령값이
                        # 아니라 아래 _capture_one 의 pose 에서 읽는다.
                        self.do_rotate_fire(step, tag='스캔회전',
                                            settle_s=SCAN_ROT_SETTLE_CUT_S)
                        t_gate = time.monotonic() + self.scan_settle_s
        finally:
            self._scan_bg_start = None

        # --- 1.5단계: 촬영이 끝났으면 마스트를 즉시 내린다 (7/20) ---
        # 추론(수 초)은 마스트 높이와 무관하므로 YOLO 앞에 걸어 하강 시간을
        # 추론과 겹친다. 근접캠 캘리브 의존 단계 전에 stage_mast_wait()로 동기화.
        if not self.args.skip_scan:
            self.stage_mast("LIFT_TO_BOTTOM", "MAST_DOWN", wait=False)
        # --- 1.6단계: 스핀을 북향으로 닫는다 [2026-07-23 조작자 지시] ---
        # 촬영 루프는 n샷에 회전 n-1 회다(마지막 샷 뒤에는 안 돌았다). 그래서
        # 스핀이 360°가 아니라 360-step 만큼만 돌고 끝났고 — 12샷이면 330° —
        # 스캔 종료 yaw 가 북향에서 한 스텝(30°) 어긋난 채 곧장 남하 street 에
        # 들어갔다. 15:45 실기 '스캔후_남하' 레그는 시작 0.7s 를 yaw 82~103°
        # 사이에서 흔들리며 횡복구 2회를 태웠다(street 은 북향 고정 가정).
        # 마스트 하강 명령(비대기)을 먼저 쏜 **뒤에** 돌아서, 이 회전 0.77s 가
        # 마스트 하강·서쪽 배치추론과 통째로 겹치게 한다 — 순 손실 ≈ 0.
        # continuous 모드는 이미 -2pi 를 한 번에 돌아 닫혀 있으므로 제외한다.
        if (not self.args.dry_run and self.fn is not None
                and self.args.scan_mode != "continuous" and len(caps) >= 2):
            self.do_rotate_fire(-2 * math.pi / max(2, int(self.args.scan_shots)),
                                tag='스캔북향닫기',
                                settle_s=SCAN_ROT_SETTLE_CUT_S)
        cap_done.set()
        if use_bg and th_box["th"] is None:
            # 스텝 수 부족 등으로 조기 시작 못 함 — 전량 단일 phase 폴백
            th_box["th"] = threading.Thread(
                target=_bg_two_phase, args=(list(caps),), daemon=True)
            th_box["th"].start()
        if use_bg:
            # [2026-07-23 조작자 지시 — 하산 파지 버그픽스] 종전엔 여기서 곧바로
            # 하이웨이까지 남하해(추론과 병렬) **하산 파지 기회를 스스로 없앴다**:
            # 대상 선정은 이 남하가 끝난 뒤에 일어나므로 descent_plan 의 "같은
            # street·북쪽" 조건이 영원히 거짓이었다 (03:15 실기 — x=250 오렌지가
            # 첫 대상이었는데도 하이웨이에서 우회 경로로 갔다).
            # 동쪽 부분맵이 뜰 때까지만 기다렸다가(03:15 실기 실측 촬영 종료
            # +1s — 동쪽 배치는 서쪽 촬영 중에 이미 돌고 있다) **하산 후보가
            # 있을 때만** 남하를 생략한다. 후보가 없으면 그 자리에서 종전대로
            # 남하해 서쪽 배치(실측 6.4s)와의 병렬 이득을 그대로 회수한다.
            self.log("  [스캔] 동쪽 추론 대기 — 하산 후보 확인 후 남하 결정")
            deadline = time.monotonic() + 90.0
            while (time.monotonic() < deadline
                   and not self._scan_east_ready.is_set()
                   and th_box["th"].is_alive()):
                time.sleep(0.2)
            if not self._scan_east_ready.is_set():
                err = box.get("err") or "추론 스레드 90s 타임아웃"
            pose0 = self.pose()
            if self.targets:
                # [2026-07-24 조작자 지시] order="nearest" 고정 — 이 목록은
                # **하산 후보 판정 전용**이라 세트 순서 선호를 태우면 안 된다.
                # 종전엔 여기에 fruit-first 가 걸려, x=250 형상 타깃이 있어도
                # 과일이 하나라도 있으면 desc0 가 비어 그대로 하이웨이로
                # 남하해버렸다 (pick_target 이 판단하기 전에 기회가 소멸).
                cand0 = match_candidates(self.cells, self.collected,
                                         self.skip_cells, self.targets,
                                         self.placed_by_cls, "nearest")
            else:
                cand0 = [c for c in self.cells
                         if c not in self.collected and c not in self.skip_cells]
            desc0 = [c for c in cand0 if self.descent_plan(c, pose0)[1]]
            if desc0:
                # [2026-07-24] 후보는 x=250 행뿐이고 방향은 남/북 둘 다다
                # (mini-goal 이 스캔점보다 북쪽이면 북진 한 레그).
                self.log(f"  [스캔] 하산 파지 후보 {len(desc0)}셀 "
                         f"{sorted(desc0)[:4]} — 남하 생략, 첫 사이클이 "
                         f"이 street 로 {max(desc0, key=lambda c: c[1])} 까지 직행")
            else:
                # ⚠ 서쪽 부분맵에만 하산 후보가 있는 경우는 놓친다 (동쪽
                #   확정 시점에 판단하므로). 스캔 street 양옆 두 행은 대부분
                #   동반구에 들어와 실무상 영향이 작다.
                self.log("  [스캔] 하산 후보 없음 — 하이웨이 남하 "
                         "(서쪽 추론과 병렬)")
                self.street_home("스캔후_남하")   # 실패해도 추론과 무관
        else:
            err = _infer_phase(caps, first=True, final=True)
            self._scan_full_done.set()
        if err:
            self.mark("SCAN", False, t0, err.splitlines()[0])
            print(f"\n중단: {err}")
            sys.exit(6)
        self.mark("SCAN", True, t0,
                  {"cells": len(self.cells), "presence": len(self.presence),
                   "east_first": use_bg and not self._scan_full_done.is_set()})

    def _pose_for_frame(self):
        """지금 손에 있는 프레임의 **촬영 시각에 정렬된** pose.

        [2026-07-23 신규 — 조작자 지시] 종전엔 프레임을 기다린 뒤 `self.pose()`
        로 **가장 최신** status 를 읽어 그 yaw 로 역투영했다. status 가 20Hz 면
        오차가 작지만 실측은 카메라 4스트림 구독만으로 6Hz·p90 0.43s·최악 3.15s
        라(status_latency_probe.py), 회전 잔여가 조금만 있어도 그만큼 yaw 가
        틀리고 range 를 곱해 셀 경계 25cm 를 넘는다 (14:02 실기 샷 8~12 등가
        yaw 오차 10.4° = 1.5m 에서 27.6cm).

        폴백 사다리 — **어느 단계에서도 촬영을 실패시키지 않는다**:
          ① 정상: 프레임 header.stamp 평균 → fn.pose_at() 보간
          ② 클럭 어긋남(>POSE_CLOCK_SANITY_S): 카메라 드라이버가 다른 클럭을
             쓰는 구성일 수 있다 — 정렬이 오히려 해로우므로 종전 방식
          ③ stamp 없음(구 arena 빌드 / 드라이버 미기입): 종전 방식
        반환: (pose|None, info dict) — info 는 리포트에 그대로 실어 다음 런에서
        정렬이 실제로 동작했는지 사후 판정한다.
        """
        if self.fn is None:
            return self.pose(), {"src": "no_ros"}
        latest = self.pose()
        fs = self.fn.frame_stamp()
        if fs is None:
            return latest, {"src": "no_frame_stamp"}
        t_frame, rgb_spread, all_spread = fs
        buf = list(self.fn.pose_buf)
        if not buf:
            return latest, {"src": "no_buf"}
        # RGB 쌍의 시각이 크게 벌어지면 스티치 자체가 서로 다른 순간의 두 장을
        # 붙인 것이라, 어떤 pose 로 정렬해도 한쪽은 틀린다. 정렬은 그대로 하되
        # (평균이 최선이다) 진단으로 남겨 다음 런에서 원인을 가른다.
        if rgb_spread > RGB_STAMP_SPREAD_WARN_S and not self._rgb_spread_warned:
            self._rgb_spread_warned = True
            self.log(f"  ⚠ [pose정렬] top/near RGB stamp 가 {rgb_spread * 1000:.0f}ms "
                     f"벌어짐 — 스티치 시간정합 확인 필요 (정렬은 평균으로 진행)")
        # 클럭 정합성: 같은 머신·같은 ROS 클럭이면 수십 ms 안이어야 한다.
        skew = abs(buf[-1][0] - t_frame)
        if skew > fl.POSE_CLOCK_SANITY_S:
            if not self._pose_skew_warned:
                self._pose_skew_warned = True
                self.log(f"  ⚠ [pose정렬] 프레임 stamp 와 arena stamp 가 "
                         f"{skew:.1f}s 어긋남 — 클럭 기준이 다르다. 정렬을 끄고 "
                         f"종전 '최신 pose' 방식으로 진행한다")
            return latest, {"src": "clock_skew", "skew": round(skew, 2)}
        aligned, info = self.fn.pose_at(t_frame)
        info["rgb_spread"] = round(rgb_spread, 3)
        info["all_spread"] = round(all_spread, 3)
        info["cam_lag"] = round(time.time() - t_frame, 3)   # 파이프라인 지연 실측
        if aligned is None:
            return latest, {**info, "src": info.get("src", "none") + "_fallback"}
        # 진단: 정렬로 실제로 얼마를 되돌렸는가 (yaw 차이). 0 이면 정지 상태라
        # 정렬이 무해했다는 뜻이고, 크면 그만큼 종전 방식이 틀리고 있었다는 뜻.
        if latest is not None:
            d = math.degrees(wrap_angle(aligned[2] - latest[2]))
            info["yaw_corr_deg"] = round(d, 2)
        return aligned, info

    def _capture_one(self, k: int, fresh_timeout: float | None = None,
                     min_stamp: float | None = None) -> dict:
        """프레임+pose만 캡처 (모델 불필요 — 추론은 _process_shot에서 일괄).

        min_stamp: [2026-07-22 오후] 지정 시 카운트(+2프레임) 대신 수신시각
        기반 대기(wait_frames_after) — 스캔 스텝의 정착과 프레임 대기 병렬화."""
        if fresh_timeout is None:
            fresh_timeout = 2.0 if self.args.dry_run else 8.0
        fresh_ok = True
        if self.fn is not None and min_stamp is not None:
            # 대기(=spin) 후 pose 를 읽어야 정착 후 pose 가 잡힌다.
            fresh_ok = self.fn.wait_frames_after(min_stamp, timeout=fresh_timeout)
        pose, pose_info = self._pose_for_frame()
        cap = {"shot": k, "pose": list(pose) if pose else None,
               "yaw_deg": round(math.degrees(pose[2]), 1) if pose else None,
               "pose_align": pose_info}
        if self.fn is None:
            cap["note"] = "ROS 없음"
            return cap
        if min_stamp is None:
            fresh_ok = self.fn.wait_fresh_frames(timeout=fresh_timeout)
        if not fresh_ok:
            cap["note"] = "신선 프레임 타임아웃"
            return cap
        rgb = self.fn.rgb_pair()
        depth = self.fn.depth_pair()
        if rgb is None or depth is None or cap["pose"] is None:
            cap["note"] = ("프레임 없음" if rgb is None or depth is None
                           else "pose 없음")
            return cap
        cap["frames"] = (rgb[0].copy(), rgb[1].copy(),
                         depth[0].copy(), depth[1].copy())
        return cap

    def _process_shots_batch(self, caps: list, st: fl.Stitcher,
                             mounts: dict, only_cells: set | None = None,
                             exclude_cells: set | None = None) -> list:
        """전 샷 일괄 추론: A1 1배치 → 기하/투표 → face 1배치 → pair 1배치.

        7/20 병목 개선: 샷별 순차 predict(~1.4s/샷 = A1 forward + face +
        PNG 저장 동기)를 GPU 배치 forward 3회(A1/face/pair)로 축약하고,
        PNG 저장은 백그라운드 스레드로 밀어 추론 경로에서 제거.
        단계별 소요는 report.scan.batch_timing에 기록."""
        shots = []
        idxs = []  # 프레임 있는 샷의 shots 인덱스
        for cap in caps:
            shot = {"shot": cap["shot"], "pose": cap.get("pose"),
                    "yaw_deg": cap.get("yaw_deg"),
                    # [2026-07-23] pose 시각정렬 진단 — src/gap/yaw_corr_deg.
                    # 다음 런에서 "정렬이 실제로 걸렸는지 + 얼마를 되돌렸는지"를
                    # 이 필드 하나로 판정한다 (_pose_for_frame 주석 참조).
                    "pose_align": cap.get("pose_align"),
                    "detections": 0, "voted": 0, "far": 0}
            if "frames" not in cap:
                shot["note"] = cap.get("note", "프레임 없음")
            elif self.models is None:
                shot["note"] = "모델 없음"
            else:
                idxs.append(len(shots))
            shots.append(shot)
        if not idxs:
            return shots

        timing = {}
        t0 = time.monotonic()
        stitched_l, bgr_l = [], []
        for i in idxs:
            top_rgb, near_rgb, _, _ = caps[i]["frames"]
            stitched = st.stitch(top_rgb, near_rgb)
            stitched_l.append(stitched)
            bgr_l.append(np.ascontiguousarray(stitched[:, :, ::-1]))
        timing["stitch"] = time.monotonic() - t0
        # [2026-07-24] GPU warp 가 실제로 탔는지 첫 배치에서 1회 로그 + report.
        if not getattr(self, "_warp_backend_logged", False):
            self._warp_backend_logged = True
            be = st.warp_backend()
            self.report.setdefault("scan", {})["warp_backend"] = be
            self.log(f"  [스티치] warp 백엔드: {be}"
                     + ("" if "GPU" in be else "  ⚠ CPU 폴백 — 경합으로 느려짐"))

        # --- A1: 전 샷 청크 배치 forward ---
        t0 = time.monotonic()
        results = _chunked_predict(
            self.models[0], bgr_l, A1_BATCH_CHUNK,
            imgsz=fl.A1_IMGSZ_STITCHED, conf=A1_SCAN_CONF, verbose=False)
        timing["a1"] = time.monotonic() - t0

        # --- A1 근접: near 원본 프레임 @512 (하이브리드, 2026-07-23) ---
        # 스티치 @896 은 배율 0.42 라 근접 물체를 네트 입력에서 250px 까지 키운다.
        # A1 은 mosaic=1.0(140에폭 중 130) + scale=0.5 로 학습돼 물체가 작게 보이는
        # 데 적응돼 있고, 네트 150px 을 넘으면 나빠지며 250px 에서는 conf 가
        # 입력 격자에 대해 카오스가 된다(픽셀 bit 동일한데 캔버스만 바꿔도 0.80→0.07).
        # → 근접 물체는 near 원본을 따로 본다.
        # [2026-07-24 갱신] 종전 주석 "두 경로는 카메라 영역으로 자연 분할되므로
        # 중복 제거가 불필요하다" 는 NEAR_STITCH_SPLIT 도입으로 더는 성립하지
        # 않는다 — seam 근처(near 상반부)는 스티치 표를 유지하므로 겹칠 수 있고,
        # 아래 스왑 블록에서 NEAR_DUP_PX 로 명시 중복 제거를 한다.
        near_results = None
        if A1_IMGSZ_SCAN_NEAR:
            t0 = time.monotonic()
            near_results = _chunked_predict(
                self.models[0],
                [np.ascontiguousarray(caps[i]["frames"][1][:, :, ::-1]) for i in idxs],
                A1_BATCH_CHUNK, imgsz=A1_IMGSZ_SCAN_NEAR, conf=A1_SCAN_CONF,
                verbose=False)
            timing["a1_near"] = time.monotonic() - t0

        # --- 기하: 원본 캠 depth 역투영 → 맵 변환 → 격자 스냅 (샷별) ---
        t0 = time.monotonic()
        intr = {c: self.intr(c) for c in ("top", "near")}
        dets_per_shot = {}
        crop_list, crop_det = [], []   # 전 샷 큐브 크롭 일괄 수집
        for j, i in enumerate(idxs):
            cap, shot = caps[i], shots[i]
            pose = tuple(cap["pose"])
            top_rgb, near_rgb, top_d, near_d = cap["frames"]
            depth_map = {"top": top_d, "near": near_d}
            bgr = bgr_l[j]
            h, w = bgr.shape[:2]
            dets = self._dets_from_result(results[j], st)
            if near_results is not None:
                # 근접(seam 아래 = cam "near")은 스티치 표를 버리고 near 원본 표로
                # 대체한다. cam 은 to_source() 가 정한다.
                #
                # [2026-07-24 조작자 지시] "아래쪽에 걸치면 무조건 버리는 게
                # 아니라, 하단캠의 절반보다 아래일 때만 버린다."
                # 근거: seam 인수 지점(지면 0.843m)은 near 프레임이 8cm 물체를
                # 온전히 담는 한계(0.793m)보다 **멀다** — 즉 seam 바로 아래
                # 물체는 near 원본에서 프레임 상단에 잘려 윗면(과일면)이
                # 통째로 사라진다. 그런데 스티치에서는 윗부분이 top 캠 조각에서
                # 와서 **온전히 보인다**.
                # 실측(20260723_223120 전 13샷, 버려진 15건의 near v_src):
                #   상반부 8건 = 113~249px  ← 전부 near 상단 잘림
                #   하반부 7건 = 885~1080px ← 전부 잘림 없음
                #   그 사이 249~885px(636px)는 완전한 빈 구간 → 문턱 0.5 는 안전.
                # 피해 실증: (250,300) 큐브가 스티치 크롭 face `orange 0.97`,
                # near 잘린 크롭 `plain` 만 → 맵이 plain 확정(orange 2개/plain 5개
                # 로 규정 카디널리티 위반). 20점 실점.
                # 하반부를 계속 near 로 넘기는 이유는 종전 그대로 — 초근접 큐브는
                # 스티치 @896(배율 0.438)에서 네트 250px 까지 커져 A1 학습 분포를
                # 벗어난다. 그 문제는 하반부에서만 생긴다.
                n_st = len(dets)
                near_split_v = near_rgb.shape[0] * NEAR_STITCH_SPLIT
                dets = [d for d in dets
                        if d["cam"] == "top" or d["v_src"] < near_split_v]
                st_kept_near = [d for d in dets if d["cam"] == "near"]
                near_dets = self._dets_from_result(near_results[j], None)
                # [2026-07-24 조작자 지시] 잘린 조각 기각 — 중복제거보다 **먼저**
                # 돌린다(조각이 스티치 표와의 매칭을 잡아먹지 않게). 상세 근거는
                # NEAR_CLIP_DROP 상수 주석.
                n_clip = 0
                if NEAR_CLIP_DROP and near_dets:
                    nh, nw = near_rgb.shape[:2]
                    keep = [d for d in near_dets
                            if clip_edge_count(d["box"], nw, nh)
                            < NEAR_CLIP_MIN_EDGES]
                    n_clip = len(near_dets) - len(keep)
                    near_dets = keep
                shot["near_clip"] = n_clip
                if st_kept_near:
                    # 같은 물체를 스티치 표로 남겼으면 near 원본 표는 버린다 —
                    # 안 그러면 한 샷에서 같은 셀에 2표가 들어간다.
                    # 매칭은 바닥접점의 near 픽셀 거리. 실측 대응쌍 15건의 거리는
                    # 7.0~40.5px 이고, 서로 다른 물체는 격자 50cm 간격이라
                    # 상반부에서 수백 px 떨어진다 → NEAR_DUP_PX 120 은 양쪽에
                    # 넉넉한 여유가 있다.
                    near_dets = [
                        d for d in near_dets
                        if not any(math.hypot(d["u"] - s["u_src"],
                                              d["v"] - s["v_src"]) <= NEAR_DUP_PX
                                   for s in st_kept_near)]
                for d in near_dets:
                    d["src_img"] = "near"   # 크롭은 near 원본에서 떠야 한다
                    # [2026-07-23] 오버레이는 스티치 위에 그리므로 near 좌표를
                    # 스티치 좌표로 옮겨 실어 둔다. 안 하면 박스가 엉뚱한 자리에
                    # 찍힌다(054834 실기 증상). A 는 similarity 라 사각형이
                    # 살짝 기울므로 4점 폴리곤으로 남긴다.
                    x1, y1, x2, y2 = d["box"]
                    d["draw_poly"] = _near_to_stitch(
                        st, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                    d["draw_uv"] = _near_to_stitch(st, [(d["u"], d["v"])])[0]
                n_top = len(dets) - len(st_kept_near)
                dets += near_dets
                # [폐기 2026-07-24 — 2원소 형식. 롤백 시 복원]
                #   shot["near_swap"] = [n_st - len(dets) + len(near_dets), len(near_dets)]
                # 3원소: [스티치 폐기, near 원본 채택, seam근처 스티치 유지]
                shot["near_swap"] = [n_st - n_top - len(st_kept_near),
                                     len(near_dets), len(st_kept_near)]
            dets_per_shot[i] = dets
            shot["detections"] = len(dets)
            for det in dets:
                cam = det["cam"]
                det["status"] = "reject"
                if intr[cam] is None:
                    det["why"] = "camera_info 없음"
                    continue
                if self.args.range_mode == "ground":
                    # RGB 바닥평면 역투영 (마스트업 캘리브) — depth 불필요
                    gx, gy = pixel_to_ground(det["u_src"], det["v_src"],
                                             intr[cam], mounts[cam])
                    xy_rf = (gx, gy)
                    det["range_m"] = round(math.hypot(gx, gy), 3)
                    d = robust_depth_median(depth_map[cam],
                                            det["u_src"], det["v_src"])
                    if d is not None:
                        det["depth_ref_m"] = round(d, 3)  # 교차검증 기록용
                else:
                    d = robust_depth_median(depth_map[cam],
                                            det["u_src"], det["v_src"])
                    if d is None or not (0.1 < d < 4.5):
                        det["why"] = f"depth 무효({d})"
                        continue
                    det["range_m"] = round(d, 3)
                    xy_rf = pixel_depth_to_robot_xy(
                        det["u_src"], det["v_src"], d, intr[cam], mounts[cam])
                mx, my = to_map(xy_rf[0], xy_rf[1], pose)
                cell, snap_err = fl.snap_cell(mx, my)
                det["cell"] = list(cell)
                det["snap_err_m"] = round(snap_err, 3)
                # (7/19 제거) far presence 강등 — 원거리도 정상 투표로 취급.
                if snap_err > MAX_SNAP_ERR_M:
                    det["why"] = "snap_err 초과"
                    continue
                # ---- 그리퍼(로봇 자체) 오검 기각 [2026-07-23 조작자 제안] ----
                # 15:45 실기에서 검은 그리퍼 손가락이 octahedron 으로 2회 검출됐고
                # (conf 0.369@0.325m → 셀 (200,250), conf 0.283@0.323m → (250,250)),
                # votes_k=1 이라 앞의 하나가 검증 없이 그대로 지도에 올랐다.
                # 판별자 2개를 AND 가 아니라 OR 로 건다 — 둘 다 실물 손실 0 이다.
                #
                #  (1) 우세색: 게임 물체는 전부 흰색, 그리퍼는 검정. 실측(같은 런)
                #      그리퍼 V(=max RGB) 중앙 46~64 / V<90 픽셀 75%,
                #      사과 스티커(가장 어두운 게임 텍스처) 중앙 194 / V<90 0.1%,
                #      큐브 bbox 0.0%, 바닥 0.0%. 스캔 near 프레임 전체에서
                #      V<90 인 픽셀은 **전부 그리퍼**였다. 문턱 0.4 는 양쪽 마진이
                #      크다(0.001 vs 0.75).
                #  (2) near 초근접 conf 하한: near 정상 표 19건 conf 중앙 0.973,
                #      최소 0.637. 유령 둘은 0.283/0.369.
                #
                # 거리 데드존은 쓰지 않는다 — 근접 물체 미검출 위험이 실제로 있다.
                # (그리퍼가 가리는 지면 띠는 마스트업 기준 전방 0.316~0.340m,
                #  폭 2.4cm 로 로봇 반경+그리퍼(0.275m) 바로 바깥이라 무의미하다.)
                if det["cam"] == "near":
                    if det["range_m"] < GRIPPER_NEAR_RANGE_M \
                            and det["conf"] < GRIPPER_NEAR_CONF_MIN:
                        det["why"] = (f"근접 저conf 기각 "
                                      f"({det['conf']:.2f}<{GRIPPER_NEAR_CONF_MIN})")
                        continue
                    # box 는 그 검출을 만든 이미지의 좌표계다 (크롭 코드와 동일 분기).
                    dark = _dark_fraction(
                        near_rgb if det.get("src_img") == "near" else bgr,
                        det["box"])
                    det["dark_frac"] = round(dark, 3)
                    if dark >= GRIPPER_DARK_FRAC:
                        det["why"] = f"검은 물체(그리퍼) 기각 dark={dark:.2f}"
                        continue
                # 근접 4셀 전담 프리샷 분리 (7/21): 프리샷은 전담 셀만 투표,
                # 스핀 스캔은 전담 셀 표를 기각 (초근접 오검/미검 영역).
                if only_cells is not None and cell not in only_cells:
                    det["why"] = "프리샷 전담 외 셀"
                    continue
                if exclude_cells is not None and cell in exclude_cells:
                    det["why"] = "프리샷 전담 셀 (초근접 스핀표 불신)"
                    continue
                det["status"] = "vote"
                det["identity"] = det["cls"]
                if det["cls"] == "cube_like_object":
                    x1, y1, x2, y2 = det["box"]
                    # [2026-07-23 하이브리드] 박스는 그 검출을 만든 이미지의
                    # 좌표계다. near 원본 검출을 스티치에서 크롭하면 엉뚱한
                    # 영역이 잘려 face 판정이 통째로 오염된다.
                    src = (np.ascontiguousarray(near_rgb[:, :, ::-1])
                           if det.get("src_img") == "near" else bgr)
                    sh, sw = src.shape[:2]
                    pad = int(max(x2 - x1, y2 - y1) * 0.18)
                    crop = src[max(0, y1 - pad):min(sh, y2 + pad),
                               max(0, x1 - pad):min(sw, x2 + pad)]
                    if crop.size:
                        crop_list.append(crop)
                        crop_det.append(det)
        timing["geometry"] = time.monotonic() - t0

        # --- face: 전 샷 크롭 1배치 (BGR, imgsz=224 고정)
        #     + pair(구성 A): 과일면 패치 전체를 AO/BP 각 1-forward ---
        t0 = time.monotonic()
        if crop_list:
            try:
                face_res = _chunked_predict(
                    self.models[1], crop_list, FACE_BATCH_CHUNK,
                    imgsz=fl.FACE_IMGSZ, conf=0.1, verbose=False)
                per_det_faces = []
                jobs = []  # (det_idx, face_idx, route, patch)
                for i, res in enumerate(face_res):
                    faces = [[res.names[int(b.cls)], float(b.conf), False]
                             for b in (res.boxes or [])]
                    per_det_faces.append(faces)
                    if self.pair is None:
                        continue
                    polys = res.masks.xy if res.masks is not None else []
                    for fk, f in enumerate(faces):
                        if (f[0] in fl.PairVerifier.ROUTE
                                and fl.PairVerifier.ROUTE[f[0]] in self.pair_routes
                                and f[1] >= fl.FACE_FRUIT_MIN
                                and fk < len(polys)
                                and polys[fk] is not None
                                and len(polys[fk]) >= 3):
                            jobs.append((i, fk, fl.PairVerifier.ROUTE[f[0]],
                                         fl.PairVerifier.make_patch(
                                             crop_list[i], polys[fk])))
                # [2026-07-23 신규] pair 검증 이력 기록. 종전엔 라벨을 조용히
                # 교체하거나 강투표를 박탈하고 끝이라, 과일 오분류가 났을 때
                # "pair 가 안 걸린 건지 / 걸렸는데 틀린 건지"를 사후에 가를
                # 방법이 아예 없었다. 15:45 실기의 실점(사과·바나나 각 2개 유실,
                # 약 40점)이 정확히 AO/BP 두 쌍의 혼동이라 이 관측성이 원인
                # 규명의 전제조건이다. 판정에는 전혀 개입하지 않는다(기록만).
                pair_log = defaultdict(list)   # det_idx -> [기록]
                if self.pair is not None and jobs:
                    for (i, fk, route, _), rv in zip(
                            jobs,
                            self.pair.verify([(r, p) for _, _, r, p in jobs])):
                        lab, pc = rv
                        before = per_det_faces[i][fk][0]
                        if pc >= fl.PAIR_CONF:
                            per_det_faces[i][fk][0] = lab  # 라벨 교체
                            act = "keep" if lab == before else "swap"
                        else:
                            per_det_faces[i][fk][2] = True  # 강투표 박탈
                            act = "veto"
                        pair_log[i].append({
                            "route": route, "face_conf": round(per_det_faces[i][fk][1], 3),
                            "before": before, "after": lab,
                            "pair_conf": round(float(pc), 3), "action": act})
                for di, (det, faces) in enumerate(zip(crop_det, per_det_faces)):
                    # 면별 원시 라벨/conf 전량 (카디널리티 제약 배정의 입력).
                    det["faces"] = [[f[0], round(f[1], 3), bool(f[2])] for f in faces]
                    if pair_log.get(di):
                        det["pair"] = pair_log[di]
                    ident, fc = fl.face_vote(
                        [(f[0], f[1]) for f in faces if not f[2]])
                    if ident is not None:
                        det["identity"] = ident
                        det["face_conf"] = round(float(fc), 3)
            except Exception as e:  # noqa: BLE001
                self.log(f"  face/pair 배치 추론 실패(face 미반영으로 계속): {e}")
        timing["face_pair"] = time.monotonic() - t0

        # --- 투표 반영 + 샷 기록 ---
        for i in idxs:
            shot = shots[i]
            for det in dets_per_shot[i]:
                if det.get("status") != "vote":
                    continue
                self.cell_votes[tuple(det["cell"])].append(
                    {"identity": det["identity"], "a1": det["cls"],
                     "a1_conf": det["conf"], "face_conf": det.get("face_conf"),
                     "cam": det["cam"], "range_m": det["range_m"],
                     "shot": shot["shot"]})
                shot["voted"] += 1
            # [2026-07-23] faces/pair 추가 — 면별 원시 라벨·conf 와 pair 검증
            # 이력. 과일 오분류 사후분석·카디널리티 제약 배정의 입력이다.
            shot["dets"] = [{kk: det.get(kk) for kk in
                             ("cls", "identity", "conf", "face_conf", "cam",
                              "range_m", "cell", "snap_err_m", "status", "why",
                              "faces", "pair")}
                            for det in dets_per_shot[i]]

        # --- PNG 저장: 백그라운드 (추론/미션 경로 비차단, 종료 시 join) ---
        save_args = [(shots[i]["shot"], *caps[i]["frames"], stitched_l[j],
                      dets_per_shot[i]) for j, i in enumerate(idxs)]

        prev_save = getattr(self, "_save_thread", None)

        def _bg_save():
            if prev_save is not None and prev_save.is_alive():
                prev_save.join(timeout=20.0)  # 프리샷 저장 스레드 체인
            for a in save_args:
                self._save_shot_images(*a)

        self._save_thread = threading.Thread(target=_bg_save, daemon=True)
        self._save_thread.start()

        self.report["scan"]["batch_timing"] = {
            k: round(v, 3) for k, v in timing.items()}
        self.log("  [배치추론] " + " | ".join(
            f"{k} {v:.2f}s" for k, v in timing.items())
            + f" (프레임 {len(idxs)}, 크롭 {len(crop_list)})")
        return shots

    def _save_shot_images(self, k, top_rgb, near_rgb, top_d, near_d, stitched, dets):
        """raw 쌍(top/near) + depth + 스티치 + 디버그 오버레이 저장 (raw 쌍 규칙)."""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            if k == 0:
                self.log("  PIL 없음 — 스캔 이미지 저장 생략")
            return
        if getattr(self.args, "no_save_images", False):
            return
        base = self.scan_dir / f"shot{k:02d}"

        # [2026-07-24 조작자 지시] **인코딩만 즉시, 디스크 쓰기는 경기 종료 후.**
        # 종전에는 배치추론 직후 백그라운드 스레드가 곧바로 디스크에 썼는데,
        # 그 I/O 가 스캔·주행과 CPU 를 다투었다(실기 추론이 벤치 대비 3~6배 느린
        # 원인 중 하나). 원본 배열을 그대로 들고 있으면 샷당 32MB(13샷 420MB)라
        # 8GB 통합메모리에 위험하므로, **압축된 bytes 로만** 보관한다
        # (샷당 ~3MB → 13샷 40MB 수준). q95/subsampling=0 은 그대로 유지 —
        # 오프라인 리플레이 재현성(투표 크롭 37건 중 35건 동일)이 이 품질에서
        # 검증된 값이라 낮추면 분석이 깨진다.
        def _enc(img_obj, name, **opts):
            buf = io.BytesIO()
            img_obj.save(buf, **opts)
            self._pending_writes.append((f"{base}_{name}", buf.getvalue()))

        try:
            _enc(Image.fromarray(top_rgb), "top.jpg", **JPEG_OPTS)
            _enc(Image.fromarray(near_rgb), "near.jpg", **JPEG_OPTS)
            _enc(Image.fromarray(stitched), "stitched.jpg", **JPEG_OPTS)
            try:
                # depth 는 uint16 이라 JPEG 불가 — PNG 유지하되 압축률만 낮춘다
                # (level6 123ms → level1 63ms, 0.09→0.16 MiB. 무손실은 그대로).
                _enc(Image.fromarray(top_d), "top_depth.png",
                     format="PNG", compress_level=1)
                _enc(Image.fromarray(near_d), "near_depth.png",
                     format="PNG", compress_level=1)
            except (TypeError, OSError):
                pass  # 구 Pillow uint16 미지원 — depth 저장만 생략
            img = Image.fromarray(stitched.copy())
            dr = ImageDraw.Draw(img)
            colors = {"vote": (40, 200, 60), "far_presence": (255, 160, 20),
                      "reject": (150, 150, 150)}
            for det in dets:
                col = colors.get(det.get("status"), (150, 150, 150))
                # [2026-07-23] near 원본 유래 검출은 box/u/v 가 near 프레임 좌표라
                # 스티치 위에 그대로 그리면 엉뚱한 자리에 찍힌다(054834 실기 증상).
                # 병합 시 실어 둔 스티치 좌표(draw_poly/draw_uv)를 쓴다.
                poly = det.get("draw_poly")
                if poly:
                    dr.polygon([tuple(q) for q in poly], outline=col)
                    tx, ty = poly[0]
                    cu, cv = det.get("draw_uv", (det["u"], det["v"]))
                else:
                    x1, y1, x2, y2 = det["box"]
                    dr.rectangle([x1, y1, x2, y2], outline=col, width=2)
                    tx, ty = x1, y1
                    cu, cv = det["u"], det["v"]
                lbl = f"{det.get('identity', det['cls'])} {det['conf']:.2f}"
                if det.get("range_m") is not None:
                    lbl += f" {det['range_m']:.2f}m"
                if det.get("cell"):
                    lbl += f" {tuple(det['cell'])}"
                if poly:
                    lbl += " [near]"
                dr.text((tx + 2, max(0, ty - 12)), lbl, fill=col)
                dr.ellipse([cu - 3, cv - 3, cu + 3, cv + 3],
                           outline=(255, 40, 40), width=2)  # 바닥접점
            _enc(img, "overlay.jpg", **JPEG_OPTS)
            # 관객 화면으로도 보낸다. 방금 _enc 가 만든 바이트를 그대로 쓰므로
            # **재인코딩이 없다.** 단발 스캔이면 경기당 1장(약 0.9 MiB).
            self.publish_scan_overlay(self._pending_writes[-1][1], k)
        except Exception as e:  # noqa: BLE001
            self.log(f"  샷 {k} 이미지 인코딩 실패: {e}")

    def flush_pending_writes(self):
        """보관 중인 인코딩 결과를 디스크에 쓴다 (경기 종료 후 호출)."""
        pend = getattr(self, "_pending_writes", None)
        if not pend:
            return 0
        self.scan_dir.mkdir(parents=True, exist_ok=True)
        n, mb = 0, 0
        for path, blob in pend:
            try:
                Path(path).write_bytes(blob)
                n += 1
                mb += len(blob)
            except OSError as e:  # noqa: PERF203
                self.log(f"  이미지 쓰기 실패 {path}: {e}")
        pend.clear()
        self.log(f"  스캔 이미지 {n}개 디스크 기록 ({mb/1e6:.0f}MB)")
        return n

    def _finalize_scan(self, partial: bool = False):
        """셀 확정: 표>=CELL_VOTES_MIN + 비대칭 과일투표(+conflicting_fruit 보류).

        프리샷 전담 4셀은 votes_k/fruit_k 를 1로 완화한다 — 종전엔 표가 프리샷
        1회뿐이라 필수였고, 2026-07-23 스핀 표 기각 제거 후에도 남긴다(실기
        기본값이 이미 --votes-k 1 --fruit-k 1 이라 실질 차이 없음).
        partial=True: [2026-07-22 오후] 동쪽 샷만의 중간 확정 — grid map 저장·
        GT 비교는 생략하고 셀 dict 만 갱신. 수거 스레드가 self.cells 를 읽는
        동안 서쪽 phase 가 재확정하므로 **새 dict 를 만들어 원자 교체**한다."""
        pre = self._preshot_cells if self._preshot_cap is not None else set()
        new_cells = dict(self.cells)
        for cell, vs in self.cell_votes.items():
            if len(vs) < (1 if cell in pre else self.votes_k):
                continue
            fruit_hits = Counter(
                v["identity"] for v in vs
                if v["identity"] in fl.FRUITS
                and (v["face_conf"] or 0) >= fl.CELL_FRUIT_CONF)
            identity, conflict, ao_tie = None, None, False
            if fruit_hits:
                top = fruit_hits.most_common(2)
                if top[0][1] >= (1 if cell in pre else self.fruit_k):
                    # [2026-07-24 조작자 지시] AO 동수 갈림 → 무조건 apple.
                    # conf 합 타이브레이크보다 **먼저** 건다. 11:56 실기의 사과
                    # 2셀 유실이 정확히 그 타이브레이크의 양극단이었다:
                    #   (150,100) apple .940 vs orange .936 → Δ.004 < 마진
                    #             → conflicting_fruit → A1 폴백 → 'plain' 확정
                    #   (250,350) orange .920 vs apple .585 → Δ.335 ≥ 마진
                    #             → 'orange' 확정 (conflict 플래그조차 없음)
                    # 둘 다 실제로는 apple 이었다. 즉 conf 우열은 어느 방향으로도
                    # 정답을 못 맞혔다 — 갈렸다는 사실 자체가 apple 신호다.
                    # 동수(top[0][1]==top[1][1]) + 최상위 동률 클래스가 정확히
                    # {apple,orange} 일 때만 발동한다 (2:1 등 비동수는 종전 경로).
                    tied = [k for k, v in fruit_hits.items() if v == top[0][1]]
                    if (AO_TIE_PREFER is not None and len(tied) == 2
                            and set(tied) == set(AO_TIE_PAIR)):
                        identity, ao_tie = AO_TIE_PREFER, True
                    elif len(top) > 1 and (top[0][1] - top[1][1]) < 2:
                        # 표차<2 동률: 표 수 대신 face_conf 합으로 재판정
                        # (7/21 171237 실기: 오독 pineapple 0.773 1표가 정독
                        # apple 0.924 1표와 동률 → conflicting_fruit 보류 →
                        # A1 폴백이 과일표를 plain으로 강등하는 사슬로 GT
                        # apple 셀이 plain 확정. conf 합 우위가 뚜렷하면 채택,
                        # 진짜 애매(우위 < 0.10)만 종전대로 보류.
                        conf_sum = defaultdict(float)
                        for v in vs:
                            if (v["identity"] in fl.FRUITS
                                    and (v["face_conf"] or 0) >= fl.CELL_FRUIT_CONF):
                                conf_sum[v["identity"]] += v["face_conf"]
                        ranked = sorted(conf_sum.items(), key=lambda kv: -kv[1])
                        if (len(ranked) > 1 and
                                ranked[0][1] - ranked[1][1] >= FRUIT_TIE_CONF_MARGIN):
                            identity = ranked[0][0]
                        else:
                            conflict = "conflicting_fruit"
                    else:
                        identity = top[0][0]
            if identity is None:  # 과일 미확정 → A1 conf 가중 집계
                tally = defaultdict(float)
                for v in vs:
                    lbl = v["identity"] if v["identity"] in fl.POLYHEDRA else (
                        "plain" if v["identity"] in fl.FRUITS | {"plain"}
                        else v["identity"])
                    tally[lbl] += v["a1_conf"]
                identity = max(tally.items(), key=lambda kv: kv[1])[0]
            new_cells[cell] = {"identity": identity, "votes": len(vs),
                               "fruit_hits": dict(fruit_hits),
                               "conflict": conflict,
                               # [2026-07-24] AO 동수 규칙 발동 표시 — 사후분석용.
                               # load_map_json 은 이 키를 복원하지 않는다(무해).
                               "ao_tie": ao_tie}
        self.cells = new_cells   # 원자 교체 — 수거 스레드와 경합 안전

        if partial:
            self.log(f"[SCAN 부분확정(동)] {len(self.cells)}셀 — 서쪽 추론은 "
                     "수거와 병렬 진행")
            return
        self.log(f"[SCAN 결과] 확정 {len(self.cells)}셀 / presence-only "
                 f"{len(self.presence)}셀 (>{fl.FAR_PRESENCE_M}m)")
        for cell, info in sorted(self.cells.items()):
            hold = f" [{info['conflict']}]" if info["conflict"] else ""
            if info.get("ao_tie"):   # [2026-07-24] AO 동수 → apple 확정 표시
                hold += f" [AO동수→{AO_TIE_PREFER}]"
            self.log(f"  {cell} = {info['identity']} (표 {info['votes']}, "
                     f"과일표 {info['fruit_hits']}){hold}")
        for cell, n in sorted(self.presence.items()):
            if cell not in self.cells:
                self.log(f"  {cell} = presence-only x{n}")

        # grid map 저장 (--map-file 재사용 포맷)
        grid = {"mast": "up", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cells": [{"cell": list(c), **info}
                          for c, info in sorted(self.cells.items())],
                "presence": [{"cell": list(c), "count": n}
                             for c, n in sorted(self.presence.items())]}
        (self.out / "grid_map.json").write_text(
            json.dumps(grid, ensure_ascii=False, indent=2))
        self.log(f"  grid map 저장: {self.out / 'grid_map.json'}")
        self._render_grid_png()

        if self.gt:
            cmp_res = fl.compare_with_gt(self.cells, self.gt)
            print(GT_UNRELIABLE_BANNER, flush=True)
            print(cmp_res["text"], flush=True)
            self.report["log"].append(cmp_res["text"])
            self.report["scan"]["gt_compare"] = {
                kk: cmp_res[kk] for kk in ("ok", "wrong", "ghosts", "missed")}
            self.report["scan"]["gt_compare_text"] = cmp_res["text"]
            # [2026-07-23 조작자 지시] 이 수치는 성능 지표가 아니다 — 아래 참조.
            self.report["scan"]["gt_unreliable"] = GT_UNRELIABLE_REASON

    def _render_grid_png(self):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return
        wpx = hpx = 440

        def px(x_cm, y_cm):
            return 20 + int(x_cm), hpx - 20 - int(y_cm)

        img = Image.new("RGB", (wpx, hpx), (250, 250, 250))
        dr = ImageDraw.Draw(img)
        dr.rectangle([px(0, 400), px(400, 0)], outline=(70, 70, 70), width=2)
        dr.rectangle([px(0, 40), px(40, 0)], outline=(200, 60, 60), width=2)
        dr.text(px(2, 55), "STORAGE", fill=(200, 60, 60))
        sx, sy = px(380, 20)
        dr.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], outline=(60, 60, 200), width=2)
        dr.text((sx - 18, sy + 8), "START", fill=(60, 60, 200))
        for gx in fl.GRID_XS_CM:
            for gy in fl.GRID_YS_CM:
                cx, cy = px(gx, gy)
                dr.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(190, 190, 190))
        for cell, n in self.presence.items():
            if cell in self.cells:
                continue
            cx, cy = px(*cell)
            dr.ellipse([cx - 7, cy - 7, cx + 7, cy + 7],
                       outline=(255, 160, 20), width=2)
            dr.text((cx + 8, cy - 6), f"?x{n}", fill=(255, 160, 20))
        for cell, info in self.cells.items():
            cx, cy = px(*cell)
            col = (40, 160, 60) if info["identity"] in fl.FRUITS else (60, 90, 200)
            if info["conflict"]:
                col = (200, 60, 200)
            dr.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=col)
            dr.text((cx + 8, cy - 6),
                    f"{info['identity'][:6]}:{info['votes']}", fill=col)
        img.save(self.out / "grid_map.png")
        self.log(f"  grid map PNG: {self.out / 'grid_map.png'}")

    # =====================================================================
    # COLLECT
    # =====================================================================
    def build_obstacles(self, exclude=None) -> list:
        obs = []
        for cell, info in self.cells.items():
            if cell == exclude or cell in self.collected:
                continue
            obs.append({"cell": cell, "xy": fl.official_cm_to_map(*cell),
                        "votes": info["votes"]})
        for cell, n in self.presence.items():
            if cell == exclude or cell in self.cells or cell in self.collected:
                continue
            if n >= PRESENCE_OBSTACLE_MIN:
                obs.append({"cell": cell, "xy": fl.official_cm_to_map(*cell),
                            "votes": n})
        return obs

    def pick_target_gated(self):
        """[2026-07-22 오후] 동서 분할 스캔 대응 — 우선순위 충돌 방지 게이트.

        동쪽 부분맵만 확정된 동안: ① 후보가 아예 없거나 ② order=fruit-first/
        shape-first 인데 우선 세트 quota 가 남았는데도 부분맵 후보가 비우선
        세트뿐이면, 서쪽 확정(_scan_full_done)을 기다렸다가 다시 고른다.
        나머지 경우(nearest 포함)는 부분맵으로 즉시 진행 — 동쪽 우선 수거는
        의도된 동작이다."""
        cell = self.pick_target()
        ev = getattr(self, "_scan_full_done", None)
        if ev is None or ev.is_set():
            return cell
        order = getattr(self.args, "order", "nearest")
        wait = cell is None
        # [2026-07-24] 정합성 인자 모드는 세트 순서 선호를 쓰지 않는다 (rate 가
        # 과일 20 / 형상 10 을 이미 반영). 아래 pref 대기는 건너뛴다 — probe 셀의
        # 스캔 라벨('plain')로 "비우선 세트"를 오판해 헛대기하는 것을 막는다.
        if (not wait and self.targets and not self.cf_on
                and order in ("fruit-first", "shape-first")):
            pref = (fl.FRUITS if order == "fruit-first"
                    else (fl.POLYHEDRA | {"plain"}))
            pref_left = any(self.placed_by_cls[k] < q
                            for k, q in self.targets.items() if k in pref)
            if pref_left and self.cells[cell]["identity"] not in pref:
                wait = True
        # [2026-07-24 조작자 지시] 하산 후보를 고른 것이면 기다리지 않는다.
        # 하산은 세트 순서를 무시하고 뽑히므로 여기서 "비우선 세트라 서쪽을
        # 더 보자"고 기다리면 그 사이 기회가 사라진다 (7/24 01:25 실기: 동쪽에
        # 형상 타깃뿐이라 서쪽 확정을 기다렸다가 최서단 사과로 갔다 — 첫 대상
        # 으로는 최악의 선택). 서쪽에서 더 나은 하산 후보가 나올 일도 없다:
        # 하산은 지금 서 있는 street(x=225) 한 줄에서만 성립한다.
        if (wait and cell is not None and self.nav_mode == "street"
                and self.descent_plan(cell, self.pose())[1]):
            self.log(f"  [수거] 하산 후보 {cell} — 서쪽 대기 생략")
            wait = False
        if wait:
            self.log("  [수거] 서쪽 맵 확정 대기 — 동쪽 부분맵과 "
                     "우선순위 충돌 방지")
            ev.wait(timeout=60.0)
            cell = self.pick_target()
        return cell

    def cf_probs_all(self):
        """타깃 클래스별 셀 확률 → ({cell: (intent, p)}, [진단문]).  [2026-07-24]

        과일 타깃은 `consistency_probs`(혼동쌍 부분계 열거), 형상 타깃은 혼동이
        없으므로 확정 셀에 p=1. quota 소진 클래스는 통째로 뺀다 (§2.3 하드캡 —
        초과 적재는 확정 −40, 헛 probe 도 ~10s 손실).
        """
        out, diag = {}, []
        for cls, quota in sorted(self.targets.items()):
            if self.placed_by_cls[cls] >= quota:
                continue
            if cls in fl.FRUITS:
                p_map, note = consistency_probs(
                    self.cells, cls, placed=self.placed_by_cls[cls],
                    collected=self.collected, evidence=self.cf_evidence)
                diag.append(note)
                for c, p in p_map.items():
                    if p >= CF_P_MIN and (c not in out or p > out[c][1]):
                        out[c] = (cls, p)
            else:
                # 형상(다면체 / set1 plain): 혼동쌍 개념 없음 → 스캔 확정 = p 1.
                # conflict 셀은 여기서 반드시 제외한다 — 갈린표 plain 은 실은
                # 과일 큐브라 plain 으로 집으면 오픽업이다 (사양 §6.5 분류 규칙).
                n = 0
                for c, info in self.cells.items():
                    if (info.get("identity") == cls and not info.get("conflict")
                            and c not in self.collected):
                        n += 1
                        if c not in out:
                            out[c] = (cls, 1.0)
                diag.append(f"{cls}: 확정 {n}셀 p=1 (혼동쌍 없음)")
        return out, diag

    def cf_pick(self, pose):
        """정합성 인자 최댓값 셀. → (cell | None, 랭킹 리스트)   [2026-07-24]

        `정합성 인자 = p·V / T_exp` (사양 §2). 매 사이클 **현재 pose 기준으로 전
        후보를 재계산**해 최댓값을 고른다 — 실패 후 위치가 다음 후보의 출발점이라
        복귀 비용은 따로 넣지 않는다.
        """
        probs, diag = self.cf_probs_all()
        for d in diag:
            self.log(f"  [정합성] {d}")
        rank = []
        for cell, (cls, p) in probs.items():
            if cell in self.collected or cell in self.skip_cells:
                continue
            xy = fl.official_cm_to_map(*cell)
            t_exp, t_full, t_probe = cf_time_exp(xy, pose, p)
            # 하산 파지는 하이웨이 왕복(~3.3m)이 통째로 사라진다 — 종전 pick_target
            # 의 "하산 무조건 우선"을 하드 오버라이드가 아니라 **시간 모델 안으로**
            # 흡수한다 (사양 §7-3 "형상 순서 선호는 정합성 인자에 흡수됨"과 동일 취지).
            desc = (self.nav_mode == "street"
                    and self.descent_plan(cell, pose)[1])
            if desc:
                t_exp = max(1.0, t_exp - CF_DESCEND_SAVE_S)
            rate = p * POINTS.get(cls, 0) / max(t_exp, 1e-6)
            probe = self.cells[cell].get("identity") != cls
            rank.append((rate, cell, cls, p, t_exp, desc, probe))
        if not rank:
            return None, []
        rank.sort(key=lambda r: -r[0])
        for rate, cell, cls, p, t_exp, desc, probe in rank[:5]:
            self.log(f"    {cell} {cls} p={p:.2f} T={t_exp:.1f}s "
                     f"→ {rate * 24.4:.1f}/T"
                     + (" [하산]" if desc else "") + (" [probe]" if probe else ""))
        return rank[0][1], rank

    def pick_target(self):
        pose = self.pose()
        if pose is None:
            return None
        # [2026-07-24 신규] 정합성 인자 모드 — 사양 §2/§3/§6.5. 기본 off 라
        # 아래 종전 경로(하산 우선 + 최근접)가 그대로 기본값이다.
        if self.cf_on and self.targets:
            cell, rank = self.cf_pick(pose)
            if cell is not None:
                cls = rank[0][2]
                self.cf_intent[cell] = cls
                if rank[0][6]:
                    self.log(f"  [정합성] probe 대상 {cell} — 스캔 라벨 "
                             f"'{self.cells[cell].get('identity')}' 이지만 {cls} "
                             f"후보(p={rank[0][3]:.2f}). 파지는 strict verify 통과 시에만.")
                return cell
            self.log("  [정합성] 후보 없음 — 종전 최근접 규칙으로 폴백")
        cand_raw = None
        if self.targets:
            # 실전 2세트 모드: 타깃 외 수거 절대 금지 (오픽업 2배 감점)
            # [2026-07-24 조작자 지시] 하산 후보 탐색용으로 **order 필터 이전**
            # 목록을 따로 뽑는다. fruit-first 는 `if pri: cand = pri` 교체라
            # 과일이 하나라도 남아 있으면 형상이 후보에서 통째로 사라지는데,
            # 하산 판정이 그 아래에 있어 x=250 형상 타깃은 판정을 받아볼 기회조차
            # 없었다 (7/24 01:25 실기 — 동쪽 맵의 x=200/250 팔면체 2개가 전부
            # 걷혀 서쪽 사과 (50,100) 로 갔다). 클래스·quota·conflict 필터는
            # 그대로 유지된다 — 우회하는 것은 **세트 순서 선호뿐**이다.
            cand_raw = match_candidates(self.cells, self.collected,
                                        self.skip_cells, self.targets,
                                        self.placed_by_cls, "nearest")
            cand = match_candidates(self.cells, self.collected, self.skip_cells,
                                    self.targets, self.placed_by_cls,
                                    getattr(self.args, "order", "nearest"))
            if not cand:
                return None
        else:
            cand = [c for c in self.cells
                    if c not in self.collected and c not in self.skip_cells]
            if not cand:
                return None
            if self.args.target_class:
                pri = [c for c in cand
                       if self.cells[c]["identity"] == self.args.target_class
                       and not self.cells[c]["conflict"]]
                if pri:
                    cand = pri
                else:
                    self.log(f"  대상 클래스 '{self.args.target_class}' 확정 셀 없음 "
                             "— 최근접 순으로 진행")

        def dist(c):
            x, y = fl.official_cm_to_map(*c)
            return math.hypot(x - pose[0], y - pose[1])

        # [2026-07-23 조작자 지시] 하산 파지 우선. 지금 서 있는 street 를 그대로
        # 타고 붙을 수 있는 후보가 있으면 먼저 고른다 — 하이웨이 왕복(스캔점
        # 기준 ~3.3m)이 통째로 사라진다. 스캔 직후 첫 사이클에서 공식 x=250 행
        # 타깃이 정확히 여기에 걸린다 (2026-07-24 x=250 한정으로 축소).
        #
        # [2026-07-24 조작자 지시 — 무조건 우선] 하산 후보는 세트 순서(fruit-first
        # /shape-first)를 **무시**하고 먼저 고른다. 정렬은 `descend_sort_key` —
        # **무조건 가장 북쪽(y 최대) 먼저** (상수 DESCEND_SKIP_ROWS 위 목적함수
        # 참조 — 스캔 레그는 마스트에 흡수돼 무관, 나중 적재 왕복이 북쪽일수록
        # 비싸다). 점수(과일 20 / 형상 10) 우선은 넣지 않는다(조작자 판단):
        # 순서가 점수에 영향을 주는 것은 "다 못 집을 때"뿐인데, 첫 하나만 형상
        # 으로 바꿔도 2사이클부터 fruit-first 가 다시 걸려 결국 버려지는 것은
        # 형상이라 4개 이상 집는 한 점수가 같다.
        # 하산 후보가 없으면(x=250 타깃 없음) 아래 min(cand, key=dist) 로 —
        # order 필터 적용된 목록에서 최근접 = 종전 규칙 그대로다.
        if self.nav_mode == "street":
            desc = [c for c in (cand_raw if cand_raw is not None else cand)
                    if self.descent_plan(c, pose)[1]]
            if desc:
                return min(desc, key=lambda c: descend_sort_key(c, pose))
        return min(cand, key=dist)

    def measure_target(self, cell, first: bool = False):
        """스티치(down) 프레임 A1 → 대상 셀 최근접 검출 바닥접점 →
        원본캠 depth 역투영 (접점 패치 중앙값 — 2026-07-21 ground→depth 전환).

        first: mini-goal 첫 측정 여부 — 실험 기능 'gate' ON 이면 첫 측정만
        TARGET_GATE_FIRST_M(0.25) 게이트를 쓴다 (홉 후 재측정은 항상 0.6).

        ground(바닥평면 역투영)는 '실루엣 최하단 = 접지점' 가정이 핵심인데,
        구형에 가까운 클래스는 최하단이 접지점이 아니라 공중의 접선점이라
        거리가 체계적으로 멀게 나온다 (7/20 실기: icosahedron 3/3 이 +5~10cm
        → 0.5m 초과 판정 → 방어 홉 → 홉 착지 ~0.27m 에선 접점이 프레임 밖이라
        재측정 전패 = 파지 0/4). depth 는 접점 픽셀의 실측 거리라 형상 무관
        (전 클래스 공히 전면까지의 거리 ≈ 중심-반폭, GRIP_FORWARD 상수로 흡수)
        이고 하단 잘림에도 보이는 픽셀로 측정이 유지된다.
        depth 무효 픽셀만 기존 ground 로 폴백하며(src 필드로 구분), ground
        값은 가능하면 y_ground_ref 로 병기해 편향 계측을 계속한다."""
        pose = self.pose()
        exp = None
        if pose is not None:
            cx, cy = fl.official_cm_to_map(*cell)
            dxw, dyw = cx - pose[0], cy - pose[1]
            c, s = math.cos(pose[2]), math.sin(pose[2])
            exp = (dxw * s - dyw * c, dxw * c + dyw * s)  # (x_right, y_forward)

        frames_ok = (self.fn is not None and self.models is not None
                     and self.fn.wait_fresh_frames(timeout=3.0))
        rgb = self.fn.rgb_pair() if frames_ok else None
        if rgb is None:
            if self.args.dry_run and exp is not None:
                return {"x_right": exp[0], "y_forward": exp[1],
                        "cls": "(dry-sim)", "conf": 1.0, "cam": "sim",
                        "src": "sim", "y_ground_ref": None}
            return None
        depth = self.fn.depth_pair() if frames_ok else None
        depth_map = {"top": depth[0], "near": depth[1]} if depth else None
        if depth_map is None:
            self.log("  [접근] depth 프레임 없음 — 전 검출 ground 폴백")

        st = self.make_stitcher("down")
        mounts = {"top": CameraMount(**fl.TOP_MOUNT_DOWN),
                  "near": CameraMount(**fl.NEAR_MOUNT_DOWN)}
        # [2026-07-23 01시 — 조작자 지시] 접근 측정은 near 원본 우선 추론.
        #
        # [폐기 2026-07-23 04시 — 원인 진단이 틀렸음. 롤백 시 참고용으로만]
        #   "down-스티치의 near 영역은 호모그래피 워프로 초근접 큐브가 길쭉하게
        #    왜곡돼 A1 이 놓친다"
        #   → 사실이 아니다. 워프는 similarity(배율 1.023, 회전 1.0°)라 종횡비
        #     왜곡이 0.95→0.95 로 없고, 같은 프레임에서 **워프된 near 영역만**
        #     @640 으로 돌리면 conf 0.99 로 정상 검출된다.
        #
        # 실제 원인은 **letterbox 배율**이다. 전체 스티치(1912x2137)에 imgsz=896
        # 을 주면 배율 0.419 라 초근접 큐브가 네트 입력에서 250px 까지 커지는데,
        # A1 은 imgsz=640 / mosaic=1.0 / scale=0.5 로 학습돼 이 크기가 분포 밖이다.
        # 분포 밖에서 conf 는 입력 격자에 대해 사실상 카오스가 된다 — 픽셀이 bit
        # 단위로 동일한데 캔버스만 바꿔도 0.80→0.07, 무손실 5px 이동에 0.80→0.11.
        # (워프 0회 대조군까지 포함해 실측 확증. docs/07-results-and-lessons.md)
        #
        # 그래서 무검출이면 **재촬영이 아니라 imgsz 를 바꿔** 같은 프레임을 다시
        # 본다 (A1_IMGSZ_NEAR_LADDER). 원거리(방어 홉 재측정 등)는 near 시야 밖일
        # 수 있어, 사다리를 전부 소진한 뒤에만 종전 스티치 경로로 폴백한다.
        bgr = np.ascontiguousarray(rgb[1][:, :, ::-1])
        dets = []
        for imgsz in A1_IMGSZ_NEAR_LADDER:
            r_near = self.models[0].predict(bgr, imgsz=imgsz,
                                            conf=A1_APPROACH_CONF, verbose=False)[0]
            dets = self._dets_from_result(r_near, None)
            if dets:
                if imgsz != A1_IMGSZ_NEAR_LADDER[0]:
                    self.log(f"  [접근] near @{A1_IMGSZ_NEAR_LADDER[0]} 무검출 "
                             f"→ @{imgsz} 재추론에서 {len(dets)}건 회수")
                break
        if not dets:
            stitched = st.stitch(rgb[0], rgb[1])
            dets, bgr = self._detect_stitched(stitched, st, A1_APPROACH_CONF)
        intr = {c: self.intr(c) for c in ("top", "near")}
        best = None
        for det in dets:
            cam = det["cam"]
            if intr[cam] is None:
                continue
            gy = None                     # ground 참조값 (편향 계측용 병기)
            try:
                gx, gy = pixel_to_ground(det["u_src"], det["v_src"],
                                         intr[cam], mounts[cam])
            except ValueError:
                gx = None
            src = None
            if depth_map is not None and depth_map[cam] is not None:
                # 패치 중심을 접점보다 3px(640기준, 해상도 비례) 위 — 라운드
                # 클래스는 실루엣 최하단 아래가 물체 뒤 바닥이라 정중앙 패치는
                # 원거리 픽셀이 섞인다. 전면이 수직면이라 위로 올려도 y_fwd 동일.
                rs = depth_map[cam].shape[1] / 640.0
                d = robust_depth_median(depth_map[cam],
                                        det["u_src"], det["v_src"] - 3 * rs)
                if d is None:             # 접점 홀 — 패치 확장 1회 재시도
                    d = robust_depth_median(depth_map[cam], det["u_src"],
                                            det["v_src"] - 3 * rs,
                                            radius=int(round(6 * rs)))
                if d is not None and 0.1 < d < 4.5:   # 스캔 경로와 동일 게이트
                    x, y = pixel_depth_to_robot_xy(
                        det["u_src"], det["v_src"], d, intr[cam], mounts[cam])
                    src = "depth"
            if src is None:
                # ground 폴백 — 접점 가정이 필요하므로 하단 잘림은 측정 불가
                if cam == "near" and det["v_src"] >= rgb[1].shape[0] - max(4, rgb[1].shape[0] // 120):
                    continue  # (60번 계약)
                if gy is None:
                    continue
                x, y, src = gx, gy, "ground"
            d_exp = math.hypot(x - exp[0], y - exp[1]) if exp else abs(y)
            # [2026-07-23] 게이트는 항상 0.6m (구 실험 기능 'gate' 제거).
            if exp is not None and d_exp > TARGET_GATE_M:
                continue  # 오타겟 방지 게이트 (다른 물체로 접근 방지)
            # 격자 스냅 검증 [2026-07-23 00시 상시 승격 — 조작자 지시]:
            # 검출을 pose 로 역투영해 최근접 격자점 스냅 — 대상 셀과 다르면
            # 배제. 00:17/00:22 실기: (250,200) 접근이 옆 셀 (200,200) 물체에
            # 락온(d_exp 0.42~0.51 < 0.6 통과) → plain 파지·오픽업 적재.
            # 물체는 격자점에만 있으므로 셀 불일치 검출은 항상 오타겟이다.
            if pose is not None:
                wx = pose[0] + y * c + x * s   # body(우 x, 전방 y) → map
                wy = pose[1] + y * s - x * c
                # 반아레나 오프셋을 fl 경유로 — 여기 리터럴이 아레나 크기와
                # 어긋나면 역투영 셀이 통째로 밀려 **모든 접근이 배제**된다.
                wcx, wcy = fl.map_to_official_cm(wx, wy)
                snap = (int(round(wcx / fl.GRID_PITCH_CM)) * fl.GRID_PITCH_CM,
                        int(round(wcy / fl.GRID_PITCH_CM)) * fl.GRID_PITCH_CM)
                if snap != (int(cell[0]), int(cell[1])):
                    self.log(f"  [접근] 격자 스냅 배제 — [{det['cls']}] "
                             f"역투영 셀 {snap} ≠ 대상 {tuple(cell)}")
                    continue
            if best is None or d_exp < best["d_exp"]:
                best = {"x_right": x, "y_forward": y, "cls": det["cls"],
                        "conf": det["conf"], "cam": cam, "src": src,
                        "y_ground_ref": round(gy, 4) if gy is not None else None,
                        "d_exp": d_exp, "box": det["box"]}
        if (best is not None and (self.args.target_class or self.targets)
                and best["cls"] == "cube_like_object"):
            # 파지 전 face 재검증용 크롭 (스캔과 동일 pad 0.18)
            x1, y1, x2, y2 = best["box"]
            pad = int(max(x2 - x1, y2 - y1) * 0.18)
            hh, ww = bgr.shape[:2]
            crop = bgr[max(0, y1 - pad):min(hh, y2 + pad),
                       max(0, x1 - pad):min(ww, x2 + pad)]
            if crop.size:
                best["crop"] = crop
        return best

    def _verify_face_ident(self, crop) -> tuple:
        """파지 직전 face 재추론 (+pair 재검증) → (성공, ident, conf, 사유).

        [2026-07-23 조작자 지시] **VERIFY_TIMEOUT_S 초과 시 포기**하고 파지로
        진행한다 — 이 재검증은 '확실히 아닐 때만 멈추는' 보조 게이트라,
        느려질 바에는 없는 편이 낫다 (마스트다운 근접뷰라 판별불가가 다수).
        시간초과 스레드는 그대로 두고 결과만 버린다. 다음 사이클의 재검증이
        아직 살아 있는 스레드와 겹치면 모델 동시 사용이 되므로 그때는
        추론을 아예 건너뛴다 (같은 '판별불가' 처리).
        """
        prev = getattr(self, "_verify_th", None)
        if prev is not None and prev.is_alive():
            return False, None, 0.0, "이전 face 재검증 미완 — 파지 진행"
        box = {}

        def _run():
            try:
                res = self.models[1].predict(crop, imgsz=fl.FACE_IMGSZ,
                                             conf=0.1, verbose=False)[0]
                faces = [[res.names[int(b.cls)], float(b.conf), False]
                         for b in (res.boxes or [])]
                if self.pair is not None and faces:   # 스캔과 동일 pair 재검증
                    polys = res.masks.xy if res.masks is not None else []
                    jobs = []
                    for fk, f in enumerate(faces):
                        if (f[0] in fl.PairVerifier.ROUTE
                                and fl.PairVerifier.ROUTE[f[0]] in self.pair_routes
                                and f[1] >= fl.FACE_FRUIT_MIN
                                and fk < len(polys) and polys[fk] is not None
                                and len(polys[fk]) >= 3):
                            jobs.append((fk, fl.PairVerifier.ROUTE[f[0]],
                                         fl.PairVerifier.make_patch(
                                             crop, polys[fk])))
                    if jobs:
                        for (fk, _, _), (lab, pc) in zip(
                                jobs,
                                self.pair.verify([(r, p) for _, r, p in jobs])):
                            if pc >= fl.PAIR_CONF:
                                faces[fk][0] = lab      # 라벨 교체
                            else:
                                faces[fk][2] = True     # 강투표 박탈
                box["v"] = fl.face_vote(
                    [(f[0], f[1]) for f in faces if not f[2]])
            except Exception as e:  # noqa: BLE001
                box["e"] = e

        th = threading.Thread(target=_run, daemon=True)
        self._verify_th = th
        t0 = time.monotonic()
        th.start()
        th.join(VERIFY_TIMEOUT_S)
        if th.is_alive():
            return (False, None, 0.0,
                    f"face 재검증 {VERIFY_TIMEOUT_S:.1f}s 초과 — 파지 진행")
        if "e" in box:
            return False, None, 0.0, f"face 재추론 실패({box['e']}) — 파지 진행"
        ident, fc = box["v"]
        self.log(f"  [검증] face 재추론 {time.monotonic() - t0:.2f}s "
                 f"→ {ident or '무검출'}({fc:.2f})")
        return True, ident, fc, ""

    def verify_target(self, m, tgt=None, strict: bool = False) -> tuple:
        """파지 직전 대상 클래스 재검증. (통과여부, 사유) 반환.

        [2026-07-24 신규] strict — **probe 셀 전용**(스캔 identity ≠ 기대 클래스:
        conflict 셀 / 혼동쌍 파트너 라벨 셀). 아래 관용 규칙("판별불가는 통과")은
        **스캔 identity 를 신뢰할 수 있을 때만** 성립한다. probe 셀은 그 전제가
        없으므로 **적극 확인만 통과**시킨다 — 무검출·plain·시간초과·예외는 전부
        중단이다. 관용 통과를 그대로 두면 갈린표 plain 을 그냥 집어 오픽업(−40)이
        된다. 사양 §6.5 "파지 게이트 — 근접 검증 필수".
        관측 클래스는 `self._cf_last_ident` 로 남겨 소진(exhaustion) 갱신에 쓴다.

        tgt(기대 클래스) 미지정 시 --target-class (종전 동작). 실전 2세트
        모드에서는 collect_one이 셀의 스캔 identity(=타깃 중 하나)를 넘긴다.
        접근 최종 측정의 검출을 재인식해
        대상과 *확정 모순*일 때만 False(셀 skip) — 마스트다운 근접 시야의
        plain/무검출은 '아님'이 아니라 '판별불가'이기 때문이다. 이때 skip하면
        진짜 대상까지 버리므로 스캔 identity를 신뢰하고 통과시킨다.

        ⚠ [2026-07-23 정정 — 종전 주석의 근거가 틀렸었다. 폐기]
          "배치 규칙상 큐브 측면은 항상 plain이고 과일면은 윗면뿐이라"
        실제 본부 규칙은 **윗면 1 + 서로 마주보는 옆면 2 = 과일**,
        **바닥 1 + 나머지 마주보는 옆면 2 = plain** 이다
        (기준본 docs/01-competition-and-rules.md §3 "The objects"). 즉 **옆 4방위 중 2방위는 과일면**
        이고, 인접한 두 옆면은 항상 과일 1 + plain 1이라 **모서리(45°) 뷰는
        과일면 1개를 보장**한다. 근접 재확인이 실제로 잘 작동하는 이유다.
        결론('plain 관측 = 판별불가 → 통과')은 그대로 유효하다: 과일큐브라도
        옆 4방위 중 2방위는 정상적으로 plain이라 1회 plain 관측의 우도비는
        plain큐브 대비 1:2 에 불과해 단독 반증이 못 되기 때문이다.

        [2026-07-23 조작자 지시] 과일 대상의 '모순' 정의를 **혼동쌍 파트너
        검출**로 좁혔다. 쌍 밖 과일이 뜬 것은 무시하고 파지한다 (아래 주석).
        형상(다면체·plain)은 쌍 개념이 없어 종전 엄격 판정 그대로다.
        재추론은 VERIFY_TIMEOUT_S(0.5s) 초과 시 포기하고 파지로 진행."""
        self._cf_last_ident = None      # [2026-07-24] 소진 갱신용 관측 기록 초기화
        tgt = tgt or self.args.target_class
        if not tgt or m.get("src") == "sim":
            return True, None
        cls = m["cls"]
        if tgt in fl.POLYHEDRA:
            if cls != tgt:
                return False, f"A1 재인식 {cls} ≠ 대상 {tgt} — 모순"
            return True, f"A1 {cls} 일치"
        if cls != "cube_like_object":
            return False, f"A1 재인식 {cls} — 큐브 아님 (대상 {tgt} 모순)"
        crop = m.get("crop")
        if crop is None:
            if strict:
                return False, "face 크롭 없음 — probe 셀은 적극 확인 필수 (중단)"
            return True, "face 크롭 없음 — 스캔 identity 신뢰"
        ok_v, ident, fc, why = self._verify_face_ident(crop)
        if not ok_v:
            if strict:
                return False, f"{why} → probe 셀은 판별불가 = 중단"
            return True, why      # 시간초과/예외 = 판별불가 → 파지 진행
        # [2026-07-24] 소진 갱신에 쓸 **결정적** 관측만 남긴다 — 면투표 게이트를
        # 넘긴 과일면만. 무검출·plain·약한 과일면은 '아님'이 아니라 '판별불가'라
        # hard evidence 가 될 수 없다 (사양 §6.5 — p=1 의 유일한 근거는 소진).
        if ident in fl.FRUITS and fc >= fl.FACE_FRUIT_MIN:
            self._cf_last_ident = ident
        # [2026-07-23 조작자 지시] 중단은 **혼동쌍 파트너**가 떴을 때만.
        #   "바나나인 줄 알고 갔는데 파인애플이 떠버린 경우 빼고는 웬만하면 집는다"
        # 근거: 실제 오픽업 위험은 모델이 상시 헷갈리는 쌍(apple↔orange,
        # banana↔pineapple) 안에서만 생긴다 — 스캔이 banana 로 확정한 자리에
        # 접근뷰가 pineapple 을 내면 둘 중 하나가 틀린 것이고, 이때만 2배 감점
        # 리스크가 실재한다. 반대로 쌍 밖 라벨(대상 banana 에 apple/orange)은
        # 모델이 사실상 내지 않는 조합이라 노이즈·타 물체 혼입으로 보고 무시한다.
        # 종전(=대상과 다른 과일이면 전부 skip)은 이 노이즈까지 셀을 버렸다.
        if ident in fl.FRUITS and fc >= fl.FACE_FRUIT_MIN and ident != tgt:
            pair_of = fl.PairVerifier.ROUTE      # {과일: 'AO'|'BP'}
            if tgt == "plain":
                # 세트1 정육면체(과일면 없음)에 과일면이 뜬 것은 쌍 혼동이
                # 아니라 물체 자체가 다르다는 뜻 — 종전대로 skip.
                return False, (f"face 재인식 {ident}({fc:.2f}) — plain 큐브에 "
                               "과일면 = 다른 물체")
            if pair_of.get(tgt) is not None and pair_of.get(ident) == pair_of.get(tgt):
                return False, (f"face 재인식 {ident}({fc:.2f}) = 대상 {tgt} 의 "
                               "혼동쌍 파트너 — 오픽업 방지 skip")
            if strict:
                return False, (f"face 재인식 {ident}({fc:.2f}) ≠ 대상 {tgt} — "
                               "probe 셀은 다른 쌍이어도 중단 (적극 확인 실패)")
            return True, (f"face 재인식 {ident}({fc:.2f}) — 대상 {tgt} 와 다른 쌍 "
                          "(모델이 내지 않는 조합) → 무시하고 파지")
        if ident == tgt and fc >= fl.FACE_FRUIT_MIN:
            return True, f"face {ident}({fc:.2f}) 일치"
        # 무검출/plain/약한 과일면(면투표 게이트 미만) = 판별불가 — 통과
        # [2026-07-24] 단 probe 셀(strict)은 여기서 **중단**한다. 스캔 라벨이
        # 대상이 아니었으므로 "모순 없음"은 파지 근거가 못 된다.
        if strict:
            return False, (f"face {ident or '무검출'}({fc:.2f}) — probe 셀 적극 "
                           f"확인 실패 ({tgt} 미확인) → 중단")
        return True, f"face {ident or '무검출'} — {tgt} 모순 없음 (윗면 미가시 가능)"

    def _save_measure_frames(self, cell, label: str):
        """접근 측정 시점의 near/top RGB 를 런 디렉터리에 저장 (백그라운드).

        [2026-07-22 조작자 지시] 02:07 실기 '검출 실패'의 원인(셀 identity
        오기록 vs 마스트/캘리브)을 사후 판별할 증거가 없었다 — 매 사이클
        첫 시도와 최종 실패 시 1장씩 남긴다. JPEG+스레드라 경기 시간 영향 없음."""
        if self.args.dry_run or self.fn is None:
            return
        try:
            # [2026-07-23 00시] PIL 지역 import — 종전엔 모듈 스코프에 Image 가
            # 없어 저장 스레드가 매번 NameError 로 조용히 죽었다 (07-22~23 실기
            # 전 런에서 증거사진 0장). 실패도 이제 로그로 남긴다.
            from PIL import Image
            rgb = self.fn.rgb_pair()
            if rgb is None:
                self.log(f"  [증거사진] {label}: 프레임 없음 — 저장 생략")
                return
            top, near = rgb[0].copy(), rgb[1].copy()
            base = self.out / f"approach_{cell[0]}_{cell[1]}_{label}"

            def _w():
                try:
                    Image.fromarray(top).save(f"{base}_top.jpg", quality=90)
                    Image.fromarray(near).save(f"{base}_near.jpg", quality=90)
                    self.log(f"  [증거사진] 저장 — {base.name}_top/near.jpg")
                except Exception as e:  # noqa: BLE001
                    self.log(f"  [증거사진] 저장 실패(무시): {e}")
            threading.Thread(target=_w, daemon=True).start()
        except Exception as e:  # noqa: BLE001
            self.log(f"  [증거사진] 캡처 실패(무시): {e}")

    def measure_with_retry(self, cell, first: bool = False):
        """measure_target 재시도 래퍼 — (측정, 성공회차). 전패 시 (None, 상한).

        측정 실패의 태반은 프레임 기인 일시 장애(회전·이동 직후 블러, stale
        프레임)나 pose 미수렴이라 새 프레임으로 다시 재면 살아난다. 최초
        측정과 홉 후 재측정이 같은 경로를 쓴다 (7/21 통일 — 종전엔 홉 후
        1회뿐이라 이동 직후의 가장 취약한 시점에 보험이 없었다)."""
        for attempt in range(1, MEASURE_TRIES + 1):
            if attempt == 1:
                self._save_measure_frames(cell, "try1")   # 증거 사진 (2026-07-22)
            m = self.measure_target(cell, first=first)
            if m is not None:
                return m, attempt
            self.spin(0.35)
        self._save_measure_frames(cell, "fail")           # 최종 실패 증거
        return None, MEASURE_TRIES

    def wait_held(self, timeout=GRASP_WAIT_S) -> bool:
        if self.args.dry_run:
            self.log("  [dry-run] grasp_state=held 가정")
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.fn.spin_for(0.1)
            st = self.fn.grasp_state()
            if st == "held":
                return True
            if st == "empty":
                return False
        return self.fn.grasp_state() == "held"

    def collect_one(self, cell, idx: int) -> dict:
        info = self.cells.get(cell, {})
        rec = {"index": idx, "target": list(cell),
               "identity": info.get("identity"), "phases": {},
               "grasped": False, "placed": False, "ok": False}
        # 관객 피드: 이 사이클이 끝날 때까지 "지금 이 셀"로 표시된다.
        self._active_cell = tuple(cell)
        self._cycle_live = rec
        self.publish_match_state(force=True)
        cxy = fl.official_cm_to_map(*cell)
        self.log(f"\n[수거 {idx}] 셀 {cell} = {info.get('identity')} "
                 f"(맵 {cxy[0]:+.2f},{cxy[1]:+.2f})")

        # --- a. ROUTE ---
        # [2026-07-22] 기본 street 주행 (navigation/docs/control-and-routing.md §5):
        # 하이웨이 x 정렬 → street 북진 → mini-goal. 롤백: --nav legacy.
        t = time.monotonic()
        pose = self.pose()
        if pose is None:
            rec["fail"] = "pose 없음"
            return rec
        street = self.nav_mode == "street"
        gx = gy = None
        # [2026-07-23] 하산 파지 — 같은 street 면 하이웨이 왕복 생략.
        # [2026-07-24] 종전엔 이 줄이 `not street or south_row` 였다. 남단 행을
        # **운반 예외이자 경로 예외**로 겸용해서, 판정이 세 곳(여기 / stage_scan
        # 남하 게이트 / pick_target)에서 갈렸다 — 7/23 실기에서 "남하 생략해
        # 놓고 하산도 안 함"(19:11 route 5.75s, 14:02 7.68s)의 원인이다.
        # 이제 하산 여부는 descend_candidate **한 곳**만 결정한다 (제외 행은
        # DESCEND_SKIP_ROWS). 여기서 다시 막지 않는다.
        mirror, descend = ((False, False) if not street
                           else self.descent_plan(cell, pose))
        # [2026-07-23 조작자 지시 — 남단 행 예외 재정의, 상시 적용]
        # y=100 행은 **mini-goal 을 표준값(공식 y=75 = map -1.25)로 그대로 두고**,
        # 파지 후 운반만 예외로 한다: 북향 복원 + 남하(15cm)를 생략하고 mini-goal
        # 에서 곧장 적재 드리프트로 넘어간다 (조작자: "75부터 하이웨이").
        # 폐기된 구 'south' 실험 기능은 mini-goal 자체를 y=60(하이웨이)으로
        # 내렸는데, 그건 접근 거리를 0.35→0.47m 로 늘리는 별개 변경이었다.
        # 기하: mini-goal 은 격자 열 사이(x+25)라 최근접 물체는 대각 35.36cm.
        # 드리프트의 135°→-135° 회전 스윕 26cm + 물체 반폭 4cm = 30cm → 여유
        # 5.4cm 로, 지금도 같은 지점에서 도는 street_face(북향 복원)와 동일 조건.
        # [2026-07-24] **미러일 때는 이 예외를 쓰지 않는다.** 미러 정면은 +45°
        # 라 여기서 곧장 드리프트를 걸면 적재 정면(-135°)까지 180° — 검증된 건
        # 135°(표준)와 90°(남단 기본)뿐이고 180° 는 wrap_angle 부호 경계다.
        # 지금은 DESCEND_SKIP_ROWS 가 y=100 을 하산에서 빼 이 조합이 나오지
        # 않지만, 나중에 y=100 을 열더라도 운반이 자동으로 표준 경로가 되도록
        # 여기에 방어를 둔다 (판정이 다시 갈리지 않게).
        south_row = street and cell[1] == 100 and not mirror
        if street:
            # mini-goal 은 남단 행도 **표준값**(y=100 → map -1.25 = 공식 75).
            # 종전 구 'south' 실험 기능은 여기서 gy 를 하이웨이(-1.40=공식 60)로
            # 내렸는데, 접근 거리를 0.354→0.47m 로 늘리는 별개 변경이라 폐기했다
            # (2026-07-23 조작자 지시 — 예외는 운반 구간에만 둔다).
            gx, gy = snv.mini_goal_for(cell, mirror=mirror)
            rec["route"] = {"mode": "street_nav",
                            "mini_goal": [round(gx, 3), round(gy, 3)],
                            "south_row": south_row,
                            "mirror": mirror, "descend": descend}
            self.log(f"  [경로] street — mini-goal ({gx:+.2f},{gy:+.2f}) "
                     f"= 물체 {'남서' if mirror else '남동'} "
                     f"{math.hypot(gx - cxy[0], gy - cxy[1]):.2f}m"
                     + (" (남단 행: 운반 즉시 드리프트)" if south_row else "")
                     + (" [하산 파지]" if descend else ""))
            # 필드 안(스캔점 등)에서 시작하면 먼저 현재 street 로 남하 —
            # 하이웨이 밖에서의 대각 이동은 물체 행을 가로지른다.
            if not descend:
                self.street_home()
            if self.args.dry_run:
                self.sim_pose[:] = [gx, gy, math.pi / 2.0]
                self.log(f"  [dry-run] street mini-goal 도달")
                r_route = {"ok": True, "reason": "dry-run"}
            else:
                r_route = self.snav.go_to_mini_goal(
                    cell, mirror=mirror, descend=descend)
            rec["route"]["sec"] = round(r_route.get("sec", 0.0), 2)
            # [2026-07-22] 폐루프 통계 — 실기 이탈/타임아웃 분석용 (01:02 실기
            # street-north 25s 타임아웃의 원인을 로그만으론 못 밝혔다).
            if r_route.get("max_cross_m") is not None:
                rec["route"]["max_cross_m"] = r_route["max_cross_m"]
                rec["route"]["recoveries"] = r_route.get("recoveries", 0)
            # [2026-07-23 — 조작자 승인] 종전 `not ok and trace` 는 **실패한
            # 경로만** 궤적을 남겼다. 01:10~01:20 실기의 사이클3(011726)은
            # recoveries=1 인데 ok=True 라 `trace: null` 로 저장돼, 10.5cm
            # 이탈의 틱 궤적이 사라졌다. 복구가 있었으면 성공이어도 남긴다.
            # (trace 는 이제 레그별로 r_route["legs"] 안에 들어 있다 —
            #  street_nav.merge_legs 참조. 하이웨이 레그 것도 이때부터 남는다.)
            if r_route.get("legs") and (not r_route.get("ok")
                                        or r_route.get("recoveries")):
                rec["route"]["legs"] = r_route["legs"]
            if not r_route.get("ok"):
                rec["fail"] = (f"route: street {r_route.get('phase')} 실패 "
                               f"({r_route.get('reason')})")
                rec["phases"]["route"] = round(time.monotonic() - t, 2)
                self.log(f"  [경로] street 실패 — {r_route.get('reason')} — 셀 skip")
                self.skip_cells.add(cell)
                self.street_home("실패복귀_남하")
                return rec
        else:
            # [legacy] 셀 앞 standoff 까지 코리도 라우팅 (2026-07-22 이전 기본)
            dvx, dvy = pose[0] - cxy[0], pose[1] - cxy[1]
            dl = math.hypot(dvx, dvy) or 1.0
            standoff = (cxy[0] + dvx / dl * STANDOFF_M,
                        cxy[1] + dvy / dl * STANDOFF_M)
            route = plan_route((pose[0], pose[1]), standoff,
                               self.build_obstacles(exclude=cell))
            self.log(f"  [경로] {route['mode']} — {route['reason']}")
            self.log("  [경로] 경유점: " +
                     " → ".join(f"({w[0]:+.2f},{w[1]:+.2f})"
                                for w in route["waypoints"]))
            rec["route"] = {"mode": route["mode"], "reason": route["reason"],
                            "waypoints": [list(w) for w in route["waypoints"]]}
            self.follow_route(route, self.v_cruise)
        rec["phases"]["route"] = round(time.monotonic() - t, 2)

        # --- b. FACE: 대상 정면 회전 ---
        t = time.monotonic()
        if street:
            # mini-goal 에서 CCW 45° — 물체는 북서 대각 0.354m (I-2)
            # (미러 mini-goal 이면 CW 45° · 북동 대각 — 2026-07-23 하산 파지)
            self.log(f"  [정면] {'CW' if mirror else 'CCW'} 45° → 물체 정면 "
                     f"({'북동' if mirror else '북서'} 대각)")
            self.street_face(True, mirror=mirror)
        else:
            pose = self.pose()
            yaw_t = math.atan2(cxy[1] - pose[1], cxy[0] - pose[0])
            self.log(f"  [정면] 회전 → {math.degrees(yaw_t):+.1f}°")
            self.do_rotate(yaw_t)
        rec["phases"]["face"] = round(time.monotonic() - t, 2)

        # street 실패 복귀: 접근 이동 되감기(요 135° 유지) → 북향 → 하이웨이.
        # 회전은 mini-goal 에서만 한다 (I-2) — 되감기가 선행되는 이유.
        adv = [0.0, 0.0]   # 접근 중 몸체 (전방, 좌) 누적 이동

        def street_bail(tag: str = "실패복귀"):
            if not street:
                return
            if self.args.dry_run:
                self.sim_pose[:] = [gx, snv.HIGHWAY_Y_M, math.pi / 2.0]
                return
            if abs(adv[0]) > 0.01 or abs(adv[1]) > 0.01:
                self.do_move(-adv[0], -adv[1], max_v=self.v_approach,
                             tag=f'{tag}_되감기')
            # [2026-07-23] 남단 행도 **실패 복귀는 표준 경로**를 쓴다. mini-goal
            # 이 공식 75 로 돌아왔으므로 남하는 15cm 뿐이고, 물체를 못 문 상태의
            # 요 135° 잔류를 다음 사이클 드리프트에 떠넘기는 것보다 안전하다
            # (구 'south' 는 mini-goal 이 이미 하이웨이라 생략이 성립했다).
            self.street_face(False)
            self.street_south(gx, f"{tag}_남하")

        # --- c. APPROACH: 60번 --fast 계약 (측정1회→단발 정렬+전진) ---
        t = time.monotonic()
        self.stage_mast_wait()  # 병렬 마스트다운 동기화 (근접캠 캘리브가 높이 의존)
        # 개방 대기 1.0→0.3s (7/21): 직후 측정(프레임 대기+추론 ≥1s)이 나머지
        # 개방 시간을 흡수한다 — 접근 전진 시작 전엔 항상 완전 개방 상태.
        self.do_gripper("OPEN", wait=0.3)
        m, attempt = self.measure_with_retry(cell, first=True)
        if m is None:
            rec["fail"] = "approach: 대상 검출 실패"
            rec["phases"]["approach"] = round(time.monotonic() - t, 2)
            self.log("  [접근] 검출 실패 — 셀 skip")
            self.skip_cells.add(cell)
            street_bail()
            return rec
        self.log(f"  [접근] 검출 [{m['cls']}|{m['cam']}|{m.get('src', '?')}] "
                 f"전방 {m['y_forward']:.3f}m 좌우 {-m['x_right']:+.3f}m "
                 f"(시도 {attempt}/{MEASURE_TRIES}, conf {m['conf']:.2f}"
                 + (f", ground참조 {m['y_ground_ref']:.3f}m"
                    if m.get("y_ground_ref") is not None else "") + ")")
        rec["measure"] = {kk: round(m[kk], 4) if isinstance(m[kk], float) else m[kk]
                          for kk in ("x_right", "y_forward", "cls", "conf", "cam",
                                     "src", "y_ground_ref")}
        rec["measure"]["attempt"] = attempt
        if m["y_forward"] > FAST_HOP_THRESHOLD_M:
            hop_dx = max(0.05, m["y_forward"] - HOP_LAND_M)
            self.log(f"  [접근] {FAST_HOP_THRESHOLD_M}m 초과 측정 — 방어 홉 "
                     f"{hop_dx:.2f}m (잔여 {HOP_LAND_M}m 착지) 후 재측정")
            self.do_move(hop_dx, -m["x_right"], max_v=self.v_approach)
            adv[0] += hop_dx
            adv[1] += -m["x_right"]
            self.spin(0.3)
            m2, hop_attempt = self.measure_with_retry(cell)
            if m2 is None:
                rec["fail"] = "approach: 홉 후 재검출 실패"
                rec["phases"]["approach"] = round(time.monotonic() - t, 2)
                self.skip_cells.add(cell)
                street_bail()
                return rec
            m = m2
            rec["measure"]["hop_attempt"] = hop_attempt
            self.log(f"  [접근] 재측정 전방 {m['y_forward']:.3f}m "
                     f"좌우 {-m['x_right']:+.3f}m "
                     f"(시도 {hop_attempt}/{MEASURE_TRIES})")
        # 파지(CLOSE)로 이어지는 최종 측정마다 대상 클래스 재검증 —
        # 전진 후엔 물체가 그리퍼 앞 ~11cm라 재인식 불가, 여기가 마지막 기회.
        # 실전 모드는 셀의 스캔 identity(타깃 중 하나)가 기대 클래스.
        exp_cls = rec["identity"] if self.targets else None
        # [2026-07-24 정합성 인자 모드] probe 셀은 스캔 라벨이 아니라 **사냥 대상**이
        # 기대 클래스이고, 검증은 strict(적극 확인만 통과)다. 사양 §6.5.
        strict = False
        if self.cf_on and cell in self.cf_intent:
            intent = self.cf_intent[cell]
            strict = intent != rec["identity"]
            if strict:
                self.log(f"  [대상검증] probe 셀 — 기대 {intent} "
                         f"(스캔 라벨 {rec['identity']}), strict 게이트")
            exp_cls = intent
        ok_t, why_t = self.verify_target(m, exp_cls, strict=strict)
        # 소진(exhaustion) 갱신 — 결정적 근접 관측만 hard evidence 로 적립한다.
        # 이것이 남은 셀의 p 를 끌어올리고, 상대 클래스가 다 차면 p=1 을 만든다.
        if self.cf_on and self._cf_last_ident in fl.FRUITS:
            self.cf_evidence[cell] = self._cf_last_ident
            rec["cf_evidence"] = self._cf_last_ident
        if why_t:
            self.log(f"  [대상검증] {why_t}")
            rec["target_verify"] = {"ok": ok_t, "why": why_t,
                                    "strict": strict, "expected": exp_cls}
        if not ok_t:
            rec["fail"] = f"approach: 대상 클래스 모순 ({why_t})"
            rec["phases"]["approach"] = round(time.monotonic() - t, 2)
            self.log("  [대상검증] 대상 아님 확정 — 셀 skip")
            self.skip_cells.add(cell)
            street_bail()
            return rec
        # strict 통과 = 이 셀이 intent 클래스임을 근접에서 **적극 확인**했다.
        # 적재 집계(stage_collect)가 스캔 라벨('plain')이 아니라 실제 클래스로
        # 잡히도록 여기서 identity 를 승격한다.
        if strict:
            self.log(f"  [정합성] probe 성공 — {cell} identity "
                     f"{rec['identity']} → {exp_cls} 승격")
            rec["cf_probe"] = {"scan": rec["identity"], "resolved": exp_cls}
            rec["identity"] = exp_cls
        dx = max(-0.3, min(0.9, m["y_forward"] - fl.GRIP_FORWARD_M))
        dy = max(-0.45, min(0.45, -m["x_right"]))
        # [2026-07-24] 접근 정착 컷 도입 — 종전 blocking do_move 는 데드존 그라인드로
        # 4~7s 를 전소했다(오늘 접근 35건 중 4건, 정상군 1.0~1.4s). 롤백 시 아래
        # 주석 한 줄로 복원. 근거·분포: docs/07-results-and-lessons.md §7.5.
        # res = self.do_move(dx, dy, max_v=self.v_approach)   # [폐기 2026-07-24]
        res = self.do_approach_fire(dx, dy, max_v=self.v_approach)
        adv[0] += dx
        adv[1] += dy
        move_ok = bool(res.get("ok")) or res.get("reason") == "firmware move timeout"
        self.log(f"  [접근] 단발이동 ({dx:+.3f},{dy:+.3f}) → "
                 f"{res.get('reason') or 'ok'}")
        rec["phases"]["approach"] = round(time.monotonic() - t, 2)
        if not move_ok:
            rec["fail"] = f"approach: 이동 실패 ({res.get('reason')})"
            self.skip_cells.add(cell)
            street_bail()   # 이동 실패 시 실이동량 불명 — 되감기는 최선 시도
            return rec

        # --- d. GRASP: CLOSE → grasp_state=="held" 게이트 (pos-gap 판정) ---
        # [2026-07-23 저녁 조작자 지시] 게이트 **부활 — 이게 기본값이다.**
        # 같은 날 오전에 폐지했던 이유(pos-gap 오탐으로 이미 문 물체를 놓거나
        # 셀을 통째로 버림)는 유효하지만, 판정 없이 진행하면 **빈 그리퍼로
        # 운반+적재 사이클(실측 ~12s)을 통째로 태우고 슬롯까지 한 칸 소모**하는
        # 쪽이 더 비싸다. 실측 오탐 상한은 46사이클 중 3건(6.5%, 전부 적재까지
        # 진행됨) — 재시도 1회로 흡수하고, 그래도 미확인이면 셀 skip.
        # 게이트 없이 돌리려면 --place-anyway (= 오전의 폐지 상태와 동등).
        # 브리지 판정 래치는 CLOSE + grasp_check_delay(0.6) + window(0.25)
        # = 0.85s 에 확정되므로 wait_held 는 보통 그 시점에 조기 반환한다.
        # [폐기 2026-07-23 저녁 — 게이트 없는 무조건 진행분. 롤백 시 복원]
        #   self.wait_held(timeout=GRASP_OBSERVE_S)
        #   rec["grasped"] = True            # 항상 적재로 진행
        t = time.monotonic()
        if self.fn is not None:
            self.fn.last.pop("grasp", None)  # 이전 판정 잔존 방지
        self.do_gripper("CLOSE")
        held = self.wait_held()
        state = (self.fn.grasp_state() if self.fn is not None else "") or ""
        if not held and state == "empty":
            # `empty` = 판정이 살아 있고 **빈손이라고 적극 선언**한 경우
            # (위치갭 < 8.0° — 손가락이 맞닿는 데까지 닫혔다). 이때만 재시도.
            self.log("  [파지] 빈손 판정(state=empty) "
                     "— 재시도 (OPEN → +0.02m → CLOSE)")
            self.do_gripper("OPEN", wait=1.2)
            self.do_move(0.02, 0.0, max_v=0.2, tag='파지재시도_전진')
            adv[0] += 0.02                   # street 되감기 잔량 갱신
            if self.fn is not None:
                self.fn.last.pop("grasp", None)
            self.do_gripper("CLOSE")
            held = self.wait_held()
            state = (self.fn.grasp_state() if self.fn is not None else "") or ""
        rec["grasped"] = held
        rec["grasp_state"] = state or None
        rec["phases"]["grasp"] = round(time.monotonic() - t, 2)
        if not held and state == "empty" and not self.args.place_anyway:
            rec["fail"] = "grasp: 2회 실패 (grasp_state=empty)"
            self.log("  [파지] 재실패(empty) — 셀 skip, 후퇴")
            self.do_gripper("OPEN", wait=1.0)
            self.do_move(-0.2, 0.0, max_v=self.v_approach, tag='파지실패_후퇴')
            adv[0] -= 0.2                    # street 되감기 잔량 갱신
            self.skip_cells.add(cell)
            street_bail("파지실패복귀")       # street 모드 복귀 (2026-07-22 계약)
            return rec
        if held:
            self.log("  [파지] 성공 (grasp_state=held)")
        elif state == "empty":
            rec["grasp_forced"] = True
            self.log("  [파지] empty 판정이지만 --place-anyway — 적재 강행")
        else:
            # [2026-07-23 저녁 조작자 지시] `unknown`/무응답 = **판정이 안 나온
            # 것**이지 빈손이 아니다. OpenRB 가 현재 간헐 이상이라 실제로 물었는데
            # 판정만 못 내는 사례가 잦다 — 여기서 재시도(OPEN)하면 문 물체를
            # 스스로 떨어뜨리고, skip 하면 멀쩡한 수거를 버린다. 그래서 미확인은
            # **기본적으로 통과**시킨다 (종전엔 --place-anyway 에서만 통과).
            # 통과분은 rec.grasp_unconfirmed 로 표시해 리포트에서 분리 집계한다.
            rec["grasp_unconfirmed"] = True
            self.log(f"  [파지] 판정 미확인(state={state or '피드백 없음'}) "
                     "— OpenRB 피드백 이상. 재시도 없이 적재 진행")
        self.collected.add(cell)

        # --- --no-place: 집기만 확인, 그 자리 반납 ---
        if self.args.no_place:
            self.do_move(-0.2, 0.0)
            adv[0] -= 0.2                  # street 되감기 잔량 갱신
            self.do_gripper("OPEN", wait=1.0)
            self.log("  [no-place] 후진 0.2m 후 반납 — 사이클 종료")
            self.collected.discard(cell)   # 반납품은 장애물로 유지
            self.skip_cells.add(cell)      # 재타겟 방지
            street_bail("반납복귀")
            rec["ok"] = True
            return rec

        # --- e. CARRY: 후진 → 스테이징 라우팅 → 슬롯 적재 (63번 계약) ---
        t = time.monotonic()
        carried_west = False
        if street:
            # [2026-07-22] street 운반 (§5 상태 5~7): 접근 이동 되감기(요 135°
            # 유지, 물체 물고 후진 = I-3) → mini-goal 에서 CW45 북향 →
            # street 남하로 하이웨이.
            if abs(adv[0]) > 0.01 or abs(adv[1]) > 0.01:
                self.do_move(-adv[0], -adv[1], max_v=self.v_approach,
                             tag='파지후_되감기')
            if south_row:
                # [2026-07-23 조작자 지시] 남단 행(y=100) 예외 — mini-goal 이
                # 공식 y=75 = 하이웨이 상단이라 **여기서 곧장 적재 드리프트**로
                # 간다. 북향 복원(45°)과 남하(15cm)를 통째로 생략하고, 드리프트가
                # 135°→-135° 회전을 이동에 접는다 (제자리 회전 2회 소멸).
                # 기하: 이 지점은 격자 열 사이라 최근접 물체가 대각 35.36cm,
                # 회전 스윕 26cm + 물체 반폭 4cm = 30cm → 여유 5.4cm.
                # 드리프트는 남서향으로 즉시 멀어지므로 이후 여유는 증가한다.
                south_ok = True
                self.log("  [운반] 남단 행 — 북향·남하 생략, mini-goal(공식 y=75)"
                         "에서 즉시 적재 드리프트")
            else:
                self.street_face(False)
                south_ok = self.street_south(gx, "운반_남하")
            if not south_ok and not self.args.dry_run:
                # [2026-07-22] 남하 실패는 최근접 street 재스냅 후 1회 재시도.
                # 실패 채 "계속" 하면 이후 라우팅이 필드 한복판에서 계획된다
                # (00:58:05 실기 — 조작자 "루프 탈출 후 골대 직진 금지" 지시).
                p_now = self.pose()
                if p_now is not None:
                    south_ok = self.street_south(
                        self.street_x_snap(p_now[0]), "운반_남하_재시도")
            if south_ok:
                # [2026-07-22 04시 조작자 지시] 적재 접근은 **드리프트** —
                # '서향 90° 회전 후 전진'(당일 03시안)을 다시 교체. 회전
                # 정지시간이 사라지고, 이동 중 wz 로 적재함 구석 정면
                # (STORAGE_CORNER_YAW=-135°=공식 225°)을 만들어 도착 즉시
                # 적재 전진이 가능하다. 하이웨이 자유밴드 전용.
                if self.args.dry_run:
                    self.sim_pose[:] = [STAGING_XY[0], STAGING_XY[1],
                                        STORAGE_CORNER_YAW]
                    self.log("  [dry-run] 드리프트 → 스테이징 (구석 정면)")
                    carried_west = True
                    rec["carry_route"] = {"mode": "street_drift",
                                          "waypoints": [list(STAGING_XY)]}
                else:
                    # 관객 피드: 드리프트는 경기 전체에서 **여기 한 곳뿐**이다.
                    # (street_nav 의 복귀 드리프트는 drive_to 내부 경로 로직이라
                    #  미션 단계로는 잡히지 않는다.)
                    self._drifting = True
                    try:
                        self.publish_match_state(force=True)
                        r_w = self.snav.drive_drift(STAGING_XY, STORAGE_CORNER_YAW,
                                                    "운반_드리프트",
                                                    v_max=snv.V_STREET_MPS)
                    finally:
                        self._drifting = False
                    carried_west = bool(r_w.get("ok"))
                    if carried_west:
                        self.log(f"  [운반] street_drift — 드리프트 "
                                 f"{r_w.get('sec', 0):.1f}s (구석 정면 도착)")
                        rec["carry_route"] = {"mode": "street_drift",
                                              "waypoints": [list(STAGING_XY)]}
                    else:
                        self.log(f"  ⚠ [운반] 드리프트 미수렴 "
                                 f"({r_w.get('reason')}) — 라우팅 폴백")
        elif self.place_settle:
            # 속도 미지정이면 position_max_rad_s(6.0=0.233m/s) 기본값이라
            # 느리다 — 접근 속도로 명시. 7/20 실기 8s 정체 구간.
            self.do_move(-0.2, 0.0, max_v=self.v_approach, tag='파지후_후진')
        if not carried_west:
            pose = self.pose()
            route2 = plan_route((pose[0], pose[1]), STAGING_XY,
                                self.build_obstacles())
            if (street and not self.args.dry_run and pose is not None
                    and pose[1] > HIGHWAY_FREE_Y_M + 0.05
                    and route2["mode"] in ("direct", "blocked")):
                # [2026-07-22] 남하 미완(필드 안) 상태의 대각 직진 금지 —
                # street 격자 강제. 물체는 격자점에만 있고 street 는 그 사이
                # 통로라 격자 추종이 항상 안전측이다.
                alt = (plan_route_street((pose[0], pose[1]), STAGING_XY,
                                         self.build_obstacles())
                       or plan_route_street((pose[0], pose[1]), STAGING_XY, []))
                if alt is not None:
                    route2 = {"mode": "street_forced",
                              "waypoints": alt["waypoints"],
                              "reason": "남하 미완 — 대각 직진 금지, "
                                        "street 격자 강제 (2026-07-22)"}
            self.log(f"  [운반] {route2['mode']} — {route2['reason']}")
            rec["carry_route"] = {"mode": route2["mode"],
                                  "waypoints": [list(w) for w in route2["waypoints"]]}
            self.follow_route(route2, self.v_cruise, settle=self.place_settle)

        pin_i = self.slot_idx % len(STORAGE_PINS)
        slot, obj_cm = STORAGE_PINS[pin_i]
        reuse = self.slot_idx >= len(STORAGE_PINS)  # 6개째부터 핀 재사용(적층 허용)
        # 적재는 항상 구석(45도)을 정면으로 보고 진입한다 (7/20). 슬롯마다
        # atan2 로 다른 방향을 보면 그리퍼가 보관함 턱/벽과 비스듬히 만난다.
        # [2026-07-22 04시] tol 0.20 — 드리프트 도착 잔차(±12°)의 소각 회전
        # 그라인드 방지 (조작자 지시 "적재 시작점 정착기준 완화").
        self.do_rotate(STORAGE_CORNER_YAW, tol=0.20)
        pose = self.pose()
        # 적재 전 pose 게이트 (7/21 16:18 실기 참사: 로컬라이저 이상 상태에서
        # 스테이징이라 믿고 START 코너에 투하). status 신선도(1s)와 스테이징
        # 이탈(0.45m)을 검사, 이상이면 재이동 1회. '자신있게 틀린' 미러락은
        # 여기서 못 잡는다 — 회전 총량 감소(align 완화)가 1차 방어.
        def gate_bad():
            """(stale, off_m) — pose() 는 마지막 status 캐시라 None 이 되지
            않으므로(영구 캐시) 신선도는 stamp 로만 판정할 수 있다 (7/21 리뷰:
            종전 'pose is None' 중단 조건은 도달 불가 데드코드였다)."""
            st = (self.fn is not None and not self.args.dry_run
                  and time.monotonic() - self.fn.stamp.get("status", 0.0) > 1.0)
            p = self.pose()
            o = (math.hypot(p[0] - STAGING_XY[0], p[1] - STAGING_XY[1])
                 if p is not None else 9.9)
            return st, o

        stale, off = gate_bad()
        if stale or off > 0.45:
            self.log(f"  ⚠ [적재게이트] {'status 불통(stale) ' if stale else ''}"
                     f"스테이징 이탈 {off:.2f}m — 재이동 1회 후 재확인")
            rec["place_gate"] = {"stale": bool(stale), "off_m": round(off, 2)}
            self.spin(0.6)
            self.do_goto(STAGING_XY[0], STAGING_XY[1], self.v_cruise)
            self.do_rotate(STORAGE_CORNER_YAW, tol=0.20)
            stale, off = gate_bad()
            if stale or off > 0.45:
                # fail-closed: 오적재 참사(16:18 START 투하)를 막는 게 우선.
                # 물고 다음 사이클로 가면 접근 OPEN에서 아무 데나 떨어뜨리므로
                # 현 위치에서 개방(보관함 밖 = 0점, 감점 없음)하고 사이클 종료.
                rec["fail"] = (f"place: pose 게이트 재실패 "
                               f"(stale={stale}, off={off:.2f}m) — 현장 개방")
                self.log("  ⚠ [적재게이트] 재확인 실패 — 보관함 투하 포기, "
                         "현 위치 개방 (오적재 방지)")
                self.do_gripper_release(wait=0.5)
                self.do_move(-0.25, 0.0, max_v=self.v_place, tag='게이트_후퇴')
                self.do_gripper("CLOSE", wait=0.0)
                rec["phases"]["carry"] = round(time.monotonic() - t, 2)
                return rec
            pose = self.pose()
        dxw, dyw = slot[0] - pose[0], slot[1] - pose[1]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        fwd = dxw * c + dyw * s
        left = -dxw * s + dyw * c
        # 슬롯은 *로봇 목표*다 (물체 자리가 아님) — 그리퍼 길이 보정 없음.
        adv = fwd + PLACE_PUSH_M   # [2026-07-22 04시] 2.5cm 더 밀어 넣기
        self.log(f"  [적재] 핀 {pin_i + 1}/{len(STORAGE_PINS)}"
                 f"{' (재사용-적층)' if reuse else ''} "
                 f"로봇목표 ({slot[0]:+.2f},{slot[1]:+.2f})m "
                 f"→ 물체 예상 ({obj_cm[0]:.0f},{obj_cm[1]:.0f})cm "
                 f"— 상대 전진 {adv:+.2f}m (좌 {left:+.2f})")
        # [2026-07-22 오후 조작자 지시] 적재 완료 정착 대기 1.0s 컷 —
        # +2.5cm 푸시는 의도된 밀착이라 펌웨어 위치정착(DONE)이 안 오는 게
        # 정상 경로다. 예상시간+1.0s 후 선점 컷하고 투하로 진행한다.
        # [폐기 2026-07-22 오후 — 롤백 시 이 호출 복원]
        # res_adv = self.do_move(adv, left, max_v=self.v_place,
        #                        timeout=PLACE_MOVE_TIMEOUT_S, tag='적재_전진')
        res_adv = self.do_move_fire(adv, left, max_v=self.v_place,
                                    tag='적재_전진', settle_s=1.0)
        if res_adv.get("settle_cut"):
            # 정착 컷은 정상(밀착)일 수도, 중도 스톨일 수도 있다 — localizer
            # pose 로 판별: 슬롯에서 0.25m 넘게 남았으면 종전 스톨 처리.
            p_c = self.pose()
            off_c = (math.hypot(p_c[0] - slot[0], p_c[1] - slot[1])
                     if p_c is not None else None)
            if off_c is not None and off_c > 0.25:
                res_adv = {"ok": False, "reason": f"정착컷 후 이탈 {off_c:.2f}m"}
            else:
                self.log("  [적재] 정착 컷 1.0s — 밀착 상태로 투하 진행")
        if not res_adv.get("ok"):
            # 펌웨어 move timeout = 스톨 (벽/물체를 밀며 정착 실패 — 16:13 실기
            # 8.7s ok=False 후 그대로 투하했던 결함). 재전진은 같은 스톨을 다시
            # 8s+ 갈기만 하므로 소폭 후퇴 후 투하로 전환하고 의심 표기.
            self.log("  ⚠ [적재] 전진 스톨 (펌웨어 timeout) — 0.10m 후퇴 후 투하, "
                     "적재 의심 표기")
            rec["place_stall"] = True
            self.do_move(-0.10, 0.0, max_v=self.v_place, tag='스톨_후퇴')
        # [2026-07-22 04시 조작자 지시] 마지막 타깃(이 적재로 전 quota 완성)은
        # **투하·후퇴 없이 문 채로 정지**. 룰북에 로봇 퇴장 규정이 없어 그대로
        # 있는 편이 안전하다 — 투하 튐/기존 핀 붕괴/후퇴 접촉 리스크 제거.
        # 물체는 이미 보관함 안(적재 전진 완료 상태)이다.
        if self.targets and rec["identity"] in self.targets:
            would = dict(self.placed_by_cls)
            would[rec["identity"]] = would.get(rec["identity"], 0) + 1
            if all(would.get(k, 0) >= q for k, q in self.targets.items()):
                self.hold_last = True
                rec["hold_last"] = True
                rec["slot"] = list(slot)
                rec["placed"] = True
                rec["ok"] = True
                rec["phases"]["carry"] = round(time.monotonic() - t, 2)
                self.log("  [적재] 마지막 타깃 — 투하/후퇴 생략, 문 채로 종료 "
                         "(2026-07-22 조작자 지시)")
                return rec
        # 투하: 현재 위치에서 상대 틱만큼만 살짝 벌린다(벽 간섭 회피).
        self.do_gripper_release(wait=0.5)     # 낙하에 충분 (7/21 1.0→0.5)
        # [2026-07-22 오후 조작자 지시] 후퇴는 odom 단발 + 정착 컷 — DONE
        # 블로킹 대기(최대 90s) 제거. 잔여 그라인드는 선점 컷, 미세 위치
        # 잔차는 다음 사이클 복귀 드리프트(폐루프)가 흡수한다.
        # [폐기 2026-07-22 오후 — 롤백 시 이 호출 복원]
        # self.do_move(-STORAGE_RETREAT_M, 0.0, max_v=self.v_place,
        #              timeout=PLACE_MOVE_TIMEOUT_S, tag='적재_후퇴')          # 후퇴 0.35
        self.do_move_fire(-STORAGE_RETREAT_M, 0.0, max_v=self.v_place,
                          tag='적재_후퇴', settle_s=1.0)                       # 후퇴 0.35
        # 재폐합은 완료 대기 불필요 — 후속이 경로계획/주행 (7/21 wait 1.0→0)
        self.do_gripper("CLOSE", wait=0.0)                   # 63번 계약: 재폐합
        self.slot_idx += 1
        rec["slot"] = list(slot)
        rec["placed"] = True
        rec["ok"] = True
        rec["phases"]["carry"] = round(time.monotonic() - t, 2)
        self.log(f"  [적재] 완료 — 후퇴 {STORAGE_RETREAT_M}m")
        return rec

    def apply_arena_speed(self):
        """프로파일 cruise 를 arena 노드에 실제로 반영한다.

        7/20: do_goto 는 arena goal(폐루프)로 주행하는데 max_v 를 싣지 않아
        주행 속도가 arena 의 max_linear_mps(기본 0.36)에 고정돼 있었다 —
        safe/normal/fast 의 cruise 0.5/0.8/0.95 가 전부 무시되고 접근·적재
        (move_relative 경로)에서만 차이가 보이던 원인.

        [2026-07-23] 드리프트 회전 튜닝 로그도 여기서 낸다 — 런당 1회, y 입력
        전에 실행되는 유일한 SETUP 훅이라 값 확인 시점이 가장 이르다.
        """
        # dry-run 에서도 찍는다: 실행인자 배선을 하드웨어 없이 확인하는 경로.
        if self.snav is not None:
            same = (self.snav.drift_yaw_rate == snv.DRIFT_YAW_RATE
                    and self.snav.drift_kp_yaw == snv.DRIFT_KP_YAW)
            self.log(f"[SETUP] 적재 드리프트 회전 캡 {self.snav.drift_yaw_rate} "
                     f"rad/s · 게인 {self.snav.drift_kp_yaw} /s"
                     f"{' (기본)' if same else ' (실행인자 override)'} — "
                     f"mini-goal 회전(turn_to {snv.TURN_YAW_RATE}/"
                     f"{snv.TURN_KP_YAW})과 별개")
        if self.args.dry_run:
            self.log(f"  [dry-run] arena max_linear_mps ← {self.v_cruise}")
            return
        # [2026-07-22 저녁] rclpy 서비스 직접 호출로 교체 — 종전 `ros2 param
        # set` 서브프로세스는 CLI 데몬 콜드스타트로 3~5s 를 먹어 "y 후 출발
        # 지연"의 주범이었다. 서비스 호출은 ~0.1s.
        try:
            from rcl_interfaces.msg import (Parameter, ParameterType,
                                            ParameterValue)
            from rcl_interfaces.srv import SetParameters
            import rclpy
            cli = self.fn.node.create_client(
                SetParameters, "/arena_control_node/set_parameters")
            ok = False
            if cli.wait_for_service(timeout_sec=2.0):
                req = SetParameters.Request()
                req.parameters = [Parameter(
                    name="max_linear_mps",
                    value=ParameterValue(
                        type=ParameterType.PARAMETER_DOUBLE,
                        double_value=float(self.v_cruise)))]
                fut = cli.call_async(req)
                rclpy.spin_until_future_complete(self.fn.node, fut,
                                                 timeout_sec=2.0)
                res = fut.result()
                ok = bool(res and res.results and res.results[0].successful)
            self.fn.node.destroy_client(cli)
            self.log(f"[SETUP] arena max_linear_mps ← {self.v_cruise} "
                     f"({'ok' if ok else '실패 — 기본값으로 계속'})")
        except Exception as e:  # noqa: BLE001 — 속도 반영 실패로 런을 죽이지 않는다
            self.log(f"[SETUP] arena 속도 반영 실패(기본값으로 계속): {e}")
        # [폐기 2026-07-22 저녁 — 서브프로세스 경로. 롤백 시 복원]
        # cmd = ["ros2", "param", "set", "/arena_control_node",
        #        "max_linear_mps", str(float(self.v_cruise))]
        # r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

    def safe_shutdown(self):
        """종료 시(정상/예외/Ctrl-C 공통) 안전 자세로 되돌린다 — 마스트 하강 +
        그리퍼 개방. 7/20: 프로그램이 죽으면 마스트가 올라간 채 남아 다음 런의
        MAST_UP이 하드스톱을 치거나 물체를 문 채로 방치되던 문제."""
        if self.args.dry_run or self.fn is None:
            return
        try:
            if self.hold_last:
                # [2026-07-22 04시] 마지막 타깃 홀드 종료 — 그리퍼를 열면
                # 문 물체가 이탈하므로 개방을 생략한다. LIFT_TO_BOTTOM 은
                # 이 시점에 이미 다운이라 무동작(멱등)이며, 마스트 올라간 채
                # 죽는 비정상 경로 대비 보험으로만 남긴다.
                self.log("[종료] 마지막 타깃 홀드 — 그리퍼 개방 생략 (파지 유지)")
                self.fn.lift("LIFT_TO_BOTTOM")
                self.fn.lift_wait_idle(timeout=20.0)
                return
            self.log("[종료] 안전 자세 — 마스트 하강 + 그리퍼 개방")
            self.fn.gripper("OPEN")
            self.fn.lift("LIFT_TO_BOTTOM")
            self.fn.lift_wait_idle(timeout=20.0)
        except Exception as e:  # noqa: BLE001 — 종료 경로는 절대 죽이지 않는다
            self.log(f"[종료] 안전 자세 실패(무시): {e}")

    def gripper_alive(self) -> bool:
        """그리퍼 텔레메트리 생존 확인 — 7/19 실기에서 서보 미검출
        (STATUS READY 0 / FAULT 1 / DXL_POWER 0)인데 파지를 시도해
        전부 허공 실패했던 것의 사전 가드."""
        if self.fn is None:
            return False
        m = self.fn.last.get("gripper_state")
        if not m:
            return False
        try:
            st = json.loads(m.data)
        except ValueError:
            return False
        return st.get("connected") and st.get("width_mm") is not None

    # ---- 경량 HUD 보조 (2026-07-23) ----
    def hud_score(self) -> int:
        """현재 예상점수 (오픽업 감점 미반영 — stage_report 의 score_est 와 동일 식)."""
        return sum(POINTS.get(k, 0) * n for k, n in self.placed_by_cls.items())

    def hud_key(self, identity: str) -> str:
        """`apple2` = 그 클래스에서 지금 노리는 순번. 사진 번호가 곧 quota 순번이다.
        Counter 라 미기록 클래스는 0 을 돌려주고 키를 만들지 않는다."""
        return f"{identity}{self.placed_by_cls[identity] + 1}"

    def stage_collect(self):
        t0 = time.monotonic()
        n_max = self.args.max_objects
        if n_max is None:
            # 실전 모드: quota 합(기본 7) + 실패 셀 재시도 여유 / 그 외 종전 3.
            n_max = sum(self.targets.values()) + 5 if self.targets else 3
        if self.targets:
            tgt_txt = " + ".join(f"{k} x{q}" for k, q in sorted(self.targets.items()))
            self.log(f"\n[COLLECT] 수거 루프 시작 (실전 타깃 {tgt_txt}, "
                     f"최대 {n_max}사이클, 순서 {getattr(self.args, 'order', 'nearest')}"
                     " — 비타깃 수거 금지)")
        else:
            self.log(f"\n[COLLECT] 수거 루프 시작 (최대 {n_max}개"
                     + (f", 우선 클래스 {self.args.target_class}" if
                        self.args.target_class else ", 최근접 순") + ")")
        if not self.args.dry_run and not self.gripper_alive():
            self.log("  ⚠ 그리퍼 텔레메트리 없음 (서보 미검출/전원?) — "
                     "파지는 전부 실패할 상태. OpenRB 전원 사이클(바닥에서!) "
                     "+ 서보 커넥터 확인 필요.")
            if not self.args.yes:
                a = input("[체크] 그리퍼 없이 접근까지만 진행할까요? (y/n): ").strip().lower()
                if a not in ("y", "yes", "ㅛ"):
                    self.mark("COLLECT", False, t0, "그리퍼 텔레메트리 없음 — 조작자 중단")
                    return
        ev = getattr(self, "_scan_full_done", None)
        if not self.cells and ev is not None and not ev.is_set():
            self.log("  동쪽 부분맵 비어있음 — 서쪽 확정 대기")
            ev.wait(timeout=60.0)
        if not self.cells:
            self.log("  확정 셀 없음 — 수거 생략")
            self.mark("COLLECT", False, t0, "확정 셀 없음")
            return
        if not self.args.dry_run and self.models is None:
            if not self.load_models():
                self.mark("COLLECT", False, t0, "모델 로드 실패")
                print("\n중단: 접근 측정에 YOLO 필요.")
                sys.exit(6)
        for i in range(1, n_max + 1):
            if self.targets and all(self.placed_by_cls[k] >= q
                                    for k, q in self.targets.items()):
                self.log(f"  전 타깃 quota 달성 — {i - 1}사이클에서 조기 종료")
                break
            cell = self.pick_target_gated()   # 동서 분할 우선순위 게이트
            if cell is None:
                self.log(f"  남은 대상 없음 — {i - 1}개에서 종료")
                break
            identity = self.cells.get(cell, {}).get("identity") or "plain"
            # [2026-07-24] probe 셀은 스캔 라벨('plain')이 아니라 **사냥 대상**을
            # 띄운다 — HUD 가 "지금 무엇을 노리는가"를 보여주는 화면이라서다.
            if self.cf_on:
                identity = self.cf_intent.get(cell, identity)
            hud_write(self.hud_key(identity), self.hud_score())   # 2026-07-23 경량 HUD
            if getattr(self, "rec", None) is not None:   # 프레임 기록에 단계 라벨
                self.rec.label = f"collect#{i} {identity} cell{tuple(cell)}"
            t_cyc = time.monotonic()
            rec = self.collect_one(cell, i)
            rec["total_sec"] = round(time.monotonic() - t_cyc, 2)
            self.report["cycles"].append(rec)
            # quota 집계: 적재 성공 기준 (--no-place 리허설은 파지 성공 기준)
            done = rec["placed"] or (self.args.no_place and rec["grasped"])
            if self.targets and done and rec.get("identity") in self.targets:
                self.placed_by_cls[rec["identity"]] += 1
                # 2026-07-23 경량 HUD: 점수만 올려 기록한다. 이미지 키는 방금 수거한
                # 물체 그대로 두고, HUD 가 "점수 상승"을 감지해 points 화면을 1틱 띄운다.
                # (다음 목표 확정이 1초 뒤라 러너 쪽 순서만으로는 0.1Hz 폴링이 놓친다)
                hud_write(self.hud_key(rec["identity"]), self.hud_score())
            self.save_report()
            # 관객 피드: 사이클 종료 — "지금 이것" 강조를 푼다. 셀의 다음 상태
            # (collected / skipped)는 collect_one 이 이미 갱신해 둔 집합이 정한다.
            self._active_cell = None
            self._cycle_live = None
            state = "적재" if rec["placed"] else (
                "파지만" if rec["grasped"] else f"실패({rec.get('fail')})")
            quota_txt = ""
            if self.targets:
                quota_txt = " | quota " + " ".join(
                    f"{k} {self.placed_by_cls[k]}/{q}"
                    for k, q in sorted(self.targets.items()))
            self.log(f"[수거 {i}] {state} — {rec['total_sec']}s{quota_txt}")
        hud_write("end", self.hud_score())   # 2026-07-23 경량 HUD — 경기 종료 화면
        detail = {"attempted": len(self.report["cycles"]),
                  "grasped": sum(1 for r in self.report["cycles"] if r["grasped"]),
                  "placed": sum(1 for r in self.report["cycles"] if r["placed"])}
        if self.targets:
            detail["placed_by_cls"] = dict(self.placed_by_cls)
            detail["score_est"] = sum(POINTS.get(k, 0) * n
                                      for k, n in self.placed_by_cls.items())
        self.mark("COLLECT", True, t0, detail)

    # =====================================================================
    def stage_report(self, t_mission0: float):
        total = round(time.monotonic() - t_mission0, 1)
        th = getattr(self, "_save_thread", None)
        if th is not None and th.is_alive():
            self.log("  스캔 이미지 인코딩 마무리 대기...")
            th.join(timeout=30.0)
        self.flush_pending_writes()   # [2026-07-24] 디스크 쓰기는 여기서 한 번에
        cycles = self.report["cycles"]
        self.report["move_stats"] = self.move_stats
        slow = [m for m in self.move_stats if m["sec"] > max(2.0, m["expect"] * 2.0)]
        if slow:
            self.log(f"[이동 병목] 예상의 2배 넘은 이동 {len(slow)}/{len(self.move_stats)}건:")
            for m in slow:
                self.log(f"    {m['tag'] or 'move'}: {m['sec']}s (예상 {m['expect']}s) d={m['d']}m ok={m['ok']}")
        grasped = sum(1 for r in cycles if r["grasped"])
        # [2026-07-23 저녁] held 확정분과 '판정 미확인이라 통과시킨' 분을 분리
        # 집계한다. OpenRB 피드백 이상이면 grasped 가 0 이어도 적재는 정상일 수
        # 있으므로, 이 둘을 합치면 실제 파지 실패가 리포트에서 안 보인다.
        unconfirmed = sum(1 for r in cycles if r.get("grasp_unconfirmed"))
        placed = sum(1 for r in cycles if r["placed"])
        okc = sum(1 for r in cycles if r["ok"])
        verdict = ("PASS" if (okc and okc == len(cycles)) else
                   "PARTIAL" if okc else "FAIL")
        self.report["summary"] = {
            "total_sec": total, "match_budget_sec": 180,
            "speed_profile": self.args.speed_profile,
            "speeds": {"cruise": self.v_cruise, "approach": self.v_approach,
                       "place": self.v_place},
            "cycles_attempted": len(cycles), "grasped": grasped,
            "grasp_unconfirmed": unconfirmed,
            "placed": placed, "ok": okc,
            "scan_cells": len(self.cells),
            "effective_params": {
                "votes_k": self.votes_k, "fruit_k": self.fruit_k,
                "pair": (self.args.pair if self.pair is not None else "off"),
                "scan_mode": self.args.scan_mode,
                "scan_shots": self.args.scan_shots,
                "range_mode": self.args.range_mode,
            },
            "gt_compare": self.report["scan"].get("gt_compare"),
            "targets": self.targets or None,
            "placed_by_cls": dict(self.placed_by_cls) if self.targets else None,
            "score_est": (sum(POINTS.get(k, 0) * n
                              for k, n in self.placed_by_cls.items())
                          if self.targets else None),
            "verdict": verdict}
        self.save_report()

        print("\n" + "=" * 60)
        print("== E2E 경기 테스트 리포트 ==")
        print(f"스캔: 확정 {len(self.cells)}셀, presence {len(self.presence)}셀")
        if self.report["scan"].get("gt_compare_text"):
            print(GT_UNRELIABLE_BANNER)
            print(self.report["scan"]["gt_compare_text"])
        unc = f" (+미확인통과 {unconfirmed})" if unconfirmed else ""
        print(f"수거: 시도 {len(cycles)} / 파지 {grasped}{unc} / 적재 {placed}")
        if unconfirmed:
            print(f"  ⚠ held 판정이 {unconfirmed}회 안 나왔다 (OpenRB 피드백 이상) "
                  "— 그리퍼 상태를 육안 확인할 것")
        if self.targets:
            score = sum(POINTS.get(k, 0) * n for k, n in self.placed_by_cls.items())
            print("실전 타깃: " + " ".join(
                f"{k} {self.placed_by_cls[k]}/{q}({POINTS.get(k, 0)}점)"
                for k, q in sorted(self.targets.items()))
                + f" → 추정 {score}점 / 100점 (오픽업 감점 미반영)")
        for r in cycles:
            state = ("적재 OK" if r["placed"] else
                     "파지만" if r["grasped"] else f"실패: {r.get('fail')}")
            if r.get("grasp_unconfirmed"):
                state += " [held미확인]"
            ph = " ".join(f"{k} {v}s" for k, v in r["phases"].items())
            print(f"  #{r['index']} 셀 {tuple(r['target'])} "
                  f"[{r['identity']}] {state} — {r['total_sec']}s ({ph})")
        over = " (3분 예산 초과!)" if total > 180 else ""
        print(f"속도 프로파일: {self.args.speed_profile} "
              f"(주행 {self.v_cruise} / 접근 {self.v_approach} / 적재 {self.v_place})")
        print(f"총 소요 {total}s / 경기 예산 180s{over}")
        print(f"판정: {verdict}")
        print(f"리포트: {self.out / 'report.json'}")
        print("=" * 60)

    def run(self):
        t0 = time.monotonic()
        self._t_mission0 = t0          # 관객 피드의 경과시간 기준
        hud_write("start", 0)   # 2026-07-23 경량 HUD — 경기 시작 화면 (잔상 제거)
        # [2026-07-23] 실기 전 구간 원해상도 프레임 상시 기록 (기본 3fps/캠, 0.21코어).
        # 실패 분석용 — 근거와 비용은 FrameRecorder 도크스트링 참조.
        self.rec = FrameRecorder(None if self.args.dry_run else self.fn, self.out,
                                 fps=self.args.frame_rec_fps).start() \
            if self.args.frame_rec_fps > 0 else None
        try:
            if not getattr(self, "_speed_applied", False):
                self.apply_arena_speed()   # main 밖 호출자(테스트 등) 대비
            self.stage_seed()
            if self.args.gt_map:
                # GT-map 모드: 스캔 위치까지만 가고 스캔 생략 — GT를 지도로
                # 사용해 접근/적재만 검증.
                # 마스트는 **의도적으로 다운 유지** — 스캔이 없으므로 상단캠이
                # 필요 없고, 접근 측정은 근접캠(NEAR_MOUNT)이라 마스트다운
                # 캘리브가 맞다. 7/20: 이걸 "마스트 고장"으로 오인한 사례가 있어
                # 로그로 명시한다. 하드웨어 점검이 목적이면 --mast-up.
                if self.args.mast_up:
                    self.log("[GT-MAP] --mast-up 지정 — 스캔은 없지만 마스트를 올린다"
                             " (하드웨어 점검용). 접근 측정은 근접캠이라 영향 없음.")
                    self.stage_mast("LIFT_TO_TOP", "MAST_UP", wait=False)
                else:
                    self.log("[GT-MAP] 마스트는 다운 유지 (스캔 없음 — 정상 동작이며"
                             " 고장이 아님). 올려서 확인하려면 --mast-up")
                self.stage_goto_center()
                if self.args.mast_up:
                    self.stage_mast_wait()
                self.log(f"[GT-MAP] 스캔 생략 — GT {len(self.gt)}셀을 지도로 사용")
                for cell, cls in self.gt.items():
                    self.cells[cell] = {"identity": cls, "votes": self.votes_k,
                                        "fruit_hits": {}, "conflict": None,
                                        "source": "gt"}
            elif self.args.skip_scan:
                self.log(f"[SCAN] --skip-scan: 기존 지도 사용 "
                         f"(확정 {len(self.cells)}셀) — 마스트/센터 이동 생략")
            else:
                # (YOLO 로드는 main()에서 프로그램 시작과 동시에 프리로드 중)
                # 마스트업은 레그1(서진)과 병렬 시작, 스캔지점 x 도달 시 프리샷
                # (마스트업, 인접 4셀 전담) → 레그2 (2026-07-21).
                # mid = 펌웨어 LIFT_TO_MID (홈 기준 절대 절반).
                mid = getattr(self.args, "scan_mast", "up") == "mid"
                self.stage_goto_center(
                    preshot=True,
                    mast_cmd="LIFT_TO_MID" if mid else "LIFT_TO_TOP",
                    mast_label="MAST_MID" if mid else "MAST_UP")
                self.stage_mast_wait()
                # 마스트다운은 stage_scan 내부(촬영 직후, 추론 전)에서 비대기로
                # 시작한다 — 하강 시간을 YOLO 추론과 겹치기 위함 (7/20).
                self.stage_scan()
            self.stage_collect()
        except KeyboardInterrupt:
            self.log("\n조작자 중단 (Ctrl-C) — 리포트 저장 후 종료")
            self.report["aborted"] = "KeyboardInterrupt"
        except SystemExit:
            raise
        except Exception:
            self.log("예외 발생:\n" + traceback.format_exc())
            self.report["aborted"] = traceback.format_exc()
        finally:
            if getattr(self, "rec", None) is not None:
                self.rec.stop()
                self.log(f"[frames] 원해상도 {self.args.frame_rec_fps}fps/캠 기록: "
                         f"{self.rec.saved}쌍 저장 (지연 {self.rec.dropped}회) "
                         f"→ {self.rec.dir}")
                self.report["frames"] = {"pairs": self.rec.saved,
                                         "fps": self.args.frame_rec_fps,
                                         "late_ticks": self.rec.dropped}
            self.safe_shutdown()
            self.stage_report(t0)


# =========================================================================
# --offline 자가테스트 (ROS/ultralytics 불필요)
# =========================================================================
def offline_selftest(args) -> int:
    print("== 오프라인 자가테스트 (ROS 없음) ==\n")
    fails = []

    # 1) GT 파서
    sample = "apple:150,200;plain:250,300;octahedron:100,150"
    gt = fl.parse_gt_text(sample)
    print(f"[GT 파서] 입력 '{sample}'")
    print("  " + fl.gt_summary_text(gt).replace("\n", "\n  "))
    if len(gt) != 3 or gt[(150, 200)] != "apple":
        fails.append("GT 파서")
    if args.gt_text or args.gt_file:
        user_gt = fl.prompt_gt(interactive=False, gt_text=args.gt_text,
                               gt_file=args.gt_file)
        print("[GT 파서] 사용자 GT: " + fl.gt_summary_text(user_gt).replace("\n", " / "))

    # 2) GT 비교 스모크
    cmp_res = fl.compare_with_gt(
        {(150, 200): {"identity": "apple"}, (250, 300): {"identity": "banana"}}, gt)
    print("\n[GT 비교]")
    print("  " + cmp_res["text"].replace("\n", "\n  "))
    if cmp_res["ok"] != 1 or len(cmp_res["missed"]) != 1:
        fails.append("GT 비교")

    # 3) 라우터 유닛 케이스 3개
    print("\n[코리도 라우터] inflation(표3) = "
          f"{inflation_for(3):.3f}m, inflation(표1) = {inflation_for(1):.3f}m")

    def ob(x, y, votes=5, cell=None):
        return {"cell": cell or (int((x + 2) * 100), int((y + 2) * 100)),
                "xy": (x, y), "votes": votes}

    cases = [
        ("케이스1 직선 클리어", (0.0, 0.0), (1.0, 1.0), [ob(-1.0, -1.0)], "direct"),
        ("케이스2 직선 막힘→L", (0.0, 0.0), (1.0, 1.0), [ob(0.5, 0.5)], "L"),
        ("케이스3 L 2안도 막힘→street/우회", (0.0, 0.0), (1.0, 1.0),
         [ob(0.5, 0.5), ob(1.0, 0.5), ob(0.05, 0.5)], "any"),
    ]
    for name, p0, p1, obstacles, expect in cases:
        r = plan_route(p0, p1, obstacles)
        wps = " → ".join(f"({w[0]:+.2f},{w[1]:+.2f})" for w in r["waypoints"])
        print(f"  {name}: mode={r['mode']}")
        print(f"    이유: {r['reason']}")
        print(f"    경유점: {p0} → {wps}")
        if expect == "L":
            got_ok = r["mode"].startswith("L_")
        elif expect == "any":
            got_ok = r["mode"] in ("street", "detour", "blocked")
        else:
            got_ok = r["mode"] == expect
        if not got_ok:
            fails.append(name)

    # 3.5) ★회귀: 반환 경로는 반드시 실제로 충돌이 없어야 한다.
    #     7/20 실기에서 detour 분기가 경로 검증 없이 우회점만 계산해
    #     장애물 6셀을 팽창반경 안으로 관통(최소 4.8cm)했다.
    print("\n[라우터 회귀] 가득 찬 아레나(28셀)에서 경로 충돌 검사")
    full = [{"cell": (x, y), "xy": fl.official_cm_to_map(x, y), "votes": 1}
            for x in fl.GRID_XS_CM for y in fl.GRID_YS_CM][:28]
    checks = [((0.50, 0.00), tuple(STAGING_XY), "파지→스테이징"),
              (tuple(fl.CENTER_SCAN_XY), (-1.00, -0.50), "센터→원거리셀")]
    for p0, p1, label in checks:
        obs = [o for o in full if math.hypot(o["xy"][0] - p1[0],
                                             o["xy"][1] - p1[1]) > 0.01]
        r = plan_route(p0, p1, obs)
        lim = STREET_CLEAR_M if r["mode"] == "street" else None
        cur, worst, hits = p0, 99.0, 0
        for w in r["waypoints"]:
            for o in obs:
                d = point_seg_dist(o["xy"], cur, w)
                worst = min(worst, d)
                if d < (lim if lim is not None else inflation_for(o["votes"])):
                    hits += 1
            cur = w
        print(f"  {label}: mode={r['mode']} 레그{len(r['waypoints'])} "
              f"침범 {hits} 최소거리 {worst:.3f}m")
        if r["mode"] not in ("blocked", "street_forced") and hits:
            fails.append(f"라우터 회귀 {label} (침범 {hits})")

    # 4) 격자/스냅 스모크
    cell, err = fl.snap_cell(*fl.official_cm_to_map(152, 197))
    print(f"\n[적재 볼링핀] 구석 정면 yaw={math.degrees(STORAGE_CORNER_YAW):+.0f}° "
          f"— 사용 가능 {len(STORAGE_PINS)}핀 (보관함 40x40 안에 8cm 물체 기준)")
    for i, (slot, obj) in enumerate(STORAGE_PINS, 1):
        print(f"    핀{i}: 로봇 ({fl.map_to_official_cm(*slot)[0]:5.1f},"
              f"{fl.map_to_official_cm(*slot)[1]:5.1f})cm"
              f" = 맵({slot[0]:+.3f},{slot[1]:+.3f}) → 물체 ({obj[0]:5.1f},{obj[1]:5.1f})cm")
    for slot, obj in STORAGE_PINS_DROPPED:
        print(f"    (제외) 물체 ({obj[0]:5.1f},{obj[1]:5.1f})cm — 보관함 밖")
    if not STORAGE_PINS:
        fails.append("적재 볼링핀 0개")

    print(f"\n[격자 스냅] 공식 (152,197)cm → 셀 {cell} 오차 {err * 100:.1f}cm")
    if cell != (150, 200):
        fails.append("격자 스냅")

    # 5) 실전 2세트 타깃 후보 필터 (룰북 §5/§7 — 오픽업 2배 감점, 폴백 금지)
    import types
    fake = types.SimpleNamespace(target_shape="cube", target_fruit="apple",
                                 shape_quota=4, fruit_quota=3)
    targets = build_targets(fake)
    print(f"\n[실전 타깃] --target-shape cube --target-fruit apple → {targets}")
    if targets != {"plain": 4, "apple": 3}:
        fails.append("build_targets (cube→plain alias)")
    tcells = {
        (50, 100): {"identity": "apple", "conflict": None},
        (100, 100): {"identity": "banana", "conflict": None},          # 비타깃
        (150, 100): {"identity": "plain", "conflict": None},
        (200, 100): {"identity": "plain", "conflict": "octahedron?"},  # 갈등
        (250, 100): {"identity": "apple", "conflict": None},
    }
    c0 = match_candidates(tcells, set(), set(), targets, Counter())
    ok0 = sorted(c0) == [(50, 100), (150, 100), (250, 100)]
    print(f"  후보(초기): {sorted(c0)} — 비타깃/갈등 제외 {'OK' if ok0 else 'FAIL'}")
    c1 = match_candidates(tcells, set(), set(), targets, Counter({"apple": 3}))
    ok1 = c1 == [(150, 100)]
    print(f"  후보(apple 3/3 소진): {c1} — plain만 잔존 {'OK' if ok1 else 'FAIL'}")
    c2 = match_candidates(tcells, set(), set(), targets, Counter(),
                          order="fruit-first")
    ok2 = sorted(c2) == [(50, 100), (250, 100)]
    print(f"  후보(fruit-first): {sorted(c2)} — 과일 우선 {'OK' if ok2 else 'FAIL'}")
    if not (ok0 and ok1 and ok2):
        fails.append("실전 타깃 후보 필터")

    # 5a-tie) AO 동수 갈림 = apple 확정  [2026-07-24 조작자 지시]
    #   perception/docs/grid-voting.md — conf 우열/방향과 무관하게 동수
    #   {apple,orange} 는 apple. 11:56 실기의 양극단 두 셀을 회귀로 고정한다.
    print("\n[AO 동수 규칙] apple/orange 1:1 갈림 → apple (docs/.../ao_tie_apple_rule.md)")

    def _finalize_votes(votes_by_cell):
        """cell_votes → 확정 cells. 러너 상태 최소본으로 _finalize_scan 재사용."""
        st = types.SimpleNamespace(
            cell_votes={c: list(v) for c, v in votes_by_cell.items()},
            cells={}, votes_k=1, fruit_k=1, _preshot_cells=set(),
            _preshot_cap=None, presence=Counter(), log=lambda *a, **k: None)
        E2ERunner._finalize_scan(st, partial=True)
        return st.cells

    def _v(ident, fc):   # 한 표 (과일면 conf fc)
        return {"identity": ident, "a1": "cube_like_object", "a1_conf": 0.97,
                "face_conf": fc, "cam": "top", "range_m": 1.3, "shot": 0}

    ao = _finalize_votes({
        # (150,100) 11:56: apple .94 / orange .936 — 종전 Δ<0.10 → plain(conflict)
        (150, 100): [_v("apple", 0.94), _v("orange", 0.936)],
        # (250,350) 11:56: orange .92 / apple .585 — 종전 Δ≥0.10 → orange
        (250, 350): [_v("orange", 0.92), _v("apple", 0.585)],
        # 비-AO 동수(apple/banana)는 규칙 밖 → 종전대로 conflicting_fruit 보류
        (300, 300): [_v("apple", 0.9), _v("banana", 0.9)],
        # 2:1(비동수)은 규칙 밖 → 다수 apple 그대로
        (100, 100): [_v("apple", 0.6), _v("apple", 0.55), _v("orange", 0.95)],
    })
    tie_a = (ao[(150, 100)]["identity"] == "apple"
             and not ao[(150, 100)]["conflict"] and ao[(150, 100)].get("ao_tie"))
    tie_b = (ao[(250, 350)]["identity"] == "apple"
             and not ao[(250, 350)]["conflict"] and ao[(250, 350)].get("ao_tie"))
    tie_c = (ao[(300, 300)]["conflict"] == "conflicting_fruit"
             and not ao[(300, 300)].get("ao_tie"))            # AO 아님 → 종전
    tie_d = (ao[(100, 100)]["identity"] == "apple"
             and not ao[(100, 100)].get("ao_tie"))            # 2:1 → 종전
    print(f"  (150,100) apple.94/orange.936 → {ao[(150,100)]['identity']} "
          f"{'OK' if tie_a else 'FAIL'}")
    print(f"  (250,350) orange.92/apple.585 → {ao[(250,350)]['identity']} "
          f"{'OK' if tie_b else 'FAIL'}")
    print(f"  (300,300) apple/banana 동수 → {ao[(300,300)]['identity']}"
          f"[{ao[(300,300)]['conflict']}] (규칙 밖) {'OK' if tie_c else 'FAIL'}")
    print(f"  (100,100) apple2:orange1 → {ao[(100,100)]['identity']} "
          f"(규칙 밖) {'OK' if tie_d else 'FAIL'}")
    if not (tie_a and tie_b and tie_c and tie_d):
        fails.append("AO 동수 규칙")

    # 5b) 정합성 인자 + conflict 클래스  [2026-07-24 신규]
    #     사양: mission/docs/match-strategy.md §2/§6.5
    print("\n[정합성 인자] 혼동쌍 부분계 열거 (룰북 과일당 3개)")

    def _mk(spec):
        """spec: {(x,y): identity | ('tie', {과일: 표수})} → cells dict"""
        out = {}
        for k, v in spec.items():
            if isinstance(v, tuple):
                out[k] = {"identity": "plain", "votes": 3,
                          "fruit_hits": v[1], "conflict": "conflicting_fruit"}
            else:
                out[k] = {"identity": v, "votes": 3, "fruit_hits": {},
                          "conflict": None}
        return out

    # 동점쌍 태그 추출
    tie_ok = (conflict_pair({"conflict": "conflicting_fruit",
                             "fruit_hits": {"apple": 1, "orange": 1}})
              == frozenset({"apple", "orange"}))
    tie_ok &= conflict_pair({"conflict": None, "fruit_hits": {}}) == frozenset()
    print(f"  conflict_pair 동점쌍 추출 {'OK' if tie_ok else 'FAIL'}")
    if not tie_ok:
        fails.append("conflict_pair")

    # --- 시나리오 A (조작자 제시 1): apple3 / orange2 / conflict1, 목표 apple.
    #     실제는 apple 라벨 중 1개가 orange 이고 conflict 가 apple.
    #     **핵심 회귀**: 개수만으로는 H_A(conflict=orange)와 H_B(conflict=apple)를
    #     가릴 수 없다(§6.5 구조적 축퇴) → conflict 의 p 가 0 도 1 도 아니어야 한다.
    scA = _mk({(50, 100): "apple", (100, 100): "apple", (150, 100): "apple",
               (50, 150): "orange", (100, 150): "orange",
               (200, 200): ("tie", {"apple": 1, "orange": 1}),
               (50, 200): "banana", (100, 200): "banana", (150, 200): "banana",
               (50, 250): "pineapple", (100, 250): "pineapple",
               (150, 250): "pineapple"})
    pA, noteA = consistency_probs(scA, "apple")
    pa, pc, po = pA[(50, 100)], pA[(200, 200)], pA[(50, 150)]
    print(f"  A) {noteA}")
    print(f"     apple라벨 p={pa:.3f} / conflict p={pc:.3f} / orange라벨 p={po:.3f}")
    okA = (abs(pa - 0.800) < 0.01 and abs(pc - 0.379) < 0.01
           and abs(po - 0.111) < 0.01 and abs(sum(pA.values()) - 3.0) < 1e-9)
    print(f"     축퇴 유지(0<p<1) + 합=3 + 라벨>conflict>파트너 "
          f"{'OK' if okA else 'FAIL'}")
    if not okA:
        fails.append("정합성 시나리오A")

    # --- 시나리오 B (조작자 제시 2): apple4 / orange1 / conflict1, 목표 apple.
    #     apple 4 > 3 = 정합성 게이트 이탈이지만 소진 논리는 그대로 작동한다.
    scB = _mk({(50, 100): "apple", (100, 100): "apple", (150, 100): "apple",
               (200, 100): "apple", (50, 150): "orange",
               (200, 200): ("tie", {"apple": 1, "orange": 1}),
               (50, 200): "banana", (100, 200): "banana", (150, 200): "banana",
               (50, 250): "pineapple", (100, 250): "pineapple",
               (150, 250): "pineapple"})
    pB, noteB = consistency_probs(scB, "apple")
    print(f"  B) {noteB}")
    print(f"     apple라벨 p={pB[(50, 100)]:.3f} / conflict p={pB[(200, 200)]:.3f}"
          f" / orange라벨 p={pB[(50, 150)]:.3f}")
    okB = (abs(pB[(50, 100)] - 0.666) < 0.01
           and abs(pB[(200, 200)] - 0.263) < 0.01
           and pB[(50, 100)] > pB[(200, 200)] > pB[(50, 150)])
    print(f"     apple라벨 > conflict > orange라벨 {'OK' if okB else 'FAIL'}")
    if not okB:
        fails.append("정합성 시나리오B")

    # --- 소진(exhaustion): B 에서 apple 2개 적재 + 나머지 apple라벨 2개가 근접에서
    #     orange 로 확정 → 남은 자유셀은 orange라벨 1 + conflict 1, 잔여 apple 1.
    #     conflict 가 유력해지고(0.8), orange 라벨까지 확정되면 p=1 로 수렴한다.
    ev = {(150, 100): "orange", (200, 100): "orange"}
    pB2, note2 = consistency_probs(scB, "apple", placed=2,
                                   collected={(50, 100), (100, 100)}, evidence=ev)
    print(f"  B-소진1) {note2}")
    print(f"     conflict p={pB2[(200, 200)]:.3f} / orange라벨 p={pB2[(50, 150)]:.3f}")
    ev2 = dict(ev)
    ev2[(50, 150)] = "orange"     # 튜플 키라 dict(**) 불가
    pB3, note3 = consistency_probs(scB, "apple", placed=2,
                                   collected={(50, 100), (100, 100)}, evidence=ev2)
    print(f"  B-소진2) orange 라벨까지 확정 → conflict p={pB3[(200, 200)]:.3f}")
    okE = (abs(pB2[(200, 200)] - 0.8) < 0.01
           and abs(pB3[(200, 200)] - 1.0) < 1e-9)
    print(f"     소진 갱신으로 p→1 수렴 {'OK' if okE else 'FAIL'}")
    if not okE:
        fails.append("정합성 소진 갱신")

    # --- p=1 은 오직 소진에서만. 개수-사전만으로는 절대 1 이 되지 않는다.
    no_one = all(p < 0.999 for p in pA.values()) and all(p < 0.999 for p in pB.values())
    print(f"  개수-사전 단독으로 p=1 없음(축퇴 존중) {'OK' if no_one else 'FAIL'}")
    if not no_one:
        fails.append("정합성 축퇴 위반")

    # --- 형상 타깃은 혼동쌍 없음 / quota 소진 시 빈 결과
    ps, _ = consistency_probs(scA, "plain")
    pq, _ = consistency_probs(scA, "apple", placed=3)
    if ps or pq:
        fails.append("정합성 형상·quota 처리")
    print(f"  형상 타깃/quota 소진 → 빈 결과 {'OK' if not (ps or pq) else 'FAIL'}")

    # --- 시간 모델: 보관함 최근접이 먼저(SJF)여야 한다 (§2 분모 = 전체 사이클)
    near_st = fl.official_cm_to_map(100, 100)
    far_st = fl.official_cm_to_map(300, 300)
    pose0 = (0.0, 0.0)
    t_near = cf_time_exp(near_st, pose0, 1.0)[0]
    t_far = cf_time_exp(far_st, pose0, 1.0)[0]
    sjf_ok = t_near < t_far
    print(f"  T_exp 보관함근접 {t_near:.1f}s < 원거리 {t_far:.1f}s "
          f"{'OK' if sjf_ok else 'FAIL'}")
    # 불확실 셀은 T_exp 가 T_full 보다 작아야 한다(실패 시 운반 안 함)
    t_h, t_full_h, t_probe_h = cf_time_exp(near_st, pose0, 0.5)
    part_ok = t_probe_h < t_h < t_full_h
    print(f"  T_probe {t_probe_h:.1f} < T_exp(p=.5) {t_h:.1f} < T_full "
          f"{t_full_h:.1f} {'OK' if part_ok else 'FAIL'}")
    if not (sjf_ok and part_ok):
        fails.append("정합성 시간 모델")

    # --- probe_p 로 후보 필터가 열리는가 (§7-2). 기본(None)은 종전과 동일해야 한다.
    cprobe = match_candidates(tcells, set(), set(), targets, Counter(),
                              probe_p={(200, 100): 0.4})
    open_ok = (200, 100) in cprobe and sorted(c0) == sorted(
        match_candidates(tcells, set(), set(), targets, Counter()))
    print(f"  probe_p 로 conflict 셀 개방 / 기본은 불변 "
          f"{'OK' if open_ok else 'FAIL'}")
    if not open_ok:
        fails.append("probe_p 후보 개방")

    # 5c) 비대칭 분할 1/5·2/4  [2026-07-24 조작자 지시 — 회귀 고정]
    #   mission/docs/match-strategy.md §3.5.
    #   룰북 3/3 이 스캔에서 apple k / orange (6-k) 로 기울어도 열거가 자동 흡수한다:
    #   partner(orange) 셀이 apple 후보로 열리고, 분할이 기울수록 그 p 가 커진다.
    #   **케이스 추가 코드 없이** 성립함을 여기서 고정한다(과거 회귀 방지).
    print("\n[비대칭 분할] apple k / orange (6-k) 자동 흡수 (§3.5)")

    def _split(n_apple, n_orange):
        sc = {}
        for i in range(n_apple):
            sc[(50 + i * 10, 100)] = {"identity": "apple", "votes": 3,
                                      "fruit_hits": {"apple": 3}, "conflict": None}
        for i in range(n_orange):
            sc[(50 + i * 10, 200)] = {"identity": "orange", "votes": 3,
                                      "fruit_hits": {"orange": 3}, "conflict": None}
        return sc

    # 분할이 기울수록 orange 라벨 셀의 apple 확률이 단조 증가 (0.132<0.291<0.412),
    # apple 라벨은 반대로 상승(0.868<0.918<0.941), 합은 항상 잔여 need(=3).
    split_exp = {(3, 3): (0.868, 0.132), (2, 4): (0.918, 0.291),
                 (1, 5): (0.941, 0.412)}
    split_ok = True
    prev_org = -1.0
    for (na, no), (ea, eo) in split_exp.items():
        sc = _split(na, no)
        ps, note = consistency_probs(sc, "apple")
        pa = ps[(50, 100)]
        po = ps[(50 + 0 * 10, 200)]
        s = sum(ps.values())
        ok = (abs(pa - ea) < 0.01 and abs(po - eo) < 0.01
              and abs(s - 3.0) < 1e-9 and po > prev_org
              and po >= CF_P_MIN)          # partner 셀이 실제 probe 후보로 열림
        split_ok &= ok
        prev_org = po
        print(f"  apple{na}/orange{no}: apple p={pa:.3f} / orange p={po:.3f} "
              f"(합 {s:.2f}) {'OK' if ok else 'FAIL'}")
        print(f"      {note}")
    print(f"  분할 단조성 + partner probe 후보 개방 {'OK' if split_ok else 'FAIL'}")
    if not split_ok:
        fails.append("비대칭 분할 1/5·2/4")

    # 6) street 주행 통합 (2026-07-22 — navigation/docs/control-and-routing.md)
    print(f"\n[street 통합] 기본 nav={getattr(args, 'nav', '?')}")
    if getattr(args, "nav", None) not in ("street", "legacy"):
        fails.append("--nav 인자 부재")
    mg = snv.mini_goal_for((200, 150))
    print(f"  mini_goal_for((200,150)) = ({mg[0]:+.2f},{mg[1]:+.2f}) "
          f"(기대 +0.25,-0.75)")
    if abs(mg[0] - 0.25) > 1e-9 or abs(mg[1] + 0.75) > 1e-9:
        fails.append("mini_goal_for")
    # street x 스냅: 모든 mini-goal x 는 자기 자신으로, 중간값은 최근접으로
    import types as _t
    _r = E2ERunner.street_x_snap
    _self = _t.SimpleNamespace()      # street_x_snap 은 self 미사용
    snap_ok = all(abs(_r(_self, -1.25 + 0.5 * k) - (-1.25 + 0.5 * k)) < 1e-9
                  for k in range(7))
    snap_ok &= _r(_self, 0.30) == 0.25 and _r(_self, 2.5) == 1.75 \
        and _r(_self, -1.9) == -1.25
    print(f"  street_x_snap 격자/클램프 {'OK' if snap_ok else 'FAIL'}")
    if not snap_ok:
        fails.append("street_x_snap")
    # 42셀 mini-goal 전수: street x 격자 위 + 아레나 내부 + 적재함 비겹침
    bad_mg = []
    for cx in fl.GRID_XS_CM:
        for cy in fl.GRID_YS_CM:
            x, y = snv.mini_goal_for((cx, cy))
            on_street = abs(_r(_self, x) - x) < 1e-9
            inside = -1.97 < x < 1.97 and -1.97 < y < 1.97
            if not (on_street and inside and x >= -1.25):
                bad_mg.append(((cx, cy), round(x, 2), round(y, 2)))
    print(f"  42셀 mini-goal 전수: 위반 {len(bad_mg)}건")
    if bad_mg:
        fails.append(f"mini-goal 전수 {bad_mg[:3]}")

    # 6-1) 하산 파지 판정 전수 [2026-07-24 신규]
    # 이 테스트가 없어서 7/23 하루(실기 21런) 동안 하산이 **0회 발동**한 것을
    # 아무도 못 잡았다. 스캔점/적재함 두 pose 에서 42셀 전수 판정을 고정한다.
    scan_pose = (fl.CENTER_SCAN_XY[0], fl.CENTER_SCAN_XY[1], math.pi / 2.0)
    place_pose = (-1.45, -1.45, math.radians(-135.0))   # 적재 후퇴 종점 실측
    got_scan = {c for c in ((cx, cy) for cx in fl.GRID_XS_CM
                            for cy in fl.GRID_YS_CM)
                if descend_candidate(c, scan_pose)[1]}
    # 기대: x=250 열에서 DESCEND_SKIP_ROWS 를 뺀 전부. 그 외 열은 하나도 없다.
    want_scan = {(250, cy) for cy in fl.GRID_YS_CM
                 if cy not in DESCEND_SKIP_ROWS}
    print(f"  하산 후보(스캔점) {len(got_scan)}셀 "
          f"{sorted(got_scan)} (기대 {len(want_scan)}셀)")
    if got_scan != want_scan:
        fails.append(f"하산 후보 집합 {sorted(got_scan ^ want_scan)}")
    # 미러 플래그: x=250 후보는 전부 미러여야 한다 (기본 대각은 2026-07-24 폐기)
    if any(not descend_candidate(c, scan_pose)[0] for c in got_scan):
        fails.append("하산 후보에 비미러 포함")
    # 적재 후 pose 에서는 **하나도** 걸리면 안 된다 — 걸리면 go_to_mini_goal 의
    # 복귀 드리프트를 건너뛰어 제자리 135° 회전이 되살아난다.
    got_place = [c for c in ((cx, cy) for cx in fl.GRID_XS_CM
                             for cy in fl.GRID_YS_CM)
                 if descend_candidate(c, place_pose)[1]]
    print(f"  하산 후보(적재 후 {place_pose[0]:+.2f},{place_pose[1]:+.2f}) "
          f"{len(got_place)}셀 (기대 0셀)")
    if got_place:
        fails.append(f"적재 후 하산 오발동 {got_place[:3]}")
    # 제외 행은 스캔점에서도 반드시 거짓
    if any(descend_candidate((250, cy), scan_pose)[1]
           for cy in DESCEND_SKIP_ROWS):
        fails.append("DESCEND_SKIP_ROWS 미적용")
    # 정렬: **무조건 가장 북쪽 먼저** [2026-07-24 조작자 지시]. pose 무관.
    want_ord = [(250, 350), (250, 300), (250, 250), (250, 200), (250, 150)]
    for tag, p in (("정위치", scan_pose),
                   ("남쪽 15cm 어긋남", (0.25, 0.10, math.pi / 2.0)),
                   ("북쪽 10cm 어긋남", (0.25, 0.35, math.pi / 2.0))):
        got_ord = sorted(reversed(want_ord),   # 뒤섞어 넣어 정렬이 실제로 도는지
                         key=lambda c: descend_sort_key(c, p))
        print(f"  하산 정렬({tag}): {[c[1] for c in got_ord]} "
              f"(기대 {[c[1] for c in want_ord]})")
        if got_ord != want_ord:
            fails.append(f"하산 정렬({tag}) {[c[1] for c in got_ord]}")

    # =====================================================================
    # 프로필 정합성 — 아레나 상수가 서로 어긋나지 않았는지 순수 등식으로 검산
    # =====================================================================
    # 아레나 크기를 바꾸면 격자·구역·street·하이웨이가 **함께** 따라와야 한다.
    # 하나만 놓치면 런너와 노드가 서로 다른 아레나를 믿는 조용한 실패가 되고,
    # 그건 주행을 해 봐야 드러난다. 여기서 로봇 없이 잡는다.
    H = fl.ARENA_HALF_M
    ocm = fl.official_cm_to_map
    side = H * 200.0                      # 아레나 한 변 [cm]
    print(f"\n[프로필 정합성] ARENA_HALF_M={H} (한 변 {side:.0f}cm), "
          f"격자 {len(fl.GRID_XS_CM)}x{len(fl.GRID_YS_CM)}"
          f"={len(fl.GRID_XS_CM) * len(fl.GRID_YS_CM)}칸, 피치 {fl.GRID_PITCH_CM}cm")

    def chk(tag, cond, detail=""):
        print(f"  {'OK ' if cond else '✗  '} {tag}" + (f" — {detail}" if detail else ""))
        if not cond:
            fails.append(f"정합성:{tag}")

    # P1 좌표 변환 왕복 항등 (전 격자셀)
    rt = all(abs(fl.map_to_official_cm(*ocm(x, y))[0] - x) < 1e-9
             and abs(fl.map_to_official_cm(*ocm(x, y))[1] - y) < 1e-9
             for x in fl.GRID_XS_CM for y in fl.GRID_YS_CM)
    chk("P1 좌표 왕복 항등", ocm(0, 0) == (-H, -H) and rt,
        f"ocm(0,0)={ocm(0, 0)}")
    # P2 적재함 = 좌하단 모서리의 40cm 정사각
    chk("P2 적재함 사각", fl.STORAGE_RECT_MAP ==
        (*ocm(0, 0), *ocm(fl.STORAGE_BOX_CM, fl.STORAGE_BOX_CM)),
        f"{tuple(round(v, 3) for v in fl.STORAGE_RECT_MAP)}")
    # P3 출발 포즈가 아레나 안, 벽에서 INSET 만큼
    sx, sy = fl.START_POSE[0], fl.START_POSE[1]
    chk("P3 출발 포즈", abs(abs(sx) - (H - fl.START_INSET_CM / 100.0)) < 1e-9
        and abs(abs(sy) - (H - fl.START_INSET_CM / 100.0)) < 1e-9,
        f"({sx:+.2f},{sy:+.2f}) 벽에서 {fl.START_INSET_CM:.0f}cm")
    # P4 모서리 기준 좌표는 아레나 크기와 무관 — 공식 cm 로 되돌려 확인.
    #    왕복에 부동소수 잔차가 남으므로(30 -> -1.7 -> 30.000000000000004)
    #    P1 과 같은 허용오차로 본다. 1e-6 cm = 10nm.
    def cm_eq(got, want):
        return all(abs(g - w) < 1e-6 for g, w in zip(got, want))

    chk("P4 모서리 기준 좌표",
        cm_eq(fl.map_to_official_cm(*STAGING_XY), (60.0, 60.0))
        and cm_eq(fl.map_to_official_cm(*CF_STORAGE_XY), (20.0, 20.0))
        and bool(STORAGE_PINS)
        and cm_eq(fl.map_to_official_cm(*STORAGE_PINS[0][0]), PIN1_CM),
        f"스테이징(60,60) 보관함중심(20,20) 핀1{PIN1_CM}")
    # P5 하이웨이 두 상수는 공식 cm 가 서로 다르다 — 같게 맞추면 안 된다
    chk("P5 하이웨이 상수", abs(HIGHWAY_Y_M - ocm(0, 75)[1]) < 1e-9
        and abs(HIGHWAY_FREE_Y_M - ocm(0, 60)[1]) < 1e-9
        and abs(snv.HIGHWAY_Y_M - HIGHWAY_FREE_Y_M) < 1e-9,
        f"러너 공식75={HIGHWAY_Y_M:+.2f} / 자유밴드·주행선 공식60={HIGHWAY_FREE_Y_M:+.2f}")
    # P6 street 중앙선이 공식 25+50k / 75+50k 집합과 일치
    want_xs = tuple(ocm(x, 0)[0] for x in range(25, int(side) - 24, fl.GRID_PITCH_CM))
    want_ys = tuple(ocm(0, y)[1] for y in range(75, int(side) - 24, fl.GRID_PITCH_CM))
    chk("P6 street 중앙선", STREET_XS_M == want_xs and STREET_YS_M == want_ys,
        f"x{len(STREET_XS_M)}개 y{len(STREET_YS_M)}개")
    # P7 클램프와 mini-goal 이 전부 아레나 안
    mg = [snv.mini_goal_for((x, y)) for x in fl.GRID_XS_CM for y in fl.GRID_YS_CM]
    chk("P7 아레나 클램프", abs(ARENA_CLAMP_M - (H - 0.1)) < 1e-9
        and all(abs(g[0]) < H - 0.03 and abs(g[1]) < H - 0.03 for g in mg),
        f"clamp={ARENA_CLAMP_M:.2f} mini-goal {len(mg)}개 전부 안쪽")
    # P8 navigation 패키지의 독립 사본과 대조 (일부러 합치지 않은 값들)
    try:
        sys.path.insert(0, str(REPO_ROOT / "navigation" / "ros2"
                               / "arena_lightweight_control"))
        from arena_lightweight_control import competition_layout as _cl
        same = (_cl.ARENA_HALF_M == H
                and tuple(_cl.GRID_XS_CM) == tuple(fl.GRID_XS_CM)
                and tuple(_cl.GRID_YS_CM) == tuple(fl.GRID_YS_CM)
                and tuple(_cl.STORAGE_RECT_MAP) == tuple(fl.STORAGE_RECT_MAP))
        chk("P8 competition_layout 대조", same,
            f"half={_cl.ARENA_HALF_M} 격자 {len(_cl.GRID_XS_CM)}x{len(_cl.GRID_YS_CM)}")
    except ImportError as exc:
        chk("P8 competition_layout 대조", False, f"import 실패: {exc}")
    # P9 프로세스 간: 런치 기본 맵이 이 상수와 같은 아레나인가
    #    (노드를 띄우지 않고 맵 파일만 읽는다 — 로봇도 ROS 도 필요 없다)
    try:
        from arena_lightweight_control.map_localization import OccupancyMap
        maps = (REPO_ROOT / "navigation" / "ros2" / "arena_lightweight_control"
                / "maps")
        b = OccupancyMap.from_yaml(maps / DEFAULT_MAP_YAML_NAME).inner_wall_bounds()
        got_half = max(abs(b.xmin), abs(b.xmax), abs(b.ymin), abs(b.ymax))
        chk("P9 맵 대조", abs(got_half - H) <= 0.06,
            f"{DEFAULT_MAP_YAML_NAME} half={got_half:.2f} vs 상수 {H:.2f}")
    except Exception as exc:                       # noqa: BLE001 - 진단용
        chk("P9 맵 대조", False, f"{type(exc).__name__}: {exc}")

    print("\n인자 요약: " + json.dumps(
        {k: v for k, v in vars(args).items() if v not in (None, False, "")},
        ensure_ascii=False))
    if fails:
        print(f"\n자가테스트 실패: {fails}")
        return 1
    print("\n자가테스트 전체 통과 — 실기 연결 시 --offline 제거 후 실행.")
    return 0


# =========================================================================
def load_map_json(path: Path):
    data = json.loads(Path(path).read_text())
    cells = {}
    for e in data.get("cells", []):
        cells[tuple(int(v) for v in e["cell"])] = {
            "identity": e.get("identity"), "votes": int(e.get("votes", 3)),
            "fruit_hits": e.get("fruit_hits", {}), "conflict": e.get("conflict")}
    presence = Counter({tuple(int(v) for v in e["cell"]): int(e.get("count", 1))
                        for e in data.get("presence", [])})
    return cells, presence


def main() -> int:
    # [2026-07-23 — 조작자 승인] 로컬라이저 굶주림 방지. YOLO 추론 스레드가
    # arena_control_node 의 스캔매칭과 코어를 다투면 pose 가 7.4Hz 로 반토막
    # 나고 0.3s 동결이 생겨, street_nav 가 남하 레그의 25%를 정지한다
    # (02:14 실기 84틱 trace). 상위 2코어를 로컬라이저 몫으로 비워둔다.
    # ⚠ argparse 보다도 먼저 — 이후 생성되는 모든 스레드가 이 마스크를
    #   상속해야 한다 (fieldlib.apply_cpu_affinity 주석 참조).
    # 끄려면 E2E_CPU_AFFINITY=off, 수동 지정은 E2E_CPU_AFFINITY=0,1,2,3.
    fl.apply_cpu_affinity("E2E_CPU_AFFINITY", "yolo")

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-text", default="", help='GT 배치 "cls:x,y;..." (공식좌표 cm)')
    p.add_argument("--gt-file", default="", help="GT 배치 텍스트 파일")
    p.add_argument("--yes", action="store_true", help="조작자 체크리스트/프롬프트 생략")
    # [폐기 2026-07-23 — 조작자 지시] --features 실험 토글 제거. 구성 고정:
    # 격자 스냅 검증·남단 행 운반 드리프트는 상시, 첫측정 게이트는 미채택.
    # 구 인자를 실수로 넘겨도 죽지 않게 무시 인자로만 남긴다.
    p.add_argument("--features", default="", help=argparse.SUPPRESS)
    p.add_argument("--launch-stack", action="store_true",
                   help="BridgeManager로 arena 스택 자동 기동")
    p.add_argument("--max-objects", type=int, default=None,
                   help="수거 사이클 상한 (기본: 실전 타깃 모드 quota합+5, 그 외 3)")
    p.add_argument("--target-class", default=None,
                   choices=sorted(fl.CLASSES), help="이 클래스 확정 셀 우선 수거 "
                   "(검증용 — 타깃 없으면 최근접 폴백. 실전은 --target-shape/fruit)")
    p.add_argument("--target-shape", default=None,
                   choices=("cube", "plain", "octahedron", "dodecahedron",
                            "icosahedron"),
                   help="실전 세트1 목표 형상 (당일 오전 공지, 10점x4). "
                        "cube=plain (과일면 없는 큐브 identity)")
    p.add_argument("--target-fruit", default=None, choices=sorted(fl.FRUITS),
                   help="실전 세트2 목표 과일 (경기 직전 공지, 20점x3)")
    p.add_argument("--shape-quota", type=int, default=4,
                   help="세트1 목표 형상 경기장 내 수량 (룰북 기본 4)")
    p.add_argument("--fruit-quota", type=int, default=3,
                   help="세트2 목표 과일 경기장 내 수량 (룰북 기본 3)")
    p.add_argument("--order", choices=("nearest", "fruit-first", "shape-first"),
                   default="nearest",
                   help="실전 모드 수거 순서 — nearest=최근접 / fruit-first=과일"
                        "(20점) 세트 우선 / shape-first=형상 세트 우선")
    # [2026-07-24 신규] 정합성 인자 + conflict 복구.
    # 사양: mission/docs/match-strategy.md §2/§3/§6.5
    p.add_argument("--consistency", choices=("off", "on"), default="off",
                   help="정합성 인자(p·V/T_exp) 대상 선정 + conflict 클래스 개수-사전 "
                        "복구. on 이면 --order 선호와 하산 무조건우선을 rate 랭킹이 "
                        "대체하고 conflict/혼동쌍 파트너 셀이 probe 후보로 열린다 "
                        "(파지는 strict verify 통과 시에만). 기본 off = 종전 동작. "
                        "⚠ apple/orange 근접 판별력이 선행조건(사양 §7-4)")
    # [2026-07-24 조작자 지시] 기본값 3.0 → 0 (끔). 실패 분석용 상시 기록은
    # 코어 0.21개 + 3.2MB/s 를 계속 먹으므로 실전 기본에서 뺀다. 분석이 필요한
    # 런에서만 --frame-rec-fps 3 으로 켠다.
    # [폐기 2026-07-24 — 구 기본값 3.0. 롤백 시 default=3.0 으로 되돌린다]
    p.add_argument("--frame-rec-fps", type=float, default=0.0,
                   help="[2026-07-23] 실기 전 구간 원해상도(1920x1080) 프레임 상시 "
                        "기록 fps (캠당). **기본 0 = 끔** (2026-07-24 조작자 지시). "
                        "3 = 코어 0.21개 / 3.2MB/s / 3분 0.58GB — 실패 분석 런에서만 "
                        "켠다. 전 프레임(15)은 코어 1개를 먹으니 쓰지 말 것.")
    p.add_argument("--no-save-images", action="store_true",
                   help="[2026-07-23] 스캔 디버그 이미지 저장 생략. JPEG 전환으로 "
                        "12샷 3.8초까지 내려왔지만, 실전에서는 그마저 아끼고 "
                        "코어를 라이다/제어에 온전히 넘긴다 (report.json 은 그대로).")
    p.add_argument("--no-place", action="store_true", help="집기만 (적재 생략)")
    p.add_argument("--place-anyway", action="store_true",
                   help="[2026-07-23 저녁] empty(빈손 확정) 판정까지 무시하고 "
                        "적재를 강행한다. 기본값은 empty 만 게이트 — "
                        "unknown/무응답은 기본적으로 그냥 진행한다.")
    p.add_argument("--gt-map", action="store_true",
                   help="스캔 위치까지 이동 후 스캔 생략 — GT를 지도로 사용해 "
                        "접근/적재만 검증 (마스트 다운 유지)")
    p.add_argument("--skip-scan", action="store_true",
                   help="스캔 생략 — --map-file 지도 재사용 (마스트 다운 전제)")
    p.add_argument("--map-file", default="", help="grid_map.json 경로 (--skip-scan)")
    p.add_argument("--dry-run", action="store_true",
                   help="이동/그리퍼/리프트 명령 대신 로그만 (스캔/추론은 가능하면 수행)")
    p.add_argument("--speed-profile", choices=sorted(SPEED_PROFILES),
                   default="normal",
                   help="속도 3단 프로파일 (주행/접근/적재) — 스윕 검증용")
    # [2026-07-23] 기본값 on → ao. BP 가 정답 banana 를 pineapple 로 뒤집는 것이
    # 154504 실기 리플레이에서 확인돼 BP 라우트를 기본에서 뺀다 (PAIR_ROUTES 주석).
    # 종전 기본값 "on"(AO+BP)은 --pair on 으로 계속 사용 가능 — 폐기일 미정.
    # [2026-07-24 갱신] 기본값 "ao" → "bp". 확정 조합 = face 신규 / AO off /
    # BP 신규(new/off/new). AO 는 face 구모델의 apple→orange 오분류를 메우려
    # 존재했는데, face 재학습본이 그 오분류를 10→2건으로 줄이자 AO 는 순해로
    # 뒤집혔다(교체 개선 0/훼손 5~6, 정답 apple 을 orange 로 확신 전환). 반면
    # face 신규가 남긴 pineapple→banana 오분류는 BP 신규가 정확히 되돌린다.
    # 근거 문서 perception/docs/models.md.
    # [폐기 2026-07-24 — 구 기본값 "ao"(AO만). 롤백 시 default="ao"]
    p.add_argument("--pair", choices=("on", "off", "ao", "bp"), default="bp",
                   help="face 과일면 pair 이진 재검증 라우트 "
                        "(on=AO+BP / ao=AO만 / bp=BP만(기본) / off, 검증기 없으면 자동 off)")
    p.add_argument("--votes-k", type=int, default=None,
                   help=f"셀 확정 최소 표 (기본 {fl.CELL_VOTES_MIN})")
    p.add_argument("--fruit-k", type=int, default=None,
                   help=f"과일 확정 최소 과일표 (기본 {fl.CELL_FRUIT_K})")
    p.add_argument("--scan-shots", type=int, default=SCAN_SHOTS,
                   help=f"360°를 나눌 스캔 스텝 수 (기본 {SCAN_SHOTS}). "
                        "1이면 회전 없이 단발 촬영 — 2m 데모용")
    p.add_argument("--scan-aim-yaw", type=float, default=SCAN_AIM_YAW_DEG,
                   help="촬영 전 조준할 절대 yaw [deg]. 미지정이면 회전 없음"
                        "(경기 동작). 2m 데모는 126 — 6칸이 HFOV 69° 한 "
                        "프레임에 들어오는 방위각")
    p.add_argument("--release-ticks", type=float, default=PLACE_RELEASE_TICKS,
                   help=f"적재 투하 시 *현재 위치 기준* 상대 개방 틱 "
                        f"(기본 {PLACE_RELEASE_TICKS:.0f}tick "
                        f"≈{PLACE_RELEASE_TICKS * DXL_DEG_PER_TICK:.1f}°, "
                        "전체 OPEN은 보관함 벽 간섭)")
    p.add_argument("--mast-up", action="store_true",
                   help="--gt-map 모드에서도 마스트를 올린다 (스캔이 없어 평소엔 "
                        "다운 유지가 정상 — 리프트 하드웨어 점검용)")
    p.add_argument("--no-place-settle", action="store_true",
                   help="적재 복귀 경로의 위치 안정화 생략 "
                        "(후퇴 0.2m / goal 정착 0.2s). 기본은 안정화 수행 — "
                        "7/20 생략 시도는 롤백됨")
    p.add_argument("--scan-settle", type=float, default=SCAN_SETTLE_S,
                   help=f"스캔 회전 후 정착 대기 s (기본 {SCAN_SETTLE_S}, 모션블러 방지)")
    p.add_argument("--scan-mode", choices=("step", "continuous"), default="step",
                   help="step=정지 스텝 스캔 / continuous=연속 회전하며 촬영")
    p.add_argument("--scan-mast", choices=("up", "mid"), default="up",
                   help="스캔 마스트 높이 — mid=스트로크 절반(펌웨어 LIFT_TO_MID "
                        "지원 + perception/calibration/stitch/mid.json 캘리브 필요). "
                        "근접캠 초근접 탑뷰 블라인드존(스캔 인접 4셀 누락) 완화용")
    p.add_argument("--range-mode", choices=("depth", "ground"), default="depth",
                   help="위치 추정: depth=원본캠 depth 역투영 / "
                        "ground=RGB 바닥평면 역투영 (마스트업 캘리브 tilt/height 기반)")
    # [2026-07-23 신규 — 조작자 지시 "실기에서 실행인자로 테스트"] 적재 진입/
    # 복귀 드리프트 **전용** 회전 튜닝. mini-goal ±45° 회전(turn_to)과는
    # 완전히 별개 상수라 이 값을 바꿔도 face_object/face_north 는 불변이다.
    # 롤백 = --drift-yaw-rate 0.8 --drift-kp-yaw 1.5 (종전 주행용 값).
    p.add_argument("--drift-yaw-rate", type=float, default=snv.DRIFT_YAW_RATE,
                   help=f"적재 드리프트 yaw 상한 rad/s (기본 "
                        f"{snv.DRIFT_YAW_RATE}, 종전 0.8). ⚠ 하이웨이 0.6m/s 와 "
                        f"동시 포화 시 브릿지가 균일 축소 — 2.0 이상은 실익 급감")
    p.add_argument("--drift-kp-yaw", type=float, default=snv.DRIFT_KP_YAW,
                   help=f"적재 드리프트 yaw P 게인 1/s (기본 {snv.DRIFT_KP_YAW}, "
                        f"종전 1.5). 수렴시간 t = ln(e0/tol)/KP 라 게인이 주 지렛대")
    p.add_argument("--nav", choices=("street", "legacy"), default="street",
                   help="수거 주행 모드 [2026-07-22] — street=하이웨이/스트리트 "
                        "북향 고정 폐루프 (기본, navigation/docs/control-and-routing.md) / "
                        "legacy=코리도 라우팅+goal 추종 (롤백용, 종전 동작)")
    p.add_argument("--no-descend", action="store_true",
                   help="[2026-07-24] 하산 파지(스캔 직후 x=250 미러 mini-goal "
                        "직행) 비활성 — 현장 롤백용. 끄면 스캔 후 항상 하이웨이로 "
                        "남하하는 종전 경로만 쓴다")
    p.add_argument("--max-v", type=float, default=None,
                   help="주행 속도만 프로파일 무시하고 직접 지정 [m/s]")
    p.add_argument("--offline", action="store_true",
                   help="ROS 없이 인자·GT·라우터 자가테스트 후 종료")
    p.add_argument("--no-hud", action="store_true",
                   help="경량 HUD(match_hud_lite.py) 자동 기동 생략")
    args = p.parse_args()

    if args.target_class and (args.target_shape or args.target_fruit):
        p.error("--target-class(검증용 우선순위)와 --target-shape/--target-fruit"
                "(실전 엄격 모드)는 동시 사용 불가")
    if (args.scan_mast == "mid" and not args.skip_scan and not args.gt_map
            and not args.offline and not fl.stitch_calib_path("mid").exists()):
        p.error("--scan-mast mid 는 mid 스티치 캘리브가 필요합니다 — 마스트를 "
                "LIFT_TO_MID 로 올린 상태에서 stitch_calibrator.py --mast mid "
                f"실행 후 재시도 (경로: {fl.stitch_calib_path('mid')})")

    if args.offline:
        return offline_selftest(args)

    # YOLO 프리로드 — 프로그램 시작과 동시 (GT 입력·스택 기동·헬스체크·
    # 주행과 전부 병렬. 스캔 시점엔 보통 로드 완료 상태)
    start_model_preload(pair_on=(args.pair != "off"))

    if args.skip_scan and not args.map_file:
        p.error("--skip-scan 은 --map-file 이 필요합니다")
    pre_cells, pre_presence = ({}, Counter())
    if args.skip_scan:
        map_path = Path(args.map_file)
        if not map_path.exists():
            print(f"지도 파일 없음: {map_path}")
            return 1
        pre_cells, pre_presence = load_map_json(map_path)
        print(f"기존 지도 로드: 확정 {len(pre_cells)}셀 / presence {len(pre_presence)}셀")

    out_dir = REPO_ROOT / "logs" / "field_ops" / \
        f"{time.strftime('%Y%m%d_%H%M%S')}_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"출력 디렉토리: {out_dir}")

    # [2026-07-24] HUD 자동 브링업 — 러너와 수명을 맞춘다 (백그라운드 기동).
    # [2026-07-24 — 조작자 지시] HUD 미사용 → HUD_AUTOSTART=False 로 상시 차단.
    # 종전 조건은 `if not args.no_hud:` 였다 (HUD_AUTOSTART 를 True 로 되돌리면
    # 그대로 복원). hud_write 도 같이 막아 logs/hud/target 파일을 건드리지 않는다.
    if HUD_AUTOSTART and not args.no_hud:
        hud_write("start", 0)   # 이전 경기 잔상 제거 후 기동
        start_hud(out_dir)

    # --- STARTUP: GT 입력 ---
    # [2026-07-23 00시 — 조작자 지시] 실기의 GT 입력·스캔 정확도 판정 폐지:
    # 배치는 무작위·무기록, 디버깅은 run 후 grid_map(png/json)을 실배치와
    # 육안 대조. GT 프롬프트는 dry-run(합성 스캔용)에서만 뜨고,
    # --gt-text/--gt-file 명시 시에만 셀 매핑 비교가 살아난다.
    gt = fl.prompt_gt(interactive=bool(args.dry_run) and not args.yes,
                      gt_text=args.gt_text, gt_file=args.gt_file)
    print("\n" + fl.gt_summary_text(gt))
    # [2026-07-23] 실험 기능 선택 단계 제거 — 구성 고정 (상수 주석 참조).
    if gt:
        (out_dir / "gt.json").write_text(json.dumps(
            {"cells": [{"cell": list(c), "cls": v} for c, v in sorted(gt.items())],
             "summary": fl.gt_summary_text(gt)}, ensure_ascii=False, indent=2))

    # --- STARTUP: 스택 기동 (옵션) ---
    bm = fl.BridgeManager(out_dir)
    if args.launch_stack:
        if bm.stack_running():
            print("arena 스택 이미 기동 중 — 재기동 생략")
        else:
            print("arena 스택 기동 중... (run_lightweight_arena_control.sh)")
            bm.launch_stack(camera_depth=True)

    # --- STARTUP: ROS 노드 + 헬스체크 ---
    fn = None
    try:
        import rclpy
        rclpy.init()
        fn = fl.FieldNode("e2e_match_test")
    except Exception as e:  # noqa: BLE001
        print(f"rclpy 초기화 실패: {e}")
        if not args.dry_run:
            print("중단: ROS 환경 필요 (source install/setup.bash). "
                  "--dry-run 또는 --offline 은 계속 가능.")
            return 2
        print("[dry-run] ROS 없이 시뮬 모드로 계속")

    if fn is not None:
        killed = bm.kill_duplicate_mecanum()
        if killed:
            print(f"주의: 모터 브리지 중복 {killed}개 정리 (시리얼 스터터 방지)")
        tries = 12 if args.launch_stack else 1
        health = None
        for i in range(tries):
            health = fl.healthcheck(fn, listen_sec=8.0)
            if health["ok"]:
                break
            if i < tries - 1:
                print(f"헬스체크 결손 — 스택 기동 대기 재시도 {i + 1}/{tries - 1}")
        print("\n== 토픽 헬스체크 ==")
        print(fl.health_report_text(health))
        if not health["ok"]:
            if not args.dry_run:
                print("\n중단: 필수 토픽 결손 (exit 2). 스택/케이블/전원 확인.")
                return 2
            print("[dry-run] 결손 무시하고 계속")

    # [2026-07-22 저녁] 러너 생성·arena 속도 반영을 체크리스트 **앞**으로 —
    # y 입력을 기다리는 동안 준비를 끝내 y 직후 잔여 지연이 SEED 락(~0.3s)뿐인
    # 완전 대기 상태를 만든다 (종전엔 y 후 ros2 CLI 3~5s 가 걸렸다).
    runner = E2ERunner(args, gt, out_dir, fn)
    runner.cells = pre_cells
    runner.presence = pre_presence
    runner.apply_arena_speed()
    runner._speed_applied = True

    # --- STARTUP: 조작자 체크리스트 ---
    checks = fl.operator_checklist(fl.DEFAULT_CHECKLIST, assume_yes=args.yes)
    if not all(checks.values()):
        bad = [k for k, v in checks.items() if not v]
        print(f"\n중단: 체크리스트 미충족 {bad} — 물리 상태 정리 후 재시작 (exit 3).")
        return 3

    runner.run()

    if fn is not None:
        try:
            import rclpy
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
