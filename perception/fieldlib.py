#!/usr/bin/env python3
"""현장 무보조(no-Claude) 운용 공용 라이브러리 (2026-07-19).

field_autopilot.py / match_runner.py 가 공유하는 재료:
  - GT 배치 텍스트 입력·파싱·비교 (프로그램 시작 시 조작자가 텍스트로 입력)
  - 스티치 프레임 구성 (7/15 캘리브, 마스트 다운/업) — 추론은 항상 스티치에서
  - ROS 헬퍼 노드 (토픽 계약 집약: arena/카메라/모터/그리퍼/리프트/이동)
  - 토픽 헬스체크 / 브리지 프로세스 매니저(단일 인스턴스 가드) / 배터리 OFF 판정
  - 42격자 스냅·GT 비교 출력

계약 출처: 90/91번(스티치+depth 역매핑), field72.sh(기동 env), 78/88(preflight),
gripper_bridge_node(/lift/command 패스스루 — 직접 tty 열기 금지: DTR 리셋으로
보드 재부팅→리프트 홈 소실).
"""
from __future__ import annotations

import collections
import json
import re
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]   # perception/fieldlib.py -> repo root

# =========================================================================
# CPU 친화도 — 로컬라이저 굶주림 방지 [2026-07-23 신규, 조작자 승인]
# =========================================================================
# 근거 (02:14:35~42 실기 스캔후_남하 레그, 84틱 trace):
#   · YOLO 배치추론(a1 3.66s)과 남하가 병렬로 겹치는 동안
#     arena_control_node 의 스캔매칭 지연이 p90 222ms / max 300ms 로 팽창
#     (라이다 주기 76ms → 스캔 3~4개 연속 유실).
#   · 그 결과 pose 유효 갱신이 7.4Hz(라이다 13.1Hz의 56%)로 반토막 나고
#     0.3s 이상 동결이 3회 → street_nav 의 pose-stale 가드가 레그의 25%
#     (1.55s/6.11s)를 정지시킴.
#   · 횡오차는 그 "눈 감은" 구간마다 7.5cm 씩 계단으로 뛰었고(6.1→13.6cm),
#     그래서 복구 문턱을 10→8cm 로 내려도 효과가 없었다.
# 대책: YOLO 를 도는 프로세스를 상위 코어에서 몰아내 로컬라이저가 쓸 코어를
#       비워준다. 병렬화 이득(§5.1 남하-추론 병렬, 5~8s)은 그대로 유지된다.
CPU_RESERVED_FOR_ARENA = 2   # 로컬라이저 몫으로 비워둘 상위 코어 수
CPU_AFFINITY_MIN_CORES = 4   # 이보다 코어가 적으면 분리하지 않는다(역효과)


def cpu_affinity_plan(role: str, avail=None):
    """role 에 배정할 코어 집합을 계산한다. 분리 불가면 None.

    role="yolo"  → 상위 CPU_RESERVED_FOR_ARENA 개를 **제외한** 나머지
    role="arena" → 상위 CPU_RESERVED_FOR_ARENA 개
    Jetson Orin Nano(6코어) 기준: yolo={0,1,2,3}, arena={4,5}.
    """
    cores = sorted(os.sched_getaffinity(0)) if avail is None else sorted(avail)
    if len(cores) < CPU_AFFINITY_MIN_CORES:
        return None
    reserved = set(cores[-CPU_RESERVED_FOR_ARENA:])
    if role == "arena":
        return reserved
    if role == "yolo":
        return set(cores) - reserved
    raise ValueError(f"알 수 없는 role: {role!r}")


def apply_cpu_affinity(env_var: str, role: str, default: str = "auto", log=print):
    """이 프로세스를 코어 집합에 고정한다. **어떤 경우에도 예외를 내지 않는다.**

    env 값: ""=비활성 / "auto"=cpu_affinity_plan / "0,1,2"=명시 목록.
    ⚠ 반드시 스레드를 만들기 **전에**(main 최상단) 호출할 것 — Linux 는
      affinity 를 스레드 단위로 갖고, 이후 생성된 스레드가 이를 상속한다.
      늦게 부르면 이미 뜬 워커 스레드가 옛 마스크를 그대로 유지한다.
    반환: 적용된 코어 집합 또는 None(미적용).
    """
    raw = os.environ.get(env_var, default).strip()
    if not raw or raw.lower() in ("off", "none", "0"):
        return None
    try:
        if raw.lower() == "auto":
            cores = cpu_affinity_plan(role)
            if cores is None:
                log(f"  [CPU] {env_var}=auto — 코어 {len(os.sched_getaffinity(0))}개 "
                    f"< {CPU_AFFINITY_MIN_CORES} 이라 분리 생략")
                return None
        else:
            cores = {int(t) for t in raw.replace(" ", ",").split(",") if t != ""}
        os.sched_setaffinity(0, cores)
        applied = sorted(os.sched_getaffinity(0))
        log(f"  [CPU] {role} 프로세스 → 코어 {applied} 고정 ({env_var}={raw})")
        try:    # OpenCV 워커 과다생성 방지 — 코어 수와 맞춘다
            import cv2
            cv2.setNumThreads(len(applied))
        except Exception:
            pass
        return set(applied)
    except Exception as e:   # 권한/커널/오타 — 전부 무시하고 종전대로 진행
        log(f"  ⚠ [CPU] 친화도 설정 실패({e}) — 분리 없이 계속")
        return None

FRUITS = {"apple", "orange", "banana", "pineapple"}
POLYHEDRA = {"octahedron", "dodecahedron", "icosahedron"}
CLASSES = FRUITS | POLYHEDRA | {"plain"}

# =========================================================================
# 아레나 기하 — 좌표계 원시값
# =========================================================================
# 공식 프레임: 원점 = 좌하단 적재함 모서리, 단위 cm. 룰북과 배치 지시가 쓰는 계.
# 맵 프레임:   원점 = 아레나 중심, 단위 m. 로컬라이저·컨트롤러가 쓰는 계.
# 두 계 사이 오프셋은 이 값 하나다.
#
# 종전에는 `- 2.0` / `+ 2.0` 리터럴이 런너 경로 여러 곳에 흩어져 있었고, 그중
# `match_runner.snap_gate` 에 박힌 것은 **살아 있는 안전 게이트**라 하나만
# 놓치면 접근이 전부 오판정된다. 아레나 크기를 바꿀 일이 생기면 여기 한 곳만
# 고치고, `--offline` 의 [프로필 정합성] 이 나머지가 따라왔는지 검산한다.
#
# ⚠ navigation/ros2/arena_lightweight_control/competition_layout.py 에도 같은
#   값이 독립적으로 있다 — ament 패키지에서 이 모듈을 import 하려면 노드에
#   sys.path 해킹이 들어가므로 일부러 합치지 않았다. 어긋나면 `--offline` 이 잡는다.
# [demo/arena-2m] 2.0 -> 1.0. 경기는 4m, 이 데모 경기장은 2m 다.
ARENA_HALF_M = 1.0


def official_cm_to_map(x_cm: float, y_cm: float):
    return x_cm / 100.0 - ARENA_HALF_M, y_cm / 100.0 - ARENA_HALF_M


def map_to_official_cm(x: float, y: float):
    return (x + ARENA_HALF_M) * 100.0, (y + ARENA_HALF_M) * 100.0


# 물체 후보 격자 (공식 cm). 피치 50cm 는 바꾸지 말 것 — street 여유 11.5cm 와
# MINI_GOAL_OFFSET_M ±25cm 가 전부 이 값에서 유도된 실기 튜닝값이다.
# 아레나를 줄일 때는 피치가 아니라 **점 개수**를 줄인다.
GRID_PITCH_CM = 50
# [demo/arena-2m] 7x6=42칸 -> 3x2=6칸. **피치 50cm 는 유지한다** —
# street 여유 11.5cm 와 mini-goal ±25cm 가 전부 이 피치에서 유도된 실기
# 튜닝값이라, 피치를 줄이면 street_nav 의 게인·허용오차가 통째로 무효가 된다.
# 벽 여백(좌우상 50cm, 하단 100cm 자유밴드)도 경기와 동일하게 둔다.
GRID_XS_CM = tuple(range(50, 151, GRID_PITCH_CM))
GRID_YS_CM = tuple(range(100, 151, GRID_PITCH_CM))

# ---- 카메라 마운트 (2026-07-20 확정 — 4쌍 전부 depth 바닥평면 실측 + 줄자 교차검증) ----
# 두 카메라 모두 마스트에 동승한다 (줄자 4점: 상단 35.1→49.7, 하단 30.8→45.9,
# 델타 14.6/15.1cm = 리프트 스트로크 14.89cm). 7/18의 "하단캠 섀시 고정" 결론은
# 마스트를 올리지 않은 채 label 만 mastup 으로 찍힌 무효 측정이었다
# (logs/real_validation/mastup_calib_summary.md 정정 참조).
# 구 공칭(하단 0.268/54.0)은 7/16 재조립 이전 값 — 전방거리 ~+17% 과대(파지 -2~-5cm).
TOP_MOUNT_DOWN = dict(forward_m=0.198, height_m=0.3497, tilt_deg=18.44)   # 7/18 적합 RMS 2.4mm, 줄자 35.0/35.1
NEAR_MOUNT_DOWN = dict(forward_m=0.1874, height_m=0.3143, tilt_deg=53.20)  # 7/18 적합 RMS 0.9mm, 줄자 30.5/30.8
# 마스트 업: 상단 틸트가 +0.95° 증가(상단 브래킷 처짐) — 높이만 더하는 보정 금지.
# 하단 틸트는 업/다운 동일(53.2 대), 처짐은 상단 브래킷 국소 현상.
TOP_MOUNT_UP = dict(forward_m=0.198, height_m=0.4986, tilt_deg=19.39)     # 7/18 적합 RMS 2.7mm, 줄자 50.0/49.7
NEAR_MOUNT_UP = dict(forward_m=0.1874, height_m=0.4600, tilt_deg=53.24)   # 7/20 적합 RMS 1.7mm(인라이어 79%), 줄자 45.9
# 마스트 mid = 펌웨어 LIFT_TO_MID (LIFT_TO_TOP 목표 tick 의 정확히 절반).
# up/down 실측의 중점 보간 — 높이는 스트로크 절반이라 중점이 정확하고 tilt 는
# 마스트 휨 보간. ⚠ 현장 적합 전 추정치. depth 역투영 스캔은 tilt/forward 만
# 쓰므로 민감도 낮음 (tilt 0.5° 오차 ≈ 1m 에서 수 mm).
TOP_MOUNT_MID = dict(forward_m=0.198, height_m=0.4242, tilt_deg=18.92)
NEAR_MOUNT_MID = dict(forward_m=0.1874, height_m=0.3872, tilt_deg=53.22)
# ⚠ forward_m 은 어느 캘리브로도 측정된 적 없음(평면 피팅으로 불가) — 30cm 자 검증 잔여.
# ⚠ roll(-1.0°~-2.5°)은 pixel_to_ground 가 미모델링 — 화면 가장자리 횡오차 수 cm 가능.

# ---- 스티치 캘리브 (7/15 실측, logs/real_validation/stitch_sweep) ----
STITCH_PARAMS = {
    "down": dict(x_offset=-3, crop_bottom_from_top=0, crop_top_from_bottom=20,
                 blend_overlap=12),
    "up": dict(x_offset=8, crop_bottom_from_top=4, crop_top_from_bottom=54,
               blend_overlap=12),
}

# ---- 추론 계약 (면투표 표준 7/18 저녁 + 스티치 픽셀 크기 주의) ----
# ---- pose 시각정렬 상수 [2026-07-23 신규 — 조작자 지시] ----
# 버퍼 길이. status 최악 6Hz 에서도 ~13s, 정상 20Hz 에서 4s 를 덮는다.
# 스캔 스텝 간격(~1.3s)의 10배 이상이면 충분하고, 메모리는 80x4 float 이라 무시.
POSE_BUF_N = 80
# 프레임 시각이 최신 status 보다 미래일 때 다음 status 를 기다리는 상한.
# 6Hz(주기 0.17s)에서 한 개를 확실히 받는 값 + 여유. 샷당 최악 이만큼 늘어난다.
POSE_ALIGN_WAIT_S = 0.35
# ---- 로컬라이저 전역 재수렴 "계단" 판별 ----
# ⚠ 크기 문턱만으로는 못 가른다. 인접 status 사이 yaw 변화는 정상 회전으로도
#   2.65rad/s x 주기 0.17s = 26° 까지 나오는데, 14:02 실기의 재수렴 계단은
#   +4.0°/+8.0° 였다 — 크기로는 정상 회전 안에 완전히 묻힌다.
#   판별자는 **IMU** 다: 재수렴은 pose.yaw 만 옮기고 imu_prior.yaw_integrated
#   는 건드리지 않는다 (그래서 프레임 텔레메트리의 yaw_diff 가 계단으로 튀었다).
#   그래서 잔차 |Δyaw_loc − Δyaw_imu| 로 판정한다. IMU 는 0.17s 구간에서 실회전을
#   충실히 따라가므로 잔차는 정상적으로 ~0 이고, 계단만 남는다.
POSE_JUMP_RESID_RAD = math.radians(3.0)
# 그 구간에서 로봇이 사실상 정지했는지 (IMU 기준). 정지 중의 재수렴이면
# **보정된 쪽(나중 샘플)** 이 절대값으로 더 옳다 — 로봇은 안 움직였으므로
# 프레임 시각의 실제 자세도 보정 후 값이다. 움직였으면 시간상 가까운 쪽을 쓴다.
POSE_STATIC_RAD = math.radians(2.0)
# IMU 가 없는 구 빌드용 **약한** 백업 가드. 정상 회전과 겹치므로 큰 값만 잡는다.
POSE_JUMP_RAD = math.radians(30.0)
# 프레임 header.stamp 와 arena status stamp 의 클럭이 어긋났는지 판정하는 문턱.
# 같은 머신·같은 ROS 클럭이면 수십 ms 안쪽이어야 한다. 이걸 넘으면 정렬이
# 오히려 해로우므로 폴백한다 (예: 카메라 드라이버가 하드웨어 클럭을 쓰는 설정).
POSE_CLOCK_SANITY_S = 2.0

A1_WEIGHTS = REPO_ROOT / "perception" / "models" / "a1_objectseg" / "best.pt"
# [2026-07-24 조작자 지시] face 가중치 **환경변수 오버라이드**.
# [2026-07-24 갱신] 실사 파인튜닝본을 **기본값으로 승격**. 근거: 본선 물체
# 예선 런(11:56)+7/23 15:45 런 두 실기에 face×AO×BP 18조합 전수 리플레이 결과
# face 신규(unified_face_ft_20260724)+AO off+BP 신규 = 43/44 로 최적이고, face
# 단독 과일면 정확도도 83.3%→89.2%(apple→orange 오분류 10→2건)로 개선됐다.
# 근거 문서 perception/docs/models.md.
# 구 기본값(unified_face)으로 A/B 하려면 환경변수로만:
#   FACE_WEIGHTS=perception/models/unified_face/best.pt \
#       bash scripts/run_match_day.sh
# 어느 쪽으로 돌았는지는 report.models.face 와 실행 로그에 남는다 (A/B 필수).
# [폐기 2026-07-24 — 구 기본값 unified_face. 롤백 시 아래 한 줄로 복원]
#   _FACE_DEFAULT = REPO_ROOT / "perception" / "models" / "unified_face" / "best.pt"
_FACE_DEFAULT = (REPO_ROOT / "perception" / "models"
                 / "unified_face_ft_20260724" / "best.pt")
_face_env = os.environ.get("FACE_WEIGHTS", "").strip()
FACE_WEIGHTS = (Path(_face_env) if os.path.isabs(_face_env) else REPO_ROOT / _face_env) \
    if _face_env else _FACE_DEFAULT
A1_IMGSZ_STITCHED = 896   # 스티치 세로 ~890px 원척도 (640이면 0.72배 축소 손실)

# =========================================================================
# TensorRT 엔진 선택 — **파일명 규약**으로만 채택한다 (2026-07-24 조작자 지시)
# =========================================================================
# 엔진은 빌드 시점의 shape 프로파일 밖 입력을 받으면 그냥 실패한다. 실기 중에
# 그걸 만나면 스캔이 통째로 날아가므로, "쓸 수 있다고 이름에 적혀 있는 것만"
# 쓰고 조금이라도 조건이 어긋나면 **.pt 로 떨어진다**.
#
# 파일명 규약:  <이름>_dyn_b<최대배치>_<최소imgsz>-<최대imgsz>_<정밀도>.engine
#   · `dyn`(또는 `dynamic`) 이 없으면 후보에서 제외 — 고정 배치 엔진은
#     배치가 달라지는 우리 경로에서 언제든 터진다. 실제로 7/20 빌드본
#     `a1_yolo26s_seg_896_fp16.engine` 이 배치 1 전용이라 7장 투입 시
#     "input size (7,3,896,896) not equal to max model size (1,3,896,896)" 로
#     실패했다. 그 파일은 이 규약에 안 맞으므로 자동 배제된다.
#   · `b<N>`: 이 엔진이 감당하는 최대 배치. 요청 배치보다 작으면 제외.
#   · `<min>-<max>`: 지원 imgsz 범위. 요청 imgsz 가 범위 밖이면 제외.
#     (없으면 imgsz 검증 불가로 보고 제외 — 모르면 안 쓴다)
ENGINE_NAME_RE = re.compile(
    r"(?P<dyn>dyn|dynamic).*?_b(?P<batch>\d+)_(?P<lo>\d+)-(?P<hi>\d+)_")


def select_infer_weights(pt_path, batch: int, imgsz):
    """(경로, 사유) 반환. 조건을 만족하는 엔진이 없으면 pt_path 그대로.

    imgsz 는 int 또는 여러 값(A1 은 스티치 896 · near 512 두 가지를 쓴다).
    **전부** 지원 범위 안이어야 채택한다 — 하나라도 밖이면 실기 중 그 경로에서
    터진다.
    """
    pt_path = Path(pt_path)
    sizes = [imgsz] if isinstance(imgsz, int) else list(imgsz)
    best, why = None, "엔진 없음"
    for eng in sorted(pt_path.parent.glob("*.engine")):
        m = ENGINE_NAME_RE.search(eng.name)
        if not m:
            why = f"{eng.name}: 규약 불일치(dyn/b/imgsz범위 표기 없음)"
            continue
        if int(m.group("batch")) < batch:
            why = f"{eng.name}: 최대배치 {m.group('batch')} < 요청 {batch}"
            continue
        lo, hi = int(m.group("lo")), int(m.group("hi"))
        bad = [z for z in sizes if not (lo <= z <= hi)]
        if bad:
            why = f"{eng.name}: imgsz 범위 {lo}~{hi} 밖 (요청 {bad})"
            continue
        # 조건을 만족하는 것 중 최대배치가 가장 큰 것
        if best is None or int(m.group("batch")) > best[1]:
            best = (eng, int(m.group("batch")))
    if best is None:
        return pt_path, f".pt 사용 ({why})"
    return best[0], f"engine 사용 ({best[0].name})"

FACE_IMGSZ = 224          # 학습 크기 고정 (위반 시 conf 붕괴)
FACE_FRUIT_MIN = 0.30     # 면투표: 과일면 우선 게이트
CELL_FRUIT_CONF = 0.5     # 셀 확정: 강한 과일면
CELL_FRUIT_K = 2
PAIR_CONF = 0.80          # pair 검증기: 이 conf 이상이면 라벨 교체, 미만은 강투표 박탈
CELL_VOTES_MIN = 3
FAR_PRESENCE_M = 2.5      # 초과 관측은 presence-only 강등 (하드컷 금지)

# 파지 전진 깊이: 7/15 실측 0.115(앞판 접촉) — 60번 --fast 계약과 맞출 것.
# 현장에서 파지 실패 시 ±0.01 단위로 조정.
GRIP_FORWARD_M = 0.115

# 적재함 40cm 정사각 — 좌하단 모서리에 붙어 있어 공식 cm 값이 아레나 크기와
# 무관하다. 맵 좌표만 ARENA_HALF_M 을 따라간다.
STORAGE_BOX_CM = 40.0
STORAGE_RECT_MAP = (*official_cm_to_map(0.0, 0.0),
                    *official_cm_to_map(STORAGE_BOX_CM, STORAGE_BOX_CM))
# 출발 포즈: 동벽·남벽에서 각각 20cm 안쪽, 북향(+y). 출발구역이 우하단 40cm
# 정사각이라 그 중앙이 아니라 벽 기준으로 잡는다 — 바닥 마커로 로봇을 놓을 때
# 벽까지의 거리가 유일하게 줄자 없이 재현 가능한 양이기 때문이다.
# round() 은 부동소수 잔차 제거용이다: 380/100 - 2.0 = 1.7999999999999998 로
# 종전 리터럴 1.8 과 1 ULP 어긋난다. 1e-9 m = 1nm 라 물리적 의미는 없지만,
# 유도화가 값을 바꾸지 않았다는 걸 등호로 보일 수 있게 해 둔다.
START_INSET_CM = 20.0
_start_x, _start_y = official_cm_to_map(ARENA_HALF_M * 200.0 - START_INSET_CM,
                                        START_INSET_CM)
START_POSE = (round(_start_x, 9), round(_start_y, 9), math.pi / 2.0)
# [demo/arena-2m] 스캔 지점 = **출발 포즈 그 자리**. 경기에서는 아레나 중앙
# 근처(공식 225,225)로 이동해 12샷 스핀을 돌았지만, 2m 에서는 그 방식이 못 쓴다:
#   - 중앙에서 6칸 중 4칸이 정확히 0.354m 인데, 마스트업 상단캠 시야 하한
#     40.8°(tilt 19.39°, VFOV 42°) 기준 최소 유효거리가 0.51m 라 프레임 밖이다.
#     2026-07-21 에 이 거리에서 8샷 전부 놓친 기록이 있다(A1 conf 0.05~0.11).
#   - 출발 포즈에서는 6칸이 방위각 45.4° 스팬 안에 들어와 HFOV 69° 한 프레임에
#     전부 잡히고, 거리도 0.85~1.84m 로 경기 사전샷 검증 대역과 겹친다.
# 그래서 이동 0m·스핀 0회. 제자리 회전이 각 칸의 방위각을 바꾸지 않으므로
# 스핀은 시간만 쓴다.
CENTER_SCAN_XY = (START_POSE[0], START_POSE[1])

# 프리샷(회전 없는 북향 1샷) 지점 = 공식 (225,75). 스캔점 대각 인접 4셀 전담.
# [2026-07-21] 종전에는 레그1 종료 지점(하이웨이, 공식 약 (232,14))에서 찍었다.
#   폐기 사유: 전담 셀까지 1.82~2.31m 로 멀어 과일면 크롭이 ~70px 까지 작아졌고,
#   중간 행(200,150) 물체와의 시선 여유가 2.9cm 로 가려짐 직전이었다.
#   실측(2026-07-21 4개 런): 전담 셀 중 GT 점유는 (200,200) 하나뿐인데 4번 중
#   2번만 성공 — 1회 미검출, 1회 banana→pineapple 오인.
#   (225,75)로 올리면 1.27~1.77m / 시선 여유 6cm 로 개선된다.
# ⚠ 이 지점은 물체 (200,100)의 mini-goal 과 동일하다 — 새 주행 설계에서
#   프리샷 직후 그 셀이 목표면 이동 없이 바로 파지 가능.
PRESHOT_XY = (0.25, -1.25)


# =========================================================================
# GT 배치 텍스트
# =========================================================================
def parse_gt_text(text: str) -> dict:
    """"cls:x_cm,y_cm;cls:x,y;..." -> {(x_cm,y_cm): cls}. 검증 포함."""
    out = {}
    for tok in text.replace("\n", ";").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        cls, xy = tok.split(":")
        cls = cls.strip().lower()
        if cls == "cube":
            cls = "plain"
        if cls not in CLASSES:
            raise ValueError(f"알 수 없는 클래스 '{cls}' (허용: {sorted(CLASSES)})")
        x, y = (int(float(v)) for v in xy.split(","))
        if x not in GRID_XS_CM or y not in GRID_YS_CM:
            raise ValueError(
                f"({x},{y})는 격자점이 아님 — "
                f"{len(GRID_XS_CM) * len(GRID_YS_CM)}점 "
                f"(x∈{GRID_XS_CM[0]}..{GRID_XS_CM[-1]}, "
                f"y∈{GRID_YS_CM[0]}..{GRID_YS_CM[-1]}, {GRID_PITCH_CM} 간격)")
        out[(x, y)] = cls
    return out


def prompt_gt(interactive: bool = True, gt_text: str = "", gt_file: str = "") -> dict:
    """프로그램 시작 시 GT 배치를 텍스트로 받는다 (요구사항).
    우선순위: --gt-text > --gt-file > 대화식 입력 > 빈 GT(비교 생략)."""
    if gt_text:
        return parse_gt_text(gt_text)
    if gt_file and Path(gt_file).exists():
        return parse_gt_text(Path(gt_file).read_text())
    if not interactive:
        return {}
    print("\n== 물체 배치 GT 입력 ==")
    print('형식: "apple:150,200;plain:250,300;octahedron:100,150" (공식좌표 cm)')
    print("빈 줄 입력 = GT 없음(비교 생략). 여러 줄 입력 후 빈 줄로 종료.")
    lines = []
    while True:
        try:
            line = input("GT> ").strip()
        except EOFError:
            break
        if not line:
            break
        lines.append(line)
    if not lines:
        return {}
    return parse_gt_text(";".join(lines))


def gt_summary_text(gt: dict) -> str:
    if not gt:
        return "GT 없음 (비교 생략)"
    by = {}
    for cell, cls in sorted(gt.items()):
        by.setdefault(cls, []).append(cell)
    lines = [f"GT {len(gt)}개:"]
    for cls, cells in sorted(by.items()):
        lines.append(f"  {cls:12s} {', '.join(f'({x},{y})' for x, y in cells)}")
    return "\n".join(lines)


def compare_with_gt(cells: dict, gt: dict) -> dict:
    """cells: {(x,y): {'identity':..}} vs gt -> 요약 dict + 한국어 표 문자열."""
    detected = {c for c in cells}
    ok = [c for c in detected & set(gt) if
          cells[c]["identity"] == gt[c] or (gt[c] == "plain" and cells[c]["identity"] == "plain")]
    wrong = [{"cell": list(c), "pred": cells[c]["identity"], "true": gt[c]}
             for c in detected & set(gt) if c not in ok]
    ghosts = sorted(detected - set(gt))
    missed = sorted(set(gt) - detected)
    lines = [f"셀 매핑: 정답 {len(ok)}/{len(gt)} | 클래스오류 {len(wrong)} "
             f"| 유령 {len(ghosts)} | 누락 {len(missed)}"]
    for w in wrong:
        lines.append(f"  ✗ {tuple(w['cell'])} 예측={w['pred']} 정답={w['true']}")
    for g in ghosts:
        lines.append(f"  ? 유령 {g} ({cells[g]['identity']})")
    for m in missed:
        lines.append(f"  ! 누락 {m} ({gt[m]})")
    return {"ok": len(ok), "wrong": wrong, "ghosts": [list(g) for g in ghosts],
            "missed": [list(m) for m in missed], "text": "\n".join(lines)}


# =========================================================================
# 좌표/격자 — 원시값은 파일 앞쪽 "아레나 기하" 절에 있다
# =========================================================================
def snap_cell(map_x: float, map_y: float):
    cx, cy = map_to_official_cm(map_x, map_y)
    gx = min(GRID_XS_CM, key=lambda g: abs(g - cx))
    gy = min(GRID_YS_CM, key=lambda g: abs(g - cy))
    return (gx, gy), math.hypot(cx - gx, cy - gy) / 100.0


# =========================================================================
# 스티치
# =========================================================================
STITCH_CALIB_DIR = REPO_ROOT / "perception" / "calibration" / "stitch"


def stitch_calib_path(mast: str) -> Path:
    return STITCH_CALIB_DIR / f"{mast}.json"


def load_stitch_calib(mast: str) -> dict:
    """저장된 스티치 캘리브(JSON). 없으면 {} — 레거시 평행이동 모델로 폴백."""
    p = stitch_calib_path(mast)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except ValueError:
        return {}


# ---- 해상도 전환 (2026-07-21) ----
# 스티치 캘리브(up/down.json)는 640x480 픽셀 좌표계다. RGB 해상도를 올리면
# (예: 1920x1080) 재캘리브 없이 아래 수학으로 변환한다:
#   RealSense RGB 의 4:3(640x480) 모드는 16:9 원판(1920x1080)의 중앙
#   1440x1080 크롭을 2.25배 축소한 것 (실측 근거: fx 605 = 1380·640/1440.
#   크롭 없는 축소라면 fx=460 이어야 함). 구픽셀→신픽셀은 순수 affine S 이고,
#   S = K_new · K_ref⁻¹ (같은 렌즈의 크롭+스케일이므로 정확).
#   near→top 호모그래피는 A' = S_top · A · S_near⁻¹.
CALIB_REF_RES = (640, 480)
CALIB_REF_K = {  # 캘리브 기준(640x480)에서의 실측 intrinsics (2026-07-20 라이브)
    "top": (612.4, 612.0, 323.0, 247.6),
    "near": (605.0, 605.0, 327.7, 237.5),
}


def pixel_scale_matrix(k_ref, k_new) -> "np.ndarray":
    """K_ref 해상도의 픽셀 → K_new 해상도의 픽셀 (크롭+스케일 affine)."""
    fx0, fy0, cx0, cy0 = k_ref
    fx1, fy1, cx1, cy1 = k_new
    sx, sy = fx1 / fx0, fy1 / fy0
    return np.array([[sx, 0.0, cx1 - cx0 * sx],
                     [0.0, sy, cy1 - cy0 * sy],
                     [0.0, 0.0, 1.0]])


def crop_model_k(cam: str, w: int, h: int):
    """camera_info 가 없을 때의 폴백: 중앙크롭 모델이 예측하는 K.

    지원: 640x480(기준 그대로) 및 16:9 (1280x720, 1920x1080 등).
    실기에서는 라이브 camera_info 를 쓰는 것이 우선이다 — 이 모델은
    주점이 정확히 중앙 크롭이라는 가정을 담고 있어 수 px 오차가 있을 수 있다.
    """
    if (w, h) == CALIB_REF_RES:
        return CALIB_REF_K[cam]
    if abs(w / h - 16.0 / 9.0) > 0.01:
        raise ValueError(f"지원하지 않는 해상도 {w}x{h} — 640x480 또는 16:9 만")
    fx0, fy0, cx0, cy0 = CALIB_REF_K[cam]
    k = w / 1920.0          # 원판 1920x1080 대비 축소율
    s = 2.25 * k            # 640 픽셀 → 목표 픽셀 배율
    return (fx0 * s, fy0 * s, (cx0 * 2.25 + 240.0) * k, cy0 * 2.25 * k)


class Stitcher:
    """상/근접 캠 수직 스티치 + 스티치 픽셀 -> 원본 캠 역매핑.

    3D 역투영은 반드시 원본 캠 프레임에서 (스티치 가상 프레임 금지 계약) —
    그래서 `to_source()` 는 어떤 모델을 쓰든 **정확한 역변환**이어야 한다.

    기하 모델은 `A`: **near 픽셀 → top 프레임(아래로 연장) 픽셀** 3x3 호모그래피.
    스티치 결과의 (u,v) 는 top 프레임 좌표 (u+left, v) 이므로
      - v < seam  : top 원본 (u+left, v)
      - v >= seam : near 원본 = A⁻¹ · (u+left, v)

    레거시(평행이동 전용) 파라미터는 이 모델의 특수해다:
      A = [[1,0,x_offset], [0,1, seam - crop_top], [0,0,1]]
    따라서 캘리브 파일이 없으면 STITCH_PARAMS 로 정확히 기존 동작을 재현한다
    (`camera_stitch_node.py` 조성과 동일).

    한 점 대응만으로는 평행이동밖에 못 정하므로 시야 양 끝이 어긋난다 —
    `perception/tools/stitch_calibrator.py` 로 여러 점을 잡아 배율/회전까지
    포함한 A 를 적합시키면 `perception/calibration/stitch/<mast>.json` 에 저장되고
    이 클래스가 자동으로 읽는다.
    """

    def __init__(self, mast: str, w: int = 640, h: int = 480,
                 calib: dict | None = None, intr: dict | None = None):
        """intr: 목표 해상도의 실측 intrinsics {"top": (fx,fy,cx,cy), "near": ...}.
        (w,h) 가 캘리브 기준 해상도와 다를 때만 쓰이며, 없으면 crop_model_k 폴백."""
        self.mast = mast
        self.w, self.h = w, h
        if calib is None:
            calib = load_stitch_calib(mast)
        ref_w, ref_h = tuple(calib.get("res") or CALIB_REF_RES)

        # ① 캘리브 기준 해상도(ref)에서 A/seam/left/right 를 구성 — 기존과 동일
        p = STITCH_PARAMS.get(mast)
        if p is None:
            # 레거시 파라미터가 없는 마스트 위치(mid 등): up/down 중점 부트스트랩.
            # 캘리브레이터 초기 프리뷰용 — 정식 스캔은 <mast>.json 캘리브 필수
            # (e2e 러너가 파일 존재를 게이트).
            up, dn = STITCH_PARAMS["up"], STITCH_PARAMS["down"]
            p = {k: int(round((up[k] + dn[k]) / 2)) for k in up}
        x = int(p["x_offset"])
        left = max(0, x)
        right = min(ref_w, x + ref_w)
        seam = ref_h - min(int(p["crop_bottom_from_top"]), ref_h - 1) - int(p["blend_overlap"])

        if calib.get("A") is not None:
            A = np.asarray(calib["A"], dtype=np.float64).reshape(3, 3)
            seam = int(calib.get("seam", seam))
            left = int(calib.get("left", left))
            right = int(calib.get("right", right))
            self.calib_source = str(calib.get("note") or stitch_calib_path(mast))
        else:
            A = np.array([[1.0, 0.0, float(x)],
                          [0.0, 1.0, float(seam - int(p["crop_top_from_bottom"]))],
                          [0.0, 0.0, 1.0]])
            self.calib_source = "STITCH_PARAMS (레거시 평행이동)"
        out_h_cal = calib.get("out_h")

        # ② 목표 해상도가 다르면 재스케일: A' = S_top·A·S_near⁻¹
        if (w, h) != (ref_w, ref_h):
            # 임의 해상도 쌍 변환 (2026-07-21 일반화): 기준/목표 K 를 각각
            # 크롭 모델로 유도해 S = K_new·K_ref⁻¹. ref 가 640 이면 crop_model_k 가
            # CALIB_REF_K(실측)를 그대로 돌려주므로 기존 640→FHD 경로와 비트 동일.
            # FHD 네이티브 캘리브를 640 입력으로 돌리는 역방향도 이걸로 지원된다.
            try:
                k_ref = {c: crop_model_k(c, ref_w, ref_h) for c in ("top", "near")}
            except ValueError as e:
                raise ValueError(
                    f"캘리브 기준 해상도 {ref_w}x{ref_h} → {w}x{h} 변환 불가({e}) — 재캘리브 필요")
            k_new = intr or {c: crop_model_k(c, w, h) for c in ("top", "near")}
            s_top = pixel_scale_matrix(k_ref["top"], k_new["top"])
            s_near = pixel_scale_matrix(k_ref["near"], k_new["near"])
            A = s_top @ A @ np.linalg.inv(s_near)
            seam = int(round(s_top[1, 1] * seam + s_top[1, 2]))
            left = max(0, int(round(s_top[0, 0] * left + s_top[0, 2])))
            right = min(w, int(round(s_top[0, 0] * right + s_top[0, 2])))
            out_h_cal = None                      # 새 해상도로 재계산
            self.calib_source += f" → {w}x{h} 재스케일"

        self.A, self.seam, self.left, self.right = A, seam, left, right
        self.A_inv = np.linalg.inv(self.A)
        self._gpu_state = None   # None=미시도 / False=CPU 폴백 확정 / tensor=그리드
        self.out_h = int(out_h_cal or self._auto_out_h())
        # A 가 정수 평행이동이면 warp 없이 슬라이싱 (기존 성능/픽셀 정확도 유지)
        self._pure_shift = bool(
            abs(self.A[0, 0] - 1) < 1e-9 and abs(self.A[1, 1] - 1) < 1e-9
            and abs(self.A[0, 1]) < 1e-9 and abs(self.A[1, 0]) < 1e-9
            and abs(self.A[2, 0]) < 1e-12 and abs(self.A[2, 1]) < 1e-12
            and abs(self.A[2, 2] - 1) < 1e-9
            and abs(self.A[0, 2] - round(self.A[0, 2])) < 1e-6
            and abs(self.A[1, 2] - round(self.A[1, 2])) < 1e-6)

    def _auto_out_h(self) -> int:
        """near 이미지가 top 프레임에서 덮는 최대 행 → 스티치 높이."""
        corners = np.array([[0, 0, 1], [self.w, 0, 1],
                            [self.w, self.h, 1], [0, self.h, 1]], np.float64).T
        m = self.A @ corners
        v = m[1] / m[2]
        return max(self.seam + 1, int(math.floor(v.max())))

    def _gpu_warp(self, near_img: np.ndarray):
        """[2026-07-22 저녁] near warp 의 GPU 경로 (torch grid_sample).

        실기 스캔 배치의 stitch 2.2s 는 warp 비용(68ms/쌍)이 아니라 주행
        메인스레드와의 CPU 경합(307ms/쌍으로 3배 부풀림)이 원인 — GPU 로
        빼면 경합을 회피한다 (벤치: 49ms/쌍, CPU 대비 max 4LSB/0.015% 픽셀,
        +110MB VRAM). 그리드는 A_inv 로 1회 생성 후 상주. 어떤 이유로든
        실패하면 영구 CPU 폴백 (반환 None → 호출부 cv2 경로)."""
        if self._gpu_state is False or near_img.dtype != np.uint8 \
                or near_img.ndim != 3:
            self.gpu_warp_used = False
            return None
        try:
            import torch
            if self._gpu_state is None:
                if not torch.cuda.is_available():
                    self._gpu_state = False
                    self.gpu_warp_used = False   # [2026-07-24] 플래그 누락분
                    self.gpu_warp_error = "torch.cuda.is_available()=False"
                    print("[스티치] ⚠ GPU warp 폴백 — CUDA 사용 불가", flush=True)
                    return None
                w_out = self.right - self.left
                vv, uu = np.meshgrid(
                    np.arange(self.seam, self.out_h, dtype=np.float64),
                    np.arange(0, w_out, dtype=np.float64), indexing="ij")
                pts = (np.stack([uu + self.left, vv, np.ones_like(uu)],
                                axis=-1) @ self.A_inv.T)
                gx = (pts[..., 0] / pts[..., 2] + 0.5) / self.w * 2 - 1
                gy = (pts[..., 1] / pts[..., 2] + 0.5) / self.h * 2 - 1
                grid = torch.from_numpy(
                    np.stack([gx, gy], axis=-1)).float()[None]
                self._gpu_state = grid.to("cuda")
            nt = torch.from_numpy(np.ascontiguousarray(near_img)).to("cuda")
            nt = nt.permute(2, 0, 1)[None].float()
            warped = torch.nn.functional.grid_sample(
                nt, self._gpu_state, mode="bilinear",
                padding_mode="zeros", align_corners=False)
            out = (warped[0].permute(1, 2, 0).round().clamp(0, 255)
                   .byte().cpu().numpy())
            # [2026-07-24 수정] 플래그는 **성공한 뒤에** 세운다. 종전에는 업로드/
            # grid_sample 앞에서 True 로 놨는데, 거기서 터지면 실제로는 CPU 폴백인데
            # 플래그가 True 로 남아 로그가 거짓말을 했다. 게다가 warp_backend() 를
            # 읽는 첫 배치가 프리샷 **1샷**이라 같은 배치 안에 자기교정할 두 번째
            # 호출이 없다 — 런 전체가 CPU 인데 "GPU" 로 보고될 수 있었다.
            self.gpu_warp_used = True
            return out
        except Exception as e:  # noqa: BLE001
            self._gpu_state = False   # OOM/드라이버 등 — 이후 전부 CPU
            self.gpu_warp_used = False
            # 무음 영구 폴백이 최악이다 — 사유를 1회만 남긴다 (이후 호출은
            # 위 `_gpu_state is False` 에서 즉시 반환하므로 여기 안 온다).
            self.gpu_warp_error = f"{type(e).__name__}: {e}"
            print(f"[스티치] ⚠ GPU warp 폴백 — {self.gpu_warp_error}", flush=True)
            return None

    def stitch(self, top_img: np.ndarray, near_img: np.ndarray) -> np.ndarray:
        top_part = top_img[:self.seam, self.left:self.right]
        w_out = self.right - self.left
        n_rows = self.out_h - self.seam
        if self._pure_shift:
            dx, dy = int(round(self.A[0, 2])), int(round(self.A[1, 2]))
            # near 원본 행 = v - dy, 열 = u + left - dx
            r0, c0 = self.seam - dy, self.left - dx
            near_part = near_img[r0:r0 + n_rows, c0:c0 + w_out]
            if near_part.shape[:2] != (n_rows, w_out):   # 경계 밖이면 패딩
                pad = np.zeros((n_rows, w_out) + top_img.shape[2:], top_img.dtype)
                pad[:near_part.shape[0], :near_part.shape[1]] = near_part
                near_part = pad
        else:
            near_part = self._gpu_warp(near_img)   # GPU 우선 (2026-07-22 저녁)
            if near_part is None:
                import cv2
                # near → 스티치 좌표: 먼저 A 로 top 프레임, 그다음 좌측 크롭/행 이동
                shift = np.array([[1.0, 0.0, -float(self.left)],
                                  [0.0, 1.0, -float(self.seam)],
                                  [0.0, 0.0, 1.0]])
                near_part = cv2.warpPerspective(
                    near_img, shift @ self.A, (w_out, n_rows),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return np.vstack((top_part, near_part))

    def warp_backend(self) -> str:
        """[2026-07-24] 실기 로그에서 GPU warp 가 실제로 탔는지 확인용.

        stitch 가 15:45 실기에서 3.27s 나왔는데 GPU 경로 주석의 벤치는
        49ms/쌍이라, 폴백으로 떨어졌는지 로그로 확인할 수 있어야 한다."""
        used = getattr(self, "gpu_warp_used", None)
        if used is None:
            # _pure_shift(정수 평행이동) 면 warp 자체가 필요없어 여기로 온다 —
            # 폴백이 아니라 정상이다. 캘리브 JSON 이 없어 레거시 평행이동으로
            # 떨어진 경우도 같은 값이 되므로 캘리브 출처를 함께 표기한다.
            return f"미실행(pure_shift={getattr(self, '_pure_shift', '?')})"
        if used:
            return "GPU(grid_sample)"
        return f"CPU 폴백({getattr(self, 'gpu_warp_error', '사유미상')})"

    def to_source(self, u: float, v: float):
        """스티치 픽셀 → (캠, 원본 u, 원본 v). A 의 정확한 역변환."""
        if v < self.seam:
            return "top", u + self.left, v
        p = self.A_inv @ np.array([u + self.left, v, 1.0])
        return "near", float(p[0] / p[2]), float(p[1] / p[2])

    def from_source(self, cam: str, u: float, v: float):
        """원본 캠 픽셀 → 스티치 픽셀 (to_source 의 역 — 검증/시각화용)."""
        if cam == "top":
            return u - self.left, v
        p = self.A @ np.array([u, v, 1.0])
        return float(p[0] / p[2]) - self.left, float(p[1] / p[2])


PAIR_DIR_DEFAULT = REPO_ROOT / "perception" / "models" / "verifiers"


class PairVerifier:
    """구성 A(INTEGRATION_SPEC_v3ft): face 과일면 이진 pair(v2) 재검증.

    conf>=pair_conf → 라벨 교체 / 미만 → 그 면 강투표 자격 박탈.
    패치 계약: face 마스크 poly → minAreaRect warp 224 → 128,
    BGR→RGB→/255→ImageNet norm. 프레임당 AO/BP 각 1-forward.
    """
    ROUTE = {"apple": "AO", "orange": "AO", "banana": "BP", "pineapple": "BP"}
    CLASSES = {"AO": ("apple", "orange"), "BP": ("banana", "pineapple")}
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, dir_path=PAIR_DIR_DEFAULT):
        import onnxruntime as ort
        d = Path(dir_path)
        # [2026-07-24 조작자 지시] BP onnx **기본값을 실사 재학습본으로 승격** +
        # AO 와 동형의 BP_WEIGHTS 환경변수 오버라이드 추가.
        # 근거: 본선 물체 예선 런(11:56)+7/23 15:45 두 실기 18조합 전수 리플레이에서
        # face 신규+AO off+**BP 신규(pair_banana_pineapple_real_v1)** = 43/44 최적.
        # BP 신규는 face 신규의 pineapple↔banana 오분류(합산 5건 중 pineapple→banana
        # 4건)를 정확히 되돌린다(교체 개선 4/훼손 1). 반면 구본 v2_1 은 정답 banana 를
        # conf 1.00 으로 pineapple 로 뒤집어 두 런 합산 훼손 9건 — 절대 쓰지 말 것.
        # 두 onnx 는 입출력 동형(input[N,3,128,128]→logits[N,2]) + classes.json
        # ("banana","pineapple") 순서 동일이라 CLASSES["BP"] 매핑 그대로 유효.
        # 구본으로 A/B 하려면 환경변수로만:
        #   BP_WEIGHTS=perception/models/verifiers/\
        #       pair_banana_pineapple_v2_1/weights/best.onnx bash run_match_day.sh
        # 근거 문서 perception/docs/models.md.
        # [폐기 2026-07-24 — 구 기본값 v2_1(없으면 v2). 롤백 시 아래 3줄로 복원]
        #   bp = d / "pair_banana_pineapple_v2_1" / "weights" / "best.onnx"
        #   if not bp.exists():
        #       bp = d / "pair_banana_pineapple_v2" / "weights" / "best.onnx"
        def _release_path(default_name: str):
            """릴리스 배포는 verifiers/<이름>.onnx 로 평탄하게 싣고, 원 학습
            트리는 <이름>/weights/best.onnx 였다. 둘 다 받아준다."""
            for cand in (d / f"{default_name}.onnx",
                         d / default_name / "weights" / "best.onnx"):
                if cand.exists():
                    return cand, default_name
            raise FileNotFoundError(
                f"pair verifier weight not found: {default_name} under {d} - "
                "run scripts/fetch_models.sh")

        def _env_or_release(env_path: str, default_name: str):
            if env_path:
                q = Path(env_path) if os.path.isabs(env_path) else REPO_ROOT / env_path
                # 학습 트리 형태(<이름>/weights/best.onnx)면 조부모, 평탄형이면 stem
                return q, (q.parent.parent.name if q.name == "best.onnx" else q.stem)
            return _release_path(default_name)

        bp, self.bp_version = _env_or_release(
            os.environ.get("BP_WEIGHTS", "").strip(),
            "pair_banana_pineapple_real_v1")
        # [2026-07-24 조작자 지시] AO onnx **기본값을 실사 재학습본으로 승격**.
        # 종전 기본값 pair_apple_orange_v2(합성 위주)를 실사 학습본
        # pair_apple_orange_real_20260724 으로 교체한다. 두 onnx 는 입출력이
        # 동형(input[N,3,128,128] float → logits[N,2])이고 classes.json 도
        # ("apple","orange") 순서가 같아 CLASSES["AO"] 매핑 그대로 유효하다.
        # 환경변수 오버라이드는 그대로 유지 — 구 가중치로 되돌려 A/B 하려면:
        #   AO_WEIGHTS=perception/models/verifiers/\
        #       pair_apple_orange_v2/weights/best.onnx bash run_match_day.sh
        # 어느 AO 로 돌았는지는 self.ao_version 과 실행 로그에 남는다.
        # [폐기 2026-07-24 — 구 기본값 v2. 롤백 시 아래 한 줄로 복원]
        #     else d / "pair_apple_orange_v2" / "weights" / "best.onnx"
        ao_path, self.ao_version = _env_or_release(
            os.environ.get("AO_WEIGHTS", "").strip(),
            "pair_apple_orange_real_20260724")
        self.sess = {
            "AO": ort.InferenceSession(str(ao_path),
                                       providers=["CPUExecutionProvider"]),
            "BP": ort.InferenceSession(str(bp),
                                       providers=["CPUExecutionProvider"]),
        }
        self.input_name = {k: s.get_inputs()[0].name for k, s in self.sess.items()}

    @classmethod
    def make_patch(cls, crop_bgr, poly):
        import cv2
        rect = cv2.minAreaRect(np.asarray(poly, np.float32))
        box = cv2.boxPoints(rect).astype(np.float32)
        dst = np.array([[0, 0], [223, 0], [223, 223], [0, 223]], np.float32)
        warped = cv2.warpPerspective(
            crop_bgr, cv2.getPerspectiveTransform(box, dst), (224, 224))
        x = cv2.resize(warped, (128, 128))[..., ::-1].astype(np.float32) / 255.0
        return ((x - cls.MEAN) / cls.STD).transpose(2, 0, 1)

    def verify(self, items):
        """items=[(route, patch)] → [(label, softmax_conf)]."""
        out = [None] * len(items)
        for route in ("AO", "BP"):
            idxs = [i for i, (r, _) in enumerate(items) if r == route]
            if not idxs:
                continue
            batch = np.stack([items[i][1] for i in idxs]).astype(np.float32)
            logits = self.sess[route].run(None, {self.input_name[route]: batch})[0]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            for i, p in zip(idxs, probs):
                j = int(p.argmax())
                out[i] = (self.CLASSES[route][j], float(p[j]))
        return out


def face_vote(faces, face_fruit_min: float = FACE_FRUIT_MIN):
    """faces=[(name,conf)...] -> (identity, conf). 면투표: 과일면 우선."""
    if not faces:
        return None, None
    fruit = [(n, c) for n, c in faces if n in FRUITS and c >= face_fruit_min]
    return max(fruit, key=lambda t: t[1]) if fruit else max(faces, key=lambda t: t[1])


# =========================================================================
# ROS 헬퍼 (rclpy는 지연 import — --offline 자가진단 지원)
# =========================================================================
class FieldNode:
    """토픽 계약 집약 헬퍼. 사용 전 rclpy.init() 필요."""

    TOPICS = {
        "scan": "/laser_scan",
        "status": "/arena_lightweight/status",
        "goal": "/arena_lightweight/goal",
        "pose_seed": "/arena_lightweight/pose",
        "control": "/arena_lightweight/control",
        "move": "/base/move_relative",
        "move_result": "/base/move_result",
        "motor": "/motor/state",
        "gripper_cmd": "/gripper/command",
        "gripper_state": "/gripper/state",
        "grasp": "/gripper/grasp",
        "lift_cmd": "/lift/command",
        "lift_state": "/lift/state",
        "odom": "/chassis/odom",
        "imu": "/imu/data",
        "top_rgb": "/camera_19/rgb",
        "top_depth": "/camera_19/depth",
        "near_rgb": "/camera_54/rgb",
        "near_depth": "/camera_54/depth",
        "top_info": "/camera_19/camera_info",
        "near_info": "/camera_54/camera_info",
        # [데모] 관객용 실시간 시각화 피드. match_runner 가 발행하고
        # mission/spectator_server.py 가 구독한다. 경기 로직은 이걸 읽지 않는다 —
        # 발행 실패는 무해하며, 구독자가 없어도 비용은 String 직렬화 1회뿐.
        "match_state": "/match/state",
        # 스캔 오버레이(검출 박스가 그려진 스티치 프레임). 러너가 이미 JPEG 로
        # 인코딩해 메모리에 들고 있으므로(_pending_writes) **추가 인코딩 비용이 없다.**
        # 단발 스캔에서는 경기당 1장이다.
        "scan_overlay": "/match/scan_overlay",
    }

    def __init__(self, node_name: str):
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String

        self.node = rclpy.create_node(node_name)
        self._rclpy = rclpy
        self.last = {}      # key -> 최신 msg
        self.stamp = {}     # key -> wall 수신시각
        self.count = {}
        # [2026-07-23 신규] pose 시각정렬용 링버퍼 — (t_ros, x, y, yaw).
        # 상세는 pose_at() 주석. deque 는 append/스냅샷이 GIL 하에서 원자적이라
        # 콜백 스레드와 메인 스레드가 락 없이 공유해도 안전하다.
        self.pose_buf = collections.deque(maxlen=POSE_BUF_N)
        self._pose_buf_warned = False
        q = qos_profile_sensor_data

        def cb(key):
            def _cb(msg):
                self.last[key] = msg
                self.stamp[key] = time.monotonic()
                self.count[key] = self.count.get(key, 0) + 1
                if key == "status":
                    self._buffer_pose(msg)
            return _cb

        n = self.node
        for key, typ, qos in (
            ("status", String, 10), ("move_result", String, 10),
            ("motor", String, 10), ("gripper_state", String, 10),
            ("grasp", String, 10), ("lift_state", String, 10),
        ):
            n.create_subscription(typ, self.TOPICS[key], cb(key), qos)
        for key in ("top_rgb", "top_depth", "near_rgb", "near_depth"):
            n.create_subscription(Image, self.TOPICS[key], cb(key), q)
        for key in ("top_info", "near_info"):
            n.create_subscription(CameraInfo, self.TOPICS[key], cb(key), 10)
        n.create_subscription(Odometry, self.TOPICS["odom"], cb("odom"), q)
        from sensor_msgs.msg import Imu, LaserScan
        n.create_subscription(LaserScan, self.TOPICS["scan"], cb("scan"), q)
        n.create_subscription(Imu, self.TOPICS["imu"], cb("imu"), q)

        self.pub = {}
        # 스캔 오버레이만 CompressedImage — 나머지는 String 계약 그대로.
        from sensor_msgs.msg import CompressedImage
        self.pub["scan_overlay"] = n.create_publisher(
            CompressedImage, self.TOPICS["scan_overlay"], 1)
        self._CompressedImage = CompressedImage
        for key in ("goal", "pose_seed", "control", "move", "gripper_cmd",
                    "lift_cmd", "match_state"):
            self.pub[key] = n.create_publisher(String, self.TOPICS[key], 10)

    # ---- 기본 ----
    def spin_for(self, sec: float):
        end = time.monotonic() + sec
        while time.monotonic() < end:
            self._rclpy.spin_once(self.node, timeout_sec=0.05)

    def publish(self, key: str, payload):
        from std_msgs.msg import String
        m = String()
        m.data = payload if isinstance(payload, str) else json.dumps(payload)
        self.pub[key].publish(m)

    def fresh(self, key: str, within: float) -> bool:
        return (time.monotonic() - self.stamp.get(key, 0.0)) < within

    # ---- 상태 ----
    def arena_status(self) -> dict:
        m = self.last.get("status")
        return json.loads(m.data) if m else {}

    def pose(self):
        st = self.arena_status()
        p = st.get("pose")
        return (float(p["x"]), float(p["y"]), float(p["yaw"])) if p else None

    # ---- pose 시각정렬 [2026-07-23 신규 — 조작자 지시] ----
    # 문제: 종전 스캔 촬영은 "프레임을 기다린 뒤 **가장 최신** status 를 읽어"
    # 그 yaw 로 역투영했다. status 가 20Hz 로 오면 오차가 작지만, 실측은
    # 카메라 4스트림(FHD x15fps) 구독만으로 **6Hz·p90 0.43s·최악 3.15s** 다
    # (status_latency_probe.py 3회 실행). 회전 잔여가 조금만 있어도 그 시간만큼
    # yaw 가 틀리고, 오차는 range 를 곱해 셀 경계(25cm)를 넘는다 — 14:02 실기
    # 샷 8~12 의 등가 yaw 오차 10.4°(1.5m 에서 27.6cm)가 그것이다.
    # 해법: status 를 (시각, pose) 로 쌓아두고 **프레임 시각에 맞춰 보간**한다.
    # 발행이 6Hz 로 느려도 프레임 시각을 두 샘플 사이에 끼우기만 하면 성립한다.
    def _buffer_pose(self, msg):
        """status 콜백에서 호출 — (t_ros, x, y, yaw) 적재. 절대 예외를 내지 않는다."""
        try:
            d = json.loads(msg.data)
            p = d.get("pose")
            t = d.get("stamp")
            if p is None or t is None:
                # 구 arena 빌드(‘stamp’ 없음) — 정렬 불가. 한 번만 알린다.
                if t is None and not self._pose_buf_warned:
                    self._pose_buf_warned = True
                    print("  ⚠ [pose정렬] arena status 에 stamp 없음 — 구 빌드. "
                          "종전 '최신 pose' 방식으로 폴백한다 "
                          "(arena_lightweight_control 재빌드+재기동 필요)", flush=True)
                return
            # imu_prior.yaw_integrated 를 같이 싣는다 — 로컬라이저 전역 재수렴
            # 계단을 실회전과 가르는 유일한 판별자다 (POSE_JUMP_RESID_RAD 주석).
            # 라디안·언랩·절대기준 없음. 없으면 None (구 빌드 대비).
            yi = (d.get("imu_prior") or {}).get("yaw_integrated")
            self.pose_buf.append((float(t), float(p["x"]), float(p["y"]),
                                  float(p["yaw"]),
                                  None if yi is None else float(yi)))
        except Exception:  # noqa: BLE001 — 텔레메트리 파싱은 런을 죽이지 않는다
            return

    def frame_stamp(self):
        """정렬 기준시각 = **RGB 쌍** header.stamp 평균. (t, rgb_spread, all_spread)

        RGB 쌍만 쓰는 이유 [2026-07-23 실측 후 수정]: 처음엔 4스트림 평균을
        썼는데 실측 spread 가 **656ms** 나왔다. 스트림별 실지연은 0.10~0.14s 로
        정상이고 클럭도 같은데(측정: top_rgb 0.107 / top_depth 0.096 /
        near_rgb 0.103 / near_depth 0.096 s), `self.last[key]` 는 토픽마다
        **도착이 고르지 않은 최신 메시지**라 서로 다른 시각이 섞인 것이다.
        yaw 오차가 실제로 먹히는 곳은 **검출 박스의 방위** = 스티치에 들어가는
        top_rgb/near_rgb 두 장이므로 기준을 그 쌍으로 좁힌다. depth 는 range
        (동경 방향)만 정하므로 yaw 정렬 대상이 아니다.
        all_spread(4스트림)는 진단용으로 계속 돌려준다 — 커지면 스티치 자체의
        시간 정합이 깨진 것이라 별개 문제의 신호다.
        """
        def _st(key):
            m = self.last.get(key)
            if m is None:
                return None
            try:
                s = m.header.stamp
            except AttributeError:
                return None
            t = s.sec + s.nanosec * 1e-9
            return t if t > 0.0 else None      # 드라이버가 stamp 미기입

        rgb = [_st("top_rgb"), _st("near_rgb")]
        if any(t is None for t in rgb):
            return None
        allt = [_st(k) for k in ("top_rgb", "top_depth", "near_rgb", "near_depth")]
        allt = [t for t in allt if t is not None]
        return (sum(rgb) / 2.0, max(rgb) - min(rgb),
                (max(allt) - min(allt)) if len(allt) > 1 else 0.0)

    def pose_at(self, t_ros: float, wait_s: float = POSE_ALIGN_WAIT_S):
        """t_ros(ROS초) 시점의 pose 를 버퍼에서 보간해 돌려준다.

        반환: (pose, info) — pose 는 (x,y,yaw) 또는 None,
              info 는 {"src", "age", "gap", "spread"} 진단용.
        src: "interp"=정상 보간 / "nearest_jump"=불연속 회피로 최근접 사용 /
             "clamp_old"=버퍼보다 과거 / "wait_timeout"=미래인데 못 기다림 /
             "no_buf"=버퍼 없음(구 빌드/미수신)
        """
        # 프레임이 최신 status 보다 **미래**면 보간이 아니라 외삽이 된다.
        # 다음 status 한 개를 기다리는 것이 정답 — 6Hz 라도 최대 0.17s 다.
        deadline = time.monotonic() + max(0.0, wait_s)
        while True:
            buf = list(self.pose_buf)
            if not buf:
                return None, {"src": "no_buf"}
            if buf[-1][0] >= t_ros or time.monotonic() >= deadline:
                break
            self._rclpy.spin_once(self.node, timeout_sec=0.02)

        if t_ros <= buf[0][0]:
            t, x, y, yw = buf[0][:4]
            return (x, y, yw), {"src": "clamp_old", "age": round(t - t_ros, 3)}
        if t_ros >= buf[-1][0]:
            t, x, y, yw = buf[-1][:4]
            return (x, y, yw), {"src": "wait_timeout", "age": round(t_ros - t, 3)}

        # t_ros 를 감싸는 인접 두 샘플 (버퍼는 시간 오름차순)
        for i in range(len(buf) - 1, 0, -1):
            if buf[i - 1][0] <= t_ros <= buf[i][0]:
                ta, xa, ya, wa, ia = buf[i - 1]
                tb, xb, yb, wb, ib = buf[i]
                break
        else:
            return None, {"src": "no_bracket"}

        def _wrap(a):
            return math.atan2(math.sin(a), math.cos(a))

        dt = tb - ta
        dyaw = _wrap(wb - wa)          # 최단호
        gap = round(dt, 3)

        # ---- 불연속(로컬라이저 전역 재수렴) 가드 ----
        # 계단을 가로질러 보간하면 존재한 적 없는 중간각이 나온다.
        jump = None
        if ia is not None and ib is not None:
            resid = _wrap(dyaw - _wrap(ib - ia))
            if abs(resid) > POSE_JUMP_RESID_RAD:
                jump = {"resid_deg": round(math.degrees(resid), 1),
                        "imu_d_deg": round(math.degrees(_wrap(ib - ia)), 1)}
        elif abs(dyaw) > POSE_JUMP_RAD:
            jump = {"weak": True}      # IMU 없음 — 약한 크기 가드만
        if jump is not None:
            static = (ia is not None and ib is not None
                      and abs(_wrap(ib - ia)) < POSE_STATIC_RAD)
            if static:
                # 정지 중 재수렴 → 보정된 쪽이 절대값으로 옳다 (로봇이 안 움직였으니
                # 프레임 시각의 실제 자세도 보정 후 값이다).
                pick, src = (xb, yb, wb), "jump_corrected"
            else:
                pick, src = ((xa, ya, wa) if (t_ros - ta) <= (tb - t_ros)
                             else (xb, yb, wb)), "nearest_jump"
            return pick, {"src": src, "gap": gap, **jump}

        if dt <= 1e-6:
            return (xa, ya, wa), {"src": "interp", "gap": gap}
        f = (t_ros - ta) / dt
        return (xa + (xb - xa) * f, ya + (yb - ya) * f,
                _wrap(wa + dyaw * f)), {"src": "interp", "gap": gap}

    def grasp_state(self) -> str:
        # /gripper/grasp 페이로드 키는 "state" (gripper_bridge publish_grasp).
        # 7/20 새벽까지 "grasp_state"를 읽어 항상 "" → held여도 무조건 실패
        # 판정되던 버그. 구 페이로드 호환으로 둘 다 읽는다.
        m = self.last.get("grasp")
        if not m:
            return ""
        try:
            d = json.loads(m.data)
            return d.get("state") or d.get("grasp_state") or ""
        except (ValueError, AttributeError):
            return ""

    def lift_state(self) -> dict:
        m = self.last.get("lift_state")
        try:
            return json.loads(m.data) if m else {}
        except ValueError:
            return {}

    def odom_xy(self):
        m = self.last.get("odom")
        if not m:
            return None
        p = m.pose.pose.position
        return (float(p.x), float(p.y))

    # ---- 동작 ----
    def seed_pose(self, x: float, y: float, yaw: float):
        self.publish("pose_seed", {"x": x, "y": y, "yaw": yaw})
        # 1.0→0.3s (7/21): 락 판정은 호출자 폴링이 담당 — 여기선 발행 소화만
        self.spin_for(0.3)

    def move_relative(self, dx: float, dy: float, dyaw: float = 0.0,
                      max_v: float | None = None, timeout: float = 40.0) -> dict:
        self.last.pop("move_result", None)
        payload = {"dx": float(dx), "dy": float(dy), "dyaw": float(dyaw)}
        if max_v:
            payload["max_v"] = float(max_v)
        self.publish("move", payload)
        deadline = time.monotonic() + timeout
        while "move_result" not in self.last and time.monotonic() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
        m = self.last.get("move_result")
        return json.loads(m.data) if m else {"ok": False, "error": "timeout"}

    def goto_xy(self, gx: float, gy: float, tol: float = 0.05,
                max_iter: int = 6, max_v: float = 0.5) -> bool:
        """arena pose 피드백 반복 이동 — yaw 무관(현재 heading 기준 변환)."""
        for _ in range(max_iter):
            self.spin_for(0.3)
            pose = self.pose()
            if pose is None:
                return False
            dxw, dyw = gx - pose[0], gy - pose[1]
            if math.hypot(dxw, dyw) < tol:
                return True
            c, s = math.cos(pose[2]), math.sin(pose[2])
            fwd = dxw * c + dyw * s
            left = -dxw * s + dyw * c
            self.move_relative(fwd, left, max_v=max_v)
        pose = self.pose()
        return pose is not None and math.hypot(gx - pose[0], gy - pose[1]) < 2 * tol

    def goto_xy_goal(self, gx: float, gy: float, tol: float = 0.25,
                     timeout: float = 25.0, settle: float = 0.2) -> bool:
        """arena goal 발행 — localizer 폐루프로 실시간 보정하며 주행.

        (goto_xy의 odom 단발이동+재보정 방식과 달리 arena 노드가
        /cmd_vel_direct로 연속 제어. 속도는 real.yaml 캡을 따른다.)
        ⚠ tol은 arena xy_tolerance(0.15)보다 커야 함 — 작으면 arena가
        먼저 latch해 cmd_vel=0만 계속 내보내고 여기선 영원히 미도달
        (7/19 23:08 45s 멈춤의 원인). 근거리 goal은 발행 없이 즉시 성공.
        tol 0.16→0.25 (7/21 18:18 실기 '정착 제거'): arena 접근 감속의
        데드존 주차 반경(0.17~0.35m)이 0.16 밖이라 주차→폴백→펌웨어 정착
        그라인드(8s) 체인이 반복됐다. 마지막 십수 cm는 후속 단계(접근 시각
        측정·적재 게이트·다음 레그 주행)가 흡수한다 — 주행하며 맞추는 설계.
        """
        p = self.pose()
        if p and math.hypot(gx - p[0], gy - p[1]) < tol:
            return True
        self.publish("goal", {"x": float(gx), "y": float(gy)})
        deadline = time.monotonic() + timeout
        # 정체 감지 (7/21 17:12 실기): arena 접근 감속 명령이 펌웨어 데드존
        # 아래로 떨어지면 목표 0.17~0.35m 앞에 주차된 채 타임아웃 25s를 전소
        # (경기당 3회 = 75s+ 손실). 1.0s 동안 3cm 미만 진행이면 조기 탈출해
        # 호출자(do_goto)의 odom 폴백에 넘긴다. 발진 직후 오탐 방지를 위해
        # goal 발행 후 0.8s 그레이스 (구독/제어 지연 + 가속 램프).
        t0 = time.monotonic()
        last_p, last_t = p, t0 + 0.8
        while time.monotonic() < deadline:
            self.spin_for(0.15)
            p = self.pose()
            if p and math.hypot(gx - p[0], gy - p[1]) < tol:
                self.publish("control", "STOP")  # goal 해제
                if settle:
                    self.spin_for(settle)
                return True
            now = time.monotonic()
            if p is not None:
                if (last_p is None
                        or math.hypot(p[0] - last_p[0], p[1] - last_p[1]) > 0.03):
                    last_p, last_t = p, max(now, t0 + 0.8)
                elif now - last_t > 1.0:
                    self.publish("control", "STOP")
                    return False        # 데드존 정체 — 조기 탈출 (폴백은 호출자)
        self.publish("control", "STOP")
        return False

    def rotate_to_yaw(self, target_yaw: float, tol: float = 0.05) -> bool:
        pose = self.pose()
        if pose is None:
            return False
        err = math.atan2(math.sin(target_yaw - pose[2]), math.cos(target_yaw - pose[2]))
        if abs(err) <= tol:
            return True
        r = self.move_relative(0.0, 0.0, err)
        return bool(r.get("ok"))

    # ---- 리프트/그리퍼 (전부 브리지 경유 — tty 직접 열기 금지) ----
    def lift(self, cmd: str, settle: float = 0.0):
        self.publish("lift_cmd", cmd)
        if settle:
            self.spin_for(settle)

    def lift_wait_idle(self, timeout: float = 25.0,
                       assume_done_s: float | None = None) -> tuple:
        """LIFT 이동 완료 대기. (완료?, 근거) 를 돌려준다.

        완료 판정 = ``MOVING==0`` 관측  **OR**  ``assume_done_s`` 경과.
        7/20: STATUS 폴링 기반 판정이 실측(상승 ~7s / 하강 ~6s)보다 과대평가돼
        마스트가 이미 다 올라갔는데도 계속 기다리던 문제 — 시간 기반 상한을
        OR 로 걸어 둘 중 먼저 성립하는 쪽으로 끊는다. ``assume_done_s=None``
        이면 기존 MOVING 폴링 단독 동작.
        """
        t0 = time.monotonic()
        deadline = t0 + timeout
        moved = False
        while time.monotonic() < deadline:
            self.publish("gripper_cmd", "LIFT_STATUS?")
            self.spin_for(0.5)
            el = time.monotonic() - t0
            st = self.lift_state()
            if st.get("moving") == 1:
                moved = True
            if moved and st.get("moving") == 0:
                return True, f"MOVING=0 ({el:.1f}s)"
            if st and not moved and st.get("moving") == 0 and el > 4.0:
                return True, f"idle 유지 ({el:.1f}s)"   # 이동 관측 없이 idle → 이미 도착
            if assume_done_s is not None and el >= assume_done_s:
                return True, f"시간 기반 {assume_done_s:.1f}s 경과 ({el:.1f}s)"
        return False, f"타임아웃 {timeout:.1f}s"

    def gripper_status(self) -> dict:
        """/gripper/state JSON (present_deg / present_raw / open_deg / ...)."""
        m = self.last.get("gripper_state")
        try:
            return json.loads(m.data) if m else {}
        except ValueError:
            return {}

    def gripper(self, cmd: str):
        self.publish("gripper_cmd", cmd)

    # ---- 프레임 ----
    def rgb_pair(self):
        """(top_rgb, near_rgb) numpy RGB. 없으면 None."""
        out = []
        for key in ("top_rgb", "near_rgb"):
            m = self.last.get(key)
            if m is None:
                return None
            ch = 4 if m.encoding == "rgba8" else 3
            a = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, ch)[:, :, :3]
            out.append(a[:, :, ::-1].copy() if m.encoding == "bgr8" else a.copy())
        return out

    def depth_pair(self):
        """(top_depth_mm, near_depth_mm) uint16. 없으면 None."""
        out = []
        for key in ("top_depth", "near_depth"):
            m = self.last.get(key)
            if m is None:
                return None
            if m.encoding == "16UC1":
                out.append(np.frombuffer(m.data, dtype=np.uint16)
                           .reshape(m.height, m.width).copy())
            elif m.encoding == "32FC1":
                d = np.frombuffer(m.data, dtype=np.float32).reshape(m.height, m.width)
                d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                out.append(np.clip(d * 1000.0, 0, 65535).astype(np.uint16))
            else:
                return None
        return out

    def camera_k(self, cam: str):
        m = self.last.get(f"{cam}_info")
        return [float(v) for v in m.k] if m else None

    def wait_fresh_frames(self, timeout: float = 8.0) -> bool:
        keys = ("top_rgb", "top_depth", "near_rgb", "near_depth")
        s0 = {k: self.count.get(k, 0) for k in keys}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
            if all(self.count.get(k, 0) >= s0[k] + 2 for k in keys):
                return True
        return False

    def wait_frames_after(self, t_min: float, timeout: float = 8.0) -> bool:
        """4개 스트림 전부 수신시각(stamp) > t_min 인 프레임 확보 대기.

        [2026-07-22 오후] 스캔 스텝의 정착(spin 0.35s)과 신선프레임 대기
        (+2프레임 카운트)가 직렬이라 스텝당 0.3~0.8s 가 이중으로 낭비됐다.
        t_min = 회전종료 + 정착시간 을 넘겨 받으면 정착 대기 중 도착한
        프레임이 그대로 인정돼 두 대기가 겹친다. 수신시각 기준이므로 전송
        지연(<0.35s)만큼 보수적 — 회전 중 촬영 프레임은 통과 못 한다."""
        keys = ("top_rgb", "top_depth", "near_rgb", "near_depth")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
            if all(self.stamp.get(k, 0.0) > t_min for k in keys):
                return True
        return False


# =========================================================================
# 헬스체크 / 프로세스
# =========================================================================
HEALTH_SPEC = [
    # (key, 설명, 신선도 s, 필수여부)
    ("scan", "LiDAR /laser_scan", 1.5, True),
    ("status", "arena localization status", 1.5, True),
    ("motor", "모터 브리지 /motor/state", 2.0, True),
    ("top_rgb", "상단캠 RGB", 2.0, True),
    ("near_rgb", "근접캠 RGB", 2.0, True),
    ("top_depth", "상단캠 depth", 2.0, True),
    ("near_depth", "근접캠 depth", 2.0, True),
    ("imu", "IMU /imu/data", 1.0, False),
    ("gripper_state", "그리퍼 브리지", 3.0, False),
    ("odom", "/chassis/odom", 2.0, True),
]


def gripper_board_ok(fn: FieldNode) -> tuple[bool, str]:
    """OpenRB 보드가 실제로 살아 있는지 payload 로 판정.

    토픽 존재만으로는 안 된다 — 보드가 USB 에서 빠져도 `gripper_bridge_node`
    는 계속 퍼블리시하며, 그때 값이 `{"width_mm": null, ..., "error": true}`
    이다 (2026-07-20 실측). 토픽만 보는 헬스체크는 이걸 통과시키고, 결손은
    수거 단계(`gripper_alive`)에 가서야 드러난다 — 3분 경기에서는 이미 늦다.
    """
    m = fn.last.get("gripper_state")
    if not m:
        return False, "그리퍼 텔레메트리 없음"
    try:
        st = json.loads(m.data)
    except ValueError:
        return False, "그리퍼 payload 파싱 실패"
    if st.get("error"):
        return False, "보드 error=true (USB/전원 확인: lsusb 에 ROBOTIS OpenRB-150)"
    if st.get("width_mm") is None:
        return False, "width_mm=null — 서보 미검출"
    if "connected" in st and not st.get("connected"):
        return False, "connected=false"
    return True, f"width {st['width_mm']:.1f}mm"


def healthcheck(fn: FieldNode, listen_sec: float = 8.0,
                require_gripper: bool = True) -> dict:
    fn.spin_for(listen_sec)
    items = []
    ok = True
    for key, desc, within, required in HEALTH_SPEC:
        alive = fn.fresh(key, within=max(within, 3.0)) or fn.count.get(key, 0) > 0
        items.append({"key": key, "desc": desc, "alive": alive, "required": required,
                      "count": fn.count.get(key, 0)})
        if required and not alive:
            ok = False
    g_ok, g_why = gripper_board_ok(fn)
    items.append({"key": "gripper_board", "desc": "그리퍼/마스트 보드(OpenRB) 실동작",
                  "alive": g_ok, "required": require_gripper, "detail": g_why})
    if require_gripper and not g_ok:
        ok = False

    st = fn.arena_status()
    loc = st.get("localization") or {}
    items.append({"key": "localization", "desc": "wall_range 락",
                  "alive": bool(loc), "required": False,
                  "detail": {k: loc.get(k) for k in ("reason", "latency_ms") if k in loc}})
    return {"ok": ok, "items": items}


def health_report_text(h: dict) -> str:
    lines = []
    for it in h["items"]:
        mark = "✓" if it["alive"] else ("✗필수" if it.get("required") else "–")
        extra = f" ({it.get('detail')})" if it.get("detail") else ""
        lines.append(f"  [{mark}] {it['desc']}{extra}")
    lines.append("== 전체: " + ("정상" if h["ok"] else "필수 항목 결손 — 위 ✗ 확인"))
    return "\n".join(lines)


class BridgeManager:
    """경기 브리지/arena 스택 자동 기동 (field72.sh 패턴). 단일 인스턴스 가드."""

    ARENA_PATTERN = "ros2 launch arena_lightweight_control"
    MECANUM_PATTERN = "mecanum_bridge_node"

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pgrep(pattern: str) -> list[str]:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        return [p for p in r.stdout.split() if p]

    def stack_running(self) -> bool:
        return bool(self._pgrep(self.ARENA_PATTERN))

    def kill_duplicate_mecanum(self) -> int:
        """모터 브리지 다중 실행 = 시리얼 스터터 원인. 1개만 남기고 정리."""
        pids = self._pgrep("install/robot_hardware/lib/.*/" + self.MECANUM_PATTERN)
        if len(pids) <= 1:
            return 0
        for pid in pids[:-1]:
            subprocess.run(["kill", "-INT", pid])
        time.sleep(1.0)
        return len(pids) - 1

    def launch_stack(self, camera_depth: bool = True) -> bool:
        if self.stack_running():
            return True
        env = os.environ.copy()
        rs = str(Path.home() / ".local/opt/librealsense-rsusb-2.58.2/lib")
        env["REALSENSE_RSUSB_LD_LIBRARY_PATH"] = rs + ":" + env.get("LD_LIBRARY_PATH", "")
        env.update({
            "BASE_SCAN_YAW": "0.0",
            "DRIVE_TYPE": "mecanum",
            "IMU_TOPIC": "/imu/data",
            "LAUNCH_CAMERAS": "true",
            "LAUNCH_PYTHON_UI": "false",
        })
        script = REPO_ROOT / "scripts" / "run_lightweight_arena_control.sh"
        log = self.log_dir / "stack_launch.log"
        args = ["bash", str(script)]
        if camera_depth:
            args.append("enable_camera_depth:=true")
        # 해상도 전환 (2026-07-21): env 로 opt-in. 예)
        #   CAMERA_RGB_PROFILE=1920x1080x15 CAMERA_DEPTH_PROFILE=640x480x15
        for env_key, launch_arg in (("CAMERA_RGB_PROFILE", "camera_rgb_profile"),
                                    ("CAMERA_DEPTH_PROFILE", "camera_depth_profile")):
            v = os.environ.get(env_key)
            if v:
                args.append(f"{launch_arg}:={v}")
        with open(log, "ab") as f:
            subprocess.Popen(["nohup"] + args, stdout=f, stderr=f,
                             env=env, start_new_session=True)
        return True

    def stop_stack(self):
        for pid in self._pgrep(self.ARENA_PATTERN):
            subprocess.run(["kill", "-INT", pid])

    # ---- 그리퍼 브리지 자가복구 (2026-07-24) --------------------------------
    # 부모 arena 런치는 살아있는데 gripper_bridge_node 자식만 죽은 "반쪽 스택"을
    # 감지·복구한다. launch_stack() 의 stack_running() 가드는 부모 PID 만 봐서 이
    # 상태를 재기동하지 않았고, up 은 require_gripper=False 로 묵인했다 (7/24 사고).
    # 복구 우선순위: ① 이미 정상이면 no-op ② launch respawn 복귀 대기
    # ③ launch 자식이 살아있는데 보드가 안 붙으면 **그 자식만 1회 kill** 하고
    # 되살리는 건 respawn 에 맡긴다 ④ launch 자식이 아예 없을 때만 standalone
    # 재기동(카메라/라이다 재초기화 없음). 근거:
    # docs/06-troubleshooting.md
    #
    # [2026-07-24 저녁 수정] ③ 을 신설하고 ④ 의 조건을 좁혔다. 종전 구현은
    # 살아있는 launch 자식까지 kill 한 뒤 1.0s 만에 standalone 을 띄웠는데,
    # launch 는 respawn_delay=2.0 으로 같은 자식을 되살리므로 **반드시 두
    # 인스턴스**가 됐다. OpenRB 시리얼은 exclusive=True 라 진 쪽이
    # `connected:false` + Errno 11 을 5Hz 로 영원히 퍼블리시하고, /gripper/state
    # 가 healthy/broken 혼재 스트림이 되어 러너의 gripper_alive() (최신 1샘플
    # 판정)가 동전던지기가 된다 → 05:12 사고, 이후 실기 2런 수거 0사이클 전멸.
    GRIPPER_NODE_PATTERN = "robot_hardware/lib/.*gripper_bridge_node"
    RESPAWN_DELAY_S = 2.0   # real_competition_bridge.launch.py 의 respawn_delay

    def gripper_proc_running(self) -> bool:
        return bool(self._pgrep(self.GRIPPER_NODE_PATTERN))

    @staticmethod
    def _ppid(pid: str) -> str:
        r = subprocess.run(["ps", "-o", "ppid=", "-p", pid],
                           capture_output=True, text=True)
        return r.stdout.strip()

    def gripper_launch_child_pids(self) -> list[str]:
        """launch(부모)가 respawn 을 책임지는 gripper_bridge_node PID 목록.

        이 목록이 비어있지 않으면 **우리가 새로 띄우면 안 된다** — 죽여도
        respawn 이 되살리므로 우리가 띄운 것과 겹친다.
        """
        parents = set(self._pgrep(self.ARENA_PATTERN))
        if not parents:
            return []
        return [pid for pid in self._pgrep(self.GRIPPER_NODE_PATTERN)
                if self._ppid(pid) in parents]

    def kill_orphan_gripper_nodes(self) -> int:
        """launch 자식이 아닌 잔존 그리퍼 브리지를 정리한다 (중복 해소).

        launch 자식이 하나라도 살아있을 때만 동작한다 — 시리얼을 다투는
        고아 standalone 이 곧 /gripper/state 오염원이다. kill_duplicate_mecanum
        과 같은 취지의 단일 인스턴스 가드.
        """
        if not self.gripper_launch_child_pids():
            return 0
        keep = set(self.gripper_launch_child_pids())
        orphans = [p for p in self._pgrep(self.GRIPPER_NODE_PATTERN) if p not in keep]
        for pid in orphans:
            subprocess.run(["kill", "-INT", pid])
        if orphans:
            time.sleep(1.0)
        return len(orphans)

    def restart_gripper_node(self) -> bool:
        """launch 가 관리하지 **않는** 그리퍼 브리지만 standalone 으로 되살린다.

        launch 자식이 살아 있으면 아무것도 하지 않고 False 를 돌려준다 —
        죽이면 respawn 이 되살리므로 이중 점유가 되기 때문(위 사고 주석 참조).
        그 경우의 복구는 ensure_gripper_alive() 의 ③ (자식만 kick) 이 맡는다.
        이 경로는 respawn 없는 구 스택이나 launch 자체가 없는 경우의 최후 복구다.

        [폐기 2026-07-24 저녁 — 구 구현. 롤백 시 아래 2줄을 함수 선두로 복원]
            for pid in self._pgrep(self.GRIPPER_NODE_PATTERN):
                subprocess.run(["kill", "-INT", pid])
        """
        if self.gripper_launch_child_pids():
            return False
        for pid in self._pgrep(self.GRIPPER_NODE_PATTERN):
            subprocess.run(["kill", "-INT", pid])
        time.sleep(1.0)
        cfg = REPO_ROOT / "install/robot_bringup/share/robot_bringup/config/real.yaml"
        if not cfg.exists():
            cfg = REPO_ROOT / "hardware/ros2/robot_bringup/config/real.yaml"
        log = self.log_dir / "gripper_restart.log"
        args = ["ros2", "run", "robot_hardware", "gripper_bridge_node",
                "--ros-args", "-r", "__node:=gripper_bridge_node",
                "--params-file", str(cfg)]
        with open(log, "ab") as f:
            subprocess.Popen(["nohup"] + args, stdout=f, stderr=f,
                             env=os.environ.copy(), start_new_session=True)
        return True

    def ensure_gripper_alive(self, fn, wait_s: float = 6.0,
                             poll: float = 1.0) -> dict:
        """그리퍼 브리지가 실제 텔레메트리를 낼 때까지 자가복구.

        반환 {ok, action, detail}. action ∈ {none, dedup, respawn, respawn_kick,
        respawn_kick_failed, restart, restart_failed}.
        gripper_board_ok() 로 판정(토픽 존재가 아니라 payload).
        """
        # [2026-07-24 저녁] ⓪ 중복 먼저 정리. 고아 standalone 이 살아 있으면
        # /gripper/state 에 connected:false 가 섞여 판정 자체를 못 믿는다.
        if self.kill_orphan_gripper_nodes():
            fn.spin_for(poll)
            ok, why = gripper_board_ok(fn)
            if ok:
                return {"ok": True, "action": "dedup", "detail": why}
        ok, why = gripper_board_ok(fn)
        if ok:
            return {"ok": True, "action": "none", "detail": why}
        # ② launch respawn 복귀 대기
        waited = 0.0
        while waited < wait_s:
            fn.spin_for(poll)
            waited += poll
            ok, why = gripper_board_ok(fn)
            if ok:
                return {"ok": True, "action": "respawn", "detail": why}
        # ③ [2026-07-24 저녁 신설] launch 자식은 살아 있는데 보드가 안 붙는다
        #    (노드가 멎었거나 포트를 못 잡은 상태). **자식만 1회 kill** 하고
        #    되살리는 건 respawn 에 맡긴다 — standalone 을 띄우면 중복이 된다.
        kids = self.gripper_launch_child_pids()
        if kids:
            for pid in kids:
                subprocess.run(["kill", "-INT", pid])
            waited = 0.0
            budget = self.RESPAWN_DELAY_S + max(wait_s, 8.0)
            while waited < budget:
                fn.spin_for(poll)
                waited += poll
                ok, why = gripper_board_ok(fn)
                if ok:
                    return {"ok": True, "action": "respawn_kick", "detail": why}
            return {"ok": False, "action": "respawn_kick_failed", "detail": why}
        # ④ launch 자식이 아예 없다 → 최후 수단으로 standalone 재기동
        self.restart_gripper_node()
        waited = 0.0
        while waited < max(wait_s, 8.0):
            fn.spin_for(poll)
            waited += poll
            ok, why = gripper_board_ok(fn)
            if ok:
                return {"ok": True, "action": "restart", "detail": why}
        return {"ok": False, "action": "restart_failed", "detail": why}


def battery_off_signature(fn: FieldNode, probe_dist: float = 0.03) -> bool:
    """배터리 OFF 판정: 소이동 명령이 타임아웃/실패 + odom 변위 0 (motor connected 유지)."""
    o0 = fn.odom_xy()
    r = fn.move_relative(probe_dist, 0.0, max_v=0.15, timeout=8.0)
    fn.spin_for(0.5)
    o1 = fn.odom_xy()
    moved = (o0 is not None and o1 is not None
             and math.hypot(o1[0] - o0[0], o1[1] - o0[1]) > 0.005)
    if r.get("ok") and moved:
        fn.move_relative(-probe_dist, 0.0, max_v=0.15, timeout=8.0)  # 원위치
        return False
    return not moved


def rosbag_record(out_dir: Path, extra_topics: list[str] | None = None) -> subprocess.Popen:
    """진단 rosbag (record_diagnosis_bag.sh 확장: 카메라/odom 포함)."""
    topics = [
        "/laser_scan", "/arena_lightweight/status", "/motor/state",
        "/chassis/odom", "/imu/data", "/base/move_relative", "/base/move_result",
        "/gripper/command", "/gripper/state", "/gripper/grasp",
        "/lift/command", "/lift/state", "/cmd_vel_direct",
    ] + (extra_topics or [])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "rosbag.log", "ab")
    return subprocess.Popen(
        ["ros2", "bag", "record", "-o", str(out_dir / "bag")] + topics,
        stdout=log, stderr=log, start_new_session=True)


# =========================================================================
# 조작자 체크리스트 (물리 상태는 센서로 모를 수 있음 — y/n 프롬프트)
# =========================================================================
def operator_checklist(items: list[tuple[str, str]], assume_yes: bool = False) -> dict:
    """items: [(key, 질문)]. 반환 {key: bool}. --yes면 전부 True."""
    out = {}
    for key, q in items:
        if assume_yes:
            out[key] = True
            continue
        while True:
            a = input(f"[체크] {q} (y/n): ").strip().lower()
            if a in ("y", "yes", "ㅛ"):
                out[key] = True
                break
            if a in ("n", "no", "ㅜ"):
                out[key] = False
                break
    return out


DEFAULT_CHECKLIST = [
    ("ready", "물리 상태 일괄 확인 — ①배터리 ON ②우하단 출발구역 북향 배치 "
              "③마스트 최저(다운) ④그리퍼 주변 클리어 ⑤경기장 안 클리어. "
              "전부 확인했나요?"),
]
