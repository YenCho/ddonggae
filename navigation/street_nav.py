#!/usr/bin/env python3
"""street_nav — 목표 물체까지의 이동 전담 모듈 (주행 재설계 2단계).

[2026-07-21 신규] 기존 match_runner.py 의 경로계획(plan_route/follow_route/
align_leg_yaw)과 **완전히 분리된** 새 파일이다. 기존 코드는 건드리지 않으므로
실패 시 이 파일을 쓰지 않는 것만으로 롤백된다.

규칙 문서(단일 소스): navigation/docs/control-and-routing.md

--------------------------------------------------------------------------
설계 요약 (조작자 확정 2026-07-21)
--------------------------------------------------------------------------
목표 물체 하나를 잡으러 갈 때의 이동은 항상 아래 순서로만 일어난다.

  1) 하이웨이에서 **map x 를 먼저 맞춘다** (yaw 는 북향 고정, 메카넘 횡이동).
  2) 그 street 를 따라 **북진**해서 mini-goal 로 간다 (횡오차 P 보정).
  3) mini-goal 에서 **CCW 45도** 회전 → 물체 정면(북서 대각).
  4) 기존 final approach 로 접근·파지 (이 모듈 밖, match_runner 재사용).
  5) mini-goal 복귀 → **CW 45도** (북향 복귀).
  6) **후진**으로 하이웨이까지 내려온다 (yaw 북향 유지, 횡오차 P 보정).
  7) 적재 기준점으로 이동 → 적재 (이 모듈 밖).

mini-goal = 물체 좌표 + (+0.25, -0.25) m  →  물체의 **남동쪽** street 교차점.
  · 물체는 mini-goal 에서 북서 대각 35.36cm.
  · 남서(-0.25,-0.25) 안은 2026-07-21 폐기: 최서열(공식 x=50) 물체의 mini-goal
    이 공식 x=25 가 되어 그 street 남단이 적재함(공식 0~40)과 겹쳤다.
    남동안은 mini-goal x 가 75~375 라 적재함을 절대 지나지 않는다.

--------------------------------------------------------------------------
왜 yaw 를 북향으로 고정하는가 (불변식)
--------------------------------------------------------------------------
그리퍼가 라이다 중심 기준 **전방 26cm** 튀어나와 있다 (2026-07-21 실측).
진행 방향과 그리퍼가 수직이면 측방 돌출 26cm > 통로 허용 21cm 라 물체를 친다.
yaw 를 북향으로 고정하면 그리퍼는 항상 진행축과 평행(전진 시 선행, 후진 시
후행)이라 측방 돌출이 본체 반폭 9.5cm 로 떨어진다 → 여유 11.5cm.

  ⚠ 이 모듈이 만드는 모든 직선 이동은 yaw 를 북향으로 유지한다.
    회전은 mini-goal 에서의 ±45도 두 번뿐이다. (종전 4축 정렬은 사이클당
    회전 수가 들쭉날쭉했고 그리퍼 수직 통과를 허용해 2026-07-21 폐기)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))
import fieldlib as fl  # noqa: E402

# =========================================================================
# 기하 상수
# =========================================================================
# mini-goal 오프셋 [m]. 물체 기준 남동쪽 street 교차점.
MINI_GOAL_OFFSET_M = (+0.25, -0.25)
# [2026-07-23 신규 — 조작자 지시 "스캔 street 옆 물체는 내려오면서 집는다"]
# 미러 mini-goal = 물체 기준 **남서쪽** 교차점. 스캔점(공식 225,225)이 놓인
# street x=225 를 기준으로, x=200 셀은 남동(기본) 대각이 같은 street 위에
# 있지만 x=250 셀은 남서 대각이라야 같은 street 다. 두 방향을 모두 지원해
# "하산 중 파지"의 대상 폭을 x=200/250 양쪽으로 넓힌다.
MINI_GOAL_OFFSET_MIRROR_M = (-0.25, -0.25)

# 북향 heading [rad]. map 규약: 북 = +y = yaw +90도 (START_POSE 로 확정).
HEADING_NORTH_RAD = math.pi / 2.0

# mini-goal 에서 물체를 보는 방향 = 북서 대각(135도). 북향에서 +45도 = CCW.
FACE_OBJECT_TURN_RAD = +math.pi / 4.0    # CCW 45도 (물체 정면)
RETURN_NORTH_TURN_RAD = -math.pi / 4.0   # CW 45도 (북향 복귀)
# 미러 mini-goal(남서 대각)에서는 물체가 **북동 대각(45도)** — 부호만 반전.
FACE_OBJECT_TURN_MIRROR_RAD = -math.pi / 4.0   # CW 45도

# 하이웨이 y [m]. 이 남쪽은 규칙상 물체가 없다 (공식 y<60cm).
# HIGHWAY_Y_M = -1.80   # [폐기 2026-07-22] 공식 20cm 라인 — 남하/북진이 사이클당
#                       # 왕복 0.8m 길어 시간 낭비 (00:50 실기, 조작자 지시로 상향).
HIGHWAY_Y_M = -0.40   # [demo/arena-2m] 경기 -1.40. 공식 y=60cm 라인 = e2e HIGHWAY_FREE_Y_M 과 동일.
#   기하 근거: 최남단 물체행은 공식 y=100(실루엣 하단 ~96cm = map -1.04).
#   북향 그리퍼 선단은 중심 +26cm → y=-1.40 에서 선단 -1.14 (공식 86cm),
#   물체 실루엣까지 10cm 마진. 회전 스윕(26cm)도 동일 마진. 60 은 보수선.

# =========================================================================
# 폐루프 주행 게인/한계
# =========================================================================
# [2026-07-22 개명] 게인·상한은 **몸체 축**(전방/횡)이 아니라 레그에서의
# **역할**(진행축 along / 횡축 cross)에 붙인다. yaw 북향 고정이라 몸체 축과
# map 축은 아래처럼 고정 대응하는데, 레그마다 어느 쪽이 진행축인지가 다르다.
#   body 전방(+vx) = map +y ,  body 좌측(+vy) = map −x
#   · HIGHWAY_X 레그  : map x 이동 → **몸체 횡**이 진행축, 몸체 전방이 횡축
#   · STREET/RETREAT  : map y 이동 → **몸체 전방**이 진행축, 몸체 횡이 횡축
# 종전에는 이름 그대로 몸체 축에 붙어 있어서 하이웨이 레그의 진행 성분이
# "횡오차"로 취급됐고, 그 결과 이탈 게이트가 레그 시작 즉시 발동했다
# (2026-07-22 실기 즉사 버그 — §아래 drive_to 주석).
#
# 횡축 P 게인 [1/s]. 오차 10cm → 0.15 m/s (시정수 ~0.67s). 종전 KP_LATERAL.
# ⚠ D 항 없음: 속도명령→위치 플랜트가 이미 적분기라 P 만으로 1차 안정계이고
#   정상편차도 0 이다. 게다가 status 20Hz vs x/y 갱신 12.5Hz 라 약 37% 샘플이
#   직전값과 동일해 고정 dt 미분은 0/2배로 튄다 (2026-07-21 분석).
#   실기에서 오버슈트가 보이면 그때 이벤트구동+저역통과 D 를 추가할 것.
# KP_CROSS = 1.5   # [폐기 2026-07-22] 00:50 실기: 진행 0.75m/s 대비 보정
#                  # (10cm→0.15m/s, 경로각 ~11°)이 너무 약해 후진 남하에서
#                  # 횡오차가 0.22m 까지 성장 (00:58:05 운반_남하 이탈).
# KP_CROSS = 3.0  # [폐기 2026-07-22 02시대 실기] 시뮬(지연 250ms)은 안정이었으나
#                 # 실기에서 좌우 진동 관측 — 실기 지연/슬립이 모델보다 크다.
# KP_CROSS = 1.8  # [폐기 2026-07-22 오후] 진동 원인이 localizer 7.5cm 계단
#                 # 피드백으로 규명·refine(1.9cm)으로 소멸 — 과도 응답 회복
#                 # 부족(04:39:46 highway-x +17cm 팽창)이라 2.5 복원.
KP_CROSS = 2.5   # [2026-07-22 오후] 오차 10cm → 0.21 m/s (데드밴드 감산 후).
                 # 진동 재발 시 1.8 롤백 — 실기 1런 확인 필요.
# 진행축 P 게인 [1/s]. 종전 KP_FORWARD.
KP_ALONG = 1.2
# yaw P 게인 [1/s].
KP_YAW = 1.5

# MAX_CROSS_MPS = 0.25   # [폐기 2026-07-22] 상한이 진행 0.75 의 1/3 이라 최대
#                        # 경로각 18° — 실기 드리프트(후진 메카넘)를 못 이겼다.
MAX_CROSS_MPS = 0.45      # 횡축 **보정** 상한 (종전 MAX_LATERAL_MPS).
                          # 진행축 속도는 여기에 걸리지 않는다 — v_max 가 정한다.
                          # 예산: street 0.75+0.45=1.20 > 1.01 은 clamp_body_twist
                          # 균일 축소가 흡수 (포화 동시 발생은 과도기뿐).

# ---- 횡오차 기반 진행속도 적응 (2026-07-22 신규 — "강한 adoption") ----
# 횡오차가 클수록 진행축 속도를 줄여 보정 권한을 몰아준다. 드리프트가 속도에
# 비례하는 실기 특성상(메카넘 슬립) 감속 자체가 교란도 줄이는 자기안정 구조.
#   |e_cross| <= START: 감속 없음 / >= FULL: 진행 0 (횡복구 전용, 루프 유지)
# CROSS_SLOW_START_M = 0.05  # [폐기 2026-07-22 오후] 허용 봉투가 하이웨이
# CROSS_SLOW_FULL_M = 0.15   # 마진 10cm 를 초과 — 04:39:46 실기 +17cm 팽창이
#                            # 물체행 침범(마진 10cm) 후에야 복구 발동.
CROSS_SLOW_START_M = 0.04
# CROSS_SLOW_FULL_M = 0.10  # [폐기 2026-07-23 — 조작자 승인] 하이웨이 기하
#                           # 마진(10cm) 기준값이라 **street 여유 11.5cm**
#                           # (통로 반폭 25 - 물체 반폭 4 - 본체 반폭 9.5)에선
#                           # 발동 시점의 잔여 여유가 1.5cm 뿐이었다.
#                           # 01:10~01:20 실기: 스캔후_남하 레그가 발동(-0.12m)
#                           # 후에도 13.6cm(011250)/15.5cm(011726)까지 벌어져
#                           # 물체행을 2.1~4.0cm 침범. 발동 후 계속 커진 3.6~
#                           # 5.5cm 는 피드백 지연 x 드리프트 속도의 서명이라,
#                           # 문턱을 그 폭만큼 앞당기는 것이 직접적인 대책이다.
CROSS_SLOW_FULL_M = 0.08   # 진행 0 (횡복구) 문턱 = street 여유 11.5cm - 지연분
# [2026-07-23 신규] 횡복구 **해제** 문턱. 종전엔 해제 판정에 CROSS_ARM_M(0.08)
# 을 재사용했고, 발동이 0.10 이라 우연히 2cm 히스테리시스가 있었다. 발동을
# 0.08 로 내리면 발동==해제가 되어 경계에서 발동/해제가 매 틱 번갈아 일어난다
# (복구 카운터 폭증 + 로그 도배 + 진행속도 채터링). 같은 2cm 폭을 명시 상수로
# 분리해 보존한다. 무장 문턱(CROSS_ARM_M)과는 역할이 다르므로 공유하지 않는다.
CROSS_RECOVER_EXIT_M = 0.06
# MAX_YAW_RATE = 0.56  # [폐기 2026-07-22 오후] 0.7배 하향분 — 실기 테스트에서
#                      # 0.8 원복해도 문제 없음 확인 (조작자 지시 롤백).
MAX_YAW_RATE = 0.8        # [rad/s] 주행 중 yaw 유지/드리프트용
TURN_YAW_RATE = 1.2       # [rad/s] [2026-07-22 저녁 신규] **제자리 회전 전용**
                          # 상한 = 0.8 x 1.5 (조작자 지시 "제자리 회전 1.5배").
                          # 주행 중 캡(MAX_YAW_RATE)과 분리 — 병진+회전 동시엔
                          # 종전 값 유지로 속도예산·드리프트 특성 불변.
MIN_SPEED_MPS = 0.06      # 이보다 느리면 정지마찰로 안 움직인다
MIN_YAW_RATE = 0.15       # 제자리 회전 정지마찰 하한 [rad/s] (turn_to 와 공용)
# [2026-07-23 신규 — 조작자 "제자리 회전 각속도가 안 올랐다" 로그 검증 결과]
# 07-22 저녁에 올린 것은 **상한**(TURN_YAW_RATE 0.8→1.2)뿐이라 ±45° 회전은
# 전혀 빨라지지 않았다: turn_to 의 명령은 wz = KP_YAW·e 이고 KP_YAW=1.5 이므로
# e=45°(0.785rad)에서 wz=1.18 — 애초에 종전 상한 0.8 을 넘는 구간이 0.25s 뿐,
# 나머지는 게인이 결정한다. 실측도 일치: face 위상 중앙값이 07-22 오후 1.43~
# 1.48s → 저녁 이후 1.15~1.52s 로 사실상 불변(모델 예측 1.40→1.34s).
# 지수 수렴 모델 t = ln(e0/tol)/KP 이므로 **게인이 유일한 지렛대**다.
#   KP 1.5 → 45° 회전 1.34s / KP 2.5 → 0.86s (상한 1.2 포화 0.25s 포함)
# 지연 여유: status 지연 중앙값 65ms 에서 KP·τ=0.16 rad ≪ π/2 라 안정.
# 주행 중 yaw 유지에는 쓰지 않는다 (병진과 결합해 지그재그를 키운다).
TURN_KP_YAW = 2.5         # 제자리 회전 전용 yaw P 게인 [1/s]

# ---- 적재 드리프트 전용 회전 [2026-07-23 신규 — 조작자 지시] ----
# 배경: 조작자 관측 "적재하고 나서 출발하면서 yaw 돌릴 때가 각속도 예산보다
# 많이 느리다". 회전 경로가 4갈래인데 적재 전후 드리프트만 가장 느린 걸 탔다:
#   ① 펌웨어 position move (do_rotate_fire)  ROTATE_MAX_V 0.525 → 본체 2.65 rad/s
#   ② turn_to (mini-goal ±45°)              TURN_YAW_RATE 1.2 / TURN_KP_YAW 2.5
#   ③ drive_to 주행 중 yaw 유지               MAX_YAW_RATE 0.8 / KP_YAW 1.5
#   ④ drive_drift (적재 진입/복귀)            ← ③을 그대로 쓰고 있었다
# 07-22 저녁 "제자리 회전 1.5배"와 07-23 게인 상향이 ①②에만 걸려 ④는 누락.
# ④는 **mini-goal 회전(②)과 완전히 분리된 별개 상수**다 — 여기를 올려도
# face_object/face_north 속도는 변하지 않고, 그 반대도 같다.
#
# 실측 근거 (2026-07-23 실기 전량, n=55/67):
#   복귀 드리프트(적재→출발, 135°) 하한 3s  — 모델 예측 2.90s 와 일치
#   운반 드리프트(파지→적재,  90°) 하한 2.1s — 모델 예측 1.92s 와 일치
#   → 두 레그의 **하한은 병진이 아니라 yaw 각속도**가 잡고 있다.
#
# ⚠ 속도 예산: 바퀴 상한 26.0 rad/s x r 0.0388 = 1.0088 m/s ≥ |vx|+|vy|+k|wz|
#   (k=0.198). 드리프트는 world 속도 norm 을 v_max(하이웨이 0.6)로 자르므로
#   몸체 45° 방향일 때 |vx|+|vy| = 0.6√2 = 0.849 → wz 여유는 0.81 rad/s 뿐이다
#   (0°/90° 방향이면 2.06). 즉 캡 1.5 는 45° 국면에서 브릿지
#   clamp_body_twist 의 균일 축소에 걸린다: 명령 (0.849 + 0.297)=1.146 →
#   scale 0.88 → 실효 v 0.53 / wz 1.32. **회전이 하한을 잡는 레그이므로 병진을
#   조금 내주고 회전을 사는 것이 net 이득**이고, 축소는 균일이라 궤적 형상은
#   보존된다(방향 불변). 2.0 이상은 축소율이 커져 실익이 급감한다.
# 실기 튜닝은 match_runner.py --drift-yaw-rate / --drift-kp-yaw 로 한다
# (StreetNavigator 인스턴스 속성 override — 아래 __init__ 참조).
DRIFT_YAW_RATE = 1.5      # [rad/s] 드리프트 중 yaw 상한 (종전 = MAX_YAW_RATE 0.8)
DRIFT_KP_YAW = 2.5        # [1/s]   드리프트 중 yaw P 게인 (종전 = KP_YAW 1.5)

# ---- 주행 속도 [m/s] ----
# [2026-07-21] 3분 경기라 시간이 부족해 종전 0.45 에서 상향 (조작자 지시).
# 메카넘 속도 예산: |vx| + |vy| + k|wz| <= max_wheel_rad_s x r
#                 = 26.0 x 0.0388 = 1.01 m/s  (real.yaml 실기값)
V_HIGHWAY_MPS = 0.6   # 하이웨이(공식 y<60cm)는 규칙상 물체가 없어 **하드웨어
                      # 상한까지** 쓴다 (조작자 지시 2026-07-22 "제한 다 풀어도 됨").
                      # [2026-07-22 하향 0.9 → 0.6] 이 레그는 전진이 아니라
                      # **횡이동**이라 브릿지 상한이 max_linear_x_mps(0.9)가 아니라
                      # max_linear_y_mps(0.6)다 (real.yaml:70-71). 0.9 를 명령해도
                      # mecanum_bridge_node 가 0.6 으로 자르므로 종전 값은 실효가
                      # 없었다. 0.6 이 현재 구성에서 실제로 낼 수 있는 최대다.
V_STREET_MPS = 0.75   # street 는 0.75 = 1.01 - MAX_CROSS(0.25). 이 지점이
                      # **횡보정 권한을 온전히 남기는 최대 속도**다. 더 올리면
                      # clamp_body_twist 가 트위스트 전체를 균일 축소해 보정이
                      # 전진속도를 깎는다(치명적이진 않고 점진적 열화).
                      # 실기에서 통로 유지가 확인되면 0.9 까지 올려도 된다.

# ---- localizer 피드백 특성 (2026-07-22 02시 실기 trace 로 실측) ----
POSE_QUANTUM_M = 0.075
# status pose 의 x/y 는 wall_range refine 격자(0.3 coarse x 1/4) 위로
# 양자화되어 나온다: 값이 전부 0.02 + k*0.075. 동결(0.2~0.5s) 후 여러 칸을
# 한 번에 갱신하는 버스트도 관측됐다 (0.75m/s x 0.3s = 0.22m). 아래 도착
# 허용/데드밴드/점프 허용치는 전부 이 양자에 맞춘 값이다.
# (arena 노드 미세 refine(→~1.9cm) 배포 후에도 안전측이므로 유지)

# 도착 판정
# POS_TOL_M = 0.04   # [폐기 2026-07-22] 양자 7.5cm 의 절반 이하라 "운 좋은
#                    # 양자"가 목표 4cm 안에 떨어질 때까지 기어다니며 대기
#                    # (02시 실기 정착 병목). 잔차는 접근 시각측정이 흡수.
POS_TOL_M = 0.06          # mini-goal 등 기본 도착 허용
POS_TOL_ROUGH_M = 0.08    # [2026-07-22 신규] 중간 레그(하이웨이 x/남하/서진)
YAW_TOL_RAD = math.radians(6.0)   # turn_to(제자리 회전) 허용
# [2026-07-22 4°→6°] 정착 병목 완화 (조작자 지시 "rough하게"). do_rotate 의
# 재회전 수용 문턱(15°)보다 훨씬 정밀하므로 후속 단계 지장 없음.
YAW_ARRIVE_TOL_RAD = math.radians(8.0)
YAW_ARRIVE_ROUGH_RAD = math.radians(12.0)   # 중간 레그용 (2026-07-22 신규)
# 병진 레그 도착의 yaw 허용은 완화값을 쓴다. 도착 직후엔 반드시
# turn_to(±45°/서향)나 접근 시각측정이 따라와 yaw 를 각자 정밀하게 잡는다.
# 4° 동시 요구는 01:02 실기 street-north 25s 타임아웃(라이브락)을 만들었다.

# 횡 보정 데드밴드 [m] (2026-07-22 신규, 수정안 6): 양자 노이즈(±1칸 순간
# 오차)에 보정이 반응해 지그재그가 나는 것을 막는다. 감산형(오차-데드밴드)
# 이라 경계에서 명령이 연속이다. 게이트/감속 판정에는 원오차를 그대로 쓴다.
# CROSS_DEADBAND_M = 0.04  # [폐기 2026-07-22 오후] 7.5cm 양자화 시절 값 —
#                          # street 물리 마진 3cm(통로 21cm-반폭 18cm)보다 커서
#                          # 4cm 오프셋 방치 = 물체 접촉 보장. refine(1.9cm) 후
#                          # 노이즈 플로어에 맞춰 하향.
CROSS_DEADBAND_M = 0.015

# 안전 게이트
# LATERAL_ABORT_M = 0.15  # [폐기 2026-07-22] 초과 시 실패 반환 → 호출부가
#                         # "계속" 하며 필드 한복판에서 다음 단계(대각 운반
#                         # 계획)로 넘어갔다 (00:58:05 실기 — 조작자 관측
#                         # "루프 탈출 후 골대 직진"). 이제 이탈은 실패가
#                         # 아니라 **횡복구**(진행 0, 횡만 보정)로 처리한다.
CROSS_JUMP_M = 0.35       # [2026-07-22 신규] 무장 후 횡오차가 이 이상이면
                          # 정상 주행으론 불가능 — 로컬라이제이션 튐으로 보고
                          # 정지+실패. (통로 반폭 0.25 + 여유 0.10)
# JUMP_TICK_M = 0.12 / JUMP_PAUSE_S / JUMP_MAX_PER_LEG
# [폐기 2026-07-22 03시 — 틱 점프 감지기 전면 제거, 조작자 결정]
#   ① 각도 점프는 IMU 피드포워드 + yaw 고정 탐색이 이미 방어.
#   ② 거리 점프(02시 실기 0.21~0.29m)의 정체는 localizer 7.5cm 양자화의
#      동결→버스트 갱신 = 실이동의 지연 방출이었고, wall_range 미세 refine
#      (~1.9cm, map_localization 2026-07-22)으로 원인이 소멸.
#   ③ 잔여 위험(회전 후 보정 스냅 0.45~0.76m)은 아래 CROSS_JUMP_M(무장 후
#      횡 0.35m 초과 → 정지) 게이트가 잡는다 — 물체행 강보정 진입 차단용
#      마지막 한 겹으로 이것만 유지.
#   오늘 실기에서 이 감지기의 오판이 수거 3사이클을 죽였다(순피해 > 순이익).
CROSS_ARM_M = 0.08        # 게이트 무장 문턱. 횡오차가 한 번 이 값 안에
                          # 들어온 뒤부터 CROSS_JUMP 감시·복구 집계를 한다.
                          # 레그 시작 시점의 큰 초기 오차를 튐으로 오판해
                          # 즉사하는 것을 막는다 (2026-07-22 실기 즉사 수정).
POSE_STALE_ABORT_S = 0.30  # pose 가 이만큼 낡으면 정지
CONTROL_HZ = 20.0          # /cmd_vel_direct 발행 주기 (mux 타임아웃 0.4s)

# ---- 위치 이상표본 게이트 [2026-07-23] ----
# ⚠ 위 250행에서 폐기된 JUMP_TICK_M 감지기와 **동작이 다르다.** 그 감지기는
# 점프를 보면 정지·일시중단·레그 실패로 갔고, 오판 한 번이 수거 사이클을
# 통째로 죽였다(7/22, 순피해 > 순이익). 이번 것은:
#   · 절대 실패를 반환하지 않고, 절대 로봇을 세우지 않는다.
#   · 의심 표본의 **x/y 만** 직전 채택값으로 대체한다 (yaw 는 IMU 피드포워드로
#     매끄러우므로 항상 새 값을 그대로 쓴다).
#   · 연속 POSE_GATE_MAX_REJECT 틱을 넘기면 **강제로 받아들여 재동기**한다 —
#     진짜 전역 재수렴 점프에서 영원히 눈감는 것을 막는다.
# 즉 최악의 오판 비용이 "2틱(약 0.1초) 동안 직전 위치 사용"으로 상한이 있다.
#
# 판정은 **포즈가 실제로 바뀐 시점 기준**으로 한다. 제어틱 기준으로 재면
# 라이다 10Hz vs 제어 17Hz 때문에 생기는 정상적인 "동결→버스트" 캐치업까지
# 이상표본으로 잡는다(250행 ②가 지적한 바로 그 현상). 전 런 트레이스
# 16개·908샘플 역검증: 제어틱 기준은 p90 환산속도가 1.56 m/s 로 물리 상한을
# 넘어 못 쓰고, 갱신시각 기준 + margin 0.06 + K=2 는 기각 7.9%,
# 최대 연속 2틱, 횡복구 오발동 23건 차단.
# 15:45 실기 문제 구간에서 e_cross −0.099/+0.126/+0.164 를 만든 표본 3개
# (d=0.210/0.228/0.269, 한계 0.117~0.174)를 정확히 잡는다.
POSE_GATE_V_MAX = 0.95     # 물리 상한 (cruise). 이 속도 x dt 에 여유를 더해 판정
POSE_GATE_MARGIN_M = 0.06  # 양자화(~1.9cm)+지연 흡수 여유
POSE_GATE_MAX_REJECT = 2   # 연속 기각 상한 — 넘으면 강제 재동기
YAW_DRIVE_MAX_RAD = math.radians(15.0)
# [2026-07-22 신규] yaw 가 북향에서 이만큼 넘게 어긋나면 **병진 금지**, 회전만.
# 근거 두 가지:
#  (1) 불변식 I-1 — 그리퍼가 진행축과 어긋나면 측방 돌출이 최대 26cm 로 커져
#      통로 허용 21cm 를 넘는다. 어긋난 채로 달리면 옆 물체를 친다.
#  (2) 어긋난 각도가 클수록 아래 hold→body 회전 보정이 커져 진행축 속도가
#      깎인다. 먼저 북향으로 돌리는 편이 항상 빠르다.
# 회전은 '북향 쪽으로만' 하므로 그리퍼 스윕이 더 나쁜 각도로 가지 않는다.


# =========================================================================
# 순수 기하 (로봇 없이 테스트 가능)
# =========================================================================
def mini_goal_for(cell_cm, mirror: bool = False) -> tuple:
    """공식 격자 셀 (x_cm, y_cm) -> mini-goal map 좌표 (x, y).

    mirror=True 면 남서 대각(2026-07-23 신규) — 물체 서쪽 street 를 쓴다.

    >>> mini_goal_for((200, 100))      # 공식 (225,75) = 프리샷 지점
    (0.25, -1.25)
    >>> mini_goal_for((250, 200), mirror=True)   # 공식 (225,175)
    (0.25, -0.25)
    """
    mx, my = fl.official_cm_to_map(float(cell_cm[0]), float(cell_cm[1]))
    off = MINI_GOAL_OFFSET_MIRROR_M if mirror else MINI_GOAL_OFFSET_M
    return (mx + off[0], my + off[1])


def object_bearing_from_mini_goal() -> float:
    """mini-goal 에서 물체를 보는 절대 heading [rad] = 135도 (북서 대각)."""
    return math.atan2(-MINI_GOAL_OFFSET_M[1], -MINI_GOAL_OFFSET_M[0])


def object_range_from_mini_goal() -> float:
    """mini-goal ~ 물체 거리 [m] = 0.3536."""
    return math.hypot(*MINI_GOAL_OFFSET_M)


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def body_error(pose, target_xy, hold_yaw: float) -> tuple:
    """map 오차를 **몸체 좌표**(전방, 좌측)로 변환.

    yaw=hold_yaw(북향)일 때 body-x = map +y(북), body-y = map -x(서).
    전진/후진/횡이동 어느 경우에도 같은 식이 성립하므로 한 함수로 처리한다.
    """
    ex = target_xy[0] - pose[0]
    ey = target_xy[1] - pose[1]
    c, s = math.cos(hold_yaw), math.sin(hold_yaw)
    return (ex * c + ey * s,      # 전방 오차
            -ex * s + ey * c)     # 좌측 오차


def merge_legs(phase: str, ok: bool, legs) -> dict:
    """레그별 `drive_to`/`drive_drift` 결과를 route 결과 하나로 합친다.

    [2026-07-23 신규 — 조작자 승인] 종전 `go_to_mini_goal` 은 마지막 레그를
    `**r2` 로 펼쳐 반환해서, **레그1(하이웨이 x 정렬 — 0.6 m/s 횡이동)의
    max_cross_m / recoveries / trace 가 통째로 버려졌다.** 01:10~01:20 실기
    분석에서 사이클별 `route.max_cross_m`(3.6~10.5cm)이 전부 street-north
    값이라 하이웨이 레그의 이탈 여부를 아예 확인할 수 없었다.

    합침 규칙:
      · `max_cross_m` = 전 레그 최대 (route 전체의 최악 이탈)
      · `recoveries`  = 전 레그 합
      · `sec`         = 전 레그 합 (종전엔 마지막 레그만이라 과소보고였다)
      · `legs`        = 레그별 통계 + **레그별 trace** (틱 궤적은 여기에만 둔다.
                        최상위 `trace` 는 어느 레그 것인지 알 수 없어 폐기)
    나머지 키(reason/pose 등)는 마지막 레그 값을 쓴다.

    legs: [(태그, drive 결과 dict), ...] — 실행된 순서.
    """
    out = dict(legs[-1][1])
    out.pop("trace", None)        # 레그별 trace 는 out["legs"] 안에만 둔다
    out["ok"] = ok
    out["phase"] = phase
    crosses = [r.get("max_cross_m") for _, r in legs
               if r.get("max_cross_m") is not None]
    if crosses:
        out["max_cross_m"] = max(crosses)
    out["recoveries"] = sum(int(r.get("recoveries") or 0) for _, r in legs)
    out["sec"] = round(sum(float(r.get("sec") or 0.0) for _, r in legs), 2)
    # [2026-07-23] 위치 게이트가 대체한 표본 수 — 게이트가 실제로 얼마나
    # 개입했는지 다음 런에서 이 한 필드로 확인한다 (0 이면 무해, 급증하면 과민).
    out["pose_gate_dropped"] = sum(
        int(r.get("pose_gate_dropped") or 0) for _, r in legs)
    out["legs"] = [
        {"leg": tag, "ok": bool(r.get("ok")), "reason": r.get("reason"),
         "sec": r.get("sec"), "max_cross_m": r.get("max_cross_m"),
         "recoveries": r.get("recoveries"),
         "pose_gate_dropped": r.get("pose_gate_dropped"), "trace": r.get("trace")}
        for tag, r in legs
    ]
    return out


# =========================================================================
# 폐루프 주행기
# =========================================================================
class StreetNavigator:
    """/cmd_vel_direct 직접 발행으로 street 를 폐루프 주행한다.

    arena_control_node 는 **pose 제공자로만** 쓴다 (goal 추종은 STOP 으로 해제).
    노드 코드는 수정하지 않는다 — 런치 인자 status_period_sec:=0.05 만 필요.

    ⚠ 종료 시 0 트위스트를 **한 번만** 보내고 발행을 멈춘다. 0을 계속 발행하면
      mux 최우선 소스를 점유해 teleop(/cmd_vel)이 영구 차단된다
      (cmd_vel_mux_node.py 주석의 과거 사고).
    """

    def __init__(self, fn, log=print, dry_run: bool = False):
        self.fn = fn
        self.log = log
        self.dry_run = dry_run
        # [2026-07-23 신규] 드리프트 회전 튜닝은 **인스턴스 속성**으로 둔다 —
        # 실기에서 실행인자로 스윕하려면 값이 바뀌어야 하는데, 모듈 전역을
        # 밖에서 대입하면 같은 프로세스의 다른 사용자(단독 CLI 테스트 등)까지
        # 오염된다. 기본값은 위 상수, override 는 e2e 의 --drift-yaw-rate /
        # --drift-kp-yaw 가 이 두 속성에만 쓴다.
        self.drift_yaw_rate = DRIFT_YAW_RATE
        self.drift_kp_yaw = DRIFT_KP_YAW
        self._cmd_pub = None
        self._Twist = None
        if fn is not None and not dry_run:
            from geometry_msgs.msg import Twist
            self._Twist = Twist
            self._cmd_pub = fn.node.create_publisher(Twist, "/cmd_vel_direct", 10)

    # ---- 저수준 ----
    def _publish(self, vx: float, vy: float, wz: float):
        if self._cmd_pub is None:
            return
        t = self._Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
        self._cmd_pub.publish(t)

    def _stop(self):
        """0 트위스트 1회 — 위 주의사항대로 그 뒤로는 발행하지 않는다."""
        self._publish(0.0, 0.0, 0.0)

    def _pose_age(self) -> float:
        return time.monotonic() - self.fn.stamp.get("status", 0.0)

    def release_goal(self):
        """arena 노드의 goal 추종 해제 — 우리 cmd_vel 과 싸우지 않게."""
        if self.fn is not None and not self.dry_run:
            self.fn.publish("control", "STOP")
            time.sleep(0.2)

    # ---- 핵심: yaw 고정 폐루프 이동 ----
    def drive_to(self, target_xy, tag: str, v_max: float = 0.45,
                 timeout: float = 25.0, along: str = "forward",
                 hold_yaw: float = HEADING_NORTH_RAD,
                 pos_tol: float = POS_TOL_M,
                 yaw_tol: float = YAW_ARRIVE_TOL_RAD) -> dict:
        """yaw 를 hold_yaw(기본 북향)로 고정한 채 target_xy 로 이동한다.

        전진/후진/횡이동을 구분하지 않는다 — 몸체 좌표 오차를 그대로 vx/vy 로
        낸다. 메카넘은 holonomic 이라 세 성분이 독립으로 합성된다
        (mecanum_bridge_node.cmd_vel_callback -> body_to_wheels 확인, 2026-07-21).

        along: 이 레그의 **진행축**. "forward"=몸체 전방(street/후진),
               "lateral"=몸체 횡(하이웨이). 속도상한 v_max 는 진행축에,
               MAX_CROSS_MPS 와 안전 게이트는 그 **수직축**에 적용된다.
        hold_yaw: [2026-07-22 신규] 유지할 heading. 운반 서진 레그(서향 180°)
               를 위해 일반화 — body_error/e_yaw 가 hold_yaw 기준이라 수식은
               북향과 동일하게 성립한다.

        [2026-07-22 v2 — 00:50 실기 반영]
        · 이탈 abort 폐지 → **횡복구**: |e_cross| 가 CROSS_SLOW_FULL(0.15)을
          넘으면 진행축 속도만 0으로 하고 루프를 유지하며 횡만 보정한다.
          실패 반환은 로컬 튐(무장 후 CROSS_JUMP)·stale·timeout 뿐이다.
          (종전: 실패 반환 → 호출부 "계속" → 필드 한복판에서 대각 운반 계획)
        · 진행속도 적응: |e_cross| 0.05~0.15 구간에서 진행축을 선형 감속 —
          보정 권한을 횡에 몰아주고, 속도 비례 드리프트 자체도 줄인다.
        · [2026-07-22 03시] 틱 점프 감지기는 제거 — IMU 가 yaw 점프를 막고,
          거리 점프의 원인(7.5cm 양자화 버스트)은 localizer 미세 refine 으로
          소멸. 잔여 방어는 무장 후 CROSS_JUMP_M 게이트뿐.
        · 텔레메트리: 매 틱 (t,x,y,yaw,e_along,e_cross,vx,vy,wz) 기록.
          성공 시 stats 만, 실패/복구 발생 시 trace 전체를 반환에 포함.

        반환: {"ok","reason","pose","sec","max_cross_m","recoveries",("trace")}
        """
        if self.dry_run:
            self.log(f"  [dry-run] drive_to {tag} → ({target_xy[0]:+.2f},{target_xy[1]:+.2f})")
            return {"ok": True, "reason": "dry-run", "pose": None, "sec": 0.0}
        if along not in ("forward", "lateral"):
            raise ValueError(f"along 은 'forward'|'lateral' 이어야 한다: {along!r}")

        t0 = time.monotonic()
        period = 1.0 / CONTROL_HZ
        armed = False          # 게이트 무장 여부 (코스 진입 후 True)
        n_stale = 0
        in_recovery = False    # 횡복구 에피소드 진행 중 (집계/로그 1회용)
        recoveries = 0
        max_cross = 0.0        # 무장 후 최대 |e_cross|
        trace = []             # [t, x, y, yaw°, e_along, e_cross, vx, vy, wz]
        gate_ref = None        # 위치 게이트: 마지막 채택 (t, x, y)
        gate_reject = 0        # 연속 기각 수 (POSE_GATE_MAX_REJECT 에서 강제 재동기)
        gate_dropped = 0       # 이 레그에서 대체한 표본 총수 (텔레메트리)

        def _result(ok, reason, pose, elapsed):
            r = {"ok": ok, "reason": reason, "pose": pose, "sec": elapsed,
                 "max_cross_m": round(max_cross, 3), "recoveries": recoveries,
                 "pose_gate_dropped": gate_dropped}
            if not ok or recoveries:
                r["trace"] = trace   # 실기 분석용 — 성공+무복구면 생략(리포트 절약)
            return r

        # [2026-07-22 03시] 틱 점프 감지기 제거 — 상수 블록의 폐기 사유 참조.
        # 위치 급변 방어는 무장 후 CROSS_JUMP_M 게이트 하나만 남긴다.
        while True:
            elapsed = time.monotonic() - t0
            if elapsed > timeout:
                self._stop()
                return _result(False, f"timeout {timeout:.0f}s",
                               self.fn.pose(), elapsed)

            self.fn.spin_for(period)
            pose = self.fn.pose()
            if pose is None or self._pose_age() > POSE_STALE_ABORT_S:
                # pose 없음/낡음 → 보정하지 말고 정지. 튐을 따라가면 옆으로 밀린다.
                n_stale += 1
                self._stop()
                if n_stale > int(CONTROL_HZ * 1.5):   # 1.5초 이상 지속 = 실패
                    return _result(False, "pose stale/없음", pose, elapsed)
                continue
            n_stale = 0

            # ---- 위치 이상표본 게이트 (상수 블록 주석 참조) ----
            # 위치가 바뀐 표본만 판정한다. 미갱신 틱(제어의 ~47%)은 그대로 통과.
            now_t = time.monotonic()
            if gate_ref is None:
                gate_ref = (now_t, pose[0], pose[1])
            elif pose[0] != gate_ref[1] or pose[1] != gate_ref[2]:
                d = math.hypot(pose[0] - gate_ref[1], pose[1] - gate_ref[2])
                dt = max(1e-3, now_t - gate_ref[0])
                if (d > POSE_GATE_V_MAX * dt + POSE_GATE_MARGIN_M
                        and gate_reject < POSE_GATE_MAX_REJECT):
                    # 의심 — x/y 만 직전 채택값으로 대체하고 계속 달린다.
                    # (세우지 않는다. 실패로 만들지 않는다. yaw 는 새 값 유지.)
                    gate_reject += 1
                    gate_dropped += 1
                    pose = (gate_ref[1], gate_ref[2], pose[2])
                else:
                    gate_reject = 0
                    gate_ref = (now_t, pose[0], pose[1])

            e_fwd, e_left = body_error(pose, target_xy, hold_yaw)
            e_yaw = wrap_angle(hold_yaw - pose[2])
            pos_err = math.hypot(e_fwd, e_left)
            # 진행축/횡축 분해 (2026-07-22). yaw 고정이라 몸체↔map 대응은 고정.
            if along == "lateral":
                e_along, e_cross = e_left, e_fwd
            else:
                e_along, e_cross = e_fwd, e_left
            ac = abs(e_cross)

            # 병진 도착 판정 — pos_tol/yaw_tol 은 레그 성격별 인자 (2026-07-22
            # v3: 중간 레그는 rough 8cm/12°, mini-goal 은 6cm/8°. 양자 7.5cm
            # 미만의 허용은 도달 불가 대기를 만든다). 직후 turn_to/시각측정이
            # yaw 를 각자 정밀하게 잡는다.
            if pos_err < pos_tol and abs(e_yaw) < yaw_tol:
                self._stop()
                return _result(True, "도착", pose, elapsed)

            # 게이트 무장: 코스에 한 번 들어온 뒤부터 — 레그 시작 시점의 큰
            # 초기 오차는 정상이고, 튐으로 오판하면 첫 루프에서 즉사한다.
            if not armed and ac <= CROSS_ARM_M:
                armed = True
            if armed:
                max_cross = max(max_cross, ac)
                if ac > CROSS_JUMP_M:
                    # 정상 주행으론 못 만드는 이탈 — 로컬 튐. 보정 추종 금지.
                    self._stop()
                    return _result(False,
                                   f"횡오차 급증 {e_cross:+.3f}m (>{CROSS_JUMP_M}) — 로컬 튐 의심",
                                   pose, elapsed)

            wz = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, KP_YAW * e_yaw))
            if abs(e_yaw) > YAW_DRIVE_MAX_RAD:
                # yaw 가 크게 어긋난 채 병진하면 그리퍼가 옆으로 튀어나온다(I-1).
                # hold_yaw 로 되돌린 뒤에 다시 간다 (2026-07-22 신규).
                # [2026-07-23 조작자 지시 — "제자리 회전이 각속도 예산보다 느리다"
                # 로그 검증] 이 분기는 병진을 0 으로 두는 **순수 제자리 회전**인데
                # 위에서 계산한 주행용 캡/게인(MAX_YAW_RATE 0.8 / KP_YAW 1.5)을
                # 그대로 쓰고 있었다. 07-22 저녁의 "제자리 회전 1.5배"와 07-23 의
                # 게인 상향은 turn_to 와 펌웨어 position move 경로에만 걸려
                # 이 게이트는 전혀 안 빨라졌다. 병진이 0 이라 드리프트/횡오차
                # 특성과 무관하므로 회전 전용 상수를 그대로 쓴다 (turn_to 와 동일).
                # 효과: 135° 회전 2.90s → 1.90s (0.8/1.5 → 1.2/2.5 모델값).
                # 주행 중 yaw 유지(아래 병진 구간)는 종전 값 그대로 — 병진과
                # 결합한 고게인은 지그재그를 키운다 (TURN_KP_YAW 상수 주석).
                # [폐기 2026-07-23 — 롤백 시 이 두 줄을 지우면 위 wz 가 그대로 쓰인다]
                wz = max(-TURN_YAW_RATE,
                         min(TURN_YAW_RATE, TURN_KP_YAW * e_yaw))
                if abs(wz) < MIN_YAW_RATE:
                    wz = math.copysign(MIN_YAW_RATE, wz)
                self._publish(0.0, 0.0, wz)
                continue

            v_along = max(-v_max, min(v_max, KP_ALONG * e_along))
            # 감산형 데드밴드 (2026-07-22 수정안 6): 양자 노이즈 무시.
            e_c_cmd = math.copysign(max(0.0, ac - CROSS_DEADBAND_M), e_cross)
            v_cross = max(-MAX_CROSS_MPS, min(MAX_CROSS_MPS, KP_CROSS * e_c_cmd))
            # ---- 진행속도 적응 (2026-07-22 "강한 adoption") ----
            # slow_f: 1(횡오차 작음, 전속) → 0(0.15 이상, 진행 정지=횡복구).
            slow_f = 1.0
            if ac > CROSS_SLOW_START_M:
                slow_f = max(0.0, (CROSS_SLOW_FULL_M - ac)
                             / (CROSS_SLOW_FULL_M - CROSS_SLOW_START_M))
            if slow_f <= 0.0 and not in_recovery:
                in_recovery = True
                recoveries += 1
                self.log(f"  [street] {tag} 횡이탈 {e_cross:+.2f}m — "
                         f"진행 정지·횡복구 (복구 {recoveries}회째)")
            elif in_recovery and ac <= CROSS_RECOVER_EXIT_M:
                # [2026-07-23] 해제 문턱을 CROSS_ARM_M(무장) 재사용에서 전용
                # 상수로 분리 — 발동 0.08 과 같아지면 경계에서 채터링한다.
                in_recovery = False   # 코스 복귀 — 진행 재개

            if pos_err < pos_tol:
                # 위치는 이미 허용오차 안 — yaw 만 남았다. 계속 기어가면
                # MIN_SPEED 하한 때문에 목표를 지나쳐 왕복한다 (2026-07-22).
                # ⚠ 부등호·문턱은 위 도착 판정과 **같아야** 한다. 다르면
                # pos_err 가 문턱 사이일 때 병진 정지+도착 미판정으로 멈춰 선다.
                v_along = v_cross = 0.0
                if 0.0 < abs(wz) < MIN_YAW_RATE:
                    wz = math.copysign(MIN_YAW_RATE, wz)   # 제자리 회전 정지마찰
            else:
                if slow_f >= 0.999 and 0.0 < abs(v_along) < MIN_SPEED_MPS:
                    # 정지마찰 하한은 **의도적 감속이 아닐 때만** — 적응 감속을
                    # 하한이 되살리면 복구 중에도 전진해 이탈이 계속 자란다.
                    v_along = math.copysign(MIN_SPEED_MPS, v_along)
                v_along *= slow_f

            if along == "lateral":
                vx_hold, vy_hold = v_cross, v_along
            else:
                vx_hold, vy_hold = v_along, v_cross
            # hold 프레임 → **실제 몸체** 프레임 회전 보정 (2026-07-22).
            # body_error 는 hold_yaw 기준인데 브릿지는 트위스트를 로봇의 현재
            # yaw 기준으로 해석한다. 회전각은 e_yaw = hold_yaw - 현재 yaw.
            # (종전 미보정 → 오프라인 시뮬 case6: yaw 170° 출발 0.55s 이탈)
            cy, sy = math.cos(e_yaw), math.sin(e_yaw)
            vx = vx_hold * cy - vy_hold * sy
            vy = vx_hold * sy + vy_hold * cy
            self._publish(vx, vy, wz)
            trace.append([round(elapsed, 2), round(pose[0], 3), round(pose[1], 3),
                          round(math.degrees(pose[2]), 1), round(e_along, 3),
                          round(e_cross, 3), round(vx, 2), round(vy, 2),
                          round(wz, 2)])

        # 도달 불가 (while True)

    def drive_free(self, target_xy, tag: str, v_max: float = 0.5,
                   timeout: float = 20.0,
                   hold_yaw: float = HEADING_NORTH_RAD,
                   pos_tol: float = POS_TOL_ROUGH_M,
                   yaw_tol: float = YAW_ARRIVE_ROUGH_RAD) -> dict:
        """자유밴드 전용 **직선 홀로노믹** 주행 — 대각 허용, 통로 게이트 없음.

        [2026-07-22 신규 — 조작자 지시] first-scan 진입(START→프리샷 x)을
        서진→북진 L 대신 단일 대각으로. 물체가 없는 남쪽 자유밴드
        (y <= HIGHWAY_Y_M) 안에서만 쓸 것 — street 횡복구/게이트가 없다.
        yaw 는 hold_yaw(기본 북) 고정(I-1), I-5 회전우선 게이트 동일 적용.
        stale 가드는 drive_to 와 같다 (틱 점프 감지기는 2026-07-22 03시 제거).
        """
        if self.dry_run:
            self.log(f"  [dry-run] drive_free {tag} → ({target_xy[0]:+.2f},{target_xy[1]:+.2f})")
            return {"ok": True, "reason": "dry-run", "pose": None, "sec": 0.0}
        t0 = time.monotonic()
        period = 1.0 / CONTROL_HZ
        n_stale = 0
        # [2026-07-22 03시] 틱 점프 감지기 제거 (drive_to 와 동일 사유) —
        # 자유밴드 전용이라 잔여 방어는 stale/timeout 으로 충분.
        while True:
            elapsed = time.monotonic() - t0
            if elapsed > timeout:
                self._stop()
                return {"ok": False, "reason": f"timeout {timeout:.0f}s",
                        "pose": self.fn.pose(), "sec": elapsed}
            self.fn.spin_for(period)
            pose = self.fn.pose()
            if pose is None or self._pose_age() > POSE_STALE_ABORT_S:
                n_stale += 1
                self._stop()
                if n_stale > int(CONTROL_HZ * 1.5):
                    return {"ok": False, "reason": "pose stale/없음",
                            "pose": pose, "sec": elapsed}
                continue
            n_stale = 0

            e_fwd, e_left = body_error(pose, target_xy, hold_yaw)
            e_yaw = wrap_angle(hold_yaw - pose[2])
            pos_err = math.hypot(e_fwd, e_left)
            if pos_err < pos_tol and abs(e_yaw) < yaw_tol:
                self._stop()
                return {"ok": True, "reason": "도착", "pose": pose,
                        "sec": elapsed}
            wz = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, KP_YAW * e_yaw))
            if abs(e_yaw) > YAW_DRIVE_MAX_RAD:
                if abs(wz) < MIN_YAW_RATE:
                    wz = math.copysign(MIN_YAW_RATE, wz)
                self._publish(0.0, 0.0, wz)
                continue
            # 직선 벡터 P 제어: 오차 방향 그대로, 크기만 v_max 로 클램프.
            vx_hold = KP_ALONG * e_fwd
            vy_hold = KP_ALONG * e_left
            vn = math.hypot(vx_hold, vy_hold)
            if vn > v_max:
                vx_hold, vy_hold = vx_hold / vn * v_max, vy_hold / vn * v_max
            elif 0.0 < vn < MIN_SPEED_MPS:
                vx_hold, vy_hold = (vx_hold / vn * MIN_SPEED_MPS,
                                    vy_hold / vn * MIN_SPEED_MPS)
            cy, sy = math.cos(e_yaw), math.sin(e_yaw)
            vx = vx_hold * cy - vy_hold * sy
            vy = vx_hold * sy + vy_hold * cy
            self._publish(vx, vy, wz)

    # ---- 조작자 설계의 각 단계 ----
    def go_to_mini_goal(self, cell_cm, v_max: float = None,
                        stop_at_entry: bool = False,
                        mirror: bool = False, descend: bool = False) -> dict:
        """하이웨이에서 x 정렬 → street 북진 → mini-goal.

        x 를 하이웨이(물체 없음)에서 먼저 0으로 만든 뒤 북진하므로, street 진입
        시점의 횡오차가 이미 작다. 진입 후에는 P 보정이 잔차만 잡는다.

        레그1 은 map x 를 맞추는 **횡이동**이라 along="lateral" 이다
        (2026-07-22 — 이 인자가 없어 레그1 이 첫 루프에서 즉사했다).

        stop_at_entry: [폐기 2026-07-23 — 호출부 없음] 구 e2e 실험 기능 'south'
        전용이었다. 남단 행 예외가 "mini-goal 을 하이웨이로 내리는" 방식에서
        "mini-goal 은 표준(공식 75) 유지 + 운반만 즉시 드리프트"로 재정의되며
        e2e 가 더 이상 넘기지 않는다. 인자는 단독 테스트용으로만 남긴다.
        (기본 False = 종전 동작).
        y=100 남단 행에서 street 진입 지점 (gx, HIGHWAY_Y_M) 자체를
        mini-goal 로 쓰고 북진 레그를 생략한다. 진입점이 파지 시작점이
        되므로 레그1 을 최종 톨러런스(POS_TOL_M/YAW_ARRIVE_TOL_RAD)로 정착.

        mirror/descend: [2026-07-23 조작자 지시] 스캔 직후 "내려오면서 파지".
        mirror 는 mini-goal 을 남서 대각으로 (물체 동쪽 street 사용),
        descend 는 **이미 같은 street 위 북쪽**에 있을 때 하이웨이 왕복
        (남하→x정렬→북진)을 통째로 생략하고 남하 한 레그로 붙인다.
        호출부(e2e collect_one)가 기하 조건을 검사한 뒤에만 넘긴다.
        """
        gx, gy = mini_goal_for(cell_cm, mirror=mirror)
        p_tol1 = POS_TOL_M if stop_at_entry else POS_TOL_ROUGH_M
        y_tol1 = YAW_ARRIVE_TOL_RAD if stop_at_entry else YAW_ARRIVE_ROUGH_RAD
        # 하이웨이는 물체가 없어 빠르게, street 는 횡보정 권한을 남기는 속도로.
        v_hw = V_HIGHWAY_MPS if v_max is None else v_max
        v_st = V_STREET_MPS if v_max is None else v_max
        self.release_goal()
        # [2026-07-23 신규] 하산 진입 — 같은 street 북쪽에서 남하 한 레그.
        # 스캔점(공식 225,225)에서 x=200/250 행의 mini-goal 은 같은 street
        # (x=225) 위 남쪽이라, 종전 경로는 하이웨이(-1.40)까지 내려갔다가
        # 같은 x 로 다시 북진하는 왕복이 통째로 낭비였다.
        if descend:
            self.log(f"  [street] 하산 진입 → mini-goal ({gx:+.2f},{gy:+.2f})"
                     f" — 하이웨이 왕복 생략 ({v_st} m/s)")
            rd = self.drive_to((gx, gy), "street-descend", v_max=v_st,
                               pos_tol=POS_TOL_M, yaw_tol=YAW_ARRIVE_TOL_RAD)
            return merge_legs("street-descend", rd["ok"],
                              [("street-descend", rd)])
        # [2026-07-22 오후 신규 — 조작자 승인] 적재 후 복귀 드리프트.
        # 적재 직후엔 자유밴드 안(y<=-1.35)에서 yaw -135° — 종전엔 레그1
        # drive_to 의 회전우선 게이트(I-5)가 제자리 135° 재회전을 만들었다.
        # 운반 드리프트의 역방향(같은 자유밴드 회랑)으로 하이웨이 x 정렬과
        # 북향 복원을 동시에 수행해 제자리 회전 3~5s 를 제거한다.
        # 도착 잔차 yaw <=12°(rough)는 게이트 15° 미만이라 이어지는 street
        # 북진이 주행 중 흡수. 일반 사이클(이미 북향)은 종전 경로 그대로다.
        pose = None if (self.dry_run or self.fn is None) else self.fn.pose()
        # 문턱 +0.15 (2026-07-22 오후): 얕은 핀(2·3) 후퇴 후 y≈-1.32 가
        # +0.05(-1.35) 밖이라 드리프트 미발동 — 물체 최남단행(연 -1.04)까지
        # 하강 경로 여유 확인 후 -1.25 로 완화 (e2e street_home 스킵과 동일값).
        drift_return = (pose is not None
                        and pose[1] <= HIGHWAY_Y_M + 0.15
                        and abs(wrap_angle(HEADING_NORTH_RAD - pose[2]))
                        > YAW_DRIVE_MAX_RAD)
        # [폐기 2026-07-22 오후 — 롤백 시 아래 분기를 지우고 이 호출 복원]
        # r1 = self.drive_to((gx, HIGHWAY_Y_M), "highway-x", v_max=v_hw,
        #                    along="lateral", pos_tol=POS_TOL_ROUGH_M,
        #                    yaw_tol=YAW_ARRIVE_ROUGH_RAD)  # 중간 레그 rough (2026-07-22)
        if drift_return:
            self.log(f"  [street] 복귀 드리프트 → x={gx:+.2f} (하이웨이 x 정렬"
                     f" + 북향 복원 동시, {v_hw} m/s)")
            r1 = self.drive_drift((gx, HIGHWAY_Y_M), HEADING_NORTH_RAD,
                                  "복귀_드리프트", v_max=v_hw,
                                  pos_tol=p_tol1, yaw_tol=y_tol1)
        else:
            self.log(f"  [street] 하이웨이 x 정렬 → x={gx:+.2f} ({v_hw} m/s, 횡이동)")
            r1 = self.drive_to((gx, HIGHWAY_Y_M), "highway-x", v_max=v_hw,
                               along="lateral", pos_tol=p_tol1,
                               yaw_tol=y_tol1)   # 중간 레그 rough (2026-07-22)
        # [2026-07-23] 반환을 merge_legs 경유로 — 종전 `**r1`/`**r2` 는 마지막
        # 레그 통계만 남겨 하이웨이 레그의 이탈/trace 를 버렸다 (헬퍼 주석 참조).
        leg1_tag = "복귀_드리프트" if drift_return else "highway-x"
        if not r1["ok"]:
            return merge_legs("highway-x", False, [(leg1_tag, r1)])
        if stop_at_entry:
            self.log(f"  [street] 남단 행 — 진입점 ({gx:+.2f},{HIGHWAY_Y_M:+.2f})"
                     f" = mini-goal (북진 생략)")
            return merge_legs("highway-entry", True, [(leg1_tag, r1)])
        self.log(f"  [street] 북진 → mini-goal ({gx:+.2f},{gy:+.2f}) ({v_st} m/s)")
        r2 = self.drive_to((gx, gy), "street-north", v_max=v_st)
        return merge_legs("street-north", r2["ok"],
                          [(leg1_tag, r1), ("street-north", r2)])

    def turn_to(self, heading_rad: float, tag: str, timeout: float = 12.0) -> dict:
        """제자리 회전 (병진 0). mini-goal ±45도 + 하이웨이 서향(2026-07-22).

        ⚠ 회전 스윕 반경 26cm, mini-goal 대각 인접 물체까지 35.36cm, 물체 반폭
          4cm → 여유 5.4cm. 스캔 스핀이 같은 기하로 통과해온 값이다.
          하이웨이(y=-1.40)에서의 회전은 최남단 실루엣까지 10cm 마진.
        """
        if self.dry_run:
            self.log(f"  [dry-run] turn_to {tag} → {math.degrees(heading_rad):+.0f}deg")
            return {"ok": True, "reason": "dry-run", "sec": 0.0}
        t0 = time.monotonic()
        period = 1.0 / CONTROL_HZ
        while True:
            elapsed = time.monotonic() - t0
            if elapsed > timeout:
                self._stop()
                return {"ok": False, "reason": f"timeout {timeout:.0f}s", "sec": elapsed}
            self.fn.spin_for(period)
            pose = self.fn.pose()
            if pose is None or self._pose_age() > POSE_STALE_ABORT_S:
                self._stop()
                continue
            e_yaw = wrap_angle(heading_rad - pose[2])
            if abs(e_yaw) < YAW_TOL_RAD:
                self._stop()
                return {"ok": True, "reason": "도착",
                        "yaw_deg": math.degrees(pose[2]), "sec": elapsed}
            # [2026-07-22 저녁] 제자리 회전은 TURN_YAW_RATE(=0.8x1.5) 상한
            # [2026-07-23] 게인도 전용값(TURN_KP_YAW) — 상한만 올렸을 때
            # 회전이 안 빨라진 원인이 게인이었다 (상수 주석 참조).
            wz = max(-TURN_YAW_RATE, min(TURN_YAW_RATE, TURN_KP_YAW * e_yaw))
            if abs(wz) < MIN_YAW_RATE:    # 정지마찰 — 너무 느리면 안 돈다
                wz = math.copysign(MIN_YAW_RATE, wz)   # (2026-07-22 상수화)
            self._publish(0.0, 0.0, wz)

    def face_object(self, mirror: bool = False) -> dict:
        """mini-goal 에서 CCW 45도 → 물체(북서 대각) 정면.

        mirror=True(남서 대각 mini-goal)면 물체가 북동 대각이라 CW 45도."""
        turn = FACE_OBJECT_TURN_MIRROR_RAD if mirror else FACE_OBJECT_TURN_RAD
        return self.turn_to(HEADING_NORTH_RAD + turn,
                            "face-object" + ("-mirror" if mirror else ""))

    def face_north(self) -> dict:
        """물체 정면에서 CW 45도 → 북향 복귀."""
        return self.turn_to(HEADING_NORTH_RAD, "face-north")

    def face_west(self) -> dict:
        """하이웨이에서 서향(180°) — 운반 서진 레그 진입용 (2026-07-22 신규).

        조작자 지시: 적재 기준점까지 횡이동(북향 유지) 대신 서향으로 돌아
        전진한다. 하이웨이는 자유밴드라 회전 스윕(26cm) 안전.
        [2026-07-22 04시] 운반 경로는 drive_drift 로 대체 — 이 메서드는
        단독 테스트/폴백용으로 유지."""
        return self.turn_to(math.pi, "face-west")

    def drive_drift(self, target_xy, target_yaw: float, tag: str,
                    v_max: float = V_STREET_MPS, timeout: float = 20.0,
                    pos_tol: float = POS_TOL_ROUGH_M,
                    yaw_tol: float = YAW_ARRIVE_ROUGH_RAD) -> dict:
        """자유밴드 전용 **드리프트** 주행 — 직선 병진 + 동시 yaw 수렴.

        [2026-07-22 04시 신규 — 조작자 지시] 하이웨이 운반 종단을 '서향 90°
        회전 후 전진'에서 드리프트로 교체: 제자리 회전 정지시간(±2~7s)이
        사라지고, 적재 정면(target_yaw=-135°)을 **이동 중에** 만들어 도착
        직후 바로 적재 전진이 가능하다.
        ⚠ 물체 없는 남쪽 자유밴드(y<=HIGHWAY_Y_M 부근) 전용 — yaw 북향
        불변식(I-1)과 회전 게이트(I-5)를 적용하지 않는다. street 진입 금지.
        """
        if self.dry_run:
            self.log(f"  [dry-run] drive_drift {tag} → "
                     f"({target_xy[0]:+.2f},{target_xy[1]:+.2f}, "
                     f"{math.degrees(target_yaw):+.0f}°)")
            return {"ok": True, "reason": "dry-run", "pose": None, "sec": 0.0}
        t0 = time.monotonic()
        period = 1.0 / CONTROL_HZ
        n_stale = 0
        while True:
            elapsed = time.monotonic() - t0
            if elapsed > timeout:
                self._stop()
                return {"ok": False, "reason": f"timeout {timeout:.0f}s",
                        "pose": self.fn.pose(), "sec": elapsed}
            self.fn.spin_for(period)
            pose = self.fn.pose()
            if pose is None or self._pose_age() > POSE_STALE_ABORT_S:
                n_stale += 1
                self._stop()
                if n_stale > int(CONTROL_HZ * 1.5):
                    return {"ok": False, "reason": "pose stale/없음",
                            "pose": pose, "sec": elapsed}
                continue
            n_stale = 0
            ex = target_xy[0] - pose[0]
            ey = target_xy[1] - pose[1]
            pos_err = math.hypot(ex, ey)
            e_yaw = wrap_angle(target_yaw - pose[2])
            if pos_err < pos_tol and abs(e_yaw) < yaw_tol:
                self._stop()
                return {"ok": True, "reason": "도착", "pose": pose,
                        "sec": elapsed}
            # [폐기 2026-07-23 — 롤백은 --drift-yaw-rate 0.8 --drift-kp-yaw 1.5]
            # wz = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, KP_YAW * e_yaw))
            # [2026-07-23 조작자 지시] 드리프트 전용 캡/게인. 이 레그의 하한을
            # 잡고 있던 것이 병진이 아니라 yaw 였다 (상수 블록 실측 근거).
            #
            # [2026-07-23 오후 수정 — 조작자 "드리프트해서 street 진입할 때 yaw
            # 품질이 떨어졌어"] 위 값을 레그 **전 구간**에 걸었던 것이 원인이다.
            # 큰 오차를 쓸어내는 구간(시간을 먹는 곳)과, yaw 가 이미 수렴해
            # 병진만 남은 종단 구간은 요구가 정반대다:
            #   · 종단에서 고게인 = 병진과 결합해 지그재그 (TURN_KP_YAW 주석의
            #     "주행 중 yaw 유지에는 쓰지 않는다"가 정확히 이 경우다)
            #   · 종단의 고 wz = 라이다 스캔 스미어(A2M12 100ms/회전: 0.8 →
            #     4.6°, 1.5 → 8.6°)로 wall_range 정합 열화 → 도착 yaw 품질 저하
            # 그래서 |e_yaw| 가 I-5 문턱(15°)을 넘는 동안만 드리프트 전용값을
            # 쓰고, 그 아래로 들어오면 **종전 주행용 값**으로 되돌린다. 시간
            # 이득은 대부분 보존된다 (135°→15° 구간이 소요의 90%).
            # 롤백(전 구간 종전값) = --drift-yaw-rate 0.8 --drift-kp-yaw 1.5.
            if abs(e_yaw) > YAW_DRIVE_MAX_RAD:
                wz = max(-self.drift_yaw_rate,
                         min(self.drift_yaw_rate, self.drift_kp_yaw * e_yaw))
            else:
                wz = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, KP_YAW * e_yaw))
            # world 오차 벡터 P → 크기 클램프 → 현재 몸체 프레임으로 회전.
            vxw = KP_ALONG * ex
            vyw = KP_ALONG * ey
            vn = math.hypot(vxw, vyw)
            if pos_err < pos_tol:
                vxw = vyw = 0.0            # 위치 도달 — yaw 만 마무리
                if 0.0 < abs(wz) < MIN_YAW_RATE:
                    wz = math.copysign(MIN_YAW_RATE, wz)
            elif vn > v_max:
                vxw, vyw = vxw / vn * v_max, vyw / vn * v_max
            elif 0.0 < vn < MIN_SPEED_MPS:
                vxw, vyw = (vxw / vn * MIN_SPEED_MPS,
                            vyw / vn * MIN_SPEED_MPS)
            cy, sy = math.cos(pose[2]), math.sin(pose[2])
            vx = vxw * cy + vyw * sy       # world → body
            vy = -vxw * sy + vyw * cy
            self._publish(vx, vy, wz)

    def retreat_to_highway(self, v_max: float = None) -> dict:
        """현재 x 를 유지한 채 **후진**으로 하이웨이까지 내려온다.

        yaw 는 북향 그대로라 물린 물체를 든 그리퍼가 후미가 된다 (진행 방향
        선행부가 본체라 통로 여유는 전진과 동일).
        """
        pose = self.fn.pose() if self.fn is not None else None
        if pose is None and not self.dry_run:
            return {"ok": False, "reason": "pose 없음", "phase": "retreat"}
        x = pose[0] if pose else 0.0
        # 물체를 물고 내려오므로 street 속도를 쓴다 (하이웨이 속도 아님).
        v = V_STREET_MPS if v_max is None else v_max
        self.log(f"  [street] 후진 하이웨이 복귀 → ({x:+.2f},{HIGHWAY_Y_M:+.2f}) ({v} m/s)")
        r = self.drive_to((x, HIGHWAY_Y_M), "retreat", v_max=v,
                          pos_tol=POS_TOL_ROUGH_M,
                          yaw_tol=YAW_ARRIVE_ROUGH_RAD)   # 중간 레그 rough
        return {"ok": r["ok"], "phase": "retreat", **r}


# =========================================================================
# 단독 실기 테스트 CLI — 파지 없이 "이동만" 검증한다
# =========================================================================
def _preflight(fn, log) -> bool:
    """실기 투입 전 자동 점검. 하나라도 실패하면 주행하지 않는다."""
    ok = True
    fn.spin_for(1.5)
    pose = fn.pose()
    if pose is None:
        log("✗ pose 없음 — arena_control_node 미기동이거나 START pose 미시드")
        ok = False
    else:
        log(f"✓ pose ({pose[0]:+.2f},{pose[1]:+.2f}, {math.degrees(pose[2]):+.0f}deg)")

    # status 주기 실측 — 20Hz 가 아니면 P 보정 피드백이 느리다
    t0, n0 = time.monotonic(), fn.count.get("status", 0)
    fn.spin_for(2.0)
    hz = (fn.count.get("status", 0) - n0) / max(1e-6, time.monotonic() - t0)
    if hz < 12.0:
        log(f"⚠ status {hz:.1f}Hz — 4Hz 기본값으로 보인다. arena 스택을 "
            "status_period_sec:=0.05 로 재기동해야 제어 피드백이 제대로 붙는다.")
        log("   (그대로 진행은 가능하지만 횡보정이 느리고 오버슈트가 날 수 있다)")
    else:
        log(f"✓ status {hz:.1f}Hz")

    st = fn.arena_status()
    imu = (st.get("imu_prior") or {})
    log(f"{'✓' if imu.get('fresh') else '⚠'} IMU prior fresh={imu.get('fresh')} "
        f"(yaw 유지에 필요)")
    return ok


def main():
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="street_nav 단독 실기 테스트 — 파지 없이 이동만 검증")
    ap.add_argument("--cell", default="200,150",
                    help="목표 공식 격자 셀 'x_cm,y_cm' (기본 200,150)")
    ap.add_argument("--speed", type=float, default=None,
                    help=f"주행 속도 m/s. 미지정 시 하이웨이 {V_HIGHWAY_MPS} / "
                         f"street {V_STREET_MPS}. 첫 검증은 --speed 0.4 권장")
    ap.add_argument("--no-turn", action="store_true",
                    help="mini-goal 에서의 ±45도 회전 생략 (이동만)")
    ap.add_argument("--no-retreat", action="store_true",
                    help="후진 복귀 생략 (mini-goal 에서 정지)")
    ap.add_argument("--dry-run", action="store_true",
                    help="ROS 없이 기하만 출력")
    args = ap.parse_args()

    try:
        cx, cy = (int(v) for v in args.cell.split(","))
    except ValueError:
        sys.exit("--cell 형식은 'x_cm,y_cm' (예: 200,150)")

    gx, gy = mini_goal_for((cx, cy))
    print(f"[목표] 공식 셀 ({cx},{cy}) → mini-goal map ({gx:+.3f},{gy:+.3f}) "
          f"= 공식 ({(gx + 2) * 100:.0f},{(gy + 2) * 100:.0f})")
    print(f"[목표] 물체는 mini-goal 에서 북서 대각 "
          f"{object_range_from_mini_goal():.3f}m "
          f"(방위 {math.degrees(object_bearing_from_mini_goal()):.0f}deg)")

    if args.dry_run:
        nav = StreetNavigator(None, dry_run=True)
        nav.go_to_mini_goal((cx, cy))
        nav.face_object()
        nav.face_north()
        print("[dry-run] 기하 검증만 수행 — 실제 주행 없음")
        return

    import rclpy
    rclpy.init()
    fn = fl.FieldNode("street_nav_test")
    nav = StreetNavigator(fn)
    try:
        if not _preflight(fn, print):
            sys.exit("사전점검 실패 — 주행하지 않음")
        input("\n>>> 주행 시작하려면 Enter (중단은 Ctrl-C) ")

        t0 = time.monotonic()
        r = nav.go_to_mini_goal((cx, cy), v_max=args.speed)
        print(f"[결과] mini-goal 도달: {r['ok']} ({r.get('reason')}) "
              f"{r.get('sec', 0):.1f}s")
        if not r["ok"]:
            return

        p = fn.pose()
        if p is None:   # [2026-07-22] pose 유실 시 p[0] 인덱싱으로 죽던 자리
            print("[정밀도] pose 없음 — 정밀도 출력 생략")
        else:
            print(f"[정밀도] 도달 pose ({p[0]:+.3f},{p[1]:+.3f}) "
                  f"목표 대비 오차 {math.hypot(p[0] - gx, p[1] - gy) * 100:.1f}cm, "
                  f"yaw {math.degrees(p[2]):+.1f}deg")

        if not args.no_turn:
            print(f"[회전] CCW 45도 (물체 정면) …")
            print(f"       {nav.face_object()}")
            time.sleep(1.0)
            print(f"[회전] CW 45도 (북향 복귀) …")
            print(f"       {nav.face_north()}")

        if not args.no_retreat:
            r2 = nav.retreat_to_highway(v_max=args.speed)
            print(f"[결과] 하이웨이 복귀: {r2['ok']} ({r2.get('reason')})")

        print(f"\n[총 소요] {time.monotonic() - t0:.1f}s")
    except KeyboardInterrupt:
        print("\n조작자 중단 — 정지")
    finally:
        nav._stop()          # 0 트위스트 1회 (계속 발행 금지 — mux 점유 방지)
        time.sleep(0.2)
        try:
            fn.node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
