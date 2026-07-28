"""경기 공식 레이아웃 (rulebook.md) — 맵 좌표계 상수.

공식 좌표계: 원점 = 좌측 하단(보관함 쪽) 모서리, 단위 cm.
맵 좌표계: stadium.yaml (4x4m 벽이 -2..+2), 변환 = cm/100 - 2.0
(grid_match_strategy.md 단계 A1 — 2026-07-15 실기 검증 완료: 출발구역
포즈 시딩 후 실제 태극기 코너 = UI STORAGE 방향 일치 확인).
"""

ARENA_HALF_M = 2.0

GRID_XS_CM = tuple(range(50, 351, 50))   # 7개
GRID_YS_CM = tuple(range(100, 351, 50))  # 6개


def official_cm_to_map(x_cm: float, y_cm: float) -> tuple[float, float]:
    return x_cm / 100.0 - ARENA_HALF_M, y_cm / 100.0 - ARENA_HALF_M


GRID_POINTS_MAP = tuple(
    official_cm_to_map(x, y) for y in GRID_YS_CM for x in GRID_XS_CM
)

# 40x40cm 구역 (x1, y1, x2, y2) — 맵 좌표
STORAGE_RECT_MAP = (-2.0, -2.0, -1.6, -1.6)   # 좌하단, 태극기
START_RECT_MAP = (1.6, -2.0, 2.0, -1.6)       # 우하단

# 중앙 스캔 기준점 — 아레나 정중앙(200,200)이 아니라 그 북동쪽 street 교차점이다.
# 격자점이 50cm 간격이라 (200,200)에는 물체가 서 있고, 로봇은 물체 사이 통로에
# 서서 돈다. 실기 로그와 fieldlib.CENTER_SCAN_XY 둘 다 (0.25, 0.25).
CENTER_SCAN_MAP = official_cm_to_map(225.0, 225.0)
