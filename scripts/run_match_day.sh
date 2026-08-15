#!/usr/bin/env bash
# =============================================================================
# 경기 당일 원커맨드 실행기 — 브릿지 기동부터 E2E 시작 대기까지
#   bash scripts/run_match_day.sh
#
# 경기 진행(본부 공지):
#   1) 오브젝트(타깃) 공지  2) 1분 카운트(소프트웨어 수정)  3) 오브젝트 위치 공지
#   4) 조교 배치            5) "3 2 1 시작!" → 3분
#
# 이 스크립트의 대응:
#   [준비]  배치 전에 미리 실행 → 스택 기동·헬스체크·포즈 시딩까지 끝내 둔다.
#   [1분]   타깃 형상/과일을 숫자 두 번으로 입력한다 (공지 직후).
#   [위치]  위치 공지를 붙여넣을 수 있다 (선택 — 기록용 / 스캔 생략용).
#   [시작]  E2E 가 모델 프리로드까지 마치고 "[체크] ... (y/n)" 에서 대기한다.
#           **"시작!" 구호에 y** 를 누르면 그 즉시 출발한다 (종전과 동일).
#
# 파라미터는 2026-07-23 04:13 / 04:18 실기 런에서 실제로 쓴 값을 고정한 것이다
# (logs/field_ops/20260723_0413*/report.json 의 args). 바꾸려면 아래 상수만.
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

RUNNER="mission/match_runner.py"
AUTOPILOT="mission/field_autopilot.py"
STATE_FILE="logs/field_ops/.match_day_state"
ROS_SETUP="/opt/ros/humble/setup.bash"       # AGENTS.md: 기본 distro 는 humble

# ---- 경기 고정 파라미터 (실기 04:13/04:18 런과 동일) ------------------------
SPEED_PROFILE="fast"      # 주행 0.95 / 접근 0.7 / 적재 0.45
ORDER="fruit-first"       # 과일(20점) 세트 우선
SCAN_SHOTS=12             # 센터 스핀 촬영 수
SCAN_SETTLE=0.5           # 스캔 스텝 정착 대기 [s]
RELEASE_TICKS=600         # 적재 투하 상대 개방 [tick]
VOTES_K=1                 # 셀 확정 최소 득표
FRUIT_K=1                 # 과일면 확정 최소 히트
SHAPE_QUOTA=4             # 룰북 §5: 형상 4개
FRUIT_QUOTA=3             # 룰북 §7: 과일 3개
# [2026-07-23] "on"(AO+BP) → "ao"(AO만). 154504 실기 리플레이에서 구 BP(v2_1)가
# 정답 banana 를 pineapple conf 1.0 으로 뒤집어 2셀 손실 확인.
# [2026-07-24 갱신] "ao" → "bp". 확정 조합 = face 신규 / AO off / BP 신규
# (new/off/new). 본선 물체 예선 런(11:56)+15:45 두 실기 18조합 전수 리플레이에서
# 43/44 로 최적. face 재학습본이 apple→orange 오분류를 스스로 줄여 AO 는 순해로
# 뒤집혔고, 남은 pineapple→banana 오분류는 BP 실사 재학습본이 되돌린다.
# face/BP 기본값도 실사본으로 승격됨(fieldlib.py) — 이 스크립트는 환경변수
# 오버라이드를 주지 않아 러너 기본값 그대로 new/off/new 로 돈다.
# 근거: perception/docs/models.md
# [폐기 2026-07-24 — 구 값 "ao". 롤백 시 PAIR="ao"]
PAIR="bp"                 # banana↔pineapple 쌍 검증기(실사본)만, AO 는 비활성
RANGE_MODE="depth"        # 접점 depth 실측 역투영
NAV="street"              # street 주행 (navigation/docs/control-and-routing.md)
# [2026-07-24 조작자 지시] 정합성 인자(Consistency Factor) **기본 on**.
# 러너 자체의 argparse 기본값은 off 로 남긴다(오프라인 리플레이·A/B 보존) —
# 경기 실행 경로인 이 스크립트에서만 켠다. 켜지면 대상 선정이 "하산 우선 +
# 최근접"에서 `p·점수/T_exp` 기댓값 랭킹으로 바뀌고, 혼동쌍 부분계(과일 클래스당
# 정확히 3개)의 개수 제약으로 파트너 라벨 셀까지 probe 후보로 열린다. 파지는
# 여전히 근접 strict verify 통과 시에만 — "타깃 외 수거 금지" 불변식은 유지된다.
# 사양: mission/docs/match-strategy.md §2/§3/§6.5
# 되돌리기(경기 중 즉시): bash scripts/run_match_day.sh -- --consistency off
#   (-- 뒤 인자가 뒤에 붙어 argparse 가 마지막 값을 채택한다)
CONSISTENCY="on"
SCAN_MODE="step"
SCAN_MAST="up"

SHAPES=(cube octahedron dodecahedron icosahedron)
FRUITS=(apple orange banana pineapple)

C_OK=$'\033[1;32m'; C_WARN=$'\033[1;33m'; C_ERR=$'\033[1;31m'
C_HL=$'\033[1;36m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'

TARGET_SHAPE=""; TARGET_FRUIT=""; LAYOUT=""; USE_LAYOUT_AS_MAP=0
DO_BRINGUP=1; EXTRA_ARGS=(); CHECK_ONLY=0

usage() {
    cat <<EOF
사용법: bash scripts/run_match_day.sh [옵션]

  --shape <이름>     타깃 형상 (${SHAPES[*]}) — 지정 시 메뉴 생략
  --fruit <이름>     타깃 과일 (${FRUITS[*]}) — 지정 시 메뉴 생략
  --no-bringup       스택 기동 단계 생략 (이미 떠 있을 때)
  --check            실행 없이 인자/구성만 검증 (--offline 자가테스트)
  --dry-run          로봇 없이 러너 리허설 (--dry-run 전달)
  --                 이후 인자는 E2E 러너로 그대로 전달
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shape) TARGET_SHAPE="${2:-}"; shift 2 ;;
        --fruit) TARGET_FRUIT="${2:-}"; shift 2 ;;
        --no-bringup) DO_BRINGUP=0; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        --dry-run) EXTRA_ARGS+=(--dry-run); shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) echo "${C_ERR}알 수 없는 옵션: $1${C_OFF}"; usage; exit 2 ;;
    esac
done

banner() { printf '\n%s══ %s ══%s\n' "$C_HL" "$1" "$C_OFF"; }

# 한 글자 입력 (엔터 불필요). 실패 시 한 줄 입력으로 폴백.
# ⚠ 여기서 stdout 에 아무것도 쓰지 않는다 — pick_from 이 명령치환으로 값을
#   받으므로, 입력 에코가 stdout 으로 새면 선택값이 오염된다(초기 버그).
# ⚠ 내부 변수명은 __rk_* 로 고정한다 — local 로 잡은 이름이 호출자가 넘긴
#   변수명과 겹치면(예: read_key k) 지역변수를 대신 채워 호출자 값이 안 잡힌다.
read_key() {
    local __rk_var="$1" __rk_ch=""
    read -rsn1 __rk_ch 2>/dev/null || read -r __rk_ch
    printf -v "$__rk_var" '%s' "$__rk_ch"
}

pick_from() {   # pick_from <배열명> <프롬프트> → 선택값을 stdout
    local -n _arr="$1"; local prompt="$2" key="" i
    while true; do
        printf '%s\n' "$prompt" >&2
        for i in "${!_arr[@]}"; do
            printf '   %s%d%s) %s\n' "$C_HL" "$((i + 1))" "$C_OFF" "${_arr[$i]}" >&2
        done
        printf '  선택> ' >&2
        read_key key
        printf '%s\n' "$key" >&2
        if [[ "$key" =~ ^[1-9]$ ]] && (( key >= 1 && key <= ${#_arr[@]} )); then
            printf '%s\n' "${_arr[$((key - 1))]}"; return 0
        fi
        printf '%s  1~%d 중에서 골라주세요.%s\n' "$C_WARN" "${#_arr[@]}" "$C_OFF" >&2
    done
}

in_list() {   # in_list <값> <배열명>
    local v="$1"; local -n _a="$2"; local x
    for x in "${_a[@]}"; do [[ "$x" == "$v" ]] && return 0; done
    return 1
}

# =============================================================================
# 0. 환경
# =============================================================================
banner "0. 환경 준비"
if [[ ! -f "$ROS_SETUP" ]]; then
    echo "${C_ERR}✗ ROS 2 humble 없음: $ROS_SETUP${C_OFF}"; exit 1
fi
# ROS/colcon 의 setup.bash 는 미설정 변수를 참조한다 — set -u 를 잠시 끈다.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
if [[ -f "$REPO_ROOT/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/install/setup.bash"
    set -u
    echo "${C_OK}✓${C_OFF} ROS humble + 워크스페이스 소싱"
else
    set -u
    echo "${C_WARN}! install/setup.bash 없음 — colcon build 먼저 (계속은 가능)${C_OFF}"
fi
[[ -f "$RUNNER" ]] || { echo "${C_ERR}✗ 러너 없음: $RUNNER${C_OFF}"; exit 1; }

if (( CHECK_ONLY )); then
    banner "구성 검증 (--check)"
    python3 "$RUNNER" --offline \
        --speed-profile "$SPEED_PROFILE" --order "$ORDER" \
        --scan-shots "$SCAN_SHOTS" --scan-settle "$SCAN_SETTLE" \
        --release-ticks "$RELEASE_TICKS" --votes-k "$VOTES_K" --fruit-k "$FRUIT_K" \
        --shape-quota "$SHAPE_QUOTA" --fruit-quota "$FRUIT_QUOTA" \
        --pair "$PAIR" --range-mode "$RANGE_MODE" --nav "$NAV" \
        --scan-mode "$SCAN_MODE" --scan-mast "$SCAN_MAST" \
        --consistency "$CONSISTENCY" \
        --target-shape "${TARGET_SHAPE:-cube}" --target-fruit "${TARGET_FRUIT:-apple}"
    exit $?
fi

# =============================================================================
# 1. 스택 기동 (배치 전에 미리 — 여기서 시간을 다 쓴다)
# =============================================================================
if (( DO_BRINGUP )); then
    banner "1. 스택 기동 + 헬스체크 + 포즈 시딩"
    echo "${C_DIM}(브릿지 재사용 가능 — 이미 떠 있으면 그대로 씁니다)${C_OFF}"
    python3 "$AUTOPILOT" up --yes
    rc=$?
    if (( rc != 0 )); then
        echo "${C_ERR}✗ 스택 기동/헬스체크 실패 (exit $rc)${C_OFF}"
        echo "  조치 후 재실행하거나, 이미 정상이면 --no-bringup 으로 건너뛰세요."
        printf '  그래도 계속할까요? (y/N) '
        read_key k; printf '\n'
        [[ "$k" == "y" || "$k" == "Y" ]] || exit "$rc"
    else
        echo "${C_OK}✓ 스택 정상 — 이제 타깃 공지를 기다립니다${C_OFF}"
    fi
else
    banner "1. 스택 기동 생략 (--no-bringup)"
fi

# =============================================================================
# 2. 타깃 공지 반영 (1분 안에 끝내야 하는 단계)
# =============================================================================
banner "2. 타깃 공지 입력"
if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE" 2>/dev/null || true
    [[ -n "${LAST_SHAPE:-}" && -n "${LAST_FRUIT:-}" ]] && \
        echo "${C_DIM}직전 실행: ${LAST_SHAPE} + ${LAST_FRUIT}${C_OFF}"
fi

if [[ -n "$TARGET_SHAPE" ]] && ! in_list "$TARGET_SHAPE" SHAPES; then
    echo "${C_ERR}✗ 형상 이름 오류: $TARGET_SHAPE${C_OFF}"; TARGET_SHAPE=""
fi
if [[ -n "$TARGET_FRUIT" ]] && ! in_list "$TARGET_FRUIT" FRUITS; then
    echo "${C_ERR}✗ 과일 이름 오류: $TARGET_FRUIT${C_OFF}"; TARGET_FRUIT=""
fi
[[ -z "$TARGET_SHAPE" ]] && TARGET_SHAPE="$(pick_from SHAPES "[형상 세트] 목표 형상 (x${SHAPE_QUOTA}, 10점):")"
[[ -z "$TARGET_FRUIT" ]] && TARGET_FRUIT="$(pick_from FRUITS "[과일 세트] 목표 과일 (x${FRUIT_QUOTA}, 20점):")"
printf '%s✓ 타깃: %s x%d + %s x%d%s\n' \
    "$C_OK" "$TARGET_SHAPE" "$SHAPE_QUOTA" "$TARGET_FRUIT" "$FRUIT_QUOTA" "$C_OFF"

mkdir -p "$(dirname "$STATE_FILE")"
printf 'LAST_SHAPE=%s\nLAST_FRUIT=%s\n' "$TARGET_SHAPE" "$TARGET_FRUIT" > "$STATE_FILE"

# =============================================================================
# 3. 오브젝트 위치 공지 (선택)
# =============================================================================
banner "3. 오브젝트 위치 공지 (선택 — 엔터로 건너뜀)"
cat <<EOF
  형식: ${C_HL}cls:x,y;cls:x,y;...${C_OFF}  (공식 좌표 cm, 예: orange:250,200;octahedron:100,150)
  ${C_DIM}입력하면 run 폴더에 gt.json 으로 남고 스캔 정오표가 출력됩니다.${C_OFF}
EOF
printf '  위치> '
read -r LAYOUT
if [[ -n "${LAYOUT// /}" ]]; then
    n_cells=$(python3 - "$LAYOUT" <<'PY'
import sys
sys.path.insert(0, "perception")
try:
    import fieldlib as fl
    print(len(fl.parse_gt_text(sys.argv[1])))
except Exception as e:                      # noqa: BLE001
    print(f"ERR {e}")
PY
)
    if [[ "$n_cells" == ERR* || -z "$n_cells" ]]; then
        echo "${C_ERR}✗ 위치 파싱 실패 ($n_cells) — 위치 없이 진행합니다${C_OFF}"
        LAYOUT=""
    else
        echo "${C_OK}✓ ${n_cells}셀 인식${C_OFF}"
        cat <<EOF
  사용 방식:
   ${C_HL}1${C_OFF}) 스캔은 그대로 하고 ${C_HL}참고용${C_OFF}으로만 기록      ${C_DIM}(권장 — 실측 우선)${C_OFF}
   ${C_HL}2${C_OFF}) 이 위치를 ${C_HL}지도로 사용해 스캔 생략${C_OFF} (--gt-map)  ${C_WARN}(약 20~25s 절약 / 오타=전멸)${C_OFF}
EOF
        printf '  선택> '
        read_key k; printf '\n'
        if [[ "$k" == "2" ]]; then
            USE_LAYOUT_AS_MAP=1
            echo "${C_WARN}! 스캔 생략 모드 — 입력한 위치가 곧 지도입니다. 좌표를 다시 확인하세요.${C_OFF}"
        fi
    fi
else
    echo "${C_DIM}위치 입력 없음 — 스캔으로 직접 찾습니다 (기본).${C_OFF}"
fi

# =============================================================================
# 4. 실행
# =============================================================================
CMD=(python3 "$RUNNER"
     --target-shape "$TARGET_SHAPE" --target-fruit "$TARGET_FRUIT"
     --shape-quota "$SHAPE_QUOTA" --fruit-quota "$FRUIT_QUOTA"
     --order "$ORDER" --speed-profile "$SPEED_PROFILE"
     --scan-shots "$SCAN_SHOTS" --scan-settle "$SCAN_SETTLE"
     --release-ticks "$RELEASE_TICKS"
     --votes-k "$VOTES_K" --fruit-k "$FRUIT_K"
     --pair "$PAIR" --range-mode "$RANGE_MODE" --nav "$NAV"
     --scan-mode "$SCAN_MODE" --scan-mast "$SCAN_MAST"
     --consistency "$CONSISTENCY")
[[ -n "$LAYOUT" ]] && CMD+=(--gt-text "$LAYOUT")
(( USE_LAYOUT_AS_MAP )) && CMD+=(--gt-map)
(( ${#EXTRA_ARGS[@]} )) && CMD+=("${EXTRA_ARGS[@]}")

banner "4. 실행 명령"
# 붙여넣기 가능하도록 셸 이스케이프해서 보여준다 (위치 텍스트의 ';' 때문에 필수).
CMD_STR="$(printf '%q ' "${CMD[@]}")"
printf '%s\n' "$CMD_STR" | sed 's/ --/ \\\n    --/g'
cat <<EOF

${C_OK}이제 러너를 띄웁니다.${C_OFF} 모델 프리로드·arena 속도 반영까지 끝내고
${C_HL}[체크] 물리 상태 일괄 확인 ... (y/n)${C_OFF} 에서 멈춰 대기합니다.

  → 조교 배치가 끝나고 ${C_HL}"3 2 1 시작!"${C_OFF} 구호에 ${C_HL}y${C_OFF} 를 누르세요. 그 즉시 출발합니다.
  → 중단은 Ctrl-C. 다시 이 스크립트를 실행하면 스택은 재사용됩니다.
EOF
printf '\n%s러너를 띄울까요? (Enter=예 / n=취소)%s ' "$C_HL" "$C_OFF"
read -r go
if [[ "${go,,}" == "n" ]]; then
    echo "취소했습니다. 같은 구성으로 다시 실행하려면:"
    printf '  %s\n' "$CMD_STR"
    exit 0
fi

exec "${CMD[@]}"
