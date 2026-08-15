#!/usr/bin/env python3
"""경량 경기 HUD — 목표물체별 고정 이미지를 저주기로만 디스플레이에 올린다.

왜 새로 만들었나 (2026-07-22 Jetson 실측):
  구 HUD(scripts/match_hud_display.py)는 매 프레임 PIL 로 4패널을 그리고,
  벤더 라이브러리가 그 결과를 PNG(compress_level=9)로 재인코딩했다.
  프레임당 CPU = 렌더 38.4 ms + 회전 2.7 ms + PNG 인코딩 332.8 ms ≈ 374 ms.
  기본값 --fps 4 는 주기 250 ms 이므로 sleep 이 항상 0 이 되어, 이 루프는
  쉬지 않고 코어 1개를 영구 점유했다. 그만큼 sllidar 시리얼 읽기 스레드
  스케줄링이 밀려 스캔 드롭/지터 → localization 품질이 열화됐다.

이 스크립트는 런타임에 아무것도 그리지 않는다:
  1) --build : 이미지를 미리 회전+인코딩해 캐시로 저장 (오프라인 1회)
  2) 런타임  : 상태 파일을 저주기(기본 0.1 Hz)로 stat 하고, 띄울 것이
               바뀔 때만 캐시된 바이트를 USB 로 그대로 write

벤더 라이브러리의 DisplayPILImage()(paste → transpose → PNG 재인코딩)를
우회하고 send_image()/send_jpeg() 를 직접 호출하므로, 라이브러리를 고칠
필요가 없다. 정상 상태 CPU 는 사실상 0, 전환 순간에만 USB write 수 ms.

상태 파일 (2줄, 러너가 원자적으로 교체):
    1줄: 띄울 이미지 키 — 예 `apple2`, `icosahedron1`, `start`, `end`
    2줄: 현재 예상점수 정수 — 예 `20`
  점수가 오른 것을 감지한 틱에서만 `points<점수>` 를 1회 띄우고,
  그 다음 틱부터 다시 1줄의 목표물체로 돌아간다 (2026-07-23 운영 결정).

사진 자산: hardware/hud/photos/<키>.png 를 그대로 쓴다. 키에 붙는 숫자는
룰북 quota 순번이다 (과일 3개 = No.1~3, 다면체 4개 = No.1~4).
없는 키는 숫자를 떼고 → `<클래스>1` → `<클래스>` 순으로 폴백하며,
사진이 아예 없는 클래스는 자동 생성 슬레이트로 대체된다.

사용:
  python3 scripts/match_hud_lite.py --build          # 캐시 생성
  python3 scripts/match_hud_lite.py                  # 라이브 (상태 파일 추종)
  python3 scripts/match_hud_lite.py --class apple2    # 수동으로 1장만 띄우기
  python3 scripts/match_hud_lite.py --build --preview-dir /tmp/hud  # 미리보기 PNG

turing-smart-screen-python 위치는 TURING_SCREEN_LIB 로 바꿀 수 있다.
상태 파일 경로는 HUD_STATE_FILE 로 바꿀 수 있다 (러너와 동일 값이어야 함).
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PHOTO_DIR = REPO_ROOT / "hardware" / "hud" / "photos"
DEFAULT_CACHE_DIR = REPO_ROOT / "hardware" / "hud" / "cache"
DEFAULT_STATE_FILE = REPO_ROOT / "logs" / "hud" / "target"

# 논리(가로) 패널 해상도. 장치는 세로 462x1920 이라 전송 직전 ROTATE_270 한다.
# Turing 9.2" (PID 0x0092) 기준이며 --size 로 바꿀 수 있다.
DEFAULT_PANEL = (1920, 462)
MAX_PAYLOAD = 1024 * 1024  # 장치 1회 업로드 한도 (벤더 MAX_CHUNK_BYTES)

PHOTO_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# 클래스 목록의 원본은 perception/fieldlib.py:29-31.
# 런타임에 numpy/cv2 를 끌어오지 않으려고 여기서는 값만 복제한다.
FRUITS = ("apple", "orange", "banana", "pineapple")
POLYHEDRA = ("octahedron", "dodecahedron", "icosahedron")
# start=경기 시작, end=경기 종료. 사진 파일명(start.png/end.png)과 같은 이름을 쓴다.
STATES = ("start", "end")
SLATE_CLASSES = FRUITS + POLYHEDRA + ("plain",) + STATES

IDLE_KEY = "start"          # 상태 파일이 없거나 폴백이 전부 실패했을 때
POINTS_PREFIX = "points"

BG = (0, 0, 0)
TEXT = (245, 248, 252)
MUTED = (140, 150, 165)
ACCENT = {"fruit": (255, 186, 64), "shape": (64, 205, 255), "other": (150, 158, 172)}

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
)

_KEY_RE = re.compile(r"^([a-z_]+?)(\d*)$")


def base_of(key):
    """`apple2` → `apple`. 숫자가 없으면 그대로. `points20` 은 예외로 통째 유지."""
    if key.startswith(POINTS_PREFIX):
        return key
    m = _KEY_RE.match(key)
    return m.group(1) if m else key


def category(cls):
    if cls in FRUITS:
        return "fruit"
    if cls in POLYHEDRA:
        return "shape"
    return "other"


# ------------------------------------------------------------------ 캐시 빌드
def _font(size):
    """설치된 첫 후보 폰트. 없으면 PIL 기본 비트맵 폰트로 폴백."""
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _fit_font(draw, text, max_width, sizes):
    """max_width 안에 들어가는 가장 큰 폰트."""
    font = _font(sizes[-1])
    for size in sizes:
        cand = _font(size)
        if draw.textlength(text, font=cand) <= max_width:
            return cand
        font = cand
    return font


def _draw_icon(draw, box, cls):
    """사진이 없을 때 쓰는 최소 아이콘 — 과일=원, 다면체=다각형, 그 외=사각."""
    import math
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = min(x1 - x0, y1 - y0) * 0.38
    kind = category(cls)
    color = ACCENT[kind]
    if kind == "fruit":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=8)
    elif kind == "shape":
        # 면 수를 대충 반영한 정다각형 (octa=8, dodeca=12, icosa=20 → 상한 10)
        n = {"octahedron": 8, "dodecahedron": 12, "icosahedron": 20}.get(cls, 6)
        n = min(10, n)
        pts = [(cx + r * math.cos(-math.pi / 2 + i * 2 * math.pi / n),
                cy + r * math.sin(-math.pi / 2 + i * 2 * math.pi / n))
               for i in range(n)]
        draw.polygon(pts, outline=color, width=8)
    else:
        draw.rounded_rectangle([cx - r, cy - r, cx + r, cy + r], radius=int(r * 0.2),
                               outline=color, width=8)


def _fit_photo(src, size):
    """사진 1장을 패널에 채운다.

    자산(1280x321 = 3.99:1)과 패널(1920x462 = 4.16:1)의 비율 차이는 4% 뿐이라
    중앙 크롭으로 전체화면을 채운다 — 세로 3% 손실. 40장 전부 렌더해 글자·얼굴이
    잘리지 않는 것을 확인했다 (2026-07-23). 여백 채우기는 사진이 가장자리까지
    찬 자산(pineapple3 = 피자)에서 갈색 띠가 생겨 채택하지 않았다.
    비율이 크게 다른 사진(±25% 초과)만 코너색 여백으로 폴백한다.
    """
    from PIL import Image
    w, h = size
    panel_ar, src_ar = w / h, src.width / max(1, src.height)
    if abs(src_ar - panel_ar) <= panel_ar * 0.25:
        scale = max(w / src.width, h / src.height)          # 채우고 넘치는 만큼 크롭
        fit = src.resize((max(w, round(src.width * scale)),
                          max(h, round(src.height * scale))), Image.LANCZOS)
        return fit.crop(((fit.width - w) // 2, (fit.height - h) // 2,
                         (fit.width - w) // 2 + w, (fit.height - h) // 2 + h))
    scale = min(w / src.width, h / src.height)              # 비율 유지 + 코너색 여백
    fit = src.resize((round(src.width * scale), round(src.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", size, src.getpixel((0, 0)))
    canvas.paste(fit, ((w - fit.width) // 2, (h - fit.height) // 2))
    return canvas


def _slate(cls, size):
    """사진이 없는 클래스용 자동 생성 화면 (아이콘 + 클래스명)."""
    from PIL import Image, ImageDraw
    w, h = size
    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)
    margin = max(12, h // 20)
    box_side = h - 2 * margin
    box = (margin, margin, margin + box_side, margin + box_side)
    _draw_icon(draw, box, cls)
    tx = box[2] + margin * 2
    avail = w - margin - tx
    kind = category(cls)
    label = {"fruit": "TARGET FRUIT", "shape": "TARGET SHAPE"}.get(kind, "STATUS")
    draw.text((tx, margin + box_side * 0.12), label, font=_font(max(14, h // 20)),
              fill=ACCENT[kind])
    name = cls.replace("_", " ").upper()
    f_name = _fit_font(draw, name, avail, (h // 3, h // 4, h // 5, h // 7, h // 9))
    draw.text((tx, margin + box_side * 0.30), name, font=f_name, fill=TEXT)
    draw.text((tx, margin + box_side * 0.78), "SNU PHYSICAL AI - TEAM 14",
              font=_font(max(12, h // 26)), fill=MUTED)
    return img


def _encode(img):
    """장치 업로드 한도 안에서 (바이트, 확장자). PNG 우선, 넘치면 JPEG.

    벤더 send_pil_image_auto() 와 같은 정책이지만 오프라인 1회라 압축률을
    맘껏 올려도 된다 (런타임 비용 0).

    [2026-07-23] PNG 는 반드시 **RGBA(colortype 6)** 로 저장한다. 장치 펌웨어가
    4바이트/픽셀을 가정하고 디코드하기 때문에, RGB(colortype 2, 3바이트/픽셀)로
    보내면 행마다 stride 가 어긋나 화면이 왼쪽으로 밀리고 번지며 3/4 지점에서
    데이터가 끊긴다(실기 증상). 벤더 DisplayPILImage 는 current_state 가 RGBA 라
    우연히 항상 RGBA 로 보내고 있었고, 그래서 이 제약이 코드에 안 드러나 있었다.
    JPEG 폴백은 알파를 못 담으므로 RGB 그대로 둔다 (벤더 JPEG 경로도 동일).
    """
    from io import BytesIO
    buf = BytesIO()
    img.convert("RGBA").save(buf, format="PNG", compress_level=9)
    data = buf.getvalue()
    # IHDR 25번째 바이트 = colortype. 6(RGBA) 이 아니면 실기에서 화면이 밀린다.
    # 오프라인 빌드에서만 도므로 런타임 비용 없음.
    if data[25] != 6:
        raise SystemExit(f"PNG colortype {data[25]} (6=RGBA 이어야 함) — "
                         "장치 펌웨어가 4바이트/픽셀로 디코드한다")
    if len(data) <= MAX_PAYLOAD:
        return data, "png"
    best = None
    for quality in (92, 85, 78, 70, 60, 50, 40, 30):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, subsampling=0, optimize=True)
        data = buf.getvalue()
        best = data if best is None or len(data) < len(best) else best
        if len(data) <= MAX_PAYLOAD:
            return data, "jpg"
    raise SystemExit(f"인코딩 실패: {len(best)}B > 한도 {MAX_PAYLOAD}B — 사진을 줄이세요")


def _short(path):
    """로그용 경로 — repo 안이면 상대경로, 밖이면(스크래치 등) 절대경로 그대로."""
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def scan_photos(photo_dir):
    """{키: 경로}. 키 = 파일 stem 소문자 (예 apple2, points30, start)."""
    found = {}
    for path in sorted(photo_dir.glob("*")):
        if path.suffix.lower() in PHOTO_EXTS:
            found.setdefault(path.stem.lower(), path)
    return found


def build_cache(size, photo_dir, cache_dir, preview_dir=None, extra_slates=()):
    from PIL import Image
    cache_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
    for stale in cache_dir.glob("*.*"):
        if stale.suffix in (".png", ".jpg"):
            stale.unlink()

    photos = scan_photos(photo_dir)
    have_base = {base_of(k) for k in photos}
    # 사진이 한 장도 없는 클래스만 슬레이트로 보충한다 (오타/신규 클래스 안전망).
    slates = [c for c in tuple(SLATE_CLASSES) + tuple(extra_slates) if c not in have_base]

    print(f"패널 {size[0]}x{size[1]} (전송 시 {size[1]}x{size[0]} 세로) "
          f"→ {_short(cache_dir)}")
    total = 0
    for key in sorted(photos) + sorted(slates):
        if key in photos:
            src = Image.open(photos[key])
            img = _fit_photo(src.convert("RGB") if src.mode != "RGB" else src, size)
            kind = "photo"
        else:
            img = _slate(key, size)
            kind = "slate"
        if preview_dir:
            img.save(preview_dir / f"{key}.png")
        # 장치는 세로 방향으로만 받는다. 벤더 DisplayPILImage 의 LANDSCAPE
        # 경로와 동일하게 미리 돌려서 캐시에 저장한다 (런타임 회전 비용 0).
        rot = img.transpose(Image.Transpose.ROTATE_270)
        data, ext = _encode(rot)
        (cache_dir / f"{key}.{ext}").write_bytes(data)
        total += len(data)
        print(f"  {key:14s} {kind:5s} {ext}  {len(data) / 1024:7.1f} KiB")
    print(f"총 {len(photos) + len(slates)}장 / {total / 1024 / 1024:.2f} MiB "
          f"(런타임 상주 메모리)")


def load_cache(cache_dir):
    """캐시 전체를 메모리로 (40장 ≈ 9 MiB)."""
    cache = {}
    for path in sorted(cache_dir.glob("*.*")):
        if path.suffix in (".png", ".jpg"):
            cache[path.stem] = (path.suffix[1:], path.read_bytes())
    return cache


def resolve(cache, key):
    """`apple2` 가 없으면 `apple1` → `apple` 순으로 폴백. 전부 없으면 None."""
    if key in cache:
        return key
    base = base_of(key)
    for cand in (base + "1", base):
        if cand in cache:
            return cand
    return None


def points_key(cache, score):
    """점수 이하의 가장 큰 points 이미지. 없으면 None (점수 화면 생략)."""
    if score is None or score <= 0:
        return None
    if f"{POINTS_PREFIX}{score}" in cache:
        return f"{POINTS_PREFIX}{score}"
    avail = []
    for k in cache:
        if k.startswith(POINTS_PREFIX) and k[len(POINTS_PREFIX):].isdigit():
            avail.append(int(k[len(POINTS_PREFIX):]))
    below = [v for v in avail if v <= score]
    return f"{POINTS_PREFIX}{max(below)}" if below else None


# -------------------------------------------------------------------- 디스플레이
def load_vendor():
    lib = Path(os.environ.get("TURING_SCREEN_LIB",
                              Path.home() / "turing-smart-screen-python"))
    if not (lib / "library").is_dir():
        raise SystemExit(f"turing-smart-screen-python 라이브러리 없음: {lib} "
                         "(TURING_SCREEN_LIB 환경변수로 지정 가능)")
    sys.path.insert(0, str(lib))
    from library.lcd import lcd_comm_turing_usb as vendor
    return vendor


class Display:
    """벤더 LcdCommTuringUSB 를 쓰지 않고 모듈 함수만 직접 호출한다.

    클래스를 쓰면 462x1920 RGBA 상태 이미지(약 3.5 MB)를 들고 매 전송마다
    paste/transpose/PNG 재인코딩을 하는데, 우리는 이미 인코딩된 바이트를
    그대로 보내므로 전부 불필요하다.
    """

    def __init__(self, brightness):
        self.vendor = load_vendor()
        self.dev, pid = self.vendor.find_usb_device()
        self.panel = self.vendor.PRODUCT_ID[pid]  # (w, h) 세로 기준
        self.vendor.send_sync_command(self.dev)
        self.set_brightness(brightness)

    def set_brightness(self, level):
        self.vendor.send_brightness_command(self.dev, int(max(0, min(100, level)) / 100 * 102))

    def push(self, blob):
        fmt, data = blob
        if fmt == "png":
            self.vendor.send_image(self.dev, data)
        else:
            self.vendor.send_jpeg(self.dev, data)


# ------------------------------------------------------------------ 런타임 루프
def read_state(state_file):
    """(키, 점수). 1줄=이미지 키, 2줄=예상점수. 못 읽으면 (None, None)."""
    try:
        with open(state_file, "r") as fh:
            key = fh.readline().strip().lower() or None
            raw = fh.readline().strip()
    except OSError:
        return None, None
    try:
        score = int(raw)
    except ValueError:
        score = None
    return key, score


def run(args, cache):
    display = None if args.no_display else Display(args.brightness)
    if display is not None:
        want = (args.size[1], args.size[0])  # 캐시는 세로로 저장돼 있음
        if display.panel != want:
            print(f"경고: 장치 {display.panel} != 캐시 {want} — "
                  f"--size {display.panel[1]}x{display.panel[0]} 로 재빌드 권장",
                  file=sys.stderr)

    state = {"shown": None}

    def show(key, why=""):
        """key 를 (폴백 포함) 띄운다. 이미 떠 있으면 USB 전송 자체를 생략."""
        real = resolve(cache, key) or resolve(cache, IDLE_KEY)
        if real is None:
            print(f"캐시에 '{key}' 도 '{IDLE_KEY}' 도 없음 — 건너뜀", file=sys.stderr)
            return
        if real == state["shown"]:
            return
        state["shown"] = real
        blob = cache[real]
        if display is not None:
            display.push(blob)
        note = f" (요청 '{key}')" if real != key else ""
        # 운영 중 tail -f 로 보므로 즉시 flush (기본 블록 버퍼링이면 죽을 때 유실된다)
        print(f"[{time.strftime('%H:%M:%S')}] HUD → {real}{note}{why} "
              f"({len(blob[1]) / 1024:.0f} KiB)", flush=True)

    if args.klass:
        show(args.klass)
        return

    period = 1.0 / max(0.01, args.hz)
    last_stamp = object()   # 첫 회는 반드시 갱신되도록 어떤 값과도 다르게
    key, score = IDLE_KEY, None
    last_score = None
    print(f"상태 파일 {args.state_file} 을 {args.hz} Hz 로 감시 "
          f"(변경 시에만 전송, nice={os.nice(0)})", flush=True)
    try:
        while True:
            try:
                st = os.stat(args.state_file)
                stamp = (st.st_mtime_ns, st.st_size)
            except OSError:
                stamp = None
            if stamp != last_stamp:
                last_stamp = stamp
                new_key, score = read_state(args.state_file)
                key = new_key or IDLE_KEY

            # 점수가 오른 것을 처음 본 틱에서만 점수 화면 1회 (2026-07-23 운영 결정).
            # 적재 성공과 다음 목표 확정 사이가 1초 미만이라, 러너 쪽 순서만으로는
            # 0.1 Hz 폴링이 점수 화면을 매번 놓친다.
            flash = None
            if score is not None and last_score is not None and score > last_score:
                flash = points_key(cache, score)
            if score is not None:
                last_score = score

            show(flash or key, f" [{score}점]" if flash else "")
            time.sleep(period)
    except KeyboardInterrupt:
        pass


# ------------------------------------------------------------------------ CLI
def parse_size(text):
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("WxH 형식이어야 합니다 (예: 1920x462)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", action="store_true", help="캐시 생성 후 종료")
    parser.add_argument("--class", dest="klass", metavar="KEY",
                        help="상태 파일 무시하고 이 키 1장만 띄우고 종료 (예 apple2)")
    parser.add_argument("--size", type=parse_size, default=DEFAULT_PANEL,
                        metavar="WxH", help="가로 패널 해상도 (기본 1920x462)")
    parser.add_argument("--photo-dir", type=Path, default=DEFAULT_PHOTO_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--preview-dir", type=Path, default=None,
                        help="빌드 시 회전 전 미리보기 PNG 도 저장")
    parser.add_argument("--state-file", type=Path,
                        default=Path(os.environ.get("HUD_STATE_FILE", DEFAULT_STATE_FILE)))
    parser.add_argument("--hz", type=float, default=0.1, help="상태 파일 확인 주기 (기본 0.1)")
    parser.add_argument("--brightness", type=int, default=40)
    parser.add_argument("--nice", type=int, default=19,
                        help="자기 자신의 nice 값 (기본 19 — 라이다/제어에 양보)")
    parser.add_argument("--cpu", type=int, nargs="+", default=None,
                        help="이 코어에만 고정 (예: --cpu 5)")
    parser.add_argument("--no-display", action="store_true", help="USB 전송 없이 동작만")
    args = parser.parse_args()

    if args.build:
        build_cache(args.size, args.photo_dir, args.cache_dir, args.preview_dir)
        return

    cache = load_cache(args.cache_dir)
    if not cache:
        raise SystemExit(f"캐시 비어있음: {args.cache_dir} — 먼저 --build 를 실행하세요")

    try:
        os.nice(args.nice)
    except OSError:
        pass
    if args.cpu:
        try:
            os.sched_setaffinity(0, set(args.cpu))
        except OSError as exc:
            print(f"경고: CPU 고정 실패 {exc}", file=sys.stderr)
    run(args, cache)


if __name__ == "__main__":
    main()
