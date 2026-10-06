#!/usr/bin/env bash
set -euo pipefail

PIPELINE="unified"
PYTHON_BIN="python3"
CAMERA="0"
CAMERA_BACKEND="v4l2"
DEVICE="0"
MODEL=""
A1_MODEL=""
IMG_SIZE="224"
A1_IMG_SIZE="640"
A2_IMG_SIZE="224"
A1_CONF="0.25"
FACE_CONF="0.25"
CROP_PAD="0.18"
TARGET_SHAPE="cube"
TARGET_FRUIT="apple"
RUNTIME_PRESET="manual"
CPU=0
ONNX=0
DECISION_OVERLAY=0
SHOW_C_INPUTS=0
NO_MODEL_OUTPUT_PANEL=0
PRINT_MODEL_OUTPUT=0
PRINT_TIMING=0
NO_DISPLAY=0
MAX_FRAMES="0"
LIST_MODELS=0

usage() {
  cat <<'EOF'
Usage:
  jetson/run_webcam_preview.sh [options]

Common:
  --pipeline unified|abc|yolo      Default: unified
  --python PATH                    Default: python3
  --camera VALUE                   Camera index, video path, or GStreamer pipeline. Default: 0
  --camera-backend auto|any|dshow|v4l2|gstreamer
                                   Default: v4l2 for Jetson/Linux USB cameras
  --device VALUE                   Default: 0. Use --cpu for CPU.
  --cpu                            Force --device cpu
  --onnx                           Unified mode: use preferred ONNX aliases
                                   (face-classifier-onnx / cube-detector-onnx;
                                   legacy: preferred-unified-onnx / preferred-a1-onnx)
  --max-frames N                   Stop after N frames
  --no-display                     Run headless
  --print-timing                   Print stage timing
  --print-model-output             Print raw model output
  --no-model-output-panel          Hide right text panel
  --list-models                    Print model aliases and exit

Unified:
  --a1-model PATH_OR_ALIAS         Default: preferred-a1 (legacy synonym of cube-detector)
  --model PATH_OR_ALIAS            Default: preferred-unified (legacy synonym of face-classifier)
  --a1-imgsz N                     Default: 640
  --imgsz N                        Unified face model crop imgsz. Default: 224
  --a1-conf FLOAT                  Default: 0.25
  --face-conf FLOAT                Default: 0.25
  --crop-pad FLOAT                 Default: 0.18

ABC cascade:
  --runtime-preset NAME            Default: manual on Jetson/Linux
  --a2-imgsz N                     Default: 224
  --decision-overlay               Show decision overlay instead of raw overlay
  --show-c-inputs                  Show C perspective-warp inputs panel

Rulebook target:
  --target-shape cube|octahedron|dodecahedron|icosahedron
  --target-fruit apple|orange|banana|pineapple

Model aliases:
  Primary: cube-detector, cube-detector-onnx, face-classifier, face-classifier-onnx
  Legacy (still accepted): preferred-a1, preferred-a1-onnx, preferred-unified,
  preferred-unified-onnx, latest-unified
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline) PIPELINE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --camera) CAMERA="$2"; shift 2 ;;
    --camera-backend) CAMERA_BACKEND="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --a1-model) A1_MODEL="$2"; shift 2 ;;
    --imgsz) IMG_SIZE="$2"; shift 2 ;;
    --a1-imgsz) A1_IMG_SIZE="$2"; shift 2 ;;
    --a2-imgsz) A2_IMG_SIZE="$2"; shift 2 ;;
    --a1-conf) A1_CONF="$2"; shift 2 ;;
    --face-conf) FACE_CONF="$2"; shift 2 ;;
    --crop-pad) CROP_PAD="$2"; shift 2 ;;
    --target-shape) TARGET_SHAPE="$2"; shift 2 ;;
    --target-fruit) TARGET_FRUIT="$2"; shift 2 ;;
    --runtime-preset) RUNTIME_PRESET="$2"; shift 2 ;;
    --cpu) CPU=1; shift ;;
    --onnx) ONNX=1; shift ;;
    --decision-overlay) DECISION_OVERLAY=1; shift ;;
    --show-c-inputs) SHOW_C_INPUTS=1; shift ;;
    --no-model-output-panel) NO_MODEL_OUTPUT_PANEL=1; shift ;;
    --print-model-output) PRINT_MODEL_OUTPUT=1; shift ;;
    --print-timing) PRINT_TIMING=1; shift ;;
    --no-display) NO_DISPLAY=1; shift ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --list-models) LIST_MODELS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$PIPELINE" in
  unified|abc|yolo) ;;
  *)
    echo "--pipeline must be unified, abc, or yolo" >&2
    exit 2
    ;;
esac

if [[ "$CPU" == "1" ]]; then
  DEVICE="cpu"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$LIST_MODELS" == "1" ]]; then
  exec "$PYTHON_BIN" jetson/realtime_seg_cam.py --list-model-aliases
fi

ARGS=(
  jetson/realtime_seg_cam.py
  --pipeline "$PIPELINE"
  --camera "$CAMERA"
  --camera-backend "$CAMERA_BACKEND"
  --device "$DEVICE"
  --display-max-width 1600
  --display-max-height 900
)

if [[ "$PIPELINE" == "unified" ]]; then
  # Defaults stay on the legacy alias names for compatibility; the new primary
  # names (face-classifier / cube-detector) resolve to the same weights.
  if [[ -z "$MODEL" ]]; then
    if [[ "$ONNX" == "1" ]]; then
      MODEL="preferred-unified-onnx"
    else
      MODEL="preferred-unified"
    fi
  fi
  if [[ -z "$A1_MODEL" ]]; then
    if [[ "$ONNX" == "1" ]]; then
      A1_MODEL="preferred-a1-onnx"
    else
      A1_MODEL="preferred-a1"
    fi
  fi
  ARGS+=(
    --a1-model "$A1_MODEL"
    --model "$MODEL"
    --task segment
    --imgsz "$IMG_SIZE"
    --conf "$FACE_CONF"
    --unified-a1-imgsz "$A1_IMG_SIZE"
    --unified-a1-conf "$A1_CONF"
    --unified-crop-pad "$CROP_PAD"
    --target-shape "$TARGET_SHAPE"
    --target-fruit "$TARGET_FRUIT"
  )
elif [[ "$PIPELINE" == "abc" ]]; then
  ARGS+=(
    --runtime-preset "$RUNTIME_PRESET"
    --target-shape "$TARGET_SHAPE"
    --target-fruit "$TARGET_FRUIT"
    --imgsz "$A1_IMG_SIZE"
    --a2-imgsz "$A2_IMG_SIZE"
  )
  if [[ -n "$A1_MODEL" ]]; then
    ARGS+=(--a1-model "$A1_MODEL")
  fi
  if [[ "$DECISION_OVERLAY" == "1" ]]; then
    ARGS+=(--abc-overlay decision)
  fi
  if [[ "$SHOW_C_INPUTS" == "1" ]]; then
    ARGS+=(--show-c-inputs)
  fi
else
  if [[ -n "$MODEL" ]]; then
    ARGS+=(--model "$MODEL")
  fi
  ARGS+=(--task segment --imgsz "$IMG_SIZE")
fi

if [[ "$NO_MODEL_OUTPUT_PANEL" == "1" ]]; then
  ARGS+=(--no-model-output-panel)
fi
if [[ "$PRINT_MODEL_OUTPUT" == "1" ]]; then
  ARGS+=(--print-model-output)
fi
if [[ "$PRINT_TIMING" == "1" ]]; then
  ARGS+=(--print-timing)
fi
if [[ "$NO_DISPLAY" == "1" ]]; then
  ARGS+=(--no-display)
fi
if [[ "$MAX_FRAMES" != "0" ]]; then
  ARGS+=(--max-frames "$MAX_FRAMES")
fi

printf 'Running: %q' "$PYTHON_BIN"
printf ' %q' "${ARGS[@]}"
printf '\n'
exec "$PYTHON_BIN" "${ARGS[@]}"
