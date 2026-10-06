import argparse
import csv
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from send_progress_email import send_email as send_email_impl


PROJECT_ROOT = Path(r"C:\Users\user\Documents\vscode\Data_Generation_Blender")
DEFAULT_DATASET = PROJECT_ROOT / r"datasets\yolo_8class_v3_50000"
RUN_CANDIDATES = [
    PROJECT_ROOT / r"runs\segment\runs\yolo26_seg_train\yolo26n_seg_50000_ideal_debug",
    PROJECT_ROOT / r"runs\detect\runs\yolo_train\yolo11n_v3_50000_fast_noamp",
    PROJECT_ROOT / r"runs\yolo_train\yolo11n_v3_50000_fast_noamp",
]
DEFAULT_TO = "jaeyoungi@snu.ac.kr"
DEFAULT_SUBJECT = "[30min check] YOLO 50000 training status"
DEFAULT_TOTAL_EPOCHS = 200
DEFAULT_PROCESS_PATTERNS = ["train_yolo_with_email.py", "yolo segment train", "yolo26n_seg_50000", "yolo detect train", "yolo11n_v3_50000"]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect YOLO 50000 training status and send the status email.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--to", default=DEFAULT_TO)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TOTAL_EPOCHS)
    parser.add_argument("--config", default="config/email_smtp.json")
    parser.add_argument("--task-label", default="YOLO 50000 training")
    parser.add_argument("--no-attach", action="store_true", help="Send only the status text without attaching weight files.")
    parser.add_argument(
        "--process-pattern",
        action="append",
        default=None,
        help="Command-line pattern used to find matching training processes. Can be repeated.",
    )
    return parser.parse_args()


def choose_run_dir():
    for candidate in RUN_CANDIDATES:
        if candidate.exists():
            return candidate
    return RUN_CANDIDATES[0]


def read_args_yaml(path: Path):
    data = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def read_latest_metrics(results_csv: Path):
    if not results_csv.exists():
        return None
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def count_files(path: Path, pattern: str):
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def format_timestamp(ts: datetime | None):
    if ts is None:
        return "unknown"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float | None):
    if seconds is None or seconds < 0:
        return "calculating"
    hours = seconds / 3600
    if hours >= 1:
        whole_hours = int(hours)
        minutes = int(round((hours - whole_hours) * 60))
        return f"about {whole_hours}h {minutes}m"
    minutes = max(1, int(round(seconds / 60)))
    return f"about {minutes}m"


def find_training_processes(patterns):
    escaped_patterns = ", ".join("'" + pattern.replace("'", "''") + "'" for pattern in patterns)
    ps_script = r"""
$patterns = @(__PATTERNS__)
Get-CimInstance Win32_Process |
  Where-Object {
    $cmd = $_.CommandLine
    $_.Name -in @('python.exe', 'yolo.exe') -and
    $null -ne $cmd -and
    ($patterns | Where-Object { $_.Length -gt 0 -and $cmd -match [regex]::Escape($_) })
  } |
  Select-Object ProcessId, Name, CreationDate, CommandLine |
  ConvertTo-Json -Compress
""".replace("__PATTERNS__", escaped_patterns)
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=PROJECT_ROOT,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return [], f"process query failed: {exc}"

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        fallback_script = r"""
Get-Process python,yolo -ErrorAction SilentlyContinue |
  Where-Object { $null -ne $_.Path -and $_.Path -match 'ai_robotics|yolo' } |
  Select-Object @{n='ProcessId';e={$_.Id}}, Name, @{n='CreationDate';e={$_.StartTime.ToString('yyyyMMddHHmmss.000000+540')}}, @{n='CommandLine';e={$_.Path}} |
  ConvertTo-Json -Compress
"""
        try:
            fallback = subprocess.run(
                ["powershell", "-NoProfile", "-Command", fallback_script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=PROJECT_ROOT,
                timeout=30,
                check=False,
            )
        except Exception:
            return [], f"process query failed: {stderr or f'exit {result.returncode}'}"
        raw = (fallback.stdout or "").strip()
        if fallback.returncode == 0 and raw:
            import json

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = []
            if isinstance(parsed, dict):
                parsed = [parsed]
            if parsed:
                return parsed, "Win32_Process unavailable; used Get-Process fallback"
        return [], f"process query failed: {stderr or f'exit {result.returncode}'}"

    raw = (result.stdout or "").strip()
    if not raw:
        return [], ""

    import json

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [], f"failed to parse process query result: {raw[:300]}"

    if isinstance(parsed, dict):
        parsed = [parsed]
    return parsed, ""


def parse_cim_datetime(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def build_body(run_dir: Path, dataset_dir: Path, total_epochs: int, task_label: str, process_patterns):
    now = datetime.now()
    process_rows, process_note = find_training_processes(process_patterns)

    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"
    weights_dir = run_dir / "weights"
    last_pt = weights_dir / "last.pt"
    best_pt = weights_dir / "best.pt"

    metrics = read_latest_metrics(results_csv)
    args_data = read_args_yaml(args_yaml)
    total_epochs = int(args_data.get("epochs", total_epochs) or total_epochs)

    image_count = count_files(dataset_dir / "images" / "train", "*.jpg")
    label_count = count_files(dataset_dir / "labels" / "train", "*.txt")

    epoch_index = -1
    completed_epochs = 0
    train_box = ""
    train_seg = ""
    train_cls = ""
    train_dfl = ""
    box_map50 = ""
    box_map5095 = ""
    mask_map50 = ""
    mask_map5095 = ""
    elapsed_seconds = None

    if metrics:
        epoch_index = int(float(metrics.get("epoch", -1)))
        completed_epochs = epoch_index + 1
        train_box = metrics.get("train/box_loss", "")
        train_seg = metrics.get("train/seg_loss", "")
        train_cls = metrics.get("train/cls_loss", "")
        train_dfl = metrics.get("train/dfl_loss", "")
        box_map50 = metrics.get("metrics/mAP50(B)", "")
        box_map5095 = metrics.get("metrics/mAP50-95(B)", "")
        mask_map50 = metrics.get("metrics/mAP50(M)", "")
        mask_map5095 = metrics.get("metrics/mAP50-95(M)", "")
        try:
            elapsed_seconds = float(metrics.get("time", "")) if metrics.get("time", "") != "" else None
        except ValueError:
            elapsed_seconds = None

    remaining_epochs = max(total_epochs - completed_epochs, 0)
    eta_seconds = None
    finish_time = None
    if elapsed_seconds and completed_epochs > 0:
        per_epoch = elapsed_seconds / completed_epochs
        eta_seconds = per_epoch * remaining_epochs
        finish_time = now + timedelta(seconds=eta_seconds)

    process_lines = []
    for row in process_rows:
        started = parse_cim_datetime(row.get("CreationDate"))
        process_lines.append(
            f"- {row.get('Name')} pid={row.get('ProcessId')} start={format_timestamp(started)}"
        )
    if not process_lines:
        process_lines.append("- no matching training process")
    if process_note:
        process_lines.append(f"- {process_note}")

    best_text = "missing"
    if best_pt.exists():
        best_text = f"exists, modified {format_timestamp(datetime.fromtimestamp(best_pt.stat().st_mtime))}"

    body = (
        f"{task_label} 30-minute status check.\n\n"
        f"check time (Asia/Seoul): {format_timestamp(now)}\n"
        f"run path: {run_dir}\n"
        f"dataset path: {dataset_dir}\n\n"
        "training progress:\n"
        f"completed epochs: {completed_epochs} / {total_epochs}\n"
        f"ultralytics epoch index: {epoch_index}\n"
        f"train/box_loss: {train_box}\n"
        f"train/seg_loss: {train_seg}\n"
        f"train/cls_loss: {train_cls}\n"
        f"train/dfl_loss: {train_dfl}\n"
        f"box metrics/mAP50(B): {box_map50}\n"
        f"box metrics/mAP50-95(B): {box_map5095}\n"
        f"mask metrics/mAP50(M): {mask_map50}\n"
        f"mask metrics/mAP50-95(M): {mask_map5095}\n"
        f"results.csv modified: {format_timestamp(datetime.fromtimestamp(results_csv.stat().st_mtime)) if results_csv.exists() else 'missing'}\n\n"
        "dataset counts:\n"
        f"images/train JPG: {image_count}\n"
        f"labels/train TXT: {label_count}\n\n"
        "ETA:\n"
        f"elapsed training time from results.csv: {round(elapsed_seconds / 60, 2) if elapsed_seconds else 'unknown'} minutes\n"
        f"remaining epochs: {remaining_epochs}\n"
        f"estimated remaining time: {format_duration(eta_seconds)}\n"
        f"estimated finish time (Asia/Seoul): {format_timestamp(finish_time)}\n\n"
        "weight files:\n"
        f"last.pt exists: {'yes' if last_pt.exists() else 'no'}\n"
        f"last.pt modified: {format_timestamp(datetime.fromtimestamp(last_pt.stat().st_mtime)) if last_pt.exists() else 'missing'}\n"
        f"best.pt exists: {'yes' if best_pt.exists() else 'no'}\n"
        f"best.pt modified: {best_text}\n\n"
        "observed processes:\n"
        + "\n".join(process_lines)
        + "\n"
    )
    return body, last_pt


def main():
    args = parse_args()
    run_dir = args.run_dir if args.run_dir is not None else choose_run_dir()
    process_patterns = args.process_pattern or DEFAULT_PROCESS_PATTERNS
    body, last_pt = build_body(run_dir, args.dataset, args.epochs, args.task_label, process_patterns)
    best_pt = run_dir / "weights" / "best.pt"

    class EmailArgs:
        pass

    email_args = EmailArgs()
    email_args.config = args.config
    email_args.to = args.to
    email_args.subject = args.subject
    email_args.body = body
    if args.no_attach:
        email_args.attach = []
    elif best_pt.exists():
        email_args.attach = [str(best_pt)]
    elif last_pt.exists():
        email_args.attach = [str(last_pt)]
    else:
        email_args.attach = []
    email_args.dry_run = False
    send_email_impl(email_args)


if __name__ == "__main__":
    main()
