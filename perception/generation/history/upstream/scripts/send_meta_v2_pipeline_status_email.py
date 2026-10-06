import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from send_progress_email import send_email as send_email_impl


def parse_args():
    parser = argparse.ArgumentParser(description="Send one Meta V2 pipeline status email.")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--to", required=True)
    parser.add_argument("--subject", default="[Meta V2] pipeline status")
    parser.add_argument("--config", default="config/email_smtp.json")
    return parser.parse_args()


def count_files(path, pattern):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def latest_csv_row(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else None
    except Exception:
        return None


def format_time(value):
    if not value:
        return "unknown"
    return str(value)


def dataset_summary(source_dataset, model_root):
    source_dataset = Path(source_dataset) if source_dataset else None
    model_root = Path(model_root) if model_root else None
    lines = []
    if source_dataset:
        lines.extend(
            [
                "source dataset:",
                f"- path: {source_dataset}",
                f"- images/train: {count_files(source_dataset / 'images' / 'train', '*.jpg')}",
                f"- labels/train: {count_files(source_dataset / 'labels' / 'train', '*.txt')}",
                f"- _meta/train: {count_files(source_dataset / '_meta' / 'train', '*.json')}",
                "",
            ]
        )
    if model_root:
        c_counts = {}
        for class_dir in (model_root / "c_facecls" / "train").glob("*"):
            if class_dir.is_dir():
                c_counts[class_dir.name] = count_files(class_dir, "*.jpg")
        lines.extend(
            [
                "exported model datasets:",
                f"- A1 images: {count_files(model_root / 'a1_objectseg' / 'images' / 'train', '*.jpg')}",
                f"- A2 crops: {count_files(model_root / 'a2_faceseg' / 'images' / 'train', '*.jpg')}",
                f"- B mask samples: {count_files(model_root / 'b_facequad' / 'images' / 'train', '*.jpg')}",
                f"- C crops: {sum(c_counts.values()) if c_counts else 0}",
                f"- C class counts: {c_counts}",
                "",
            ]
        )
    return "\n".join(lines)


def training_summary(run_dir, task):
    if not run_dir:
        return ""
    run_dir = Path(run_dir)
    row = latest_csv_row(run_dir / "results.csv")
    lines = [
        f"{task} run:",
        f"- path: {run_dir}",
        f"- results.csv: {'yes' if (run_dir / 'results.csv').exists() else 'no'}",
        f"- best.pt: {'yes' if (run_dir / 'weights' / 'best.pt').exists() else 'no'}",
        f"- last.pt: {'yes' if (run_dir / 'weights' / 'last.pt').exists() else 'no'}",
    ]
    if row:
        compact = ", ".join(f"{k}={v}" for k, v in row.items() if k in {
            "epoch",
            "time",
            "train/box_loss",
            "train/seg_loss",
            "train/cls_loss",
            "metrics/mAP50(B)",
            "metrics/mAP50(M)",
            "train_loss",
            "val_loss",
            "train_mae",
            "val_mae",
            "train_acc",
            "val_acc",
        })
        lines.append(f"- latest: {compact}")
    lines.append("")
    return "\n".join(lines)


def all_known_runs(state):
    runs = []
    for key in ("a1_run_dir", "a2_run_dir", "b_run_dir", "c_run_dir"):
        if state.get(key):
            runs.append((key.replace("_run_dir", "").upper(), state[key]))
    current = state.get("current_run_dir")
    task = state.get("current_task", "current")
    if current and all(Path(current) != Path(path) for _name, path in runs):
        runs.append((task, current))
    return runs


def build_body(state):
    lines = [
        "Meta V2 A1/A2/B/C pipeline status",
        "",
        f"check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"status: {state.get('status', 'unknown')}",
        f"stage: {state.get('stage', 'unknown')}",
        f"current task: {state.get('current_task', '')}",
        f"started at: {format_time(state.get('started_at'))}",
        f"updated at: {format_time(state.get('updated_at'))}",
        "",
        dataset_summary(state.get("source_dataset"), state.get("model_root")),
    ]
    for task, run_dir in all_known_runs(state):
        lines.append(training_summary(run_dir, task))
    if state.get("last_error"):
        lines.extend(["last error:", str(state["last_error"]), ""])
    return "\n".join(lines)


def main():
    args = parse_args()
    state = read_json(args.state)
    body = build_body(state)

    class EmailArgs:
        pass

    email_args = EmailArgs()
    email_args.config = args.config
    email_args.to = args.to
    email_args.subject = args.subject
    email_args.body = body
    email_args.attach = []
    email_args.dry_run = False
    send_email_impl(email_args)


if __name__ == "__main__":
    main()
