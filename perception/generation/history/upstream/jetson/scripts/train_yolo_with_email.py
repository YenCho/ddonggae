import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


EPOCH_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\b")
SEOUL_TZ = ZoneInfo("Asia/Seoul")


def send_email(args, subject, body, attachments=None):
    if not args.email_to:
        return
    cmd = [
        sys.executable,
        "scripts/send_progress_email.py",
        "--config", args.email_config,
        "--to", args.email_to,
        "--subject", subject,
        "--body", body,
    ]
    for attachment in attachments or []:
        if attachment and Path(attachment).exists():
            cmd.extend(["--attach", str(attachment)])
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"email notification failed: {exc}", flush=True)


def format_duration(seconds):
    minutes = int(round(max(seconds, 0) / 60))
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def format_seoul_time(timestamp=None):
    if timestamp is None:
        dt = datetime.now(SEOUL_TZ)
    else:
        dt = datetime.fromtimestamp(timestamp, SEOUL_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S KST")


def build_train_cmd(args):
    cmd = [
        "yolo",
        args.task,
        "train",
        f"model={args.model}",
        f"data={args.data}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"device={args.device}",
        f"workers={args.workers}",
        f"cache={args.cache}",
        f"amp={args.amp}",
        f"deterministic={args.deterministic}",
        f"patience={args.patience}",
        f"project={args.project}",
        f"name={args.name}",
        f"exist_ok={args.exist_ok}",
    ]
    cmd.extend(args.extra)
    return cmd


def weights_for(args):
    weights_dir = Path(args.project) / args.name / "weights"
    return [weights_dir / "best.pt", weights_dir / "last.pt"]


def train_body(args, epoch, total_epochs, start_time, last_line=""):
    elapsed = time.time() - start_time
    per_epoch = elapsed / max(epoch, 1)
    remaining_epochs = max(total_epochs - epoch, 0)
    remaining_seconds = per_epoch * remaining_epochs if epoch > 0 else None
    eta = format_duration(remaining_seconds) if remaining_seconds is not None else "calculating"
    expected_finish = format_seoul_time(time.time() + remaining_seconds) if remaining_seconds is not None else "calculating"
    return (
        "YOLO 학습 진행 상태\n\n"
        f"run: {args.project}/{args.name}\n"
        f"epoch: {epoch}/{total_epochs}\n"
        f"elapsed: {format_duration(elapsed)}\n"
        f"remaining time: {eta}\n"
        f"expected finish (Asia/Seoul): {expected_finish}\n"
        f"email sent (Asia/Seoul): {format_seoul_time()}\n"
        f"data: {args.data}\n"
        f"model: {args.model}\n"
        f"imgsz: {args.imgsz}\n"
        f"batch: {args.batch}\n"
        f"device: {args.device}\n\n"
        f"latest log:\n{last_line}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Run YOLO training and send non-blocking email progress notifications.")
    parser.add_argument("--task", choices=["detect", "segment", "classify", "pose"], default="detect")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", default="ram")
    parser.add_argument("--amp", default="False")
    parser.add_argument("--deterministic", default="False")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="runs/yolo_train")
    parser.add_argument("--name", default="yolo11n_v3_30000_fast_noamp")
    parser.add_argument("--exist_ok", default="True")
    parser.add_argument("--email_to", default="")
    parser.add_argument("--email_every", type=int, default=20)
    parser.add_argument("--email_config", default="config/email_smtp.json")
    parser.add_argument("--attach_final", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = build_train_cmd(args)
    print(" ".join(cmd), flush=True)
    start_time = time.time()
    last_epoch_mailed = 0
    last_epoch = 0
    last_total = args.epochs
    last_line = ""

    send_email(
        args,
        f"[시작] YOLO 학습 {args.name}",
        train_body(args, 0, args.epochs, start_time, "training started"),
    )

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stripped = line.strip()
            if stripped:
                last_line = stripped
            match = EPOCH_RE.match(line)
            if not match:
                continue
            epoch = int(match.group(1))
            total_epochs = int(match.group(2))
            last_epoch = epoch
            last_total = total_epochs
            if args.email_every > 0 and epoch - last_epoch_mailed >= args.email_every:
                send_email(
                    args,
                    f"[진행] YOLO 학습 epoch {epoch}/{total_epochs}",
                    train_body(args, epoch, total_epochs, start_time, last_line),
                )
                last_epoch_mailed = epoch
    finally:
        return_code = process.wait()

    attachments = weights_for(args) if args.attach_final else []
    if return_code == 0:
        send_email(
            args,
            f"[완료] YOLO 학습 {args.name}",
            train_body(args, max(last_epoch, last_total), last_total, start_time, last_line),
            attachments,
        )
    else:
        send_email(
            args,
            f"[오류] YOLO 학습 중단 {args.name}",
            train_body(args, last_epoch, last_total, start_time, last_line) + f"\nreturn code: {return_code}\n",
        )
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
