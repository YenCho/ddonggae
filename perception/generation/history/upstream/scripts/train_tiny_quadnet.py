import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyQuadNet for B: visible face mask -> amodal 4-corner quad.")
    parser.add_argument("--data", type=Path, required=True, help="b_facequad dataset root.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/meta_v2_b_tinyquadnet"))
    parser.add_argument("--name", default="b_tinyquadnet_meta_v2_10000")
    parser.add_argument("--email_to", default="")
    parser.add_argument("--email_config", default="config/email_smtp.json")
    parser.add_argument("--email_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_seconds(seconds):
    minutes = int(max(seconds, 0) // 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def send_email(args, subject, body):
    if not args.email_to:
        return
    cmd = [
        sys.executable,
        "scripts/send_progress_email.py",
        "--config",
        args.email_config,
        "--to",
        args.email_to,
        "--subject",
        subject,
        "--body",
        body,
    ]
    try:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"email failed: {exc}", flush=True)


def parse_quad_label(path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = [float(v) for v in text.split()]
    if len(parts) != 17:
        return None
    kpts = parts[5:17]
    xy = []
    vis = []
    for idx in range(0, len(kpts), 3):
        xy.extend([kpts[idx], kpts[idx + 1]])
        vis.append(kpts[idx + 2])
    if any(v < -1e-6 or v > 1 + 1e-6 for v in xy):
        return None
    return np.asarray(xy, dtype=np.float32), np.asarray(vis, dtype=np.float32)


class FaceQuadDataset(Dataset):
    def __init__(self, root, split="train", imgsz=128):
        self.root = Path(root)
        self.imgsz = int(imgsz)
        image_dir = self.root / "images" / split
        label_dir = self.root / "labels" / split
        self.samples = []
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            parsed = parse_quad_label(label_path)
            if parsed is None:
                continue
            xy, vis = parsed
            self.samples.append((image_path, xy, vis))
        if not self.samples:
            raise RuntimeError(f"no B samples found: {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, xy, vis = self.samples[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"failed to read {image_path}")
        image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image[None, :, :])
        target = torch.from_numpy(xy.astype(np.float32))
        visibility = torch.from_numpy(vis.astype(np.float32))
        return image, target, visibility


class TinyQuadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(128, 8),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.features(x))


def choose_device(device_arg):
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    index = int(device_arg.split(",")[0])
    return torch.device(f"cuda:{index}")


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss = 0.0
    total_mae = 0.0
    total = 0
    for image, target, _vis in loader:
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            pred = model(image)
            loss = criterion(pred, target)
            if train:
                loss.backward()
                optimizer.step()
        batch = image.size(0)
        total_loss += float(loss.detach().cpu()) * batch
        total_mae += float((pred.detach() - target).abs().mean().cpu()) * batch
        total += batch
    return total_loss / max(total, 1), total_mae / max(total, 1)


def save_checkpoint(path, model, optimizer, epoch, best_val, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val,
            "args": vars(args),
            "model_name": "TinyQuadNet",
        },
        path,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = args.project / args.name
    weights_dir = run_dir / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    dataset = FaceQuadDataset(args.data, "train", args.imgsz)
    val_len = max(1, int(round(len(dataset) * args.val_fraction))) if len(dataset) > 10 else max(1, len(dataset) // 5)
    train_len = max(1, len(dataset) - val_len)
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = choose_device(args.device)
    model = TinyQuadNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss(beta=0.03)

    results_path = run_dir / "results.csv"
    start_epoch = 1
    best_val = float("inf")
    last_path = weights_dir / "last.pt"
    best_path = weights_dir / "best.pt"
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))

    if not results_path.exists() or start_epoch == 1:
        with results_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "time", "train_loss", "val_loss", "train_mae", "val_mae", "lr"])

    start_time = time.time()
    send_email(
        args,
        f"[start] B TinyQuadNet {args.name}",
        f"B TinyQuadNet training started\nrun: {run_dir}\ndata: {args.data}\nsamples: {len(dataset)}\ntrain/val: {train_len}/{val_len}\ntime: {now_text()}",
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_mae = run_epoch(model, val_loader, criterion, optimizer, device, False)
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        with results_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{elapsed:.3f}", f"{train_loss:.6f}", f"{val_loss:.6f}", f"{train_mae:.6f}", f"{val_mae:.6f}", f"{lr:.8f}"])

        save_checkpoint(last_path, model, optimizer, epoch, best_val, args)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, best_val, args)

        line = (
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"train_mae={train_mae:.5f} val_mae={val_mae:.5f}"
        )
        print(line, flush=True)
        if args.email_every > 0 and (epoch == 1 or epoch % args.email_every == 0 or epoch == args.epochs):
            per_epoch = elapsed / max(epoch - start_epoch + 1, 1)
            eta = per_epoch * max(args.epochs - epoch, 0)
            send_email(
                args,
                f"[progress] B TinyQuadNet {epoch}/{args.epochs}",
                f"{line}\nrun: {run_dir}\nelapsed: {format_seconds(elapsed)}\nremaining: {format_seconds(eta)}\ntime: {now_text()}",
            )

    send_email(
        args,
        f"[done] B TinyQuadNet {args.name}",
        f"B TinyQuadNet training done\nrun: {run_dir}\nbest_val_loss: {best_val:.6f}\ntime: {now_text()}",
    )


if __name__ == "__main__":
    main()
