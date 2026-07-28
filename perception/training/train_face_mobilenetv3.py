import argparse
import csv
import json
import pathlib
import sys as _sys
if _sys.platform == "win32":   # load linux-saved checkpoints (PosixPath) on Windows
    pathlib.PosixPath = pathlib.WindowsPath
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


CLASSES = ("apple", "orange", "banana", "pineapple", "plain", "unknown")


def parse_args():
    parser = argparse.ArgumentParser(description="Train C: MobileNetV3-Small face classifier.")
    parser.add_argument("--data", type=Path, required=True, help="c_facecls dataset root.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/meta_v2_c_facecls"))
    parser.add_argument("--name", default="c_mobilenetv3small_meta_v2_10000")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--init_checkpoint", type=Path, default=None, help="Optional C checkpoint to initialize model weights before training.")
    parser.add_argument("--amp", action="store_true", default=False, help="Use CUDA mixed precision training.")
    parser.add_argument("--channels_last", action="store_true", default=False, help="Use channels_last memory format on CUDA.")
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--email_to", default="")
    # Training-progress email notifications. Disabled by default in the public
    # release; supply your own SMTP config to re-enable.
    parser.add_argument("--email_config", default="")
    parser.add_argument("--email_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--classes", default="", help="Comma list overriding default class set, e.g. apple,orange")
    parser.add_argument("--backbone", default="mobilenet_v3_small", choices=["mobilenet_v3_small","mobilenet_v3_large","efficientnet_b0","convnext_tiny"])
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


class FaceClsDataset(Dataset):
    def __init__(self, root, split="train", imgsz=128, augment=False):
        self.root = Path(root)
        self.split = split
        self.imgsz = int(imgsz)
        self.augment = augment
        self.samples = []
        for class_id, class_name in enumerate(CLASSES):
            class_dir = self.root / split / class_name
            for path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((path, class_id))
        if not self.samples:
            raise RuntimeError(f"no C classifier samples found: {self.root / split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        if self.augment:
            image = self.apply_aug(image)
        image = image.astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = torch.from_numpy(image.transpose(2, 0, 1))
        return image, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def apply_aug(image):
        out = image
        if random.random() < 0.5:
            out = cv2.flip(out, 1)
        if random.random() < 0.6:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-10, 10)
            out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.25:
            k = random.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)
        return out


def class_counts(dataset):
    counts = {name: 0 for name in CLASSES}
    for _path, label in dataset.samples:
        counts[CLASSES[label]] += 1
    return counts


def make_model(num_classes, pretrained=True, backbone="mobilenet_v3_small"):
    try:
        import torchvision.models as tvm
        if backbone == "mobilenet_v3_small":
            m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        elif backbone == "mobilenet_v3_large":
            m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        elif backbone == "efficientnet_b0":
            m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        elif backbone == "convnext_tiny":
            m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        else:
            raise ValueError(f"unknown backbone {backbone}")
        return m, backbone, bool(pretrained)
    except Exception as exc:
        print(f"backbone {backbone} unavailable/pretrained load failed: {exc}", flush=True)
        return MiniClassifier(num_classes), "mini_classifier_fallback", False


class MiniClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 160, 3, stride=2, padding=1),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(160, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def choose_device(device_arg):
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    index = int(device_arg.split(",")[0])
    return torch.device(f"cuda:{index}")


def make_sampler(dataset, subset):
    labels = [dataset.samples[idx][1] for idx in subset.indices]
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float32)
    weights = np.zeros(len(CLASSES), dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = 1.0 / counts[nonzero]
    sample_weights = [float(weights[label]) for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def run_epoch(model, loader, criterion, optimizer, device, train, scaler=None, use_amp=False, channels_last=False):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for image, label in loader:
        image = image.to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            image = image.contiguous(memory_format=torch.channels_last)
        label = label.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=device.type, enabled=bool(use_amp and device.type == "cuda")):
            logits = model(image)
            loss = criterion(logits, label)
            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        batch = image.size(0)
        pred = logits.detach().argmax(dim=1)
        total_loss += float(loss.detach().cpu()) * batch
        total_correct += int((pred == label).sum().detach().cpu())
        total += batch
    return total_loss / max(total, 1), total_correct / max(total, 1)


def save_checkpoint(path, model, optimizer, epoch, best_metric, args, model_name, pretrained_loaded):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_acc": best_metric,
            "classes": CLASSES,
            "args": vars(args),
            "model_name": model_name,
            "pretrained_loaded": pretrained_loaded,
        },
        path,
    )


def main():
    global CLASSES
    args = parse_args()
    if args.classes:
        CLASSES = tuple(c.strip() for c in args.classes.split(",") if c.strip())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    run_dir = args.project / args.name
    weights_dir = run_dir / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    train_dataset = FaceClsDataset(args.data, "train", args.imgsz, augment=True)
    counts = class_counts(train_dataset)
    has_val_dir = all((args.data / "val" / c).is_dir() for c in CLASSES)
    if has_val_dir:
        # source-grouped holdout: val/ built from cutouts held out of train (no
        # augmented-variant leakage). Use all of train as train.
        val_dataset = FaceClsDataset(args.data, "val", args.imgsz, augment=False)
        train_set = Subset(train_dataset, list(range(len(train_dataset))))
        val_set = Subset(val_dataset, list(range(len(val_dataset))))
        print(f"[val] using held-out source split: train={len(train_set)} val={len(val_set)}")
    else:
        val_dataset = FaceClsDataset(args.data, "train", args.imgsz, augment=False)
        val_len = max(1, int(round(len(train_dataset) * args.val_fraction))) if len(train_dataset) > 10 else max(1, len(train_dataset) // 5)
        indices = list(range(len(train_dataset)))
        random.Random(args.seed).shuffle(indices)
        train_set = Subset(train_dataset, indices[val_len:])
        val_set = Subset(val_dataset, indices[:val_len])
    train_len, val_len = len(train_set), len(val_set)

    train_sampler = make_sampler(train_dataset, train_set)
    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = max(2, int(args.prefetch_factor))
    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, **loader_kwargs)

    device = choose_device(args.device)
    model, model_name, pretrained_loaded = make_model(len(CLASSES), args.pretrained, args.backbone)
    model = model.to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_count_array = np.asarray([counts[name] for name in CLASSES], dtype=np.float32)
    class_weights = np.ones(len(CLASSES), dtype=np.float32)
    nonzero = class_count_array > 0
    class_weights[nonzero] = class_count_array[nonzero].sum() / (class_count_array[nonzero] * max(nonzero.sum(), 1))
    class_weights[~nonzero] = 0.0
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))

    results_path = run_dir / "results.csv"
    start_epoch = 1
    best_val_acc = 0.0
    last_path = weights_dir / "last.pt"
    best_path = weights_dir / "best.pt"
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))
    elif args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        print(f"initialized from checkpoint: {args.init_checkpoint}", flush=True)

    if not results_path.exists() or start_epoch == 1:
        with results_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "time", "train_loss", "val_loss", "train_acc", "val_acc", "lr"])

    start_time = time.time()
    send_email(
        args,
        f"[start] C MobileNetV3 {args.name}",
        (
            f"C classifier training started\nrun: {run_dir}\ndata: {args.data}\n"
            f"model: {model_name}, pretrained_loaded={pretrained_loaded}\n"
            f"init_checkpoint: {args.init_checkpoint}\n"
            f"amp: {args.amp}, channels_last: {args.channels_last}, batch: {args.batch}, workers: {args.workers}\n"
            f"samples: {len(train_dataset)}\nclass_counts: {counts}\ntrain/val: {train_len}/{val_len}\ntime: {now_text()}"
        ),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            True,
            scaler=scaler,
            use_amp=args.amp,
            channels_last=args.channels_last,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            False,
            scaler=None,
            use_amp=args.amp,
            channels_last=args.channels_last,
        )
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        with results_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{elapsed:.3f}", f"{train_loss:.6f}", f"{val_loss:.6f}", f"{train_acc:.6f}", f"{val_acc:.6f}", f"{lr:.8f}"])

        save_checkpoint(last_path, model, optimizer, epoch, best_val_acc, args, model_name, pretrained_loaded)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_path, model, optimizer, epoch, best_val_acc, args, model_name, pretrained_loaded)

        line = (
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )
        print(line, flush=True)
        if args.email_every > 0 and (epoch == 1 or epoch % args.email_every == 0 or epoch == args.epochs):
            per_epoch = elapsed / max(epoch - start_epoch + 1, 1)
            eta = per_epoch * max(args.epochs - epoch, 0)
            send_email(
                args,
                f"[progress] C MobileNetV3 {epoch}/{args.epochs}",
                f"{line}\nrun: {run_dir}\nelapsed: {format_seconds(elapsed)}\nremaining: {format_seconds(eta)}\ntime: {now_text()}",
            )

    send_email(
        args,
        f"[done] C MobileNetV3 {args.name}",
        f"C classifier training done\nrun: {run_dir}\nbest_val_acc: {best_val_acc:.4f}\ntime: {now_text()}",
    )


if __name__ == "__main__":
    main()
