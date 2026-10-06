import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


CLASSES = ("apple", "orange", "banana", "pineapple", "plain", "unknown")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge one or more C face-classifier datasets into train/class/*.jpg."
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def reset_output(path):
    if path.exists():
        shutil.rmtree(path)


def ensure_dirs(path, splits):
    for split in splits:
        for class_name in CLASSES:
            (path / split / class_name).mkdir(parents=True, exist_ok=True)


def list_images(root, split, class_name):
    class_dir = root / split / class_name
    if not class_dir.exists():
        return []
    return sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def link_or_copy(src, dst, mode, resume=False):
    if resume and dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        try:
            dst.hardlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
    return True


def main():
    args = parse_args()
    inputs = [p.resolve() for p in args.inputs]
    output_root = args.output_root.resolve()
    if args.reset:
        reset_output(output_root)
    ensure_dirs(output_root, args.splits)

    counts = defaultdict(Counter)
    sources = defaultdict(Counter)
    skipped = 0
    written = 0
    jobs = []
    for source_idx, root in enumerate(inputs):
        source_name = root.name
        for split in args.splits:
            for class_name in CLASSES:
                for src in list_images(root, split, class_name):
                    ext = ".jpg" if src.suffix.lower() not in IMAGE_EXTS else src.suffix.lower()
                    dst_name = f"s{source_idx:02d}_{source_name}_{src.stem}{ext}"
                    dst = output_root / split / class_name / dst_name
                    jobs.append((src, dst, split, class_name, source_name))

    for src, dst, split, class_name, source_name in tqdm(jobs, desc="merge facecls", unit="img"):
        if link_or_copy(src, dst, args.mode, resume=args.resume):
            written += 1
        else:
            skipped += 1
        counts[split][class_name] += 1
        sources[source_name][class_name] += 1

    (output_root / "classes.json").write_text(
        json.dumps({"classes": list(CLASSES), "format": "split/class/*.jpg"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit = {
        "inputs": [str(p) for p in inputs],
        "output_root": str(output_root),
        "splits": list(args.splits),
        "mode": args.mode,
        "written": written,
        "skipped": skipped,
        "counts": {split: dict(counter) for split, counter in counts.items()},
        "sources": {source: dict(counter) for source, counter in sources.items()},
    }
    audit_path = output_root / "merge_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"wrote: {output_root}")


if __name__ == "__main__":
    main()
