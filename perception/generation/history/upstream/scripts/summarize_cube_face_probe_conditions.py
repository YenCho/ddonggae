from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


FRUITS = {"apple", "orange", "banana", "pineapple"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize condition-wise cube-face unified probe reports.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate_output", type=Path, required=True)
    parser.add_argument("--orange_recall_threshold", type=float, default=0.92)
    parser.add_argument("--orange_to_apple_threshold", type=float, default=0.05)
    parser.add_argument("--orange_to_plain_threshold", type=float, default=0.08)
    parser.add_argument("--apple_to_orange_threshold", type=float, default=0.05)
    return parser.parse_args()


def read_rows(report: Path) -> list[dict[str, str]]:
    csv_path = report / "crop_top_class_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def metrics_for_rows(rows: list[dict[str, str]]) -> dict:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        matrix[row.get("true", "none")][row.get("pred", "none")] += 1

    orange_total = sum(matrix["orange"].values())
    apple_total = sum(matrix["apple"].values())
    fruit_total = sum(matrix[t][p] for t in FRUITS for p in list(FRUITS) + ["plain", "none"])
    fruit_wrong = sum(matrix[t][p] for t in FRUITS for p in FRUITS if p != t)
    fruit_plain = sum(matrix[t][p] for t in FRUITS for p in ["plain", "none"])
    true_conf_lt080 = sum(
        1
        for row in rows
        if row.get("true") in FRUITS and float(row.get("true_conf") or 0.0) < 0.80
    )
    return {
        "images": len(rows),
        "orange_total": orange_total,
        "orange_correct": matrix["orange"]["orange"],
        "orange_recall": matrix["orange"]["orange"] / max(1, orange_total),
        "orange_to_apple": matrix["orange"]["apple"],
        "orange_to_apple_rate": matrix["orange"]["apple"] / max(1, orange_total),
        "orange_to_plain_or_blank": matrix["orange"]["plain"] + matrix["orange"]["none"],
        "orange_to_plain_or_blank_rate": (matrix["orange"]["plain"] + matrix["orange"]["none"]) / max(1, orange_total),
        "orange_to_banana_pineapple_noise": matrix["orange"]["banana"] + matrix["orange"]["pineapple"],
        "orange_to_banana_pineapple_noise_rate": (matrix["orange"]["banana"] + matrix["orange"]["pineapple"]) / max(1, orange_total),
        "apple_total": apple_total,
        "apple_to_orange": matrix["apple"]["orange"],
        "apple_to_orange_rate": matrix["apple"]["orange"] / max(1, apple_total),
        "fruit_to_fruit_wrong": fruit_wrong,
        "fruit_to_fruit_wrong_rate": fruit_wrong / max(1, fruit_total),
        "fruit_to_plain_or_blank": fruit_plain,
        "fruit_to_plain_or_blank_rate": fruit_plain / max(1, fruit_total),
        "true_conf_lt080": true_conf_lt080,
        "matrix": {t: dict(matrix[t]) for t in sorted(matrix)},
    }


def copy_candidate_rows(rows: list[dict[str, str]], output: Path, prefix: str) -> int:
    copied = 0
    for row in rows:
        true = row.get("true", "none")
        pred = row.get("pred", "none")
        weak = float(row.get("true_conf") or 0.0) < 0.80
        if true not in FRUITS:
            continue
        if pred == true and not weak:
            continue
        image = Path(row["image"])
        label = label_for_image(image)
        if not image.exists() or not label.exists():
            continue
        stem = f"{prefix}_{copied:05d}_{true}_pred_{pred}_{image.stem}"
        image_out = output / "images" / "val" / f"{stem}{image.suffix.lower()}"
        label_out = output / "labels" / "val" / f"{stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_out)
        shutil.copy2(label, label_out)
        copied += 1
    return copied


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.candidate_output.exists():
        shutil.rmtree(args.candidate_output)
    args.candidate_output.mkdir(parents=True, exist_ok=True)

    entries = []
    candidate_sources = []
    for item in manifest["reports"]:
        rows = read_rows(Path(item["report"]))
        metrics = metrics_for_rows(rows)
        reasons = []
        if item["class"] == "orange":
            if metrics["orange_recall"] < args.orange_recall_threshold:
                reasons.append("low orange recall")
            if metrics["orange_to_apple_rate"] >= args.orange_to_apple_threshold:
                reasons.append("orange->apple boundary")
            if metrics["orange_to_plain_or_blank_rate"] >= args.orange_to_plain_threshold:
                reasons.append("orange->plain/blank; do not train as orange positive")
            if item["condition"] == "white_face_dominant" and (
                metrics["orange_recall"] < 0.95 or metrics["orange_to_banana_pineapple_noise_rate"] >= 0.05
            ):
                reasons.append("white/blank-dominant; separate as blank/ambiguous negative after visual review")
        if item["class"] == "apple" and metrics["apple_to_orange_rate"] >= args.apple_to_orange_threshold:
            reasons.append("apple->orange boundary negative")
        if metrics["orange_to_banana_pineapple_noise_rate"] >= 0.05:
            reasons.append("banana/pineapple noise in orange probe")

        entry = {**item, **metrics, "candidate": bool(reasons), "reasons": reasons}
        entries.append(entry)
        if reasons:
            copied = copy_candidate_rows(rows, args.candidate_output, f'{item["condition"]}_{item["class"]}')
            candidate_sources.append({**item, "reasons": reasons, "candidate_crops": copied})

    (args.output / "condition_summary.json").write_text(
        json.dumps({"reports": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.candidate_output / "candidate_sources.json").write_text(
        json.dumps({"sources": candidate_sources}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.candidate_output / "DO_NOT_TRAIN_UNTIL_REVIEW.md").write_text(
        "# Review Required\n\n"
        "This folder contains candidate hard crops copied from low-performing probe conditions.\n"
        "Review contact sheets and condition reports before promoting anything to a train split.\n"
        "White/blank-dominant orange cases must be handled as blank/ambiguous negatives, not orange positives.\n",
        encoding="utf-8",
    )

    lines = [
        "# Cube-Face Unified Probe Condition Summary",
        "",
        "Preferred unified is evaluated only on exported A1-style 224x224 cube-face crops, not full-frame images.",
        "",
        "| condition | class | orange recall | orange->apple | orange->plain/blank | banana/pineapple noise | apple->orange | candidate | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for entry in entries:
        lines.append(
            "| {condition} | {class_name} | {orange_recall:.3f} | {orange_apple:.3f} | {orange_plain:.3f} | {noise:.3f} | {apple_orange:.3f} | {candidate} | {reasons} |".format(
                condition=entry["condition"],
                class_name=entry["class"],
                orange_recall=entry["orange_recall"],
                orange_apple=entry["orange_to_apple_rate"],
                orange_plain=entry["orange_to_plain_or_blank_rate"],
                noise=entry["orange_to_banana_pineapple_noise_rate"],
                apple_orange=entry["apple_to_orange_rate"],
                candidate="yes" if entry["candidate"] else "no",
                reasons=", ".join(entry["reasons"]) or "-",
            )
        )

    lines.extend(["", "## Boost Priority", ""])
    if not candidate_sources:
        lines.append("- No condition crossed the candidate thresholds. Keep these as probe-only validation sets.")
    else:
        for source in candidate_sources:
            reason_text = " ".join(source["reasons"])
            if "white/blank-dominant" in reason_text or "plain/blank" in reason_text:
                action = "separate as blank/ambiguous negative; do not use as orange positive"
            elif source["class"] == "apple":
                action = "use as apple-orange boundary negative probe before training"
            else:
                action = "candidate for conservative booster after visual review"
            lines.append(f'- `{source["condition"]}` `{source["class"]}`: {", ".join(source["reasons"])} -> {action}')

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{args.output / 'condition_summary.json'}`",
            f"- boost candidate folder: `{args.candidate_output}`",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
