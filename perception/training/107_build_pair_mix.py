#!/usr/bin/env python3
"""pair 검증기 **합성+실사 혼합 학습셋** 구성 (2026-07-24 조작자 지시).

배경: 15:45 실기의 apple→orange 오분류는 face 가 틀린 것만이 아니라 **AO
검증기가 그 틀린 답에 동의**해서 확정됐다. face 만 실사로 파인튜닝하면 AO 가
되돌려버리므로 AO 도 같은 실사로 재학습해야 한다.

합성 14,959/클래스 vs 실사 171/131 — 그대로 합치면 실사 지분이 1% 라 학습
신호가 묻힌다. 105_build_mix_dataset.py 와 같은 처방을 쓴다:
  · 합성은 클래스별 상한까지 **무작위 서브샘플**
  · 실사는 심볼릭 링크를 N개 만들어 에폭당 N번 보이게 (**오버샘플**)

주의 — 실사 val 은 절대 train 에 넣지 않는다. 학습 스크립트
(train_face_mobilenetv3.py)는 train/ 만 읽고 --val_fraction 으로 자체 분할하는데,
오버샘플된 실사가 양쪽에 걸치면 val_acc 가 누수로 오염된다(7/23 BP 에서 겪음).
그래서 학습 중 val_acc 는 참고만 하고, **정직한 평가는 별도로**
100_eval_pair_on_val.py 로 실사 val 에 대해 돌린다.

사용(원격 2080Ti):
  python3 107_build_pair_mix.py \
      --synth ~/team14_face/datasets/pair_apple_orange_v2 \
      --real  ~/team14_face/datasets/pair_ao_real_20260724 \
      --out   ~/team14_face/datasets/pair_ao_mix_20260724 \
      --synth-cap 3000 --real-repeat 8
"""
import argparse
import random
import shutil
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def images(d: Path):
    return sorted(p for p in d.glob("*") if p.suffix.lower() in EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--synth-cap", type=int, default=3000,
                    help="클래스별 합성 상한")
    ap.add_argument("--real-repeat", type=int, default=8,
                    help="실사 1장을 링크 몇 개로 늘릴지")
    ap.add_argument("--seed", type=int, default=20260724)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    classes = sorted(d.name for d in (args.synth / "train").iterdir() if d.is_dir())
    real_classes = sorted(d.name for d in (args.real / "train").iterdir() if d.is_dir())
    if classes != real_classes:
        raise SystemExit(f"클래스 불일치 — 중단\n 합성 {classes}\n 실사 {real_classes}")
    print(f"클래스 {classes}")

    if args.out.exists():
        shutil.rmtree(args.out)
    stat = {}
    for c in classes:
        dst = args.out / "train" / c
        dst.mkdir(parents=True)

        syn = images(args.synth / "train" / c)
        rng.shuffle(syn)
        syn = syn[:args.synth_cap]
        for p in syn:
            (dst / f"syn_{p.name}").symlink_to(p.resolve())

        real = images(args.real / "train" / c)
        for r in range(args.real_repeat):
            for p in real:
                (dst / f"real_r{r}_{p.name}").symlink_to(p.resolve())

        n_real = len(real) * args.real_repeat
        stat[c] = (len(syn), len(real), n_real)
        print(f"  {c:10s} 합성 {len(syn):5d} + 실사 {len(real):4d}x{args.real_repeat}"
              f"={n_real:5d}  = {len(syn)+n_real:5d}  (실사 지분 "
              f"{n_real/(len(syn)+n_real)*100:.0f}%)")

    # 실사 val 은 **복사하지 않는다** — 별도 평가용으로 원본 경로를 그대로 쓴다.
    print(f"\n{args.out}")
    val_desc = ", ".join(f"{c} {len(images(args.real / 'val' / c))}" for c in classes)
    print(f"평가는 원본 실사 val 로: {args.real / 'val'} ({val_desc})")


if __name__ == "__main__":
    main()
