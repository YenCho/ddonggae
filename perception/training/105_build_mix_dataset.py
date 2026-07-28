#!/usr/bin/env python3
"""합성 + 실사 **혼합 학습 목록** 구성 (2080Ti 원격 실행용, 2026-07-24).

문제: 합성 121,597 인스턴스 vs 실사 807 — 그대로 합치면 실사가 0.7% 라
학습 신호가 묻혀 apple→orange 오분류가 그대로 남는다. 반대로 합성을 빼면
pineapple·plain 이 무너진다(catastrophic forgetting). 그래서 양쪽을 맞춘다.

  · 합성: 클래스별 인스턴스 상한까지만 이미지를 고른다. **희소 클래스를 포함한
    이미지부터** 고르는 그리디 — plain 이 66k 로 과다해 무작위로 뽑으면
    plain 만 차고 과일이 모자란다.
  · 실사: 파일을 복제하지 않고 **목록에 N번 기재**해 에폭당 N번 보이게 한다
    (ultralytics 는 train 에 txt 목록 경로를 받는다).

클래스 순서는 양쪽 data.yaml 이 동일해야 한다 — 다르면 라벨이 뒤섞인다.
스크립트가 먼저 검사하고 다르면 중단한다.

사용(원격):
  python3 105_build_mix_dataset.py \
      --synth ~/team14_face/datasets/face_crops_full \
      --real  ~/team14_face/datasets/face_ft_real_20260724 \
      --out   ~/team14_face/datasets/mix_ft_20260724 \
      --synth-cap 3000 --real-repeat 8
"""
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_yaml_names(p: Path):
    names, in_names = {}, False
    for line in p.read_text().splitlines():
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            s = line.strip()
            if not s or not s[0].isdigit():
                break
            k, v = s.split(":", 1)
            names[int(k)] = v.strip()
    return names


def scan_labels(label_dir: Path):
    """이미지별 클래스 카운트."""
    out = {}
    for txt in label_dir.glob("*.txt"):
        c = Counter()
        for ln in txt.read_text().splitlines():
            ln = ln.strip()
            if ln:
                c[int(ln.split()[0])] += 1
        if c:
            out[txt.stem] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--synth-cap", default="3000",
                    help="합성 인스턴스 상한. 전체 공통 정수이거나 "
                         "'apple:3000,orange:2500,...' 클래스별 지정")
    # [2026-07-24] 상한 초과 허용 배율. 그리디가 부족한 클래스를 채우려고
    # 이미지를 고르면 같이 든 클래스가 딸려 들어와 상한을 크게 넘는다.
    # 첫 시도에서 orange 가 3000 상한인데 5847 까지 찼고, 그러면
    # apple→orange 편향을 되레 강화한다(우리가 고치려는 실패 그 자체).
    ap.add_argument("--over-ratio", type=float, default=1.15,
                    help="이 배율을 넘게 만드는 이미지는 채택하지 않음")
    ap.add_argument("--real-repeat", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sn = read_yaml_names(args.synth / "data.yaml")
    rn = read_yaml_names(args.real / "data.yaml")
    if sn != rn:
        raise SystemExit(f"클래스 순서 불일치 — 중단\n 합성 {sn}\n 실사 {rn}")
    print(f"클래스 순서 일치 확인: {sn}")

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- 합성 서브샘플 (희소 클래스 우선 그리디) ----
    syn = scan_labels(args.synth / "labels" / "train")
    print(f"합성 train 라벨 {len(syn)}장 스캔 완료")
    total = Counter()
    for c in syn.values():
        total.update(c)
    print("합성 전체 인스턴스:", {sn[k]: v for k, v in sorted(total.items())})

    inv = {v: k for k, v in sn.items()}
    if ":" in args.synth_cap:
        cap = {}
        for tok in args.synth_cap.split(","):
            k, v = tok.split(":")
            cap[inv[k.strip()]] = int(v)
        for k in sn:
            cap.setdefault(k, 0)
    else:
        cap = {k: int(args.synth_cap) for k in sn}
    print("합성 클래스별 상한:", {sn[k]: v for k, v in sorted(cap.items())})

    picked, got = [], Counter()
    stems = list(syn)
    rng.shuffle(stems)
    # 희소 클래스(전체 수가 적은 순)를 많이 가진 이미지를 먼저
    rank = {k: i for i, (k, _) in enumerate(sorted(total.items(), key=lambda kv: kv[1]))}
    stems.sort(key=lambda s: -sum(syn[s][k] * (len(rank) - rank[k]) for k in syn[s]))
    for s in stems:
        if all(got[k] >= cap[k] for k in sn):
            break
        # 아직 부족한 클래스를 하나도 안 채우면 건너뛴다
        if not any(got[k] < cap[k] for k in syn[s]):
            continue
        # 어떤 클래스든 상한*over_ratio 를 넘기게 되면 채택하지 않는다
        if any(got[k] + syn[s][k] > cap[k] * args.over_ratio for k in syn[s]):
            continue
        picked.append(s)
        got.update(syn[s])
    print(f"합성 선택 {len(picked)}장 / 인스턴스",
          {sn[k]: v for k, v in sorted(got.items())})

    # ---- 목록 작성 ----
    def img_path(root, split, stem):
        for ext in (".png", ".jpg", ".jpeg"):
            p = root / "images" / split / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    lines = []
    miss = 0
    for s in picked:
        p = img_path(args.synth, "train", s)
        if p is None:
            miss += 1
            continue
        lines.append(str(p.resolve()))
    real_train = sorted((args.real / "images" / "train").glob("*"))
    for _ in range(args.real_repeat):
        lines += [str(p.resolve()) for p in real_train]
    rng.shuffle(lines)
    (args.out / "mix_train.txt").write_text("\n".join(lines) + "\n")

    val_lines = [str(p.resolve()) for p in sorted((args.real / "images" / "val").glob("*"))]
    (args.out / "mix_val.txt").write_text("\n".join(val_lines) + "\n")

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        f"train: {(args.out / 'mix_train.txt').resolve()}\n"
        f"val: {(args.out / 'mix_val.txt').resolve()}\n"
        "names:\n" + "".join(f"  {k}: {v}\n" for k, v in sorted(sn.items())))

    n_real = len(real_train) * args.real_repeat
    n_syn = len(lines) - n_real
    print(f"\nmix_train {len(lines)}줄 = 합성 {n_syn} + 실사 {len(real_train)}x{args.real_repeat}={n_real}")
    print(f"  실사 지분 {n_real/len(lines)*100:.0f}%  (합성:실사 = {n_syn/n_real:.1f}:1)")
    print(f"mix_val {len(val_lines)}줄 (실사 val — 셀 GT 기반)")
    if miss:
        print(f"⚠ 합성 이미지 누락 {miss}장")
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
