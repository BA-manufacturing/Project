"""전경 마스크 점검 (노트북 2번을 여러 카테고리에 한 번에). 장서연.

    uv run python example/exp_fgcheck.py elec_01 elec_02 --thr 20 12 None

카테고리·임계값마다 출력:
    fg_area     전경이 사진에서 차지하는 비율 (val 정상 평균, 최소, 최대)
    cut10       결함의 10% 넘게 마스크 밖으로 잘리는 불량 사진 수 / 불량 수
    cut_mean    불량 사진의 결함 면적 중 잘린 비율 평균
runs/fg_<category>.png 에 정상 4장·불량 4장의 마스크 겹침 그림을 저장함.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import Dataset                       # noqa: E402
from src.features import upsample             # noqa: E402
from src.model import foreground              # noqa: E402
from src.paths import ROOT                    # noqa: E402


def thr_arg(s):
    return None if s == "None" else float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("categories", nargs="+")
    ap.add_argument("--thr", nargs="+", type=thr_arg, default=[20])
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    ds = Dataset(None, args.categories)
    for name in args.categories:
        cat = ds.category(name)
        x = cat.images("val")
        masks = cat.masks("val") > 0
        y = cat.val.is_anomaly.to_numpy() == 1
        for t in args.thr:
            fg = upsample(foreground(x, t).astype(np.float32)) > 0.5
            area = fg[~y].mean(axis=(1, 2))
            cut = np.array([(m & ~f).sum() / max(m.sum(), 1) for m, f in zip(masks[y], fg[y])])
            print(f"{name:8} thr={str(t):5} fg_area {area.mean():.2f} ({area.min():.2f}~{area.max():.2f})  "
                  f"cut10 {(cut > 0.1).sum():2d}/{y.sum():2d}  cut_mean {cut.mean():.3f}", flush=True)
        if args.plot:
            plot(name, x, masks, y, foreground(x, args.thr[0]))


def plot(name, x, masks, y, fg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = list(np.flatnonzero(~y)[:4]) + list(np.flatnonzero(y)[:4])
    fig, ax = plt.subplots(2, 8, figsize=(24, 6.5))
    for j, i in enumerate(idx):
        ax[0, j].imshow(x[i])
        ax[0, j].set_title(("good " if not y[i] else "bad ") + str(i))
        ov = x[i].astype(float) / 255 * 0.5
        f = upsample(fg[i:i + 1].astype(np.float32))[0] > 0.5
        ov[f] += [0, 0.4, 0]
        ov[masks[i]] = [1, 0, 0]
        ax[1, j].imshow(np.clip(ov, 0, 1))
        for a in ax[:, j]:
            a.axis("off")
    fig.suptitle(f"{name}: top=image, bottom=foreground (green) + defect (red)")
    fig.tight_layout()
    fig.savefig(ROOT / "runs" / f"fg_{name}.png", dpi=60)
    plt.close(fig)


if __name__ == "__main__":
    main()
