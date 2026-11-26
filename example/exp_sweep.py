"""옵션 비교를 빠르게 하는 실험 도구 (장서연). 저장소의 모델 코드는 건드리지 않음.

    uv run python example/exp_sweep.py elec_01 elec_02            저장된 모델로 top·sigma 비교
    uv run python example/exp_sweep.py elec_03 --fit fg_thr=None  옵션 하나를 바꿔 새로 학습해 비교

val 의 칸별 최근접 거리(격자)를 한 번만 계산해 runs/cache/ 에 두고, top·sigma 처럼 학습과 무관한
옵션은 그 격자에서 맵과 점수만 다시 만듦. 그래서 재학습 없이 몇 초 만에 비교됨.
--fit 은 train 으로 새 모델을 학습하되 artifacts/ 에는 저장하지 않음. val 은 채점에만 씀.

결과는 runs/sweep.csv 에 한 줄씩 덧붙임 (AUROC, AUPRO, F1@q). F1@q 는 정상 점수 상위 q 지점 임계값.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import Dataset, load_model, metrics                     # noqa: E402
from src.features import blur, cnn_features, upsample           # noqa: E402
from src.model import CONFIGS, Model, foreground                 # noqa: E402
from src.paths import ROOT                                       # noqa: E402

RUNS = ROOT / "runs"
CACHE = RUNS / "cache"
TOPS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
SIGMAS = [2, 4, 6]


def parse(kv):
    k, v = kv.split("=")
    return k, json.loads(v.replace("None", "null").replace("True", "true").replace("False", "false"))


def grids(model, images, batch=64):
    """(n, 32, 32) 마스크 전 최근접 거리."""
    out = []
    for i in range(0, len(images), batch):
        f = ((cnn_features(model.net, images[i:i + batch]) - model.mu) / model.sd).astype(np.float32)
        n, h, w, d = f.shape
        out.append(model._nearest(f.reshape(-1, d)).reshape(n, h, w))
    return np.concatenate(out)


def get_cache(cat, fit):
    tag = cat.name + ("" if not fit else "__" + "_".join(f"{k}-{v}" for k, v in fit.items()))
    path = CACHE / f"{tag}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return tag, z["grid"], json.loads(str(z["cfg"]))
    t0 = time.time()
    if fit:
        cfg = {**CONFIGS.get(cat.name, {}), **fit}
        model = Model(**cfg).fit(cat.images("train"))
    else:
        model = load_model(cat.name)
    g = grids(model, cat.images("val"))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, grid=g, cfg=json.dumps(model.cfg))
    print(f"  {tag}: 격자 계산 {time.time() - t0:.0f}초", flush=True)
    return tag, g, model.cfg


def scores_of(maps, top):
    m = maps.reshape(len(maps), -1)
    k = max(1, int(m.shape[1] * top))
    return np.partition(m, -k, axis=1)[:, -k:].mean(axis=1)


def f1_at(y, s, q):
    thr = np.quantile(s[y == 0], q)
    return metrics.f1(y, s >= thr)["f1"]


def sweep(name, fit, ds, only_default=False):
    cat = ds.category(name)
    tag, grid, cfg = get_cache(cat, fit)
    images = cat.images("val")
    fg = foreground(images, cfg["fg_thr"])
    masks = cat.masks("val")
    y = cat.val.is_anomaly.to_numpy()
    del images

    rows = []
    sigmas = [cfg["sigma"]] if only_default else SIGMAS
    for sigma in sigmas:
        maps = blur(upsample(grid * fg), sigma=sigma)
        pro = metrics.aupro(masks, maps)
        tops = [cfg["top"]] if (only_default or sigma != cfg["sigma"]) else TOPS
        for top in tops:
            s = scores_of(maps, top)
            rows.append(dict(category=name, run=tag, **{k: cfg[k] for k in ("fg_thr", "bank_size", "aug")},
                             top=top, sigma=sigma, auroc=metrics.auroc(y, s), aupro=pro,
                             f1_q95=f1_at(y, s, 0.95), f1_q99=f1_at(y, s, 0.99),
                             best_f1=max(metrics.f1(y, s >= t)["f1"] for t in np.unique(s[y == 1]))))
        del maps
    out = pd.DataFrame(rows)
    print(out.drop(columns=["category"]).round(3).to_string(index=False), "\n", flush=True)
    file = RUNS / "sweep.csv"
    out.to_csv(file, mode="a", header=not file.exists(), index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("categories", nargs="+")
    ap.add_argument("--fit", nargs="*", default=[], help="바꿀 옵션 key=value. 주면 새로 학습")
    ap.add_argument("--default-only", action="store_true", help="모델 옵션 그대로 한 줄만")
    args = ap.parse_args()
    fit = dict(parse(kv) for kv in args.fit)
    ds = Dataset(None, args.categories)
    for name in args.categories:
        sweep(name, fit, ds, args.default_only)


if __name__ == "__main__":
    main()
