#!/usr/bin/env python
"""팀 공용 도구: fold 배정과 5-fold 임계값 평가.

    python team.py folds                          folds.csv 생성. 한 번만. 이미 있으면 건드리지 않음
    python team.py cv scores.csv                  5-fold 로 임계값 규칙을 비교하고 F1 을 추정
    python team.py cv scores.csv --table t.csv    최종 점수 (AUROC + F1 + AUPRO) / 3 추정까지
    python team.py cv scores.csv --save           val 전체로 임계값을 정해 artifacts/<카테고리>/threshold.txt 에 저장

scores.csv 는 `python main.py eval --out scores.csv` 가 만든 이미지별 점수임.
모델을 다시 돌리지 않고 그 표만 나눠 봄.

F1 은 "fold 밖" 판정으로 잼. 각 사진을, 그 사진이 빠진 나머지 4조각으로 정한 임계값으로 판정한 뒤
카테고리별 F1 을 계산함. 임계값을 정하는 데 쓰지 않은 사진으로만 재므로 test 점수에 가까움.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.paths import ARTIFACTS, DATASET, ROOT

FOLDS = ROOT / "folds.csv"
K, SEED = 5, 0
QUANTILES = np.round(np.arange(0.50, 0.995, 0.01), 2)


def make_folds():
    """val 을 카테고리별로 5조각. 정상과 각 불량 유형이 조각마다 고르게 들어가게 나눔.

    유형별로 섞은 뒤 한 장씩 차례로 조각에 돌려 넣음. 그래서 어떤 유형이든 조각 간 장수 차이가
    1 이하이고, 불량 전체와 정상의 조각 간 차이도 1 이하임. 5장 미만인 유형은 서로 다른 조각에 흩어짐.
    """
    val = pd.read_csv(DATASET / "index.csv").query("split == 'val'")
    rng = np.random.default_rng(SEED)
    parts = []
    for name, g in val.groupby("category"):
        for _, side in g.groupby("is_anomaly"):                   # 정상과 불량을 따로 돌려 넣음
            # 장수가 많은 유형부터. 이어서 돌려 넣으므로 유형이 바뀌어도 조각별 합계가 고르게 유지됨
            labels = side.label.value_counts().index
            paths = np.concatenate([rng.permutation(side.path[side.label == lab].to_numpy()) for lab in labels])
            parts.append(pd.DataFrame({"category": name, "path": paths, "fold": np.arange(len(paths)) % K}))
    folds = pd.concat(parts, ignore_index=True)
    assert len(folds) == len(val) and folds.path.is_unique
    return folds


def best_f1_threshold(y, s):
    """F1 이 가장 높은 임계값. 후보는 점수 값들."""
    cand = np.unique(s)
    return cand[np.argmax([f1_score(y, s >= t, zero_division=0) for t in cand])]


def rule_quantile(tune, q=None):
    """정상 점수의 상위 분위수. q 는 전 카테고리 공통 하나 (카테고리 평균 F1 이 가장 높은 값)."""
    groups = [(c, g.is_anomaly.to_numpy(), g.score.to_numpy()) for c, g in tune.groupby("category")]
    if q is None:
        mean_f1 = [np.mean([f1_score(y, s >= np.quantile(s[y == 0], q), zero_division=0)
                            for _, y, s in groups]) for q in QUANTILES]
        q = QUANTILES[int(np.argmax(mean_f1))]
    return {c: float(np.quantile(s[y == 0], q)) for c, y, s in groups}, q


def rule_per_category(tune, _=None):
    """카테고리마다 F1 이 가장 높은 임계값. 불량이 적어 val 에 과적합되기 쉬움."""
    return {c: float(best_f1_threshold(g.is_anomaly.to_numpy(), g.score.to_numpy()))
            for c, g in tune.groupby("category")}, None


RULES = {"분위수(공통 q)": rule_quantile, "카테고리별 최적": rule_per_category}


def f1_by_category(d, pred):
    return d.assign(pred=pred).groupby("category").apply(
        lambda g: f1_score(g.is_anomaly, g.pred, zero_division=0), include_groups=False)


def cv(scores, table=None, save=False, rule_name="분위수(공통 q)"):
    d = pd.read_csv(scores).merge(pd.read_csv(FOLDS)[["path", "fold"]], on="path", validate="1:1")
    out = pd.DataFrame({"n_bad": d.groupby("category").is_anomaly.sum(),
                        "AUROC": d.groupby("category").apply(
                            lambda g: roc_auc_score(g.is_anomaly, g.score), include_groups=False),
                        "F1 기본(상위 1%)": f1_by_category(d, d.pred.to_numpy())})
    fold_means = {}
    for name, rule in RULES.items():
        pred = np.zeros(len(d), int)
        for k in range(K):
            hold = (d.fold == k).to_numpy()
            thr, _ = rule(d[~hold])
            pred[hold] = d.score[hold] >= d.category[hold].map(thr)
        out[f"F1 fold밖 {name}"] = f1_by_category(d, pred)
        # 낙관적인 값: 임계값을 정한 바로 그 사진으로 잰 F1. fold밖 값과의 차이가 과적합 정도
        thr_all, _ = rule(d)
        out[f"F1 val맞춤 {name}"] = f1_by_category(d, (d.score >= d.category.map(thr_all)).to_numpy())
        fold_means[name] = [f1_by_category(d[d.fold == k], pred[d.fold == k]).mean() for k in range(K)]

    pd.options.display.float_format = "{:.3f}".format
    print(out.to_string())
    print("\n카테고리 평균")
    print(out.drop(columns="n_bad").mean().to_string())
    print("\nfold 별 평균 F1 의 표준편차 (흔들리는 정도)")
    for name, v in fold_means.items():
        print(f"  {name:12} {np.std(v):.3f}")

    # 기본은 분위수 규칙. 카테고리별 최적은 과적합이 커서 fold밖 F1 이 확실히 높을 때만 --rule 로 고를 것
    best = rule_name
    f1 = out[f"F1 fold밖 {best}"].mean()
    print(f"\n사용 규칙: {best} (fold밖 F1 {f1:.3f})")
    if table:
        t = pd.read_csv(table).set_index("category").loc[out.index]
        print(f"최종 점수 추정 (AUROC {t.image_auroc.mean():.3f} + F1 {f1:.3f} + AUPRO {t.pixel_aupro.mean():.3f}) / 3"
              f" = {(t.image_auroc.mean() + f1 + t.pixel_aupro.mean()) / 3:.3f}")
    if save:
        thr, q = RULES[best](d)
        for c, t in thr.items():
            (ARTIFACTS / c / "threshold.txt").write_text(repr(t))
        print(f"\n임계값 {len(thr)}개 저장 (규칙: {best}{'' if q is None else f', q={q}'}). "
              f"다시 학습하면 지워지니 그때는 eval 과 cv --save 를 다시 할 것")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["folds", "cv"])
    ap.add_argument("scores", nargs="?", help="main.py eval --out 으로 저장한 csv")
    ap.add_argument("--table", help="main.py eval --table 로 저장한 csv. 주면 최종 점수 추정을 냄")
    ap.add_argument("--save", action="store_true", help="임계값을 artifacts 에 저장")
    ap.add_argument("--rule", default="quantile", choices=["quantile", "per_category"],
                    help="최종 점수 추정과 --save 에 쓸 임계값 규칙. 기본 quantile")
    args = ap.parse_args()

    if args.command == "folds":
        if FOLDS.exists():
            print(f"{FOLDS.name} 이 이미 있음. 모두 같은 배정을 써야 하므로 다시 만들지 않음")
        else:
            folds = make_folds()
            folds.to_csv(FOLDS, index=False)
            print(f"{FOLDS.name} 저장: {len(folds)}장, 카테고리 {folds.category.nunique()}개")
    elif not args.scores:
        ap.error("cv 에는 scores.csv 경로가 필요함")
    else:
        cv(args.scores, args.table, args.save, list(RULES)[args.rule == "per_category"])


if __name__ == "__main__":
    main()
