"""runs/sweep.csv 를 기록표 형식으로 바꿈 (장서연).

    uv run python example/exp_record.py > runs/record.md

카테고리마다 기준(모델 옵션 그대로) 한 줄과, 옵션 하나만 바꾼 줄들을 전→후로 적음.
채택 기준(README_TEAM 규칙 4): AUROC 가 0.03 이상 오르고 AUPRO 가 0.01 넘게 떨어지지 않을 때.
AUROC 는 그대로이고 AUPRO 만 0.03 이상 오를 때도 채택.
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAIN, KEEP = 0.03, 0.01
OWNER = "장서연"


def main():
    d = pd.read_csv(ROOT / "runs" / "sweep.csv").drop_duplicates(
        ["category", "run", "top", "sigma"], keep="last")
    base_cfg = dict(top=0.01, sigma=4)
    lines = ["| 날짜 | 담당 | 카테고리 | 바꾼 것 | AUROC 전→후 | AUPRO 전→후 | 채택 | 메모 |",
             "|---|---|---|---|---|---|---|---|"]
    d[["auroc", "aupro"]] = d[["auroc", "aupro"]].round(3)        # 표에 적힌 값으로 판정
    for cat, g in d.groupby("category"):
        mine = g[g.run == cat]
        base = mine[(mine.top == base_cfg["top"]) & (mine.sigma == base_cfg["sigma"])]
        if base.empty:
            continue
        for _, r in g.iterrows():
            b = base.iloc[0]
            if r.run == cat:
                if r.top == b.top and r.sigma == b.sigma:
                    continue
                what = f"top={r.top:g}" if r.sigma == b.sigma else f"sigma={r.sigma:g}"
            else:
                what = r.run.split("__", 1)[1].replace("-", "=")
                # 재학습 실험은 그때의 top·sigma 와 같은 기준 줄과 비교 (CONFIGS 에 top 을 먼저 적은 경우)
                same = mine[(mine.top == r.top) & (mine.sigma == r.sigma)]
                if not same.empty:
                    b = same.iloc[0]
                    what += f" (top={r.top:g} 기준)" if r.top != base_cfg["top"] else ""
            da, dp = r.auroc - b.auroc, r.aupro - b.aupro
            ok = (da >= GAIN and dp > -KEEP) or (abs(da) < GAIN and dp >= GAIN and da > -KEEP)
            lines.append(f"| {date.today()} | {OWNER} | {cat} | {what} | {b.auroc:.3f}→{r.auroc:.3f} | "
                         f"{b.aupro:.3f}→{r.aupro:.3f} | {'O' if ok else 'X'} | "
                         f"F1@q95 {b.f1_q95:.3f}→{r.f1_q95:.3f} |")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
