"""제출 파일. 사전학습 ResNet 특징 + 정상 패치 메모리뱅크 (PatchCore 를 단순화한 것).

    load_model(category)    그 카테고리를 예측할 모델을 돌려줌
    Model.score_map(images) (n, 256, 256) 이상 맵. 클수록 이상
    Model.score(images)     (n,) 이미지 점수
    Model.threshold         판정 임계값. `team.py cv --save` 로 저장한 값이 있으면 읽음

모델 코드는 전 카테고리 공통이고, 카테고리마다 다른 것은 아래 CONFIGS 의 옵션뿐임.
사용법은 README_TEAM.md 참고.
"""

import json

import numpy as np

from src.features import blur, cnn_features, pool, resnet, upsample
from src.paths import ARTIFACTS

# 카테고리별 옵션. 각자 자기 카테고리 줄만 고칠 것. 없는 카테고리는 Model 의 기본값을 씀.
# 고친 뒤에는 그 카테고리를 다시 학습해야 반영됨 (main.py train --categories ...).
CONFIGS = {
    "elec_04": dict(fg_thr=12),      # 20 이면 어두운 결함 2장이 마스크 밖으로 잘림
    "elec_05": dict(fg_thr=12),

    "mech_02": dict(top=0.05),
    "mech_04": dict(top=0.10),
    "mech_07": dict(top=0.10),
    "med_08":  dict(top=0.10), 
}


def foreground(images, thr):
    """(n, 256, 256, 3) uint8 -> (n, 32, 32) bool. 배경보다 밝은 칸이 물체.

    thr 가 None 이면 마스크를 쓰지 않음 (전부 True).
    """
    n = len(images)
    if thr is None:
        return np.ones((n, 32, 32), bool)
    # shortcut: 어두운 배경을 가정한 고정 임계값, 배경이 밝은 카테고리는 fg_thr=None 으로 끌 것
    from scipy.ndimage import binary_dilation, label

    gray = np.asarray(images, np.float32).mean(axis=3, keepdims=True)
    fg = pool(gray, 8)[..., 0] > thr                  # 8픽셀 칸 평균 밝기
    for i in range(n):                                # 사진마다 가장 큰 덩어리만 남김 (배경 반사광 제거)
        lab, k = label(fg[i])
        if k > 1:
            fg[i] = lab == np.bincount(lab.ravel())[1:].argmax() + 1
    fg = binary_dilation(fg, structure=np.ones((1, 3, 3), bool))   # 한 칸 넓힘 (테두리, 싹·돌출)
    fg[fg.mean(axis=(1, 2)) < 0.10] = True            # 물체를 못 찾은 사진은 마스크 포기
    return fg

def greedy_coreset(x, k, rng, proj_dim=128):
    """서로 최대한 멀리 떨어진 k 개 행의 순번 (k-center greedy, PatchCore).
    무작위 투영으로 차원을 줄여 거리를 계산함. 고른 순번은 원래 특징에 그대로 씀."""
    import torch
    n, d = x.shape
    if k >= n:
        return np.arange(n)
    P = rng.standard_normal((d, proj_dim)).astype(np.float32) / np.sqrt(proj_dim)
    z = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32) @ P)
    zz = (z * z).sum(1)
    sel = np.empty(k, dtype=np.int64)
    sel[0] = int(rng.integers(n))
    mind = zz + zz[sel[0]] - 2 * (z @ z[sel[0]])          # 뽑힌 칸들까지의 최소 거리(제곱)
    for i in range(1, k):
        j = int(torch.argmax(mind))                         # 가장 덜 덮인 칸을 추가
        sel[i] = j
        mind = torch.minimum(mind, zz + zz[j] - 2 * (z @ z[j]))
    return sel


class Model:
    """패치 최근접 이웃.

    Args:
        bank_size: 메모리뱅크에 남길 정상 칸 수. 크면 정확하지만 느림.
        top: 이미지 점수로 쓸 맵 상위 비율. 작을수록 작은 결함에 민감하고 잡음에 약함.
        sigma: 이상 맵 블러 세기. 작을수록 좁은 결함이 덜 깎이고 잡음이 늘어남.
        fg_thr: 전경 마스크 밝기 임계값(0~255). None 이면 마스크를 끔.
        aug: 정상 이미지의 회전·뒤집기 사본 8가지를 뱅크에 넣음. 방향이 자유로운 물체에만.
    """

    threshold = None

    def __init__(self, bank_size=20000, top=0.01, sigma=4, fg_thr=20, aug=False, seed=0, coreset=False):
        self.cfg = dict(bank_size=bank_size, top=top, sigma=sigma, fg_thr=fg_thr, aug=aug, seed=seed, coreset=coreset)
        self.__dict__.update(self.cfg)
        self.net = resnet()

    def fit(self, images):
        """정상 이미지 (n, 256, 256, 3) uint8 로 학습. 메모리 때문에 100장씩 나눠 처리함."""
        rng = np.random.default_rng(self.seed)
        views = [lambda x: x]
        if self.aug:                                  # 90도 회전 4가지 x 좌우 뒤집기
            views = [lambda x, k=k, fl=fl: np.rot90(x[:, :, ::-1] if fl else x, k, axes=(1, 2))
                     for k in range(4) for fl in (False, True)]
        step = 100
        per = 5 * self.bank_size // (len(views) * -(-len(images) // step)) + 1   # 묶음당 뽑을 칸 수
        cand = []
        for i in range(0, len(images), step):
            for v in views:
                x = np.ascontiguousarray(v(images[i:i + step]))
                flat = cnn_features(self.net, x)[foreground(x, self.fg_thr)]     # 전경 칸만
                cand.append(flat[rng.choice(len(flat), min(per, len(flat)), replace=False)])
        flat = np.concatenate(cand)                   # 뱅크 크기의 약 5배 후보
        # 특징마다 단위가 달라서 정상 기준으로 표준화함
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-6
        flat = (flat - self.mu) / self.sd
        k = min(self.bank_size, len(flat))
        if self.coreset:                              # 골고루 덮도록 고름 (새 기능)
            keep = greedy_coreset(flat, k, rng)
        else:                                         # 원본: 무작위로 고름
            keep = rng.choice(len(flat), k, replace=False)
        self._set_bank(flat[keep].astype(np.float32))
        return self

    def _set_bank(self, bank):
        self.bank = bank
        self.bank_sq = (bank ** 2).sum(1)

    def _nearest(self, q, chunk=2048):
        """q 의 각 행에서 메모리뱅크 최근접 거리."""
        out = np.empty(len(q), dtype=np.float32)
        for i in range(0, len(q), chunk):
            x = q[i:i + chunk]
            d = (x ** 2).sum(1)[:, None] + self.bank_sq[None] - 2 * x @ self.bank.T
            out[i:i + chunk] = np.sqrt(np.maximum(d.min(1), 0))
        return out

    def score_map(self, images):
        """(n, 256, 256, 3) uint8 -> (n, 256, 256) float."""
        f = ((cnn_features(self.net, images) - self.mu) / self.sd).astype(np.float32)
        n, h, w, d = f.shape
        grid = self._nearest(f.reshape(-1, d)).reshape(n, h, w)
        grid = grid * foreground(images, self.fg_thr)     # 배경 칸은 0
        return blur(upsample(grid), sigma=self.sigma)

    def score(self, images):
        """(n,) 이미지 점수. 맵 상위 top 비율의 평균."""
        m = self.score_map(images).reshape(len(images), -1)
        k = max(1, int(m.shape[1] * self.top))
        return np.sort(m, axis=1)[:, -k:].mean(axis=1)

    # ------------------------------------------------------------ 저장과 불러오기
    # 학습 결과(배열)와 옵션만 저장함. 옵션을 함께 저장해야 채점 때 학습과 같은 처리가 보장됨.

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "model.npz", mu=self.mu, sd=self.sd, bank=self.bank,
                 config=json.dumps(self.cfg))
        # 다시 학습하면 점수 눈금이 바뀌므로 옛 임계값은 지움
        (folder / "threshold.txt").unlink(missing_ok=True)

    @classmethod
    def load(cls, folder):
        z = np.load(folder / "model.npz")
        m = cls(**json.loads(str(z["config"])))
        m.mu, m.sd = z["mu"], z["sd"]
        m._set_bank(z["bank"])
        thr = folder / "threshold.txt"
        if thr.exists():
            m.threshold = float(thr.read_text())
        return m

'''
def train(category, images):
    """카테고리 하나를 학습해서 artifacts/<category>/ 에 저장함. `main.py train` 이 부름."""
    Model(**CONFIGS.get(category, {})).fit(images).save(ARTIFACTS / category)


def load_model(category):
    """그 카테고리를 예측할 모델을 돌려줌. 채점할 때 이 함수로 가져와 score_map 만 돌림."""
    folder = ARTIFACTS / category
    if not (folder / "model.npz").exists():
        raise FileNotFoundError(f"{folder} 에 model.npz 가 없음. 먼저 학습할 것 "
                                f"(python main.py train --categories {category})")
    return Model.load(folder)'''
