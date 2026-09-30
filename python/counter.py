"""塞頭計數演算法（純 OpenCV，與介面無關）。

一顆塞頭 = 一根管子，所以數塞頭就是數管子。

作法是「面積相除」：把照片裡的彩色塞頭面切出來，量出單顆的面積，
再讓每一塊區域各自除以單顆面積。黏在一起的兩顆面積約兩倍，自然
被算成 2。

這份是瀏覽器版 (index.html) 的 Python 移植，演算法與參數相同，
唯一的差別寫在 cross_cores() 的註解裡。

用法：
    import cv2
    from counter import count_plugs, Params

    img = cv2.imread("photo.jpg")           # BGR
    res = count_plugs(img, Params())
    print(res.count)                        # 顆數
    cv2.imwrite("out.jpg", res.annotate())  # 標了編號的圖
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------- 參數


@dataclass
class Params:
    """所有可調參數。預設值已用一張 20 顆塞頭的照片驗證過。"""

    # --- 顏色（色相以「度」為單位，0-360；OpenCV 內部是 0-179，程式會自動換算）---
    hue_center: float = 168.0   # 綠色塞頭的色相中心
    hue_width: float = 50.0     # 容許的色相範圍（±）
    sat_min: float = 0.18       # 粗略遮罩的飽和度下限
    val_min: float = 0.12       # 粗略遮罩的亮度下限

    # --- 演算法 ---
    unit_adj: float = 1.00      # 直接乘在自動量到的單顆面積上，最直接影響總數
    min_comp: float = 0.35      # 區塊面積下限（相對單顆面積）
    tight: float = 1.00         # 顏色門檻鬆緊，調大只認最飽和的面
    focus: float = 1.00         # 背景剔除強度（對焦清楚程度的門檻倍率）

    # --- 尺寸 ---
    locate_px: int = 600        # 第一階段（找整捆）的工作邊長
    work_px: int = 1000         # 第二階段（切塞頭）的工作邊長
    max_src_px: int = 2000      # 超過這個邊長的照片先縮小；實測不影響結果

    # --- 外形檢查 ---
    shape_guard: bool = True    # 外形只容得下一顆時，即使面積偏大也判一顆


@dataclass
class Blob:
    """一塊連通的彩色區域，可能是一顆塞頭，也可能是黏在一起的好幾顆。"""

    label: int
    area: int
    ratio: float          # 面積 / 單顆面積
    k_area: int           # 純面積法的判定顆數
    k: int                # 最終判定顆數（可能被外形檢查修正）
    cx: float
    cy: float
    w: float              # 最小外接矩形的短邊
    length: float         # 最小外接矩形的長邊
    fill: float           # 面積 / 矩形面積
    border: bool          # 有沒有碰到畫面邊緣（可能被裁掉）
    note: str = ""        # 需要留意的地方

    @property
    def edgy(self) -> bool:
        """比值接近 .5 —— 這是唯一可能算錯的地方。"""
        return self.ratio > 0.65 and abs(self.ratio - math.floor(self.ratio) - 0.5) < 0.15


@dataclass
class Result:
    count: int                       # 最終顆數
    blobs: List[Blob]
    markers: List[Tuple[float, float]]   # 每顆塞頭一個標記（工作影像座標）
    crop: np.ndarray                 # 放大後的工作影像（BGR）
    mask: np.ndarray                 # 同尺寸的彩色遮罩（uint8 0/1）
    unit: float                      # 單顆面積（工作影像的 px²）
    w_typ: float                     # 單顆的典型短邊
    l_typ: float                     # 單顆的典型長邊
    ratio_sum: float                 # 所有比值的總和
    by_total: int                    # 交叉核對一：總面積 ÷ 單顆面積
    by_cores: int                    # 交叉核對二：距離轉換核心數
    roi: Tuple[int, int, int, int]   # 整捆在原始照片中的範圍
    src_size: Tuple[int, int]        # 原始照片尺寸 (w, h)
    plug_px: int                     # 單顆在原始照片中約幾 px 見方
    w_typ_px: int                    # 單顆短邊，換算回原始照片的 px
    l_typ_px: int                    # 單顆長邊，換算回原始照片的 px
    work_px: int                     # 單顆在工作影像中約幾 px 見方
    frame_share: float               # 整捆佔畫面寬度的比例
    ms: float                        # 運算耗時

    @property
    def agrees(self) -> bool:
        """三個算法是否一致。一致時最可靠。"""
        return self.by_total == self.count == self.by_cores

    @property
    def thin(self) -> bool:
        """解析度是否偏低。實測單顆小於約 25 px 就開始不可靠。"""
        return self.work_px < 25

    def annotate(self, markers: Optional[List[Tuple[float, float]]] = None,
                 manual: Optional[List[bool]] = None) -> np.ndarray:
        """在放大後的工作影像上畫編號。markers 可以換成人工修正後的版本。"""
        pts = self.markers if markers is None else markers
        out = self.crop.copy()
        tint = out.copy()
        tint[self.mask.astype(bool)] = (134, 156, 14)     # BGR
        cv2.addWeighted(tint, 0.26, out, 0.74, 0, out)
        draw_markers(out, pts, manual, math.sqrt(max(self.unit, 1)) * 0.20)
        return out

    def to_source(self, pts: Optional[List[Tuple[float, float]]] = None
                  ) -> List[Tuple[float, float]]:
        """把工作影像的座標換算回原始輸入影像的座標。"""
        pts = self.markers if pts is None else pts
        x0, y0, x1, _ = self.roi
        k = (x1 - x0) / max(self.crop.shape[1], 1)
        return [(x0 + x * k, y0 + y * k) for x, y in pts]

    def annotate_on(self, frame: np.ndarray,
                    markers: Optional[List[Tuple[float, float]]] = None,
                    manual: Optional[List[bool]] = None,
                    count_badge: bool = True) -> np.ndarray:
        """直接在原始畫面上標號 —— 即時預覽用，看得到整個鏡頭視野。"""
        out = frame.copy()
        x0, y0, x1, y1 = self.roi
        k = (x1 - x0) / max(self.crop.shape[1], 1)
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (134, 156, 14), 2)
        pts = self.to_source(markers)
        draw_markers(out, pts, manual, math.sqrt(max(self.unit, 1)) * 0.20 * k)
        if count_badge:
            _count_badge(out, len(pts))
        return out


def draw_markers(img: np.ndarray, pts, manual=None, radius: float = 14.0) -> None:
    """畫上編號。所有圈圈同一尺寸，並且一定容得下最大的那個編號 ——
    圈圈只夠放一位數時，兩位數會滿出來被背景吃掉。"""
    if not len(pts):
        return
    flags = [False] * len(pts) if manual is None else manual
    font = cv2.FONT_HERSHEY_DUPLEX
    r = max(9, int(round(radius)))
    scale = max(0.34, r / 15.0)
    thick = 2 if r >= 13 else 1
    (tw, th), _ = cv2.getTextSize(str(len(pts)), font, scale, thick)
    r = max(r, int(tw * 0.62) + 3)          # 依最大編號的寬度把圈圈撐開
    ring_w = max(1, r // 7)
    for i, (x, y) in enumerate(pts, 1):
        p = (int(round(x)), int(round(y)))
        ring = (31, 86, 176) if flags[i - 1] else (74, 85, 6)   # BGR
        cv2.circle(img, p, r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, r, ring, ring_w, cv2.LINE_AA)
        txt = str(i)
        (tw_i, th_i), _ = cv2.getTextSize(txt, font, scale, thick)
        cv2.putText(img, txt, (p[0] - tw_i // 2, p[1] + th_i // 2),
                    font, scale, ring, thick, cv2.LINE_AA)


def _count_badge(img: np.ndarray, n: int) -> None:
    """左上角的大數字，即時預覽時一眼就看得到。"""
    h, w = img.shape[:2]
    scale = max(1.0, w / 640.0)
    txt = str(n)
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(txt, font, scale * 1.9, int(3 * scale))
    pad = int(10 * scale)
    cv2.rectangle(img, (0, 0), (tw + pad * 2, th + pad * 2), (255, 255, 255), -1)
    cv2.rectangle(img, (0, 0), (tw + pad * 2, th + pad * 2), (74, 85, 6), max(2, int(2 * scale)))
    cv2.putText(img, txt, (pad, th + pad), font, scale * 1.9, (74, 85, 6),
                int(3 * scale), cv2.LINE_AA)


class NoPlugsFound(Exception):
    """照片裡找不到可以數的東西。訊息會說明是哪一步失敗。"""


# ---------------------------------------------------------------- 小工具


def _resize_max(img: np.ndarray, max_side: int, allow_up: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    z = min(max_side / max(w, h), allow_up)
    if abs(z - 1.0) < 0.02:
        return img
    interp = cv2.INTER_AREA if z < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, round(w * z)), max(1, round(h * z))), interpolation=interp)


def _box(img: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), normalize=True,
                         borderType=cv2.BORDER_REPLICATE)


def _otsu(values8: np.ndarray) -> float:
    """對一維 uint8 陣列求 Otsu 門檻。"""
    if values8.size == 0:
        return 128.0
    t, _ = cv2.threshold(values8.reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def _hue_mask(hue: np.ndarray, center_deg: float, width_deg: float) -> np.ndarray:
    """OpenCV 的色相是 0-179（實際角度的一半），所以要先換算。"""
    c = (center_deg / 2.0) % 180.0
    d = np.abs(((hue.astype(np.float32) - c + 90.0) % 180.0) - 90.0)
    return d <= (width_deg / 2.0)


def _focus_field(bgr: np.ndarray, strength: float) -> np.ndarray:
    """整張照片裡對焦清楚的區域。

    先算局部邊緣能量並以 Otsu 取門檻，再做一次覆蓋率測試 —— 這樣塞頭
    平坦的中央（本身沒有邊緣）會留下，而整片模糊的背景不會。
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)) + \
           np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    en = _box(grad, max(3, round(min(w, h) / 80)))
    peak = float(en.max())
    en8 = np.clip(en * (255.0 / (peak if peak > 1e-6 else 1.0)), 0, 255).astype(np.uint8)
    thr = _otsu(en8) * strength
    sharp = (en8 >= thr).astype(np.float32)
    cov = _box(sharp, max(6, round(min(w, h) / 35)))
    return cov >= 0.12


def _color_masks(bgr: np.ndarray, focus_ok: np.ndarray, p: Params):
    """回傳 (粗略遮罩, 精細遮罩)。

    精細遮罩對飽和度再做一次 Otsu，把塞頭之間那圈淡色的透明套環排除掉
    —— 相鄰的塞頭能分開，靠的就是這一步。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    coarse = (_hue_mask(hue, p.hue_center, p.hue_width)
              & (sat >= p.sat_min * 255) & (val >= p.val_min * 255) & focus_ok)
    if int(coarse.sum()) < 100:
        return coarse, coarse
    t = _otsu(sat[coarse])
    return coarse, coarse & (sat >= t * p.tight)


def _unit_of(areas: np.ndarray) -> float:
    """佔多數的那個物件大小。

    中位數反覆往自己的鄰域重新擬合，讓黏在一起的大塊與零星雜訊都拉不動它。
    不能改用最小值：最小的那塊往往是只露出一部分的塞頭，拿它當單位會把
    總數算成將近兩倍（實測 36 vs 正解 20）。
    """
    if areas.size == 0:
        return 1.0
    u = float(np.median(areas))
    for _ in range(8):
        sel = areas[(areas >= 0.45 * u) & (areas <= 2.2 * u)]
        if sel.size == 0:
            break
        nu = float(np.median(sel))
        if abs(nu - u) < 1:
            break
        u = nu
    return max(u, 1.0)


def _biggest_cluster(stats: np.ndarray, labels: List[int], unit: float) -> List[int]:
    """單鏈接分群後取最大的一群，藉此甩掉背景裡零星的同色雜物。"""
    if len(labels) < 2:
        return labels
    cent = np.array([[stats[i][0], stats[i][1]] for i in labels], dtype=np.float32)
    link2 = (2.6 * math.sqrt(unit)) ** 2
    parent = list(range(len(labels)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(labels)):
        d = ((cent - cent[i]) ** 2).sum(axis=1)
        for j in np.nonzero(d <= link2)[0]:
            a, b = find(i), find(int(j))
            if a != b:
                parent[b] = a

    groups: dict = {}
    for i, lab in enumerate(labels):
        groups.setdefault(find(i), []).append(lab)
    return max(groups.values(), key=lambda g: sum(stats[l][2] for l in g))


def _place(dist: np.ndarray, lab_img: np.ndarray, label: int, k: int,
           unit: float) -> List[Tuple[float, float]]:
    """在一塊區域裡挑 k 個標記位置：距離圖最深的點，彼此互相排開。"""
    ys, xs = np.nonzero(lab_img == label)
    if ys.size == 0:
        return []
    d = dist[ys, xs]
    if k <= 1:
        i = int(np.argmax(d))
        return [(float(xs[i]), float(ys[i]))]

    keep = d >= 0.3 * d.max()          # 只有夠深的點才可能是塞頭中心
    ys, xs, d = ys[keep], xs[keep], d[keep]
    order = np.argsort(-d)
    supr = 0.72 * math.sqrt(unit)
    while True:
        pts: List[Tuple[float, float]] = []
        s2 = supr * supr
        for oi in order:
            x, y = float(xs[oi]), float(ys[oi])
            if all((x - px) ** 2 + (y - py) ** 2 > s2 for px, py in pts):
                pts.append((x, y))
                if len(pts) >= k:
                    return pts
        if len(pts) >= k or supr < 3:
            return pts
        supr *= 0.7


# ---------------------------------------------------------------- 主流程


def count_plugs(bgr: np.ndarray, p: Params = Params()) -> Result:
    """數出照片裡的塞頭數量。輸入 BGR 影像（cv2.imread / VideoCapture 的格式）。"""
    t0 = cv2.getTickCount()
    if bgr is None or bgr.size == 0:
        raise NoPlugsFound("沒有影像")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    src_h, src_w = bgr.shape[:2]
    # 12MP 的照片多花好幾秒卻換不到準確度：實測縮到 2000 px 結果完全相同
    src = _resize_max(bgr, p.max_src_px)
    k0 = src_w / src.shape[1]

    # --- 第一階段：找出整捆在哪裡 ---
    small = _resize_max(src, p.locate_px)
    sh, sw = small.shape[:2]
    focus_small = _focus_field(small, p.focus)
    coarse_small, _ = _color_masks(small, focus_small, p)
    if int(coarse_small.sum()) < 100:
        raise NoPlugsFound("照片裡找不到這個顏色的塞頭（試著調整色相或降低顏色門檻）")

    # 每顆塞頭都被自己的透明套環圍住，整捆從來不會連成一塊（實測會碎成
    # 58 塊），所以先把遮罩模糊化讓各面融成一個區域，再取最大的區域。
    # 直接取「最大的連通區塊」只會框到整捆的一角，把邊緣的塞頭切掉。
    dens = _box(coarse_small.astype(np.float32), max(4, round(min(sw, sh) / 28)))
    n_reg, reg, reg_stats, _ = cv2.connectedComponentsWithStats(
        (dens >= 0.22).astype(np.uint8), 8)
    if n_reg < 2:
        raise NoPlugsFound("找不到成群的塞頭")
    big = 1 + int(np.argmax(reg_stats[1:, cv2.CC_STAT_AREA]))

    inside = (reg == big) & coarse_small
    if int(inside.sum()) < 50:
        inside = reg == big
    ys, xs = np.nonzero(inside)
    pad = 0.12 * max(xs.max() - xs.min(), ys.max() - ys.min())
    f = src.shape[1] / sw
    x0 = max(0, int((xs.min() - pad) * f))
    x1 = min(src.shape[1], int(math.ceil((xs.max() + pad) * f)))
    y0 = max(0, int((ys.min() - pad) * f))
    y1 = min(src.shape[0], int(math.ceil((ys.max() + pad) * f)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise NoPlugsFound("找到的區域太小")

    # --- 第二階段：把整捆裁出來放大再處理 ---
    # 手機照片整張直接算，塞頭只有 20 幾 px，會全部黏成一團
    crop_src = src[y0:y1, x0:x1]
    crop = _resize_max(crop_src, p.work_px, allow_up=1.5)
    ch, cw = crop.shape[:2]
    back = (x1 - x0) / cw          # 工作影像 px -> 縮小後照片 px

    # 對焦判斷沿用整張照片的結果：在裁切區裡重算 Otsu 沒有意義，
    # 因為那一塊幾乎全部都是清楚的，門檻會落在塞頭自己身上。
    gx = np.clip(((x0 + (np.arange(cw) + 0.5) * (x1 - x0) / cw) / f).astype(int), 0, sw - 1)
    gy = np.clip(((y0 + (np.arange(ch) + 0.5) * (y1 - y0) / ch) / f).astype(int), 0, sh - 1)
    focus_crop = focus_small[np.ix_(gy, gx)]

    _, fine = _color_masks(crop, focus_crop, p)
    er = max(1, round(min(cw, ch) / 330))
    mask = cv2.erode(fine.astype(np.uint8), np.ones((2 * er + 1, 2 * er + 1), np.uint8))

    n, lab_img, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    cand = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 12]
    if not cand:
        raise NoPlugsFound("彩色區塊太小，試著調低「顏色門檻」")

    unit = _unit_of(np.array([stats[i, cv2.CC_STAT_AREA] for i in cand], float)) * p.unit_adj
    kept = [i for i in cand if stats[i, cv2.CC_STAT_AREA] >= p.min_comp * unit]
    if not kept:
        raise NoPlugsFound("沒有符合大小的區塊，試著調低「最小面積」")
    info = {i: (cents[i][0], cents[i][1], stats[i, cv2.CC_STAT_AREA]) for i in kept}
    kept = _biggest_cluster(info, kept, unit)

    keep_mask = np.isin(lab_img, kept).astype(np.uint8)
    dist = cv2.distanceTransform(keep_mask, cv2.DIST_L2, 5)

    # --- 每一塊各自算幾顆 ---
    blobs: List[Blob] = []
    for i in sorted(kept, key=lambda i: -stats[i, cv2.CC_STAT_AREA]):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ratio = area / unit
        k_area = max(1, int(round(ratio)))
        cs, _ = cv2.findContours((lab_img == i).astype(np.uint8),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = np.vstack(cs) if cs else np.zeros((1, 1, 2), np.int32)
        (_, _), (rw, rl), _ = cv2.minAreaRect(pts)
        rw, rl = min(rw, rl), max(rw, rl)
        x, y, bw, bh = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        blobs.append(Blob(
            label=i, area=area, ratio=ratio, k_area=k_area, k=k_area,
            cx=float(cents[i][0]), cy=float(cents[i][1]),
            w=rw, length=rl, fill=area / max(rw * rl, 1e-6),
            border=bool(x <= 2 or y <= 2 or x + bw >= cw - 2 or y + bh >= ch - 2),
        ))

    # --- 外形檢查 ---
    # 塞頭從側面拍時長邊幾乎不變（實測 ±1%），只有短邊會被拉寬（±23%），
    # 所以長邊是比面積（±5%）更穩的尺。
    singles = [b for b in blobs if b.k_area == 1 and b.length > 0]
    w_typ = float(np.median([b.w for b in singles])) if singles else 0.0
    l_typ = float(np.median([b.length for b in singles])) if singles else 0.0
    if p.shape_guard and len(singles) >= 3 and l_typ > 0:
        for b in blobs:
            if b.length <= 0:
                continue
            if b.w <= 1.35 * w_typ and b.length <= 1.12 * l_typ:
                if b.k_area > 1:
                    b.k = 1
                    b.note = "外形只容得下一顆，面積偏大（側拍）"
            elif b.k_area == 1 and b.length > 1.2 * l_typ:
                b.note = "外形偏長，可能其實是兩顆"

    for b in blobs:
        if b.note:
            continue
        if b.border:
            b.note = "碰到畫面邊緣，可能被切到"
        elif b.edgy:
            b.note = "比值接近 .5，可能差一顆"
        elif b.k > 1:
            b.note = f"{b.k} 顆黏在一起"
        elif b.ratio < 0.75:
            b.note = "只露出一部分，仍算一顆"

    total = sum(b.k for b in blobs)

    # --- 標記位置 ---
    markers: List[Tuple[float, float]] = []
    for b in blobs:
        markers += _place(dist, lab_img, b.label, b.k, unit)
    row = 2.0 * max(math.sqrt(unit) * 0.5, 1.0)
    markers.sort(key=lambda pt: (round(pt[1] / row), pt[0]))

    # --- 兩個交叉核對 ---
    ratio_sum = float(sum(b.ratio for b in blobs))
    by_total = max(1, int(round(ratio_sum)))
    by_cores = _cross_cores(dist, keep_mask, unit)

    plug_side = math.sqrt(unit)
    ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
    return Result(
        count=total, blobs=blobs, markers=markers, crop=crop, mask=keep_mask,
        unit=unit, w_typ=w_typ, l_typ=l_typ,
        ratio_sum=ratio_sum, by_total=by_total, by_cores=by_cores,
        roi=(int(x0 * k0), int(y0 * k0), int(x1 * k0), int(y1 * k0)),
        src_size=(src_w, src_h),
        plug_px=int(round(plug_side * back * k0)),
        w_typ_px=int(round(w_typ * back * k0)),
        l_typ_px=int(round(l_typ * back * k0)),
        work_px=int(round(plug_side)),
        frame_share=(x1 - x0) / src.shape[1],
        ms=ms,
    )


def _cross_cores(dist: np.ndarray, keep_mask: np.ndarray, unit: float) -> int:
    """交叉核對：距離圖上彼此分離的「核心」有幾個。

    瀏覽器版這裡跑的是 h-maxima 抑制的分水嶺氾濫。那個作法要逐像素
    迴圈，在 Python 裡對 30 萬個像素要好幾秒，所以改用向量化的核心
    計數：塞頭中央的距離值高，兩顆之間的縫隙會把它們切開。兩者都是
    獨立於面積法的第二意見，但不是同一個演算法，數字可能略有出入。
    """
    d = dist[keep_mask.astype(bool)]
    if d.size == 0:
        return 0
    r_typ = float(np.median(d[d >= np.percentile(d, 90)]))
    if r_typ <= 0:
        return 0
    cores = (dist >= 0.55 * r_typ).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(cores, 8)
    floor = max(4.0, 0.03 * unit)
    return int(sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= floor))


# ---------------------------------------------------------------- 批次


def count_file(path: str, p: Params = Params()) -> Result:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise NoPlugsFound(f"讀不到影像：{path}")
    return count_plugs(img, p)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python counter.py <照片> [照片...]")
        raise SystemExit(1)
    for path in sys.argv[1:]:
        try:
            r = count_file(path)
            flag = "一致" if r.agrees else f"不一致(面積總和 {r.by_total}, 核心 {r.by_cores})"
            print(f"{path}: {r.count} 顆  [{flag}]  單顆 {int(r.unit)} px²  {r.ms:.0f} ms")
        except NoPlugsFound as e:
            print(f"{path}: 失敗 — {e}")
