"""塞頭計數器 — Streamlit + OpenCV，用筆電外接的 webcam 計數。

啟動：
    streamlit run app.py

演算法在 counter.py，這裡只負責介面、相機與存檔。
"""

from __future__ import annotations

import csv
import math
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from counter import NoPlugsFound, Params, count_plugs

# 點圖修正需要這個小套件；沒裝就自動退回按鈕模式
try:
    from streamlit_image_coordinates import streamlit_image_coordinates as click_image
    HAS_CLICK = True
except Exception:
    HAS_CLICK = False

APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="塞頭計數器", page_icon="🔢", layout="wide")

CSV_COLUMNS = [
    "時間", "標註圖", "原始圖", "自動顆數", "最終顆數", "人工新增", "人工刪除",
    "單顆面積", "區塊數", "比值總和", "面積總和核對", "核心核對", "三法一致",
    "單顆px", "佔畫面寬", "色相中心", "色相寬度", "單顆面積微調", "最小面積",
    "顏色門檻", "背景剔除", "耗時ms",
]


# ---------------------------------------------------------------- 狀態


def init_state() -> None:
    d = {
        "cap": None, "cap_key": None, "live": True,
        "shot": None, "result": None, "markers": [], "manual": [],
        "last_click": None, "record": None, "saved_row": None,
        "error": "", "click_mode": "修正計數",
    }
    for k, v in d.items():
        st.session_state.setdefault(k, v)


def _badge_zero(img) -> None:
    """畫面裡找不到塞頭時也要顯示 0，不能只是沒有反應。"""
    h, w = img.shape[:2]
    sc = max(1.0, w / 640.0)
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize("0", font, sc * 1.9, int(3 * sc))
    pad = int(10 * sc)
    cv2.rectangle(img, (0, 0), (tw + pad * 2, th + pad * 2), (255, 255, 255), -1)
    cv2.rectangle(img, (0, 0), (tw + pad * 2, th + pad * 2), (110, 110, 110), max(2, int(2 * sc)))
    cv2.putText(img, "0", (pad, th + pad), font, sc * 1.9, (110, 110, 110),
                int(3 * sc), cv2.LINE_AA)


def release_cap() -> None:
    cap = st.session_state.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.cap = None
    st.session_state.cap_key = None


def get_cap(index: int, width: int, height: int):
    """開啟相機並記在 session 裡，避免每次重繪都重開（重開很慢）。"""
    key = (index, width, height)
    if st.session_state.cap_key != key or st.session_state.cap is None:
        release_cap()
        # Windows 預設的 MSMF 後端開相機常要好幾秒，DSHOW 快很多
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        st.session_state.cap = cap
        st.session_state.cap_key = key
    return st.session_state.cap


def grab(cap, flush: int = 4):
    """取一張新的畫面。先丟掉幾張，否則拿到的是驅動緩衝裡的舊畫面。"""
    frame = None
    for _ in range(max(1, flush)):
        ok, f = cap.read()
        if ok:
            frame = f
    return frame


# ---------------------------------------------------------------- 計數


def analyse(frame: np.ndarray, p: Params) -> None:
    st.session_state.shot = frame
    st.session_state.error = ""
    try:
        res = count_plugs(frame, p)
    except NoPlugsFound as e:
        st.session_state.result = None
        st.session_state.markers = []
        st.session_state.manual = []
        st.session_state.error = str(e)
        return
    st.session_state.result = res
    st.session_state.markers = list(res.markers)
    st.session_state.manual = [False] * len(res.markers)
    st.session_state.last_click = None
    st.session_state.saved_row = None


def toggle_marker(x: float, y: float) -> None:
    """點空白處新增一顆，點已有的標記則刪除它。"""
    res = st.session_state.result
    if res is None:
        return
    hit = max(16.0, math.sqrt(max(res.unit, 1)) * 0.55)
    best, best_d = -1, hit * hit
    for i, (mx, my) in enumerate(st.session_state.markers):
        d = (mx - x) ** 2 + (my - y) ** 2
        if d < best_d:
            best, best_d = i, d
    if best >= 0:
        st.session_state.markers.pop(best)
        st.session_state.manual.pop(best)
    else:
        st.session_state.markers.append((x, y))
        st.session_state.manual.append(True)


def counts() -> tuple[int, int, int]:
    """(最終顆數, 人工新增, 人工刪除)"""
    res = st.session_state.result
    total = len(st.session_state.markers)
    added = sum(1 for m in st.session_state.manual if m)
    auto_left = total - added
    removed = (res.count - auto_left) if res else 0
    return total, added, max(0, removed)


# ---------------------------------------------------------------- 存檔


def save_record(out_dir: Path, p: Params, save_raw: bool) -> dict:
    """存下標註圖（與原始圖）並回寫一列 CSV。回傳那一列。"""
    res = st.session_state.result
    total, added, removed = counts()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotated").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ann_path = out_dir / "annotated" / f"{stamp}.jpg"
    ann = res.annotate(st.session_state.markers, st.session_state.manual)
    cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(ann_path))

    raw_path = ""
    if save_raw and st.session_state.shot is not None:
        (out_dir / "raw").mkdir(exist_ok=True)
        rp = out_dir / "raw" / f"{stamp}.jpg"
        cv2.imencode(".jpg", st.session_state.shot,
                     [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(rp))
        raw_path = rp.name

    row = {
        "時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "標註圖": ann_path.name, "原始圖": raw_path,
        "自動顆數": res.count, "最終顆數": total,
        "人工新增": added, "人工刪除": removed,
        "單顆面積": int(res.unit), "區塊數": len(res.blobs),
        "比值總和": round(res.ratio_sum, 2),
        "面積總和核對": res.by_total, "核心核對": res.by_cores,
        "三法一致": "是" if res.agrees else "否",
        "單顆px": res.plug_px, "佔畫面寬": f"{res.frame_share:.0%}",
        "色相中心": p.hue_center, "色相寬度": p.hue_width,
        "單顆面積微調": p.unit_adj, "最小面積": p.min_comp,
        "顏色門檻": p.tight, "背景剔除": p.focus,
        "耗時ms": int(res.ms),
    }
    csv_path = out_dir / "log.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            wr.writeheader()
        wr.writerow(row)
    return row


# ---------------------------------------------------------------- 介面

init_state()

with st.sidebar:
    st.header("設定")

    st.subheader("相機")
    cam_index = st.number_input("相機編號", 0, 10, 1, 1,
                                help="外接 webcam 通常是 1，筆電內建鏡頭多半是 0。開不起來就把這個數字換掉試試。")
    res_label = st.selectbox("解析度", ["1920×1080", "1280×720", "3840×2160", "640×480"], index=0)
    cam_w, cam_h = (int(v) for v in res_label.replace("×", "x").split("x"))
    st.caption("相機不一定支援所選解析度，會自動退到最接近的。")

    st.subheader("顏色")
    hue_center = st.slider("色相中心（度）", 0, 359, 168, 1,
                           help="綠色塞頭是 168。要數別的顏色就改這裡，或在圖上用「取色」點一下。")
    hue_width = st.slider("色相寬度（±度）", 5, 90, 50, 1)
    st.caption("白色塞頭無法用顏色法：白色與手指亮部同為低飽和高亮度，實測會把手指算進去。")

    st.subheader("演算法")
    unit_adj = st.slider("單顆面積微調", 0.75, 1.30, 1.00, 0.01,
                         help="最直接影響總數。總數偏多就調大，偏少就調小。")
    min_comp = st.slider("最小面積", 0.08, 0.70, 0.35, 0.01,
                         help="調小會納入只露一角的塞頭，但也容易把雜訊算進來。")
    tight = st.slider("顏色門檻", 0.80, 1.25, 1.00, 0.01)
    focus = st.slider("背景剔除", 0.50, 1.80, 1.00, 0.05)
    shape_guard = st.checkbox("外形檢查", True,
                              help="外形只容得下一顆時，即使面積偏大也判一顆。防的是側拍把單顆拉大。")

    st.subheader("存檔")
    out_dir = Path(st.text_input("輸出資料夾", str(APP_DIR / "captures")))
    save_raw = st.checkbox("同時保留原始照片", True,
                           help="原始照片加上你的人工修正，就是之後訓練模型的資料來源。")
    st.caption("按下「拍照並存檔」才會寫檔；即時預覽不會產生檔案。")

params = Params(hue_center=float(hue_center), hue_width=float(hue_width),
                unit_adj=float(unit_adj), min_comp=float(min_comp),
                tight=float(tight), focus=float(focus), shape_guard=shape_guard)

left, right = st.columns([3, 2], gap="large")
# 即時預覽的迴圈永遠不會結束，所以必須放在整頁都畫完之後，
# 否則迴圈後面的右半欄（大數字、判定明細）永遠不會出現。
run_live = False
img_ph = cap_ph = num_ph = live_cap = None

# ---- 左：畫面 ----
with left:
    if st.session_state.result is None and st.session_state.shot is None:
        st.subheader("相機")
        c1, c2, c3 = st.columns([1.2, 1, 1])
        # 不要給 toggle 綁 key：綁了之後程式就不能再改 st.session_state.live，
        # 而拍照與上傳都需要把預覽關掉
        live = c1.toggle("即時預覽（持續計數）", value=st.session_state.live)
        st.session_state.live = live
        shoot = c2.button("📷 拍照並存檔", type="primary", use_container_width=True)
        if c3.button("重新連線", use_container_width=True):
            release_cap()
            st.rerun()

        cap = get_cap(int(cam_index), cam_w, cam_h)
        if cap is None or not cap.isOpened():
            st.error(f"開不了相機 {int(cam_index)}。換一個相機編號，或確認沒有其他程式正在使用它。")
        else:
            if shoot:
                # 用預覽中最後那張畫面：使用者看到什麼就存什麼。
                # 沒有預覽（暫停中）才重新抓一張。
                frame = st.session_state.get("live_frame")
                if frame is None:
                    frame = grab(cap)
                if frame is None:
                    st.error("拿不到畫面，按「重新連線」再試一次。")
                else:
                    st.session_state.live = False
                    analyse(frame, params)
                    if st.session_state.result is not None:
                        try:
                            st.session_state.saved_row = save_record(out_dir, params, save_raw)
                        except Exception as e:
                            st.session_state.error = f"存檔失敗：{e}"
                    st.rerun()

            img_ph = st.empty()
            cap_ph = st.empty()
            live_cap = cap
            run_live = bool(st.session_state.live)

            def show(frame, res, fps=None, num_ph=None):
                """把畫面（含即時計數的標號）送到瀏覽器。"""
                view = res.annotate_on(frame) if res is not None else frame.copy()
                if res is None:
                    _badge_zero(view)
                # 傳輸用的縮圖：計數仍然用完整畫面，只有顯示縮小
                if view.shape[1] > 960:
                    k = 960 / view.shape[1]
                    view = cv2.resize(view, (960, int(view.shape[0] * k)),
                                      interpolation=cv2.INTER_AREA)
                img_ph.image(cv2.cvtColor(view, cv2.COLOR_BGR2RGB), use_container_width=True)
                n = res.count if res is not None else 0
                extra = f"　·　{fps:.0f} fps" if fps else ""
                cap_ph.caption(f"即時計數 **{n}** 顆{extra}　·　按「拍照並存檔」定格並記錄")
                if num_ph is not None:
                    num_ph.metric("塞頭數量", n, help="一顆塞頭 = 一根管子")

            if not run_live:
                frame = st.session_state.get("live_frame")
                if frame is None:
                    frame = grab(cap)
                if frame is not None:
                    try:
                        res = count_plugs(frame, params)
                    except NoPlugsFound:
                        res = None
                    st.session_state.live_frame = frame
                    show(frame, res)
                    cap_ph.caption("預覽已暫停　·　打開「即時預覽」繼續")

        up = st.file_uploader("或直接上傳一張照片", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up is not None:
            data = np.frombuffer(up.getvalue(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                st.error("這個檔案讀不進來。")
            else:
                st.session_state.live = False
                analyse(img, params)
                st.rerun()

    else:
        res = st.session_state.result
        a1, a2, a3 = st.columns(3)
        if a1.button("重新拍攝", use_container_width=True):
            st.session_state.shot = None
            st.session_state.result = None
            st.session_state.live = True
            st.rerun()
        if a2.button("用目前參數重算", use_container_width=True,
                     help="改了側邊欄的參數之後按這個，不用重拍"):
            analyse(st.session_state.shot, params)
            st.rerun()
        if a3.button("回到自動結果", use_container_width=True, disabled=res is None):
            st.session_state.markers = list(res.markers)
            st.session_state.manual = [False] * len(res.markers)
            st.rerun()

        if st.session_state.error:
            st.error(st.session_state.error)
            if st.session_state.shot is not None:
                st.image(cv2.cvtColor(st.session_state.shot, cv2.COLOR_BGR2RGB),
                         use_container_width=True, caption="拍到的畫面")
        elif res is not None:
            if HAS_CLICK:
                st.session_state.click_mode = st.radio(
                    "點圖做什麼", ["修正計數", "取色"], horizontal=True,
                    help="修正計數：點空白處補一顆、點圈圈刪一顆。取色：點一顆塞頭，自動把色相中心設成它的顏色。")

            ann = res.annotate(st.session_state.markers, st.session_state.manual)
            rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)

            if HAS_CLICK:
                from PIL import Image
                # 固定寬度會超出欄寬被裁掉，改成填滿欄寬
                pt = click_image(Image.fromarray(rgb), key="canvas",
                                 use_column_width="always")
                if pt and pt != st.session_state.last_click:
                    st.session_state.last_click = pt
                    # 元件回傳的是顯示尺寸的座標，換算回工作影像座標
                    shown_w = pt.get("width") or rgb.shape[1]
                    scale = rgb.shape[1] / float(shown_w)
                    x, y = pt["x"] * scale, pt["y"] * scale
                    if st.session_state.click_mode == "取色":
                        px = res.crop[int(np.clip(y, 0, res.crop.shape[0] - 1)),
                                      int(np.clip(x, 0, res.crop.shape[1] - 1))]
                        hsv = cv2.cvtColor(np.uint8([[px]]), cv2.COLOR_BGR2HSV)[0][0]
                        st.info(f"抓到的色相是 {int(hsv[0]) * 2}°（飽和度 {hsv[1]}／亮度 {hsv[2]}）。"
                                "把側邊欄的「色相中心」改成這個數字，再按「用目前參數重算」。")
                    else:
                        toggle_marker(x, y)
                        st.rerun()
                st.caption("點空白處補上漏掉的塞頭　·　點圈圈刪掉算錯的")
            else:
                st.image(rgb, use_container_width=True)
                st.warning("沒有安裝 streamlit-image-coordinates，只能用按鈕調整總數。"
                           "安裝後可以直接點圖修正：pip install streamlit-image-coordinates")
                b1, b2 = st.columns(2)
                if b1.button("＋ 一顆", use_container_width=True):
                    st.session_state.markers.append((10.0, 10.0))
                    st.session_state.manual.append(True)
                    st.rerun()
                if b2.button("− 一顆", use_container_width=True,
                             disabled=not st.session_state.markers):
                    st.session_state.markers.pop()
                    st.session_state.manual.pop()
                    st.rerun()

# ---- 右：數字與明細 ----
with right:
    res = st.session_state.result
    total, added, removed = counts()

    # 同一個 metric：沒有結果時由即時迴圈更新，有結果時顯示定格後的數字
    num_ph = st.empty()
    num_ph.metric("塞頭數量", total if res else "—", help="一顆塞頭 = 一根管子")

    if res is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("自動偵測", res.count)
        m2.metric("人工修正", f"+{added} / −{removed}" if (added or removed) else "未修正")
        m3.metric("耗時", f"{res.ms:.0f} ms")

        ok_total = res.by_total == total
        ok_cores = res.by_cores == total
        c1, c2 = st.columns(2)
        c1.metric("面積總和核對", f"{res.ratio_sum:.1f} → {res.by_total}",
                  delta="一致" if ok_total else "不一致",
                  delta_color="normal" if ok_total else "inverse")
        c2.metric("核心核對", res.by_cores,
                  delta="一致" if ok_cores else "不一致",
                  delta_color="normal" if ok_cores else "inverse")

        st.caption(
            f"照片 {res.src_size[0]}×{res.src_size[1]} · "
            f"單顆矩形 {res.w_typ_px}×{res.l_typ_px} px · "
            f"單顆面積 {int(res.unit)} px² · 整捆佔畫面寬 {res.frame_share:.0%}"
            + ("　⚠ 解析度偏低，靠近一點重拍會更準" if res.thin else "")
        )

        st.markdown("**判定明細** — 每塊區域 ÷ 單顆面積")
        df = pd.DataFrame([{
            "面積": b.area, "÷單顆": round(b.ratio, 2),
            "短×長": f"{b.w:.0f}×{b.length:.0f}", "顆數": b.k, "備註": b.note,
        } for b in res.blobs])
        st.dataframe(df, use_container_width=True, hide_index=True, height=min(420, 40 + 35 * len(df)))

        if st.session_state.saved_row is not None:
            st.success(f"已存檔：{st.session_state.saved_row['標註圖']}")
        if st.button("存檔" if st.session_state.saved_row is None
                     else "另存一筆（含人工修正）", use_container_width=True):
            try:
                row = save_record(out_dir, params, save_raw)
                st.session_state.saved_row = row
                st.success(f"已存檔：{row['標註圖']}")
            except Exception as e:
                st.error(f"存檔失敗：{e}")

        log = out_dir / "log.csv"
        if log.exists():
            with st.expander("計數紀錄"):
                try:
                    hist = pd.read_csv(log, encoding="utf-8-sig")
                    st.dataframe(hist.tail(30), use_container_width=True, hide_index=True)
                    st.download_button("下載 log.csv", log.read_bytes(),
                                       file_name="log.csv", mime="text/csv")
                except Exception as e:
                    st.warning(f"讀不了紀錄檔：{e}")
    else:
        st.info("開啟相機、對準塞頭，數字會即時更新。按「📷 拍照並存檔」定格並記錄。")
        st.markdown(
            "**怎麼拍**\n"
            "- 讓整捆塞頭佔畫面大一點，背景單純\n"
            "- 對焦在塞頭上（程式靠對焦清楚度剔除背景）\n"
            "- 解析度不用擔心：實測每顆只剩 19 px 都還算得對\n"
        )


# ---------------------------------------------------------------- 即時迴圈
# 放在最後：使用者一按任何元件，Streamlit 會從頭重跑，迴圈就會被中斷。
if run_live and live_cap is not None:
    t_prev = time.time()
    while st.session_state.live:
        ok, frame = live_cap.read()
        if not ok:
            img_ph.error("畫面中斷，按「重新連線」。")
            break
        st.session_state.live_frame = frame
        try:
            res_live = count_plugs(frame, params)
        except NoPlugsFound:
            res_live = None
        st.session_state.live_res = res_live
        now = time.time()
        dt = now - t_prev
        t_prev = now
        show(frame, res_live, (1.0 / dt) if dt > 0 else None, num_ph)
