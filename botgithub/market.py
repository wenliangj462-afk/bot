"""
market_data.py — 数据爬虫与指标计算
"""
import requests, logging, time, json, math, re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, wait

from common import log, CFG
from core import UTC

# ── 新闻情绪关键词（中文 + 英文）─────────────────────────────────────────
_POSITIVE_KW = ["突破", "创新高", "流入", "买入", "批准", "降息", "牛市", "ETF通过",
                "增持", "看涨", "升级", "质押", "Layer2扩容", "机构买入", "上海升级",
                "复苏", "放量", "金叉", "主升浪", "停火", "和平", "缓和"]
_NEGATIVE_KW = ["暴跌", "崩盘", "流出", "卖出", "监管", "加息", "熊市", "黑客",
                "禁止", "看空", "漏洞", "51%攻击", "清算", "破产", "脱钩",
                "死叉", "缩量", "派发", "M顶", "制裁", "战争", "冲突", "关税"]
_POSITIVE_KW_EN = ["breakout", "all-time high", "inflow", "approved", "bullish",
                   "rally", "upgrade", "institutional", "recovery", "golden cross",
                   "rate cut", "dovish", "ceasefire", "peace", "stimulus",
                   "etf approv", "accumulation", "adoption"]
_NEGATIVE_KW_EN = ["crash", "plunge", "outflow", "hack", "banned", "bearish",
                   "liquidat", "bankrupt", "exploit", "death cross",
                   "rate hike", "hawkish", "sanction", "war", "conflict", "tariff",
                   "default", "recession", "contagion", "sec sue"]
_NEGATION_PREFIX = ["不", "未", "无", "非", "没有", "不会", "不曾"]

def _has_negation(text: str, kw: str) -> bool:
    """判断关键词前 5 字内是否有否定词，有则不计入情绪"""
    idx = text.find(kw)
    if idx < 0:
        return False
    prefix = text[max(0, idx-5):idx]
    return any(neg in prefix for neg in _NEGATION_PREFIX)

def _score_title(t: str, source_weight: float = 1.0) -> float:
    """计算标题得分（中英文关键词），乘以来源权重"""
    pos = sum(1 for k in _POSITIVE_KW if k in t and not _has_negation(t, k))
    neg = sum(1 for k in _NEGATIVE_KW if k in t and not _has_negation(t, k))
    # 英文关键词：大小写不敏感匹配
    t_lower = t.lower()
    pos += sum(1 for k in _POSITIVE_KW_EN if k in t_lower)
    neg += sum(1 for k in _NEGATIVE_KW_EN if k in t_lower)
    return (pos - neg) * source_weight


# ── 新闻爬虫 ─────────────────────────────────────────────────────────────
def fetch_fear_greed() -> Dict:
    log.debug("fetch_fear_greed 开始...")
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1",
                         timeout=(3, 5))
        json_data = r.json()
        data = json_data.get("data")
        if not data:
            log.debug("fetch_fear_greed 完成（无数据）")
            return {"value": 50, "label": "Neutral"}
        d = data[0]
        log.debug(f"fetch_fear_greed 完成: {d['value_classification']}({d['value']})")
        return {"value": int(d["value"]), "label": d["value_classification"]}
    except Exception as e:
        log.warning(f"恐贪指数获取失败: {e}")
        log.debug("fetch_fear_greed 完成（异常）")
        return {"value": 50, "label": "Neutral"}

def _fetch_wallstreetcn() -> list:
    """华尔街见闻快讯：地缘政治、美债、加息降息、宏观经济"""
    titles = []
    try:
        import xml.etree.ElementTree as ET
        rss = requests.get("https://rsshub.app/wallstreetcn/live/global",
                           headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        macro_kw = ["美联储", "Fed", "CPI", "通胀", "利率", "加息", "降息", "FOMC",
                    "非农", "GDP", "美债", "国债", "衰退", "就业", "PMI", "关税",
                    "制裁", "战争", "冲突", "地缘", "原油", "黄金", "美元",
                    "ETH", "BTC", "比特币", "以太坊", "加密", "ETF", "SEC",
                    "流动性", "缩表", "QE", "QT", "鲍威尔", "Powell"]
        if rss.status_code == 200:
            root = ET.fromstring(rss.content)
            for item in root.findall("./channel/item")[:15]:
                t = (item.find("title").text or "").strip()
                if any(kw in t for kw in macro_kw):
                    titles.append((f"[华尔街见闻] {t}", 1.3))
    except Exception as e:
        log.debug(f"华尔街见闻 失败: {e}")
    return titles

def _fetch_jinshi_rss() -> list:
    """金十数据：加密 + 宏观经济快讯"""
    titles = []
    try:
        import xml.etree.ElementTree as ET
        rss    = requests.get("https://rsshub.app/jinshi/information",
                              headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        eth_kw = ["ETH", "以太坊", "Ethereum", "加密", "美联储", "CPI", "通胀",
                  "FOMC", "ETF", "利率", "Layer2", "升级", "质押", "Dencun", "EIP", "DeFi",
                  "美债", "国债", "非农", "GDP", "PMI", "关税", "制裁", "衰退",
                  "降息", "加息", "鲍威尔", "就业", "SEC", "BTC", "比特币"]
        if rss.status_code == 200:
            root = ET.fromstring(rss.content)
            for item in root.findall("./channel/item")[:10]:
                t = (item.find("title").text or "").strip()
                if any(kw in t for kw in eth_kw):
                    titles.append((f"[金十] {t}", 1.0))
    except Exception as e:
        log.debug(f"金十 RSS 失败: {e}")
    return titles

def _fetch_cointelegraph() -> list:
    """CoinTelegraph RSS：加密 + 宏观英文新闻"""
    titles = []
    try:
        import xml.etree.ElementTree as ET
        rss = requests.get("https://cointelegraph.com/rss",
                           headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        ct_kw = ["Ethereum", "ETH", "crypto", "Bitcoin", "BTC", "DeFi",
                 "Fed", "SEC", "ETF", "rate", "CPI", "inflation", "Layer",
                 "tariff", "sanction", "war", "recession", "stablecoin",
                 "regulation", "institutional", "whale", "liquidat"]
        if rss.status_code == 200:
            root = ET.fromstring(rss.content)
            for item in root.findall("./channel/item")[:10]:
                t = (item.find("title").text or "").strip()
                if any(kw.lower() in t.lower() for kw in ct_kw):
                    titles.append((f"[CoinTelegraph] {t}", 1.2))
    except Exception as e:
        log.debug(f"CoinTelegraph 失败: {e}")
    return titles

def fetch_global_news(top_n: int = 5) -> Dict:
    log.debug("fetch_global_news 开始（15秒超时保护）...")
    all_titles = []
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="news-fetch") as executor:
        futures = [executor.submit(_fetch_wallstreetcn),
                   executor.submit(_fetch_jinshi_rss),
                   executor.submit(_fetch_cointelegraph)]
        done, _ = wait(futures, timeout=15)
        for f in done:
            try:
                all_titles.extend(f.result())
            except Exception:
                pass

    if not all_titles:
        log.debug("fetch_global_news 完成（无数据）")
        return {"text": "无重大 ETH 新闻，市场情绪中性", "keywords": {}, "sentiment": 0.0}

    scored = []
    for t, weight in all_titles:
        score = _score_title(t, weight)
        scored.append((t, score, weight))

    scored.sort(key=lambda x: abs(x[1]), reverse=True)
    top_scored = scored[:top_n]

    lines = []
    total_score = 0
    top_weight = 0
    keywords = {}
    for t, s, w in top_scored:
        arrow = "↑" if s > 0 else ("↓" if s < 0 else "→")
        lines.append(f"{arrow} {t}")
        total_score += s
        top_weight += w
        for kw in _POSITIVE_KW + _NEGATIVE_KW:
            if kw in t:
                keywords[kw] = keywords.get(kw, 0) + 1
        t_lower = t.lower()
        for kw in _POSITIVE_KW_EN + _NEGATIVE_KW_EN:
            if kw in t_lower:
                keywords[kw] = keywords.get(kw, 0) + 1

    sentiment = total_score / top_weight if top_weight > 0 else 0
    sentiment = max(-1.0, min(1.0, sentiment))

    lines.append(f"\n[新闻情绪汇总] 得分: {total_score:+.2f} (加权均值 {sentiment:.2f})")
    log.debug(f"fetch_global_news 完成（{len(all_titles)}条新闻）")
    return {
        "text": "\n".join(lines),
        "keywords": keywords,
        "sentiment": sentiment,
    }


# ── 市场情绪深度数据 ──────────────────────────────────────────────────────
def fetch_market_sentiment_data(symbol: str = "ETH-USDT-SWAP") -> Dict:
    """
    L4：市场情绪深度数据（OKX 公开接口，无需鉴权）
    包含：多空比（L/S Ratio）、持仓量（OI）及 OI 变化、
    主动买卖比（Taker Buy/Sell Ratio）
    """
    result = {
        "ls_ratio":       None,
        "oi":             None,
        "oi_change_pct":  None,
        "taker_buy_ratio": None,
        "_valid":         False,
    }
    base = "https://www.okx.com"
    try:
        ls_resp = requests.get(
            f"{base}/api/v5/rubik/stat/contracts/long-short-account-ratio",
            params={"instId": symbol, "period": "1H", "limit": "2"},
            timeout=(3, 5),
        )
        if ls_resp.status_code == 200:
            ls_data = ls_resp.json().get("data", [])
            if len(ls_data) >= 1:
                result["ls_ratio"] = round(float(ls_data[0].get("longShortAccRatio", 1.0)), 3)

        oi_resp = requests.get(
            f"{base}/api/v5/rubik/stat/contracts/open-interest-volume",
            params={"instId": symbol, "period": "1H", "limit": "3"},
            timeout=(3, 5),
        )
        if oi_resp.status_code == 200:
            oi_data = oi_resp.json().get("data", [])
            if len(oi_data) >= 2:
                oi_cur  = float(oi_data[0].get("oi",  0))
                oi_prev = float(oi_data[-1].get("oi", 0))
                result["oi"] = round(oi_cur, 0)
                if oi_prev > 0:
                    result["oi_change_pct"] = round((oi_cur - oi_prev) / oi_prev * 100, 2)

        taker_resp = requests.get(
            f"{base}/api/v5/rubik/stat/taker-volume",
            params={"instId": symbol, "instType": "CONTRACTS", "period": "1H", "limit": "1"},
            timeout=(3, 5),
        )
        if taker_resp.status_code == 200:
            t_data = taker_resp.json().get("data", [])
            if t_data:
                buy_vol  = float(t_data[0].get("buyVol",  0))
                sell_vol = float(t_data[0].get("sellVol", 0))
                total    = buy_vol + sell_vol
                if total > 0:
                    result["taker_buy_ratio"] = round(buy_vol / total, 3)

        result["_valid"] = any(v is not None for v in [
            result["ls_ratio"], result["oi"], result["taker_buy_ratio"]
        ])
    except Exception as e:
        log.debug(f"市场情绪数据获取失败: {e}")

    return result


# ============================================================
# 宏观记忆：自动计算 ATH/ATL 距离 + 波动率环境
# ============================================================
def build_macro_context(daily_data: Optional[list], current_price: float) -> str:
    """
    宏观记忆层：纯客观数据驱动，从日线K线自动计算。
    不注入任何人工主观判断，消除认知偏见。
    输出原则：语义描述而非绝对价格，让 AI 直接获得市场结构结论。
    """
    parts = []
    if not (daily_data and len(daily_data) >= 20 and current_price > 0):
        return "\n".join(parts) if parts else ""
    try:
        df = pd.DataFrame(daily_data,
                          columns=["ts","o","h","l","c","v","volCcy","volCcyQuote","confirm"])
        for col in ["h","l","c","v"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["h","l","c"]).sort_values("ts").reset_index(drop=True)
        n_days = len(df)

        hi_all = float(df["h"].max())
        lo_all = float(df["l"].min())
        df_90  = df.tail(90)
        hi_90  = float(df_90["h"].max())
        lo_90  = float(df_90["l"].min())

        range_pos = (current_price - lo_90) / (hi_90 - lo_90 + 1e-9) * 100
        dist_ath_pct = (current_price - hi_all) / hi_all * 100

        if range_pos >= 80:
            pos_desc = f"处于90日区间顶部({range_pos:.0f}%)，接近压力区，谨防回调"
        elif range_pos <= 20:
            pos_desc = f"处于90日区间底部({range_pos:.0f}%)，接近支撑区，关注反弹"
        elif 40 <= range_pos <= 60:
            pos_desc = f"处于90日区间中轴({range_pos:.0f}%)，方向中性，等待突破方向"
        elif range_pos > 60:
            pos_desc = f"处于90日区间中上部({range_pos:.0f}%)，偏强，但需确认量能"
        else:
            pos_desc = f"处于90日区间中下部({range_pos:.0f}%)，偏弱，反弹需谨慎"

        ath_desc = f"距{n_days}日高点{dist_ath_pct:+.1f}%"
        parts.append(f"价格位置: {pos_desc}，{ath_desc}")

        daily_range = (df["h"] - df["l"]).astype(float)
        atr_now5   = float(daily_range.tail(5).mean())
        atr_base30 = float(daily_range.tail(30).mean())
        atr_ratio  = atr_now5 / (atr_base30 + 1e-9)

        if atr_ratio < 0.6:
            vol_desc = (f"低波动缩量期（当前5日均振幅={atr_now5:.0f}，为30日均值的{atr_ratio:.0%}），"
                        f"突破信号多为假突破，建议提高入场置信度阈值至0.75以上")
        elif atr_ratio > 1.8:
            vol_desc = (f"高波动放量期（当前5日均振幅={atr_now5:.0f}，为30日均值的{atr_ratio:.0%}），"
                        f"止损应至少2×ATR以上，避免被噪波扫损")
        elif atr_ratio > 1.3:
            vol_desc = (f"波动率偏高（{atr_ratio:.0%}），趋势行情概率上升，可适当放宽止损")
        else:
            vol_desc = (f"波动率正常（{atr_ratio:.0%}），常规止损策略适用")
        parts.append(f"波动率环境: {vol_desc}")
    except Exception as e:
        log.debug(f"宏观上下文计算失败: {e}")

    return "\n".join(parts) if parts else ""


# ============================================================
# 指标辅助函数
# ============================================================
def _get_rsi_interval(rsi: float) -> int:
    """将RSI映射到区间：0-30(0), 30-50(1), 50-70(2), 70-100(3)"""
    if rsi < 30:
        return 0
    elif rsi < 50:
        return 1
    elif rsi < 70:
        return 2
    else:
        return 3


def _get_ma_alignment(ind_4h: Dict) -> int:
    """均线排列：空头排列(-1), 震荡(0), 多头排列(1)"""
    ema9 = ind_4h.get("ema9", 0)
    ema21 = ind_4h.get("ema21", 0)
    if ema9 > ema21:
        return 1
    elif ema9 < ema21:
        return -1
    else:
        return 0


def build_market_context(key_levels: Dict, sentiment: Dict, current_price: float) -> str:
    """
    将 L3（关键价位）和 L4（市场情绪深度）合并为紧凑文本，供 AI Prompt 使用。
    """
    lines = []

    if key_levels.get("_valid"):
        resistances = key_levels.get("resistances", [])
        supports    = key_levels.get("supports", [])
        atr_d       = key_levels.get("atr_daily", 0)

        lines.append("[L3 历史关键价位（日线Pivot，近3月）]")
        if resistances:
            r_parts = []
            for r in resistances[:3]:
                if current_price > 0:
                    dist_pct = (r["price"] - current_price) / current_price * 100
                    r_parts.append(f"{r['price']:.2f}(测试{r['count']}次,距今+{dist_pct:.1f}%)")
                else:
                    r_parts.append(f"{r['price']:.2f}(测试{r['count']}次)")
            lines.append(f"  阻力: {' | '.join(r_parts)}")
        else:
            lines.append("  阻力: 上方无明显历史阻力")

        if supports:
            s_parts = []
            for s in supports[:3]:
                if current_price > 0:
                    dist_pct = (current_price - s["price"]) / current_price * 100
                    s_parts.append(f"{s['price']:.2f}(测试{s['count']}次,距今-{dist_pct:.1f}%)")
                else:
                    s_parts.append(f"{s['price']:.2f}(测试{s['count']}次)")
            lines.append(f"  支撑: {' | '.join(s_parts)}")
        else:
            lines.append("  支撑: 下方无明显历史支撑")

        if atr_d > 0:
            lines.append(f"  日线ATR参考: {atr_d:.2f} USDT")

    if sentiment.get("_valid"):
        lines.append("\n[L4 市场情绪深度（OKX统计，1H周期）]")

        ls = sentiment.get("ls_ratio")
        if ls is not None:
            ls_desc = "多头拥挤⚠️" if ls > 1.5 else ("空头拥挤⚠️" if ls < 0.7 else "多空均衡")
            lines.append(f"  多空账户比: {ls:.3f} → {ls_desc}")

        oi    = sentiment.get("oi")
        oi_ch = sentiment.get("oi_change_pct")
        if oi is not None:
            oi_str = f"  持仓量(OI): {oi:,.0f} ETH"
            if oi_ch is not None:
                oi_signal = "🔴大规模清算/减仓" if oi_ch < -3 else ("🟡小幅减仓" if oi_ch < -1 else ("🟢持仓增加" if oi_ch > 2 else "稳定"))
                oi_str += f"  1H变化: {oi_ch:+.2f}% → {oi_signal}"
            lines.append(oi_str)

        tbr = sentiment.get("taker_buy_ratio")
        if tbr is not None:
            tbr_desc = "买盘主导🟢" if tbr > 0.55 else ("卖盘主导🔴" if tbr < 0.45 else "买卖均衡")
            lines.append(f"  主动买入比: {tbr:.3f} → {tbr_desc}")

    return "\n".join(lines) if lines else "（L3/L4数据不可用）"


# ── 技术指标计算 ─────────────────────────────────────────────────────────
def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=1).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=1).mean()
    rs       = avg_gain / avg_loss.replace(0, 1e-10)
    rsi      = 100 - 100 / (1 + rs)
    val      = float(rsi.iloc[-1])
    return val if not np.isnan(val) else 50.0

def _calc_macd(close: pd.Series,
               fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, bool, bool]:
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist        = macd_line - signal_line
    cur         = float(hist.iloc[-1])
    prev        = float(hist.iloc[-2]) if len(hist) >= 2 else 0.0
    return cur, prev, (prev <= 0 < cur), (prev >= 0 > cur)

def _calc_bb(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> Tuple[float, float, float]:
    ma  = close.rolling(period).mean()
    std = close.rolling(period).std()
    up  = float((ma + std_mult * std).iloc[-1])
    low = float((ma - std_mult * std).iloc[-1])
    pct = float((close.iloc[-1] - low) / (up - low + 1e-9))
    return up, low, pct

def _calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> float:
    tr  = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    val = float(tr.ewm(com=period - 1, min_periods=1).mean().iloc[-1])
    return val

def _calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    up_move = high.diff()
    down_move = low.diff() * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr.replace(0, 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr.replace(0, 1e-9))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(span=period, adjust=False).mean().iloc[-1]
    return float(adx) if not np.isnan(adx) else 25.0

def calc_indicators(data: list) -> Dict:
    if not data or len(data) < 50:
        return {"_valid": False}
    try:
        df = pd.DataFrame(data,
                          columns=["ts","o","h","l","c","v","volCcy","volCcyQuote","confirm"])
        for col in ["ts","o","h","l","c","v"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("ts").reset_index(drop=True)
        df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].ffill().bfill()
        df = df.dropna(subset=["c","h","l","v"])
        if str(data[0][-1]) != "1":
            df = df.iloc[:-1]
        if len(df) < 50:
            return {"_valid": False}

        _sentinel_close = df["c"]
        _tr_raw = pd.concat([
            df["h"] - df["l"],
            (df["h"] - _sentinel_close.shift(1)).abs(),
            (df["l"] - _sentinel_close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        _atr_raw = float(_tr_raw.ewm(span=14, adjust=False).mean().iloc[-1])
        # 动态 ATR 截断阈值：根据波动率分位数调整
        # 高波动 (ATR 比率>1.5) → 3.0×ATR，正常→2.0×ATR，低波动 (<0.8) → 1.5×ATR
        _atr_series = _tr_raw.ewm(span=14, adjust=False).mean()
        _atr_avg = float(_atr_series.tail(min(200, len(_atr_series))).mean())
        _atr_ratio = _atr_raw / (_atr_avg + 1e-9) if _atr_avg > 0 else 1.0
        if _atr_ratio > 1.5:
            _atr_mult = 3.0  # 高波动放宽截断
        elif _atr_ratio < 0.8:
            _atr_mult = 1.5  # 低波动收紧截断
        else:
            _atr_mult = 2.0  # 正常
        _cap_up  = _sentinel_close + _atr_mult * _atr_raw
        _cap_dn  = _sentinel_close - _atr_mult * _atr_raw
        _recent  = min(5, len(df))
        df.loc[df.index[-_recent:], "h"] = df["h"].clip(lower=_sentinel_close, upper=_cap_up)
        df.loc[df.index[-_recent:], "l"] = df["l"].clip(lower=_cap_dn, upper=_sentinel_close)
        df["h"] = df[["h", "c"]].max(axis=1)
        df["l"] = df[["l", "c"]].min(axis=1)

        close = df["c"]
        high  = df["h"]
        low   = df["l"]
        vol   = df["v"]

        rsi_val                               = _calc_rsi(close)
        hist_cur, hist_prev, cross_up, cross_dn = _calc_macd(close)
        bb_up, bb_low, bb_pct                 = _calc_bb(close)
        atr_val                               = _calc_atr(high, low, close)
        atr_series                            = _tr_raw.ewm(span=14, adjust=False).mean()
        atr_avg                               = float(atr_series.tail(min(200, len(atr_series))).mean())
        atr_ratio                             = atr_val / (atr_avg + 1e-9)
        adx_val                               = _calc_adx(high, low, close)

        roc_val  = float(close.pct_change(10).iloc[-1] * 100)
        typical  = (high + low + close) / 3.0
        sma_typical = typical.rolling(20).mean()
        mad = typical.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci_val = float((typical.iloc[-1] - sma_typical.iloc[-1]) / (0.015 * mad.iloc[-1] + 1e-9))

        ema9_val  = float(close.ewm(span=9,  adjust=False).mean().iloc[-1])
        ema21_val = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        ma50_val  = float(close.rolling(50, min_periods=1).mean().iloc[-1])

        def _isnan(v): return v != v
        nan_fields = {
            "rsi":    _isnan(rsi_val),
            "macd":   _isnan(hist_cur),
            "atr":    _isnan(atr_val),
            "ema9":   _isnan(ema9_val),
            "ema21":  _isnan(ema21_val),
            "bb_up":  _isnan(bb_up),
            "bb_pct": _isnan(bb_pct),
            "adx":    _isnan(adx_val),
            "roc":    _isnan(roc_val),
            "cci":    _isnan(cci_val),
        }
        bad = [k for k, v in nan_fields.items() if v]
        if bad:
            log.error(f"指标计算异常：{bad} 为 NaN，数据量不足（{len(df)}条），跳过本轮")
            return {"_valid": False}

        vol_avg   = float(vol.tail(20).mean())
        vol_surge = float(vol.iloc[-1]) / (vol_avg + 1e-9)

        price_rising = close.iloc[-1] > close.iloc[-5]
        vol_rising   = vol.iloc[-1] > vol.iloc[-5]
        divergence   = "bullish" if (price_rising and not vol_rising) else ("bearish" if (not price_rising and vol_rising) else "none")

        recent = df.tail(20)

        if "volCcyQuote" in df.columns and not df["volCcyQuote"].isnull().all():
            quote_vol = df["volCcyQuote"].astype(float)
            vwap = (quote_vol.sum()) / (vol.sum() + 1e-9)
        else:
            quote_vol = close * vol
            vwap = quote_vol.sum() / (vol.sum() + 1e-9)
        vwap_dist_pct = (close.iloc[-1] - vwap) / vwap * 100
        vwap_dist_pct = max(-20.0, min(20.0, vwap_dist_pct))

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=13, min_periods=1).mean()
        avg_loss = loss.ewm(com=13, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi_series = 100 - 100 / (1 + rs)
        rsi_series = rsi_series.fillna(50).clip(0, 100)
        ema_fast_s   = close.ewm(span=12, adjust=False).mean()
        ema_slow_s   = close.ewm(span=26, adjust=False).mean()
        macd_hist_s  = (ema_fast_s - ema_slow_s) - (ema_fast_s - ema_slow_s).ewm(span=9, adjust=False).mean()
        bb_mid_s     = close.rolling(20).mean()
        bb_std_s     = close.rolling(20).std()
        bb_pct_s     = (close - (bb_mid_s - 1.8 * bb_std_s)) / (3.6 * bb_std_s + 1e-9)
        vol_ma_s     = vol.rolling(20).mean()
        vol_ratio_s  = vol / (vol_ma_s + 1e-9)
        pct_chg_s    = close.pct_change() * 100

        df_enriched = df.copy()
        df_enriched["rsi"]       = rsi_series
        df_enriched["macd_hist"] = macd_hist_s
        df_enriched["bb_pct"]    = bb_pct_s
        df_enriched["vol_ratio"] = vol_ratio_s
        df_enriched["pct_chg"]   = pct_chg_s

        _adx_score = min(1.0, adx_val / 100.0)
        _ema_align = (1.0 if ema9_val > ema21_val else 0.0)
        _macd_ma5 = macd_hist_s.rolling(5, min_periods=1).mean().iloc[-1]
        _macd_dir = 1.0 if hist_cur > _macd_ma5 else (-1.0 if hist_cur < _macd_ma5 else 0.0)
        _macd_strength = min(1.0, abs(hist_cur) / (atr_val / close.iloc[-1] + 1e-9))
        _macd_score = _macd_dir * _macd_strength
        _price_ema_dist = (close.iloc[-1] - ema9_val) / (ema9_val + 1e-9)
        _ema_slope = (ema9_val - float(close.ewm(span=9, adjust=False).mean().iloc[-3])) / (ema9_val + 1e-9)
        _ma_stability = (1.0 if _price_ema_dist > 0 else 0.0) * (1.0 if _ema_slope > 0 else 0.5)
        # ADX 门控：趋势强度不足时，EMA/稳定性得分被压缩，避免假趋势给高分
        adx_gate = min(1.0, adx_val / 30.0)
        regime_score = round(
            _adx_score * 0.40 +
            (_ema_align * 0.30 + _ma_stability * 0.20) * adx_gate +
            (_macd_score * 0.5 + 0.5) * 0.10,
            3
        )

        return {
            "_valid":          True,
            "price":           float(close.iloc[-1]),
            "rsi":             rsi_val,
            "macd_hist":       hist_cur,
            "macd_cross_up":   cross_up,
            "macd_cross_down": cross_dn,
            "bb_pct":          bb_pct,
            "bb_upper":        bb_up,
            "bb_lower":        bb_low,
            "bb_width":        (bb_up - bb_low) / float(close.iloc[-1]) if close.iloc[-1] > 0 else 0,
            "atr":             atr_val,
            "atr_ratio":       atr_ratio,
            "adx":             adx_val,
            "roc":             roc_val,
            "cci":             cci_val,
            "ema9":            ema9_val,
            "ema21":           ema21_val,
            "ema_bull":        ema9_val > ema21_val,
            "trend":           "UP" if close.iloc[-1] > ma50_val else "DOWN",
            "vol_surge":       vol_surge,
            "support":         float(recent["l"].min()),
            "resistance":      float(recent["h"].max()),
            "divergence":      divergence,
            "vwap":            vwap,
            "vwap_dist_pct":   vwap_dist_pct,
            "regime_score":    regime_score,
            "_df":             df_enriched,
        }
    except Exception as e:
        log.debug(f"指标计算失败: {e}")
        return {"_valid": False}


def build_kline_series(ind_15m: Dict, ind_1h: Dict, n_15m: int = 15, n_1h: int = 5) -> str:
    """
    从 calc_indicators 返回的 _df 中提取最近 N 根已计算特征的 K 线，
    构造供 AI 感知趋势动态的紧凑文本。
    """
    lines = []

    df_15m: Optional[pd.DataFrame] = ind_15m.get("_df")
    if df_15m is not None and len(df_15m) >= n_15m:
        lines.append(f"[15m最近{n_15m}根 | 格式: 涨跌% RSI MACD方向 BB位置 量能]")
        subset = df_15m.tail(n_15m).copy()
        for i, (_, row) in enumerate(subset.iterrows()):
            pct    = row.get("pct_chg",   0.0)
            rsi    = row.get("rsi",       50.0)
            mhist  = row.get("macd_hist", 0.0)
            bbp    = row.get("bb_pct",    0.5)
            vratio = row.get("vol_ratio", 1.0)

            arrow    = "↑" if pct > 0.15 else ("↓" if pct < -0.15 else "→")
            prev_mh  = subset.iloc[i-1].get("macd_hist", 0.0) if i > 0 else 0.0
            macd_sym = ("▲" if mhist > prev_mh else "▼") if not (mhist != mhist) else "-"
            vol_sym  = "🔥" if vratio > 2.0 else ("↑" if vratio > 1.3 else ("↓" if vratio < 0.7 else " "))

            lines.append(
                f"  {arrow} {pct:+.2f}% | {rsi:.1f} | {macd_sym}     | {bbp:.2f} | {vol_sym}"
            )

        last5       = subset.tail(5)
        rsi_arr     = last5["rsi"].values
        pct_arr     = last5["pct_chg"].values
        vratio_arr  = last5["vol_ratio"].values
        narrative   = []

        if rsi_arr[-1] > rsi_arr[-3] + 5:
            narrative.append("RSI持续上行")
        elif rsi_arr[-1] < rsi_arr[-3] - 5:
            narrative.append("RSI持续下行")

        price_up   = pct_arr[-1] > 0
        vol_shrink = vratio_arr[-1] < vratio_arr[-3]
        if price_up and vol_shrink:
            narrative.append("价涨量缩(顶背离风险)")
        elif not price_up and not vol_shrink:
            narrative.append("价跌量增(抛压延续)")
        elif not price_up and vol_shrink:
            narrative.append("价跌量缩(下跌动能衰减)")
        elif price_up and not vol_shrink:
            narrative.append("价涨量增(趋势健康)")

        if rsi_arr[-1] > 72:
            narrative.append("RSI超买区")
        elif rsi_arr[-1] < 30:
            narrative.append("RSI超卖区")

        if narrative:
            lines.append(f"  📊 15m形态: {' | '.join(narrative)}")

    df_1h: Optional[pd.DataFrame] = ind_1h.get("_df")
    if df_1h is not None and len(df_1h) >= n_1h:
        lines.append(f"\n[1H最近{n_1h}根趋势摘要]")
        for _, row in df_1h.tail(n_1h).iterrows():
            pct   = row.get("pct_chg", 0.0)
            rsi   = row.get("rsi",     50.0)
            mh    = row.get("macd_hist", 0.0)
            arrow = "↑" if pct > 0.3 else ("↓" if pct < -0.3 else "→")
            lines.append(f"  {arrow} Δ{pct:+.2f}% RSI={rsi:.0f} MACD={'▲' if mh > 0 else '▼'}")

    return "\n".join(lines) if lines else "（K线序列数据不可用）"


def calc_key_levels(ind_1h: Dict, ind_4h: Dict, current_price: float) -> Dict:
    """
    L3 记忆层：从已有的 1H/4H K 线数据自动计算关键支撑阻力位。
    """
    result = {"supports": [], "resistances": [], "pivot": 0.0, "_valid": False}
    try:
        df_1h: Optional[pd.DataFrame] = ind_1h.get("_df")
        df_4h: Optional[pd.DataFrame] = ind_4h.get("_df")
        if df_1h is None or len(df_1h) < 20:
            return result

        dfs = [df_1h]
        if df_4h is not None and len(df_4h) >= 10:
            df_4h_weighted = df_4h.copy()
            df_4h_weighted = df_4h_weighted.assign(v=df_4h_weighted["v"].astype(float) * 3)
            dfs.append(df_4h_weighted)
        df = pd.concat(dfs, ignore_index=True).sort_values("ts").reset_index(drop=True)

        close  = df["c"].astype(float)
        high   = df["h"].astype(float)
        low    = df["l"].astype(float)
        volume = df["v"].astype(float)

        pivot = float((high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3)

        price_min = float(low.min())
        price_max = float(high.max())
        if price_max <= price_min:
            return result

        n_bins    = 50
        bin_size  = (price_max - price_min) / n_bins
        vol_bins  = np.zeros(n_bins)

        for i in range(len(df)):
            h_i = float(high.iloc[i])
            l_i = float(low.iloc[i])
            v_i = float(volume.iloc[i])
            b_lo = int((l_i - price_min) / bin_size)
            b_hi = int((h_i - price_min) / bin_size)
            b_lo = max(0, min(b_lo, n_bins - 1))
            b_hi = max(0, min(b_hi, n_bins - 1))
            span = b_hi - b_lo + 1
            for b in range(b_lo, b_hi + 1):
                vol_bins[b] += v_i / span

        threshold = np.percentile(vol_bins, 60)
        dense_prices = []
        for i, vb in enumerate(vol_bins):
            if vb >= threshold:
                center_price = price_min + (i + 0.5) * bin_size
                dense_prices.append((center_price, vb))

        supports    = sorted(
            [(p, v) for p, v in dense_prices if p < current_price * 0.995],
            key=lambda x: x[0], reverse=True
        )
        resistances = sorted(
            [(p, v) for p, v in dense_prices if p > current_price * 1.005],
            key=lambda x: x[0]
        )

        def _merge_levels(levels, merge_pct=0.003):
            if not levels:
                return []
            merged = [levels[0]]
            for price, vol in levels[1:]:
                last_price = merged[-1][0]
                if abs(price - last_price) / last_price > merge_pct:
                    merged.append((price, vol))
            return merged[:3]

        sup3 = _merge_levels(supports)
        res3 = _merge_levels(resistances)

        result = {
            "supports":    [{"price": round(p, 2), "count": int(v)} for p, v in sup3],
            "resistances": [{"price": round(p, 2), "count": int(v)} for p, v in res3],
            "pivot":       round(pivot, 2),
            "_valid":      True,
        }
    except Exception as e:
        log.debug(f"关键价位计算失败: {e}")
    return result


# ============================================================
# SignalsModule - 指标计算、数据获取、规则引擎
# 从 ETHTrader 拆分而出
# ============================================================

def bbsignal_to_reason(signal: str) -> str:
    return {
        "long": "下轨支撑，看多反弹",
        "short": "上轨压力，看空回落",
    }.get(signal, "")

def rsi_signal_to_reason(signal: str, rsi: float) -> str:
    return {
        "long": f"超卖 RSI={rsi:.1f}，反弹信号",
        "short": f"超买 RSI={rsi:.1f}，回落信号",
    }.get(signal, "")

class SignalsModule:
    """指标计算、数据获取、规则引擎模块"""

    # K线缓存TTL配置
    _RAW_KLINE_TTL = {
        "1m": 30,
        "3m": 45,
        "5m": 60,
        "15m": 90,
        "1H": 120,
        "4H": 300,
        "1D": 600,
    }

    def __init__(self, trader, config):
        self.trader = trader
        self.cfg = config
        self._ind_15m_cache: Dict[str, Any] = {}
        self._atr_history: List[float] = []
        self._raw_kline_cache: Dict[str, tuple] = {}  # (data, fetch_ts, last_ts)

    def fetch_data(self, bar: str, limit: int = None, symbol: str = None) -> Optional[list]:
        """获取K线数据"""
        if limit is None:
            limit = self.cfg.kline_limit
        sym = self.cfg.symbol
        params = {"instId": sym, "bar": bar, "limit": str(limit)}
        import requests
        for base in self.cfg.okx_kline_urls:
            try:
                r = requests.get(f"{base}/api/v5/market/candles", params=params, timeout=(3, 5))
                d = r.json()
                if d.get("code") == "0":
                    return d["data"]
                log.warning(f"K线返回code={d.get('code')} {base} bar={bar}")
            except Exception as e:
                log.warning(f"K线获取失败 {base} bar={bar}: {type(e).__name__}: {e}")
        return None

    def _fetch_kline_cached(self, bar: str) -> Optional[list]:
        """
        带缓存的 K 线拉取：
        - 若缓存未过期（TTL 内）且最新 candle 时间戳未变化，直接返回缓存数据
        - 否则发起 REST 请求并更新缓存
        """
        now_mono = time.monotonic()
        ttl = self._RAW_KLINE_TTL.get(bar, 60)
        cached = self._raw_kline_cache.get(bar)
        if cached:
            data, fetch_ts, last_ts = cached
            if (now_mono - fetch_ts) < ttl and data:
                return data
        fresh = self.fetch_data(bar, None, self.cfg.symbol)
        if fresh:
            last_ts = fresh[0][0] if fresh else ""
            self._raw_kline_cache[bar] = (fresh, now_mono, last_ts)
            log.debug(f"[{bar}] K线已刷新（TTL={ttl}s，{len(fresh)}根）")
        return fresh

    def _get_indicators_for_symbol(self, symbol: str, calc_indicators_func) -> tuple:
        """
        刷新获取指定品种的K线指标（供AI数据过期时重新获取）
        返回 (ind_15m, ind_1h, ind_4h) 元组
        calc_indicators_func: 从 ETHTrader 传入的 calc_indicators 函数引用
        """
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ind-fetch") as ex:
            f15 = ex.submit(self.fetch_data, "15m", None, symbol)
            f1h = ex.submit(self.fetch_data, "1H",  None, symbol)
            f4h = ex.submit(self.fetch_data, "4H",  None, symbol)
            raw_15m, raw_1h, raw_4h = f15.result(), f1h.result(), f4h.result()
        ind_15m = calc_indicators_func(raw_15m)
        ind_1h  = calc_indicators_func(raw_1h)
        ind_4h  = calc_indicators_func(raw_4h)
        return ind_15m, ind_1h, ind_4h

    def evaluate_rules(self, ind_15m: Dict, price: float, prev_indicators: Dict = None,
                      market_mode: str = "趋势", key_levels: Dict = None) -> Dict:
        """
        规则引擎主评估函数。
        使用布林带、RSI、成交量等确定性规则快速评估市场状态。
        明确信号（超买超卖）直接返回，无需触发AI。
        """
        if ind_15m is None or not ind_15m.get("_valid"):
            return {"trigger_ai": True, "signal_type": "hold", "confidence": 0.5, "reason": "数据无效，触发AI"}

        bb_upper = ind_15m.get("bb_upper", 0)
        bb_lower = ind_15m.get("bb_lower", 0)
        bb_pct = ind_15m.get("bb_pct", 0.5)
        rsi = ind_15m.get("rsi", 50)
        vol_surge = ind_15m.get("vol_surge", 1.0)

        prev_rsi = (prev_indicators or {}).get("rsi") if prev_indicators else None

        # 规则1：布林带位置检测
        bb_trigger, bb_signal = self._eval_bollinger_band_rule(price, bb_upper, bb_lower, bb_pct)

        # 规则2：RSI阈值检测
        rsi_trigger, rsi_signal = self._eval_rsi_rule(rsi, prev_rsi)

        # 规则3：成交量放量检测
        vol_trigger, vol_signal = self._eval_volume_rule(vol_surge)

        # 综合评估：任一规则给出明确信号则 bypass AI
        # 修复：confidence 改为动态计算，反映信号强度差异
        if not bb_trigger:
            # 布林带：根据偏离 BB 轨道的程度给分（0.65~0.85）
            _bb_extreme = max(abs(price / bb_upper - 1.0), abs(price / bb_lower - 1.0)) if bb_lower > 0 else 0
            bb_conf = min(0.85, 0.65 + _bb_extreme * 15)
            return {"trigger_ai": False, "signal_type": bb_signal, "confidence": round(bb_conf, 3), "reason": f"布林带{bbsignal_to_reason(bb_signal)}"}
        if not rsi_trigger:
            # RSI：根据极端程度给分（0.60~0.80）
            rsi_conf = min(0.80, 0.60 + abs(rsi - 50) / 50 * 0.3)
            return {"trigger_ai": False, "signal_type": rsi_signal, "confidence": round(rsi_conf, 3), "reason": f"RSI{rsi_signal_to_reason(rsi_signal, rsi)}"}
        if not vol_trigger:
            # 放量：根据 vol_surge 倍数给分（0.60~0.80）
            vol_conf = min(0.80, 0.55 + (vol_surge - 1.0) * 0.1)
            return {"trigger_ai": False, "signal_type": vol_signal, "confidence": round(vol_conf, 3), "reason": f"成交量异常放量{vol_surge:.1f}x"}

        return {"trigger_ai": True, "signal_type": "hold", "confidence": 0.5, "reason": "无明确规则信号，触发AI决策"}

    def _eval_bollinger_band_rule(self, price: float, bb_upper: float, bb_lower: float, bb_pct: float) -> Tuple[bool, str]:
        if bb_upper <= 0 or bb_lower <= 0:
            return True, "hold"

        if price >= bb_upper * 0.99:
            return False, "short"
        if price <= bb_lower * 1.01:
            return False, "long"

        return True, "hold"

    def _eval_rsi_rule(self, rsi: float, prev_rsi: float = None) -> Tuple[bool, str]:
        if rsi >= 75:
            return False, "short"
        if rsi <= 25:
            return False, "long"
        if prev_rsi and prev_rsi < 30 and rsi >= 30 and rsi < 50:
            return False, "long"
        if prev_rsi and prev_rsi > 70 and rsi <= 70 and rsi > 50:
            return False, "short"
        return True, "hold"

    def _eval_volume_rule(self, vol_surge: float) -> Tuple[bool, str]:
        # 提高门槛至 2.5/2.0，1.5x 在 15m 上太常见（尤其美股开盘时段）
        if vol_surge >= 2.5:
            return False, "volume_surge"
        if vol_surge >= 2.0:
            return False, "volume_confirm"
        return True, "hold"

    def _detect_breakout(self, raw_klines: list, current_price: float, direction: str = "long",
                        ind_15m: Dict = None, market_mode: str = "趋势") -> dict:
        if not raw_klines or len(raw_klines) < 20:
            return {"breakout": False}

        filtered = self._filter_raw_klines_flash_crash(raw_klines)
        closes = [float(k[4]) for k in filtered[-20:]]
        highs  = [float(k[2]) for k in filtered[-20:]]
        lows   = [float(k[3]) for k in filtered[-20:]]

        if len(closes) < 20:
            return {"breakout": False}

        atr = self._get_atr_from_klines(filtered[-21:])
        if atr <= 0:
            atr = ind_15m.get("atr", 20) if ind_15m else 20

        # 修复1：放量基线改用截尾均值（去最高最低各10%），防单根插针 K 线拉高基线
        _vols = sorted([float(k[5]) for k in filtered[-20:]])
        _trim = max(1, len(_vols) // 10)  # 去头尾各10%
        avg_vol = sum(_vols[_trim:-_trim]) / max(len(_vols) - 2 * _trim, 1) if len(_vols) > 4 else sum(_vols) / len(_vols)
        current_vol = float(filtered[-1][5]) if filtered else 0
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        params = self._get_adaptive_breakout_params(atr, avg_vol, current_vol, current_price)
        vol_threshold = params["vol_threshold"]
        sl_atr_mult = params["sl_atr_mult"]

        # 修复2：ADX 门控，过滤假突破（无趋势动能时突破极易被拉回）
        adx = (ind_15m or {}).get("adx", 30)
        if adx < 18:
            return {"breakout": False}  # 趋势强度不足，直接过滤
        if adx < 25:
            vol_threshold *= 1.3  # 弱趋势下提高放量要求

        if market_mode == "震荡激进":
            vol_threshold = max(vol_threshold, 2.0)

        recent_high = max(highs[-params["period"]:])
        recent_low = min(lows[-params["period"]:])

        if direction == "long":
            if current_price > recent_high and current_price > closes[-2]:
                if vol_ratio >= vol_threshold:
                    sl = current_price - atr * sl_atr_mult
                    reason = f"突破{params['period']}高點+放量{vol_ratio:.1f}x"
                    confidence = min(0.60 + (vol_ratio - vol_threshold) * 0.05, 0.80)
                    return {"breakout": True, "confidence": confidence, "reason": reason, "sl": sl, "atr": atr}
        else:
            if current_price < recent_low and current_price < closes[-2]:
                if vol_ratio >= vol_threshold:
                    sl = current_price + atr * sl_atr_mult
                    reason = f"跌破{params['period']}低點+放量{vol_ratio:.1f}x"
                    confidence = min(0.60 + (vol_ratio - vol_threshold) * 0.05, 0.80)
                    return {"breakout": True, "confidence": confidence, "reason": reason, "sl": sl, "atr": atr}

        return {"breakout": False}

    def _filter_raw_klines_flash_crash(self, raw_klines: list) -> list:
        if not raw_klines or len(raw_klines) < 5:
            return raw_klines
        _recent_klines = raw_klines[-50:]
        _closes = [float(k[4]) for k in _recent_klines]
        _highs  = [float(k[2]) for k in _recent_klines]
        _lows   = [float(k[3]) for k in _recent_klines]
        _trs = []
        for i in range(1, len(_closes)):
            _tr = max(abs(_highs[i-1] - _lows[i-1]), abs(_highs[i-1] - _closes[i]), abs(_lows[i-1] - _closes[i]))
            _trs.append(_tr)
        _atr = sum(_trs) / len(_trs) if _trs else 0
        _cap_mult = 2.0
        _filtered = [list(k) for k in raw_klines]
        for i in range(max(0, len(_filtered) - 5), len(_filtered)):
            k = _filtered[i]
            c, h, l = float(k[4]), float(k[2]), float(k[3])
            _cap_h = c + _cap_mult * _atr
            _cap_l = c - _cap_mult * _atr
            if h > _cap_h:
                k[2] = str(_cap_h)
            if l < _cap_l:
                k[3] = str(_cap_l)
            k[2] = str(max(float(k[2]), c))
            k[3] = str(min(float(k[3]), c))
        return _filtered

    def _get_adaptive_breakout_params(self, atr: float, avg_vol: float, current_vol: float, current_price: float = 0) -> dict:
        # 修复：改用 ATR 比率（相对值）代替 ATR 绝对值分档，适配 ETH 不同价格区间
        _atr_pct = atr / current_price if current_price > 0 and atr > 0 else 0.01
        if _atr_pct > 0.015:  # 高波动（>1.5% ATR）
            vol_thresh = 1.8
            sl_mult = 2.0
            period = 20
        elif _atr_pct > 0.008:  # 中等波动（0.8%~1.5% ATR）
            vol_thresh = 1.5
            sl_mult = 1.5
            period = 20
        else:  # 低波动（<0.8% ATR）
            vol_thresh = 1.2
            sl_mult = 1.2
            period = 15
        return {"vol_threshold": vol_thresh, "sl_atr_mult": sl_mult, "period": period}

    def _get_atr_from_klines(self, klines: list) -> float:
        if not klines or len(klines) < 2:
            return 0
        trs = []
        for i in range(1, len(klines)):
            h_i = float(klines[i][2])      # 当前K线 High
            l_i = float(klines[i][3])      # 当前K线 Low
            c_prev = float(klines[i-1][4]) # 前一根 Close
            if h_i > 0 and l_i > 0:
                tr = max(h_i - l_i, abs(h_i - c_prev), abs(l_i - c_prev))
                trs.append(tr)
        return sum(trs) / len(trs) if trs else 0
