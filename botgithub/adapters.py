"""
Adapters Module - AI接口适配器
包含 SmartAIConsultant, AIGatekeeper, ArbitrationTrigger, ConvictionScorer
以及 FastLaneModule（AI缓存与快速决策通道）
从 ETHTrader 拆分而出
"""

import time
import re
import json
import math
import threading
import logging
from typing import Optional, Dict, List, Any, TYPE_CHECKING
from datetime import datetime, timezone, timedelta
from openai import OpenAI

if TYPE_CHECKING:
    from market import SignalsModule

# ── 共享基础设施（common.py）──────────────────────────────────────────────
from common import (
    CFG, log, _webhook, _parse_dt, UTC,
    gs_get, gs_set, gs_increment, _call_reasoner,
    SYSTEM_PROMPT_TRADE,
)
# ── 数据模型（core.py）──────────────────────────────────────────────────
from core import GLOBAL_STATE
# ── market_data 模块 ───────────────────────────────────────────────────
from market import build_kline_series, calc_key_levels
from position_exec import _get_atr_quantile


# ── 辅助函数（从主模块迁移）─────────────────────────────────────────────
def _price_of_level(item) -> float:
    """从支撑/阻力位对象提取价格"""
    if isinstance(item, dict):
        return float(item.get("price", item))
    return float(item)


def _call_reasoner_for_json(ai_client, messages: list, max_tokens: int = 2000,
                           timeout: int = 120) -> Dict:
    """调用 reasoner 并解析 JSON 响应"""
    raw = _call_reasoner(ai_client, messages, max_tokens=max_tokens, timeout=timeout)
    return _parse_llm_json(raw)


def get_market_mode(ind_15m: Dict, current_price: float, prev_mode: str = "趋势") -> str:
    """
    计算当前市场模式（震荡激进/震荡/趋势），带滞后阈值防止频繁切换。
    三种模式：
      震荡激进：ADX极低(< osc_aggressive_adx_thresh) 且布林带较窄(< osc_aggressive_bb_thresh)
               适合快进快出的均值回归，止损更紧，冷却更短
      震荡：    布林带较窄(< osc_bb_width_thresh)，适合支撑/阻力位反转
      趋势：    布林带较宽，适合追势
    滞后逻辑：±10%死区防抖，以及震荡激进ADX±3死区，避免模式频繁切换。
    """
    bb_upper = ind_15m.get("bb_upper", 0)
    bb_lower = ind_15m.get("bb_lower", 0)
    p = current_price or 1
    bb_width_raw = (bb_upper - bb_lower) / p if bb_upper > 0 else 0.05
    # BB物理宽度地板（市值越高，绝对波动越大）
    _physical_floor = 0.015
    bb_width = max(bb_width_raw, _physical_floor)
    adx      = ind_15m.get("adx", 30)
    thresh   = CFG.osc_bb_width_thresh
    agg_bb   = CFG.osc_aggressive_bb_thresh
    agg_adx  = CFG.osc_aggressive_adx_thresh

    # 波动率自适应死区：根据 ATR 比率调整滞后阈值
    _atr_ratio = ind_15m.get('atr_ratio', 1.0)
    if _atr_ratio > 1.5:
        _hysteresis_mult = 1.15
        _adx_buffer = 4
    elif _atr_ratio < 0.8:
        _hysteresis_mult = 1.05
        _adx_buffer = 2
    else:
        _hysteresis_mult = 1.10
        _adx_buffer = 3

    if prev_mode == "震荡激进":
        if bb_width < agg_bb * _hysteresis_mult and adx < agg_adx + _adx_buffer:
            return "震荡激进"
        elif bb_width < thresh * 1.1:
            return "震荡"
        else:
            return "趋势"
    elif prev_mode == "震荡":
        if bb_width < thresh * 1.1:
            if bb_width < agg_bb and adx < agg_adx:
                return "震荡激进"
            return "震荡"
        else:
            return "趋势"
    elif prev_mode == "趋势":
        if bb_width > thresh * (2 - _hysteresis_mult):
            return "趋势"
        if bb_width < agg_bb * (2 - _hysteresis_mult) and adx < agg_adx * (2 - _hysteresis_mult/1.1):
            return "震荡激进"
        return "震荡"
    else:
        if bb_width < agg_bb and adx < agg_adx:
            return "震荡激进"
        elif bb_width < thresh:
            return "震荡"
        else:
            return "趋势"


def _calc_period_score(ind_15m: Dict, ind_1h: Dict, ind_4h: Dict) -> tuple:
    """
    预计算多周期评分（仅算一次，供 simple prompt 和 reasoner prompt 共用）。
    返回 (score_float, score_desc)。
    """
    def _sc(t, r, m):
        s = 0.5 + (0.5 if t == "UP" else -0.5 if t == "DOWN" else 0)
        s += (r - 50) / 100
        s += 0.2 if m > 0 else -0.2 if m < 0 else 0
        return max(0, min(1, s))
    sc = (_sc(ind_15m.get("trend",""), ind_15m.get("rsi",50), ind_15m.get("macd_hist",0)) * 0.3
          + _sc(ind_1h.get("trend",""),  ind_1h.get("rsi",50), ind_1h.get("macd_hist",0))  * 0.3
          + _sc(ind_4h.get("trend",""),  ind_4h.get("rsi",50), ind_4h.get("macd_hist",0))  * 0.4)
    score_desc = ("强烈看多" if sc > 0.65 else "看多" if sc > 0.55
                  else "中性" if sc > 0.45 else "看空" if sc > 0.35 else "强烈看空")
    return sc, score_desc


def build_funding_trend(funding_history: List[Dict]) -> str:
    """根据资金费率历史生成趋势描述"""
    if not funding_history:
        return "费率趋势: 数据不足"
    recent = funding_history[-6:]
    rates = [f.get("funding_rate", 0) for f in recent]
    avg = sum(rates) / len(rates) if rates else 0
    direction = "正" if avg > 0 else "负"
    return f"近3h平均费率: {avg*100:+.4f}% ({direction}费率环境)"

# ── JSON 解析辅助 ─────────────────────────────────────────────────────────
def _clean_json_text(raw_text: str) -> str:
    """剥离 Markdown 围栏，返回纯文本"""
    return re.sub(r'```(?:json)?\s*', '', raw_text).replace('```', '').strip()

def _parse_llm_json(raw_text: str) -> Dict:
    """从 LLM 原始输出中稳健提取 JSON 决策字典（栈匹配）"""
    cleaned = _clean_json_text(raw_text)
    start_idx = cleaned.find('{')
    if start_idx != -1:
        stack = 0
        for i in range(start_idx, len(cleaned)):
            if cleaned[i] == '{':
                stack += 1
            elif cleaned[i] == '}':
                stack -= 1
                if stack == 0:
                    candidate = cleaned[start_idx:i+1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, dict) and "action" in result:
                            return result
                    except json.JSONDecodeError:
                        pass
                    break
    # 最终兜底
    try:
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end > start:
            return json.loads(raw_text[start:end+1])
    except Exception:
        pass
    # 详细记录解析失败原因，便于排查 AI 决策异常
    _error_ctx = raw_text[:300].replace("\n", " ")
    log.error(f"JSON 解析失败 | 长度={len(raw_text)} | 含 action={'action' in raw_text} | 原始输出 (前 300 字): {_error_ctx}")
    return {"action": "hold", "confidence": 0.0, "suggested_sl": 0,
            "suggested_tp": 0, "suggested_leverage": 1,
            "reason": "JSON解析失败，自动 hold", "thought_process": ""}



# ============================================================
# AIGatekeeper - AI请求门控（缓存/熔断/防抖）
# ============================================================
class AIGatekeeper:
    """
    将原来散落在 ETHTrader 中的三套「不调 AI」机制收口到一处：
      Silence  — 持仓无变化时的静默期
      Cache    — 相同市场状态复用决策
      Debounce — 快速信号去重
    对外唯一接口：should_skip() / get_cached() / set_cache() / clear()
    """

    def __init__(self):
        # Cache
        self._cache:          Optional[Dict] = None
        self._cache_ts:       float = 0.0
        self._cache_hash:     str   = ""
        self._cache_lock:     threading.Lock = threading.Lock()
        # 持久化最近一次有效AI决策，防止异步调用间隙返回 None 导致 "AI 未就绪"
        self._last_decision:  Optional[Dict] = None
        self._last_decision_ts: float = 0.0
        # Drift snapshot
        self._last_price:     float = 0.0
        self._last_rsi_bkt:   Optional[str] = None
        self._last_bb_zone:   Optional[str] = None
        # Silence
        self._last_request_ts:    float = 0.0
        self._last_decision_price: float = 0.0
        self._last_decision_rsi:  float = 50.0
        self._last_decision_macd: float = 0.0
        # Debounce
        self._signal_cache:   Dict = {}
        self._last_breakout_ts: Dict = {"up": 0.0, "down": 0.0}
        # Circuit breaker
        self._circuit_broken_until: float = 0.0
        self._failure_count:        int   = 0
        self._degraded_mode:     bool  = False
        self._cache_decay_mode:  bool  = False
        self._next_retry_delay:  float = 0.0
        self._rate_limit_until:     float = 0.0
        # FastLane
        self._entry_fasttrack_mult:  float = 0.0
        self._entry_fasttrack_until: float = 0.0

    def get_cached(self, market_mode: str = "") -> Optional[Dict]:
        """返回有效缓存；过期则清除并返回最近一次有效决策（而非 None）"""
        with self._cache_lock:
            if self._cache is not None:
                ttl = self._get_ttl(market_mode)
                age = time.monotonic() - self._cache_ts
                if age > ttl:
                    log.info(f"AI缓存已过期（{age:.0f}s>{ttl}s），强制刷新")
                    self._cache = None
                    self._cache_hash = ""
                else:
                    return self._cache
            # 缓存为空或已过期时，返回最近一次有效决策，避免异步调用间隙出现 "AI 未就绪"
            if self._last_decision is not None:
                return dict(self._last_decision)  # 返回副本防止外部修改
            return None

    def set_cache(self, decision: Dict, input_sig: str,
                  price: float, rsi: float, macd: float,
                  rsi_bkt: str, bb_zone: str):
        with self._cache_lock:
            self._cache           = decision
            self._cache_ts        = time.monotonic()
            self._cache_hash      = input_sig
            self._last_request_ts = time.monotonic()
            self._last_decision_ts    = time.monotonic()
            self._last_decision_price = price
            self._last_decision_rsi   = rsi
            self._last_decision_macd  = macd
            self._last_rsi_bkt    = rsi_bkt
            self._last_bb_zone   = bb_zone
            # 持久化最近一次有效AI决策
            self._last_decision  = dict(decision)

    def clear(self):
        with self._cache_lock:
            self._cache       = None
            self._cache_ts    = 0.0
            self._cache_hash  = ""
            self._last_rsi_bkt  = None
            self._last_bb_zone  = None

    @property
    def cache_hash(self) -> str:
        return self._cache_hash

    @property
    def last_request_ts(self) -> float:
        return self._last_request_ts

    @property
    def last_decision_price(self) -> float:
        return self._last_decision_price

    @property
    def last_decision_rsi(self) -> float:
        return self._last_decision_rsi

    @property
    def last_decision_macd(self) -> float:
        return self._last_decision_macd

    @property
    def circuit_broken(self) -> bool:
        return (
            time.monotonic() < self._circuit_broken_until
            or time.monotonic() < self._rate_limit_until
        )

    def record_rate_limit(self, retry_after_seconds: int = 60):
        self._rate_limit_until = time.monotonic() + retry_after_seconds
        log.warning(f"AIGatekeeper: 429限流冷却{retry_after_seconds}s")

    def mark_entry_fasttrack(self, vspike_mult: float):
        self._entry_fasttrack_mult  = vspike_mult
        self._entry_fasttrack_until = time.monotonic() + 30.0
        log.info(f"[AIGatekeeper] Mode: AGGRESSIVE | VSpike={vspike_mult:.1f}x | FastTrack激活30s")

    @property
    def entry_fasttrack_mult(self) -> float:
        if time.monotonic() < self._entry_fasttrack_until:
            return self._entry_fasttrack_mult
        return 0.0

    def record_failure(self):
        self._failure_count += 1
        n = self._failure_count
        base = CFG.ai_failure_exp_backoff
        self._next_retry_delay = min(base ** n, 120.0)
        if n <= 3:
            log.warning(f"AIGatekeeper: AI失败第{n}次，退避{self._next_retry_delay:.0f}s")
        elif n <= 6:
            self._degraded_mode = True
            self._cache_decay_mode = False
            self._circuit_broken_until = time.monotonic() + 300.0
            log.warning(f"AIGatekeeper: AI失败第{n}次，进入ConvictionOnly降级模式(Kelly×0.6)，熔断5min")
        else:
            self._cache_decay_mode = True
            self._circuit_broken_until = time.monotonic() + 600.0
            log.error(f"AIGatekeeper: AI失败第{n}次，进入缓存衰减模式，熔断10min")

    def reset_failure(self):
        self._failure_count = 0
        self._degraded_mode = False
        self._cache_decay_mode = False
        self._next_retry_delay = 0.0

    @property
    def is_degraded(self) -> bool:
        return self._degraded_mode

    @property
    def is_cache_decay(self) -> bool:
        return self._cache_decay_mode

    @property
    def next_retry_delay(self) -> float:
        return self._next_retry_delay

    @staticmethod
    def _get_ttl(market_mode: str) -> int:
        if market_mode == "趋势":
            return CFG.cache_ttl_trend
        elif market_mode == "震荡激进":
            return int(CFG.cache_ttl_osc * 0.67)
        return CFG.cache_ttl_osc


# ============================================================
# ArbitrationTrigger - 千问并行投票
# ============================================================
class ArbitrationTrigger:
    """
    千问并行投票触发器。
    只在 ConvictionScorer fd_score 处于边缘区（70~82）时触发。
    """

    def __init__(self):
        self._qwen_client = None
        self._qwen_available = False
        self._init_qwen()

    def _init_qwen(self):
        if gs_get("qwen_timeout_count") is None:
            gs_set("qwen_timeout_count", 0)
        if gs_get("qwen_timeout_window") is None:
            gs_set("qwen_timeout_window", time.time())
        if not CFG.qwen_api_key:
            self._qwen_available = False
            return
        try:
            import openai
            self._qwen_client = openai.OpenAI(
                api_key=CFG.qwen_api_key,
                base_url=CFG.qwen_base_url,
                timeout=CFG.qwen_timeout,
            )
            self._qwen_available = True
            log.info(f"[ArbitrationTrigger] 千问已就绪: model={CFG.qwen_model}")
        except Exception as e:
            log.warning(f"[ArbitrationTrigger] 千问初始化失败: {e}，仲裁功能禁用")
            self._qwen_available = False

    def _record_timeout(self, err: str):
        _now = time.time()
        _window = gs_get("qwen_timeout_window", _now)
        if _now - _window > 86400:
            gs_set("qwen_timeout_count", 0)
            gs_set("qwen_timeout_window", _now)
        gs_increment("qwen_timeout_count", 1)
        _cnt = gs_get("qwen_timeout_count", 0)
        log.warning(f"[ArbitrationTrigger] 千问超时/失败×{_cnt}（近24h）: {err[:80]}")
        if _cnt >= CFG.ai_timeout_alert_count:
            log.critical(f"千问API稳定性告警：24h内超时{_cnt}次，建议检查网络或扩展timeout={CFG.qwen_timeout}s")
            _webhook(
                f"千问API不稳定",
                f"24h内超时{_cnt}次，当前timeout={CFG.qwen_timeout}s"
            )

    def should_trigger(self, fd_score: float) -> bool:
        if not self._qwen_available:
            return False
        return CFG.arbitration_min_score <= fd_score < CFG.arbitration_max_score

    def call_qwen(self,
                   action: str,
                   vspike_mult: float,
                   ob_imbalance: float,
                   rsi: float,
                   market_mode: str,
                   depth_dir: str,
                   price: float,
                   reason: str,
    ) -> Optional[Dict]:
        if not self._qwen_available:
            return None

        side_tag = "做多(买)" if action == "open_long" else "做空(卖)"
        direction_tag = {
            "买方主导": "买盘强势（顺势做多）",
            "卖方主导": "卖盘强势（顺势做空）",
            "均衡": "多空均衡（谨慎）",
        }.get(depth_dir, "未知")

        prompt = f"""你是 ETH 永续合约的交易顾问，只回答方向和置信度。
当前简明局势：
- 订单流方向：{direction_tag}
- VSpike 倍数：{vspike_mult:.1f}x（成交量突增）
- 订单簿失衡度：{ob_imbalance:.2f}（>0买强，<0卖强）
- RSI（14）：{rsi:.1f}
- 市场模式：{market_mode}
- 当前价格：${price:.2f}
你的任务：
仅根据以上信息，判断是否{side_tag}。
输出格式（严格JSON）：
{{"action": "open_long" 或 "open_short" 或 "hold", "confidence": 0.0~1.0, "reason": "一句话理由"}}
注意：不要复述数据，只给判断和置信度。"""

        try:
            response = self._qwen_client.chat.completions.create(
                model=CFG.qwen_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            content = response.choices[0].message.content.strip()
            import json as _json
            try:
                data = _json.loads(content)
            except json.JSONDecodeError as e:
                log.error(f"[ArbitrationTrigger] Qwen JSON 解析失败：{e} | 原始输出：{content[:200]}")
                data = {"action": "hold", "confidence": 0.0, "reason": "JSON 解析失败"}
            log.info(f"[ArbitrationTrigger] 千问投票: {data.get('action')} conf={data.get('confidence', 0):.2f}")
            return {
                "action": data.get("action", "hold"),
                "confidence": float(data.get("confidence", 0.5)),
                "reason": data.get("reason", ""),
            }
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                retry_after = 60
                if "retry-after" in err_str.lower():
                    try:
                        retry_after = int(''.join(c for c in err_str if c.isdigit())) or 60
                    except:
                        pass
                self.record_rate_limit(retry_after)
                log.warning(f"[ArbitrationTrigger] 千问触发429限流，冷却{retry_after}s")
            else:
                self._record_timeout(err_str)
            return None

    def resolve(self,
                 qwen_result: Optional[Dict],
                 ds_score: float,
                 ds_kelly: float,
                 ds_action: str,
                 ds_conf: float,
                 sym: str = "ETH") -> Dict:
        if qwen_result is None:
            return {"action": ds_action, "confidence": ds_conf,
                    "kelly_override": ds_kelly, "sl_tighten_mult": 1.0,
                    "source": "deepseek_only"}
        q_action = qwen_result.get("action", "hold")
        q_conf   = qwen_result.get("confidence", 0.5)

        if q_action == "hold":
            return {"action": "hold", "confidence": min(ds_conf, q_conf) * 0.5,
                    "kelly_override": 0.0, "sl_tighten_mult": 1.0,
                    "source": "qwen_veto"}

        if q_action == ds_action:
            final_conf = (ds_conf + q_conf) / 2
            final_kelly = min(ds_kelly * 1.1, 1.0)
            return {"action": ds_action, "confidence": final_conf,
                    "kelly_override": final_kelly, "sl_tighten_mult": 1.0,
                    "source": "qwen_agree"}
        else:
            return {"action": q_action, "confidence": min(ds_conf, q_conf) * 0.7,
                    "kelly_override": ds_kelly * 0.7, "sl_tighten_mult": 0.85,
                    "source": "qwen_disagree"}


# ============================================================
# 趋势对齐分数（Trend Alignment Score）
# 多时间框架 ADX + EMA 一致性 + 关键位突破确认 → 0~1 连续分数
# 取代旧的单一 4H EMA 二元判断
# ============================================================
def get_trend_alignment_score(ind_15m: Dict, ind_1h: Dict, ind_4h: Dict,
                              ind_1d: Optional[Dict] = None) -> tuple:
    """
    返回 (score, direction)
    - score: 0.0~1.0，连续趋势强度分数（越大代表单边趋势越强）
    - direction: "bull" / "bear" / "neutral"
    """
    price = ind_15m.get("price", 0) if ind_15m else 0

    # ── 1. ADX 趋势强度（45%）─────────────────────────────────────────────
    adx_1h = (ind_1h.get("adx", 20) or 20) if ind_1h else 20
    adx_4h = (ind_4h.get("adx", 20) or 20) if ind_4h else 20
    adx_score = max(0.0, min(1.0, (max(adx_1h, adx_4h) - 15) / 25))

    # ── 2. 多时间框架 EMA 一致性（40%）───────────────────────────────────
    # 权重：4H(×2) > 1H(×1) > 1D(×1, 可选)
    ema_1h = ind_1h.get("ema_bull", False) if ind_1h else None
    ema_4h = ind_4h.get("ema_bull", False) if ind_4h else None
    ema_1d = (ind_1d.get("ema_bull", False) if ind_1d else None) or ema_4h

    weights = []
    if ema_1h is not None:
        weights.append((1, ema_1h))
    if ema_4h is not None:
        weights.append((2, ema_4h))
    if ema_1d is not None:
        weights.append((1, ema_1d))

    if not weights:
        # 无任何 EMA 数据，降级为 neutral
        return 0.0, "neutral"

    total_w = sum(w for w, _ in weights)
    # 净方向：[-1.0, 1.0]
    net_alignment = sum(w if bull else -w for w, bull in weights) / total_w
    # 一致性强度：取绝对值，0=完全对立，1=完全一致
    alignment_strength = abs(net_alignment)

    # ── 3. 关键位突破确认（15%）───────────────────────────────────────────
    proximity_bonus = 0.0
    if price > 0 and ind_1h and ind_4h:
        try:
            key_levels = calc_key_levels(ind_1h, ind_4h, price)
            if key_levels.get("_valid"):
                supports = key_levels.get("supports", [])
                resistances = key_levels.get("resistances", [])
                if supports:
                    nearest_support = min(l["price"] for l in supports)
                else:
                    nearest_support = price
                if resistances:
                    nearest_resist = max(l["price"] for l in resistances)
                else:
                    nearest_resist = price
                if net_alignment > 0 and price > nearest_resist * 1.005:
                    proximity_bonus = 0.15
                elif net_alignment < 0 and price < nearest_support * 0.995:
                    proximity_bonus = 0.15
        except Exception:
            pass  # 关键位计算失败不影响主逻辑

    # ── 4. 最终分数 ───────────────────────────────────────────────────────
    score = max(0.0, min(1.0,
        0.45 * adx_score
        + 0.40 * alignment_strength
        + 0.15 * proximity_bonus
    ))

    # ── 5. 方向判断 ───────────────────────────────────────────────────────
    if net_alignment >= 0.5:
        direction = "bull"
    elif net_alignment <= -0.5:
        direction = "bear"
    else:
        direction = "neutral"

    return score, direction


# ============================================================
# ConvictionScorer - 信心评分
# ============================================================
class ConvictionScorer:
    """
    五维信心评分器，输出综合分数 (0-100) 和 Kelly 比率。
    """

    def _arbitrate_final_score(self, score_val: float, kelly_ratio: float,
                               context: Dict) -> Dict:
        """
        统一仲裁引擎：AI 分数 × 环境乘数 × 风险乘数
        - 环境乘数：trend_score(0.6~1.0 连续) × ATR 波动系数
        - 风险乘数：回撤因子 × 连亏因子 (地板 0.5)
        - 最终分最低保底 35 分
        """
        ctx = context or {}
        _atr_ratio   = ctx.get("atr_ratio", 1.0)
        _trend_score = ctx.get("trend_alignment_score", 0.5)  # 默认中性

        # 环境乘数：trend_score(0.6~1.0) × ATR修正
        env_mult = (0.6 + 0.4 * _trend_score) * (
            1.12 if _atr_ratio > 1.5 else 0.90 if _atr_ratio < 0.7 else 1.00
        )

        # 风险乘数
        _dd_mult     = float(gs_get("dd_kelly_mult", 1.0))
        _consec      = float(gs_get("consecutive_losses", 0))
        _consec_mult = max(0.5, 1.0 - min(_consec * 0.05, 0.45))
        risk_mult    = _dd_mult * _consec_mult

        _arb_score = max(35.0, min(100.0, score_val * env_mult * risk_mult))
        _kelly_adj = kelly_ratio * env_mult * risk_mult

        return {
            "score":             round(_arb_score, 2),
            "kelly_ratio":       _kelly_adj,
            "env_mult":          round(env_mult, 3),
            "risk_mult":         round(risk_mult, 3),
            "trend_alignment_score": round(_trend_score, 3),
        }

    def score(self,
              ai_conf:      float,
              action:       str,
              vspike_mult:  float = 0.0,
              ob_imbalance: float = 0.0,
              rsi:          float = 50.0,
              at_key_level: bool  = False,
              market_mode:  str   = "趋势",
              context:       Dict  = None,
    ) -> Dict:
        is_long = action == "open_long"
        _ai_w_mult = float(gs_get("ai_weight_mult", 1.0))
        ai_score = ai_conf * 65.0 * _ai_w_mult  # 降低 AI 主导权，从 80→65
        tau = 4.5  # 从 8.0→4.5，让 2x~6x 区间分数梯度更明显
        spike_score = 25.0 * math.tanh(vspike_mult / tau) if vspike_mult > 0 else 0.0
        # VSpike ≥6.0x 提供逃生窗口：额外 +15 bonus，不受 tanh 饱和限制
        if vspike_mult >= 6.0:
            spike_score += 15.0
        if is_long:
            ob_score = ob_imbalance * 15.0  # 从 10→15，提升订单簿权重
        else:
            ob_score = -ob_imbalance * 15.0
        level_score = 8.0 if at_key_level else 0.0
        rsi_ideal = 35.0 if is_long else 65.0
        rsi_dist = abs(rsi - rsi_ideal) / 40.0
        rsi_penalty = min(6.0, rsi_dist * 6.0)  # 上限从 12→6，RSI 不应比盘口更重要
        raw = ai_score + spike_score + ob_score + level_score - rsi_penalty
        mode_coeff = {"趋势": 1.0, "震荡": 0.9, "震荡激进": 0.85}.get(market_mode, 1.0)
        score_val = max(0.0, min(100.0, raw * mode_coeff))
        sigmoid_k = 2.5 * (score_val - 50) / 50
        kelly_ratio = 1.0 / (1.0 + math.exp(-sigmoid_k))

        final = self._arbitrate_final_score(score_val, kelly_ratio, context)
        return {
            "score":       final["score"],
            "kelly_ratio": final["kelly_ratio"],
            "env_mult":    final["env_mult"],
            "risk_mult":   final["risk_mult"],
            "components": {
                "ai_raw":      round(score_val, 2),
                "spike":       spike_score,
                "ob":          ob_score,
                "level":       level_score,
                "rsi_penalty": rsi_penalty,
            }
        }


# ============================================================
# SmartAIConsultant - AI决策顾问
# ============================================================
class SmartAIConsultant:
    """
    AI 决策顾问：惰性双温投票 + RAG + 策略委员会加权。
    结构：_build_prompt() → 低温度单次请求 → (仅在有交易信号时)高温度二次确认 → _vote()。
    核心优化：大多数轮次只调用一次（hold 不触发第二温），节省 50%+ Token。
    """
    def __init__(self, client: OpenAI, trader=None):
        self.client = client
        self.trader = trader  # OkxTrader（= ETHTrader），供 _build_prompt 访问 _atr_history
        self.tick_size = 0.01
        # ── 千问仲裁 client（持仓/平仓分歧时使用）─────────────────────────
        self._qwen_client = None
        self._qwen_available = False
        self._init_qwen()

    # ── 千问仲裁：持仓/平仓分歧专用 ────────────────────────────────────────
    def _init_qwen(self):
        if not CFG.qwen_api_key:
            self._qwen_available = False
            return
        try:
            import openai
            self._qwen_client = openai.OpenAI(
                api_key=CFG.qwen_api_key,
                base_url=CFG.qwen_base_url,
                timeout=CFG.qwen_timeout,
            )
            self._qwen_available = True
            log.info(f"[SmartAIConsultant] 千问仲裁已就绪: model={CFG.qwen_model}")
        except Exception as e:
            log.warning(f"[SmartAIConsultant] 千问仲裁初始化失败: {e}，持仓仲裁功能禁用")
            self._qwen_available = False

    def _arbitrate_exit_dispute(self, r1: Dict, r2: Dict, a1: str, a2: str,
                                 c1: float, c2: float, pos_info: Dict,
                                 ind_15m: Dict, market_mode: str,
                                 vspike_mult: float, ob_imbalance: float) -> Dict:
        """
        持仓分歧仲裁：当投票在 hold vs close/close vs adjust_sl_tp 之间分歧时，
        调千问做最终裁决，替代原来的"强制 hold"。
        """
        if not self._qwen_available:
            log.warning("⚠️ 持仓分歧需仲裁但千问未就绪，降级为 hold")
            return {
                "action": "hold", "confidence": 0.0,
                "reason": f"投票分歧({a1} vs {a2})，千问未就绪，保守观望",
            }

        side = pos_info.get("side", "")
        pnl = pos_info.get("pnl_pct", 0)
        holding = pos_info.get("holding_minutes", 0)
        side_tag = f"持{side}" if side else "空仓"

        prompt = f"""你是ETH量化交易顾问，当前持仓状态：
- 持仓方向：{side_tag}
- 浮盈/浮亏：{pnl*100:+.2f}%
- 持仓时长：{holding:.0f}分钟
- 市场模式：{market_mode}
- VSpike：{vspike_mult:.1f}x
- 订单簿失衡：{ob_imbalance:.2f}（>0买强，<0卖强）
- 15m RSI：{ind_15m.get('rsi', 50):.1f}
- 15m MACD：{ind_15m.get('macd_hist', 0):.4f}
- 15m BB%：{ind_15m.get('bb_pct', 0.5):.2f}

两个独立模型给出分歧意见：
- 保守方：{a1}（置信度 {c1:.2f}）→ {r1.get('reason', '无')}
- 创意方：{a2}（置信度 {c2:.2f}）→ {r2.get('reason', '无')}

你的任务：裁决是否应该平仓。
输出格式（严格JSON）：
{{"action": "close" 或 "hold", "confidence": 0.0~1.0, "reason": "一句话裁决理由"}}"""

        try:
            response = self._qwen_client.chat.completions.create(
                model=CFG.qwen_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            content = response.choices[0].message.content.strip()
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = {"action": "hold", "confidence": 0.5, "reason": "千问JSON解析失败"}

            final_action = data.get("action", "hold")
            final_conf = float(data.get("confidence", 0.5))
            log.info(f"⚖️ 千问仲裁（持仓分歧）: {a1}(conf={c1:.2f}) vs {a2}(conf={c2:.2f}) → "
                     f"裁决: {final_action}(conf={final_conf:.2f})")
            return {
                "action": final_action,
                "confidence": final_conf,
                "reason": f"千问仲裁: {data.get('reason', '')}",
                "source": "qwen_exit_arbitration",
            }
        except Exception as e:
            log.warning(f"⚠️ 千问仲裁异常: {e}，降级为 hold")
            return {
                "action": "hold", "confidence": 0.0,
                "reason": f"投票分歧({a1} vs {a2})，千问仲裁异常，保守观望",
            }

    # ── 子方法：构造 prompt ────────────────────────────────────────────────

    def _build_simple_prompt(self, ind_15m: Dict, ind_1h: Dict, ind_4h: Dict,
                              news_data: Dict, fg_index: Dict, funding: Dict,
                              depth: Dict, pos_info: Dict,
                              key_levels: Optional[Dict] = None,
                              funding_history: Optional[List[Dict]] = None,
                              macro_context: str = "",
                              rag_warning: str = "",
                              market_sentiment: Optional[Dict] = None,
                              prev_market_mode: Optional[str] = None,
                              sentiment_alert: str = "",
                              fast_context: str = "",
                              # ── 预计算值（由 get_decision 顶部算好再传入，避免重复计算）──
                              _osc_market: Optional[str] = None,
                              _period_score: Optional[float] = None,
                              _score_desc: Optional[str] = None) -> tuple:
        price_now = ind_15m.get("price", 0)
        has_pos   = bool(pos_info.get("side"))
        _bb_upper = ind_15m.get("bb_upper", 0)
        _bb_lower = ind_15m.get("bb_lower", 0)
        _bb_price = price_now or 1
        _bb_w     = (_bb_upper - _bb_lower) / _bb_price if _bb_upper > 0 else 0

        # key levels
        levels_str = ""
        if key_levels and key_levels.get("_valid"):
            res = key_levels.get("resistances", [])[:3]
            sup = key_levels.get("supports", [])[:3]
            piv = key_levels.get("pivot", 0)
            def _lp(item):
                p = _price_of_level(item)
                cnt = item.get("count", 0) if isinstance(item, dict) else 0
                dist = abs(p - price_now) / price_now * 100 if price_now > 0 else 0
                suffix = f"x{cnt}" if cnt else ""
                return f"{p:.2f}({dist:.1f}%){suffix}"
            parts = []
            if res: parts.append("阻:" + "/".join(_lp(r) for r in res))
            if sup: parts.append("撑:" + "/".join(_lp(s) for s in sup))
            if piv: parts.append(f"Piv:{float(piv):.2f}")
            levels_str = " | ".join(parts)

        # market mode（预计算传入，避免重复）
        _market_mode = _osc_market if _osc_market else get_market_mode(ind_15m, _bb_price, prev_market_mode)
        mode_hint = ("顺势追击" if _market_mode == "趋势"
                     else "均值回归" if _market_mode == "震荡"
                     else "快进快出")

        # period score（预计算传入，避免重复）
        sc = _period_score if _period_score is not None else _calc_period_score(ind_15m, ind_1h, ind_4h)[0]
        score_desc = _score_desc if _score_desc else ("中性" if sc > 0.45 else "看空" if sc > 0.35 else "强烈看空")

        # position
        if has_pos:
            allowed_actions = ["hold", f"close_{pos_info['side']}", "adjust_sl_tp"]
            pos_str = (f"{pos_info['side']} 入:{pos_info['entry_price']:.2f} "
                       f"PnL:{pos_info.get('pnl_pct',0)*100:+.2f}% "
                       f"持:{pos_info.get('holding_minutes',0):.0f}m")
            pos_block = (f"持有{pos_info['side']}，浮盈{pos_info.get('pnl_pct',0)*100:+.2f}%。"
                         f"可选: hold / close_{pos_info['side']} / adjust_sl_tp")
        else:
            allowed_actions = ["hold", "open_long", "open_short"]
            pos_str = "空仓"
            pos_block = "当前空仓。可选: open_long / open_short / hold"

        # stop context
        stop_block = ""
        _lst = gs_get("last_stop_time")
        if _lst:
            _now = datetime.now(UTC)
            _lst_dt = _parse_dt(_lst)
            _min_ago = (_now - _lst_dt).total_seconds() / 60 if _lst_dt else 999
            if _min_ago < CFG.min_cooldown_after_loss * 3:
                _lsd = gs_get("last_stop_direction", "")
                _lsp = gs_get("last_stop_pnl_pct", 0)
                _lspx = gs_get("last_stop_price", 0)
                _pct = abs(price_now - _lspx) / _lspx * 100 if _lspx > 0 else 0
                stop_block = (f"近止损({_lsd} {_lsp*100:.1f}%) {_min_ago:.0f}min前。"
                              f"原价{_lspx:.2f}现{price_now:.2f}({_pct:.1f}%)。"
                              f"同向开仓需conf>=0.75。")

        # kline
        kline_str = build_kline_series(ind_15m, ind_1h, n_15m=8, n_1h=4)

        bw = depth.get('bid_wall_mult', 0)
        sw = depth.get('ask_wall_mult', 0)
        bw_d = depth.get('bid_wall_dist_pct', 0)
        sw_d = depth.get('ask_wall_dist_pct', 0)
        wall_str = ""
        if bw > 0: wall_str += f"买墙:{bw:.1f}x {bw_d*100:.1f}%距 "
        if sw > 0: wall_str += f"卖墙:{sw:.1f}x {sw_d*100:.1f}%距"

        user_prompt = (
            f"你是ETH量化交易顾问。根据以下指标判断交易动作。\n\n"
            f"[指标]\n"
            f"[15m] rsi={ind_15m.get('rsi',50):.1f} macd={ind_15m.get('macd_hist',0):.4f} "
            f"bb%={ind_15m.get('bb_pct',0.5):.2f} vol={ind_15m.get('vol_surge',1):.2f}x "
            f"trend={ind_15m.get('trend','?')} div={ind_15m.get('divergence','无')}\n"
            f"[1h]  rsi={ind_1h.get('rsi',50):.1f} macd={ind_1h.get('macd_hist',0):.4f} ema={ind_1h.get('ema_bull','?')}\n"
            f"[4h]  rsi={ind_4h.get('rsi',50):.1f} macd={ind_4h.get('macd_hist',0):.4f} ema={ind_4h.get('ema_bull','?')}\n"
            f"现价:{price_now:.2f} | 市场:{_market_mode}(BB={_bb_w*100:.1f}%) | 评分:{score_desc}({sc:.2f}) | {mode_hint}\n"
            f"VSpike={ind_15m.get('vol_surge',1):.1f}x | 恐贪={fg_index.get('value',50)} | 资金费={funding.get('funding_rate',0)*100:+.3f}%\n"
            f"{levels_str if levels_str else '关键位:暂无'}\n\n"
            f"[K线序列]{kline_str}\n\n"
            f"[盘口] 失衡={depth.get('imbalance',0):.2f} 斜率={depth.get('slope_ratio',1.0):.2f} {wall_str}\n\n"
            f"[持仓]{pos_str}\n"
            f"{stop_block}\n"
            f"{fast_context}\n"
            + (f"[历史案例]{rag_warning}\n\n" if rag_warning else "\n")
            + f"{pos_block}\n\n"
            f"[5条铁律]\n"
            f"1.方向中性：只看信号强度，不追涨杀跌。\n"
            f"2.止损必给：open必须同时给suggested_sl和suggested_tp。\n"
            f"3.关键位锚定：止损优先锚定最近支撑/阻力，再考虑ATR。\n"
            f"4.数据自洽：说超买需RSI>=70(震荡>=65)，背离需div=bullish/bearish。\n"
            f"5.趋势市顺势追击，震荡市均值回归。\n\n"
            f"只输出JSON:\n"
            f'{{"action":"","confidence":0.0~1.0,"suggested_sl":数值,"suggested_tp":数值,'
            f'"suggested_leverage":1~{CFG.max_leverage},"reason":"一句话","wait_seconds":0~300}}'
        )
        return user_prompt, allowed_actions


    def _build_reasoner_prompt(self, ind_15m: Dict, ind_1h: Dict, ind_4h: Dict,
                                funding: Dict, depth: Dict, pos_info: Dict,
                                prev_market_mode: Optional[str] = None,
                                fast_context: str = "",
                                # ── 预计算值（由 get_decision 顶部算好再传入）────────────────
                                _osc_market: Optional[str] = None) -> tuple:
        price_now = ind_15m.get("price", 0)
        has_pos   = bool(pos_info.get("side"))
        _bb_upper = ind_15m.get("bb_upper", 0)
        _bb_lower = ind_15m.get("bb_lower", 0)
        _bb_price = price_now or 1
        _bb_w     = (_bb_upper - _bb_lower) / _bb_price if _bb_upper > 0 else 0
        _market_mode = _osc_market if _osc_market else get_market_mode(ind_15m, _bb_price, prev_market_mode)

        if has_pos:
            allowed_actions = ["hold", f"close_{pos_info['side']}", "adjust_sl_tp"]
            pos_block = (f"持有{pos_info['side']}，浮盈{pos_info.get('pnl_pct',0)*100:+.2f}%，"
                         f"持仓{int(pos_info.get('holding_minutes',0))}分钟，"
                         f"止损{pos_info.get('current_sl',0):.2f}")
        else:
            allowed_actions = ["hold", "open_long", "open_short"]
            pos_block = "空仓"

        user_prompt = (
            f"你是ETH量化交易顾问。请仔细分析以下指标，决定交易动作。\n\n"
            f"[15m] rsi={ind_15m.get('rsi',50):.1f} macd={ind_15m.get('macd_hist',0):.4f} "
            f"bb%={ind_15m.get('bb_pct',0.5):.2f} vol={ind_15m.get('vol_surge',1):.2f}x "
            f"trend={ind_15m.get('trend','?')} div={ind_15m.get('divergence','无')} "
            f"adx={ind_15m.get('adx',25):.1f}\n"
            f"[1h]  rsi={ind_1h.get('rsi',50):.1f} macd={ind_1h.get('macd_hist',0):.4f} ema_bull={ind_1h.get('ema_bull','?')}\n"
            f"[4h]  rsi={ind_4h.get('rsi',50):.1f} macd={ind_4h.get('macd_hist',0):.4f} ema_bull={ind_4h.get('ema_bull','?')}\n"
            f"现价:{price_now:.2f} | 市场:{_market_mode}(BB={_bb_w*100:.1f}%) | VSpike={ind_15m.get('vol_surge',1):.1f}x\n"
            f"资金费={funding.get('funding_rate',0)*100:+.3f}% | 盘口失衡={depth.get('imbalance',0):.2f}\n"
            f"{fast_context}\n\n"
            f"[持仓]{pos_block}\n\n"
            f"[可选动作]{', '.join(allowed_actions)}\n\n"
            f"深度思考后输出JSON，不要任何解释文字:\n"
            f'{{"action":"","confidence":0.0~1.0,"suggested_sl":数值,"suggested_tp":数值,'
            f'"suggested_leverage":1~{CFG.max_leverage},"reason":"一句话","wait_seconds":0~300}}'
        )
        return user_prompt, allowed_actions


    # 兼容旧接口
    def _build_prompt(self, ind_15m: Dict, ind_1h: Dict, ind_4h: Dict,
                      news_data: Dict, fg_index: Dict, funding: Dict,
                      depth: Dict, pos_info: Dict,
                      key_levels: Optional[Dict] = None,
                      funding_history: Optional[List[Dict]] = None,
                      macro_context: str = "",
                      rag_warning: str = "",
                      market_sentiment: Optional[Dict] = None,
                      prev_market_mode: Optional[str] = None,
                      sentiment_alert: str = "",
                      fast_context: str = "") -> tuple:
        return self._build_simple_prompt(
            ind_15m, ind_1h, ind_4h,
            news_data, fg_index, funding, depth, pos_info,
            key_levels, funding_history, macro_context, rag_warning,
            market_sentiment, prev_market_mode, sentiment_alert, fast_context,
        )




    # ── 子方法：投票 ───────────────────────────────────────────────────────
    def _vote(self, r1: Dict, r2: Dict, for_exit_arbitration: bool = False) -> Dict:
        """
        双温度投票融合（r1=T0.25保守/Reasoner风格，r2=T0.70创意/Chat风格）。

        优先级规则（P0/P1）：
        1. 多空完全冲突（open_long vs open_short）→ 强制 hold，双模型打架不做任何方向偏倚
        2. 任意一方要求 hold/skip → 采纳保守结果（Reasoner 否决权）
        3. 置信度差 ≥ 0.2 → 采信高置信度方
        4. 置信度差 < 0.2 → 强制 hold，避免在模糊信号下强行决策

        ⚠️ SL/TP 处理：只取赢家完整点位，绝不融合。
        原因：止损位是两个独立的技术判断融合，悬在半空容易被插针扫损。
        """
        a1, a2 = r1.get("action"), r2.get("action")
        c1, c2 = r1.get("confidence", 0.0), r2.get("confidence", 0.0)
        conf_gap = abs(c1 - c2)

        # ── P0 Fix 1：多空完全冲突 → 强制熔断为 hold ────────────────────────
        if {a1, a2} == {"open_long", "open_short"}:
            log.warning(f"🚨 AI投票多空完全冲突（{a1} vs {a2}），强制熔断为 hold")
            result = {
                "action": "hold", "confidence": 0.0,
                "suggested_sl": 0, "suggested_tp": 0,
                "suggested_leverage": 1,
                "reason": f"双温度投票多空冲突({a1} vs {a2})，强制观望",
                "thought_process": "",
            }
            return self._record_vote(result, a1, a2, c1, c2, conf_gap), result

        # ── P1：Reasoner 否决权（保守方 hold/skip 拦截创意方开仓）────────────
        # 否决条件收紧：hold方置信度比open方高出≥0.10才生效
        # 避免Reasoner以更低或相近的置信度无条件压制Chat的开仓信号
        if a1 in ("hold", "skip") or a2 in ("hold", "skip"):
            hold_res  = r1 if a1 in ("hold", "skip") else r2
            open_res  = r2 if a1 in ("hold", "skip") else r1
            hold_conf = c1 if a1 in ("hold", "skip") else c2
            open_conf = c2 if a1 in ("hold", "skip") else c1
            veto_margin = hold_conf - open_conf
            if veto_margin >= 0.10:
                result = hold_res
                if result.get("confidence", 0) == 0.0:
                    result = {**result, "confidence": 0.5}
                log.info(f"⚖️ AI投票：Reasoner否决（{a1} vs {a2}），hold_conf={hold_conf:.2f}高出open_conf={open_conf:.2f} margin={veto_margin:.2f}≥0.10，采保守结果: hold")
            else:
                result = open_res
                log.info(f"⚖️ AI投票：否决条件不足（hold_conf={hold_conf:.2f} open_conf={open_conf:.2f} margin={veto_margin:.2f}<0.10），尊重开仓信号: {open_res.get('action')} conf={open_conf:.2f}")
            return self._record_vote(result, a1, a2, c1, c2, conf_gap), result

        # ── 共识：双方一致 → 取高置信度方（全套点位继承）────────────────────
        if a1 == a2:
            result = r1 if c1 >= c2 else r2
            if result.get("action") in ("hold", "skip") and result.get("confidence", 0) == 0.0:
                result = {**result, "confidence": 0.5}
            log.info(f"✅ AI投票共识: {a1} (conf={result.get('confidence', max(c1,c2)):.2f})")
            return self._record_vote(result, a1, a2, c1, c2, conf_gap), result

        # ── 置信度差 ≥ 0.2 → 采信高置信度方 ──────────────────────────────
        if conf_gap >= 0.2:
            result = r1 if c1 > c2 else r2
            log.info(f"⚖️ AI投票分歧({a1} vs {a2})，置信度差{conf_gap:.2f}，采信 {result['action']}(conf={max(c1,c2):.2f})")
            return self._record_vote(result, a1, a2, c1, c2, conf_gap), result

        # ── 持仓分歧仲裁标记（开仓/平仓通用）────────────────────────────────
        # 当双方分歧且置信度差 < 0.2 时，标记 need_exit_arbitration=True
        # 调用方（eth_trader.py）检测到该标记后会触发千问仲裁
        _exit_intents = {"close_long", "close_short", "close", "adjust_sl_tp"}
        _is_exit_dispute = for_exit_arbitration and (
            (a1 in _exit_intents and a2 in ("hold", "skip")) or
            (a2 in _exit_intents and a1 in ("hold", "skip")) or
            (a1 in _exit_intents and a2 in _exit_intents)
        )

        # ── 双方均为离场/保护意图 → 取保守动作（adjust_sl_tp 优先）────────────
        # 避免 close_short vs adjust_sl_tp 这类"方向一致、方式不同"的分歧被强制 hold
        _contradictory = {"close_long", "close_short"}.issubset({a1, a2})
        if a1 in _exit_intents and a2 in _exit_intents and not _contradictory:
            if "adjust_sl_tp" in (a1, a2):
                result = r1 if a1 == "adjust_sl_tp" else r2
            else:
                result = r1 if c1 >= c2 else r2
            if _is_exit_dispute:
                result["_need_exit_arbitration"] = True
                log.info(f"⚖️ AI投票：双方均为离场意图({a1} vs {a2})，差距{conf_gap:.2f}<0.2，取保守动作但标记仲裁: {result['action']}(conf={result.get('confidence',0):.2f})")
            else:
                log.info(f"⚖️ AI投票：双方均为离场意图({a1} vs {a2})，差距{conf_gap:.2f}<0.2，取保守动作: {result['action']}(conf={result.get('confidence',0):.2f})")
            return self._record_vote(result, a1, a2, c1, c2, conf_gap), result

        # ── 模糊信号 → 强制 hold ─────────────────────────────────────────
        log.info(f"🚫 AI投票分歧({a1} conf={c1:.2f} vs {a2} conf={c2:.2f})，差距<0.2，强制观望")
        result = {
            "action": "hold", "confidence": 0.0,
            "suggested_sl": 0, "suggested_tp": 0,
            "suggested_leverage": 1,
            "reason": f"双温度投票分歧({a1} vs {a2})，置信度差={conf_gap:.2f}<0.2",
            "thought_process": f"保守:{r1.get('reason','')} | 创意:{r2.get('reason','')}",
        }
        if _is_exit_dispute:
            result["_need_exit_arbitration"] = True
            result["_dispute_actions"] = [a1, a2]
            result["_dispute_confs"] = [c1, c2]
            log.info(f"📎 [{a1} vs {a2}] 投票分歧标记仲裁（hold vs close 模糊冲突）")
        return self._record_vote(result, a1, a2, c1, c2, conf_gap), result

    def _record_vote(self, result: Dict, a1: str, a2: str, c1: float, c2: float, conf_gap: float) -> Dict:
        """将投票结果写入结构化日志（JSON 格式），供回溯分析。"""
        try:
            decision_record = {
                "ts": datetime.now(UTC).isoformat(),
                "action": result.get("action"),
                "confidence": result.get("confidence"),
                "suggested_leverage": result.get("suggested_leverage"),
                "suggested_sl": result.get("suggested_sl"),
                "suggested_tp": result.get("suggested_tp"),
                "reason": (result.get("reason") or "")[:120],
                "pyramid_plan": result.get("pyramid_plan"),
                "vote_r1_action": a1, "vote_r1_conf": round(c1, 3),
                "vote_r2_action": a2, "vote_r2_conf": round(c2, 3),
                "conf_gap": round(conf_gap, 3),
            }
            log.info(f"[AI_DECISION] {json.dumps(decision_record, ensure_ascii=False)}")
        except Exception:
            pass
        return result

    # ── P2 辅助：持仓PnL计算 ──────────────────────────────────────────────
    def get_pnl_pct(self, pos_info: Dict, current_price: float) -> float:
        """计算当前持仓浮盈/浮亏百分比（正=盈利，负=亏损）。"""
        if not pos_info or not pos_info.get("side") or not pos_info.get("entry_price"):
            return 0.0
        if current_price <= 0:
            return 0.0
        entry = float(pos_info["entry_price"])
        if pos_info["side"] == "long":
            return (current_price - entry) / entry
        else:
            return (entry - current_price) / entry

    def get_roe_pct(self, pos_info: Dict, current_price: float) -> float:
        """计算权益回报率 = 价格变动 × 杠杆（含双边手续费估算）。"""
        fee_estimate = 0.0005   # 双边手续费估算（约 0.05%）
        pnl_pct = self.get_pnl_pct(pos_info, current_price)
        lev = float(pos_info.get("leverage", 1))
        return (pnl_pct - fee_estimate) * lev

    # ── P2 核心：Reasoner 按需触发判断 ─────────────────────────────────
    def _should_use_reasoner(self, pos_info: Dict, ind_15m: Dict,
                             prev_market_mode: str, current_price: float) -> bool:
        """
        判断本次决策是否需要 deepseek-reasoner 的强逻辑能力。
        极端场景（高波动/高风险/刚止损）→ 用 Reasoner 做一票否决兜底。
        普通场景 → 只用 deepseek-chat，省 token。
        """
        pnl = self.get_pnl_pct(pos_info, current_price)
        roe = self.get_roe_pct(pos_info, current_price)
        market = prev_market_mode or "震荡"
        ind = ind_15m or {}

        # 条件1：连续亏损 ≥ 阈值（需要强逻辑复盘）
        if gs_get("consecutive_losses", 0) >= CFG.reasoner_consec_loss_thresh:
            return True

        # 条件2：ROE ≤ 阈值（合约杠杆感知，无论几倍杠杆统一标准）
        if pos_info.get("side") and roe <= CFG.reasoner_roe_thresh:
            return True

        # 条件3：震荡激进市场（ADX低+BB窄，均值回归失效，需强逻辑判断）
        if market == "震荡激进":
            return True

        # 条件4：高波动（ATR 相对均值 > 阈值）
        atr_ratio = ind.get("atr_ratio", 1.0) if ind else 1.0
        if atr_ratio > CFG.reasoner_atr_ratio_thresh:
            return True

        # 条件5：止损后冷却期内（需 Reasoner 做归因）
        last_stop = gs_get("last_stop_time")
        if last_stop:
            try:
                _dt = _parse_dt(last_stop)
                if _dt and (datetime.now(UTC) - _dt).total_seconds() < CFG.reasoner_stop_cooldown_sec:
                    return True
            except Exception:
                pass

        # 默认：普通场景只用 deepseek-chat
        return False

    # ── P2 改造：_single_call 支持 model 参数 ─────────────────────────────────
    def _single_call(self, temperature: float, user_prompt: str,
                     allowed_actions: List[str],
                     model: str = "deepseek-chat") -> Dict:
        """执行单次 AI 调用并解析返回的决策。model 默认为 deepseek-chat。"""

        # ── deepseek-reasoner 专用路径（不接受 temperature，输出 <think>+JSON）────
        if model == "deepseek-reasoner":
            _msgs = [
                {"role": "system", "content": SYSTEM_PROMPT_TRADE},
                {"role": "user",   "content": user_prompt},
            ]
            _result = _call_reasoner_for_json(
                self.client, _msgs,
                max_tokens=4000,   # 限制 token 数，超出部分被截断（截断部分通常是重复推理，不影响结论）
                timeout=CFG.reasoner_timeout_seconds,  # 独立超时，reasoner 深度思考可能需要 3 分钟
            )
            if _result.get("action") not in allowed_actions:
                log.warning(f"AI({model})输出了不允许的动作 {_result.get('action')}，强制改为 hold")
                _result["action"] = "hold"
                _result["reason"] = f"[强制hold] {_result.get('reason', '')}"
            log.info(f"⚡ Reasoner 响应 | action={_result.get('action')} conf={_result.get('confidence',0):.2f}")
            return _result

        # ── deepseek-chat 标准路径（包含 429 限流处理）─────────────────────────
        try:
            r = self.client.chat.completions.create(
                model=model,
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TRADE},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                timeout=CFG.ai_timeout_seconds,
            )
        except Exception as e:
            err_str = str(e)
            # 检测 429 限流
            if "429" in err_str or "rate_limit" in err_str.lower():
                retry_after = CFG.ai_retry_delay
                if "retry-after" in err_str.lower():
                    try:
                        import re
                        match = re.search(r'retry-after["\']?\s*[:\=]\s*(\d+)', err_str, re.IGNORECASE)
                        retry_after = int(match.group(1)) if match else CFG.ai_retry_delay
                    except:
                        pass
                log.warning(f"⚠️ DeepSeek 触发 429 限流，冷却 {retry_after}s")
                raise  # 重新抛出，由上层 get_decision 的重试机制处理
            raise

        content = r.choices[0].message.content

        _thought_markers = ["<think>", "```json", "```"]
        thought, json_part = "", ""
        _found_marker = False
        for _marker in _thought_markers:
            if _marker in content:
                _parts = content.split(_marker, 1)
                _after = _parts[1]
                if _marker == "<think>" and "</think>" in _after:
                    thought = _after.split("</think>")[0].strip()
                    json_part = _after.split("\n</think>\n", 1)[1].strip()
                else:
                    thought = _after.split("\n", 1)[0] if "\n" in _after else _after
                    json_part = _after.split("\n", 1)[1] if "\n" in _after else _after
                _found_marker = True
                break
        if not _found_marker:
            _brace_idx = content.find("{")
            if _brace_idx > 0:
                thought = content[:_brace_idx].strip()
                json_part = content[_brace_idx:]
            else:
                json_part = content

        result = _parse_llm_json(json_part)
        result["thought_process"] = thought.strip()
        if isinstance(result.get("action"), str):
            result["action"] = result["action"].strip().lower()
        if result.get("action") not in allowed_actions:
            log.warning(f"AI({model})输出了不允许的动作 {result.get('action')}，强制改为 hold")
            result["action"] = "hold"
            result["reason"] = f"[强制hold] {result.get('reason', '')}"
        log.info(f"⚡ AI({model}) 响应 | action={result.get('action')} conf={result.get('confidence',0):.2f} "
                 f"| thought={len(thought)}chars")
        return result

    # ── 主方法：AI 决策入口 ─────────────────────────────────────────────────
    def get_decision(self, ind_15m: Dict, ind_1h: Dict, ind_4h: Dict,
                     news_data: Dict, fg_index: Dict, funding: Dict,
                     depth: Dict, pos_info: Dict,
                     key_levels: Optional[Dict] = None,
                     funding_history: Optional[List[Dict]] = None,
                     macro_context: str = "",
                     rag_warning: str = "",
                     market_sentiment: Optional[Dict] = None,
                     prev_market_mode: str = None,
                     sentiment_alert: str = "",
                     fast_context: str = "",
                     trend_alignment_score: float = 0.5,
                     trend_dir: str = "neutral") -> Dict:
        """
        AI 决策主流程（Prompt Diet 版本）：
        - deepseek-chat T=0.25：精简 prompt（~600 tokens）
        - deepseek-reasoner：极简 prompt（~300 tokens，不含规则）
        - hold 直接采纳，节省 token
        """
        last_attempt_err = None
        # ── 注入趋势对齐分数到 prompt ─────────────────────────────────────
        _trend_inj = f"趋势对齐分数：{trend_alignment_score:.2f}（1.0=极强顺势），方向：{trend_dir}。"
        fast_context = (_trend_inj + "\n\n" + (fast_context or "")).strip()

        # ── 预计算 market_mode 和 period_score（仅算一次，供两个 prompt 共用）────
        _price_bb = ind_15m.get("price", 0) or 1
        _osc_pre  = get_market_mode(ind_15m, _price_bb, prev_market_mode)
        _sc_pre, _desc_pre = _calc_period_score(ind_15m, ind_1h, ind_4h)

        for attempt in range(max(1, CFG.ai_max_retries)):
            try:
                # ── Step A: 精简 prompt + deepseek-chat T=0.25 先探路 ─────────
                simple_prompt, allowed_actions = self._build_simple_prompt(
                    ind_15m, ind_1h, ind_4h, news_data, fg_index, funding,
                    depth, pos_info, key_levels, funding_history,
                    macro_context, rag_warning, market_sentiment,
                    prev_market_mode, sentiment_alert,
                    fast_context=fast_context,
                    _osc_market=_osc_pre, _period_score=_sc_pre, _score_desc=_desc_pre,
                )
                r1 = self._single_call(0.25, simple_prompt, allowed_actions,
                                        model="deepseek-chat")
                a1 = r1.get("action", "hold")

                # hold 直接采纳
                if a1 in ("hold", "skip"):
                    log.info(f"🤖 Chat(T0.25): {a1} conf={r1.get('confidence', 0):.2f} — 直接采纳")
                    gs_set("ai_consecutive_timeout", 0)
                    return r1

                # ── Step B: 有交易信号 → 按需选择模型 ─────────────────────────
                use_reasoner = self._should_use_reasoner(
                    pos_info, ind_15m, prev_market_mode,
                    depth.get("mid_price", 0) if depth else 0
                )
                if use_reasoner:
                    # 极简 prompt 给 reasoner，让它充分思考
                    reasoner_prompt, _ = self._build_reasoner_prompt(
                        ind_15m, ind_1h, ind_4h,
                        funding, depth, pos_info,
                        prev_market_mode, fast_context,
                        _osc_market=_osc_pre,
                    )
                    model_name = "deepseek-reasoner"
                    model_temp = 0.35
                    log.info(f"🧠 极端场景 → {model_name}（极简 prompt）")
                    r2 = self._single_call(model_temp, reasoner_prompt, allowed_actions,
                                            model=model_name)
                else:
                    # 普通场景用精简 prompt + chat
                    model_name = "deepseek-chat"
                    model_temp = 0.70
                    log.info(f"📊 普通场景 → {model_name}(T{model_temp})")
                    r2 = self._single_call(model_temp, simple_prompt, allowed_actions,
                                            model=model_name)

                gs_set("ai_consecutive_timeout", 0)
                # ── 投票：持仓时标记分歧仲裁 ──────────────────────────────
                has_pos = bool(pos_info.get("side"))
                _, result = self._vote(r1, r2, for_exit_arbitration=has_pos)

                # ── 持仓分歧千问仲裁 ──────────────────────────────────────
                if has_pos and result.get("_need_exit_arbitration"):
                    _vs = ind_15m.get("vol_surge", 1.0)
                    _ob = depth.get("imbalance", 0.0) if hasattr(depth, "get") else 0.0
                    _a1, _a2 = r1.get("action", ""), r2.get("action", "")
                    _c1, _c2 = r1.get("confidence", 0.0), r2.get("confidence", 0.0)
                    arb_result = self._arbitrate_exit_dispute(
                        r1, r2, _a1, _a2, _c1, _c2, pos_info,
                        ind_15m, _osc_pre, _vs, _ob,
                    )
                    log.info(f"⚖️ 千问仲裁裁决持仓分歧: {arb_result['action']}(conf={arb_result['confidence']:.2f})")
                    result = arb_result
                return result

            except Exception as e:
                last_attempt_err = e
                err_str = str(e)
                is_reasoner_err = "reasoner" in err_str.lower()
                is_timeout = any(kw in err_str.lower()
                                 for kw in ("timeout", "timed out", "read timeout"))

                if is_reasoner_err and 'r1' in locals() and r1:
                    log.warning(f"⏱️ Reasoner 超时，采纳 r1={r1.get('action')} conf={r1.get('confidence', 0):.2f}")
                    gs_set("ai_consecutive_timeout", 0)
                    return r1

                if is_timeout:
                    n = gs_increment("ai_consecutive_timeout")
                    log.warning(f"⏱️ AI 超时（第{attempt+1}/{CFG.ai_max_retries}次）")
                    if n >= CFG.ai_timeout_alert_count:
                        log.error(f"🚨 AI 连续超时 {n} 次！")
                        _webhook("🚨 AI 持续超时", f"连续超时 {n} 次")
                    if n >= CFG.ai_timeout_alert_count * 2:
                        pause_until = (datetime.now(UTC) + timedelta(minutes=15)).isoformat()
                        gs_set("pause_until", pause_until)
                        log.critical(f"🛑 AI 连续超时 {n} 次，暂停开仓 15 分钟")
                else:
                    log.exception(f"AI 异常（第{attempt+1}次）: {e}")

                if attempt < CFG.ai_max_retries - 1:
                    time.sleep(CFG.ai_retry_delay)

        log.error(f"AI 决策失败，已重试 {CFG.ai_max_retries} 次")
        return {"action": "hold", "confidence": 0.0, "suggested_sl": 0,
                "suggested_tp": 0, "suggested_leverage": 1,
                "reason": f"AI Error after {CFG.ai_max_retries} retries"}


# ============================================================
# FastLaneModule - AI缓存与快速决策通道
# ============================================================
class FastLaneModule:
    """AI缓存与快速决策通道模块"""

    def __init__(self, eth_trader, signals: 'SignalsModule', ai):
        self.trader = eth_trader
        self.signals = signals
        self.ai = ai
        self._ai_decision_cache: Dict[str, Any] = {}
        self._last_decision_sig: Dict[str, str] = {}
        self._fast_signal_cache: Dict[str, tuple] = {}  # (action, conf, ts)

    def _is_redundant_fast_signal(self, sym: str, signal_key: str,
                                   action: str, conf: float,
                                   cooldown: int = 300,
                                   force_bypass: bool = False) -> bool:
        """
        快速信号防抖：同一信号源在 cooldown 秒内出现相同 action + conf 变化 <= 0.05 则拦截
        """
        if force_bypass:
            self._fast_signal_cache[f"{sym}_{signal_key}"] = (action, conf, time.monotonic())
            return False
        now = time.monotonic()
        cache_key = f"{sym}_{signal_key}"
        cached = self._fast_signal_cache.get(cache_key)
        if cached:
            prev_action, prev_conf, prev_ts = cached
            conf_improved = conf > prev_conf + 0.10
            if (prev_action == action
                    and abs(prev_conf - conf) <= 0.05
                    and not conf_improved
                    and (now - prev_ts) < cooldown):
                return True
        self._fast_signal_cache[cache_key] = (action, conf, now)
        return False

    def _should_skip_ai_request(self, symbol: str, ind_15m: Dict, ind_1h: Dict, current_price: float) -> bool:
        """
        静默拦截：持仓时若市场无明显变化且未超过静默期，则跳过AI请求。
        """
        return self.trader._should_skip_ai_request(symbol, ind_15m, ind_1h, current_price)

    def _trigger_ai_async_sym(self, symbol: str, *args, **kwargs):
        """触发异步AI决策"""
        return self.trader._trigger_ai_async_sym(symbol, *args, **kwargs)

    def _get_ai_decision(self, symbol: str = None) -> Optional[Dict]:
        return self.trader._get_ai_decision(symbol)

    def _clear_ai_cache(self, symbol: str = None):
        self.trader._clear_ai_cache(symbol)

    def _get_cache_ttl(self, market_mode: str = None) -> int:
        """缓存 TTL 动态化"""
        mode = market_mode if market_mode else (self.trader._market_mode if hasattr(self.trader, '_market_mode') else "趋势")
        if mode == "趋势":
            return 480
        elif mode == "震荡激进":
            return int(900 * 0.67)
        else:
            return 900
