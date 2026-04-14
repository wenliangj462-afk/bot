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
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Optional, Dict, List, Any, TYPE_CHECKING, Literal
from datetime import datetime, timezone, timedelta
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from market import SignalsModule

# ── 共享基础设施（common.py）──────────────────────────────────────────────
from common import (
    CFG, log, _webhook, _parse_dt, UTC,
    gs_get, gs_set, gs_increment, _call_reasoner,
    SYSTEM_PROMPT_TRADE,
    ai_cache_query_counter, ai_cache_hit_counter, ai_cache_miss_counter,
    _clean_json_text,
)
# ── 数据模型（core.py）──────────────────────────────────────────────────
# ── market_data 模块 ───────────────────────────────────────────────────
from market import build_kline_series, build_multi_tf_alignment, calc_key_levels, calc_composite_regime_score, classify_regime
from position_exec import _get_atr_quantile


# ── AI 响应 Pydantic 模型（Structured Output 客户端校验层）───────────────
class TradeDecisionModel(BaseModel):
    """
    AI 决策响应结构模型（仅用于客户端校验，不做强制构造）。
    使用 Optional + None 而非默认值，避免静默填充掩盖缺失字段。
    extra="forbid" 拒绝未知字段，防止 AI 混入乱字段。
    """
    action: Literal["open_long", "open_short", "close_long", "close_short", "hold", "adjust_sl_tp", "skip"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_sl: Optional[float] = None
    suggested_tp: Optional[float] = None
    suggested_leverage: Optional[int] = Field(None, ge=1, le=10)
    reason: str = Field(..., min_length=1)
    wait_seconds: Optional[int] = Field(None, ge=0, le=300)
    thought_process: Optional[str] = ""

    class Config:
        extra = "forbid"


def _validate_with_pydantic(raw_dict: Dict) -> Optional[TradeDecisionModel]:
    """尝试用 Pydantic 校验 AI 输出，失败返回 None（触发回退解析器）"""
    try:
        return TradeDecisionModel(**raw_dict)
    except ValidationError as e:
        log.debug(f"[Pydantic] 校验失败: {e}")
        return None


# ── 辅助函数（从主模块迁移）─────────────────────────────────────────────
def _price_of_level(item) -> float:
    """从支撑/阻力位对象提取价格"""
    if isinstance(item, dict):
        return float(item.get("price", item))
    return float(item)


def _call_reasoner_for_json(ai_client, messages: list, max_tokens: int = 2000,
                           timeout: int = 120) -> Dict:
    """调用 reasoner 并解析 JSON 响应（含 Pydantic 校验层）"""
    raw = _call_reasoner(ai_client, messages, max_tokens=max_tokens, timeout=timeout)
    result = _parse_llm_json(raw)

    validated = _validate_with_pydantic(result)
    if validated is None:
        log.debug(f"[Reasoner Pydantic] 校验失败，使用回退值。原始: {result}")
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
    else:
        result = validated.dict()
    return result


def get_market_mode(ind_15m: Dict, current_price: float,
                    prev_market_mode: str | None = None,
                    funding: Dict = None,
                    returns_30: np.ndarray | None = None) -> tuple[str, float]:
    """
    计算市场模式及复合评分。
    返回 (categorical_mode: str, regime_score: float)。
    - mode: 震荡激进 / 震荡 / 趋势（带滞后阈值防频繁切换）
    - score: 0~1，连续复合评分
    """
    if prev_market_mode is None:
        prev_market_mode = "趋势"
    score = calc_composite_regime_score(ind_15m, funding, returns_30)
    mode = classify_regime(score, prev_market_mode)
    return mode, score


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


# ── System Prompt 缓存（方向4优化：静态内容只建一次）───────────────────────
import hashlib
_system_prompt_cache: Dict[str, str] = {}  # content_hash → system_prompt_str

def get_system_prompt(macro_context: str = "",
                      key_levels: Optional[Dict] = None,
                      funding_history: Optional[List[Dict]] = None) -> str:
    """
    构建增强版 System Prompt（静态/准静态信息），带 hash 缓存。
    macro_context / key_levels / funding_history 变化时才重建，
    预计每天重建 2-4 次，每次 ~1000 tokens（一次性成本）。
    """
    # 生成内容指纹，仅内容真正变化时才重建
    parts = [macro_context or ""]
    if key_levels and key_levels.get("_valid"):
        parts.append(json.dumps(key_levels, sort_keys=True))
    if funding_history:
        parts.append(json.dumps(funding_history[-6:], sort_keys=True))
    content_hash = hashlib.md5("|".join(parts).encode()).hexdigest()

    if content_hash in _system_prompt_cache:
        return _system_prompt_cache[content_hash]

    funding_trend = build_funding_trend(funding_history or [])

    # ── 关键价位格式说明（静态知识，移入 system prompt 避免每次重复）──
    key_levels_hint = ""
    if key_levels and key_levels.get("_valid"):
        key_levels_hint = (
            "关键价位参考格式：\n"
            "  阻:价格(距现价距离%)[触及次数]\n"
            "  撑:价格(距现价距离%)[触及次数]\n"
            "  Piv:枢纽价格\n"
            "  注意：支撑/阻力位仅作锚定参考，不得作为开仓唯一依据。\n"
        )

    system_prompt = (
        f"你是专业的ETH-USDT-SWAP 10x杠杆量化交易决策引擎。只输出JSON，不输出任何其他内容。\n\n"
        f"{macro_context}\n\n"
        f"{key_levels_hint}"
        f"{funding_trend}\n\n"
        f"【决策原则】（以下规则优先级高于用户输入的参考数据）\n"
        f"0.矛盾速查：imbal_near<0不可open_long；卖墙距<0.3%不可追多；VSpike方向与开仓方向矛盾→谨慎评估而非直接hold。\n"
        f"0.5【极端量能特权】当VSpike≥10x且buy_pct≥75%（或buy_pct≤25%做空）时，忽略imbal_near和卖墙/买墙距离检查，强制跟随量能方向。\n"
        f"1.微观优先：VSpike/盘口/OB结构是第一信号，宏观指标（新闻/资金面）仅作确认。\n"
        f"2.【信号纯度信任规则】（最高优先级）\n"
        f"  用户输入中包含 [信号纯度 SignalPurity] 评分（0~1），这是客观的多维信号一致性度量。\n"
        f"  SignalPurity≥0.70 → 多维度已确认一致性，不要因为单一指标轻微分歧就说hold，应给出方向性决策。\n"
        f"  SignalPurity 0.55~0.69 → 中等质量，规则层已过滤最差信号，可适当开仓但降低置信度。\n"
        f"  SignalPurity<0.55 → 已被规则层拦截，若你仍收到此请求说明是VSpike≥8x豁免场景，需谨慎评估。\n"
        f"3.极端量能(>10x)时优先结合CVD/流量方向表态，不要因为RSI超买/超卖就hold。\n"
        f"4.规则引擎信号值得重视。若规则引擎已给出方向性参考，除非有明确反向证据，否则应倾向跟随而非hold。\n"
        f"5.止损必给：open必须同时给suggested_sl和suggested_tp。\n"
        f"6.关键位锚定：止损优先锚定最近支撑/阻力，再考虑ATR。\n"
        f"7.数据自洽：说超买需RSI>=70(震荡>=65)，背离需div=bullish/bearish。\n"
        f"8.趋势市顺势追击，震荡市均值回归。\n"
        f"9.新闻/资金面可作为辅助确认，但需与微观数据自洽。\n"
        # === 新增持仓平仓原则（核心）===
        f"10.持仓果断原则：已有明显浮盈(>1.5%)但微观信号转坏(VSpike反向、RSI≥68、买方支撑墙减弱、CVD净卖出)时，"
        f"**优先选择close锁定利润**，宁可早平也不让利润回吐。\n"
        f"11.持仓翻转原则：VSpike≥5.0x 且方向与当前持仓完全相反 → 立即评估close或反手，不得恋战。\n"
        f"12.亏损快刀原则：浮亏>0.5% 且微观数据不支持当前方向 → 果断close，不赌反弹。\n\n"
        f"【输出格式】只输出JSON:\n"
        f'{{"action":"","confidence":0.0~1.0,"suggested_sl":数值,"suggested_tp":数值,'
        f'"suggested_leverage":1~10,"reason":"一句话","wait_seconds":0~300}}'
    )

    _system_prompt_cache[content_hash] = system_prompt
    return system_prompt

# ── JSON 解析辅助 ─────────────────────────────────────────────────────────
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
        # 最后一次 cache clear 的时间戳（monotonic），供 worker 检测竞态
        self._cache_cleared_ts: float = 0.0
        # Drift snapshot
        self._last_price:     float = 0.0
        self._last_rsi_bkt:   Optional[str] = None
        self._last_bb_zone:   Optional[str] = None
        self._last_atr_ratio: Optional[float] = None
        # VSpike 缓存快照：用于检测量能方向翻转
        self._last_vspike_mult: float = 0.0
        self._last_vspike_dir:  str   = ""  # "买方主导"/"卖方主导"/""
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

    def get_cached(self, market_mode: str = "", atr_ratio: Optional[float] = None,
                   vspike_mult: float = 0.0, vspike_dir: str = "") -> Optional[Dict]:
        """返回有效缓存；过期则清除并返回最近一次有效决策（而非 None）
        atr_ratio: 当前ATR比率，用于动态调整TTL。如为None，则使用上次缓存时的ATR比率
        vspike_mult/vspike_dir: 当前 VSpike 状态，用于检测量能方向翻转"""
        ai_cache_query_counter.inc()
        with self._cache_lock:
            if self._cache is not None:
                # ── VSpike 方向翻转检测：量能方向完全相反且倍数 ≥6x → 清除缓存 ──
                if (vspike_mult >= 6.0
                        and self._last_vspike_mult >= 6.0
                        and self._last_vspike_dir
                        and vspike_dir
                        and self._last_vspike_dir != vspike_dir):
                    log.info(
                        f"🔥 VSpike方向翻转（缓存时={self._last_vspike_dir}{self._last_vspike_mult:.1f}x → "
                        f"当前={vspike_dir}{vspike_mult:.1f}x），强制清除缓存"
                    )
                    self._cache = None
                    self._cache_hash = ""
                    ai_cache_miss_counter.inc()
                    # 返回最近有效决策
                    if self._last_decision is not None:
                        return dict(self._last_decision)
                    return None

                # 使用传入的atr_ratio或上次缓存时的atr_ratio
                effective_atr = atr_ratio if atr_ratio is not None else self._last_atr_ratio
                ttl = self._get_ttl(market_mode, effective_atr)
                age = time.monotonic() - self._cache_ts
                if age > ttl:
                    log.info(f"AI缓存已过期（{age:.0f}s>{ttl}s，ATR比率={effective_atr:.2f}），强制刷新")
                    self._cache = None
                    self._cache_hash = ""
                    ai_cache_miss_counter.inc()
                else:
                    ai_cache_hit_counter.inc()
                    return self._cache
            # 缓存为空或已过期时，返回最近一次有效决策，避免异步调用间隙出现 "AI 未就绪"
            if self._last_decision is not None:
                ai_cache_miss_counter.inc()
                return dict(self._last_decision)
            ai_cache_miss_counter.inc()
            return None

    def set_cache(self, decision: Dict, input_sig: str,
                  price: float, rsi: float, macd: float,
                  rsi_bkt: str, bb_zone: str, atr_ratio: float = 1.0,
                  vspike_mult: float = 0.0, vspike_dir: str = ""):
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
            self._last_atr_ratio = atr_ratio
            # 记录 VSpike 快照
            self._last_vspike_mult = vspike_mult
            self._last_vspike_dir  = vspike_dir
            # 持久化最近一次有效AI决策
            self._last_decision  = dict(decision)

    def clear(self, clear_last: bool = False):
        with self._cache_lock:
            self._cache       = None
            self._cache_ts    = 0.0
            self._cache_hash  = ""
            self._last_rsi_bkt  = None
            self._last_bb_zone  = None
            self._last_vspike_mult = 0.0
            self._last_vspike_dir  = ""
            # VSpike 紧急事件：同时清除 _last_decision，防止 fallback 返回旧决策
            if clear_last:
                self._last_decision = None
                self._last_decision_ts = 0.0
            # 记录 clear 时间戳，供 worker 竞态检测
            self._cache_cleared_ts = time.monotonic()

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
    def _get_ttl(market_mode: str, atr_ratio: Optional[float] = None) -> int:
        """动态TTL调整：基于市场模式和ATR波动率自适应
        高波动（ATR比率>1.5）时缩短TTL，低波动（ATR比率<0.7）时延长TTL"""
        # 基础TTL
        if market_mode == "趋势":
            base_ttl = CFG.cache_ttl_trend
        elif market_mode == "震荡激进":
            base_ttl = int(CFG.cache_ttl_osc * 0.67)
        else:
            base_ttl = CFG.cache_ttl_osc

        # ATR动态调整
        if atr_ratio is not None:
            if atr_ratio > 1.5:  # 高波动
                adjusted_ttl = int(base_ttl * 0.5)  # 缩短50%
                log.debug(f"TTL动态调整: 高波动(atr_ratio={atr_ratio:.2f})，{base_ttl}s→{adjusted_ttl}s")
                return max(60, adjusted_ttl)  # 最低60秒
            elif atr_ratio < 0.7:  # 低波动
                adjusted_ttl = int(base_ttl * 1.5)  # 延长50%
                log.debug(f"TTL动态调整: 低波动(atr_ratio={atr_ratio:.2f})，{base_ttl}s→{adjusted_ttl}s")
                return adjusted_ttl
            else:  # 正常波动
                log.debug(f"TTL动态调整: 正常波动(atr_ratio={atr_ratio:.2f})，使用基础TTL {base_ttl}s")
                return base_ttl
        else:
            # 无ATR数据，使用基础TTL
            return base_ttl


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
                   # ── 新增：多周期共振 + 平仓原因 + 微观深度 ──
                   rsi_1h: float = 50.0,
                   rsi_4h: float = 50.0,
                   exit_reason: str = "",
                   imbal_near: float = 0.0,
                   bid_wall: str = "",
                   ask_wall: str = "",
    ) -> Optional[Dict]:
        if not self._qwen_available:
            return None

        side_tag = "做多(买)" if action == "open_long" else "做空(卖)"
        direction_tag = {
            "买方主导": "买盘强势（顺势做多）",
            "卖方主导": "卖盘强势（顺势做空）",
            "均衡": "多空均衡（谨慎）",
        }.get(depth_dir, "未知")

        # 多周期 RSI 共振
        _rsi_tf = f"RSI: 15m={rsi:.1f} | 1h={rsi_1h:.1f} | 4h={rsi_4h:.1f}"
        if rsi > 70 and rsi_1h > 70:
            _rsi_tf += " ⚠️超买共振"
        elif rsi < 30 and rsi_1h < 30:
            _rsi_tf += " ⚠️超卖共振"

        # 微观深度补充
        _micro_extra = ""
        if abs(imbal_near) > 0.20:
            _micro_extra += f"近场失衡:{imbal_near:+.3f} {'买方压制' if imbal_near > 0 else '卖方压制'} "
        if bid_wall:
            _micro_extra += f"买方墙:{bid_wall} "
        if ask_wall:
            _micro_extra += f"卖方墙:{ask_wall}"

        # 平仓原因
        _exit_ctx = ""
        if exit_reason:
            _exit_ctx = f"- 上次平仓原因：{exit_reason}\n"

        prompt = f"""你是 ETH 永续合约的交易顾问，只回答方向和置信度。
当前简明局势：
- 订单流方向：{direction_tag}
- VSpike 倍数：{vspike_mult:.1f}x（成交量突增）
- 订单簿失衡度：{ob_imbalance:.2f}（>0买强，<0卖强）
- {_rsi_tf}
- 市场模式：{market_mode}
- 当前价格：${price:.2f}
{_exit_ctx}- L1建议{reason}
{f'- 微观补充：{_micro_extra}' if _micro_extra else ''}
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
            except _json.JSONDecodeError as e:
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

        # 环境乘数：仅 ATR 修正，trend_score 已由 mode_coeff 惩罚，不再重复
        env_mult = (
            1.12 if _atr_ratio > 1.5 else 0.92 if _atr_ratio < 0.7 else 1.00
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
        ctx = context or {}

        # ── 硬门槛：AI 置信度不足 → 大幅压缩 AI 权重，但其他维度仍参与评分 ──
        # 修复：原代码直接 return 0.0，导致 VSpike/盘口/关键价位全部归零
        _ai_floor_penalty = ai_conf < 0.55
        if _ai_floor_penalty:
            # AI 不确信时，AI 部分按 0.40 系数计算（而非直接归零）
            ai_conf = ai_conf * 0.40 / 0.55  # 线性映射 0~0.55 → 0~0.40
        is_long = action == "open_long"
        _ai_w_mult = float(gs_get("ai_weight_mult", 1.0))
        ai_score = ai_conf * 85.0 * _ai_w_mult  # 从 65→85，AI 权重提升到 ~0.70
        tau = 4.5  # 从 8.0→4.5，让 2x~6x 区间分数梯度更明显

        # ── VSpike 极端量能保底通道：≥10x 且方向一致时，保底分 45 ──
        # 修复：从 15x 降至 10x，方向门槛从 80% 降至 70%，放在 AI 地板之后
        _extreme_vspike = vspike_mult >= 10.0 and (
            (is_long and ctx.get("buy_pct", 0.5) >= 0.70) or
            (not is_long and ctx.get("buy_pct", 0.5) <= 0.30)
        )
        if _extreme_vspike:
            return {
                "score":       45.0,
                "kelly_ratio": 0.45,
                "env_mult":    1.0,
                "risk_mult":   1.0,
                "components": {
                    "ai_raw":      round(ai_score, 2),
                    "spike":       vspike_mult,
                    "ob":          0.0,
                    "level":       0.0,
                    "rsi_penalty": 0.0,
                },
                "extreme_bypass": True,
            }

        # === VSpike 方向一致性检查（修复反向极端量能漏洞）===
        if vspike_mult >= 6.0:
            buy_pct = ctx.get("buy_pct", 0.5)
            # 强反向极端量能：直接倒扣
            if (is_long and buy_pct <= 0.25) or (not is_long and buy_pct >= 0.75):
                spike_score = -12.0
                log.warning(f"🛡️ [VSpike反向惩罚] 极端量能 {vspike_mult:.1f}x "
                            f"({'买方' if buy_pct >= 0.75 else '卖方'}主导) "
                            f"与 {'开多' if is_long else '开空'} 方向冲突，spike_score=-12")
            else:
                spike_score = 18.0 * math.tanh(vspike_mult / tau) + 15.0
                spike_score = min(spike_score, 30.0)
        else:
            spike_score = 18.0 * math.tanh(vspike_mult / tau) if vspike_mult > 0 else 0.0
        spike_score = max(spike_score, -15.0)  # 防止极端负分拖垮总分
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
# 辅助函数：构造 Funding Rate 行 + 关键价位距离行
# ============================================================

def _build_funding_rate_line(funding: Dict, funding_history: Optional[List[Dict]] = None) -> str:
    """构造资金费率行：当前费率 + 8h变化（覆盖8h，需funding_history≥32条）"""
    _fr = funding.get("funding_rate", 0)
    _fr_pct = _fr * 100
    _fg = gs_get("fear_greed", 50)
    _fg_val = _fg if isinstance(_fg, (int, float)) else 50

    _change_8h = None
    if funding_history and len(funding_history) >= 2:
        _oldest = funding_history[0].get("funding_rate", 0)
        _change_8h = (_fr - _oldest) * 100

    if _change_8h is not None:
        return f"恐贪={_fg_val} | 资金费={_fr_pct:+.3f}% (8h变化: {_change_8h:+.3f}%)"
    return f"恐贪={_fg_val} | 资金费={_fr_pct:+.3f}%"


def _build_key_level_distance(key_levels: Optional[Dict], price_now: float) -> str:
    """构造关键价位距离行：离最近支撑/阻力距离"""
    if not key_levels or not key_levels.get("_valid") or price_now <= 0:
        return "关键位:暂无"

    res = key_levels.get("resistances", [])[:3]
    sup = key_levels.get("supports", [])[:3]

    def _price_of(item):
        if isinstance(item, dict):
            return float(item.get("price", 0))
        return float(item) if item else 0

    parts = []
    # 最近阻力（高于现价）
    if res:
        _above_r = [_price_of(r) for r in res if _price_of(r) > price_now]
        if _above_r:
            _near_r = min(_above_r)
            _dist_r = (_near_r - price_now) / price_now * 100
            parts.append(f"阻力+{_dist_r:.1f}%")
    # 最近支撑（低于现价）
    if sup:
        _below_s = [_price_of(s) for s in sup if _price_of(s) < price_now]
        if _below_s:
            _near_s = max(_below_s)
            _dist_s = (price_now - _near_s) / price_now * 100
            parts.append(f"支撑-{_dist_s:.1f}%")

    if parts:
        return " | ".join(parts)
    return "关键位:暂无"


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
        _recent_wr = float(gs_get("last_24h_win_rate", 0.5))
        _consec_losses = gs_get("consecutive_losses", 0)

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
- 近期胜率：{_recent_wr*100:.1f}% | 连续亏损：{_consec_losses}次

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
    # ── 千问并行轻量调用（供主决策并行使用）────────────────────────────────────
    def _call_qwen_decision(self, simple_prompt: str, allowed_actions: list,
                            market_mode: str, action_hint: str,
                            cvd_delta: float = 0.0,
                            trend_score: float = 0.5,
                            key_level_dist: float = 0.0) -> Optional[Dict]:
        """
        千问轻量并行调用：复用 deepseek 的 prompt（~600 tokens），
        用独立 prompt 做第二意见投票。超时 15s 返回 None。
        """
        if not self._qwen_available:
            return None

        # 构建精简基础 prompt
        _base = simple_prompt.split('[5条铁律]')[0] if '[5条铁律]' in simple_prompt \
            else simple_prompt.split('[决策原则]')[0] if '[决策原则]' in simple_prompt \
            else simple_prompt

        # 补充微观深度数据（极端量能场景下帮助 Qwen 理解 CVD 净量含义）
        deep_ctx = ""
        if cvd_delta != 0:
            _dir_txt = "净买入" if cvd_delta > 0 else "净卖出"
            deep_ctx += f"CVD累计{_dir_txt}: {abs(cvd_delta):+.0f}张 | "
        if trend_score > 0:
            deep_ctx += f"趋势对齐分数: {trend_score:.2f} | "
        if key_level_dist > 0:
            deep_ctx += f"距最近关键位: {key_level_dist*100:.1f}%"

        prompt = f"""你是ETH量化交易顾问，请给出交易建议。
{_base}
[决策原则]
1.你只给第二意见，不重复主模型判断。
2.如果主模型建议{action_hint}但你认为数据不支持，可直接说hold。
3.输出严格JSON。
[补充微观深度数据]
{deep_ctx}
只输出JSON:
{{"action":"","confidence":0.0~1.0,"reason":"一句话"}}"""

        try:
            response = self._qwen_client.chat.completions.create(
                model=CFG.qwen_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
                timeout=CFG.qwen_timeout,
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            _qa = data.get("action", "hold")
            if _qa not in allowed_actions:
                _qa = "hold"
            return {
                "action": _qa,
                "confidence": float(data.get("confidence", 0.5)),
                "reason": f"[Qwen] {data.get('reason', '')}",
            }
        except Exception as e:
            log.debug(f"🔇 Qwen并行调用失败: {e}")
            return None

    def _fast_ai_direction_check(self, simple_prompt: str, allowed_actions: list, system_prompt: str) -> Dict:
        """
        【规则引擎专用快速方向校验】
        只调用 L1 DeepSeek-Chat，不走 L2 Qwen 和 L3 Reasoner
        严格控制在 25 秒内，专门用于「快速判断规则引擎建议的方向是否合理」
        """
        try:
            log.info("🔍 [Fast Direction Check] 只调用 L1 DeepSeek-Chat（快速方向校验 ≤25s）")
            result = self._single_call(
                temperature=0.25,
                user_prompt=simple_prompt,
                allowed_actions=allowed_actions,
                model="deepseek-chat",
                system_prompt=system_prompt
            )
            return result
        except Exception as e:
            log.warning(f"[Fast Direction Check] 异常: {e}")
            return {"action": "hold", "confidence": 0.5, "reason": "快速方向校验异常"}

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
                              aggressive_context: str = "",
                              # ── 预计算值（由 get_decision 顶部算好再传入，避免重复计算）──
                              _osc_market: Optional[str] = None,
                              _period_score: Optional[float] = None,
                              _score_desc: Optional[str] = None,
                              vs_status: Optional[Dict] = None,
                              # ── System Prompt（由 get_decision 构造后传入，静态内容只建一次）──
                              _system_prompt: Optional[str] = None) -> tuple:
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
        _market_mode = _osc_market if _osc_market else get_market_mode(ind_15m, _bb_price, prev_market_mode)[0]
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

        # ── AI 可读风险状态（仅连亏≥1+空仓时注入，让 AI 自己理解"不能急着开仓"）──
        _risk_block = ""
        _consec = gs_get("consecutive_losses", 0)
        if _consec >= 1 and not has_pos:
            _vs_r = vs_status or {}
            _vs_mult_r = _vs_r.get("mult", 0.0)
            _vs_dir_r = _vs_r.get("direction", "")
            _stop_dir_r = gs_get("last_stop_direction", "")
            _entry_px = gs_get("last_entry_price", 0.0)
            _hold_min = gs_get("last_holding_minutes", 0)
            _stop_pnl = gs_get("last_stop_pnl_pct", 0.0)
            _stop_px = gs_get("last_stop_price", 0.0)
            _exit_r = gs_get("last_exit_reason", "")[:50]
            if _stop_dir_r:
                _dir_name = "long" if _stop_dir_r == "long" else "short"
                _side_cn = "多" if _stop_dir_r == "long" else "空"
                _risk_block = (
                    f"[上一笔交易] {_side_cn} 开@{_entry_px:.2f} → {_exit_r or '止损'} → "
                    f"{_stop_pnl*100:.1f}%（持仓{_hold_min:.0f}分钟）\n"
                    f"[当前风险状态] 连亏{_consec}笔"
                    f"，当前VSpike={_vs_mult_r:.1f}x{_vs_dir_r}，"
                    f"若建议开仓，请在reason中说明为何当前场景与上一笔不同\n\n"
                )

        # kline
        kline_str = build_kline_series(ind_15m, ind_1h, ind_4h, n_15m=8, n_1h=6)

        # 多周期指标对齐摘要
        if ind_4h is not None:
            _tf_align = build_multi_tf_alignment(ind_15m, ind_1h, ind_4h)
            tf_align_block = f"\n{_tf_align}\n"
        else:
            tf_align_block = ""

        # ===== [微观] 区块：VSpike / 盘口微观结构 =====
        # 注：详细 CVD/吃单流量 数据已通过 fast_context 传入，此处仅提取 ind_15m/depth 可直接获取的微观信号
        _vs_mult = ind_15m.get('vol_surge', 1.0)
        _vs_is_spike = _vs_mult >= float(getattr(CFG, 'v_spike_mult_thresh', 2.9))
        _imbal = depth.get('imbalance', 0)
        _sr = depth.get('slope_ratio', 1.0)
        _bwm = depth.get('bid_wall_mult', 0.0)
        _awm = depth.get('ask_wall_mult', 0.0)
        _bwd = depth.get('bid_wall_dist_pct', 0.0)
        _awd = depth.get('ask_wall_dist_pct', 0.0)
        _imbal_n = depth.get('imbal_near', 0.0)

        micro_lines = []
        # VSpike
        micro_lines.append(f"VSpike={_vs_mult:.1f}x{' 🔥突增' if _vs_is_spike else ''}")
        # OB 微观结构
        _ob_wall_t = float(getattr(CFG, 'ob_wall_mult', 3.5))
        if _awm >= _ob_wall_t and _awd < 0.003:
            micro_lines.append(f"🧱 卖方冰山墙:距{_awd*100:.2f}% 强度{_awm:.1f}x 慎追多")
        if _bwm >= _ob_wall_t and _bwd < 0.003:
            micro_lines.append(f"🧱 买方支撑墙:距{_bwd*100:.2f}% 强度{_bwm:.1f}x 有利做多")
        if _sr >= 1.4:
            micro_lines.append(f"📈 OB买方斜率(ratio={_sr:.2f})")
        elif _sr <= 0.65:
            micro_lines.append(f"📉 OB卖方斜率(ratio={_sr:.2f})")
        if abs(_imbal_n) > 0.20:
            micro_lines.append(f"近场失衡:{_imbal_n:+.3f} {'买方压制' if _imbal_n > 0 else '卖方压制'}")
        # 盘口失衡速报
        micro_lines.append(f"盘口失衡:{_imbal:+.2f}")

        micro_block = "\n".join(micro_lines)

        bw = depth.get('bid_wall_mult', 0)
        sw = depth.get('ask_wall_mult', 0)
        bw_d = depth.get('bid_wall_dist_pct', 0)
        sw_d = depth.get('ask_wall_dist_pct', 0)
        wall_str = ""
        if bw > 0: wall_str += f"买墙:{bw:.1f}x {bw_d*100:.1f}%距 "
        if sw > 0: wall_str += f"卖墙:{sw:.1f}x {sw_d*100:.1f}%距"

        # ── 新闻面（有内容时才注入，避免空 prompt 占 token）──
        _news_block = ""
        _news_text = news_data.get("text", "") if news_data else ""
        _news_sent = news_data.get("sentiment", 0.0) if news_data else 0.0
        if _news_text and "无重大 ETH 新闻" not in _news_text:
            _sent_arrow = "↑偏多" if _news_sent > 0.1 else ("↓偏空" if _news_sent < -0.1 else "→中性")
            _news_block = f"[新闻面]\n{_news_text}\n综合情绪:{_sent_arrow}({ _news_sent:+.2f})\n\n"

        # ── 资金面（L/S 多空比 / OI 变化 / 主动买卖比）──
        _funding_block = ""
        _ms = market_sentiment or {}
        if _ms.get("_valid"):
            _ls = _ms.get("ls_ratio")
            # OI 双窗口（15min + 1h）
            _oi_15m = _ms.get("oi_change_15m")
            _oi_1h = _ms.get("oi_change_1h")
            _oi_15m_str = f"{_oi_15m:+.1f}%" if _oi_15m is not None else "暂无"
            _oi_1h_str = f"{_oi_1h:+.1f}%" if _oi_1h is not None else "暂无"
            # Taker Buy Ratio 百分比（5min短期 + 15min中期）
            _tbr_5m = _ms.get("taker_buy_5m")
            _tbr_15m = _ms.get("taker_buy_15m")
            _tbr_pct_5m = f"{_tbr_5m*100:.0f}%" if _tbr_5m is not None else "?"
            _tbr_pct_15m = f"{_tbr_15m*100:.0f}%" if _tbr_15m is not None else "?"
            _tbr_dir_5m = "买方进攻" if _tbr_5m is not None and _tbr_5m > 0.55 else ("卖方进攻" if _tbr_5m is not None and _tbr_5m < 0.45 else "均衡")
            _tbr_dir_15m = "买方主导" if _tbr_15m is not None and _tbr_15m > 0.55 else ("卖方主导" if _tbr_15m is not None and _tbr_15m < 0.45 else "均衡")

            _funding_block = f"[资金面] L/S多空比={_ls if _ls is not None else '?'}"
            _funding_block += f" | OI变化:15m={_oi_15m_str} 1h={_oi_1h_str}"
            _funding_block += f" | Taker买方占比:5m={_tbr_pct_5m}({_tbr_dir_5m}) 15m={_tbr_pct_15m}({_tbr_dir_15m})"
            _funding_block += "\n\n"

        # ── 市场情绪警报（仅在触发时注入，token 友好）──
        _sent_alert_block = ""
        if sentiment_alert:
            _sent_alert_block = f"[情绪警报]{sentiment_alert}\n\n"

        # ── CVD 累积成交量差（零额外计算，vs_status已有完整数据）──
        _cvd_delta = vs_status.get("cum_delta", 0) if vs_status else 0
        _cvd_trend = vs_status.get("delta_trend", "") if vs_status else ""
        if _cvd_delta != 0:
            _cvd_dir = "净买" if _cvd_delta > 0 else "净卖"
            _cvd_trend_label = f"({_cvd_trend})" if _cvd_trend and _cvd_trend != "数据积累中" else ""
            _cvd_line = f"CVD趋势: 近15min {_cvd_dir}{abs(_cvd_delta):.0f}张{_cvd_trend_label}"
        else:
            _cvd_line = "CVD趋势: 均衡(无明显方向)"

        user_prompt = (
            f"{aggressive_context}"
            f"{_risk_block}"
            f"[微观]{micro_block}\n\n"
            f"[盘口] 失衡={_imbal:.2f} 斜率={_sr:.2f} {wall_str}\n\n"
            f"[K线序列]{kline_str}\n{tf_align_block}\n"
            f"[CVD]{_cvd_line}\n\n"
            f"[持仓]{pos_str}\n"
            f"{stop_block}\n"
            f"{fast_context}\n"
            + (f"[历史案例]{rag_warning}\n\n" if rag_warning else "\n")
            + f"{pos_block}\n\n"
            + _news_block
            + _sent_alert_block
            + _funding_block
            + f"[指标](参考)\n"
            f"[15m] rsi={ind_15m.get('rsi',50):.1f} macd={ind_15m.get('macd_hist',0):.4f} "
            f"bb%={ind_15m.get('bb_pct',0.5):.2f} trend={ind_15m.get('trend','?')} div={ind_15m.get('divergence','无')}\n"
            f"[1h]  rsi={ind_1h.get('rsi',50):.1f} macd={ind_1h.get('macd_hist',0):.4f} ema={ind_1h.get('ema_bull','?')}\n"
            f"[4h]  rsi={ind_4h.get('rsi',50):.1f} macd={ind_4h.get('macd_hist',0):.4f} ema={ind_4h.get('ema_bull','?')}\n"
            f"现价:{price_now:.2f} | 市场:{_market_mode}(BB={_bb_w*100:.1f}%) | 评分:{score_desc}({sc:.2f}) | {mode_hint}\n"
            + _build_funding_rate_line(funding, funding_history) + "\n"
            + _build_key_level_distance(key_levels, price_now) + "\n"
            # 决策原则和输出格式已移至 System Prompt，避免每次重复发送 ~600 tokens
        )
        return user_prompt, allowed_actions


    def _build_reasoner_prompt(self, ind_15m: Dict, ind_1h: Dict, ind_4h: Dict,
                                funding: Dict, depth: Dict, pos_info: Dict,
                                prev_market_mode: Optional[str] = None,
                                fast_context: str = "",
                                # ── 预计算值（由 get_decision 顶部算好再传入）────────────────
                                _osc_market: Optional[str] = None,
                                vs_status: Optional[Dict] = None) -> tuple:
        price_now = ind_15m.get("price", 0)
        has_pos   = bool(pos_info.get("side"))
        _bb_upper = ind_15m.get("bb_upper", 0)
        _bb_lower = ind_15m.get("bb_lower", 0)
        _bb_price = price_now or 1
        _bb_w     = (_bb_upper - _bb_lower) / _bb_price if _bb_upper > 0 else 0
        _market_mode = _osc_market if _osc_market else get_market_mode(ind_15m, _bb_price, prev_market_mode)[0]

        if has_pos:
            allowed_actions = ["hold", f"close_{pos_info['side']}", "adjust_sl_tp"]
            pos_block = (f"持有{pos_info['side']}，浮盈{pos_info.get('pnl_pct',0)*100:+.2f}%，"
                         f"持仓{int(pos_info.get('holding_minutes',0))}分钟，"
                         f"止损{pos_info.get('current_sl',0):.2f}")
        else:
            allowed_actions = ["hold", "open_long", "open_short"]
            pos_block = "空仓"

        # ===== [微观] 区块：VSpike / 吃单流量 / CVD / OB结构 =====
        _vs_mult_r = ind_15m.get('vol_surge', 1.0)
        _vs_spike_r = _vs_mult_r >= float(getattr(CFG, 'v_spike_mult_thresh', 2.9))
        _imbal_r = depth.get('imbalance', 0)
        _sr_r = depth.get('slope_ratio', 1.0)
        _bwm_r = depth.get('bid_wall_mult', 0.0)
        _awm_r = depth.get('ask_wall_mult', 0.0)
        _bwd_r = depth.get('bid_wall_dist_pct', 0.0)
        _awd_r = depth.get('ask_wall_dist_pct', 0.0)
        _imbal_n_r = depth.get('imbal_near', 0.0)

        micro_lines_r = []
        # VSpike（带累计净量标签，消除 AI 对 cum_delta 正负含义的歧义）
        _cum_delta_r = (vs_status.get("cum_delta", 0) if vs_status else 0) or 0
        _cum_label_r = "净买" if _cum_delta_r > 0 else ("净卖" if _cum_delta_r < 0 else "均衡")
        _cum_str_r = f" {_cum_label_r}{abs(_cum_delta_r):.0f}张" if _cum_delta_r != 0 else ""
        micro_lines_r.append(f"VSpike={_vs_mult_r:.1f}x{' 🔥突增' if _vs_spike_r else ''}{_cum_str_r}")
        if _vs_spike_r:
            micro_lines_r.append(f"  方向:{ind_15m.get('trend','?')}(买占50%+?需结合CVD)")
        if _sr_r >= 1.4:
            micro_lines_r.append(f"📈 OB买方斜率(ratio={_sr_r:.2f})")
        elif _sr_r <= 0.65:
            micro_lines_r.append(f"📉 OB卖方斜率(ratio={_sr_r:.2f})")
        _ob_wall_t_r = float(getattr(CFG, 'ob_wall_mult', 3.5))
        if _awm_r >= _ob_wall_t_r and _awd_r < 0.003:
            micro_lines_r.append(f"🧱 卖方冰山墙:距{_awd_r*100:.2f}% 强度{_awm_r:.1f}x")
        if _bwm_r >= _ob_wall_t_r and _bwd_r < 0.003:
            micro_lines_r.append(f"🧱 买方支撑墙:距{_bwd_r*100:.2f}% 强度{_bwm_r:.1f}x")
        if abs(_imbal_n_r) > 0.20:
            micro_lines_r.append(f"近场失衡:{_imbal_n_r:+.3f}")
        micro_block_r = "\n".join(micro_lines_r)

        user_prompt = (
            f"你是ETH量化交易顾问。请仔细分析以下指标，决定交易动作。\n\n"
            f"[微观]{micro_block_r}\n\n"
            f"[盘口] 失衡={_imbal_r:.2f} 斜率={_sr_r:.2f}\n\n"
            f"[持仓]{pos_block}\n\n"
            f"{fast_context}\n\n"
            f"[指标](参考)\n"
            f"[15m] rsi={ind_15m.get('rsi',50):.1f} macd={ind_15m.get('macd_hist',0):.4f} "
            f"bb%={ind_15m.get('bb_pct',0.5):.2f} adx={ind_15m.get('adx',25):.1f}\n"
            f"[1h]  rsi={ind_1h.get('rsi',50):.1f} macd={ind_1h.get('macd_hist',0):.4f} ema_bull={ind_1h.get('ema_bull','?')}\n"
            f"[4h]  rsi={ind_4h.get('rsi',50):.1f} macd={ind_4h.get('macd_hist',0):.4f} ema_bull={ind_4h.get('ema_bull','?')}\n"
            f"现价:{price_now:.2f} | 市场:{_market_mode}(BB={_bb_w*100:.1f}%)\n"
            f"资金费={funding.get('funding_rate',0)*100:+.3f}%\n\n"
            f"[决策原则] 微观数据(VSpike/流量/OB)优先，宏观指标仅作确认。极端量能(>10x)时优先结合CVD/流量方向表态，不要因RSI超买/超卖就hold。规则引擎信号仅作参考，若数据不支持直接否决。深度思考后输出。\n\n"
            f"[可选动作]{', '.join(allowed_actions)}\n\n"
            f"深度思考后输出JSON，不要任何解释文字:\n"
            f'{{"action":"","confidence":0.0~1.0,"suggested_sl":数值,"suggested_tp":数值,'
            f'"suggested_leverage":1~{CFG.max_leverage},"reason":"一句话","wait_seconds":0~300}}'
        )
        return user_prompt, allowed_actions


    # 兼容旧接口


    # ── P2 改造：_single_call 支持 model 参数 ─────────────────────────────────
    def _single_call(self, temperature: float, user_prompt: str,
                     allowed_actions: List[str],
                     model: str = "deepseek-chat",
                     system_prompt: str = None) -> Dict:
        """执行单次 AI 调用并解析返回的决策。model 默认为 deepseek-chat。"""
        _sys = system_prompt if system_prompt else SYSTEM_PROMPT_TRADE

        # ── deepseek-reasoner 专用路径（不接受 temperature，输出 <think>+JSON）────
        if model == "deepseek-reasoner":
            _msgs = [
                {"role": "system", "content": _sys},
                {"role": "user",   "content": user_prompt},
            ]
            _result = _call_reasoner_for_json(
                self.client, _msgs,
                max_tokens=4000,   # 限制 token 数，超出部分被截断
                timeout=CFG.reasoner_timeout_seconds,
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
                    {"role": "system", "content": _sys},
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

        # ── L1 Pydantic 优先解析（严格 schema 校验）──────────────────────
        try:
            validated = TradeDecisionModel.model_validate_json(json_part)
            result = validated.model_dump()
        except Exception as e:
            log.debug(f"[L1 Pydantic] model_validate_json 失败，fallback 到老解析器: {e}")
            result = _parse_llm_json(json_part)
        result["thought_process"] = thought.strip()
        if result.get("confidence") is not None:
            result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

        # ── action 白名单校验（兜底防护）──────────────────────────────────
        if isinstance(result.get("action"), str):
            result["action"] = result["action"].strip().lower()
        if result.get("action") not in allowed_actions:
            log.warning(f"AI({model})输出了不允许的动作 {result.get('action')}，强制改为 hold")
            result["action"] = "hold"
            result["reason"] = f"[强制hold] {result.get('reason', '')}"
        log.info(f"⚡ AI({model}) 响应 | action={result.get('action')} conf={result.get('confidence',0):.2f} "
                 f"| thought={len(thought)}chars")
        return result

    # ── 主方法：AI 决策入口（三层模型架构）────────────────────────────────────
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
                     aggressive_context: str = "",
                     trend_alignment_score: float = 0.5,
                     trend_dir: str = "neutral",
                     vs_status: Optional[Dict] = None) -> Dict:
        """
        三层模型调用架构：
        L1 (DeepSeek-Chat)：主力快速决策，每次轮询无条件调用
        L2 (Qwen)：第二意见/仲裁，仅当 L1 开仓且 conf < 0.85 时调用
        L3 (Reasoner)：一票否决，仅当共识开仓且极端场景(VSpike≥5.0 / trending+ATR≥1.5)时调用
        """
        last_attempt_err = None
        _exit_intents = {"close_long", "close_short", "close", "adjust_sl_tp"}

        # ── 预计算 market_mode（供 prompt 和 L3 触发共用）───────────────────
        _price_bb = ind_15m.get("price", 0) or 1
        _osc_pre = get_market_mode(ind_15m, _price_bb, prev_market_mode)[0]

        # ── 构建增强版 System Prompt（静态内容，hash 缓存，每天重建 2-4 次）──
        _system_prompt = get_system_prompt(
            macro_context=macro_context,
            key_levels=key_levels,
            funding_history=funding_history,
        )

        # ── 注入趋势对齐分数到 fast_context ────────────────────────────────
        _trend_inj = f"趋势对齐分数：{trend_alignment_score:.2f}（1.0=极强顺势），方向：{trend_dir}。"
        fast_context = (_trend_inj + "\n\n" + (fast_context or "")).strip()

        for attempt in range(max(1, CFG.ai_max_retries)):
            try:
                # ═══════════════════════════════════════════════════════
                # L1: DeepSeek-Chat（主力快速决策，每次无条件调用）
                # ═══════════════════════════════════════════════════════
                simple_prompt, allowed_actions = self._build_simple_prompt(
                    ind_15m, ind_1h, ind_4h, news_data, fg_index, funding,
                    depth, pos_info, key_levels, funding_history,
                    macro_context, rag_warning, market_sentiment,
                    prev_market_mode, sentiment_alert,
                    fast_context=fast_context,
                    aggressive_context=aggressive_context,
                    _osc_market=_osc_pre,
                    vs_status=vs_status,
                    _system_prompt=_system_prompt,
                )
                chat_temp = 0.35 if _osc_pre in ("震荡", "震荡激进") else 0.25
                log.info(f"📊 [{CFG.symbol}] L1 DeepSeek-Chat T={chat_temp}（{_osc_pre}）")
                r1 = self._single_call(chat_temp, simple_prompt, allowed_actions,
                                        model="deepseek-chat",
                                        system_prompt=_system_prompt)
                l1_action = r1.get("action", "hold")
                l1_conf = r1.get("confidence", 0.0)
                log.info(f"L1 原始决策: {l1_action} conf={l1_conf:.2f}")

                # ── 统一 L2 Qwen 触发判断 ──
                _has_strong_signal = any(kw in (fast_context or "") for kw in (
                    "规则引擎参考信号", "打板信号", "快速决策", "🔥 秒级成交量突增",
                ))
                _has_grade_signal = any(kw in (fast_context or "") for kw in ("S级", "A级"))
                _need_l2 = (
                    (l1_action not in ("hold", "skip") and l1_conf < 0.80)
                    or (l1_action in ("hold", "skip") and (_has_strong_signal or _has_grade_signal))
                    or (_osc_pre == "趋势" and l1_action in ("hold", "skip") and l1_conf < 0.85)
                )

                if not _need_l2:
                    if l1_action in ("hold", "skip"):
                        log.info(f"✅ L1 决定 {l1_action}，直接采纳，跳过 L2/L3")
                    else:
                        log.info(f"✅ L1 高置信度({l1_conf:.2f}≥0.80)，跳过 L2")
                    gs_set("ai_consecutive_timeout", 0)
                    return r1

                # L2 触发原因
                if l1_action in ("hold", "skip") and (_has_strong_signal or _has_grade_signal):
                    _sig_label = "S/A级" if _has_grade_signal else "规则/打板"
                    log.info(f"⚠️ L1 决定 {l1_action}，但 fast_context 存在{_sig_label}强信号，转 L2 Qwen 仲裁")
                elif l1_action in ("hold", "skip") and _osc_pre == "趋势":
                    log.info(f"⚠️ L1 决定 {l1_action}，但当前为趋势市，调 Qwen 补漏检查")
                else:
                    log.info(f"📝 L1 conf={l1_conf:.2f}<0.80，调用 L2 Qwen 仲裁")

                # ── close/adjust → 逃生通道直接执行，不等 L2/L3 ──
                if l1_action in _exit_intents:
                    log.info(f"✅ L1 决定 {l1_action}(conf={l1_conf:.2f})，逃生通道直接执行")
                    gs_set("ai_consecutive_timeout", 0)
                    return r1

                # ═══════════════════════════════════════════════════════
                # L2: Qwen 仲裁（持仓 close 且 L1 conf≥0.68 → 直接放行，不等 L2）
                # ═══════════════════════════════════════════════════════
                final_action = l1_action
                final_conf = l1_conf
                final_reason = r1.get("reason", "")

                _is_close_l1 = l1_action in ("close_short", "close_long", "close")
                if _is_close_l1 and l1_conf >= 0.68:
                    log.info(f"✅ L1 close 置信度达标 ({l1_action} conf={l1_conf:.2f}≥0.68)，跳过 L2 仲裁，直接执行")
                    gs_set("ai_consecutive_timeout", 0)
                    return r1

                qwen_result = None
                # Qwen 内部已 try/except 返回 None，最多重试2次
                for _qwen_attempt in range(2):
                    qwen_result = self._call_qwen_decision(
                        simple_prompt, allowed_actions, _osc_pre, l1_action,
                        cvd_delta=(vs_status.get("cum_delta", 0.0) if vs_status else 0.0),
                        trend_score=trend_alignment_score,
                    )
                    if qwen_result:
                        break
                    log.debug(f"Qwen 未返回，第{_qwen_attempt+1}/2次重试")

                if qwen_result:
                    q_action = qwen_result.get("action", "hold")
                    q_conf = qwen_result.get("confidence", 0.5)
                    log.info(f"🔵 L2 Qwen: {q_action} conf={q_conf:.2f}")

                    # ── 强否决：Qwen 方向相反且 conf ≥ 0.75 ──
                    if q_action != l1_action and q_conf >= 0.82:
                        log.warning(
                            f"🚫 Qwen 强否决：L1={l1_action}(conf={l1_conf:.2f}) "
                            f"vs Qwen={q_action}(conf={q_conf:.2f}≥0.75)，采纳 Qwen"
                        )
                        final_action = "hold"
                        final_conf = q_conf
                        final_reason = f"[Qwen否决L1] {qwen_result.get('reason', '')}"
                        r1["action"] = "hold"
                        r1["confidence"] = q_conf
                        r1["reason"] = final_reason
                    # ── 方向相同 → 共识，取平均置信度 ──
                    elif q_action == l1_action:
                        final_conf = (l1_conf + q_conf) / 2
                        final_reason = f"{r1.get('reason','')} [Qwen:{qwen_result.get('reason','')}]"
                        log.info(f"✅ L1+L2 共识: {l1_action} conf={final_conf:.2f}")
                        r1["confidence"] = round(final_conf, 3)
                        r1["reason"] = final_reason
                    # ── 方向不同但未达强否决 → 分歧处理 ──
                    else:
                        # ── 趋势市优先通道：Qwen 明确方向性信号且 conf≥0.60 → 采用 Qwen 方向 ──
                        if (_osc_pre == "趋势" and q_action != "hold"
                                and q_conf >= 0.60):
                            final_conf = q_conf
                            final_reason = (
                                f"{r1.get('reason','')} "
                                f"[趋势市采纳Qwen方向:{q_action} conf={q_conf:.2f}]"
                            )
                            log.info(
                                f"⚖️ L1+L2 分歧（趋势市采纳Qwen）：L1={l1_action}(conf={l1_conf:.2f}) "
                                f"vs Qwen={q_action}(conf={q_conf:.2f}≥0.60)，"
                                f"采用 Qwen 方向 conf={final_conf:.2f}"
                            )
                            r1["action"] = q_action
                            r1["confidence"] = round(final_conf, 3)
                            r1["reason"] = final_reason
                        else:
                            # 非趋势市或 Qwen 方向性不足 → 保留 L1，按信心差距动态降权
                            _conf_gap = abs(l1_conf - q_conf)
                            if l1_conf > q_conf:
                                # L1 更自信：信心差距越大，降权越小（Qwen 信心不足，分歧权重低）
                                final_conf = l1_conf * max(0.78, 1.0 - _conf_gap * 0.5)
                            else:
                                # Qwen 更自信但未达强否决线：较大降权
                                final_conf = l1_conf * 0.72
                            final_reason = f"{r1.get('reason','')} [Qwen分歧:{q_action} conf={q_conf:.2f}]"
                            log.info(
                                f"⚖️ L1+L2 弱分歧：L1={l1_action}(conf={l1_conf:.2f}) "
                                f"vs Qwen={q_action}(conf={q_conf:.2f})，"
                                f"保留 L1 动态降权 conf={final_conf:.2f}"
                            )
                            r1["confidence"] = round(final_conf, 3)
                            r1["reason"] = final_reason

                    r1["_qwen_action"] = q_action
                    r1["_qwen_conf"] = q_conf
                else:
                    log.info(f"⚠️ Qwen 未返回，仅用 L1: {l1_action} conf={l1_conf:.2f}")

                # ═══════════════════════════════════════════════════════
                # L3: Reasoner 一票否决（极端场景 + per-symbol 冷却）
                # ═══════════════════════════════════════════════════════
                use_reasoner = False
                _sym_key = f"reasoner_usage_{CFG.symbol}"
                _reasoner_stats = gs_get(_sym_key, {"count": 0, "window_start": 0.0})
                _now_s = time.monotonic()
                _window = 15 * 60  # 15分钟滚动窗口

                if _now_s - _reasoner_stats["window_start"] > _window:
                    _reasoner_stats = {"count": 0, "window_start": _now_s}

                # ── 提前初始化 VSpike 变量（供 L3 触发判断 + 后续方向守卫共用）──
                _vs = vs_status or {}
                _vs_mult = _vs.get("mult", 0.0)
                _vs_dir = _vs.get("direction", "")
                _vs_buy_pct = _vs.get("buy_pct", 0.5)

                # 共识 hold → 不需要 Reasoner
                if r1.get("action") in ("hold", "skip"):
                    use_reasoner = False
                elif _reasoner_stats["count"] >= 2:
                    log.info(f"⏳ Reasoner cooldown（{CFG.symbol}，15min内已达2次上限）")
                    use_reasoner = False
                else:
                    # 【持仓场景跳过 L3】持仓翻转需要果断平仓，不是深度思考
                    if pos_info.get("side"):
                        use_reasoner = False
                        log.debug(
                            f"⏭️ [{CFG.symbol}] 持仓场景，跳过 L3 Reasoner（避免超时拖延平仓）"
                        )
                    else:
                        _atr_ratio = ind_15m.get("atr_ratio", 1.0) if ind_15m else 1.0
                        _regime = ind_15m.get("regime_score", 0.5) if ind_15m else 0.5

                        # 触发条件1：VSpike ≥ 10.0 + 极强趋势（regime < 0.20）
                        if _vs_mult >= 10.0 and _regime < 0.20:
                            use_reasoner = True
                            _reasoner_stats["count"] += 1
                            log.info(f"🔥 L3 触发（黑天鹅防护）：VSpike={_vs_mult:.1f}x + Regime={_regime:.2f}<0.20")

                        # 触发条件2：趋势市 + ATR ≥ 2.8
                        elif _osc_pre == "趋势" and _atr_ratio >= 2.8:
                            use_reasoner = True
                            _reasoner_stats["count"] += 1
                            log.info(f"🔥 L3 触发（黑天鹅防护）：趋势市 + ATR={_atr_ratio:.2f} ≥ 2.8")

                        gs_set(_sym_key, _reasoner_stats)

                if use_reasoner:
                    reasoner_prompt, _ = self._build_reasoner_prompt(
                        ind_15m, ind_1h, ind_4h,
                        funding, depth, pos_info,
                        prev_market_mode, fast_context,
                        _osc_market=_osc_pre,
                        vs_status=vs_status,
                    )
                    log.info(f"🧠 L3 deepseek-reasoner 一票否决")
                    try:
                        r3 = self._single_call(0.35, reasoner_prompt, allowed_actions,
                                                model="deepseek-reasoner")
                        r3_action = r3.get("action", "hold")
                        r3_conf = r3.get("confidence", 0.5)
                        log.info(f"L3 Reasoner: {r3_action} conf={r3_conf:.2f}")

                        # 一票否决：Reasoner 方向相反且 conf ≥ 0.65
                        if r3_action != final_action and r3_conf >= 0.65:
                            log.warning(
                                f"🚫 Reasoner 一票否决：L1/L2={final_action}(conf={final_conf:.2f}) "
                                f"vs Reasoner={r3_action}(conf={r3_conf:.2f}≥0.65)，改为 hold"
                            )
                            r1["action"] = "hold"
                            r1["confidence"] = r3_conf
                            r1["reason"] = f"[Reasoner否决] {r3.get('reason', '')}"
                        else:
                            log.info(f"✅ Reasoner 未否决，维持 {final_action} conf={final_conf:.2f}")
                            r1["reason"] = f"{r1.get('reason','')} [Reasoner:{r3.get('reason','')}]"
                        r1["_reasoner_action"] = r3_action
                        r1["_reasoner_conf"] = r3_conf
                    except Exception as e:
                        log.warning(f"⚠️ L3 Reasoner 调用失败: {e}，跳过一票否决，维持 {final_action}")
                        r1["reason"] = f"{r1.get('reason','')} [Reasoner:调用失败]"
                    result = r1.copy()
                else:
                    result = r1.copy()

                gs_set("ai_consecutive_timeout", 0)

                # ── VSpike 方向守卫：极端反向量能硬拦截 ──
                _vs_dir = _vs.get("direction", "")
                _vs_buy_pct = _vs.get("buy_pct", 0.5)
                _vs_guard_thresh = 15.0
                _is_against = (
                    (result["action"] == "open_long" and "卖方主导" in _vs_dir and _vs_buy_pct < 0.25) or
                    (result["action"] == "open_short" and "买方主导" in _vs_dir and _vs_buy_pct > 0.75)
                )
                if _vs_mult >= _vs_guard_thresh and _is_against:
                    _orig = result.get("action", "")
                    result["action"] = "hold"
                    result["confidence"] = 0.0
                    result["reason"] = (result.get("reason", "") +
                        f" | ⚠️ VSpike方向守卫：{_vs_mult:.1f}x {_vs_dir}(buy_pct={_vs_buy_pct:.2f})，否决{_orig}")
                    log.warning(
                        f"🛡️ VSpike方向守卫拦截：{_vs_mult:.1f}x {_vs_dir} "
                        f"buy_pct={_vs_buy_pct:.2f}，否决{_orig}"
                    )

                return result

            except Exception as e:
                last_attempt_err = e
                err_str = str(e)
                is_timeout = any(kw in err_str.lower()
                                 for kw in ("timeout", "timed out", "read timeout"))

                # 任何模型超时 → 智能 fallback（按 L1 意图分类处理）
                if 'r1' in locals() and r1:
                    _l1_a = r1.get("action", "hold")
                    _l1_c = r1.get("confidence", 0.0)
                    if _l1_a in ("close_short", "close_long", "close"):
                        log.warning(
                            f"⏱️ L2超时，但 L1 原本想 close ({_l1_a} conf={_l1_c:.2f})，"
                            f"降级执行 close（conf={max(0.55, _l1_c*0.85):.2f}）"
                        )
                        r1["confidence"] = max(0.55, _l1_c * 0.85)
                        gs_set("ai_consecutive_timeout", 0)
                        return r1
                    elif _l1_a in ("open_long", "open_short"):
                        log.warning(f"⏱️ L2超时，L1 想开仓 → 安全 fallback hold")
                        r1["action"] = "hold"
                        r1["confidence"] = 0.0
                        r1["reason"] = f"[L2超时安全降级] {r1.get('reason','')}"
                        gs_set("ai_consecutive_timeout", 0)
                        return r1
                    else:
                        log.warning(f"⏱️ 模型调用超时/异常，采纳 r1 hold")
                        r1["action"] = "hold"
                        r1["confidence"] = 0.0
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

    def _get_ai_decision(self, symbol: str = None, atr_ratio: float = None) -> Optional[Dict]:
        return self.trader._get_ai_decision(symbol, atr_ratio=atr_ratio)

    def _clear_ai_cache(self, symbol: str = None, clear_last: bool = False):
        self.trader._clear_ai_cache(symbol, clear_last=clear_last)

    def _get_cache_ttl(self, market_mode: str = None) -> int:
        """缓存 TTL 动态化"""
        mode = market_mode if market_mode else (self.trader._market_mode if hasattr(self.trader, '_market_mode') else "趋势")
        if mode == "趋势":
            return 480
        elif mode == "震荡激进":
            return int(900 * 0.67)
        else:
            return 900
