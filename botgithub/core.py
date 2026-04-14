# ============================================================
# core.py — 数据模型层
# Position、TradingState、PositionIntent、EventBus、全局状态
# ============================================================
import os, time, threading, json, logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum

_log = logging.getLogger("ETH_Quant_V6.0")

# ============================================================
# 交易状态枚举
# ============================================================
class TradingState(Enum):
    IDLE     = "idle"
    SCANNING = "scanning"
    HOLDING  = "holding"
    EXITING  = "exiting"

class TradingEvent(Enum):
    PRICE_SPIKE   = "price_spike"
    SIGNAL_DETECTED = "signal_detected"
    ORDER_FILLED   = "order_filled"
    ORDER_FAILED   = "order_failed"
    STOP_LOSS      = "stop_loss"
    TAKE_PROFIT    = "take_profit"

UTC = timezone.utc

# ============================================================
# 全局状态（GLOBAL_STATE + 专用锁 + 操作函数）
# ============================================================
GLOBAL_STATE: Dict = {
    "start_balance":           0.0,
    "last_reset_date":         datetime.now(UTC).date().isoformat(),
    "daily_locked":            False,
    "consecutive_losses":      0,
    "pause_until":             None,
    # 止损上下文（供 AI 决策参考，防止扫损后追单）
    "last_stop_time":          None,
    "last_stop_direction":     None,
    "last_stop_pnl_pct":      0.0,
    "last_stop_reason":        None,
    "last_stop_market_mode":   None,
    "last_stop_price":         0.0,
    "last_adjust_time":        None,
    "last_state_sync":         None,
    "current_trade_id":        None,
    "last_watchdog":           None,
    "consecutive_slippage":    0,
    "ai_consecutive_timeout":  0,
    "today_realized_pnl":      0.0,
    "partial_tp_triggered":     False,
    "last_24h_win_rate":       0.5,
    "last_risk_update":        None,
    "today_opened_risk":       0.0,
    "dd_kelly_mult":           1.0,
}
_gs_lock = threading.Lock()

def gs_get(key: str, default=None):
    with _gs_lock:
        return GLOBAL_STATE.get(key, default)

def gs_set(key: str, value):
    with _gs_lock:
        GLOBAL_STATE[key] = value

def gs_update(d: dict):
    with _gs_lock:
        GLOBAL_STATE.update(d)

def gs_increment(key: str, default: int = 0) -> int:
    with _gs_lock:
        GLOBAL_STATE[key] = GLOBAL_STATE.get(key, default) + 1
        return GLOBAL_STATE[key]

def gs_add(key: str, value: float) -> float:
    with _gs_lock:
        GLOBAL_STATE[key] = GLOBAL_STATE.get(key, 0.0) + value
        return GLOBAL_STATE[key]

# ============================================================
# 持仓数据结构
# ============================================================
@dataclass
class Position:
    side:             Optional[str]      = None
    entry_price:      float              = 0.0
    size:             float              = 0.0
    leverage:         int                = 1
    open_time:        Optional[datetime] = None
    last_open_time:   Optional[datetime] = None
    peak_price:       float              = 0.0
    trailing_active:  bool               = False
    pending_ord_id:   str                = ""
    partial_filled:   float              = 0.0
    liq_price:        float              = 0.0
    stop_loss:        float              = 0.0
    take_profit:      float              = 0.0
    trade_id:         Optional[int]      = None
    moved_stop:               bool    = False
    breakeven_triggered:       bool    = False
    partial_tp_triggered:      bool    = False
    partial_tp_2_5R_triggered: bool    = False
    trailing_dist_atr_mult:     Optional[float] = None
    pyramid_plan:     Optional[dict] = None
    pyramid_count:    int            = 0
    initial_size:     float          = 0.0
    initial_risk_usd: float          = 0.0
    entry_market_mode:  Optional[str]  = None
    entry_rsi:          float          = 0.0
    entry_bb_pct:       float          = 0.0
    entry_atr_pct:      float          = 0.0
    ai_confidence:      float          = 0.0
    ai_reason:          Optional[str]  = None
    ai_conf_at_open:    float          = 0.0
    sl_tp_algo_ids:     list           = field(default_factory=list)
    # VSpike 趋势捕捉模式（极端量能与持仓方向一致时激活，降低trailing阈值+拉宽TP）
    trend_capture_ts:   float          = 0.0   # 激活时间(monotonic)，0=未激活
    trend_capture_mult: float          = 0.0   # 触发的VSpike倍率
    exit_rsi:          float          = 0.0
    exit_bb_pct:       float          = 0.0
    exit_atr_pct:      float          = 0.0
    exit_market_mode:  Optional[str]  = None
    trailing_activate_ts: float       = 0.0   # 追踪止损激活时间戳（防止"边激活边触发"）
    ladder_level:        int          = 0     # 阶梯盈利锁当前层级（0=未触发）

# ============================================================
# 信号层统一输出对象
# ============================================================
@dataclass
class MarketSignal:
    """
    信号生成层的统一输出对象。
    _run_symbol 先填充此对象，决策层只读它，不再直接访问原始 ind_xx dict。
    """
    price:          float = 0.0
    rsi:            float = 50.0
    bb_pct:         float = 0.5
    macd_hist:      float = 0.0
    atr:            float = 0.0
    adx:            float = 20.0
    vol_surge:      float = 1.0
    ema_bull:       bool  = False
    vspike_active:  bool  = False
    vspike_mult:    float = 0.0
    vspike_dir:     str   = "均衡"
    vspike_buy_pct: float = 0.5
    vspike_baseline: float = 0.0
    market_mode:    str   = "震荡"
    ob_imbalance:   float = 0.0
    ob_wall_side:   str   = ""
    funding_rate:   float = 0.0
    ai_action:      str   = "hold"
    ai_conf:        float = 0.5
    ai_reason:      str   = ""
    fast_decision:  Optional[dict] = None
    fast_reason:    str   = ""

    def from_indicators(self, ind_15m: dict, vs_status: dict,
                        market_mode: str, depth: dict, funding: dict) -> "MarketSignal":
        """从原始指标 dict 填充信号对象"""
        self.rsi        = ind_15m.get("rsi", 50.0)
        self.bb_pct     = ind_15m.get("bb_pct", 0.5)
        self.macd_hist  = ind_15m.get("macd_hist", 0.0)
        self.atr        = ind_15m.get("atr", 0.0)
        self.adx        = ind_15m.get("adx", 20.0)
        self.vol_surge  = ind_15m.get("vol_surge", 1.0)
        self.ema_bull   = bool(ind_15m.get("ema_bull", False))
        self.price      = ind_15m.get("price", 0.0)
        self.market_mode = market_mode
        self.vspike_active  = bool(vs_status.get("is_spike") or vs_status.get("spike_recent"))
        self.vspike_mult    = vs_status.get("mult", 0.0)
        self.vspike_dir     = vs_status.get("direction", "均衡")
        self.vspike_buy_pct = vs_status.get("buy_pct", 0.5)
        self.vspike_baseline = vs_status.get("baseline_vol", 0.0)
        if hasattr(depth, "get"):
            self.ob_imbalance = depth.get("imbalance", 0.0)
            self.ob_wall_side = depth.get("wall_side", "")
        if hasattr(funding, "get"):
            self.funding_rate = funding.get("funding_rate", 0.0)
        return self

# ============================================================
# 仓位意图枚举（模块间通信用）
# ============================================================
class PositionIntentType(Enum):
    OPEN               = "open"
    CLOSE              = "close"
    UPDATE_SL           = "update_sl"
    UPDATE_TP           = "update_tp"
    UPDATE_PEAK         = "update_peak"
    SYNC_FROM_EXCHANGE  = "sync_exchange"
    RESET               = "reset"

# ============================================================
# 仓位意图对象（任何模块想修改仓位必须通过 PositionManager.submit）
# ============================================================
@dataclass
class PositionIntent:
    intent_type:    PositionIntentType
    side:           Optional[str]      = None
    size:           float              = 0.0
    entry_price:    float              = 0.0
    stop_loss:      float              = 0.0
    take_profit:    float              = 0.0
    leverage:       int                = 1
    liq_price:      float              = 0.0
    margin:         float              = 0.0
    reason:         str                = ""
    decision_id:    Optional[int]      = None
    pyramid_plan:   Optional[dict]      = None
    trailing_dist_atr_mult: Optional[float] = None
    # ── PositionManager 专用字段 ──────────────────────────────────
    source:         str                = ""   # 来源模块名
    payload:        Dict[str, Any]     = field(default_factory=dict)  # 变更数据

# ============================================================
# 事件总线（模块间通信）
# ============================================================
class EventBus:
    """简单的发布-订阅事件总线"""
    def __init__(self):
        self._handlers: Dict[str, list] = {}

    def subscribe(self, event_type: str, handler: callable):
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: callable):
        """订阅所有事件类型的通配处理器"""
        if "_all" not in self._handlers:
            self._handlers["_all"] = []
        self._handlers["_all"].append(handler)

    def publish(self, event_type: str, data: Any = None):
        # 快照 handler 列表，防止迭代时并发修改
        for handler in list(self._handlers.get(event_type, [])):
            try:
                handler(data)
            except Exception:
                _log.warning(f"[EventBus] handler {getattr(handler, '__name__', '?')} 处理 {event_type} 异常", exc_info=True)
        for handler in list(self._handlers.get("_all", [])):
            try:
                handler(data)
            except Exception:
                _log.warning(f"[EventBus] 通配 handler {getattr(handler, '__name__', '?')} 处理 {event_type} 异常", exc_info=True)

# ── EventBus 单例（模块加载时创建，Python import 自带线程安全）──────────────
_event_bus_instance: EventBus = EventBus()

def get_event_bus() -> EventBus:
    return _event_bus_instance

# ============================================================
# 状态持久化
# ============================================================
def _state_file() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_state.json")

def save_state_to_disk(pos: Position):
    from common import log
    sf  = _state_file()
    tmp = sf + ".tmp"
    with _gs_lock:
        gs_snapshot = dict(GLOBAL_STATE)
    state_data = {
        "global": gs_snapshot,
        "position": {
            **asdict(pos),
            "open_time":      pos.open_time.isoformat()      if pos.open_time      else None,
            "last_open_time": pos.last_open_time.isoformat() if pos.last_open_time else None,
        },
    }
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, sf)
    except Exception as e:
        log.exception(f"保存状态失败: {e}")
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass

def load_state_from_disk(pos: Position):
    from common import _parse_dt, log
    sf = _state_file()
    if not os.path.exists(sf):
        return
    try:
        with open(sf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "global" in data:
            gs_update(data["global"])
            _saved_date = data["global"].get("last_reset_date", "")
            _today_str  = datetime.now(UTC).date().isoformat()
            if _saved_date != _today_str:
                gs_set("today_opened_risk",  0.0)
                gs_set("today_realized_pnl", 0.0)
        if "position" in data:
            pos_data = data["position"]
            # 自动恢复所有标量字段（datetime 字段单独处理）
            _dt_fields = {"open_time", "last_open_time"}
            _skip_fields = _dt_fields  # 不自动赋值，下面手动处理
            for k, v in pos_data.items():
                if k in _skip_fields:
                    continue
                if hasattr(pos, k):
                    setattr(pos, k, v)
            # datetime 字段用 _parse_dt 恢复
            ot  = pos_data.get("open_time")
            pos.open_time        = _parse_dt(ot)
            lot = pos_data.get("last_open_time")
            pos.last_open_time   = _parse_dt(lot)
    except Exception as e:
        _log.exception(f"恢复状态失败: {e}")


# ============================================================
# TradingStateMachine — 从 state.py 迁移（交易状态机）
# ============================================================
class TradingStateMachine:
    """
    交易状态机 — 显式管理仓位生命周期。
    状态转换规则：
      IDLE     → SCANNING  : VSpike / 时间超时 / 关键价位临近
      SCANNING → HOLDING   : 开仓成功
      SCANNING → IDLE      : AI持续 hold 超过阈值
      HOLDING  → EXITING   : AI/规则触发平仓
      EXITING  → IDLE      : 平仓确认
      EXITING  → HOLDING   : 平仓失败（订单被拒）
    """
    _VALID_TRANSITIONS = {
        TradingState.IDLE:     {TradingState.SCANNING},
        TradingState.SCANNING: {TradingState.HOLDING, TradingState.IDLE},
        TradingState.HOLDING:  {TradingState.EXITING},
        TradingState.EXITING:  {TradingState.IDLE, TradingState.HOLDING},
    }

    def __init__(self):
        self._state  = TradingState.IDLE
        self._lock   = threading.Lock()
        self._entered_at: float = time.monotonic()
        self._history: List[tuple] = []

    @property
    def state(self) -> TradingState:
        return self._state

    def transition(self, new_state: TradingState, reason: str = "") -> bool:
        with self._lock:
            allowed = self._VALID_TRANSITIONS.get(self._state, set())
            if new_state not in allowed:
                _log.warning(
                    f"[StateMachine] 非法转换 {self._state.value}→{new_state.value} "
                    f"(reason={reason})，已拒绝"
                )
                return False
            self._history.append((self._state, self._entered_at))
            if len(self._history) > 50:
                self._history.pop(0)
            _log.info(f"[StateMachine] {self._state.value} → {new_state.value}  ({reason})")
            self._state      = new_state
            self._entered_at = time.monotonic()
            return True

    def force_state(self, new_state: TradingState, reason: str = ""):
        with self._lock:
            _log.info(f"[StateMachine] 强制设置 {self._state.value}→{new_state.value} ({reason})")
            self._state      = new_state
            self._entered_at = time.monotonic()

    def time_in_state(self) -> float:
        return time.monotonic() - self._entered_at

    def is_holding(self)  -> bool: return self._state == TradingState.HOLDING
    def is_scanning(self) -> bool: return self._state == TradingState.SCANNING
    def is_idle(self)     -> bool: return self._state == TradingState.IDLE
    def is_exiting(self)  -> bool: return self._state == TradingState.EXITING

    def sync_from_pos(self, pos: "Position") -> "TradingState":
        has_pos = bool(pos.side)
        cur = self._state

        if has_pos and cur in (TradingState.IDLE, TradingState.SCANNING):
            self.force_state(TradingState.HOLDING, "sync: 检测到持仓")
        elif not has_pos and cur == TradingState.HOLDING:
            self.force_state(TradingState.IDLE, "sync: 持仓已清空")
        elif not has_pos and cur == TradingState.EXITING:
            self.transition(TradingState.IDLE, "sync: 平仓确认")
        return self._state
