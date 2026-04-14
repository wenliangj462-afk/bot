# ============================================================
# config.py — 配置层（无依赖的基础层）
# 包含 CFG 单例和所有配置基础设施，不依赖 common 或 config_manager
# ============================================================
import os, threading, queue, time, json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

# ── 允许通过数据库热更新的参数白名单 ─────────────────────────────────────
_HOTPATCH_WHITELIST = {
    "OKX_CONFIDENCE_THRESHOLD", "OPEN_CONFIDENCE_MIN", "ADJUST_CONFIDENCE_THRESH",
    "SL_ATR_MULT", "TP_RR_RATIO", "RISK_PER_TRADE", "RISK_MULTIPLIER_MAX",
    "MAX_LEVERAGE", "MAX_DAILY_LOSS_PCT", "TRAILING_ACT_PCT", "TRAILING_DIST_PCT",
    "RAG_RSI_TOLERANCE", "RAG_SIMILAR_TRADES", "RAG_WIN_CASES", "RAG_MAX_AGE_DAYS",
    "MAX_TOTAL_EXPOSURE", "LEVEL_PROXIMITY_THRESH",
    "MIN_OPEN_INTERVAL_MINUTES", "FUNDING_RATE_THRESH",
    "EXIT_TP_ATR_MULT", "TP_RR_RATIO_MIN",
    "KELLY_FRACTION", "KELLY_MAX_F",
    "MAX_DAILY_RISK_PCT",
    "STRONG_SIGNAL_KELLY_BOOST",
    "V_SPIKE_MULT_THRESH", "OB_WALL_MULT",
}

# ── 模块级 cfg_field_map ─────────────────────────────────────────────────
_CFG_FIELD_MAP: Dict[str, tuple] = {
    "OKX_CONFIDENCE_THRESHOLD": ("confidence_thresh",         float),
    "OPEN_CONFIDENCE_MIN":       ("open_confidence_min",       float),
    "ADJUST_CONFIDENCE_THRESH":  ("adjust_confidence_thresh",  float),
    "SL_ATR_MULT":               ("sl_atr_mult",               float),
    "TP_RR_RATIO":               ("tp_rr_ratio",               float),
    "RISK_PER_TRADE":            ("risk_per_trade",            float),
    "RISK_MULTIPLIER_MAX":       ("risk_multiplier_max",       float),
    "MAX_LEVERAGE":              ("max_leverage",              int),
    "MAX_DAILY_LOSS_PCT":        ("max_daily_loss_pct",        float),
    "TRAILING_ACT_PCT":          ("trailing_act_pct",          float),
    "TRAILING_DIST_PCT":         ("trailing_dist_pct",         float),
    "RAG_RSI_TOLERANCE":         ("rag_rsi_tolerance",         float),
    "RAG_SIMILAR_TRADES":        ("rag_similar_trades",        int),
    "RAG_WIN_CASES":             ("rag_win_cases",             int),
    "RAG_MAX_AGE_DAYS":          ("rag_max_age_days",          int),
    "MAX_TOTAL_EXPOSURE":        ("max_total_exposure",        float),
    "LEVEL_PROXIMITY_THRESH":    ("level_proximity_thresh",    float),
    "MIN_OPEN_INTERVAL_MINUTES": ("min_open_interval_m",       int),
    "FUNDING_RATE_THRESH":       ("funding_rate_thresh",       float),
    "KELLY_FRACTION":            ("kelly_fraction",            float),
    "KELLY_MAX_F":               ("kelly_max_f",                float),
    "MAX_DAILY_RISK_PCT":        ("max_daily_risk_pct",        float),
    "STRONG_SIGNAL_KELLY_BOOST": ("strong_signal_kelly_boost", float),
    "V_SPIKE_MULT_THRESH":       ("v_spike_mult_thresh",       float),
    "V_SPIKE_MIN_CONTRACTS":     ("v_spike_min_contracts",     float),
    "OB_SLOPE_LEVELS":           ("ob_slope_levels",            int),
    "OB_WALL_MULT":              ("ob_wall_mult",               float),
    "MIN_HOLD_SECONDS":           ("min_hold_seconds",          int),
    "EXIT_TP_ATR_MULT":          ("exit_tp_atr_mult",          float),
    "OSC_RISK_RATIO":            ("osc_risk_ratio",             float),
    "CONSECUTIVE_LOSS_REDUCE_FACTOR": ("consecutive_loss_reduce_factor", float),
    "CLOSE_CONFIDENCE_THRESHOLD": ("close_confidence_threshold", float),
    "REASONER_CONSEC_LOSS_THRESH":   ("reasoner_consec_loss_thresh",   int,   2,     6),
    "REASONER_ROE_THRESH":            ("reasoner_roe_thresh",          float, -3.0, -0.05),
    "REASONER_ATR_RATIO_THRESH":     ("reasoner_atr_ratio_thresh",    float, 1.2,   3.0),
    "TIME_STOP_MINUTES":         ("time_stop_minutes",         int),
    "OSC_AGGRESSIVE_ADX_THRESH": ("osc_aggressive_adx_thresh", float),
    "SL_MIN_ATR_MULT":           ("sl_min_atr_mult",           float),
    "OSC_LEVEL_PROXIMITY":       ("osc_level_proximity",       float),
    "RSI_SILENCE_LOW":          ("rsi_silence_low",          float),
    "RSI_SILENCE_HIGH":         ("rsi_silence_high",         float),
    "ATR_TRUNC_MULT_HIGH":      ("atr_trunc_mult_high",      float),
    "ATR_TRUNC_MULT_NORMAL":     ("atr_trunc_mult_normal",    float),
    "ATR_TRUNC_MULT_LOW":       ("atr_trunc_mult_low",       float),
    "BB_WIDTH_PHYSICAL_FLOOR":  ("bb_width_physical_floor", float),
    "TREND_TP_TIGHTEN_R":       ("trend_tp_tighten_r",       float),
    "OSC_TP_RR_RATIO":          ("osc_tp_rr_ratio",         float),
    "OSC_AGGR_TP_RR_RATIO":     ("osc_aggr_tp_rr_ratio",    float),
    "TP_2_5R_CLOSE_PCT":         ("tp_2_5R_close_pct",        float),
    # ── Regime 复合化阈值（方向1）───────────────────────────────────────
    "REGIME_THRESH_TREND_TO_OSC":  ("regime_thresh_trend_to_osc",  float),
    "REGIME_THRESH_TREND_TO_AGGR": ("regime_thresh_trend_to_aggr", float),
    "REGIME_THRESH_OSC_TO_TREND":  ("regime_thresh_osc_to_trend",  float),
    "REGIME_THRESH_OSC_TO_AGGR":  ("regime_thresh_osc_to_aggr",  float),
    "REGIME_THRESH_AGGR_TO_OSC":  ("regime_thresh_aggr_to_osc",  float),
    "REGIME_THRESH_AGGR_TO_TREND":("regime_thresh_aggr_to_trend", float),
}

# ── Level 0：永久锁定值 ─────────────────────────────────────────────────
# 这些参数在运行时不可修改，防止关键安全参数被意外更改
_LEVEL0_LOCKED: Dict[str, float] = {
    "hard_stop_loss_pct": 0.04,      # 硬止损阈值
    "max_daily_loss_pct": 0.10,      # 日损熔断
    "max_leverage": 10,              # 最大杠杆（防止超额风险）
    "max_total_exposure": 10.0,      # 总名义杠杆率上限
    "max_margin_pct": 0.25,          # 最大保证金使用率
    "max_position_pct": 0.40,        # 单一仓位最大权益占比
}

# ── Level 2：AI 可自主调整的参数边界 ────────────────────────────────────
_LEVEL2_BOUNDS: Dict[str, tuple] = {
    "SL_ATR_MULT":         ("sl_atr_mult",         float, 1.0,   3.5),
    "TP_RR_RATIO":         ("tp_rr_ratio",          float, 1.5,   4.0),
    "EXIT_TP_ATR_MULT":    ("exit_tp_atr_mult",    float, 2.0,   5.0),
    "TRAILING_ACT_PCT":    ("trailing_act_pct",     float, 0.008, 0.030),
    "TRAILING_DIST_PCT":   ("trailing_dist_pct",    float, 0.008, 0.030),
    "OSC_BB_WIDTH_THRESH": ("osc_bb_width_thresh",  float, 0.020, 0.060),
    "OSC_SL_ATR_MULT":     ("osc_sl_atr_mult",      float, 1.0,   2.5),
    "CACHE_TTL_TREND":     ("cache_ttl_trend",      int,   120,   1200),
    "CACHE_TTL_OSC":       ("cache_ttl_osc",        int,   120,   1800),
    "CHECK_INTERVAL_EMPTY":   ("check_interval_empty",   int,   60,    300),
    "CHECK_INTERVAL_HOLD":    ("check_interval_hold",    int,   30,    180),
    "CHECK_INTERVAL_LEVEL":   ("check_interval_level",  int,   60,    120),
    "CHECK_INTERVAL_EXTREME": ("check_interval_extreme", int,   15,     30),
    "VSPIKE_EXTREME_THRESH":  ("vspike_extreme_thresh",  float, 2.0,   5.0),
    "OB_FASTLANE_THRESH":     ("ob_fastlane_thresh",    float, 0.50,  0.80),
    "V_SPIKE_MULT_THRESH":    ("v_spike_mult_thresh",    float, 1.5,   4.0),
    "V_SPIKE_MIN_CONTRACTS":  ("v_spike_min_contracts",  float, 10.0,  500.0),
    "OB_SLOPE_LEVELS":        ("ob_slope_levels",        int,   5,     20),
    "OB_WALL_MULT":           ("ob_wall_mult",           float, 2.0,   4.5),
    "RISK_PER_TRADE":             ("risk_per_trade",             float, 0.012, 0.032),
    "OSC_RISK_RATIO":             ("osc_risk_ratio",             float, 0.40,  1.00),
    "CONSECUTIVE_LOSS_REDUCE_FACTOR": ("consecutive_loss_reduce_factor", float, 0.50, 0.90),
    "OKX_CONFIDENCE_THRESHOLD":   ("confidence_thresh",          float, 0.60,  0.75),
    "CLOSE_CONFIDENCE_THRESHOLD": ("close_confidence_threshold", float, 0.55,  0.80),
    "TREND_BASE_CONF_THRESH":     ("trend_base_conf_thresh",     float, 0.40,  0.70),
    "TREND_CONF_CLAMP_MIN":       ("trend_conf_clamp_min",       float, 0.35,  0.60),
    "TREND_CONF_CLAMP_MAX":       ("trend_conf_clamp_max",       float, 0.60,  0.80),
    "SL_ATR_FLOOR_OSC_AGGR":     ("sl_atr_floor_osc_aggr",    float, 0.5,   1.5),
    "SL_ATR_CAP_OSC_AGGR":        ("sl_atr_cap_osc_aggr",       float, 1.5,   3.0),
    "SL_ATR_FLOOR_OSC":           ("sl_atr_floor_osc",          float, 0.5,   2.0),
    "SL_ATR_CAP_OSC":             ("sl_atr_cap_osc",            float, 1.5,   3.5),
    "SL_ATR_FLOOR_TREND":         ("sl_atr_floor_trend",        float, 1.0,   2.5),
    "SL_ATR_CAP_TREND":           ("sl_atr_cap_trend",          float, 2.0,   4.0),
    "BB_WIDTH_WIDE_THRESH":       ("bb_width_wide_thresh",     float, 0.03,  0.08),
    "BB_WIDTH_NARROW_THRESH":    ("bb_width_narrow_thresh",   float, 0.015, 0.040),
    "TIME_STOP_MINUTES":          ("time_stop_minutes",          int,   150,   420),
    "FUNDING_RATE_THRESH":        ("funding_rate_thresh",        float, 0.001, 0.006),
    "OSC_AGGRESSIVE_ADX_THRESH":  ("osc_aggressive_adx_thresh",  float, 16.0,  26.0),
    "SL_MIN_ATR_MULT":             ("sl_min_atr_mult",            float, 0.30,  1.00),
    "OSC_LEVEL_PROXIMITY":        ("osc_level_proximity",        float, 0.001, 0.005),
    "OSC_CONVICTION_OPEN_MIN":    ("osc_conviction_open_min",    float, 50.0,  65.0),
    "CONVICTION_OPEN_MIN":        ("conviction_open_min",        float, 50.0,  65.0),
    "ATR_TRUNC_MULT_HIGH":         ("atr_trunc_mult_high",      float, 2.0,   4.0),
    "ATR_TRUNC_MULT_NORMAL":       ("atr_trunc_mult_normal",    float, 1.5,   3.0),
    "ATR_TRUNC_MULT_LOW":          ("atr_trunc_mult_low",       float, 1.0,   2.5),
    "BB_WIDTH_PHYSICAL_FLOOR":     ("bb_width_physical_floor", float, 0.010, 0.025),
    "RSI_SILENCE_LOW":             ("rsi_silence_low",          float, 30.0,  50.0),
    "RSI_SILENCE_HIGH":            ("rsi_silence_high",         float, 50.0,  70.0),
    "TREND_TP_TIGHTEN_R":         ("trend_tp_tighten_r",      float, 1.5,   3.0),
    "OSC_TP_RR_RATIO":            ("osc_tp_rr_ratio",         float, 1.2,   2.5),
    "OSC_AGGR_TP_RR_RATIO":       ("osc_aggr_tp_rr_ratio",    float, 1.0,   2.0),
    "TP_2_5R_CLOSE_PCT":          ("tp_2_5R_close_pct",       float, 0.15,  0.40),
    # ── Regime 复合化阈值（方向1）───────────────────────────────────────
    "REGIME_THRESH_TREND_TO_OSC":  ("regime_thresh_trend_to_osc",  float, 0.35,  0.55),
    "REGIME_THRESH_TREND_TO_AGGR": ("regime_thresh_trend_to_aggr", float, 0.65,  0.85),
    "REGIME_THRESH_OSC_TO_TREND":  ("regime_thresh_osc_to_trend",  float, 0.30,  0.50),
    "REGIME_THRESH_OSC_TO_AGGR":  ("regime_thresh_osc_to_aggr",  float, 0.60,  0.80),
    "REGIME_THRESH_AGGR_TO_OSC":  ("regime_thresh_aggr_to_osc",  float, 0.32,  0.52),
    "REGIME_THRESH_AGGR_TO_TREND":("regime_thresh_aggr_to_trend", float, 0.30,  0.50),
}

_dyn_cfg: Dict[str, Any] = {}
_dyn_lock = threading.Lock()

def _dyn_set(field: str, value: Any) -> None:
    """显式写入动态字段（供热更新函数内部使用）"""
    if field in _LEVEL0_LOCKED:
        raise RuntimeError(f"Level-0 锁定参数 '{field}' 不可在运行时修改")
    with _dyn_lock:
        _dyn_cfg[field] = value

def _dyn_get(field: str, default: Any = None) -> Any:
    return _dyn_cfg.get(field, default)

def _safe_cast(value: str, target_type: type):
    if target_type == int:
        return int(float(value))
    return target_type(value)


# ══════════════════════════════════════════════════════════════════════════════
# BotConfig — 所有配置项的默认值（从环境变量读取）
# ══════════════════════════════════════════════════════════════════════════════
class BotConfig:
    symbol:                str   = os.getenv("TRADING_SYMBOL", "ETH-USDT-SWAP").strip()
    risk_check_interval:   int   = int(os.getenv("RISK_CHECK_INTERVAL", "2"))
    check_interval_hold:   int   = int(os.getenv("CHECK_INTERVAL_HOLD",   "90"))
    check_interval_level:  int   = int(os.getenv("CHECK_INTERVAL_LEVEL",  "60"))
    check_interval_extreme: int  = int(os.getenv("CHECK_INTERVAL_EXTREME", "15"))
    vspike_extreme_thresh: float = float(os.getenv("VSPIKE_EXTREME_THRESH", "2.0"))
    ob_fastlane_thresh:   float = float(os.getenv("OB_FASTLANE_THRESH",   "0.50"))
    check_interval_empty:  int   = int(os.getenv("CHECK_INTERVAL_EMPTY", "60"))
    level_proximity_thresh: float = float(os.getenv("LEVEL_PROXIMITY_THRESH", "0.005"))
    level_hard_override_thresh: float = float(os.getenv("LEVEL_HARD_OVERRIDE_THRESH", "0.002"))
    confidence_thresh:     float = float(os.getenv("OKX_CONFIDENCE_THRESHOLD", "0.65"))
    adjust_confidence_thresh: float = float(os.getenv("ADJUST_CONFIDENCE_THRESH", "0.75"))
    max_leverage:          int   = int(os.getenv("MAX_LEVERAGE", "10"))
    full_position_mode:    bool  = os.getenv("FULL_POSITION_MODE", "false").lower() == "true"
    max_daily_loss_pct:    float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05"))
    max_daily_risk_pct:    float = float(os.getenv("MAX_DAILY_RISK_PCT", "0.07"))
    risk_per_trade:        float = float(os.getenv("RISK_PER_TRADE", "0.02"))
    risk_multiplier_max:   float = float(os.getenv("RISK_MULTIPLIER_MAX", "1.5"))
    hard_stop_loss_pct:    float = float(os.getenv("HARD_STOP_LOSS_PCT", "0.08"))
    time_stop_minutes:     int   = int(os.getenv("TIME_STOP_MINUTES", "720"))
    time_stop_profit_minutes: int = int(os.getenv("TIME_STOP_PROFIT_MINUTES", "60"))
    time_stop_profit_pct:    float = float(os.getenv("TIME_STOP_PROFIT_PCT", "0.005"))
    min_order_notional:    float = float(os.getenv("MIN_ORDER_NOTIONAL", "10"))
    min_open_interval_m:   int   = int(os.getenv("MIN_OPEN_INTERVAL_MINUTES", "5"))
    news_refresh_minutes:  int   = int(os.getenv("NEWS_REFRESH_MINUTES", "15"))
    news_refresh_minutes_osc: int = int(os.getenv("NEWS_REFRESH_MINUTES_OSC", "60"))
    fg_refresh_minutes_osc: int  = int(os.getenv("FG_REFRESH_MINUTES_OSC", "120"))
    slippage_pct:          float = float(os.getenv("SLIPPAGE_PCT", "0.0005"))
    max_slippage_thresh:   float = float(os.getenv("MAX_SLIPPAGE_THRESH", "0.002"))
    trailing_act_pct:      float = float(os.getenv("TRAILING_ACT_PCT", "0.003"))
    trailing_dist_pct:     float = float(os.getenv("TRAILING_DIST_PCT", "0.010"))
    order_wait_seconds:    int   = int(os.getenv("ORDER_WAIT_SECONDS", "30"))
    price_decimals:        int   = int(os.getenv("PRICE_DECIMALS", "2"))
    funding_rate_thresh:   float = float(os.getenv("FUNDING_RATE_THRESH", "0.003"))
    liq_warn_pct:          float = float(os.getenv("LIQ_WARN_PCT", "0.05"))
    max_consecutive_loss:  int   = int(os.getenv("MAX_CONSECUTIVE_LOSS", "4"))
    min_cooldown_after_loss: int = int(os.getenv("MIN_COOLDOWN_AFTER_LOSS", "20"))
    consecutive_loss_reduce_factor: float = float(os.getenv("CONSECUTIVE_LOSS_REDUCE_FACTOR", "0.7"))
    reasoner_min_pnl_pct:     float = float(os.getenv("REASONER_MIN_PNL_PCT", "-2.0"))
    reasoner_min_interval_sec: int = int(os.getenv("REASONER_MIN_INTERVAL_SEC", "300"))
    maintenance_margin_rate: float = float(os.getenv("MAINTENANCE_MARGIN_RATE", "0.004"))
    health_port:           int   = int(os.getenv("HEALTH_PORT", "8080"))
    webhook_url:           str   = os.getenv("WEBHOOK_URL", "")
    webhook_level:         int   = int(os.getenv("WEBHOOK_LEVEL", "1"))
    webhook_retry:        int   = int(os.getenv("WEBHOOK_RETRY", "3"))
    webhook_queue_size:   int   = int(os.getenv("WEBHOOK_QUEUE_SIZE", "500"))
    webhook_fail_alert:   int   = int(os.getenv("WEBHOOK_FAIL_ALERT", "5"))
    cryptopanic_api_key:   str   = os.getenv("CRYPTOPANIC_API_KEY", "")
    state_file:            str   = "bot_state.json"
    okx_kline_urls: list        = None  # 在 __init__ 中设置默认值
    min_adjust_interval_minutes: int = int(os.getenv("MIN_ADJUST_INTERVAL_MINUTES", "15"))
    min_hold_seconds:           int = int(os.getenv("MIN_HOLD_SECONDS", "900"))
    exit_tp_atr_mult:          float = float(os.getenv("EXIT_TP_ATR_MULT", "3.5"))
    state_sync_interval_minutes: int = int(os.getenv("STATE_SYNC_INTERVAL_MINUTES", "5"))
    ws_ping_interval:      int   = int(os.getenv("WS_PING_INTERVAL", "20"))
    ws_max_retries:        int   = int(os.getenv("WS_MAX_RETRIES", "3"))
    ws_initial_retry_delay: float = float(os.getenv("WS_INITIAL_RETRY_DELAY", "1"))
    watchdog_interval:     int   = int(os.getenv("WATCHDOG_INTERVAL", "60"))
    kline_limit:           int   = int(os.getenv("KLINE_LIMIT", "300"))
    slippage_fuse_threshold: float = float(os.getenv("SLIPPAGE_FUSE_THRESHOLD", "0.01"))
    slippage_fuse_count:    int   = int(os.getenv("SLIPPAGE_FUSE_COUNT", "3"))
    slippage_fuse_pause_hours: float = float(os.getenv("SLIPPAGE_FUSE_PAUSE_HOURS", "1"))
    max_equity_drawdown_pct: float = float(os.getenv("MAX_EQUITY_DRAWDOWN_PCT", "0.08"))
    drawdown_pause_hours:    float = float(os.getenv("DRAWDOWN_PAUSE_HOURS", "1"))
    max_total_margin_ratio: float = float(os.getenv("MAX_TOTAL_MARGIN_RATIO", "0.7"))
    open_confidence_min:    float = float(os.getenv("OPEN_CONFIDENCE_MIN", "0.60"))
    check_margin_enabled:   bool  = os.getenv("CHECK_MARGIN_ENABLED", "true").lower() == "true"
    funding_settlement_guard_minutes: int   = int(os.getenv("FUNDING_SETTLEMENT_GUARD_MINUTES", "5"))
    funding_settlement_guard_rate:    float = float(os.getenv("FUNDING_SETTLEMENT_GUARD_RATE", "0.002"))
    funding_settlement_guard_confidence: float = float(os.getenv("FUNDING_SETTLEMENT_GUARD_CONFIDENCE", "0.9"))
    price_stale_seconds:    int   = int(os.getenv("PRICE_STALE_SECONDS", "10"))
    ai_timeout_seconds:     int   = int(os.getenv("AI_TIMEOUT_SECONDS", "40"))
    reasoner_timeout_seconds: int = int(os.getenv("REASONER_TIMEOUT_SECONDS", "180"))  # reasoner 思考时间长，单独设置
    ai_max_retries:         int   = int(os.getenv("AI_MAX_RETRIES", "3"))
    ai_retry_delay:         float = float(os.getenv("AI_RETRY_DELAY", "5"))
    ai_timeout_alert_count: int   = int(os.getenv("AI_TIMEOUT_ALERT_COUNT", "3"))
    ai_min_request_interval: int = int(os.getenv("AI_MIN_REQUEST_INTERVAL", "10"))
    max_hold_silence_minutes: int   = int(os.getenv("MAX_HOLD_SILENCE_MINUTES", "15"))
    silence_force_wakeup_loss_pct:  float = float(os.getenv("SILENCE_FORCE_WAKEUP_LOSS_PCT", "-3.0"))
    silence_force_wakeup_atr_mult: float = float(os.getenv("SILENCE_FORCE_WAKEUP_ATR_MULT", "3.5"))
    silence_wakeup_alert_cooldown: int = int(os.getenv("SILENCE_WAKEUP_ALERT_COOLDOWN", "600"))
    hold_silence_price_thresh_trend: float = float(os.getenv("HOLD_SILENCE_PRICE_THRESH_TREND", "0.003"))
    hold_silence_price_thresh_osc:   float = float(os.getenv("HOLD_SILENCE_PRICE_THRESH_OSC", "0.008"))
    cache_ttl_trend: int = int(os.getenv("CACHE_TTL_TREND", "480"))
    cache_ttl_osc:   int = int(os.getenv("CACHE_TTL_OSC",   "900"))
    mark_price_deviation_thresh: float = float(os.getenv("MARK_PRICE_DEVIATION_THRESH", "0.005"))
    open_wait_price_drift_pct:   float = float(os.getenv("OPEN_WAIT_PRICE_DRIFT_PCT", "0.001"))
    # ── Reasoner 触发阈值（可配置化，无需改代码）─────────────────────────────
    reasoner_consec_loss_thresh: int   = int(os.getenv("REASONER_CONSEC_LOSS_THRESH",   "3"))
    reasoner_roe_thresh:         float = float(os.getenv("REASONER_ROE_THRESH",          "-2.0"))
    reasoner_atr_ratio_thresh:   float = float(os.getenv("REASONER_ATR_RATIO_THRESH",  "2.5"))
    # ── SL ATR 自适应参数（各市场模式 floor/cap）─────────────────────────────
    sl_atr_floor_osc_aggr: float = float(os.getenv("SL_ATR_FLOOR_OSC_AGGR", "0.8"))
    sl_atr_cap_osc_aggr:    float = float(os.getenv("SL_ATR_CAP_OSC_AGGR",    "2.2"))
    sl_atr_floor_osc:       float = float(os.getenv("SL_ATR_FLOOR_OSC",       "1.3"))
    sl_atr_cap_osc:         float = float(os.getenv("SL_ATR_CAP_OSC",         "2.5"))
    sl_atr_floor_trend:     float = float(os.getenv("SL_ATR_FLOOR_TREND",     "1.8"))
    sl_atr_cap_trend:       float = float(os.getenv("SL_ATR_CAP_TREND",       "3.0"))
    # ── BB 宽度轮询阈值（宽→快车道，窄→正常轮询）─────────────────────────────
    bb_width_wide_thresh:   float = float(os.getenv("BB_WIDTH_WIDE_THRESH",   "0.05"))
    bb_width_narrow_thresh: float = float(os.getenv("BB_WIDTH_NARROW_THRESH", "0.03"))
    close_confidence_threshold: float = float(os.getenv("CLOSE_CONFIDENCE_THRESHOLD", "0.65"))
    trend_base_conf_thresh:    float = float(os.getenv("TREND_BASE_CONF_THRESH",     "0.55"))
    trend_conf_clamp_min:      float = float(os.getenv("TREND_CONF_CLAMP_MIN",       "0.45"))
    trend_conf_clamp_max:      float = float(os.getenv("TREND_CONF_CLAMP_MAX",       "0.70"))
    sl_min_atr_mult:      float = float(os.getenv("SL_MIN_ATR_MULT",      "0.5"))
    partial_tp_min_pct:   int   = int(os.getenv("PARTIAL_TP_MIN_PCT",   "20"))
    partial_tp_max_pct:   int   = int(os.getenv("PARTIAL_TP_MAX_PCT",   "50"))
    pyramid_max_entries:   int   = int(os.getenv("PYRAMID_MAX_ENTRIES",   "2"))
    pyramid_max_ratio:     float = float(os.getenv("PYRAMID_MAX_RATIO",   "0.5"))
    pyramid_max_risk_mult: float = float(os.getenv("PYRAMID_MAX_RISK_MULT", "2.0"))
    slippage_log_file:      str   = os.getenv("SLIPPAGE_LOG_FILE", "slippage_history.jsonl")
    data_retention_days:    int   = int(os.getenv("DATA_RETENTION_DAYS", "30"))
    log_file:               str   = os.getenv("LOG_FILE", "eth_trader_v6.log")
    kelly_log_file:        str   = os.getenv("KELLY_LOG_FILE", "kelly_metrics.jsonl")
    macro_kline_days:       int   = int(os.getenv("MACRO_KLINE_DAYS", "365"))
    rag_similar_trades:     int   = int(os.getenv("RAG_SIMILAR_TRADES", "2"))
    rag_win_cases:          int   = int(os.getenv("RAG_WIN_CASES", "2"))
    rag_max_age_days:       int   = int(os.getenv("RAG_MAX_AGE_DAYS", "60"))
    rag_rsi_noise:          float = float(os.getenv("RAG_RSI_NOISE", "2.0"))
    rag_rsi_tolerance:      float = float(os.getenv("RAG_RSI_TOLERANCE", "8.0"))
    rag_require_same_trend: bool  = os.getenv("RAG_REQUIRE_SAME_TREND", "true").lower() == "true"
    sl_atr_mult:            float = float(os.getenv("SL_ATR_MULT", "2.0"))
    sl_atr_adapt_enable:    bool  = os.getenv("SL_ATR_ADAPT_ENABLE", "true").lower() == "true"
    sl_atr_adapt_factor:    float = float(os.getenv("SL_ATR_ADAPT_FACTOR", "0.5"))
    tp_rr_ratio:            float = float(os.getenv("TP_RR_RATIO", "2.5"))
    tp_rr_ratio_min:       float = float(os.getenv("TP_RR_RATIO_MIN", "1.2"))
    osc_bb_width_thresh:       float = float(os.getenv("OSC_BB_WIDTH_THRESH", "0.03"))
    osc_level_proximity:       float = float(os.getenv("OSC_LEVEL_PROXIMITY", "0.002"))
    osc_risk_ratio:            float = float(os.getenv("OSC_RISK_RATIO", "0.8"))
    osc_sl_atr_mult:           float = float(os.getenv("OSC_SL_ATR_MULT", "1.5"))
    osc_aggressive_bb_thresh:  float = float(os.getenv("OSC_AGGRESSIVE_BB_THRESH", "0.020"))  # 必须小于 osc_bb_width_thresh，保证层级单调递减
    osc_aggressive_adx_thresh: float = float(os.getenv("OSC_AGGRESSIVE_ADX_THRESH", "22.0"))
    ob_slope_levels: int   = int(os.getenv("OB_SLOPE_LEVELS", "10"))
    ob_wall_mult:    float = float(os.getenv("OB_WALL_MULT",   "3.0"))
    v_spike_mult_thresh: float = float(os.getenv("V_SPIKE_MULT_THRESH", "2.8"))
    v_spike_min_contracts: float = float(os.getenv("V_SPIKE_MIN_CONTRACTS", "50.0"))
    vspike_escape_level1:  float = float(os.getenv("VSPIKE_ESCAPE_LEVEL1",  "4.0"))
    vspike_escape_level2:  float = float(os.getenv("VSPIKE_ESCAPE_LEVEL2",  "5.0"))
    vspike_extreme_mult:   float = float(os.getenv("VSPIKE_EXTREME_MULT",   "10.0"))
    vspike_silence_break_cd: int  = int(os.getenv("VSPIKE_SILENCE_BREAK_CD", "180"))  # 非极端VSpike打破静默最小间隔(秒)
    vspike_escape_baseline: float = float(os.getenv("VSPIKE_ESCAPE_BASELINE", "20.0"))
    escape_loss_min:       float = float(os.getenv("ESCAPE_LOSS_MIN",       "0.005"))
    profit_protect_thresh: float = float(os.getenv("PROFIT_PROTECT_THRESH", "0.01"))
    startup_cooldown_seconds: int = int(os.getenv("STARTUP_COOLDOWN_SECONDS", "180"))
    drawdown_kelly_decay_start: float = float(os.getenv("DRAWDOWN_KELLY_DECAY_START", "0.04"))
    drawdown_kelly_floor:       float = float(os.getenv("DRAWDOWN_KELLY_FLOOR",       "0.25"))
    osc_conviction_open_min:    float = float(os.getenv("OSC_CONVICTION_OPEN_MIN",   "57.0"))
    conviction_open_min:        float = float(os.getenv("CONVICTION_OPEN_MIN",        "55.0"))
    osc_conviction_min:         float = float(os.getenv("OSC_CONVICTION_MIN",         "40.0"))
    conviction_vspike_tau:      float = float(os.getenv("CONVICTION_VSPIKE_TAU",      "8.0"))
    conviction_full_score:      float = float(os.getenv("CONVICTION_FULL_SCORE",      "88.0"))
    price_stale_fallback_secs:  int   = int(os.getenv("PRICE_STALE_FALLBACK_SECONDS", "15"))
    stale_sl_expansion:         float = float(os.getenv("STALE_DATA_SL_EXPANSION",    "1.2"))
    stale_lev_reduction:        float = float(os.getenv("STALE_DATA_LEV_REDUCTION",   "0.8"))
    ai_failure_exp_backoff:     float = float(os.getenv("AI_FAILURE_EXP_BACKOFF",     "1.5"))
    vspike_priority_threshold:  float = float(os.getenv("VSPIKE_PRIORITY_THRESHOLD",  "15.0"))
    aggressive_conflict_cooldown: float = float(os.getenv("AGGRESSIVE_CONFLICT_COOLDOWN", "120.0"))
    ai_fastlane_min_interval:   int   = int(os.getenv("AI_FASTLANE_MIN_INTERVAL",     "5"))
    ob_fastlane_imbalance:      float = float(os.getenv("OB_FASTLANE_IMBALANCE",      "0.35"))
    qwen_api_key:             str   = os.getenv("QWEN_API_KEY",              "").strip()
    qwen_base_url:            str   = os.getenv("QWEN_BASE_URL",   "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
    qwen_model:               str   = os.getenv("QWEN_MODEL",     "qwen-plus").strip()
    qwen_timeout:             int   = int(os.getenv("QWEN_TIMEOUT",             "20"))
    arbitration_min_score:    float = float(os.getenv("ARBITRATION_MIN_SCORE", "70.0"))
    arbitration_max_score:   float = float(os.getenv("ARBITRATION_MAX_SCORE", "82.0"))
    max_margin_pct:         float = float(os.getenv("MAX_MARGIN_PCT", "0.25"))
    max_position_pct:       float = float(os.getenv("MAX_POSITION_PCT", "0.40"))
    slippage_adapt_enable:  bool  = os.getenv("SLIPPAGE_ADAPT_ENABLE", "true").lower() == "true"
    slippage_adapt_window:  int   = int(os.getenv("SLIPPAGE_ADAPT_WINDOW", "10"))
    slippage_adapt_mult:    float = float(os.getenv("SLIPPAGE_ADAPT_MULT", "1.2"))
    slippage_fuse_avg_thresh: float = float(os.getenv("SLIPPAGE_FUSE_AVG_THRESH", "0.002"))
    api_rate_limit_public:  int   = int(os.getenv("API_RATE_LIMIT_PUBLIC", "10"))
    api_rate_limit_private: int   = int(os.getenv("API_RATE_LIMIT_PRIVATE", "5"))
    risk_adapt_enable:      bool  = os.getenv("RISK_ADAPT_ENABLE", "true").lower() == "true"
    risk_adapt_window_hours: int  = int(os.getenv("RISK_ADAPT_WINDOW_HOURS", "24"))
    risk_adapt_win_rate_target: float = float(os.getenv("RISK_ADAPT_WIN_RATE_TARGET", "0.5"))
    cache_price_bucket_trending: int = int(os.getenv("CACHE_PRICE_BUCKET_TRENDING", "20"))
    cache_price_bucket_osc:      int = int(os.getenv("CACHE_PRICE_BUCKET_OSC", "10"))
    cache_force_refresh_conf:    float = float(os.getenv("CACHE_FORCE_REFRESH_CONF", "0.85"))
    max_total_exposure: float = float(os.getenv("MAX_TOTAL_EXPOSURE", "10.0"))
    kelly_fraction: float = float(os.getenv("KELLY_FRACTION", "0.5"))
    kelly_max_f:     float = float(os.getenv("KELLY_MAX_F", "0.6"))
    strong_signal_kelly_boost: float = float(os.getenv("STRONG_SIGNAL_KELLY_BOOST", "0.65"))
    rag_case_quality_threshold: float = float(os.getenv("RAG_CASE_QUALITY_THRESHOLD", "7.5"))
    max_historical_cases: int = int(os.getenv("MAX_HISTORICAL_CASES", "15"))
    case_max_age_days: int = int(os.getenv("CASE_MAX_AGE_DAYS", "90"))
    enable_auto_case_pool: bool = os.getenv("ENABLE_AUTO_CASE_POOL", "true").lower() == "true"
    # ── Magic Numbers 抽取（方向3）─────────────────────────────────────────────
    rsi_silence_low:        float = float(os.getenv("RSI_SILENCE_LOW",        "40.0"))
    rsi_silence_high:        float = float(os.getenv("RSI_SILENCE_HIGH",        "60.0"))
    atr_trunc_mult_high:    float = float(os.getenv("ATR_TRUNC_MULT_HIGH",    "3.0"))
    atr_trunc_mult_normal:  float = float(os.getenv("ATR_TRUNC_MULT_NORMAL",  "2.0"))
    atr_trunc_mult_low:     float = float(os.getenv("ATR_TRUNC_MULT_LOW",     "1.5"))
    bb_width_physical_floor:float = float(os.getenv("BB_WIDTH_PHYSICAL_FLOOR","0.015"))
    trend_tp_tighten_r:     float = float(os.getenv("TREND_TP_TIGHTEN_R",    "2.0"))
    osc_tp_rr_ratio:       float = float(os.getenv("OSC_TP_RR_RATIO",       "1.5"))
    osc_aggr_tp_rr_ratio:  float = float(os.getenv("OSC_AGGR_TP_RR_RATIO",  "1.2"))
    tp_2_5R_close_pct:      float = float(os.getenv("TP_2_5R_CLOSE_PCT",    "0.25"))
    # ── Regime 复合化阈值（方向1）─────────────────────────────────────────────
    regime_thresh_trend_to_osc:  float = float(os.getenv("REGIME_THRESH_TREND_TO_OSC",  "0.45"))
    regime_thresh_trend_to_aggr: float = float(os.getenv("REGIME_THRESH_TREND_TO_AGGR", "0.78"))
    regime_thresh_osc_to_trend:  float = float(os.getenv("REGIME_THRESH_OSC_TO_TREND",  "0.40"))
    regime_thresh_osc_to_aggr:  float = float(os.getenv("REGIME_THRESH_OSC_TO_AGGR",  "0.72"))
    regime_thresh_aggr_to_osc:  float = float(os.getenv("REGIME_THRESH_AGGR_TO_OSC",  "0.42"))
    regime_thresh_aggr_to_trend: float = float(os.getenv("REGIME_THRESH_AGGR_TO_TREND", "0.38"))
    # 规则引擎直出门槛（0.72 配合 Fast AI 双保险，允许布林带等常规信号 0.72+ 通过）
    rule_engine_bypass_conf:  float = float(os.getenv("RULE_ENGINE_BYPASS_CONF",  "0.72"))
    # ── 打板突破阈值 ──────────────────────────────────────────────────────
    breakout_vol_min_trend:   float = float(os.getenv("BREAKOUT_VOL_MIN_TREND",   "1.4"))
    breakout_vol_min_osc:     float = float(os.getenv("BREAKOUT_VOL_MIN_OSC",     "1.2"))
    breakout_ob_min_trend:    float = float(os.getenv("BREAKOUT_OB_MIN_TREND",    "0.25"))
    breakout_ob_min_osc:      float = float(os.getenv("BREAKOUT_OB_MIN_OSC",      "0.20"))
    breakout_conf_min:        float = float(os.getenv("BREAKOUT_CONF_MIN",        "0.60"))

    def __init__(self):
        if self.okx_kline_urls is None:
            self.okx_kline_urls = ["https://www.okx.com", "https://aws.okx.com"]


# ══════════════════════════════════════════════════════════════════════════════
# DynamicConfigProxy — 动态配置代理
# ══════════════════════════════════════════════════════════════════════════════
_DYNAMIC_FIELDS: set = {
    tup[0] for tup in _CFG_FIELD_MAP.values()
}

class DynamicConfigProxy:
    """
    动态配置代理：BotConfig 的所有字段仍然可通过 CFG.xxx 访问。
    - 静态字段（不在 _DYNAMIC_FIELDS）：直接透传到底层 BotConfig 单例
    - 动态字段（在 _DYNAMIC_FIELDS）：读写 _dyn_cfg（独立字典，线程安全）
    """
    __slots__ = ('_static',)

    def __init__(self):
        object.__setattr__(self, '_static', _static_cfg)

    def __getattr__(self, name: str):
        if name in _DYNAMIC_FIELDS:
            return _dyn_cfg.get(name, getattr(self._static, name))
        return getattr(self._static, name)

    def __setattr__(self, name: str, value: Any):
        if name in _LEVEL0_LOCKED:
            raise RuntimeError(f"Level-0 锁定参数 '{name}' 不可在运行时修改")
        if name in _DYNAMIC_FIELDS:
            with _dyn_lock:
                _dyn_cfg[name] = value
        else:
            setattr(self._static, name, value)

    def __repr__(self):
        return f"<DCFG dynamic={_dyn_cfg}>"


# ── 创建配置单例 ─────────────────────────────────────────────────────────────
_static_cfg = BotConfig()
CFG = DynamicConfigProxy()


# ══════════════════════════════════════════════════════════════════════════════
# 热更新框架（从 config_manager.py 合并）
# ══════════════════════════════════════════════════════════════════════════════
UTC = timezone.utc

# ── 滑点日志缓冲队列 ─────────────────────────────────────────────────────
_slippage_queue: queue.Queue = queue.Queue(maxsize=500)
_slippage_batch_size = 20
_slippage_last_flush = time.time()

def _slippage_writer_loop():
    """后台线程：批量写入滑点日志，减少文件 I/O 次数"""
    global _slippage_last_flush
    from common import log as _log
    while True:
        try:
            records = []
            try:
                records.append(_slippage_queue.get(timeout=5))
                while not _slippage_queue.empty():
                    records.append(_slippage_queue.get_nowait())
            except queue.Empty:
                pass
            if records:
                try:
                    with open(CFG.slippage_log_file, "a", encoding="utf-8") as f:
                        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
                    _slippage_last_flush = time.time()
                except Exception:
                    pass
        except Exception as e:
            try:
                _log.error(f"[slippage_writer] 异常: {e}")
            except Exception:
                pass
            time.sleep(5)

threading.Thread(target=_slippage_writer_loop, daemon=True, name="slippage-writer").start()

def log_slippage(side: str, expected_px: float, fill_px: float,
                 size: float, slippage_pct: float, decision_id: int = None):
    """每笔成交记录滑点，通过缓冲队列异步写入，不阻塞主线程。"""
    try:
        record = {
            "ts":           datetime.now(UTC).isoformat(),
            "side":         side,
            "expected_px":  round(expected_px, 4),
            "fill_px":      round(fill_px, 4),
            "slippage_pct": round(slippage_pct * 100, 4),
            "size":         size,
            "decision_id":  decision_id,
            "above_thresh": slippage_pct > CFG.max_slippage_thresh,
        }
        _slippage_queue.put_nowait(record)
    except Exception:
        pass


# ── Pending Config Functions ──────────────────────────────────────────────

def submit_pending_config(param_key: str, old_value: str, new_value: str,
                          reason: str, source: str = "reasoner") -> bool:
    """将 Reasoner 建议写入待审批表，不立即生效。"""
    from common import log, get_db_conn
    if param_key not in _HOTPATCH_WHITELIST:
        log.warning(f"参数 {param_key} 不在热更新白名单，拒绝写入")
        return False
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute("UPDATE pending_config SET status='superseded' WHERE param_key=? AND status='pending'",
                  (param_key,))
        c.execute("""INSERT INTO pending_config
                     (param_key, old_value, new_value, reason, source, status, created_ts)
                     VALUES (?,?,?,?,?,'pending',?)""",
                  (param_key, str(old_value), str(new_value), reason, source,
                   datetime.now(UTC).isoformat()))
        conn.commit()
        log.info(f"[待审批] {param_key}: {old_value} -> {new_value}（{reason[:60]}）")
        return True
    except Exception as e:
        log.error(f"写入待审批配置失败: {e}")
        return False

def get_pending_configs() -> list:
    """获取所有待审批的配置变更。"""
    from common import get_db_conn
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute("""SELECT id, param_key, old_value, new_value, reason, created_ts
                     FROM pending_config WHERE status='pending' ORDER BY created_ts""")
        rows = c.fetchall()
        return [{"id": r[0], "key": r[1], "old": r[2], "new": r[3],
                 "reason": r[4], "ts": r[5]} for r in rows]
    except Exception:
        return []

def approve_pending_config(config_id: int):
    """审批通过某条配置变更，立即更新 CFG 运行时值（热更新，无需重启）。"""
    from common import log, get_db_conn, set_sys_config, _webhook, bot_instance
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute("SELECT param_key, new_value, reason FROM pending_config WHERE id=? AND status='pending'",
                  (config_id,))
        row = c.fetchone()
        if not row:
            return None, None
        param_key, new_value, reason = row
        c.execute("UPDATE pending_config SET status='approved', applied_ts=? WHERE id=?",
                  (datetime.now(UTC).isoformat(), config_id))
        conn.commit()

        if param_key in _CFG_FIELD_MAP:
            _tup = _CFG_FIELD_MAP[param_key]
            field, cast = _tup[0], _tup[1]
            try:
                converted = _safe_cast(new_value, cast)
                _dyn_set(field, converted)
                set_sys_config(f"hotpatch_{param_key}", new_value)
                log.info(f"[热更新] {param_key} = {converted}（已审批）")
                _webhook("参数热更新", f"{param_key}: -> {new_value}\n理由: {reason[:100]}")
                if bot_instance:
                    bot_instance._clear_ai_cache()
                    bot_instance._prev_indicators = {}
                    bot_instance._market_mode = "趋势"
            except Exception as e:
                log.error(f"热更新 CFG 字段失败: {e}")
        return param_key, new_value
    except Exception as e:
        log.error(f"审批配置失败: {e}")
        return None, None

def reject_pending_config(config_id: int):
    """拒绝某条待审批配置。"""
    from common import log, get_db_conn
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute("UPDATE pending_config SET status='rejected', applied_ts=? WHERE id=?",
                  (datetime.now(UTC).isoformat(), config_id))
        conn.commit()
        log.info(f"[已拒绝] 配置变更 id={config_id}")
    except Exception as e:
        log.error(f"拒绝配置失败: {e}")

def try_apply_level2_suggestions(suggestions: dict, symbol: str = "") -> list:
    """
    Level 2 动态参数自动调整。
    AI 在决策 JSON 中可以附带 param_suggestions 字段，本函数自动校验边界并应用，
    无需人工审批。
    """
    from common import log, get_db_conn, set_sys_config, _webhook, bot_instance
    applied = []
    if not suggestions:
        return applied

    for env_key, raw_value in suggestions.items():
        if env_key not in _LEVEL2_BOUNDS:
            continue
        field, cast_type, lo, hi = _LEVEL2_BOUNDS[env_key]
        if field in _LEVEL0_LOCKED:
            continue
        try:
            new_val = _safe_cast(str(raw_value), cast_type)
        except (ValueError, TypeError):
            continue
        if not (lo <= new_val <= hi):
            continue
        old_val = _dyn_get(field)
        if old_val == new_val:
            continue
        try:
            _dyn_set(field, new_val)
            set_sys_config(f"l2_{env_key}", str(new_val))
            applied.append((env_key, old_val, new_val))
            log.info(f"[L2] 自动调参: {env_key} {old_val} -> {new_val}")
        except Exception as e:
            log.error(f"[L2] 写入 {env_key} 失败: {e}")
    if applied:
        lines = "\n".join(f"  {k}: {o} -> {n}" for k, o, n in applied)
        _webhook("AI 自动调参 (L2)", f"品种: {symbol or 'N/A'}\n{lines}", level=2)
        if bot_instance:
            bot_instance._clear_ai_cache()
    return applied

def _load_dynamic_config():
    """遍历 _CFG_FIELD_MAP，将热更新参数写入 _dyn_cfg。启动时调用。"""
    from common import log, get_sys_config
    loaded = 0
    for param_key, tup in _CFG_FIELD_MAP.items():
        field_name, cast = tup[0], tup[1]
        saved = get_sys_config(f"hotpatch_{param_key}")
        if saved:
            try:
                converted = _safe_cast(saved, cast)
                _dyn_set(field_name, converted)
                loaded += 1
            except Exception as e:
                log.error(f"热加载参数 {param_key} 失败: {e}")
    if loaded:
        log.info(f"热加载完成，共恢复 {loaded} 个热更新参数")
