"""
ETH-USDT-SWAP 量化交易系统 V6.0（超级无敌ai动态维护）

【代码质量 TODO / FIXME - 待改进项】
  TODO(高优先级):
    - 核心业务逻辑（calc_indicators、calc_liq_price 等）与 I/O 分离，便于单元测试。
    - 依赖注入：将 OkxTrader 作为 ETHTrader 构造参数，便于 mock。
  TODO(长期):
    - 使用 black/ruff 格式化代码，确保单行≤120字符。

  ✅ 已完成:
    - SmartAIConsultant.get_decision() 拆分为 _build_prompt()、_single_call()、_vote()，主方法仅做编排。
    - ETHTrader._do_open() 拆分为 _calc_size_and_margin()、_pre_reserve()、_place_order_phase()、_post_fill_sync()，_do_open 为纯协调器（~45行）。
    - 全自动动态历史案例池：historical_cases 表启动时自动建表，平仓后自动生成 + DeepSeek reasoner 评估（质量≥7.5才入池），每日 UTC 00:05 自动维护（删老/去重/胜败平衡），retrieve_similar_failures 优先从案例池检索。
    - 每日重置可靠化：_daily_reset_loop 合并到 UTC 00:05，同时重置 today_opened_risk + today_realized_pnl + 案例池维护。
    - _handle_pending_order I/O 移出决策热路径：_run_symbol 仅做内存快读，实际 I/O 在 trailing 线程每45s处理。
    - 所有线程池（_async_executor + _auto_case_executor + postmortem._executor）均在 finally 中优雅 shutdown。
    - 金字塔加仓执行逻辑：在持仓分支（_run_symbol）中每轮检查浮盈触发，满足条件立即执行加仓（三阶段锁 + 累计风险上限）。
    - _apply_pyramid_plan() + _apply_market_mode_adjustments()：从 _run_symbol 提取为独立函数，消除代码散落。
    - bot_instance 弱引用清理：Telegram /status 命令中所有 getattr(bot_instance, ...) 替换为 bot=_get_bot_instance() 后安全访问。
    - _FEW_SHOT_EXAMPLES 硬编码废弃：历史案例池≥6条时完全由真实案例替代，不再混入硬编码样例。
    - JSON 解析器合并：`_clean_json_text` 提取为共享辅助函数，`_parse_llm_json` 与 `_parse_llm_json_array` 共用，减少重复代码。
    - AI 决策日志结构化：`_vote()` 输出 JSON 格式决策日志（包含双模型投票详情），便于回溯分析。
    - 滑点日志关联交易：`_place_order_phase` 成交后自动记录 expected_px / fill_px / slippage_pct 与 decision_id。

【已完成优化 (本轮)】
  ✅ BotConfig → StaticConfig(BotConfig) + DynamicConfig(DCFG) 分离，专用锁
  ✅ 数据库 → DatabaseManager 类（队列式写入线程 + WAL 并发读）
  ✅ RAG 查询：ORDER BY ts DESC LIMIT 500 + random.sample() 内存采样
  ✅ 主循环日志：log.debug + 每10轮摘要（权益/胜率/布林带宽/市场模式）
"""

import os, time, json, logging, traceback, re, hmac, hashlib, base64, requests, math
import xml.etree.ElementTree as ET
import threading
import queue
import sqlite3
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any, Callable
from functools import wraps
from contextlib import closing
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from http.server import HTTPServer, BaseHTTPRequestHandler
from logging.handlers import RotatingFileHandler
from collections import deque

import pandas as pd
import numpy as np
import socket
import websocket
from dotenv import load_dotenv
from openai import OpenAI
from prometheus_client import Gauge, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

load_dotenv()

# ── 数据模型（core.py）──────────────────────────────────────────────────
from core import (
    GLOBAL_STATE, Position, PositionIntent, PositionIntentType,
    TradingEvent, TradingState, MarketSignal,
    gs_get, gs_set, gs_update, gs_increment, gs_add,
    _gs_lock, get_event_bus, UTC,
    save_state_to_disk, load_state_from_disk,
)

# ── 共享基础设施（common.py）────────────────────────────────────────────
from common import (
    log, ai_log, log_event, kelly_optimal_size, bayesian_posterior,
    log_kelly_metrics, init_db, get_db_conn, get_db_manager, get_sys_config, set_sys_config,
    submit_pending_config, get_pending_configs, approve_pending_config,
    reject_pending_config, try_apply_level2_suggestions, _load_dynamic_config,
    save_decision_to_db, save_trade_open, _webhook, bot_instance,
    _LEVEL2_BOUNDS, _LEVEL0_LOCKED, _parse_dt, price_reclaimed,
    retrieve_similar_failures, build_rag_warning,
    Watchdog, watchdog_loop, start_health_server, _webhook_queue,
    TradePostmortem, DailyReportModule, get_trades_in_range,
    DatabaseManager, PositionManager,
    update_trade_close, get_recent_win_rate, get_pending_order_by_id,
    _health_data,
)
from config import CFG
from core import TradingStateMachine

# ── 模块导入（拆分后）──────────────────────────────────────────────────────
from adapters import (
    AIGatekeeper, ArbitrationTrigger, ConvictionScorer, SmartAIConsultant,
    FastLaneModule, get_market_mode, get_trend_alignment_score, _price_of_level,
)
from market import (
    SignalsModule, fetch_global_news, fetch_market_sentiment_data,
    build_macro_context,
    calc_indicators, build_kline_series, calc_key_levels,
    fetch_fear_greed, _get_rsi_interval, _get_ma_alignment,
)
from exchange import (
    OkxTrader, OkxWebSocket, VolumeSpikeDetector,
    _api_need_heal, _public_limiter, _private_limiter,
)
from position_exec import PositionExec, calc_liq_price, _get_atr_quantile
from risk_guard import RiskGuard


# ============================================================
# StateManager — 从 state.py 迁移（状态同步核心）
# ============================================================
def _safe_float(v, default: float = 0.0) -> float:
    """安全转换为浮点数"""
    try:
        return float(v) if v is not None else default
    except (ValueError, TypeError):
        return default


class StateManager:
    """
    状态管理器：从 state.py 迁移。
    将从 ETHTrader 提取的独立方法封装为 StateManager 的方法，
    通过 self.trader 访问 ETHTrader 实例的状态。
    """

    def __init__(self, trader):
        self.trader = trader
        self.okx = trader.trader  # OkxTrader 快捷引用

    def _full_state_sync(self):
        """全量状态同步：余额 + 单品种持仓（20秒超时保护）"""
        log.debug("执行全量状态同步...")
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTTimeout

        def _sync_worker():
            bal_full = self.okx.get_account_balance_full()
            self.trader.latest_avail_bal = bal_full["avail_bal"]
            self.trader.latest_equity    = bal_full["equity"]

            resp = self.okx.get_positions()
            exch_pos = {}
            if resp.get("code") == "0":
                for p in resp.get("data", []):
                    sid = p.get("instId", "")
                    if sid == CFG.symbol and abs(float(p.get("pos", 0))) > 0:
                        exch_pos[sid] = p

            pos  = self.trader.pos
            lock = self.trader.lock
            needs_reset = False
            acquired = lock.acquire(timeout=5)
            if not acquired:
                log.warning(f"[{CFG.symbol}] 全量同步获取锁超时(5s)，跳过该品种")
                gs_set("last_state_sync", datetime.now(UTC).isoformat())
                return
            else:
                _snap_trade_id    = None
                _snap_side        = None
                _snap_entry_price = 0.0
                try:
                    if CFG.symbol in exch_pos:
                        p       = exch_pos[CFG.symbol]
                        posSide = p.get("posSide")
                        if posSide in ["long", "short"]:
                            if not pos.side:
                                log.warning(f"[{CFG.symbol}] 孤儿仓位！本地无但交易所有")
                                _webhook("孤儿仓位检测", f"[{CFG.symbol}] {posSide} {p.get('pos')}张")
                            pos.side        = posSide
                            pos.size        = abs(float(p["pos"]))
                            pos.entry_price = float(p["avgPx"])
                            pos.liq_price   = float(p.get("liqPx", 0))
                            if not pos.open_time:
                                pos.open_time = datetime.now(UTC)
                    else:
                        if pos.side:
                            log.warning(f"[{CFG.symbol}] 本地有持仓但交易所无，尝试补写平仓记录")
                            _webhook("状态不一致", f"[{CFG.symbol}] 本地有持仓但交易所无，尝试补写平仓记录")
                            if pos.trade_id and pos.side and pos.entry_price > 0:
                                _snap_trade_id    = pos.trade_id
                                _snap_side        = pos.side
                                _snap_entry_price = pos.entry_price
                                _snap_leverage    = pos.leverage if pos.leverage > 0 else 1
                            needs_reset = True
                finally:
                    lock.release()

                if _snap_trade_id:
                    try:
                        current_price = self.trader._get_price(CFG.symbol)
                        if current_price > 0:
                            update_trade_close(
                                _snap_trade_id,
                                current_price,
                                "同步补录：本地有持仓但交易所丢失",
                                leverage=_snap_leverage
                            )
                            log.info(f"[{CFG.symbol}] 已补写平仓记录 trade_id={_snap_trade_id}")
                    except Exception as e:
                        log.error(f"补写平仓记录失败: {e}")

                if needs_reset:
                    self._reset_pos()
                else:
                    save_state_to_disk(pos)

            calculated_reserved = 0.0
            pos = self.trader.pos
            if pos.side and pos.entry_price > 0 and pos.size > 0:
                price = self.trader._get_price(CFG.symbol)
                if price > 0:
                    ct_val = self.okx.contract_sizes.get(CFG.symbol, 0.01)
                    notional = pos.size * ct_val * price
                    lev = pos.leverage if pos.leverage > 0 else 1
                    required_margin = notional / lev
                    calculated_reserved += required_margin
            if pos.pending_ord_id:
                pending = get_pending_order_by_id(pos.pending_ord_id)
                if pending and pending.get("margin", 0) > 0:
                    calculated_reserved += float(pending["margin"])
            with self.trader._margin_lock:
                self.trader._reserved_margin = calculated_reserved
            log.debug(f"全量同步后重新计算 _reserved_margin: {calculated_reserved:.2f}")

            gs_set("last_state_sync", datetime.now(UTC).isoformat())

        try:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="state-sync") as ex:
                future = ex.submit(_sync_worker)
                future.result(timeout=20)
            log.info(f"全量同步完成 | 权益:{self.trader.latest_equity:.2f} 可用:{self.trader.latest_avail_bal:.2f}")
            # 打印持仓详情，便于排查
            _p = self.trader.pos
            if _p.side:
                log.info(f"📋 [{CFG.symbol}] 持仓: {_p.side} x{_p.leverage} {_p.size}张 @ {_p.entry_price:.2f} "
                         f"SL={_p.stop_loss:.2f} TP={_p.take_profit:.2f} 强平={_p.liq_price:.2f}")
            else:
                log.info(f"📋 [{CFG.symbol}] 无持仓")
        except FTTimeout:
            log.error("全量状态同步超时(20s)，跳过本轮")
        except Exception as e:
            log.error(f"状态同步失败: {e}")

    def _reset_pos(self):
        """重置仓位状态"""
        pos = self.trader.pos
        pos.side             = None
        pos.entry_price      = 0.0
        pos.size             = 0.0
        pos.leverage         = 1
        pos.peak_price       = 0.0
        pos.trailing_active  = False
        pos.pending_ord_id   = ""
        pos.partial_filled   = 0.0
        pos.liq_price        = 0.0
        pos.stop_loss        = 0.0
        pos.take_profit      = 0.0
        pos.trade_id         = None
        pos.open_time        = None
        pos.last_open_time   = None
        pos.moved_stop       = False
        pos.breakeven_triggered  = False
        pos.partial_tp_triggered = False
        pos.partial_tp_2_5R_triggered = False
        pos.trailing_dist_atr_mult = None
        pos.pyramid_plan      = None
        pos.pyramid_count     = 0
        pos.initial_size      = 0.0
        pos.initial_risk_usd  = 0.0
        if pos.sl_tp_algo_ids:
            log.debug(f"_reset_pos: 清除 {len(pos.sl_tp_algo_ids)} 个残留 algo ID: {pos.sl_tp_algo_ids}")
            pos.sl_tp_algo_ids = []
        pos.entry_market_mode = None
        pos.entry_rsi         = 0.0
        pos.entry_bb_pct      = 0.0
        pos.entry_atr_pct     = 0.0
        pos.ai_confidence     = 0.0
        pos.ai_reason         = None
        pos.exit_rsi          = 0.0
        pos.exit_bb_pct       = 0.0
        pos.exit_atr_pct      = 0.0
        pos.exit_market_mode  = None
        save_state_to_disk(pos)

        # 清理 DB pending 订单记录（幽灵仓位/算法单成交后残留）
        try:
            from common import get_all_pending_orders, delete_pending_order
            for pending in get_all_pending_orders():
                if pending.get("symbol") == CFG.symbol:
                    delete_pending_order(pending["ord_id"])
                    log.debug(f"_reset_pos: 清理 DB pending 记录 ord_id={pending['ord_id']} margin={pending.get('margin',0):.2f}U")
        except Exception as e:
            log.warning(f"_reset_pos: 清理 DB pending 订单失败: {e}")

        with self.trader._margin_lock:
            if self.trader._reserved_margin > 0:
                log.debug(f"_reset_pos: 清理残留 _reserved_margin={self.trader._reserved_margin:.2f}U")
                self.trader._reserved_margin = 0.0

        # 清除 SL/TP 去重记录，避免新仓位首次调整被误判为重复
        self.trader.position_exec._last_submitted_sl = 0.0
        self.trader.position_exec._last_submitted_tp = 0.0

    def sync_position(self, symbol: str = None):
        sym = CFG.symbol
        pos  = self.trader.pos
        lock = self.trader.lock

        log.debug(f"[{sym}] sync_position 开始（锁外网络请求）")
        try:
            res = self.okx.get_positions()
        except Exception as e:
            log.warning(f"[{sym}] sync_position 网络异常，跳过本次同步: {e}")
            return

        if res.get("code") != "0":
            log.warning(f"[{sym}] sync_position API 错误码={res.get('code')}，跳过本次同步（保留本地仓位状态）")
            return

        data = [d for d in res.get("data", []) if d.get("instId") == sym]

        with lock:
            log.debug(f"[{sym}] sync_position 已获取锁，更新内存状态")
            if not data:
                self._reset_pos()
                return
            p       = data[0]
            posSide = p.get("posSide")
            if posSide not in ["long", "short"]:
                self._reset_pos()
                return
            size  = abs(_safe_float(p.get("pos"),   0.0))
            entry = _safe_float(p.get("avgPx"), 0.0)
            if size > 0 and entry > 0:
                pos.side        = posSide
                pos.size        = size
                pos.entry_price = entry
                pos.liq_price   = float(p.get("liqPx", 0))
                # 从交易所恢复真实开仓时间（cTime=毫秒时间戳），避免重启后计时器归零
                _ctime_ms = p.get("cTime")
                if _ctime_ms:
                    pos.open_time = datetime.fromtimestamp(int(_ctime_ms) / 1000, tz=UTC)
                elif not pos.open_time:
                    pos.open_time = datetime.now(UTC)
                if pos.peak_price == 0.0:
                    pos.peak_price = entry
                save_state_to_disk(pos)
            else:
                self._reset_pos()

    def update_dynamic_params(self):
        """根据24小时胜率动态调整 dynamic_risk_factor"""
        now = datetime.now(UTC)
        if (now - self.trader._last_param_update).total_seconds() < 3600:
            return

        self.trader._last_param_update = now

        try:
            win_rate, n = get_recent_win_rate(n=25, min_sample=8)

            gs_set("last_24h_win_rate", win_rate)
            if n < 8:
                log.debug(f"胜率样本不足（近25笔仅{n}笔），last_24h_win_rate 保持中性先验 0.5")
                return

            if n < 10:
                log.debug(f"胜率已更新={win_rate:.1%}（近{n}笔），样本<10笔，dynamic_risk_factor 暂不调整")
                return

            if win_rate >= 0.60:
                target = 1.2
            elif win_rate >= 0.50:
                target = 1.0
            elif win_rate >= 0.40:
                target = 0.8
            else:
                target = 0.5

            old = self.trader.dynamic_risk_factor
            self.trader.dynamic_risk_factor = round(old + (target - old) * 0.2, 3)

            if abs(self.trader.dynamic_risk_factor - old) > 0.01:
                direction = "up" if self.trader.dynamic_risk_factor > old else "down"
                log.info(
                    f"动态风险因子更新: {old:.2f}→{self.trader.dynamic_risk_factor:.2f} "
                    f"（胜率={win_rate*100:.1f}% 样本=近{n}笔）"
                )
                if self.trader.dynamic_risk_factor <= 0.6:
                    _webhook(
                        "风险因子告警",
                        f"近期胜率={win_rate*100:.1f}%（近{n}笔），动态风险降至{self.trader.dynamic_risk_factor:.2f}x"
                    )
        except Exception as e:
            log.debug(f"动态参数更新失败: {e}")

    def check_daily_stop(self) -> bool:
        """日损熔断只看今天的已实现盈亏"""
        with self.trader.lock:
            if gs_get("daily_locked"):
                log.critical("今日亏损已锁定，系统暂停至明日")
                return False

            pause_until = gs_get("pause_until")
            if pause_until:
                try:
                    pause_dt = _parse_dt(pause_until)
                    if pause_dt and datetime.now(UTC) < pause_dt:
                        log.warning(f"滑点熔断暂停中，恢复时间: {pause_until}")
                        return False
                    else:
                        gs_set("pause_until", None)
                except Exception:
                    gs_set("pause_until", None)

            start_bal        = gs_get("start_balance") or (self.trader.latest_equity or 0.0)
            today_realized   = gs_get("today_realized_pnl", 0.0) or 0.0
            if start_bal and start_bal > 0:
                realized_loss_pct = today_realized / (start_bal + 1e-9)
            else:
                realized_loss_pct = 0.0

            if realized_loss_pct <= -CFG.max_daily_loss_pct:
                log.critical(
                    f"今日已实现亏损达 {realized_loss_pct*100:.2f}%"
                    f"（已实现:{today_realized:+.2f}U / 基准:{start_bal:.2f}U）"
                )
                _webhook(
                    "今日实现亏损上限",
                    f"已实现亏损: {today_realized:+.2f}U ({realized_loss_pct*100:.2f}%)"
                )
                log_event("daily_realized_loss_limit", {
                    "realized_pnl": today_realized,
                    "pct": realized_loss_pct,
                    "threshold": -CFG.max_daily_loss_pct
                })
                return False

            return True

    def _daily_reset_loop(self):
        """独立后台线程：每天 UTC 00:05 统一执行三项任务"""
        log.info("每日重置线程启动（UTC 00:05 统一执行）")
        while not self.trader._stop:
            try:
                now = datetime.now(UTC)
                tomorrow = now.date() + timedelta(days=1)
                next_run = datetime.combine(tomorrow, datetime.min.time(), tzinfo=UTC) + timedelta(minutes=5)
                secs = (next_run - now).total_seconds()
                time.sleep(max(secs, 0))
                if self.trader._stop:
                    break

                gs_set("today_opened_risk", 0.0)
                gs_set("today_realized_pnl", 0.0)
                gs_set("last_reset_date", now.date().isoformat())
                log.warning(f"[每日重置] UTC {now.isoformat()} 清零: today_opened_risk=0.0 today_realized_pnl=0.0")

                if CFG.enable_auto_case_pool:
                    try:
                        DB = get_db_manager()
                        DB.write_sync(
                            "DELETE FROM historical_cases WHERE created_at < datetime('now', '-' || ? || ' days')",
                            (CFG.case_max_age_days,)
                        )
                        DB.write_sync(
                            "DELETE FROM historical_cases WHERE id NOT IN (SELECT id FROM historical_cases ORDER BY quality_score DESC, created_at DESC LIMIT ?)",
                            (CFG.max_historical_cases,)
                        )
                        rows = DB.read("SELECT COUNT(*), SUM(quality_score) FROM historical_cases")
                        cnt = rows[0][0] if rows else 0
                        avg_q = rows[0][1] / cnt if rows and rows[0][1] and cnt > 0 else 0
                        log.info(f"[案例池] 维护完成，当前 {cnt} 条，均分 {avg_q:.1f}")
                    except Exception as e_maintain:
                        log.warning(f"[_daily_reset_loop] 案例维护异常: {e_maintain}")

            except Exception as e:
                log.warning(f"[_daily_reset_loop] 异常: {e}")
                time.sleep(3600)

    def _prewarm_atr_history(self):
        """启动时预热 ATR 历史"""
        if self.trader._atr_history and len(self.trader._atr_history) >= 30:
            return
        try:
            log.info("[ATR预热] 开始获取历史15m K线填充ATR历史...")
            raw = self.trader.signals.fetch_data("15m", 60, CFG.symbol)
            if not raw or len(raw) < 20:
                log.warning(f"[ATR预热] 历史数据不足（{len(raw) if raw else 0}根），跳过")
                return
            closes = [float(c[4]) for c in raw]
            trs = []
            for i in range(1, len(raw)):
                h, l, c_prev = float(raw[i-1][2]), float(raw[i-1][3]), closes[i-1]
                tr = max(abs(closes[i] - c_prev), abs(h - c_prev), abs(l - c_prev))
                trs.append(tr)
            period = min(14, len(trs))
            for start in range(len(trs) - period + 1):
                atr = sum(trs[start:start + period]) / period
                if atr > 0:
                    self.trader._atr_history.append(atr)
            if len(self.trader._atr_history) > 50:
                self.trader._atr_history = self.trader._atr_history[-50:]
            log.info(f"[ATR预热] 完成：已填充 {len(self.trader._atr_history)} 个 ATR 值")
        except Exception as e:
            log.warning(f"[ATR预热] 失败: {e}")


def _compute_1h_regime_score(ind_1h: Dict, price: float, prev_mode: str, funding: Dict) -> float:
    """计算 1H 时间框架的 regime_score（复用 get_market_mode 函数）。"""
    try:
        _, _score = get_market_mode(ind_1h, price, prev_mode, funding)
        return _score
    except Exception:
        return 0.5  # 异常时返回中性值，不拦截


# ============================================================
# 信号纯度评分（Signal Purity Score）
# 多维度确定性评估 0~1，过滤低质量信号，防止 AI 被矛盾输入干扰
# ============================================================
def _calc_signal_purity(ind_15m: dict, ind_1h: dict, ind_4h: dict,
                        vs_status: dict, depth: dict, funding: dict) -> float:
    """
    信号纯度评分 0~1：多维度确定性评估，过滤低质量信号。

    评分项（总分 1.0）：
      1. VSpike 量能方向纯度   (0.30) — 量能是否强烈且方向明确
      2. 多时间框架趋势一致性 (0.25) — 15m/1H/4H EMA+ADX 对齐
      3. 订单簿不平衡强度     (0.20) — OB imbalance 方向性
      4. 资金费率方向一致性   (0.10) — 连续打分
      5. CVD 累计方向确认     (0.15) — 吃单流量是否确认 VSpike 方向
    """
    score = 0.0

    # ── 1. VSpike 量能方向纯度 (0.30) ──
    # 不仅看当前 buy_pct，还结合 CVD 确认方向持续性
    vs_mult   = vs_status.get("mult", 1.0)
    vs_buy_pct = vs_status.get("buy_pct", 0.5)
    vs_dir    = vs_status.get("direction", "均衡")

    if vs_mult >= 4.0:
        # 量能充足时，方向偏度决定纯度
        dir_extreme = max(vs_buy_pct, 1.0 - vs_buy_pct)  # 0.5~1.0
        vs_base = (dir_extreme - 0.5) / 0.5              # 归一化到 0~1
        score += 0.30 * min(1.0, vs_base)
    else:
        # 量能不足时，给少量基础分（不完全否定）
        score += 0.10 * (vs_mult / 4.0)  # 0~0.10

    # ── 2. 多时间框架趋势一致性 (0.25) ──
    _trend_score, _trend_dir = get_trend_alignment_score(ind_15m, ind_1h, ind_4h)
    score += _trend_score * 0.25
    # 震荡极高波动（BB>0.7）额外惩罚 5%
    _bb_w = ind_15m.get("bb_width", 0.0)
    if _bb_w > 0.7:
        score = max(0, score - 0.05)

    # ── 3. 订单簿不平衡强度 (0.20) ──
    imbal = depth.get("imbalance", 0.0) if depth else 0.0
    # |imbalance| > 0.5 即给满分
    score += min(0.20, abs(imbal) * 0.4)

    # ── 4. 资金费率方向一致性 (0.10) — 连续打分 ──
    fund_rate = funding.get("funding_rate", 0.0) if funding else 0.0
    if abs(fund_rate) > 0.0001:
        # |fund| 0.0001→0, 0.0005→0.05, ≥0.001→0.10
        score += min(0.10, abs(fund_rate) / 0.001 * 0.10)

    # ── 5. CVD 累计方向确认 (0.15) ──
    # CVD 确认 VSpike 方向 → 加分；CVD 与 VSpike 反向 → 不给分
    cvd = vs_status.get("cum_delta", 0)
    cvd_abs = abs(cvd)
    if cvd_abs > 50:  # 有方向性流量
        cvd_buy = (cvd > 0)
        vs_buy_dir = (vs_buy_pct > 0.55)
        if (cvd_buy and vs_buy_dir) or (not cvd_buy and not vs_buy_dir):
            score += 0.15  # CVD 确认 VSpike 方向
        # CVD 与 VSpike 方向冲突 → 不给分（但不扣，避免双重惩罚）
    else:
        # CVD 微弱，给一半基础分
        score += 0.07 * (cvd_abs / 50.0)

    return min(1.0, max(0.0, score))


# ============================================================
# 核心交易引擎
# ============================================================
class ETHTrader:
    """
    OKX ETH-USDT-SWAP 量化交易机器人。

    核心组件：
      - SmartAIConsultant：AI 决策（双温度投票 + RAG + 策略委员会）
      - OkxTrader：OKX API 封装（下单/查询/WS）
      - 风控循环（_risk_control_loop）：独立线程，30-180秒动态轮询

    线程模型：
      - 主线程：run() 循环
      - 风控线程：_risk_control_loop()，每轮检查持仓状态
      - WebSocket 线程：接收实时价格/持仓更新
      - Webhook 线程：异步发送通知
      - 滑点写入线程：_slippage_writer_loop，后台批量写文件

    状态管理：
      - Position dataclass：持仓快照
      - _reserved_margin：下单中但未成交的保证金
      - pending_ord_id：挂单 ID，崩溃后由 _handle_pending_order 恢复

    重要锁：
      - self.lock：操作锁，_do_open/_close/_adjust_sl_tp 共用
      - self._open_lock：开仓锁，防止余额超额占用
      - self._margin_lock：保证金读写专用锁
    """
    def __init__(self, ai_client: OpenAI):
        self.trader = OkxTrader()  # 先创建 OkxTrader
        self.trader.eth_trader = self  # 让 OkxTrader 能反向访问 ETHTrader._atr_history
        self.ai     = SmartAIConsultant(ai_client, trader=self.trader)  # 再创建 AI，传入 trader

        # ── 单品种状态 ────────────────────────────────────────────────────
        self.pos   = Position()
        self.lock  = threading.RLock()
        # WS 价格缓存
        self._price_val:      float = 0.0
        self._price_ts:       float = 0.0
        self._mark_price_val: float = 0.0
        self._mark_price_ts2: float = 0.0
        self._mark_warn_ts:   float = 0.0
        self._price_warn_ts:  float = 0.0
        # SL/TP 算法单成交检测缓存（Bug 修复：WS positions 清仓前保存，供 orders channel 使用）
        self._sl_tp_pending_trade_id:  Optional[int] = None
        self._sl_tp_pending_algo_ids:  list           = []
        self._sl_tp_pending_leverage:  int            = 1
        # SL/TP 缓存专用锁，防止 orders/positions channel 竞态
        self._sl_tp_cache_lock: threading.Lock = threading.Lock()
        # 降级为 hold 日志节流（同一条消息 120s 内只打一次 INFO）
        self._last_downgrade_log_ts: float = 0.0
        # 开仓时间戳，用于 WS pos=0 延迟消息保护窗口
        self._last_open_ts: float = 0.0
        # 翻转计数器（10 分钟滑动窗口，仅监控告警不拦截）
        self._flip_timestamps: list = []

        # ── 缓存 ──────────────────────────────────────────────────────────
        _empty_cache = lambda: {"data": None, "time": datetime.min.replace(tzinfo=UTC)}
        self.funding_cache     = _empty_cache()
        self.key_levels_cache  = _empty_cache()
        self.funding_history:  List = []
        # 共享缓存
        self.news_cache      = _empty_cache()
        self.fg_cache        = _empty_cache()
        self.sentiment_cache = _empty_cache()
        self.macro_cache     = {"full": "", "short": "", "time": datetime.min.replace(tzinfo=UTC)}

        # ── 动态风险因子（根据近7日胜率自动调整，每小时更新）────────────────
        self.dynamic_risk_factor:  float    = 1.0
        self._last_param_update:   datetime = datetime.min.replace(tzinfo=UTC)
        self._liq_warn_ts:         float    = 0.0
        self._market_mode:          str      = "趋势"
        self._prev_ls_ratio:        Optional[float] = None
        self._prev_taker_ratio:     Optional[float] = None
        self._atr_val:              float    = 0.0
        self._atr_history:          List     = []   # 最近100个ATR值，用于RAG波动率分位计算
        self._ob_rolling_avg:       List[float] = []  # 最近N帧订单簿均档容量，用于动态冰山墙基准
        self._last_price_val:       float    = 0.0
        self._last_15m_raw:         List     = []
        self._last_3m_raw:          List     = []
        self._last_1h_raw:          List     = []
        self._ind_15m_cache:        Tuple[float, Dict, str] = (0.0, {}, "")
        self._ind_3m_cache:         Tuple[float, Dict, str] = (0.0, {}, "")
        self._ind_4h_cache:         Tuple[float, Dict, str] = (0.0, {}, "")  # P1: 4H指标600s缓存
        self._last_ob_imbalance:    float = 0.0  # 上轮 OB 失衡度，供 get_dynamic_interval 次轮读取
        # raw K 线缓存：bar → (data, fetch_monotonic, last_candle_ts)
        # TTL 按周期设定：3m=45s, 15m=120s, 1H=300s, 4H=600s
        self._raw_kline_cache: Dict[str, tuple] = {}
        self._RAW_KLINE_TTL: Dict[str, int] = {"3m": 45, "15m": 120, "1H": 300, "4H": 600}
        # Hurst exponent 缓存：symbol → 近30根15m收益率序列（用于 Regime 复合化评分）
        self._returns_cache: Dict[str, np.ndarray] = {}
        self._last_adjust_time:     datetime = datetime.min.replace(tzinfo=UTC)
        self._last_trailing_adjust: datetime = datetime.min.replace(tzinfo=UTC)

        # ── 账户余额 ─────────────────────────────────────────────────────
        self.latest_equity:    float = 0.0
        self.latest_avail_bal: float = 0.0

        # ── AI 异步决策缓存 ───────────────────────────────────────────────
        self._ai_cache:        Optional[Dict] = None
        self._ai_cache_ts:     float = 0.0     # AI 缓存时间戳（monotonic），用于判断缓存是否过期
        self._ai_cache_lock:   threading.Lock = threading.Lock()
        self._ai_hash:         str  = ""
        self._ai_running_flag: bool = False
        self._ai_gen:          int  = 0     # generation counter，防止旧 worker 覆写新决策
        self._ai_urgent_gen:   int  = 0     # 紧急请求的最高 generation
        self._ai_thread:       Optional[threading.Thread] = None
        self._last_ai_request_time:   float = 0.0
        self._last_ai_decision_time:  float = 0.0
        self._last_ai_decision_price: float = 0.0
        self._last_ai_decision_rsi:   float = 50.0
        self._last_ai_decision_macd:  float = 0.0
        self._last_ai_rsi_bkt: Optional[str] = None  # 宽分桶快照（漂移容忍用）
        self._last_ai_bb_zone:  Optional[str] = None  # BB% 三分桶快照
        self._prev_indicators:  Dict = {}
        self._last_cvd_reset_ts: str = ""  # 上次CVD重置的15m蜡烛时间戳（用于检测新K线周期）

        self._stop = False
        self._zero_pos_seen_ts: float = 0.0   # 幽灵仓位检测：WS连续返回pos=0的时间戳
        # ── 人工审核控制 ──────────────────────────────────────────────────
        self._ai_blocked: bool = False              # /block_ai 禁用 AI 决策
        # ── 决策摘要队列（最近10次，供 /health 和 /ai/summaries 消费）──────
        self._ai_summaries: deque = deque(maxlen=10)
        # ── 告警去重时间戳 dict（monotonic），防循环告警 ──────────────────
        self._last_alert_ts: Dict[str, float] = {}
        # ── 打板信号冷却时间戳（防同一突破在冷却期内重复触发）────────────
        self._last_breakout_ts: Dict[str, float] = {"up": 0.0, "down": 0.0}
        # ── VSpike 非极端事件冷却（防震荡市密集触发浪费 Token）────────────
        # 格式：{symbol: monotonic时间戳}
        self._last_vspike_break_ts: Dict[str, float] = {}
        # AI 平仓协调标志：防止 AI 平仓与追踪止损重复下单
        self._ai_close_pending_until: float = 0.0  # monotonic 时间戳
        # ── 决策去重签名（防相同决策每轮输出完整 thought_process）──────────
        self._last_decision_sig: Dict[str, str] = {}
        # ── 连续 hold 计数 + AI 建议的下次调用间隔（秒）────────────────────
        self._consecutive_hold: int = 0
        self._ai_hold_wait:     int = -1   # -1=未设定，0=AI说立即，其他=指定秒数
        # ── P2: L1 重复 hold 跳过缓存（120s 内市场状态不变且 L1 仍 hold → 跳过 L1）──
        self._last_l1_hold_ts:   float = 0.0   # 上次 L1 hold 的 monotonic 时间戳
        self._last_l1_hold_hash: str = ""      # 上次 L1 hold 时的 input_sig
        # P1-3: 日志限频（key → last_log_monotonic），避免高频循环刷屏
        self._log_throttle: Dict[str, float] = {}
        # ── AI force_wakeup 标记（hold 时 AI 可主动要求下轮重评）───────────
        self._last_force_wakeup: Dict[str, bool] = {}   # symbol -> bool
        # ── AI 动态 RSI 唤醒阈值（hold 时 AI 可建议 next_wakeup_rsi）────────
        # None 表示未设定，使用默认静态阈值（40~60）
        self._next_wakeup_rsi: Optional[tuple] = None  # (low, high)
        # 开仓锁：防止并发读取相同余额导致超额下单
        self._open_lock = threading.Lock()
        # 已预留保证金（下单中但尚未反映到 WS 余额的部分）
        self._reserved_margin: float = 0.0
        self._margin_lock = threading.Lock()  # 中优先级5 Fix：保证金操作专用锁
        # AI 并发限流信号量：同一时刻只允许1个品种请求 DeepSeek
        self._ai_semaphore = threading.Semaphore(1)
        # AI 熔断器：连续5次失败后暂停30分钟，降级为规则引擎
        self._ai_failure_count: int = 0
        self._ai_circuit_broken_until: float = 0.0  # monotonic 时间戳
        # 启动时间戳：用于启动冷却期（防止首次扫描立即开仓）
        self._boot_ts: float = time.monotonic()
        self._boot_dt: datetime = datetime.now(UTC)  # 用于识别重启恢复持仓
        # 启动冷却期间被拦截的信号跟踪（冷却过期后要求信号/行情已变化）
        self._post_cooldown_check: Dict[str, Dict] = {}  # {symbol: {"reason": str, "seen_ts": float}}

        # ── 同价位止损冷却 + 连续被洗记忆 ──────────────────────────────
        self._last_sl_price: float = 0.0          # 上次止损价格
        self._last_sl_time: float = 0.0            # 上次止损时间(monotonic)
        self._wash_count_at_price: int = 0         # 同价位区间连续被洗次数

        # ── 双管道 AI 调用优化：空仓决策 vs 持仓管理独立间隔 ────────────
        self._last_entry_decision_ts:  float = 0.0   # 上次空仓决策时间戳(monotonic)
        self._last_holding_adjust_ts:   float = 0.0   # 上次持仓调整时间戳(monotonic)

        # ── Phase 1: AIGatekeeper ────────────────────────────
        self._ai_gate = AIGatekeeper()
        self._conviction = ConvictionScorer()
        # ── 千问仲裁触发器 ───────────────────────────────────────
        self._arbitration = ArbitrationTrigger()
        # ── AI 表现追踪全局状态初始化（占位，实际初始化在 load_state_from_disk 之后）──
        # 见下方 load_state_from_disk() 后的重置逻辑
        # ── Phase 2: PositionManager ─────────────────────────
        self._pos_mgr = PositionManager(self.pos, self.lock)
        # ── Phase 3: StateMachine + EventBus ─────────────────
        self._state_machine = TradingStateMachine()
        # 根据当前仓位恢复状态
        if self.pos.side:
            self._state_machine.force_state(TradingState.HOLDING, "启动恢复：检测到持仓")
        self._event_bus = get_event_bus()
        # 订阅交易事件 → Telegram 通知
        self._event_bus.subscribe_all(self._on_trading_event)

        # ── 新模块初始化 ───────────────────────────────────────────────
        # SignalsModule：指标计算、数据获取、规则引擎
        self.signals = SignalsModule(self.trader, CFG)
        # FastLaneModule：AI缓存与快速决策通道
        self.fast_lane = FastLaneModule(self, self.signals, self.ai)
        # RiskGuard：风控循环、追踪止损
        self.risk_guard = RiskGuard(self)
        # PositionExec：开仓/平仓执行
        self.position_exec = PositionExec(self)
        # StateManager：状态同步
        self.state = StateManager(self)
        # 注册重置回调（依赖 StateManager 已初始化）
        self._pos_mgr.set_reset_callback(self.state._reset_pos)

        # 秒级成交量突增检测器（零额外 WS 连接，复用公共 WS trades 频道）
        self.vspike = VolumeSpikeDetector()

        self.ws_client = OkxWebSocket(
            self.trader,
            on_ticker_callback=self._on_ws_ticker,
            on_private_callback=self._on_ws_private,
            on_mark_price_callback=self._on_ws_mark_price,
            on_trades_callback=self._on_ws_trades,
            eth_trader=self,
        )
        self.watchdog   = Watchdog()
        self.reporter   = DailyReportModule(self.trader, log, ai_client=ai_client)
        self.postmortem = TradePostmortem(ai_client)
        # latest_price 改为 (value, monotonic_timestamp) 元组，支持陈旧检测
        self._mark_price_ts:  float = 0.0
        self._mark_price_degraded: bool = False  # Mark Price REST 降级标记

        # ── AI 异步决策缓存 ────────────────────────────────────────────────
        # AI 决策在独立后台线程运行，主循环消费最新缓存结果。
        # 好处：
        #   1. 极端行情下 AI 超时 40s 不会延迟行情数据采集和下一轮判断
        #   2. 风控线程完全不受 AI 调用影响（本来就独立，但现在连主循环也不阻塞）
        #   3. AI 新结果就绪时立刻可被下一轮 run_once 消费，响应更快
        self._ai_running: bool = False
        self._ai_thread: Optional[threading.Thread] = None

        # ── 主循环统计计数器（Task4: 每10轮输出摘要）────────────────────────
        self._run_counter: int = 0
        self._last_summary_time: Optional[datetime] = None

        # ── 启动初始化 ────────────────────────────────────────────────────
        SYM = CFG.symbol
        load_state_from_disk(self.pos)

        # ── AI 绩效历史清洗（必须在 load_state_from_disk 之后，否则会被磁盘数据覆盖）──
        _old_hist = gs_get("ai_win_history", None)
        if _old_hist and len(_old_hist) >= 5:
            log.info(f"🔄 策略升级：清除旧 AI 绩效历史({len(_old_hist)}笔)，重置权重为 0.75")
            gs_set("ai_win_history", [])
            gs_set("ai_decision_conf_history", [])
            gs_set("ai_recent_win_rate", 0.5)
            gs_set("ai_weight_mult", 0.75)
        else:
            if _old_hist is None:
                gs_set("ai_win_history", [])
            if gs_get("ai_decision_conf_history", None) is None:
                gs_set("ai_decision_conf_history", [])
            if gs_get("ai_recent_win_rate", None) is None:
                gs_set("ai_recent_win_rate", 0.5)
            if gs_get("ai_weight_mult", None) is None:
                gs_set("ai_weight_mult", 1.0)

        self.trader.fetch_contract_sizes()
        # __init__ 中显式调用，确保热更新参数在交易开始前生效
        _load_dynamic_config()
        self.ai.tick_size = self.trader.tick_sizes.get(SYM, 0.01)

        init_bal_full = self.trader.get_account_balance_full()
        init_equity   = init_bal_full["equity"]
        init_avail    = init_bal_full["avail_bal"]
        self.latest_equity    = init_equity
        self.latest_avail_bal = init_avail
        if gs_get("start_balance") == 0.0:
            gs_set("start_balance", init_equity)

        # 启动时自检：清除误锁
        if gs_get("daily_locked", False):
            start_bal = gs_get("start_balance", init_equity)
            threshold = start_bal * (1 - CFG.max_daily_loss_pct)
            if init_equity >= threshold:
                log.warning(
                    f"⚠️ 检测到持久化的 daily_locked=True，但当前权益 {init_equity:.2f} "
                    f"≥ 阈值 {threshold:.2f}，判定为误锁，自动解除"
                )
                gs_set("daily_locked", False)
                _webhook("🔓 误锁自动解除", f"权益 {init_equity:.2f} USDT 健康，已恢复交易")
                log_event("daily_lock_cleared", {"equity": init_equity})
            else:
                log.critical(f"🚨 daily_locked=True 且权益 {init_equity:.2f} < 阈值 {threshold:.2f}")

        log.info(f"💼 基准权益: {gs_get('start_balance'):.2f} | 当前权益: {init_equity:.2f} | 可用: {init_avail:.2f} USDT")
        log.info(f"📋 交易品种: {CFG.symbol}")
        log_event("system_start", {"equity": init_equity, "symbol": CFG.symbol, "state": GLOBAL_STATE})

        self.ws_client.start()
        self.risk_thread = threading.Thread(target=self.risk_guard._risk_control_loop, daemon=True)
        self.risk_thread.start()
        threading.Thread(target=self.state._daily_reset_loop, daemon=True, name="daily-reset").start()
        threading.Thread(target=self.monitoring_loop, daemon=True, name="monitor").start()
        threading.Thread(target=watchdog_loop, args=(self.watchdog,), daemon=True).start()
        # 启动后等待 WS 建立再做全量同步，清除可能的幽灵仓位
        time.sleep(3)
        self.state._full_state_sync()
        # 启动时恢复崩溃中间态订单
        self.position_exec._handle_pending_order()
        # 启动时预热 ATR 历史（确保 RAG 波动率分位在首次决策时即有效）
        self.state._prewarm_atr_history()

        # ── 快速决策拒绝反馈缓存 ──
        # ── AI 最近方向性信号跟踪 ──
        # 防止 Path B 用规则引擎信号覆盖 AI 的方向性判断
        # 格式: {"sym": (action, monotonic_ts)}
        self._last_ai_directional: Dict[str, tuple] = {}
    # ============================================================
    # 主交易循环（从 trading_loop.py 合并）
    # ============================================================
    def run(self):
        """主交易循环"""
        log.info("🚀 主交易循环启动")
        while not self._stop:
            try:
                self.run_once()
                sleep_sec = self.get_dynamic_interval()
                # 用 spike_event.wait 替代 time.sleep，VSpike 触发时立即唤醒
                self.vspike.spike_event.wait(timeout=sleep_sec)
                self.vspike.reset_event()
            except Exception as e:
                log.exception(f"主循环异常: {e}")
                time.sleep(5)

    def get_dynamic_interval(self) -> int:
        """动态轮询间隔（OB失衡 × 成交量 × 关键位 × BB宽度 四层联动）"""
        pos = self.pos
        has_pos = bool(pos.side)

        # ── Step 0: 极端快车道（最高优先级）──────────────────────────────
        # vol_surge ≥ 2.0 或 |OB失衡| ≥ ob_fastlane_thresh → 15s 轮询
        try:
            _, ind_15m, _ = self._ind_15m_cache
            if ind_15m and ind_15m.get("_valid"):
                vol_surge = float(ind_15m.get("vol_surge", 1.0))
                ob_imbal  = abs(self._last_ob_imbalance)
                if (vol_surge >= CFG.vspike_extreme_thresh
                        or ob_imbal >= CFG.ob_fastlane_thresh):
                    self._consecutive_hold = 0
                    self._ai_hold_wait = -1
                    log.debug(f"⚡ [{CFG.symbol}] 极端快车道唤醒 VSpike={vol_surge:.2f} OB={ob_imbal:.2f}")
                    return CFG.check_interval_extreme
        except Exception:
            pass

        # ── Step 1: 接近关键价位 → 快车道 ────────────────────────────────
        price = self._get_price(CFG.symbol)
        if price > 0:
            kl = self.key_levels_cache.get("data") or {}
            if kl.get("_valid"):
                all_lvls = []
                for item in kl.get("resistances", []) + kl.get("supports", []):
                    lp = float(item.get("price", item) if isinstance(item, dict) else item)
                    all_lvls.append(lp)
                if kl.get("pivot"):
                    all_lvls.append(float(kl["pivot"]))
                for lp in all_lvls:
                    if abs(price - lp) / price <= CFG.level_proximity_thresh:
                        return CFG.check_interval_level

        # ── Step 1.5: 高波动快车道（成交量或OB轻度失衡）──────────────────
        try:
            _, ind_15m, _ = self._ind_15m_cache
            if ind_15m and ind_15m.get("_valid"):
                vol_surge = float(ind_15m.get("vol_surge", 1.0))
                ob_imbal  = abs(self._last_ob_imbalance)
                if vol_surge >= CFG.v_spike_mult_thresh or ob_imbal >= CFG.ob_fastlane_imbalance:
                    self._consecutive_hold = 0
                    self._ai_hold_wait = -1
                    log.debug(f"📊 [{CFG.symbol}] 高波动快车道 VSpike={vol_surge:.2f} OB={ob_imbal:.2f}")
                    return CFG.check_interval_level
        except Exception:
            pass

        # ── Step 2: AI 静默间隔（仅空仓）──────────────────────────────────
        if not has_pos and self._ai_hold_wait >= 0:
            wait = self._ai_hold_wait
            self._ai_hold_wait = -1
            return wait

        # ── Step 3: 布林带宽度判断 ───────────────────────────────────────
        bb_width = None
        try:
            _, ind_15m, _ = self._ind_15m_cache
            if ind_15m and ind_15m.get("_valid"):
                bb_width = ind_15m.get("bb_width")
        except Exception:
            pass

        if bb_width is not None:
            if bb_width > CFG.bb_width_wide_thresh:
                return max(20, (CFG.check_interval_hold if has_pos else CFG.check_interval_empty) // 2)
            if bb_width < CFG.bb_width_narrow_thresh:
                if has_pos:
                    # 震荡市持仓：低波动无需频繁轮询，延长至120s减少噪声
                    return 120 if self._market_mode in ("震荡", "震荡激进") else 30
                mult = min(5, 1 + self._consecutive_hold // 3)
                return min(300, int(60 * mult))  # 慢车道上限从 600s → 300s

        # ── Step 4: 默认间隔 ─────────────────────────────────────────────
        _base = CFG.check_interval_hold if has_pos else CFG.check_interval_empty

        # ── Step 4b: 空仓连续 hold → 拉长间隔，减少空烧 ─────────────────
        # 连续 2 次 L1 hold + 无 VSpike/快车道信号 → 120s 慢扫
        if not has_pos and self._consecutive_hold >= 2:
            log.debug(f"⏸️ [{CFG.symbol}] 空仓连续{self._consecutive_hold}次hold，拉长轮询至120s")
            return 120

        return _base

    def monitoring_loop(self):
        """每10分钟检查一次近期交易表现及 WS 连接健康度"""
        log.info("📡 异步监控线程启动（每10分钟）")
        time.sleep(60)
        while not self._stop:
            try:
                now = datetime.now(UTC)
                now_mono = time.monotonic()

                # ── WS 连接健康检查 ─────────────────────────────────────
                price_age = now_mono - self._price_ts if self._price_ts > 0 else None
                mark_age  = now_mono - self._mark_price_ts2 if self._mark_price_ts2 > 0 else None

                # 若最后更新超过 120 秒（2 倍心跳），判定 WS 连接断裂
                if price_age and price_age > 120:
                    log.warning(f"⚠️ WS ticker 连接老化（{price_age:.0f}s 无更新），尝试重连")
                    if hasattr(self.ws_client, '_reconnect_public'):
                        self.ws_client._reconnect_public()

                if mark_age and mark_age > 120:
                    log.warning(f"⚠️ WS mark-price 连接老化（{mark_age:.0f}s 无更新），尝试重连")
                    if hasattr(self.ws_client, '_reconnect_public'):
                        self.ws_client._reconnect_public()

                # ── 交易表现检查 ────────────────────────────────────────
                trades = get_trades_in_range(now - timedelta(hours=24), now)
                closed = [t for t in trades if t.get("pnl") is not None]

                if len(closed) >= 3:
                    recent = sorted(closed, key=lambda x: x.get("ts", ""), reverse=True)[:5]
                    loss_streak = sum(1 for t in recent if t.get("pnl", 0) < 0)

                    if loss_streak >= 3:
                        key = "recent_loss_streak"
                        last = self._last_alert_ts.get(key, 0)
                        if now_mono - last >= 3600:
                            self._last_alert_ts[key] = now_mono
                            avg_loss = sum(t.get("pnl_pct", 0) for t in recent if t.get("pnl_pct", 0) < 0) / max(loss_streak, 1)
                            _webhook(
                                "🚨 近期连续亏损告警",
                                f"最近5笔交易中连续亏损 {loss_streak} 笔，平均亏损 {avg_loss*100:.2f}%\n"
                                f"建议检查策略是否失效",
                                level=2
                            )

                time.sleep(600)
            except Exception as e:
                log.exception(f"监控线程异常: {e}")
                time.sleep(60)

    # ============================================================
    # WebSocket 回调
    # ============================================================
    def _on_ws_ticker(self, message):
        """公共 WS 回调：tickers（所有品种）"""
        try:
            if "arg" in message and "data" in message:
                channel = message["arg"].get("channel")
                sym     = message["arg"].get("instId", "")
                if channel == "tickers" and sym == CFG.symbol:
                    if message.get("data"):
                        self._price_val = float(message["data"][0]["last"])
                    self._price_ts  = time.monotonic()
        except Exception as e:
            log.error(f"Ticker WS 消息处理异常: {e}")

    def _on_ws_mark_price(self, message):
        """公共 WS 标记价回调（所有品种）"""
        try:
            if "arg" in message and "data" in message:
                channel = message["arg"].get("channel")
                sym     = message["arg"].get("instId", "")
                if channel == "mark-price" and sym == CFG.symbol:
                    if message.get("data"):
                        self._mark_price_val = float(message["data"][0]["markPx"])
                    self._mark_price_ts2  = time.monotonic()
        except Exception as e:
            log.error(f"MarkPrice WS 消息处理异常: {e}")

    def _on_ws_trades(self, message):
        """公共 WS trades 回调 → 喂给 VolumeSpikeDetector"""
        try:
            trades = message.get("data", [])
            for t in trades:
                sz   = float(t.get("sz", 0))
                side = t.get("side", "")
                if sz > 0 and side:
                    self.vspike.record_trade(sz, side)
        except Exception as e:
            log.debug(f"VSpike trades 解析异常: {e}")

    def _get_conviction_open_thresh(self) -> float:
        """根据市场模式返回开仓 ConvictionScore 门槛：震荡 57 / 趋势 53。"""
        if self._market_mode in ("震荡", "震荡激进"):
            return CFG.osc_conviction_open_min
        return CFG.conviction_open_min

    def _check_aggressive_conflict_cooldown(self) -> bool:
        """AGGRESSIVE 冲突冷却：多空冲突后方向未变，跳过 AI 调用复用 hold。
        返回 True = 应跳过 AI，复用上次冲突 hold。
        """
        if self._ai_gate.entry_fasttrack_mult < CFG.vspike_priority_threshold:
            return False  # 非 AGGRESSIVE 模式

        _conflict_ts = gs_get("last_aggressive_conflict_ts", 0)
        if _conflict_ts == 0:
            return False  # 没有历史冲突记录

        _cooldown = getattr(CFG, "aggressive_conflict_cooldown", 180.0)
        elapsed = time.monotonic() - _conflict_ts
        if elapsed > _cooldown:
            return False  # 冷却过期

        # 检查 VSpike 方向是否相同
        _vs_dir_old = gs_get("last_aggressive_conflict_dir", "")
        _current_vs = self.vspike.get_status()
        _current_dir = _current_vs.get("direction", "")

        if _vs_dir_old and _current_dir and _vs_dir_old == _current_dir:
            log.debug(
                f"🔒 AGGRESSIVE冲突冷却中：{_vs_dir_old} 方向未变 "
                f"({elapsed:.0f}s/{_cooldown:.0f}s)"
            )
            return True

        return False  # 方向变了，需要重新判断

    def _should_log(self, key: str, min_interval: float = 120.0) -> bool:
        """P1-3: 日志限频。同一 key 在 min_interval 秒内只输出一次 INFO，其余降级 DEBUG。"""
        now = time.monotonic()
        last = self._log_throttle.get(key, 0.0)
        if now - last >= min_interval:
            self._log_throttle[key] = now
            return True
        return False

    def _should_skip_ai_request(self, symbol: str, ind_15m: Dict, ind_1h: Dict, current_price: float) -> bool:
        """
        检查是否应该跳过 AI 请求。
        静默拦截已全面禁用，持仓/空仓均由缓存 TTL 控制频率。
        保留：AI 熔断、VSpike 冷却/反转检测、浮亏/浮盈告警。
        返回 True 表示跳过，False 表示继续调 AI。
        """
        # AI 熔断器
        if self._ai_gate.circuit_broken:
            log.warning(f"🛑 [{symbol}] AI 熔断中，跳过 AI 请求")
            return True

        # AI force_wakeup —— 清除缓存立即重评
        if self._last_force_wakeup.get(symbol, False):
            log.info(f"⚡ [{symbol}] 上轮 AI force_wakeup，清除缓存重评")
            self._last_force_wakeup[symbol] = False
            self.fast_lane._clear_ai_cache(symbol=symbol, clear_last=True)
            return False

        # VSpike 突增 → 分级冷却清缓存
        _vs_status = self.vspike.get_status()
        if _vs_status.get("is_spike"):
            _vs_mult = _vs_status.get("mult", 1.0)
            _now_mono_vs = time.monotonic()

            # ── 分级冷却：极端→零延迟 / 中强→120s / 弱→180s ──
            _last_clear = gs_get("last_vspike_cache_clear_ts", 0.0)
            if _vs_mult >= 8.0:
                _min_gap = 0.0
                _tag = "极端"
            elif _vs_mult >= 5.0:
                _min_gap = 120.0
                _tag = "中强"
            else:
                _min_gap = 180.0
                _tag = "弱"

            if _now_mono_vs - _last_clear < _min_gap:
                log.debug(
                    f"⏸️ [{symbol}] VSpike {_vs_mult:.1f}x ({_tag}) 清缓存冷却中 "
                    f"（需{_min_gap:.0f}s，已过{(_now_mono_vs - _last_clear):.0f}s），跳过清缓存，AI 请求继续"
                )
                # 只跳过清缓存，AI 请求正常走
                return False

            # ── 全局最小间隔保护：任何情况下清缓存后 45s 内不允许再次清缓存 ──
            _last_ai_req = self._ai_gate.last_request_ts
            if _last_ai_req > 0 and (_now_mono_vs - _last_ai_req) < 45.0:
                log.debug(
                    f"⏸️ [{symbol}] VSpike {_vs_mult:.1f}x 清缓存被全局最小间隔保护 "
                    f"（距上次AI调用 {(_now_mono_vs - _last_ai_req):.0f}s < 45s），跳过清缓存，AI 请求继续"
                )
                # 只跳过清缓存，AI 请求正常走
                return False

            # ── 反转检测（震荡市防双向来回砍）──
            _rev_min_mult = 4.0 if self._market_mode in ("震荡", "震荡激进") else 3.0
            _reversal = self.vspike.has_recent_reversal(lookback_secs=600.0, min_mult=_rev_min_mult)
            if _reversal["reversed"] and _vs_mult < 8.0:
                _first_mult = _reversal.get("first_mult", 0.0)
                _magnitude_surge = (_vs_mult >= _first_mult * 1.8)
                _has_pos = bool(self.pos.side)
                _dir_consistent = False
                if _has_pos:
                    _new_dir_is_long = (_reversal["second_dir"] == "买方主导")
                    _pos_is_long = (self.pos.side == "long")
                    _dir_consistent = (_new_dir_is_long == _pos_is_long)
                else:
                    _dir_consistent = True
                if _magnitude_surge and _dir_consistent:
                    log.info(
                        f"🚦 [{symbol}] 反转检测豁免：新VSpike {_vs_mult:.1f}x ≥ "
                        f"{_reversal['first_mult']:.1f}x×1.8，放行决策"
                    )
                else:
                    if self._market_mode in ("震荡", "震荡激进"):
                        log.warning(
                            f"🔄 [{symbol}] VSpike 反转检测：{_reversal['detail']} | "
                            f"震荡市降级 hold，避免双向来回砍"
                        )
                        return True
                    else:
                        log.info(
                            f"🔄 [{symbol}] VSpike 反转警告（趋势市放行）：{_reversal['detail']}"
                        )

            gs_set("last_vspike_cache_clear_ts", _now_mono_vs)
            log.info(f"🔥 [{symbol}] VSpike {_vs_mult:.1f}x ({_tag}) → 清除缓存刷新决策")
            self.fast_lane._clear_ai_cache(symbol=symbol, clear_last=True)
            if _vs_mult >= CFG.vspike_priority_threshold:
                self._ai_gate.mark_entry_fasttrack(_vs_mult)
                log.info(
                    f"[AIGatekeeper] Mode: AGGRESSIVE | "
                    f"Reason: VSpike={_vs_mult:.1f}x >= {CFG.vspike_priority_threshold:.0f}x | "
                    f"Bypassing Cache: True"
                )
            return False

        # 浮亏/浮盈告警（仅持仓）
        pos = self.pos
        if pos and pos.side and pos.entry_price > 0 and current_price > 0:
            if pos.side == "long":
                _pos_pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            else:
                _pos_pnl_pct = (pos.entry_price - current_price) / pos.entry_price
            _now_mono_alert = time.monotonic()
            if _pos_pnl_pct <= CFG.silence_force_wakeup_loss_pct:
                _cooldown = CFG.silence_wakeup_alert_cooldown
                if _now_mono_alert - self._last_alert_ts.get("wakeup_loss", 0) >= _cooldown:
                    self._last_alert_ts["wakeup_loss"] = _now_mono_alert
                    log.warning(f"🚨 [{symbol}] 浮亏 {_pos_pnl_pct*100:.2f}%，当前价:{current_price}")
                    _webhook(
                        f"🚨 [{CFG.symbol}] 浮亏告警",
                        f"方向:{pos.side} 浮亏:{_pos_pnl_pct*100:.2f}% 当前价:{current_price}"
                    )
            _atr_wakeup = ind_15m.get("atr", 0)
            _atr_dollar_trigger = _atr_wakeup * CFG.silence_force_wakeup_atr_mult
            if _atr_wakeup > 0 and (_pos_pnl_pct * current_price) >= _atr_dollar_trigger:
                _cooldown = CFG.silence_wakeup_alert_cooldown
                if _now_mono_alert - self._last_alert_ts.get("wakeup_profit", 0) >= _cooldown:
                    self._last_alert_ts["wakeup_profit"] = _now_mono_alert
                    log.warning(f"📈 [{symbol}] 浮盈 {_pos_pnl_pct*100:.2f}% ≥ {CFG.silence_force_wakeup_atr_mult}×ATR，当前价:{current_price}")
                    _webhook(
                        f"📈 [{CFG.symbol}] 浮盈告警",
                        f"方向:{pos.side} 浮盈:{_pos_pnl_pct*100:.2f}% 当前价:{current_price}"
                    )

        # 不做任何静默拦截，由缓存 TTL 控制频率
        return False

    # ── P4 辅助方法：统一安全读取AI决策（防None降级）──
    def _safe_ai_decision(self, symbol: str, ind_15m, fallback_reason: str) -> Dict:
        """统一入口：读取缓存决策，None 时降级为 hold。"""
        decision = self.fast_lane._get_ai_decision(symbol, atr_ratio=ind_15m.get("atr_ratio") if ind_15m else None)
        if decision is None:
            return {"action": "hold", "confidence": 0.5, "reason": fallback_reason}
        return decision

    # ── P0 辅助函数：多周期对齐状态粗粒度桶（缓存Key用）──
    @staticmethod
    def _calc_tf_align_bucket(ind_1h: Dict, ind_4h: Dict) -> str:
        """返回4种多周期对齐状态之一，用于AI缓存Key。
        仅依赖1H/4H的trend+RSI+MACD，零额外计算。
        """
        _bull = 0
        for _ind in (ind_1h, ind_4h):
            if not _ind:
                continue
            if _ind.get("trend", "") == "UP":
                _bull += 1
            elif _ind.get("trend", "") == "DOWN":
                _bull -= 1
            _rsi = _ind.get("rsi", 50)
            if _rsi > 55:
                _bull += 1
            elif _rsi < 45:
                _bull -= 1
            _macd = _ind.get("macd_hist", 0)
            if _macd > 0:
                _bull += 1
            elif _macd < 0:
                _bull -= 1
        if _bull >= 3:
            return "TF_BULL"
        elif _bull <= -3:
            return "TF_BEAR"
        elif _bull >= 1:
            return "TF_SLIGHT_BULL"
        elif _bull <= -1:
            return "TF_SLIGHT_BEAR"
        return "TF_NEUTRAL"

    def _trigger_ai_async_sym(self, symbol: str, ind_15m, ind_1h, ind_4h, ind_3m,
                               news_data, fg_index, funding, depth, pos_info,
                               key_levels=None, market_sentiment=None,
                               funding_history=None, macro_context="", rag_warning="",
                               sentiment_alert="", fast_context="", vs_frozen=None):
        """Per-symbol non-blocking AI trigger"""
        # ── 人工禁用 AI 检查（/block_ai 命令触发）───────────────────────
        if self._ai_blocked:
            log.debug(f"🔒 [{symbol}] AI 决策已被人工禁用，跳过（使用 /unblock_ai 恢复）")
            return
        # 优化8：最小请求间隔由 AI_MIN_REQUEST_INTERVAL 控制（默认10s），避免DeepSeek API限流
        now_mono = time.monotonic()
        min_interval = CFG.ai_min_request_interval
        if now_mono - self._ai_gate.last_request_ts < min_interval:
            log.info(f"⏩ [{symbol}] AI 请求间隔未达 {min_interval}s，跳过")
            return

        # 注：持仓请求拦截已由 _should_skip_ai_request 在 _run_symbol 中统一处理，此处无需重复

        if self._ai_running_flag:
            # VSpike 极端量能时，废弃当前运行中的旧 worker，启动新 worker
            _vs_pre = self.vspike.get_status()
            _vs_mult_pre = _vs_pre.get("mult", 0.0) if hasattr(_vs_pre, "get") else 0.0
            if _vs_mult_pre >= 6.0:
                self._ai_gen += 1
                self._ai_urgent_gen = self._ai_gen
                log.info(f"⚡ [{symbol}] VSpike {_vs_mult_pre:.1f}x 紧急抢占，废弃旧AI worker (gen→{self._ai_gen})，立即启动新决策")
            else:
                log.info(f"⏩ [{symbol}] AI 上轮仍运行，跳过（需等待完成或 VSpike≥6x 紧急抢占）")
                return
        try:
            # ── 缓存键模糊化（降低敏感度，提升命中率）──────────────────────
            _price_raw = ind_15m.get('price', 0)
            _osc_mode, _regime_score = get_market_mode(
                ind_15m, _price_raw, self._market_mode,
                funding=funding, returns_30=self._returns_cache.get(symbol)
            )
            ind_15m["regime_score"] = _regime_score
            self._market_mode = _osc_mode

            # ① RSI 宽分桶：0-30(超卖) / 30-45(偏弱) / 45-55(中性) / 55-70(偏强) / 70-100(超买)
            _rsi = ind_15m.get('rsi', 50)
            if _rsi <= 30:
                _rsi_bkt = "0_30"
            elif _rsi <= 45:
                _rsi_bkt = "30_45"
            elif _rsi <= 55:
                _rsi_bkt = "45_55"
            elif _rsi <= 70:
                _rsi_bkt = "55_70"
            else:
                _rsi_bkt = "70_100"

            # ② BB% 三分桶：0=下沿(超卖) / 1=中部 / 2=上沿(超买)
            _bb_pct = ind_15m.get("bb_pct", 0.5)
            _bb_zone = "0" if _bb_pct <= 0.2 else ("2" if _bb_pct >= 0.8 else "1")

            # ③ 价格漂移容忍：偏离上次决策价 < 0.3% → 连哈希都不用算，直接复用缓存
            # 修复：必须前提是当前缓存还有效（未过期且不为 None）
            _active_cache = self.fast_lane._get_ai_decision(symbol, atr_ratio=ind_15m.get("atr_ratio") if ind_15m else None)
            _last_price = self._ai_gate.last_decision_price
            _last_rsi_bkt = self._ai_gate._last_rsi_bkt
            _last_bb_zone = self._ai_gate._last_bb_zone
            # VSpike 活跃时跳过漂移检查（量能突增代表市场结构变化，不应复用旧缓存）
            _vs_skip_drift = False
            if _active_cache is not None:
                _vs_check = self.vspike.get_status()
                _vs_skip_drift = _vs_check.get("is_spike") or _vs_check.get("spike_recent")
            if (not _vs_skip_drift
                    and _active_cache is not None
                    and _last_price > 0 and _last_rsi_bkt is not None and _last_bb_zone is not None):
                _price_drift = abs(_price_raw - _last_price) / _last_price
                if _price_drift < 0.003 and _rsi_bkt == _last_rsi_bkt and _bb_zone == _last_bb_zone:
                    # 价格稳定 + RSI/BB% 未跨桶 → 复用缓存
                    cached = _active_cache
                    log.info(f"⏩ [{symbol}] 价格漂移 {_price_drift*100:.2f}% < 0.3% + RSI/BB%未变，复用缓存 → "
                              f"{cached.get('action','?')} (conf={cached.get('confidence',0):.2f})")
                    return

            # ④ 价格分桶：趋势市 $20 / 震荡市 $10
            if _osc_mode == "趋势":
                price_bucket = CFG.cache_price_bucket_trending
            else:
                price_bucket = CFG.cache_price_bucket_osc
            _price_bkt = int(round(_price_raw / price_bucket) * price_bucket) if _price_raw > 0 else 0

            _macd_dir = "up" if ind_15m.get("macd_hist", 0) > 0 else "dn"
            _imbal    = round(depth.get("imbalance", 0) if hasattr(depth, 'get') else 0, 1)
            _fund_sgn = "pos" if (funding.get("funding_rate", 0) if hasattr(funding, 'get') else 0) > 0 else "neg"
            holding_minutes = pos_info.get('holding_minutes', 0) // 5 if pos_info.get('side') else 0
            _vol_bkt  = round(ind_15m.get('vol_surge', 1.0) * 2) / 2.0  # 0.5, 1.0, 1.5, 2.0, ...

            input_sig = (
                f"{_price_bkt}_"
                f"{_rsi_bkt}_"
                f"{_macd_dir}_"
                f"{_bb_zone}_"
                f"{_imbal}_"
                f"{_fund_sgn}_"
                f"{holding_minutes}_"
                f"{pos_info.get('side')}_"
                f"{_osc_mode}_"
                f"{_vol_bkt}_"          # 量能分桶
                f"{_calc_tf_align_bucket(ind_1h, ind_4h)}"  # P0: 多周期对齐状态
            )
        except Exception:
            input_sig = ""

        # ── 先获取当前缓存状态 ────────────────────────────────────────────────
        _active_cache = self.fast_lane._get_ai_decision(symbol, atr_ratio=ind_15m.get("atr_ratio") if ind_15m else None)

        # ── 持仓 + 接近关键价位 → 缓存缩短到 30 秒（关键位置不容忍延迟）───
        if pos_info.get("side"):
            _near_level = False
            _price_for_level = ind_15m.get("price", 0) if ind_15m else 0
            if key_levels and key_levels.get("_valid") and _price_for_level > 0:
                for _ll in key_levels.get("supports", []) + key_levels.get("resistances", []):
                    _lvl_price = _ll.get("price", 0) if isinstance(_ll, dict) else 0
                    if _lvl_price > 0 and abs(_price_for_level - _lvl_price) / _price_for_level <= CFG.level_proximity_thresh:
                        _near_level = True
                        break
            if _near_level and _active_cache is not None:
                _cache_age = time.monotonic() - self._ai_gate._cache_ts if self._ai_gate._cache_ts > 0 else 999
                if _cache_age > 30:
                    log.info(f"⚡ [{symbol}] 持仓接近关键价位，缓存已{_cache_age:.0f}s>30s，强制刷新")
                    self.fast_lane._clear_ai_cache(symbol=symbol)
                    _active_cache = None  # 强制走 AI 调用

        prev_conf = _active_cache.get("confidence", 0) if _active_cache else 0
        if prev_conf >= CFG.cache_force_refresh_conf and self._ai_hash == input_sig:
            log.debug(f"🔄 [{symbol}] 高置信度缓存({prev_conf:.2f}>={CFG.cache_force_refresh_conf})，强制刷新")
        elif input_sig == self._ai_hash and _active_cache is not None:
            # 缓存命中
            log.info(f"⏩ [{symbol}] AI缓存命中 → {_active_cache.get('action','?')} (conf={_active_cache.get('confidence',0):.2f})")
            return

        # ── P2/P3: L1 重复 hold 跳过 → 已禁用（用户要求不省 token，每次缓存 miss 都调 L1）──
        _now_mono_p2 = time.monotonic()
        _l1_hold_age = _now_mono_p2 - self._last_l1_hold_ts
        _l1_hold_limit = 0  # 禁用重复 hold 跳过，每次都调 L1
        if (_l1_hold_age < _l1_hold_limit
                and input_sig == self._last_l1_hold_hash
                and _active_cache is not None
                and _active_cache.get("action") in ("hold", "skip")):
            log.debug(
                f"⏭️ [{symbol}] L1 重复 hold 跳过（{_l1_hold_age:.0f}s < {_l1_hold_limit}s({_osc_mode}) + "
                f"市场状态未变），复用缓存 → {_active_cache.get('reason','')[:40]}"
            )
            return

        def _worker():
            self._ai_running_flag = True
            _my_gen = self._ai_gen  # 捕获当前 generation，用于检测是否有紧急抢占
            _worker_start_ts = time.monotonic()  # 用于竞态检测：clear 在启动之后发生 → 丢弃
            _vs_info = self.vspike.get_status()
            log.info(f"🚀 [{symbol}] AI worker 启动（mode={self._market_mode}, vspike={_vs_info.get('mult',0):.1f}x）")
            semaphore_acquired = False
            _failure_counted = False  # 防止双重计数
            # 在任何局部赋值之前，先提取 ind_15m 的值（避免 Python 将 ind_15m 视为局部变量）
            try:
                _saved_price_val = ind_15m["price"] if ind_15m else 0
                _saved_rsi_val = ind_15m["rsi"] if ind_15m else 50
                _saved_macd_val = ind_15m["macd_hist"] if ind_15m else 0
            except (TypeError, KeyError):
                _saved_price_val, _saved_rsi_val, _saved_macd_val = 0, 50, 0
            try:
                tick = self.trader.tick_sizes.get(symbol, 0.01)
                self.ai.tick_size = tick
                # AI 并发限流：同一时刻只有1个品种占用 DeepSeek
                _acquire_start = time.monotonic()
                acquired = self._ai_semaphore.acquire(timeout=CFG.ai_timeout_seconds * max(1, CFG.ai_max_retries) + 30)
                if not acquired:
                    _sem_to = CFG.ai_timeout_seconds * max(1, CFG.ai_max_retries) + 30
                    log.warning(f"⚠️ [{symbol}] AI 信号量等待超时({_sem_to}s)，跳过本轮")
                    self._ai_running_flag = False
                    # 超时表示 permit 被本次失败的 acquire 消耗，必须补 release 恢复，否则后续所有 AI 调用永久阻塞
                    self._ai_semaphore.release()
                    return
                semaphore_acquired = True  # 修复3：标记已获取信号量
                # ── 拿到锁后检查数据新鲜度 ──────────────────────────────────
                # 优化8：若等待时间>15s，K线数据可能过期，重新获取后再调用
                wait_elapsed = time.monotonic() - _acquire_start
                _use_ind15, _use_ind1h, _use_ind4h = ind_15m, ind_1h, ind_4h
                if wait_elapsed > 15:
                    log.info(f"⏳ [{symbol}] 等待{wait_elapsed:.0f}s，刷新K线数据")
                    try:
                        _ind15, _ind1h, _ind4h = self.signals._get_indicators_for_symbol(symbol)
                        if _ind15:
                            _use_ind15 = _ind15
                            _use_ind1h = _ind1h
                            _use_ind4h = _ind4h
                    except Exception as refresh_e:
                        log.warning(f"刷新K线失败: {refresh_e}，继续使用旧数据")
                if wait_elapsed > 30:
                    log.info(f"⏭️ [{symbol}] AI数据严重过期({wait_elapsed:.0f}s)，跳过，等下轮新行情")
                    return  # 修复3：不在此释放信号量，统一在finally中释放

                # ── 趋势对齐分数（取代单一 4H EMA 二元判断）──────────────────────
                _trend_score, _trend_dir = get_trend_alignment_score(
                    _use_ind15, _use_ind1h, _use_ind4h)

                try:
                    # ── Prompt 注入量能事件指令（三级：无/中间级/AGGRESSIVE）──
                    _ft_mult = self._ai_gate.entry_fasttrack_mult
                    _vs_ctx = self.vspike.get_status()
                    _vs_dir_ctx = _vs_ctx.get("direction", "均衡")
                    _vs_bp = _vs_ctx.get("buy_pct", 0.5)
                    _vs_mult_now = _vs_ctx.get("mult", 0.0)
                    _vs_spike_active = _vs_ctx.get("is_spike") or _vs_ctx.get("spike_recent")
                    _effective_mult = max(_ft_mult, _vs_mult_now) if _vs_spike_active else _ft_mult

                    if _effective_mult >= CFG.vspike_priority_threshold:
                        # ── AGGRESSIVE 模式（≥15x）：极端量能特权 ──
                        _vs_dir_warn = ""
                        if _effective_mult >= 15.0:
                            if _vs_dir_ctx == "卖方主导" and _vs_bp < 0.25:
                                _vs_dir_warn = (
                                    f"⚠️ 当前为极端卖方主导量能事件（{100*_vs_bp:.0f}%卖盘），"
                                    f"订单簿买墙大概率被瞬间击穿，禁止做多。\n"
                                    f"已成交流量方向优先级 > 订单簿静态结构。\n"
                                )
                            elif _vs_dir_ctx == "买方主导" and _vs_bp > 0.75:
                                _vs_dir_warn = (
                                    f"⚠️ 当前为极端买方主导量能事件（{100*_vs_bp:.0f}%买盘），"
                                    f"订单簿卖墙大概率被瞬间吃穿，禁止做空。\n"
                                    f"已成交流量方向优先级 > 订单簿静态结构。\n"
                                )
                        _aggressive_ctx = (
                            f"【⚡ 极端量能特权模式 VSpike={_effective_mult:.1f}x】\n"
                            f"当前成交量是基准的{_effective_mult:.0f}倍，属于历史极值事件。\n"
                            f"决策指令：将订单流/成交量动力学权重提升至80%，"
                            f"RSI/MACD等滞后指标仅作辅助参考（权重≤20%）。\n"
                            f"若订单流方向明确（buy_pct>65%看多/buy_pct<35%看空），"
                            f"即使技术指标有背离，也应果断顺势决策。\n"
                            f"{_vs_dir_warn}"
                            f"conf可放宽至0.62以上即可开仓，系统将自动采用试探仓控制风险。\n\n"
                        )
                    elif _effective_mult >= 6.0:
                        # ── 中间级（6x~14x）：显著量能事件，提升订单流权重但不强制方向 ──
                        _dir_threshold_lo = "35%" if _vs_bp > 0.5 else "65%"
                        _dir_threshold_hi = "65%" if _vs_bp > 0.5 else "35%"
                        _aggressive_ctx = (
                            f"【⚡ 显著量能事件 VSpike={_effective_mult:.1f}x】\n"
                            f"当前成交量为基准的{_effective_mult:.0f}倍，属于显著放量事件。\n"
                            f"决策调整：\n"
                            f"- 订单流/成交量动力学权重提升至60%（常规为40%）\n"
                            f"- 静态订单簿结构（挂单墙）可靠性下降——大量能事件中挂单墙经常被吃穿\n"
                            f"- RSI/MACD等滞后指标仍有参考价值，但不应作为主要反对理由\n"
                            f"- 若订单流方向明确（buy_pct>{_dir_threshold_hi}看多/buy_pct<{_dir_threshold_lo}看空），"
                            f"conf可适当放宽至0.65\n\n"
                        )
                    else:
                        _aggressive_ctx = ""
                    result = self.ai.get_decision(
                        _use_ind15, _use_ind1h, _use_ind4h, news_data, fg_index, funding, depth, pos_info,
                        key_levels=key_levels,
                        funding_history=funding_history or [],
                        aggressive_context=_aggressive_ctx,
                        macro_context=macro_context,
                        rag_warning=rag_warning,
                        market_sentiment=market_sentiment,
                        prev_market_mode=self._market_mode,
                        sentiment_alert=sentiment_alert,
                        fast_context=fast_context or "",
                        trend_alignment_score=_trend_score,
                        trend_dir=_trend_dir,
                        vs_status=vs_frozen,
                    )
                except Exception as ai_e:
                    log.error(f"[{symbol}] AI调用异常: {ai_e}")
                    result = {"action": "hold", "confidence": 0.0, "reason": f"AI异常: {str(ai_e)[:100]}", "thought_process": ""}
                # 无论成功或异常，都统一在finally中释放信号量
                # ── generation 检查：如果有紧急抢占请求，废弃旧结果 ──
                if self._ai_urgent_gen > _my_gen:
                    log.info(f"🗑️ [{symbol}] 旧AI worker(gen={_my_gen})被废弃，结果不写入缓存 (urgent_gen={self._ai_urgent_gen})")
                    self._ai_gate.reset_failure()  # 即使废弃也重置失败计数
                # ── 竞态检查：如果 cache clear 在自己启动之后发生 → 结果过时 ──
                elif self._ai_gate._cache_cleared_ts > _worker_start_ts:
                    log.info(f"🗑️ [{symbol}] 旧AI worker 结果过时（cache 在启动后被清除），丢弃")
                    self._ai_gate.reset_failure()
                else:
                    with self._ai_cache_lock:
                        self._ai_cache = result
                        self._ai_cache_ts = time.monotonic()
                        self._ai_hash = input_sig
                        self._last_ai_request_time = time.monotonic()
                        self._last_ai_decision_time = time.monotonic()
                        self._last_ai_decision_price = _saved_price_val
                        self._last_ai_decision_rsi = _saved_rsi_val
                        self._last_ai_decision_macd = _saved_macd_val
                        self._last_ai_rsi_bkt = _rsi_bkt
                        self._last_ai_bb_zone = _bb_zone
                        self._ai_failure_count = 0
                        if result.get("action") in ("hold", "skip"):
                            self._last_l1_hold_ts = time.monotonic()
                            self._last_l1_hold_hash = input_sig
                    # Phase 1 双写过渡：同步写入 AIGatekeeper
                    _vs_for_cache = vs_frozen if vs_frozen else {}
                    self._ai_gate.set_cache(result, input_sig, _saved_price_val,
                                            _saved_rsi_val, _saved_macd_val, _rsi_bkt, _bb_zone,
                                            vspike_mult=_vs_for_cache.get("mult", 0.0),
                                            vspike_dir=_vs_for_cache.get("direction", ""))
                    self._ai_gate.reset_failure()
                    # ── AI 完成后唤醒主循环：避免等 60s 轮询间隔才读到新决策 ──
                    try:
                        self.vspike.spike_event.set()
                    except Exception:
                        pass
            except Exception as e:
                log.error(f"[{symbol}] AI后台决策异常: {e}")
                # 竞态检查：异常路径也需检测过时结果
                if self._ai_urgent_gen > _my_gen or self._ai_gate._cache_cleared_ts > _worker_start_ts:
                    log.info(f"🗑️ [{symbol}] 旧AI worker 异常结果过时，丢弃（不写入缓存/不增加失败计数）")
                    self._ai_gate.reset_failure()
                else:
                    # 即使异常也要更新缓存，防止缓存过期导致后续决策错误
                    with self._ai_cache_lock:
                        self._ai_cache = {
                            "action": "hold", "confidence": 0.0,
                            "reason": f"AI异常: {str(e)[:100]}",
                            "thought_process": ""
                        }
                        self._ai_cache_ts = time.monotonic()
                        self._ai_hash = input_sig
                        self._last_ai_request_time = time.monotonic()
                        # AI 熔断器：失败计数+1，连续5次则激活30分钟熔断
                        if not _failure_counted:
                            self._ai_failure_count += 1
                            _failure_counted = True
                        if self._ai_failure_count >= 5:
                            self._ai_circuit_broken_until = time.monotonic() + 1800
                            log.error(f"🛑 AI 连续失败{self._ai_failure_count}次，激活熔断器30分钟，降级为规则引擎")
                            log.error(f"🛑 Telegram告警: AI 连续失败{self._ai_failure_count}次，熔断30分钟")
                            _webhook("🛑 AI 熔断激活", f"连续{self._ai_failure_count}次失败，暂停30分钟，降级规则引擎")
                            self._ai_failure_count = 0
                # 识别429限流错误，触发专用冷却（区别于普通失败熔断）
                # 只有非过时的 worker 才记录失败
                if self._ai_urgent_gen <= _my_gen and self._ai_gate._cache_cleared_ts <= _worker_start_ts:
                    _err_str = str(e).lower()
                    if "429" in _err_str or "rate limit" in _err_str or "too many requests" in _err_str:
                        self._ai_gate.record_rate_limit(retry_after_seconds=60)
                    # Phase 1 双写过渡：同步异常状态到 AIGatekeeper
                    self._ai_gate.record_failure()
            finally:
                # 修复3：统一信号量释放，确保不会漏掉或重复释放
                if semaphore_acquired:
                    self._ai_semaphore.release()
                self._ai_running_flag = False

        t = threading.Thread(target=_worker, daemon=True, name=f"ai-{symbol[:4]}")
        self._ai_thread = t
        t.start()

    def _get_cache_ttl(self) -> int:
        """缓存 TTL 动态化：趋势市 8 分钟（快），震荡激进 10 分钟，震荡市 15 分钟（慢）"""
        if self._market_mode == "趋势":
            return CFG.cache_ttl_trend  # 默认 480s
        elif self._market_mode == "震荡激进":
            return int(CFG.cache_ttl_osc * 0.67)  # 默认 600s
        else:
            return CFG.cache_ttl_osc  # 默认 900s

    def _get_ai_decision(self, symbol: str = None, atr_ratio: float = None) -> Optional[Dict]:
        _vs = self.vspike.get_status()
        return self._ai_gate.get_cached(
            self._market_mode, atr_ratio=atr_ratio,
            vspike_mult=_vs.get("mult", 0.0),
            vspike_dir=_vs.get("direction", "")
        )

    def _clear_ai_cache(self, symbol: str = None, clear_last: bool = False):
        self._ai_gate.clear(clear_last=clear_last)
        # 同步清除本地 hash，防止缓存已清除但哈希仍匹配
        self._ai_hash = ""


    def _on_trading_event(self, data: Dict):
        """EventBus 订阅者：将交易事件转发到 Telegram/日志"""
        # publish(event_type, data_dict) 通配 handler 只收到 data_dict
        _sym = data.get("sym", "")
        if isinstance(data, dict):
            if "entry" in data and "side" in data:
                # trade_open
                _lev = data.get("lev", "?")
                _tp = data.get("tp", 0)
                _sl = data.get("sl", 0)
                _liq = data.get("liq_price", 0)
                _title = f"{'🚀' if data['side'] == 'long' else '📉'} 开仓 {_sym}"
                _content = (f"方向:{data['side']} x{_lev} | 张数:{data['size']} | "
                           f"入口:{data['entry']:.2f} | SL:{_sl:.2f} | TP:{_tp:.2f} | "
                           f"强平价:{_liq:.2f}" if _liq else f"方向:{data['side']} x{_lev}")
                _webhook(_title, _content)
                log_event("trade_open", {"title": _title, "content": _content, **data})
            elif "reason" in data:
                # trade_close
                _pnl = data.get("pnl_usdt", 0)
                _title = f"{'✅' if _pnl > 0 else '❌'} 平仓 {_sym}"
                _content = f"原因:{data['reason']} | 盈亏:{_pnl:.2f}U"
                _webhook(_title, _content)
                log_event("trade_close", {"title": _title, "content": _content, **data})
            else:
                log_event("unknown", {"title": "未知事件", "content": str(data), **data})

    def _update_ai_performance(self, is_win: bool, ai_conf_at_entry: float = 0.0):
        """
        仅统计 AI 真正开口（conf>0.5）且已平仓的交易。
        Hold 不计入，AI没开口的交易不计入。
        有效样本≥5笔后根据胜率动态调整 AI 权重。
        """
        if ai_conf_at_entry <= 0.5:
            return  # AI没真正开口，不计入

        win_history = gs_get("ai_win_history", [])
        win_history.append(1 if is_win else 0)
        win_history = win_history[-25:]
        gs_set("ai_win_history", win_history)

        n = len(win_history)
        win_rate = sum(win_history) / n if n > 0 else 0.5
        gs_set("ai_recent_win_rate", win_rate)

        # 冷启动保护：少于10笔有效样本时强制默认权重
        if n < 10:
            ai_weight_mult = 0.75 if n >= 5 else 1.0
            log.debug(f"[AI绩效] 冷启动({n}笔)，权重={ai_weight_mult}")
        else:
            if win_rate >= 0.65:
                ai_weight_mult = 1.0
            elif win_rate >= 0.50:
                ai_weight_mult = 0.85
            elif win_rate >= 0.40:
                ai_weight_mult = 0.70
            else:
                ai_weight_mult = 0.60
            log.info(f"[AI绩效] 近{n}笔胜率={win_rate:.0%} AI权重×{ai_weight_mult:.2f}")

        gs_set("ai_weight_mult", ai_weight_mult)

    def _on_ws_private(self, message):
        """私有 WS 回调：positions / account / orders（所有品种）"""
        try:
            if "arg" in message and "data" in message:
                channel = message["arg"].get("channel")
                data    = message["data"]
                if channel == "account":
                    if not data:
                        return
                    for detail in data[0].get("details", []):
                        if detail["ccy"] == "USDT":
                            avail  = float(detail.get("availBal",  0))
                            frozen = float(detail.get("frozenBal", 0))
                            eq_raw = detail.get("eq")
                            equity = float(eq_raw) if eq_raw is not None else (avail + frozen)
                            self.latest_avail_bal = avail
                            self.latest_equity    = equity
                elif channel == "orders":
                    # 检测 SL/TP 算法单成交（附随主单的止盈止损触发平仓）
                    # Bug 修复：OKX 自动平仓时 _reset_pos 清零 trade_id，导致平仓数据未写入 DB
                    # positions channel 在清空前已保存数据到实例变量，这里直接使用
                    for order_data in (data or []):
                        state = order_data.get("state", "")
                        if state != "filled":
                            continue
                        algo_id = order_data.get("algoId", "")
                        if not algo_id:
                            continue
                        # 使用锁保护读取缓存，防止与 positions channel 竞态
                        with self._sl_tp_cache_lock:
                            if algo_id not in (self._sl_tp_pending_algo_ids or []):
                                continue
                            current_trade_id = self._sl_tp_pending_trade_id
                            current_leverage = self._sl_tp_pending_leverage
                        if not current_trade_id:
                            continue
                        # 获取实际成交价
                        fill_px = float(order_data.get("fillPx") or 0)
                        if fill_px <= 0:
                            fill_px = self._get_price(CFG.symbol)
                        # 判断止盈/止损
                        ord_type = order_data.get("ordType", "")
                        close_reason = "止盈触发" if ord_type in ("tp", "take_profit", "tpsl") else "止损触发"
                        try:
                            update_trade_close(
                                current_trade_id,
                                fill_px,
                                close_reason=close_reason,
                                leverage=current_leverage,
                            )
                            log.info(f"📝 [{CFG.symbol}] SL/TP触发平仓记录 trade_id={current_trade_id}，"
                                     f"exit={fill_px:.4f}，原因={close_reason}")

                            # ── 同步更新连亏计数（修复：止损路径绕过 _close，连亏永不更新）──
                            with self.lock:
                                _sl_entry = self.pos.entry_price
                            if _sl_entry and _sl_entry > 0:
                                _sl_side = "long" if self.pos.side == "long" else "short"
                                _sl_pnl = (fill_px - _sl_entry) if _sl_side == "long" else (_sl_entry - fill_px)
                                if _sl_pnl < 0:
                                    _n = gs_increment("consecutive_losses")
                                    log.warning(f"📉 [{CFG.symbol}] SL止损亏损，连续亏损 {_n} 次（亏损{_sl_pnl/_sl_entry*current_leverage*100:.2f}%）")
                                else:
                                    gs_set("consecutive_losses", 0)
                        except Exception as e:
                            log.error(f"SL/TP平仓DB记录失败 trade_id={current_trade_id}: {e}")
                        finally:
                            # 清理已使用的缓存，防止重复处理（锁保护）
                            with self._sl_tp_cache_lock:
                                self._sl_tp_pending_trade_id = None
                                self._sl_tp_pending_algo_ids = []
                                self._sl_tp_pending_leverage = 1
                elif channel == "positions":
                    for pos_data in data:
                        sym = pos_data.get("instId", "")
                        if sym != CFG.symbol:
                            continue
                        lock = self.lock
                        pos  = self.pos
                        with lock:
                            if abs(float(pos_data.get("pos", 0))) > 0:
                                self._zero_pos_seen_ts = 0.0   # 交易所确认有持仓，重置计时器
                                pos.side        = pos_data["posSide"]
                                pos.size        = abs(float(pos_data["pos"]))
                                pos.entry_price = float(pos_data["avgPx"])
                                # ── SL/TP 算法单成交检测缓存：将 pos.sl_tp_algo_ids 同步到 orders 频道可用的缓存 ──
                                if pos.sl_tp_algo_ids:
                                    with self._sl_tp_cache_lock:
                                        self._sl_tp_pending_algo_ids = list(pos.sl_tp_algo_ids)
                                        self._sl_tp_pending_leverage = pos.leverage or 1
                                        self._sl_tp_pending_trade_id = pos.trade_id
                                # ── 全量同步后主动更新最新一条摘要（确保健康数据反映真实持仓）──
                                if self._ai_summaries:
                                    _last = self._ai_summaries[-1]
                                    _curr_price = self._price_val or 0
                                    _entry = pos.entry_price or 0
                                    if _last.get("actual_side") != pos.side and _entry > 0 and _curr_price > 0:
                                        _sync_pnl = ((_curr_price - _entry) / _entry) if pos.side == "long" \
                                            else ((_entry - _curr_price) / _entry)
                                        _last["actual_side"] = pos.side
                                        _last["pnl_pct"] = round(_sync_pnl * 100, 2)
                                        log.debug(f"🔄 [{CFG.symbol}] 同步持仓方向→{pos.side} PnL→{_sync_pnl*100:.2f}%")
                            else:
                                # WS positions 推送 pos=0
                                if pos.side:
                                    _open_age = time.monotonic() - self._last_open_ts
                                    if _open_age < 30.0:
                                        # 开仓 30s 内，WS 可能返回旧快照，静默丢弃
                                        log.debug(f"WS pos=0 但本地有 {pos.side} 仓位（开仓后{_open_age:.0f}s），静默丢弃")
                                        self._zero_pos_seen_ts = 0.0   # 重置计时器
                                    else:
                                        # 开仓超过 30 秒，开始累积连续收到 pos=0 的时间
                                        now_mono = time.monotonic()
                                        if self._zero_pos_seen_ts == 0.0:
                                            self._zero_pos_seen_ts = now_mono
                                        elif now_mono - self._zero_pos_seen_ts > 2.0:
                                            log.warning(
                                                f"👻 [{sym}] WS 连续 {now_mono - self._zero_pos_seen_ts:.1f}s 返回 pos=0，"
                                                f"但本地仍有 {pos.side} 仓位，触发强制全量同步"
                                            )
                                            self._zero_pos_seen_ts = 0.0   # 重置，避免重复触发
                                            threading.Thread(target=self.state._full_state_sync, daemon=True).start()
                                else:
                                    # 本地无持仓，重置计时器
                                    self._zero_pos_seen_ts = 0.0
        except Exception as e:
            log.error(f"私有 WS 消息处理异常: {e}")

    # ---------- 每品种价格获取 ----------
    def _get_price(self, sym: str = None) -> float:
        sym = sym or CFG.symbol
        age = time.monotonic() - self._price_ts
        if self._price_val > 0 and age <= CFG.price_stale_seconds:
            return self._price_val
        if age > CFG.price_stale_seconds and self._price_val > 0:
            if (time.monotonic() - self._price_warn_ts) > 30:
                log.warning(f"⚠️ [{sym}] WS价格已 {age:.1f}s 未更新，降级REST")
                self._price_warn_ts = time.monotonic()
        rest_price = self.trader.get_current_price()
        if rest_price > 0:
            self._price_val = rest_price
            self._price_ts  = time.monotonic()
        return self._price_val

    def _get_mark_price(self) -> float:
        sym = CFG.symbol
        age      = time.monotonic() - self._mark_price_ts2
        mark_val = self._mark_price_val
        if mark_val > 0 and age <= CFG.price_stale_seconds:
            return mark_val
        fallback = self._get_price(sym)
        if mark_val > 0 and age > CFG.price_stale_seconds:
            if (time.monotonic() - self._mark_warn_ts) > 30:
                log.warning(f"⚠️ [{sym}] 标记价已 {age:.1f}s 未更新，降级使用最新价 {fallback:.4f}")
                self._mark_warn_ts = time.monotonic()
        # REST降级时标记为degraded状态，不覆盖原始时间戳（防止误判为新鲜数据）
        if fallback > 0:
            self._mark_price_val = fallback
            self._mark_price_degraded = True  # 标记为降级状态
            self._mark_price_ts2 = time.monotonic()
        return fallback if fallback > 0 else mark_val

    def run_once(self):
        now = datetime.now(UTC)

        # ── API 鲁棒性 3/3：API 自我修复 ─────────────────────────────────
        # retry_with_backoff 检测到 429/5xx/网络异常重试耗尽后设置此标志
        # run_once 在下一轮检测到后立即执行状态修复，防止错误雪崩
        global _api_need_heal
        if _api_need_heal.get("flag"):
            _api_need_heal["flag"] = False  # 先清除，避免重复触发
            _reason = _api_need_heal.get("reason", "")
            log.warning(f"🔧 [API自愈] 执行全量状态同步，原因: {_reason}")
            self.state._full_state_sync()
            # 设置 5 分钟 PAUSE，防止连续错误请求消耗限额
            _pause_until = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
            gs_set("pause_until", _pause_until)
            log.warning(f"⏸️ [API自愈] 已暂停交易 5分钟，恢复时间: {_pause_until}")
            _webhook("⚠️ [API自愈触发]", f"原因: {_reason[:80]}，已暂停5分钟")
            return  # 本轮不再继续，等下一轮

        # 每日报告（共用）
        if now.hour == 8 and now.minute == 0:
            last_report = get_sys_config("last_report_date")
            today_str   = now.date().isoformat()
            if last_report != today_str:
                self.reporter.generate_24h_report()
                set_sys_config("last_report_date", today_str)

        # 全量状态同步
        last_sync = gs_get("last_state_sync")
        last_sync_dt = _parse_dt(last_sync)
        if not last_sync_dt or (now - last_sync_dt).total_seconds() > CFG.state_sync_interval_minutes * 60:
            self.state._full_state_sync()
            # ── 启动时强制设置逐仓杠杆（仅在首次同步时执行一次）────────────
            if not hasattr(self, '_margin_mode_set'):
                self._margin_mode_set = True
                ok_long  = self.trader.set_leverage(CFG.max_leverage, symbol=CFG.symbol, posSide="long")
                ok_short = self.trader.set_leverage(CFG.max_leverage, symbol=CFG.symbol, posSide="short")
                if ok_long and ok_short:
                    log.info(f"✅ 逐仓杠杆设置完成 | 多空均为 {CFG.max_leverage}x")
                else:
                    log.warning(f"⚠️ 逐仓杠杆设置部分失败（long={ok_long} short={ok_short}），请手动在OKX确认")

        # ── API 鲁棒性 1/3：WebSocket 数据新鲜度检测 ───────────────────────
        # tickers 频道正常每 1~5s 推送，若 30s 无更新说明 WS 已断流
        # 强制重连以恢复数据流，防止用 stale 数据做错误决策
        if self.ws_client and self.ws_client._is_data_stale("tickers", max_age=30.0):
            log.warning(f"📡 [WS断流检测] tickers 频道超时，触发强制重连...")
            self.ws_client._reconnect_public()
            self.ws_client._reconnect_private()

        # ── API 鲁棒性 2/3：API 速率限制预测性避让 ─────────────────────────
        # 令牌桶填充比例 > 80% 时，主动延迟 1s 再请求，避免触发 429 被动等待
        _pub_fill = _public_limiter.get_fill_ratio()
        _priv_fill = _private_limiter.get_fill_ratio()
        if _pub_fill > 0.8 or _priv_fill > 0.8:
            log.debug(f"⏳ [速率预测] 令牌桶填充率 公开={_pub_fill:.0%} 私有={_priv_fill:.0%}，主动延迟 1s 避让")
            time.sleep(1.0)

        self.risk_guard._check_daily_reset()
        if not self.state.check_daily_stop():
            return

        # 动态风险因子（每小时根据7日胜率自动调整）
        self.state.update_dynamic_params()

        # 每小时一致性检查：pending_orders 与内存状态、_reserved_margin 校验
        now_ts = time.monotonic()
        if not hasattr(self, '_last_consistency_check') or now_ts - self._last_consistency_check > 3600:
            self.risk_guard._check_pending_consistency()
            self._last_consistency_check = now_ts

        # 每5分钟检查一次 system_config 是否有新的热更新
        # 覆盖「直接 SQL 改库」或「接口崩溃后人工写库」的场景
        now_ts = time.monotonic()
        if not hasattr(self, '_last_cfg_reload') or now_ts - self._last_cfg_reload > 300:
            _load_dynamic_config()
            self._last_cfg_reload = now_ts

        # 账户余额（共用）
        balance = self.latest_avail_bal
        if balance <= 0:
            bal_full = self.trader.get_account_balance_full()
            self.latest_avail_bal = bal_full["avail_bal"]
            self.latest_equity    = bal_full["equity"]
            balance = self.latest_avail_bal

        log.debug(
            f"💰 权益: {self.latest_equity:.2f} USDT | 可用: {balance:.2f} USDT | "
            f"持仓: {'有' if self.pos.side else '无'}"
        )
        log.debug("📰 获取共享数据（新闻/恐贪/宏观）...")

        # 震荡市宏观数据变化慢，延长刷新间隔减少无效 API 调用
        _is_osc = self._market_mode in ("震荡", "震荡激进")
        _news_ttl = (CFG.news_refresh_minutes_osc if _is_osc else CFG.news_refresh_minutes) * 60
        _fg_ttl   = (CFG.fg_refresh_minutes_osc   if _is_osc else 60) * 60  # 趋势市1h，震荡市2h

        # 共享数据：新闻、恐贪、宏观（所有品种共用）
        if (now - self.news_cache["time"]).total_seconds() > _news_ttl:
            news_data = fetch_global_news(top_n=5)
            self.news_cache = {"data": news_data, "time": now}
            log.debug(f"📰 新闻已刷新（{self._market_mode}模式，TTL={_news_ttl//60:.0f}min）")
            # ── 建议4：重大新闻触发 AI 缓存强制刷新（不等缓存过期）──
            _crisis_kw = ("黑客", "暴跌", "崩盘", "破产", "清算", "脱钩", "51%攻击", "漏洞",
                          "黑客攻击", "FTX", "Luna", "Terra", "爆仓", "闪崩")
            _news_raw = news_data.get("text", "")
            if any(kw in _news_raw for kw in _crisis_kw):
                self.fast_lane._clear_ai_cache(clear_last=True)
                log.warning("🚨 检测到重大危机新闻，已强制清除 AI 缓存，确保下一轮决策立即响应")
        else:
            news_data = self.news_cache["data"]

        if (now - self.fg_cache["time"]).total_seconds() > _fg_ttl:
            fg_index = fetch_fear_greed()
            self.fg_cache = {"data": fg_index, "time": now}
            log.debug(f"😰 恐贪指数已刷新（{self._market_mode}模式，TTL={_fg_ttl//60:.0f}min）")
        else:
            fg_index = self.fg_cache["data"]

        log.debug("📊 开始获取市场情绪数据...")

        if (now - self.macro_cache["time"]).total_seconds() > 12 * 3600:
            log.debug("📊 获取1D宏观K线...")
            raw_daily = self.signals.fetch_data("1D", limit=CFG.macro_kline_days, symbol=CFG.symbol)
            log.debug(f"📊 1D K线获取完成: {len(raw_daily) if raw_daily else 0}根")
            log.debug("📊 获取当前价格用于宏观分析...")
            current_price = self._get_price(CFG.symbol)
            log.debug(f"📊 当前价格获取完成: {current_price}")
            macro_full, macro_short = build_macro_context(raw_daily, current_price)
            self.macro_cache = {"full": macro_full, "short": macro_short, "time": now}
        macro_context = self.macro_cache.get("short", "")  # AI prompt 使用短版本节省 token

        # ── 单品种决策 ───────────────────────────────────────────────────
        log.debug("🔄 开始品种决策...")
        try:
            self._run_symbol(CFG.symbol, now, balance, news_data, fg_index, macro_context)
        except Exception as e:
            log.error(f"[{CFG.symbol}] 决策循环异常: {e}\n{traceback.format_exc()}")

        # ── Task4: 每10轮总结一次关键指标（权益/胜率/布林带宽/市场模式）────────
        self._run_counter += 1
        if self._run_counter % 10 == 0:
            win_rate = gs_get("last_24h_win_rate", 0.0)
            equity   = self.latest_equity
            avail    = self.latest_avail_bal
            start_bal = gs_get("start_balance", equity)
            pnl_pct  = (equity - start_bal) / start_bal * 100 if start_bal > 0 else 0.0

            # 收集单品种指标
            pos = self.pos
            cached = self._ind_15m_cache
            if cached:
                _, ind, _ = cached
                bb_pct  = ind.get("bb_pct", -1)
                rsi     = ind.get("rsi", -1)
                trend   = ind.get("trend", "N/A")
                bb_w    = ind.get("bb_width", 0)
                market  = self._market_mode
                pos_side = pos.side or "空仓"
                pos_pnl  = 0.0
                if pos.side and pos.entry_price > 0:
                    price = self._last_price_val or pos.entry_price
                    pos_pnl = ((price - pos.entry_price) / pos.entry_price * 100
                               if pos.side == "long"
                               else (pos.entry_price - price) / pos.entry_price * 100)
                sym_summary = (
                    f"{CFG.symbol}: {pos_side} {'+' if pos_pnl >= 0 else ''}{pos_pnl:.1f}%"
                    f"│BB{bb_w*100:.1f}%│RSI{rsi:.0f}│{market[:2]}│{trend[:2]}"
                )
            else:
                sym_summary = f"{CFG.symbol}: 无指标数据"

            elapsed_s = (now - self._last_summary_time).total_seconds() if self._last_summary_time else 0
            self._last_summary_time = now
            log.info(
                f"📊【10轮摘要】#{self._run_counter} │"
                f"权益={equity:.2f}U(可用{avail:.2f}U) │"
                f"累计PNL={pnl_pct:+.1f}% │"
                f"近期胜率={win_rate:.0%} │"
                f"{sym_summary} │"
                f"间隔{elapsed_s:.0f}s"
            )

    # ============================================================
    # 双管道 AI 调用优化：空仓决策 vs 持仓管理
    # ============================================================
    def _run_holding_cycle(self, sym: str, now: datetime, balance: float,
                           price: float, pos: Position, lock,
                           ind_15m: Dict, ind_1h: Dict, ind_4h: Dict,
                           market_mode: str, depth: Dict, funding: Dict,
                           raw_15m, raw_1h, raw_4h,
                           news_data: Dict, fg_index: Dict, macro_context: str,
                           rag_warning: str, market_sentiment: Dict,
                           _sentiment_alert: str, key_levels: Dict,
                           vs_status: Dict) -> None:
        """
        持仓管理轻量循环 —— 只做「护利」，不做「开仓」。

        职责：
          1. VSpike 反向紧急逃生窗（≥8x 反向量能 → 立即平仓）
          2. 定期轻量 AI 持仓评估（趋势市 120s / 震荡市 240s）
          3. 规则引擎反向信号检测（强反向信号 → 清除 AI 缓存强制重评）

        风控循环（trailing 线程，每2s）已负责：
          - 追踪止损、阶梯盈利锁、止损移动
          → 本方法不再重复这些逻辑
        """
        from core import gs_set as _gs_set
        import time as _time

        _now_mono = _time.monotonic()

        # ── 持仓盈亏 ────────────────────────────────────────────────────
        pnl_pct = 0.0
        with lock:
            if pos.side and pos.entry_price > 0:
                pnl_pct = (
                    (price - pos.entry_price) / pos.entry_price if pos.side == "long"
                    else (pos.entry_price - price) / pos.entry_price
                )

        # ── ① VSpike 反向紧急逃生窗（每轮检查，零成本）─────────────────
        _vs_mult = vs_status.get("mult", 0.0)
        _vs_dir = vs_status.get("direction", "均衡")
        _vs_buy_pct = vs_status.get("buy_pct", 0.5)
        _is_vspike = vs_status.get("is_spike", False) or vs_status.get("spike_recent", False)

        if _is_vspike and _vs_mult >= 8.0 and pos.side:
            _against = (
                (pos.side == "short" and "买方主导" in _vs_dir) or
                (pos.side == "long" and "卖方主导" in _vs_dir)
            )
            if _against:
                _pnl_tag = f"盈利{pnl_pct*100:.2f}%" if pnl_pct > 0 else (
                    f"保本区({pnl_pct*100:.2f}%)" if pnl_pct > -0.003 else f"亏损{pnl_pct*100:.1f}%"
                )
                log.critical(
                    f"💥 [{sym}] [持仓逃生窗] VSpike {_vs_mult:.1f}x 反向{_vs_dir} "
                    f"(buy={_vs_buy_pct:.0%}) + {_pnl_tag} → 紧急平仓"
                )
                log_event("holding_vspike_escape", {
                    "sym": sym, "side": pos.side, "pnl_pct": pnl_pct,
                    "vspike_mult": _vs_mult, "vspike_dir": _vs_dir,
                })
                self._ai_close_pending_until = _now_mono + 5.0
                self.position_exec._close(
                    f"VSpike逃生窗: {_vs_mult:.1f}x反向{_vs_dir} ({_pnl_tag})",
                    symbol=sym
                )
                # 清除 AI 缓存，防止缓存决策干扰
                self.fast_lane._clear_ai_cache(symbol=sym)
                _gs_set("last_holding_adjust_ts", _now_mono)
                return

        # ── ② 规则引擎反向信号检测 ──────────────────────────────────────
        if pos.side:
            _hold_rule = self.signals.evaluate_rules(
                ind_15m, price, self._prev_indicators.get(sym),
                market_mode, key_levels, depth,
                vs_status=vs_status
            )
            _hold_counter = (
                (_hold_rule.get("signal_type") == "short" and pos.side == "long") or
                (_hold_rule.get("signal_type") == "long" and pos.side == "short")
            )
            _hold_conf = _hold_rule.get("confidence", 0.0)
            if _hold_counter and _hold_conf >= CFG.close_confidence_threshold:
                log.warning(
                    f"⚠️ [{sym}] 规则引擎反向信号(conf={_hold_conf:.2f})与当前持仓{pos.side}相反，"
                    f"清除AI缓存强制重评 | {_hold_rule.get('reason', '')}"
                )
                self.fast_lane._clear_ai_cache(symbol=sym)

        # ── ③ 定期轻量 AI 持仓评估 ──────────────────────────────────────
        # 动态间隔：趋势市 120s / 震荡市 240s / VSpike 突增 → 立即评估
        _base_interval = 120.0 if market_mode == "趋势" else 240.0

        # VSpike ≥5x → 缩短间隔，加快响应
        if _vs_mult >= 10.0:
            _eff_interval = 25.0
        elif _vs_mult >= 5.0:
            _eff_interval = 40.0
        else:
            _eff_interval = _base_interval

        _last_adj = getattr(self, "_last_holding_adjust_ts", 0.0)
        _age = _now_mono - _last_adj if _last_adj > 0 else 999.0

        if _age < _eff_interval:
            # 间隔未到 → 跳过 AI 调用，仅做内存检查
            _remaining = _eff_interval - _age
            if self._should_log(f"hold_skip_{sym}", 60.0):
                log.debug(
                    f"⏳ [{sym}] [持仓循环] 距下次AI评估还有 {_remaining:.0f}s "
                    f"(interval={_eff_interval:.0f}s, mode={market_mode})，跳过"
                )
            return

        # ── AI 持仓评估 ─────────────────────────────────────────────────
        # 构造精简 prompt（只做持仓管理，不需要开仓相关上下文）
        pos_info = {
            "side":            pos.side,
            "entry_price":     pos.entry_price,
            "pnl_pct":         pnl_pct,
            "holding_minutes": (now - pos.open_time).total_seconds() / 60 if pos.open_time else 0,
            "peak_price":      pos.peak_price,
            "current_sl":      pos.stop_loss,
            "current_tp":      pos.take_profit,
        }

        # 构建精简的持仓管理 prompt
        _holding_prompt = (
            f"【持仓管理评估】\n"
            f"当前{pos.side}仓 | 入场价={pos.entry_price:.2f} | 现价={price:.2f}\n"
            f"浮盈/亏={pnl_pct*100:+.2f}% | 持仓{pos_info['holding_minutes']:.1f}分钟\n"
            f"止损={pos.stop_loss:.2f} | 止盈={pos.take_profit:.2f} | 峰值={pos.peak_price:.2f}\n"
            f"RSI={ind_15m.get('rsi', 50):.1f} | BB%={ind_15m.get('bb_pct', 0.5):.2f} | "
            f"ATR={ind_15m.get('atr', 0):.2f} | 量能突增={ind_15m.get('vol_surge', 1.0):.1f}x\n"
            f"市场模式={market_mode} | 盘口失衡={depth.get('imbalance', 0):+.3f}\n"
            f"VSpike: {_vs_mult:.1f}x {_vs_dir} (buy={_vs_buy_pct:.0%})\n"
            f"请评估是否应该：close（平仓）、hold（继续持有）、adjust_sl_tp（调整止损止盈）\n"
            f"输出JSON: " + '{{"action":"close|hold|adjust_sl_tp","confidence":0.0~1.0,"reason":"一句话","suggested_sl":数值或null,"suggested_tp":数值或null}}'
        )

        _sys_prompt = (
            "你是ETH-USDT-SWAP量化交易的持仓管理专家。"
            "当前已有持仓，你的任务是评估是否应该平仓、继续持有或调整止损止盈。"
            "只输出JSON，不输出任何其他内容。"
            "\n\n【持仓管理原则】"
            "\n1. VSpike≥8x反向量能 → 立即close"
            "\n2. 浮盈>1%但微观转坏(VSpike反向、RSI极端) → close锁定利润"
            "\n3. 浮亏>0.5%且无恢复信号 → close止损"
            "\n4. 趋势持续且微观支持 → hold"
            "\n5. 止损必给：adjust_sl_tp必须同时给suggested_sl和suggested_tp"
        )

        try:
            _hold_decision = self.ai._fast_ai_direction_check(
                simple_prompt=_holding_prompt,
                allowed_actions=["close", "hold", "adjust_sl_tp"],
                system_prompt=_sys_prompt,
            )
        except Exception as e:
            log.warning(f"⚠️ [{sym}] 持仓AI评估异常: {e}，跳过本轮")
            _gs_set("last_holding_adjust_ts", _now_mono)
            return

        _hd_action = _hold_decision.get("action", "hold")
        _hd_conf = _hold_decision.get("confidence", 0.0)
        _hd_reason = _hold_decision.get("reason", "")[:100]

        # 更新持仓调整时间戳
        _gs_set("last_holding_adjust_ts", _now_mono)
        self._last_holding_adjust_ts = _now_mono

        log.info(
            f"📋 [{sym}] [持仓AI评估] {_hd_action} conf={_hd_conf:.2f} | {_hd_reason}"
        )

        # ── 执行决策 ────────────────────────────────────────────────────
        if _hd_action == "close" and _hd_conf >= 0.65:
            log.info(
                f"🔻 [{sym}] [持仓AI平仓] conf={_hd_conf:.2f} | {_hd_reason}"
            )
            self._ai_close_pending_until = _now_mono + 5.0
            self.position_exec._close(f"持仓管理平仓: {_hd_reason}", symbol=sym)
            return

        if _hd_action == "adjust_sl_tp" and _hd_conf >= 0.65:
            new_sl = _hold_decision.get("suggested_sl")
            new_tp = _hold_decision.get("suggested_tp")
            if new_sl and new_tp and new_sl > 0 and new_tp > 0:
                self.position_exec._adjust_sl_tp(new_sl, new_tp, _hd_reason, symbol=sym)
            return

        # hold → 静默
        if self._should_log(f"hold_ai_{sym}", 120.0):
            log.debug(
                f"😶 [{sym}] [持仓AI评估] hold conf={_hd_conf:.2f} | {_hd_reason}"
            )

    def _run_symbol(self, sym: str, now: datetime, balance: float,
                    news_data: Dict, fg_index: Dict, macro_context: str):
        """单品种决策：获取数据 → AI触发 → 执行动作"""
        log.debug(f"🔄 [{sym}] _run_symbol 开始")
        # ── 初始化规则引擎相关状态（防止极端竞态）────────────────────────
        fast_decision = None
        rule_result = None
        pos  = self.pos
        lock = self.lock

        # 挂单中：快速内存检查（无 I/O），实际处理由 trailing 线程每45s执行
        # 避免在决策热路径上引入网络延迟
        pending_id = pos.pending_ord_id
        if pending_id:
            log.debug(f"🔄 [{sym}] 挂单中 pending_id={pending_id!r}，跳过本轮决策（由trailing线程处理）")
            return

        price = self._get_price(sym)
        self._last_price_val = price   # Task4: 缓存最新价格供10轮摘要使用
        if price <= 0:
            log.warning(f"⚠️ [{sym}] 无法获取价格，跳过")
            return

        # ── 价格陈旧检测 → Mark Price 兜底（15s 内用 mark price 替代，跳过超过 15s 的陈旧数据）────
        _price_age = time.monotonic() - self._price_ts
        if _price_age > CFG.price_stale_fallback_secs:
            log.warning(f"⚠️ [{sym}] 价格陈旧>{CFG.price_stale_fallback_secs}s，跳过")
            return
        if _price_age > CFG.price_stale_seconds:
            log.info(f"⚠️ [{sym}] 价格轻微陈旧({_price_age:.1f}s)，使用mark price兜底")
            _mark = self._get_mark_price()
            if _mark > 0:
                price = _mark

        # ── 状态机：sync_from_pos 对齐真实仓位 ──────────────────
        _trading_state = self._state_machine.sync_from_pos(pos)
        self._current_trading_state = _trading_state

        # IDLE + VSpike → 升为 SCANNING（高优先级唤醒）
        _has_spike = self.vspike.get_status().get("is_spike", False)
        if _trading_state == TradingState.IDLE and _has_spike:
            self._state_machine.transition(TradingState.SCANNING, "VSpike触发扫描")

        # IDLE + 无 VSpike → 仍然进入 SCANNING，由规则引擎和 AI 决定是否交易
        # 删除原先的直接 return，改为正常流转（动态间隔已通过 get_dynamic_interval 控制频率）
        if _trading_state == TradingState.IDLE and not _has_spike:
            self._state_machine.transition(TradingState.SCANNING, "常规扫描")

        # 获取 K 线（带缓存：3m=45s, 15m=120s, 1H=300s, 4H=600s）
        # 缓存命中时无 REST 调用；缓存未命中时并行拉取，减少率限风险
        log.debug(f"🔄 [{sym}] 开始获取3m/15m/1H/4H K线（带缓存）...")
        _bars_needed = ["3m", "15m", "1H", "4H"]
        _bar_ttl = self._RAW_KLINE_TTL
        _now_mono = time.monotonic()
        _miss = [b for b in _bars_needed
                 if b not in self._raw_kline_cache
                 or (_now_mono - self._raw_kline_cache[b][1]) >= _bar_ttl.get(b, 60)]
        if _miss:
            with ThreadPoolExecutor(max_workers=len(_miss), thread_name_prefix="kline-fetch") as ex:
                _futs = {b: ex.submit(self.signals.fetch_data, b, None, sym) for b in _miss}
                try:
                    for b, fut in _futs.items():
                        data = fut.result(timeout=10)
                        if data:
                            self._raw_kline_cache[b] = (data, _now_mono, data[0][0] if data else "")
                            log.debug(f"🔄 [{sym}] {b} K线已更新: {len(data)}根")
                        else:
                            log.warning(f"⚠️ [{sym}] {b} K线拉取返回空")
                except TimeoutError:
                    log.error(f"K线获取超时（10秒），跳过本轮 {sym}")
                    return
        else:
            log.debug(f"🔄 [{sym}] K线全部命中缓存，跳过 REST 请求")
        # 缓存清理：只保留需要的 bar，防止内存泄漏
        _needed_bars = {"3m", "15m", "1H", "4H"}
        _extra_keys = [k for k in self._raw_kline_cache if k not in _needed_bars]
        for k in _extra_keys:
            del self._raw_kline_cache[k]
        raw_3m  = (self._raw_kline_cache.get("3m")  or (None,))[0]
        raw_15m = (self._raw_kline_cache.get("15m") or (None,))[0]
        raw_1h  = (self._raw_kline_cache.get("1H")  or (None,))[0]
        raw_4h  = (self._raw_kline_cache.get("4H")  or (None,))[0]
        if not raw_15m:
            log.warning(f"⚠️ [{sym}] 15m K线不可用，跳过本轮")
            return

        # ── Hurst 指数用：计算近30根15m收益率序列（用于 Regime 复合化评分）───
        if raw_15m and len(raw_15m) >= 31:
            _close_prices = np.array([float(c[4]) for c in raw_15m[-31:]], dtype=float)
            returns_30 = np.diff(np.log(_close_prices))
            self._returns_cache[sym] = returns_30
        else:
            returns_30 = self._returns_cache.get(sym)

        # 策略委员会用：存储原始K线供ATR突破和RSI策略使用
        self._last_15m_raw = raw_15m
        self._last_3m_raw  = raw_3m
        self._last_1h_raw  = raw_1h

        # 15m指标缓存15秒（仅针对raw_15m未变化的情况）
        # 增强：使用K线时间戳哈希，避免数据相同但重复计算
        _raw_hash = str(raw_15m[-1][0]) if raw_15m else ""
        # ── CVD 15分钟周期重置 ───────────────────────────────────────────
        # 当新的15分钟K线开始时（hash变化），重置成交量累计计数器
        if _raw_hash and _raw_hash != self._last_cvd_reset_ts:
            self.vspike.reset_cvd()
            self._last_cvd_reset_ts = _raw_hash
        _now = time.monotonic()
        _cached_ts, _cached_ind, _cached_hash = self._ind_15m_cache
        if (_now - _cached_ts < 15.0 and _cached_ind.get("_valid") and _raw_hash == _cached_hash):
            ind_15m = _cached_ind
        else:
            ind_15m = calc_indicators(raw_15m)
            if ind_15m.get("_valid"):
                self._ind_15m_cache = (_now, ind_15m, _raw_hash)
        ind_1h  = calc_indicators(raw_1h)

        # P1: 4H指标600s TTL缓存（4H K-line每4小时才变一根，避免每tick重复计算）
        _raw_4h_hash = str(raw_4h[-1][0]) if raw_4h else ""
        _cached_4h_ts, _cached_4h_ind, _cached_4h_hash = self._ind_4h_cache
        if raw_4h and (_now - _cached_4h_ts < 600.0 and _cached_4h_ind.get("_valid") and _raw_4h_hash == _cached_4h_hash):
            ind_4h = _cached_4h_ind
        else:
            ind_4h = calc_indicators(raw_4h) if raw_4h else {"_valid": False}
            if ind_4h.get("_valid"):
                self._ind_4h_cache = (_now, ind_4h, _raw_4h_hash)

        # 激进优化：3m 指标（独立缓存，快速捕捉突破信号）
        _raw_3m_hash = str(raw_3m[-1][0]) if raw_3m else ""
        _cached_3m_ts, _cached_3m_ind, _cached_3m_hash = self._ind_3m_cache
        if (_now - _cached_3m_ts < 10.0 and _cached_3m_ind.get("_valid") and _raw_3m_hash == _cached_3m_hash):
            ind_3m = _cached_3m_ind
        else:
            ind_3m = calc_indicators(raw_3m) if raw_3m and len(raw_3m) >= 20 else {"_valid": False}
            if ind_3m.get("_valid"):
                self._ind_3m_cache = (_now, ind_3m, _raw_3m_hash)

        # 优化移动止损：更新ATR缓存
        self._atr_val = ind_15m.get("atr", 0.0)
        # 维护最近50个ATR值（用于RAG波动率分位计算，替代单值假数据）
        atr_val = ind_15m.get("atr", 0.0)
        if atr_val > 0:
            self._atr_history.append(atr_val)
            if len(self._atr_history) > 100:
                self._atr_history = self._atr_history[-100:]

        # ── BB 宽度物理地板（与 get_market_mode 保持一致，保证 prompt 值 = 判定值）────
        _bb_w_raw = ind_15m.get("bb_width", 0)
        if _bb_w_raw > 0:
            ind_15m["bb_width"] = max(_bb_w_raw, 0.015)

        if not ind_15m.get("_valid"):
            log.warning(f"⚠️ [{sym}] 15m K线无效，跳过")
            return

        depth   = self.trader.analyze_orderbook()

        # ── 订单簿滚动均值基准（用于动态冰山墙判断）──────────────────────────
        # 计算当前快照的 top-N 均档容量，滚动维护历史序列
        _top5 = depth.get("top5_bids", []) + depth.get("top5_asks", [])
        if _top5:
            _snap_avg = sum(s for _, s in _top5) / max(len(_top5), 1)
            self._ob_rolling_avg.append(_snap_avg)
            if len(self._ob_rolling_avg) > 10:
                self._ob_rolling_avg = self._ob_rolling_avg[-10:]
        # 注入 rolling 基准到 depth dict，供 OB 结构判断使用
        depth["ob_rolling_avg"] = round(sum(self._ob_rolling_avg) / max(len(self._ob_rolling_avg), 1), 2) if self._ob_rolling_avg else 0.0

        # 资金费率（每品种独立缓存）
        fc = self.funding_cache
        if (now - fc["time"]).total_seconds() > 900:
            funding = self.trader.get_funding_rate()
            self.funding_cache = {"data": funding, "time": now}
            fh = self.funding_history
            fh.append({"funding_rate": funding.get("funding_rate", 0)})
            if len(fh) > 32:
                self.funding_history = fh[-12:]
        else:
            funding = fc["data"]

        # ── Phase 3: 构建统一信号对象 ────────────────────────────
        _signal = MarketSignal().from_indicators(
            ind_15m, self.vspike.get_status(), self._market_mode,
            depth, funding
        )
        _signal.price = price

        # ── 信号纯度评分（Signal Purity Score）────────────────────────────
        # 多维度确定性评估 0~1，过滤低质量信号，防止 AI 被矛盾输入干扰
        # 空仓时用于决定 entry_cooldown 是否延长；持仓时仅记录供复盘
        _vs = self.vspike.get_status()
        _purity = _calc_signal_purity(ind_15m, ind_1h, ind_4h, _vs, depth, funding)
        gs_set("signal_purity", round(_purity, 3))
        # 中低纯度提示（节流日志）
        if _purity < 0.55 and self._should_log(f"purity_low_{sym}", 300.0):
            log.debug(f"📉 [{sym}] 信号纯度低: {_purity:.2f}（多指标矛盾）")
        elif _purity < 0.70 and self._should_log(f"purity_mid_{sym}", 300.0):
            log.debug(f"📊 [{sym}] 信号纯度中等: {_purity:.2f}（部分分歧）")

        # L3 关键价位（每品种，4H刷新）
        klc = self.key_levels_cache
        if (now - klc["time"]).total_seconds() > 4 * 3600:
            key_levels = calc_key_levels(ind_1h, ind_4h, price)
            self.key_levels_cache = {"data": key_levels, "time": now}
        else:
            key_levels = klc.get("data") or {"supports": [], "resistances": [], "_valid": False}

        # 持仓盈亏计算
        pnl_pct = liq_dist_pct = 0.0
        with lock:
            if pos.side and pos.entry_price > 0:
                pnl_pct = (
                    (price - pos.entry_price) / pos.entry_price if pos.side == "long"
                    else (pos.entry_price - price) / pos.entry_price
                )
                if pos.liq_price > 0 and price > 0:
                    liq = pos.liq_price
                    liq_dist_pct = max(
                        (price - liq) / price if pos.side == "long" else (liq - price) / price, 0
                    )

        # RAG（传入当前ATR比率用于波动率相似度）
        rag_warning = ""
        if CFG.rag_similar_trades > 0:
            # 收集历史ATR用于分位计算
            hist_atrs = self._atr_history if self._atr_history else [ind_15m.get("atr", 1)]  # 使用真实ATR历史
            similar = retrieve_similar_failures(
                current_rsi=ind_15m.get("rsi", 50),
                current_trend=ind_15m.get("trend", ""),
                current_bb_pct=ind_15m.get("bb_pct", 0.5),
                current_side=pos.side or "",
                symbol=sym,
                n=CFG.rag_similar_trades,
                current_market_mode=self._market_mode,  # 市场模式参数
                current_atr=ind_15m.get("atr", 0),
                hist_atrs=hist_atrs,
                current_ma_alignment=_get_ma_alignment(ind_4h),
            )
            if similar:
                rag_warning = build_rag_warning(similar, ind_15m.get("rsi", 50), ind_15m.get("bb_pct", 0.5))

            # 记录市场模式和关键参数到日志，方便复盘
            _cached_mode = self._market_mode
            log.debug(f"🔍 [{sym}] 市场模式={_cached_mode} RSI={ind_15m.get('rsi',50):.1f} "
                      f"BB%={ind_15m.get('bb_pct',0.5):.2f} ATR={ind_15m.get('atr',0):.2f}")

        # L4 市场情绪（每品种独立，1H刷新）
        if (now - self.sentiment_cache["time"]).total_seconds() > 3600:
            msd = fetch_market_sentiment_data(sym)
            self.sentiment_cache = {"data": msd, "time": now}
        market_sentiment = self.sentiment_cache.get("data") or {"_valid": False}

        # ── 计算情绪警示（供 AI 决策参考）──────────────────────────────
        # 1.1 在 ETHTrader 中计算情绪突变信号（需要访问 self._prev_ls_ratio 等）
        _sentiment_alert = ""
        if market_sentiment.get("_valid"):
            msd = market_sentiment
            ls = msd.get("ls_ratio")
            tbr = msd.get("taker_buy_ratio")
            oi_ch = msd.get("oi_change_pct")
            # 多空比极值反转
            if ls is not None:
                if ls > 2.0 and price > ind_1h.get('ema21', price) * 1.02:
                    _sentiment_alert += "⚠️多空比极值(>2.0)+价格高位→强烈抑制做多，可寻找做空机会 "
                elif ls < 0.5 and price < ind_1h.get('ema21', price) * 0.98:
                    _sentiment_alert += "⚠️多空比极值(<0.5)+价格低位→强烈抑制做空，可寻找做多机会 "
                # 多空比快速回归（情绪退潮）
                prev_ls = self._prev_ls_ratio
                if prev_ls is not None and abs(ls - prev_ls) > 0.5:
                    if ls > prev_ls and ls > 1.5:
                        _sentiment_alert += "⚠️多空比快速上升→情绪亢奋，随时可能退潮 "
                    elif ls < prev_ls and ls < 0.7:
                        _sentiment_alert += "⚠️多空比快速下降→恐慌蔓延，可能超跌反弹 "
                self._prev_ls_ratio = ls
            # Taker买卖比突变
            if tbr is not None:
                prev_tbr = self._prev_taker_ratio
                if prev_tbr is not None and abs(tbr - prev_tbr) > 0.15:
                    if tbr > 0.7 and prev_tbr < 0.5:
                        _sentiment_alert += "🟢Taker买入比突变(+>{:.0f}%)→主力主动买入，跟随做多 ".format((tbr - prev_tbr) * 100)
                    elif tbr < 0.3 and prev_tbr > 0.5:
                        _sentiment_alert += "🔴Taker买入比突变(-{:.0f}%)→主力主动卖出，平多仓 ".format((prev_tbr - tbr) * 100)
                self._prev_taker_ratio = tbr
                # 资金流向判断
                if ind_15m.get('trend') == 'UP' and tbr > 0.6:
                    _sentiment_alert += "🟢价格涨+Taker买比>{:.0f}%→主动买盘驱动，趋势可持续 ".format(tbr * 100)
                elif ind_15m.get('trend') == 'UP' and tbr < 0.4:
                    _sentiment_alert += "🟡价格涨但Taker买比<{:.0f}%→被动上涨，动力可能不足 ".format(tbr * 100)
            # OI与价格背离
            if oi_ch is not None:
                price_high = price >= ind_15m.get('bb_upper', price) * 0.99
                price_low = price <= ind_15m.get('bb_lower', price) * 1.01
                if price_high and oi_ch < -2:
                    _sentiment_alert += "🔴价格新高+OI下降→多头获利了结，潜在反转 "
                elif price_low and oi_ch > 2:
                    _sentiment_alert += "🟢价格新低+OI上升→空头加仓，可能继续下跌 "
                price_dir = 1 if ind_15m.get('trend') == 'UP' else (-1 if ind_15m.get('trend') == 'DOWN' else 0)
                if oi_ch > 10 and price_dir < 0:
                    _sentiment_alert += "🟢OI暴增(+>10%)+价格下跌→大户多头建仓，可跟随 "
                elif oi_ch < -10 and price_dir > 0:
                    _sentiment_alert += "🔴OI暴跌(-<-10%)+价格上涨→大户减仓，平多仓 "
                elif oi_ch < -10:
                    _sentiment_alert += "🔴OI剧烈下降(-<-10%)→大规模清算发生，注意风控 "

        pos_info = {
            "side":            pos.side,
            "entry_price":     pos.entry_price,
            "pnl_pct":         pnl_pct,
            "holding_minutes": (now - pos.open_time).total_seconds() / 60 if pos.open_time else 0,
            "peak_price":      pos.peak_price,
            "liq_dist_pct":    liq_dist_pct,
            "current_sl":      pos.stop_loss,
            "current_tp":      pos.take_profit,
        }

        log.debug(f"😱 [{sym}] 恐贪={fg_index['value']} 费率={funding['funding_rate']*100:.4f}% 失衡={depth['imbalance']:.3f}")  # 主循环每轮输出，改为DEBUG减少刷屏

        # ── 市场模式（在所有使用点之前计算，供规则引擎/打板检测/逆势开仓等共用）──
        # 返回 (categorical_mode, regime_score)，regime_score 注入 ind_15m 供后续规则引擎使用
        if ind_15m and ind_15m.get("_valid"):
            market_mode, _regime_score = get_market_mode(
                ind_15m, price, self._market_mode,
                funding=funding, returns_30=returns_30
            )
            ind_15m["regime_score"] = _regime_score
        else:
            market_mode = self._market_mode or "趋势"
        self._market_mode = market_mode  # 缓存供本轮所有逻辑及追踪止损使用
        self.vspike.market_mode = market_mode  # 同步给检测器，用于自适应阈值

        # ═══════════════════════════════════════════════════════════════════
        # 双管道分流：空仓决策 vs 持仓管理
        # ═══════════════════════════════════════════════════════════════════
        if pos.side:
            # 持仓管理：轻量循环（VSpike逃生 + 定期AI评估 + 规则反向检测）
            # 风控线程（trailing）已负责：追踪止损、阶梯盈利锁
            self._run_holding_cycle(
                sym, now, balance, price, pos, lock,
                ind_15m, ind_1h, ind_4h, market_mode, depth, funding,
                raw_15m, raw_1h, raw_4h,
                news_data, fg_index, macro_context,
                rag_warning, market_sentiment, _sentiment_alert, key_levels,
                vs_status=self.vspike.get_status(),
            )
            return

        # ── 空仓决策：完整管线（规则引擎 + 打板 + MA/MACD + 震荡信号 + AI）──
        # 日风险用满 → 跳过规则引擎 + AI 调用，避免无谓消耗
        # ── 1.1 规则引擎预筛选 + 核心覆盖修复 ─────────────────────────────
        _today_risk = float(gs_get("today_opened_risk", 0.0))
        _eq = self.latest_equity if self.latest_equity > 0 else balance
        if _today_risk > 0 and _today_risk >= _eq * CFG.max_daily_risk_pct:
            if self._should_log("daily_risk_full", 300.0):
                log.info(f"🛑 [{sym}] 日风险已用满({_today_risk:.2f}U/{_eq*CFG.max_daily_risk_pct:.2f}U)，跳过规则引擎+AI调用，等待下轮重置")
            return
        rule_result = self.signals.evaluate_rules(
            ind_15m, price, self._prev_indicators.get(sym),
            market_mode, key_levels, depth,
            vs_status=self.vspike.get_status()
        )
        self._prev_indicators = {"rsi": ind_15m.get("rsi"), "vol_surge": ind_15m.get("vol_surge")}

        # ── 规则引擎：仅作为信号收集器，注入 prompt 供 AI 参考，不再直出 ──
        if rule_result.get("signal_type") in ("long", "short"):
            _rs = rule_result.get("signal_type")
            _rc = rule_result.get("confidence", 0)
            _rr = rule_result.get("reason", "")
            log.debug(f"📋 [{sym}] 规则引擎信号: {_rs} (conf={_rc:.2f}) — 注入 prompt 供 AI 参考")


        # ── 请求拦截检查（静默已禁用，仅保留熔断/VSpike 冷却/EAT 检测）───
        silence_triggered = False
        if fast_decision is None and self.fast_lane._should_skip_ai_request(sym, ind_15m, ind_1h, price):
            log.debug(f"⏸️ [{sym}] AI 请求被拦截（熔断/冷却），使用缓存决策")
            decision = self._safe_ai_decision(sym, ind_15m, "请求拦截")
            silence_triggered = True


        # ── Fast Decision 上下文（signal_hint 非绑定参考）────────────────────
        _fast_context = ""
        # ── 规则引擎信号注入 prompt（非绑定，仅参考）─────────────────────
        _rs_type = rule_result.get("signal_type", "") if rule_result else ""
        _rs_conf = rule_result.get("confidence", 0) if rule_result else 0
        _rs_reason = rule_result.get("reason", "") if rule_result else ""
        if _rs_type in ("long", "short"):
            _rs_grade = "S" if _rs_conf >= 0.90 else ("A" if _rs_conf >= 0.80 else "B")
            _rs_dir = "short" if _rs_type == "short" else "long"
            _rs_vol = ind_15m.get("vol_surge", 0)
            _rs_ob  = depth.get("imbalance", 0)
            _fast_context = (
                f"\n[规则引擎参考信号(非绑定)]\n"
                f"等级: {_rs_grade}级 | 方向: {_rs_dir} | 置信度: {_rs_conf:.2f} | "
                f"成交量: {_rs_vol:.1f}x | 盘口失衡: {_rs_ob:+.3f} | "
                f"原因: {_rs_reason[:80]}\n"
            )

        # ── 打板检测信号注入（不再直出，仅注入 prompt）────────────────────
        if not pos.side:
            _bo_up = self.signals._detect_breakout(raw_15m, price, direction="long", ind_15m=ind_15m, market_mode=market_mode)
            _bo_dn = self.signals._detect_breakout(raw_15m, price, direction="short", ind_15m=ind_15m, market_mode=market_mode)
            if _bo_up.get("breakout") and _bo_up.get("confidence", 0) >= CFG.breakout_conf_min:
                _fast_context += f"\n[打板信号] 向上突破 conf={_bo_up.get('confidence'):.2f} | {_bo_up.get('reason','')[:80]}\n"
            elif _bo_dn.get("breakout") and _bo_dn.get("confidence", 0) >= CFG.breakout_conf_min:
                _fast_context += f"\n[打板信号] 向下跌破 conf={_bo_dn.get('confidence'):.2f} | {_bo_dn.get('reason','')[:80]}\n"

        # ── 信号纯度评分注入（供 AI 参考）───────────────────────────────
        _fast_context += f"\n[信号纯度 SignalPurity] {_purity:.2f}/1.00 | "
        if _purity >= 0.70:
            _fast_context += "高质量信号，多维度一致\n"
        elif _purity >= 0.55:
            _fast_context += "中等质量，部分指标有分歧\n"
        else:
            _fast_context += "低质量信号，多指标矛盾，建议保守\n"

        # ── EAT-FLOW 吃单流量 + VSpike 突增（注入 fast_context）────────────
        _vs = self.vspike.get_status()

        # EAT-FLOW：仅在有意义的结构信号时注入（避免平静期 token 浪费）
        # 触发条件：VSpike 突增 / 震荡激进市 / absorption 冰山单信号
        _ef_trigger = (
            _vs.get("is_spike")
            or market_mode == "震荡激进"
            or _vs.get("absorption")
        )
        if _vs.get("has_flow_data") and _ef_trigger:
            _fp  = _vs["flow_per_sec"]
            _cd  = _vs["cum_delta_6"]
            _fast_context += (
                f"\n[吃单流量 EAT-FLOW]\n"
                f"净流量: {_fp:+.1f}张/s ({_vs['direction']}) | "
                f"1min累计Delta: {_cd:+.0f}张 | 趋势: {_vs['delta_trend']}\n"
            )
            if _vs.get("absorption"):
                _fast_context += (
                    "⚠️ 吸筹/出货信号：单边主导(买占"
                    f"{_vs['buy_pct']*100:.0f}%)但1min净Delta≈0，"
                    "疑似冰山单在消化方向性流量，等待价格明确表态\n"
                )

        # ── CVD 累计成交量Delta（每15分钟K线周期清零）──────────────────────
        # 阈值：|cum_delta| > 50 张 → 有方向性；否则中性
        _cvd = _vs.get("cum_delta", 0)
        _cvd_dir = (
            "净买入📈" if _cvd > 50
            else "净卖出📉" if _cvd < -50
            else "中性"
        )
        _cvd_total = (_vs.get("cum_buy", 0) or 0) + (_vs.get("cum_sell", 0) or 0)
        if _cvd_total > 0:  # 有实际成交数据时才展示（避免开局无数据时干扰）
            _cvd_pct = abs(_cvd) / max(_cvd_total, 1) * 100
            _fast_context += (
                f"\n[CVD成交量净额] {_cvd_dir} | "
                f"累计净量 {_cvd:+.0f}张 (正值=主动买入主导 / 负值=主动卖出主导) / 总成交{_cvd_total:.0f}张(占比{_cvd_pct:.0f}%)"
                f" | {'主动买入主导' if _cvd > 0 else '主动卖出主导' if _cvd < 0 else '多空均衡'}\n"
            )

        # ── 持仓时 VSpike 翻转警告（关键！）──
        _vs_dir = _vs.get("direction", "")
        _vs_buy_pct = _vs.get("buy_pct", 0.5)
        _vs_mult = _vs.get("mult", 0.0)
        if pos.side and _vs.get("is_spike") and _vs_mult >= 5.0:
            _against = (
                (pos.side == "short" and "买方主导" in _vs_dir) or
                (pos.side == "long" and "卖方主导" in _vs_dir)
            )
            if _against:
                _fast_context += (
                    f"\n⚠️ 【VSpike反向警告】当前持仓{pos.side}，"
                    f"但VSpike {_vs_mult:.1f}x {_vs_dir} (buy={_vs_buy_pct:.0%})，"
                    f"量能强烈反向！立即评估是否close！\n"
                )

        # VSpike：仅在突增事件发生时额外提示
        if _vs.get("is_spike"):
            # 宏观高密度窗口检测（用于提示 AI 调节 V_SPIKE_MULT_THRESH）
            _macro_kw = ("CPI", "FOMC", "非农", "NFP", "利率", "美联储", "Fed", "通胀",
                         "鹰派", "鸽派", "加息", "降息", "央行", "GDP", "PCE", "PPI")
            _news_txt = (news_data.get("text", "") if news_data else "")[:300]
            _macro_active = any(kw in _news_txt for kw in _macro_kw)
            _macro_hint = "⚠️ 宏观高密度窗口：近期有央行/CPI/FOMC新闻，建议适当提高VSpike阈值至3.0~4.0过滤噪音；" if _macro_active else ""
            _fast_context += (
                f"\n[🔥 秒级成交量突增]\n"
                f"{_macro_hint}"
                f"倍数: {_vs['mult']}x 均量 | 方向: {_vs['direction']}"
                f" (买占{_vs['buy_pct']*100:.0f}%)\n"
                f"本桶成交量: {_vs['raw_vol']:.0f}张 | 基线均量: {_vs['baseline_vol']:.0f}张\n"
                f"→ 结合价格位置判断：突破/插针/情绪快速反转均可能\n"
            )

        # ── OB 结构信号（按市场模式分级注入，控制 token 用量）──────────────
        _ob_wall_thresh = float(CFG.ob_wall_mult)
        _sr      = depth.get("slope_ratio", 1.0)
        _bwm     = depth.get("bid_wall_mult", 0.0)
        _awm     = depth.get("ask_wall_mult", 0.0)
        _bwd     = depth.get("bid_wall_dist_pct", 0.0)
        _awd     = depth.get("ask_wall_dist_pct", 0.0)
        _imbal_n = depth.get("imbal_near", 0.0)

        # 冰山墙/假突破：紧贴中价（<0.3%）且强度达阈值
        _has_iceberg = (
            (_awm >= _ob_wall_thresh and _awd < 0.003) or
            (_bwm >= _ob_wall_thresh and _bwd < 0.003)
        )

        # 注入策略（三档）：
        #   震荡激进 / 冰山信号 → 完整注入
        #   趋势市             → 仅在 slope_ratio 极值时注入单行
        #   普通震荡           → 有明显 wall 或 imbal_near 偏斜时才完整注入
        if market_mode == "震荡激进" or _has_iceberg:
            _full_ob = True
        elif market_mode == "趋势":
            _full_ob = False
            if _sr >= 1.4 or _sr <= 0.65:
                _fast_context += (
                    f"\n[OB Slope] slope_ratio={_sr:.2f}"
                    f"({'买方主导' if _sr >= 1.4 else '卖方主导'}，趋势市参考)\n"
                )
        else:  # 普通震荡
            _full_ob = (
                max(_bwm, _awm) >= 2.5 or
                abs(_imbal_n) > 0.35 or
                _sr >= 1.4 or _sr <= 0.65   # slope 极值同样值得关注
            )

        if _full_ob:
            _ob_lines = []
            if _awm >= _ob_wall_thresh and _awd < 0.003:
                _ob_lines.append(
                    f"🧱 卖方冰山墙：距中价{_awd*100:.2f}% 强度{_awm:.1f}x — 假突破风险高，慎追多"
                )
            if _bwm >= _ob_wall_thresh and _bwd < 0.003:
                _ob_lines.append(
                    f"🧱 买方支撑墙：距中价{_bwd*100:.2f}% 强度{_bwm:.1f}x — 下方空间受限，有利做多"
                )
            if _sr >= 1.4:
                _ob_lines.append(f"📈 买方斜率主导(ratio={_sr:.2f})，微观结构偏多")
            elif _sr <= 0.65:
                _ob_lines.append(f"📉 卖方斜率主导(ratio={_sr:.2f})，微观结构偏空")
            if abs(_imbal_n) > 0.20:
                _ob_lines.append(
                    f"近场失衡={_imbal_n:+.3f}"
                    f"({'买方压制' if _imbal_n > 0 else '卖方压制'})"
                )
            if _ob_lines:
                _fast_context += "\n[🏗 OB订单簿结构]\n" + "\n".join(_ob_lines) + "\n"

        if not silence_triggered and fast_decision is None:
            # 冻结当前 VSpike 快照，供 AI 决策与后续 ConvictionScore 共用
            _vs_frozen = self.vspike.get_status()
            _vs_mult_fr = _vs_frozen.get("mult", 0.0) if (
                _vs_frozen.get("is_spike") or _vs_frozen.get("spike_recent")
            ) else 0.0
            _vs_dir_fr = _vs_frozen.get("direction", "均衡")

            # AI 异步触发（每品种独立缓存）；即使有 fast_decision 也传上下文让 AI 最终裁决
            self.fast_lane._trigger_ai_async_sym(
                sym, ind_15m, ind_1h, ind_4h, ind_3m, news_data, fg_index, funding,
                depth, pos_info, key_levels=key_levels,
                funding_history=self.funding_history,
                macro_context=macro_context, rag_warning=rag_warning,
                market_sentiment=market_sentiment,
                sentiment_alert=_sentiment_alert,
                fast_context=_fast_context,
                vs_frozen=_vs_frozen,
            )

            decision = self._safe_ai_decision(sym, ind_15m, "AI 未就绪")

        # ── 防御：静默路径下 _vs_frozen 等变量可能未定义 ──
        if '_vs_frozen' not in locals():
            _vs_frozen = self.vspike.get_status()
        if '_vs_mult_fr' not in locals():
            _vs_mult_fr = _vs_frozen.get("mult", 0.0) if (
                _vs_frozen.get("is_spike") or _vs_frozen.get("spike_recent")
            ) else 0.0
        if '_vs_dir_fr' not in locals():
            _vs_dir_fr = _vs_frozen.get("direction", "均衡")

        # ── AI 决策后的方向解析与 VSpike Bonus 计算 ──
        _action_fr = decision.get("action", "")
        _is_long_fr = _action_fr == "open_long"
        _is_short_fr = _action_fr == "open_short"
        # 仅在明确开仓动作时才检查方向对齐，hold/close 不参与 bonus 计算
        _vs_dir_ok_fr = (
            (_is_long_fr and _vs_dir_fr == "买方主导") or
            (_is_short_fr and _vs_dir_fr == "卖方主导")
        )
        _vs_score_mult_fr = _vs_mult_fr if _vs_dir_ok_fr else 0.0
        # VSpike ≥6.0x bonus：仅方向对齐时才给 bonus，避免反向 Spike 误导开仓
        if _vs_mult_fr >= 6.0 and _vs_dir_ok_fr:
            _vs_score_mult_fr += 15.0
        # P0-3: 上限 cap=30，防止 103x VSpike 拿到 118 分独占 ConvictionScore
        _vs_score_mult_fr = min(_vs_score_mult_fr, 30.0)
        decision["_vspike_status"] = _vs_frozen
        decision["_vs_score_mult_frozen"] = _vs_score_mult_fr
        decision["_vs_score_mult_frozen_ts"] = time.monotonic()  # 过期时间戳

        # 【核心修复】高置信度规则引擎信号（conf≥0.75）→ 跳过 ConvictionScore pre-check
        # 避免 AI hold 决策覆盖 rule engine 的 open 信号，导致错过快速行情
        _rule_high_conf_bypass = (
            decision.get("action", "") in ("open_long", "open_short")
            and decision.get("confidence", 0) >= 0.75
        )
        if fast_decision is not None and not silence_triggered and not _rule_high_conf_bypass:
            _use_fast = decision.get("use_fast_decision", False)
            _ai_conf_for_fast = decision.get("confidence", 0)
            _fast_conf = fast_decision.get("confidence", 0)
            # 路径A：AI 明确认可快速决策（原有逻辑）
            _path_a = _use_fast and _ai_conf_for_fast >= 0.68
            # 路径B：已禁用。AI 决策为唯一出口，规则引擎信号统一经 ConvictionScore 评估。
            # 原逻辑：AI hold 时用规则引擎信号强推——在震荡市里方向反复横跳，风险大于收益。
            _ai_action = decision.get("action", "hold")
            # 记录 AI 方向性信号（供未来方向性仲裁使用）
            if _ai_action in ("open_long", "open_short"):
                self._last_ai_directional[sym] = (_ai_action, time.monotonic())
            _path_b = False  # 禁用 Path B，以 AI 决策为主
            # 路径C：规则引擎明确信号（conf≥0.65）直出，经 ConvictionScore 把关
            _path_c = (
                fast_decision is not None
                and fast_decision.get("action", "") in ("open_long", "open_short")
                and fast_decision.get("confidence", 0) >= 0.65
            )
            if _path_a or _path_b or _path_c:
                # ── fast_decision 独立 Bypass Lane（规则引擎高分 bypass AI门槛）────────
                if fast_decision is not None:
                    # 获取当前 VSpike 状态（方向对齐倍数，供 BypassLane 使用）
                    _vs_now_bl = self.vspike.get_status()
                    _vs_mult_bl  = _vs_now_bl.get("mult", 0.0)
                    _vs_dir_bl   = _vs_now_bl.get("direction", "均衡")
                    _is_long_bl  = fast_decision.get("action", "") == "open_long"
                    _vs_dir_ok   = (_is_long_bl and _vs_dir_bl == "买方主导") or (not _is_long_bl and _vs_dir_bl == "卖方主导")
                    _vs_score_mult   = _vs_mult_bl if _vs_dir_ok else 0.0
                    # 计算是否在关键价位
                    _near_level = False
                    if ind_15m:
                        _price_bl = ind_15m.get("price", 0)
                        _sup_bl = ind_15m.get("support", 0)
                        _res_bl = ind_15m.get("resistance", 0)
                        _near_level = (_sup_bl > 0 and abs(_price_bl - _sup_bl) / _price_bl < 0.003) or (_res_bl > 0 and abs(_price_bl - _res_bl) / _price_bl < 0.003)
                    _fd_action = fast_decision.get("action", "")
                    _fd_conf   = fast_decision.get("confidence", 0.5)
                    # ── 趋势对齐分数（FastLane 也注入）──────────────────────────
                    _ts_fast, _td_fast = get_trend_alignment_score(ind_15m, ind_1h, ind_4h)
                    _fd_score  = self._conviction.score(
                        ai_conf      = _fd_conf,
                        action       = _fd_action,
                        vspike_mult  = _vs_score_mult,
                        ob_imbalance = depth.get("imbalance", 0.0) if hasattr(depth, "get") else 0.0,
                        rsi          = ind_15m.get("rsi", 50.0) if ind_15m else 50.0,
                        at_key_level = _near_level,
                        market_mode  = self._market_mode,
                        context      = {
                            "atr_ratio": ind_15m.get("atr_ratio", 1.0) if ind_15m else 1.0,
                            "trend_alignment_score": _ts_fast,
                            # 极端量能逃生通道
                            "cvd_delta": _vs_frozen.get("cum_delta", 0.0),
                            "flow_direction": _vs_frozen.get("direction", ""),
                            "buy_pct": _vs_frozen.get("buy_pct", 0.5),
                        },
                    )
                    if _fd_score["score"] >= CFG.conviction_full_score:  # 88分 bypass
                        # Grok微调：黑天鹅波动过滤
                        _atr_now = ind_15m.get("atr", 0) if ind_15m else 0
                        _vol_surge_now = ind_15m.get("vol_surge", 1.0) if ind_15m else 1.0
                        _is_black_swan = _vol_surge_now >= 5.0 and _atr_now > 0
                        _bypass_kelly = _fd_score["kelly_ratio"]
                        if _is_black_swan:
                            _bypass_kelly = min(_bypass_kelly, 0.9)
                            log.info(f"⚡ [{sym}] 黑天鹅过滤: vol_surge={_vol_surge_now:.1f}x Kelly cap→0.9")
                        _fd_ai_conf_bypass = _fd_score.get("components", {}).get("ai_raw", 0)
                        log.info(
                            f"⚡ [{sym}] fast_decision Bypass | score={_fd_score['score']} ai_conf(raw)={_fd_conf:.2f} "
                            f"ai_component={_fd_score['components'].get('ai_raw',0):.0f} kelly={_bypass_kelly:.2f} | "
                            f"reason={fast_decision.get('reason', '规则高分共振')}"
                        )
                        decision = fast_decision
                        self.fast_lane._clear_ai_cache(symbol=sym)
                        self._bypass_kelly_override = _bypass_kelly
                        return

                    # ── 仲裁触发：fd_score 处于边缘区（70~82），并行调千问 ─────────
                    if self._arbitration.should_trigger(_fd_score["score"]):
                        _vs_now_q  = self.vspike.get_status()
                        _vs_dir_q  = _vs_now_q.get("direction", "均衡")
                        _price_q   = ind_15m.get("price", price) if ind_15m else price
                        _rsi_q     = ind_15m.get("rsi", 50.0) if ind_15m else 50.0
                        _ob_q      = depth.get("imbalance", 0.0) if hasattr(depth, "get") else 0.0
                        _qwen_res  = self._arbitration.call_qwen(
                            action       = _fd_action,
                            vspike_mult = _vs_mult_bl,
                            ob_imbalance= _ob_q,
                            rsi         = _rsi_q,
                            market_mode  = self._market_mode,
                            depth_dir    = _vs_dir_q,
                            price        = _price_q,
                            reason       = fast_decision.get("reason", ""),
                        )
                        _ar_res = self._arbitration.resolve(
                            qwen_result  = _qwen_res,
                            ds_score     = _fd_score["score"],
                            ds_kelly     = _fd_score["kelly_ratio"],
                            ds_action    = _fd_action,
                            ds_conf      = _fd_conf,
                            sym          = sym,
                        )
                        log.info(
                            f"⚖️ [{sym}] 仲裁裁决 | source={_ar_res['source']} "
                            f"→ action={_ar_res['action']} conf={_ar_res['confidence']:.2f} kelly_override={_ar_res['kelly_override']:.2f}"
                        )
                        # 用仲裁结果覆盖 fast_decision
                        _ar_action = _ar_res["action"]
                        if _ar_action == "hold":
                            decision = {"action": "hold", "confidence": 0.5,
                                        "reason": "仲裁否决：千问与DeepSeek信号冲突"}
                            self.fast_lane._clear_ai_cache(symbol=sym)
                            return
                        # 仲裁同意开仓：注入 kelly_override + sl_tighten_mult
                        decision = {
                            "action": _ar_action,
                            "confidence": _ar_res["confidence"],
                            "reason": f"仲裁[{_ar_res['source']}]: {fast_decision.get('reason', '')}",
                            "sl_tighten_mult": _ar_res.get("sl_tighten_mult", 1.0),
                        }
                        self.fast_lane._clear_ai_cache(symbol=sym)
                        self._bypass_kelly_override = _ar_res["kelly_override"]
                        return
                # ── ConvictionScore 降级为参考指标：记录分数，不再拦截/降级快速决策 ──
                if fast_decision is not None:
                    _fd_score_q = _fd_score.get("score", 0.0)
                    _base_thresh_q = self._get_conviction_open_thresh()
                    if _vs_score_mult >= 6.0:
                        _fd_thresh_q = max(48.0, _base_thresh_q - 8.0)
                    elif _vs_score_mult >= 4.0:
                        _fd_thresh_q = max(50.0, _base_thresh_q - 6.0)
                    elif _vs_score_mult >= 3.0:
                        _fd_thresh_q = max(52.0, _base_thresh_q - 4.0)
                    else:
                        _fd_thresh_q = _base_thresh_q
                    if _fd_score_q < _fd_thresh_q:
                        log.warning(
                            f"⚠️ [{sym}] [ConvictionScore参考] 快速决策 Score={_fd_score_q:.1f}"
                            f" < {_fd_thresh_q:.1f}({self._market_mode})，作为参考记录，不拦交易"
                        )
                    else:
                        log.debug(
                            f"✅ [{sym}] ConvictionScore 快速预检={_fd_score_q:.1f} ≥ {_fd_thresh_q:.1f}，通过"
                        )
                if fast_decision is not None:
                    _fd_reason = fast_decision.get("reason", "")
                    log.info(f"✅ [{sym}] 快速决策执行（{_fd_reason}）→ {fast_decision['action']}")
                    decision = fast_decision
                    self.fast_lane._clear_ai_cache(symbol=sym)
                else:
                    log.info(f"⏳ [{sym}] AI 否决快速决策（use_fast={_use_fast} ai_conf={_ai_conf_for_fast:.2f} fast_conf={_fast_conf:.2f}），转完整推理")
                    fast_decision = None  # 丢弃快速决策，由 AI 推理接管

        # ── AI hold 时主动要求重评（force_wakeup）───────────────────────────
        # 只有 action="hold" 时 AI 才会输出 force_wakeup，此时本轮立即重评
        _fw = bool(decision.get("force_wakeup", False)) if decision.get("action") == "hold" else False
        if _fw:
            log.info(f"⚡ [{sym}] AI hold 但 force_wakeup=true → 清除缓存，下轮强制重评")
            self._last_force_wakeup[sym] = True
            self.fast_lane._clear_ai_cache(symbol=sym, clear_last=True)
            silence_triggered = False  # 取消本轮静默，下一轮强制走完整 AI

        # ── AI 动态 RSI 唤醒阈值 + Level-2 参数建议 ──────────────────────────
        if decision.get("action") == "hold":
            _params = decision.get("param_suggestions") or {}
            # 动态 RSI 唤醒范围：AI 建议则覆盖，否则清除（回归默认阈值 40~60）
            _ai_rsi = _params.get("next_wakeup_rsi")
            if _ai_rsi and isinstance(_ai_rsi, (list, tuple)) and len(_ai_rsi) == 2:
                _lo = float(_ai_rsi[0])
                _hi = float(_ai_rsi[1])
                # 钳位：low ∈ [15, 45]，high ∈ [55, 85]，且 low < high
                _lo = max(15.0, min(45.0, _lo))
                _hi = max(55.0, min(85.0, _hi))
                if _lo < _hi:
                    self._next_wakeup_rsi = (_lo, _hi)
                    log.info(f"🎯 [{sym}] AI 动态 RSI 唤醒阈值: [{_lo:.0f}, {_hi:.0f}]")
                else:
                    self._next_wakeup_rsi = None
                    log.debug(f"🗑️ [{sym}] AI next_wakeup_rsi 无效（lo≥hi），清除动态阈值")
            else:
                self._next_wakeup_rsi = None  # 未建议，清除动态阈值

            # Level-2 参数自动调整（param_suggestions 中其他字段）
            _suggestions = {k: v for k, v in _params.items() if k != "next_wakeup_rsi"}
            if _suggestions:
                try:
                    applied = try_apply_level2_suggestions(_suggestions, symbol=sym)
                    for env_key, old_val, new_val in applied:
                        log.info(f"⚙️ [{sym}] AI 参数建议生效: {env_key} {old_val} → {new_val}")
                except Exception as e:
                    log.debug(f"Level-2 参数应用失败: {e}")

        # 去重日志：相同决策只在内容变化时输出 INFO，重复时降级为 DEBUG
        # 规则引擎已有 🚀 WARNING 日志，不再重复输出 AI 决策日志
        _disp = {k: v for k, v in decision.items() if k != "thought_process"}
        _sig  = f"{decision.get('action')}|{decision.get('confidence', 0):.2f}|{(decision.get('reason') or '')[:80]}"
        _is_repeat = (_sig == self._last_decision_sig.get(sym, ""))
        self._last_decision_sig[sym] = _sig
        if not _is_repeat:
            _decision_src = "[静默缓存]" if silence_triggered else ("[快速决策]" if fast_decision is not None else "[新决策]")
            if not decision.get("reason", "").startswith(("规则", "S级", "A级")):
                log.info(f"🤖 [{sym}] AI决策{_decision_src}: {_disp}")
            if decision.get("thought_process"):
                log.debug(f"🤖 [{sym}] 思考过程: {decision['thought_process']}")
        else:
            _decision_src = "[静默缓存]" if silence_triggered else "[重复]"
            log.debug(f"🤖 [{sym}] AI决策{_decision_src}: action={decision.get('action')} conf={decision.get('confidence', 0):.2f} — {(decision.get('reason') or '')[:60]}")

        # ── 提前冷却检查（在委员会/贝叶斯之前，避免白跑）────────────────────────
        _early_action = decision.get("action", "")
        if (_early_action in ("open_long", "open_short")
                and not pos.side):
            _early_dir = "long" if _early_action == "open_long" else "short"
            _early_blocked = False
            _early_vspike = self.vspike.get_status()
            _early_vs_dir_ok = (
                (_early_dir == "long" and "买方主导" in _early_vspike.get("direction", "")) or
                (_early_dir == "short" and "卖方主导" in _early_vspike.get("direction", ""))
            )

            # 市场模式感知（平仓/止损冷却期共用）
            _trend_exit = market_mode == "趋势"
            # VSpike豁免：趋势市 ≥10x 不要求方向一致（卖方洗盘=做多机会）
            _vs_exempt_mult = 10.0 if _trend_exit else 15.0
            _vs_exempt_dir_ok = True if _trend_exit else _early_vs_dir_ok

            # ① 平仓冷却期：趋势市减半（方向明确，机会转瞬即逝）
            _last_exit_ts = gs_get("last_exit_ts")
            _last_exit_side = gs_get("last_exit_side", "")
            if _last_exit_ts and not _early_blocked:
                _exit_cd_same = 45.0 if _trend_exit else 50.0   # 同向冷却
                _exit_cd_opp  = 45.0 if _trend_exit else 50.0    # 反向冷却
                _now = datetime.now(UTC)
                _exit_dt = _parse_dt(_last_exit_ts)
                _seconds_since_exit = (_now - _exit_dt).total_seconds() if _exit_dt else 999
                _exit_cooldown = _exit_cd_same if _early_dir == _last_exit_side else _exit_cd_opp
                if _seconds_since_exit < _exit_cooldown:
                    _vs_active = _early_vspike.get("is_spike") or _early_vspike.get("spike_recent")
                    _vs_mult = _early_vspike.get("mult", 0)
                    if _vs_active and _vs_mult >= _vs_exempt_mult and _vs_exempt_dir_ok:
                        log.warning(
                            f"🔥 [{sym}] 平仓冷却期内，VSpike={_vs_mult:.1f}x≥{_vs_exempt_mult:.0f}x豁免 | "
                            f"距退出:{_seconds_since_exit:.0f}s/{_exit_cooldown:.0f}s"
                        )
                    else:
                        _early_blocked = True
                        if self._should_log(f"exit_cooldown_{sym}", 120.0):
                            log.info(
                                f"🛑 [{sym}] 平仓后冷却期内，禁止{'同向' if _early_dir == _last_exit_side else '反向'}开仓 | "
                                f"距退出:{_seconds_since_exit:.0f}s < {_exit_cooldown:.0f}s"
                            )
                        else:
                            log.debug(f"🛑 [{sym}] 平仓冷却期拦截（节流）")

            # ② 止损冷却期
            if not _early_blocked and gs_get("consecutive_losses", 0) >= 1:
                _last_stop_dir = gs_get("last_stop_direction")
                _last_stop_time = gs_get("last_stop_time")
                _last_stop_price = gs_get("last_stop_price", 0.0)
                if _last_stop_time and _last_stop_dir and _early_dir == _last_stop_dir:
                    _now = datetime.now(UTC)
                    _last_stop_dt = _parse_dt(_last_stop_time)
                    _minutes_ago = (_now - _last_stop_dt).total_seconds() / 60 if _last_stop_dt else 999
                    # 止损冷却期：趋势市减半（方向明确，机会转瞬即逝）
                    _stop_cooldown = CFG.min_cooldown_after_loss / 2.0 if _trend_exit else CFG.min_cooldown_after_loss
                    if _minutes_ago < _stop_cooldown:
                        _conf = decision.get("confidence", 0)
                        _price_rc = price_reclaimed(price, _last_stop_price, _last_stop_dir)
                        _spike_ok = _early_vspike.get("spike_just_triggered", False) and _early_vspike.get("mult", 0) >= 2.8
                        _rsi = ind_15m.get("rsi", 50) if ind_15m else 50
                        _rsi_ext = (_early_dir == "long" and _rsi < 30) or (_early_dir == "short" and _rsi > 70)
                        # ── VSpike 反向否决：量能 ≥3x 且方向相反时，禁止条件A ──
                        _vs_mult_rc = _early_vspike.get("mult", 0)
                        _vs_dir_rc = _early_vspike.get("direction", "均衡")
                        _vs_against = (
                            (_early_dir == "long" and "卖方主导" in _vs_dir_rc) or
                            (_early_dir == "short" and "买方主导" in _vs_dir_rc)
                        )
                        _vs_reverse = _vs_mult_rc >= 3.0 and _vs_against
                        # ── 规则引擎信号限制：固定 conf（0.72~0.76）非 AI 判断，
                        #    止损冷却豁免仅限 AI 驱动决策（fast_decision is None）──
                        _is_rule_signal = (fast_decision is not None)
                        _condA_thresh = _cooldown_progress = 0.0  # 提前定义
                        if _is_rule_signal:
                            # 规则引擎信号不享受条件A/B豁免，仅条件C（极端量能≥10x）可豁免
                            _condA = False
                            _condB = False
                            _condC = (
                                (_early_vspike.get("is_spike") or _early_vspike.get("spike_recent"))
                                and _early_vspike.get("mult", 0) >= _vs_exempt_mult and _vs_exempt_dir_ok and _conf >= 0.62
                            )
                        else:
                            # AI 驱动决策：完整豁免逻辑
                            _cooldown_progress = _minutes_ago / _stop_cooldown  # 0~1
                            _condA_thresh = 0.70 if _cooldown_progress >= 0.5 else 0.75
                            _condA = _price_rc and _conf >= _condA_thresh and not _vs_reverse
                            _condB = _spike_ok and _early_vs_dir_ok and _rsi_ext and _conf >= 0.68
                            _condC = (
                                (_early_vspike.get("is_spike") or _early_vspike.get("spike_recent"))
                                and _early_vspike.get("mult", 0) >= _vs_exempt_mult and _vs_exempt_dir_ok and _conf >= 0.62
                            )
                        if _condA:
                            log.info(
                                f"⚡ [{sym}] 止损冷却期内，条件A(顺势回归)豁免："
                                f"AI conf={_conf:.2f}≥{_condA_thresh:.2f}，价格已重新站稳，"
                                f"冷却进度={_cooldown_progress:.0%}，批准开仓"
                            )
                        elif _condB:
                            log.warning(
                                f"🔥 [{sym}] 止损冷却期内，条件B(极端衰竭)豁免："
                                f"VSpike={_early_vspike.get('mult', 0):.1f}x + {_early_vspike.get('direction', '')} + "
                                f"RSI={_rsi:.1f}(极端区) + AI conf={_conf:.2f}≥0.68，批准左侧摸底"
                            )
                        elif _condC:
                            log.warning(
                                f"🔥 [{sym}] 止损冷却期内，条件C(极端量能)豁免："
                                f"VSpike={_early_vspike.get('mult', 0):.1f}x≥{_vs_exempt_mult:.0f}x + {_early_vspike.get('direction', '')} + "
                                f"AI conf={_conf:.2f}≥0.62，豁免冷却期"
                            )
                        else:
                            _early_blocked = True
                            if self._should_log(f"stop_cooldown_{sym}", 120.0):
                                log.info(
                                    f"🛑 [{sym}] 止损冷却期内，禁止同方向开仓 | "
                                    f"方向:{_last_stop_dir} 距止损:{_minutes_ago:.0f}分钟 "
                                    f"RSI={_rsi:.1f} VSpike={_early_vspike.get('mult', 0):.1f}x {_early_vspike.get('direction', '')}"
                                )
                            else:
                                log.debug(f"🛑 [{sym}] 止损冷却期拦截（节流）")

            if _early_blocked:
                self.fast_lane._clear_ai_cache(symbol=sym)
                return  # 跳过后续委员会/贝叶斯/执行

        # ── 贝叶斯后验置信度（结合 AI confidence + 盘口失衡度）──────────────────
        if decision.get("action") in ("open_long", "open_short") and not silence_triggered:
            ai_conf = decision.get("confidence", 0.5)
            imbal = (depth.get("imbalance", 0) if isinstance(depth, dict) else 0)
            # 先验 = 近25笔已平仓胜率（每小时由 update_dynamic_params 动态维护）
            # 下限 0.50：承认"错过"≠"做错"，32%低胜率主要来自不敢下单而非方向错误
            prior = max(0.50, gs_get("last_24h_win_rate", 0.5))
            # 震荡市 prior 上限下调：24h 滚动胜率包含趋势市高胜率，震荡市实际胜率更低
            # 用 0.55 上限防止震荡市系统性高估信号质量
            if self._market_mode in ("震荡", "震荡激进"):
                prior = min(prior, 0.55)

            # P1-1: VSpike≥6x 时压低 imbalance 权重
            _vs_bay = self.vspike.get_status()
            _vs_mult_bay = _vs_bay.get("mult", 0) if (
                _vs_bay.get("is_spike") or _vs_bay.get("spike_recent")
            ) else 0
            # 极端成交量下挂单簿是不可靠的（做市商挂假单/扫单），不应让它否决真实交易流信号
            # imbalance 权重从 1.0 降到 0.3，让 AI 置信度主导似然计算
            if _vs_mult_bay >= 6.0:
                _imbal_weight = 0.3  # 极端量能下挂单簿权重大幅降低
                log.debug(f"🧮 [{sym}] VSpike={_vs_mult_bay:.1f}x≥6x, imbalance 权重降至 {_imbal_weight}")
            elif _vs_mult_bay >= 3.0:
                _imbal_weight = 0.6  # 中等量能下适度降低
            else:
                _imbal_weight = 1.0  # 正常权重
            # 似然 = AI置信度 × (1 + 失衡度×权重)，截断到 [0.05, 0.95]
            # 上限 0.95 而非 1.0：likelihood=1.0 时 Bayesian 分母退化，posterior 跳至 1.0
            likelihood = max(0.05, min(0.95, ai_conf * (1 + imbal * _imbal_weight)))
            posterior = bayesian_posterior(prior=prior, likelihood=likelihood)
            decision["posterior_confidence"] = posterior
            log.info(f"🧮 [{sym}] 贝叶斯后验: prior={prior:.3f} AI_conf={ai_conf:.3f} imbalance={imbal:.3f}(w={_imbal_weight}) → posterior={posterior:.3f}")

        features = {
            "ind_15m": {k: v for k, v in ind_15m.items() if k not in ["_valid", "_df"]},
            "depth": depth,
            "news_sentiment": news_data.get("sentiment"),
            "fg_index": fg_index["value"],
            "funding_rate": funding["funding_rate"],
            "atr_quantile": _get_atr_quantile(ind_15m.get("atr", 0), self._atr_history) if ind_15m.get("atr", 0) > 0 else 0.5,
            "rsi_interval": _get_rsi_interval(ind_15m.get("rsi",50)),
            "ma_alignment": _get_ma_alignment(ind_4h),
            "posterior_confidence": decision.get("posterior_confidence", decision.get("confidence", 0.5)),
        }
        decision_id = save_decision_to_db(decision, price, balance, features, symbol=sym)

        # ── 决策摘要：写入队列 + 更新全局健康数据 ─────────────────────────
        # 记录真实持仓方向（解决缓存决策方向与实际持仓方向不一致的问题）
        _actual_side = pos.side or "none"
        _actual_pnl = 0.0
        if pos.side and pos.entry_price > 0 and price > 0:
            if pos.side == "long":
                _actual_pnl = (price - pos.entry_price) / pos.entry_price
            else:  # short
                _actual_pnl = (pos.entry_price - price) / pos.entry_price
        _summary = {
            "ts":           datetime.now(UTC).isoformat(),
            "symbol":       sym,
            "action":       decision.get("action"),
            "actual_side": _actual_side,   # 真实持仓方向（不受缓存影响）
            "pnl_pct":     round(_actual_pnl * 100, 2),  # 真实浮亏/浮盈百分比
            "confidence":   decision.get("confidence", 0),
            "reason":       decision.get("reason", "")[:120],
            "price":        price,
            "rsi":         ind_15m.get("rsi", 0),
            "atr":         round(ind_15m.get("atr", 0), 4),
            "market_mode":  market_mode,
            "signal_purity": round(_purity, 2),
            "is_ai":        fast_decision is None,
        }
        self._ai_summaries.append(_summary)
        _health_data["ai_summaries"] = list(self._ai_summaries)
        _health_data["last_update"]  = _summary["ts"]
        _health_data["status"]       = "running"

        # ── 异常决策告警（5分钟去重，防循环告警）─────────────────────────
        _now_mono = time.monotonic()
        def _can_alert(key: str, cooldown: int = 300) -> bool:
            return _now_mono - self._last_alert_ts.get(key, 0) >= cooldown

        _act  = decision.get("action", "")
        _conf = decision.get("confidence", 0)
        _rsi  = ind_15m.get("rsi", 50)

        # 告警1：极高置信度 + 极端 RSI（AI 可能过度拟合当前情绪）
        if _act in ("open_long", "open_short") and _conf > 0.9:
            if _rsi > 85 or _rsi < 15:
                _key = f"extreme_conf_rsi_{sym}"
                if _can_alert(_key):
                    self._last_alert_ts[_key] = _now_mono
                    _webhook("⚠️ AI极高置信度+极端RSI",
                             f"[{sym}] 置信度={_conf:.2f} RSI={_rsi:.1f}\n"
                             f"决策={_act} | {decision.get('reason','')[:80]}\n"
                             f"⚠️ 请关注是否存在过度拟合情绪的风险")

        # 告警2：逆势高置信度（4H 反向但 conf 较高，复盘价值高）
        if _act in ("open_long", "open_short") and _conf >= 0.75:
            _ema_bull = ind_4h.get("ema_bull", True)
            _is_long  = (_act == "open_long")
            _contra   = (_is_long and not _ema_bull) or (not _is_long and _ema_bull)
            if _contra:
                _key = f"contra_high_conf_{sym}"
                if _can_alert(_key, cooldown=600):
                    self._last_alert_ts[_key] = _now_mono
                    _webhook("⚠️ 逆势高置信度开仓",
                             f"[{sym}] {_act} 置信度={_conf:.2f}，但4H EMA方向相反\n"
                             f"已执行（逆势阈值0.75已满足），请关注后续表现")

        action = decision.get("action", "hold")

        # ── 平仓后强制 Qwen 仲裁：120s 内开仓必须经过 Qwen 二次确认 ──
        if action in ("open_long", "open_short") and not pos.side:
            _exit_ts = gs_get("last_exit_ts")
            if _exit_ts:
                _exit_dt = _parse_dt(_exit_ts)
                _exit_age = (datetime.now(UTC) - _exit_dt).total_seconds() if _exit_dt else 999
                if _exit_age < 120:
                    _vs_q = self.vspike.get_status()
                    _exit_reason = gs_get("last_exit_reason", "")[:80]
                    _qwen_q = self._arbitration.call_qwen(
                        action       = action,
                        vspike_mult  = _vs_q.get("mult", 0.0),
                        ob_imbalance = depth.get("imbalance", 0.0) if isinstance(depth, dict) else 0.0,
                        rsi          = ind_15m.get("rsi", 50.0) if ind_15m else 50.0,
                        market_mode  = self._market_mode,
                        depth_dir    = _vs_q.get("direction", "均衡"),
                        price        = price,
                        reason       = decision.get("reason", "")[:60],
                        rsi_1h       = ind_1h.get("rsi", 50.0) if ind_1h else 50.0,
                        rsi_4h       = ind_4h.get("rsi", 50.0) if ind_4h else 50.0,
                        exit_reason  = _exit_reason,
                        imbal_near   = depth.get("imbal_near", 0.0) if isinstance(depth, dict) else 0.0,
                        bid_wall     = f"距{depth.get('bid_wall_dist_pct',0)*100:.1f}% {depth.get('bid_wall_mult',0):.1f}x"
                                       if isinstance(depth, dict) and depth.get("bid_wall_mult", 0) >= 3.5 else "",
                        ask_wall     = f"距{depth.get('ask_wall_dist_pct',0)*100:.1f}% {depth.get('ask_wall_mult',0):.1f}x"
                                       if isinstance(depth, dict) and depth.get("ask_wall_mult", 0) >= 3.5 else "",
                    )
                    if _qwen_q:
                        _qa = _qwen_q.get("action", "hold")
                        _qc = _qwen_q.get("confidence", 0.5)
                        # Qwen 反对 → 拦截
                        if _qa == "hold" or (_qa != action and _qc >= 0.60):
                            log.warning(
                                f"🚫 [{sym}] [平仓后Qwen仲裁] L1建议{action} conf={decision.get('confidence',0):.2f}，"
                                f"Qwen {_qa} conf={_qc:.2f}（距上次平仓{_exit_age:.0f}s < 120s），拦截"
                            )
                            action = "hold"
                            decision["action"] = "hold"
                            decision["confidence"] = 0.5
                            decision["reason"] = (
                                f"平仓后Qwen仲裁未通过 | L1={decision.get('action','')} "
                                f"vs Qwen={_qa}(conf={_qc:.2f}) | 等待{_exit_age:.0f}s < 120s冷却"
                            )
                            return
                        # Qwen 同意 → 放行
                        else:
                            _merged_conf = round((decision.get("confidence", 0) + _qc) / 2, 3)
                            log.info(
                                f"✅ [{sym}] [平仓后Qwen仲裁] L1={action} vs Qwen={_qa}(conf={_qc:.2f}) 一致，"
                                f"合并conf={_merged_conf:.2f}，放行"
                            )
                            decision["confidence"] = _merged_conf

        # ── 归一化：AI 输出 close_long / close_short → "close"（统一处理路径）
        if action in ("close_long", "close_short"):
            action = "close"
        if action == "skip":
            log.info(f"⏭️ [{sym}] AI主动回避: {decision.get('reason', '')}")
            return

        # 多空翻转保护：AI 输出反向动作时由专用控制器处理（含 conf 门控 + 冷却期）
        if action in ("open_long", "open_short") and pos.side:
            is_reverse = (action == "open_long" and pos.side == "short") or \
                          (action == "open_short" and pos.side == "long")
            if is_reverse:
                # 进入专用翻转处理器（三层保护），处理完直接返回
                if self.position_exec._handle_reverse_logic(sym, action, decision, ind_15m, funding):
                    return
            else:
                _now_mono = time.monotonic()
                if _now_mono - self._last_downgrade_log_ts > 120:
                    log.info(
                        f"ℹ️ [{sym}] 已有{pos.side}仓位，AI 再次建议{action}，"
                        f"视为同向信号，降级为 hold（金字塔加仓由独立检查处理）"
                    )
                    self._last_downgrade_log_ts = _now_mono
                else:
                    log.debug(
                        f"ℹ️ [{sym}] 已有{pos.side}仓位，AI 再次建议{action}，"
                        f"视为同向信号，降级为 hold（节流）"
                    )
                action = "hold"  # 降级为 hold，继续走后续流程（金字塔加仓、状态更新等）

        # ── 离场信号收集器：将 Rule 1-7 转化为 AI 离场的参考信号（不再阻断）──────
        def _build_exit_context(now_utc: datetime, decision: Dict) -> Dict:
            """
            AI 驱动平仓架构：Rule 1-7 不再作为硬阻断器，仅收集信号供日志和 AI 参考。
            唯一硬阻断：核按钮 VSpike（Rule 0.5，≥8x 极端反向量能）。
            冷静期（Rule 0）保留紧急逃生窗，高置信度 AI 决策可直接通过。

            返回:
              - signals: dict of {rule_name: hit_bool}  各规则命中情况
              - msg: str  人类可读的触发信号摘要
              - nuclear_safeguard: bool  核按钮 VSpike 是否触发（唯一硬阻断）
              - calm_period_ok: bool  冷静期是否已过或满足紧急逃生条件
            """
            _signals = {}
            _reasons = []

            # ── VSpike 反向量能（Rule 0 逃生窗 & Rule 6 共用）──────────────────
            _vs_now      = self.vspike.get_status()
            _vs_mult     = _vs_now.get("mult", 1.0)
            _vs_buypct   = _vs_now.get("buy_pct", 0.5)
            _vs_baseline = _vs_now.get("baseline_vol", 0)
            _vs_is_realtime = bool(_vs_now.get("is_spike"))
            _vs_is_recent   = bool(_vs_now.get("spike_recent", False))
            _vs_is_active = _vs_is_realtime or _vs_is_recent
            _vs_peak_tag  = f"(历史峰值{_vs_now.get('spike_recent_age',0):.0f}s前)" if _vs_is_recent and not _vs_is_realtime else ""
            _vs_dir_against = (
                (pos.side == "short" and _vs_buypct > 0.65) or
                (pos.side == "long"  and _vs_buypct < 0.35)
            )
            _vspike_opp  = (
                _vs_is_active and _vs_mult >= CFG.vspike_escape_level1
                and _vs_baseline >= CFG.vspike_escape_baseline
                and _vs_dir_against
            )

            # ── Rule 0: 冷静期 ──────────────────────────────────────────────────
            _calm_ok = True  # 默认冷静期已过或不适用
            _nuclear_safeguard = False
            if pos.open_time:
                hold_secs = (now_utc - pos.open_time).total_seconds()
                # VSpike 趋势捕捉激活中 → 豁免冷静期，允许快进快出
                _capture_active = (getattr(pos, 'trend_capture_ts', 0) > 0 and
                                   (time.monotonic() - pos.trend_capture_ts) < 1800)
                _eff_hold = 0 if _capture_active else (600 if market_mode == "震荡激进" else CFG.min_hold_seconds)
                if hold_secs < _eff_hold:
                    conf = decision.get("confidence", 0)
                    reason = decision.get("reason", "")
                    _urgent_keywords = ("止损", "破位")
                    _kw_hit   = any(k in reason for k in _urgent_keywords)
                    _hi_conf  = conf >= 0.88
                    _vs_mid   = conf >= 0.68 and _vspike_opp and _vs_mult >= CFG.vspike_escape_level1 + 3.0
                    _vs_str   = conf >= 0.65 and _vspike_opp and _vs_mult >= CFG.vspike_escape_level2 + 5.0
                    is_urgent = _hi_conf or _kw_hit or _vs_mid or _vs_str
                    _calm_ok = is_urgent
                    if not is_urgent:
                        _signals["calm_period"] = False
                        _reasons.append(f"冷静期保护中({hold_secs:.0f}s<{_eff_hold}s)")

            # ── Rule 0.5：核按钮 VSpike（唯一硬阻断）────────────────────────────
            _vs_is_active_r05 = _vs_is_realtime
            _vspike_extreme = (
                _vs_is_active_r05
                and _vs_mult >= CFG.vspike_extreme_mult
                and _vs_baseline >= CFG.vspike_escape_baseline
                and _vs_dir_against
            )
            _pnl_r05 = 0.0
            if pos.entry_price > 0:
                _pnl_r05 = (price - pos.entry_price) / pos.entry_price if pos.side == "long" else (pos.entry_price - price) / pos.entry_price
            _conf_now = decision.get("confidence", 0)
            _rule05_hit = (
                _vspike_extreme
                and _conf_now >= 0.65
                and _pnl_r05 <= CFG.profit_protect_thresh
            )
            if _rule05_hit:
                _nuclear_safeguard = True
                _pnl_tag = f"盈利{_pnl_r05*100:.2f}%" if _pnl_r05 > 0 else (f"保本区({_pnl_r05*100:.2f}%)" if _pnl_r05 > -0.003 else f"亏损{_pnl_r05*100:.1f}%")
                _signals["nuclear_vspike"] = True
                _reasons.append(f"⓪核按钮VSpike({_vs_mult:.1f}x{_vs_peak_tag},buy={_vs_buypct:.0%})conf={_conf_now:.2f}+{_pnl_tag}")

            # ── PnL 计算（后续规则共用）─────────────────────────────────────────
            _pnl_pct = 0.0
            if pos.entry_price > 0:
                _pnl_pct = (price - pos.entry_price) / pos.entry_price if pos.side == "long" else (pos.entry_price - price) / pos.entry_price
            _sl_dist_pct = abs(pos.entry_price - pos.stop_loss) / pos.entry_price if pos.stop_loss > 0 else 0.03
            rsi_now = ind_15m.get("rsi", 50)
            bb_mid = ind_15m.get("bb_mid", 0)
            bb_pct_now = ind_15m.get("bb_pct", 0.5)
            atr_now = ind_15m.get("atr", 0)
            _osc_mode = market_mode in ("震荡", "震荡激进")
            _no_partial_tp = not getattr(pos, 'partial_tp_triggered', False)

            # ── Rule 1: 动态 ATR 止盈 ───────────────────────────────────────────
            rule1_hit = False
            if atr_now > 0 and _pnl_pct > 0:
                tp_threshold = CFG.exit_tp_atr_mult * atr_now
                tp_pct = _pnl_pct
                rule1_hit = tp_pct * pos.entry_price >= tp_threshold
            _signals["rule1_atr_tp"] = rule1_hit
            if rule1_hit: _reasons.append(f"①动态止盈(>{CFG.exit_tp_atr_mult}×ATR)")

            # ── Rule 2: 1H EMA20 反向贯穿 ──────────────────────────────────────
            rule2_hit = False
            if raw_1h and len(raw_1h) >= 20:
                try:
                    closes_1h = [float(k[4]) for k in raw_1h[-25:]]
                    ema20_val = float(pd.Series(closes_1h).ewm(span=20, adjust=False).mean().iloc[-1])
                    if pos.side == "long" and price < ema20_val:
                        rule2_hit = True
                    elif pos.side == "short" and price > ema20_val:
                        rule2_hit = True
                except Exception:
                    pass
            _signals["rule2_ema20"] = rule2_hit
            if rule2_hit: _reasons.append("②1H EMA20反向贯穿")

            # ── Rule 3: 浮亏超止损 ──────────────────────────────────────────────
            rule3_hit = _pnl_pct <= -max(0.03, _sl_dist_pct * 1.2)
            _signals["rule3_float_loss"] = rule3_hit
            if rule3_hit: _reasons.append(f"③浮亏>{abs(_pnl_pct)*100:.1f}%")

            # ── Rule 4: RSI 极端 + 价格远离成本 ─────────────────────────────────
            rsi_extreme = (rsi_now >= 72 or rsi_now <= 28)
            price_dist_cost = abs(_pnl_pct)
            rule4_hit = rsi_extreme and price_dist_cost > 0.02
            _signals["rule4_rsi_extreme"] = rule4_hit
            if rule4_hit: _reasons.append(f"④RSI极端({rsi_now:.0f})+远离成本({price_dist_cost*100:.1f}%)")

            # ── Rule 5: 日线布林中轨破位 ────────────────────────────────────────
            rule5_hit = False
            if bb_mid > 0:
                if pos.side == "long" and price < bb_mid:
                    rule5_hit = True
                elif pos.side == "short" and price > bb_mid:
                    rule5_hit = True
            _signals["rule5_bb_mid"] = rule5_hit
            if rule5_hit: _reasons.append("⑤日线BB中轨破位")

            # ── Rule 6: VSpike 极端反向 + 亏损/微利 ─────────────────────────────
            _vspike_strong_opp = _vspike_opp and _vs_mult >= CFG.vspike_escape_level2
            _atr_15m_val = ind_15m.get("atr", 0)
            _escape_loss_dyn = max(_atr_15m_val * 0.25 / price, 0.003) if _atr_15m_val > 0 and price > 0 else CFG.escape_loss_min
            if _vs_mult >= 8.0 and _vspike_opp:
                rule6_hit = True
            elif _pnl_pct > CFG.profit_protect_thresh * 2:
                rule6_hit = False
            else:
                rule6_hit = _vspike_strong_opp and (-0.03 < _pnl_pct < _escape_loss_dyn)
            _signals["rule6_vspike_escape"] = rule6_hit
            if rule6_hit:
                _r6_pnl_tag = f"亏损{_pnl_pct*100:.1f}%" if _pnl_pct < 0 else f"保本区({_pnl_pct*100:.2f}%)"
                _reasons.append(f"⑥VSpike极端反向({_vs_mult:.1f}x{_vs_peak_tag},buy={_vs_buypct:.0%})+{_r6_pnl_tag}(动态阈值{_escape_loss_dyn*100:.2f}%)")

            # ── Rule 7: 震荡市 BB%/RSI 背离 + 浮盈 ─────────────────────────────
            _bb_extreme_against = (
                (pos.side == "short" and bb_pct_now >= 0.75) or
                (pos.side == "long"  and bb_pct_now <= 0.25)
            )
            _rsi_extreme_against = (
                (pos.side == "short" and rsi_now >= 68) or
                (pos.side == "long"  and rsi_now <= 32)
            )
            rule7_hit = _osc_mode and _no_partial_tp and (_bb_extreme_against or _rsi_extreme_against) and _pnl_pct > 0
            _signals["rule7_osc_divergence"] = rule7_hit
            if rule7_hit:
                _r7_why = f"BB%={bb_pct_now:.2f}" if _bb_extreme_against else f"RSI={rsi_now:.0f}"
                _reasons.append(f"⑦震荡市极端背离({_r7_why})+浮盈{_pnl_pct*100:.2f}%")

            _msg = " / ".join(_reasons) if _reasons else "无明确离场信号"
            return {
                "signals": _signals,
                "msg": _msg,
                "nuclear_safeguard": _nuclear_safeguard,
                "calm_period_ok": _calm_ok,
                "pnl_pct": _pnl_pct,
                "hit_count": sum(1 for v in _signals.values() if v),
            }

        if action == "close" and pos.side:
            # ── AI 驱动平仓架构：AI 置信度为主导，Rule 1-7 仅作参考 ─────────────
            conf_close   = decision.get("confidence", 0.0)
            close_reason = decision.get("reason", "AI平仓")
            _exit_ctx    = _build_exit_context(now, decision)

            # ── 核按钮 VSpike：极端量能无条件逃生（最高优先级）───────────────
            if _exit_ctx["nuclear_safeguard"]:
                log.critical(
                    f"💥 [{sym}] 核按钮VSpike触发！极端反向量能，立即执行平仓 | "
                    f"AI理由: {close_reason} | conf={conf_close:.2f} | {_exit_ctx['msg']}"
                )
                log_event("nuclear_vspike_close", {
                    "sym": sym, "confidence": conf_close,
                    "reason": close_reason, "trigger": "nuclear_vspike",
                    "exit_signals": _exit_ctx["msg"]
                })
                self._ai_close_pending_until = time.monotonic() + 5.0
                self.position_exec._close(f"核按钮VSpike平仓 | {close_reason}", decision_id, symbol=sym)
                # 极端量能平仓后，下一轮立即重评（可能反手）
                self._last_force_wakeup[sym] = True
                self.fast_lane._clear_ai_cache(symbol=sym, clear_last=True)
                log.info(f"⚡ [{sym}] 核按钮平仓后 force_wakeup：下一轮立即重评（极端VSpike上下文）")
                return

            # ── 置信度门槛（千问仲裁直通）───────────────────────────────────
            _is_qwen_arb = decision.get("source") == "qwen_exit_arbitration"
            if not _is_qwen_arb:
                _close_thresh = 0.68
                if conf_close < _close_thresh:
                    log.info(
                        f"🛡️ [{sym}] 平仓置信度 {conf_close:.2f} < {_close_thresh:.2f}，"
                        f"降级为 hold | {close_reason}"
                    )
                    return

            # ── AI 置信度达标 → 执行平仓 ──────────────────────────────────
            _arb_tag = "[千问仲裁]" if _is_qwen_arb else ""
            _sig_tag = f"[信号:{_exit_ctx['hit_count']}条]" if _exit_ctx["hit_count"] > 0 else "[AI独立判断]"
            _phase_tag = "趋势市" if market_mode == "趋势" else "震荡市"
            log.info(
                f"🔻 [{sym}] AI平仓指令 conf={conf_close:.2f} "
                f"{'仲裁直通' if _is_qwen_arb else f'(阈值{_close_thresh:.2f} {_phase_tag})'} "
                f"{_arb_tag}{_sig_tag} | {close_reason}"
            )
            log_event("ai_close_triggered", {
                "sym": sym, "confidence": conf_close,
                "reason": close_reason, "trigger": "ai_phase_approved",
                "exit_signals": _exit_ctx["msg"],
                "hit_count": _exit_ctx["hit_count"],
            })
            self.position_exec._close(
                f"AI平仓(conf={conf_close:.2f}) {_sig_tag} {_exit_ctx['msg']} | {close_reason}",
                decision_id, symbol=sym
            )
            # 极端量能平仓后，下一轮立即重评（可能反手）
            _vs_fw = self.vspike.get_status()
            if _vs_fw.get("mult", 0.0) >= CFG.vspike_extreme_mult:
                self._last_force_wakeup[sym] = True
                self.fast_lane._clear_ai_cache(symbol=sym, clear_last=True)
                log.info(f"⚡ [{sym}] AI平仓后 force_wakeup：VSpike={_vs_fw.get('mult',0):.1f}x极端量能，下一轮立即重评")
            return

        if action == "adjust_sl_tp" and pos.side:
            if decision.get("confidence", 0) >= CFG.adjust_confidence_thresh:
                new_sl, new_tp = decision.get("suggested_sl"), decision.get("suggested_tp")
                if new_sl and new_tp:
                    self.position_exec._adjust_sl_tp(new_sl, new_tp, decision.get("reason", ""), symbol=sym)
            return

        # ── 金字塔加仓触发检查（持仓时）───────────────────────────────────
        # 在主决策循环中直接检查，满足条件立即加仓（比 trailing 线程更及时）
        # 不在 action 条件分支内：即使 AI 说 hold/rerank，只要浮盈达标就加仓
        if pos.side and pnl_pct > 0:
            try:
                self.position_exec._do_pyramid_add(sym)
            except Exception as e_pyr:
                log.debug(f"[_run_symbol] 金字塔检查异常: {e_pyr}")

        if action in ("open_long", "open_short") and not pos.side:
            # ── 启动冷却期：Bot 重启后至少等待 N 秒再允许开仓（等 ATR 稳定 + 观察初始波动）──
            _startup_age = time.monotonic() - self._boot_ts
            _startup_cd = getattr(CFG, "startup_cooldown_seconds", 180)
            if _startup_age < _startup_cd:
                log.info(f"🚫 [{sym}] 启动冷却中（{_startup_age:.0f}s < {_startup_cd}s），跳过开仓")
                # 记录首次出现的信号reason，冷却过期后要求信号已变化
                _sig_reason = decision.get("reason", "")
                self._post_cooldown_check[sym] = {
                    "reason": _sig_reason,
                    "seen_ts": time.monotonic(),
                }
                self.fast_lane._clear_ai_cache(symbol=sym)
                return

            # ── 冷却过期后首次检查：信号新鲜度 + VSpike 方向一致性 ──
            _blocked = self._post_cooldown_check.pop(sym, None)
            if _blocked:
                _sig_reason_now = decision.get("reason", "")
                _reason_changed = _sig_reason_now != _blocked["reason"]
                _signal_age = time.monotonic() - _blocked["seen_ts"]
                # ① 信号新鲜度检查：同一信号持续 ≥120s → 视为陈旧信号
                if not _reason_changed and _signal_age >= 120:
                    log.info(
                        f"🚫 [{sym}] 启动冷却已过，但同信号已持续 {_signal_age:.0f}s ≥120s，"
                        f"视为陈旧信号（reason={_sig_reason_now}），等待新信号"
                    )
                    self.fast_lane._clear_ai_cache(symbol=sym)
                    return

            # ── 同价位止损冷却：止损后 ±1% 范围内 120s 内不重入 ──
            _sl_cool_price = self._last_sl_price
            _sl_cool_time = self._last_sl_time
            if _sl_cool_price > 0 and price > 0:
                _sl_dist_pct = abs(price - _sl_cool_price) / _sl_cool_price
                _sl_cool_age = time.monotonic() - _sl_cool_time
                if _sl_dist_pct <= 0.01 and _sl_cool_age < 120:
                    log.warning(
                        f"🚫 [{sym}] [同价位冷却] 上次止损价={_sl_cool_price:.4f}，"
                        f"当前价={price:.4f}(±{_sl_dist_pct*100:.2f}%)，冷却剩余{120 - _sl_cool_age:.0f}s，"
                        f"拦截重入"
                    )
                    self.fast_lane._clear_ai_cache(symbol=sym)
                    return

            # ── 连续被洗记忆：同价位已洗≥2次 → 要求更强信号 ──
            if self._wash_count_at_price >= 2:
                _vs_now_wash = self.vspike.get_status()
                _vs_mult_wash = _vs_now_wash.get("mult", 0.0)
                _wash_conf = decision.get("posterior_confidence") or decision.get("confidence", 0)
                if _vs_mult_wash < 5.0 or _wash_conf < 0.65:
                    log.warning(
                        f"🚫 [{sym}] [连续被洗] 同价位已洗{self._wash_count_at_price}次，"
                        f"要求更强信号（VSpike≥5.0x 实际={_vs_mult_wash:.1f}x, "
                        f"conf≥0.65 实际={_wash_conf:.2f}），拦截"
                    )
                    self.fast_lane._clear_ai_cache(symbol=sym)
                    return

            is_long   = (action == "open_long")

            # ── 优先使用贝叶斯后验置信度（融合历史胜率+盘口失衡），无后验时用原始置信度 ─
            _raw_conf = decision.get("confidence", 0)
            _post_conf = decision.get("posterior_confidence")
            conf = _post_conf if _post_conf is not None else _raw_conf

            # ── 提前获取 VSpike 状态（1H Veto 豁免和 Regime Score 共用）──
            _vs_now = self.vspike.get_status()
            _vs_mult = _vs_now.get("mult", 0.0)

            # ── 1H 大周期趋势 Veto（防止 15m 假突破逆势开仓）──
            _1h_ema_bull = ind_1h.get("ema_bull", True)
            _1h_rsi = ind_1h.get("rsi", 50)
            _1h_opp = (is_long and not _1h_ema_bull) or (not is_long and _1h_ema_bull)
            # ── VSpike 豁免：极端量能 + AI共识 + 方向一致时，1H 趋势已被实时打破，不应以旧状态否决 ──
            _vs_dir_match = (
                (is_long and _vs_now.get("direction") == "买方主导") or
                (not is_long and _vs_now.get("direction") == "卖方主导")
            )
            _1h_veto_vspike_exempt = (
                _vs_mult >= 8.0
                and (_vs_now.get("is_spike") or _vs_now.get("spike_recent"))
                and conf >= 0.55
                and _vs_dir_match
            )
            if _1h_opp and not _1h_veto_vspike_exempt:
                # 1H 强烈反趋势 → 高置信度才放行
                _1h_rsi_extreme = (
                    (is_long and _1h_rsi < 35) or  # 1H RSI 已经超卖，做空是顺势
                    (not is_long and _1h_rsi > 65)  # 1H RSI 已经超买，做空是顺势
                )
                if not _1h_rsi_extreme and conf < 0.75:
                    _dir_tag = "1H看跌" if is_long else "1H看涨"
                    log.info(
                        f"🚫 [{sym}] [1H Veto] {action} 被1H趋势拦截（{_dir_tag}，1H RSI={_1h_rsi:.1f}），"
                        f"conf={conf:.2f}<0.75，跳过"
                    )
                    self.fast_lane._clear_ai_cache(symbol=sym)
                    return
                elif _1h_rsi_extreme:
                    log.info(
                        f"⚡ [{sym}] [1H Veto] 1H反趋势但RSI极端({_1h_rsi:.1f})，"
                        f"降低门槛至 0.68 允许左侧开仓"
                    )
                    conf = max(conf, 0.68)  # 左侧交易至少 0.68 conf
                else:
                    log.info(
                        f"⚠️ [{sym}] [1H Veto] 1H反趋势但 conf={conf:.2f}≥0.75，"
                        f"高置信度放行（可能是左侧交易）"
                    )
            elif _1h_opp and _1h_veto_vspike_exempt:
                log.info(
                    f"⚡ [{sym}] [1H Veto] 1H反趋势但 VSpike={_vs_mult:.1f}x 极端量能 + conf={conf:.2f}≥0.55，"
                    f"豁免通过（1H 趋势正被实时打破）"
                )

            # ── 1H Regime Score 硬过滤（补充现有 veto，比 ema_bull 更能反映真实趋势强度）──
            _1h_regime_score = _compute_1h_regime_score(
                ind_1h, price, self._market_mode, funding
            )
            # 1H Regime Score 硬过滤：纯度评分已过滤最差信号，阈值适度放宽
            _vspk_priv = _vs_mult >= 6.0 and (_vs_now.get("is_spike") or _vs_now.get("spike_recent"))
            _reg_lo, _reg_hi = (0.28, 0.72) if _vspk_priv else (0.30, 0.70)
            _vspk_tag = "" if not _vspk_priv else " [VSpike特权通道]"
            if action == "open_long" and _1h_regime_score < _reg_lo:
                log.info(
                    f"🚫 [{sym}] [1H Regime Veto{_vspk_tag}] long 被拦截 | "
                    f"1H_regime={_1h_regime_score:.2f}（极弱趋势，阈值={_reg_lo:.2f}）"
                )
                self.fast_lane._clear_ai_cache(symbol=sym)
                return
            if action == "open_short" and _1h_regime_score > _reg_hi:
                log.info(
                    f"🚫 [{sym}] [1H Regime Veto{_vspk_tag}] short 被拦截 | "
                    f"1H_regime={_1h_regime_score:.2f}（极强趋势，阈值={_reg_hi:.2f}）"
                )
                self.fast_lane._clear_ai_cache(symbol=sym)
                return

            # ── 置信度动态阈值：三层分离（基础 + 修正 + 惩罚）──
            _trend_score, _trend_dir = get_trend_alignment_score(ind_15m, ind_1h, ind_4h)

            # 层1：基础门槛（按市场模式查表）
            _base_mode = market_mode if market_mode in ("趋势", "震荡", "震荡激进") else "震荡"
            _conf_base = {"趋势": 0.55, "震荡": 0.60, "震荡激进": 0.62}[_base_mode]

            # 层2：唯一修正项 — VSpike 特权（≥6x + 方向明确 + 连亏<3 时降门槛）
            _conf_threshold = _conf_base
            _vspk_priv = (
                _vs_mult >= 6.0
                and (_vs_now.get("is_spike") or _vs_now.get("spike_recent"))
                and gs_get("consecutive_losses", 0) < 3
            )
            if _vspk_priv:
                _vs_bp = _vs_now.get("buy_pct", 0.5)
                if _vs_bp >= 0.70 or _vs_bp <= 0.30:
                    _conf_threshold = max(0.40, _conf_threshold - 0.15)
                    log.info(
                        f"⚡ [{sym}] [VSpike特权通道] {_vs_mult:.1f}x + 方向明确(buy={_vs_bp:.0%})"
                        f" → conf门槛降至 {_conf_threshold:.2f}，首单仓位×0.5"
                    )

            # 层3：唯一惩罚项 — 连亏加价（非线性递增）
            _consec = gs_get("consecutive_losses", 0)
            if _consec >= 5:
                _extra_conf = 0.12
            elif _consec >= 3:
                _extra_conf = 0.06
            else:
                _extra_conf = 0.0
            if _extra_conf > 0:
                _conf_threshold = min(0.85, _conf_threshold + _extra_conf)
                log.info(
                    f"⚠️ [{sym}] 连亏 {_consec} 次，提升开仓 conf 门槛 +{_extra_conf:.2f} → {_conf_threshold:.2f}"
                )

            conf_threshold = _conf_threshold

            if conf < conf_threshold:
                _src = f"(贝叶斯后验={_post_conf:.3f})" if _post_conf else ""
                # P1-3: 限频，置信度不足跳过日志每90秒最多输出一次INFO
                if self._should_log(f"conf_skip_{sym}", 90.0):
                    log.info(f"⚖️ [{sym}] 置信度 {conf:.2f}{_src} < {conf_threshold:.2f}，跳过")
                else:
                    log.debug(f"⚖️ [{sym}] 置信度 {conf:.2f}{_src} < {conf_threshold:.2f}，跳过（限频）")
                self.fast_lane._clear_ai_cache(symbol=sym)  # 清缓存，防止同一条注定被拒的决策重复撞墙
                return

            # ── ConvictionScore 综合质量底线（仅拦截极端劣质信号）──
            _cv_score = self._conviction.score(
                ai_conf=_raw_conf,
                action=action,
                vspike_mult=_vs_mult,
                ob_imbalance=depth.get("imbalance", 0.0) if depth else 0.0,
                rsi=ind_15m.get("rsi", 50.0),
                at_key_level=False,
                market_mode=_sym_market_mode,
                context={
                    "buy_pct": _vs_now.get("buy_pct", 0.5),
                    "atr_ratio": ind_15m.get("atr_ratio", 1.0),
                },
            ).get("score", 0.0)
            if _cv_score < 35.0 and not _vspk_priv:
                log.warning(f"🚫 [{sym}] ConvictionScore综合质量底线 {_cv_score:.1f}<35，拦截")
                self.fast_lane._clear_ai_cache(symbol=sym)
                return
            decision["_cv_score_frozen"] = _cv_score
            decision["_cv_score_frozen_ts"] = time.monotonic()

            # ── VSpike 方向一致性检查（防陈旧信号+市场微观翻转）──
            # 复用 _vs_now（3729行已获取），时间差<3s不影响
            _vs_mult_open = _vs_now.get("mult", 1.0)
            _vs_bp_open = _vs_now.get("buy_pct", 0.5)
            _vs_dir_open = _vs_now.get("direction", "均衡")
            # ① 显式方向冲突：buy_pct ≥70% 且方向与信号相反 → 拦截
            _vs_conflict = (
                (action == "open_long" and "卖方主导" in _vs_dir_open and _vs_bp_open < 0.30) or
                (action == "open_short" and "买方主导" in _vs_dir_open and _vs_bp_open > 0.70)
            )
            if _vs_conflict:
                log.warning(
                    f"🚫 [{sym}] [VSpike硬拦截|开仓阶段] {action} 与量能{_vs_dir_open}冲突"
                    f"（mult={_vs_mult_open:.1f}x, buy={_vs_bp_open:.0%}），取消开仓"
                )
                self.fast_lane._clear_ai_cache(symbol=sym)
                return
            # ② 极端反向量能 ≥8x → 最后防线
            _vs_extreme_against = (
                (action == "open_long" and "卖方主导" in _vs_dir_open and _vs_bp_open < 0.20) or
                (action == "open_short" and "买方主导" in _vs_dir_open and _vs_bp_open > 0.80)
            )
            if _vs_mult_open >= 8.0 and _vs_extreme_against:
                log.warning(
                    f"🛡️ [{sym}] [极端防线|开仓阶段] VSpike≥8x反向拦截: {action} vs {_vs_dir_open}"
                    f"（mult={_vs_mult_open:.1f}x, buy={_vs_bp_open:.0%}）"
                )
                self.fast_lane._clear_ai_cache(symbol=sym)
                return

            # ── 统一风险因子链（Kelly 直接作为风险预算，乘以此链）──────────────
            market_factor, level_proximity_thresh = self.position_exec._apply_market_mode_adjustments(
                _sym_market_mode, conf
            )
            res_levels = key_levels.get("resistances", [])
            sup_levels = key_levels.get("supports", [])

            # ── 关键位距离预警（供 AI 决策参考，提前计算）──────────────────────
            # 计算并缓存最近关键位距离，在 AI 决策后用于软/硬拦截判断
            _level_warn = None  # (方向, 价位, 距离百分比, 软拦截阈值)
            if is_long and res_levels:
                nearest_res = min(res_levels, key=lambda x: abs(_price_of_level(x) - price))
                _dist_pct  = (_price_of_level(nearest_res) - price) / price
                if 0 < _dist_pct < level_proximity_thresh:
                    _level_warn = ("long", _price_of_level(nearest_res), _dist_pct, level_proximity_thresh)
            elif not is_long and sup_levels:
                nearest_sup = min(sup_levels, key=lambda x: abs(_price_of_level(x) - price))
                _dist_pct  = (price - _price_of_level(nearest_sup)) / price
                if 0 < _dist_pct < level_proximity_thresh:
                    _level_warn = ("short", _price_of_level(nearest_sup), _dist_pct, level_proximity_thresh)

            # ── 关键位 + 近场失衡硬拦截（离关键位 <0.3% 且盘口明显反向）──
            _imbalance = depth.get("imbalance", 0.0) if depth else 0.0
            if action == "open_long" and res_levels:
                _nr = min(res_levels, key=lambda x: abs(_price_of_level(x) - price))
                _nr_dist = (_price_of_level(_nr) - price) / price
                if 0 < _nr_dist < 0.003 and _imbalance < -0.30:
                    if conf < 0.78:
                        log.info(
                            f"🚫 [{sym}] [关键位+OB Veto] long 被拦截 | "
                            f"距阻力 {_nr_dist*100:.2f}%, imbalance={_imbalance:.2f}"
                        )
                        self.fast_lane._clear_ai_cache(symbol=sym)
                        return
                    else:
                        log.warning(
                            f"⚠️ [{sym}] [关键位+OB Veto] 高置信度逃逸 | "
                            f"距阻力 {_nr_dist*100:.2f}%, imbalance={_imbalance:.2f}, conf={conf:.2f}"
                        )
            elif action == "open_short" and sup_levels:
                _ns = min(sup_levels, key=lambda x: abs(_price_of_level(x) - price))
                _ns_dist = (price - _price_of_level(_ns)) / price
                if 0 < _ns_dist < 0.003 and _imbalance > 0.30:
                    if conf < 0.78:
                        log.info(
                            f"🚫 [{sym}] [关键位+OB Veto] short 被拦截 | "
                            f"距支撑 {_ns_dist*100:.2f}%, imbalance={_imbalance:.2f}"
                        )
                        self.fast_lane._clear_ai_cache(symbol=sym)
                        return
                    else:
                        log.warning(
                            f"⚠️ [{sym}] [关键位+OB Veto] 高置信度逃逸 | "
                            f"距支撑 {_ns_dist*100:.2f}%, imbalance={_imbalance:.2f}, conf={conf:.2f}"
                        )

            # ── 已有持仓，禁止重复开仓 ────────────────────────────────────
            if self.pos.side:
                log.info(f"⏸️ [{sym}] 当前已有持仓，不再开新仓")
                return

            # ── 总名义杠杆率上限（防三箭齐发超额敞口）──────────────────────
            exposure = self.risk_guard._total_exposure_pct()
            if exposure >= CFG.max_total_exposure:
                log.info(
                    f"⏸️ [{sym}] 总名义杠杆率 {exposure:.2f}x ≥ 上限 {CFG.max_total_exposure}x，"
                    f"暂不开新仓（当前总敞口过高，等待已有仓位平仓后再入场）"
                )
                return

            if self.risk_guard._check_funding_risk(funding, action, confidence=conf):
                return
            with self.lock:
                if (pos.last_open_time and
                        (now - pos.last_open_time).total_seconds() < CFG.min_open_interval_m * 60):
                    log.info(f"⏳ [{sym}] 距上次开仓不足 {CFG.min_open_interval_m} 分钟，跳过")
                    return

            # ── 统一风险因子链 ────────────────────────────────────────────────
            # AI 动态风险倍数优先：若 AI 输出 dynamic_risk_mult，则跳过固定系数链
            _ai_risk_mult = decision.get("dynamic_risk_mult")
            _risk_mult_from_ai = False
            if (isinstance(_ai_risk_mult, (int, float))
                    and 0.3 <= _ai_risk_mult <= 2.0
                    and action in ("open_long", "open_short")):
                risk_mult = round(float(_ai_risk_mult), 3)
                _risk_mult_from_ai = True
                log.info(f"📊 [{sym}] AI 动态 risk_mult={risk_mult:.3f}（覆盖固定系数）")
            else:
                consecutive  = gs_get("consecutive_losses", 0)
                # 非线性衰减：指数衰减曲线，连亏越多衰减越快
                # 连亏 1 次→0.90, 2 次→0.81, 3 次→0.73, 4 次→0.66, 5 次→0.59, 10 次→0.35
                _decay_base = 0.90
                consec_factor = max(0.25, _decay_base ** consecutive)
                risk_mult = round(market_factor * consec_factor * self.dynamic_risk_factor, 3)
                risk_mult = max(0.3, min(1.5, risk_mult))  # 硬限：[0.3, 1.5]

            # ── AI 分步建仓：首单比例折入统一风险因子链 ──────────────────────
            _pyramid_plan = self.position_exec._apply_pyramid_plan(decision, action, _conf)
            if _pyramid_plan and _conf >= 0.70 and not _risk_mult_from_ai:
                _initial_ratio = float(_pyramid_plan.get("initial_ratio", 1.0) or 1.0)
                _initial_ratio = max(0.1, min(1.0, _initial_ratio))
                if _initial_ratio < 1.0:
                    log.info(f"📐 [{sym}] AI首单比例 {_initial_ratio:.0%}，risk_mult: {risk_mult:.3f}→{round(risk_mult*_initial_ratio,3):.3f}")
                    risk_mult = round(risk_mult * _initial_ratio, 3)
                    risk_mult = max(0.1, min(1.5, risk_mult))  # 首单比例调整时降低下限，允许更小试探仓
            else:
                _pyramid_plan = None  # 无计划或非开仓动作则置空
            if not _risk_mult_from_ai:
                log.debug(f"📐 [{sym}] 统一风险因子链: market={market_factor:.2f} × consec={consec_factor:.2f} × dyn={self.dynamic_risk_factor:.2f} → risk_mult={risk_mult:.3f}")

            # ── VSpike ≥6x 特权通道首单仓位减半（连亏≥3时不生效）──
            _vspk_priv_size = (
                _vs_mult >= 6.0
                and (_vs_now.get("is_spike") or _vs_now.get("spike_recent"))
                and gs_get("consecutive_losses", 0) < 3
            )
            if _vspk_priv_size:
                _vs_bp_s = _vs_now.get("buy_pct", 0.5)
                _vs_dir_strong_s = _vs_bp_s >= 0.70 or _vs_bp_s <= 0.30
                if _vs_dir_strong_s:
                    risk_mult = round(risk_mult * 0.5, 4)
                    log.info(
                        f"🔒 [{sym}] [VSpike特权通道] 首单仓位×0.5 → risk_mult={risk_mult:.4f}"
                    )

            # ── 滑点自适应（高波动时放宽漂移容忍 + 降仓位降风险）──────────────
            _atr_val = ind_15m.get("atr", 0)
            _atr_q = _get_atr_quantile(_atr_val, self._atr_history) if _atr_val > 0 else 0.5
            _vs_now_slip = self.vspike.get_status()
            _vs_mult_slip = _vs_now_slip.get("mult", 0.0)
            _slip_adaptive = _atr_q > 0.85 or _vs_mult_slip >= 5.0
            _slip_risk_mult = 1.0  # 默认不调整
            _adaptive_drift = CFG.open_wait_price_drift_pct
            if _slip_adaptive:
                _adaptive_drift = CFG.open_wait_price_drift_pct * 2.5
                _slip_risk_mult = 0.7
                log.info(
                    f"🌊 [{sym}] 滑点自适应激活（ATR分位={_atr_q:.2f}, VSpike={_vs_mult_slip:.1f}x）："
                    f"漂移阈值 {CFG.open_wait_price_drift_pct*100:.1f}%→{_adaptive_drift*100:.2f}%, risk_mult×{_slip_risk_mult}"
                )

            # ── AI 指导的开仓等待 + 价格稳定性检查 ─────────────────────────
            _wait_sec = int(decision.get("wait_seconds", 0))
            _wait_sec = max(0, min(_wait_sec, 5))   # 硬限 0~5 秒，防 AI 给出过长等待
            if _wait_sec > 0:
                log.info(f"⏱️ [{sym}] AI 建议等待 {_wait_sec}s 后开仓（价格漂移阈值={_adaptive_drift*100:.1f}%）")
                time.sleep(_wait_sec)
                fresh_price = self._get_price(sym)
                if fresh_price > 0 and price > 0:
                    drift = abs(fresh_price - price) / price
                    if drift > _adaptive_drift:
                        log.info(
                            f"🚫 [{sym}] 价格稳定性检查失败：等待 {_wait_sec}s 后价格从 {price:.4f} "
                            f"漂移至 {fresh_price:.4f}（偏离 {drift*100:.3f}% > "
                            f"{_adaptive_drift*100:.1f}%），自动取消开仓"
                        )
                        return
                    log.info(f"✅ [{sym}] 价格稳定性检查通过（漂移 {drift*100:.3f}%），执行开仓")

            # ── 关键位软/硬拦截检查（在 AI 决策之后执行）────────────────────────
            # _level_warn = None：无需检查；_level_warn = (方向, 价位, 距离%, 软阈值)
            _near_level_mult = 1.0  # 默认不降仓位，由软拦截动态调整
            if _level_warn is not None:
                _lvl_dir, _lvl_price, _lvl_dist, _lvl_thresh = _level_warn
                _lvl_type = "阻力" if _lvl_dir == "long" else "支撑"
                _lvl_ai_override = decision.get("override_level_proximity", False)

                # ── 动态 ATR 硬阈值 ─────────────────────────────────────────
                # 波动率高 → 阈值放宽（避免正常波动被拦截）；波动率低 → 收紧（下杀即阻）
                _atr_pct = ind_15m.get("atr", 0) / max(price, 1e-9)
                _dynamic_hard_thresh = max(0.0008, min(0.002, 0.2 * _atr_pct))

                if _lvl_dist < _dynamic_hard_thresh:
                    # 硬熔断：距离 < 动态阈值，无论 AI 如何判断均强制拦截
                    log.warning(
                        f"🚫 [{sym}] 关键位硬拦截：价格距L3{_lvl_type}仅{_lvl_dist*100:.2f}% "
                        f"（< 动态阈值{_dynamic_hard_thresh*100:.2f}%，ATR%={_atr_pct*100:.2f}%），强制禁止开仓"
                    )
                    log_event("blocked_by_level_hard", {
                        "sym": sym, "action": action,
                        "price": price, "level": _lvl_price, "dist_pct": _lvl_dist,
                        "dynamic_thresh": _dynamic_hard_thresh, "atr_pct": _atr_pct,
                    })
                    return

                # ── 软拦截：动态阈值 ~ 软阈值区间 → 降仓位 + 同步调 SL ─────────
                # 计算降权系数：距关键位越近，仓位越小（50%~100% 线性）
                if _lvl_dist < _lvl_thresh and not _lvl_ai_override:
                    _near_level_mult = max(0.5, min(1.0, _lvl_dist / max(_lvl_thresh, 1e-9)))
                    _near_level_mult = max(0.5, _near_level_mult)  # 最低保留 50%
                    decision["near_level_mult"] = _near_level_mult
                    # 同步调 SL：向关键位方向收紧（降低被扫止损风险）
                    _sl_tighten = max(0.85, min(0.97, _near_level_mult + 0.05))
                    decision["sl_tighten_mult"] = min(
                        decision.get("sl_tighten_mult", 1.0),
                        _sl_tighten
                    )
                    log.info(
                        f"🚧 [{sym}] 关键位软拦截降仓位：做{_lvl_dir}，距L3{_lvl_type}"
                        f"{_lvl_dist*100:.2f}% < 软阈值{_lvl_thresh*100:.1f}%，"
                        f"降权×{_near_level_mult:.2f} + SL收紧×{_sl_tighten:.2f}（AI override={_lvl_ai_override}）"
                    )
                    log_event("level_soft_reduction", {
                        "sym": sym, "action": action,
                        "level": _lvl_price, "dist_pct": _lvl_dist,
                        "near_level_mult": _near_level_mult, "sl_tighten": _sl_tighten,
                    })
                elif _lvl_ai_override:
                    log.info(
                        f"✅ [{sym}] AI _override=true，价格已突破L3{_lvl_type} "
                        f"（距{_lvl_dist*100:.2f}%，AI 判断可突破），继续执行"
                    )

            # ── 资金费率仓位放大器 ────────────────────────────────────────────
            # 资金费率方向与开仓方向一致时（多头付费率→做多、空头收费率→做空），
            # 意味着套利资金在推高费率，吸引更多同向资金入场，视为正向加持 ×1.2~1.5
            _fund_rate = funding.get("funding_rate", 0) if funding else 0
            _open_dir_match_funding = (
                (_fund_rate > CFG.funding_rate_thresh and action == "open_long")
                or (_fund_rate < -CFG.funding_rate_thresh and action == "open_short")
            )
            if _open_dir_match_funding and abs(_fund_rate) > 0:
                _fund_amp = 1.0 + min(0.5, (abs(_fund_rate) - CFG.funding_rate_thresh) / 0.005 * 0.3)
                _fund_amp = min(_fund_amp, 1.5)
                risk_mult = round(risk_mult * _fund_amp, 3)
                log.info(f"💰 [{sym}] 资金费率加持放大: ×{_fund_amp:.2f} → risk_mult={risk_mult:.3f} (费率={_fund_rate*100:.4f}%)")

            # ── Regime < 0.3 突破追单熔断 ────────────────────────────────────
            # 即使 Prompt 已约束，AI 仍可能在弱势市场输出追突破动作
            # 执行层二次验证：Regime极低时将追突破降级为 hold，节省 Gas
            _reg_open = ind_15m.get("regime_score", 0.5)
            if _reg_open < 0.3 and action in ("open_long", "open_short"):
                _ai_reason = decision.get("reason", "") or ""
                _breakout_kw = ("突破", "breakout", "break out", "破位", "新高", "新低", "突穿", "站上", "跌破")
                # 排除语义完整的否定/反义短语（取代原来的 3 字符前缀检测，覆盖更准）
                _false_breakout_phrases = (
                    "回踩突破", "突破失败", "假突破", "突破后回落", "前期突破",
                    "未突破", "没有突破", "不突破", "突破未果", "突破后回踩",
                    "突破无效", "突破被吞没",
                )
                # 优先：任何假突破短语出现在 reason 中 → 整个不算追突破
                _is_breakout_ai = any(fp in _ai_reason for fp in _false_breakout_phrases)
                if not _is_breakout_ai:
                    # 兜底：关键词匹配 + 前缀否定检查
                    _negation = ("未", "没", "不", "非", "无", "假", "疑", "而非", "并非", "不算", "不是")
                    def _is_true_breakout(reason: str, kw: str) -> bool:
                        idx = reason.lower().find(kw)
                        if idx < 0:
                            return False
                        prefix = reason[max(0, idx - 3):idx]
                        return not any(neg in prefix for neg in _negation)
                    _is_breakout_ai = any(_is_true_breakout(_ai_reason, kw) for kw in _breakout_kw)
                if _is_breakout_ai:
                    log.warning(
                        f"⚠️ [逻辑熔断] Regime({_reg_open:.2f})<0.3，AI 疑似追突破"
                        f"（reason={_ai_reason[:60]}），已降级为 hold"
                    )
                    log_event("regime_circuit_break", {
                        "sym": sym, "regime": _reg_open,
                        "action": action, "reason": _ai_reason[:80]
                    })
                    self.fast_lane._clear_ai_cache(symbol=sym)
                    return

            # ── 开仓前健康检查：REST 验证本地空仓与交易所一致 ──────────────
            try:
                _exch_resp = self.trader.get_positions()
                if _exch_resp.get("code") == "0":
                    _exch_sym_pos = [
                        p for p in _exch_resp.get("data", [])
                        if p.get("instId") == sym and abs(float(p.get("pos", 0))) > 0
                    ]
                    if _exch_sym_pos:
                        _ep = _exch_sym_pos[0]
                        _ex_side = _ep.get("posSide", "?")
                        _ex_size = _ep.get("pos", "?")
                        log.warning(f"⚠️ [{sym}] 开仓前检测到交易所有持仓({_ex_side} {_ex_size})，跳过开仓，先同步状态")
                        self.state.sync_position(sym)
                        return  # 跳过本次开仓
                else:
                    log.debug(f"[{sym}] 开仓前REST校验失败: {_exch_resp}")
            except Exception as _e:
                log.debug(f"[{sym}] 开仓前健康检查异常: {_e}")

            self.position_exec._do_open(decision, price, balance, ind_15m["atr"], funding, decision_id,
                          risk_mult=risk_mult, symbol=sym,
                          market_mode=_sym_market_mode, ind_15m=ind_15m,
                          pyramid_plan=_pyramid_plan,
                          depth=depth,
                          trend_score=_trend_score,
                          news_sentiment=news_data.get("sentiment", 0.0) if news_data else 0.0,
                          fear_greed=fg_index.get("value", 50) if fg_index else 50,
                          slip_risk_mult=_slip_risk_mult)
            # 开仓后重置 AI 门控器时间戳
            self._ai_gate._last_request_ts = time.monotonic()
            self._ai_gate._last_decision_price = price
            self._ai_gate._last_decision_rsi = ind_15m.get("rsi", 50)
            self._ai_gate._last_decision_macd = ind_15m.get("macd_hist", 0)

        # ── 连续 hold 计数 ───────────────────────────────────────────────
        if action in ("hold", "skip", "adjust_sl_tp"):
            self._consecutive_hold += 1
            _ai_wait = int(decision.get("wait_seconds", -1))  # -1 = AI 未给出
            self._ai_hold_wait = max(15, min(90, _ai_wait)) if _ai_wait >= 0 else 15
        else:
            self._consecutive_hold = 0
            self._ai_hold_wait = -1  # 重置为"未给出"，下次由 AI 重新决定

        # ── OB 失衡度缓存（供 get_dynamic_interval 次轮读取）──────────────
        if depth:
            self._last_ob_imbalance = float(depth.get("imbalance", 0))



# ============================================================
# 启动入口
# ============================================================
def main():
    from common import log
    # ── DeepSeek 主决策模型（必须）──────────────────────────────────────
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not deepseek_key:
        log.error("❌ DEEPSEEK_API_KEY 未设置，请检查 .env 文件")
        return

    # ── 创建 DeepSeek client（SmartAIConsultant 用）────────────────────
    ai_client = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",
        timeout=60,
    )
    log.info("🤖 ETH Trader 启动中...")
    bot = ETHTrader(ai_client)
    log.info("✅ ETH Trader 已启动（DeepSeek 主决策 + Qwen 仲裁/趋势补漏）")
    bot.run()


if __name__ == "__main__":
    main()
