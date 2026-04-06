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
# import pandas_ta as ta  # 未使用，手动计算所有指标
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
    log_kelly_metrics, init_db, get_db_conn, get_sys_config, set_sys_config,
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
    build_market_context, build_macro_context,
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
        with self.trader._margin_lock:
            if self.trader._reserved_margin > 0:
                log.debug(f"_reset_pos: 清理残留 _reserved_margin={self.trader._reserved_margin:.2f}U")
                self.trader._reserved_margin = 0.0

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
                if not pos.open_time:
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
                    f"（加权胜率={win_rate*100:.1f}% 样本=近{n}笔）"
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
                        DB = self.trader.db
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
        SYM = CFG.symbol  # 整个类内部都用这个常量
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

        # ── 缓存 ──────────────────────────────────────────────────────────
        _empty_cache = lambda: {"data": None, "time": datetime.min.replace(tzinfo=UTC)}
        self.funding_cache     = _empty_cache()
        self.key_levels_cache  = _empty_cache()
        self.funding_history:  List = []
        # 共享缓存
        self.news_cache      = _empty_cache()
        self.fg_cache        = _empty_cache()
        self.sentiment_cache = _empty_cache()
        self.macro_cache     = {"data": "", "time": datetime.min.replace(tzinfo=UTC)}

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
        self._last_ob_imbalance:    float = 0.0  # 上轮 OB 失衡度，供 get_dynamic_interval 次轮读取
        # raw K 线缓存：bar → (data, fetch_monotonic, last_candle_ts)
        # TTL 按周期设定：3m=45s, 15m=120s, 1H=300s, 4H=600s
        self._raw_kline_cache: Dict[str, tuple] = {}
        self._RAW_KLINE_TTL: Dict[str, int] = {"3m": 45, "15m": 120, "1H": 300, "4H": 600}
        self._last_adjust_time:     datetime = datetime.min.replace(tzinfo=UTC)

        # ── 账户余额 ─────────────────────────────────────────────────────
        self.latest_equity:    float = 0.0
        self.latest_avail_bal: float = 0.0

        # ── AI 异步决策缓存 ───────────────────────────────────────────────
        self._ai_cache:        Optional[Dict] = None
        self._ai_cache_ts:     float = 0.0     # AI 缓存时间戳（monotonic），用于判断缓存是否过期
        self._ai_cache_lock:   threading.Lock = threading.Lock()
        self._ai_hash:         str  = ""
        self._ai_running_flag: bool = False
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
        # ── 人工审核控制 ──────────────────────────────────────────────────
        self._ai_blocked: bool = False              # /block_ai 禁用 AI 决策
        # ── 决策摘要队列（最近10次，供 /health 和 /ai/summaries 消费）──────
        self._ai_summaries: deque = deque(maxlen=10)
        # ── 告警去重时间戳 dict（monotonic），防循环告警 ──────────────────
        self._last_alert_ts: Dict[str, float] = {}
        # ── 打板信号冷却时间戳（防同一突破在冷却期内重复触发）────────────
        self._last_breakout_ts: Dict[str, float] = {"up": 0.0, "down": 0.0}
        # ── close被拦截冷却（防同一VSpike事件触发Token死循环）─────────────
        # 格式：{symbol: {"spike_ts": float, "blocked_until": float}}
        self._blocked_close_ctx: Dict[str, Dict] = {}
        # AI 平仓协调标志：防止 AI 平仓与追踪止损重复下单
        self._ai_close_pending_until: float = 0.0  # monotonic 时间戳
        # ── 决策去重签名（防相同决策每轮输出完整 thought_process）──────────
        self._last_decision_sig: Dict[str, str] = {}
        # ── 连续 hold 计数 + AI 建议的下次调用间隔（秒）────────────────────
        self._consecutive_hold: int = 0
        self._ai_hold_wait:     int = -1   # -1=未设定，0=AI说立即，其他=指定秒数
        # ── AI 强择打破静默（hold 时 AI 可主动要求 force_wakeup）───────────
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

        # ── Phase 1: AIGatekeeper ────────────────────────────
        self._ai_gate = AIGatekeeper()
        self._conviction = ConvictionScorer()
        # ── 千问仲裁触发器 ───────────────────────────────────────
        self._arbitration = ArbitrationTrigger()
        # ── AI 表现追踪全局状态初始化 ──────────────────────────
        if gs_get("ai_win_history", None) is None:
            gs_set("ai_win_history", [])
        if gs_get("ai_decision_conf_history", None) is None:
            gs_set("ai_decision_conf_history", [])
        if gs_get("ai_recent_win_rate", None) is None:
            gs_set("ai_recent_win_rate", 0.5)
        if gs_get("ai_weight_mult", None) is None:
            gs_set("ai_weight_mult", 1.0)
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

        # ── 快速决策通道防抖缓存 ──
        # 格式: {"sym_SIGNAL_KEY": (action, conf, monotonic_ts)}
        self._fast_signal_cache: Dict[str, tuple] = {}
        # ── 快速决策拒绝反馈缓存 ──
        # ConvictionScore 连续拒绝的信号进入冷却，防止低质量信号无限重试
        # 格式: {"sym_action": (action, reject_count, monotonic_ts)}
        self._fast_rejection_cache: Dict[str, tuple] = {}
        # ── AI 最近方向性信号跟踪 ──
        # 防止 Path B 用规则引擎信号覆盖 AI 的方向性判断
        # 格式: {"sym": (action, monotonic_ts)}
        self._last_ai_directional: Dict[str, tuple] = {}
        # ── ConvictionScore 拒绝的决策跟踪 ──
        # 格式: (action, score, monotonic_ts) — 被 ConvictionScore 拒绝后设置，驱动静默等待
        self._cv_rejected_decision: tuple = ()

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
                    return 30
                mult = min(5, 1 + self._consecutive_hold // 3)
                return min(300, int(60 * mult))  # 慢车道上限从 600s → 300s

        # ── Step 4: 默认间隔 ─────────────────────────────────────────────
        return CFG.check_interval_hold if has_pos else CFG.check_interval_empty

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

    def _should_skip_ai_request(self, symbol: str, ind_15m: Dict, ind_1h: Dict, current_price: float) -> bool:
        """
        静默拦截：持仓时若市场无明显变化且未超过静默期，则跳过AI请求。
        对空仓同样生效（垃圾时间直接跳过）。
        返回 True 表示应该跳过 AI 请求，使用缓存决策或默认 hold。
        """
        # AI 熔断器：连续5次失败后暂停30分钟，降级为规则引擎
        now_mono = time.monotonic()
        if self._ai_gate.circuit_broken:
            log.warning(f"🛑 [{symbol}] AI 熔断中（AIGatekeeper），跳过 AI 请求")
            return True

        # ── AI 主动打破静默（force_wakeup）—— 最高优先级，立即生效 ─────────
        if self._last_force_wakeup.get(symbol, False):
            log.info(f"⚡ [{symbol}] 上轮 AI 输出 force_wakeup=true，本轮强制唤醒")
            self._last_force_wakeup[symbol] = False  # 用完清除
            self.fast_lane._clear_ai_cache(symbol=symbol)  # 清缓存，确保真实 AI 调用
            return False

        # ── 秒级成交量突增 → 打破静默，让 AI 看到 VSpike 上下文 ─────────
        if self.vspike.get_status().get("is_spike"):
            # close被拦截冷却检查：同一spike事件内AI已反复拦截，暂停重复调用
            _bctx = self._blocked_close_ctx.get(symbol, {})
            _now_mono_vs = time.monotonic()
            _spike_peak_ts = self.vspike._spike_peak_ts
            # 若是同一spike事件（peak_ts相同）且冷却期未过 → 跳过，不打破静默
            # VSpike 极端值 (≥10x) 豁免 close 拦截冷却
            _vs_mult_extreme = self.vspike.get_status().get("mult", 0.0)
            if (_bctx.get("spike_ts") == _spike_peak_ts
                    and _now_mono_vs < _bctx.get("blocked_until", 0)
                    and _vs_mult_extreme < CFG.vspike_extreme_mult):  # 极端 VSpike 豁免冷却
                _remain = _bctx["blocked_until"] - _now_mono_vs
                log.debug(f"🛑 [{symbol}] close拦截冷却中（同一spike事件，剩余{_remain:.0f}s），跳过AI唤醒")
                return True
            if _vs_mult_extreme >= CFG.vspike_extreme_mult:
                log.info(f"⚡ [{symbol}] VSpike 极端值 ({_vs_mult_extreme:.1f}x) 豁免 close 拦截冷却")
            log.info(f"🔥 [{symbol}] 秒级成交量突增，打破静默重新决策")
            self.fast_lane._clear_ai_cache(symbol=symbol)  # 强制清缓存，确保 AI 真实调用（而非返回旧 hold）
            # VSpike打破静默时同步清除BREAKOUT防抖缓存，防止同一轮打板信号因300s冷却被误杀
            for _bkey in (f"{symbol}_BREAKOUT_UP", f"{symbol}_BREAKOUT_DOWN"):
                self._fast_signal_cache.pop(_bkey, None)
            # ── VSpike 快车道打标：极端量能时激活 AGGRESSIVE 模式 ─────────────
            _vs_mult_now = self.vspike.get_status().get("mult", 0.0)
            if _vs_mult_now >= CFG.vspike_priority_threshold:
                self._ai_gate.mark_entry_fasttrack(_vs_mult_now)
                log.info(
                    f"[AIGatekeeper] Mode: AGGRESSIVE | "
                    f"Reason: VSpike={_vs_mult_now:.1f}x >= {CFG.vspike_priority_threshold:.0f}x | "
                    f"Bypassing Cache: True"
                )
            return False

        # ── 垃圾时间过滤（对空仓和持仓都生效）───────────────────────────
        # 注意：_consecutive_hold 计数器统一在 _run_symbol 主逻辑中管理，
        # 此处仅负责判断是否跳过，不单独增量（避免重复计数）。
        # ── 1H ADX + Regime 双重过滤：极弱震荡 = 物理垃圾时间 ───────────────
        # 1H ADX < 20 → 无趋势；Regime < 0.3 → 市场骨架极弱
        # 两者同时满足 → "10x杠杆进去就是送手续费"，直接跳过 AI 节省 Token
        # 豁免条件：VSpike ≥6.0x 时，强成交量信号本身已构成开仓依据，值得 AI 判断
        _adx_1h = ind_1h.get("adx", 25) if ind_1h else 25
        _regime = ind_15m.get("regime_score", 0.5)
        _phys_spike = self.vspike.get_status()
        _phys_spike_mult = _phys_spike.get("mult", 0.0) if (
            _phys_spike.get("is_spike") or _phys_spike.get("spike_recent")
        ) else 0.0
        if _adx_1h < 20 and _regime < 0.3 and _phys_spike_mult < 6.0:
            log.info(
                f"🛡️ [{symbol}] 物理拦截：1H_ADX({_adx_1h:.1f})<20 且 Regime({_regime:.2f})<0.3"
                f" → 极弱震荡，无操作价值，跳过AI节省Token"
                f"（VSpike={_phys_spike_mult:.1f}x < 6.0 无豁免）"
            )
            return True

        _market_mode = get_market_mode(ind_15m, current_price, self._market_mode)
        _rsi = ind_15m.get("rsi", 50)
        # 动态 RSI 阈值：优先使用 AI 上次 hold 决策时建议的范围，否则使用默认 [40, 60]
        _rsi_low, _rsi_high = self._next_wakeup_rsi if self._next_wakeup_rsi else (40, 60)
        if (_market_mode == "震荡"
                and _rsi_low <= _rsi <= _rsi_high):
            kl = self.key_levels_cache.get("data") or {}
            if kl.get("_valid"):
                res = kl.get("resistances", [])
                sup = kl.get("supports", [])
                if res or sup:
                    def _d(levels):
                        return min(abs(current_price - (float(l["price"]) if isinstance(l, dict) else float(l))) / current_price
                                   for l in levels) if levels else float("inf")
                    nearest = min(_d(res), _d(sup))
                    if nearest > 0.01:
                        _dyn_tag = f"(AI动态阈值{self._next_wakeup_rsi})" if self._next_wakeup_rsi else "(默认阈值40~60)"
                        log.debug(f"🗑️ [{symbol}] 垃圾时间跳过AI：震荡/RSI={_rsi:.0f}{_dyn_tag}/距关键位>{nearest*100:.1f}%")
                        return True

        pos = self.pos
        if not pos or not pos.side:
            # 无持仓（且非垃圾时间），不触发静默拦截
            return False

        # ── 浮亏/浮盈超限强制唤醒（防止静默拦截冻住亏损或踏空浮盈持仓）─────────
        if pos.entry_price > 0 and current_price > 0:
            if pos.side == "long":
                _pos_pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            else:
                _pos_pnl_pct = (pos.entry_price - current_price) / pos.entry_price
            if _pos_pnl_pct <= CFG.silence_force_wakeup_loss_pct:
                log.warning(
                    f"⚠️ [{symbol}] 浮亏 {_pos_pnl_pct*100:.2f}% ≤ {abs(CFG.silence_force_wakeup_loss_pct)*100:.0f}%，"
                    f"强制唤醒 AI 重新决策"
                )
                _now_mono_alert = time.monotonic()
                _cooldown = CFG.silence_wakeup_alert_cooldown
                if _now_mono_alert - self._last_alert_ts.get("wakeup_loss", 0) >= _cooldown:
                    self._last_alert_ts["wakeup_loss"] = _now_mono_alert
                    _webhook(
                        f"🚨 [{CFG.symbol}] 浮亏强制唤醒",
                        f"方向:{pos.side} 浮亏:{_pos_pnl_pct*100:.2f}% 当前价:{current_price}"
                    )
                self.fast_lane._clear_ai_cache(symbol=symbol)  # 清缓存，确保 AI 真实看到浮亏状态
                return False
            _atr_wakeup = ind_15m.get("atr", 0)
            _atr_dollar_trigger = _atr_wakeup * CFG.silence_force_wakeup_atr_mult
            if _atr_wakeup > 0 and (_pos_pnl_pct * current_price) >= _atr_dollar_trigger:
                log.warning(
                    f"⚠️ [{symbol}] 浮盈 {_pos_pnl_pct*100:.2f}% ≥ {CFG.silence_force_wakeup_atr_mult}×ATR(≈${_atr_dollar_trigger:.2f})，"
                    f"强制唤醒 AI 考虑止盈/加仓"
                )
                _now_mono_alert = time.monotonic()
                _cooldown = CFG.silence_wakeup_alert_cooldown
                if _now_mono_alert - self._last_alert_ts.get("wakeup_profit", 0) >= _cooldown:
                    self._last_alert_ts["wakeup_profit"] = _now_mono_alert
                    _webhook(
                        f"📈 [{CFG.symbol}] 浮盈强制唤醒",
                        f"方向:{pos.side} 浮盈:{_pos_pnl_pct*100:.2f}% 当前价:{current_price}"
                    )
                self.fast_lane._clear_ai_cache(symbol=symbol)  # 清缓存，确保 AI 真实看到浮盈状态
                return False

        now_mono = time.monotonic()
        holding_seconds = (datetime.now(UTC) - pos.open_time).total_seconds() if pos.open_time else 0
        holding_minutes = holding_seconds / 60

        # 距离上次 AI 请求时间
        last_request_time = self._ai_gate.last_request_ts
        time_since_request = now_mono - last_request_time
        silence_threshold = CFG.max_hold_silence_minutes * 60  # 转换为秒

        # 超过最大静默时长，强制唤醒（硬熔断，防止"冻死"）
        if time_since_request >= silence_threshold:
            log.warning(
                f"⏰ [{symbol}] 静默超限（{holding_minutes:.0f}分钟），强制刷新 AI"
            )
            self.fast_lane._clear_ai_cache(symbol=symbol)  # 清缓存，确保触发真实 AI 调用
            return False

        # 无持仓，不拦截
        if not pos.side:
            return False

        # ── 动态价格波动阈值（趋势市更敏感，震荡市更宽容）─────────────────
        _price_thresh = (
            CFG.hold_silence_price_thresh_trend
            if _market_mode == "趋势"
            else CFG.hold_silence_price_thresh_osc
        )
        # 检查价格波动（若上次AI决策价格不存在，用持仓成本价作基准）
        last_price = self._ai_gate.last_decision_price
        price_change_pct = 0.0
        if last_price > 0:
            price_change_pct = abs(current_price - last_price) / last_price
            if price_change_pct >= _price_thresh:
                # 价格波动超过动态阈值，不拦截
                return False

        # 检查关键指标变化
        current_rsi = ind_15m.get("rsi", 50)
        current_macd_hist = ind_15m.get("macd_hist", 0)
        last_rsi = self._ai_gate.last_decision_rsi
        last_macd_hist = self._ai_gate.last_decision_macd

        # RSI 从非超买/超卖区进入超买/超卖区
        rsi_was_overbought = last_rsi >= 65
        rsi_now_overbought = current_rsi >= 65
        rsi_was_oversold = last_rsi <= 35
        rsi_now_oversold = current_rsi <= 35
        rsi_entered_extreme = (rsi_now_overbought and not rsi_was_overbought) or (rsi_now_oversold and not rsi_was_oversold)

        # MACD 死叉/金叉（方向变化）
        macd_crossed = (current_macd_hist > 0 and last_macd_hist <= 0) or (current_macd_hist < 0 and last_macd_hist >= 0)

        # 有显著指标变化，不拦截
        if rsi_entered_extreme or macd_crossed:
            return False

        # 满足所有静默条件，跳过 AI 请求
        log.debug(f"🛡️ [{symbol}] 静默拦截：持仓{holding_minutes:.0f}分钟 价格波动{price_change_pct*100:.2f}% <{_price_thresh*100:.1f}% RSI={current_rsi:.1f} MACD={current_macd_hist:.4f}")
        return True

    def _is_redundant_fast_signal(self, sym: str, signal_key: str,
                                   action: str, conf: float,
                                   cooldown: int = 300,
                                   force_bypass: bool = False) -> bool:
        """
        快速信号防抖：同一信号源在 cooldown 秒内出现相同 action + conf 变化 <= 0.05 则拦截，
        避免重复 AI 调用烧 token。
        两个逃生条件：
        - force_bypass=True（VSpike 剧烈波动时绕过防抖）
        - new_conf > prev_conf + 0.10（信号显著增强，允许突破）
        使用 time.monotonic() 防止系统时钟回拨。
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
                log.debug(
                    f"[{sym}] {signal_key} 防抖拦截 "
                    f"(action={action} conf={conf:.2f} vs 前次 {prev_conf:.2f} "
                    f"{'+0.10突破' if conf_improved else ''})"
                )
                return True

        self._fast_signal_cache[cache_key] = (action, conf, now)
        return False

    def _trigger_ai_async_sym(self, symbol: str, ind_15m, ind_1h, ind_4h, ind_3m,
                               news_data, fg_index, funding, depth, pos_info,
                               key_levels=None, market_sentiment=None,
                               funding_history=None, macro_context="", rag_warning="",
                               sentiment_alert="", fast_context=""):
        """Per-symbol non-blocking AI trigger"""
        # ── 人工禁用 AI 检查（/block_ai 命令触发）───────────────────────
        if self._ai_blocked:
            log.debug(f"🔒 [{symbol}] AI 决策已被人工禁用，跳过（使用 /unblock_ai 恢复）")
            return
        # 优化8：最小请求间隔由 AI_MIN_REQUEST_INTERVAL 控制（默认10s），避免DeepSeek API限流
        now_mono = time.monotonic()
        min_interval = CFG.ai_min_request_interval
        if now_mono - self._ai_gate.last_request_ts < min_interval:
            log.debug(f"⏳ [{symbol}] AI请求间隔小于{min_interval}秒，跳过")
            return

        # 注：持仓静默拦截已由 _should_skip_ai_request 在 _run_symbol 中统一处理，此处无需重复

        if self._ai_running_flag:
            log.debug(f"⏩ [{symbol}] AI 上轮仍运行，跳过")
            return
        try:
            # ── 缓存键模糊化（降低敏感度，提升命中率）──────────────────────
            _price_raw = ind_15m.get('price', 0)
            market_mode = get_market_mode(ind_15m, _price_raw, self._market_mode)
            self._market_mode = market_mode

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
            _active_cache = self.fast_lane._get_ai_decision(symbol)
            _last_price = self._ai_gate.last_decision_price
            _last_rsi_bkt = self._ai_gate._last_rsi_bkt
            _last_bb_zone = self._ai_gate._last_bb_zone
            if (_active_cache is not None
                    and _last_price > 0 and _last_rsi_bkt is not None and _last_bb_zone is not None):
                _price_drift = abs(_price_raw - _last_price) / _last_price
                if _price_drift < 0.003 and _rsi_bkt == _last_rsi_bkt and _bb_zone == _last_bb_zone:
                    # 价格稳定 + RSI/BB% 未跨桶 → 复用缓存
                    cached = _active_cache
                    log.debug(f"⏭️ [{symbol}] 漂移 {_price_drift*100:.2f}% < 0.3% + RSI/BB%未变，直接复用缓存 → "
                              f"{cached.get('action','?')} (conf={cached.get('confidence',0):.2f})")
                    return

            # ④ 价格分桶：趋势市 $20 / 震荡市 $10
            if market_mode == "趋势":
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
                f"{market_mode}_"
                f"{_vol_bkt}"  # 量能分桶：量能突变（如放量2倍）时触发AI刷新
            )
        except Exception:
            input_sig = ""

        # 强制刷新逻辑：高置信度信号（>= cache_force_refresh_conf）触发强制刷新
        # 修复：再次通过 _get_ai_decision 确认缓存未过期，避免引用已被清除的缓存
        _active_cache = self.fast_lane._get_ai_decision(symbol)
        prev_conf = _active_cache.get("confidence", 0) if _active_cache else 0
        if prev_conf >= CFG.cache_force_refresh_conf and self._ai_hash == input_sig:
            log.debug(f"🔄 [{symbol}] 高置信度缓存({prev_conf:.2f}>={CFG.cache_force_refresh_conf})，强制刷新")
        elif input_sig == self._ai_hash and _active_cache is not None:
            # 缓存命中时只打印精简日志，避免刷屏
            log.debug(f"⏭️ [{symbol}] 命中AI缓存 → {_active_cache.get('action','?')} (conf={_active_cache.get('confidence',0):.2f})")
            return

        def _worker():
            self._ai_running_flag = True
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
                acquired = self._ai_semaphore.acquire(timeout=CFG.ai_timeout_seconds + 10)
                if not acquired:
                    log.warning(f"⚠️ [{symbol}] AI 信号量等待超时({CFG.ai_timeout_seconds + 10}s)，跳过本轮")
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
                    # ── Prompt 注入极端量能特权指令 ───────────────────────────────
                    _ft_mult = self._ai_gate.entry_fasttrack_mult
                    if _ft_mult >= CFG.vspike_priority_threshold:
                        _fasttrack_injection = (
                            f"\n\n【⚡ 极端量能特权模式 VSpike={_ft_mult:.1f}x】\n"
                            f"当前成交量是基准的{_ft_mult:.0f}倍，属于历史极值事件。\n"
                            f"决策指令：将订单流/成交量动力学权重提升至80%，"
                            f"RSI/MACD等滞后指标仅作辅助参考（权重≤20%）。\n"
                            f"若订单流方向明确（buy_pct>65%看多/buy_pct<35%看空），"
                            f"即使技术指标有背离，也应果断顺势决策。\n"
                            f"conf可放宽至0.62以上即可开仓，系统将自动采用试探仓控制风险。"
                        )
                        _fc_combined = (fast_context or "") + _fasttrack_injection
                    else:
                        _fc_combined = fast_context or ""
                    result = self.ai.get_decision(
                        _use_ind15, _use_ind1h, _use_ind4h, news_data, fg_index, funding, depth, pos_info,
                        key_levels=key_levels,
                        funding_history=funding_history or [],
                        macro_context=macro_context,
                        rag_warning=rag_warning,
                        market_sentiment=market_sentiment,
                        prev_market_mode=self._market_mode,
                        sentiment_alert=sentiment_alert,
                        fast_context=_fc_combined,
                        trend_alignment_score=_trend_score,
                        trend_dir=_trend_dir,
                    )
                except Exception as ai_e:
                    log.error(f"[{symbol}] AI调用异常: {ai_e}")
                    result = {"action": "hold", "confidence": 0.0, "reason": f"AI异常: {str(ai_e)[:100]}", "thought_process": ""}
                # 无论成功或异常，都统一在finally中释放信号量
                with self._ai_cache_lock:
                    self._ai_cache = result
                    self._ai_cache_ts = time.monotonic()
                    self._ai_hash = input_sig
                    self._last_ai_request_time = time.monotonic()
                    # 更新死区拦截所需的决策记录（使用开头捕获的值）
                    self._last_ai_decision_time = time.monotonic()
                    self._last_ai_decision_price = _saved_price_val
                    self._last_ai_decision_rsi = _saved_rsi_val
                    self._last_ai_decision_macd = _saved_macd_val
                    # 宽分桶快照（用于漂移容忍判断）
                    self._last_ai_rsi_bkt = _rsi_bkt
                    self._last_ai_bb_zone = _bb_zone
                    # AI 熔断器：AI 调用成功后重置失败计数
                    self._ai_failure_count = 0
                # Phase 1 双写过渡：同步写入 AIGatekeeper（旧变量保留，后续可清理）
                self._ai_gate.set_cache(result, input_sig, _saved_price_val,
                                        _saved_rsi_val, _saved_macd_val, _rsi_bkt, _bb_zone)
                self._ai_gate.reset_failure()
            except Exception as e:
                log.error(f"[{symbol}] AI后台决策异常: {e}")
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

    def _get_ai_decision(self, symbol: str = None) -> Optional[Dict]:
        return self._ai_gate.get_cached(self._market_mode)

    def _clear_ai_cache(self, symbol: str = None):
        self._ai_gate.clear()
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
        win_history = win_history[-10:]
        gs_set("ai_win_history", win_history)

        n = len(win_history)
        win_rate = sum(win_history) / n if n > 0 else 0.5
        gs_set("ai_recent_win_rate", win_rate)

        # 冷启动保护：少于5笔有效样本时强制默认权重
        if n < 5:
            ai_weight_mult = 1.0
            log.debug(f"[AI绩效] 冷启动({n}笔)，权重不调整")
        else:
            if win_rate >= 0.65:
                ai_weight_mult = 1.0
            elif win_rate >= 0.50:
                ai_weight_mult = 0.75
            elif win_rate >= 0.40:
                ai_weight_mult = 0.50
            else:
                ai_weight_mult = 0.40
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
                                pos.side        = pos_data["posSide"]
                                pos.size        = abs(float(pos_data["pos"]))
                                pos.entry_price = float(pos_data["avgPx"])
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
                                # Bug 修复（续）：在 _reset_pos 清空 sl_tp_algo_ids 之前保存
                                # 这样 orders channel 到达时仍能通过 _sl_tp_pending_close 判断是否为 SL/TP 触发
                                # 锁保护防止与 orders channel 竞态
                                with self._sl_tp_cache_lock:
                                    self._sl_tp_pending_trade_id = pos.trade_id
                                    self._sl_tp_pending_algo_ids = list(pos.sl_tp_algo_ids or [])
                                    self._sl_tp_pending_leverage = pos.leverage if pos.leverage > 0 else 1
                                self.state._reset_pos()
                                # 平仓后更新摘要为"无持仓"
                                if self._ai_summaries:
                                    _last = self._ai_summaries[-1]
                                    _last["actual_side"] = "none"
                                    _last["pnl_pct"] = 0.0
                                    log.debug(f"🔄 [{CFG.symbol}] 平仓同步→ none")
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

    # ═══════════════════════════════════════════════════════════════════════
    # 策略委员会：ATR突破趋势策略 + RSI均值回归策略
    # ═══════════════════════════════════════════════════════════════════════

    def _strategy_committee(self, sym: str, ai_decision: Dict,
                             ind_15m: Dict, ind_1h: Dict, ind_3m: Dict, current_price: float,
                             depth: Dict = None) -> Dict:
        """
        策略委员会（建议模式）：AI 完全主导，委员会不修改决策。
        仅在 >=2 个策略反对 且 AI 置信度 < 0.7 时，输出警示日志供人工关注。
        """
        # 策略委员会暂未实装（策略子方法待迁移），直接返回 AI 决策
        return ai_decision

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
            macro_ctx = build_macro_context(raw_daily, current_price)
            self.macro_cache = {"data": macro_ctx, "time": now}
        macro_context = self.macro_cache.get("data", "")

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

    def _run_symbol(self, sym: str, now: datetime, balance: float,
                    news_data: Dict, fg_index: Dict, macro_context: str):
        """单品种决策：获取数据 → AI触发 → 执行动作"""
        log.debug(f"🔄 [{sym}] _run_symbol 开始")
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
        ind_4h  = calc_indicators(raw_4h)

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

        # ── close拦截冷却解封检查（价格偏移 > 0.5×ATR → 行情已变化，提前解封）────
        _bctx = self._blocked_close_ctx.get(sym)
        if _bctx and time.monotonic() < _bctx.get("blocked_until", 0):
            _blk_price = _bctx.get("block_price", 0)
            _blk_atr   = _bctx.get("atr", 0)
            if _blk_price > 0 and _blk_atr > 0:
                _price_shift = abs(price - _blk_price)
                if _price_shift >= _blk_atr * 0.5:
                    log.info(
                        f"🔓 [{sym}] 价格偏移{_price_shift:.2f}≥0.5×ATR({_blk_atr:.2f})，"
                        f"close拦截冷却提前解封（拦截价={_blk_price:.2f} 当前={price:.2f}）"
                    )
                    self._blocked_close_ctx.pop(sym, None)

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
            if len(fh) > 12:
                self.funding_history = fh[-12:]
        else:
            funding = fc["data"]

        # ── Phase 3: 构建统一信号对象 ────────────────────────────
        _signal = MarketSignal().from_indicators(
            ind_15m, self.vspike.get_status(), self._market_mode,
            depth, funding
        )
        _signal.price = price

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
                if ls > 2.0 and price > ind_1h.get('ema_20', price) * 1.02:
                    _sentiment_alert += "⚠️多空比极值(>2.0)+价格高位→强烈抑制做多，可寻找做空机会 "
                elif ls < 0.5 and price < ind_1h.get('ema_20', price) * 0.98:
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
        _osc_market = (
            get_market_mode(ind_15m, price, self._market_mode)
            if (ind_15m and ind_15m.get("_valid"))
            else (self._market_mode or "趋势")
        )
        self._market_mode = _osc_market  # 缓存供本轮所有逻辑及追踪止损使用
        self.vspike.market_mode = _osc_market  # 同步给检测器，用于自适应阈值

        # ── 1.1 规则引擎预筛选（仅在空仓时执行）──────────────────────────────
        # 规则引擎快速评估布林带、RSI、成交量等确定性信号
        # 明确信号（如RSI超买超卖）直接给出决策，无需触发AI
        # 修复5：统一信号判断优先级，规则引擎信号纳入fast_decision统一管理
        fast_decision = None  # 统一快速决策信号初始为空
        if not pos.side:
            rule_result = self.signals.evaluate_rules(
                ind_15m, price,
                prev_indicators=self._prev_indicators,
                market_mode=_osc_market,
                key_levels=key_levels,
            )
            self._prev_indicators = {"rsi": ind_15m.get("rsi"), "vol_surge": ind_15m.get("vol_surge")}
            if not rule_result["trigger_ai"] and rule_result["signal_type"] in ("long", "short"):
                action = f"open_{rule_result['signal_type']}"
                rule_conf = rule_result["confidence"]
                # 获取VSpike状态，用于防抖逃生窗
                _vs4debow = self.vspike.get_status()
                _spike_bypass = _vs4debow.get("spike_just_triggered", False)
                if self.fast_lane._is_redundant_fast_signal(sym, "RULE_RSI", action, rule_conf,
                                                  cooldown=300, force_bypass=_spike_bypass):
                    pass  # 防抖命中，不生成新的 fast_decision，跳到AI推理路径
                else:
                    # ── 拒绝反馈：被 ConvictionScore 连续拒绝的信号进入冷却 ──
                    _rej_key = f"{sym}_{action}"
                    _prev_rej = self._fast_rejection_cache.get(_rej_key)
                    _rej_suppressed = False
                    if _prev_rej:
                        _rej_act, _rej_cnt, _rej_ts = _prev_rej
                        _rej_age = time.monotonic() - _rej_ts
                        if _rej_age >= 600:  # 10 min 过期自动清除
                            self._fast_rejection_cache.pop(_rej_key, None)
                        elif _rej_cnt >= 2 and not (_spike_bypass and _vs4debow.get("mult", 0) >= 6.0):
                            log.info(
                                f"🛡️ [{sym}] 规则引擎信号抑制: {action} 已被ConvictionScore"
                                f"连续{_rej_cnt}次拒绝，冷却剩余{600 - _rej_age:.0f}s"
                            )
                            _rej_suppressed = True
                    if not _rej_suppressed:
                        # ── 分模式硬过滤 ──
                        _vol_rule = ind_15m.get("vol_surge", 0)
                        _ob_rule  = depth.get("imbalance", 0)
                        _adx_rule = ind_15m.get("adx", 0)
                        _is_long_rule = (action == "open_long")
                        if _osc_market == "趋势":
                            _vol_pass = _vol_rule >= 1.4
                            _ob_pass  = (_is_long_rule and _ob_rule > 0.25) or (not _is_long_rule and _ob_rule < -0.25)
                            _adx_pass = _adx_rule >= 22
                            _ok_rule = _vol_pass and _ob_pass and _adx_pass
                            _why_fail = []
                            if not _vol_pass: _why_fail.append(f"vol={_vol_rule:.1f}x<1.4")
                            if not _ob_pass:  _why_fail.append(f"imb={_ob_rule:+.3f}")
                            if not _adx_pass: _why_fail.append(f"adx={_adx_rule:.1f}<22")
                        else:  # 震荡 / 震荡激进
                            _vol_pass = _vol_rule >= 1.2
                            _ob_pass  = (_is_long_rule and _ob_rule > 0.20) or (not _is_long_rule and _ob_rule < -0.20)
                            _ok_rule = _vol_pass and _ob_pass
                            _why_fail = []
                            if not _vol_pass: _why_fail.append(f"vol={_vol_rule:.1f}x<1.2")
                            if not _ob_pass:  _why_fail.append(f"imb={_ob_rule:+.3f}")
                        if not _ok_rule:
                            log.debug(f"🔇 [{sym}] 规则引擎信号被分模式过滤拦截({', '.join(_why_fail)})，降级为AI推理")
                        else:
                            _spike_tag = "  🔥波动唤醒" if _spike_bypass else ""
                            # 置信度封顶：规则引擎信号最高 0.72，强制 AI 最终把关
                            _capped_conf = min(rule_conf, 0.72)
                            if _capped_conf < rule_conf and not _spike_bypass:
                                log.debug(f"📎 [{sym}] 规则引擎置信度封顶: {rule_conf:.2f}→{_capped_conf:.2f}")
                            log.info(f"📋 [{sym}] 规则引擎预筛选{_spike_tag}: {rule_result['reason']}，置信度={_capped_conf:.2f}，直接{action}")
                            fast_decision = {
                                "action": action,
                                "confidence": _capped_conf,
                                "suggested_sl": 0,
                                "suggested_tp": 0,
                                "suggested_leverage": min(CFG.max_leverage, 5),
                                "reason": rule_result["reason"],
                                "thought_process": f"[规则引擎] {rule_result['reason']}"
                            }
                # 规则引擎信号优先级最高，不执行后续快速决策（ breakout/MA/MACD）

        # 静默拦截：持仓时若市场无明显变化且未超过静默期，则跳过AI请求
        silence_triggered = False
        if self.fast_lane._should_skip_ai_request(sym, ind_15m, ind_1h, price):
            log.debug(f"⏸️ [{sym}] 静默拦截：持仓无显著变化，跳过AI请求，使用缓存决策")
            decision = self.fast_lane._get_ai_decision(symbol=sym)
            if decision is None:
                decision = {"action": "hold", "confidence": 0.5, "reason": "静默拦截", "thought_process": ""}
            silence_triggered = True
            # 静默拦截时，直接进入后续动作判断，跳过 AI 触发

        # ── 1.2 打板检测（快速决策通道）──────────────────────────────────────
        # 若检测到强势突破且当前空仓，优先使用快速决策（绕过AI等待）
        # 修复5：仅当规则引擎未给出信号时才进行打板检测
        # 传入 ind_15m 和 market_mode 使 ADX/震荡激进过滤生效
        if not pos.side and not silence_triggered and fast_decision is None:
            # 检测向上突破
            breakout_up = self.signals._detect_breakout(raw_15m, price, direction="long", ind_15m=ind_15m, market_mode=_osc_market)
            breakout_down = self.signals._detect_breakout(raw_15m, price, direction="short", ind_15m=ind_15m, market_mode=_osc_market)
            # 优先处理有明显方向的突破（冷却：趋势市5min，震荡市10min）
            _bo_cooldown = 600 if _osc_market != "趋势" else 300
            _now_mono = time.monotonic()
            # 分模式硬过滤（突破属于趋势信号，需要 ADX 确认）
            _adx_bo = ind_15m.get("adx", 0)
            if _osc_market == "趋势":
                _adx_bo_ok = _adx_bo >= 22
                _vol_bo_min = 1.4
                _ob_bo_min = 0.25
            else:  # 震荡 / 震荡激进
                _adx_bo_ok = True  # 震荡市不要求 ADX
                _vol_bo_min = 1.2
                _ob_bo_min = 0.20
            if breakout_up.get("breakout") and breakout_up.get("confidence", 0) >= 0.6:
                _vol_bo_up = ind_15m.get("vol_surge", 0)
                _ob_bo_up  = depth.get("imbalance", 0)
                _bo_up_ok = _vol_bo_up >= _vol_bo_min and _ob_bo_up > _ob_bo_min and _adx_bo_ok
                if _bo_up_ok and _now_mono - self._last_breakout_ts["up"] >= _bo_cooldown:
                    self._last_breakout_ts["up"] = _now_mono
                    self._consecutive_hold = 0       # 突破信号唤醒，重置冷静期
                    self._ai_hold_wait = 0
                    if not self.fast_lane._is_redundant_fast_signal(sym, "BREAKOUT_UP", "open_long", breakout_up["confidence"]):
                        log.warning(f"🚀 [{sym}] 打板信号：{breakout_up.get('reason')}，置信度={breakout_up.get('confidence'):.2f}，使用快速开多")
                        fast_decision = {
                            "action": "open_long",
                            "confidence": breakout_up["confidence"],
                            "suggested_sl": breakout_up.get("sl", price - breakout_up.get("atr", 20) * 1.5),
                            "suggested_tp": 0,
                            "suggested_leverage": min(CFG.max_leverage, 8),
                            "reason": breakout_up["reason"],
                            "thought_process": f"[打板快速决策] {breakout_up['reason']}"
                        }
                elif not _bo_up_ok:
                    _why_fail = []
                    if _vol_bo_up < _vol_bo_min: _why_fail.append(f"vol={_vol_bo_up:.1f}x")
                    if _ob_bo_up <= _ob_bo_min:  _why_fail.append(f"imb={_ob_bo_up:+.3f}")
                    if not _adx_bo_ok:           _why_fail.append(f"adx={_adx_bo:.1f}")
                    log.debug(f"🔇 [{sym}] 打板多头信号被分模式过滤拦截({', '.join(_why_fail)})，降级为AI推理")
                else:
                    log.debug(f"⏳ [{sym}] 打板多头信号冷却中（距上次 {_now_mono - self._last_breakout_ts['up']:.0f}s < {_bo_cooldown}s），跳过")
            elif breakout_down.get("breakout") and breakout_down.get("confidence", 0) >= 0.6:
                _vol_bo_dn = ind_15m.get("vol_surge", 0)
                _ob_bo_dn  = depth.get("imbalance", 0)
                _bo_dn_ok = _vol_bo_dn >= _vol_bo_min and _ob_bo_dn < -_ob_bo_min and _adx_bo_ok
                if _bo_dn_ok and _now_mono - self._last_breakout_ts["down"] >= _bo_cooldown:
                    self._last_breakout_ts["down"] = _now_mono
                    self._consecutive_hold = 0       # 突破信号唤醒，重置冷静期
                    self._ai_hold_wait = 0
                    if not self.fast_lane._is_redundant_fast_signal(sym, "BREAKOUT_DOWN", "open_short", breakout_down["confidence"]):
                        log.warning(f"🚀 [{sym}] 打板信号：{breakout_down.get('reason')}，置信度={breakout_down.get('confidence'):.2f}，使用快速开空")
                        fast_decision = {
                            "action": "open_short",
                            "confidence": breakout_down["confidence"],
                            "suggested_sl": breakout_down.get("sl", price + breakout_down.get("atr", 20) * 1.5),
                            "suggested_tp": 0,
                            "suggested_leverage": min(CFG.max_leverage, 8),
                            "reason": breakout_down["reason"],
                            "thought_process": f"[打板快速决策] {breakout_down['reason']}"
                        }
                elif not _bo_dn_ok:
                    _why_fail = []
                    if _vol_bo_dn < _vol_bo_min: _why_fail.append(f"vol={_vol_bo_dn:.1f}x")
                    if _ob_bo_dn >= -_ob_bo_min: _why_fail.append(f"imb={_ob_bo_dn:+.3f}")
                    if not _adx_bo_ok:           _why_fail.append(f"adx={_adx_bo:.1f}")
                    log.debug(f"🔇 [{sym}] 打板空头信号被分模式过滤拦截({', '.join(_why_fail)})，降级为AI推理")
                else:
                    log.debug(f"⏳ [{sym}] 打板空头信号冷却中（距上次 {_now_mono - self._last_breakout_ts['down']:.0f}s < {_bo_cooldown}s），跳过")

        # ── 1.3 新增强快速决策信号（MA交叉 + MACD交叉）──────────────────────
        # MA5/MA10 黄金交叉 + 放量 → 快速做多
        # MACD柱状线从负转正 + 放量 → 快速做多
        if not pos.side and not silence_triggered and fast_decision is None:
            closes_for_ma = [float(k[4]) for k in raw_15m[-10:]]
            if len(closes_for_ma) >= 10:
                # pandas已在文件顶部全局导入，无需重复导入
                ma5_series = pd.Series(closes_for_ma).ewm(span=5, adjust=False)
                ma10_series = pd.Series(closes_for_ma).ewm(span=10, adjust=False)
                ma5_curr = float(ma5_series.mean().iloc[-1])
                ma10_curr = float(ma10_series.mean().iloc[-1])
                ma5_prev = float(ma5_series.mean().iloc[-2])
                ma10_prev = float(ma10_series.mean().iloc[-2])

                # MA黄金交叉：MA5从下方穿越MA10
                ma_golden_cross = (ma5_curr > ma10_curr) and (ma5_prev <= ma10_prev)
                # MA死叉：MA5从上方穿越MA10
                ma_death_cross = (ma5_curr < ma10_curr) and (ma5_prev >= ma10_prev)

                # 获取已计算的指标
                macd_cross_up = ind_15m.get("macd_cross_up", False)
                macd_cross_dn = ind_15m.get("macd_cross_down", False)
                vol_surge = ind_15m.get("vol_surge", 1.0)
                atr = ind_15m.get("atr", 20)

                # 分模式硬过滤（MA/MACD属于趋势信号，趋势市需要 ADX 确认）
                _adx_ma = ind_15m.get("adx", 0)
                _ma_adx_ok = _osc_market == "趋势" and _adx_ma < 22  # 趋势市ADX不足时拦截
                # ── OB盘口方向对齐（4个子信号共用）──
                _ob_ma = depth.get("imbalance", 0)
                # MA黄金交叉 + 强放量(>=1.8) → 快速做多
                # 门槛高于MACD：MA5/MA10在15m上噪音大，需更强放量确认
                if ma_golden_cross and vol_surge >= 1.8:
                    if _ma_adx_ok:
                        log.debug(f"🔇 [{sym}] MA黄金交叉被ADX过滤拦截(adx={_adx_ma:.1f}<22, market={_osc_market})，降级为AI推理")
                    elif _ob_ma <= 0.20:
                        log.debug(f"🔇 [{sym}] MA黄金交叉被OB盘口过滤拦截(imb={_ob_ma:+.3f}≤0.20)，降级为AI推理")
                    elif not self.fast_lane._is_redundant_fast_signal(sym, "MA_GOLDEN", "open_long", 0.65):
                        log.warning(f"🚀 [{sym}] MA黄金交叉+放量信号，使用快速开多")
                        fast_decision = {
                            "action": "open_long",
                            "confidence": 0.65,
                            "suggested_sl": ma10_curr - atr * 1.5,
                            "suggested_tp": 0,
                            "suggested_leverage": min(CFG.max_leverage, 8),
                            "reason": f"MA5/MA10黄金交叉+放量{vol_surge:.1f}倍",
                            "thought_process": f"[MA黄金交叉快速决策] MA5={ma5_curr:.2f} MA10={ma10_curr:.2f}"
                        }
                # MACD柱状线从负转正 + 放量(>=1.3) → 快速做多（MACD已平滑，门槛低于MA交叉）
                elif macd_cross_up and vol_surge >= 1.3:
                    if _ma_adx_ok:
                        log.debug(f"🔇 [{sym}] MACD金叉被ADX过滤拦截(adx={_adx_ma:.1f}<22, market={_osc_market})，降级为AI推理")
                    elif _ob_ma <= 0.20:
                        log.debug(f"🔇 [{sym}] MACD金叉被OB盘口过滤拦截(imb={_ob_ma:+.3f}≤0.20)，降级为AI推理")
                    elif not self.fast_lane._is_redundant_fast_signal(sym, "MACD_GOLDEN", "open_long", 0.68):
                        log.warning(f"🚀 [{sym}] MACD金叉+放量信号，使用快速开多")
                        fast_decision = {
                            "action": "open_long",
                            "confidence": 0.68,
                            "suggested_sl": price - atr * 1.5,
                            "suggested_tp": 0,
                            "suggested_leverage": min(CFG.max_leverage, 8),
                            "reason": f"MACD金叉+放量{vol_surge:.1f}倍",
                            "thought_process": "[MACD金叉快速决策]"
                        }
                # MA死叉 + 强放量(>=1.8) → 快速做空
                elif ma_death_cross and vol_surge >= 1.8:
                    if _ma_adx_ok:
                        log.debug(f"🔇 [{sym}] MA死叉被ADX过滤拦截(adx={_adx_ma:.1f}<22, market={_osc_market})，降级为AI推理")
                    elif _ob_ma >= -0.20:
                        log.debug(f"🔇 [{sym}] MA死叉被OB盘口过滤拦截(imb={_ob_ma:+.3f}≥-0.20)，降级为AI推理")
                    elif not self.fast_lane._is_redundant_fast_signal(sym, "MA_DEATH", "open_short", 0.65):
                        log.warning(f"🚀 [{sym}] MA死叉+放量信号，使用快速开空")
                        fast_decision = {
                            "action": "open_short",
                            "confidence": 0.65,
                            "suggested_sl": ma10_curr + atr * 1.5,
                            "suggested_tp": 0,
                            "suggested_leverage": min(CFG.max_leverage, 8),
                            "reason": f"MA5/MA10死叉+放量{vol_surge:.1f}倍",
                            "thought_process": f"[MA死叉快速决策] MA5={ma5_curr:.2f} MA10={ma10_curr:.2f}"
                        }
                # MACD柱状线从正转负 + 放量(>=1.3) → 快速做空
                elif macd_cross_dn and vol_surge >= 1.3:
                    if _ma_adx_ok:
                        log.debug(f"🔇 [{sym}] MACD死叉被ADX过滤拦截(adx={_adx_ma:.1f}<22, market={_osc_market})，降级为AI推理")
                    elif _ob_ma >= -0.20:
                        log.debug(f"🔇 [{sym}] MACD死叉被OB盘口过滤拦截(imb={_ob_ma:+.3f}≥-0.20)，降级为AI推理")
                    elif not self.fast_lane._is_redundant_fast_signal(sym, "MACD_DEATH", "open_short", 0.68):
                        log.warning(f"🚀 [{sym}] MACD死叉+放量信号，使用快速开空")
                        fast_decision = {
                            "action": "open_short",
                            "confidence": 0.68,
                            "suggested_sl": price + atr * 1.5,
                            "suggested_tp": 0,
                            "suggested_leverage": min(CFG.max_leverage, 8),
                            "reason": f"MACD死叉+放量{vol_surge:.1f}倍",
                            "thought_process": "[MACD死叉快速决策]"
                        }

        # ── 1.4a 震荡市专用评分函数（快速决策通道）──────────────────────────
        # 注：_osc_market 已在 1.2 之前统一计算，此处直接复用
        # 当市场为震荡市且空仓时，基于RSI+关键位+成交量直接开仓，无需等待AI
        if not pos.side and _osc_market == "震荡" and not silence_triggered and fast_decision is None:
            rsi = ind_15m.get("rsi", 50)
            price_now = price
            supports = key_levels.get("supports", []) if key_levels else []
            resistances = key_levels.get("resistances", []) if key_levels else []
            atr = ind_15m.get("atr", 20)
            vol_surge = ind_15m.get("vol_surge", 1.0)

            # 超卖 + 支撑附近 → 快速做多
            if (rsi <= 35 and price_now > 0
                    and any(abs(price_now - s["price"]) / price_now < CFG.osc_level_proximity for s in supports)):
                _ob_osc = depth.get("imbalance", 0)
                if _ob_osc <= 0.20:
                    log.debug(f"🔇 [{sym}] 震荡市超卖信号被OB盘口过滤拦截(imb={_ob_osc:+.3f}≤0.20)，降级为AI推理")
                elif not self.fast_lane._is_redundant_fast_signal(sym, "OSC_OVERSOLD", "open_long", 0.60, cooldown=600):
                    log.info(f"📊 [{sym}] 震荡市超卖+支撑，快速做多 RSI={rsi:.1f}")
                    fast_decision = {
                        "action": "open_long",
                        "confidence": 0.60,
                        "suggested_sl": price_now - atr * CFG.osc_sl_atr_mult,
                        "suggested_tp": 0,
                        "suggested_leverage": min(CFG.max_leverage, 6),
                        "reason": f"震荡市超卖触及支撑 RSI={rsi:.1f}",
                        "thought_process": "[震荡市快速决策] 超卖+支撑"
                    }
            # 超买 + 阻力附近 → 快速做空
            elif (rsi >= 67 and price_now > 0
                    and any(abs(price_now - r["price"]) / price_now < CFG.osc_level_proximity for r in resistances)):
                _ob_osc2 = depth.get("imbalance", 0)
                if _ob_osc2 >= -0.20:
                    log.debug(f"🔇 [{sym}] 震荡市超买信号被OB盘口过滤拦截(imb={_ob_osc2:+.3f}≥-0.20)，降级为AI推理")
                elif not self.fast_lane._is_redundant_fast_signal(sym, "OSC_OVERBOUGHT", "open_short", 0.60, cooldown=600):
                    log.info(f"📊 [{sym}] 震荡市超买+阻力，快速做空 RSI={rsi:.1f}")
                    fast_decision = {
                        "action": "open_short",
                        "confidence": 0.60,
                        "suggested_sl": price_now + atr * CFG.osc_sl_atr_mult,
                        "suggested_tp": 0,
                        "suggested_leverage": min(CFG.max_leverage, 6),
                        "reason": f"震荡市超买触及阻力 RSI={rsi:.1f}",
                        "thought_process": "[震荡市快速决策] 超买+阻力"
                    }
            # 缩量回踩支撑 + RSI回升 → 快速做多（量价背离）
            elif rsi <= 45 and rsi > 35 and vol_surge < 0.7 and price_now > 0:
                # 缩量且RSI从低位回升，视为有效反弹
                if any(abs(price_now - s["price"]) / price_now < CFG.osc_level_proximity * 2 for s in supports):
                    _ob_osc3 = depth.get("imbalance", 0)
                    if _ob_osc3 <= 0.15:
                        log.debug(f"🔇 [{sym}] 震荡市缩量回踩被OB盘口过滤拦截(imb={_ob_osc3:+.3f}≤0.15)，降级为AI推理")
                    elif not self.fast_lane._is_redundant_fast_signal(sym, "OSC_PULLBACK", "open_long", 0.58, cooldown=600):
                        log.info(f"📊 [{sym}] 震荡市缩量回踩支撑+RSI修复，快速做多 RSI={rsi:.1f} vol_surge={vol_surge:.1f}")
                        fast_decision = {
                            "action": "open_long",
                            "confidence": 0.58,
                            "suggested_sl": price_now - atr * CFG.osc_sl_atr_mult,
                            "suggested_tp": 0,
                            "suggested_leverage": min(CFG.max_leverage, 6),
                            "reason": f"震荡市缩量回踩支撑 RSI={rsi:.1f} vol={vol_surge:.1f}x",
                            "thought_process": "[震荡市快速决策] 缩量回踩支撑+RSI修复"
                        }

        # ── 1.4b 震荡激进专用评分函数（快速决策通道）────────────────────────
        # BB极值+放量+RSI极端时直接开仓，无需等待AI（比普通震荡更高的量能要求）
        if not pos.side and _osc_market == "震荡激进" and not silence_triggered and fast_decision is None:
            rsi_agg     = ind_15m.get("rsi", 50)
            price_now   = price
            atr_agg     = ind_15m.get("atr", 20)
            vol_surge_agg = ind_15m.get("vol_surge", 1.0)
            bb_pct_agg  = ind_15m.get("bb_pct", 0.5)

            # BB下沿+放量+超卖 → 快速做多（区间底部强支撑反弹）
            # 第二确认：imbalance > 0.6（买盘深度明显占优）避免在均衡市被假信号骗
            imb_agg = depth.get("imbalance", 0)
            if bb_pct_agg <= 0.10 and vol_surge_agg >= 1.5 and rsi_agg <= 35 and abs(imb_agg) > 0.6:
                # 方向对齐：做多看买盘失衡(imb > 0)
                if imb_agg <= 0:
                    log.debug(f"🔇 [{sym}] 震荡激进BB做多信号被OB方向过滤拦截(imb={imb_agg:+.3f}≤0，买方未主导)，降级为AI推理")
                elif not self.fast_lane._is_redundant_fast_signal(sym, "OSC_BB_LOW", "open_long", 0.65, cooldown=600):
                    log.info(f"📊 [{sym}] 震荡激进BB下沿+放量+超卖+失衡，快速做多 bb_pct={bb_pct_agg:.2f} vol={vol_surge_agg:.1f} RSI={rsi_agg:.1f} imb={imb_agg:.2f}")
                    fast_decision = {
                        "action": "open_long",
                        "confidence": 0.65,
                        "suggested_sl": price_now - atr_agg * 1.2,
                        "suggested_tp": 0,
                        "suggested_leverage": min(CFG.max_leverage, 8),
                        "reason": f"震荡激进BB下沿+放量+超卖+失衡 imb={imb_agg:.2f} vol={vol_surge_agg:.1f}x RSI={rsi_agg:.1f}",
                        "thought_process": "[震荡激进快速决策] BB下沿+放量+超卖+盘口失衡确认"
                    }
            # BB上沿+放量+超买 → 快速做空（区间顶部强压力回落）
            elif bb_pct_agg >= 0.90 and vol_surge_agg >= 1.5 and rsi_agg >= 65 and abs(imb_agg) > 0.6:
                # 方向对齐：做空看卖盘失衡(imb < 0)
                if imb_agg >= 0:
                    log.debug(f"🔇 [{sym}] 震荡激进BB做空信号被OB方向过滤拦截(imb={imb_agg:+.3f}≥0，卖方未主导)，降级为AI推理")
                elif not self.fast_lane._is_redundant_fast_signal(sym, "OSC_BB_HIGH", "open_short", 0.65, cooldown=600):
                    log.info(f"📊 [{sym}] 震荡激进BB上沿+放量+超买+失衡，快速做空 bb_pct={bb_pct_agg:.2f} vol={vol_surge_agg:.1f} RSI={rsi_agg:.1f} imb={imb_agg:.2f}")
                    fast_decision = {
                        "action": "open_short",
                        "confidence": 0.65,
                        "suggested_sl": price_now + atr_agg * 1.2,
                        "suggested_tp": 0,
                        "suggested_leverage": min(CFG.max_leverage, 8),
                        "reason": f"震荡激进BB上沿+放量+超买+失衡 imb={imb_agg:.2f} vol={vol_surge_agg:.1f}x RSI={rsi_agg:.1f}",
                        "thought_process": "[震荡激进快速决策] BB上沿+放量+超买+盘口失衡确认"
                    }

        # ── Fast Decision 上下文（signal_hint 非绑定参考）────────────────────
        _fast_context = ""
        if fast_decision is not None:
            _fd_src = fast_decision.get("thought_process", "")
            _fd_act = fast_decision["action"]
            _fd_dir = "short" if _fd_act == "open_short" else "long"
            _fd_vol = ind_15m.get("vol_surge", 0)
            _fd_ob  = depth.get("imbalance", 0)
            _fast_context = (
                f"\n[规则引擎参考信号(非绑定)]\n"
                f"方向: {_fd_dir} | 置信度: {fast_decision.get('confidence', 0):.2f} | "
                f"来源: {_fd_src[:50]}\n"
                f"成交量: {_fd_vol:.1f}x | 盘口失衡: {_fd_ob:+.3f} | "
                f"原因: {fast_decision.get('reason', '')[:60]}\n"
            )

        # ── EAT-FLOW 吃单流量 + VSpike 突增（注入 fast_context）────────────
        _vs = self.vspike.get_status()

        # EAT-FLOW：仅在有意义的结构信号时注入（避免平静期 token 浪费）
        # 触发条件：VSpike 突增 / 震荡激进市 / absorption 冰山单信号
        _ef_trigger = (
            _vs.get("is_spike")
            or _osc_market == "震荡激进"
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
                f"累计 {_cvd:+.0f}张 / {_cvd_total:.0f}张(占比{_cvd_pct:.0f}%)"
                f" | {'主动买入主导' if _cvd > 0 else '主动卖出主导' if _cvd < 0 else '多空均衡'}\n"
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
        if _osc_market == "震荡激进" or _has_iceberg:
            _full_ob = True
        elif _osc_market == "趋势":
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

        # ── 规则引擎触发AI路径防抖：防止 trigger_ai=True 时无限刷屏 ───────────
        _rule_full_triggered = (
            pos.side is None   # 空仓
            and rule_result is not None
            and rule_result.get("trigger_ai") is True
            and rule_result.get("signal_type") not in ("long", "short")  # 非快速决策信号
        )
        _rule_debounce_hit = False
        if _rule_full_triggered:
            _vs4r = self.vspike.get_status()
            _spike_bypass_r = _vs4r.get("spike_just_triggered", False)
            _rule_conf_r = rule_result.get("confidence", 0)
            _rule_reason_r = rule_result.get("reason", "")
            _urgent_kw = ("控险", "止损", "风险", "假突破", "逆势", "危险", "破位", "逃生")
            _is_urgent_rule = _rule_conf_r >= 0.75 or any(k in _rule_reason_r for k in _urgent_kw)
            if self.fast_lane._is_redundant_fast_signal(sym, "RULE_FULL", "trigger_full_ai", _rule_conf_r,
                                               cooldown=300, force_bypass=(_spike_bypass_r or _is_urgent_rule)):
                _rule_debounce_hit = True
                log.debug(f"🛡️ [{sym}] 规则引擎AI触发防抖命中（conf={_rule_conf_r:.2f}），跳过本次完整推理")
            elif _is_urgent_rule:
                log.warning(f"⚡ [{sym}] 规则引擎紧急信号绕过防抖: {_rule_reason_r}")

        if not silence_triggered and not _rule_debounce_hit:
            # AI 异步触发（每品种独立缓存）；即使有 fast_decision 也传上下文让 AI 最终裁决
            self.fast_lane._trigger_ai_async_sym(
                sym, ind_15m, ind_1h, ind_4h, ind_3m, news_data, fg_index, funding,
                depth, pos_info, key_levels=key_levels,
                funding_history=self.funding_history,
                macro_context=macro_context, rag_warning=rag_warning,
                market_sentiment=market_sentiment,
                sentiment_alert=_sentiment_alert,
                fast_context=_fast_context,
            )

            decision = self.fast_lane._get_ai_decision(symbol=sym)
            if decision is None:
                log.debug(f"⏳ [{sym}] AI 决策未就绪，降级 hold")
                decision = {"action": "hold", "confidence": 0.5, "reason": "AI 未就绪"}
        elif _rule_debounce_hit:
            # 防抖命中时，直接用缓存决策（不调用 AI）
            decision = self.fast_lane._get_ai_decision(symbol=sym)
            if decision is None:
                decision = {"action": "hold", "confidence": 0.5, "reason": "AI 未就绪"}

        # ── 冻结 AI 决策时刻的 VSpike 状态，供后续 ConvictionScore 计算复用 ──
        # 避免在 _do_open/_calc_size_and_margin 中重新查询导致时间差引入的 VSpike=0 问题
        _vs_frozen = self.vspike.get_status()
        _vs_mult_fr = _vs_frozen.get("mult", 0.0) if (
            _vs_frozen.get("is_spike") or _vs_frozen.get("spike_recent")
        ) else 0.0
        _vs_dir_fr = _vs_frozen.get("direction", "均衡")
        _is_long_fr = decision.get("action", "") == "open_long"
        _vs_dir_ok_fr = (
            (_is_long_fr and _vs_dir_fr == "买方主导") or
            (not _is_long_fr and _vs_dir_fr == "卖方主导")
        )
        _vs_score_mult_fr = _vs_mult_fr if _vs_dir_ok_fr else 0.0
        # VSpike ≥6.0x bonus：仅方向对齐时才给 bonus，避免反向 Spike 误导开仓
        if _vs_mult_fr >= 6.0 and _vs_dir_ok_fr:
            _vs_score_mult_fr += 15.0
        decision["_vspike_status"] = _vs_frozen
        decision["_vs_score_mult_frozen"] = _vs_score_mult_fr
        decision["_vs_score_mult_frozen_ts"] = time.monotonic()  # 过期时间戳

        # ── AI 对快速决策的最终裁决（适用于所有AI路径，含防抖命中路径）──────────
        # 修复：原代码裁决块在 if not silence 内，_rule_debounce_hit 时 fast_decision 被静默丢弃
        if fast_decision is not None and not silence_triggered:
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
            if _path_a or _path_b:
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
                # ── ConvictionScore 预检：低分信号提前拦截，避免空跑开仓流程 ────
                if _fd_score["score"] < CFG.conviction_open_min:
                    _fd_action_rej = fast_decision.get("action", "")
                    _rej_key = f"{sym}_{_fd_action_rej}"
                    _prev_rej = self._fast_rejection_cache.get(_rej_key)
                    _rej_count = (_prev_rej[1] + 1) if _prev_rej else 1
                    self._fast_rejection_cache[_rej_key] = (_fd_action_rej, _rej_count, time.monotonic())
                    _fd_comps = _fd_score.get("components", {})
                    log.info(
                        f"🚫 [{sym}] 快速决策预检拦截: ConvictionScore={_fd_score['score']:.1f}"
                        f" < {CFG.conviction_open_min} "
                        f"(ai_raw={_fd_comps.get('ai_raw',0):.0f} spike={_fd_comps.get('spike',0):.0f} "
                        f"ob={_fd_comps.get('ob',0):.1f}) "
                        f"第{_rej_count}次拒绝，信号降级"
                    )
                    fast_decision = None
                    return
                _debounce_tag = "[防抖路径]" if _rule_debounce_hit else ""
                _fd_reason = fast_decision.get("reason", "")
                log.info(f"✅ [{sym}] 快速决策执行{_debounce_tag}（{_fd_reason}）→ {fast_decision['action']}")
                decision = fast_decision
                self.fast_lane._clear_ai_cache(symbol=sym)
                # 成功执行快速决策时，清除该信号的拒绝记录
                _rej_key_ok = f"{sym}_{fast_decision.get('action', '')}"
                self._fast_rejection_cache.pop(_rej_key_ok, None)
            else:
                log.info(f"⏳ [{sym}] AI 否决快速决策（use_fast={_use_fast} ai_conf={_ai_conf_for_fast:.2f} fast_conf={_fast_conf:.2f}），转完整推理")
                fast_decision = None  # 丢弃快速决策，由 AI 推理接管

        # ── AI hold 时主动打破静默（force_wakeup）───────────────────────────
        # 只有 action="hold" 时 AI 才会输出 force_wakeup，此时本轮立即打破静默重评
        _fw = bool(decision.get("force_wakeup", False)) if decision.get("action") == "hold" else False
        if _fw:
            log.info(f"⚡ [{sym}] AI 输出 hold 但 force_wakeup=true → 强制打破静默，重新决策")
            self._last_force_wakeup[sym] = True
            self.fast_lane._clear_ai_cache(symbol=sym)
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
        _disp = {k: v for k, v in decision.items() if k != "thought_process"}
        _sig  = f"{decision.get('action')}|{decision.get('confidence', 0):.2f}|{(decision.get('reason') or '')[:80]}"
        _is_repeat = (_sig == self._last_decision_sig.get(sym, ""))
        self._last_decision_sig[sym] = _sig
        if not _is_repeat:
            _decision_src = "[静默缓存]" if silence_triggered else ("[快速决策]" if fast_decision is not None else "[新决策]")
            log.info(f"🤖 [{sym}] AI决策{_decision_src}: {_disp}")
            if decision.get("thought_process"):
                log.debug(f"🤖 [{sym}] 思考过程: {decision['thought_process']}")
        else:
            _decision_src = "[静默缓存]" if silence_triggered else "[重复]"
            log.debug(f"🤖 [{sym}] AI决策{_decision_src}: action={decision.get('action')} conf={decision.get('confidence', 0):.2f} — {(decision.get('reason') or '')[:60]}")

        # ── 策略委员会校验（快速决策打板信号跳过委员会）─────────────────────
        # fast_decision 已包含独立技术确认，无需再被委员会审查
        committee_opposing = 0
        if (decision.get("action") in ("open_long", "open_short")
                and not pos.side and fast_decision is None):
            decision = self._strategy_committee(
                sym, decision, ind_15m, ind_1h, ind_3m, price, depth=depth
            )

        # ── 贝叶斯后验置信度（结合 AI confidence + 盘口失衡度）──────────────────
        if decision.get("action") in ("open_long", "open_short") and not silence_triggered:
            ai_conf = decision.get("confidence", 0.5)
            imbal = (depth.get("imbalance", 0) if isinstance(depth, dict) else 0)
            # 先验 = 近25笔已平仓胜率（每小时由 update_dynamic_params 动态维护）
            # 下限 0.40：防止历史极端低胜率（如0.23）压垮 Kelly 算出负仓位导致系统停摆
            prior = max(0.40, gs_get("last_24h_win_rate", 0.5))
            # 似然 = AI置信度 × (1 + 失衡度)，截断到 [0.05, 0.95]
            # 上限 0.95 而非 1.0：likelihood=1.0 时 Bayesian 分母退化，posterior 跳至 1.0
            likelihood = max(0.05, min(0.95, ai_conf * (1 + imbal)))
            posterior = bayesian_posterior(prior=prior, likelihood=likelihood)
            decision["posterior_confidence"] = posterior
            log.info(f"🧮 [{sym}] 贝叶斯后验: prior={prior:.3f} AI_conf={ai_conf:.3f} imbalance={imbal:.3f} → posterior={posterior:.3f}")

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
            "market_mode":  _osc_market,
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
                log.info(
                    f"ℹ️ [{sym}] 已有{pos.side}仓位，AI 再次建议{action}，"
                    f"视为同向信号，降级为 hold（金字塔加仓由独立检查处理）"
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
                _eff_hold = 600 if _osc_market == "震荡激进" else CFG.min_hold_seconds
                if hold_secs < _eff_hold:
                    conf = decision.get("confidence", 0)
                    reason = decision.get("reason", "")
                    _urgent_keywords = ("控险", "止损", "风险", "假突破", "危险", "破位", "逃生")
                    _kw_hit   = any(k in reason for k in _urgent_keywords)
                    _hi_conf  = conf >= 0.75
                    _vs_mid   = conf >= 0.68 and _vspike_opp and _vs_mult >= CFG.vspike_escape_level1
                    _vs_str   = conf >= 0.65 and _vspike_opp and _vs_mult >= CFG.vspike_escape_level2
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
            _osc_mode = _osc_market in ("震荡", "震荡激进")
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
            # 核心理念：开仓需要 DeepSeek/Qwen 多模型确认，平仓同样由 AI 主导决策
            # Rule 1-7 已降级为信号收集器（_build_exit_context），不再阻断 AI 平仓请求
            # 唯一硬阻断：核按钮 VSpike（Rule 0.5，≥8x 极端反向量能）→ 立即执行
            conf_close   = decision.get("confidence", 0.0)
            close_reason = decision.get("reason", "AI平仓")
            _exit_ctx    = _build_exit_context(now, decision)

            # ── 核按钮 VSpike：极端量能无条件逃生（最高优先级，绕过所有检查）────
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
                return

            # ── 冷静期拦截（仅在非紧急情况下阻断）─────────────────────────────
            if not _exit_ctx["calm_period_ok"]:
                log.warning(
                    f"🛡️ [{sym}] 【冷静期拦截】AI平仓请求暂停 | {_exit_ctx['msg']}\n"
                    f"   AI理由: {close_reason} | conf={conf_close:.2f}"
                )
                log_event("calm_period_blocked", {
                    "sym": sym, "ai_reason": close_reason,
                    "ai_conf": conf_close, "block_msg": _exit_ctx["msg"]
                })
                _spike_peak_ts_now = self.vspike._spike_peak_ts
                _blocked_until = time.monotonic() + CFG.blocked_close_cooldown
                self._blocked_close_ctx[sym] = {
                    "spike_ts":     _spike_peak_ts_now,
                    "blocked_until": _blocked_until,
                    "block_price":  price,
                    "atr":          ind_15m.get("atr", 0),
                }
                log.info(f"⏳ [{sym}] 冷静期拦截冷却启动 {CFG.blocked_close_cooldown}s")
                return

            # ── AI 置信度达标 → 直接执行（不再需要 Rule 1-7 放行）─────────────
            # 千问仲裁特殊处理：仲裁源 = 千问已做最终裁决，不再二次置信度门控
            _is_qwen_arb = decision.get("source") == "qwen_exit_arbitration"
            _arb_close = conf_close >= CFG.close_confidence_threshold if not _is_qwen_arb else True
            if _arb_close:
                self._ai_close_pending_until = time.monotonic() + 5.0
                _arb_tag = "[千问仲裁]" if _is_qwen_arb else ""
                _sig_tag = f"[信号:{_exit_ctx['hit_count']}条]" if _exit_ctx["hit_count"] > 0 else "[AI独立判断]"
                log.info(
                    f"🔻 [{sym}] AI平仓指令 {conf_close:.2f} {'≥' if not _is_qwen_arb else '=='} {CFG.close_confidence_threshold if not _is_qwen_arb else '仲裁直通'} "
                    f"{_arb_tag}{_sig_tag} | {close_reason}"
                )
                log_event("ai_close_triggered", {
                    "sym": sym, "confidence": conf_close,
                    "reason": close_reason, "trigger": "ai_high_conf",
                    "exit_signals": _exit_ctx["msg"],
                    "hit_count": _exit_ctx["hit_count"],
                })
                self.position_exec._close(
                    f"AI平仓(conf={conf_close:.2f}) {_sig_tag} {_exit_ctx['msg']} | {close_reason}",
                    decision_id, symbol=sym
                )
            else:
                # ── AI 置信度不足，延迟执行，由追踪止损控制 ────────────────────
                log.info(
                    f"⚠️ [{sym}] AI平仓置信度 {conf_close:.2f} < {CFG.close_confidence_threshold}，"
                    f"延迟执行，由追踪止损控制离场 | 原因: {close_reason}"
                    f" | 离场信号: {_exit_ctx['msg']}"
                )
                log_event("ai_close_deferred", {
                    "sym": sym, "confidence": conf_close,
                    "reason": close_reason, "trigger": "deferred_to_trailing",
                    "exit_signals": _exit_ctx["msg"]
                })
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
            is_long   = (action == "open_long")

            # ── 优先使用贝叶斯后验置信度（融合历史胜率+盘口失衡），无后验时用原始置信度 ─
            _raw_conf = decision.get("confidence", 0)
            _post_conf = decision.get("posterior_confidence")
            conf = _post_conf if _post_conf is not None else _raw_conf

            # ── 置信度动态阈值（trend_score 连续化，取代 4H EMA 二元判断）─────────
            _trend_score, _trend_dir = get_trend_alignment_score(ind_15m, ind_1h, ind_4h)
            _sym_market_mode = _osc_market
            adaptive_thresh = CFG.trend_base_conf_thresh * (1.2 - 0.4 * _trend_score)
            conf_threshold = max(CFG.trend_conf_clamp_min, min(CFG.trend_conf_clamp_max, adaptive_thresh))

            if conf < conf_threshold:
                _src = f"(贝叶斯后验={_post_conf:.3f})" if _post_conf else ""
                log.info(f"⚖️ [{sym}] 置信度 {conf:.2f}{_src} < {conf_threshold:.2f}（趋势={_trend_dir} 强度={_trend_score:.2f}），跳过")
                self.fast_lane._clear_ai_cache(symbol=sym)  # 清缓存，防止同一条注定被拒的决策重复撞墙
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

            # ── 止损后同方向冷却（条件A或条件B可旁路）──────────────────────────
            if action in ("open_long", "open_short") and gs_get("consecutive_losses", 0) >= 1:
                _last_stop_dir = gs_get("last_stop_direction")
                _last_stop_time = gs_get("last_stop_time")
                _last_stop_price = gs_get("last_stop_price", 0.0)
                if _last_stop_time and _last_stop_dir:
                    _now = datetime.now(UTC)
                    _last_stop_dt = _parse_dt(_last_stop_time)
                    _minutes_ago = (_now - _last_stop_dt).total_seconds() / 60 if _last_stop_dt else 999
                    _action_dir = "long" if action == "open_long" else "short"
                    if _action_dir == _last_stop_dir and _minutes_ago < CFG.min_cooldown_after_loss:
                        _conf_override = decision.get("confidence", 0)
                        _price_reclaimed_flag = price_reclaimed(price, _last_stop_price, _last_stop_dir)
                        # 条件A：顺势回归（需全部满足）
                        _condA = _price_reclaimed_flag and _conf_override >= 0.85
                        # 条件B：极端衰竭 / 插针反转（需全部满足）
                        # 只在 spike 刚触发时检查（spike_just_triggered），避免误判
                        _vs = self.vspike.get_status()
                        _rsi = ind_15m.get("rsi", 50) if ind_15m else 50
                        _spike_ok = _vs.get("spike_just_triggered", False) and _vs.get("mult", 0) >= 2.8
                        _eat_dir_supports = (
                            (_action_dir == "long"  and "买方主导" in _vs.get("direction", "")) or
                            (_action_dir == "short" and "卖方主导" in _vs.get("direction", ""))
                        )
                        _rsi_extreme = (_action_dir == "long" and _rsi < 30) or (_action_dir == "short" and _rsi > 70)
                        _condB = _spike_ok and _eat_dir_supports and _rsi_extreme and _conf_override >= 0.68
                        if _condA:
                            log.info(
                                f"⚡ [{sym}] 止损后冷却期内，条件A(顺势回归)满足："
                                f"AI conf={_conf_override:.2f}≥0.85，价格已重新站稳，批准开仓"
                            )
                        elif _condB:
                            log.warning(
                                f"🔥 [{sym}] 止损后冷却期内，条件B(极端衰竭)满足："
                                f"成交量突增{_vs.get('mult', 0):.1f}x + EAT-FLOW {_vs.get('direction', '')} + "
                                f"RSI={_rsi:.1f}(极端区) + AI conf={_conf_override:.2f}≥0.68，批准左侧摸底"
                            )
                        else:
                            log.info(
                                f"🛑 [{sym}] 止损后冷却期内，禁止同方向开仓 | "
                                f"方向:{_last_stop_dir} 距止损:{_minutes_ago:.0f}分钟 "
                                f"条件A(顺势): {'满足' if _condA else '不满足'} "
                                f"条件B(极端衰竭): {'满足' if _condB else '不满足'} | "
                                f"RSI={_rsi:.1f} VSpike={_vs.get('mult', 0):.1f}x {_vs.get('direction', '')}"
                            )
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

            # ── AI 指导的开仓等待 + 价格稳定性检查 ─────────────────────────
            _wait_sec = int(decision.get("wait_seconds", 0))
            _wait_sec = max(0, min(_wait_sec, 5))   # 硬限 0~5 秒，防 AI 给出过长等待
            if _wait_sec > 0:
                log.info(f"⏱️ [{sym}] AI 建议等待 {_wait_sec}s 后开仓（价格漂移阈值={CFG.open_wait_price_drift_pct*100:.1f}%）")
                time.sleep(_wait_sec)
                fresh_price = self._get_price(sym)
                if fresh_price > 0 and price > 0:
                    drift = abs(fresh_price - price) / price
                    if drift > CFG.open_wait_price_drift_pct:
                        log.info(
                            f"🚫 [{sym}] 价格稳定性检查失败：等待 {_wait_sec}s 后价格从 {price:.4f} "
                            f"漂移至 {fresh_price:.4f}（偏离 {drift*100:.3f}% > "
                            f"{CFG.open_wait_price_drift_pct*100:.1f}%），自动取消开仓"
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
                _is_breakout_ai = any(kw in _ai_reason.lower() for kw in _breakout_kw)
                if _is_breakout_ai:
                    log.warning(
                        f"⚠️ [逻辑熔断] Regime({_reg_open:.2f})<0.3，AI 疑似追突破"
                        f"（reason={_ai_reason[:60]}），已降级为 hold"
                    )
                    log_event("regime_circuit_break", {
                        "sym": sym, "regime": _reg_open,
                        "action": action, "reason": _ai_reason[:80]
                    })
                    return

            self.position_exec._do_open(decision, price, balance, ind_15m["atr"], funding, decision_id,
                          risk_mult=risk_mult, symbol=sym,
                          market_mode=_sym_market_mode, ind_15m=ind_15m,
                          pyramid_plan=_pyramid_plan,
                          committee_opposing=committee_opposing,
                          depth=depth,
                          trend_score=_trend_score)

        # ── ConvictionScore 拒绝处理：强制静默等待，防止非 hold 动作无限重试 ───
        # 当 _do_open 因 ConvictionScore < 阈值而 skip 时，该信息通过 _cv_rejected_decision 传递
        _cv_rej = getattr(self, '_cv_rejected_decision', None)
        if _cv_rej:
            _cv_rej_action, _cv_rej_score, _cv_rej_ts = _cv_rej
            self._consecutive_hold += 1
            # 使用 AI 建议的 wait_seconds 或强制最少 45s 静默（防止 13s 超频重试）
            _ai_wait = int(decision.get("wait_seconds", -1))
            self._ai_hold_wait = max(45, min(300, _ai_wait)) if _ai_wait >= 0 else 60
            # 清除 AI 决策缓存，避免同一条被拒绝的决策被反复使用
            self.fast_lane._clear_ai_cache(symbol=sym)
            log.info(
                f"🛡️ [{sym}] ConvictionScore 拒绝 → 强制静默 "
                f"(action={_cv_rej_action} score={_cv_rej_score:.1f} "
                f"wait={self._ai_hold_wait}s)"
            )
            self._cv_rejected_decision = ()  # 处理后清除
        # ── 连续 hold 计数（用于自适应静默间隔）────────────────────────────
        elif action in ("hold", "skip", "adjust_sl_tp"):
            self._consecutive_hold += 1
            _ai_wait = int(decision.get("wait_seconds", -1))  # -1 = AI 未给出
            self._ai_hold_wait = max(0, min(300, _ai_wait)) if _ai_wait >= 0 else -1
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
    log.info("✅ ETH Trader 已启动（DeepSeek 主决策 + Qwen 仲裁）")
    bot.run()


if __name__ == "__main__":
    main()
