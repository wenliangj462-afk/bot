"""
Risk Guard Module - 风控循环、追踪止损、资金费率检查
从 ETHTrader 拆分而出
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

# ── 共享基础设施（common.py）──────────────────────────────────────────────
from common import (
    CFG, log, _webhook, gs_get, gs_set, gs_increment, gs_update,
    UTC, log_event, get_pending_order_by_id, delete_pending_order,
    get_all_pending_orders,
)
# ── 数据模型（core.py）──────────────────────────────────────────────────
from core import GLOBAL_STATE, PositionIntent, PositionIntentType, save_state_to_disk
# ── market_data ───────────────────────────────────────────────────────
from market import fetch_market_sentiment_data

# ── 辅助函数 ─────────────────────────────────────────────────────────────
def dynamic_kelly_mult(drawdown_pct: float, win_rate: float, market_mode: str) -> float:
    """
    三因子联动 Kelly 倍数修正因子：
    1. 回撤因子：回撤 < 4% → 1.0，≥ 4% 线性衰减至地板 0.20
    2. 胜率因子：< 40% → 0.75，> 60% → 1.15，40-60% 线性插值
    3. 市场模式因子：趋势市 → 1.12，其余 → 1.0
    最终三因子连乘，钳制在 [0.18, 1.25]
    """
    # 因子1：回撤
    if drawdown_pct <= 0:
        dd_mult = 1.0
    elif drawdown_pct < 0.04:
        dd_mult = 1.0
    else:
        dd_mult = max(0.20, 1.0 - (drawdown_pct - 0.04) * 10)

    # 因子2：胜率（min_sample=8 已保证 win_rate 有意义）
    if win_rate < 0.40:
        wr_mult = 0.75
    elif win_rate > 0.60:
        wr_mult = 1.15
    else:
        wr_mult = 0.75 + (win_rate - 0.40) / 0.20 * 0.40  # 线性插值 0.75→1.15

    # 因子3：市场模式
    mode_mult = 1.12 if market_mode == "趋势" else 1.0

    result = dd_mult * wr_mult * mode_mult
    return max(0.18, min(1.25, result))

# 类定义开始
class RiskGuard:
    def __init__(self, trader):
        self.trader = trader
        self.okx = trader.trader  # OkxTrader 快捷引用

    # ── 8 methods extracted from ETHTrader ────────────────────────────────

    # ---------- pending订单一致性检查 ----------
    def _check_pending_consistency(self):
        """
        每小时一致性检查：
        1. pending_orders DB 中的记录是否与内存 pos.pending_ord_id 一致
        2. _reserved_margin 是否与 DB 中的 margin 总和一致
        3. 幽灵仓位：本地有持仓但交易所无
        4. 孤儿订单：交易所有持仓但本地无
        """
        try:
            db_orders = get_all_pending_orders()
            total_db_margin = sum(float(p.get("margin", 0)) for p in db_orders)

            pos = self.trader.pos
            if pos:
                mem_pending_id = pos.pending_ord_id

                # DB有记录但内存无 → 清理孤儿DB记录
                if db_orders and not mem_pending_id:
                    for p in db_orders:
                        if p.get("symbol") == CFG.symbol:
                            log.warning(f"⚠️ [{CFG.symbol}] DB有pending记录但内存无，清理: {p['ord_id']}")
                            delete_pending_order(p["ord_id"])
                            _webhook("⚠️ 状态不一致", f"[{CFG.symbol}] DB有pending但内存无，清理 {p['ord_id']}")
                    # 重新查询DB，获取清理后的真实合计值
                    db_orders = get_all_pending_orders()
                    total_db_margin = sum(float(p.get("margin", 0)) for p in db_orders)

                # 内存有pending但DB无 → 记录告警（可能是正常情况，如新下单）
                if mem_pending_id and not any(p["ord_id"] == mem_pending_id for p in db_orders):
                    log.warning(f"⚠️ [{CFG.symbol}] 内存有pending_ord_id但DB无记录: {mem_pending_id}")

            # 校验 _reserved_margin
            with self.trader._margin_lock:
                diff = abs(self.trader._reserved_margin - total_db_margin)
                if total_db_margin == 0 and self.trader._reserved_margin > 0:
                    # DB无pending订单但内存有值：历史遗留，直接清零（上次运行的订单已成交/取消）
                    log.debug(f"🔓 清除历史遗留 _reserved_margin: {self.trader._reserved_margin:.2f}U")
                    self.trader._reserved_margin = 0.0
                elif diff > 0.5:  # 差异>0.5U时告警
                    log.warning(f"⚠️ _reserved_margin 不一致: 内存={self.trader._reserved_margin:.2f} DB合计={total_db_margin:.2f} 差异={diff:.2f}")
                    _webhook("⚠️ 保证金不一致", f"内存={self.trader._reserved_margin:.2f}U vs DB={total_db_margin:.2f}U")

            # 幽灵仓位检查：本地有持仓但交易所无
            try:
                resp = self.okx.get_positions()
                if resp.get("code") == "0":
                    exch_syms = {p.get("instId") for p in resp.get("data", []) if abs(float(p.get("pos", 0))) > 0}
                    pos = self.trader.pos
                    if pos and pos.side and CFG.symbol not in exch_syms:
                        log.error(f"⚠️ [{CFG.symbol}] 幽灵仓位！本地有{pos.side}持仓但交易所无")
                        _webhook("⚠️ 幽灵仓位", f"[{CFG.symbol}] 本地有{pos.side}持仓但交易所无，请人工确认")
            except Exception as pos_e:
                log.warning(f"幽灵仓位检查失败: {pos_e}")
        except Exception as e:
            log.error(f"一致性检查异常: {e}")

    # ---------- 权益暴露率 ----------
    def _total_exposure_pct(self) -> float:
        """计算当前持仓的名义杠杆率 = 名义价值 / 总权益"""
        equity = self.trader.latest_equity or 0.0
        if equity <= 0 or not self.trader.pos.side:
            return 0.0
        ct_val = self.okx.contract_sizes.get(CFG.symbol, 0.01)
        notional = self.trader.pos.size * ct_val * self.trader.pos.entry_price
        return notional / equity

    # ---------- 风控循环 ----------
    def _risk_control_loop(self):
        """风控线程：每 2s 检查单品种持仓

        关键设计：风控线程只读取持仓快照（持锁），然后释放锁再做决策。
        _close / _adjust_sl_tp 内部会自己获取锁，不能在持锁时调用它们。
        """
        while not self.trader._stop:
            try:
                pos  = self.trader.pos
                lock = self.trader.lock
                price      = self.trader._get_price(CFG.symbol)
                mark_price = self.trader._get_mark_price()

                # ── 持锁读取快照（毫秒级，无 I/O）────────────────────────
                # 超时检测：若 1s 内抢不到锁说明主线程有卡顿，记录告警但不阻塞风控
                _lock_t0 = time.monotonic()
                if not lock.acquire(timeout=1.0):
                    log.warning(f"⚠️ [{CFG.symbol}] 风控线程等锁超时(>1s)，跳过本轮（主线程可能卡顿）")
                    time.sleep(CFG.risk_check_interval)
                    continue
                try:
                    _lock_wait = time.monotonic() - _lock_t0
                    if _lock_wait > 0.2:
                        log.debug(f"[{CFG.symbol}] 风控锁等待 {_lock_wait*1000:.0f}ms")
                    side               = pos.side
                    entry              = pos.entry_price
                    liq                = pos.liq_price
                    sl                 = pos.stop_loss
                    tp                 = pos.take_profit
                    size               = pos.size
                    peak               = pos.peak_price
                    trailing           = pos.trailing_active
                    moved_stop         = pos.moved_stop
                    pending_ord_id      = pos.pending_ord_id
                    breakeven_triggered      = pos.breakeven_triggered  # 高优先级3 Fix
                    partial_tp_triggered     = pos.partial_tp_triggered
                    partial_tp_2_5R_triggered = pos.partial_tp_2_5R_triggered
                finally:
                    lock.release()

                if not side or price <= 0:
                    time.sleep(CFG.risk_check_interval)
                    continue

                if entry <= 0:
                    time.sleep(CFG.risk_check_interval)
                    continue
                pnl_pct = (
                    (price - entry) / entry if side == "long"
                    else (entry - price) / entry
                )

                # ── 释放锁后做决策（_close/_adjust_sl_tp 内部自己加锁）──
                # ── 状态不同步检测（幽灵仓位 & 真实仓位消失）──────────────
                # 节流：每30s检查一次，避免每2s调用REST浪费配额
                if side and (time.monotonic() - getattr(self.trader, '_last_ghost_chk_ts', 0) > 10):
                    self.trader._last_ghost_chk_ts = time.monotonic()
                    try:
                        pos_resp = self.okx.get_positions()
                        if pos_resp.get("code") == "0":
                            # 按品种过滤：只检查 CFG.symbol，不能用 data 非空代替
                            has_sym_pos = any(
                                p.get("instId") == CFG.symbol and abs(float(p.get("pos", 0))) > 0
                                for p in pos_resp.get("data", [])
                            )
                            if not has_sym_pos:
                                # 幽灵仓位：本地认为有持仓，但交易所没有
                                log.error(f"👻 [{CFG.symbol}] 幽灵仓位检测：本地持仓{side}但交易所无持仓，尝试平仓清除")
                                _webhook("👻 幽灵仓位", f"[{CFG.symbol}] 本地持仓{side}但交易所无持仓，尝试平仓")
                                self.trader.position_exec._close("幽灵仓位清除", symbol=CFG.symbol)
                                time.sleep(CFG.risk_check_interval)
                                continue
                    except Exception as e:
                        log.warning(f"[{CFG.symbol}] 幽灵仓位检测异常（将30s后重试）: {e}")

                # 硬止损
                if pnl_pct <= -CFG.hard_stop_loss_pct:
                    log_event("hard_stop_close", {
                        "sym": CFG.symbol, "price": price,
                        "pnl_pct": pnl_pct, "trigger": "program_hard_stop",
                        "threshold": CFG.hard_stop_loss_pct
                    })
                    self.trader.position_exec._close("硬止损触发", symbol=CFG.symbol)
                    time.sleep(CFG.risk_check_interval)
                    continue

                # 强平预警（标记价）
                if liq > 0 and mark_price > 0:
                    # 计算距强平的距离（百分比）
                    if side == "long":
                        # long: 标记价 > 强平价 才安全，距离 = (mark - liq) / mark
                        dist_pct = (mark_price - liq) / mark_price
                    else:
                        # short: 标记价 < 强平价 才安全，距离 = (liq - mark) / liq
                        dist_pct = (liq - mark_price) / liq

                    # 负值表示已被强平（标记价已超过强平价）
                    if dist_pct < 0:
                        log.critical(f"💥 [{CFG.symbol}] 已被强平！标记价={mark_price:.4f} 强平价={liq:.4f}")
                        self.trader.position_exec._close("强平触发", symbol=CFG.symbol)
                        time.sleep(CFG.risk_check_interval)
                        continue

                    # 安全距离低于阈值时警告
                    if dist_pct < CFG.liq_warn_pct:
                        # 12.12：加冷却防止重复触发（同一仓位 60s 内只警告一次）
                        if time.monotonic() - self.trader._liq_warn_ts > 60:
                            self.trader._liq_warn_ts = time.monotonic()
                            log.warning(f"⚠️ [{CFG.symbol}] 强平预警！标记价={mark_price:.4f} 距强平={dist_pct*100:.2f}%")
                            self.trader.position_exec._close("强平预警自动减仓", symbol=CFG.symbol)
                        time.sleep(CFG.risk_check_interval)
                        continue

                # ── SL止损单验证 & 强制市价止损保护 ────────────────────────
                # 如果pending_ord_id还存在，检查是否已成交/取消
                if pending_ord_id:
                    try:
                        status = self.okx.get_order_status(pending_ord_id)
                        if status.get("code") == "0" and status.get("data"):
                            ord_state = status["data"][0].get("state", "")
                            # 如果挂单已消失（非live/partially_filled）但本地仍有持仓
                            # 说明可能是SL附损单被触发成交，或挂单被取消但未同步
                            if ord_state not in ("live", "partially_filled"):
                                log.warning(f"⚠️ [{CFG.symbol}] 挂单{pending_ord_id}状态={ord_state}，本地持仓未清算，触发市价止损保护")
                                _webhook("🛡️ SL丢失保护", f"[{CFG.symbol}] 挂单已{ord_state}但本地仍有持仓，执行市价止损")
                                self.trader.position_exec._close("SL丢失市价保护", symbol=CFG.symbol)
                                time.sleep(CFG.risk_check_interval)
                                continue
                        else:
                            # API失败时跳过本次检查，不影响其他逻辑
                            log.debug(f"[{CFG.symbol}] 查询pending_ord状态失败: {status}")
                    except Exception as e:
                        log.debug(f"[{CFG.symbol}] SL验证异常: {e}")

                # 追踪止损（使用ATR动态调整）
                atr_trailing = self.trader._atr_val
                market_mode_trailing = self.trader._market_mode
                # P0: 追踪止损激活阈值全面ATR自适应（随波动率动态缩放，静态值仅作无ATR时的兜底）
                _tr_atr_r = atr_trailing / price if atr_trailing > 0 and price > 0 else 0
                if market_mode_trailing == "震荡激进":
                    act_pct = max(0.005, min(0.015, _tr_atr_r * 1.5)) if _tr_atr_r > 0 else 0.008  # 下限从0.3%→0.5%，激活稍晚
                elif market_mode_trailing == "震荡":
                    act_pct = max(0.008, min(0.020, _tr_atr_r * 1.5)) if _tr_atr_r > 0 else 0.010  # 下限从0.5%→0.8%
                else:
                    # 趋势：ATR×2.5，下限1.5%防过于敏感，上限3.5%防过于迟钝
                    act_pct = max(0.015, min(0.035, _tr_atr_r * 2.5)) if _tr_atr_r > 0 else CFG.trailing_act_pct
                # 检查 AI 平仓协调标志，防止与 AI 平仓重复下单
                if hasattr(self.trader, "_ai_close_pending_until") and time.monotonic() < self.trader._ai_close_pending_until:
                    log.debug(f"[{CFG.symbol}] AI 平仓协调中，跳过追踪止损 (剩余{self.trader._ai_close_pending_until - time.monotonic():.1f}s)")
                else:
                    if self._check_trailing_stop(price, symbol=CFG.symbol, atr=atr_trailing, market_mode=market_mode_trailing, act_pct=act_pct):
                        log_event("trailing_stop_close", {
                            "sym": CFG.symbol, "price": price,
                            "peak": self.trader.pos.peak_price, "trigger": "program_trailing",
                            "market_mode": market_mode_trailing
                        })
                        self.trader.position_exec._close("追踪止损触发", symbol=CFG.symbol)
                        time.sleep(CFG.risk_check_interval)
                        continue

                # ── P2: 超时强制平仓（time_stop_minutes，市场模式感知）─────────────
                if self.trader.pos.open_time:
                    _hold_mins_now = (datetime.now(UTC) - self.trader.pos.open_time).total_seconds() / 60
                    _eff_ts = (
                        CFG.time_stop_minutes * 0.25 if market_mode_trailing == "震荡激进"  # 25%
                        else CFG.time_stop_minutes * 0.33 if market_mode_trailing == "震荡"  # 33%
                        else CFG.time_stop_minutes  # 趋势：100%
                    )
                    _eff_ts = max(120, int(_eff_ts))  # 绝对下限 2h
                    if _hold_mins_now >= _eff_ts:
                        log.warning(
                            f"⏰ [{CFG.symbol}] 超时强制平仓：持仓{_hold_mins_now:.0f}分钟"
                            f" ≥ {_eff_ts}分钟（{market_mode_trailing}，配置={CFG.time_stop_minutes}分钟）"
                        )
                        self.trader.position_exec._close(f"超时平仓({market_mode_trailing},{_eff_ts}min)", symbol=CFG.symbol)
                        time.sleep(CFG.risk_check_interval)
                        continue

                # ── 强制平仓信号检测（与硬止损同级）────────────────────────────
                # 优化10：极端市场条件触发无条件平仓
                try:
                    # 获取市场情绪（1小时缓存，若缓存过期则刷新）
                    now = datetime.now(UTC)
                    msd = None
                    if (now - self.trader.sentiment_cache["time"]).total_seconds() > 3600:
                        msd = fetch_market_sentiment_data(CFG.symbol)
                        self.trader.sentiment_cache = {"data": msd, "time": now}
                    msd = self.trader.sentiment_cache.get("data") or {}

                    # 获取资金费率
                    fc = self.trader.funding_cache
                    funding = fc.get("data", {})
                    funding_rate = funding.get("funding_rate", 0) if funding else 0

                    ls_ratio = msd.get("ls_ratio")
                    oi_change = msd.get("oi_change_pct")

                    force_close_reason = None
                    # 资金费率极端值（从 0.5%→0.8%，降低误杀）
                    if abs(funding_rate) > 0.008:  # >0.8%
                        if funding_rate > 0 and side == "long":
                            force_close_reason = f"资金费率极端正值({funding_rate*100:.3f}%)抑制做多"
                        elif funding_rate < 0 and side == "short":
                            force_close_reason = f"资金费率极端负值({funding_rate*100:.3f}%)抑制做空"
                    # 多空比极端
                    if ls_ratio is not None:
                        if ls_ratio > 3.0 and side == "long":
                            force_close_reason = f"多空比极值({ls_ratio:.2f})多头拥挤"
                        elif ls_ratio < 0.33 and side == "short":
                            force_close_reason = f"多空比极值({ls_ratio:.2f})空头拥挤"
                        # 2.5 增强：多空比从极端值快速回归（情绪退潮）
                        prev_ls = self.trader._prev_ls_ratio
                        if prev_ls is not None:
                            if prev_ls > 2.0 and ls_ratio < 1.2 and side == "long":
                                force_close_reason = f"多空比从{prev_ls:.2f}快速降至{ls_ratio:.2f}，情绪退潮，平多仓"
                            elif prev_ls < 0.5 and ls_ratio > 0.8 and side == "short":
                                force_close_reason = f"多空比从{prev_ls:.2f}快速升至{ls_ratio:.2f}，恐慌缓解，平空仓"
                    # OI剧烈下降（从 -20%→-30%，降低误杀）
                    if oi_change is not None and oi_change < -0.30:  # <-30%
                        force_close_reason = f"OI暴跌({oi_change*100:.1f}%)大规模清算"

                    if force_close_reason:
                        log.warning(f"🚨 [{CFG.symbol}] 强制平仓触发: {force_close_reason}")
                        self.trader.position_exec._close(force_close_reason, symbol=CFG.symbol)
                        time.sleep(CFG.risk_check_interval)
                        continue
                except Exception as force_e:
                    log.debug(f"强制平仓检测异常: {force_e}")

                # 移动止损优化（盈利后移至成本+0.5R）
                if not moved_stop and pnl_pct >= 0.01:  # 盈利≥1%
                    if (side == "long" and sl < entry) or (side == "short" and sl > entry):
                        # 计算1R距离 = abs(entry - sl)
                        risk_dist = abs(entry - sl) if sl > 0 else 0
                        if risk_dist > 0:
                            new_sl = entry + 0.5 * risk_dist if side == "short" else entry - 0.5 * risk_dist
                            with lock:
                                pos.stop_loss = new_sl
                                pos.moved_stop = True
                            save_state_to_disk(pos)
                            sz_str = str(max(1, int(size)))
                            be_ok = self.okx.update_algo_orders(side, sz_str, new_sl, tp, symbol=CFG.symbol)
                            if be_ok:
                                log.debug(f"🛡️ [{CFG.symbol}] 移动止损至成本+0.5R: {new_sl:.4f}")
                            else:
                                log.error(f"❌ [{CFG.symbol}] 移动止损挂单失败！")
                        else:
                            # 若无止损，直接设为成本
                            with lock:
                                pos.stop_loss = entry
                                pos.moved_stop = True
                            save_state_to_disk(pos)
                            sz_str = str(max(1, int(size)))
                            be_ok = self.okx.update_algo_orders(side, sz_str, entry, tp, symbol=CFG.symbol)
                            if be_ok:
                                log.debug(f"🛡️ [{CFG.symbol}] 保本止损已强制设置={entry:.4f}")

                # 1.5R 分批止盈（持仓≥2张时）：平25%仓位 + SL移至0.5R（成本以下，给正常波动留空间）
                # 2.5R 再平25%：多级分批止盈（需配合 partial_tp_2_5R_triggered 标记）
                _tp_2_5R_close_pct = 0.25   # 每次平25%（从30%降低，让趋势延续时赚更多）
                _sl_move_r = 0.50             # SL移至成本以下0.5R（从0.3R放宽，防止正常波动扫掉剩余仓位）
                _tp_tighten_r = 2.00          # ADX<20震荡市：TP收紧至2.0R（从1.8R放宽）
                _tp_cool_seconds = 600         # 10分钟TP调整冷却

                # 读取ADX和当前TP用于ADX规则判断
                _cached = self.trader._ind_15m_cache
                _, _ind_cached, _ = _cached
                adx_15m = _ind_cached.get("adx", 25)
                market_mode_tp = self.trader._market_mode
                tp_dist_pct = abs(tp - entry) / entry if tp > 0 and entry > 0 else 0
                sl_dist_pct = abs(entry - sl) / entry if sl > 0 and entry > 0 else 0

                # ── 规则化风控：ADX<20震荡市收紧TP至2.0R（双保险，不依赖AI）────────
                # 修复：已触发1.5R分批止盈后不执行TP收紧（让剩余仓位自由运行到原始TP）
                if (market_mode_tp == "震荡" and adx_15m < 20
                        and tp > 0 and entry > 0 and tp_dist_pct > sl_dist_pct * _tp_tighten_r
                        and not partial_tp_triggered):
                    last_adj = self.trader._last_adjust_time
                    if (datetime.now(UTC) - last_adj).total_seconds() > _tp_cool_seconds:
                        new_tp = entry + _tp_tighten_r * abs(entry - sl) if side == "long" else entry - _tp_tighten_r * abs(entry - sl)
                        with lock:
                            pos.take_profit = new_tp
                        save_state_to_disk(pos)
                        sz_str = str(max(1, int(size)))
                        ok = self.okx.update_algo_orders(side, sz_str, sl, new_tp, symbol=CFG.symbol)
                        self.trader._last_adjust_time = datetime.now(UTC)
                        if ok:
                            log.warning(f"🛡️ [{CFG.symbol}] ADX={adx_15m:.1f}<20震荡市，TP从{tp:.4f}收紧至{new_tp:.4f}(2.0R)")
                        _webhook("🛡️ TP自动收紧", f"[{CFG.symbol}] ADX={adx_15m:.1f}震荡市，TP→{new_tp:.4f}")

                # ── 1.5R 分批止盈：平25% + SL移至0.5R ─────────────────────────────
                # 修复：size >= 1 即可触发（1 张仓位也享受分批止盈保护）
                if (size >= 1 and not partial_tp_triggered
                        and sl > 0 and entry > 0):
                    # 1.5R触发：盈利达到止损距离的1.5倍
                    if pnl_pct >= sl_dist_pct * 1.5:
                        close_sz = max(1, int(size * 0.25))   # 平25%
                        log.debug(f"🎯 [{CFG.symbol}] 盈利达1.5R({pnl_pct*100:.2f}%)，分批止盈 {close_sz}张(25%)")
                        try:
                            close_side = "sell" if side == "long" else "buy"
                            # 注：风控线程直接发起市价平仓，不经过 position_exec（紧急平仓逻辑）
                            close_res  = self.okx._request(
                                "POST", "/api/v5/trade/order",
                                body_data={"instId": CFG.symbol, "tdMode": "isolated",
                                           "side": close_side, "posSide": side,
                                           "ordType": "market", "sz": str(close_sz),
                                           "reduceOnly": "true"}
                            )
                            if close_res.get("code") == "0":
                                sz_remain_1r = max(1, int(size - close_sz))
                                # 锁内只做内存状态计算和状态持久化
                                with lock:
                                    partial_tp_triggered = True
                                    pos.partial_tp_triggered = True
                                    breakeven_triggered = True
                                    pos.breakeven_triggered = True
                                    pos.size = sz_remain_1r
                                    if side == "long":
                                        new_sl = entry - _sl_move_r * abs(entry - sl)
                                    else:
                                        new_sl = entry + _sl_move_r * abs(entry - sl)
                                    pos.stop_loss = new_sl
                                save_state_to_disk(pos)
                                # 锁外才发起网络API调用，遵守"持锁期间禁止网络IO"规则
                                _sz_for_algo = str(sz_remain_1r)
                                _sl_for_algo = new_sl
                                sz_remain = _sz_for_algo
                                new_sl = _sl_for_algo
                                algo_ok = self.okx.update_algo_orders(side, sz_remain, new_sl, tp, symbol=CFG.symbol)
                                self.trader._last_adjust_time = datetime.now(UTC)
                                if algo_ok:
                                    log.warning(f"🛡️ [{CFG.symbol}] 1.5R分批平{close_sz}张，SL移至{new_sl:.4f}(0.3R成本以下)")
                                else:
                                    log.error(f"❌ [{CFG.symbol}] 1.5R分批后更新算法单失败！")
                                    _webhook("❌ 算法单更新失败", f"[{CFG.symbol}] 1.5R分批后SL挂单失败，SL={new_sl:.4f}，请手动设置")
                                _webhook("🎯 1.5R分批止盈", f"[{CFG.symbol}] 平{close_sz}张(25%)，剩余SL=0.5R({new_sl:.4f})")
                            else:
                                log.error(f"❌ [{CFG.symbol}] 1.5R分批止盈失败: {close_res.get('msg', '')}")
                        except Exception as e:
                            log.error(f"[{CFG.symbol}] 1.5R分批止盈异常: {e}")

                # ── 2.5R 再平25%：多级分批止盈 ─────────────────────────────────
                if (size >= 1 and not partial_tp_2_5R_triggered
                        and sl > 0 and entry > 0):
                    # 2.5R触发：盈利达到止损距离的2.5倍
                    if pnl_pct >= sl_dist_pct * 2.5:
                        close_sz = max(1, int(size * _tp_2_5R_close_pct))
                        log.warning(f"🎯 [{CFG.symbol}] 盈利达2.5R({pnl_pct*100:.2f}%)，再次分批止盈 {close_sz}张({int(_tp_2_5R_close_pct*100)}%)")
                        try:
                            close_side = "sell" if side == "long" else "buy"
                            # 注：风控线程直接发起市价平仓，不经过 position_exec（紧急平仓逻辑）
                            close_res  = self.okx._request(
                                "POST", "/api/v5/trade/order",
                                body_data={"instId": CFG.symbol, "tdMode": "isolated",
                                           "side": close_side, "posSide": side,
                                           "ordType": "market", "sz": str(close_sz),
                                           "reduceOnly": "true"}
                            )
                            if close_res.get("code") == "0":
                                sz_new = max(0, int(size - close_sz))
                                # 锁内只做内存状态计算和状态持久化
                                with lock:
                                    partial_tp_2_5R_triggered = True
                                    pos.partial_tp_2_5R_triggered = True
                                    pos.size = sz_new
                                    _sl_2r = pos.stop_loss or sl
                                    _tp_2r = pos.take_profit or tp
                                save_state_to_disk(pos)
                                # 锁外才发起网络API调用，遵守"持锁期间禁止网络IO"规则
                                _sz_for_algo = str(sz_new)
                                _sl_for_algo_2 = _sl_2r
                                _tp_for_algo_2 = _tp_2r
                                self.trader._last_adjust_time = datetime.now(UTC)
                                if sz_new > 0:
                                    # 同步更新算法单（SL/TP）到剩余合约数，与 AI 分批止盈保持一致
                                    self.okx.update_algo_orders(side, _sz_for_algo, _sl_for_algo_2, _tp_for_algo_2, symbol=CFG.symbol)
                                    _webhook("🎯 2.5R分批止盈", f"[{CFG.symbol}] 再平{close_sz}张(30%)，剩余 sz={_sz_for_algo}")
                                else:
                                    # 最后一张合约已平，直接重置本地状态（无需等 WS pos=0）
                                    log.info(f"✅ [{CFG.symbol}] 2.5R分批止盈后仓位归零，直接重置状态")
                                    self.trader.state._reset_pos()
                                    _webhook("🎯 2.5R分批止盈（全平）", f"[{CFG.symbol}] 再平{close_sz}张，仓位清零")
                            else:
                                log.error(f"❌ [{CFG.symbol}] 2.5R分批止盈失败: {close_res.get('msg', '')}")
                        except Exception as e:
                            log.error(f"[{CFG.symbol}] 2.5R分批止盈异常: {e}")

                # ── 全局回撤因子写入 GS（position_exec 读取后联用胜率+市场模式）────
                current_equity = self.trader.latest_equity
                if current_equity > 0:
                    start_bal = gs_get("start_balance", current_equity)
                    drawdown = 0.0
                    if start_bal > 0:
                        drawdown = (start_bal - current_equity) / start_bal
                    # 纯回撤因子
                    if drawdown <= 0 or drawdown < 0.04:
                        _dd_factor = 1.0
                    else:
                        _dd_factor = max(0.20, 1.0 - (drawdown - 0.04) * 10)
                    gs_set("dd_kelly_mult", _dd_factor)
                    if drawdown >= CFG.max_equity_drawdown_pct:
                        win_rate = gs_get("last_24h_win_rate", 0.5)
                        _kelly_mult = dynamic_kelly_mult(drawdown, win_rate, "震荡")
                        log.warning(
                            f"⚠️ 回撤预警：{drawdown*100:.2f}% | Kelly衰减至×{_kelly_mult:.2f} | "
                            f"系统继续运行（不暂停）"
                        )
                        _webhook(
                            "⚠️ 回撤预警",
                            f"权益回撤 {drawdown*100:.2f}%，Kelly系数衰减至 ×{_kelly_mult:.2f}\n"
                            f"当前权益: {current_equity:.2f}U | 基准: {start_bal:.2f}U\n"
                            f"系统继续运行，仓位自动缩减"
                        )

                time.sleep(CFG.risk_check_interval)
            except Exception as e:
                log.error(f"风控循环异常: {e}")
                time.sleep(5)

    # ---------- 追踪止损 ----------
    def _check_trailing_stop(self, price: float, symbol: str = None, atr: float = 0, market_mode: str = "趋势", act_pct: float = None) -> bool:
        """
        优化移动止损：使用ATR动态调整追踪距离
        趋势市：追踪距离 1.5×ATR
        震荡市：追踪距离 0.8×ATR
        """
        sym = CFG.symbol
        pos = self.trader.pos
        if not pos.side or pos.entry_price == 0 or price <= 0:
            return False
        is_update = False
        trigger   = False

        # ── ATR动态追踪距离 ─────────────────────────────────────────────────
        # 优先使用 AI 在 _do_light_adjust 中指定的倍数；否则按市场模式选择
        if atr > 0:
            if pos.trailing_dist_atr_mult is not None:
                trailing_dist = pos.trailing_dist_atr_mult * atr / price  # AI 动态指定
            else:
                # 从 0.8→1.2 ATR（震荡）和 1.0→1.5 ATR（震荡激进），给正常波动留出呼吸空间
                _td_mult = 2.0 if market_mode == "趋势" else (1.5 if market_mode == "震荡激进" else 1.2)
                trailing_dist = _td_mult * atr / price
        else:
            trailing_dist = CFG.trailing_dist_pct  # 兜底用配置值

        # ── 动态激活阈值：震荡市更早锁定利润 ────────────────────────────────
        eff_act_pct = act_pct if act_pct is not None else CFG.trailing_act_pct

        # ── 时间止损：持仓超时且盈利不足时移动止损至成本 ──────────────────────
        holding_seconds = (datetime.now(UTC) - pos.open_time).total_seconds() if pos.open_time else 0
        holding_minutes = holding_seconds / 60
        if holding_seconds >= CFG.time_stop_profit_minutes * 60:  # 转换为秒
            if pos.side == "long":
                pnl_pct = (price - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - price) / pos.entry_price
            if 0 < pnl_pct < CFG.time_stop_profit_pct:
                # 盈利不足，将止损移至成本价
                if pos.side == "long":
                    new_sl = pos.entry_price * (1 - 0.001)  # 成本-0.1%作为缓冲（止损低于成本）
                else:
                    new_sl = pos.entry_price * (1 + 0.001)  # 成本+0.1%作为缓冲（止损高于成本）
                if new_sl != pos.stop_loss:
                    self.trader._pos_mgr.submit(PositionIntent(
                        intent_type=PositionIntentType.UPDATE_SL,
                        payload={"sl": new_sl},
                        reason="time_stop"
                    ))
                    is_update = True
                    log.info(f"⏰ [{sym}] 时间止损触发：持仓{holding_minutes/60:.0f}分钟 盈利{pnl_pct*100:.2f}% 移动SL至{new_sl:.4f}")
        # ────────────────────────────────────────────────────────────────────────

        if pos.side == "long":
            pnl_pct = (price - pos.entry_price) / pos.entry_price
            if price > pos.peak_price:
                self.trader._pos_mgr.submit(PositionIntent(
                    intent_type=PositionIntentType.UPDATE_PEAK,
                    payload={"peak_price": price},
                    reason="trailing_long"
                ))
                is_update = True
            if pnl_pct >= eff_act_pct and not pos.trailing_active:
                self.trader._pos_mgr.submit(PositionIntent(
                    intent_type=PositionIntentType.UPDATE_PEAK,
                    payload={"trailing_active": True},
                    reason="trailing_long_activate"
                ))
                is_update = True
                log.info(f"🔔 [{sym}] 追踪止损已激活(多) 盈利={pnl_pct*100:.2f}%，ATR动态距离={trailing_dist*100:.2f}%")
            if pos.trailing_active:
                drawdown = (pos.peak_price - price) / pos.peak_price
                if drawdown >= trailing_dist:
                    log.info(f"📉 [{sym}] 追踪止损触发(多) 峰值={pos.peak_price:.4f} 回撤={drawdown*100:.2f}%")
                    trigger = True
        else:
            pnl_pct = (pos.entry_price - price) / pos.entry_price
            if price < pos.peak_price or pos.peak_price == 0:
                self.trader._pos_mgr.submit(PositionIntent(
                    intent_type=PositionIntentType.UPDATE_PEAK,
                    payload={"peak_price": price},
                    reason="trailing_short"
                ))
                is_update = True
            if pnl_pct >= eff_act_pct and not pos.trailing_active:
                self.trader._pos_mgr.submit(PositionIntent(
                    intent_type=PositionIntentType.UPDATE_PEAK,
                    payload={"trailing_active": True},
                    reason="trailing_short_activate"
                ))
                is_update = True
                log.info(f"🔔 [{sym}] 追踪止损已激活(空) 盈利={pnl_pct*100:.2f}%，ATR动态距离={trailing_dist*100:.2f}%")
            if pos.trailing_active:
                rally = (price - pos.peak_price) / pos.peak_price
                if rally >= trailing_dist:
                    log.info(f"📈 [{sym}] 追踪止损触发(空) 低点={pos.peak_price:.4f} 反弹={rally*100:.2f}%")
                    trigger = True
        if is_update:
            save_state_to_disk(pos)
        return trigger

    # ---------- 资金费率风险检查 ----------
    def _check_funding_risk(self, funding: Dict, action: str, confidence: float = 0.0) -> bool:
        """
        除原有费率方向抑制外，新增结算前保护：
        距下次结算 ≤ funding_settlement_guard_minutes 分钟，
        且费率绝对值 ≥ funding_settlement_guard_rate，
        除非 AI 置信度 > funding_settlement_guard_confidence，否则跳过本轮开仓。
        """
        rate = funding.get("funding_rate", 0)

        # ── 结算前保护 ──────────────────────────────────────────────────────
        next_ts_str = funding.get("next_funding_time", "")
        if next_ts_str and next_ts_str != "N/A":
            try:
                next_ts_ms = int(next_ts_str)
                next_dt    = datetime.fromtimestamp(next_ts_ms / 1000, tz=UTC)
                mins_to_settle = (next_dt - datetime.now(UTC)).total_seconds() / 60
                if (0 < mins_to_settle <= CFG.funding_settlement_guard_minutes
                        and abs(rate) >= CFG.funding_settlement_guard_rate):
                    if confidence <= CFG.funding_settlement_guard_confidence:
                        log.warning(
                            f"⚠️ 距资金费率结算仅 {mins_to_settle:.1f} 分钟，"
                            f"费率 {rate*100:.4f}%，AI置信度 {confidence:.2f} ≤ "
                            f"{CFG.funding_settlement_guard_confidence}，跳过开仓"
                        )
                        return True
                    else:
                        log.info(
                            f"💡 结算前 {mins_to_settle:.1f} 分钟，但AI高置信度 {confidence:.2f}，允许开仓"
                        )
            except (ValueError, TypeError):
                pass

        # ── 原有费率方向抑制 ─────────────────────────────────────────────────
        if abs(rate) < CFG.funding_rate_thresh:
            return False
        if rate > CFG.funding_rate_thresh and action == "open_long":
            log.warning(f"⚠️ 资金费率偏高({rate*100:.4f}%)，抑制开多")
            return True
        if rate < -CFG.funding_rate_thresh and action == "open_short":
            log.warning(f"⚠️ 资金费率偏低({rate*100:.4f}%)，抑制开空")
            return True
        return False

    # ---------- 每日UTC重置检查 ----------
    def _check_daily_reset(self):
        with self.trader.lock:
            today = datetime.now(UTC).date().isoformat()
            # 双保险：每日 UTC 00:00 ~ 00:10 之间若有任何开仓检查，直接触发重置
            # 即使主循环未及时调用，下次访问时 gs_get 读取默认值 0.0 + 日期比对也能兜底
            if today != gs_get("last_reset_date"):
                new_equity = self.trader.latest_equity
                # Bug F 修复：有持仓时保留 consecutive_losses，防止连亏保护在零点跨日后失效
                # 连亏计数以"平仓事件"为单位，不应被日历日期打断
                has_open_pos = bool(self.trader.pos.side)
                update_dict = {
                    "start_balance":        new_equity,
                    "last_reset_date":      today,
                    "daily_locked":         False,
                    "last_adjust_time":     None,
                    "breakeven_triggered":  False,
                    "consecutive_slippage": 0,
                    "today_realized_pnl":   0.0,
                }
                if not has_open_pos:
                    update_dict["consecutive_losses"] = 0  # 空仓时才重置连亏计数
                gs_update(update_dict)
                gs_set("today_opened_risk", 0.0)  # 每日风险累计重置
                reset_note = "（持仓中，连亏计数保留）" if has_open_pos else ""
                log.info(f"📅 新交易日重置 | 基准权益: {new_equity:.2f} USDT | 已实现盈亏清零 {reset_note}")
                # 保存单品种状态（新日期 + 新基准权益）
                save_state_to_disk(self.trader.pos)

    # ---------- 风险自适应调整 ----------
    def adapt_risk_per_trade(self):
        if not CFG.risk_adapt_enable:
            return CFG.risk_per_trade
        win_rate = gs_get("last_24h_win_rate", 0.5)
        # 目标胜率 0.5，实际高于0.5则增加风险，低于则降低
        ratio = win_rate / CFG.risk_adapt_win_rate_target
        # 限制在[0.5, 1.5]之间
        factor = max(0.5, min(1.5, ratio))
        adapted = CFG.risk_per_trade * factor
        log.debug(f"风险自适应: 24h胜率={win_rate:.2f}, 因子={factor:.2f}, 新risk_per_trade={adapted:.4f}")
        return adapted

    # ---------- 释放pending订单预留保证金 ----------
    def _release_pending_margin(self, ord_id: str, sym: str):
        """从pending_orders表读取margin并释放预留保证金"""
        try:
            log.info(f"🔓 [{sym}] 开始释放挂单 {ord_id} 的预留保证金...")
            pending = get_pending_order_by_id(ord_id)
            log.debug(f"🔓 [{sym}] 查询到pending订单: {pending is not None}")
            if pending and pending.get("margin", 0) > 0:
                margin_to_release = float(pending["margin"])
                with self.trader._margin_lock:
                    self.trader._reserved_margin = max(0.0, self.trader._reserved_margin - margin_to_release)
                log.debug(f"🔓 [{sym}] 挂单取消，释放预留保证金 {margin_to_release:.2f}U")
                delete_pending_order(ord_id)
                log.debug(f"🔓 [{sym}] 已删除pending订单记录")
            else:
                log.debug(f"🔓 [{sym}] pending订单不存在或margin为0")
                delete_pending_order(ord_id)
        except Exception as e:
            log.warning(f"[{sym}] 释放预留保证金异常: {e}")
