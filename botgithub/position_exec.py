"""
Position Exec Module - 开仓/平仓执行、仓位计算
从 ETHTrader 拆分而出
"""

import time
import math
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

# ── 共享基础设施（common.py）──────────────────────────────────────────────
from common import (
    CFG, log, _webhook, gs_get, gs_set, gs_increment, gs_add,
    UTC, log_event, log_kelly_metrics, kelly_optimal_size, _call_chat,
    get_db_conn, SYSTEM_PROMPT_RISK,
)
# ── 数据模型（core.py）──────────────────────────────────────────────────
from core import (
    GLOBAL_STATE, PositionIntent, PositionIntentType, Position, _gs_lock,
    save_state_to_disk, TradingState, TradingEvent, gs_add,
)
# ── config ──────────────────────────────────────────────────────────────
from config import log_slippage

# ── 辅助函数（待实现或从其他模块迁移）────────────────────────────────────
def _get_atr_quantile(current_atr: float, atr_history: List[float]) -> float:
    """计算当前 ATR 在历史 ATR 序列中的分位数"""
    if not atr_history or current_atr <= 0:
        return 0.5
    sorted_atrs = sorted(atr_history)
    rank = sum(1 for a in sorted_atrs if a < current_atr)
    return rank / max(len(sorted_atrs), 1)

def calc_liq_price(side: str, entry: float, leverage: int, balance: float,
                   size: float, ct_val: float) -> float:
    """
    计算逐仓强平价（OKX 简化公式 + 缓冲）
    - 考虑维持保证金率 (MMR=0.4%)
    - 增加 0.1% 缓冲以覆盖手续费/滑点影响
    """
    if entry <= 0 or leverage <= 0:
        return 0.0
    mmr = 0.004       # 维持保证金率 0.4%
    buffer = 0.001    # 额外缓冲 0.1%，覆盖手续费和滑点
    if side == "long":
        return entry * (1 - (mmr + buffer) / leverage)
    else:
        return entry * (1 + (mmr + buffer) / leverage)

class OrderFailedError(Exception):
    """订单失败异常"""
    pass

# ── 从 common.py 导入已实现的函数 ─────────────────────────────────────────
from common import save_pending_order, get_pending_order_by_id, \
    generate_fail_reason_async, _auto_generate_historical_case, \
    update_trade_close, save_trade_open

class PositionExec:
    def __init__(self, trader):
        self.trader = trader
        self.okx = trader.trader  # OkxTrader 快捷引用
        # ConvictionScore 拒绝标志：由 _calc_size_and_margin 设置，_do_open 传递到 _run_symbol
        self._cv_rejected = False
        self._cv_rejected_score = 0.0
        self._cv_rejected_action = ""

    def _do_light_adjust(self, sym: str):
        """
        AI 完全动态决定：止损/止盈调整、分批止盈（20~50%）、移至保本、追踪止损距离。
        硬止损（hard_stop_loss_pct）由风控循环独立守护，不受此函数影响。
        冷却：至少间隔 30s（_adjust_sl_tp 内部也有同等保护，防并发重入）。
        """
        try:
            pos = self.trader.pos
            if not pos or not pos.side:
                return

            price      = self.trader._price_val
            entry      = pos.entry_price
            current_sl = pos.stop_loss
            current_tp = pos.take_profit
            atr        = self.trader._atr_val
            market_mode = self.trader._market_mode or "趋势"
            pnl_pct    = ((price - entry) / entry if pos.side == "long"
                          else (entry - price) / entry) if entry > 0 else 0.0
            _fc_data     = self.trader.funding_cache.get("data") or {}
            funding_rate = _fc_data.get("funding_rate", 0)

            # ── 冷却：至少间隔 30s ──────────────────────────────────────────
            if (datetime.now(UTC) - self.trader._last_adjust_time).total_seconds() < 30:
                return

            sl_min_dist = CFG.sl_min_atr_mult * atr  # 止损距当前价最小保护距离

            prompt = f"""你是专业量化交易员，负责动态管理持仓的止损/止盈。

当前持仓：
- 品种: {sym} | 方向: {pos.side} | 入场价: {entry:.4f} | 当前价: {price:.4f}
- 浮盈: {pnl_pct*100:+.2f}% | SL: {current_sl:.4f} | TP: {current_tp:.4f}
- ATR: {atr:.4f} | 市场模式: {market_mode} | 资金费率: {funding_rate*100:.4f}%
- 持仓量: {int(pos.size)}张 | 追踪止损: {"已激活(峰值=" + f"{pos.peak_price:.4f})" if pos.trailing_active else "未激活"}
- 当前追踪距离: {f"{pos.trailing_dist_atr_mult:.1f}×ATR" if pos.trailing_dist_atr_mult is not None else "自动(市场模式)"}

【硬性约束】
1. SL 必须距当前价至少 {CFG.sl_min_atr_mult}×ATR = {sl_min_dist:.4f}（{"做多时 SL ≤ " + f"{price - sl_min_dist:.4f}" if pos.side == "long" else "做空时 SL ≥ " + f"{price + sl_min_dist:.4f}"}），防频繁扫损
2. partial_tp_pct 仅限 {CFG.partial_tp_min_pct}~{CFG.partial_tp_max_pct} 整数，0=不分批
3. set_breakeven=true 仅在浮盈为正时有意义（SL 移至入场价附近）
4. trailing_dist_atr 范围 0.5~2.5（0=不修改）；趋势市建议 1.5~2.0，震荡市建议 0.8~1.2

【决策指引】
- 浮盈 > 0.5%：开始考虑微调（上移 SL、收紧追踪），为利润提供初步保护
- 浮盈 > 0.8%：考虑移至保本（set_breakeven），SL 移至入场价附近（留 0.1% 缓冲防噪音扫损）
- 浮盈 > 1.2%：收紧追踪距离（trailing_dist_atr 适度缩小），锁定更多浮盈
- 浮盈 > 1.5R：可分批止盈 {CFG.partial_tp_min_pct}~{CFG.partial_tp_max_pct}%，同时 SL 移至成本附近
- 浮亏状态：不移动 SL（只可适度调整 TP 到更保守位置）
- ATR 扩大且趋势强：可放宽追踪距离（trailing_dist_atr 增大）
- 资金费率绝对值 > 0.3% 且方向不利：SL 收紧

只输出 JSON（不含任何其他文字）：
{{"adjust":"sl_only"|"tp_only"|"both"|"partial_tp"|"breakeven"|"no","new_sl":数值,"new_tp":数值,"partial_tp_pct":0,"set_breakeven":false,"trailing_dist_atr":0,"reason":"一句话"}}"""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RISK},
                {"role": "user",   "content": prompt},
            ]
            content = _call_chat(self.trader.ai.client, messages, max_tokens=400, temperature=0.15, timeout=15)

            import json as _json
            try:
                data = _json.loads(content)
            except Exception:
                brace = content.find("{")
                if brace >= 0:
                    try:
                        data = _json.loads(content[brace:content.rfind("}")+1])
                    except Exception:
                        return
                else:
                    return

            adjust        = data.get("adjust", "no")
            new_sl        = float(data.get("new_sl", current_sl) or current_sl)
            new_tp        = float(data.get("new_tp", current_tp) or current_tp)
            partial_pct   = int(data.get("partial_tp_pct", 0) or 0)
            set_be        = bool(data.get("set_breakeven", False))
            trail_atr     = float(data.get("trailing_dist_atr", 0) or 0)
            reason        = data.get("reason", "AI动态调整")

            if adjust == "no":
                return

            log.info(f"🤖 [{sym}] light_adjust: {adjust} | sl={new_sl:.4f} tp={new_tp:.4f} "
                     f"partial={partial_pct}% be={set_be} trail_atr={trail_atr:.1f} | {reason}")

            # ── 1. 分批止盈（优先执行，执行后自动更新剩余仓位的 SL/TP）────────
            if adjust == "partial_tp" or partial_pct > 0:
                partial_pct = max(CFG.partial_tp_min_pct, min(CFG.partial_tp_max_pct, partial_pct))
                cur_size = int(abs(pos.size))
                if cur_size >= 2:
                    close_sz = max(1, int(cur_size * partial_pct / 100))
                    try:
                        close_side = "sell" if pos.side == "long" else "buy"
                        close_res  = self.okx._request(
                            "POST", "/api/v5/trade/order",
                            body_data={"instId": sym, "tdMode": "isolated",
                                       "side": close_side, "posSide": pos.side,
                                       "ordType": "market", "sz": str(close_sz),
                                       "reduceOnly": "true"}
                        )
                        if close_res.get("code") == "0":
                            sz_remain = str(max(1, cur_size - close_sz))
                            self.okx.update_algo_orders(pos.side, sz_remain, new_sl, new_tp, symbol=sym)
                            self.trader._last_adjust_time = datetime.now(UTC)
                            log.warning(f"🎯 [{sym}] AI分批止盈: 平{close_sz}张({partial_pct}%) "
                                        f"剩余{sz_remain}张 SL={new_sl:.4f}")
                            _webhook("🎯 AI分批止盈",
                                     f"[{sym}] 平{close_sz}张({partial_pct}%) 剩余{sz_remain}张\n{reason}")
                        else:
                            log.error(f"❌ [{sym}] AI分批止盈下单失败: {close_res}")
                    except Exception as pe:
                        log.error(f"❌ [{sym}] AI分批止盈异常: {pe}")
                else:
                    log.info(f"⚠️ [{sym}] 分批止盈需要 ≥2 张，当前 {cur_size} 张，跳过")
                return  # 成功执行后返回

            # ── 2. 移至保本（breakeven）────────────────────────────────────────
            if set_be or adjust == "breakeven":
                if pnl_pct > 0 and entry > 0:
                    buf = entry * 0.001  # 0.1% 缓冲
                    be_sl = (entry + buf) if pos.side == "long" else (entry - buf)
                    # breakeven 豁免最小距离检查（保本是安全操作）
                    self._adjust_sl_tp(be_sl, current_tp, f"AI保本({reason})", symbol=sym)
                    log.info(f"⚖️ [{sym}] AI移至保本 SL={be_sl:.4f}")
                else:
                    log.info(f"⚠️ [{sym}] 当前浮亏({pnl_pct*100:.2f}%)，跳过保本操作")
                # breakeven 之后不再做其他调整
                return

            # ── 3. 追踪止损距离更新 ──────────────────────────────────────────
            if trail_atr > 0:
                # 震荡模式保护：不低于 1.0x ATR
                if market_mode in ("震荡", "震荡激进"):
                    trail_atr = max(trail_atr, 1.0)
                trail_atr = max(0.8, min(3.0, trail_atr))
                with self.trader.lock:
                    pos.trailing_dist_atr_mult = trail_atr
                    save_state_to_disk(pos)
                log.info(f"📐 [{sym}] AI更新追踪距离: {trail_atr:.1f}×ATR")

            # ── 4. SL/TP 调整（带最小保护距离校验）─────────────────────────
            if adjust in ("sl_only", "both"):
                # 确保 SL 不低于最小保护距离
                if pos.side == "long":
                    min_allowed_sl = price - sl_min_dist
                    if new_sl > min_allowed_sl:
                        log.info(f"⚠️ [{sym}] AI SL {new_sl:.4f} 过近（最小允许 {min_allowed_sl:.4f}），截断")
                        new_sl = min_allowed_sl
                else:
                    max_allowed_sl = price + sl_min_dist
                    if new_sl < max_allowed_sl:
                        log.info(f"⚠️ [{sym}] AI SL {new_sl:.4f} 过近（最小允许 {max_allowed_sl:.4f}），截断")
                        new_sl = max_allowed_sl

            _final_sl = new_sl if adjust in ("sl_only", "both") else current_sl
            _final_tp = new_tp if adjust in ("tp_only", "both") else current_tp
            # 校验 SL 有效性：tp_only 模式下 current_sl 可能为 0
            if _final_sl <= 0 and pos.side:
                log.warning(f"[{sym}] {adjust} 模式但 SL 无效(={_final_sl})，跳过调整")
                return
            self._adjust_sl_tp(_final_sl, _final_tp, f"AI动态({reason})", symbol=sym)

        except Exception as e:
            log.warning(f"[_do_light_adjust] {sym} 异常: {e}")

    def _do_pyramid_add(self, sym: str):
        """
        检查并执行 AI 制定的加仓计划（pyramid_plan）中的下一步。
        硬性约束：
          - 累计风险（首单 + 所有加仓）≤ pyramid_max_risk_mult × risk_per_trade × 余额
          - 单次加仓量 ≤ initial_size × pyramid_max_ratio
          - 总加仓次数 ≤ pyramid_max_entries
          - 保证金使用率 ≤ max_total_margin_ratio（与 full_position_mode 上限相同）

        锁策略：锁内读取 pos 快照 → 锁外 I/O → 锁内写入
        """
        # ── 阶段1：锁内读取快照 ─────────────────────────────────────────────
        add_risk_usd = 0.0
        with self.trader.lock:
            pos = self.trader.pos
            if not pos or not pos.side or not pos.pyramid_plan:
                return
            add_entries   = pos.pyramid_plan.get("add_entries") or []
            idx           = pos.pyramid_count
            if idx >= len(add_entries) or idx >= CFG.pyramid_max_entries:
                return
            entry_plan    = add_entries[idx]
            trigger_pct   = float(entry_plan.get("price_trigger_pct", 0.5)) / 100.0
            vol_confirm   = bool(entry_plan.get("vol_confirm", False))
            add_ratio     = float(entry_plan.get("ratio", 0.3))
            initial_size  = pos.initial_size if pos.initial_size > 0 else pos.size
            sl            = pos.stop_loss
            pos_side      = pos.side
            pos_size      = pos.size
            pos_entry     = pos.entry_price
            pos_leverage  = pos.leverage
            pos_tp        = pos.take_profit
            pos_init_risk = pos.initial_risk_usd
        # 锁已释放，后续使用快照值

        # ── 阶段2：锁外 I/O ─────────────────────────────────────────────────
        price = self.trader._get_price(sym)
        if price <= 0 or pos_entry <= 0:
            return

        # 价格条件
        pnl_pct = ((price - pos_entry) / pos_entry) if pos_side == "long" else ((pos_entry - price) / pos_entry)
        if pnl_pct < trigger_pct:
            return

        # 成交量条件（可选）
        if vol_confirm:
            _, _ind_15m, _ = self.trader._ind_15m_cache
            vol_surge = (_ind_15m.get("vol_surge", 1.0) if _ind_15m else 1.0)
            if vol_surge < 1.3:
                log.debug(f"[{sym}] 加仓条件未满足：vol_surge={vol_surge:.1f} < 1.3")
                return

        # 加仓量计算
        add_ratio = min(add_ratio, CFG.pyramid_max_ratio)
        # 金字塔加仓风险因子调整：连亏状态下降低加仓比例
        consecutive = gs_get('consecutive_losses', 0)
        if consecutive > 0:
            _consec_mult = max(0.5, 1.0 - consecutive * 0.1)  # 与首单一致的衰减逻辑
            add_ratio = round(add_ratio * _consec_mult, 2)
            log.debug(f'[{sym}] 金字塔加仓风险调整：连亏{consecutive}次，add_ratio {add_ratio:.2f} → {round(add_ratio*_consec_mult, 2):.2f}')
        add_size  = max(1, int(initial_size * add_ratio))

        # 累计风险校验
        atr     = self.trader._atr_val
        ct_val  = self.okx.contract_sizes.get(sym, 0.01)
        dist    = abs(price - sl) if sl > 0 else (atr * CFG.sl_atr_mult if atr > 0 else price * 0.02)
        add_risk_usd    = add_size * dist * ct_val
        cumulative_risk  = pos_init_risk + add_risk_usd

        bal              = self.trader.latest_equity if self.trader.latest_equity > 0 else 1000.0
        adapted_risk     = self.trader.risk_guard.adapt_risk_per_trade()
        max_risk_budget = bal * adapted_risk * CFG.pyramid_max_risk_mult
        max_from_initial = pos_init_risk * CFG.pyramid_max_risk_mult
        if cumulative_risk > max_from_initial:
            log.info(f"⛔ [{sym}] 加仓[{idx+1}]累计风险 ${cumulative_risk:.2f} > 2倍初始风险 ${max_from_initial:.2f}，跳过")
            return
        if cumulative_risk > max_risk_budget:
            log.info(f"⛔ [{sym}] 加仓[{idx+1}]累计风险 ${cumulative_risk:.2f} > 动态上限 ${max_risk_budget:.2f}，跳过")
            return

        # 保证金使用率校验
        lev          = pos_leverage if pos_leverage > 0 else 1
        cur_margin   = pos_size * ct_val * pos_entry / lev
        add_margin   = add_size * ct_val * price / lev
        equity       = self.trader.latest_equity if self.trader.latest_equity > 0 else bal
        margin_ratio = (cur_margin + add_margin) / equity if equity > 0 else 1.0
        if margin_ratio > CFG.max_total_margin_ratio:
            log.info(f"⛔ [{sym}] 加仓[{idx+1}]保证金占比 {margin_ratio*100:.1f}% > "
                     f"{CFG.max_total_margin_ratio*100:.0f}%，跳过")
            return

        # ── 执行加仓市价单 ────────────────────────────────────────────────
        ord_side = "buy" if pos_side == "long" else "sell"
        log.warning(f"🔼 [{sym}] 执行加仓[{idx+1}/{CFG.pyramid_max_entries}]: {add_size}张 "
                    f"({add_ratio*100:.0f}%×首单{initial_size:.0f}张) "
                    f"price={price:.4f} 浮盈={pnl_pct*100:.2f}% "
                    f"累计风险=${cumulative_risk:.2f}/${max_risk_budget:.2f}")
        res = self.okx._request(
            "POST", "/api/v5/trade/order",
            body_data={"instId": sym, "tdMode": "isolated",
                       "side": ord_side, "posSide": pos_side,
                       "ordType": "market", "sz": str(int(add_size))}
        )
        if res.get("code") != "0":
            log.error(f"❌ [{sym}] 加仓[{idx+1}]下单失败: {res}")
            return

        # ── 阶段3：锁内写入 ─────────────────────────────────────────────────
        with self.trader.lock:
            pos = self.trader.pos  # 重新读取（防止重复加仓）
            if pos.pyramid_count >= idx + 1:
                return  # 已由其他线程更新，跳过
            pos.pyramid_count    += 1
            # 修复：不覆盖 initial_risk_usd（首单风险，用于平仓释放 today_opened_risk）
            # cumulative_risk 仅用于后续加仓的风险上限校验，平仓释放由各次开仓时的增量管理
            # pos.initial_risk_usd 保持首单值不变
            save_state_to_disk(pos)

        log.warning(f"✅ [{sym}] 加仓[{idx+1}]成功，已加仓 {idx+1}/{CFG.pyramid_max_entries} 次")
        _webhook("🔼 加仓执行",
                 f"[{sym}] 第{idx+1}次加仓 {add_size}张\n"
                 f"触发浮盈={pnl_pct*100:.2f}% 累计风险=${cumulative_risk:.2f}")

        # ── 加仓后：自动将 SL 移至保本线或原 SL 上方 ────────────────────────
        try:
            with self.trader.lock:
                entry = pos.entry_price
                cur_sl = pos.stop_loss
            if entry > 0:
                new_sl = entry * (1.002 if pos_side == "long" else 0.998)
                if (pos_side == "long" and (cur_sl <= 0 or new_sl > cur_sl)) or \
                   (pos_side == "short" and (cur_sl <= 0 or new_sl < cur_sl)):
                    adj = self._adjust_sl_tp(new_sl, pos_tp, reason="pyramid breakeven", symbol=sym)
                    if adj:
                        log.warning(f"🛡️ [{sym}] 加仓后 SL 移至保本: {new_sl:.4f}")
        except Exception as e_sl:
            log.warning(f"[_do_pyramid_add] SL保本移动失败: {e_sl}")

    def _handle_pending_order(self, sym: str = None) -> bool:
        """
        处理追踪挂单（限价开仓单）的成交回调。
        返回 True 表示订单已完成（filled/canceled），调用方应重新同步仓位。
        """
        sym = sym if sym else CFG.symbol
        lock = self.trader.lock
        pos = self.trader.pos

        # ── 阶段1：锁内读取 pending_ord_id（毫秒级，无 I/O）────────────────
        try:
            lock.acquire()
            if not pos.pending_ord_id:
                return False
            pending_id = pos.pending_ord_id
        finally:
            lock.release()

        # ── 阶段2：锁外网络I/O（查订单状态，3秒超时）──────────────────────────
        resp = None
        try:
            resp = self.okx.get_order_status(pending_id)
        except Exception as e:
            log.warning(f"⚠️ [{sym}] pending订单状态查询异常: {e}")
            return False
        if not resp or resp.get("code") != "0" or not resp.get("data"):
            # 订单不存在/已取消 → 释放预留保证金
            self.trader.risk_guard._release_pending_margin(pending_id, sym)
            with lock:
                pos.pending_ord_id = ""
                pos.partial_filled = 0.0
                save_state_to_disk(pos)
            return False

        data     = resp["data"][0]
        state    = data.get("state", "")
        acc_fill = float(data.get("accFillSz", 0))
        log.debug(f"🔄 [{sym}] pending订单状态: state={state}, acc_fill={acc_fill}")

        # ── 阶段3：锁内更新状态（释放锁后再做网络I/O）──────────────────────────
        if acc_fill > pos.partial_filled:
            with lock:
                pos.partial_filled = acc_fill
                save_state_to_disk(pos)

        if state == "filled":
            log.info(f"🔄 [{sym}] 订单已成交，清理pending状态...")
            # 读取并释放预留保证金（锁外）
            pending = get_pending_order_by_id(pending_id)
            if pending and pending.get("margin", 0) > 0:
                margin_to_release = float(pending["margin"])
                with self.trader._margin_lock:
                    self.trader._reserved_margin = max(0.0, self.trader._reserved_margin - margin_to_release)
                log.info(f"🔓 [{sym}] 成交释放预留保证金: {margin_to_release:.2f}U")
            # 状态清理（锁内）
            with lock:
                pos.pending_ord_id = ""
                pos.partial_filled = 0.0
                save_state_to_disk(pos)
            # 同步仓位（锁外，无锁保护，但持仓已清pending_ord_id，不会重复进入）
            self.trader.state.sync_position(symbol=sym)
            self.trader.state._full_state_sync()
            log.info(f"🔄 [{sym}] pending订单处理完成")
            return True
        elif state in ("canceled", "mmp_canceled"):
            # 取消时也需查询 pending 记录来计算应释放保证金
            pending = get_pending_order_by_id(pending_id)
            total_margin = 0.0
            if pending and pending.get("margin", 0) > 0:
                total_margin = float(pending.get("margin", 0))
                total_size = float(pending.get("size", 0))
                unfilled_ratio = max(0.0, 1.0 - acc_fill / total_size) if total_size > 0 else 1.0
                margin_to_release = total_margin * unfilled_ratio
                if margin_to_release > 0:
                    with self.trader._margin_lock:
                        self.trader._reserved_margin = max(0.0, self.trader._reserved_margin - margin_to_release)
            with lock:
                pos.pending_ord_id = ""
                pos.partial_filled = 0.0
                save_state_to_disk(pos)
            log.info(f"🔄 [{sym}] 订单已取消，清理pending（释放margin={margin_to_release:.2f}U）")
            return False
        return False

    def _adjust_sl_tp(self, new_sl: float, new_tp: float, reason: str, symbol: str = None):
        sym = symbol if symbol else CFG.symbol
        pos  = self.trader.pos
        lock = self.trader.lock
        # 冷却时间：固定 30s（_do_light_adjust 已按 45s 周期限流，此处仅防并发重复写入）
        cool_down_seconds = 30

        # 阶段1：锁内校验 + 读快照（毫秒级，无 I/O）
        with lock:
            if not pos.side:
                return
            last_dt = self.trader._last_adjust_time
            if last_dt and (datetime.now(UTC) - last_dt).total_seconds() < cool_down_seconds:
                return
            price = self.trader._get_price(sym)
            if price <= 0:
                price = pos.entry_price
            if pos.side == "long":
                if new_sl >= price or new_sl <= 0: return
                if new_tp <= price: return
            else:
                if new_sl <= price or new_sl <= 0: return
                if new_tp >= price: return
            sz      = str(int(abs(pos.size)))
            sz_side = pos.side  # 快照方向，防竞态用

        # 阶段2：锁外做网络 I/O，避免 API 超时阻塞风控线程
        success = self.okx.update_algo_orders(sz_side, sz, new_sl, new_tp, symbol=sym)

        # 阶段3：锁内写回状态（防竞态：校验仓位方向未变）
        if success:
            with lock:
                if pos.side != sz_side:  # 持仓已被平掉，放弃写回
                    return
                self.trader._pos_mgr.submit(PositionIntent(PositionIntentType.UPDATE_SL, {"sl": new_sl}, "_adjust_sl_tp"))
                self.trader._pos_mgr.submit(PositionIntent(PositionIntentType.UPDATE_TP, {"tp": new_tp}, "_adjust_sl_tp"))
                self.trader._last_adjust_time = datetime.now(UTC)
                save_state_to_disk(pos)
            log.info(f"🛡️ [{sym}] SL={new_sl:.4f} TP={new_tp:.4f} | {reason}")
            _webhook("调整止盈止损", f"[{sym}] SL={new_sl:.4f} TP={new_tp:.4f}\n{reason}")
            # 12.11：止损价已变化，清除AI缓存强制下轮重新决策
            # 否则AI会基于旧止损价持续输出相同的 adjust_sl_tp
            self.trader._clear_ai_cache(symbol=sym)

    def _calc_size_and_margin(self, dec: Dict, price: float, bal: float, atr: float,
                               is_long: bool, sym: str, market_mode: str, ind_15m: Dict,
                               risk_mult: float = 1.0, committee_opposing: int = 0,
                               action: str = "", depth: Dict = None, trend_score: float = 0.5) -> Dict:
        """
        计算最优开仓张数和所需保证金（Kelly 公式 + 动态风控）。
        返回结构包含 skip 标志和完整仓位参数。
        """
        lev = int(dec.get("suggested_leverage") or CFG.max_leverage)
        ct_val = self.okx.contract_sizes.get(sym, 0.01)
        lot_sz = self.okx.lot_sizes.get(sym, 1.0)
        # latest_avail_bal 已经由 OKX WS 实时推送（挂单冻结后已扣除），无需再减本地 _reserved_margin
        effective_bal = max(0.0, bal)
        equity_for_cap = self.trader.latest_equity if self.trader.latest_equity > 0 else bal

        sl = float(dec.get("suggested_sl") or 0)
        tp = float(dec.get("suggested_tp") or 0)

        # ── SL ATR 倍数：三联动（市场模式 + 波动率 atr_ratio + AI 置信度）─────────
        # 联动规则：趋势市 base 高（2.0）+ 高波动放宽；震荡市 base 低（1.5）+ 低波动收紧
        ai_conf = dec.get("confidence", 0.5)
        _atr_ratio = (ind_15m or {}).get("atr_ratio", 1.0)
        _adapt_factor = CFG.sl_atr_adapt_factor  # 波动率敏感度，默认 0.5

        if market_mode == "震荡激进":
            # base=1.5，强信号（conf≥0.70）允许稍宽至 1.5×，普通信号收紧至 1.0×
            _base = 1.5 if ai_conf >= 0.70 else 1.0
            _floor, _cap = CFG.sl_atr_floor_osc_aggr, CFG.sl_atr_cap_osc_aggr
        elif market_mode == "震荡":
            _base = CFG.osc_sl_atr_mult
            _floor, _cap = CFG.sl_atr_floor_osc, CFG.sl_atr_cap_osc
        else:  # 趋势市
            _base = CFG.sl_atr_mult
            _floor, _cap = CFG.sl_atr_floor_trend, CFG.sl_atr_cap_trend

        # 波动率自适应：高波动（atr_ratio>1）放宽，低波动（atr_ratio<1）收紧
        _eff_sl_mult = _base * (1 + (_atr_ratio - 1.0) * _adapt_factor)
        _eff_sl_mult = max(_floor, min(_cap, _eff_sl_mult))

        if abs(_eff_sl_mult - _base) > 0.05:
            log.debug(f"📊 [{sym}] ATR自适应SL: {market_mode} ATR={_atr_ratio:.2f} "
                      f"base={_base} → eff={_eff_sl_mult:.2f}（floor={_floor} cap={_cap}）")

        # ── 杠杆动态加成：震荡激进 + 高置信 → 允许在建议杠杆基础上 +3 ───────────
        if market_mode == "震荡激进" and ai_conf >= 0.72:
            base_lev = int(dec.get("suggested_leverage") or CFG.max_leverage)
            boosted_lev = min(CFG.max_leverage, base_lev + 3)
            if boosted_lev > lev:
                log.info(f"📈 [{sym}] 震荡激进高置信杠杆加成: {lev}x → {boosted_lev}x（conf={ai_conf:.2f}）")
                lev = boosted_lev

        sl_dist_min = _eff_sl_mult * atr

        if sl <= 0:
            sl = price - sl_dist_min if is_long else price + sl_dist_min
        else:
            actual_dist = abs(price - sl)
            if actual_dist < atr:
                log.info(f"⚠️ AI建议SL距离={actual_dist:.2f} < 1×ATR={atr:.2f}，扩宽至{sl_dist_min:.2f}")
                sl = price - sl_dist_min if is_long else price + sl_dist_min

        dist = abs(price - sl)

        # SL 收紧：仲裁方向冲突时 sl_tighten_mult=0.85（SL距离缩小15%）
        _sl_tighten = dec.get("sl_tighten_mult", 1.0) if isinstance(dec, dict) else 1.0
        if _sl_tighten < 1.0:
            _tightened_dist = dist * _sl_tighten
            _tightened_sl = price - _tightened_dist if is_long else price + _tightened_dist
            log.info(f"⚖️ [{sym}] SL收紧: {dist:.4f}→{_tightened_dist:.4f}（×{_sl_tighten:.2f}），新SL={_tightened_sl:.4f}")
            dist = _tightened_dist
            sl = _tightened_sl

        # TP 动态盈亏比（震荡激进：1.2 快进快出；普通震荡：1.5；趋势：配置值）
        if market_mode == "震荡激进":
            tp_rr = 1.5
        elif market_mode == "震荡":
            tp_rr = 1.5
        else:
            tp_rr = CFG.tp_rr_ratio
        # 强制下限：AI 通过 L2 热更新可调低 RR，但永远不能低于 tp_rr_ratio_min
        tp_rr = max(tp_rr, CFG.tp_rr_ratio_min)
        tp_dist_min = tp_rr * dist
        if tp <= 0 or abs(price - tp) < tp_dist_min:
            tp = price + tp_dist_min if is_long else price - tp_dist_min

        # 资金竞争防护后的可用余额
        single_cap = equity_for_cap
        max_by_margin = int(effective_bal * lev / (price * ct_val))
        if max_by_margin < 1:
            min_lev = int(price * ct_val / effective_bal) + 1 if effective_bal > 0 else CFG.max_leverage + 1
            if min_lev <= CFG.max_leverage:
                log.warning(f"⚠️ [{sym}] AI建议杠杆{lev}x不足，自动升至{min_lev}x")
                lev = min_lev
                max_by_margin = int(effective_bal * lev / (price * ct_val))
            else:
                return {"skip": True, "reason": f"余额{effective_bal:.2f}不足"}

        # 保证金超限时自动升杠杆
        min_margin_1_lot = (1 * ct_val * price) / lev
        if min_margin_1_lot > equity_for_cap * CFG.max_margin_pct:
            required_lev = (1 * ct_val * price) / (equity_for_cap * CFG.max_margin_pct)
            auto_lev = min(CFG.max_leverage, max(lev, int(required_lev) + 1))
            if auto_lev > lev:
                log.info(f"📈 [{sym}] 保证金超限，自动升杠杆 {lev}x→{auto_lev}x")
                lev = auto_lev
                max_by_margin = int(effective_bal * lev / (price * ct_val))

        max_margin_budget = min(single_cap, equity_for_cap * CFG.max_margin_pct)
        max_size_by_margin_cap = max(int(max_margin_budget * lev / (ct_val * price + 1e-9)), 1)

        # 计算张数（使用 Kelly 公式优化仓位）
        # p_win 钳位 [0.45, 0.75]：LLM 置信度非统计概率，Bayesian likelihood=1.0 时
        # posterior 可达 1.0，直接代入 Kelly 会使 f→1.0 虽有 kelly_max_f 兜底但逻辑错误
        _raw_p = dec.get("posterior_confidence", dec.get("confidence", 0.5))
        p_win = max(0.45, min(0.75, _raw_p))
        # 盈亏比：TP距离 / SL距离（TP=0 时 Kelly=0，需给出警告）
        tp_dist = abs(tp - price) if tp > 0 else 0
        if tp_dist <= 0:
            log.warning(f"⚠️ [{sym}] TP 未有效设置（{tp:.4f}），Kelly 公式返回 0，仓位将被拒绝")
        b = tp_dist / max(dist, 1e-9)
        # Kelly 公式计算最优资金占比（直接决定风险预算比例）
        ai_conf = dec.get("confidence", 0.5)
        # ── 强信号动态 Kelly 上浮机制 ─────────────────────────────────────────
        # 条件：趋势市 conf≥0.75 或 震荡激进 conf≥0.70 或 快速决策 conf≥0.60
        is_strong = (
            (market_mode in ("趋势",) and ai_conf >= 0.75 and p_win >= 0.70)
            or (market_mode == "震荡激进" and ai_conf >= 0.70)
            or dec.get("is_ai", True) is False  # 快速决策（规则引擎触发）天然有技术确认
        )
        kelly_base = CFG.strong_signal_kelly_boost if is_strong else CFG.kelly_fraction
        if is_strong:
            log.debug(f"🚀 [{sym}] 强信号 Kelly 上浮: {CFG.kelly_fraction:.2f} → {kelly_base:.2f} (conf={ai_conf:.2f} p_win={p_win:.2f} mode={market_mode})")
        kelly_f = kelly_optimal_size(p_win, b, kelly_base)
        kelly_f = min(kelly_f, CFG.kelly_max_f)  # 应用 Kelly 上限
        # 委员会强烈反对时（≥2 票反对 且 AI conf < 0.7），Kelly 折扣 50%
        if committee_opposing >= 2 and ai_conf < 0.7:
            kelly_f *= 0.5
            log.warning(f"⚠️ [{sym}] 委员会反对{committee_opposing}票，Kelly 折扣 50% → kelly_f={kelly_f:.3f}")

        # 初始化默认值（满仓模式不走 Kelly，但仍需返回值有默认值）
        slippage_mult = 1.0
        risk_per_trade_dynamic = 1.0
        risk_budget = effective_bal

        if CFG.full_position_mode:
            # 满仓模式：直接用最大可用保证金，不走凯利（超高风险）
            # 风险检查：如果权益回撤已经很大，不允许满仓
            current_dd = gs_get("dd_kelly_mult", 1.0)
            if current_dd < 0.5:
                log.warning(f"⚠️ [{sym}] 满仓模式被拒：当前权益回撤较大（Kelly倍数={current_dd:.2f} < 0.5），为保护账户禁止开仓")
                return {"skip": True, "reason": f"回撤过大，满仓模式风险过高"}
            size = int(effective_bal * lev / (price * ct_val))
            size = max(1, min(size, max_by_margin, max_size_by_margin_cap))
            log.warning(f"🚀 [{sym}] 满仓梭哈模式激活（高风险）：张数={size}，余额={effective_bal:.2f}U，杠杆={lev}x")
        else:
            # 统一风险因子链：Kelly * risk_mult → 最终风险比例
            # Kelly 是天花板，risk_mult 只能降低 Kelly 结果，不能放大它
            # risk_mult 已在 _run_symbol 中合并（market * consec * dyn * pyramid）
            # 滑点熔断作为链中最后一环，追加到 risk_mult
            slippage_mult = 0.5 if self.okx.get_hourly_avg_slippage(sym) > CFG.slippage_fuse_avg_thresh else 1.0
            effective_risk_mult = min(risk_mult, 1.0)  # 禁止 risk_mult 放大 Kelly

            # ── 三因子联动 Kelly 倍数：胜率 × 回撤 × 市场模式 ─────────────────
            _win_rate  = gs_get("last_24h_win_rate", 0.5)
            _dd_factor = float(gs_get("dd_kelly_mult", 1.0))
            # wr_mult 和 mode_mult 手动联用（dd_factor 来自 risk_guard.py 每轮写入的 GS 值）
            _wr_mult   = (0.75 if _win_rate < 0.40
                          else 1.15 if _win_rate > 0.60
                          else 0.75 + (_win_rate - 0.40) / 0.20 * 0.40)
            _mode_mult = 1.12 if market_mode == "趋势" else 1.0
            _dd_mult   = max(0.18, min(1.25, _dd_factor * _wr_mult * _mode_mult))
            if _dd_mult < 1.0 or market_mode == "趋势":
                log.debug(f"📊 [{sym}] Kelly三因子: 胜率={_win_rate:.2f} 回撤={_dd_factor:.2f} 趋势={market_mode} → ×{_dd_mult:.2f}")

            # ── ConvictionScorer：软打分（取代硬阈值 AND 门）──────────────
            # 优先使用 AI 决策时预计算的 VSpike 分数（含方向检查 + bonus），直接复用不做重算
            _vs_score_mult = dec.get("_vs_score_mult_frozen", 0.0)
            # 检查过期时间（90s），避免使用已失效的 Spike 数据
            _vs_frozen_ts = dec.get("_vs_score_mult_frozen_ts", 0.0)
            if _vs_frozen_ts > 0 and (time.monotonic() - _vs_frozen_ts) > 90.0:
                _vs_score_mult = 0.0  # 过期清除

            _near_level  = False
            if ind_15m:
                _price_cs = ind_15m.get("price", price)
                _sup = ind_15m.get("support", 0)
                _res = ind_15m.get("resistance", 0)
                if _sup > 0 and abs(_price_cs - _sup) / _price_cs < 0.003:
                    _near_level = True
                if _res > 0 and abs(_price_cs - _res) / _price_cs < 0.003:
                    _near_level = True

            _conviction = self.trader._conviction.score(
                ai_conf      = ai_conf,
                action       = dec.get("action", "open_long"),
                vspike_mult  = _vs_score_mult,
                ob_imbalance = depth.get("imbalance", 0.0) if hasattr(depth, "get") else 0.0,
                rsi          = ind_15m.get("rsi", 50.0) if ind_15m else 50.0,
                at_key_level = _near_level,
                market_mode  = market_mode,
                context      = {
                    "atr_ratio": (ind_15m or {}).get("atr_ratio", 1.0),
                    "trend_alignment_score": trend_score,
                },
            )
            _cv_score      = _conviction["score"]
            _cv_kelly      = _conviction["kelly_ratio"]
            _cv_components = _conviction["components"]

            log.info(
                f"🎯 [{sym}] [Entry] Score={_cv_score} | "
                f"Components: AI({_cv_components.get('ai_raw', 0)}) + "
                f"Spike({_cv_components.get('spike', 0)}) + "
                f"OB({_cv_components.get('ob', 0)}) + "
                f"Levels({_cv_components.get('level', 0)}) + "
                f"RSI({_cv_components.get('rsi_penalty', 0)}) | "
                f"Kelly_Adj={_cv_kelly:.2f}"
            )

            # ── 方案 C：VSpike 条件性阈值 ─────────────────────────────────────
            # VSpike 越强，阈值越低，允许边缘信号在高成交量确认时下注
            _base_thresh = CFG.osc_conviction_min if market_mode in ("震荡", "震荡激进") else CFG.conviction_open_min
            if _vs_score_mult >= 6.0:
                _cv_thresh = max(35.0, _base_thresh - 12.0)  # VSpike≥6x：阈值 -12
            elif _vs_score_mult >= 4.0:
                _cv_thresh = max(38.0, _base_thresh - 8.0)   # VSpike≥4x：阈值 -8
            elif _vs_score_mult >= 3.0:
                _cv_thresh = max(40.0, _base_thresh - 4.0)   # VSpike≥3x：阈值 -4
            else:
                _cv_thresh = _base_thresh                     # 无 VSpike：基准阈值
            # ────────────────────────────────────────────────────────────────

            if _cv_score < _cv_thresh:
                log.info(f"🚫 [{sym}] ConvictionScore={_cv_score:.1f} < {_cv_thresh}（{'震荡' if market_mode in ('震荡', '震荡激进') else '趋势'}市），跳过开仓")
                # 记录 ConvictionScore 拒绝的决策，供 _do_open 传递到 _run_symbol
                self._cv_rejected = True
                self._cv_rejected_score = _cv_score
                self._cv_rejected_action = dec.get("action", "")
                return {"skip": True, "size": 0, "sl": sl, "tp": tp, "risk_per_trade_dynamic": 0.0,
                        "calc": {"kelly_p_win": p_win, "kelly_b": b, "kelly_f": kelly_f,
                                 "kelly_risk_mult": risk_mult, "conviction_score": _cv_score,
                                 "kelly_slippage_mult": slippage_mult,
                                 "risk_per_trade_dynamic": 0.0, "kelly_risk_budget": 0,
                                 "posterior_confidence": p_win}}

            # ── 去重复惩罚：ConvictionScore 的 _arbitrate_final_score 已包含
            #    env_mult(trend_score+ATR) 和 risk_mult(回撤×胜率×连亏)，
            #    _dd_mult 同样包含回撤×胜率×市场模式，双重惩罚导致仓位被压缩到 Kelly 的 13%
            #    简化：Kelly × risk_mult × slippage，ConvictionScore 质量已通过阈值门控体现
            final_risk_mult = kelly_f * effective_risk_mult * slippage_mult
            # fast_decision BypassLane：bypass时用旁路kelly替代
            _bypass_k = getattr(self.trader, '_bypass_kelly_override', None)
            if _bypass_k is not None:
                final_risk_mult = kelly_f * effective_risk_mult * slippage_mult * _bypass_k
                delattr(self.trader, '_bypass_kelly_override')
                log.debug(f"⚡ [{sym}] BypassLane kelly override: {_bypass_k:.2f}")
            risk_per_trade_dynamic = final_risk_mult
            # ── 软拦截降仓位（近关键位降权 + SL 收紧，同步生效）────────────────
            _near_mult = dec.get("near_level_mult", 1.0) if isinstance(dec, dict) else 1.0
            if _near_mult < 1.0:
                _orig_budget = risk_budget
                risk_per_trade_dynamic *= _near_mult
                risk_budget = effective_bal * risk_per_trade_dynamic
                log.info(f"🚧 [{sym}] 软拦截降仓位: ×{_near_mult:.2f} "
                          f"→ risk_per_trade={risk_per_trade_dynamic:.4f} "
                          f"budget={risk_budget:.2f}U（原={_orig_budget:.2f}U）")
            # ── 每日风险统计（仅记录，不拦截）────────────────────────────────
            today_risk = float(gs_get("today_opened_risk", 0.0))
            daily_cap  = effective_bal * CFG.max_daily_risk_pct
            log.debug(f"📊 [{sym}] 今日已用风险: {today_risk:.2f}U / {daily_cap:.2f}U（统计用，不拦截）")
            log.debug(f"🎯 [{sym}] Kelly链: p_win={p_win:.3f} b={b:.2f} kelly_f={kelly_f:.3f} risk_mult={risk_mult:.3f} slippage_mult={slippage_mult:.1f} final_risk={final_risk_mult:.4f} budget={risk_budget:.2f}U")
            raw_size = risk_budget / (dist * ct_val + 1e-9)
            if raw_size < 1.0:
                overshoot_pct = (dist * ct_val - risk_budget) / (effective_bal + 1e-9) * 100
                log.warning(
                    f"⚠️ [{sym}] 止损距离过宽：理论张数={raw_size:.2f}<1张，"
                    f"强制1张将使实际风险超出预算约{overshoot_pct:.1f}%"
                )
            size_by_risk = max(int(raw_size), 1)
            size = min(size_by_risk, max_by_margin, max_size_by_margin_cap)

        if size > max_size_by_margin_cap:
            log.info(f"📉 [{sym}] 仓位上限约束：{size}→{max_size_by_margin_cap}张")
            size = max_size_by_margin_cap

        # 精度截断
        if lot_sz > 0:
            lots = math.floor(size / lot_sz + 1e-9)
            lots = max(lots, 1)
            size = lots * lot_sz
            size_str = str(int(size)) if lot_sz >= 1 else f"{size:.8f}".rstrip('0').rstrip('.')
        else:
            lots = max(int(size), 1)
            size = float(lots)
            size_str = str(lots)

        notional = size * ct_val * price
        # 逐仓模式保证金 = 名义价值 / 杠杆（OKX 逐仓保证金率约 1/lev，维持保证金率 900%-1500%）
        required_margin = notional / lev

        # 总保证金使用率熔断：超限时按上限裁剪仓位，而非直接拒绝
        current_total_margin = 0.0
        p = self.trader.pos
        if p.side and p.entry_price > 0 and p.size > 0:
            px = self.trader._get_price(CFG.symbol)
            if px > 0:
                cv = self.okx.contract_sizes.get(CFG.symbol, 0.01)
                nom = p.size * cv * px
                lv = p.leverage if p.leverage > 0 else 1
                current_total_margin = nom / lv
        equity_for_margin = self.trader.latest_equity if self.trader.latest_equity > 0 else bal
        total_margin_if_open = current_total_margin + required_margin
        margin_ratio = total_margin_if_open / equity_for_margin if equity_for_margin > 0 else 1.0
        if margin_ratio > CFG.max_total_margin_ratio:
            # 计算允许的最大名义值 → 最大张数
            max_allowed_margin = equity_for_margin * CFG.max_total_margin_ratio - current_total_margin
            max_allowed_notional = max_allowed_margin * lev
            max_allowed_size_raw = max_allowed_notional / (price * ct_val + 1e-9)
            max_allowed_size = max(1, int(max_allowed_size_raw))
            size_before = size
            size = min(size, max_allowed_size)
            # 重新计算 notional / required_margin（裁剪后）
            notional = size * ct_val * price
            required_margin = notional / lev
            log.warning(
                f"⚠️ [{sym}] 总保证金使用率熔断：当前 {current_total_margin:.2f}U + 本次 {required_margin:.2f}U "
                f"占比 {margin_ratio*100:.1f}% > {CFG.max_total_margin_ratio*100:.0f}% "
                f"→ 裁剪仓位 {size_before}→{size}张（上限{max_allowed_size}）"
            )

        if notional < CFG.min_order_notional:
            log.warning(f"⚠️ [{sym}] 名义价值{notional:.2f} < {CFG.min_order_notional}，跳过")
            return {"skip": True}

        margin_usage_pct = required_margin / (equity_for_cap + 1e-9) * 100
        dyn_slippage = self.okx.get_dynamic_slippage(sym)
        hourly_slippage = self.okx.get_hourly_avg_slippage(sym)
        use_limit = hourly_slippage > CFG.slippage_fuse_avg_thresh
        # 滑点熔断已纳入统一 Kelly 链（slippage_mult），此处仅处理下单价格和订单类型
        if use_limit:
            log.warning(f"⚠️ [{sym}] 过去1小时平均滑点 {hourly_slippage*100:.3f}% 过高，已在 Kelly 链中压缩仓位")

        px = price * (1 + dyn_slippage) if is_long else price * (1 - dyn_slippage)
        if use_limit:
            px = price * (1 + dyn_slippage * 0.5) if is_long else price * (1 - dyn_slippage * 0.5)
        ord_type = "limit" if use_limit else "market"

        liq_price_est = calc_liq_price("long" if is_long else "short", price, lev, bal, size, ct_val)

        log.info(
            f"📐 [{sym}] 张数计算: 原始={size_by_risk}张 "
            f"lotSz截断后={size}张({size_str}) ctVal={ct_val} 名义={notional:.2f}U"
        )

        return {
            "skip": False,
            "size": size, "size_str": size_str, "notional": notional,
            "required_margin": required_margin, "lev": lev,
            "sl": sl, "tp": tp, "dist": dist,
            "px": px, "ord_type": ord_type,
            "liq_price_est": liq_price_est,
            "margin_usage_pct": margin_usage_pct,
            "risk_per_trade_dynamic": risk_per_trade_dynamic,
            "dyn_slippage": dyn_slippage,
            "sl_dist_min": sl_dist_min,
            "ct_val": ct_val, "lot_sz": lot_sz,
            "atr": atr,  # 用于 log_kelly_metrics 计算预估滑点
            # Kelly 监控指标
            "kelly_p_win": p_win,
            "kelly_b": b,
            "kelly_f": kelly_f,
            "kelly_risk_mult": risk_mult,
            "kelly_slippage_mult": slippage_mult,
            "kelly_risk_budget": risk_budget,
            "kelly_committee_opposing": committee_opposing,
            "posterior_confidence": p_win,  # 贝叶斯后验胜率，用于日志记录
        }

    def _pre_reserve(self, sym: str, required_margin: float) -> float:
        """
        在 _margin_lock 保护下预留保证金。
        返回 _margin_to_release（= required_margin），供异常时释放。
        """
        with self.trader._margin_lock:
            self.trader._reserved_margin += required_margin
        return required_margin

    def _place_order_phase(self, sym: str, is_long: bool, lev: int, size: float,
                           size_str: str, price: float, sl: float, tp: float,
                           ct_val: float, lot_sz: float, dyn_slippage: float,
                           decision_id: int, ord_type: str = "market") -> tuple:
        """
        发送开仓订单（限价/市价），返回 (ord_id, algo_ids, res)。
        成交后由 _post_fill_sync 处理仓位同步。
        """
        posSide = "long" if is_long else "short"
        side = "buy" if is_long else "sell"

        # 流动性感知下单
        ct_val_for_liq = self.okx.contract_sizes.get(sym, 0.01)
        try:
            book_check = self.okx.get_orderbook(sz=20, symbol=sym)
            avail_levels = book_check["asks"] if is_long else book_check["bids"]
            avail_qty = sum(s for _, s in avail_levels)
            if avail_qty > 0 and size > avail_qty:
                clipped_raw = avail_qty * 0.8
                # 确保是 lot_sz 的整数倍（向下取整）
                clipped = max(lot_sz, int(clipped_raw / lot_sz) * lot_sz)
                clipped = max(lot_sz, clipped)  # 至少1个lot
                log.warning(
                    f"⚠️ [{sym}] 流动性不足！需求 {size}张，盘口可吃 {avail_qty:.0f}张，"
                    f"压缩至 {clipped}张（80%安全边际）"
                )
                size = clipped
                size_str = str(int(size)) if lot_sz >= 1 else f"{size:.8f}".rstrip('0').rstrip('.')
                extra_slippage = 0.5 * dyn_slippage
                price = price * (1 + dyn_slippage + extra_slippage) if is_long                         else price * (1 - dyn_slippage - extra_slippage)
                tick = self.okx.tick_sizes.get(sym, 0.01)
                price = math.floor(price / tick + 1e-9) * tick
                log.info(f"📐 [{sym}] 收紧限价至 {price:.4f}，size_str={size_str}")
        except Exception as e:
            log.debug(f"流动性检查失败（非致命，继续下单）: {e}")

        self.okx.set_leverage(lev, symbol=sym, posSide=posSide)
        res = self.okx.place_order(side, posSide, size_str, price, sl, tp, symbol=sym, ord_type=ord_type)
        if res.get("code") != "0":
            log.error(f"❌ 开仓失败: {res}")
            log_event("order_failed", {"reason": res, "decision_id": decision_id})
            raise OrderFailedError(res)

        data_list = res.get("data") or []
        if not data_list:
            log.error(f"❌ 开仓返回空data（code=0但无订单）: {res}")
            raise OrderFailedError({"code": "-1", "msg": "empty data"})
        ord_id = data_list[0].get("ordId", "")
        # 捕获附加 SL/TP 算法单的 algoId（用于 WS 成交检测）
        algo_ids = []
        for item in data_list:
            _aid = item.get("algoId", "")
            if _aid:
                algo_ids.append(_aid)

        # 滑点日志关联交易记录（expected vs actual fill price）
        fill_px = float(data_list[0].get("avgPx") or price)
        filled_sz = float(data_list[0].get("accFillSz") or 0)
        if filled_sz > 0:
            slip = abs(fill_px - price) / price if price > 0 else 0.0
            log_slippage(
                side=side,
                expected_px=price,
                fill_px=fill_px,
                size=filled_sz,
                slippage_pct=slip,
                decision_id=decision_id,
            )

        return ord_id, algo_ids, res

    def _post_fill_sync(self, sym: str, ord_id: str, posSide: str, size: float,
                         price: float, sl: float, tp: float, lev: int,
                         liq_price_est: float, decision_id: int, required_margin: float,
                         pyramid_plan: dict = None, initial_risk_usd: float = 0.0,
                         ind_15m: Dict = None, dec: Dict = None, market_mode: str = None,
                         algo_ids: list = None):
        """
        订单提交成功后：持久化 pending_order、更新 pos 状态、写 trade 记录。
        pyramid_plan / initial_risk_usd：有加仓计划时写入 pos，供后续加仓执行使用。
        ind_15m / dec / market_mode：开仓时的市场快照，供动态案例池使用。
        algo_ids：SL/TP 算法单 algoId 列表，供 WS 成交检测使用。
        """
        side = "buy" if posSide == "long" else "sell"
        lock = self.trader.lock
        pos = self.trader.pos

        # ── 动态案例池：提取开仓时的技术指标快照 ──────────────────────────
        _entry_rsi = ind_15m.get("rsi", 0.0) if ind_15m else 0.0
        _entry_bb  = ind_15m.get("bb_pct", 0.5) if ind_15m else 0.5
        _entry_atr = ind_15m.get("atr", 0.0) if ind_15m else 0.0
        _ai_conf   = dec.get("confidence", 0.0) if dec else 0.0
        _ai_reason = dec.get("reason", "") if dec else ""
        _mkt_mode  = market_mode or "趋势"
        # 计算入场 ATR 分位
        _entry_atr_pct = _get_atr_quantile(_entry_atr, self.trader._atr_history) if _entry_atr > 0 else 0.5

        save_pending_order(
            ord_id=ord_id, side=side, pos_side=posSide, size=size,
            entry_price=price, sl=sl, tp=tp, leverage=lev,
            liq_price=liq_price_est, decision_id=decision_id,
            symbol=sym, margin=float(required_margin)
        )

        with lock:
            pos.pending_ord_id  = ord_id
            pos.last_open_time  = datetime.now(UTC)
            pos.partial_filled       = 0.0
            pos.liq_price            = liq_price_est
            pos.leverage             = lev
            pos.stop_loss            = sl
            pos.take_profit          = tp
            pos.breakeven_triggered       = False
            pos.partial_tp_triggered      = False
            pos.partial_tp_2_5R_triggered  = False
            # 金字塔加仓计划状态初始化
            pos.pyramid_plan      = pyramid_plan
            pos.pyramid_count     = 0
            pos.initial_size      = float(size) if pyramid_plan else 0.0
            # 修复：所有交易均记录开仓风险额（不限于金字塔），用于精确释放 today_opened_risk
            # 若 initial_risk_usd=0（非金字塔路径），此处补算，防止关仓时用 sl_snapshot 错误释放
            _fallback_ct_init = self.okx.contract_sizes.get(sym, 0.01)
            _risk_for_pos = initial_risk_usd if initial_risk_usd > 0 else (
                float(size) * abs(sl - price) * _fallback_ct_init if (sl > 0 and price > 0) else 0.0
            )
            pos.initial_risk_usd  = _risk_for_pos
            # 动态案例池：记录开仓时的市场快照
            pos.entry_market_mode = _mkt_mode
            pos.entry_rsi        = _entry_rsi
            pos.entry_bb_pct     = _entry_bb
            pos.entry_atr_pct    = _entry_atr_pct
            pos.ai_confidence    = _ai_conf
            pos.ai_reason        = _ai_reason
            save_state_to_disk(pos)

        # ── 每日风险上限累计 ─────────────────────────────────────────────
        _fallback_ct = self.okx.contract_sizes.get(sym, 0.01)
        _opened_risk = initial_risk_usd if initial_risk_usd > 0 else (float(size) * abs(sl - price) * _fallback_ct if (sl > 0 and price > 0) else 0.0)
        if _opened_risk > 0:
            _ = gs_add("today_opened_risk", _opened_risk)

        trade_id = save_trade_open(decision_id, posSide, size, price, symbol=sym)
        # 动态案例池：将 entry 市场快照写入 trades 表（用于后续案例生成）
        if trade_id and CFG.enable_auto_case_pool:
            try:
                conn = get_db_conn()
                c = conn.cursor()
                c.execute(
                    "UPDATE trades SET entry_market_mode=?, entry_rsi=?, entry_bb_pct=?, "
                    "entry_atr_pct=?, ai_confidence=?, ai_reason=? WHERE id=?",
                    (_mkt_mode, _entry_rsi, _entry_bb, _entry_atr_pct,
                     _ai_conf, (_ai_reason or "")[:300], trade_id)
                )
                conn.commit()
            except Exception as e:
                log.debug(f"[案例池] 开仓快照写入失败: {e}")

        with lock:
            pos.trade_id = trade_id
            pos.sl_tp_algo_ids = list(algo_ids) if algo_ids else []
            pos.ai_conf_at_open = (dec.get("confidence", 0.0) if dec else 0.0)

        # Phase 3: 状态机推进 → HOLDING
        if hasattr(self.trader, '_state_machine'):
            self.trader._state_machine.transition(TradingState.HOLDING, "开仓确认完成")
        # Phase 4: EventBus 开仓通知（原代码无此通知，新增）
        if hasattr(self.trader, '_event_bus'):
            self.trader._event_bus.publish("trade_open", {
                "sym": sym, "side": posSide, "size": size,
                "entry": price, "sl": sl, "tp": tp, "lev": lev,
                "liq_price": liq_price_est,
            })
        log.debug(f"📋 限价单已提交 ordId={ord_id}，预估强平价={liq_price_est:.2f}，由_handle_pending_order异步处理")

    def _apply_pyramid_plan(self, decision: Dict, action: str, conf: float) -> Optional[dict]:
        """
        提取金字塔加仓计划（仅使用 AI 显式输出的计划，不再自动生成 fallback）。
        - AI 已输出 pyramid_plan → 直接使用
        - AI 未输出 → None（不加仓，由 AI 全权决定是否加仓）
        """
        plan = decision.get("pyramid_plan")
        if plan:
            return plan
        return None

    def _apply_market_mode_adjustments(self, market_mode: str, conf: float) -> tuple:
        """
        根据市场模式返回 (market_factor, level_proximity_thresh)。
        集中所有模式相关调整逻辑，避免散落在 _run_symbol 中。
        """
        market_factor = 1.0
        if market_mode in ("震荡", "震荡激进"):
            base_ratio = CFG.osc_risk_ratio * (0.9 if market_mode == "震荡激进" else 1.0)
            if conf >= 0.75:
                market_factor = 1.0
            elif conf >= 0.70:
                market_factor = 0.9
            else:
                market_factor = base_ratio
        market_factor = max(0.3, min(1.5, market_factor))

        level_thresh = (
            CFG.osc_level_proximity
            if market_mode in ("震荡", "震荡激进")
            else CFG.level_proximity_thresh
        )
        return market_factor, level_thresh

    def _do_open(self, dec: Dict, price: float, bal: float, atr: float, funding: Dict,
                 decision_id: int, risk_mult: float = 1.0, symbol: str = None,
                 market_mode: str = None, ind_15m: Dict = None, pyramid_plan: dict = None,
                 committee_opposing: int = 0, depth: Dict = None,
                 trend_score: float = 0.5):
        """
        三阶段锁策略协调器：[锁内]计算+预留 → [无锁]下单 → [锁内]同步状态
        不含任何具体计算逻辑，仅做流程编排 + 异常兜底。
        risk_mult: 统一风险因子链（Kelly * market * consec * dyn * pyramid），已在 _run_symbol 中合并完毕
        pyramid_plan: AI 制定的加仓计划（None = 不加仓）
        committee_opposing: 策略委员会反对票数（≥2 且 AI conf<0.7 时 Kelly 折扣 50%）
        """
        sym = CFG.symbol
        is_long = "long" in dec.get("action", "")
        posSide = "long" if is_long else "short"
        _margin_to_release = 0.0
        # 每次 _do_open 调用时重置 ConvictionScore 拒绝标志
        self._cv_rejected = False
        self._cv_rejected_score = 0.0
        self._cv_rejected_action = ""

        try:
            # ── 阶段1：全局开仓锁，纯内存计算 ───────────────────────────────
            with self.trader._open_lock:
                # 标记价偏离熔断（网络 I/O 前的快速卫兵）
                mark_px = self.trader._mark_price_val
                if mark_px > 0 and price > 0:
                    deviation = abs(price - mark_px) / mark_px
                    if deviation > CFG.mark_price_deviation_thresh:
                        log.warning(
                            f"⛔ [{sym}] 开仓熔断：最新价 {price:.4f} 与标记价 {mark_px:.4f} "
                            f"偏离 {deviation*100:.2f}%，暂缓开仓"
                        )
                        log_event("open_blocked_deviation", {"symbol": sym, "price": price,
                            "mark_px": mark_px, "deviation_pct": deviation * 100})
                        return

                calc = self._calc_size_and_margin(
                    dec, price, bal, atr, is_long, sym, market_mode or self.trader._market_mode, ind_15m,
                    risk_mult,  # 统一风险因子链（已在 _run_symbol 中合并 market*consec*dyn*pyramid）
                    committee_opposing=committee_opposing,
                    action=dec.get("action", ""),
                    depth=depth,
                    trend_score=trend_score,
                )
                if calc.get("skip"):
                    # ConvictionScore 拒绝：将拒绝信息传递给 trader，供 _run_symbol 处理
                    if self._cv_rejected:
                        self.trader._cv_rejected_decision = (
                            self._cv_rejected_action,
                            self._cv_rejected_score,
                            time.monotonic(),
                        )
                    return

                _margin_to_release = self._pre_reserve(sym, calc["required_margin"])

                # Kelly 指标结构化日志
                log_kelly_metrics(
                    sym=sym,
                    p_win=calc.get("kelly_p_win", 0.5),
                    b=calc.get("kelly_b", 0),
                    kelly_f=calc.get("kelly_f", 0),
                    risk_mult=calc.get("kelly_risk_mult", 1.0),
                    slippage_mult=calc.get("kelly_slippage_mult", 1.0),
                    final_risk=calc.get("risk_per_trade_dynamic", 0),
                    risk_budget=calc.get("kelly_risk_budget", 0),
                    size=calc["size"],
                    price=price,
                    decision_id=decision_id,
                    committee_opposing=calc.get("kelly_committee_opposing", 0),
                    market_mode=market_mode or "",
                    posterior_confidence=calc.get("posterior_confidence", 0.5),
                    kelly_fraction=CFG.kelly_fraction,
                    atr=calc.get("atr", 0.0),
                )

            # ── 阶段2：无锁 API 调用（风控线程可并发抢锁）────────────────────
            log.info(
                f"🚀 [{sym}] 开仓 {posSide} x{calc['lev']} | "
                f"{calc['size']}张/{calc['size'] * calc['ct_val']:.4f} | "
                f"px={calc['px']:.4f}({calc['ord_type']}) "
                f"sl={calc['sl']:.4f}(×{calc['dist'] / max(atr, 1e-9):.1f}ATR) "
                f"tp={calc['tp']:.4f}(RR={abs(calc['tp'] - price) / max(calc['dist'], 1e-9):.1f}x) "
                f"名义={calc['notional']:.1f}U 保证金={calc['required_margin']:.2f}U"
                f"({calc['margin_usage_pct']:.1f}%eq) "
                f"风险={calc['risk_per_trade_dynamic'] * 100:.2f}% "
                f"滑点系数={calc['dyn_slippage'] * 100:.3f}%"
            )
            log_event("order_attempt", {
                "symbol": sym, "side": posSide, "lev": calc["lev"],
                "size": calc["size"], "px": calc["px"],
                "sl": calc["sl"], "tp": calc["tp"], "decision_id": decision_id,
            })

            ord_id, algo_ids, _ = self._place_order_phase(
                sym, is_long, calc["lev"], calc["size"], calc["size_str"], price,
                calc["sl"], calc["tp"], calc["ct_val"], calc["lot_sz"],
                calc["dyn_slippage"], decision_id, ord_type=calc["ord_type"]
            )

            # ── 阶段3：持久化 + 更新内存状态（重新加锁）─────────────────────
            _initial_risk_usd = calc["size"] * calc["dist"] * calc["ct_val"]
            self._post_fill_sync(
                sym, ord_id, posSide, calc["size"], price,
                calc["sl"], calc["tp"], calc["lev"],
                calc["liq_price_est"], decision_id, calc["required_margin"],
                pyramid_plan=pyramid_plan, initial_risk_usd=_initial_risk_usd,
                ind_15m=ind_15m, dec=dec, market_mode=market_mode,
                algo_ids=algo_ids,
            )

        except Exception as e:
            log.exception(f"❌ [{sym}] 开仓异常: {e}")
            if _margin_to_release > 0:
                with self.trader._margin_lock:
                    self.trader._reserved_margin = max(0.0, self.trader._reserved_margin - _margin_to_release)
                log.info(f"🔓 [{sym}] 异常退出，释放预留保证金 {_margin_to_release:.2f}U")

    def _close(self, reason: str, decision_id: Optional[int] = None, symbol: str = None):
        """
        与 _do_open 相同的三阶段锁策略。
        原版在持锁期间调用 close_position API，导致 API 超时或 rate-limit
        backoff 时风控线程被阻塞。

        阶段一（加锁）：读取当前仓位快照，立即释放锁。
        阶段二（无锁）：发送平仓 API 请求。
        阶段三（加锁）：写回平仓后的状态。
        """
        # ── 解析品种 ──────────────────────────────────────────────────────
        sym = CFG.symbol
        pos  = self.trader.pos
        lock = self.trader.lock

        # ── 阶段一：加锁，捕获快照 ──────────────────────────────────────────
        with lock:
            if not pos.side:
                return
            side_snapshot      = pos.side
            entry_snapshot     = pos.entry_price
            size_snapshot      = pos.size
            leverage_snapshot  = pos.leverage if pos.leverage > 0 else 1
            sl_snapshot        = pos.stop_loss
            # 动态案例池：读取入场快照（平仓时写入 exit 指标）
            _entry_mode_snap = pos.entry_market_mode
            _entry_rsi_snap  = pos.entry_rsi
            _entry_bb_snap   = pos.entry_bb_pct
            _entry_atr_snap  = pos.entry_atr_pct
            _ai_conf_snap    = pos.ai_confidence
            _ai_reason_snap  = pos.ai_reason
        # ── 锁释放 ──────────────────────────────────────────────────────────

        log.info(f"🔔 触发平仓: {reason}")

        # ── 阶段二：无锁 API 调用 ────────────────────────────────────────────
        res = self.okx.close_position(side_snapshot)

        if res.get("code") == "0":
            exit_price = self.trader._get_price(sym)   # 修复：必须传 sym，否则多品种时取错价格
            if exit_price <= 0:
                exit_price = self.okx.get_current_price()
            if side_snapshot == "long":
                pnl = exit_price - entry_snapshot
            else:
                pnl = entry_snapshot - exit_price
            pnl_pct = (pnl / entry_snapshot) * leverage_snapshot if entry_snapshot > 0 else 0.0

            if pnl < 0:
                n = gs_increment("consecutive_losses")
                log.warning(f"📉 [{sym}] 连续亏损 {n} 次（本次亏损 {pnl_pct*100:.2f}%）")
                if n >= CFG.max_consecutive_loss:
                    log.warning(f"⚠️ 已达到连续亏损上限 {CFG.max_consecutive_loss}，后续交易将降杠杆")
                # ── 止损上下文：保存关键信息供 AI 后续决策参考 ─────────────────
                _stop_dir = "long" if side_snapshot in ("buy", "long") else "short"
                _stop_reason = reason if reason else "未知"
                _stop_mode = _entry_mode_snap or "趋势"
                gs_set("last_stop_time", datetime.now(UTC).isoformat())
                gs_set("last_stop_direction", _stop_dir)
                gs_set("last_stop_pnl_pct", pnl_pct)
                gs_set("last_stop_reason", _stop_reason)
                gs_set("last_stop_market_mode", _stop_mode)
                gs_set("last_stop_price", sl_snapshot if sl_snapshot > 0 else exit_price)
                # 异步生成失败原因摘要（RAG闭环）
                _trade_id_for_fail = pos.trade_id or 0
                ctx = (
                    f"方向:{side_snapshot} 入场:{entry_snapshot:.2f} 出场:{exit_price:.2f} "
                    f"盈亏:{pnl_pct*100:+.2f}% 平仓原因:{reason}"
                )
                generate_fail_reason_async(
                    trade_id=_trade_id_for_fail,
                    ai_client=self.trader.ai.client,
                    context=ctx,
                )
            else:
                gs_set("consecutive_losses", 0)

            # Bug C 修复：先写 DB 再清零 trade_id，若 DB 写失败保留 trade_id 以便重试
            # （原版在锁内同时清零，DB 异常时 trade_id 已丢失且 _reset_pos 不会执行）
            with lock:
                current_trade_id = pos.trade_id
            if current_trade_id:
                try:
                    update_trade_close(current_trade_id, exit_price, close_reason=reason, leverage=leverage_snapshot)
                    with lock:
                        if pos.trade_id == current_trade_id:  # 防竞态
                            pos.trade_id = None
                except Exception as db_e:
                    log.error(f"❌ [{sym}] 平仓 DB 记录写入失败 trade_id={current_trade_id}: {db_e}，trade_id 保留以便重试")

            # 累计今日已实现盈亏
            pnl_usdt = pnl * size_snapshot * self.okx.contract_sizes.get(sym, 0.01)
            gs_add("today_realized_pnl", pnl_usdt)

            # 释放今日已用风险额度（平仓成功后归还，使额度可在当日内复用）
            # 修复：优先用 pos.initial_risk_usd（开仓时精确记录），
            # 防止追踪止损移动SL后 sl_snapshot≈entry 导致释放量接近0（幻影积累 bug）
            _ct = self.okx.contract_sizes.get(sym, 0.01)
            _risk_to_release = (
                pos.initial_risk_usd if pos.initial_risk_usd > 0
                else abs(sl_snapshot - entry_snapshot) * size_snapshot * _ct
            )
            if _risk_to_release > 0:
                remaining_risk = gs_add("today_opened_risk", -_risk_to_release)
                log.debug(f"♻️ [{sym}] 释放风险额度 {_risk_to_release:.2f}U（来源={'pos.initial_risk_usd' if pos.initial_risk_usd > 0 else 'sl_snapshot'}），今日剩余已用: {max(0.0, remaining_risk):.2f}U")

            # 动态案例池：提前计算持仓时长（在锁外，用 entry_snapshot_time 推算）
            # pos.open_time 在锁保护下读取（锁外读取也能接受，差异秒级）
            _exit_ts = datetime.now(UTC)
            try:
                _entry_ts = pos.open_time  # datetime，锁外读取（竞态容忍：秒级误差）
                _hold_minutes = int((_exit_ts - _entry_ts).total_seconds() / 60) if _entry_ts else 0
            except Exception:
                _hold_minutes = 0

            # ── 动态案例池：捕获 exit 指标快照（lock-free，直接读取缓存，无需额外 API） ──
            _exit_rsi     = 0.0
            _exit_bb      = 0.5
            _exit_atr_pct = 0.5
            _exit_mode    = self.trader._market_mode or "趋势"
            try:
                _cached = self.trader._ind_15m_cache[1] if self.trader._ind_15m_cache else {}
                if _cached and _cached.get("_valid"):
                    _exit_rsi = _cached.get("rsi", 0.0)
                    _exit_bb  = _cached.get("bb_pct", 0.5)
                _exit_atr_pct = _get_atr_quantile(self.trader._atr_val, self.trader._atr_history)
            except Exception:
                pass

            # 动态案例池：写入 exit 指标到 DB + 异步触发 AI 质量评估
            if CFG.enable_auto_case_pool and current_trade_id:
                try:
                    conn = get_db_conn()
                    c = conn.cursor()
                    c.execute(
                        "UPDATE trades SET exit_market_mode=? WHERE id=?",
                        (_exit_mode, current_trade_id)
                    )
                    conn.commit()
                except Exception as e:
                    log.debug(f"[案例池] 平仓快照写入失败: {e}")
                _auto_generate_historical_case(
                    trade_id=current_trade_id,
                    ai_client=self.trader.ai.client,
                    pos_snapshot={
                        "entry_market_mode": _entry_mode_snap,
                        "entry_rsi":        _entry_rsi_snap,
                        "entry_bb_pct":     _entry_bb_snap,
                        "entry_atr_pct":    _entry_atr_snap,
                        "exit_market_mode": _exit_mode,
                        "exit_rsi":         _exit_rsi,
                        "exit_bb_pct":      _exit_bb,
                        "exit_atr_pct":     _exit_atr_pct,
                        "ai_confidence":    _ai_conf_snap,
                        "ai_reason":        _ai_reason_snap,
                        "pnl_pct":         pnl_pct,
                        "pnl_usd":         pnl_usdt,
                        "entry_price":     entry_snapshot,
                        "exit_price":      exit_price,
                        "direction":       "long" if side_snapshot == "buy" else "short",
                        "hold_minutes":    _hold_minutes,
                        "symbol":          sym,
                    }
                )

            log.info(f"✅ [{sym}] 平仓成功 | 原因: {reason} | 盈亏: {pnl_pct*100:.2f}%")
            # ── AI 表现追踪：仅对AI真正开口(conf>0.5)的已平仓交易更新胜率 ──
            _ai_conf_open = pos.ai_conf_at_open if hasattr(pos, "ai_conf_at_open") else 0.0
            if _ai_conf_open > 0.5:
                self.trader._update_ai_performance(is_win=(pnl_pct > 0), ai_conf_at_entry=_ai_conf_open)
                log.debug(f"[AI绩效] 平仓更新: conf_at_open={_ai_conf_open:.2f} pnl={pnl_pct*100:.2f}%")
            # Phase 3: 状态机推进 → EXITING
            if hasattr(self.trader, '_state_machine'):
                self.trader._state_machine.transition(TradingState.EXITING, f"平仓订单已提交: {reason}")
            self.trader._event_bus.publish("trade_close", {
                "sym": sym, "reason": reason, "pnl_usdt": pnl_usdt,
            })
            log_event("position_closed", {
                "symbol": sym, "reason": reason, "side": side_snapshot,
                "entry": entry_snapshot, "exit": exit_price,
                "size": size_snapshot, "pnl_pct": pnl_pct,
                "decision_id": decision_id
            })

            with lock:
                # 动态案例池：将 exit 指标写入 position 对象（持久化保存）
                pos.exit_market_mode = _exit_mode
                pos.exit_rsi         = _exit_rsi
                pos.exit_bb_pct      = _exit_bb
                pos.exit_atr_pct     = _exit_atr_pct
                save_state_to_disk(pos)

            # ── 复盘频率优化：仅严重亏损(≥硬止损)或大盈利(>5%)时触发AI分析 ───
            # 其他情况（追踪止损/分批止盈/小幅盈亏）只记录基础数据，不消耗AI Token
            if pnl_pct <= -CFG.hard_stop_loss_pct or pnl_pct > 0.05:
                self.trader.postmortem.trigger({
                    "trade_id":        current_trade_id,
                    "decision_id":     decision_id,
                    "side":            side_snapshot,
                    "entry":           entry_snapshot,
                    "exit":            exit_price,
                    "pnl_pct":         pnl_pct,
                    "close_reason":    reason,
                    "close_ts":        datetime.now(UTC).isoformat(),
                    "holding_minutes": _hold_minutes,
                })
            else:
                log.debug(f"📝 [{sym}] 平仓记录已写入（pnl={pnl_pct*100:.2f}%），无需AI复盘")

            # ── 阶段三：加锁，写回状态 ──────────────────────────────────────
            with lock:
                if pos.side == side_snapshot:
                    self.trader.state._reset_pos()
            # Phase 3: 状态机推进 → IDLE
            if hasattr(self.trader, '_state_machine') and self.trader._state_machine.is_exiting():
                self.trader._state_machine.transition(TradingState.IDLE, "平仓确认完成")
            self.trader._clear_ai_cache(symbol=sym)
        else:
            code = res.get("code", "")
            log.error(f"❌ 平仓失败: {res}")
            log_event("close_failed", {"reason": reason, "response": res})

            # ── 51023：交易所无仓位，但本地 pos 有值 → 必须同步清除 ──────────
            # 原因：上次平仓 API 成功但 WS 回调丢失，或重启时加载了旧状态
            # 不清除会导致：AI 每轮发 close → 51023 → 死循环刷屏
            if code == "51023":
                log.warning("⚠️ 交易所返回「仓位不存在」，本地状态与交易所不一致，强制同步清除本地仓位")
                try:
                    pos_resp = self.okx.get_positions()
                    exch_positions = [
                        p for p in pos_resp.get("data", [])
                        if p.get("instId") == CFG.symbol and abs(float(p.get("pos", 0))) > 0
                    ] if pos_resp.get("code") == "0" else []

                    if not exch_positions:
                        # ── 交易所确实无仓位：补写 DB 平仓记录 ───────────────
                        # 从近期账单查询实际成交价，补全这笔"半截交易"
                        reconstructed_exit = 0.0
                        try:
                            bills_resp = self.okx.get_bills_archive(instType="SWAP", limit=20)
                            if bills_resp.get("code") == "0":
                                for bill in bills_resp.get("data", []):
                                    # type=2：平仓类型账单；instId 匹配
                                    if (bill.get("instId") == CFG.symbol
                                            and bill.get("type") in ("2", "1")):
                                        fill_px = float(bill.get("fillPx") or bill.get("px") or 0)
                                        if fill_px > 0:
                                            reconstructed_exit = fill_px
                                            log.info(f"📋 从账单重建出场价: {fill_px:.2f}")
                                            break
                        except Exception as bill_e:
                            log.debug(f"账单查询失败: {bill_e}")

                        # 兜底：用当前最新价（需验证价格合理性，防止插针引入错误价格）
                        if reconstructed_exit <= 0:
                            reconstructed_exit = self.trader._get_price(sym)
                            if reconstructed_exit > 0:
                                # 验证价格合理性：与入场价偏离超过 20% 则认为价格异常
                                price_deviation = abs(reconstructed_exit - entry_snapshot) / entry_snapshot if entry_snapshot > 0 else 1.0
                                if price_deviation > 0.20:
                                    log.warning(f"⚠️ 当前价偏离入场价过多({price_deviation*100:.1f}%)，疑似插针，拒绝用于补写平仓记录")
                                    reconstructed_exit = 0.0  # 放弃补写
                                else:
                                    log.info(f"📋 账单未找到，用当前价代替出场价: {reconstructed_exit:.4f}")
                            else:
                                log.warning("📋 账单未找到且当前价无效")

                        # 修复：若仍无法获取有效出场价，放弃补写DB记录（防止0价格污染）
                        with lock:
                            current_trade_id  = pos.trade_id
                            _recon_leverage   = pos.leverage if pos.leverage > 0 else 1
                        if current_trade_id and reconstructed_exit > 0:
                            update_trade_close(current_trade_id, reconstructed_exit, leverage=_recon_leverage)
                            with lock:
                                pos.trade_id = None
                            log.info(f"📋 已补写平仓记录 trade_id={current_trade_id} exit={reconstructed_exit:.4f}")

                            # 补计已实现盈亏
                            ct_val = self.okx.contract_sizes.get(sym, 0.01)
                            pnl_recon = (
                                (reconstructed_exit - entry_snapshot) if side_snapshot == "long"
                                else (entry_snapshot - reconstructed_exit)
                            )
                            pnl_usdt_recon = pnl_recon * size_snapshot * ct_val
                            gs_add("today_realized_pnl", pnl_usdt_recon)
                            log.info(f"📋 补计已实现盈亏: {pnl_usdt_recon:+.2f}U")
                        elif current_trade_id:
                            # 高优先级2 Fix：无法重建出场价时强制清零本地全状态，防止幽灵仓位循环
                            log.critical(
                                f"🚨 [{sym}] 无法重建出场价(reconstructed_exit={reconstructed_exit:.4f})，"
                                f"强制清零本地状态（trade_id={current_trade_id}）"
                            )
                            with lock:
                                pos.trade_id = None
                                self.trader.state._reset_pos()
                            self.trader._clear_ai_cache(symbol=sym)
                            log.warning(f"⚠️ 放弃补写平仓记录：无法获取有效出场价（trade_id={current_trade_id}）")

                        # 清除本地状态
                        with self.trader.lock:
                            if pos.side == side_snapshot:
                                self.trader.state._reset_pos()
                        self.trader._clear_ai_cache(symbol=sym)
                        log.info("✅ 幽灵仓位已清除，DB 记录已补写，恢复正常运行")
                        _webhook(
                            "⚠️ 幽灵仓位修复",
                            f"本地残留 {side_snapshot} 仓位与交易所不符\n"
                            f"重建出场价: {reconstructed_exit:.2f}\n已补写平仓记录"
                        )
                    else:
                        log.warning("⚠️ 交易所仍有持仓，可能是平仓参数错误，等待下轮状态同步")
                except Exception as e:
                    log.error(f"验证仓位状态失败: {e}，下轮全量同步")

    def _handle_reverse_logic(self, sym: str, action: str, decision: Dict,
                               ind_15m: Dict, funding: Dict) -> bool:
        """
        处理 AI 输出的反向开仓指令（持有多单时输出 open_short，或反之）。
        返回 True 表示已处理并拦截后续流程，返回 False 表示未处理（应走默认逻辑）。

        三层保护：
          1. 硬性 conf 门控：conf < 0.80 → 降级为仅平仓，不反手
          2. 翻转防抖：600s 内不重复翻转（防连续打脸）
          3. conf ≥ 0.80 + 冷却通过 → 执行翻转（consecutive_losses 不清零，新仓仍受冷却约束）
        """
        conf = decision.get("confidence", 0)
        decision_id = decision.get("decision_id") or 0

        # ── 第一层：硬性 conf 门控 ─────────────────────────────────────────
        if conf < 0.80:
            log.warning(
                f"🛑 [{sym}] AI 试图反手({action})但 conf={conf:.2f}<0.80 | "
                f"降级为【仅平仓】，不执行反向开仓"
            )
            self._close(
                reason=f"翻转置信度不足({conf:.2f}<0.80)，强制平仓避险",
                decision_id=decision_id,
                symbol=sym,
            )
            return True

        # ── 第二层：翻转防抖冷却 ───────────────────────────────────────────
        if self.trader._is_redundant_fast_signal(sym, "REVERSE_LOCK", action, conf, cooldown=600):
            log.info(
                f"⏳ [{sym}] 翻转动作处于 600s 冷却期内，防止连续打脸，忽略此动作"
            )
            return True

        # ── 第三层：执行翻转 ───────────────────────────────────────────────
        log.warning(
            f"🔄 [{sym}] 满足高置信度({conf:.2f})翻转条件，正在执行【多空翻转】"
        )

        # 注意：consecutive_losses 保持不变，翻转后新仓仍受"止损后冷却"约束
        # 但 conf ≥ 0.80 时冷却期闯关条件（条件A）仍可放行

        # 步骤1：平仓当前持仓
        close_reason = f"AI确信趋势反转(conf={conf:.2f})"
        self._close(reason=close_reason, decision_id=decision_id, symbol=sym)

        # 步骤2：等待平仓确认（最多 5 秒）
        for _ in range(5):
            time.sleep(1)
            self.trader.state.sync_position(symbol=sym)
            if not self.trader.pos.side and self.trader.pos.size == 0:  # 双重确认：side=None 且 size=0
                break

        if self.trader.pos.side or self.trader.pos.size > 0:
            log.error(f"⚠️ [{sym}] 平仓未能在 5s 内完成（side={self.trader.pos.side}, size={self.trader.pos.size}），跳过翻转开仓")
            return True

        # 步骤3：同步余额（平仓保证金已释放）
        self.trader.latest_avail_bal = (
            self.okx.get_account_balance_full().get("avail_bal") or 0.0
        )
        balance = self.trader.latest_avail_bal
        # 余额二次校验：极端行情下交易所结算可能延迟，余额未更新时跳过反向开仓
        if balance <= 0:
            log.error(
                f"❌ [{sym}] 翻转失败：平仓后余额={balance:.2f}，"
                f"可能因交易所结算延迟未及时释放保证金，取消反向开仓"
            )
            return True
        price = self.trader._get_price(sym)
        if price <= 0:
            price = self.okx.get_current_price()

        # 步骤4：构建反向开仓决策（复用 AI 原始决策结构，仅改 action）
        reverse_dec = dict(decision)
        reverse_dec["action"] = action
        reverse_dec["decision_id"] = decision_id

        # 步骤5：执行反向开仓（使用 _do_open 完整参数链）
        try:
            self._do_open(
                reverse_dec, price, balance, ind_15m["atr"], funding, decision_id,
                symbol=sym, ind_15m=ind_15m, market_mode=self.trader._market_mode,
            )
            log.warning(f"🔄 [{sym}] 多空翻转完成：反向开仓 {action}")
        except Exception as e:
            log.exception(f"❌ [{sym}] 翻转开仓失败: {e}")

        return True
