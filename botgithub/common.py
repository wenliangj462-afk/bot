# ============================================================
# common.py — 共享基础设施层
# 所有模块从 common.py 导入，不反向依赖主文件
# ============================================================
import os, time, json, logging, traceback, re, hmac, hashlib, base64, requests, math
import xml.etree.ElementTree as ET
import threading
import queue
import sqlite3
import weakref
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any, Callable
from functools import wraps
from contextlib import closing
from urllib.parse import urlencode, urlparse, parse_qs
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

# ── 配置单例（从 config.py 获取，打破循环依赖）──────────────────────────────
from config import CFG, submit_pending_config, get_pending_configs, approve_pending_config, \
    reject_pending_config, try_apply_level2_suggestions, _load_dynamic_config, \
    _CFG_FIELD_MAP, _LEVEL2_BOUNDS, _LEVEL0_LOCKED, log_slippage

# ── 全局状态函数（从 core.py 统一获取）─────────────────────────────────────
from core import gs_get, gs_set, gs_update, gs_increment, gs_add, UTC, \
    PositionIntent, PositionIntentType

# ── 通用时间解析 ─────────────────────────────────────────────────────────────
def _parse_dt(s: str) -> Optional[datetime]:
    """
    安全解析 ISO 时间字符串，统一返回带时区（UTC-aware）的 datetime。
    兼容新格式（含 +00:00）和旧状态文件格式（不含时区后缀）。
    """
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return None

def price_reclaimed(current_price: float, stop_price: float, direction: str) -> bool:
    """
    判断价格是否重新站上（多）或站下（空）原止损位。
    多头止损后：current_price > stop_price * 1.005（超出0.5%确认站稳）
    空头止损后：current_price < stop_price * 0.995（超出0.5%确认站稳）
    """
    if stop_price <= 0 or current_price <= 0 or not direction:
        return False
    if direction == "long":
        return current_price >= stop_price * 1.005
    else:
        return current_price <= stop_price * 0.995

# ============================================================
# 日志配置（自动切分 + AI决策独立日志）
# ============================================================
log = logging.getLogger("ETH_Quant_V6.0")
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ── 全局 System Prompt（统一管理，避免散落硬编码）─────────────────────────
SYSTEM_PROMPT_TRADE = "你是专业的ETH量化交易决策引擎。只输出JSON，不输出任何其他内容。"
SYSTEM_PROMPT_RISK  = "你是专业量化交易风控专家。只输出JSON，不输出任何其他内容。"

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

file_handler = RotatingFileHandler(
    os.getenv("LOG_FILE", "eth_trader_v4.log"),
    maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
)
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

ai_log = logging.getLogger("AI_Decision")
ai_log.setLevel(logging.INFO)
ai_handler = RotatingFileHandler("ai_decisions.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
ai_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
ai_log.addHandler(ai_handler)

_event_log_path = "eth_events.jsonl"
def log_event(event_type: str, data: Dict):
    record = {"ts": datetime.now(UTC).isoformat(), "event": event_type, **data}
    try:
        with open(_event_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ============================================================
# Kelly 公式 + 贝叶斯置信度计算
# ============================================================
def kelly_optimal_size(p_win: float, b: float, fraction: float = 1.0) -> float:
    """
    Kelly 公式计算最优仓位比例。
    p_win: 胜率（可用贝叶斯后验概率）
    b: 盈亏比 (TP距离 / SL距离)
    fraction: Kelly 比例系数（1.0=全 Kelly，0.5=半 Kelly 更保守）
    """
    q = 1 - p_win
    if b <= 0:
        return 0.0
    f = (p_win * b - q) / b
    f = max(0.0, min(f, 1.0)) * fraction
    return f

def bayesian_posterior(prior: float, likelihood: float) -> float:
    """
    简化的贝叶斯后验概率计算。
    prior: 先验概率（默认 0.5）
    likelihood: 似然（结合 AI 置信度 + 盘口失衡度）
    返回后验概率
    """
    numerator = likelihood * prior
    denominator = (likelihood * prior) + ((1 - likelihood) * (1 - prior))
    if denominator <= 0:
        return prior
    return numerator / denominator

def log_kelly_metrics(sym: str, p_win: float, b: float, kelly_f: float,
                      risk_mult: float, slippage_mult: float,
                      final_risk: float, risk_budget: float,
                      size: int, price: float, decision_id: int,
                      committee_opposing: int = 0,
                      market_mode: str = "",
                      posterior_confidence: float = 0.5,
                      kelly_fraction: float = 0.5,
                      atr: float = 0.0):
    """
    将 Kelly 公式关键参数写入结构化日志，便于事后分析模型有效性。
    """
    BASE_BP = 5
    VOL_MULT = 3
    if price > 0 and atr > 0:
        vol_component = (atr / price) * VOL_MULT * 100
        expected_slippage_bp = round(BASE_BP + vol_component, 2)
    else:
        expected_slippage_bp = BASE_BP

    record = {
        "ts": datetime.now(UTC).isoformat(),
        "symbol": sym,
        "p_win": round(p_win, 4),
        "b": round(b, 3),
        "kelly_f": round(kelly_f, 4),
        "kelly_fraction": round(kelly_fraction, 4),
        "risk_mult": round(risk_mult, 3),
        "slippage_mult": slippage_mult,
        "final_risk": round(final_risk, 4),
        "risk_budget": round(risk_budget, 2),
        "size": size,
        "price": round(price, 4),
        "decision_id": decision_id,
        "committee_opposing": committee_opposing,
        "market_mode": market_mode,
        "posterior_confidence": round(posterior_confidence, 4),
        "expected_slippage_bp": expected_slippage_bp,
        "atr": round(atr, 4) if atr > 0 else 0.0,
    }
    try:
        with open(os.getenv("KELLY_LOG_FILE", "kelly_metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ============================================================
# SQLite 数据库（WAL模式 + 线程本地连接）
# ============================================================
DB_PATH = "eth_trading.db"
_thread_local = threading.local()
_db_init_lock = threading.Lock()  # 保护 threading.local() 初始化的原子性

def _create_db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

def get_db_conn():
    # 双检锁模式：先无锁检查，再有锁初始化，保证原子性
    if not getattr(_thread_local, 'conn', None):
        with _db_init_lock:
            if not getattr(_thread_local, 'conn', None):
                _thread_local.conn = _create_db_conn()
    try:
        _thread_local.conn.execute("SELECT 1")
    except Exception:
        _thread_local.conn = _create_db_conn()
    return _thread_local.conn

def init_db():
    conn = get_db_conn()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT DEFAULT 'ETH-USDT-SWAP',
            ts DATETIME, thought_process TEXT, action TEXT,
            confidence REAL, suggested_sl REAL, suggested_tp REAL,
            reason TEXT, price REAL, balance REAL, features TEXT,
            slippage_pct REAL DEFAULT 0.0
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT DEFAULT 'ETH-USDT-SWAP',
            open_ts DATETIME, close_ts DATETIME, side TEXT,
            size REAL, entry REAL, exit REAL, pnl REAL, pnl_pct REAL,
            decision_id INTEGER, close_reason TEXT DEFAULT '',
            fail_reason TEXT DEFAULT '',
            FOREIGN KEY(decision_id) REFERENCES decisions(id)
        )
    ''')
    # 补充 trades 表缺失的列（兼容旧数据库）
    for col, dtype in [
        ("confidence", "REAL DEFAULT 0"),
        ("entry_market_mode", "TEXT"),
        ("entry_rsi", "REAL DEFAULT 0"),
        ("entry_bb_pct", "REAL DEFAULT 0"),
        ("entry_atr_pct", "REAL DEFAULT 0"),
        ("ai_confidence", "REAL DEFAULT 0"),
        ("ai_reason", "TEXT"),
    ]:
        try:
            c.execute(f"ALTER TABLE trades ADD COLUMN {col} {dtype}")
        except Exception:
            pass  # 列已存在
    c.execute('''
        CREATE TABLE IF NOT EXISTS system_config (
            key TEXT PRIMARY KEY, value TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS pending_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            param_key TEXT NOT NULL, old_value TEXT,
            new_value TEXT NOT NULL, reason TEXT,
            source TEXT DEFAULT 'reasoner',
            status TEXT DEFAULT 'pending',
            created_ts DATETIME, applied_ts DATETIME
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS pending_orders (
            ord_id TEXT PRIMARY KEY, symbol TEXT DEFAULT 'ETH-USDT-SWAP',
            side TEXT, pos_side TEXT, size REAL,
            entry_price REAL, sl REAL, tp REAL,
            leverage INTEGER, liq_price REAL, margin REAL,
            decision_id INTEGER, created_ts DATETIME
        )
    ''')

    existing_decisions = {row[1] for row in c.execute("PRAGMA table_info(decisions)")}
    for col, ddl in [
        ("thought_process", "TEXT"), ("slippage_pct", "REAL DEFAULT 0.0"),
        ("symbol", "TEXT DEFAULT 'ETH-USDT-SWAP'"),
    ]:
        if col not in existing_decisions:
            c.execute(f"ALTER TABLE decisions ADD COLUMN {col} {ddl}")

    existing_trades = {row[1] for row in c.execute("PRAGMA table_info(trades)")}
    for col, ddl in [
        ("side", "TEXT"), ("fail_reason", "TEXT DEFAULT ''"),
        ("close_reason", "TEXT DEFAULT ''"), ("symbol", "TEXT DEFAULT 'ETH-USDT-SWAP'"),
    ]:
        if col not in existing_trades:
            c.execute(f"ALTER TABLE trades ADD COLUMN {col} {ddl}")

    existing_pending = {row[1] for row in c.execute("PRAGMA table_info(pending_orders)")}
    if "symbol" not in existing_pending:
        c.execute("ALTER TABLE pending_orders ADD COLUMN symbol TEXT DEFAULT 'ETH-USDT-SWAP'")
    if "margin" not in existing_pending:
        c.execute("ALTER TABLE pending_orders ADD COLUMN margin REAL DEFAULT 0.0")
    if "retry_count" not in existing_pending:
        c.execute("ALTER TABLE pending_orders ADD COLUMN retry_count INTEGER DEFAULT 0")

    c.execute("CREATE INDEX IF NOT EXISTS idx_pending_orders_symbol ON pending_orders(symbol)")
    conn.commit()

def _ensure_db_indexes():
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_decision_id ON trades(decision_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol, close_ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_decisions_symbol_ts ON decisions(symbol, ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_pending_status_created ON pending_config(status, created_ts)")
        conn.commit()
    except Exception:
        pass

# ============================================================
# 系统配置读写
# ============================================================
def get_sys_config(key: str, default: Any = None) -> Any:
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT value FROM system_config WHERE key=?", (key,))
    row = c.fetchone()
    return row[0] if row else default

def set_sys_config(key: str, value: Any):
    """
    设置系统配置（异步写入，避免阻塞）
    
    Args:
        key: 配置键
        value: 配置值（任意类型，自动转为字符串）
    """
    sql = "REPLACE INTO system_config (key, value) VALUES (?, ?)"
    params = (key, str(value))
    _db_manager.write(sql, params)

# ============================================================
# Pending Orders 操作
# ============================================================
def save_pending_order(ord_id: str, side: str, pos_side: str, size: float,
                       entry_price: float, sl: float, tp: float, leverage: int,
                       liq_price: float, decision_id: int, symbol: str = "ETH-USDT-SWAP",
                       margin: float = 0.0):
    """保存挂单到数据库"""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO pending_orders
        (ord_id, symbol, side, pos_side, size, entry_price, sl, tp, leverage, liq_price, margin, decision_id, created_ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (ord_id, symbol, side, pos_side, size, entry_price, sl, tp, leverage, liq_price, margin, decision_id,
          datetime.now(UTC).isoformat()))
    conn.commit()

def get_pending_order_by_id(ord_id: str) -> Optional[Dict]:
    """根据订单ID获取pending订单"""
    if not ord_id:
        return None
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM pending_orders WHERE ord_id=?", (ord_id,))
    row = c.fetchone()
    if not row:
        return None
    cols = [desc[0] for desc in c.description]
    return dict(zip(cols, row))

def delete_pending_order(ord_id: str):
    """
    删除 pending 订单（异步写入，避免阻塞）
    
    Args:
        ord_id: 订单 ID
    """
    sql = "DELETE FROM pending_orders WHERE ord_id=?"
    params = (ord_id,)
    _db_manager.write(sql, params)

def get_all_pending_orders() -> List[Dict]:
    """获取所有pending订单"""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM pending_orders ORDER BY created_ts DESC")
    rows = c.fetchall()
    if not rows:
        return []
    cols = [desc[0] for desc in c.description]
    return [dict(zip(cols, row)) for row in rows]

# ============================================================
# Trade 操作
# ============================================================
def update_trade_close(trade_id: int, exit_price: float, close_reason: str = "",
                       leverage: int = 1, pnl: float = None, pnl_pct: float = None,
                       fail_reason: str = None):
    """更新交易平仓信息"""
    for attempt in range(3):
        try:
            conn = get_db_conn()
            if pnl is not None:
                conn.execute("UPDATE trades SET exit=?, close_reason=?, pnl=?, pnl_pct=? WHERE id=?",
                             (exit_price, close_reason, pnl, pnl_pct, trade_id))
            else:
                conn.execute("UPDATE trades SET exit=?, close_reason=? WHERE id=?",
                             (exit_price, close_reason, trade_id))
            conn.commit()
            return
        except Exception as e:
            if "locked" in str(e) and attempt < 2:
                time.sleep(0.5); continue
            log.exception(f"update_trade_close 写入失败: {e}")

# ============================================================
# Win Rate 查询
# ============================================================
# ============================================================
# AI 异步任务
# ============================================================
def generate_fail_reason_async(trade_id: int, ai_client, context: str):
    """异步生成失败原因摘要（RAG闭环）"""
    threading.Thread(
        target=_generate_fail_reason_worker,
        args=(trade_id, ai_client, context),
        daemon=True,
        name="fail-reason-gen"
    ).start()

def _generate_fail_reason_worker(trade_id: int, ai_client, context: str):
    """失败原因生成的工作线程"""
    try:
        from common import _call_reasoner
        prompt = f"""分析以下交易失败的潜在原因：

{context}

请简要分析失败原因（100字以内），用于RAG案例库改进。"""
        reason = _call_reasoner(ai_client, prompt, max_tokens=200, timeout=30)
        if reason:
            conn = get_db_conn()
            c = conn.cursor()
            c.execute("UPDATE trades SET fail_reason=? WHERE id=?", (reason[:200], trade_id))
            conn.commit()
            log.debug(f"[RAG] 交易 {trade_id} 失败原因已生成: {reason[:50]}...")
    except Exception as e:
        log.debug(f"[RAG] 生成失败原因异常: {e}")

def _auto_generate_historical_case(trade_id: int, ai_client, pos_snapshot: Dict):
    """自动生成历史案例到RAG库"""
    threading.Thread(
        target=_auto_gen_case_worker,
        args=(trade_id, ai_client, pos_snapshot),
        daemon=True,
        name="auto-case-gen"
    ).start()

def _auto_gen_case_worker(trade_id: int, ai_client, pos_snapshot: Dict):
    """自动案例生成的工作线程"""
    try:
        from common import _call_chat
        prompt = f"""评估以下交易案例的质量，生成简短总结：

方向: {pos_snapshot.get('side', 'unknown')}
入场模式: {pos_snapshot.get('entry_market_mode', 'unknown')}
RSI: {pos_snapshot.get('entry_rsi', 0):.2f}
盈亏: {pos_snapshot.get('pnl_pct', 0)*100:+.2f}%
置信度: {pos_snapshot.get('ai_confidence', 0):.2f}

生成5-10字的质量评分标签，如"高质量趋势跟踪"或"低质量震荡交易"。"""
        response = _call_chat(ai_client, [{"role": "user", "content": prompt}],
                             max_tokens=50, timeout=20)
        log.debug(f"[RAG] 案例 {trade_id} 自动标签: {response[:30] if response else '无'}")
    except Exception as e:
        log.debug(f"[RAG] 自动案例生成异常: {e}")

# ============================================================
# Bot 实例弱引用（全局单例）
# ============================================================
_bot_instance_ref: Optional[weakref.ref] = None

def _set_bot_instance(bot):
    global _bot_instance_ref
    _bot_instance_ref = weakref.ref(bot)

def _get_bot_instance():
    if _bot_instance_ref is None:
        return None
    return _bot_instance_ref()

class _BotInstanceProxy:
    def __bool__(self):
        return _get_bot_instance() is not None
    def __getattr__(self, name):
        bot = _get_bot_instance()
        if bot is None:
            raise AttributeError("bot_instance is not set")
        return getattr(bot, name)
    def __setattr__(self, name, value):
        bot = _get_bot_instance()
        if bot is None:
            raise AttributeError("bot_instance is not set")
        setattr(bot, name, value)

bot_instance = _BotInstanceProxy()

# ============================================================
# 动态配置管理（从 config.py 统一获取，已在顶部导入）
# ============================================================

# ============================================================
# 决策记录写入
# ============================================================
def save_decision_to_db(decision: Dict, price: float, balance: float, features: Dict,
                        slippage_pct: float = 0.0, symbol: str = "ETH-USDT-SWAP") -> Optional[int]:
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute('''
            INSERT INTO decisions
            (symbol, ts, thought_process, action, confidence, suggested_sl, suggested_tp,
             reason, price, balance, features, slippage_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, datetime.now(UTC).isoformat(),
            decision.get("thought_process", "")[:500],
            decision.get("action"), decision.get("confidence", 0.0),
            decision.get("suggested_sl", 0.0), decision.get("suggested_tp", 0.0),
            decision.get("reason", ""),
            price, balance, json.dumps(features, ensure_ascii=False), slippage_pct
        ))
        conn.commit()
        return c.lastrowid
    except Exception as e:
        log.exception(f"决策写入失败: {e}")
        return None

def save_trade_open(decision_id: Optional[int], side: str, size: float, entry: float,
                    symbol: str = "ETH-USDT-SWAP") -> Optional[int]:
    for attempt in range(3):
        try:
            conn = get_db_conn()
            c = conn.cursor()
            c.execute('''
                INSERT INTO trades (symbol, open_ts, side, size, entry, decision_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, datetime.now(UTC).isoformat(), side, size, entry, decision_id))
            conn.commit()
            return c.lastrowid
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < 2:
                time.sleep(0.5); continue
            log.exception(f"数据库写入失败: {e}"); return None
    return None

# ============================================================
# Webhook
# ============================================================
_webhook_queue: queue.Queue = queue.Queue(maxsize=500)
_webhook_thread: Optional[threading.Thread] = None

def _webhook_worker():
    """Webhook worker - 处理 _webhook_queue 中的消息"""
    global _webhook_consecutive_failures
    max_retries = max(getattr(CFG, "webhook_retry", 3), 1)
    while True:
        try:
            title, content = _webhook_queue.get(timeout=5)
        except queue.Empty:
            continue
        if not CFG.webhook_url:
            _webhook_queue.task_done()
            continue
        try:
            platform = _get_webhook_platform()
            method, url, kwargs = _build_webhook_payload(platform, title, content)

            success = False
            for attempt in range(max_retries):
                try:
                    resp = requests.request(method, url, **kwargs)
                    if resp.status_code == 200:
                        success = True
                        _webhook_consecutive_failures = 0
                        log.debug(f"Webhook 发送成功 [{platform}]")
                        break
                    else:
                        log.debug(f"Webhook HTTP {resp.status_code} [attempt {attempt+1}]: {resp.text[:120]}")
                except Exception as e:
                    log.debug(f"Webhook 失败 [{platform}] attempt {attempt+1}: {e}")

                if not success and attempt < (max_retries - 1):
                    time.sleep(2 ** attempt)

            if not success:
                _webhook_consecutive_failures += 1
                log.warning(f"⚠️ Webhook 发送失败，已连续 {_webhook_consecutive_failures} 次")
                if _webhook_consecutive_failures >= _WEBHOOK_FAILURE_ALERT_THRESHOLD:
                    log.error(f"Webhook 连续失败达到阈值 ({_WEBHOOK_FAILURE_ALERT_THRESHOLD}次)，请检查网络或 WEBHOOK_URL 配置")
                    _webhook_consecutive_failures = 0
        finally:
            _webhook_queue.task_done()

def _send_webhook(msg: str, data: str = "", level: int = 1):
    """已废弃，使用 _notify 代替"""
    pass

def _webhook(msg: str, data: str = "", level: int = 1):
    """兼容性别名 - 将 (msg, data, level) 转为 (title, content) 格式"""
    try:
        _webhook_queue.put_nowait((msg, data))
    except queue.Full:
        log.warning(f"Webhook 队列已满，跳过: {msg[:50]}")

def start_webhook_thread():
    global _webhook_thread
    if _webhook_thread is None or not _webhook_thread.is_alive():
        _webhook_thread = threading.Thread(target=_webhook_worker, daemon=True)
        _webhook_thread.start()

# ============================================================
# AI 调用（Reasoner + Chat）
# ============================================================
_reasoner_semaphore = threading.Semaphore(2)

def _call_reasoner(ai_client, messages: list, max_tokens: int = 2000,
                   timeout: int = 120) -> str:
    _reasoner_semaphore.acquire()
    try:
        last_err = None
        max_r = max(1, int(os.getenv("AI_MAX_RETRIES", "3")))
        wait_times = [0] + [5 * (3 ** i) for i in range(max_r - 1)]
        for attempt, wait in enumerate(wait_times[:max_r]):
            if wait:
                time.sleep(wait)
            try:
                res = ai_client.chat.completions.create(
                    model="deepseek-reasoner",
                    max_tokens=max_tokens,
                    messages=messages,
                    timeout=timeout,
                )
                return res.choices[0].message.content.strip()
            except TimeoutError as te:
                last_err = te
                log.warning(f"Reasoner 超时(attempt {attempt+1}/3): {te}")
            except Exception as e:
                last_err = e
                if "429" in str(e) or "rate" in str(e).lower():
                    log.warning(f"Reasoner Rate Limit，第{attempt+1}次重试: {e}")
                    continue
                raise
        raise RuntimeError(f"Reasoner 重试耗尽: {last_err}")
    finally:
        _reasoner_semaphore.release()

_chat_semaphore = threading.Semaphore(2)

def _call_chat(ai_client, messages: list, max_tokens: int = 600,
               temperature: float = 0.5, timeout: int = 60) -> str:
    _chat_semaphore.acquire()
    try:
        last_err = None
        for attempt, wait in enumerate([0, 3, 10]):
            if wait:
                time.sleep(wait)
            try:
                res = ai_client.chat.completions.create(
                    model="deepseek-chat",
                    max_tokens=max_tokens,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout,
                )
                return res.choices[0].message.content.strip()
            except TimeoutError as te:
                last_err = te
                log.warning(f"Chat 超时(attempt {attempt+1}/3)，重试: {te}")
            except Exception as e:
                last_err = e
                if "429" in str(e) or "rate" in str(e).lower():
                    log.warning(f"Chat Rate Limit，第{attempt+1}次重试，等待{wait}s: {e}")
                    continue
                raise
        raise RuntimeError(f"Chat 重试耗尽: {last_err}")
    finally:
        _chat_semaphore.release()

def _clean_reasoner_json(raw_text: str) -> dict:
    """从 Reasoner 输出中提取 JSON（支持 think 标签包裹）"""
    result = {"action": "hold", "confidence": 0.5}
    _think_start = raw_text.find("████")
    _think_end = raw_text.find("████████")
    if _think_start != -1 and _think_end != -1 and _think_end > _think_start:
        json_candidate = raw_text[_think_end + 24:].strip().lstrip()
        try:
            result = json.loads(json_candidate)
        except json.JSONDecodeError:
            log.warning(f"Reasoner JSON 解析失败: {json_candidate[:200]}")
    else:
        try:
            result = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            log.warning(f"Reasoner 无think标签且JSON解析失败: {raw_text[:200]}")
    return result

# ============================================================
# 全局状态单例 GS_STATE
# ============================================================
class _GSState:
    """
    全局状态单例，替代 getattr(bot_instance, ...) 模式。
    所有需要共享的状态通过 GS_STATE 读写，模块间无循环依赖。
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {
            "position": None,           # Position 对象
            "entry_price": 0.0,
            "current_pnl_pct": 0.0,
            "unrealised_pnl": 0.0,
            "mark_price": 0.0,
            "last_ticker_price": 0.0,
            "funding_rate": 0.0,
            "market_mode": "趋势",
            "ws_connected": False,
            "ai_cache": {},             # AI 缓存
            "pending_orders": [],       # 待确认订单列表
            "daily_pnl": 0.0,
            "daily_trades": 0,
            "consecutive_loss": 0,
            "drawdown_mult": 1.0,
            "atr": 0.0,
            "trailing_activated": False,
            "trailing_stop": 0.0,
            "in_trade": False,
            "ev_hour": datetime.now(UTC).hour,
        }

    def update(self, **kwargs):
        with self._lock:
            self._data.update(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: str, value: Any):
        with self._lock:
            self._data[key] = value

GS_STATE = _GSState()

def _gs_state_update(**kwargs):
    """GS_STATE 的线程安全批量更新（仅限 common.py 内部使用）"""
    GS_STATE.update(**kwargs)

# ============================================================
# 初始化（启动时调用一次）
# ============================================================
init_db()
_ensure_db_indexes()
_load_dynamic_config()
start_webhook_thread()

# ── Prometheus 指标（供 health endpoint 使用）──────────────────────────────
REGISTRY = CollectorRegistry(auto_describe=True)
balance_gauge = Gauge('account_balance', 'Current USDT balance', registry=REGISTRY)
position_side_gauge = Gauge('position_side', '1 for long, -1 for short, 0 for flat', registry=REGISTRY)
pnl_pct_gauge = Gauge('position_pnl_pct', 'Current position PnL percentage', registry=REGISTRY)
last_ai_confidence_gauge = Gauge('last_ai_confidence', 'Confidence of last AI decision', registry=REGISTRY)
consecutive_losses_gauge = Gauge('consecutive_losses', 'Number of consecutive losing trades', registry=REGISTRY)
consecutive_slippage_gauge = Gauge('consecutive_slippage', 'Number of consecutive high slippage trades', registry=REGISTRY)
ai_request_count = Gauge('ai_request_total', 'Total number of AI requests', registry=REGISTRY)
slippage_avg_gauge = Gauge('slippage_avg_pct', 'Average slippage percentage over last N trades', registry=REGISTRY)
open_positions_gauge = Gauge('open_positions', 'Number of currently open positions', registry=REGISTRY)

_health_data: dict = {
    "status":       "starting",
    "ai_summaries": [],
    "last_update":  None,
}

# ── Webhook 辅助函数 ─────────────────────────────────────────────────────
def _detect_webhook_platform(url: str) -> str:
    """根据 URL 特征自动识别通知平台"""
    if not url:
        return "unknown"
    url_lower = url.lower()
    if "sctapi.ftqq.com" in url_lower or "sc.ftqq.com" in url_lower:
        return "serverchan"
    if "qyapi.weixin.qq.com" in url_lower:
        return "wecom"
    if "open.feishu.cn" in url_lower or "open.larksuite.com" in url_lower:
        return "feishu"
    if "oapi.dingtalk.com" in url_lower:
        return "dingtalk"
    if "api.telegram.org" in url_lower:
        return "telegram"
    return "generic"

def _build_webhook_payload(platform: str, title: str, content: str):
    """根据平台构造对应的请求参数 (method, url, kwargs)"""
    full_text = f"【ETH量化V6.0】{title}\n{content}"
    url = CFG.webhook_url

    if platform == "serverchan":
        return "POST", url, {
            "data": {"title": f"【ETH量化V6.0】{title}", "desp": content},
            "timeout": 8
        }
    elif platform == "feishu":
        return "POST", url, {
            "json": {"msg_type": "text", "content": {"text": full_text}},
            "timeout": 8
        }
    elif platform == "dingtalk":
        return "POST", url, {
            "json": {"msgtype": "text", "text": {"content": full_text}},
            "timeout": 8
        }
    elif platform == "telegram":
        text = f"{title}\n{content}"[:4096]
        return "POST", url, {
            "data":    json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"),
            "headers": {"Content-Type": "application/json; charset=utf-8"},
            "timeout": 10
        }
    else:
        return "POST", url, {
            "json": {"msgtype": "text", "text": {"content": full_text}},
            "timeout": 8
        }

# 缓存平台类型
_webhook_platform: Optional[str] = None

def _safe_url_for_log(url: str) -> str:
    """安全打印 URL：隐藏 token，只显示域名"""
    if not url:
        return "<未配置>"
    try:
        parsed = urlparse(url)
        if "api.telegram.org" in parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}/bot***"
        safe = f"{parsed.scheme}://{parsed.netloc}"
        if len(safe) > 50:
            safe = safe[:50] + "..."
        return safe
    except Exception:
        return "<解析失败>"


# ── Telegram 命令处理 ───────────────────────────────────────────────────
_tg_last_update_id: int = 0

def _parse_tg_url(url: str) -> tuple:
    """从 Telegram URL 解析 token 和 chat_id"""
    import re as _re
    m = _re.search(r'bot([^/]+)/\w+\?chat_id=(-?\d+)', url)
    if m:
        return m.group(1), m.group(2)
    return None, None


def _tg_poll_commands():
    """Telegram 命令轮询线程。"""
    global _tg_last_update_id
    if _detect_webhook_platform(CFG.webhook_url) != "telegram":
        return
    token, chat_id = _parse_tg_url(CFG.webhook_url)
    if not token or not chat_id:
        log.warning("Telegram URL 格式不正确，命令轮询未启动")
        return

    base = f"https://api.telegram.org/bot{token}"
    log.info(f"Telegram 命令轮询已启动 (chat_id={chat_id})")

    def send(text: str):
        try:
            requests.post(f"{base}/sendMessage",
                          json={"chat_id": chat_id, "text": text[:4096],
                                "parse_mode": "Markdown"}, timeout=8)
        except Exception:
            pass

    while True:
        try:
            resp = requests.get(f"{base}/getUpdates",
                                params={"offset": _tg_last_update_id + 1, "timeout": 30},
                                timeout=35)
            if resp.status_code != 200:
                time.sleep(5)
                continue
            updates = resp.json().get("result", [])
            for upd in updates:
                _tg_last_update_id = upd["update_id"]
                msg = upd.get("message", {})
                if str(msg.get("chat", {}).get("id", "")) != str(chat_id):
                    continue
                text = msg.get("text", "").strip()
                if not text.startswith("/"):
                    continue

                cmd = text.split()[0].lower()
                log.info(f"Telegram 命令: {cmd}")

                if cmd == "/status":
                    bot = _get_bot_instance()
                    lines = ["📊 *系统状态*\n"]
                    equity_s = getattr(bot, 'latest_equity', 0) if bot else 0
                    lines.append(f"💰 权益: `{equity_s:.2f}U` | 可用: `{getattr(bot, 'latest_avail_bal', 0):.2f}U`\n")
                    lines.append(f"📈 动态风险因子: `{getattr(bot, 'dynamic_risk_factor', 1.0):.2f}` | Kelly系数: `{CFG.kelly_fraction:.2f}`\n")
                    today_risk = float(gs_get("today_opened_risk", 0.0))
                    daily_cap  = equity_s * CFG.max_daily_risk_pct
                    lines.append(f"📅 今日已用风险: `{today_risk:.2f}U` / `{daily_cap:.2f}U`\n")
                    lines.append(f"📅 今日已实现盈亏: `{gs_get('today_realized_pnl', 0.0):+.2f}U`\n")
                    lines.append(f"📉 连续亏损次数: `{gs_get('consecutive_losses', 0)}`\n\n")
                    pos_s = getattr(bot, 'pos', None) if bot else None
                    if pos_s and pos_s.side:
                        price_s = getattr(bot, '_price_val', 0) if bot else 0
                        pnl = (
                            (price_s - pos_s.entry_price) / pos_s.entry_price
                            if pos_s.side == "long"
                            else (pos_s.entry_price - price_s) / pos_s.entry_price
                        ) if pos_s.entry_price > 0 else 0
                        lines.append(
                            f"📌 *{CFG.symbol}*\n"
                            f"  方向: `{pos_s.side}` x{pos_s.leverage} | {pos_s.size}张\n"
                            f"  开仓价: `{pos_s.entry_price:.4f}` | 浮盈: `{pnl*100:+.2f}%`\n"
                            f"  SL: `{pos_s.stop_loss:.4f}` | TP: `{pos_s.take_profit:.4f}`\n"
                        )
                    else:
                        lines.append("📭 当前无持仓\n")
                    pause_until = gs_get("pause_until")
                    if pause_until:
                        lines.append(f"\n⏸️ 开仓已暂停至: `{pause_until[:16]}`")
                    send("".join(lines))

                elif cmd == "/pending":
                    from config import get_pending_configs
                    items = get_pending_configs()
                    if not items:
                        send("✅ 当前无待审批的参数变更")
                    else:
                        lines = ["📋 *待审批参数变更*\n"]
                        for p in items:
                            lines.append(
                                f"`[{p['id']}]` {p['key']}: {p['old']} → *{p['new']}*\n"
                                f"  理由: {p['reason'][:80]}\n"
                            )
                        lines.append("\n审批: /approve\\_N  拒绝: /reject\\_N")
                        send("".join(lines))

                elif cmd.startswith("/approve_"):
                    from config import approve_pending_config
                    try:
                        cid = int(cmd.split("_")[1])
                        key, val = approve_pending_config(cid)
                        if key:
                            send(f"✅ 已审批: `{key}` = `{val}`\n参数已立即生效（无需重启）")
                        else:
                            send(f"❌ 未找到 id={cid} 的待审批记录")
                    except (IndexError, ValueError):
                        send("格式错误，示例: /approve\\_1")

                elif cmd.startswith("/reject_"):
                    from config import reject_pending_config
                    try:
                        cid = int(cmd.split("_")[1])
                        reject_pending_config(cid)
                        send(f"❌ 已拒绝 id={cid} 的变更建议")
                    except (IndexError, ValueError):
                        send("格式错误，示例: /reject\\_1")

                elif cmd == "/pause":
                    pause_until = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
                    gs_set("pause_until", pause_until)
                    send("⏸️ 已暂停开新仓 1 小时（风控线程继续运行）")

                elif cmd == "/resume":
                    gs_set("pause_until", None)
                    send("▶️ 已恢复开仓，下一轮循环生效")

                elif cmd == "/block_ai":
                    bot = _get_bot_instance()
                    if bot:
                        bot._ai_blocked = True
                        send("🔒 AI 决策已禁用，系统降级为规则引擎快速决策。使用 /unblock\\_ai 恢复")
                        log.warning("🔒 [Telegram] AI 决策已被人工禁用")
                    else:
                        send("❌ 无法获取 bot 实例")

                elif cmd == "/unblock_ai":
                    bot = _get_bot_instance()
                    if bot:
                        bot._ai_blocked = False
                        bot._ai_circuit_broken_until = 0.0
                        send("🔓 AI 决策及熔断器已恢复正常")
                        log.info("🔓 [Telegram] AI 决策及熔断器已恢复")
                    else:
                        send("❌ 无法获取 bot 实例")

                elif cmd == "/help":
                    send(
                        "🤖 *可用指令*\n"
                        "/status — 当前持仓和今日盈亏\n"
                        "/pending — 待审批参数变更\n"
                        "/approve\\_N — 审批第N条变更\n"
                        "/reject\\_N — 拒绝第N条变更\n"
                        "/pause — 暂停开仓1小时\n"
                        "/resume — 恢复开仓\n"
                        "/block\\_ai — 紧急禁用 AI 决策（降级规则引擎）\n"
                        "/unblock\\_ai — 恢复 AI 决策"
                    )
        except Exception as e:
            log.debug(f"Telegram 轮询异常: {e}")
            time.sleep(10)

# 启动 Telegram 命令轮询线程
threading.Thread(target=_tg_poll_commands, daemon=True, name="tg-poll").start()



_webhook_consecutive_failures: int = 0
_WEBHOOK_FAILURE_ALERT_THRESHOLD: int = max(getattr(CFG, "webhook_fail_alert", 5), 3)

def _get_webhook_platform() -> str:
    global _webhook_platform
    if _webhook_platform is None:
        _webhook_platform = _detect_webhook_platform(CFG.webhook_url)
        if CFG.webhook_url:
            log.info(f"Webhook 平台识别: {_webhook_platform}（URL: {_safe_url_for_log(CFG.webhook_url)}）")
    return _webhook_platform



# ── _notify 入口函数（供外部调用）─────────────────────────────────────────
def _notify(title: str, content: str, level: int = 1):
    """
    非阻塞：将消息投入队列，立即返回。
    Level 1：核心交易事件
    Level 2：运营事件
    Level 3：全部
    """
    if not CFG.webhook_url:
        return

    level_cfg = CFG.webhook_level

    _L1_PREFIXES = ("开仓", "平仓", "系统启动", "系统停止",
                    "崩溃恢复", "孤儿仓位", "状态不一致",
                    "滑点熔断", "今日实现亏损", "AI 持续超时",
                    "仓位已恢复")
    is_l1 = any(title.startswith(p) or p in title for p in _L1_PREFIXES)

    _L2_PREFIXES = ("每日战报", "调整止盈止损", "误锁自动解除")
    is_l2 = any(title.startswith(p) or p in title for p in _L2_PREFIXES)

    if level_cfg >= 3:
        pass
    elif level_cfg >= 2:
        if not (is_l1 or is_l2):
            return
    else:
        if not is_l1:
            return

    try:
        if is_l1:
            _webhook_queue.put((title, content), timeout=5)
        else:
            _webhook_queue.put_nowait((title, content))
    except queue.Full:
        if is_l1:
            log.warning(f"Webhook 队列已满（L1事件），消息阻塞超时: {title}")
        else:
            log.debug(f"Webhook 队列已满，丢弃消息: {title}")


# ── Health Handler & Server ──────────────────────────────────────────────
class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(generate_latest(REGISTRY))
        elif self.path == "/config/pending":
            from config import get_pending_configs
            pending = get_pending_configs()
            body = json.dumps({"pending": pending}, ensure_ascii=False, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/ai/latest":
            summaries = _health_data.get("ai_summaries", [])
            latest = summaries[-1] if summaries else {}
            body = json.dumps(latest, ensure_ascii=False, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/ai/summaries":
            body = json.dumps(
                {"summaries": _health_data.get("ai_summaries", [])},
                ensure_ascii=False, indent=2
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            body = json.dumps(_health_data, ensure_ascii=False, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        try:
            id_raw = params.get("id", [None])[0]
            if id_raw is None:
                raise ValueError("缺少 id 参数")
            config_id = int(float(id_raw))
            if config_id <= 0:
                raise ValueError("id 必须为正整数")
        except (ValueError, TypeError) as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"无效的 id: {e}"}).encode())
            return

        if parsed.path == "/config/approve":
            from config import approve_pending_config
            key, val = approve_pending_config(config_id)
            if key:
                body = json.dumps({"status": "approved", "param": key, "new_value": val},
                                  ensure_ascii=False).encode()
                self.send_response(200)
            else:
                body = json.dumps({"error": f"id={config_id} 不存在或已处理"}).encode()
                self.send_response(404)

        elif parsed.path == "/config/reject":
            from config import reject_pending_config
            reject_pending_config(config_id)
            body = json.dumps({"status": "rejected", "id": config_id}).encode()
            self.send_response(200)

        else:
            body = json.dumps({"error": f"未知接口: {parsed.path}"}).encode()
            self.send_response(400)

        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args): pass

def start_health_server():
    try:
        srv = HTTPServer(("0.0.0.0", CFG.health_port), _HealthHandler)
        t   = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        log.info(f"健康检查服务已启动: http://0.0.0.0:{CFG.health_port}/health , /metrics")
    except Exception as e:
        log.warning(f"健康检查服务启动失败（不影响主流程）: {e}")


# ── Watchdog ─────────────────────────────────────────────────────────────
class Watchdog:
    def __init__(self):
        self.last_beat = None

    def beat(self):
        self.last_beat = datetime.now(UTC)
        if CFG.webhook_url:
            _notify("看门狗心跳", f"系统运行正常\n余额: {gs_get('start_balance', 0):.2f} USDT", level=3)
        log.debug("看门狗心跳")

def watchdog_loop(watchdog):
    while True:
        try:
            watchdog.beat()
            gs_set("last_watchdog", datetime.now(UTC).isoformat())
            time.sleep(CFG.watchdog_interval)
        except Exception as e:
            log.exception(f"看门狗异常: {e}")
            time.sleep(10)
# ── 复盘系统提示词 ─────────────────────────────────────────────────────────
_POSTMORTEM_SYSTEM_PROMPT = """你是一位资深的量化交易复盘专家。擅长从交易日志、K线数据、订单薄失衡、技术指标中推导亏损/盈利的因果链，并给出可执行的改进建议。"""

# ── 数据库查询辅助函数 ──────────────────────────────────────────────────────

def get_trades_in_range(start_dt: datetime, end_dt: datetime) -> List[Dict]:
    """查询指定时间范围内的已平仓交易"""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, symbol, open_ts, close_ts, side, size, entry, exit, "
        "pnl, pnl_pct, decision_id, close_reason, fail_reason, confidence, "
        "entry_market_mode, entry_rsi, entry_bb_pct, entry_atr_pct, "
        "ai_confidence, ai_reason "
        "FROM trades "
        "WHERE close_ts >= ? AND close_ts <= ? AND pnl IS NOT NULL "
        "ORDER BY close_ts DESC",
        (start_dt.isoformat(), end_dt.isoformat())
    )
    cols = [desc[0] for desc in c.description]
    return [dict(zip(cols, row)) for row in c.fetchall()]

def get_decisions_in_range(start_dt: datetime, end_dt: datetime) -> List[Dict]:
    """查询指定时间范围内的AI决策记录"""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, symbol, ts, thought_process, action, confidence, "
        "suggested_sl, suggested_tp, reason, price, balance, features, slippage_pct "
        "FROM decisions "
        "WHERE ts >= ? AND ts <= ? "
        "ORDER BY ts DESC",
        (start_dt.isoformat(), end_dt.isoformat())
    )
    cols = [desc[0] for desc in c.description]
    rows = []
    for row in c.fetchall():
        d = dict(zip(cols, row))
        if d.get('features'):
            try:
                d['features'] = json.loads(d['features'])
            except Exception:
                d['features'] = {}
        rows.append(d)
    return rows

def cleanup_old_data(days: int = 30):
    """清理 days 天之前的旧数据（用于每日报告维护）"""
    conn = get_db_conn()
    c = conn.cursor()
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        c.execute("DELETE FROM decisions WHERE ts < ?", (cutoff,))
        c.execute("DELETE FROM trades WHERE close_ts < ?", (cutoff,))
        # 清理过期的待处理订单（7天以上未处理的）
        old_ord_cutoff = (datetime.now(UTC) - timedelta(days=7)).isoformat()
        c.execute("DELETE FROM pending_orders WHERE created_ts < ?", (old_ord_cutoff,))
        conn.commit()
        log.debug(f"[cleanup] 已清理 {days} 天前的旧数据及 7 天前的待处理订单")
    except Exception as e:
        log.debug(f"[cleanup] 清理旧数据失败: {e}")

# ============================================================
# RAG：基于指标相似度检索历史交易案例
# ============================================================
def retrieve_similar_failures(current_rsi: float, current_trend: str,
                               current_bb_pct: float = 0.5,
                               current_side: str = "",
                               symbol: str = "ETH-USDT-SWAP",
                               n: int = 2,
                               current_market_mode: str = "",
                               current_atr: float = 0,
                               hist_atrs: list = None,
                               current_ma_alignment: int = 0) -> list:
    """
    增强版 RAG：同时检索失败和盈利案例，引入随机性防过拟合。
    引入 [ATR分位, RSI区间, 均线排列] 三元组相似度。
    """
    if n <= 0:
        return []
    import random as _random
    from position_exec import _get_atr_quantile
    from market import _get_rsi_interval

    # ── 动态案例池模式 ─────────
    if CFG.enable_auto_case_pool:
        try:
            current_atr_q = _get_atr_quantile(current_atr, hist_atrs or [])
            rsi_lo, rsi_hi = max(0, current_rsi - 12), min(100, current_rsi + 12)
            cutoff = (datetime.now(UTC) - timedelta(days=getattr(CFG, 'case_max_age_days', 90))).isoformat()
            conn = get_db_conn()
            c = conn.cursor()
            c.execute(
                """SELECT timestamp, direction, entry_rsi, entry_bb_pct, entry_atr_pct,
                          market_mode, actual_result, pnl_pct, quality_score, core_reason,
                          ai_decision_reason
                   FROM historical_cases
                   WHERE is_approved = 1
                     AND timestamp >= ?
                     AND entry_rsi BETWEEN ? AND ?
                   ORDER BY created_at DESC
                   LIMIT 60""",
                (cutoff, rsi_lo, rsi_hi)
            )
            rows = c.fetchall()
            if not rows:
                return []
            scored = []
            for r in rows:
                ts, direction, e_rsi, e_bb, e_atr, mode, result, pnl, q_score, core, reason = r
                bb_diff = abs(current_bb_pct - (e_bb or 0.5))
                if bb_diff > 0.15:
                    continue
                mode_match = 1 if (current_market_mode and mode == current_market_mode) else 0
                try:
                    age_days = (datetime.now(UTC) - datetime.fromisoformat(ts.replace("Z", "+00:00"))).days
                    time_w = max(0.3, 1.0 - age_days / 90)
                except Exception:
                    time_w = 0.5
                score = (q_score or 5.0) * (0.5 + 0.5 * mode_match) * time_w
                scored.append({
                    "ts":          str(ts)[:10],
                    "reason":      (reason or core or "")[:80],
                    "confidence":  0.5,
                    "pnl_pct":     pnl or 0.0,
                    "hist_rsi":    e_rsi or 50,
                    "hist_bb":     e_bb or 0.5,
                    "score":       score,
                    "fail_reason": core or "",
                    "is_win":      result == "win",
                    "market_mode": mode or "趋势",
                })
            scored.sort(key=lambda x: x["score"], reverse=True)
            wins  = [s for s in scored if s["is_win"]][:getattr(CFG, 'rag_win_cases', 1)]
            losses = [s for s in scored if not s["is_win"]]
            if losses:
                top = min(3, len(losses))
                losses = _random.sample(losses[:top * 2], min(n, top))
            return wins[:getattr(CFG, 'rag_win_cases', 1)] + losses[:n]
        except Exception as e:
            log.debug(f"[案例池] RAG查询失败，fallback到旧逻辑: {e}")

    # ── 旧模式：基于 decisions + trades 表 ─────────
    noise = getattr(CFG, 'rag_rsi_noise', 3)
    perturbed_rsi = current_rsi + _random.uniform(-noise, noise) if noise > 0 else current_rsi
    current_atr_quantile = _get_atr_quantile(current_atr, hist_atrs or [])
    current_rsi_interval = _get_rsi_interval(perturbed_rsi)
    cutoff = (datetime.now(UTC) - timedelta(days=getattr(CFG, 'rag_max_age_days', 60))).isoformat()

    def _query_cases(pnl_condition: str, limit: int) -> list:
        try:
            conn = get_db_conn()
            c    = conn.cursor()
            c.execute(f'''
                SELECT d.ts, d.reason, d.confidence, d.features,
                       t.pnl_pct, t.close_ts, t.side, t.fail_reason
                FROM decisions d
                JOIN trades t ON d.id = t.decision_id
                WHERE {pnl_condition}
                  AND d.features IS NOT NULL
                  AND d.symbol = ?
                  AND t.close_ts >= ?
                ORDER BY d.ts DESC
                LIMIT 500
            ''', (symbol, cutoff))
            rows = c.fetchall()
            if len(rows) > limit:
                rows = list(_random.sample(rows, limit))
            return rows
        except Exception as e:
            log.debug(f"RAG 查询失败: {e}")
            return []

    def _score_rows(rows: list, n_keep: int, is_win: bool) -> list:
        bb_tol = 0.1
        atr_quantile_tol = 0.2
        candidates = []
        for row in rows:
            try:
                feats      = json.loads(row[3]) if row[3] else {}
                ind        = feats.get("ind_15m", {})
                hist_rsi   = float(ind.get("rsi",    50))
                hist_bb    = float(ind.get("bb_pct",  0.5))
                hist_trend = str(ind.get("trend",    ""))
                hist_side  = str(row[6] or "")
                hist_atr_quantile = float(feats.get("atr_quantile", 0.5))
                hist_rsi_interval = int(feats.get("rsi_interval", 1))
                hist_ma_alignment = int(feats.get("ma_alignment", 0))

                rsi_diff = abs(perturbed_rsi - hist_rsi)
                rag_rsi_tolerance = getattr(CFG, 'rag_rsi_tolerance', 12)
                if rsi_diff > rag_rsi_tolerance:
                    continue
                bb_diff = abs(current_bb_pct - hist_bb)
                if bb_diff > bb_tol:
                    continue
                atr_quantile_diff = abs(current_atr_quantile - hist_atr_quantile)
                if atr_quantile_diff > atr_quantile_tol:
                    continue
                if getattr(CFG, 'rag_require_same_trend', False) and hist_trend and hist_trend != current_trend:
                    continue
                if current_side and hist_side and hist_side != current_side:
                    continue

                score = (1.0 - rsi_diff / rag_rsi_tolerance) * 0.3 + \
                        (1.0 - bb_diff  / bb_tol)                * 0.2 + \
                        (1.0 - atr_quantile_diff / atr_quantile_tol) * 0.2
                if hist_rsi_interval == current_rsi_interval:
                    score += 0.1
                if hist_ma_alignment == current_ma_alignment:
                    score += 0.1
                hist_bb_width = float(feats.get("bb_width", 0.05))
                hist_mode = "震荡" if hist_bb_width < CFG.osc_bb_width_thresh else "趋势"
                if current_market_mode and hist_mode == current_market_mode:
                    score = min(1.0, score + 0.1)
                try:
                    hist_ts = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
                    age_days = (datetime.now(UTC) - hist_ts).total_seconds() / 86400
                    if age_days <= 7:
                        time_weight = 1.2
                    elif age_days <= 30:
                        time_weight = 1.0
                    else:
                        time_weight = 0.8
                    score *= time_weight
                except Exception:
                    pass
                candidates.append({
                    "ts":          row[0][:10],
                    "reason":      str(row[1] or ""),
                    "confidence":  row[2],
                    "pnl_pct":     row[4],
                    "hist_rsi":    hist_rsi,
                    "hist_bb":     hist_bb,
                    "score":       score,
                    "fail_reason": str(row[7] or "").strip(),
                    "is_win":      is_win,
                    "market_mode": hist_mode,
                })
            except Exception:
                continue
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:n_keep]

    loss_rows = _query_cases("t.pnl < 0", 300)
    win_rows  = _query_cases("t.pnl > 0", 300) if getattr(CFG, 'rag_win_cases', 1) > 0 else []
    losses = _score_rows(loss_rows, n,                                is_win=False)
    wins   = _score_rows(win_rows,  getattr(CFG, 'rag_win_cases', 1), is_win=True)
    return losses + wins


def build_rag_warning(similar_failures: list, current_rsi: float, current_bb_pct: float = 0.5) -> str:
    """
    格式化 RAG 检索结果为 AI 可直接理解的「历史参考」。
    失败案例：风险提示语气；盈利案例：正向强化语气。
    """
    if not similar_failures:
        return ""
    losses = [f for f in similar_failures if not f.get("is_win")]
    wins   = [f for f in similar_failures if f.get("is_win")]
    lines  = ["[历史相似市场环境参考（RSI+BB%+波动率）]"]
    if losses:
        lines.append("▼ 类似条件下的失败案例（仅供参考，非硬性禁止）:")
        for i, f in enumerate(losses, 1):
            core = f["fail_reason"] if f["fail_reason"] else (f["reason"][:50] if f["reason"] else "原因未记录")
            mode_tag = f"[{f.get('market_mode','?')}市]" if f.get('market_mode') else ""
            lines.append(
                f"  止损{i}{mode_tag}[{f['ts']}] RSI={f['hist_rsi']:.1f}(当前{current_rsi:.1f}) "
                f"BB%={f['hist_bb']:.2f}(当前{current_bb_pct:.2f}) "
                f"结果={f['pnl_pct']*100:+.1f}% → {core}"
            )
        lines.append(
            "  ⚠️ 如当前1H/4H趋势强劲+盘口买压明显，允许忽略上述历史止损记录顺势开仓。"
        )
    if wins:
        lines.append("▲ 类似条件下的盈利案例（正向参考）:")
        for i, f in enumerate(wins, 1):
            core = f["reason"][:50] if f["reason"] else "原因未记录"
            lines.append(
                f"  盈利{i}[{f['ts']}] RSI={f['hist_rsi']:.1f} "
                f"BB%={f['hist_bb']:.2f} 结果={f['pnl_pct']*100:+.1f}% → {core}"
            )
        lines.append("  ✅ 以上盈利案例可作为入场信心参考，但需验证当前指标与彼时一致。")
    return "\n".join(lines)



# 全局 DatabaseManager 单例在类定义后实例化（见文件末尾）

def get_db_manager() -> "DatabaseManager":
    """获取全局 DatabaseManager 实例"""
    return _db_manager

def get_recent_win_rate(n: int = 25, min_sample: int = 8) -> tuple:
    """
    计算最近 N 笔已平仓交易的胜率。
    返回 (win_rate, sample_count)
    """
    conn = get_db_conn()
    c = conn.cursor()
    c.execute(
        "SELECT pnl FROM trades WHERE pnl IS NOT NULL ORDER BY close_ts DESC LIMIT ?",
        (n,)
    )
    rows = c.fetchall()
    if not rows or len(rows) < min_sample:
        return 0.5, len(rows)
    wins = sum(1 for r in rows if r[0] is not None and r[0] > 0)
    return wins / len(rows), len(rows)


# ── 交易复盘模块 ─────────────────────────────────────────────────────────────

class TradePostmortem:
    def __init__(self, client):
        self.client   = client
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="postmortem")

    def trigger(self, trade_snapshot: Dict):
        """异步触发复盘，不阻塞主交易线程"""
        self._executor.submit(self._run, trade_snapshot)

    def _build_context(self, snap: Dict) -> str:
        """从数据库拉取完整上下文，构建复盘 Prompt"""
        decision_id = snap.get("decision_id")
        trade_id    = snap.get("trade_id")

        # 拉取开仓决策
        open_decision = {}
        if decision_id:
            try:
                conn = get_db_conn()
                c    = conn.cursor()
                c.execute(
                    "SELECT thought_process, action, confidence, reason, features, ts "
                    "FROM decisions WHERE id=?", (decision_id,)
                )
                row = c.fetchone()
                if row:
                    open_decision = {
                        "thought_process": row[0] or "",
                        "action":          row[1],
                        "confidence":      row[2],
                        "reason":          row[3],
                        "features":        json.loads(row[4]) if row[4] else {},
                        "open_ts":         row[5],
                    }
            except Exception as e:
                log.debug(f"[复盘] 查询开仓决策失败: {e}")

        # 拉取持仓期间的决策序列（hold/adjust_sl_tp）
        mid_decisions: List[Dict] = []
        if open_decision.get("open_ts") and snap.get("close_ts"):
            try:
                open_dt  = _parse_dt(open_decision["open_ts"])
                close_dt = _parse_dt(snap["close_ts"])
                if open_dt and close_dt:
                    mid_decisions = get_decisions_in_range(open_dt, close_dt)
            except Exception as e:
                log.debug(f"[复盘] 查询持仓期间决策失败: {e}")

        # 提取技术指标特征
        feats = open_decision.get("features", {})
        ind   = feats.get("ind_15m", {})
        depth = feats.get("depth", {})

        context = f"""== 本次交易基本信息 ==
方向: {snap['side']}
开仓价: {snap['entry']:.2f}  平仓价: {snap['exit']:.2f}
盈亏: {snap['pnl_pct']*100:+.2f}%  持仓时长: {snap.get('holding_minutes', 0):.0f} 分钟
平仓原因: {snap['close_reason']}

== 开仓时刻决策 ==
AI 置信度: {open_decision.get('confidence', 0):.2f}
开仓理由: {open_decision.get('reason', '未知')}
AI 思考过程: {(open_decision.get('thought_process') or '无')[:300]}

== 开仓时刻市场指标（15m） ==
RSI={ind.get('rsi', 0):.2f}  MACD={ind.get('macd_hist', 0):.4f}  BB%={ind.get('bb_pct', 0):.2f}
ATR={ind.get('atr', 0):.2f}  成交量异常倍数={ind.get('vol_surge', 0):.2f}
趋势={ind.get('trend', '?')}  背离={ind.get('divergence', '?')}
支撑={ind.get('support', 0):.2f}  阻力={ind.get('resistance', 0):.2f}
资金费率={feats.get('funding_rate', 0)*100:.4f}%
恐贪指数={feats.get('fg_index', 50)}

== 盘口（开仓时） ==
买卖失衡={depth.get('imbalance', 0):.3f}  价差={depth.get('spread', 0):.2f}"""

        if mid_decisions:
            mid_summary = []
            for d in mid_decisions[-6:]:   # 最多展示最后6条，避免 Prompt 过长
                mid_summary.append(
                    f"  {d['ts'][11:16]} {d['action']}(conf={d['confidence']:.2f}): {d['reason'][:60]}"
                )
            context += "\n\n== 持仓期间 AI 决策序列 ==\n" + "\n".join(mid_summary)

        return context

    def _run(self, snap: Dict):
        """实际执行复盘的工作函数（后台线程，使用 deepseek-chat 深度推理）"""
        try:
            log.info(f"🔍 [Chat复盘] 开始复盘 trade_id={snap.get('trade_id')} "
                     f"side={snap['side']} pnl={snap['pnl_pct']*100:+.2f}%")

            context    = self._build_context(snap)
            user_prompt = f"""{context}

请对本次交易进行深度复盘分析，重点推导因果链，给出可执行的改进建议（中文，400字以内）："""

            # ── 使用 deepseek-chat 复盘（异步，不阻塞交易线程）──────
            review = _call_chat(
                self.client,
                messages=[
                    {"role": "system", "content": _POSTMORTEM_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=800,
                timeout=120,
            )

            pnl_emoji = "🟢" if snap['pnl_pct'] >= 0 else "🔴"
            summary = (
                f"{pnl_emoji} 交易复盘 | {snap['side']} | "
                f"{snap['pnl_pct']*100:+.2f}% | {snap['close_reason']}\n"
                f"{'─'*40}\n"
                f"{review}"
            )

            # 写入独立复盘日志文件
            postmortem_log_path = "trade_postmortem.log"
            with open(postmortem_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"[{datetime.now(UTC).isoformat()}]\n")
                f.write(f"开仓:{snap['entry']:.2f} 平仓:{snap['exit']:.2f} "
                        f"盈亏:{snap['pnl_pct']*100:+.2f}%\n")
                f.write(f"平仓原因: {snap['close_reason']}\n")
                f.write(f"{'─'*40}\n{review}\n")

            log.info(f"📋 [复盘完成]\n{summary}")
            ai_log.info(json.dumps({
                "type": "postmortem",
                "trade": snap,
                "review": review
            }, ensure_ascii=False))

            # Webhook 推送（消息较长，只发摘要前 200 字）
            webhook_msg = (
                f"{'🟢盈利' if snap['pnl_pct']>=0 else '🔴亏损'} "
                f"{snap['side']} {snap['pnl_pct']*100:+.2f}%\n"
                f"原因: {snap['close_reason']}\n\n"
                f"{review[:200]}{'...' if len(review)>200 else ''}"
            )
            _notify("📋 交易复盘", webhook_msg)

        except Exception as e:
            log.error(f"[复盘] 执行失败: {e}")


# ── 每日战报模块 ─────────────────────────────────────────────────────────────

class DailyReportModule:
    def __init__(self, trader, logger, ai_client=None):
        self.trader = trader
        self.log    = logger
        self.client = ai_client   # Reasoner 战报归因（可选）

    def generate_24h_report(self):
        """抓取过去24小时的账单数据并生成报告（分页获取全部）"""
        try:
            self.log.info("📊 正在生成过去 24 小时交易报告...")
            now = datetime.now(UTC)
            yesterday = now - timedelta(days=1)
            after_ts = int(yesterday.timestamp() * 1000)

            all_bills = []
            after = None
            while True:
                bills_resp = self.trader.get_bills_archive(instType="SWAP", limit=100, after=after)
                if bills_resp.get("code") != "0":
                    self.log.error(f"获取账单失败: {bills_resp}")
                    break
                bills = bills_resp.get("data", [])
                if not bills:
                    break
                all_bills.extend(bills)
                if int(bills[-1]['ts']) < after_ts:
                    break
                after = bills[-1].get('billId')

            total_pnl = 0.0
            total_fee = 0.0
            trade_count = 0
            win_count = 0

            for bill in all_bills:
                bill_time = int(bill['ts'])
                if bill_time < after_ts:
                    continue
                if bill.get('type') in ['1', '2']:
                    pnl = float(bill.get('pnl', 0))
                    total_pnl += pnl
                    trade_count += 1
                    if pnl > 0:
                        win_count += 1
                fee = float(bill.get('fee', 0))
                total_fee += abs(fee)

            win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
            net_profit = total_pnl - total_fee
            # 修复除零：当毛盈亏接近0时，用极小值替代
            abs_pnl = abs(total_pnl) if abs(total_pnl) > 1e-6 else 1e-9
            fee_ratio = total_fee / abs_pnl

            trades = get_trades_in_range(yesterday, now)
            high_conf_trades = [t for t in trades if t.get('confidence', 0) >= 0.8]
            low_conf_trades  = [t for t in trades if t.get('confidence', 0) < 0.6]
            high_conf_win = sum(1 for t in high_conf_trades if t.get('pnl', 0) > 0)
            low_conf_win  = sum(1 for t in low_conf_trades  if t.get('pnl', 0) > 0)
            high_conf_rate = (high_conf_win / len(high_conf_trades) * 100) if high_conf_trades else 0
            low_conf_rate  = (low_conf_win  / len(low_conf_trades)  * 100) if low_conf_trades  else 0

            decisions = get_decisions_in_range(yesterday, now)
            keyword_counts = {}
            for dec in decisions:
                reason = dec.get("reason", "")
                for kw in ["周期分歧", "极端恐慌/恐惧", "极端贪婪", "盘口失衡", "资金费率"]:
                    if kw.replace("/", "") in reason.replace("/", "") or kw in reason:
                        keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

            # 手续费磨损告警（生死线监控）
            fee_warning = ""
            if total_fee > 0 and abs(total_pnl) > 0:
                fee_to_pnl = total_fee / abs(total_pnl)
                if fee_to_pnl > 0.5:
                    fee_warning = f"\n⚠️ 手续费磨损危险({fee_to_pnl*100:.0f}%)！建议提高OPEN_CONFIDENCE_MIN至0.65"
                elif fee_to_pnl > 0.3:
                    fee_warning = f"\n⚠️ 手续费磨损偏高({fee_to_pnl*100:.0f}%)，关注交易频率"

            # 更新全局胜率（用近25笔保持与 update_dynamic_params 口径一致）
            # 每日报告中的 win_rate 是从 OKX 账单统计的展示值，不直接写入全局状态
            # 全局 last_24h_win_rate 由 update_dynamic_params() 每小时维护（近25笔动态视窗）
            wr_recent, n_recent = get_recent_win_rate(n=25, min_sample=8)
            gs_set("last_24h_win_rate", wr_recent)

            report_msg = (
                f"\n{'='*10} 📅 每日战报 ({now.strftime('%m-%d')}) {'='*10}\n"
                f"💰 净 盈 亏: {net_profit:+.2f} USDT " + ("🚀" if net_profit > 0 else "📉") + "\n"
                f"📈 毛 盈 亏: {total_pnl:+.2f} USDT\n"
                f"💸 手 续 费: {total_fee:.2f} USDT\n"
                f"📊 资金损耗率: {fee_ratio*100:.2f}%{fee_warning}\n"
                f"🔢 交易次数: {trade_count} 次\n"
                f"🎯 胜    率: {win_rate:.2f}%\n"
                f"📈 高置信度(≥0.8)胜率: {high_conf_rate:.2f}% ({high_conf_win}/{len(high_conf_trades)})\n"
                f"📉 低置信度(<0.6)胜率: {low_conf_rate:.2f}% ({low_conf_win}/{len(low_conf_trades)})\n"
                f"🛡️ 连亏次数: {gs_get('consecutive_losses', 0)} 次\n"
                f"📝 AI关键词统计:\n"
            )
            for kw, cnt in keyword_counts.items():
                report_msg += f"   - {kw}: {cnt} 次\n"
            report_msg += f"{'='*32}"

            self.log.info(report_msg)
            _notify("📊 每日战报", report_msg)

            # ── Chat 归因分析（异步，不阻塞战报发送）─────────────────────────
            # 用 deepseek-chat 总结连亏原因，给出止损/杠杆调整建议（战报级，无需 Reasoner）
            if trade_count > 0 and self.client:
                def _reasoner_analysis():
                    try:
                        loss_trades = [t for t in trades if t.get('pnl', 0) < 0]
                        # 截断保护：最多取最近5笔，每行限80字，防止超长Prompt淹没核心指令
                        loss_lines = [
                            f"  {t.get('close_ts', '')[:16]} {t.get('side', '')} 入:{t.get('entry', 0):.4f} "
                            f"出:{t.get('exit', 0):.4f} 亏:{t.get('pnl_pct', 0)*100:+.2f}% conf:{t.get('confidence', 0):.2f}"
                            for t in loss_trades[-5:]
                        ]
                        loss_summary = "\n".join(loss_lines) or "  无亏损交易"
                        # 关键词统计截断：最多10个词
                        kw_str = ", ".join(
                            f"{k}×{v}" for k, v in list(keyword_counts.items())[:10]
                        ) or "无"

                        analysis_prompt = f"""以下是过去24小时的交易统计：
净盈亏:{net_profit:+.2f}U 胜率:{win_rate:.1f}% 手续费:{total_fee:.2f}U
高置信度胜率:{high_conf_rate:.1f}% 低置信度胜率:{low_conf_rate:.1f}%
连亏次数:{gs_get('consecutive_losses', 0)}

最近亏损交易明细：
{loss_summary}

AI决策关键词频率：{kw_str}

请通过逻辑推理分析：
1. 今日亏损的底层原因是什么（趋势变化？流动性恶化？某品种特殊波动？）
2. 高/低置信度的胜率差异说明了什么？
3. 给出明天可直接执行的1-2条参数或策略调整建议（如：提高最低置信度阈值、调整SL倍数等）
请简洁推理，中文，300字以内。"""

                        insight = _call_chat(
                            self.client,
                            messages=[{"role": "user", "content": analysis_prompt}],
                            max_tokens=600,
                            timeout=120,
                        )
                        self.log.info(f"🧠 [Chat战报归因]\n{insight}")
                        _notify("🧠 Chat战报归因", insight)
                    except Exception as e:
                        self.log.debug(f"Chat战报归因失败: {e}")

                threading.Thread(target=_reasoner_analysis, daemon=True,
                                 name="chat-daily").start()

            cleanup_old_data(30)

            # ── 每周一：Reasoner 生成参数调整建议（自我进化闭环）──────────────
            # 读取过去7天数据，让 Reasoner 分析并给出具体的 .env 参数建议
            now_local = datetime.now(UTC)
            if now_local.weekday() == 0 and self.client:  # 0 = Monday
                last_weekly = get_sys_config("last_weekly_reasoner")
                this_week   = now_local.strftime("%Y-W%W")
                if last_weekly != this_week:
                    set_sys_config("last_weekly_reasoner", this_week)
                    def _weekly_evolution():
                        try:
                            week_start = now_local - timedelta(days=7)
                            w_trades   = get_trades_in_range(week_start, now_local)
                            if len(w_trades) < 5:
                                return
                            w_wins   = [t for t in w_trades if t.get('pnl', 0) > 0]
                            w_loss   = [t for t in w_trades if t.get('pnl', 0) < 0]
                            win_rate_w = len(w_wins) / len(w_trades) * 100
                            avg_win  = sum(t.get('pnl_pct', 0) for t in w_wins)  / max(len(w_wins), 1) * 100
                            avg_loss = sum(t.get('pnl_pct', 0) for t in w_loss)  / max(len(w_loss), 1) * 100

                            prompt = f"""以下是过去7天的量化交易系统运行数据：
交易总笔数: {len(w_trades)} | 胜率: {win_rate_w:.1f}%
平均盈利: {avg_win:+.2f}% | 平均亏损: {avg_loss:+.2f}%
当前关键参数:
  OKX_CONFIDENCE_THRESHOLD={CFG.confidence_thresh}
  OPEN_CONFIDENCE_MIN={CFG.open_confidence_min}
  SL_ATR_MULT={CFG.sl_atr_mult}
  TP_RR_RATIO={CFG.tp_rr_ratio}
  MAX_LEVERAGE={CFG.max_leverage}
  RISK_PER_TRADE={CFG.risk_per_trade}
  RAG_RSI_TOLERANCE={CFG.rag_rsi_tolerance}

请给出3-5条具体参数调整建议。
必须以如下JSON格式输出（方便系统解析），每条建议一个对象：
[
  {{"param": "SL_ATR_MULT", "old": "2.0", "new": "2.5", "reason": "本周平均亏损偏大，止损距离需放宽"}},
  ...
]
只输出JSON数组，不要其他文字。"""

                            raw = _call_reasoner(
                                self.client,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=600, timeout=120,
                            )

                            # 用健壮解析器替代脆弱的正则
                            suggestions = _parse_llm_json_array(raw)

                            submitted = 0
                            for s in suggestions:
                                if isinstance(s, dict) and all(k in s for k in ("param","old","new","reason")):
                                    from config import submit_pending_config
                                    ok = submit_pending_config(
                                        s["param"], s["old"], s["new"], s["reason"]
                                    )
                                    if ok:
                                        submitted += 1

                            # 格式化为可读摘要推送 Webhook
                            from config import get_pending_configs
                            pending = get_pending_configs()
                            summary_lines = [f"🧬 Reasoner 本周进化建议（{submitted}条待审批）：\n"]
                            for p in pending[-5:]:
                                summary_lines.append(
                                    f"  [{p['id']}] {p['key']}: {p['old']} → {p['new']}\n"
                                    f"       理由: {p['reason'][:80]}\n"
                                )
                            summary_lines.append(
                                "\n✅ 审批: POST /config/approve?id=N\n"
                                "❌ 拒绝: POST /config/reject?id=N"
                            )
                            summary = "".join(summary_lines)

                            self.log.info(f"🧬 [每周进化建议]\n{summary}")
                            _notify("🧬 每周参数进化建议", summary)
                            with open("weekly_evolution.log", "a", encoding="utf-8") as f:
                                f.write(f"\n{'='*50}\n[{now_local.isoformat()}]\n{raw}\n")
                        except Exception as e:
                            self.log.debug(f"每周Reasoner进化失败: {e}")
                    threading.Thread(target=_weekly_evolution, daemon=True,
                                     name="weekly-evolution").start()

            return report_msg

        except Exception as e:
            self.log.error(f"❌ 生成报告失败: {e}")
            return None


# ── JSON 解析辅助函数（供每周进化使用）────────────────────────────────────

def _clean_json_text(raw_text: str) -> str:
    """剥离 Markdown 围栏，返回纯文本"""
    import re
    return re.sub(r'```(?:json)?\s*', '', raw_text).replace('```', '').strip()

def _parse_llm_json_array(raw_text: str) -> list:
    """
    健壮解析 LLM 输出的 JSON 数组（替代脆弱的正则 r'[.*]'）。
    处理 Reasoner 常见的 Markdown 包裹（```json [...] ```）和额外文字。
    兼容 deepseek-reasoner 的 <think>...</think> 标签包裹，自动剥离后解析。
    返回 list，失败返回 []。
    """
    cleaned = _clean_json_text(raw_text)
    # 剥离 <think>...</think> 包裹（Reasoner 输出结构）
    _think_s = cleaned.find("████")
    _think_e = cleaned.find("</think>")
    if _think_s != -1 and _think_e != -1 and _think_e > _think_s:
        cleaned = cleaned[_think_e + 8:].strip()

    # 步骤2：先尝试整段直接 parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    # 步骤3：找第一个 [ 到最后一个 ] 之间的内容（健壮版）
    start = cleaned.find('[')
    end   = cleaned.rfind(']')
    if start != -1 and end > start:
        try:
            result = json.loads(cleaned[start:end+1])
            if isinstance(result, list):
                return result
        except Exception:
            pass

    # 步骤4：使用 raw_decode 逐步解析，正确处理嵌套花括号
    items = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(cleaned):
        # 跳过非 JSON 字符
        start = cleaned.find('{', pos)
        if start == -1:
            break
        try:
            obj, end_pos = decoder.raw_decode(cleaned, start)
            if isinstance(obj, dict):
                items.append(obj)
            pos = end_pos  # raw_decode 返回的是 clean 中 obj 结束处的绝对索引
        except json.JSONDecodeError:
            pos = start + 1
    return items


# ============================================================
# DatabaseManager — 从 state.py 迁移（队列式异步写入）
# ============================================================
class DatabaseManager:
    """
    数据库管理器。
    - 写入：enqueue → 后台写线程串行处理（消除 SQLite 写锁冲突）
    - 读取：线程本地连接（WAL 模式下读写可并发）
    - busy_timeout=30s：写线程内遇到锁时自动等待，不失败
    """
    __slots__ = ('_write_queue', '_writer_thread', '_write_conn', '_init_event')

    def __init__(self):
        import queue
        self._write_queue: queue.Queue = queue.Queue(maxsize=5000)
        self._write_conn = None  # 在 _writer_loop 线程内创建
        self._init_event = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="db-writer")
        self._writer_thread.start()
        # 等待 writer 线程完成表初始化
        self._init_event.wait(timeout=10)

    def _writer_loop(self):
        import queue
        import sqlite3

        # 在当前线程内创建连接，避免跨线程使用
        self._write_conn = _create_db_conn()
        self._write_conn.execute("PRAGMA busy_timeout = 30000")

        # === 全自动动态案例池：启动时自动建表（IF NOT EXISTS） ===
        self._write_conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                pnl_usd REAL NOT NULL,
                hold_minutes INTEGER NOT NULL,
                market_mode TEXT,
                entry_rsi REAL,
                entry_bb_pct REAL,
                entry_atr_pct REAL,
                exit_rsi REAL,
                exit_bb_pct REAL,
                exit_atr_pct REAL,
                ai_confidence REAL,
                ai_decision_reason TEXT,
                actual_result TEXT NOT NULL,
                quality_score REAL,
                core_reason TEXT,
                is_approved INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        self._write_conn.commit()
        log.info("historical_cases 表已创建/已存在（全自动案例池）")

        # 动态案例池：为 trades 表补充缺失列（兼容已有数据库）
        _existing_trades = {row[1] for row in self._write_conn.execute("PRAGMA table_info(trades)")}
        for _col, _ctype in [
            ("entry_market_mode",  "TEXT"),
            ("entry_rsi",          "REAL"),
            ("entry_bb_pct",       "REAL"),
            ("entry_atr_pct",      "REAL"),
            ("ai_confidence",      "REAL"),
            ("ai_reason",         "TEXT"),
            ("exit_market_mode",   "TEXT"),
        ]:
            if _col not in _existing_trades:
                try:
                    self._write_conn.execute(
                        f"ALTER TABLE trades ADD COLUMN {_col} {_ctype}"
                    )
                    self._write_conn.commit()
                    log.debug(f"[DB MIGRATION] trades 表已补充 {_col} 列")
                except sqlite3.OperationalError:
                    pass

        # 通知 __init__ 初始化完成
        self._init_event.set()

        while True:
            try:
                item = self._write_queue.get(timeout=1)
                if item is None:
                    break
                sql, params = item
                for attempt in range(3):
                    try:
                        self._write_conn.execute(sql, params)
                        self._write_conn.commit()
                        break
                    except sqlite3.OperationalError as db_e:
                        if attempt < 2:
                            time.sleep(0.1 * (attempt + 1))
                        else:
                            log.error(f"DB写入最终失败(3次重试): {db_e} | SQL: {sql[:80]}")
                    except Exception as db_e:
                        log.error(f"DB写入异常: {db_e} | SQL: {sql[:80]}")
                        break
            except queue.Empty:
                pass
            except Exception as e:
                log.warning(f"DB写线程顶层异常: {e}")

    def write(self, sql: str, params: tuple = ()) -> bool:
        import queue
        try:
            self._write_queue.put_nowait((sql, params))
            return True
        except queue.Full:
            log.warning(f"DB写队列满，跳过: {sql[:50]}")
            return False

    def write_sync(self, sql: str, params: tuple = (), max_retries: int = 3) -> bool:
        import sqlite3
        for attempt in range(max_retries):
            try:
                conn = get_db_conn()
                conn.execute(sql, params)
                conn.commit()
                return True
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                log.warning(f"DB同步写入失败(attempt {attempt+1}): {e}")
        return False

    def read(self, sql: str, params: tuple = ()) -> list:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute(sql, params)
        return c.fetchall()

    def close(self):
        self._write_queue.put_nowait(None)
        self._writer_thread.join(timeout=5)


# 全局 DatabaseManager 单例（WAL 模式 + 队列写入）
_db_manager: DatabaseManager = DatabaseManager()


# ============================================================
# PositionManager — 从 state.py 迁移（仓位意图唯一写入者）
# ============================================================
class PositionManager:
    """
    仓位状态的唯一写入者。
    - 读取：任何模块可调用 get_pos() 获取只读快照
    - 写入：通过 submit(intent) 提交变更意图，由本类统一验证后执行
    - 好处：所有仓位变更有唯一入口，lock 只需在这里管理
    """

    def __init__(self, pos: "Position", lock: threading.RLock):
        self._pos  = pos
        self._lock = lock
        self._reset_callback = None

    def get_pos(self) -> "Position":
        return self._pos

    def submit(self, intent: PositionIntent) -> bool:
        with self._lock:
            t   = intent.intent_type
            p   = intent.payload
            src = intent.source

            if not self._pos.side and t in (
                PositionIntentType.UPDATE_SL,
                PositionIntentType.UPDATE_TP,
                PositionIntentType.UPDATE_PEAK,
            ):
                log.warning(f"[PosMgr] 空仓时收到 {t.value} intent，src={src}，已忽略")
                return False

            if t == PositionIntentType.RESET:
                if self._reset_callback:
                    self._reset_callback()
                return True

            if t == PositionIntentType.UPDATE_SL:
                new_sl = p.get("sl")
                if new_sl and new_sl != self._pos.stop_loss:
                    old = self._pos.stop_loss
                    self._pos.stop_loss = new_sl
                    log.debug(f"[PosMgr] UPDATE_SL | {old:.4f} → {new_sl:.4f} | src={src}")
                return True

            if t == PositionIntentType.UPDATE_TP:
                new_tp = p.get("tp")
                if new_tp is not None:
                    old = self._pos.take_profit
                    self._pos.take_profit = new_tp
                    log.debug(f"[PosMgr] UPDATE_TP | {old:.4f} → {new_tp:.4f} | src={src}")
                return True

            if t == PositionIntentType.UPDATE_PEAK:
                new_peak = p.get("peak_price")
                if new_peak is not None:
                    old = self._pos.peak_price
                    self._pos.peak_price = new_peak
                    log.debug(f"[PosMgr] UPDATE_PEAK | {old:.4f} → {new_peak:.4f} | src={src}")
                active = p.get("trailing_active")
                if active is not None:
                    self._pos.trailing_active = active
                    log.debug(f"[PosMgr] TRAILING_ACTIVE → {active} | src={src}")
                return True

            if t == PositionIntentType.SYNC_FROM_EXCHANGE:
                for k, v in p.items():
                    if hasattr(self._pos, k):
                        old = getattr(self._pos, k)
                        setattr(self._pos, k, v)
                        log.debug(f"[PosMgr] SYNC {k} | {old} → {v} | src={src}")
                return True

            log.warning(f"[PosMgr] 未知 intent_type={t}, src={src}")
            return False

    def set_reset_callback(self, cb):
        self._reset_callback = cb