"""
exchange.py — OKX 交易所接口、网络层、量能检测
"""
import os, time, threading, logging, hmac, hashlib, base64, requests, json, queue, socket, websocket
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from functools import wraps
from collections import deque
from urllib.parse import urlencode
import math

from common import log, CFG, gs_get, gs_set, gs_add
from core import gs_update, UTC

# ── OKX 错误码分类 ──────────────────────────────────────────────────────────
_RATE_LIMIT_CODES   = {"50011", "50013"}
_AUTH_ERROR_CODES   = {"50001", "50002", "50004", "50005", "50006", "50007"}
_PARAM_ERROR_CODES  = {"50008", "50009", "50010", "50014", "50015"}
_ORDER_ERROR_CODES  = {"51000", "51001", "51002", "51006", "51008",
                       "51010", "51020", "51023", "51100", "51110"}

# ── API 自愈标志（供 run_once 检测并执行修复）────────────────────────────
_api_need_heal = {"flag": False, "reason": "", "ts": 0.0}

_SERVER_ERROR_CODES = {"50000", "50001", "50004", "50025", "50026"}

def classify_okx_error(code: str) -> str:
    if code in _RATE_LIMIT_CODES:  return "rate_limit"
    if code in _AUTH_ERROR_CODES:  return "auth_error"
    if code in _PARAM_ERROR_CODES: return "param_error"
    if code in _ORDER_ERROR_CODES: return "order_error"
    if code in _SERVER_ERROR_CODES: return "server_error"
    return "unknown"

def _safe_float(val, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

# ── API限流器（滑动窗口）───────────────────────────────────────────────────
class RateLimiter:
    """滑动窗口限流器，限制每秒请求数"""
    def __init__(self, rate: int):
        self.rate = rate
        self.tokens = rate
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self, block: bool = True) -> bool:
        """
        获取令牌（线程安全）
        
        Args:
            block: 是否阻塞等待令牌
            
        Returns:
            bool: 是否成功获取令牌
            
        Note:
            令牌桶算法：每秒生成 rate 个令牌，上限为 rate 个
            非阻塞模式：令牌不足时立即返回 False
            阻塞模式：等待直到令牌可用
        """
        while True:
            try:
                with self.lock:
                    now = time.time()
                    elapsed = now - self.last_update
                    self.tokens += elapsed * self.rate
                    if self.tokens > self.rate:
                        self.tokens = self.rate
                    self.last_update = now
                    if self.tokens >= 1:
                        self.tokens -= 1
                        return True
                    if not block:
                        return False
                    sleep_time = (1 - self.tokens) / self.rate
                time.sleep(sleep_time)
            except KeyboardInterrupt:
                # 允许用户中断等待
                return False
            except Exception:
                # 防御性兜底：异常时返回 False，避免阻塞调用方
                # 日志由调用方记录，避免日志风暴
                return False

    def get_fill_ratio(self) -> float:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            current_tokens = min(self.rate, self.tokens + elapsed * self.rate)
            return current_tokens / self.rate

# 全局限流器
_public_limiter = RateLimiter(CFG.api_rate_limit_public)
_private_limiter = RateLimiter(CFG.api_rate_limit_private)

def rate_limited(is_private: bool = False):
    """限流装饰器，区分公共和私有接口"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = _private_limiter if is_private else _public_limiter
            limiter.acquire(block=True)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def retry_with_backoff(max_retries: int = 3, is_private: bool = False):
    """
    精化重试策略 + API 自我修复触发
    """
    global _api_need_heal

    def _trigger_heal(reason: str):
        _api_need_heal["flag"] = True
        _api_need_heal["reason"] = reason
        _api_need_heal["ts"] = time.time()
        log.error(f"[API自愈触发] {reason}，将在下一轮 run_once 中执行状态修复")

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = _private_limiter if is_private else _public_limiter
            limiter.acquire(block=True)

            last_result = None
            for i in range(max_retries):
                try:
                    res = func(*args, **kwargs)
                    last_result = res
                    if isinstance(res, dict):
                        code = res.get("code", "0")
                        if code != "0":
                            err_type = classify_okx_error(code)
                            if err_type == "rate_limit":
                                wait = 2 ** i
                                log.warning(f"OKX 频率限制({code})，等待 {wait}s 后重试")
                                time.sleep(wait)
                                continue
                            elif err_type in ("auth_error", "param_error", "order_error"):
                                log.error(f"OKX 不可重试错误 [{err_type}]({code})，立即返回")
                                return res
                            elif err_type == "server_error":
                                wait = 2 ** i
                                log.warning(f"OKX 服务器错误({code})，等待 {wait}s 后重试")
                                time.sleep(wait)
                                continue
                    return res
                except Exception as e:
                    if i == max_retries - 1:
                        log.error(f"请求最终失败，已重试{max_retries}次: {e}")
                        _trigger_heal(f"网络异常重试耗尽: {str(e)[:80]}")
                        return last_result if last_result is not None else {"code": "-1", "msg": str(e)}
                    log.warning(f"请求网络异常，第{i+1}次重试: {e}")
                    time.sleep(2 ** i)
            _trigger_heal("API请求重试耗尽（rate_limit/5xx/网络）")
            return last_result if last_result is not None else {"code": "-1", "msg": "max retries exceeded"}
        return wrapper
    return decorator


# ============================================================
# VolumeSpikeDetector — 秒级成交量突增检测
# ============================================================
class VolumeSpikeDetector:
    _WINDOW_SECS      = 10
    _BASELINE_WINDOWS = 30
    _COOLDOWN_SECS    = 60
    _MIN_BASELINE_VOL = 5.0
    _SPIKE_PERSIST_SECS = 90.0

    def __init__(self):
        self._buf: deque = deque(maxlen=self._BASELINE_WINDOWS + 2)
        self._cur_bucket: int = 0
        self._cur_buy:    float = 0.0
        self._cur_sell:   float = 0.0
        self._lock = threading.Lock()
        self._last_spike_ts: float = 0.0
        self._spike_peak_ts:   float = 0.0
        self._spike_peak_data: Dict  = {}
        self._cum_buy:  float = 0.0
        self._cum_sell: float = 0.0
        self.market_mode: str = "趋势"
        self._status: Dict = {
            "is_spike": False, "mult": 1.0,
            "buy_pct": 0.5, "direction": "均衡",
            "raw_vol": 0.0, "baseline_vol": 0.0,
            "has_flow_data": False,
            "flow_per_sec": 0.0,
            "cum_delta_6":  0.0,
            "delta_trend":  "数据积累中",
            "absorption":   False,
            "cum_buy":  0.0,
            "cum_sell": 0.0,
            "cum_delta":0.0,
        }
        self.spike_event = threading.Event()

    def get_threshold(self) -> float:
        return float(CFG.v_spike_mult_thresh)

    def record_trade(self, sz: float, side: str) -> None:
        now_bucket = int(time.time() / self._WINDOW_SECS)
        with self._lock:
            if now_bucket != self._cur_bucket:
                if self._cur_bucket > 0:
                    self._buf.append((self._cur_bucket, self._cur_buy, self._cur_sell))
                    self._detect()
                self._cur_bucket = now_bucket
                self._cur_buy  = 0.0
                self._cur_sell = 0.0
            if side == "buy":
                self._cur_buy  += sz
                self._cum_buy  += sz
            else:
                self._cur_sell += sz
                self._cum_sell += sz

    def reset_cvd(self) -> None:
        with self._lock:
            self._cum_buy  = 0.0
            self._cum_sell = 0.0

    def _detect(self) -> None:
        if len(self._buf) < 5:
            return
        last = self._buf[-1]
        cur_vol = last[1] + last[2]
        prev = list(self._buf)[:-1]
        baseline = sum(b[1] + b[2] for b in prev) / len(prev)
        if baseline < self._MIN_BASELINE_VOL:
            return

        mult      = cur_vol / baseline if baseline > 0 else 1.0
        buy_pct   = last[1] / cur_vol if cur_vol > 0 else 0.5
        direction = ("买方主导" if buy_pct > 0.65
                     else "卖方主导" if buy_pct < 0.35
                     else "均衡")
        thresh    = self.get_threshold()
        is_spike  = (mult >= thresh
                     and cur_vol >= float(CFG.v_spike_min_contracts))

        flow_per_sec = round((last[1] - last[2]) / self._WINDOW_SECS, 2)

        _n6 = min(6, len(self._buf))
        cum_delta_6 = round(
            sum(b[1] - b[2] for b in list(self._buf)[-_n6:]), 1
        )

        if len(self._buf) >= 6:
            _recent3 = sum(b[1] - b[2] for b in list(self._buf)[-3:])
            _prior3  = sum(b[1] - b[2] for b in list(self._buf)[-6:-3])
            _accel   = _recent3 - _prior3
            _noise   = baseline * 0.25
            if _accel > _noise:
                delta_trend = "买压加速"
            elif _accel < -_noise:
                delta_trend = "卖压加速"
            else:
                delta_trend = "趋势平稳"
        else:
            delta_trend = "数据积累中"

        _one_sided  = (buy_pct >= 0.75 or buy_pct <= 0.25)
        _total_6    = sum(b[1] + b[2] for b in list(self._buf)[-_n6:])
        _delta_ratio = abs(cum_delta_6) / max(_total_6, 1.0)
        absorption  = is_spike and _one_sided and (_delta_ratio < 0.15)

        now = time.time()
        _just_triggered = is_spike and (now - self._last_spike_ts) >= self._COOLDOWN_SECS
        self._status = {
            "is_spike":              is_spike,
            "spike_just_triggered":  _just_triggered,
            "mult":         round(mult, 2),
            "buy_pct":      round(buy_pct, 3),
            "direction":    direction,
            "raw_vol":      round(cur_vol, 1),
            "baseline_vol": round(baseline, 1),
            "has_flow_data": True,
            "flow_per_sec":  flow_per_sec,
            "cum_delta_6":   cum_delta_6,
            "delta_trend":   delta_trend,
            "absorption":    absorption,
            "cum_buy":   round(self._cum_buy,  1),
            "cum_sell":  round(self._cum_sell, 1),
            "cum_delta": round(self._cum_buy - self._cum_sell, 1),
        }

        if is_spike:
            self._spike_peak_ts = now
            if mult > self._spike_peak_data.get("mult", 0):
                self._spike_peak_data = {
                    "mult":         round(mult, 2),
                    "buy_pct":      round(buy_pct, 3),
                    "direction":    direction,
                    "baseline_vol": round(baseline, 1),
                }

        if _just_triggered:
            self._last_spike_ts = now
            log.info(
                f"[VSpike+EAT-FLOW] {self.market_mode} "
                f"突增 {mult:.1f}x(阈值{thresh:.1f}x) "
                f"本桶={cur_vol:.0f}张(>={CFG.v_spike_min_contracts:.0f}) "
                f"基线={baseline:.0f}张 方向={direction} | "
                f"flow={flow_per_sec:+.1f}张/s cum_delta={cum_delta_6:+.0f} {delta_trend}"
                + (" 吸筹/出货" if absorption else "")
            )
            self.spike_event.set()

    def get_status(self) -> Dict:
        status = self._status.copy()
        if not status.get("is_spike") and self._spike_peak_data:
            _peak_age = time.time() - self._spike_peak_ts
            if _peak_age < self._SPIKE_PERSIST_SECS:
                status["spike_recent"]    = True
                status["spike_recent_age"] = round(_peak_age, 1)
                status["mult"]         = self._spike_peak_data["mult"]
                status["buy_pct"]      = self._spike_peak_data["buy_pct"]
                status["direction"]    = self._spike_peak_data["direction"]
                status["baseline_vol"] = self._spike_peak_data["baseline_vol"]
            else:
                # 峰值过期，清除旧数据，避免返回 stale 数据
                self._spike_peak_data = {}
        return status

    def reset_event(self) -> None:
        self.spike_event.clear()


# ============================================================
# WebSocket 客户端
# ============================================================
class OkxWebSocket:
    def __init__(self, trader, on_ticker_callback, on_private_callback,
                 on_mark_price_callback=None, on_trades_callback=None,
                 eth_trader=None):
        self.trader = trader
        self.eth_trader = eth_trader
        self.on_ticker      = on_ticker_callback
        self.on_private     = on_private_callback
        self.on_mark_price  = on_mark_price_callback
        self.on_trades      = on_trades_callback
        self.ws_private  = None
        self.ws_public   = None
        self.should_stop = False
        self.retry_delay_priv = CFG.ws_initial_retry_delay
        self.retry_delay_pub  = CFG.ws_initial_retry_delay
        self.max_retries = CFG.ws_max_retries
        self.retry_count_priv = 0
        self.retry_count_pub  = 0
        self.ping_thread_priv: Optional[threading.Thread] = None
        self.ping_thread_pub:  Optional[threading.Thread] = None
        self.ping_interval = 25
        self._last_msg_time: Dict[str, float] = {}

    def _is_data_stale(self, channel: str = "tickers", max_age: float = 30.0) -> bool:
        last = self._last_msg_time.get(channel)
        if last is None:
            return False
        age = time.time() - last
        if age > max_age:
            log.warning(
                f"[WS数据断流] 频道'{channel}' {age:.1f}s 无更新（阈值{max_age}s），"
                f"连接可能已失效，将触发强制重连"
            )
            return True
        return False

    def _get_auth_args(self):
        ts = int(time.time())
        msg = str(ts) + 'GET' + '/users/self/verify'
        sign = base64.b64encode(
            hmac.new(self.trader.secret_key.encode(), msg.encode(), hashlib.sha256).digest()
        ).decode()
        return {
            "apiKey": self.trader.api_key,
            "passphrase": self.trader.passphrase,
            "timestamp": str(ts),
            "sign": sign
        }

    def _send_ping(self, ws):
        if ws and not self.should_stop:
            try:
                ws.send(json.dumps({"op": "ping"}))
                log.debug("发送 OKX 应用层 ping")
            except Exception as e:
                log.debug(f"Ping 发送失败: {e}")

    def _ping_loop(self, ws_getter, stop_event: threading.Event):
        while not stop_event.wait(timeout=self.ping_interval):
            if self.should_stop:
                break
            ws = ws_getter()
            if ws:
                self._send_ping(ws)

    def _start_ping(self, ws_getter, attr_name: str):
        old_event_name = attr_name + "_stop_event"
        old_event: Optional[threading.Event] = getattr(self, old_event_name, None)
        if old_event:
            old_event.set()
        stop_event = threading.Event()
        setattr(self, old_event_name, stop_event)
        existing: Optional[threading.Thread] = getattr(self, attr_name, None)
        if existing and existing.is_alive():
            existing.join(timeout=2)
        t = threading.Thread(target=self._ping_loop, args=(ws_getter, stop_event),
                             daemon=True, name=f"ping-{attr_name}")
        setattr(self, attr_name, t)
        t.start()

    # ---------- 私有 WS ----------
    def _connect_private(self):
        url = "wss://ws.okx.com:8443/ws/v5/private"
        self.ws_private = websocket.WebSocketApp(
            url,
            on_open=self._on_open_private,
            on_message=self._on_message_private,
            on_error=self._on_error_private,
            on_close=self._on_close_private,
        )
        self.ws_private.run_forever(ping_interval=CFG.ws_ping_interval, ping_timeout=10)

    def _on_open_private(self, ws):
        self.retry_count_priv = 0
        self.retry_delay_priv = CFG.ws_initial_retry_delay
        self._private_logged_in = False
        log.info("私有 WebSocket 连接建立，尝试登录...")
        login_args = self._get_auth_args()
        ws.send(json.dumps({"op": "login", "args": [login_args]}))
        self._start_ping(lambda: self.ws_private, "ping_thread_priv")
        gs_set("last_state_sync", None)

    def _subscribe_private_channels(self, ws):
        """登录确认后再订阅私有频道"""
        channels = [{"channel": "account", "ccy": "USDT"}]
        channels.append({"channel": "positions", "instId": CFG.symbol})
        channels.append({"channel": "orders", "instType": "SWAP", "instId": CFG.symbol})
        ws.send(json.dumps({"op": "subscribe", "args": channels}))
        log.info(f"私有频道订阅完成（{CFG.symbol} positions / account / orders）")

    def _on_message_private(self, ws, message):
        try:
            data = json.loads(message)
            self._last_msg_time["private"] = time.time()
            # 检测 login 成功后再订阅
            if data.get("event") == "login" and data.get("code") == "0":
                self._private_logged_in = True
                log.info("私有 WebSocket 登录成功，开始订阅频道...")
                self._subscribe_private_channels(ws)
                return
            if data.get("event") == "login" and data.get("code") != "0":
                log.error(f"私有 WebSocket 登录失败: {data}")
                return
            self.on_private(data)
        except json.JSONDecodeError:
            pass

    def _on_error_private(self, ws, error):
        log.exception(f"私有 WebSocket 错误: {error}")

    def _on_close_private(self, ws, close_status_code, close_msg):
        log.warning(f"私有 WebSocket 关闭，code={close_status_code}, msg={close_msg}")
        if not self.should_stop:
            self._reconnect_private()

    def _reconnect_private(self):
        while not self.should_stop:
            if self.max_retries > 0 and self.retry_count_priv >= self.max_retries:
                log.critical("私有 WS 重连次数已达上限，停止")
                return
            delay = self.retry_delay_priv
            log.info(f"私有 WS 将在 {delay:.1f}s 后重连...")
            time.sleep(delay)
            self.retry_count_priv += 1
            self.retry_delay_priv = min(self.retry_delay_priv * 2, 60)
            try:
                self._connect_private()
                break  # run_forever 正常退出后跳出循环
            except Exception as e:
                log.error(f"私有 WS 重连异常: {e}")

    # ---------- 公共 WS ----------
    def _connect_public(self):
        url = "wss://ws.okx.com:8443/ws/v5/public"
        self.ws_public = websocket.WebSocketApp(
            url,
            on_open=self._on_open_public,
            on_message=self._on_message_public,
            on_error=self._on_error_public,
            on_close=self._on_close_public,
        )
        self.ws_public.run_forever(ping_interval=CFG.ws_ping_interval, ping_timeout=10)

    def _on_open_public(self, ws):
        self.retry_count_pub = 0
        self.retry_delay_pub = CFG.ws_initial_retry_delay
        log.info(f"公共 WebSocket 连接建立，订阅 {CFG.symbol} tickers + mark-price + trades...")
        args = [
            {"channel": "tickers",    "instId": CFG.symbol},
            {"channel": "mark-price", "instId": CFG.symbol},
            {"channel": "trades",     "instId": CFG.symbol},
        ]
        ws.send(json.dumps({"op": "subscribe", "args": args}))
        log.info(f"公共频道订阅完成（{CFG.symbol} tickers / mark-price / trades）")
        self.retry_delay_pub = CFG.ws_initial_retry_delay
        self._start_ping(lambda: self.ws_public, "ping_thread_pub")

    def _on_message_public(self, ws, message):
        try:
            data = json.loads(message)
            channel = data.get("arg", {}).get("channel", "")
            self._last_msg_time[channel] = time.time()
            if channel == "mark-price" and self.on_mark_price:
                self.on_mark_price(data)
            elif channel == "trades" and self.on_trades:
                self.on_trades(data)
            else:
                self.on_ticker(data)
        except json.JSONDecodeError:
            pass

    def _on_error_public(self, ws, error):
        log.exception(f"公共 WebSocket 错误: {error}")

    def _on_close_public(self, ws, close_status_code, close_msg):
        log.warning(f"公共 WebSocket 关闭，code={close_status_code}, msg={close_msg}")
        if not self.should_stop:
            self._reconnect_public()

    def _reconnect_public(self):
        while not self.should_stop:
            if self.max_retries > 0 and self.retry_count_pub >= self.max_retries:
                log.critical("公共 WS 重连次数已达上限，停止")
                return
            delay = self.retry_delay_pub
            log.info(f"公共 WS 将在 {delay:.1f}s 后重连...")
            time.sleep(delay)
            self.retry_count_pub += 1
            self.retry_delay_pub = min(self.retry_delay_pub * 2, 60)
            try:
                self._connect_public()
                break
            except Exception as e:
                log.error(f"公共 WS 重连异常: {e}")

    # ---------- 启动 / 停止 ----------
    def start(self):
        self.should_stop = False
        threading.Thread(target=self._connect_private, daemon=True).start()
        threading.Thread(target=self._connect_public,  daemon=True).start()

    def stop(self):
        self.should_stop = True
        if self.ws_private:
            self.ws_private.close()
        if self.ws_public:
            self.ws_public.close()


# ============================================================
# OKX Trader — REST API 封装
# ============================================================
class OkxTrader:
    def __init__(self, eth_trader=None):
        self.api_key    = os.getenv("OKX_API_KEY",    "").strip()
        self.secret_key = os.getenv("OKX_SECRET_KEY", "").strip()
        self.passphrase = os.getenv("OKX_PASSWORD",   "").strip()
        self.base_url   = os.getenv("OKX_REST_URL", "https://www.okx.com")
        self.contract_sizes: Dict[str, float] = {}
        self.tick_sizes:     Dict[str, float] = {}
        self.lot_sizes:      Dict[str, float] = {}
        self._slippage_history: Dict[str, deque] = {CFG.symbol: deque(maxlen=CFG.slippage_adapt_window)}
        self._slippage_hourly: Dict[str, deque] = {CFG.symbol: deque(maxlen=3600//CFG.risk_check_interval)}
        self.eth_trader = eth_trader
        if not all([self.api_key, self.secret_key, self.passphrase]):
            raise ValueError("OKX API 配置不完整，请检查 .env")

    def _sign(self, ts: str, method: str, path: str, body: str = "") -> str:
        msg = ts + method.upper() + path + body
        mac = hmac.new(self.secret_key.encode(), msg.encode(), digestmod=hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    @retry_with_backoff()
    def _request(self, method: str, endpoint: str,
                 params: Dict = None, body_data: Dict = None) -> Dict:
        ts        = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        body_str  = json.dumps(body_data, separators=(",", ":")) if body_data else ""
        sign_path = endpoint + ("?" + urlencode(params) if params else "")
        headers   = {
            "OK-ACCESS-KEY":        self.api_key,
            "OK-ACCESS-SIGN":       self._sign(ts, method, sign_path, body_str),
            "OK-ACCESS-TIMESTAMP":  ts,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type":         "application/json",
        }
        url = self.base_url + endpoint
        try:
            if method.upper() == "GET":
                resp = requests.get(url + ("?" + urlencode(params) if params else ""),
                                    headers=headers, timeout=(5, 10))
            else:
                resp = requests.post(url, headers=headers, data=body_str, timeout=(5, 10))
            if resp.status_code >= 500:
                resp.raise_for_status()
            try:
                return resp.json()
            except json.JSONDecodeError:
                log.error(f"API返回非JSON格式: {resp.text[:200]}")
                return {"code": "-1", "msg": "Invalid JSON response"}
        except requests.exceptions.Timeout:
            log.warning(f"API请求超时: {endpoint} {method}")
            return {"code": "-1", "msg": "Request timeout"}
        except requests.exceptions.RequestException as e:
            log.exception(f"API请求异常: {e}")
            return {"code": "-1", "msg": str(e)}

    def fetch_contract_sizes(self):
        resp = self._request("GET", "/api/v5/public/instruments", params={"instType": "SWAP"})
        if resp.get("code") == "0":
            for inst in resp.get("data", []):
                sid = inst.get("instId", "")
                self.contract_sizes[sid] = float(inst.get("ctVal",  1.0))
                self.tick_sizes[sid]     = float(inst.get("tickSz", 0.01))
                self.lot_sizes[sid]      = float(inst.get("lotSz",  1.0))
        sym = CFG.symbol
        ct   = self.contract_sizes.get(sym, "未知")
        tick = self.tick_sizes.get(sym, 0.01)
        lot  = self.lot_sizes.get(sym, 1.0)
        log.info(f"{sym} 合约面值: {ct}/张 价格步长: {tick} 数量步长: {lot}张")

    def get_account_balance(self) -> float:
        resp = self._request("GET", "/api/v5/account/balance")
        if resp.get("code") == "0" and resp.get("data"):
            for d in resp.get("data", [{}])[0].get("details", []):
                if d.get("ccy") == "USDT":
                    try: return float(d.get("availBal", 0))
                    except Exception: return 0.0
        return 0.0

    def get_account_equity(self) -> float:
        resp = self._request("GET", "/api/v5/account/balance")
        if resp.get("code") == "0" and resp.get("data"):
            for d in resp.get("data", [{}])[0].get("details", []):
                if d.get("ccy") == "USDT":
                    try:
                        eq = d.get("eq")
                        if eq is not None:
                            return float(eq)
                        avail  = float(d.get("availBal", 0))
                        frozen = float(d.get("frozenBal", 0))
                        return avail + frozen
                    except Exception:
                        return 0.0
        return 0.0

    def get_account_balance_full(self) -> Dict[str, float]:
        resp = self._request("GET", "/api/v5/account/balance")
        if resp.get("code") == "0" and resp.get("data"):
            for d in resp.get("data", [{}])[0].get("details", []):
                if d.get("ccy") == "USDT":
                    try:
                        avail  = float(d.get("availBal",  0))
                        frozen = float(d.get("frozenBal", 0))
                        eq_raw = d.get("eq")
                        equity = float(eq_raw) if eq_raw is not None else (avail + frozen)
                        return {"equity": equity, "avail_bal": avail}
                    except Exception:
                        return {"equity": 0.0, "avail_bal": 0.0}
        return {"equity": 0.0, "avail_bal": 0.0}

    def get_positions(self) -> Dict:
        sym = CFG.symbol
        return self._request("GET", "/api/v5/account/positions",
                             params={"instId": sym})

    def get_current_price(self) -> float:
        sym = CFG.symbol
        resp = self._request("GET", "/api/v5/market/ticker", params={"instId": sym})
        if resp.get("code") == "0" and resp.get("data"):
            return float(resp["data"][0].get("last", 0))
        return 0.0

    def set_leverage(self, lever: int, symbol: str = None, posSide: str = None):
        sym = CFG.symbol
        body_data = {"instId": sym, "lever": str(lever), "mgnMode": "isolated", "posSide": posSide}
        return self._request("POST", "/api/v5/account/set-leverage", body_data=body_data)

    def _fmt_price(self, price: float, symbol: str, side: str = "buy") -> str:
        """价格精度对齐：买单向下取整(floor)，卖单向上取整(ceil)"""
        tick = self.tick_sizes.get(symbol, 0.01)
        if side in ("sell", "short"):
            aligned = math.ceil(price / tick - 1e-9) * tick
        else:
            aligned = math.floor(price / tick + 1e-9) * tick
        if tick >= 1:
            decimals = 0
        else:
            decimals = max(0, round(-math.log10(tick)))
        return f"{aligned:.{decimals}f}"

    def place_order(self, side: str, posSide: str, sz: str,
                    px: float = None, sl_px: float = None, tp_px: float = None,
                    symbol: str = None, ord_type: str = "market") -> Dict:
        sym = CFG.symbol
        params: Dict = {
            "instId": sym, "tdMode": "isolated",
            "side": side, "posSide": posSide,
            "ordType": ord_type,
            "sz": sz,
        }
        if ord_type == "limit" and px is not None:
            params["px"] = self._fmt_price(px, sym)
        algo = {}
        if sl_px and sl_px > 0:
            algo["slTriggerPx"] = self._fmt_price(sl_px, sym)
            algo["slOrdPx"]     = "-1"
        if tp_px and tp_px > 0:
            algo["tpTriggerPx"] = self._fmt_price(tp_px, sym)
            algo["tpOrdPx"]     = "-1"
        if algo:
            params["attachAlgoOrds"] = [algo]
        return self._request("POST", "/api/v5/trade/order", body_data=params)

    def place_ioc_order(self, side: str, posSide: str, sz: str, px: float,
                        symbol: str = None) -> Dict:
        sym = CFG.symbol
        return self._request("POST", "/api/v5/trade/order", body_data={
            "instId": sym, "tdMode": "isolated",
            "side": side, "posSide": posSide,
            "ordType": "limit",
            "tifType": "IOC",
            "px": self._fmt_price(px, sym),
            "sz": sz,
            "tgtCcy": "base_ccy",
        })

    def place_market_order(self, side: str, posSide: str, sz: str,
                           symbol: str = None) -> Dict:
        sym = CFG.symbol
        return self._request("POST", "/api/v5/trade/order", body_data={
            "instId": sym, "tdMode": "isolated",
            "side": side, "posSide": posSide,
            "ordType": "market", "sz": sz,
        })

    def get_order_status(self, ord_id: str) -> Dict:
        sym = CFG.symbol
        return self._request("GET", "/api/v5/trade/order",
                             params={"instId": sym, "ordId": ord_id})

    def get_orderbook(self, sz: int = 400, symbol: str = None) -> Dict:
        sym = symbol or CFG.symbol
        resp = self._request("GET", "/api/v5/market/books",
                             params={"instId": sym, "sz": str(sz)})
        if resp.get("code") == "0" and resp.get("data"):
            book = resp["data"][0]
            bids = [[float(p), float(s)] for p, s, *_ in book.get("bids", [])]
            asks = [[float(p), float(s)] for p, s, *_ in book.get("asks", [])]
            return {"bids": bids, "asks": asks, "ts": book.get("ts", 0)}
        return {"bids": [], "asks": []}

    def analyze_orderbook(self) -> Dict:
        book = self.get_orderbook(sz=400)
        _empty = {
            "bid_depth": 0, "ask_depth": 0, "imbalance": 0,
            "big_bid": 0, "big_ask": 0, "spread": 0,
            "top5_bids": [], "top5_asks": [],
            "bid_slope": 0, "ask_slope": 0, "slope_ratio": 1.0,
            "depth_ratio": 0,
            "bid_wall_mult": 0.0, "ask_wall_mult": 0.0,
            "bid_wall_dist_pct": 0.0, "ask_wall_dist_pct": 0.0,
            "imbal_near": 0.0,
        }
        if not book["bids"] or not book["asks"]:
            return _empty

        bids = book["bids"]
        asks = book["asks"]
        mid  = (bids[0][0] + asks[0][0]) / 2.0

        bid_depth = sum(s for _, s in bids)
        ask_depth = sum(s for _, s in asks)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-9)
        big_bid   = max(s for _, s in bids)
        big_ask   = max(s for _, s in asks)
        top5_bids = bids[:5]
        top5_asks = asks[:5]
        spread    = asks[0][0] - bids[0][0] if asks and bids else 0

        n = max(2, min(int(CFG.ob_slope_levels), len(bids), len(asks)))

        def _lin_slope(levels: list, n: int) -> float:
            cum, xs, ys = 0.0, [], []
            for i, (_, sz) in enumerate(levels[:n]):
                cum += sz
                xs.append(i)
                ys.append(cum)
            if len(xs) < 2:
                return 0.0
            x_mean = sum(xs) / len(xs)
            y_mean = sum(ys) / len(ys)
            num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
            den = sum((x - x_mean) ** 2 for x in xs)
            return num / den if den > 1e-9 else 0.0

        bid_slope   = _lin_slope(bids, n)
        ask_slope   = _lin_slope(asks, n)
        slope_ratio = bid_slope / max(ask_slope, 1e-9)

        def _wall(levels: list, n: int, mid_price: float, side: str):
            if not levels:
                return 0.0, 0.0
            sizes = [s for _, s in levels[:n]]
            snap_avg = max(sum(sizes) / max(len(sizes), 1), 5.0)
            rolling_list = getattr(self.eth_trader, "_ob_rolling_avg", []) if self.eth_trader else []
            rolling_avg = sum(rolling_list) / max(len(rolling_list), 1) if rolling_list else snap_avg
            baseline = max(rolling_avg * 0.8, 5.0)
            max_sz  = max(sizes)
            mult    = max_sz / baseline
            max_idx = sizes.index(max_sz)
            wall_px = levels[max_idx][0]
            dist_pct = abs(wall_px - mid_price) / mid_price
            return round(mult, 2), round(dist_pct, 5)

        bid_wall_mult, bid_wall_dist_pct = _wall(bids, n, mid, "bid")
        ask_wall_mult, ask_wall_dist_pct = _wall(asks, n, mid, "ask")

        near_k    = min(5, n)
        near_bid  = sum(s for _, s in bids[:near_k])
        near_ask  = sum(s for _, s in asks[:near_k])
        imbal_near = (near_bid - near_ask) / (near_bid + near_ask + 1e-9)

        return {
            "bid_depth":  bid_depth,
            "ask_depth":  ask_depth,
            "imbalance":  imbalance,
            "big_bid":    big_bid,
            "big_ask":    big_ask,
            "spread":     spread,
            "top5_bids":  top5_bids,
            "top5_asks":  top5_asks,
            "bid_slope":  round(bid_slope,  2),
            "ask_slope":  round(ask_slope,  2),
            "depth_ratio": bid_depth / (ask_depth + 1e-9),
            "slope_ratio":        round(slope_ratio, 3),
            "bid_wall_mult":      bid_wall_mult,
            "ask_wall_mult":      ask_wall_mult,
            "bid_wall_dist_pct":  bid_wall_dist_pct,
            "ask_wall_dist_pct":  ask_wall_dist_pct,
            "imbal_near":         round(imbal_near, 3),
        }

    def get_orderbook_price(self, side: str) -> float:
        sym = CFG.symbol
        try:
            resp = self._request("GET", "/api/v5/market/books",
                                 params={"instId": sym, "sz": "1"})
            if resp.get("code") == "0" and resp.get("data"):
                book = resp["data"][0]
                if side == "buy":
                    return float(book["asks"][0][0])
                else:
                    return float(book["bids"][0][0])
        except Exception as e:
            log.debug(f"获取盘口失败: {e}")
        return 0.0

    def cancel_order(self, ord_id: str) -> Dict:
        sym = CFG.symbol
        return self._request("POST", "/api/v5/trade/cancel-order",
                             body_data={"instId": sym, "ordId": ord_id})

    def cancel_all_orders(self, symbol: str = None) -> List[Dict]:
        sym = symbol if symbol else CFG.symbol
        try:
            resp = self._request("POST", "/api/v5/trade/cancel-all",
                                 body_data={"instId": sym})
            if resp.get("code") == "0":
                log.info(f"[{sym}] 已取消全部挂单")
                return resp.get("data", [])
            else:
                log.warning(f"[{sym}] 取消全部挂单失败: {resp}")
                return []
        except Exception as e:
            log.warning(f"[{sym}] 取消全部挂单异常: {e}")
            return []

    def close_position(self, posSide: str) -> Dict:
        sym = CFG.symbol
        return self._request("POST", "/api/v5/trade/close-position",
                             body_data={"instId": sym,
                                        "mgnMode": "isolated", "posSide": posSide})

    def get_funding_rate(self) -> Dict:
        sym = CFG.symbol
        resp = self._request("GET", "/api/v5/public/funding-rate",
                             params={"instId": sym})
        if resp.get("code") == "0" and resp.get("data"):
            d = resp["data"][0]
            return {"funding_rate": float(d.get("fundingRate", 0)),
                    "next_funding_time": d.get("nextFundingTime", "N/A")}
        return {"funding_rate": 0.0, "next_funding_time": "N/A"}

    def get_bills_archive(self, instType: str = "SWAP", limit: int = 100, after: str = None) -> Dict:
        params = {"instType": instType, "limit": str(limit)}
        if after:
            params["after"] = after
        return self._request("GET", "/api/v5/account/bills-archive", params=params)

    def get_algo_orders(self) -> Dict:
        sym = CFG.symbol
        base_params = {"instType": "SWAP", "instId": sym}
        combined = {"code": "0", "data": []}
        for ord_type in ("oco", "conditional"):
            resp = self._request("GET", "/api/v5/trade/orders-algo-pending",
                                 params={**base_params, "ordType": ord_type})
            if resp.get("code") == "0":
                combined["data"].extend(resp.get("data", []))
            else:
                log.debug(f"查询 {ord_type} 算法单: {resp.get('msg','')}")
        return combined

    def cancel_algo_order(self, algo_id: str) -> Dict:
        sym = CFG.symbol
        return self._request("POST", "/api/v5/trade/cancel-algos",
                             body_data=[{"algoId": algo_id, "instId": sym}])

    def place_algo_order(self, side: str, posSide: str, sz: str,
                         sl_px: float, tp_px: float) -> Dict:
        sym = CFG.symbol
        has_sl = sl_px and sl_px > 0
        has_tp = tp_px and tp_px > 0
        ord_type = "oco" if (has_sl and has_tp) else "conditional"

        params: Dict = {
            "instId":  sym,
            "tdMode":  "isolated",
            "side":    side,
            "posSide": posSide,
            "ordType": ord_type,
            "sz":      sz,
        }
        if has_sl:
            params["slTriggerPx"] = self._fmt_price(sl_px, sym)
            params["slOrdPx"]     = "-1"
        if has_tp:
            params["tpTriggerPx"] = self._fmt_price(tp_px, sym)
            params["tpOrdPx"]     = "-1"
        return self._request("POST", "/api/v5/trade/order-algo", body_data=params)

    def update_algo_orders(self, posSide: str, sz: str, new_sl: float, new_tp: float,
                           symbol: str = None) -> bool:
        sym = CFG.symbol
        resp = self.get_algo_orders()
        if resp.get("code") != "0":
            log.error(f"查询算法单失败: {resp}")
            return False
        algo_list = [a for a in resp.get("data", []) if a.get("instId") == sym]
        for algo in algo_list:
            algo_id   = algo.get("algoId")
            cancel_res = self.cancel_algo_order(algo_id)
            if cancel_res.get("code") != "0":
                log.error(f"撤销算法单 {algo_id} 失败: {cancel_res}")
                return False
            log.info(f"已撤销原算法单 {algo_id}")
        time.sleep(1 + __import__("numpy").random.random())
        confirm_resp = self.get_algo_orders()
        if confirm_resp.get("code") == "0" and confirm_resp.get("data"):
            log.warning("撤单后仍有活跃算法单，等待后重试")
            time.sleep(2)
        side = "sell" if posSide == "long" else "buy"
        res  = self.place_algo_order(side, posSide, sz, new_sl, new_tp)
        if res.get("code") == "0":
            log.info(f"新算法单挂载成功: sl={new_sl:.4f}, tp={new_tp:.4f}")
            return True
        else:
            log.error(f"挂载新算法单失败: {res}")
            return False

    def record_slippage(self, symbol: str, slippage_pct: float):
        if symbol in self._slippage_history:
            self._slippage_history[symbol].append(slippage_pct)
        if symbol in self._slippage_hourly:
            self._slippage_hourly[symbol].append((time.time(), slippage_pct))

    def get_avg_slippage(self, symbol: str) -> float:
        hist = self._slippage_history.get(symbol, [])
        if not hist:
            return CFG.slippage_pct
        return sum(hist) / len(hist)

    def get_hourly_avg_slippage(self, symbol: str) -> float:
        hist = self._slippage_hourly.get(symbol, [])
        cutoff = time.time() - 3600
        recent = [s for ts, s in hist if ts >= cutoff]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def get_dynamic_slippage(self, symbol: str) -> float:
        if not CFG.slippage_adapt_enable:
            return CFG.slippage_pct
        avg = self.get_avg_slippage(symbol)
        dynamic = avg * CFG.slippage_adapt_mult
        return min(dynamic, CFG.slippage_fuse_threshold * 0.5)
