#!/usr/bin/env python3
"""
一次性修复脚本：回填 trades 表中 pnl/close_ts 为 NULL 的脏数据。
用法：停止交易服务后运行  python3 fix_historical_data.py
运行完毕后可删除本脚本。
"""
import sqlite3
import os
from datetime import datetime, timezone

UTC = timezone.utc
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eth_trading.db")

def main():
    if not os.path.exists(DB_PATH):
        print(f"❌ 数据库不存在: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()

    # ── 1. 统计现状 ──────────────────────────────────────────────
    c.execute("SELECT COUNT(*) FROM trades")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM trades WHERE exit IS NOT NULL")
    closed = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM trades WHERE exit IS NOT NULL AND pnl IS NULL")
    null_pnl = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM trades WHERE exit IS NOT NULL AND close_ts IS NULL")
    null_close_ts = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM trades WHERE exit IS NULL AND open_ts IS NOT NULL")
    orphan = c.fetchone()[0]

    print(f"📊 trades 表现状:")
    print(f"   总记录: {total}")
    print(f"   已平仓（有exit）: {closed}")
    print(f"   已平仓但 pnl=NULL: {null_pnl}")
    print(f"   已平仓但 close_ts=NULL: {null_close_ts}")
    print(f"   孤儿记录（有open无exit）: {orphan}")
    print()

    if null_pnl == 0 and null_close_ts == 0:
        print("✅ 无需修复，数据全部正常")
    else:
        # ── 2. 回填 pnl ─────────────────────────────────────────────
        if null_pnl > 0:
            c.execute("""
                SELECT id, side, entry, exit
                FROM trades
                WHERE exit IS NOT NULL AND pnl IS NULL AND entry > 0
            """)
            rows = c.fetchall()
            fixed_pnl = 0
            for tid, side, entry, exit_price in rows:
                if side == "long":
                    pnl_pct = (exit_price - entry) / entry
                elif side == "short":
                    pnl_pct = (entry - exit_price) / entry
                else:
                    print(f"   ⚠️ trade#{tid} side='{side}' 未知，跳过")
                    continue
                # 默认用 10x 杠杆（系统固定杠杆）
                pnl = pnl_pct * 10
                c.execute(
                    "UPDATE trades SET pnl=?, pnl_pct=? WHERE id=?",
                    (round(pnl, 6), round(pnl_pct, 6), tid)
                )
                fixed_pnl += 1
                result = "盈" if pnl_pct > 0 else "亏"
                print(f"   ✅ trade#{tid:4d} {side:5s} entry={entry:.2f} exit={exit_price:.2f} → pnl_pct={pnl_pct*100:+.2f}% ({result})")

            print(f"\n   回填 pnl 完成: {fixed_pnl}/{null_pnl} 条")

        # ── 3. 回填 close_ts ─────────────────────────────────────────
        if null_close_ts > 0:
            # 优先用 open_ts + 推算，无法推算则用 open_ts 做兜底
            c.execute("""
                SELECT id, open_ts
                FROM trades
                WHERE exit IS NOT NULL AND close_ts IS NULL
            """)
            rows = c.fetchall()
            fixed_ts = 0
            for tid, open_ts in rows:
                if open_ts:
                    # close_ts 无法精确恢复，用 open_ts 标记为"已平仓时间不详"
                    c.execute(
                        "UPDATE trades SET close_ts=? WHERE id=?",
                        (open_ts, tid)  # 用 open_ts 兜底，至少不为 NULL
                    )
                    fixed_ts += 1
            print(f"   回填 close_ts 完成: {fixed_ts}/{null_close_ts} 条（用 open_ts 兜底）")

        conn.commit()

    # ── 4. 清理孤儿记录（只有开仓没有平仓，且超过7天）──────────────
    if orphan > 0:
        cutoff = (datetime.now(UTC) - __import__('datetime').timedelta(days=7)).isoformat()
        c.execute("""
            SELECT id, side, entry, open_ts
            FROM trades
            WHERE exit IS NULL AND open_ts < ?
        """, (cutoff,))
        stale = c.fetchall()
        if stale:
            print(f"\n🧹 发现 {len(stale)} 条超过7天的孤儿记录:")
            for tid, side, entry, ots in stale:
                print(f"   trade#{tid:4d} {side:5s} entry={entry:.2f} open={ots}")
            # 不自动删除，只标记
            print(f"   （未自动删除，如需清理请手动: DELETE FROM trades WHERE exit IS NULL AND open_ts < '{cutoff}'）")

    # ── 5. 修复后统计 ────────────────────────────────────────────
    c.execute("SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL")
    valid = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL AND pnl > 0")
    wins = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL AND pnl <= 0")
    losses = c.fetchone()[0]
    win_rate = wins / valid * 100 if valid > 0 else 0

    print(f"\n📊 修复后统计:")
    print(f"   有效记录: {valid}")
    print(f"   盈利: {wins} | 亏损: {losses}")
    print(f"   真实胜率: {win_rate:.1f}%")

    # ── 6. 检查 historical_cases 表 ──────────────────────────────
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_cases'")
    if not c.fetchone():
        print(f"\n⚠️ historical_cases 表不存在，将在下次启动交易服务时自动创建")
    else:
        c.execute("SELECT COUNT(*) FROM historical_cases")
        hc = c.fetchone()[0]
        print(f"\n📊 historical_cases 表: {hc} 条案例")

    conn.close()
    print("\n✅ 修复完成")

if __name__ == "__main__":
    main()
