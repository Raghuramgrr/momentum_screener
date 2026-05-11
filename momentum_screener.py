#!/usr/bin/env python3
"""
MOMENTUM SCREENER
=================
Usage:
  python momentum_screener.py                        # scan default watchlist
  python momentum_screener.py FLEX AMD JBL NVDA      # custom tickers
  python momentum_screener.py --preset ai            # preset watchlist
  python momentum_screener.py --watch                # auto-refresh every 5 min
  python momentum_screener.py --serve                # start API server on :5000

Install:
  pip install yfinance pandas tabulate colorama flask flask-cors
"""

import sys
import time
import argparse
from datetime import datetime

# ── deps ──────────────────────────────────────────────────────────────────────
try:
    import yfinance as yf
    import pandas as pd
    from tabulate import tabulate
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError as e:
    print(f"\n  Missing dependency: {e}")
    print("  pip install yfinance pandas tabulate colorama flask flask-cors\n")
    sys.exit(1)

# ── PRESETS ───────────────────────────────────────────────────────────────────
PRESETS = {
    "watchlist": ["FLEX", "AMD", "JBL", "COCO", "MPC"],
    "ai":        ["NVDA", "AMD", "FLEX", "JBL", "PLTR", "SMCI"],
    "momentum":  ["AAPL", "MSFT", "AMZN", "META", "TSLA", "NFLX"],
    "energy":    ["MPC", "XOM", "CVX", "COP", "SLB"],
    "mag7":      ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"],
}

# ── COLOURS ───────────────────────────────────────────────────────────────────
G  = Fore.GREEN
R  = Fore.RED
Y  = Fore.YELLOW
C  = Fore.CYAN
DM = Style.DIM
B  = Style.BRIGHT
RS = Style.RESET_ALL

def green(t):  return f"{G}{B}{t}{RS}"
def red(t):    return f"{R}{B}{t}{RS}"
def yellow(t): return f"{Y}{t}{RS}"
def cyan(t):   return f"{C}{t}{RS}"
def dim(t):    return f"{DM}{t}{RS}"
def bold(t):   return f"{B}{t}{RS}"

# ── LOGGING ───────────────────────────────────────────────────────────────────
def log(msg, level="info"):
    ts  = datetime.now().strftime("%H:%M:%S")
    tag = {"ok": green("✓"), "err": red("✗"), "warn": yellow("⚠"), "info": dim("·")}[level]
    print(f"  {dim(ts)}  {tag}  {msg}", flush=True)

# ── DATA FETCH ────────────────────────────────────────────────────────────────
def fetch(ticker):
    log(f"[{ticker}] Fetching...", "info")
    try:
        raw = yf.download(ticker, period="65d", interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty or len(raw) < 15:
            log(f"[{ticker}] Only {len(raw)} bars — need 15+", "warn")
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        closes  = raw["Close"].dropna().tolist()
        volumes = raw["Volume"].dropna().tolist()
        highs   = raw["High"].dropna().tolist()
        lows    = raw["Low"].dropna().tolist()

        price   = closes[-1]
        prev    = closes[-2]
        chg_pct = (price - prev) / prev * 100

        try:
            name = yf.Ticker(ticker).info.get("longName", ticker)
        except Exception:
            name = ticker

        vol_m = volumes[-1] / 1e6
        log(f"[{ticker}] OK  ${price:.2f}  {chg_pct:+.2f}%  {len(closes)} bars  vol {vol_m:.1f}M", "ok")
        return dict(ticker=ticker, name=name, price=price,
                    change_pct=chg_pct, closes=closes,
                    volumes=volumes, highs=highs, lows=lows)
    except Exception as e:
        log(f"[{ticker}] Failed — {e}", "err")
        return None

# ── INDICATORS ────────────────────────────────────────────────────────────────
def ema(data, period):
    k = 2 / (period + 1)
    e = data[0]
    for v in data[1:]:
        e = v * k + e * (1 - k)
    return e

def rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    gains = losses = 0.0
    for i in range(len(closes) - period, len(closes)):
        d = closes[i] - closes[i - 1]
        if d > 0: gains  += d
        else:     losses -= d
    return 100 - 100 / (1 + gains / (losses or 1e-9))

def macd(closes):
    m = ema(closes, 12) - ema(closes, 26)
    return m, m > 0

def adx(highs, lows, closes, period=14):
    if len(closes) < period + 2:
        return 20.0
    n = min(len(closes) - 1, period)
    sp = sm = st = 0.0
    for i in range(len(closes) - n, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i]  - closes[i - 1]))
        dp = max(highs[i] - highs[i - 1], 0)
        dm = max(lows[i - 1] - lows[i], 0)
        st += tr
        sp += dp if dp > dm else 0
        sm += dm if dm > dp else 0
    pip = (sp / st) * 100
    mim = (sm / st) * 100
    return abs(pip - mim) / (pip + mim or 1) * 100

def rvol(volumes):
    if len(volumes) < 21:
        return 1.0
    avg = sum(volumes[-21:-1]) / 20
    return volumes[-1] / (avg or 1)

def atr(highs, lows, closes, period=14):
    n = min(len(closes) - 1, period)
    total = 0.0
    for i in range(len(closes) - n, len(closes)):
        total += max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i]  - closes[i - 1]))
    return total / n

def bb_squeeze(closes, period=20):
    if len(closes) < period:
        return False
    sl  = closes[-period:]
    avg = sum(sl) / period
    std = (sum((v - avg) ** 2 for v in sl) / period) ** 0.5
    return (2 * std) / avg < 0.05

# ── SIGNAL ENGINE ─────────────────────────────────────────────────────────────
def analyze(data):
    closes  = data["closes"]
    volumes = data["volumes"]
    highs   = data["highs"]
    lows    = data["lows"]
    price   = data["price"]
    chg_pct = data["change_pct"]

    e9      = ema(closes, 9)
    e20     = ema(closes, 20)
    e50     = ema(closes, 50)
    rsi_v   = rsi(closes)
    macd_v, macd_bull = macd(closes)
    adx_v   = adx(highs, lows, closes)
    rvol_v  = rvol(volumes)
    atr_v   = atr(highs, lows, closes)
    squeeze = bb_squeeze(closes)

    pct20    = (price - e20) / e20 * 100
    extended = pct20 > 8
    rsi_bull = 55 <= rsi_v <= 70
    trend_ok = adx_v > 25

    score = 30
    if trend_ok:                       score += 15
    if adx_v > 35:                     score += 5
    if rvol_v > 1.5:                   score += 15
    if rvol_v > 2.0:                   score += 5
    if macd_bull:                      score += 10
    if price > e9 > 0 and price > e20: score += 10
    if price > e50:                    score += 5
    if rsi_bull:                       score += 10
    elif rsi_v >= 45:                  score += 5
    if squeeze:                        score += 10
    if extended:                       score -= 15
    if rsi_v > 80:                     score -= 10
    score = max(10, min(95, score))

    setup = "NEUTRAL"
    if extended and rsi_v > 70:
        setup = "EXTENDED"
    elif squeeze and adx_v < 25:
        setup = "BB SQUEEZE"
    elif trend_ok and macd_bull and price > e20:
        setup = "MOMENTUM CONT."
    elif chg_pct > 5 and rvol_v > 1.5:
        setup = "BREAKOUT"
    elif chg_pct < -3 and price > e50:
        setup = "PULLBACK ENTRY"
    elif macd_bull and price > e20:
        setup = "TREND FOLLOWING"

    stop = round(price - 1.5 * atr_v, 2)
    t1   = round(price + 2.0 * atr_v, 2)
    t2   = round(price + 3.5 * atr_v, 2)
    rr   = round((t1 - price) / max(price - stop, 0.01), 1)

    reasons, risks = [], []
    if adx_v > 35:     reasons.append(f"ADX {adx_v:.0f} — very strong trend")
    elif trend_ok:     reasons.append(f"ADX {adx_v:.0f} — trend confirmed")
    else:              risks.append(f"ADX {adx_v:.0f} — weak trend")

    if rvol_v > 2:     reasons.append(f"RVOL {rvol_v:.1f}x — institutional surge")
    elif rvol_v > 1.5: reasons.append(f"RVOL {rvol_v:.1f}x — vol confirms move")
    else:              risks.append(f"RVOL {rvol_v:.1f}x — low volume")

    if rsi_bull:       reasons.append(f"RSI {rsi_v:.0f} — ideal bull zone")
    elif rsi_v > 70:   risks.append(f"RSI {rsi_v:.0f} — overbought")
    elif rsi_v < 45:   risks.append(f"RSI {rsi_v:.0f} — bearish momentum")

    if macd_bull:      reasons.append("MACD positive")
    else:              risks.append("MACD negative")

    if 0 < pct20 <= 5: reasons.append(f"{pct20:.1f}% above 20 EMA — clean structure")
    elif pct20 > 8:    risks.append(f"{pct20:.1f}% above 20 EMA — extended")

    if squeeze:        reasons.append("BB squeeze — volatility compression")

    if score >= 65:
        when = "Enter 9:30-10:00 AM ET open OR Power Hour 3-4 PM ET on volume spike"
    elif score >= 45:
        when = "Wait for RVOL > 1.5x before entry. Skip 11:30 AM-2 PM midday chop"
    else:
        when = "Low confidence — observe only, skip or wait for stronger setup"

    macd_label = "BULL" if macd_bull else "BEAR"
    log(f"[{data['ticker']}] Score {score}/100 | {setup} | "
        f"ADX {adx_v:.0f} | RSI {rsi_v:.0f} | "
        f"RVOL {rvol_v:.1f}x | MACD {macd_label}",
        "ok" if score >= 65 else "warn" if score >= 45 else "err")

    return dict(
        ticker=data["ticker"], name=data["name"],
        price=price, change_pct=chg_pct,
        setup=setup, score=score,
        rsi=rsi_v, adx=adx_v, rvol=rvol_v,
        macd=macd_v, macd_bull=macd_bull,
        atr=atr_v, pct20=pct20, squeeze=squeeze,
        stop=stop, t1=t1, t2=t2, rr=rr,
        reasons=reasons, risks=risks, when=when,
    )

# ── REGIME ────────────────────────────────────────────────────────────────────
def regime_data():
    """Fetch SPY/QQQ/VIX and return plain dict (used by both CLI and API)."""
    results = {}
    for sym in ["SPY", "QQQ", "^VIX"]:
        d = fetch(sym)
        if d:
            results[sym] = d
    if len(results) < 3:
        return None
    return {
        "spy": {"price": results["SPY"]["price"],  "changePct": results["SPY"]["change_pct"]},
        "qqq": {"price": results["QQQ"]["price"],  "changePct": results["QQQ"]["change_pct"]},
        "vix": {"price": results["^VIX"]["price"], "changePct": results["^VIX"]["change_pct"]},
    }

def market_regime():
    """Return coloured regime string for terminal output."""
    log("Fetching market regime (SPY, QQQ, VIX)...", "info")
    rd = regime_data()
    if not rd:
        return "UNKNOWN"
    spy_up  = rd["spy"]["changePct"] > 0
    qqq_up  = rd["qqq"]["changePct"] > 0
    vix_val = rd["vix"]["price"]
    if spy_up and qqq_up and vix_val < 20:
        return green("RISK ON") + f"  — Long setups preferred. VIX {vix_val:.1f}"
    elif vix_val < 25:
        return yellow("NEUTRAL") + f" — Selective only. VIX {vix_val:.1f}"
    else:
        return red("RISK OFF") + f" — Avoid new longs. VIX {vix_val:.1f}"

# ── TERMINAL DISPLAY ──────────────────────────────────────────────────────────
def conf_bar(score, width=20):
    filled = int(score / 100 * width)
    bar    = "█" * filled + "░" * (width - filled)
    if score >= 65: return green(bar)
    if score >= 45: return yellow(bar)
    return red(bar)

def print_card(a, rank):
    chg_str  = f"{a['change_pct']:+.2f}%"
    chg_col  = green(chg_str) if a["change_pct"] >= 0 else red(chg_str)
    rank_line = bold(f"#{rank}  {a['ticker']}") + "  " + dim(a["name"])

    print()
    print(f"  {'─'*62}")
    print(f"  {rank_line}")
    print(f"  {cyan(a['setup'])}    ${a['price']:.2f}  {chg_col} today")
    print(f"  {'─'*62}")

    adx_c  = green(f"{a['adx']:.0f}")    if a["adx"] > 25   else yellow(f"{a['adx']:.0f}")
    rsi_c  = (red if a["rsi"] > 70 else green if a["rsi"] >= 55 else dim)(f"{a['rsi']:.0f}")
    rvol_c = green(f"{a['rvol']:.1f}x")  if a["rvol"] > 1.5 else dim(f"{a['rvol']:.1f}x")
    macd_c = green(f"▲{a['macd']:.1f}")  if a["macd_bull"]  else red(f"▼{abs(a['macd']):.1f}")
    pct_c  = green(f"+{a['pct20']:.1f}%") if a["pct20"] >= 0 else red(f"{a['pct20']:.1f}%")
    sqz    = yellow("SQUEEZE") if a["squeeze"] else dim("no squeeze")

    print(f"  ADX {adx_c}  RSI {rsi_c}  RVOL {rvol_c}  "
          f"MACD {macd_c}  vs20EMA {pct_c}  ATR ${a['atr']:.2f}  {sqz}")

    entry_s = dim(f"${a['price']:.2f}")
    stop_s  = red(f"${a['stop']:.2f}")
    t1_s    = green(f"${a['t1']:.2f}")
    t2_s    = green(f"${a['t2']:.2f}")
    rr_s    = cyan(f"1:{a['rr']}")

    print(f"\n  {'ENTRY':>10}  {'STOP':>10}  {'TARGET 1':>10}  {'TARGET 2':>10}  {'R:R':>6}")
    print(f"  {entry_s:>10}  {stop_s:>10}  {t1_s:>10}  {t2_s:>10}  {rr_s:>6}")
    print(f"\n  CONFIDENCE  {conf_bar(a['score'])}  {bold(str(a['score']))}/100")

    if a["reasons"]:
        print(f"\n  {green('WHY:')}   {dim('  ·  '.join(a['reasons']))}")
    if a["risks"]:
        print(f"  {red('RISKS:')}  {dim('  ·  '.join(a['risks']))}")
    print(f"  {yellow('WHEN:')}   {a['when']}")

def print_summary_table(results):
    rows = []
    for i, a in enumerate(results):
        rows.append([
            f"#{i+1}  {a['ticker']}",
            a["setup"],
            f"${a['price']:.2f}",
            f"{a['change_pct']:+.2f}%",
            f"{a['adx']:.0f}",
            f"{a['rsi']:.0f}",
            f"{a['rvol']:.1f}x",
            "BULL" if a["macd_bull"] else "BEAR",
            f"${a['stop']}",
            f"${a['t1']}",
            f"{a['score']}/100",
        ])
    headers = ["TICKER","SETUP","PRICE","CHG","ADX","RSI","RVOL","MACD","STOP","T1","SCORE"]
    print("\n" + tabulate(rows, headers=headers, tablefmt="simple"))

def print_timing_guide():
    best  = green("9:30-10:30 AM ET")
    chop  = yellow("11:30 AM-2 PM ET")
    power = cyan("3:00-4:00 PM ET ")
    print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  {best}   BEST WINDOW — highest volume, enter breakouts  │
  │  {chop}   MIDDAY CHOP — avoid new entries, monitor only  │
  │  {power}   POWER HOUR  — trail stops or exit positions   │
  └──────────────────────────────────────────────────────────────┘""")

# ── TERMINAL SCAN LOOP ────────────────────────────────────────────────────────
def run(tickers, watch=False):
    while True:
        now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        print(f"\n{'═'*64}")
        print(f"  {bold('MOMENTUM / SCAN')}  {dim(now)}")
        print(f"{'═'*64}")
        print_timing_guide()

        regime = market_regime()
        print(f"\n  MARKET REGIME:  {regime}\n")
        print(f"  {'─'*62}")
        print(f"  Scanning: {', '.join(tickers)}")
        print(f"  {'─'*62}")

        raw_data = [d for d in (fetch(t) for t in tickers) if d]

        if not raw_data:
            print(red("\n  All fetches failed. Check internet connection.\n"))
            if not watch:
                break
            time.sleep(300)
            continue

        results = sorted([analyze(d) for d in raw_data],
                         key=lambda x: x["score"], reverse=True)

        failed     = len(tickers) - len(raw_data)
        failed_str = red(str(failed)) if failed else dim("0")
        print(f"\n  {green(str(len(results)))} analyzed  {failed_str} failed\n")

        print_summary_table(results)
        for i, a in enumerate(results, 1):
            print_card(a, i)

        print(f"\n  {'═'*62}")
        print(f"  {dim('Not financial advice. Use stop-losses. Manage risk.')}")
        print(f"  {'═'*62}\n")

        if not watch:
            break
        print(f"\n  {dim('Next scan in 5 minutes... (Ctrl+C to stop)')}\n")
        time.sleep(300)

# ── FLASK API SERVER ──────────────────────────────────────────────────────────
def serve(port=5000):
    try:
        from flask import Flask, jsonify, request
        from flask_cors import CORS
    except ImportError:
        print(red("\n  Flask not installed."))
        print("  Run:  pip install flask flask-cors\n")
        sys.exit(1)

    app = Flask(__name__)
    CORS(app)

    # ── GET /api/quote?ticker=FLEX ──────────────────────────────────────────
    @app.route("/api/quote")
    def api_quote():
        ticker = request.args.get("ticker", "").upper().strip()
        if not ticker:
            return jsonify({"error": "ticker param required"}), 400
        data = fetch(ticker)
        if not data:
            return jsonify({"error": f"Could not fetch {ticker}"}), 500
        return jsonify(analyze(data))

    # ── GET /api/regime ─────────────────────────────────────────────────────
    @app.route("/api/regime")
    def api_regime():
        rd = regime_data()
        if not rd:
            return jsonify({"error": "regime fetch failed"}), 500
        return jsonify(rd)

    # ── GET /api/scan?tickers=FLEX,AMD,JBL ─────────────────────────────────
    # ── GET /api/scan?preset=watchlist      ─────────────────────────────────
    @app.route("/api/scan")
    def api_scan():
        preset = request.args.get("preset", "").strip()
        raw    = request.args.get("tickers", "").strip()
        if preset and preset in PRESETS:
            tickers = PRESETS[preset]
        elif raw:
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        else:
            return jsonify({"error": "pass ?tickers=FLEX,AMD or ?preset=watchlist"}), 400

        results = []
        for t in tickers:
            d = fetch(t)
            if d:
                results.append(analyze(d))
        results.sort(key=lambda x: x["score"], reverse=True)
        return jsonify(results)

    # ── startup banner ──────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print(f"  {bold('MOMENTUM SCREENER — API SERVER')}")
    print(f"{'═'*64}")
    print(f"\n  {green('Routes registered:')}")
    print(f"  {cyan('GET')}  http://localhost:{port}/api/quote?ticker=FLEX")
    print(f"  {cyan('GET')}  http://localhost:{port}/api/regime")
    print(f"  {cyan('GET')}  http://localhost:{port}/api/scan?tickers=FLEX,AMD,JBL")
    print(f"  {cyan('GET')}  http://localhost:{port}/api/scan?preset=watchlist")
    print(f"\n  {dim('Set USE_PYTHON_API = true in momentum-screener.html')}")
    print(f"  {dim('Press Ctrl+C to stop.')}\n")

    app.run(host="0.0.0.0", port=port, debug=False)

# ── CLI ENTRY POINT ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Momentum Screener — multi-factor swing & intraday signal engine"
    )
    parser.add_argument("tickers", nargs="*",
                        help="Ticker symbols e.g. FLEX AMD JBL")
    parser.add_argument("--preset", choices=list(PRESETS.keys()),
                        default="watchlist",
                        help="Preset watchlist (default: watchlist)")
    parser.add_argument("--watch",  action="store_true",
                        help="Auto-refresh every 5 minutes")
    parser.add_argument("--serve",  action="store_true",
                        help="Start Flask API server on port 5000")
    parser.add_argument("--port",   type=int, default=5000,
                        help="Port for --serve mode (default: 5000)")
    args = parser.parse_args()

    if args.serve:
        serve(port=args.port)
        return

    tickers = ([t.upper() for t in args.tickers]
               if args.tickers else PRESETS[args.preset])

    try:
        run(tickers, watch=args.watch)
    except KeyboardInterrupt:
        print(f"\n  {dim('Stopped.')}\n")

if __name__ == "__main__":
    main()
