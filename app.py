# XAU/USD Deep AI Engine v3.0
# ICT / SMC + Multi-Timeframe Quant + ML Meta-Labeling + Groq
#
# IMPORTANT:
# This system is research software. Backtest and validate before live trading.

from datetime import datetime, timezone
import json
import os
import sqlite3
import time

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="XAU/USD Deep AI Engine v3",
    layout="wide",
    page_icon="🧠",
)

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

.stApp {
    background:#07090e;
    color:#f3f4f6;
}

section[data-testid="stSidebar"] {
    background:#0f172a;
}

.card {
    background:linear-gradient(135deg,#172554,#0f172a);
    padding:18px;
    border-radius:16px;
    border:1px solid #334155;
    margin-bottom:12px;
}

.good {
    color:#22c55e;
}

.bad {
    color:#ef4444;
}

.neutral {
    color:#94a3b8;
}

.small {
    font-size:.85rem;
    color:#94a3b8;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CONFIG
# ============================================================

DB_FILE = "xau_deep_ai_v3.db"

MODEL_FILE = "xau_deep_meta_model_v3.pkl"
SCALER_FILE = "xau_deep_meta_scaler_v3.pkl"

SYMBOL = "XAU/USD"

FEATURES = [
    "direction_num",
    "ict_conf",
    "ict_buy",
    "ict_sell",
    "h4_bias",
    "h1_bias",
    "m15_trend",
    "m5_trend",
    "rsi",
    "mom3",
    "mom6",
    "range_rel",
    "body_ratio",
    "volatility_pct",
    "sweep",
    "ote_inside",
    "premium_discount",
    "fvg_present",
    "ob_present",
    "displacement_present",
]


# ============================================================
# DATABASE
# ============================================================

def get_db():
    return sqlite3.connect(DB_FILE, timeout=20)


def init_db():

    con = get_db()
    c = con.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS settings(
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS trades(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            symbol TEXT,
            direction TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            result REAL,
            win INTEGER,
            setup TEXT,
            confidence REAL,
            ml_confidence REAL,
            groq_confidence REAL,
            note TEXT,
            features TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            direction TEXT,
            setup TEXT,
            score REAL,
            ml_confidence REAL,
            accepted INTEGER,
            snapshot TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS active_trade(
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            direction TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            time TEXT,
            setup TEXT,
            confidence REAL,
            ml_confidence REAL,
            groq_confidence REAL,
            features TEXT,
            snapshot TEXT
        )
        """
    )

    con.commit()
    con.close()


init_db()


def save_setting(key, value):

    con = get_db()

    con.execute(
        """
        INSERT OR REPLACE INTO settings(key,value)
        VALUES(?,?)
        """,
        (key, str(value)),
    )

    con.commit()
    con.close()


def load_setting(key, default=""):

    con = get_db()

    row = con.execute(
        """
        SELECT value
        FROM settings
        WHERE key=?
        """,
        (key,),
    ).fetchone()

    con.close()

    return row[0] if row else default


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Data / AI")

twelve_key = st.sidebar.text_input(
    "Twelve Data API",
    type="password",
    value=load_setting("twelve_key", ""),
)

save_setting("twelve_key", twelve_key)


ntfy_channel = st.sidebar.text_input(
    "Ntfy Channel",
    value=load_setting(
        "ntfy",
        "xau_deep_channel",
    ),
)

save_setting("ntfy", ntfy_channel)


# ------------------------------------------------------------
# GROQ
# ------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.header("🧠 Groq Risk Reviewer")

use_groq = st.sidebar.checkbox(
    "تفعيل مراجعة Groq",
    value=load_setting("use_groq", "1") == "1",
)

save_setting(
    "use_groq",
    "1" if use_groq else "0",
)


groq_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=load_setting("groq_key", ""),
)

save_setting("groq_key", groq_key)


groq_model = st.sidebar.text_input(
    "Groq Model",
    value=load_setting(
        "groq_model",
        "openai/gpt-oss-20b",
    ),
)

save_setting("groq_model", groq_model)


groq_min = st.sidebar.slider(
    "Minimum Groq confidence %",
    40,
    95,
    60,
)


# ------------------------------------------------------------
# AI
# ------------------------------------------------------------

ml_min = st.sidebar.slider(
    "Minimum ML confidence %",
    50,
    95,
    65,
)

candidate_min = st.sidebar.slider(
    "Minimum ICT / Quant score",
    40,
    100,
    68,
)


# ------------------------------------------------------------
# ICT
# ------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.header("🧭 ICT / SMC")

swing = st.sidebar.slider(
    "Swing sensitivity",
    2,
    8,
    3,
)

disp_mult = st.sidebar.slider(
    "Displacement × ATR",
    0.8,
    2.5,
    1.2,
    0.1,
)

fvg_min_atr = st.sidebar.slider(
    "Minimum FVG × ATR",
    0.0,
    1.0,
    0.05,
    0.05,
)

ob_lookback = st.sidebar.slider(
    "Order Block lookback",
    20,
    120,
    60,
)

liquidity_lookback = st.sidebar.slider(
    "Liquidity lookback",
    20,
    150,
    80,
)


# ------------------------------------------------------------
# RISK
# ------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.header("🎯 Risk Management")

atr_sl = st.sidebar.slider(
    "SL ATR multiplier",
    0.8,
    3.0,
    1.3,
    0.1,
)

rr = st.sidebar.slider(
    "Reward / Risk",
    1.2,
    5.0,
    2.0,
    0.1,
)

risk_percent = st.sidebar.slider(
    "Risk % per trade",
    0.1,
    2.0,
    0.5,
    0.1,
)

max_daily_losses = st.sidebar.slider(
    "Maximum daily losses",
    1,
    5,
    2,
)

max_daily_loss = st.sidebar.slider(
    "Maximum daily loss %",
    0.5,
    5.0,
    1.5,
    0.5,
)

one_position = st.sidebar.checkbox(
    "One position at a time",
    True,
)


# ------------------------------------------------------------
# SESSION
# ------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.header("🕒 Trading Sessions — UTC")

session_start = st.sidebar.slider(
    "Session start",
    0,
    23,
    7,
)

session_end = st.sidebar.slider(
    "Session end",
    0,
    23,
    20,
)

avoid_friday = st.sidebar.checkbox(
    "Avoid late Friday",
    True,
)


# ============================================================
# ALERTS
# ============================================================

def send_alert(
    message,
    title="🧠 XAU Deep AI",
):

    if not ntfy_channel:
        return

    try:

        channel = ntfy_channel.strip().split("/")[-1]

        requests.post(
            f"https://ntfy.sh/{channel}",
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": "high",
            },
            timeout=5,
        )

    except Exception:
        pass


# ============================================================
# TWELVE DATA
# ============================================================

@st.cache_data(ttl=240)
def fetch_twelve_data(
    api_key,
    symbol,
    interval,
    outputsize=500,
):

    if not api_key:
        return pd.DataFrame()

    try:

        response = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "apikey": api_key,
            },
            timeout=10,
        )

        data = response.json()

        if "values" not in data:

            st.session_state["td_error"] = data.get(
                "message",
                "Twelve Data error",
            )

            return pd.DataFrame()

        df = pd.DataFrame(data["values"])

        df = df[
            [
                "datetime",
                "open",
                "high",
                "low",
                "close",
            ]
        ].copy()

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
            utc=True,
        )

        for column in [
            "open",
            "high",
            "low",
            "close",
        ]:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = (
            df.dropna()
            .sort_values("datetime")
            .drop_duplicates("datetime")
            .reset_index(drop=True)
        )

        return df

    except Exception as e:

        st.session_state["td_error"] = str(e)

        return pd.DataFrame()


# ============================================================
# QUANT INDICATORS
# ============================================================

def add_indicators(df):

    if df.empty:
        return df

    x = df.copy()

    tr = pd.concat(
        [
            x.high - x.low,
            (x.high - x.close.shift()).abs(),
            (x.low - x.close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr"] = tr.rolling(14).mean()

    x["ema20"] = x.close.ewm(
        span=20,
        adjust=False,
    ).mean()

    x["ema50"] = x.close.ewm(
        span=50,
        adjust=False,
    ).mean()

    x["ema200"] = x.close.ewm(
        span=200,
        adjust=False,
    ).mean()

    delta = x.close.diff()

    gain = (
        delta.clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
        .rolling(14)
        .mean()
    )

    rs = gain / (loss + 1e-9)

    x["rsi"] = (
        100
        - 100 / (1 + rs)
    )

    x["mom3"] = (
        x.close.pct_change(3)
        * 100
    )

    x["mom6"] = (
        x.close.pct_change(6)
        * 100
    )

    x["range"] = x.high - x.low

    x["body"] = (
        x.close - x.open
    ).abs()

    x["body_ratio"] = (
        x.body
        / (x.range + 1e-9)
    )

    x["range_rel"] = (
        x.range
        / (x.atr + 1e-9)
    )

    x["volatility_pct"] = (
        x.atr.rolling(100)
        .rank(pct=True)
        * 100
    )

    return (
        x.dropna()
        .reset_index(drop=True)
    )


def closed_candles(df):

    if len(df) <= 2:
        return df.copy()

    return df.iloc[:-1].copy()


# ============================================================
# SWING STRUCTURE
# ============================================================

def find_swings(df, n=3):

    highs = []
    lows = []

    h = df.high.values
    l = df.low.values

    for i in range(
        n,
        len(df) - n,
    ):

        if (
            h[i]
            >= h[i - n:i + n + 1].max()
        ):

            highs.append(
                (
                    i,
                    float(h[i]),
                )
            )

        if (
            l[i]
            <= l[i - n:i + n + 1].min()
        ):

            lows.append(
                (
                    i,
                    float(l[i]),
                )
            )

    return highs, lows


# ============================================================
# MARKET STRUCTURE
# ============================================================

def analyze_structure(
    df,
    n=3,
):

    highs, lows = find_swings(
        df,
        n,
    )

    events = (
        [
            (i, "H", p)
            for i, p in highs
        ]
        +
        [
            (i, "L", p)
            for i, p in lows
        ]
    )

    events.sort()

    bias = "NEUTRAL"

    last_high = None
    last_low = None

    breaks = []

    for i, kind, price in events:

        if kind == "H":

            if (
                last_high is not None
                and price > last_high
            ):

                label = (
                    "BOS"
                    if bias == "BULLISH"
                    else "CHoCH"
                )

                breaks.append(
                    {
                        "type": label,
                        "direction": "BULLISH",
                        "price": price,
                        "index": i,
                    }
                )

                bias = "BULLISH"

            last_high = price

        else:

            if (
                last_low is not None
                and price < last_low
            ):

                label = (
                    "BOS"
                    if bias == "BEARISH"
                    else "CHoCH"
                )

                breaks.append(
                    {
                        "type": label,
                        "direction": "BEARISH",
                        "price": price,
                        "index": i,
                    }
                )

                bias = "BEARISH"

            last_low = price

    return {
        "bias": bias,
        "highs": highs,
        "lows": lows,
        "breaks": breaks[-10:],
    }


# ============================================================
# DISPLACEMENT
# ============================================================

def detect_displacement(
    df,
    multiplier=1.2,
):

    results = []

    start = max(
        0,
        len(df) - 20,
    )

    for i in range(
        start,
        len(df),
    ):

        row = df.iloc[i]

        if row.atr <= 0:
            continue

        body = row.close - row.open

        if (
            abs(body)
            > multiplier * row.atr
            and row.body_ratio >= 0.65
        ):

            results.append(
                {
                    "index": i,
                    "direction": (
                        "BULLISH"
                        if body > 0
                        else "BEARISH"
                    ),
                    "strength": round(
                        abs(body)
                        / row.atr,
                        2,
                    ),
                }
            )

    return results


# ============================================================
# FAIR VALUE GAPS
# ============================================================

def detect_fvg(
    df,
    minimum_atr=0.05,
):

    bullish = []
    bearish = []

    for i in range(
        2,
        len(df),
    ):

        first = df.iloc[i - 2]
        middle = df.iloc[i - 1]
        third = df.iloc[i]

        atr = middle.atr

        if atr <= 0:
            continue

        # Bullish FVG
        if first.high < third.low:

            size = (
                third.low
                - first.high
            )

            if size >= minimum_atr * atr:

                bullish.append(
                    {
                        "index": i,
                        "bottom": float(first.high),
                        "top": float(third.low),
                        "size": float(size),
                    }
                )

        # Bearish FVG
        if first.low > third.high:

            size = (
                first.low
                - third.high
            )

            if size >= minimum_atr * atr:

                bearish.append(
                    {
                        "index": i,
                        "bottom": float(third.high),
                        "top": float(first.low),
                        "size": float(size),
                    }
                )

    return bullish[-5:], bearish[-5:]


# ============================================================
# ORDER BLOCKS
# ============================================================

def detect_order_blocks(
    df,
    lookback=60,
    displacement_multiplier=1.2,
):

    start = max(
        1,
        len(df) - lookback,
    )

    bullish = []
    bearish = []

    for i in range(
        start,
        len(df),
    ):

        row = df.iloc[i]

        if row.atr <= 0:
            continue

        body = row.close - row.open

        if (
            abs(body)
            <= displacement_multiplier
            * row.atr
        ):

            continue

        previous = df.iloc[i - 1]

        # Bullish displacement after bearish candle
        if (
            body > 0
            and previous.close
            < previous.open
        ):

            bullish.append(
                {
                    "index": i - 1,
                    "top": float(previous.open),
                    "bottom": float(previous.low),
                    "strength": round(
                        abs(body)
                        / row.atr,
                        2,
                    ),
                }
            )

        # Bearish displacement after bullish candle
        if (
            body < 0
            and previous.close
            > previous.open
        ):

            bearish.append(
                {
                    "index": i - 1,
                    "top": float(previous.high),
                    "bottom": float(previous.open),
                    "strength": round(
                        abs(body)
                        / row.atr,
                        2,
                    ),
                }
            )

    return (
        bullish[-1] if bullish else None,
        bearish[-1] if bearish else None,
    )


# ============================================================
# LIQUIDITY
# ============================================================

def detect_liquidity(
    df,
    structure,
):

    highs = structure["highs"][-8:]
    lows = structure["lows"][-8:]

    if not highs or not lows:
        return {}

    bsl = max(
        price
        for _, price in highs
    )

    ssl = min(
        price
        for _, price in lows
    )

    last = df.iloc[-1]

    sweep = None

    if (
        last.high > bsl
        and last.close < bsl
    ):

        sweep = "BSL_SWEEP"

    elif (
        last.low < ssl
        and last.close > ssl
    ):

        sweep = "SSL_SWEEP"

    return {
        "bsl": float(bsl),
        "ssl": float(ssl),
        "sweep": sweep,
    }


# ============================================================
# PREMIUM / DISCOUNT
# ============================================================

def calculate_premium_discount(
    structure,
    current_price,
):

    if (
        not structure["highs"]
        or not structure["lows"]
    ):

        return {}

    high = structure["highs"][-1][1]
    low = structure["lows"][-1][1]

    top = max(high, low)
    bottom = min(high, low)

    equilibrium = (
        top + bottom
    ) / 2

    zone = (
        "PREMIUM"
        if current_price > equilibrium
        else "DISCOUNT"
    )

    return {
        "high": top,
        "low": bottom,
        "equilibrium": equilibrium,
        "zone": zone,
    }


# ============================================================
# OTE
# ============================================================

def calculate_ote(
    structure,
    current_price,
    bias,
):

    if (
        not structure["highs"]
        or not structure["lows"]
    ):

        return None

    high = structure["highs"][-1][1]
    low = structure["lows"][-1][1]

    difference = high - low

    if difference <= 0:
        return None

    if bias == "BULLISH":

        a = high - difference * 0.79
        b = high - difference * 0.618

    elif bias == "BEARISH":

        a = low + difference * 0.618
        b = low + difference * 0.79

    else:

        return None

    bottom = min(a, b)
    top = max(a, b)

    return {
        "bottom": bottom,
        "top": top,
        "inside": (
            bottom
            <= current_price
            <= top
        ),
    }


# ============================================================
# BREAKER BLOCKS
# ============================================================

def detect_breakers(df):

    bullish_ob, bearish_ob = detect_order_blocks(
        df,
        60,
        1.0,
    )

    last = df.iloc[-1]

    breakers = []

    if (
        bullish_ob
        and last.close
        < bullish_ob["bottom"]
    ):

        breakers.append(
            {
                "type": "BEARISH_BREAKER",
                "top": bullish_ob["top"],
                "bottom": bullish_ob["bottom"],
            }
        )

    if (
        bearish_ob
        and last.close
        > bearish_ob["top"]
    ):

        breakers.append(
            {
                "type": "BULLISH_BREAKER",
                "top": bearish_ob["top"],
                "bottom": bearish_ob["bottom"],
            }
        )

    return breakers[-2:]


# ============================================================
# SESSION ANALYSIS
# ============================================================

def analyze_sessions(df):

    if df.empty:
        return {}

    x = df.copy()

    x["hour"] = x.datetime.dt.hour

    sessions = {
        "ASIA": (0, 7),
        "LONDON": (7, 13),
        "NEW_YORK": (13, 20),
    }

    result = {}

    for name, (
        start,
        end,
    ) in sessions.items():

        window = x[
            (x.hour >= start)
            & (x.hour < end)
        ].tail(24)

        if window.empty:
            continue

        result[name] = {
            "high": float(
                window.high.max()
            ),
            "low": float(
                window.low.min()
            ),
            "range": float(
                window.high.max()
                - window.low.min()
            ),
        }

    return result


# ============================================================
# COMPLETE ICT ENGINE
# ============================================================

def run_ict_engine(
    df,
    swing_lookback=3,
    displacement_multiplier=1.2,
    minimum_fvg_atr=0.05,
    ob_lookback=60,
):

    if len(df) < 80:
        return None

    x = closed_candles(df)

    structure_data = analyze_structure(
        x,
        swing_lookback,
    )

    current_price = float(
        x.close.iloc[-1]
    )

    bullish_fvg, bearish_fvg = detect_fvg(
        x,
        minimum_fvg_atr,
    )

    bullish_ob, bearish_ob = detect_order_blocks(
        x,
        ob_lookback,
        displacement_multiplier,
    )

    liquidity = detect_liquidity(
        x,
        structure_data,
    )

    displacement_data = detect_displacement(
        x,
        displacement_multiplier,
    )

    premium_discount = calculate_premium_discount(
        structure_data,
        current_price,
    )

    ote = calculate_ote(
        structure_data,
        current_price,
        structure_data["bias"],
    )

    breakers = detect_breakers(x)

    sessions = analyze_sessions(x)

    buy_score = 0
    sell_score = 0

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    if structure_data["bias"] == "BULLISH":
        buy_score += 25

    elif structure_data["bias"] == "BEARISH":
        sell_score += 25

    # --------------------------------------------------------
    # Liquidity
    # --------------------------------------------------------

    if liquidity.get("sweep") == "SSL_SWEEP":
        buy_score += 25

    elif liquidity.get("sweep") == "BSL_SWEEP":
        sell_score += 25

    # --------------------------------------------------------
    # Order Blocks
    # --------------------------------------------------------

    if bullish_ob:
        buy_score += 15

    if bearish_ob:
        sell_score += 15

    # --------------------------------------------------------
    # FVG
    # --------------------------------------------------------

    if bullish_fvg:
        buy_score += 10

    if bearish_fvg:
        sell_score += 10

    # --------------------------------------------------------
    # Displacement
    # --------------------------------------------------------

    if any(
        x["direction"] == "BULLISH"
        for x in displacement_data[-3:]
    ):

        buy_score += 10

    if any(
        x["direction"] == "BEARISH"
        for x in displacement_data[-3:]
    ):

        sell_score += 10

    # --------------------------------------------------------
    # Premium / Discount
    # --------------------------------------------------------

    if premium_discount.get("zone") == "DISCOUNT":
        buy_score += 10

    if premium_discount.get("zone") == "PREMIUM":
        sell_score += 10

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if buy_score > sell_score:
        direction = "BUY"

    elif sell_score > buy_score:
        direction = "SELL"

    else:
        direction = "NONE"

    confidence = min(
        100,
        max(
            buy_score,
            sell_score,
        ),
    )

    return {
        "bias": structure_data["bias"],
        "structure": structure_data,
        "liquidity": liquidity,
        "bull_fvg": (
            bullish_fvg[-1]
            if bullish_fvg
            else None
        ),
        "bear_fvg": (
            bearish_fvg[-1]
            if bearish_fvg
            else None
        ),
        "bull_ob": bullish_ob,
        "bear_ob": bearish_ob,
        "displacements": displacement_data[-3:],
        "pd": premium_discount,
        "ote": ote,
        "breakers": breakers,
        "sessions": sessions,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "direction": direction,
        "ict_confidence": confidence,
        "price": current_price,
        "atr": float(x.atr.iloc[-1]),
        "rsi": float(x.rsi.iloc[-1]),
    }


# ============================================================
# QUANT REGIME
# ============================================================

def calculate_quant_regime(
    h1,
    h4,
    m15,
    m5,
    ict,
):

    h1_last = h1.iloc[-1]
    h4_last = h4.iloc[-1]
    m15_last = m15.iloc[-1]
    m5_last = m5.iloc[-1]

    if (
        h1_last.close
        > h1_last.ema200
        and h1_last.ema50
        > h1_last.ema200
    ):

        h1_bias = 1

    elif (
        h1_last.close
        < h1_last.ema200
        and h1_last.ema50
        < h1_last.ema200
    ):

        h1_bias = -1

    else:

        h1_bias = 0

    if (
        h4_last.close
        > h4_last.ema200
        and h4_last.ema50
        > h4_last.ema200
    ):

        h4_bias = 1

    elif (
        h4_last.close
        < h4_last.ema200
        and h4_last.ema50
        < h4_last.ema200
    ):

        h4_bias = -1

    else:

        h4_bias = 0

    if (
        m15_last.close
        > m15_last.ema50
        and m15_last.ema20
        > m15_last.ema50
    ):

        m15_trend = 1

    elif (
        m15_last.close
        < m15_last.ema50
        and m15_last.ema20
        < m15_last.ema50
    ):

        m15_trend = -1

    else:

        m15_trend = 0

    if m5_last.close > m5_last.ema50:
        m5_trend = 1

    elif m5_last.close < m5_last.ema50:
        m5_trend = -1

    else:

        m5_trend = 0

    return {
        "h4_bias": h4_bias,
        "h1_bias": h1_bias,
        "m15_trend": m15_trend,
        "m5_trend": m5_trend,
        "rsi": float(m5_last.rsi),
        "atr": float(m5_last.atr),
        "mom3": float(m5_last.mom3),
        "mom6": float(m5_last.mom6),
        "range_rel": float(m5_last.range_rel),
        "body_ratio": float(m5_last.body_ratio),
        "volatility_pct": float(
            m5_last.volatility_pct
        ),
        "ict_buy": ict["buy_score"],
        "ict_sell": ict["sell_score"],
        "ict_conf": ict["ict_confidence"],
    }


# ============================================================
# CANDIDATE GENERATOR
# ============================================================

def generate_candidate(
    h4,
    h1,
    m15,
    m5,
    ict,
):

    quant = calculate_quant_regime(
        h1,
        h4,
        m15,
        m5,
        ict,
    )

    direction = ict["direction"]

    if direction == "NONE":
        return None

    sweep = ict[
        "liquidity"
    ].get("sweep")

    sweep_flag = (
        1
        if sweep
        else 0
    )

    ote_inside = (
        1
        if (
            ict["ote"]
            and ict["ote"]["inside"]
        )
        else 0
    )

    premium_discount = (
        1
        if ict["pd"].get("zone")
        == "PREMIUM"
        else -1
    )

    fvg_present = (
        1
        if (
            ict["bull_fvg"]
            or ict["bear_fvg"]
        )
        else 0
    )

    ob_present = (
        1
        if (
            ict["bull_ob"]
            or ict["bear_ob"]
        )
        else 0
    )

    displacement_present = (
        1
        if ict["displacements"]
        else 0
    )

    alignment = 0

    if direction == "BUY":

        if quant["h4_bias"] == 1:
            alignment += 10

        if quant["h1_bias"] == 1:
            alignment += 10

        if quant["m15_trend"] == 1:
            alignment += 10

        if (
            sweep == "SSL_SWEEP"
        ):
            alignment += 15

        if ote_inside:
            alignment += 10

        if ict["bull_ob"]:
            alignment += 10

        if ict["bull_fvg"]:
            alignment += 5

    else:

        if quant["h4_bias"] == -1:
            alignment += 10

        if quant["h1_bias"] == -1:
            alignment += 10

        if quant["m15_trend"] == -1:
            alignment += 10

        if (
            sweep == "BSL_SWEEP"
        ):
            alignment += 15

        if ote_inside:
            alignment += 10

        if ict["bear_ob"]:
            alignment += 10

        if ict["bear_fvg"]:
            alignment += 5

    score = min(
        100,
        ict["ict_confidence"] * 0.5
        + alignment * 0.5,
    )

    if sweep:

        setup = "ICT_LIQUIDITY_SWEEP"

    elif ob_present or fvg_present:

        setup = "ICT_FVG_OB"

    else:

        setup = "ICT_STRUCTURE"

    return {
        "direction": direction,
        "score": round(score, 1),
        "setup": setup,
        "q": quant,
        "sweep": sweep_flag,
        "ote_inside": ote_inside,
        "premium_discount": premium_discount,
        "fvg_present": fvg_present,
        "ob_present": ob_present,
        "displacement_present": displacement_present,
        "price": ict["price"],
        "atr": ict["atr"],
    }


# ============================================================
# ML FEATURES
# ============================================================

def make_feature_vector(candidate):

    q = candidate["q"]

    direction_num = (
        1
        if candidate["direction"] == "BUY"
        else -1
    )

    return np.array(
        [[
            direction_num,
            q["ict_conf"],
            q["ict_buy"],
            q["ict_sell"],
            q["h4_bias"],
            q["h1_bias"],
            q["m15_trend"],
            q["m5_trend"],
            q["rsi"],
            q["mom3"],
            q["mom6"],
            q["range_rel"],
            q["body_ratio"],
            q["volatility_pct"],
            candidate["sweep"],
            candidate["ote_inside"],
            candidate["premium_discount"],
            candidate["fvg_present"],
            candidate["ob_present"],
            candidate["displacement_present"],
        ]],
        dtype=float,
    )


# ============================================================
# MODEL
# ============================================================

def load_model():

    if (
        os.path.exists(MODEL_FILE)
        and os.path.exists(SCALER_FILE)
    ):

        try:

            return (
                joblib.load(MODEL_FILE),
                joblib.load(SCALER_FILE),
            )

        except Exception:
            pass

    return None, None


model, scaler = load_model()


def train_model_from_history():

    global model
    global scaler

    con = get_db()

    trades = pd.read_sql(
        """
        SELECT *
        FROM trades
        WHERE features IS NOT NULL
        """,
        con,
    )

    con.close()

    if len(trades) < 30:
        return False

    X = []
    y = []

    for _, row in trades.iterrows():

        try:

            feature_dict = json.loads(
                row["features"]
            )

            X.append(
                [
                    feature_dict[key]
                    for key in FEATURES
                ]
            )

            y.append(
                int(row["win"])
            )

        except Exception:
            continue

    if (
        len(X) < 30
        or len(set(y)) < 2
    ):

        return False

    scaler_new = StandardScaler()

    X_scaled = scaler_new.fit_transform(
        np.asarray(
            X,
            dtype=float,
        )
    )

    model_new = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=2.0,
        random_state=42,
    )

    model_new.fit(
        X_scaled,
        y,
    )

    joblib.dump(
        model_new,
        MODEL_FILE,
    )

    joblib.dump(
        scaler_new,
        SCALER_FILE,
    )

    model = model_new
    scaler = scaler_new

    return True


def get_ml_probability(
    candidate,
):

    if (
        model is None
        or scaler is None
    ):

        return None

    try:

        features = make_feature_vector(
            candidate
        )

        probabilities = model.predict_proba(
            scaler.transform(features)
        )[0]

        if len(probabilities) != 2:
            return None

        return float(
            probabilities[1]
        )

    except Exception:

        return None


# ============================================================
# GROQ
# ============================================================

def groq_review(
    candidate,
    ict,
):

    if (
        not use_groq
        or not groq_key
    ):

        return None

    snapshot = {
        "direction": candidate["direction"],
        "setup": candidate["setup"],
        "score": candidate["score"],
        "ml_confidence": candidate.get(
            "ml_confidence"
        ),
        "ict": {
            "bias": ict["bias"],
            "buy_score": ict["buy_score"],
            "sell_score": ict["sell_score"],
            "liquidity_sweep": ict[
                "liquidity"
            ].get("sweep"),
            "bullish_ob": bool(
                ict["bull_ob"]
            ),
            "bearish_ob": bool(
                ict["bear_ob"]
            ),
            "bullish_fvg": bool(
                ict["bull_fvg"]
            ),
            "bearish_fvg": bool(
                ict["bear_fvg"]
            ),
            "ote": ict["ote"],
            "premium_discount": ict["pd"],
            "displacements": ict[
                "displacements"
            ][-2:],
        },
        "quant": candidate["q"],
    }

    prompt = (
        "You are a risk-control reviewer for a "
        "systematic XAU/USD trading engine. "
        "Evaluate only the supplied data. "
        "Do not invent missing information. "
        "You are a filter, never the trade generator. "
        "Reject weak confluence, conflicting higher "
        "timeframe direction, late entries and dangerous "
        "volatility. Return JSON only using exactly: "
        "{agree:boolean, confidence:number, reason:string, "
        "risk_flags:[string]}\n\n"
        + json.dumps(
            snapshot,
            ensure_ascii=False,
        )
    )

    try:

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": (
                    f"Bearer {groq_key}"
                ),
                "Content-Type": (
                    "application/json"
                ),
            },
            json={
                "model": groq_model,
                "temperature": 0,
                "max_completion_tokens": 300,
                "response_format": {
                    "type": "json_object"
                },
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            },
            timeout=15,
        )

        data = response.json()

        text = data[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]

        parsed = json.loads(
            text.replace(
                "```json",
                "",
            )
            .replace(
                "```",
                "",
            )
            .strip()
        )

        return {
            "agree": bool(
                parsed.get(
                    "agree"
                )
            ),
            "confidence": float(
                parsed.get(
                    "confidence",
                    0,
                )
            ),
            "reason": str(
                parsed.get(
                    "reason",
                    "",
                )
            ),
            "risk_flags": parsed.get(
                "risk_flags",
                [],
            ),
        }

    except Exception:

        return None


# ============================================================
# SESSION GUARD
# ============================================================

def session_allowed():

    now = datetime.now(
        timezone.utc
    )

    if (
        avoid_friday
        and now.weekday() == 4
        and now.hour >= 18
    ):

        return False

    if session_start < session_end:

        return (
            session_start
            <= now.hour
            < session_end
        )

    return (
        now.hour >= session_start
        or now.hour < session_end
    )


# ============================================================
# DAILY GUARD
# ============================================================

def daily_guard():

    today = datetime.now(
        timezone.utc
    ).date().isoformat()

    con = get_db()

    trades = pd.read_sql(
        """
        SELECT *
        FROM trades
        WHERE date=?
        """,
        con,
        params=(today,),
    )

    con.close()

    if trades.empty:
        return True

    losses = int(
        (trades.win == 0).sum()
    )

    return losses < max_daily_losses


# ============================================================
# TRADE LEVELS
# ============================================================

def calculate_trade_levels(
    candidate,
    ict,
):

    price = candidate["price"]
    atr = candidate["atr"]
    direction = candidate["direction"]

    structure = ict["structure"]

    lows = [
        price
        for _, price
        in structure["lows"][-3:]
    ]

    highs = [
        price
        for _, price
        in structure["highs"][-3:]
    ]

    if direction == "BUY":

        structural_stop = (
            min(lows)
            if lows
            else price
            - atr * atr_sl
        )

        sl = min(
            price - atr * atr_sl,
            structural_stop,
        )

        tp = (
            price
            + (price - sl) * rr
        )

    else:

        structural_stop = (
            max(highs)
            if highs
            else price
            + atr * atr_sl
        )

        sl = max(
            price + atr * atr_sl,
            structural_stop,
        )

        tp = (
            price
            - (sl - price) * rr
        )

    return (
        round(price, 2),
        round(sl, 2),
        round(tp, 2),
    )


# ============================================================
# CANDIDATE AUDIT
# ============================================================

def save_candidate_audit(
    candidate,
    accepted,
    ml_confidence,
):

    con = get_db()

    con.execute(
        """
        INSERT INTO candidates(
            time,
            direction,
            setup,
            score,
            ml_confidence,
            accepted,
            snapshot
        )
        VALUES(?,?,?,?,?,?,?)
        """,
        (
            datetime.now(
                timezone.utc
            ).isoformat(),
            candidate["direction"],
            candidate["setup"],
            candidate["score"],
            (
                ml_confidence
                if ml_confidence is not None
                else -1
            ),
            int(accepted),
            json.dumps(
                candidate,
                default=str,
            ),
        ),
    )

    con.commit()
    con.close()


# ============================================================
# OPEN TRADE
# ============================================================

def open_trade(
    candidate,
    ict,
    ml_confidence,
    groq_result,
):

    entry, sl, tp = calculate_trade_levels(
        candidate,
        ict,
    )

    vector = make_feature_vector(
        candidate
    )[0]

    feature_dict = {
        key: float(value)
        for key, value
        in zip(
            FEATURES,
            vector,
        )
    }

    snapshot = {
        "ict": ict,
        "candidate": candidate,
    }

    con = get_db()

    con.execute(
        "DELETE FROM active_trade"
    )

    con.execute(
        """
        INSERT INTO active_trade(
            id,
            symbol,
            direction,
            entry,
            sl,
            tp,
            time,
            setup,
            confidence,
            ml_confidence,
            groq_confidence,
            features,
            snapshot
        )
        VALUES(
            1,?,?,?,?,?,?,?,?,?,?,?,?
        )
        """,
        (
            SYMBOL,
            candidate["direction"],
            entry,
            sl,
            tp,
            datetime.now(
                timezone.utc
            ).isoformat(),
            candidate["setup"],
            candidate["score"],
            (
                ml_confidence
                if ml_confidence is not None
                else -1
            ),
            (
                groq_result["confidence"]
                if groq_result
                else -1
            ),
            json.dumps(
                feature_dict
            ),
            json.dumps(
                snapshot,
                default=str,
            ),
        ),
    )

    con.commit()
    con.close()

    groq_conf = (
        groq_result["confidence"]
        if groq_result
        else -1
    )

    send_alert(
        f"""
XAU/USD {candidate["direction"]}

Setup: {candidate["setup"]}

Entry: {entry}
SL: {sl}
TP: {tp}

ICT Score:
{candidate["score"]:.1f}%

ML Confidence:
{ml_confidence:.1f}%

Groq:
{groq_conf:.1f}%
""",
        "🚨 XAU Deep AI Trade",
    )


# ============================================================
# MULTI-TIMEFRAME PIPELINE
# ============================================================

def build_market_pipeline():

    datasets = {}

    timeframes = {
        "1h": 500,
        "4h": 300,
        "15min": 700,
        "5min": 900,
    }

    for timeframe, size in timeframes.items():

        raw = fetch_twelve_data(
            twelve_key,
            SYMBOL,
            timeframe,
            size,
        )

        datasets[timeframe] = add_indicators(
            raw
        )

    if any(
        len(df) < 100
        for df in datasets.values()
    ):

        return None, datasets

    h1 = datasets["1h"]
    h4 = datasets["4h"]
    m15 = datasets["15min"]
    m5 = datasets["5min"]

    ict = run_ict_engine(
        m15,
        swing,
        disp_mult,
        fvg_min_atr,
        ob_lookback,
    )

    if ict is None:
        return None, datasets

    candidate = generate_candidate(
        h4,
        h1,
        m15,
        m5,
        ict,
    )

    if candidate is None:
        return None, datasets

    return {
        "candidate": candidate,
        "ict": ict,
        "h1": h1,
        "h4": h4,
        "m15": m15,
        "m5": m5,
    }, datasets


# ============================================================
# AUTONOMOUS DECISION ENGINE
# ============================================================

def autonomous_decision(
    packet,
):

    if packet is None:

        return (
            "NO_DATA",
            None,
        )

    candidate = packet[
        "candidate"
    ]

    ict = packet["ict"]

    # --------------------------------------------------------
    # Active position
    # --------------------------------------------------------

    if one_position:

        con = get_db()

        active = con.execute(
            """
            SELECT 1
            FROM active_trade
            WHERE id=1
            """
        ).fetchone()

        con.close()

        if active:

            return (
                "ACTIVE_POSITION",
                None,
            )

    # --------------------------------------------------------
    # Session
    # --------------------------------------------------------

    if not session_allowed():

        return (
            "OUT_OF_SESSION",
            candidate,
        )

    # --------------------------------------------------------
    # Daily protection
    # --------------------------------------------------------

    if not daily_guard():

        return (
            "DAILY_GUARD",
            candidate,
        )

    # --------------------------------------------------------
    # ICT score
    # --------------------------------------------------------

    if (
        candidate["score"]
        < candidate_min
    ):

        save_candidate_audit(
            candidate,
            False,
            None,
        )

        return (
            "ICT_SCORE_TOO_LOW",
            candidate,
        )

    q = candidate["q"]

    # --------------------------------------------------------
    # H4/H1/M15 conflict
    # --------------------------------------------------------

    if candidate["direction"] == "BUY":

        if (
            q["h4_bias"] == -1
            or q["h1_bias"] == -1
            or q["m15_trend"] == -1
        ):

            save_candidate_audit(
                candidate,
                False,
                None,
            )

            return (
                "HTF_CONFLICT",
                candidate,
            )

    else:

        if (
            q["h4_bias"] == 1
            or q["h1_bias"] == 1
            or q["m15_trend"] == 1
        ):

            save_candidate_audit(
                candidate,
                False,
                None,
            )

            return (
                "HTF_CONFLICT",
                candidate,
            )

    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------

    ml_probability = get_ml_probability(
        candidate
    )

    if ml_probability is not None:

        ml_confidence = (
            ml_probability * 100
        )

        candidate[
            "ml_confidence"
        ] = ml_confidence

        if (
            ml_confidence
            < ml_min
        ):

            save_candidate_audit(
                candidate,
                False,
                ml_confidence,
            )

            return (
                "ML_REJECT",
                candidate,
            )

    else:

        ml_confidence = None

    # --------------------------------------------------------
    # Groq
    # --------------------------------------------------------

    groq_result = groq_review(
        candidate,
        ict,
    )

    if (
        use_groq
        and groq_result is not None
    ):

        if (
            not groq_result["agree"]
            or groq_result[
                "confidence"
            ] < groq_min
        ):

            save_candidate_audit(
                candidate,
                False,
                ml_confidence,
            )

            return (
                "GROQ_REJECT",
                candidate,
            )

    # --------------------------------------------------------
    # Accept
    # --------------------------------------------------------

    save_candidate_audit(
        candidate,
        True,
        ml_confidence,
    )

    open_trade(
        candidate,
        ict,
        ml_confidence
        if ml_confidence is not None
        else 0,
        groq_result,
    )

    return (
        "TRADE_OPENED",
        candidate,
    )


# ============================================================
# ACTIVE TRADE MONITOR
# ============================================================

def monitor_active_trade(
    m5,
):

    con = get_db()

    active = pd.read_sql(
        """
        SELECT *
        FROM active_trade
        WHERE id=1
        """,
        con,
    )

    con.close()

    if (
        active.empty
        or m5.empty
    ):

        return None

    trade = active.iloc[0]

    closed = closed_candles(m5)

    if closed.empty:
        return None

    candle = closed.iloc[-1]

    is_buy = (
        "BUY"
        in trade.direction
    )

    hit_sl = False
    hit_tp = False

    if is_buy:

        if candle.low <= trade.sl:
            hit_sl = True

        elif candle.high >= trade.tp:
            hit_tp = True

    else:

        if candle.high >= trade.sl:
            hit_sl = True

        elif candle.low <= trade.tp:
            hit_tp = True

    if not (
        hit_sl
        or hit_tp
    ):

        return None

    # Conservative classification:
    # if both barriers were touched within
    # the same candle, SL wins unless
    # lower timeframe data proves otherwise.

    win = (
        0
        if hit_sl
        else 1
    )

    result = (
        trade.tp
        - trade.entry
        if win
        else trade.sl
        - trade.entry
    )

    if not is_buy:
        result = -result

    con = get_db()

    con.execute(
        """
        INSERT INTO trades(
            date,
            symbol,
            direction,
            entry,
            sl,
            tp,
            result,
            win,
            setup,
            confidence,
            ml_confidence,
            groq_confidence,
            note,
            features
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            datetime.now(
                timezone.utc
            ).date().isoformat(),
            trade.symbol,
            trade.direction,
            trade.entry,
            trade.sl,
            trade.tp,
            result,
            win,
            trade.setup,
            trade.confidence,
            trade.ml_confidence,
            trade.groq_confidence,
            "TP" if win else "SL",
            trade.features,
        ),
    )

    con.execute(
        "DELETE FROM active_trade"
    )

    con.commit()
    con.close()

    # Re-train after every newly settled
    # trade once sufficient data exists.

    train_model_from_history()

    send_alert(
        f"""
Closed {trade.symbol}

Direction:
{trade.direction}

Result:
{"WIN" if win else "LOSS"}

Entry:
{trade.entry}

SL:
{trade.sl}

TP:
{trade.tp}
""",
        "🧠 AI Trade Settled",
    )

    return win


# ============================================================
# UI
# ============================================================

st.title(
    "🧠 XAU/USD Deep AI Engine v3.0"
)

st.caption(
    "ICT/SMC + Multi-Timeframe Quant + "
    "ML Meta-Labeling + Groq Risk Review"
)


packet, datasets = build_market_pipeline()


if packet is None:

    st.warning(
        "⚠️ البيانات غير كافية أو Twelve Data "
        "لم يُرجع جميع الفريمات المطلوبة."
    )

    if st.session_state.get(
        "td_error"
    ):

        st.error(
            st.session_state[
                "td_error"
            ]
        )

else:

    status, candidate = autonomous_decision(
        packet
    )

    ict = packet["ict"]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "XAU/USD",
        f'{ict["price"]:.2f}',
    )

    c2.metric(
        "ICT Bias",
        ict["bias"],
    )

    c3.metric(
        "ICT BUY",
        ict["buy_score"],
    )

    c4.metric(
        "ICT SELL",
        ict["sell_score"],
    )

    c5.metric(
        "ICT Confidence",
        f'{ict["ict_confidence"]:.0f}%',
    )

    st.info(
        f"Decision Engine: **{status}**"
    )

    tabs = st.tabs(
        [
            "🧭 ICT / SMC",
            "🤖 AI Decision",
            "🏦 Active Trade",
            "📚 Learning",
            "🔬 Audit",
        ]
    )

    # ========================================================
    # ICT TAB
    # ========================================================

    with tabs[0]:

        s = ict["structure"]

        a, b, c, d = st.columns(4)

        a.metric(
            "Structure",
            s["bias"],
        )

        b.metric(
            "Liquidity",
            ict[
                "liquidity"
            ].get(
                "sweep",
                "NONE",
            ),
        )

        c.metric(
            "Premium / Discount",
            ict[
                "pd"
            ].get(
                "zone",
                "N/A",
            ),
        )

        d.metric(
            "ATR",
            f'{ict["atr"]:.2f}',
        )

        st.subheader(
            "🏗️ BOS / CHoCH"
        )

        if s["breaks"]:

            st.dataframe(
                pd.DataFrame(
                    s["breaks"]
                ),
                use_container_width=True,
            )

        else:

            st.caption(
                "لا توجد كسور هيكلية حديثة."
            )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "📦 Order Blocks"
            )

            st.write(
                "Bullish:",
                ict["bull_ob"],
            )

            st.write(
                "Bearish:",
                ict["bear_ob"],
            )

            st.subheader(
                "🔄 Breaker Blocks"
            )

            st.write(
                ict["breakers"]
            )

        with col2:

            st.subheader(
                "🌀 Fair Value Gaps"
            )

            st.write(
                "Bullish:",
                ict["bull_fvg"],
            )

            st.write(
                "Bearish:",
                ict["bear_fvg"],
            )

            st.subheader(
                "⚡ Displacement"
            )

            st.write(
                ict[
                    "displacements"
                ]
            )

        st.subheader(
            "🎯 OTE"
        )

        st.write(
            ict["ote"]
        )

        st.subheader(
            "💧 Liquidity"
        )

        st.write(
            ict["liquidity"]
        )

        st.subheader(
            "🕒 Sessions"
        )

        st.json(
            ict["sessions"]
        )

    # ========================================================
    # AI TAB
    # ========================================================

    with tabs[1]:

        if candidate is not None:

            q = candidate["q"]

            x1, x2, x3, x4, x5 = st.columns(5)

            x1.metric(
                "Setup",
                candidate["setup"],
            )

            x2.metric(
                "Score",
                f'{candidate["score"]:.1f}',
            )

            if (
                candidate.get(
                    "ml_confidence"
                )
                is not None
            ):

                x3.metric(
                    "ML Confidence",
                    f'{candidate["ml_confidence"]:.1f}%',
                )

            else:

                x3.metric(
                    "ML Confidence",
                    "Training",
                )

            x4.metric(
                "RSI",
                f'{q["rsi"]:.1f}',
            )

            x5.metric(
                "Volatility",
                f'{q["volatility_pct"]:.1f}%',
            )

            st.subheader(
                "Decision Factors"
            )

            st.json(
                {
                    "direction": candidate[
                        "direction"
                    ],
                    "setup": candidate[
                        "setup"
                    ],
                    "score": candidate[
                        "score"
                    ],
                    "H4": q[
                        "h4_bias"
                    ],
                    "H1": q[
                        "h1_bias"
                    ],
                    "M15": q[
                        "m15_trend"
                    ],
                    "M5": q[
                        "m5_trend"
                    ],
                    "Liquidity Sweep": candidate[
                        "sweep"
                    ],
                    "OTE": candidate[
                        "ote_inside"
                    ],
                    "FVG": candidate[
                        "fvg_present"
                    ],
                    "Order Block": candidate[
                        "ob_present"
                    ],
                    "Displacement": candidate[
                        "displacement_present"
                    ],
                }
            )

    # ========================================================
    # ACTIVE TRADE
    # ========================================================

    with tabs[2]:

        con = get_db()

        active = pd.read_sql(
            """
            SELECT *
            FROM active_trade
            WHERE id=1
            """,
            con,
        )

        con.close()

        if active.empty:

            st.info(
                "لا توجد صفقة نشطة."
            )

        else:

            st.dataframe(
                active,
                use_container_width=True,
            )

    # ========================================================
    # LEARNING
    # ========================================================

    with tabs[3]:

        con = get_db()

        trades = pd.read_sql(
            """
            SELECT *
            FROM trades
            ORDER BY id DESC
            """,
            con,
        )

        con.close()

        if trades.empty:

            st.info(
                "النموذج ينتظر نتائج صفقات "
                "كافية لبدء Meta-Learning."
            )

        else:

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Closed Trades",
                len(trades),
            )

            m2.metric(
                "Win Rate",
                f'{trades.win.mean()*100:.1f}%',
            )

            m3.metric(
                "Wins",
                int(trades.win.sum()),
            )

            st.dataframe(
                trades,
                use_container_width=True,
            )

    # ========================================================
    # AUDIT
    # ========================================================

    with tabs[4]:

        con = get_db()

        candidates = pd.read_sql(
            """
            SELECT *
            FROM candidates
            ORDER BY id DESC
            LIMIT 100
            """,
            con,
        )

        con.close()

        st.dataframe(
            candidates,
            use_container_width=True,
        )


# ============================================================
# MONITOR ACTIVE POSITION
# ============================================================

if packet is not None:

    monitor_active_trade(
        packet["m5"]
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
---
### ⚠️ Research Notice

هذا النظام لا يعتبر ICT أو SMC أو الذكاء الاصطناعي
ضماناً للربحية.

ICT/SMC هنا عبارة عن features وhypotheses قابلة للاختبار
إحصائياً.

قبل التداول الحقيقي يجب اختبار النظام باستخدام:

- Walk-Forward Validation
- Out-of-Sample Testing
- Monte Carlo
- Spread Stress
- Slippage Stress
- Parameter Stability
- Regime Stability
- BUY/SELL Separation
- Session Analysis
- Maximum Drawdown
- Profit Factor
- Sharpe
- Expectancy
- Consecutive Losses

والأهم: لا ينبغي اعتبار نتائج الـBacktest دليلاً على
الربحية المستقبلية.
"""
)

time.sleep(0.01)
