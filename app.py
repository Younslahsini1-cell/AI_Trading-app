"""
XAU/USD Deep AI Engine — v4 (Background Engine Edition)
=========================================================
ICT / Smart Money + Institutional Liquidity + Neural Network + Groq

[قاعدة الترند صديقك]:
- إذا كان الترند هابطاً: نبحث عن فرص SELL فقط.
- إذا كان الترند صاعداً: نبحث عن فرص BUY فقط.
- لا يوجد حظر إطلاقاً، فقط توجيه ذكي نحو اتجاه الترند.

[الاستراتيجيات]:
1) ICT / SMC
2) Institutional Liquidity
   - HTF Trend
   - Previous Day High / Low
   - Asian Liquidity
   - Liquidity Sweep
   - Market Structure Shift (MSS)
   - Displacement
   - Fair Value Gap (FVG)
   - Institutional Risk / Reward

كل استراتيجية تعمل بشكل مستقل.
يمكن أن تكون هناك صفقة ICT وصفقة Institutional في نفس الوقت.
كل صفقة تحفظ اسم الاستراتيجية التي اكتشفتها.
"""

from datetime import datetime, timezone, timedelta
import json
import os
import sqlite3
import threading
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except Exception:
    from requests.packages.urllib3.util.retry import Retry

import streamlit as st

from streamlit_autorefresh import st_autorefresh
from streamlit_local_storage import LocalStorage

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ============================================================
# جلسة HTTP مشتركة
# ============================================================

def _build_http_session():
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        connect=3,
        read=2,
        backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        pool_connections=20,
        pool_maxsize=20,
        max_retries=retry_strategy,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"Connection": "keep-alive"})

    return session

HTTP_SESSION = _build_http_session()

# ============================================================
# إعدادات الصفحة
# ============================================================

st.set_page_config(
    page_title="XAU/USD Deep AI Engine",
    layout="wide",
    page_icon="🧠",
)

# ============================================================
# HTML Renderer
# ============================================================

def render_html(html_content):
    try:
        if hasattr(st, "html"):
            st.html(html_content)
        else:
            st.markdown(
                html_content,
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            html_content,
            unsafe_allow_html=True,
        )

# ============================================================
# CSS
# ============================================================

render_html(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900');

    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif !important;
    }

    .stApp {
        background-color: #07090e;
        color: #f3f4f6;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }

    .ai-level-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid #3b82f6;
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.4);
        margin-bottom: 20px;
    }

    .ai-level-title {
        font-size: 1.2rem;
        color: #93c5fd;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .ai-level-value {
        font-size: 4rem;
        font-weight: 900;
        color: #fbbf24;
        line-height: 1;
    }

    .ai-level-sub {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 10px;
    }

    .trade-status-card {
        background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 20px;
        text-align: center;
    }

    .trade-status-title {
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 700;
    }

    .trade-status-value {
        font-size: 2.3rem;
        font-weight: 900;
        margin-top: 8px;
    }

    .trade-buy {
        color: #22c55e;
    }

    .trade-sell {
        color: #ef4444;
    }

    .trade-neutral {
        color: #94a3b8;
    }

    .engine-pulse {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 10px #22c55e;
        margin-inline-end: 8px;
    }

    .strategy-card {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 12px;
    }

    .strategy-title {
        font-size: 1.05rem;
        font-weight: 900;
        margin-bottom: 8px;
    }
</style>
"""
)

# ============================================================
# SECRETS
# ============================================================

def get_secret_value(key, default=""):
    try:
        value = st.secrets.get(key, default)

        if value is None:
            return default

        return str(value)

    except Exception:
        return default

# ============================================================
# الملفات والثوابت
# ============================================================

DB_FILE = "xau_deep_ai.db"
MODEL_FILE = "xau_deep_mlp_v2.pkl"
SCALER_FILE = "xau_deep_scaler_v2.pkl"
TRAINING_LOCK_FILE = "training.lock"

TRAINING_OUTPUT_SIZE = 5000
LIVE_OUTPUT_SIZE = 220

TRAINING_LOCK_MAX_AGE = 60 * 60
RETRAIN_INTERVAL_SECONDS = 6 * 60 * 60

WORKER_LOOP_SECONDS = 45

HEARTBEAT_INTERVAL_SECONDS = 6 * 60 * 60

FEATURES = [
    "atr",
    "ema_50",
    "ema_200",
    "rsi",
]

ICT_SWING_LOOKBACK = 3
ICT_OB_DISPLACEMENT_MULT = 1.2

# Institutional settings
INSTITUTIONAL_SWEEP_LOOKBACK = 12
INSTITUTIONAL_MSS_LOOKBACK = 5
INSTITUTIONAL_DISPLACEMENT_ATR = 1.10
INSTITUTIONAL_BODY_PERCENT = 55.0
INSTITUTIONAL_SIGNAL_LOOKBACK = 10

MODEL_IO_LOCK = threading.RLock()

# يمنع تشغيل عمليتي حفظ متداخلتين للصفقات
TRADE_DB_LOCK = threading.RLock()

# ============================================================
# Database
# ============================================================

def get_db_connection():
    return sqlite3.connect(
        DB_FILE,
        timeout=20,
        check_same_thread=False,
    )

def init_db():
    conn = get_db_connection()

    try:
        c = conn.cursor()

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                symbol TEXT,
                direction TEXT,
                entry REAL,
                sl REAL,
                tp REAL,
                win INTEGER,
                note TEXT,
                claude_conf REAL,
                claude_note TEXT,
                groq_conf REAL,
                groq_note TEXT,
                ai_conf_before_groq REAL,
                ai_conf_after_groq REAL,
                final_confidence REAL,
                strategy TEXT
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS active_trade (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                direction TEXT,
                entry REAL,
                sl REAL,
                tp REAL,
                time TEXT,
                features TEXT,
                ai_conf REAL,
                groq_conf REAL,
                groq_note TEXT,
                signal_bar_time TEXT,
                final_confidence REAL,
                strategy TEXT
            )
            """
        )

        conn.commit()

        migrations = [
            "ALTER TABLE trades ADD COLUMN claude_conf REAL",
            "ALTER TABLE trades ADD COLUMN claude_note TEXT",
            "ALTER TABLE trades ADD COLUMN groq_conf REAL",
            "ALTER TABLE trades ADD COLUMN groq_note TEXT",
            "ALTER TABLE trades ADD COLUMN ai_conf_before_groq REAL",
            "ALTER TABLE trades ADD COLUMN ai_conf_after_groq REAL",
            "ALTER TABLE trades ADD COLUMN final_confidence REAL",
            "ALTER TABLE trades ADD COLUMN strategy TEXT",

            "ALTER TABLE active_trade ADD COLUMN features TEXT",
            "ALTER TABLE active_trade ADD COLUMN ai_conf REAL",
            "ALTER TABLE active_trade ADD COLUMN groq_conf REAL",
            "ALTER TABLE active_trade ADD COLUMN groq_note TEXT",
            "ALTER TABLE active_trade ADD COLUMN signal_bar_time TEXT",
            "ALTER TABLE active_trade ADD COLUMN final_confidence REAL",
            "ALTER TABLE active_trade ADD COLUMN strategy TEXT",
        ]

        for stmt in migrations:
            try:
                c.execute(stmt)
                conn.commit()
            except sqlite3.OperationalError:
                pass

        # ----------------------------------------------------
        # ترحيل الصفقات القديمة إلى ICT / SMC
        # ----------------------------------------------------

        try:
            c.execute(
                """
                UPDATE trades
                SET strategy = 'ICT / SMC'
                WHERE strategy IS NULL
                   OR TRIM(strategy) = ''
                """
            )

            c.execute(
                """
                UPDATE active_trade
                SET strategy = 'ICT / SMC'
                WHERE strategy IS NULL
                   OR TRIM(strategy) = ''
                """
            )

            conn.commit()

        except Exception:
            pass

    finally:
        conn.close()

init_db()

# ============================================================
# Settings
# ============================================================

def save_setting(key, val):
    for attempt in range(3):

        conn = get_db_connection()

        try:
            c = conn.cursor()

            c.execute(
                """
                INSERT OR REPLACE INTO settings (key, value)
                VALUES (?, ?)
                """,
                (
                    key,
                    str(val),
                ),
            )

            conn.commit()

            return

        except sqlite3.OperationalError:

            if attempt == 2:
                raise

            time.sleep(
                0.15 * (attempt + 1)
            )

        finally:
            conn.close()

def load_setting(key, default=""):
    conn = get_db_connection()

    try:
        c = conn.cursor()

        c.execute(
            """
            SELECT value
            FROM settings
            WHERE key = ?
            """,
            (key,),
        )

        row = c.fetchone()

        return (
            row[0]
            if row
            else default
        )

    finally:
        conn.close()

def get_successful_trades_count():
    conn = get_db_connection()

    try:
        c = conn.cursor()

        c.execute(
            """
            SELECT COUNT(*)
            FROM trades
            WHERE win = 1
            """
        )

        return int(
            c.fetchone()[0]
        )

    finally:
        conn.close()

def get_total_trades_count():
    conn = get_db_connection()

    try:
        c = conn.cursor()

        c.execute(
            """
            SELECT COUNT(*)
            FROM trades
            """
        )

        return int(
            c.fetchone()[0]
        )

    finally:
        conn.close()

def get_active_trades_df():
    conn = get_db_connection()

    try:
        df = pd.read_sql(
            """
            SELECT *
            FROM active_trade
            ORDER BY id ASC
            """,
            conn,
        )

        return df

    finally:
        conn.close()

def get_active_trade_for_strategy(strategy):
    conn = get_db_connection()

    try:
        df = pd.read_sql(
            """
            SELECT *
            FROM active_trade
            WHERE strategy = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            conn,
            params=(strategy,),
        )

        return df

    finally:
        conn.close()

def has_active_trade_for_strategy(strategy):
    df = get_active_trade_for_strategy(
        strategy
    )

    return not df.empty

def get_next_active_trade_id(conn):
    c = conn.cursor()

    c.execute(
        """
        SELECT COALESCE(MAX(id), 0) + 1
        FROM active_trade
        """
    )

    row = c.fetchone()

    return int(
        row[0] if row else 1
    )

# ============================================================
# Sidebar
# ============================================================

st.sidebar.header(
    "⚙️ إعدادات الذكاء الاصطناعي"
)

localS = LocalStorage()

twelve_secret = get_secret_value(
    "TWELVE_DATA_API_KEY",
    "",
)

if "twelve_key" not in st.session_state:

    try:
        stored_twelve_key = (
            localS.getItem(
                "twelve_key_ls"
            )
            or load_setting(
                "twelve_key",
                "",
            )
        )

    except Exception:
        stored_twelve_key = load_setting(
            "twelve_key",
            "",
        )

    st.session_state[
        "twelve_key"
    ] = (
        twelve_secret
        or stored_twelve_key
    )

twelve_key = st.sidebar.text_input(
    "مفتاح Twelve Data API",
    type="password",
    key="twelve_key",
)

if twelve_key:

    save_setting(
        "twelve_key",
        twelve_key,
    )

    try:
        localS.setItem(
            "twelve_key_ls",
            twelve_key,
        )
    except Exception:
        pass

ntfy_channel = st.sidebar.text_input(
    "قناة Ntfy للتنبيهات",
    value=load_setting(
        "ntfy",
        "xau_deep_channel",
    ),
)

save_setting(
    "ntfy",
    ntfy_channel,
)

st.sidebar.markdown("---")
st.sidebar.header(
    "🧠 الرأي الثاني (Groq)"
)

use_groq = st.sidebar.checkbox(
    "تفعيل مراجعة Groq",
    value=(
        load_setting(
            "use_groq",
            "1",
        )
        == "1"
    ),
)

save_setting(
    "use_groq",
    "1" if use_groq else "0",
)

groq_secret = get_secret_value(
    "GROQ_API_KEY",
    "",
)

if "groq_key" not in st.session_state:

    try:
        stored_groq_key = (
            localS.getItem(
                "groq_key_ls"
            )
            or load_setting(
                "groq_key",
                "",
            )
        )

    except Exception:
        stored_groq_key = load_setting(
            "groq_key",
            "",
        )

    st.session_state[
        "groq_key"
    ] = (
        groq_secret
        or stored_groq_key
    )

groq_key = st.sidebar.text_input(
    "مفتاح Groq API",
    type="password",
    key="groq_key",
)

if groq_key:

    save_setting(
        "groq_key",
        groq_key,
    )

    try:
        localS.setItem(
            "groq_key_ls",
            groq_key,
        )
    except Exception:
        pass

groq_model = st.sidebar.text_input(
    "اسم نموذج Groq",
    value=load_setting(
        "groq_model",
        "openai/gpt-oss-120b",
    ),
)

save_setting(
    "groq_model",
    groq_model,
)

min_groq_conf = st.sidebar.slider(
    "أدنى ثقة مطلوبة من Groq",
    30,
    95,
    50,
    1,
)

save_setting(
    "min_groq_conf",
    min_groq_conf,
)

st.sidebar.markdown("---")
st.sidebar.header(
    "🎯 إدارة المخاطر"
)

atr_mult = st.sidebar.slider(
    "معامل الوقف ATR",
    1.0,
    3.0,
    1.5,
    0.1,
)

save_setting(
    "atr_mult",
    atr_mult,
)

risk_reward = st.sidebar.slider(
    "نسبة العائد R:R",
    1.5,
    4.0,
    2.0,
    0.5,
)

save_setting(
    "risk_reward",
    risk_reward,
)

min_conf = st.sidebar.slider(
    "أدنى ثقة نهائية مطلوبة لفتح صفقة",
    50,
    95,
    65,
    1,
)

save_setting(
    "min_conf",
    min_conf,
)

if st.sidebar.button(
    "🔄 إعادة تدريب النموذج من الصفر"
):

    for file_path in (
        MODEL_FILE,
        SCALER_FILE,
        TRAINING_LOCK_FILE,
    ):

        if os.path.exists(
            file_path
        ):

            try:
                os.remove(
                    file_path
                )
            except OSError:
                pass

    try:
        st.cache_data.clear()
    except Exception:
        pass

    st.rerun()

# ============================================================
# Ntfy & Data Functions
# ============================================================

def send_alert(
    msg,
    title="🧠 Deep AI Alert",
):
    channel_setting = load_setting(
        "ntfy",
        "",
    )

    if not channel_setting:
        return

    channel = (
        channel_setting
        .strip()
        .split("/")[-1]
    )

    if not channel:
        return

    try:
        HTTP_SESSION.post(
            f"https://ntfy.sh/{channel}",
            data=msg.encode(
                "utf-8"
            ),
            headers={
                "Title": title,
                "Priority": "high",
            },
            timeout=5,
        )
    except Exception:
        pass

def send_trade_confirmation_alert(
    direction,
    entry,
    sl,
    tp,
    final_confidence,
    risk_reward_ratio,
    ai_conf,
    groq_conf=None,
    strategy="ICT / SMC",
):
    groq_line = (
        f"\nGroq Confidence: "
        f"{groq_conf:.1f}%"
        if groq_conf is not None
        else ""
    )

    msg = (
        f"✅ تم تأكيد الصفقة\n"
        f"الاستراتيجية: {strategy}\n"
        f"الاتجاه: {direction}\n"
        f"الدخول: ${entry}\n"
        f"وقف الخسارة (SL): ${sl}\n"
        f"جني الأرباح (TP): ${tp}\n"
        f"نسبة R:R = 1:{risk_reward_ratio}\n"
        f"ثقة AI الخام: {ai_conf:.1f}%"
        f"{groq_line}\n"
        f"الثقة النهائية: "
        f"{final_confidence:.1f}%"
    )

    send_alert(
        msg,
        title=(
            f"✅ {strategy} — XAU/USD"
        ),
    )

def fetch_twelve_series(
    api_key,
    symbol="XAU/USD",
    interval="1h",
    outputsize=150,
):
    if not api_key:
        return pd.DataFrame()

    try:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": min(
                int(outputsize),
                5000,
            ),
            "timezone": "UTC",
            "apikey": api_key,
        }

        response = HTTP_SESSION.get(
            "https://api.twelvedata.com/time_series",
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        if "values" not in result:

            APP_STATE_set(
                "last_twelve_error",
                result.get(
                    "message",
                    "استجابة غير متوقعة",
                ),
            )

            return pd.DataFrame()

        values = result["values"]

        if not values:
            return pd.DataFrame()

        df = pd.DataFrame(
            values
        )

        required = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
        ]

        if not all(
            col in df.columns
            for col in required
        ):
            return pd.DataFrame()

        df = df[
            required
        ].copy()

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
            utc=True,
        )

        for col in (
            "open",
            "high",
            "low",
            "close",
        ):

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        df.dropna(
            subset=required,
            inplace=True,
        )

        df.sort_values(
            "datetime",
            inplace=True,
        )

        df.drop_duplicates(
            subset=["datetime"],
            keep="last",
            inplace=True,
        )

        df.reset_index(
            drop=True,
            inplace=True,
        )

        APP_STATE_set(
            "last_twelve_error",
            None,
        )

        return df

    except Exception as exc:

        APP_STATE_set(
            "last_twelve_error",
            f"تعذّر الاتصال: {exc}",
        )

        return pd.DataFrame()

def keep_closed_candles(
    df,
    interval_hours=1,
):
    if (
        df is None
        or df.empty
        or "datetime"
        not in df.columns
    ):
        return pd.DataFrame()

    df = df.copy()

    now_utc = datetime.now(
        timezone.utc
    )

    candle_delta = timedelta(
        hours=interval_hours
    )

    mask = (
        df["datetime"]
        + candle_delta
        <= pd.Timestamp(
            now_utc
        )
    )

    closed = df.loc[
        mask
    ].copy()

    return closed.reset_index(
        drop=True
    )

def fetch_training_data_twelve(
    api_key,
):
    return fetch_twelve_series(
        api_key,
        symbol="XAU/USD",
        interval="1h",
        outputsize=TRAINING_OUTPUT_SIZE,
    )

def fetch_free_training_series(
    symbol="XAUUSD=X",
    interval="60m",
    range_="730d",
):
    try:

        url = (
            "https://query1.finance.yahoo.com/"
            f"v8/finance/chart/{symbol}"
        )

        response = HTTP_SESSION.get(
            url,
            params={
                "interval": interval,
                "range": range_,
            },
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=15,
        )

        response.raise_for_status()

        payload = response.json()

        result_list = (
            (
                payload.get("chart")
                or {}
            ).get("result")
            or []
        )

        if not result_list:
            return pd.DataFrame()

        result = result_list[0]

        timestamps = (
            result.get("timestamp")
            or []
        )

        quote = (
            (
                result.get(
                    "indicators"
                )
                or {}
            ).get("quote")
            or [{}]
        )[0]

        opens = (
            quote.get("open")
            or []
        )

        highs = (
            quote.get("high")
            or []
        )

        lows = (
            quote.get("low")
            or []
        )

        closes = (
            quote.get("close")
            or []
        )

        if (
            not timestamps
            or not closes
        ):
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    timestamps,
                    unit="s",
                    utc=True,
                ),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
            }
        )

        for col in (
            "open",
            "high",
            "low",
            "close",
        ):

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        df.dropna(
            subset=[
                "datetime",
                "open",
                "high",
                "low",
                "close",
            ],
            inplace=True,
        )

        df.sort_values(
            "datetime",
            inplace=True,
        )

        df.drop_duplicates(
            subset=["datetime"],
            keep="last",
            inplace=True,
        )

        df.reset_index(
            drop=True,
            inplace=True,
        )

        return df

    except Exception:
        return pd.DataFrame()

def fetch_training_data_free():
    return fetch_free_training_series(
        symbol="XAUUSD=X",
        interval="60m",
        range_="730d",
    )

def fetch_live_series(
    symbol_twelve,
    symbol_yahoo,
    interval_twelve,
    interval_yahoo,
    range_yahoo,
    outputsize_twelve,
    twelve_api_key,
):
    df = fetch_free_training_series(
        symbol=symbol_yahoo,
        interval=interval_yahoo,
        range_=range_yahoo,
    )

    if (
        df is not None
        and not df.empty
    ):

        APP_STATE_set(
            "last_data_source",
            f"Yahoo Finance ({interval_yahoo})",
        )

        return df

    if not twelve_api_key:
        return pd.DataFrame()

    df = fetch_twelve_series(
        twelve_api_key,
        symbol=symbol_twelve,
        interval=interval_twelve,
        outputsize=outputsize_twelve,
    )

    if (
        df is not None
        and not df.empty
    ):

        APP_STATE_set(
            "last_data_source",
            f"Twelve Data ({interval_twelve})",
        )

    return df

# ============================================================
# Indicators & Model
# ============================================================

def apply_deep_indicators(df):
    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    if len(df) < 60:
        return pd.DataFrame()

    df = df.copy()

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"]
                - df["close"].shift()
            ).abs(),
            (
                df["low"]
                - df["close"].shift()
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = (
        tr.rolling(14).mean()
    )

    df["ema_50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    df["ema_200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
    )

    delta = df["close"].diff()

    gain = (
        delta
        .where(delta > 0, 0.0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta
        .where(delta < 0, 0.0)
        .rolling(14)
        .mean()
    )

    rs = (
        gain
        / (loss + 1e-6)
    )

    df["rsi"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    df.dropna(
        subset=FEATURES,
        inplace=True,
    )

    return df.reset_index(
        drop=True
    )

def model_is_ready(
    model_obj,
    scaler_obj,
):
    if (
        model_obj is None
        or scaler_obj is None
    ):
        return False

    if not hasattr(
        scaler_obj,
        "mean_",
    ):
        return False

    if not hasattr(
        scaler_obj,
        "scale_",
    ):
        return False

    if not hasattr(
        model_obj,
        "classes_",
    ):
        return False

    try:

        if len(
            model_obj.classes_
        ) < 2:
            return False

        if (
            getattr(
                model_obj,
                "n_features_in_",
                len(FEATURES),
            )
            != len(FEATURES)
        ):
            return False

        if len(
            scaler_obj.mean_
        ) != len(FEATURES):
            return False

        if len(
            scaler_obj.scale_
        ) != len(FEATURES):
            return False

    except Exception:
        return False

    return True

def _atomic_joblib_dump(
    obj,
    target_file,
):
    temp_file = (
        target_file
        + f".tmp.{os.getpid()}."
        f"{threading.get_ident()}"
    )

    try:

        joblib.dump(
            obj,
            temp_file,
        )

        os.replace(
            temp_file,
            target_file,
        )

    finally:

        if os.path.exists(
            temp_file
        ):

            try:
                os.remove(
                    temp_file
                )
            except OSError:
                pass

# ============================================================
# Background Training
# ============================================================

def _background_train_and_save(
    api_key,
):
    try:

        df_train = (
            fetch_training_data_free()
        )

        if (
            df_train.empty
            and api_key
        ):
            df_train = (
                fetch_training_data_twelve(
                    api_key
                )
            )

        df_train = (
            keep_closed_candles(
                df_train,
                interval_hours=1,
            )
        )

        df_train = (
            apply_deep_indicators(
                df_train
            )
        )

        if (
            df_train.empty
            or len(df_train) < 60
        ):
            return

        future_close = (
            df_train["close"].shift(-1)
        )

        valid_mask = (
            future_close.notna()
        )

        X_df = df_train.loc[
            valid_mask,
            FEATURES,
        ].copy()

        close_now = (
            df_train.loc[
                valid_mask,
                "close",
            ].astype(float)
        )

        close_future = (
            future_close.loc[
                valid_mask
            ].astype(float)
        )

        valid_features = np.isfinite(
            X_df.astype(float).values
        ).all(axis=1)

        X_df = X_df.loc[
            valid_features
        ]

        close_now = close_now.loc[
            X_df.index
        ]

        close_future = close_future.loc[
            X_df.index
        ]

        if len(X_df) < 30:
            return

        X = (
            X_df.astype(float)
            .values
        )

        y = (
            close_future.values
            > close_now.values
        ).astype(int)

        if (
            len(X) < 30
            or len(y) < 30
        ):
            return

        if len(X) != len(y):
            return

        if len(
            np.unique(y)
        ) < 2:
            return

        new_scaler = (
            StandardScaler()
        )

        X_scaled = (
            new_scaler.fit_transform(
                X
            )
        )

        new_model = MLPClassifier(
            hidden_layer_sizes=(
                128,
                64,
                32,
            ),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            random_state=42,
            shuffle=False,
        )

        new_model.fit(
            X_scaled,
            y,
        )

        if not model_is_ready(
            new_model,
            new_scaler,
        ):
            return

        with MODEL_IO_LOCK:

            _atomic_joblib_dump(
                new_model,
                MODEL_FILE,
            )

            _atomic_joblib_dump(
                new_scaler,
                SCALER_FILE,
            )

        APP_STATE_set(
            "last_train_time",
            datetime.now(
                timezone.utc
            ).isoformat(),
        )

    except Exception:
        pass

    finally:

        if os.path.exists(
            TRAINING_LOCK_FILE
        ):

            try:
                os.remove(
                    TRAINING_LOCK_FILE
                )
            except OSError:
                pass

def clean_stale_training_lock():
    if not os.path.exists(
        TRAINING_LOCK_FILE
    ):
        return

    try:

        age = (
            time.time()
            - os.path.getmtime(
                TRAINING_LOCK_FILE
            )
        )

        if (
            age
            > TRAINING_LOCK_MAX_AGE
        ):
            os.remove(
                TRAINING_LOCK_FILE
            )

    except Exception:
        pass

def maybe_spawn_training(
    api_key,
    force=False,
):
    if os.path.exists(
        TRAINING_LOCK_FILE
    ):
        return

    needs_training = (
        force
        or not (
            os.path.exists(
                MODEL_FILE
            )
            and os.path.exists(
                SCALER_FILE
            )
        )
    )

    if not needs_training:

        try:

            age = (
                time.time()
                - os.path.getmtime(
                    MODEL_FILE
                )
            )

            if (
                age
                > RETRAIN_INTERVAL_SECONDS
            ):
                needs_training = True

        except OSError:
            needs_training = True

    if not needs_training:
        return

    try:

        with open(
            TRAINING_LOCK_FILE,
            "x",
            encoding="utf-8",
        ) as file:

            file.write(
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

        thread = threading.Thread(
            target=_background_train_and_save,
            args=(api_key,),
            daemon=True,
            name="bg_trainer",
        )

        thread.start()

    except FileExistsError:
        pass

    except Exception:
        pass

def load_current_model():
    if not (
        os.path.exists(
            MODEL_FILE
        )
        and os.path.exists(
            SCALER_FILE
        )
    ):
        return None, None

    with MODEL_IO_LOCK:

        try:

            loaded_model = (
                joblib.load(
                    MODEL_FILE
                )
            )

            loaded_scaler = (
                joblib.load(
                    SCALER_FILE
                )
            )

            if model_is_ready(
                loaded_model,
                loaded_scaler,
            ):

                return (
                    loaded_model,
                    loaded_scaler,
                )

        except Exception:
            pass

    return None, None

clean_stale_training_lock()

# ============================================================
# Experience Layer & Groq
# ============================================================

def get_experience_adjustment(
    direction,
    ai_conf,
):
    total = (
        get_total_trades_count()
    )

    if total < 20:

        return {
            "available": False,
            "confidence": ai_conf,
            "win_rate": None,
            "sample": total,
        }

    normalized_direction = (
        "BUY"
        if "BUY"
        in str(direction)
        else "SELL"
    )

    conn = get_db_connection()

    try:

        df = pd.read_sql(
            """
            SELECT direction, win
            FROM trades
            WHERE direction LIKE ?
            """,
            conn,
            params=(
                f"%{normalized_direction}%",
            ),
        )

    finally:
        conn.close()

    if len(df) < 10:

        return {
            "available": False,
            "confidence": ai_conf,
            "win_rate": None,
            "sample": len(df),
        }

    wins = float(
        df["win"].sum()
    )

    n = len(df)

    smoothed_rate = (
        (wins + 5.0)
        / (n + 10.0)
        * 100
    )

    adjusted = (
        ai_conf * 0.70
        + smoothed_rate * 0.30
    )

    return {
        "available": True,
        "confidence": round(
            float(adjusted),
            1,
        ),
        "win_rate": round(
            smoothed_rate,
            1,
        ),
        "sample": n,
    }

def _parse_groq_bool(value):
    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        (int, float),
    ):
        return bool(value)

    if isinstance(
        value,
        str,
    ):

        normalized = (
            value
            .strip()
            .lower()
        )

        if normalized in {
            "true",
            "1",
            "yes",
            "y",
            "agree",
            "approved",
            "موافق",
            "نعم",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
            "n",
            "disagree",
            "rejected",
            "غير موافق",
            "لا",
        }:
            return False

    return False

def get_groq_review(
    direction,
    last_row,
    ai_conf,
    api_key,
    model_name,
    extra_context="",
):
    if not api_key:

        APP_STATE_set(
            "last_groq_error",
            "لم يتم إدخال مفتاح Groq API.",
        )

        return None

    try:

        prompt = (
            "أنت محلل فني مساعد "
            "لصفقة محتملة على XAU/USD.\n"
            f"الاتجاه المقترح: {direction}.\n"
            f"ثقة نموذج AI الخام: "
            f"{ai_conf:.1f}%.\n"
            f"ATR={last_row['atr']:.2f}.\n"
            f"EMA50={last_row['ema_50']:.2f}.\n"
            f"EMA200={last_row['ema_200']:.2f}.\n"
            f"RSI={last_row['rsi']:.1f}.\n"
            f"السعر={last_row['close']:.2f}.\n"
            f"{extra_context}\n"
            "راجع الاتجاه بشكل مستقل بناءً "
            "على البيانات المعطاة فقط. "
            "لا تفترض أن نموذج AI صحيح، "
            "لكن أيضًا لا تكن متشدداً "
            "بلا داعٍ.\n"
            'يجب أن يكون الرد JSON فقط بهذا الشكل: '
            '{"agree": true, "confidence": 0, '
            '"reason": "..."}'
        )

        response = HTTP_SESSION.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": (
                    f"Bearer {api_key}"
                ),
                "Content-Type": (
                    "application/json"
                ),
            },
            json={
                "model": model_name,
                "temperature": 0,
                "max_completion_tokens": 2000,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "أنت محلل فني متوازن. "
                            "أعد JSON صالح فقط."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "response_format": {
                    "type": "json_object"
                },
            },
            timeout=20,
        )

        if not response.ok:

            APP_STATE_set(
                "last_groq_error",
                (
                    f"HTTP "
                    f"{response.status_code} "
                    f"من Groq: "
                    f"{response.text[:400]}"
                ),
            )

            return None

        data = response.json()

        choices = data.get(
            "choices",
            [],
        )

        if not choices:

            APP_STATE_set(
                "last_groq_error",
                (
                    "رد Groq بلا choices: "
                    + json.dumps(
                        data,
                        ensure_ascii=False,
                    )[:400]
                ),
            )

            return None

        message = choices[0].get(
            "message",
            {},
        )

        text = message.get(
            "content",
            "",
        )

        if not text:

            APP_STATE_set(
                "last_groq_error",
                "رد Groq وصل لكن بلا محتوى نصي.",
            )

            return None

        cleaned = (
            str(text)
            .replace(
                "```json",
                "",
            )
            .replace(
                "```",
                "",
            )
            .strip()
        )

        try:

            parsed = json.loads(
                cleaned
            )

        except Exception as parse_exc:

            APP_STATE_set(
                "last_groq_error",
                (
                    "تعذّر تحليل رد Groq "
                    f"كـ JSON: {parse_exc}"
                    f" — النص: {cleaned[:400]}"
                ),
            )

            return None

        agree = _parse_groq_bool(
            parsed.get(
                "agree",
                False,
            )
        )

        try:

            confidence = float(
                parsed.get(
                    "confidence",
                    0,
                )
            )

        except Exception:
            confidence = 0.0

        confidence = max(
            0,
            min(
                100,
                confidence,
            ),
        )

        reason = str(
            parsed.get(
                "reason",
                "",
            )
        )

        APP_STATE_set(
            "last_groq_error",
            None,
        )

        return {
            "agree": agree,
            "confidence": confidence,
            "reason": reason,
        }

    except Exception as exc:

        APP_STATE_set(
            "last_groq_error",
            (
                "استثناء أثناء الاتصال "
                f"بـ Groq: {exc}"
            ),
        )

        return None

# ============================================================
# ICT Engine
# ============================================================

def find_swing_points(
    df,
    lookback=3,
):
    if (
        df is None
        or df.empty
    ):
        return [], []

    highs = (
        df["high"]
        .astype(float)
        .values
    )

    lows = (
        df["low"]
        .astype(float)
        .values
    )

    n = len(df)

    swing_highs = []
    swing_lows = []

    for i in range(
        lookback,
        n - lookback,
    ):

        high_window = highs[
            i - lookback:
            i + lookback + 1
        ]

        low_window = lows[
            i - lookback:
            i + lookback + 1
        ]

        if highs[i] == np.max(
            high_window
        ):

            swing_highs.append(
                (
                    i,
                    float(
                        highs[i]
                    ),
                )
            )

        if lows[i] == np.min(
            low_window
        ):

            swing_lows.append(
                (
                    i,
                    float(
                        lows[i]
                    ),
                )
            )

    return (
        swing_highs,
        swing_lows,
    )

def analyze_market_structure(
    swing_highs,
    swing_lows,
):
    events = (
        [
            (
                i,
                "high",
                p,
            )
            for i, p in swing_highs
        ]
        + [
            (
                i,
                "low",
                p,
            )
            for i, p in swing_lows
        ]
    )

    events.sort(
        key=lambda x: x[0]
    )

    structure_breaks = []

    trend = None
    last_high = None
    last_low = None

    for (
        i,
        kind,
        price,
    ) in events:

        if kind == "high":

            if (
                last_high is not None
                and price > last_high
            ):

                label = (
                    "BOS"
                    if trend == "bullish"
                    else "CHoCH"
                )

                structure_breaks.append(
                    {
                        "type": label,
                        "direction": "BULLISH",
                        "price": round(
                            price,
                            2,
                        ),
                    }
                )

                trend = "bullish"

            last_high = price

        else:

            if (
                last_low is not None
                and price < last_low
            ):

                label = (
                    "BOS"
                    if trend == "bearish"
                    else "CHoCH"
                )

                structure_breaks.append(
                    {
                        "type": label,
                        "direction": "BEARISH",
                        "price": round(
                            price,
                            2,
                        ),
                    }
                )

                trend = "bearish"

            last_low = price

    current_bias = (
        structure_breaks[-1]["direction"]
        if structure_breaks
        else "NEUTRAL"
    )

    return (
        current_bias,
        structure_breaks[-8:],
    )

def detect_order_blocks(
    df,
    lookback=40,
    displacement_atr_mult=1.2,
):
    if (
        df is None
        or df.empty
        or "atr" not in df.columns
        or len(df) < 5
    ):
        return None, None

    recent = (
        df.iloc[-lookback:]
        .reset_index(drop=True)
    )

    bullish_ob = None
    bearish_ob = None

    for i in range(
        1,
        len(recent),
    ):

        atr_value = float(
            recent["atr"].iloc[i]
        )

        if (
            not np.isfinite(
                atr_value
            )
            or atr_value <= 0
        ):
            continue

        body = (
            recent["close"].iloc[i]
            - recent["open"].iloc[i]
        )

        is_displacement = (
            abs(body)
            > displacement_atr_mult
            * atr_value
        )

        previous = (
            recent.iloc[i - 1]
        )

        if (
            is_displacement
            and body > 0
            and previous["close"]
            < previous["open"]
        ):

            bullish_ob = {
                "top": round(
                    float(
                        previous["open"]
                    ),
                    2,
                ),
                "bottom": round(
                    float(
                        previous["low"]
                    ),
                    2,
                ),
                "bias": "BULLISH",
                "strength": round(
                    min(
                        (
                            abs(body)
                            / atr_value
                        )
                        * 100,
                        999,
                    ),
                    1,
                ),
            }

        elif (
            is_displacement
            and body < 0
            and previous["close"]
            > previous["open"]
        ):

            bearish_ob = {
                "top": round(
                    float(
                        previous["high"]
                    ),
                    2,
                ),
                "bottom": round(
                    float(
                        previous["open"]
                    ),
                    2,
                ),
                "bias": "BEARISH",
                "strength": round(
                    min(
                        (
                            abs(body)
                            / atr_value
                        )
                        * 100,
                        999,
                    ),
                    1,
                ),
            }

    return (
        bullish_ob,
        bearish_ob,
    )

def detect_fvg(
    df,
    lookback=60,
):
    if (
        df is None
        or df.empty
        or len(df) < 3
    ):
        return None, None

    recent = (
        df.iloc[-lookback:]
        .reset_index(drop=True)
    )

    bullish_fvg = None
    bearish_fvg = None

    for i in range(
        2,
        len(recent),
    ):

        c1 = recent.iloc[
            i - 2
        ]

        c3 = recent.iloc[i]

        if (
            c1["high"]
            < c3["low"]
        ):

            bullish_fvg = {
                "top": round(
                    float(
                        c3["low"]
                    ),
                    2,
                ),
                "bottom": round(
                    float(
                        c1["high"]
                    ),
                    2,
                ),
                "bias": "BULLISH",
            }

        if (
            c1["low"]
            > c3["high"]
        ):

            bearish_fvg = {
                "top": round(
                    float(
                        c1["low"]
                    ),
                    2,
                ),
                "bottom": round(
                    float(
                        c3["high"]
                    ),
                    2,
                ),
                "bias": "BEARISH",
            }

    return (
        bullish_fvg,
        bearish_fvg,
    )

def detect_liquidity_and_manipulation(
    df,
    swing_highs,
    swing_lows,
):
    if (
        not swing_highs
        or not swing_lows
        or df is None
        or df.empty
    ):
        return (
            None,
            None,
            False,
            "",
        )

    recent_highs = (
        swing_highs[-5:]
    )

    recent_lows = (
        swing_lows[-5:]
    )

    bsl_price = max(
        p
        for _, p in recent_highs
    )

    ssl_price = min(
        p
        for _, p in recent_lows
    )

    last = df.iloc[-1]

    manipulation = False
    note = ""

    if (
        last["high"]
        > bsl_price
        and last["close"]
        < bsl_price
    ):

        manipulation = True

        note = (
            "BSL swept — احتمال انعكاس."
        )

    elif (
        last["low"]
        < ssl_price
        and last["close"]
        > ssl_price
    ):

        manipulation = True

        note = (
            "SSL swept — احتمال انعكاس."
        )

    return (
        round(
            bsl_price,
            2,
        ),
        round(
            ssl_price,
            2,
        ),
        manipulation,
        note,
    )

def session_analyzer(
    df,
    session_len=24,
):
    if (
        df is None
        or df.empty
        or len(df) < 5
    ):
        return None

    window = df.iloc[
        -min(
            session_len,
            len(df),
        ):
    ]

    session_high = float(
        window["high"].max()
    )

    session_low = float(
        window["low"].min()
    )

    session_range = (
        session_high
        - session_low
    )

    net_change = (
        float(
            window["close"].iloc[-1]
        )
        - float(
            window["open"].iloc[0]
        )
    )

    if net_change < 0:
        bias = "BEARISH"
    elif net_change > 0:
        bias = "BULLISH"
    else:
        bias = "NEUTRAL"

    main_score = round(
        min(
            abs(net_change)
            / (
                session_range
                + 1e-6
            )
            * 100,
            100,
        ),
        1,
    )

    return {
        "bias": bias,
        "main_score": main_score,
        "range": round(
            session_range,
            2,
        ),
        "high": round(
            session_high,
            2,
        ),
        "low": round(
            session_low,
            2,
        ),
    }

def compute_asian_session_levels(
    df,
):
    if (
        df is None
        or df.empty
        or len(df) < 24
    ):
        return None

    window = df.iloc[-24:]

    return {
        "asian_high": round(
            float(
                window["high"].max()
            ),
            2,
        ),
        "asian_low": round(
            float(
                window["low"].min()
            ),
            2,
        ),
    }

def compute_fibonacci_extension(
    swing_low,
    swing_high,
    direction,
):
    diff = (
        swing_high
        - swing_low
    )

    if diff <= 0:
        return {}

    ratios = [
        (
            1.0,
            "100% EXTENSION",
        ),
        (
            1.27,
            "127% EXTENSION",
        ),
        (
            1.618,
            "161.8% EXTENSION",
        ),
        (
            2.0,
            "200% EXTENSION",
        ),
        (
            2.618,
            "261.8% EXTENSION",
        ),
    ]

    levels = {}

    for ratio, label in ratios:

        if direction == "BULLISH":

            levels[label] = round(
                swing_high
                + diff
                * (ratio - 1),
                2,
            )

        else:

            levels[label] = round(
                swing_low
                - diff
                * (ratio - 1),
                2,
            )

    levels[
        "EQUILIBRIUM TARGET"
    ] = round(
        (
            swing_high
            + swing_low
        )
        / 2,
        2,
    )

    return levels

def compute_ote_zone(
    swing_low,
    swing_high,
    direction,
    current_price,
):
    diff = (
        swing_high
        - swing_low
    )

    if diff <= 0:
        return None

    if direction == "BULLISH":

        top = (
            swing_high
            - diff * 0.618
        )

        bottom = (
            swing_high
            - diff * 0.79
        )

    else:

        top = (
            swing_low
            + diff * 0.79
        )

        bottom = (
            swing_low
            + diff * 0.618
        )

    lo = min(
        top,
        bottom,
    )

    hi = max(
        top,
        bottom,
    )

    return {
        "top": round(
            hi,
            2,
        ),
        "bottom": round(
            lo,
            2,
        ),
        "inside": (
            lo
            <= current_price
            <= hi
        ),
        "direction": direction,
    }

def compute_volatility_risk(
    df,
):
    if (
        df is None
        or df.empty
        or "atr" not in df.columns
    ):
        return (
            "N/A",
            0,
            0.0,
        )

    atr_series = (
        df["atr"]
        .dropna()
    )

    if atr_series.empty:
        return (
            "N/A",
            0,
            0.0,
        )

    current_atr = float(
        atr_series.iloc[-1]
    )

    percentile = (
        atr_series
        < current_atr
    ).mean() * 100

    if percentile >= 90:
        label, score = (
            "CRISIS",
            90,
        )

    elif percentile >= 65:
        label, score = (
            "HIGH",
            70,
        )

    elif percentile >= 35:
        label, score = (
            "MEDIUM",
            50,
        )

    else:
        label, score = (
            "LOW",
            25,
        )

    return (
        label,
        score,
        round(
            current_atr,
            2,
        ),
    )

def detect_recent_displacement(
    df,
    lookback=10,
    atr_mult=1.3,
):
    if (
        df is None
        or df.empty
    ):
        return []

    recent = df.iloc[
        -lookback:
    ]

    results = []

    for _, row in recent.iterrows():

        atr_value = row.get(
            "atr",
            np.nan,
        )

        if (
            pd.isna(atr_value)
            or atr_value <= 0
        ):
            continue

        body = (
            row["close"]
            - row["open"]
        )

        candle_range = (
            row["high"]
            - row["low"]
        )

        body_pct = round(
            abs(body)
            / (
                candle_range
                + 1e-6
            )
            * 100,
            1,
        )

        if (
            abs(body)
            > atr_mult
            * atr_value
        ):

            results.append(
                {
                    "bias": (
                        "BULLISH"
                        if body > 0
                        else "BEARISH"
                    ),
                    "body_pct": body_pct,
                }
            )

    return results[-3:]

def run_ict_engine(
    df_processed,
    swing_lookback=3,
    ob_mult=1.2,
):
    if (
        df_processed is None
        or df_processed.empty
        or len(df_processed)
        < max(
            30,
            swing_lookback * 6,
        )
    ):
        return None

    (
        swing_highs,
        swing_lows,
    ) = find_swing_points(
        df_processed,
        lookback=swing_lookback,
    )

    (
        bias,
        structure_breaks,
    ) = analyze_market_structure(
        swing_highs,
        swing_lows,
    )

    (
        bull_ob,
        bear_ob,
    ) = detect_order_blocks(
        df_processed,
        displacement_atr_mult=ob_mult,
    )

    (
        bull_fvg,
        bear_fvg,
    ) = detect_fvg(
        df_processed
    )

    (
        bsl,
        ssl,
        manipulation,
        manip_note,
    ) = detect_liquidity_and_manipulation(
        df_processed,
        swing_highs,
        swing_lows,
    )

    session_info = session_analyzer(
        df_processed
    )

    asian_levels = (
        compute_asian_session_levels(
            df_processed
        )
    )

    (
        vol_label,
        vol_score,
        atr_value,
    ) = compute_volatility_risk(
        df_processed
    )

    displacements = (
        detect_recent_displacement(
            df_processed
        )
    )

    current_price = round(
        float(
            df_processed[
                "close"
            ].iloc[-1]
        ),
        2,
    )

    fib_levels = {}
    ote = None

    if (
        swing_highs
        and swing_lows
    ):

        (
            last_low_idx,
            last_low_price,
        ) = swing_lows[-1]

        (
            last_high_idx,
            last_high_price,
        ) = swing_highs[-1]

        leg_direction = (
            "BULLISH"
            if last_low_idx
            > last_high_idx
            else "BEARISH"
        )

        lo_price = min(
            last_low_price,
            last_high_price,
        )

        hi_price = max(
            last_low_price,
            last_high_price,
        )

        fib_levels = (
            compute_fibonacci_extension(
                lo_price,
                hi_price,
                leg_direction,
            )
        )

        ote = compute_ote_zone(
            lo_price,
            hi_price,
            leg_direction,
            current_price,
        )

    matching_breaks = [
        brk
        for brk in structure_breaks
        if brk["direction"] == bias
    ]

    if bias != "NEUTRAL":

        structure_score = min(
            100,
            25
            + len(matching_breaks)
            * 25,
        )

    else:
        structure_score = 50

    liquidity_score = (
        80
        if manipulation
        else 40
    )

    if bias == "BULLISH":

        order_block_score = (
            bull_ob["strength"]
            if bull_ob
            else 30
        )

        fvg_score = (
            70
            if bull_fvg
            else 35
        )

    elif bias == "BEARISH":

        order_block_score = (
            bear_ob["strength"]
            if bear_ob
            else 30
        )

        fvg_score = (
            70
            if bear_fvg
            else 35
        )

    else:

        order_block_score = 40
        fvg_score = 40

    order_block_score = min(
        100,
        order_block_score,
    )

    confidence = round(
        np.mean(
            [
                structure_score,
                liquidity_score,
                order_block_score,
                fvg_score,
            ]
        ),
        1,
    )

    return {
        "bias": bias,
        "structure_breaks": structure_breaks,
        "bull_ob": bull_ob,
        "bear_ob": bear_ob,
        "bull_fvg": bull_fvg,
        "bear_fvg": bear_fvg,
        "bsl": bsl,
        "ssl": ssl,
        "manipulation": manipulation,
        "manip_note": manip_note,
        "session": session_info,
        "asian_levels": asian_levels,
        "vol_label": vol_label,
        "vol_score": vol_score,
        "atr": atr_value,
        "displacements": displacements,
        "fib_levels": fib_levels,
        "ote": ote,
        "current_price": current_price,
        "scores": {
            "structure": round(
                structure_score,
                1,
            ),
            "liquidity": round(
                liquidity_score,
                1,
            ),
            "order_block": round(
                order_block_score,
                1,
            ),
            "fvg": round(
                fvg_score,
                1,
            ),
        },
        "confidence": confidence,
    }

# ============================================================
# Institutional Strategy
# ============================================================

def _get_h1_trend(
    df_h1_processed,
):
    if (
        df_h1_processed is None
        or df_h1_processed.empty
    ):
        return None

    last = (
        df_h1_processed.iloc[-1]
    )

    try:

        return (
            "BULLISH"
            if float(
                last["ema_50"]
            )
            > float(
                last["ema_200"]
            )
            else "BEARISH"
        )

    except Exception:
        return None

def _get_previous_day_levels(
    df,
):
    if (
        df is None
        or df.empty
        or "datetime" not in df.columns
    ):
        return None

    work = df.copy()

    work["datetime"] = pd.to_datetime(
        work["datetime"],
        utc=True,
        errors="coerce",
    )

    work.dropna(
        subset=["datetime"],
        inplace=True,
    )

    if work.empty:
        return None

    last_day = (
        work["datetime"]
        .dt.date
        .iloc[-1]
    )

    previous_days = (
        work[
            work["datetime"]
            .dt.date
            < last_day
        ]
    )

    if previous_days.empty:
        return None

    previous_date = (
        previous_days[
            "datetime"
        ]
        .dt.date
        .max()
    )

    day_data = (
        previous_days[
            previous_days[
                "datetime"
            ].dt.date
            == previous_date
        ]
    )

    if day_data.empty:
        return None

    return {
        "pdh": float(
            day_data["high"].max()
        ),
        "pdl": float(
            day_data["low"].min()
        ),
        "date": str(
            previous_date
        ),
    }

def _get_asian_range_for_latest_day(
    df,
):
    if (
        df is None
        or df.empty
        or "datetime" not in df.columns
    ):
        return None

    work = df.copy()

    work["datetime"] = pd.to_datetime(
        work["datetime"],
        utc=True,
        errors="coerce",
    )

    work.dropna(
        subset=["datetime"],
        inplace=True,
    )

    if work.empty:
        return None

    latest_date = (
        work["datetime"]
        .dt.date
        .iloc[-1]
    )

    asian = work[
        (
            work["datetime"]
            .dt.date
            == latest_date
        )
        & (
            work["datetime"]
            .dt.hour
            >= 0
        )
        & (
            work["datetime"]
            .dt.hour
            < 7
        )
    ]

    if asian.empty:
        return None

    return {
        "asian_high": float(
            asian["high"].max()
        ),
        "asian_low": float(
            asian["low"].min()
        ),
        "date": str(
            latest_date
        ),
    }

def _institutional_session_allowed(
    timestamp,
):
    try:

        ts = pd.Timestamp(
            timestamp
        )

        if ts.tzinfo is None:
            ts = ts.tz_localize(
                "UTC"
            )
        else:
            ts = ts.tz_convert(
                "UTC"
            )

        hour = ts.hour

        # London
        if 7 <= hour < 11:
            return (
                True,
                "LONDON",
            )

        # New York
        if 12 <= hour < 16:
            return (
                True,
                "NEW YORK",
            )

        return (
            False,
            "OUTSIDE_SESSION",
        )

    except Exception:
        return (
            False,
            "UNKNOWN",
        )

def _find_institutional_sweep(
    df,
    liquidity_levels,
    trend,
):
    if (
        df is None
        or df.empty
        or len(df) < 8
    ):
        return None

    recent = df.iloc[
        -INSTITUTIONAL_SWEEP_LOOKBACK:
    ].copy()

    candidates = []

    bull_levels = [
        (
            "ASIAN_LOW",
            liquidity_levels.get(
                "asian_low"
            ),
        ),
        (
            "PDL",
            liquidity_levels.get(
                "pdl"
            ),
        ),
    ]

    bear_levels = [
        (
            "ASIAN_HIGH",
            liquidity_levels.get(
                "asian_high"
            ),
        ),
        (
            "PDH",
            liquidity_levels.get(
                "pdh"
            ),
        ),
    ]

    if trend == "BULLISH":

        for idx, row in recent.iterrows():

            for name, level in bull_levels:

                if level is None:
                    continue

                try:

                    swept = (
                        float(row["low"])
                        < float(level)
                    )

                    reclaimed = (
                        float(row["close"])
                        > float(level)
                    )

                    if swept and reclaimed:

                        candidates.append(
                            {
                                "index": int(idx),
                                "type": name,
                                "level": float(level),
                                "sweep_low": float(
                                    row["low"]
                                ),
                                "sweep_high": float(
                                    row["high"]
                                ),
                                "time": str(
                                    row.get(
                                        "datetime",
                                        "",
                                    )
                                ),
                            }
                        )

                except Exception:
                    continue

    elif trend == "BEARISH":

        for idx, row in recent.iterrows():

            for name, level in bear_levels:

                if level is None:
                    continue

                try:

                    swept = (
                        float(row["high"])
                        > float(level)
                    )

                    reclaimed = (
                        float(row["close"])
                        < float(level)
                    )

                    if swept and reclaimed:

                        candidates.append(
                            {
                                "index": int(idx),
                                "type": name,
                                "level": float(level),
                                "sweep_low": float(
                                    row["low"]
                                ),
                                "sweep_high": float(
                                    row["high"]
                                ),
                                "time": str(
                                    row.get(
                                        "datetime",
                                        "",
                                    )
                                ),
                            }
                        )

                except Exception:
                    continue

    if not candidates:
        return None

    return candidates[-1]

def _institutional_mss_confirmed(
    df,
    sweep,
    direction,
):
    if (
        df is None
        or df.empty
        or sweep is None
    ):
        return False, None

    try:

        sweep_position = (
            df.index.get_loc(
                sweep["index"]
            )
        )

    except Exception:
        try:
            sweep_position = int(
                sweep["index"]
            )
        except Exception:
            return False, None

    latest_position = (
        len(df) - 1
    )

    if (
        latest_position
        <= sweep_position
    ):
        return False, None

    bars_after = (
        df.iloc[
            sweep_position + 1:
        ]
    )

    if bars_after.empty:
        return False, None

    confirmation = (
        bars_after.tail(
            INSTITUTIONAL_MSS_LOOKBACK
        )
    )

    if confirmation.empty:
        return False, None

    # لا نعتبر نفس شمعة الـ sweep هي MSS
    before_start = max(
        0,
        sweep_position
        - INSTITUTIONAL_MSS_LOOKBACK,
    )

    before = df.iloc[
        before_start:sweep_position
    ]

    if len(before) < 2:
        return False, None

    reference_high = float(
        before["high"].max()
    )

    reference_low = float(
        before["low"].min()
    )

    if direction == "BULLISH":

        for idx, row in confirmation.iterrows():

            if (
                float(row["close"])
                > reference_high
            ):

                return True, {
                    "type": "BULLISH MSS",
                    "price": float(
                        row["close"]
                    ),
                    "time": str(
                        row.get(
                            "datetime",
                            "",
                        )
                    ),
                }

    else:

        for idx, row in confirmation.iterrows():

            if (
                float(row["close"])
                < reference_low
            ):

                return True, {
                    "type": "BEARISH MSS",
                    "price": float(
                        row["close"]
                    ),
                    "time": str(
                        row.get(
                            "datetime",
                            "",
                        )
                    ),
                }

    return False, None

def _institutional_displacement(
    df,
    direction,
):
    if (
        df is None
        or df.empty
        or "atr" not in df.columns
    ):
        return False, None

    last = df.iloc[-1]

    try:

        atr_value = float(
            last["atr"]
        )

        body = (
            float(last["close"])
            - float(last["open"])
        )

        candle_range = (
            float(last["high"])
            - float(last["low"])
        )

        if (
            not np.isfinite(
                atr_value
            )
            or atr_value <= 0
        ):
            return False, None

        body_pct = (
            abs(body)
            / (
                candle_range
                + 1e-9
            )
            * 100
        )

        direction_ok = (
            (
                direction == "BULLISH"
                and body > 0
            )
            or
            (
                direction == "BEARISH"
                and body < 0
            )
        )

        strong_enough = (
            abs(body)
            >= (
                INSTITUTIONAL_DISPLACEMENT_ATR
                * atr_value
            )
        )

        body_ok = (
            body_pct
            >= INSTITUTIONAL_BODY_PERCENT
        )

        if (
            direction_ok
            and strong_enough
            and body_ok
        ):

            return True, {
                "body": round(
                    abs(body),
                    2,
                ),
                "atr": round(
                    atr_value,
                    2,
                ),
                "body_pct": round(
                    body_pct,
                    1,
                ),
                "time": str(
                    last.get(
                        "datetime",
                        "",
                    )
                ),
            }

    except Exception:
        pass

    return False, None

def _institutional_latest_fvg(
    df,
    direction,
):
    if (
        df is None
        or len(df) < 3
    ):
        return None

    c1 = df.iloc[-3]
    c2 = df.iloc[-2]
    c3 = df.iloc[-1]

    try:

        if direction == "BULLISH":

            if (
                float(c1["high"])
                < float(c3["low"])
            ):

                return {
                    "bias": "BULLISH",
                    "bottom": float(
                        c1["high"]
                    ),
                    "top": float(
                        c3["low"]
                    ),
                    "time": str(
                        c3.get(
                            "datetime",
                            "",
                        )
                    ),
                }

        else:

            if (
                float(c1["low"])
                > float(c3["high"])
            ):

                return {
                    "bias": "BEARISH",
                    "bottom": float(
                        c3["high"]
                    ),
                    "top": float(
                        c1["low"]
                    ),
                    "time": str(
                        c3.get(
                            "datetime",
                            "",
                        )
                    ),
                }

    except Exception:
        pass

    return None

def _institutional_entry_zone(
    df,
    direction,
    fvg,
):
    if (
        df is None
        or df.empty
    ):
        return False

    if fvg is None:
        return True

    try:

        price = float(
            df["close"].iloc[-1]
        )

        # السماح بالدخول أثناء تكون/تأكيد
        # المنطقة أو بالقرب منها.
        atr = float(
            df["atr"].iloc[-1]
        )

        if (
            not np.isfinite(atr)
            or atr <= 0
        ):
            return True

        zone_low = float(
            fvg["bottom"]
        )

        zone_high = float(
            fvg["top"]
        )

        tolerance = (
            atr * 0.60
        )

        return (
            zone_low - tolerance
            <= price
            <= zone_high + tolerance
        )

    except Exception:
        return False

def _institutional_score(
    trend,
    session_name,
    sweep,
    mss,
    displacement,
    fvg,
):
    score = 0.0

    if trend in {
        "BULLISH",
        "BEARISH",
    }:
        score += 20

    if session_name in {
        "LONDON",
        "NEW YORK",
    }:
        score += 10

    if sweep:
        score += 25

    if mss:
        score += 20

    if displacement:
        score += 15

    if fvg:
        score += 10

    return round(
        min(
            score,
            100,
        ),
        1,
    )

def institutional_scanner(
    df_h1_processed,
    df_m15_processed,
    df_m5_processed,
    model,
    scaler,
    cfg,
):
    strategy_name = (
        "Institutional Liquidity"
    )

    result = {
        "strategy": strategy_name,
        "trade_exists": False,
        "direction": None,
        "ai_conf_before_groq": 0.0,
        "experience_conf": 0.0,
        "experience_available": False,
        "experience_win_rate": None,
        "experience_sample": 0,
        "h1_trend": None,
        "session": None,
        "sweep": None,
        "mss": None,
        "displacement": None,
        "fvg": None,
        "pdh": None,
        "pdl": None,
        "asian_high": None,
        "asian_low": None,
        "institutional_score": 0.0,
        "groq_called": False,
        "groq_available": False,
        "groq_agree": None,
        "groq_conf": None,
        "groq_reason": "",
        "final_confidence": 0.0,
        "status": "",
    }

    # --------------------------------------------------------
    # لا تسمح إلا بصفقة واحدة لهذه الاستراتيجية
    # --------------------------------------------------------

    active_df = (
        get_active_trade_for_strategy(
            strategy_name
        )
    )

    if not active_df.empty:

        active = active_df.iloc[0]

        result["trade_exists"] = True

        result["direction"] = (
            active["direction"]
        )

        result[
            "ai_conf_before_groq"
        ] = float(
            active.get(
                "ai_conf",
                0,
            )
            or 0
        )

        saved_groq = active.get(
            "groq_conf",
            np.nan,
        )

        if pd.notna(
            saved_groq
        ):
            result[
                "groq_conf"
            ] = float(
                saved_groq
            )

        result[
            "groq_reason"
        ] = str(
            active.get(
                "groq_note",
                "",
            )
            or ""
        )

        saved_final = active.get(
            "final_confidence",
            np.nan,
        )

        if pd.notna(
            saved_final
        ):

            result[
                "final_confidence"
            ] = float(
                saved_final
            )

        elif (
            result["groq_conf"]
            is not None
        ):

            result[
                "final_confidence"
            ] = result[
                "groq_conf"
            ]

        else:

            result[
                "final_confidence"
            ] = result[
                "ai_conf_before_groq"
            ]

        result["status"] = (
            "استراتيجية Institutional "
            "تدير صفقة نشطة حالياً."
        )

        return (
            result["status"],
            result,
        )

    # --------------------------------------------------------
    # البيانات
    # --------------------------------------------------------

    if (
        df_h1_processed is None
        or df_h1_processed.empty
        or df_m5_processed is None
        or df_m5_processed.empty
    ):

        result["status"] = (
            "Institutional: "
            "بيانات السوق غير كافية."
        )

        return (
            result["status"],
            result,
        )

    h1_trend = _get_h1_trend(
        df_h1_processed
    )

    result["h1_trend"] = (
        h1_trend
    )

    if h1_trend is None:

        result["status"] = (
            "Institutional: "
            "تعذر تحديد ترند H1."
        )

        return (
            result["status"],
            result,
        )

    latest_time = (
        df_m5_processed[
            "datetime"
        ].iloc[-1]
        if "datetime"
        in df_m5_processed.columns
        else None
    )

    allowed, session_name = (
        _institutional_session_allowed(
            latest_time
        )
    )

    result["session"] = (
        session_name
    )

    if not allowed:

        result["status"] = (
            "Institutional: "
            "خارج جلسة London / New York."
        )

        return (
            result["status"],
            result,
        )

    # --------------------------------------------------------
    # السيولة
    # --------------------------------------------------------

    pd_levels = (
        _get_previous_day_levels(
            df_h1_processed
        )
    )

    asian_levels = (
        _get_asian_range_for_latest_day(
            df_m5_processed
        )
    )

    if pd_levels:

        result["pdh"] = round(
            pd_levels["pdh"],
            2,
        )

        result["pdl"] = round(
            pd_levels["pdl"],
            2,
        )

    if asian_levels:

        result["asian_high"] = round(
            asian_levels["asian_high"],
            2,
        )

        result["asian_low"] = round(
            asian_levels["asian_low"],
            2,
        )

    liquidity_levels = {
        "pdh": (
            pd_levels["pdh"]
            if pd_levels
            else None
        ),
        "pdl": (
            pd_levels["pdl"]
            if pd_levels
            else None
        ),
        "asian_high": (
            asian_levels[
                "asian_high"
            ]
            if asian_levels
            else None
        ),
        "asian_low": (
            asian_levels[
                "asian_low"
            ]
            if asian_levels
            else None
        ),
    }

    sweep = _find_institutional_sweep(
        df_m5_processed,
        liquidity_levels,
        h1_trend,
    )

    result["sweep"] = (
        sweep
    )

    if sweep is None:

        result["status"] = (
            "Institutional: "
            "لم يحدث Liquidity Sweep "
            "مطابق للترند."
        )

        return (
            result["status"],
            result,
        )

    # --------------------------------------------------------
    # MSS
    # --------------------------------------------------------

    mss_ok, mss_data = (
        _institutional_mss_confirmed(
            df_m5_processed,
            sweep,
            h1_trend,
        )
    )

    result["mss"] = (
        mss_data
    )

    if not mss_ok:

        result["status"] = (
            "Institutional: "
            "تم رصد Sweep ولكن MSS "
            "لم يتأكد بعد."
        )

        return (
            result["status"],
            result,
        )

    # --------------------------------------------------------
    # Displacement
    # --------------------------------------------------------

    displacement_ok, displacement_data = (
        _institutional_displacement(
            df_m5_processed,
            h1_trend,
        )
    )

    result[
        "displacement"
    ] = displacement_data

    if not displacement_ok:

        result["status"] = (
            "Institutional: "
            "MSS موجود لكن "
            "Displacement غير كافٍ."
        )

        return (
            result["status"],
            result,
        )

    # --------------------------------------------------------
    # FVG
    # --------------------------------------------------------

    fvg = _institutional_latest_fvg(
        df_m5_processed,
        h1_trend,
    )

    result["fvg"] = fvg

    # FVG عامل تأكيد وليس شرطاً
    # وحيداً لمنع تقليل الفرص بشكل مفرط.

    fvg_present = (
        fvg is not None
    )

    # --------------------------------------------------------
    # Entry zone
    # --------------------------------------------------------

    if not _institutional_entry_zone(
        df_m5_processed,
        h1_trend,
        fvg,
    ):

        result["status"] = (
            "Institutional: "
            "السعر لم يدخل منطقة "
            "FVG / Entry الحالية."
        )

        return (
            result["status"],
            result,
        )

    # --------------------------------------------------------
    # AI confirmation
    # --------------------------------------------------------

    if not model_is_ready(
        model,
        scaler,
    ):

        result["status"] = (
            "Institutional: "
            "الشبكة العصبية قيد "
            "التهيئة والتدريب."
        )

        return (
            result["status"],
            result,
        )

    h1_last = (
        df_h1_processed.iloc[-1]
    )

    try:

        feature_values = (
            h1_last[
                FEATURES
            ]
            .astype(float)
            .values
            .reshape(1, -1)
        )

        if not np.isfinite(
            feature_values
        ).all():

            result["status"] = (
                "Institutional: "
                "بيانات AI غير صالحة."
            )

            return (
                result["status"],
                result,
            )

        x_input = scaler.transform(
            feature_values
        )

        probabilities = (
            model.predict_proba(
                x_input
            )[0]
        )

        classes = np.asarray(
            model.classes_
        )

        target_class = (
            1
            if h1_trend == "BULLISH"
            else 0
        )

        target_indices = np.where(
            classes == target_class
        )[0]

        if len(
            target_indices
        ) == 0:

            result["status"] = (
                "Institutional: "
                "النموذج لا يدعم اتجاه "
                "الترند الحالي."
            )

            return (
                result["status"],
                result,
            )

        target_index = int(
            target_indices[0]
        )

        ai_conf = float(
            probabilities[
                target_index
            ] * 100
        )

    except Exception as exc:

        result["status"] = (
            "Institutional: "
            f"تعذر تنفيذ AI: {exc}"
        )

        return (
            result["status"],
            result,
        )

    direction = (
        "BUY 🟢"
        if h1_trend == "BULLISH"
        else "SELL 🔴"
    )

    result[
        "direction"
    ] = direction

    result[
        "ai_conf_before_groq"
    ] = ai_conf

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    experience = (
        get_experience_adjustment(
            direction,
            ai_conf,
        )
    )

    result[
        "experience_available"
    ] = experience[
        "available"
    ]

    result[
        "experience_conf"
    ] = experience[
        "confidence"
    ]

    result[
        "experience_win_rate"
    ] = experience[
        "win_rate"
    ]

    result[
        "experience_sample"
    ] = experience[
        "sample"
    ]

    working_ai_conf = (
        experience["confidence"]
        if experience["available"]
        else ai_conf
    )

    # --------------------------------------------------------
    # Institutional score
    # --------------------------------------------------------

    institutional_score = (
        _institutional_score(
            h1_trend,
            session_name,
            sweep,
            mss_ok,
            displacement_ok,
            fvg_present,
        )
    )

    result[
        "institutional_score"
    ] = institutional_score

    # دمج AI مع شروط المؤسسة
    working_conf = (
        working_ai_conf * 0.45
        + institutional_score * 0.55
    )

    # --------------------------------------------------------
    # منع نفس الشمعة
    # --------------------------------------------------------

    signal_bar_time = str(
        h1_last.get(
            "datetime",
            "",
        )
    )

    last_signal_key = load_setting(
        "last_signal_key_institutional",
        "",
    )

    if (
        signal_bar_time
        and last_signal_key
        == signal_bar_time
    ):

        result[
            "final_confidence"
        ] = round(
            working_conf,
            1,
        )

        result["status"] = (
            "Institutional: "
            "تمت معالجة إشارة H1 "
            "هذه سابقاً."
        )

        return (
            result["status"],
            result,
        )

    # --------------------------------------------------------
    # Groq
    # --------------------------------------------------------

    groq_result = None

    if cfg["use_groq"]:

        result[
            "groq_called"
        ] = True

        extra_context = (
            "\nالاستراتيجية: Institutional Liquidity.\n"
            f"الجلسة: {session_name}.\n"
            f"Previous Day High: "
            f"{result['pdh']}.\n"
            f"Previous Day Low: "
            f"{result['pdl']}.\n"
            f"Asian High: "
            f"{result['asian_high']}.\n"
            f"Asian Low: "
            f"{result['asian_low']}.\n"
            f"Liquidity Sweep: "
            f"{sweep['type']}.\n"
            f"MSS: "
            f"{mss_data['type']}.\n"
            f"Displacement ATR: "
            f"{displacement_data['atr']}.\n"
            f"Displacement Body %: "
            f"{displacement_data['body_pct']}.\n"
            f"FVG: "
            f"{'YES' if fvg_present else 'NO'}.\n"
        )

        groq_result = get_groq_review(
            direction,
            h1_last,
            working_conf,
            cfg["groq_key"],
            cfg["groq_model"],
            extra_context=extra_context,
        )

        if groq_result is not None:

            result[
                "groq_available"
            ] = True

            result[
                "groq_agree"
            ] = groq_result[
                "agree"
            ]

            result[
                "groq_conf"
            ] = groq_result[
                "confidence"
            ]

            result[
                "groq_reason"
            ] = groq_result[
                "reason"
            ]

            if (
                not groq_result[
                    "agree"
                ]
                or groq_result[
                    "confidence"
                ]
                < cfg[
                    "min_groq_conf"
                ]
            ):

                penalty_ratio = (
                    0.35
                    if not groq_result[
                        "agree"
                    ]
                    else 0.15
                )

                blended = (
                    working_conf
                    * (
                        1
                        - penalty_ratio
                    )
                    + groq_result[
                        "confidence"
                    ]
                    * penalty_ratio
                )

                result[
                    "final_confidence"
                ] = round(
                    blended,
                    1,
                )

            else:

                result[
                    "final_confidence"
                ] = round(
                    (
                        working_conf
                        + groq_result[
                            "confidence"
                        ]
                    )
                    / 2,
                    1,
                )

        else:

            result[
                "groq_available"
            ] = False

            result[
                "final_confidence"
            ] = round(
                working_conf * 0.92,
                1,
            )

    else:

        result[
            "final_confidence"
        ] = round(
            working_conf,
            1,
        )

    # --------------------------------------------------------
    # Minimum confidence
    # --------------------------------------------------------

    if (
        result[
            "final_confidence"
        ]
        < cfg["min_conf"]
    ):

        result["status"] = (
            "Institutional: "
            "الشروط موجودة لكن الثقة "
            f"{result['final_confidence']:.1f}% "
            f"< {cfg['min_conf']}%."
        )

        return (
            result["status"],
            result,
        )

    # --------------------------------------------------------
    # SL / TP
    # --------------------------------------------------------

    last_m5 = (
        df_m5_processed.iloc[-1]
    )

    curr = round(
        float(
            last_m5["close"]
        ),
        2,
    )

    atr_value = float(
        last_m5["atr"]
    )

    if (
        not np.isfinite(
            atr_value
        )
        or atr_value <= 0
    ):

        result["status"] = (
            "Institutional: "
            "ATR غير صالح."
        )

        return (
            result["status"],
            result,
        )

    # وقف أكثر منطقية:
    # BUY تحت Sweep Low
    # SELL فوق Sweep High

    safety_buffer = (
        atr_value * 0.20
    )

    if h1_trend == "BULLISH":

        sl_base = float(
            sweep["sweep_low"]
        )

        sl_price = round(
            sl_base
            - safety_buffer,
            2,
        )

        risk_distance = (
            curr
            - sl_price
        )

        if (
            risk_distance <= 0
        ):
            risk_distance = (
                atr_value
                * cfg["atr_mult"]
            )

            sl_price = round(
                curr
                - risk_distance,
                2,
            )

        tp_price = round(
            curr
            + (
                risk_distance
                * cfg["risk_reward"]
            ),
            2,
        )

    else:

        sl_base = float(
            sweep["sweep_high"]
        )

        sl_price = round(
            sl_base
            + safety_buffer,
            2,
        )

        risk_distance = (
            sl_price
            - curr
        )

        if (
            risk_distance <= 0
        ):
            risk_distance = (
                atr_value
                * cfg["atr_mult"]
            )

            sl_price = round(
                curr
                + risk_distance,
                2,
            )

        tp_price = round(
            curr
            - (
                risk_distance
                * cfg["risk_reward"]
            ),
            2,
        )

    if (
        sl_price == curr
        or tp_price == curr
    ):

        result["status"] = (
            "Institutional: "
            "SL/TP غير صالح."
        )

        return (
            result["status"],
            result,
        )

    # --------------------------------------------------------
    # حفظ الصفقة
    # --------------------------------------------------------

    with TRADE_DB_LOCK:

        conn = get_db_connection()

        try:

            # فحص أخير قبل INSERT
            c = conn.cursor()

            c.execute(
                """
                SELECT COUNT(*)
                FROM active_trade
                WHERE strategy = ?
                """,
                (
                    strategy_name,
                ),
            )

            if int(
                c.fetchone()[0]
            ) > 0:

                result[
                    "trade_exists"
                ] = True

                result["status"] = (
                    "Institutional: "
                    "هناك صفقة نشطة "
                    "لهذه الاستراتيجية."
                )

                return (
                    result["status"],
                    result,
                )

            trade_id = (
                get_next_active_trade_id(
                    conn
                )
            )

            c.execute(
                """
                INSERT INTO active_trade (
                    id,
                    symbol,
                    direction,
                    entry,
                    sl,
                    tp,
                    time,
                    features,
                    ai_conf,
                    groq_conf,
                    groq_note,
                    signal_bar_time,
                    final_confidence,
                    strategy
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    trade_id,
                    "XAU/USD",
                    direction,
                    curr,
                    sl_price,
                    tp_price,
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                    json.dumps(
                        {
                            feature: float(
                                h1_last[
                                    feature
                                ]
                            )
                            for feature
                            in FEATURES
                        }
                    ),
                    ai_conf,
                    (
                        result[
                            "groq_conf"
                        ]
                        if result[
                            "groq_conf"
                        ] is not None
                        else None
                    ),
                    result[
                        "groq_reason"
                    ],
                    signal_bar_time,
                    result[
                        "final_confidence"
                    ],
                    strategy_name,
                ),
            )

            conn.commit()

        finally:
            conn.close()

    save_setting(
        "last_signal_key_institutional",
        signal_bar_time,
    )

    groq_line = (
        f"\nGroq: "
        f"{result['groq_conf']:.1f}%"
        if result[
            "groq_available"
        ]
        else ""
    )

    send_alert(
        (
            "🏦 Institutional Trade Signal\n"
            f"Strategy: {strategy_name}\n"
            f"Direction: {direction}\n"
            f"Session: {session_name}\n"
            f"Sweep: {sweep['type']}\n"
            f"MSS: {mss_data['type']}\n"
            f"Displacement: "
            f"{displacement_data['body_pct']:.1f}% body\n"
            f"FVG: "
            f"{'YES' if fvg_present else 'NO'}\n"
            f"Entry: ${curr}\n"
            f"SL: ${sl_price}\n"
            f"TP: ${tp_price}\n"
            f"AI Raw: {ai_conf:.1f}%\n"
            f"Institutional Score: "
            f"{institutional_score:.1f}%"
            f"{groq_line}\n"
            f"Final Confidence: "
            f"{result['final_confidence']:.1f}%"
        ),
        title=(
            "🏦 Institutional Trade — XAU/USD"
        ),
    )

    send_trade_confirmation_alert(
        direction=direction,
        entry=curr,
        sl=sl_price,
        tp=tp_price,
        final_confidence=result[
            "final_confidence"
        ],
        risk_reward_ratio=(
            cfg["risk_reward"]
        ),
        ai_conf=ai_conf,
        groq_conf=(
            result["groq_conf"]
            if result[
                "groq_available"
            ]
            else None
        ),
        strategy=strategy_name,
    )

    result[
        "trade_exists"
    ] = True

    result["status"] = (
        "🏦 تم إطلاق صفقة "
        "Institutional — "
        f"{direction} — "
        f"Sweep: {sweep['type']} — "
        f"MSS — "
        f"Final: "
        f"{result['final_confidence']:.1f}%"
    )

    return (
        result["status"],
        result,
    )

# ============================================================
# AI Scanner — ICT / SMC
# ============================================================

def ai_scanner(
    df_h1_processed,
    df_m15_processed,
    df_m5_processed,
    model,
    scaler,
    ict_data_m15,
    ict_data_m5,
    cfg,
):
    strategy_name = "ICT / SMC"

    result = {
        "strategy": strategy_name,
        "trade_exists": False,
        "direction": None,
        "ai_conf_before_groq": 0.0,
        "experience_conf": 0.0,
        "experience_available": False,
        "experience_win_rate": None,
        "experience_sample": 0,
        "ict_confidence": None,
        "ict_bias": None,
        "h1_trend": None,
        "m15_bias": None,
        "m5_bias": None,
        "groq_called": False,
        "groq_available": False,
        "groq_agree": None,
        "groq_conf": None,
        "groq_reason": "",
        "final_confidence": 0.0,
        "status": "",
    }

    use_groq_local = cfg[
        "use_groq"
    ]

    groq_key_local = cfg[
        "groq_key"
    ]

    groq_model_local = cfg[
        "groq_model"
    ]

    min_groq_conf_local = cfg[
        "min_groq_conf"
    ]

    min_conf_local = cfg[
        "min_conf"
    ]

    atr_mult_local = cfg[
        "atr_mult"
    ]

    risk_reward_local = cfg[
        "risk_reward"
    ]

    # --------------------------------------------------------
    # ICT/H1
    # --------------------------------------------------------

    if (
        df_h1_processed is not None
        and not df_h1_processed.empty
    ):

        h1_last = (
            df_h1_processed.iloc[-1]
        )

        result[
            "h1_trend"
        ] = (
            "BULLISH"
            if h1_last["ema_50"]
            > h1_last["ema_200"]
            else "BEARISH"
        )

    result["m15_bias"] = (
        ict_data_m15.get(
            "bias"
        )
        if ict_data_m15
        else "NEUTRAL"
    )

    result["m5_bias"] = (
        ict_data_m5.get(
            "bias"
        )
        if ict_data_m5
        else "NEUTRAL"
    )

    # --------------------------------------------------------
    # فقط ICT active trade
    # --------------------------------------------------------

    df_act = (
        get_active_trade_for_strategy(
            strategy_name
        )
    )

    if not df_act.empty:

        active = df_act.iloc[0]

        result[
            "trade_exists"
        ] = True

        result[
            "direction"
        ] = active[
            "direction"
        ]

        result[
            "ai_conf_before_groq"
        ] = float(
            active.get(
                "ai_conf",
                0,
            )
            or 0
        )

        groq_conf = active.get(
            "groq_conf",
            np.nan,
        )

        if pd.notna(
            groq_conf
        ):

            result[
                "groq_conf"
            ] = float(
                groq_conf
            )

        result[
            "groq_reason"
        ] = str(
            active.get(
                "groq_note",
                "",
            )
            or ""
        )

        saved_final = active.get(
            "final_confidence",
            np.nan,
        )

        if pd.notna(
            saved_final
        ):

            result[
                "final_confidence"
            ] = float(
                saved_final
            )

        elif (
            result[
                "groq_conf"
            ]
            is not None
        ):

            result[
                "final_confidence"
            ] = result[
                "groq_conf"
            ]

        else:

            result[
                "final_confidence"
            ] = result[
                "ai_conf_before_groq"
            ]

        result["status"] = (
            "ICT / SMC يدير "
            "صفقة نشطة حالياً."
        )

        return (
            result["status"],
            result,
        )

    # --------------------------------------------------------
    # البيانات
    # --------------------------------------------------------

    if (
        df_h1_processed is None
        or df_h1_processed.empty
    ):

        result["status"] = (
            "ICT: لا توجد صفقة: "
            "بيانات السوق غير كافية."
        )

        return (
            result["status"],
            result,
        )

    if not model_is_ready(
        model,
        scaler,
    ):

        result["status"] = (
            "ICT: الشبكة العصبية "
            "قيد التهيئة والتدريب."
        )

        return (
            result["status"],
            result,
        )

    last = (
        df_h1_processed.iloc[-1]
    )

    signal_bar_time = str(
        last.get(
            "datetime",
            "",
        )
    )

    last_signal_key = load_setting(
        "last_signal_key",
        "",
    )

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    try:

        feature_values = (
            last[
                FEATURES
            ]
            .astype(float)
            .values
            .reshape(1, -1)
        )

        if not np.isfinite(
            feature_values
        ).all():

            result["status"] = (
                "ICT: بيانات المؤشرات "
                "غير صالحة."
            )

            return (
                result["status"],
                result,
            )

        x_input = scaler.transform(
            feature_values
        )

        probabilities = (
            model.predict_proba(
                x_input
            )[0]
        )

        classes = np.asarray(
            model.classes_
        )

        best_index = int(
            np.argmax(
                probabilities
            )
        )

        model_pred = int(
            classes[
                best_index
            ]
        )

    except Exception as exc:

        result["status"] = (
            "ICT: تعذر تنفيذ "
            f"الشبكة العصبية: {exc}"
        )

        return (
            result["status"],
            result,
        )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    h1_last = (
        df_h1_processed.iloc[-1]
    )

    h1_trend = (
        "BULLISH"
        if h1_last["ema_50"]
        > h1_last["ema_200"]
        else "BEARISH"
    )

    result[
        "h1_trend"
    ] = h1_trend

    pred = model_pred

    if (
        h1_trend == "BEARISH"
        and pred == 1
    ):

        pred = 0

        result["status"] = (
            "ICT: ترند هابط — "
            "البحث نحو فرص البيع."
        )

    elif (
        h1_trend == "BULLISH"
        and pred == 0
    ):

        pred = 1

        result["status"] = (
            "ICT: ترند صاعد — "
            "البحث نحو فرص الشراء."
        )

    # --------------------------------------------------------
    # confidence
    # --------------------------------------------------------

    try:

        target_indices = np.where(
            classes == pred
        )[0]

        if len(
            target_indices
        ) == 0:

            result["status"] = (
                "ICT: النموذج لا يدعم "
                "الاتجاه المطلوب."
            )

            return (
                result["status"],
                result,
            )

        target_index = int(
            target_indices[0]
        )

        ai_conf = float(
            probabilities[
                target_index
            ]
            * 100
        )

    except Exception as exc:

        result["status"] = (
            "ICT: تعذر حساب "
            f"الثقة: {exc}"
        )

        return (
            result["status"],
            result,
        )

    direction = (
        "BUY 🟢"
        if pred == 1
        else "SELL 🔴"
    )

    result[
        "direction"
    ] = direction

    result[
        "ai_conf_before_groq"
    ] = ai_conf

    # --------------------------------------------------------
    # M15/M5
    # --------------------------------------------------------

    m15_bias = (
        ict_data_m15.get(
            "bias"
        )
        if ict_data_m15
        else "NEUTRAL"
    )

    m5_bias = (
        ict_data_m5.get(
            "bias"
        )
        if ict_data_m5
        else "NEUTRAL"
    )

    result[
        "m15_bias"
    ] = m15_bias

    result[
        "m5_bias"
    ] = m5_bias

    confirmation_score = 1.0

    if (
        m15_bias != "NEUTRAL"
        and m15_bias != h1_trend
    ):
        confirmation_score -= 0.35

    if (
        m5_bias != "NEUTRAL"
        and m5_bias != h1_trend
    ):
        confirmation_score -= 0.35

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    experience = (
        get_experience_adjustment(
            direction,
            ai_conf,
        )
    )

    result[
        "experience_available"
    ] = experience[
        "available"
    ]

    result[
        "experience_conf"
    ] = experience[
        "confidence"
    ]

    result[
        "experience_win_rate"
    ] = experience[
        "win_rate"
    ]

    result[
        "experience_sample"
    ] = experience[
        "sample"
    ]

    working_conf = (
        experience["confidence"]
        if experience["available"]
        else ai_conf
    )

    # --------------------------------------------------------
    # ICT M15
    # --------------------------------------------------------

    if ict_data_m15 is not None:

        ict_bias = (
            ict_data_m15.get(
                "bias"
            )
        )

        ict_conf = (
            ict_data_m15.get(
                "confidence",
                50.0,
            )
        )

        result[
            "ict_bias"
        ] = ict_bias

        result[
            "ict_confidence"
        ] = ict_conf

        ai_direction_bias = (
            "BULLISH"
            if pred == 1
            else "BEARISH"
        )

        if (
            ict_bias
            == ai_direction_bias
        ):

            ict_component = (
                ict_conf
            )

        elif (
            ict_bias
            == "NEUTRAL"
        ):

            ict_component = 50.0

        else:

            ict_component = (
                100.0
                - ict_conf
            )

        working_conf = (
            working_conf * 0.85
            + ict_component * 0.15
        )

    working_conf *= (
        confirmation_score
    )

    # --------------------------------------------------------
    # Duplicate signal
    # --------------------------------------------------------

    if (
        signal_bar_time
        and last_signal_key
        == signal_bar_time
    ):

        result[
            "final_confidence"
        ] = round(
            working_conf,
            1,
        )

        result["status"] = (
            "ICT: تمت معالجة "
            "هذه الشمعة سابقاً."
        )

        return (
            result["status"],
            result,
        )

    # --------------------------------------------------------
    # Groq
    # --------------------------------------------------------

    groq_result = None

    if use_groq_local:

        result[
            "groq_called"
        ] = True

        groq_result = (
            get_groq_review(
                direction,
                last,
                working_conf,
                groq_key_local,
                groq_model_local,
            )
        )

        if groq_result is not None:

            result[
                "groq_available"
            ] = True

            result[
                "groq_agree"
            ] = groq_result[
                "agree"
            ]

            result[
                "groq_conf"
            ] = groq_result[
                "confidence"
            ]

            result[
                "groq_reason"
            ] = groq_result[
                "reason"
            ]

            if (
                not groq_result[
                    "agree"
                ]
                or groq_result[
                    "confidence"
                ]
                < min_groq_conf_local
            ):

                penalty_ratio = (
                    0.35
                    if not groq_result[
                        "agree"
                    ]
                    else 0.15
                )

                blended = (
                    working_conf
                    * (
                        1
                        - penalty_ratio
                    )
                    + groq_result[
                        "confidence"
                    ]
                    * penalty_ratio
                )

                result[
                    "final_confidence"
                ] = round(
                    blended,
                    1,
                )

            else:

                result[
                    "final_confidence"
                ] = round(
                    (
                        working_conf
                        + groq_result[
                            "confidence"
                        ]
                    )
                    / 2,
                    1,
                )

        else:

            result[
                "groq_available"
            ] = False

            result[
                "final_confidence"
            ] = round(
                working_conf * 0.92,
                1,
            )

    else:

        result[
            "final_confidence"
        ] = round(
            working_conf,
            1,
        )

    # --------------------------------------------------------
    # Minimum confidence
    # --------------------------------------------------------

    if (
        result[
            "final_confidence"
        ]
        < min_conf_local
    ):

        result["status"] = (
            f"🟡 ICT ({direction}) "
            "مطابق للترند، لكن لم "
            "تكتمل شروط التأكيد "
            f"(الثقة "
            f"{result['final_confidence']:.1f}% "
            f"< {min_conf_local}%)."
        )

        return (
            result["status"],
            result,
        )

    # --------------------------------------------------------
    # SL / TP
    # --------------------------------------------------------

    curr = round(
        float(
            last["close"]
        ),
        2,
    )

    atr_value = float(
        last["atr"]
    )

    if (
        not np.isfinite(
            atr_value
        )
        or atr_value <= 0
    ):

        result["status"] = (
            "ICT: قيمة ATR غير صالحة."
        )

        return (
            result["status"],
            result,
        )

    sl_distance = round(
        atr_value
        * atr_mult_local,
        2,
    )

    tp_distance = round(
        sl_distance
        * risk_reward_local,
        2,
    )

    if pred == 1:

        sl_price = round(
            curr
            - sl_distance,
            2,
        )

        tp_price = round(
            curr
            + tp_distance,
            2,
        )

    else:

        sl_price = round(
            curr
            + sl_distance,
            2,
        )

        tp_price = round(
            curr
            - tp_distance,
            2,
        )

    # --------------------------------------------------------
    # حفظ ICT دون حذف Institutional
    # --------------------------------------------------------

    with TRADE_DB_LOCK:

        conn = get_db_connection()

        try:

            c = conn.cursor()

            c.execute(
                """
                SELECT COUNT(*)
                FROM active_trade
                WHERE strategy = ?
                """,
                (
                    strategy_name,
                ),
            )

            if int(
                c.fetchone()[0]
            ) > 0:

                result[
                    "trade_exists"
                ] = True

                result["status"] = (
                    "ICT: توجد صفقة "
                    "نشطة بالفعل."
                )

                return (
                    result["status"],
                    result,
                )

            trade_id = (
                get_next_active_trade_id(
                    conn
                )
            )

            c.execute(
                """
                INSERT INTO active_trade (
                    id,
                    symbol,
                    direction,
                    entry,
                    sl,
                    tp,
                    time,
                    features,
                    ai_conf,
                    groq_conf,
                    groq_note,
                    signal_bar_time,
                    final_confidence,
                    strategy
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    trade_id,
                    "XAU/USD",
                    direction,
                    curr,
                    sl_price,
                    tp_price,
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                    json.dumps(
                        {
                            feature: float(
                                last[
                                    feature
                                ]
                            )
                            for feature
                            in FEATURES
                        }
                    ),
                    ai_conf,
                    (
                        result[
                            "groq_conf"
                        ]
                        if result[
                            "groq_conf"
                        ] is not None
                        else None
                    ),
                    result[
                        "groq_reason"
                    ],
                    signal_bar_time,
                    result[
                        "final_confidence"
                    ],
                    strategy_name,
                ),
            )

            conn.commit()

        finally:
            conn.close()

    save_setting(
        "last_signal_key",
        signal_bar_time,
    )

    groq_line = (
        f"\nGroq: "
        f"{result['groq_conf']:.1f}%"
        if result[
            "groq_available"
        ]
        else ""
    )

    send_alert(
        (
            "🧠 ICT / SMC Trade Signal\n"
            f"Strategy: {strategy_name}\n"
            f"Direction: {direction}\n"
            f"Entry: ${curr}\n"
            f"SL: ${sl_price}\n"
            f"TP: ${tp_price}\n"
            f"AI Raw: {ai_conf:.1f}%\n"
            f"Experience+ICT: "
            f"{working_conf:.1f}%"
            f"{groq_line}\n"
            f"Final Confidence: "
            f"{result['final_confidence']:.1f}%"
        )
    )

    send_trade_confirmation_alert(
        direction=direction,
        entry=curr,
        sl=sl_price,
        tp=tp_price,
        final_confidence=result[
            "final_confidence"
        ],
        risk_reward_ratio=(
            risk_reward_local
        ),
        ai_conf=ai_conf,
        groq_conf=(
            result["groq_conf"]
            if result[
                "groq_available"
            ]
            else None
        ),
        strategy=strategy_name,
    )

    result[
        "trade_exists"
    ] = True

    result["status"] = (
        f"🟢 ICT: تم إطلاق الإشارة "
        f"({direction}) — "
        f"AI: {ai_conf:.1f}% — "
        f"Final: "
        f"{result['final_confidence']:.1f}%"
    )

    return (
        result["status"],
        result,
    )

# ============================================================
# Background Engine State
# ============================================================

@st.cache_resource
def _get_shared_engine_state():
    return {
        "data": {
            "last_twelve_error": None,
            "last_groq_error": None,
            "last_train_time": None,
            "last_update_time": None,
            "ai_result": None,
            "institutional_result": None,
            "strategy_results": {},
            "scan_msg": "",
            "ict_confidence": None,
            "snapshot": None,
            "engine_running": False,
            "engine_error": None,
            "last_data_source": None,
        },
        "lock": threading.Lock(),
    }

_shared_engine_state = (
    _get_shared_engine_state()
)

APP_STATE = (
    _shared_engine_state["data"]
)

_APP_STATE_LOCK = (
    _shared_engine_state["lock"]
)

def APP_STATE_set(
    key,
    value,
):
    with _APP_STATE_LOCK:
        APP_STATE[key] = value

def APP_STATE_get(
    key,
    default=None,
):
    with _APP_STATE_LOCK:
        return APP_STATE.get(
            key,
            default,
        )

def _read_worker_config():
    def safe_float(
        key,
        default,
    ):
        try:

            return float(
                load_setting(
                    key,
                    str(default),
                )
                or default
            )

        except Exception:
            return float(
                default
            )

    return {
        "twelve_key": load_setting(
            "twelve_key",
            "",
        ),
        "use_groq": (
            load_setting(
                "use_groq",
                "1",
            )
            == "1"
        ),
        "groq_key": load_setting(
            "groq_key",
            "",
        ),
        "groq_model": load_setting(
            "groq_model",
            "openai/gpt-oss-120b",
        ),
        "min_groq_conf": safe_float(
            "min_groq_conf",
            50,
        ),
        "min_conf": safe_float(
            "min_conf",
            65,
        ),
        "atr_mult": safe_float(
            "atr_mult",
            1.5,
        ),
        "risk_reward": safe_float(
            "risk_reward",
            2.0,
        ),
    }

# ============================================================
# مراقبة جميع الصفقات النشطة
# ============================================================

def _monitor_active_trade(
    df_live_processed,
    model,
    scaler,
    cfg,
):
    if (
        df_live_processed is None
        or df_live_processed.empty
    ):
        return

    conn = get_db_connection()

    try:

        active_df = pd.read_sql(
            """
            SELECT *
            FROM active_trade
            ORDER BY id ASC
            """,
            conn,
        )

    finally:
        conn.close()

    if active_df.empty:
        return

    last_row = (
        df_live_processed.iloc[-1]
    )

    current_bar_time = str(
        last_row.get(
            "datetime",
            "",
        )
    )

    for _, trade_row in (
        active_df.iterrows()
    ):

        try:
            _monitor_single_active_trade(
                trade_row,
                current_bar_time,
                last_row,
                model,
                scaler,
                cfg,
            )

        except Exception:
            continue

def _monitor_single_active_trade(
    trade_row,
    current_bar_time,
    last_row,
    model,
    scaler,
    cfg,
):
    trade_id = int(
        trade_row["id"]
    )

    strategy_name = str(
        trade_row.get(
            "strategy",
            "ICT / SMC",
        )
        or "ICT / SMC"
    )

    is_buy_trade = (
        "BUY"
        in str(
            trade_row["direction"]
        )
    )

    signal_bar_time = str(
        trade_row.get(
            "signal_bar_time",
            "",
        )
        or ""
    )

    can_monitor_trade = (
        current_bar_time
        != signal_bar_time
    )

    # --------------------------------------------------------
    # Reversal warning
    # --------------------------------------------------------

    if (
        can_monitor_trade
        and model_is_ready(
            model,
            scaler,
        )
    ):

        try:

            x_current = scaler.transform(
                last_row[
                    FEATURES
                ]
                .astype(float)
                .values
                .reshape(1, -1)
            )

            current_probs = (
                model.predict_proba(
                    x_current
                )[0]
            )

            classes = np.asarray(
                model.classes_
            )

            current_index = int(
                np.argmax(
                    current_probs
                )
            )

            current_pred = int(
                classes[
                    current_index
                ]
            )

            current_conf = float(
                current_probs[
                    current_index
                ]
                * 100
            )

            reversal_detected = False

            if (
                is_buy_trade
                and current_pred == 0
                and current_conf
                >= (
                    cfg[
                        "min_conf"
                    ]
                    - 5
                )
            ):

                reversal_detected = True

            elif (
                not is_buy_trade
                and current_pred == 1
                and current_conf
                >= (
                    cfg[
                        "min_conf"
                    ]
                    - 5
                )
            ):

                reversal_detected = True

            if reversal_detected:

                send_alert(
                    (
                        "⚠️ تنبيه الشبكة العصبية\n"
                        f"الاستراتيجية: "
                        f"{strategy_name}\n"
                        f"الصفقة: "
                        f"{trade_row['direction']}\n"
                        f"انعكاس محتمل بقوة "
                        f"{current_conf:.1f}%."
                    ),
                    "🚨 AI Reversal Warning",
                )

        except Exception:
            pass

    if not can_monitor_trade:
        return

    # --------------------------------------------------------
    # SL / TP
    # --------------------------------------------------------

    high_price = float(
        last_row["high"]
    )

    low_price = float(
        last_row["low"]
    )

    sl_price = float(
        trade_row["sl"]
    )

    tp_price = float(
        trade_row["tp"]
    )

    hit_sl = False
    hit_tp = False

    if is_buy_trade:

        if (
            low_price
            <= sl_price
        ):
            hit_sl = True

        if (
            high_price
            >= tp_price
        ):
            hit_tp = True

    else:

        if (
            high_price
            >= sl_price
        ):
            hit_sl = True

        if (
            low_price
            <= tp_price
        ):
            hit_tp = True

    both_hit = (
        hit_sl
        and hit_tp
    )

    if both_hit:

        # السلوك المحافظ الموجود سابقاً
        hit_tp = False
        hit_sl = True

    if not (
        hit_sl
        or hit_tp
    ):
        return

    win_value = (
        1
        if hit_tp
        else 0
    )

    note_str = (
        "SL and TP touched in same candle; "
        "conservative SL outcome."
        if both_hit
        else (
            "AI Target Reached"
            if hit_tp
            else "AI Stop Loss Hit"
        )
    )

    # --------------------------------------------------------
    # Final confidence
    # --------------------------------------------------------

    saved_final = trade_row.get(
        "final_confidence",
        np.nan,
    )

    if pd.notna(
        saved_final
    ):

        final_confidence = float(
            saved_final
        )

    else:

        saved_groq = trade_row.get(
            "groq_conf",
            np.nan,
        )

        if pd.notna(
            saved_groq
        ):

            final_confidence = float(
                saved_groq
            )

        else:

            final_confidence = float(
                trade_row.get(
                    "ai_conf",
                    0,
                )
                or 0
            )

    # --------------------------------------------------------
    # تسجيل الصفقة المغلقة
    # --------------------------------------------------------

    with TRADE_DB_LOCK:

        conn = get_db_connection()

        try:

            c = conn.cursor()

            c.execute(
                """
                INSERT INTO trades (
                    date,
                    symbol,
                    direction,
                    entry,
                    sl,
                    tp,
                    win,
                    note,
                    claude_conf,
                    claude_note,
                    groq_conf,
                    groq_note,
                    ai_conf_before_groq,
                    ai_conf_after_groq,
                    final_confidence,
                    strategy
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    str(
                        datetime.now(
                            timezone.utc
                        ).date()
                    ),
                    trade_row[
                        "symbol"
                    ],
                    trade_row[
                        "direction"
                    ],
                    float(
                        trade_row[
                            "entry"
                        ]
                    ),
                    float(
                        trade_row[
                            "sl"
                        ]
                    ),
                    float(
                        trade_row[
                            "tp"
                        ]
                    ),
                    win_value,
                    note_str,
                    None,
                    None,
                    (
                        float(
                            trade_row.get(
                                "groq_conf"
                            )
                        )
                        if pd.notna(
                            trade_row.get(
                                "groq_conf"
                            )
                        )
                        else None
                    ),
                    str(
                        trade_row.get(
                            "groq_note",
                            "",
                        )
                        or ""
                    ),
                    float(
                        trade_row.get(
                            "ai_conf",
                            0,
                        )
                        or 0
                    ),
                    final_confidence,
                    final_confidence,
                    strategy_name,
                ),
            )

            c.execute(
                """
                DELETE FROM active_trade
                WHERE id = ?
                """,
                (
                    trade_id,
                ),
            )

            conn.commit()

        finally:
            conn.close()

    # --------------------------------------------------------
    # Online Learning
    # --------------------------------------------------------

    if model_is_ready(
        model,
        scaler,
    ):

        try:

            feat_dict = json.loads(
                trade_row.get(
                    "features"
                )
                or "{}"
            )

            if feat_dict:

                x_vec = np.array(
                    [
                        [
                            float(
                                feat_dict.get(
                                    f,
                                    np.nan,
                                )
                            )
                            for f in FEATURES
                        ]
                    ],
                    dtype=float,
                )

                if np.isfinite(
                    x_vec
                ).all():

                    x_scaled = (
                        scaler.transform(
                            x_vec
                        )
                    )

                    outcome_up = (
                        (
                            is_buy_trade
                            and win_value == 1
                        )
                        or (
                            not is_buy_trade
                            and win_value == 0
                        )
                    )

                    label = np.array(
                        [
                            1
                            if outcome_up
                            else 0
                        ]
                    )

                    with MODEL_IO_LOCK:

                        model.partial_fit(
                            x_scaled,
                            label,
                            classes=np.asarray(
                                model.classes_,
                            ),
                        )

                        _atomic_joblib_dump(
                            model,
                            MODEL_FILE,
                        )

                    APP_STATE_set(
                        "last_train_time",
                        datetime.now(
                            timezone.utc
                        ).isoformat(),
                    )

        except Exception:
            pass

    send_alert(
        (
            f"Closed {trade_row['symbol']} "
            f"{trade_row['direction']}\n"
            f"Strategy: {strategy_name}\n"
            f"-> {note_str}"
        ),
        "🧠 AI Trade Settled",
    )

# ============================================================
# Heartbeat
# ============================================================

def _maybe_send_heartbeat(
    ai_result,
    institutional_result,
    snapshot,
):
    last_heartbeat_raw = (
        load_setting(
            "last_heartbeat_time",
            "",
        )
    )

    now = datetime.now(
        timezone.utc
    )

    should_send = True

    if last_heartbeat_raw:

        try:

            last_heartbeat = (
                datetime.fromisoformat(
                    last_heartbeat_raw
                )
            )

            if (
                (now - last_heartbeat).total_seconds()
                < HEARTBEAT_INTERVAL_SECONDS
            ):
                should_send = False

        except Exception:
            should_send = True

    if not should_send:
        return

    price_txt = (
        f"${snapshot['close']}"
        if snapshot
        and snapshot.get(
            "close"
        )
        else "—"
    )

    active_df = (
        get_active_trades_df()
    )

    if active_df.empty:

        trade_txt = (
            "لا توجد صفقات نشطة حالياً"
        )

    else:

        parts = []

        for _, row in (
            active_df.iterrows()
        ):

            parts.append(
                (
                    f"{row.get('strategy', 'ICT / SMC')}: "
                    f"{row.get('direction', '')}"
                )
            )

        trade_txt = "\n".join(
            parts
        )

    send_alert(
        (
            "❤️ المحرك يعمل بشكل طبيعي\n"
            f"آخر سعر: {price_txt}\n"
            f"الصفقات النشطة:\n"
            f"{trade_txt}\n"
            f"الوقت: "
            f"{now.strftime('%Y-%m-%d %H:%M UTC')}"
        ),
        title="❤️ Engine Heartbeat",
    )

    save_setting(
        "last_heartbeat_time",
        now.isoformat(),
    )

# ============================================================
# Engine Cycle
# ============================================================

def _engine_cycle():
    cfg = (
        _read_worker_config()
    )

    twelve_key_local = (
        cfg["twelve_key"]
    )

    maybe_spawn_training(
        twelve_key_local
    )

    model, scaler = (
        load_current_model()
    )

    # --------------------------------------------------------
    # H1
    # --------------------------------------------------------

    df_live_raw_h1 = (
        fetch_live_series(
            symbol_twelve="XAU/USD",
            symbol_yahoo="XAUUSD=X",
            interval_twelve="1h",
            interval_yahoo="60m",
            range_yahoo="60d",
            outputsize_twelve=LIVE_OUTPUT_SIZE,
            twelve_api_key=twelve_key_local,
        )
    )

    # --------------------------------------------------------
    # M15
    # --------------------------------------------------------

    df_live_raw_m15 = (
        fetch_live_series(
            symbol_twelve="XAU/USD",
            symbol_yahoo="XAUUSD=X",
            interval_twelve="15min",
            interval_yahoo="15m",
            range_yahoo="5d",
            outputsize_twelve=LIVE_OUTPUT_SIZE,
            twelve_api_key=twelve_key_local,
        )
    )

    # --------------------------------------------------------
    # M5
    # --------------------------------------------------------

    df_live_raw_m5 = (
        fetch_live_series(
            symbol_twelve="XAU/USD",
            symbol_yahoo="XAUUSD=X",
            interval_twelve="5min",
            interval_yahoo="5m",
            range_yahoo="5d",
            outputsize_twelve=LIVE_OUTPUT_SIZE,
            twelve_api_key=twelve_key_local,
        )
    )

    df_live_h1 = (
        keep_closed_candles(
            df_live_raw_h1,
            interval_hours=1,
        )
    )

    df_live_m15 = (
        keep_closed_candles(
            df_live_raw_m15,
            interval_hours=0.25,
        )
    )

    df_live_m5 = (
        keep_closed_candles(
            df_live_raw_m5,
            interval_hours=5 / 60,
        )
    )

    df_h1_processed = (
        apply_deep_indicators(
            df_live_h1
        )
    )

    df_m15_processed = (
        apply_deep_indicators(
            df_live_m15
        )
    )

    df_m5_processed = (
        apply_deep_indicators(
            df_live_m5
        )
    )

    # --------------------------------------------------------
    # ICT M15
    # --------------------------------------------------------

    ict_data_m15 = (
        run_ict_engine(
            df_m15_processed,
            swing_lookback=(
                ICT_SWING_LOOKBACK
            ),
            ob_mult=(
                ICT_OB_DISPLACEMENT_MULT
            ),
        )
        if not df_m15_processed.empty
        else None
    )

    # --------------------------------------------------------
    # ICT M5
    # --------------------------------------------------------

    ict_data_m5 = (
        run_ict_engine(
            df_m5_processed,
            swing_lookback=(
                ICT_SWING_LOOKBACK
            ),
            ob_mult=(
                ICT_OB_DISPLACEMENT_MULT
            ),
        )
        if not df_m5_processed.empty
        else None
    )

    # ========================================================
    # الاستراتيجية الأولى
    # ========================================================

    scan_msg_ict, ai_result = (
        ai_scanner(
            df_h1_processed,
            df_m15_processed,
            df_m5_processed,
            model,
            scaler,
            ict_data_m15,
            ict_data_m5,
            cfg,
        )
    )

    # ========================================================
    # الاستراتيجية الثانية
    # تعمل بشكل مستقل
    # ========================================================

    scan_msg_institutional, institutional_result = (
        institutional_scanner(
            df_h1_processed,
            df_m15_processed,
            df_m5_processed,
            model,
            scaler,
            cfg,
        )
    )

    # ========================================================
    # مراقبة جميع الصفقات
    # ========================================================

    _monitor_active_trade(
        df_h1_processed,
        model,
        scaler,
        cfg,
    )

    # ========================================================
    # Snapshot
    # ========================================================

    snapshot = None

    if not df_h1_processed.empty:

        last_snapshot = (
            df_h1_processed.iloc[-1]
        )

        snapshot = {
            "close": float(
                last_snapshot[
                    "close"
                ]
            ),
            "rsi": float(
                last_snapshot[
                    "rsi"
                ]
            ),
            "ema_50": float(
                last_snapshot[
                    "ema_50"
                ]
            ),
            "ema_200": float(
                last_snapshot[
                    "ema_200"
                ]
            ),
            "atr": float(
                last_snapshot[
                    "atr"
                ]
            ),
            "datetime": str(
                last_snapshot.get(
                    "datetime",
                    "",
                )
            ),
        }

    # ========================================================
    # حفظ الحالة
    # ========================================================

    strategy_results = {
        "ICT / SMC": ai_result,
        "Institutional Liquidity": (
            institutional_result
        ),
    }

    with _APP_STATE_LOCK:

        APP_STATE[
            "ai_result"
        ] = ai_result

        APP_STATE[
            "institutional_result"
        ] = institutional_result

        APP_STATE[
            "strategy_results"
        ] = strategy_results

        APP_STATE[
            "scan_msg"
        ] = (
            f"ICT: {scan_msg_ict} | "
            f"Institutional: "
            f"{scan_msg_institutional}"
        )

        APP_STATE[
            "ict_confidence"
        ] = (
            ict_data_m15.get(
                "confidence"
            )
            if ict_data_m15
            else None
        )

        APP_STATE[
            "snapshot"
        ] = snapshot

        APP_STATE[
            "last_update_time"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

        APP_STATE[
            "engine_error"
        ] = None

    _maybe_send_heartbeat(
        ai_result,
        institutional_result,
        snapshot,
    )

# ============================================================
# Background Worker
# ============================================================

def background_worker_loop():

    APP_STATE_set(
        "engine_running",
        True,
    )

    while True:

        try:

            _engine_cycle()

        except Exception as exc:

            APP_STATE_set(
                "engine_error",
                (
                    f"{exc}\n"
                    f"{traceback.format_exc()}"
                ),
            )

        time.sleep(
            WORKER_LOOP_SECONDS
        )

def ensure_background_worker_started():

    for t in threading.enumerate():

        if t.name == "bg_worker_loop":
            return

    worker = threading.Thread(
        target=background_worker_loop,
        daemon=True,
        name="bg_worker_loop",
    )

    worker.start()

ensure_background_worker_started()

# ============================================================
# UI
# ============================================================

st.title(
    "🧠 نظام التداول العميق —
