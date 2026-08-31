"""
XAU/USD Deep AI Engine
=========================================================
ICT / Smart Money + Neural Network + Groq Second Opinion

النسخة المصححة والمراجعة:
1) Twelve Data كمصدر بيانات.
2) Neural Network على XAU/USD H1.
3) Experience Layer من نتائج الصفقات المغلقة.
4) Groq كرأي ثانٍ اختياري.
5) ICT / Smart Money:
   - Market Structure
   - BOS / CHoCH
   - Order Blocks
   - Fair Value Gaps
   - Liquidity / BSL / SSL
   - Liquidity Sweeps
   - Session Analysis
   - Asian Session
   - Fibonacci Extensions
   - OTE
   - Displacement
   - Volatility & Risk
   - Score Breakdown
6) واجهة:
   - حالة الصفقة
   - الاتجاه
   - AI Confidence
   - Groq Review
   - Final Confidence
   - ICT Dashboard
7) Ntfy Alerts
8) SQLite
9) Background Training
10) Training Lock
11) حماية من النموذج غير الجاهز
12) منع تكرار الإشارة على نفس شمعة H1
13) استخدام آخر شمعة مغلقة فقط للإشارة
14) عدم إفساد نموذج الاتجاه بنتائج win/loss
15) حفظ مفاتيح API بشكل دائم في قاعدة البيانات
16) تطبيق قواعد ICT كاستراتيجية تداول
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
import streamlit as st

from streamlit_autorefresh import st_autorefresh

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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
    """
    عرض HTML بطريقة متوافقة مع Streamlit الحديث والقديم.
    """

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

    @import url(
        'https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900'
    );

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
        background: linear-gradient(
            135deg,
            #1e3a8a 0%,
            #0f172a 100%
        );
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

    .groq-note {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 14px;
        margin-top: 10px;
    }

    .trade-status-card {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #0f172a 100%
        );
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

    .confidence-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }

    .confidence-title {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    .confidence-value {
        color: #fbbf24;
        font-size: 2rem;
        font-weight: 900;
    }

    .ict-card {
        background: linear-gradient(
            135deg,
            #1e293b 0%,
            #0f172a 100%
        );
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #334155;
        margin-bottom: 14px;
    }

    .ict-title {
        font-size: 0.85rem;
        color: #93c5fd;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 8px;
        text-transform: uppercase;
    }

    .ict-value {
        font-size: 1.8rem;
        font-weight: 900;
        color: #fbbf24;
        line-height: 1.1;
    }

    .ict-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 6px;
    }

    .ict-bullish {
        color: #22c55e !important;
    }

    .ict-bearish {
        color: #ef4444 !important;
    }

    .ict-neutral {
        color: #94a3b8 !important;
    }

    .ict-row {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }

    .ict-badge-yes {
        background: #7f1d1d;
        color: #fecaca;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 700;
    }

    .ict-badge-no {
        background: #14532d;
        color: #bbf7d0;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 700;
    }

</style>
"""
)

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

FEATURES = [
    "atr",
    "ema_50",
    "ema_200",
    "rsi",
]

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
            ai_conf_after_groq REAL
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS active_trade (
            id INTEGER PRIMARY KEY,
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
            signal_bar_time TEXT
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
        "ALTER TABLE active_trade ADD COLUMN features TEXT",
        "ALTER TABLE active_trade ADD COLUMN ai_conf REAL",
        "ALTER TABLE active_trade ADD COLUMN groq_conf REAL",
        "ALTER TABLE active_trade ADD COLUMN groq_note TEXT",
        "ALTER TABLE active_trade ADD COLUMN signal_bar_time TEXT",
    ]

    for stmt in migrations:
        try:
            c.execute(stmt)
            conn.commit()
        except sqlite3.OperationalError:
            pass

    conn.close()

init_db()

# ============================================================
# Settings
# ============================================================

def save_setting(key, val):

    conn = get_db_connection()

    try:
        c = conn.cursor()

        c.execute(
            """
            INSERT OR REPLACE INTO settings
            (key, value)
            VALUES (?, ?)
            """,
            (
                key,
                str(val),
            ),
        )

        conn.commit()

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

        return row[0] if row else default

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

        return int(c.fetchone()[0])

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

        return int(c.fetchone()[0])

    finally:
        conn.close()

# ============================================================
# Secrets
# ============================================================

def get_secret_value(key, default=""):

    try:
        value = st.secrets.get(
            key,
            default,
        )

        if value is None:
            return default

        return str(value)

    except Exception:
        return default

# ============================================================
# Sidebar
# ============================================================

st.sidebar.header(
    "⚙️ إعدادات الذكاء الاصطناعي"
)

# Twelve Data API Key مع حفظ دائم
twelve_secret = get_secret_value(
    "TWELVE_DATA_API_KEY",
    "",
)

# تحميل المفتاح من قاعدة البيانات إذا وجد
saved_twelve_key = load_setting(
    "twelve_api_key",
    "",
)

twelve_key = st.sidebar.text_input(
    "مفتاح Twelve Data API",
    type="password",
    value=(
        twelve_secret
        if twelve_secret
        else saved_twelve_key
        if saved_twelve_key
        else st.session_state.get(
            "twelve_key",
            "",
        )
    ),
)

# حفظ المفتاح في قاعدة البيانات
if twelve_key:
    save_setting(
        "twelve_api_key",
        twelve_key,
    )

st.session_state[
    "twelve_key"
] = twelve_key

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

# ============================================================
# Groq
# ============================================================

st.sidebar.markdown("---")

st.sidebar.header(
    "🧠 الرأي الثاني (Groq)"
)

use_groq = st.sidebar.checkbox(
    "تفعيل مراجعة Groq قبل فتح الصفقة",
    value=(
        load_setting(
            "use_groq",
            load_setting(
                "use_claude",
                "1",
            ),
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

# تحميل مفتاح Groq من قاعدة البيانات
saved_groq_key = load_setting(
    "groq_api_key",
    "",
)

groq_key = st.sidebar.text_input(
    "مفتاح Groq API",
    type="password",
    value=(
        groq_secret
        if groq_secret
        else saved_groq_key
        if saved_groq_key
        else st.session_state.get(
            "groq_key",
            "",
        )
    ),
)

# حفظ مفتاح Groq في قاعدة البيانات
if groq_key:
    save_setting(
        "groq_api_key",
        groq_key,
    )

st.session_state[
    "groq_key"
] = groq_key

groq_model = st.sidebar.text_input(
    "اسم نموذج Groq",
    value=load_setting(
        "groq_model",
        "llama-3.3-70b-versatile",
    ),
)

save_setting(
    "groq_model",
    groq_model,
)

st.sidebar.caption(
    "تأكد من أن اسم النموذج متاح في Groq Console."
)

min_groq_conf = st.sidebar.slider(
    "أدنى ثقة مطلوبة من Groq (%)",
    40,
    95,
    60,
    1,
)

# ============================================================
# Risk
# ============================================================

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

risk_reward = st.sidebar.slider(
    "نسبة العائد R:R",
    1.5,
    4.0,
    2.0,
    0.5,
)

min_conf = st.sidebar.slider(
    "أدنى ثقة مطلوبة من الشبكة العصبية (%)",
    60,
    95,
    75,
    1,
)

# ============================================================
# ICT Settings
# ============================================================

st.sidebar.markdown("---")

st.sidebar.header(
    "🧭 إعدادات لوحة ICT / Smart Money"
)

show_ict_tab = st.sidebar.checkbox(
    "إظهار تبويب ICT / Smart Money",
    value=True,
)

swing_lookback = st.sidebar.slider(
    "حساسية القمم/القيعان",
    2,
    8,
    3,
    1,
)

ob_displacement_mult = st.sidebar.slider(
    "معامل قوة الاندفاع",
    0.8,
    2.5,
    1.2,
    0.1,
)

st.sidebar.caption(
    "لوحة ICT تحليلية للقراءة فقط."
)

# ============================================================
# Reset Model
# ============================================================

st.sidebar.markdown("---")

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
# Ntfy
# ============================================================

def send_alert(
    msg,
    title="🧠 Deep AI Alert",
):

    if not ntfy_channel:
        return

    channel = (
        ntfy_channel
        .strip()
        .split("/")[-1]
    )

    if not channel:
        return

    try:

        requests.post(
            f"https://ntfy.sh/{channel}",
            data=msg.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": "high",
            },
            timeout=5,
        )

    except Exception:
        pass

# ============================================================
# Twelve Data
# ============================================================

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

        response = requests.get(
            "https://api.twelvedata.com/time_series",
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        if "values" not in result:

            st.session_state[
                "last_twelve_error"
            ] = result.get(
                "message",
                "استجابة غير متوقعة من Twelve Data.",
            )

            return pd.DataFrame()

        values = result[
            "values"
        ]

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

            st.session_state[
                "last_twelve_error"
            ] = (
                "بيانات Twelve Data لا تحتوي "
                "الأعمدة المطلوبة."
            )

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

        st.session_state[
            "last_twelve_error"
        ] = None

        return df

    except Exception as exc:

        st.session_state[
            "last_twelve_error"
        ] = (
            f"تعذّر الاتصال بـ Twelve Data: {exc}"
        )

        return pd.DataFrame()

# ============================================================
# Remove Incomplete Candle
# ============================================================

def keep_closed_candles(
    df,
    interval_hours=1,
):

    if (
        df is None
        or df.empty
        or "datetime" not in df.columns
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

    return (
        closed
        .reset_index(drop=True)
    )

# ============================================================
# Training Data
# ============================================================

def fetch_training_data_twelve(
    api_key
):

    return fetch_twelve_series(
        api_key,
        symbol="XAU/USD",
        interval="1h",
        outputsize=TRAINING_OUTPUT_SIZE,
    )

# ============================================================
# Indicators
# ============================================================

def apply_deep_indicators(df):

    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    if len(df) < 210:
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
        tr
        .rolling(14)
        .mean()
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

    delta = df[
        "close"
    ].diff()

    gain = (
        delta.where(
            delta > 0,
            0.0,
        )
        .rolling(14)
        .mean()
    )

    loss = (
        -delta.where(
            delta < 0,
            0.0,
        )
        .rolling(14)
        .mean()
    )

    rs = gain / (
        loss + 1e-6
    )

    df["rsi"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    df.replace(
        [
            np.inf,
            -np.inf,
        ],
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

# ============================================================
# Model Validation
# ============================================================

def model_is_ready(
    model_obj,
    scaler_obj,
):

    if model_obj is None:
        return False

    if scaler_obj is None:
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

        if (
            len(
                model_obj.classes_
            )
            < 2
        ):
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

    except Exception:

        return False

    return True

# ============================================================
# Background Training
# ============================================================

def _background_train_and_save(
    api_key
):

    try:

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
            or len(df_train) < 250
        ):
            return

        future_close = (
            df_train[
                "close"
            ].shift(-1)
        )

        valid_mask = (
            future_close.notna()
        )

        train_df = (
            df_train.loc[
                valid_mask
            ].copy()
        )

        if len(train_df) < 100:
            return

        X = (
            train_df[
                FEATURES
            ]
            .astype(float)
            .values
        )

        y = np.where(
            train_df[
                "close"
            ].shift(-1)
            > train_df[
                "close"
            ],
            1,
            0,
        )

        # إزالة آخر صف لأن هدفه غير موجود
        X = X[:-1]
        y = y[:-1]

        if (
            len(X) < 100
            or len(y) < 100
        ):
            return

        if (
            len(
                np.unique(y)
            )
            < 2
        ):
            return

        new_scaler = (
            StandardScaler()
        )

        X_scaled = (
            new_scaler.fit_transform(
                X
            )
        )

        new_model = (
            MLPClassifier(
                hidden_layer_sizes=(
                    100,
                    50,
                ),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=42,
            )
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

        # حفظ ذري قدر الإمكان
        model_tmp = (
            MODEL_FILE + ".tmp"
        )

        scaler_tmp = (
            SCALER_FILE + ".tmp"
        )

        joblib.dump(
            new_model,
            model_tmp,
        )

        joblib.dump(
            new_scaler,
            scaler_tmp,
        )

        os.replace(
            model_tmp,
            MODEL_FILE,
        )

        os.replace(
            scaler_tmp,
            SCALER_FILE,
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

# ============================================================
# Training Lock
# ============================================================

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

        if age > TRAINING_LOCK_MAX_AGE:

            os.remove(
                TRAINING_LOCK_FILE
            )

    except Exception:
        pass

clean_stale_training_lock()

# ============================================================
# Train / Load Model
# ============================================================

def train_deep_model(
    api_key
):

    if (
        os.path.exists(
            MODEL_FILE
        )
        and os.path.exists(
            SCALER_FILE
        )
    ):

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

    if (
        api_key
        and not os.path.exists(
            TRAINING_LOCK_FILE
        )
    ):

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
                target=(
                    _background_train_and_save
                ),
                args=(api_key,),
                daemon=True,
            )

            thread.start()

        except FileExistsError:

            pass

        except Exception:

            pass

    return (
        None,
        None,
    )

model, scaler = train_deep_model(
    twelve_key
)

# ============================================================
# Experience Layer
# ============================================================

def get_experience_adjustment(
    direction,
    ai_conf,
):

    total = get_total_trades_count()

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
            SELECT
                direction,
                win
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

    # Bayesian smoothing:
    # prior 50%, strength 10
    smoothed_rate = (
        (
            wins
            + 5.0
        )
        /
        (
            n
            + 10.0
        )
        * 100
    )

    # لا نجعل الخبرة تتغلب بالكامل على AI
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

# ============================================================
# Groq Review
# ============================================================

def get_groq_review(
    direction,
    last_row,
    ai_conf,
    api_key,
    model_name,
):

    if not api_key:
        return None

    try:

        prompt = (
            "أنت محلل فني مساعد لصفقة محتملة "
            "على XAU/USD.\n"
            f"الاتجاه المقترح: {direction}.\n"
            f"ثقة نموذج AI الخام: {ai_conf:.1f}%.\n"
            f"ATR={last_row['atr']:.2f}.\n"
            f"EMA50={last_row['ema_50']:.2f}.\n"
            f"EMA200={last_row['ema_200']:.2f}.\n"
            f"RSI={last_row['rsi']:.1f}.\n"
            f"السعر={last_row['close']:.2f}.\n"
            "\n"
            "راجع الاتجاه بشكل مستقل. "
            "لا تفترض أن نموذج AI صحيح. "
            "أعط رأياً محافظاً.\n"
            "\n"
            "يجب أن يكون الرد JSON فقط بهذا الشكل:\n"
            '{"agree": true, '
            '"confidence": 0, '
            '"reason": "..."}'
        )

        response = requests.post(
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
                "max_completion_tokens": 300,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "أنت محلل فني محافظ. "
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

        response.raise_for_status()

        data = response.json()

        choices = data.get(
            "choices",
            [],
        )

        if not choices:
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

        parsed = json.loads(
            cleaned
        )

        agree = bool(
            parsed.get(
                "agree",
                False,
            )
        )

        confidence = float(
            parsed.get(
                "confidence",
                0,
            )
        )

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

        return {
            "agree": agree,
            "confidence": confidence,
            "reason": reason,
        }

    except Exception:

        return None

# ============================================================
# AI Scanner
# ============================================================

def ai_scanner(
    df_live_processed,
    ict_data=None,
):

    result = {
        "trade_exists": False,
        "direction": None,
        "ai_conf_before_groq": 0.0,
        "experience_conf": 0.0,
        "experience_available": False,
        "experience_win_rate": None,
        "experience_sample": 0,
        "groq_called": False,
        "groq_available": False,
        "groq_agree": None,
        "groq_conf": None,
        "groq_reason": "",
        "final_confidence": 0.0,
        "status": "",
        "ict_used": False,
        "ict_bias": None,
        "ict_confidence": None,
        "ict_adjustment": 0.0,
        "ict_signals": [],
    }

    if not twelve_key:

        result["status"] = (
            "النظام متوقف: يرجى إدخال "
            "مفتاح Twelve Data API."
        )

        return (
            result["status"],
            result,
        )

    conn = get_db_connection()

    try:

        df_act = pd.read_sql(
            """
            SELECT *
            FROM active_trade
            WHERE id = 1
            """,
            conn,
        )

    finally:

        conn.close()

    if not df_act.empty:

        active = df_act.iloc[0]

        result[
            "trade_exists"
        ] = True

        result[
            "direction"
        ] = active["direction"]

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
            None,
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

        result[
            "final_confidence"
        ] = (
            result[
                "groq_conf"
            ]
            if result[
                "groq_conf"
            ] is not None
            else result[
                "ai_conf_before_groq"
            ]
        )

        result["status"] = (
            "الذكاء الاصطناعي يدير "
            "صفقة نشطة حالياً."
        )

        return (
            result["status"],
            result,
        )

    if (
        df_live_processed is None
        or df_live_processed.empty
    ):

        result["status"] = (
            "لا توجد صفقة: بيانات السوق "
            "غير كافية."
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
            "الشبكة العصبية قيد "
            "التهيئة والتدريب."
        )

        return (
            result["status"],
            result,
        )

    last = (
        df_live_processed.iloc[-1]
    )

    signal_bar_time = str(
        last.get(
            "datetime",
            "",
        )
    )

    # منع إعادة استخدام نفس الشمعة
    last_signal_key = load_setting(
        "last_signal_key",
        "",
    )

    try:

        feature_values = (
            last[
                FEATURES
            ]
            .astype(float)
            .values
            .reshape(
                1,
                -1,
            )
        )

        if not np.isfinite(
            feature_values
        ).all():

            result["status"] = (
                "لا توجد صفقة: بيانات "
                "المؤشرات غير صالحة."
            )

            return (
                result["status"],
                result,
            )

        x_input = (
            scaler.transform(
                feature_values
            )
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

        pred = int(
            classes[
                best_index
            ]
        )

        ai_conf = float(
            probabilities[
                best_index
            ]
            * 100
        )

    except Exception as exc:

        result["status"] = (
            "تعذر تنفيذ تنبؤ الشبكة "
            f"العصبية: {exc}"
        )

        return (
            result["status"],
            result,
        )

    result[
        "ai_conf_before_groq"
    ] = ai_conf

    if ai_conf < min_conf:

        result["status"] = (
            f"لا توجد صفقة: ثقة AI "
            f"({ai_conf:.1f}%) أقل من "
            f"المطلوب ({min_conf}%)."
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

    # ========================================================
    # Experience Layer
    # ========================================================

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
        experience[
            "confidence"
        ]
        if experience[
            "available"
        ]
        else ai_conf
    )

    # ========================================================
    # 🧭 ICT Layer (قواعد استراتيجية ICT)
    # ========================================================

    ict_adjustment = 0.0
    ict_bias = None
    ict_confidence = None
    ict_used = False
    ict_signals = []

    if ict_data is not None:

        ict_bias = ict_data.get(
            "bias",
            "NEUTRAL",
        )

        ict_confidence = ict_data.get(
            "confidence",
            0,
        )

        ict_used = True

        # تحويل الاتجاه النصي إلى قيمة عددية
        if "BUY" in direction:

            ai_direction = "BULLISH"

        else:

            ai_direction = "BEARISH"

        # ✅ قواعد ICT الأساسية:

        # 1. قاعدة الاتجاه الهيكلي (Bias)
        if ict_bias == ai_direction:
            ict_signals.append(
                f"توافق الهيكل: {ict_bias}"
            )
            ict_adjustment += 5.0  # تعزيز

        elif ict_bias != "NEUTRAL":
            ict_signals.append(
                f"تعارض الهيكل: {ict_bias} مقابل {ai_direction}"
            )
            ict_adjustment -= 10.0  # تخفيض كبير

        # 2. قاعدة السيولة (Liquidity Sweep)
        if ict_data.get("manipulation"):
            manip_note = ict_data.get(
                "manip_note",
                "",
            )
            # إذا كان هناك Sweep عكس الاتجاه، فقد يكون انعكاسًا
            if "BSL swept" in manip_note and ai_direction == "BULLISH":
                # BSL swept يعني احتمال انعكاس هابط
                ict_signals.append(
                    "BSL swept: إشارة انعكاس هابط"
                )
                ict_adjustment -= 8.0
            elif "SSL swept" in manip_note and ai_direction == "BEARISH":
                ict_signals.append(
                    "SSL swept: إشارة انعكاس صاعد"
                )
                ict_adjustment -= 8.0
            else:
                # قد يكون sweep في نفس الاتجاه
                ict_signals.append(
                    f"Liquidity Sweep: {manip_note}"
                )
                ict_adjustment += 3.0  # تعزيز بسيط

        # 3. قاعدة Order Blocks
        bull_ob = ict_data.get("bull_ob")
        bear_ob = ict_data.get("bear_ob")

        if ai_direction == "BULLISH" and bull_ob:
            ict_signals.append(
                "Bullish Order Block موجود"
            )
            ict_adjustment += 4.0
        elif ai_direction == "BEARISH" and bear_ob:
            ict_signals.append(
                "Bearish Order Block موجود"
            )
            ict_adjustment += 4.0
        else:
            ict_signals.append(
                "لا يوجد Order Block داعم"
            )
            ict_adjustment -= 2.0

        # 4. قاعدة FVG
        bull_fvg = ict_data.get("bull_fvg")
        bear_fvg = ict_data.get("bear_fvg")

        if ai_direction == "BULLISH" and bull_fvg:
            ict_signals.append(
                "Bullish FVG موجود"
            )
            ict_adjustment += 3.0
        elif ai_direction == "BEARISH" and bear_fvg:
            ict_signals.append(
                "Bearish FVG موجود"
            )
            ict_adjustment += 3.0
        else:
            ict_signals.append(
                "لا يوجد FVG داعم"
            )
            ict_adjustment -= 1.0

        # 5. قاعدة OTE
        ote = ict_data.get("ote")
        if ote and ote.get("inside") and ote.get("direction") == ai_direction:
            ict_signals.append(
                "السعر داخل منطقة OTE المثالية"
            )
            ict_adjustment += 5.0
        elif ote and ote.get("inside"):
            ict_signals.append(
                "السعر داخل OTE ولكن عكس الاتجاه"
            )
            ict_adjustment -= 3.0

        # 6. قاعدة الجلسة الآسيوية
        asian_levels = ict_data.get("asian_levels")
        if asian_levels:
            if ai_direction == "BULLISH" and last["close"] > asian_levels["asian_high"]:
                ict_signals.append(
                    "السعر فوق قمة الجلسة الآسيوية"
                )
                ict_adjustment += 2.0
            elif ai_direction == "BEARISH" and last["close"] < asian_levels["asian_low"]:
                ict_signals.append(
                    "السعر تحت قاع الجلسة الآسيوية"
                )
                ict_adjustment += 2.0

        # تحديد التعديل النهائي (بحدود معقولة)
        ict_adjustment = max(-15.0, min(15.0, ict_adjustment))

        working_conf += ict_adjustment
        working_conf = max(0.0, min(100.0, working_conf))

        result.update({
            "ict_used": True,
            "ict_bias": ict_bias,
            "ict_confidence": ict_confidence,
            "ict_adjustment": round(ict_adjustment, 2),
            "ict_signals": ict_signals,
        })

        # إذا انخفضت الثقة دون الحد الأدنى بعد تعديل ICT
        if working_conf < min_conf:
            result["final_confidence"] = working_conf
            result["status"] = (
                "🟡 إشارة AI موجودة، لكن قواعد ICT خفّضت الثقة دون الحد الأدنى."
            )
            return result["status"], result

    # ========================================================
    # Groq
    # ========================================================

    groq_result = None

    if use_groq:

        result[
            "groq_called"
        ] = True

        groq_result = (
            get_groq_review(
                direction,
                last,
                working_conf,
                groq_key,
                groq_model,
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
                ] < min_groq_conf
            ):

                result[
                    "final_confidence"
                ] = min(
                    working_conf,
                    groq_result[
                        "confidence"
                    ],
                )

                result["status"] = (
                    "🟡 إشارة AI موجودة، "
                    "لكن Groq لم يعتمد الصفقة."
                )

                return (
                    result["status"],
                    result,
                )

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
            ] = working_conf

            result["status"] = (
                "🟠 AI أعطى إشارة، "
                "لكن Groq لم يستجب. "
                "تم منع الصفقة تحفظاً."
            )

            return (
                result["status"],
                result,
            )

    else:

        result[
            "final_confidence"
        ] = working_conf

    # ========================================================
    # منع تكرار الإشارة
    # ========================================================

    if (
        signal_bar_time
        and last_signal_key
        == signal_bar_time
    ):

        result["status"] = (
            "لا توجد صفقة جديدة: "
            "تمت معالجة هذه الشمعة "
            "سابقاً."
        )

        return (
            result["status"],
            result,
        )

    # ========================================================
    # SL / TP
    # ========================================================

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
            "لا توجد صفقة: قيمة ATR "
            "غير صالحة."
        )

        return (
            result["status"],
            result,
        )

    sl_distance = round(
        atr_value
        * atr_mult,
        2,
    )

    tp_distance = round(
        sl_distance
        * risk_reward,
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

    # ========================================================
    # تخزين الصفقة
    # ========================================================

    conn = get_db_connection()

    try:

        c = conn.cursor()

        c.execute(
            "DELETE FROM active_trade"
        )

        c.execute(
            """
            INSERT INTO active_trade
            (
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
                signal_bar_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
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
            ),
        )

        conn.commit()

    finally:

        conn.close()

    save_setting(
        "last_signal_key",
        signal_bar_time,
    )

    # ========================================================
    # Alert
    # ========================================================

    groq_line = ""

    if result[
        "groq_available"
    ]:

        groq_line = (
            f"\nGroq: "
            f"{result['groq_conf']:.1f}%"
        )

    ict_line = ""

    if ict_used:

        ict_line = (
            f"\nICT: "
            f"{ict_bias} "
            f"({ict_confidence:.1f}%) "
            f"تعديل: "
            f"{ict_adjustment:+.1f}%\n"
            "إشارات:\n"
            + "\n".join(
                f"- {s}"
                for s in ict_signals
            )
        )

    send_alert(
        (
            "🧠 AI Trade Signal\n"
            f"Direction: {direction}\n"
            f"Entry: ${curr}\n"
            f"SL: ${sl_price}\n"
            f"TP: ${tp_price}\n"
            f"AI Raw: {ai_conf:.1f}%\n"
            f"Experience: "
            f"{working_conf:.1f}%"
            f"{ict_line}"
            f"{groq_line}\n"
            f"Final Confidence: "
            f"{result['final_confidence']:.1f}%"
        )
    )

    result[
        "trade_exists"
    ] = True

    result[
        "status"
    ] = (
        f"🟢 تم إطلاق الإشارة "
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
# ICT — Swing Points
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

# ============================================================
# ICT — Market Structure
# ============================================================

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
            for i, p
            in swing_highs
        ]
        +
        [
            (
                i,
                "low",
                p,
            )
            for i, p
            in swing_lows
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
                    if trend
                    == "bullish"
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
                    if trend
                    == "bearish"
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
        structure_breaks[-1][
            "direction"
        ]
        if structure_breaks
        else "NEUTRAL"
    )

    return (
        current_bias,
        structure_breaks[-8:],
    )

# ============================================================
# ICT — Order Blocks
# ============================================================

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
        df.iloc[
            -lookback:
        ]
        .reset_index(
            drop=True
        )
    )

    bullish_ob = None
    bearish_ob = None

    for i in range(
        1,
        len(recent),
    ):

        atr_value = float(
            recent[
                "atr"
            ].iloc[i]
        )

        if (
            not np.isfinite(
                atr_value
            )
            or atr_value <= 0
        ):
            continue

        body = (
            recent[
                "close"
            ].iloc[i]
            -
            recent[
                "open"
            ].iloc[i]
        )

        is_displacement = (
            abs(body)
            >
            displacement_atr_mult
            * atr_value
        )

        previous = (
            recent.iloc[
                i - 1
            ]
        )

        if (
            is_displacement
            and body > 0
            and previous[
                "close"
            ]
            <
            previous[
                "open"
            ]
        ):

            bullish_ob = {
                "top": round(
                    float(
                        previous[
                            "open"
                        ]
                    ),
                    2,
                ),
                "bottom": round(
                    float(
                        previous[
                            "low"
                        ]
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
            and previous[
                "close"
            ]
            >
            previous[
                "open"
            ]
        ):

            bearish_ob = {
                "top": round(
                    float(
                        previous[
                            "high"
                        ]
                    ),
                    2,
                ),
                "bottom": round(
                    float(
                        previous[
                            "open"
                        ]
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

# ============================================================
# ICT — FVG
# ============================================================

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
        df.iloc[
            -lookback:
        ]
        .reset_index(
            drop=True
        )
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

        c3 = recent.iloc[
            i
        ]

        if (
            c1["high"]
            <
            c3["low"]
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
            >
            c3["high"]
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

# ============================================================
# ICT — Liquidity
# ============================================================

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
        for _, p
        in recent_highs
    )

    ssl_price = min(
        p
        for _, p
        in recent_lows
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

# ============================================================
# ICT — Session
# ============================================================

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
        window[
            "high"
        ].max()
    )

    session_low = float(
        window[
            "low"
        ].min()
    )

    session_range = (
        session_high
        - session_low
    )

    net_change = (
        float(
            window[
                "close"
            ].iloc[-1]
        )
        -
        float(
            window[
                "open"
            ].iloc[0]
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
            /
            (
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

# ============================================================
# Asian Session
# ============================================================

def compute_asian_session_levels(
    df
):

    if (
        df is None
        or df.empty
        or len(df) < 24
    ):

        return None

    window = df.iloc[
        -24:
    ]

    return {
        "asian_high": round(
            float(
                window[
                    "high"
                ].max()
            ),
            2,
        ),
        "asian_low": round(
            float(
                window[
                    "low"
                ].min()
            ),
            2,
        ),
    }

# ============================================================
# Fibonacci
# ============================================================

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
                * (
                    ratio
                    - 1
                ),
                2,
            )

        else:

            levels[label] = round(
                swing_low
                - diff
                * (
                    ratio
                    - 1
                ),
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

# ============================================================
# OTE
# ============================================================

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
            - diff
            * 0.618
        )

        bottom = (
            swing_high
            - diff
            * 0.79
        )

    else:

        top = (
            swing_low
            + diff
            * 0.79
        )

        bottom = (
            swing_low
            + diff
            * 0.618
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

# ============================================================
# Volatility
# ============================================================

def compute_volatility_risk(
    df
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

        label = "CRISIS"
        score = 90

    elif percentile >= 65:

        label = "HIGH"
        score = 70

    elif percentile >= 35:

        label = "MEDIUM"
        score = 50

    else:

        label = "LOW"
        score = 25

    return (
        label,
        score,
        round(
            current_atr,
            2,
        ),
    )

# ============================================================
# Displacement
# ============================================================

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
            pd.isna(
                atr_value
            )
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
            /
            (
                candle_range
                + 1e-6
            )
            * 100,
            1,
        )

        if (
            abs(body)
            >
            atr_mult
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

# ============================================================
# ICT Engine
# ============================================================

def run_ict_engine(
    df_processed,
    swing_lookback=3,
    ob_mult=1.2,
):

    if (
        df_processed is None
        or df_processed.empty
        or len(df_processed)
        <
        max(
            30,
            swing_lookback * 6,
        )
    ):

        return None

    swing_highs, swing_lows = (
        find_swing_points(
            df_processed,
            lookback=swing_lookback,
        )
    )

    bias, structure_breaks = (
        analyze_market_structure(
            swing_highs,
            swing_lows,
        )
    )

    bull_ob, bear_ob = (
        detect_order_blocks(
            df_processed,
            displacement_atr_mult=ob_mult,
        )
    )

    bull_fvg, bear_fvg = (
        detect_fvg(
            df_processed
        )
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

    session_info = (
        session_analyzer(
            df_processed
        )
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

        last_low_idx, last_low_price = (
            swing_lows[-1]
        )

        last_high_idx, last_high_price = (
            swing_highs[-1]
        )

        if (
            last_low_idx
            > last_high_idx
        ):

            leg_direction = (
                "BULLISH"
            )

        else:

            leg_direction = (
                "BEARISH"
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

        ote = (
            compute_ote_zone(
                lo_price,
                hi_price,
                leg_direction,
                current_price,
            )
        )

    # ========================================================
    # Score Breakdown
    # ========================================================

    matching_breaks = [
        brk
        for brk
        in structure_breaks
        if brk[
            "direction"
        ]
        == bias
    ]

    if bias != "NEUTRAL":

        structure_score = min(
            100,
            25
            + len(
                matching_breaks
            )
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
            bull_ob[
                "strength"
            ]
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
            bear_ob[
                "strength"
            ]
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
# UI Header
# ============================================================

st.title(
    "🧠 نظام التداول العميق — XAU/USD"
)

success_count = (
    get_successful_trades_count()
)

total_count = (
    get_total_trades_count()
)

ai_level = max(
    1,
    int(
        success_count
        * 1.5
    ),
)

render_html(
    f"""
<div class="ai-level-card">

    <div class="ai-level-title">
        AI EVOLUTION LEVEL
    </div>

    <div class="ai-level-value">
        Lvl. {ai_level}
    </div>

    <div class="ai-level-sub">
        Successful Trades:
        {success_count}
        |
        Total Trades:
        {total_count}
        |
        Deep Neural Network Active
    </div>

</div>
"""
)

# ============================================================
# Live Data
# ============================================================

df_live_raw = (
    fetch_twelve_series(
        twelve_key,
        symbol="XAU/USD",
        interval="1h",
        outputsize=LIVE_OUTPUT_SIZE,
    )
    if twelve_key
    else pd.DataFrame()
)

# مهم:
# نستخدم فقط الشموع المغلقة
df_live_closed = (
    keep_closed_candles(
        df_live_raw,
        interval_hours=1,
    )
)

df_live_processed = (
    apply_deep_indicators(
        df_live_closed
    )
)

# ============================================================
# ICT
# ============================================================

ict_data = (
    run_ict_engine(
        df_live_processed,
        swing_lookback=swing_lookback,
        ob_mult=ob_displacement_mult,
    )
    if not df_live_processed.empty
    else None
)

# ============================================================
# AI Scan
# ============================================================

scan_msg, ai_result = (
    ai_scanner(
        df_live_processed,
        ict_data=ict_data,
    )
)

# ============================================================
# Tabs
# ============================================================

if show_ict_tab:

    tab1, tab2, tab3 = st.tabs(
        [
            "⚡ حالة الذكاء الاصطناعي",
            "📊 سجل الخبرات المكتسبة",
            "🧭 لوحة ICT / Smart Money",
        ]
    )

else:

    tab1, tab2 = st.tabs(
        [
            "⚡ حالة الذكاء الاصطناعي",
            "📊 سجل الخبرات المكتسبة",
        ]
    )

    tab3 = None

# ============================================================
# TAB 1
# ============================================================

with tab1:

    if not twelve_key:

        st.warning(
            "⚠️ النظام نائم: أدخل مفتاح Twelve Data API."
        )

    if os.path.exists(
        TRAINING_LOCK_FILE
    ):

        st.info(
            "🧠 النموذج يتدرب حالياً "
            "في الخلفية على البيانات التاريخية."
        )

    st.markdown(
        "### 🎯 حالة الصفقة"
    )

    trade_exists = bool(
        ai_result.get(
            "trade_exists",
            False,
        )
    )

    if trade_exists:

        direction = ai_result.get(
            "direction"
        )

        if (
            direction
            and "BUY"
            in str(direction)
        ):

            status_class = (
                "trade-buy"
            )

            status_text = (
                "🟢 توجد صفقة"
            )

        elif (
            direction
            and "SELL"
            in str(direction)
        ):

            status_class = (
                "trade-sell"
            )

            status_text = (
                "🔴 توجد صفقة"
            )

        else:

            status_class = (
                "trade-neutral"
            )

            status_text = (
                "🟡 توجد إشارة"
            )

    else:

        status_class = (
            "trade-neutral"
        )

        status_text = (
            "⚪ لا توجد صفقة"
        )

    render_html(
        f"""
<div class="trade-status-card">

    <div class="trade-status-title">
        TRADE STATUS
    </div>

    <div class="trade-status-value {status_class}">
        {status_text}
    </div>

</div>
"""
    )

    # ========================================================
    # Confidence
    # ========================================================

    st.markdown(
        "### 🧠 مراحل الثقة"
    )

    c1, c2, c3 = st.columns(3)

    ai_before = float(
        ai_result.get(
            "ai_conf_before_groq",
            0,
        )
        or 0
    )

    groq_conf = ai_result.get(
        "groq_conf"
    )

    final_conf = float(
        ai_result.get(
            "final_confidence",
            ai_before,
        )
        or 0
    )

    render_html(
        f"""
<div class="confidence-card">

    <div class="confidence-title">
        AI BEFORE GROQ
    </div>

    <div class="confidence-value">
        {ai_before:.1f}%
    </div>

</div>
"""
    )

    if groq_conf is None:

        groq_display = "—"

    else:

        groq_display = (
            f"{float(groq_conf):.1f}%"
        )

    c2.markdown(
        f"""
<div class="confidence-card">

    <div class="confidence-title">
        GROQ REVIEW
    </div>

    <div class="confidence-value">
        {groq_display}
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    render_html(
        f"""
<div class="confidence-card">

    <div class="confidence-title">
        FINAL CONFIDENCE
    </div>

    <div class="confidence-value">
        {final_conf:.1f}%
    </div>

</div>
"""
    )

    # ========================================================
    # Experience
    # ========================================================

    if ai_result.get(
        "experience_available",
        False,
    ):

        st.info(
            "📚 Experience Layer: "
            f"{ai_result['experience_win_rate']:.1f}% "
            "historical smoothed win-rate "
            f"on {ai_result['experience_sample']} "
            "similar-direction trades."
        )

    # ========================================================
    # Direction
    # ========================================================

    if ai_result.get(
        "direction"
    ):

        st.info(
            f"📌 الاتجاه المقترح: "
            f"**{ai_result['direction']}**"
        )

    # ========================================================
    # Groq
    # ========================================================

    if ai_result.get(
        "groq_called",
        False,
    ):

        if ai_result.get(
            "groq_available",
            False,
        ):

            if ai_result.get(
                "groq_agree"
            ):

                st.success(
                    "✅ Groq وافق على الإشارة."
                )

            else:

                st.error(
                    "❌ Groq لم يوافق على الإشارة."
                )

            if ai_result.get(
                "groq_reason"
            ):

                render_html(
                    f"""
<div class="groq-note">

<b>🧠 رأي Groq:</b>

{ai_result["groq_reason"]}

</div>
"""
                )

        else:

            st.warning(
                "🟠 تم طلب مراجعة Groq "
                "لكن لم تصل استجابة صحيحة."
            )

    # ========================================================
    # ICT Signals
    # ========================================================

    if ai_result.get(
        "ict_used",
        False,
    ):

        st.markdown(
            "### 🧭 إشارات ICT"
        )

        for signal in ai_result.get(
            "ict_signals",
            [],
        ):
            st.write(f"- {signal}")

        ict_adj = ai_result.get(
            "ict_adjustment",
            0,
        )

        if ict_adj > 0:

            st.success(
                f"تعديل ICT: +{ict_adj:.1f}%"
            )

        elif ict_adj < 0:

            st.warning(
                f"تعديل ICT: {ict_adj:.1f}%"
            )

        else:

            st.info(
                "تعديل ICT: 0.0%"
            )

    # ========================================================
    # Status
    # ========================================================

    if scan_msg:

        st.info(
            f"🔍 {scan_msg}"
        )

    # ========================================================
    # Twelve Error
    # ========================================================

    twelve_error = (
        st.session_state.get(
            "last_twelve_error"
        )
    )

    if (
        twelve_error
        and twelve_key
    ):

        st.error(
            f"⚠️ Twelve Data: "
            f"{twelve_error}"
        )

    # ========================================================
    # Active Trade
    # ========================================================

    conn = get_db_connection()

    try:

        df_active = pd.read_sql(
            """
            SELECT *
            FROM active_trade
            WHERE id = 1
            """,
            conn,
        )

    finally:

        conn.close()

    if (
        not df_active.empty
        and twelve_key
    ):

        active_trade = (
            df_active.iloc[0]
        )

        st.warning(
            f"""
🔒 **صفقة نشطة حالياً**

الاتجاه:
{active_trade['direction']}

الدخول:
${active_trade['entry']}

SL:
${active_trade['sl']}

TP:
${active_trade['tp']}

شمعة الإشارة:
{active_trade.get('signal_bar_time', '')}
"""
        )

    # ========================================================
    # Snapshot
    # ========================================================

    if not df_live_processed.empty:

        last_snapshot = (
            df_live_processed.iloc[-1]
        )

        s1, s2, s3, s4, s5 = (
            st.columns(5)
        )

        s1.metric(
            "السعر",
            f"{last_snapshot['close']:.2f}",
        )

        s2.metric(
            "RSI",
            f"{last_snapshot['rsi']:.1f}",
        )

        s3.metric(
            "EMA50",
            f"{last_snapshot['ema_50']:.2f}",
        )

        s4.metric(
            "EMA200",
            f"{last_snapshot['ema_200']:.2f}",
        )

        s5.metric(
            "ATR",
            f"{last_snapshot['atr']:.2f}",
        )

        if "datetime" in last_snapshot:

            st.caption(
                "آخر شمعة مغلقة: "
                f"{last_snapshot['datetime']}"
            )

# ============================================================
# TAB 2 — History
# ============================================================

with tab2:

    conn = get_db_connection()

    try:

        df_log = pd.read_sql(
            """
            SELECT *
            FROM trades
            ORDER BY id DESC
            """,
            conn,
        )

    finally:

        conn.close()

    if not df_log.empty:

        win_rate = (
            df_log[
                "win"
            ].sum()
            /
            len(df_log)
            * 100
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "إجمالي الصفقات",
            len(df_log),
        )

        m2.metric(
            "نسبة الربح",
            f"{win_rate:.1f}%",
        )

        m3.metric(
            "الصفقات الرابحة",
            int(
                df_log[
                    "win"
                ].sum()
            ),
        )

        st.dataframe(
            df_log,
            use_container_width=True,
        )

    else:

        st.info(
            "لا توجد صفقات مغلقة مسجلة حتى الآن."
        )

# ============================================================
# TAB 3 — ICT
# ============================================================

if tab3 is not None:

    with tab3:

        if not twelve_key:

            st.warning(
                "⚠️ أدخل مفتاح Twelve Data API."
            )

        elif ict_data is None:

            st.info(
                "بيانات غير كافية لعرض تحليل ICT."
            )

        else:

            st.caption(
                "🧭 لوحة تحليلية للقراءة فقط "
                "(Read-Only)."
            )

            sub1, sub2, sub3, sub4, sub5 = (
                st.tabs(
                    [
                        "📉 Volatility & Risk",
                        "🏗️ Market Structure",
                        "🏦 Smart Money Concepts",
                        "📐 Fibonacci & OTE",
                        "🎯 Trade Setup / Signals",
                    ]
                )
            )

            # =================================================
            # Volatility
            # =================================================

            with sub1:

                risk_label = (
                    "HIGH"
                    if ict_data[
                        "vol_score"
                    ] >= 70
                    else
                    "MEDIUM"
                    if ict_data[
                        "vol_score"
                    ] >= 50
                    else
                    "LOW"
                )

                v1, v2 = (
                    st.columns(2)
                )

                render_html(
                    f"""
<div class="ict-card">

    <div class="ict-title">
        Volatility
    </div>

    <div class="ict-value">
        {ict_data['vol_label']}
    </div>

    <div class="ict-sub">
        ATR: {ict_data['atr']}
    </div>

</div>
"""
                )

                render_html(
                    f"""
<div class="ict-card">

    <div class="ict-title">
        Risk Level
    </div>

    <div class="ict-value">
        {risk_label}
    </div>

    <div class="ict-sub">
        Score: {ict_data['vol_score']}
    </div>

</div>
"""
                )

                if ict_data[
                    "session"
                ]:

                    st.markdown(
                        "#### 🕒 Session Analyzer"
                    )

                    session = (
                        ict_data[
                            "session"
                        ]
                    )

                    if (
                        session["bias"]
                        == "BULLISH"
                    ):

                        bias_class = (
                            "ict-bullish"
                        )

                    elif (
                        session["bias"]
                        == "BEARISH"
                    ):

                        bias_class = (
                            "ict-bearish"
                        )

                    else:

                        bias_class = (
                            "ict-neutral"
                        )

                    sc1, sc2, sc3, sc4 = (
                        st.columns(4)
                    )

                    render_html(
                        f"""
<div class="ict-row">

<b>BIAS</b>

<br>

<span class="{bias_class}">
{session['bias']}
</span>

</div>
"""
                    )

                    render_html(
                        f"""
<div class="ict-row">

<b>MAIN SCORE</b>

<br>

{session['main_score']}

</div>
"""
                    )

                    render_html(
                        f"""
<div class="ict-row">

<b>SESSION RANGE</b>

<br>

{session['range']}

</div>
"""
                    )

                    render_html(
                        f"""
<div class="ict-row">

<b>SESSION HIGH/LOW</b>

<br>

{session['high']}
/
{session['low']}

</div>
"""
                    )

                if ict_data[
                    "asian_levels"
                ]:

                    st.markdown(
                        "#### 🌏 Asian Session"
                    )

                    ah, al = (
                        st.columns(2)
                    )

                    ah.metric(
                        "Asian High",
                        ict_data[
                            "asian_levels"
                        ][
                            "asian_high"
                        ],
                    )

                    al.metric(
                        "Asian Low",
                        ict_data[
                            "asian_levels"
                        ][
                            "asian_low"
                        ],
                    )

            # =================================================
            # Market Structure
            # =================================================

            with sub2:

                if (
                    ict_data[
                        "bias"
                    ]
                    == "BULLISH"
                ):

                    bias_class = (
                        "ict-bullish"
                    )

                elif (
                    ict_data[
                        "bias"
                    ]
                    == "BEARISH"
                ):

                    bias_class = (
                        "ict-bearish"
                    )

                else:

                    bias_class = (
                        "ict-neutral"
                    )

                render_html(
                    f"""
<h3>
الاتجاه الهيكلي الحالي:
</h3>

<span class="{bias_class}">
{ict_data['bias']}
</span>
"""
                )

                st.markdown(
                    "#### 📊 Score Breakdown"
                )

                b1, b2, b3, b4 = (
                    st.columns(4)
                )

                b1.metric(
                    "Structure Score",
                    ict_data[
                        "scores"
                    ][
                        "structure"
                    ],
                )

                b2.metric(
                    "Liquidity Score",
                    ict_data[
                        "scores"
                    ][
                        "liquidity"
                    ],
                )

                b3.metric(
                    "Order Block Score",
                    ict_data[
                        "scores"
                    ][
                        "order_block"
                    ],
                )

                b4.metric(
                    "FVG Score",
                    ict_data[
                        "scores"
                    ][
                        "fvg"
                    ],
                )

                st.markdown(
                    "#### 🧱 Structure Breaks"
                )

                if ict_data[
                    "structure_breaks"
                ]:

                    for brk in reversed(
                        ict_data[
                            "structure_breaks"
                        ]
                    ):

                        css = (
                            "ict-bullish"
                            if brk[
                                "direction"
                            ]
                            == "BULLISH"
                            else
                            "ict-bearish"
                        )

                        render_html(
                            f"""
<div class="ict-row">

<b>{brk['type']}</b>

—

<span class="{css}">
{brk['direction']}
</span>

@

{brk['price']}

</div>
"""
                        )

                else:

                    st.caption(
                        "لا توجد كسور هيكل واضحة."
                    )

                if ict_data[
                    "displacements"
                ]:

                    st.markdown(
                        "#### ⚡ Recent Displacements"
                    )

                    for displacement in (
                        ict_data[
                            "displacements"
                        ]
                    ):

                        css = (
                            "ict-bullish"
                            if displacement[
                                "bias"
                            ]
                            == "BULLISH"
                            else
                            "ict-bearish"
                        )

                        render_html(
                            f"""
<div class="ict-row">

<span class="{css}">
{displacement['bias']}
</span>

—

Body:
{displacement['body_pct']}%

</div>
"""
                        )

            # =================================================
            # Smart Money
            # =================================================

            with sub3:

                st.markdown(
                    "#### 📦 Order Blocks"
                )

                oc1, oc2 = (
                    st.columns(2)
                )

                if ict_data[
                    "bull_ob"
                ]:

                    ob = (
                        ict_data[
                            "bull_ob"
                        ]
                    )

                    oc1.success(
                        f"""
Bullish OB

Top:
{ob['top']}

Bottom:
{ob['bottom']}

Strength:
{ob['strength']}%
"""
                    )

                else:

                    oc1.caption(
                        "لا يوجد Bullish Order Block حديث."
                    )

                if ict_data[
                    "bear_ob"
                ]:

                    ob = (
                        ict_data[
                            "bear_ob"
                        ]
                    )

                    oc2.error(
                        f"""
Bearish OB

Top:
{ob['top']}

Bottom:
{ob['bottom']}

Strength:
{ob['strength']}%
"""
                    )

                else:

                    oc2.caption(
                        "لا يوجد Bearish Order Block حديث."
                    )

                st.markdown(
                    "#### 🌀 Fair Value Gaps"
                )

                fc1, fc2 = (
                    st.columns(2)
                )

                if ict_data[
                    "bull_fvg"
                ]:

                    fvg = (
                        ict_data[
                            "bull_fvg"
                        ]
                    )

                    fc1.success(
                        "Bullish FVG: "
                        f"{fvg['bottom']} "
                        "→ "
                        f"{fvg['top']}"
                    )

                else:

                    fc1.caption(
                        "لا توجد فجوة سعرية صاعدة حديثة."
                    )

                if ict_data[
                    "bear_fvg"
                ]:

                    fvg = (
                        ict_data[
                            "bear_fvg"
                        ]
                    )

                    fc2.error(
                        "Bearish FVG: "
                        f"{fvg['bottom']} "
                        "→ "
                        f"{fvg['top']}"
                    )

                else:

                    fc2.caption(
                        "لا توجد فجوة سعرية هابطة حديثة."
                    )

                st.markdown(
                    "#### 💧 Liquidity & Manipulation"
                )

                lc1, lc2 = (
                    st.columns(2)
                )

                lc1.metric(
                    "BSL",
                    (
                        ict_data["bsl"]
                        if ict_data[
                            "bsl"
                        ] is not None
                        else "N/A"
                    ),
                )

                lc2.metric(
                    "SSL",
                    (
                        ict_data["ssl"]
                        if ict_data[
                            "ssl"
                        ] is not None
                        else "N/A"
                    ),
                )

                if ict_data[
                    "manipulation"
                ]:

                    render_html(
                        f"""
<span class="ict-badge-yes">
MANIPULATION: YES
</span>

&nbsp;

{ict_data['manip_note']}
"""
                    )

                else:

                    render_html(
                        """
<span class="ict-badge-no">
MANIPULATION: NO
</span>

&nbsp;

لا يوجد اصطياد سيولة واضح حالياً.
"""
                    )

            # =================================================
            # Fibonacci / OTE
            # =================================================

            with sub4:

                if ict_data[
                    "ote"
                ]:

                    ote = (
                        ict_data[
                            "ote"
                        ]
                    )

                    st.markdown(
                        f"""
#### 🎯 Optimal Trade Entry (OTE)

الاتجاه:
**{ote['direction']}**
"""
                    )

                    st.write(
                        "المنطقة: "
                        f"**{ote['bottom']} "
                        f"→ "
                        f"{ote['top']}**"
                    )

                    if ote[
                        "inside"
                    ]:

                        st.success(
                            "✅ السعر الحالي داخل منطقة OTE."
                        )

                    else:

                        st.info(
                            "⚪ السعر الحالي خارج منطقة OTE."
                        )

                if ict_data[
                    "fib_levels"
                ]:

                    st.markdown(
                        "#### 📐 Fibonacci Extensions"
                    )

                    for (
                        label,
                        value,
                    ) in ict_data[
                        "fib_levels"
                    ].items():

                        render_html(
                            f"""
<div class="ict-row">

<b>{label}</b>:

{value}

</div>
"""
                        )

            # =================================================
            # Trade Setup
            # =================================================

            with sub5:

                bias_txt = (
                    ict_data[
                        "bias"
                    ]
                )

                if (
                    bias_txt
                    == "BULLISH"
                ):

                    color_class = (
                        "ict-bullish"
                    )

                elif (
                    bias_txt
                    == "BEARISH"
                ):

                    color_class = (
                        "ict-bearish"
                    )

                else:

                    color_class = (
                        "ict-neutral"
                    )

                render_html(
                    f"""
<div class="ict-card">

    <div class="ict-title">
        Bias
    </div>

    <div class="ict-value {color_class}">
        {bias_txt}
    </div>

    <div class="ict-sub">
        Confidence:
        {ict_data['confidence']}%
    </div>

</div>
"""
                )

                st.markdown(
                    "#### 📡 Signals"
                )

                signal_count = 0

                if ict_data[
                    "manipulation"
                ]:

                    st.write(
                        "- "
                        + ict_data[
                            "manip_note"
                        ]
                    )

                    signal_count += 1

                if (
                    ict_data["ote"]
                    and ict_data[
                        "ote"
                    ]["inside"]
                ):

                    st.write(
                        "- السعر داخل منطقة OTE "
                        "باتجاه "
                        f"{ict_data['ote']['direction']}"
                    )

                    signal_count += 1

                if (
                    bias_txt
                    == "BULLISH"
                    and ict_data[
                        "bull_ob"
                    ]
                ):

                    st.write(
                        "- Order Block صاعد "
                        "يدعم الاتجاه الهيكلي."
                    )

                    signal_count += 1

                if (
                    bias_txt
                    == "BEARISH"
                    and ict_data[
                        "bear_ob"
                    ]
                ):

                    st.write(
                        "- Order Block هابط "
                        "يدعم الاتجاه الهيكلي."
                    )

                    signal_count += 1

                if signal_count == 0:

                    st.caption(
                        "لا توجد إشارات إضافية بارزة حالياً."
                    )

                st.caption(
                    "هذا القسم تحليلي بالكامل."
                )

# ============================================================
# Automatic Monitoring
# ============================================================

st_autorefresh(
    interval=60000,
    key="deep_ai_loop",
)

# ============================================================
# Active Trade Monitoring
# ============================================================

if twelve_key:

    conn = get_db_connection()

    try:

        active_df = pd.read_sql(
            """
            SELECT *
            FROM active_trade
            WHERE id = 1
            """,
            conn,
        )

    finally:

        conn.close()

    if (
        not active_df.empty
        and not df_live_processed.empty
    ):

        trade_row = (
            active_df.iloc[0]
        )

        last_row = (
            df_live_processed.iloc[-1]
        )

        is_buy_trade = (
            "BUY"
            in str(
                trade_row[
                    "direction"
                ]
            )
        )

        signal_bar_time = str(
            trade_row.get(
                "signal_bar_time",
                "",
            )
            or ""
        )

        current_bar_time = str(
            last_row.get(
                "datetime",
                "",
            )
        )

        # ====================================================
        # لا نغلق الصفقة على نفس شمعة الإشارة
        # ====================================================

        can_monitor_trade = (
            current_bar_time
            != signal_bar_time
        )

        # ====================================================
        # AI Reversal Detection
        # ====================================================

        if (
            can_monitor_trade
            and model_is_ready(
                model,
                scaler,
            )
        ):

            try:

                x_current = (
                    scaler.transform(
                        last_row[
                            FEATURES
                        ]
                        .astype(float)
                        .values
                        .reshape(
                            1,
                            -1,
                        )
                    )
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
                        min_conf - 5
                    )
                ):

                    reversal_detected = True

                elif (
                    not is_buy_trade
                    and current_pred == 1
                    and current_conf
                    >= (
                        min_conf - 5
                    )
                ):

                    reversal_detected = True

                if reversal_detected:

                    send_alert(
                        (
                            "⚠️ تنبيه من الشبكة "
                            "العصبية: "
                            "رصد انعكاس محتمل "
                            f"ضد الصفقة "
                            f"({trade_row['direction']}) "
                            f"بقوة "
                            f"({current_conf:.1f}%)."
                        ),
                        "🚨 AI Reversal Warning",
                    )

            except Exception:

                pass

        # ====================================================
        # SL / TP
        # ====================================================

        if can_monitor_trade:

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

            # =================================================
            # إذا ضرب SL وTP في نفس الشمعة
            # نستخدم القرار المحافظ: SL
            # =================================================

            both_hit = (
                hit_sl
                and hit_tp
            )

            if both_hit:

                hit_tp = False
                hit_sl = True

            # =================================================
            # Close Trade
            # =================================================

            if hit_sl or hit_tp:

                win_value = (
                    1
                    if hit_tp
                    else 0
                )

                if both_hit:

                    note_str = (
                        "SL and TP touched "
                        "in same candle; "
                        "conservative SL outcome."
                    )

                else:

                    note_str = (
                        "AI Target Reached "
                        "(تم التعلم من النتيجة)"
                        if hit_tp
                        else
                        "AI Stop Loss Hit "
                        "(تم تسجيل النتيجة)"
                    )

                # =================================================
                # Experience
                # =================================================
                #
                # لا نستخدم partial_fit على نموذج الاتجاه
                # لأن النموذج الأساسي يتعلم:
                #
                # next candle direction
                #
                # وليس:
                #
                # win/loss
                #
                # لذلك يتم تسجيل النتيجة في SQLite
                # وتستخدم لاحقاً في Experience Layer.
                # =================================================

                # =================================================
                # Store Closed Trade
                # =================================================

                conn = get_db_connection()

                try:

                    c = conn.cursor()

                    c.execute(
                        """
                        INSERT INTO trades
                        (
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
                            ai_conf_after_groq
                        )
                        VALUES
                        (
                            ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?
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

                            float(
                                trade_row.get(
                                    "groq_conf"
                                )
                                if pd.notna(
                                    trade_row.get(
                                        "groq_conf"
                                    )
                                )
                                else
                                trade_row.get(
                                    "ai_conf",
                                    0,
                                )
                                or 0
                            ),
                        ),
                    )

                    c.execute(
                        """
                        DELETE FROM active_trade
                        WHERE id = 1
                        """
                    )

                    conn.commit()

                finally:

                    conn.close()

                # =================================================
                # Notification
                # =================================================

                send_alert(
                    (
                        f"Closed "
                        f"{trade_row['symbol']} "
                        f"{trade_row['direction']} "
                        "-> "
                        f"{note_str}"
                    ),
                    "🧠 AI Trade Settled",
                )
