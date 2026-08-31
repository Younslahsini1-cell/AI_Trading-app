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
    page_title="XAU/USD Deep AI Engine Pro",
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
            st.markdown(html_content, unsafe_allow_html=True)
    except Exception:
        st.markdown(html_content, unsafe_allow_html=True)

# ============================================================
# CSS
# ============================================================

render_html(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900&display=swap');

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

    .trade-buy { color: #22c55e; }
    .trade-sell { color: #ef4444; }
    .trade-neutral { color: #94a3b8; }

    .confidence-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }

    .confidence-value {
        color: #fbbf24;
        font-size: 2rem;
        font-weight: 900;
    }

    .ict-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
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
    }
</style>
"""
)

# ============================================================
# الملفات والثوابت والميزات المحدثة (FEATURES)
# ============================================================

DB_FILE = "xau_deep_ai.db"
MODEL_FILE = "xau_deep_mlp_v3.pkl"
SCALER_FILE = "xau_deep_scaler_v3.pkl"
TRAINING_LOCK_FILE = "training.lock"

TRAINING_OUTPUT_SIZE = 5000
TRAINING_LOCK_MAX_AGE = 60 * 60

# تم توسيع شريط الميزات ليشمل المؤشرات الجديدة والدعوم والمقاومات
FEATURES = [
    "atr",
    "ema_50",
    "ema_200",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "bb_pct",
    "stoch_k",
    "stoch_d",
    "dist_to_sup",
    "dist_to_res",
    "pivot_pp",
]

# ============================================================
# Database
# ============================================================

def get_db_connection():
    return sqlite3.connect(DB_FILE, timeout=20, check_same_thread=False)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    c.execute("""
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
            groq_conf REAL,
            groq_note TEXT,
            ai_conf_before_groq REAL,
            ai_conf_after_groq REAL
        )
    """)
    c.execute("""
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
    """)
    conn.commit()
    conn.close()

init_db()

# ============================================================
# Settings Helpers
# ============================================================

def save_setting(key, val):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(val)))
        conn.commit()
    finally:
        conn.close()

def load_setting(key, default=""):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = c.fetchone()
        return row[0] if row else default
    finally:
        conn.close()

def get_secret_value(key, default=""):
    try:
        val = st.secrets.get(key, default)
        return str(val) if val is not None else default
    except Exception:
        return default

# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("⚙️ إعدادات الذكاء الاصطناعي")

twelve_secret = get_secret_value("TWELVE_DATA_API_KEY", "")
saved_twelve_key = load_setting("twelve_api_key", "")
twelve_key = st.sidebar.text_input(
    "مفتاح Twelve Data API",
    type="password",
    value=twelve_secret or saved_twelve_key or st.session_state.get("twelve_key", ""),
)
if twelve_key:
    save_setting("twelve_api_key", twelve_key)
st.session_state["twelve_key"] = twelve_key

ntfy_channel = st.sidebar.text_input(
    "قناة Ntfy للتنبيهات",
    value=load_setting("ntfy", "xau_deep_channel"),
)
save_setting("ntfy", ntfy_channel)

st.sidebar.markdown("---")
st.sidebar.header("🧠 الرأي الثاني (Groq)")

use_groq = st.sidebar.checkbox(
    "تفعيل مراجعة Groq قبل فتح الصفقة",
    value=(load_setting("use_groq", "1") == "1"),
)
save_setting("use_groq", "1" if use_groq else "0")

groq_secret = get_secret_value("GROQ_API_KEY", "")
saved_groq_key = load_setting("groq_api_key", "")
groq_key = st.sidebar.text_input(
    "مفتاح Groq API",
    type="password",
    value=groq_secret or saved_groq_key or st.session_state.get("groq_key", ""),
)
if groq_key:
    save_setting("groq_api_key", groq_key)
st.session_state["groq_key"] = groq_key

groq_model = st.sidebar.text_input(
    "اسم نموذج Groq",
    value=load_setting("groq_model", "llama-3.3-70b-versatile"),
)
save_setting("groq_model", groq_model)

min_groq_conf = st.sidebar.slider("أدنى ثقة مطلوبة من Groq (%)", 40, 95, 60, 1)

st.sidebar.markdown("---")
st.sidebar.header("🎯 إدارة المخاطر")
atr_mult = st.sidebar.slider("معامل الوقف ATR", 1.0, 3.0, 1.5, 0.1)
risk_reward = st.sidebar.slider("نسبة العائد R:R", 1.5, 4.0, 2.0, 0.5)
min_conf = st.sidebar.slider("أدنى ثقة مطلوبة من الشبكة العصبية (%)", 60, 95, 75, 1)

# ============================================================
# Fetching Data
# ============================================================

def fetch_twelve_series(api_key, symbol="XAU/USD", interval="1h", outputsize=150):
    if not api_key:
        return pd.DataFrame()
    try:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": min(int(outputsize), 5000),
            "timezone": "UTC",
            "apikey": api_key,
        }
        resp = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=10)
        resp.raise_for_status()
        res = resp.json()

        if "values" not in res:
            st.session_state["last_twelve_error"] = res.get("message", "خطأ في Twelve Data")
            return pd.DataFrame()

        df = pd.DataFrame(res["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df.dropna(subset=["datetime", "open", "high", "low", "close"], inplace=True)
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.session_state["last_twelve_error"] = None
        return df
    except Exception as e:
        st.session_state["last_twelve_error"] = str(e)
        return pd.DataFrame()

def keep_closed_candles(df, interval_hours=1):
    if df is None or df.empty:
        return pd.DataFrame()
    now_utc = datetime.now(timezone.utc)
    delta = timedelta(hours=interval_hours)
    mask = (df["datetime"] + delta) <= pd.Timestamp(now_utc)
    return df.loc[mask].reset_index(drop=True)

# ============================================================
# حساب جميع المؤشرات والدعوم والمقاومات
# ============================================================

def apply_deep_indicators(df):
    if df is None or df.empty or len(df) < 210:
        return pd.DataFrame()

    df = df.copy()

    # 1. ATR
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # 2. Moving Averages (EMA)
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # 3. RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df["rsi"] = 100 - (100 / (1 + rs))

    # 4. MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 5. Bollinger Bands
    bb_middle = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    df["bb_width"] = (bb_upper - bb_lower) / (bb_middle + 1e-6)
    df["bb_pct"] = (df["close"] - bb_lower) / ((bb_upper - bb_lower) + 1e-6)

    # 6. Stochastic Oscillator
    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * ((df["close"] - low_14) / ((high_14 - low_14) + 1e-6))
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # 7. Support & Resistance & Pivot Points
    # Dynamic Swing High/Low Support & Resistance (Lookback 20)
    rolling_high = df["high"].rolling(20).max()
    rolling_low = df["low"].rolling(20).min()

    # المسافة النسبية للدعم والمقاومة (%)
    df["dist_to_res"] = (rolling_high - df["close"]) / df["close"]
    df["dist_to_sup"] = (df["close"] - rolling_low) / df["close"]

    # Classic Pivot Points
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    prev_close = df["close"].shift(1)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    df["pivot_pp"] = (df["close"] - pivot) / df["close"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURES, inplace=True)

    return df.reset_index(drop=True)

# ============================================================
# Model Validation & Training
# ============================================================

def model_is_ready(model_obj, scaler_obj):
    if model_obj is None or scaler_obj is None:
        return False
    if not hasattr(scaler_obj, "mean_") or not hasattr(model_obj, "classes_"):
        return False
    if len(model_obj.classes_) < 2:
        return False
    if getattr(model_obj, "n_features_in_", len(FEATURES)) != len(FEATURES):
        return False
    return True

def _background_train_and_save(api_key):
    try:
        df_train = fetch_twelve_series(api_key, outputsize=TRAINING_OUTPUT_SIZE)
        df_train = keep_closed_candles(df_train, interval_hours=1)
        df_train = apply_deep_indicators(df_train)

        if df_train.empty or len(df_train) < 250:
            return

        df_train["target"] = np.where(df_train["close"].shift(-1) > df_train["close"], 1, 0)
        train_df = df_train.iloc[:-1].copy()

        X = train_df[FEATURES].astype(float).values
        y = train_df["target"].values

        if len(np.unique(y)) < 2:
            return

        new_scaler = StandardScaler()
        X_scaled = new_scaler.fit_transform(X)

        new_model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=1000,
            early_stopping=True,
            random_state=42,
        )
        new_model.fit(X_scaled, y)

        if model_is_ready(new_model, new_scaler):
            joblib.dump(new_model, MODEL_FILE)
            joblib.dump(new_scaler, SCALER_FILE)
    except Exception:
        pass
    finally:
        if os.path.exists(TRAINING_LOCK_FILE):
            try:
                os.remove(TRAINING_LOCK_FILE)
            except OSError:
                pass

def train_deep_model(api_key):
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            m = joblib.load(MODEL_FILE)
            s = joblib.load(SCALER_FILE)
            if model_is_ready(m, s):
                return m, s
        except Exception:
            pass

    if api_key and not os.path.exists(TRAINING_LOCK_FILE):
        try:
            with open(TRAINING_LOCK_FILE, "w") as f:
                f.write(datetime.now(timezone.utc).isoformat())
            t = threading.Thread(target=_background_train_and_save, args=(api_key,), daemon=True)
            t.start()
        except Exception:
            pass

    return None, None

model, scaler = train_deep_model(twelve_key)

# ============================================================
# Groq Review Function
# ============================================================

def get_groq_review(direction, last_row, ai_conf, api_key, model_name):
    if not api_key:
        return None
    try:
        prompt = (
            f"أنت محلل فني خبير لمعدن الذهب XAU/USD.\n"
            f"الاتجاه المقترح: {direction}\n"
            f"نسبة ثقة الذكاء الاصطناعي: {ai_conf:.1f}%\n"
            f"بيانات المؤشرات الدقيقة:\n"
            f"- السعر: {last_row['close']:.2f}\n"
            f"- RSI: {last_row['rsi']:.1f}\n"
            f"- MACD: {last_row['macd']:.2f} (Signal: {last_row['macd_signal']:.2f})\n"
            f"- Stochastic %K: {last_row['stoch_k']:.1f}\n"
            f"- ATR: {last_row['atr']:.2f}\n"
            f"- المسافة لأقرب مقاومة: {last_row['dist_to_res']*100:.2f}%\n"
            f"- المسافة لأقرب دعم: {last_row['dist_to_sup']*100:.2f}%\n\n"
            f"راجع هذه البيانات بدقة وقرر هل توافق على الصفقة أم لا.\n"
            f"أعد الرد بصيغة JSON فقط كالتالي:\n"
            f'{{"agree": true, "confidence": 75, "reason": "سبب محدد باختصار"}}'
        )

        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model_name,
                "temperature": 0.2,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            timeout=15,
        )
        res.raise_for_status()
        data = res.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception:
        return None

# ============================================================
# Main UI & Logic
# ============================================================

st_autorefresh(interval=60000, key="auto_refresh")

st.title("🧠 XAU/USD Deep AI Engine Pro")
st.caption("نظام تحليل وتداول الذهب بالذكاء الاصطناعي والمؤشرات الفنية المتكاملة")

if not twelve_key:
    st.warning("⚠️ يرجى إدخال مفتاح Twelve Data API في الشريط الجانبي لبدء التحليل.")
    st.stop()

# جلب البيانات الحالية
df_raw = fetch_twelve_series(twelve_key, outputsize=220)
df_closed = keep_closed_candles(df_raw, interval_hours=1)
df_calc = apply_deep_indicators(df_closed)

if df_calc.empty:
    st.info("🔄 جاري تحميل البيانات وإعداد المؤشرات والدعوم والمقاومات...")
    st.stop()

last_candle = df_calc.iloc[-1]
current_price = last_candle["close"]

# التنبؤ بواسطة النموذج
if model_is_ready(model, scaler):
    input_vec = scaler.transform([last_candle[FEATURES].values])
    probs = model.predict_proba(input_vec)[0]
    buy_prob = probs[1] * 100
    sell_prob = probs[0] * 100

    if buy_prob >= min_conf:
        signal = "BUY"
        raw_conf = buy_prob
    elif sell_prob >= min_conf:
        signal = "SELL"
        raw_conf = sell_prob
    else:
        signal = "NEUTRAL"
        raw_conf = max(buy_prob, sell_prob)
else:
    signal = "NEUTRAL"
    raw_conf = 0.0

# Groq Review
groq_result = None
if signal != "NEUTRAL" and use_groq and groq_key:
    groq_result = get_groq_review(signal, last_candle, raw_conf, groq_key, groq_model)

# عرض الواجهات الرئيسية
tab1, tab2, tab3 = st.tabs(["📊 حالة السوق والصفقة", "📉 المؤشرات والدعوم والمقاومات", "⚙️ تفاصيل AI"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="trade-status-card">
                <div class="trade-status-title">التوصية الحالية</div>
                <div class="trade-status-value trade-{signal.lower()}">{signal}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="trade-status-card">
                <div class="trade-status-title">السعر الحالي للذهب</div>
                <div class="trade-status-value">${current_price:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="trade-status-card">
                <div class="trade-status-title">نسبة ثقة AI</div>
                <div class="trade-status-value" style="color: #fbbf24;">{raw_conf:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if groq_result:
        st.subheader("🤖 رأي النموذج الثاني (Groq AI)")
        st.json(groq_result)

with tab2:
    st.subheader("📌 قراءة المؤشرات والدعوم/المقاومات المعتمدة في القرار")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("RSI (14)", f"{last_candle['rsi']:.1f}")
        st.metric("MACD Hist", f"{last_candle['macd_hist']:.3f}")
    with col_b:
        st.metric("Stochastic %K", f"{last_candle['stoch_k']:.1f}")
        st.metric("ATR (14)", f"${last_candle['atr']:.2f}")
    with col_c:
        st.metric("المسافة لأقرب مقاومة", f"{last_candle['dist_to_res']*100:.2f}%")
        st.metric("المسافة لأقرب دعم", f"{last_candle['dist_to_sup']*100:.2f}%")

with tab3:
    st.subheader("🧪 ميزات شبكة الـ Neural Network المفرزة")
    st.write("القيم المدخلة للنموذج في الشمعة الأخيرة:")
    st.dataframe(pd.DataFrame([last_candle[FEATURES]]))
