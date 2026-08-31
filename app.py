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
# 1. إعدادات الصفحة والتنسيق العالمي (World-Class Futuristic UI)
# ============================================================

st.set_page_config(
    page_title="XAU/USD Institutional Deep Engine Pro",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

def render_html(html_content):
    try:
        if hasattr(st, "html"):
            st.html(html_content)
        else:
            st.markdown(html_content, unsafe_allow_html=True)
    except Exception:
        st.markdown(html_content, unsafe_allow_html=True)

render_html(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;600;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif !important;
    }

    /* الخلفية العامة الهادئة والاحترافية */
    .stApp {
        background: radial-gradient(circle at 50% -20%, #151d30, #090c15 80%);
        color: #e2e8f0;
    }

    /* الشريط الجانبي */
    section[data-testid="stSidebar"] {
        background-color: #0b101d !important;
        border-right: 1px solid #1e293b;
    }

    /* البطاقات الزجاجية Glassmorphism */
    .glass-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
    }

    /* كروت المؤشرات الفردية */
    .metric-card-pro {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f8fafc;
    }

    /* التنبيه الضوئي المستمر (Live Status) */
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #4ade80;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 10px #22c55e;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    /* علامات التوصية (BUY / SELL / NEUTRAL) */
    .signal-header {
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
    }
    .signal-buy {
        color: #10b981;
        text-shadow: 0 0 30px rgba(16, 185, 129, 0.4);
    }
    .signal-sell {
        color: #f43f5e;
        text-shadow: 0 0 30px rgba(244, 63, 94, 0.4);
    }
    .signal-neutral {
        color: #64748b;
    }

    /* ترويسة الصفحة */
    .main-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 25px;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
"""
)

# ============================================================
# 2. الثوابت وقواعد شبكة الذكاء الاصطناعي والميزات
# ============================================================

DB_FILE = "xau_deep_ai.db"
MODEL_FILE = "xau_deep_mlp_v4_advanced.pkl"
SCALER_FILE = "xau_deep_scaler_v4_advanced.pkl"
TRAINING_LOCK_FILE = "training_v4.lock"

TRAINING_OUTPUT_SIZE = 5000

# شريط الميزات الفائق (Enhanced Feature Store)
FEATURES = [
    "atr", "ema_50", "ema_200", "rsi", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pct", "stoch_k", "stoch_d", "dist_to_sup", "dist_to_res",
    "pivot_pp", "price_vs_ema50", "price_vs_ema200", "rsi_momentum"
]

# ============================================================
# 3. إدارة قاعدة البيانات SQLite
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

def send_ntfy_notification(channel, title, message):
    if not channel:
        return
    try:
        requests.post(
            f"https://ntfy.sh/{channel}",
            data=message.encode("utf-8"),
            headers={"Title": title.encode("utf-8")}
        )
    except Exception:
        pass

# ============================================================
# 4. الشريط الجانبي والإعدادات
# ============================================================

st.sidebar.markdown("### ⚡ إعدادات المحرك المؤسسي")

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
    "قناة Ntfy للتنبيهات الفورية",
    value=load_setting("ntfy", "xau_deep_institutional_channel"),
)
save_setting("ntfy", ntfy_channel)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 إعدادات الشبكة العصبية وGroq")

# تخفيف الشروط للحصول على صفقات أكثر مع الحفاظ على القوة
min_conf = st.sidebar.slider(
    "أدنى نسبة ثقة مطلوبة من الذكاء الاصطناعي (%)",
    min_value=50,
    max_value=90,
    value=int(load_setting("min_conf", "58")),
    step=1,
    help="تم ترخيف الحد الأدنى لزيادة عدد الفرص الصفقة الجيدة بمرونة."
)
save_setting("min_conf", str(min_conf))

use_groq = st.sidebar.checkbox(
    "تفعيل التحليل الثاني بواسطة Groq AI",
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
    "نموذج Groq",
    value=load_setting("groq_model", "llama-3.3-70b-versatile"),
)
save_setting("groq_model", groq_model)

min_groq_conf = st.sidebar.slider("أدنى موافقة مطلوبة من Groq (%)", 40, 90, 55, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 إدارة المخاطر والتنفيذ")
atr_mult = st.sidebar.slider("معامل ATR لوقف الخسارة", 1.0, 3.0, 1.5, 0.1)
risk_reward = st.sidebar.slider("نسبة العائد إلى المخاطرة (R:R)", 1.2, 4.0, 2.0, 0.1)

# ============================================================
# 5. جلب بيانات السوق وحساب المؤشرات المعقدة
# ============================================================

def fetch_twelve_series(api_key, symbol="XAU/USD", interval="1h", outputsize=250):
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
        resp = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=12)
        resp.raise_for_status()
        res = resp.json()

        if "values" not in res:
            st.session_state["last_twelve_error"] = res.get("message", "خطأ في Twelve Data API")
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

def apply_deep_indicators(df):
    if df is None or df.empty or len(df) < 210:
        return pd.DataFrame()

    df = df.copy()

    # 1. Average True Range (ATR)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # 2. Exponential Moving Averages (EMA)
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # 3. RSI & Momentum
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_momentum"] = df["rsi"].diff(3)

    # 4. MACD Indicator
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

    # 7. Support & Resistance & Pivots
    rolling_high = df["high"].rolling(20).max()
    rolling_low = df["low"].rolling(20).min()
    df["dist_to_res"] = (rolling_high - df["close"]) / df["close"]
    df["dist_to_sup"] = (df["close"] - rolling_low) / df["close"]

    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    prev_close = df["close"].shift(1)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    df["pivot_pp"] = (df["close"] - pivot) / df["close"]

    # 8. Dynamic Ratios (New Advanced Features)
    df["price_vs_ema50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
    df["price_vs_ema200"] = (df["close"] - df["ema_200"]) / df["ema_200"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURES, inplace=True)

    return df.reset_index(drop=True)

# ============================================================
# 6. تطوير المحرك العصبي العميق (Deep MLP Neural Network)
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
    """
    تدريب شبكة عصبية متعددة الطبقات (Deep Multi-Layer Perceptron) 
    تتكون من 3 طبقات خفية عميقة (256 -> 128 -> 64) لرفع مستوى التنبؤ.
    """
    try:
        df_train = fetch_twelve_series(api_key, outputsize=TRAINING_OUTPUT_SIZE)
        df_train = keep_closed_candles(df_train, interval_hours=1)
        df_train = apply_deep_indicators(df_train)

        if df_train.empty or len(df_train) < 300:
            return

        # تحديد الهدف: هل ارتفع السعر بنسبة واعدة في الشمعة التالية؟
        df_train["target"] = np.where(df_train["close"].shift(-1) > df_train["close"], 1, 0)
        train_df = df_train.iloc[:-1].copy()

        X = train_df[FEATURES].astype(float).values
        y = train_df["target"].values

        if len(np.unique(y)) < 2:
            return

        new_scaler = StandardScaler()
        X_scaled = new_scaler.fit_transform(X)

        # بنية عصبية فائقة القوة وعميقة (Deep Neural Architecture)
        new_model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=1200,
            early_stopping=True,
            n_iter_no_change=25,
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
# 7. استشارة نموذج Groq AI الثانوي
# ============================================================

def get_groq_review(direction, last_row, ai_conf, api_key, model_name):
    if not api_key:
        return None
    try:
        prompt = (
            f"أنت نظام تحليل مؤسسي عالي الدقة لذهب XAU/USD.\n"
            f"إشارة المحرك العصبي: {direction} بثقة {ai_conf:.1f}%\n"
            f"بيانات السوق للحظة الحالية:\n"
            f"- السعر الحالي: ${last_row['close']:.2f}\n"
            f"- RSI (14): {last_row['rsi']:.1f}\n"
            f"- MACD Hist: {last_row['macd_hist']:.3f}\n"
            f"- Stochastic %K: {last_row['stoch_k']:.1f}\n"
            f"- ATR: ${last_row['atr']:.2f}\n"
            f"- بعد السعر عن المقاومة: {last_row['dist_to_res']*100:.2f}%\n"
            f"- بعد السعر عن الدعم: {last_row['dist_to_sup']*100:.2f}%\n\n"
            f"هل تؤيد فتح هذه الصفقة؟ أجب بصيغة JSON حصرية:\n"
            f'{{"agree": true, "confidence": 75, "reason": "سبب اختصار باللغة العربية"}}'
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
# 8. التحديث التلقائي واللوحة الرئيسية UI
# ============================================================

st_autorefresh(interval=60000, key="auto_refresh_pro")

# ترويسة الصفحة العالمية
render_html(
    """
    <div class="main-header">
        <div>
            <h1 style="margin:0; font-weight:900; font-size:2.2rem; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚡ XAU/USD Institutional Deep AI Engine
            </h1>
            <p style="margin:5px 0 0 0; color:#94a3b8; font-size:0.95rem;">محرك عصبوني عميق متقدم لتحليل وتداول الذهب مع فلاتر الزخم الفنية</p>
        </div>
        <div class="live-badge">
            <div class="pulse-dot"></div>
            <span>المحرك نشط مباشر</span>
        </div>
    </div>
    """
)

if not twelve_key:
    st.warning("⚠️ يرجى إدخال مفتاح Twelve Data API في الشريط الجانبي لتشغيل المحرك.")
    st.stop()

# جلب بيانات السوق
df_raw = fetch_twelve_series(twelve_key, outputsize=260)
df_closed = keep_closed_candles(df_raw, interval_hours=1)
df_calc = apply_deep_indicators(df_closed)

if df_calc.empty:
    st.info("🔄 جاري تحميل بيانات الذهب المعمقة وبناء المستويات الديناميكية...")
    st.stop()

last_candle = df_calc.iloc[-1]
current_price = last_candle["close"]

# تقييم الإشارة بالذكاء الاصطناعي مع المرونة المشروطة (More Frequent Signals)
signal = "NEUTRAL"
raw_conf = 0.0

if model_is_ready(model, scaler):
    input_vec = scaler.transform([last_candle[FEATURES].values])
    probs = model.predict_proba(input_vec)[0]
    buy_prob = probs[1] * 100
    sell_prob = probs[0] * 100

    # دعم مرونة إضافية: تقاطع RSI أو MACD يزيد ترجيح الإشارة
    rsi_val = last_candle["rsi"]
    macd_h = last_candle["macd_hist"]

    adjusted_buy = buy_prob + (5.0 if (rsi_val < 45 and macd_h > 0) else 0.0)
    adjusted_sell = sell_prob + (5.0 if (rsi_val > 55 and macd_h < 0) else 0.0)

    if adjusted_buy >= min_conf and adjusted_buy > adjusted_sell:
        signal = "BUY"
        raw_conf = min(adjusted_buy, 99.0)
    elif adjusted_sell >= min_conf and adjusted_sell > adjusted_buy:
        signal = "SELL"
        raw_conf = min(adjusted_sell, 99.0)
    else:
        signal = "NEUTRAL"
        raw_conf = max(buy_prob, sell_prob)

# حساب وقف الخسارة والهدف بناءً على ATR
atr_val = last_candle["atr"]
if signal == "BUY":
    sl_price = current_price - (atr_val * atr_mult)
    tp_price = current_price + ((current_price - sl_price) * risk_reward)
elif signal == "SELL":
    sl_price = current_price + (atr_val * atr_mult)
    tp_price = current_price - ((sl_price - current_price) * risk_reward)
else:
    sl_price = 0.0
    tp_price = 0.0

# Groq Review
groq_result = None
if signal != "NEUTRAL" and use_groq and groq_key:
    groq_result = get_groq_review(signal, last_candle, raw_conf, groq_key, groq_model)

# ============================================================
# 9. التبويبات والعرض المتقدم (Advanced Tabs & Displays)
# ============================================================

tab_main, tab_tech, tab_ai, tab_history = st.tabs([
    "🎯 مركز التوصيات والتنفيذ", 
    "📊 الدعوم والمؤشرات الفنية", 
    "🧠 الشبكة العصبية المعمقة",
    "📜 سجل الصفقات والتدقيق"
])

with tab_main:
    col_sig, col_price, col_conf = st.columns([1.2, 1, 1])
    
    with col_sig:
        sig_class = f"signal-{signal.lower()}"
        render_html(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="color:#94a3b8; font-weight:700; font-size:0.9rem;">التوصية المؤسسية الحالية</div>
                <div class="signal-header {sig_class}">{signal}</div>
            </div>
            """
        )

    with col_price:
        render_html(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="color:#94a3b8; font-weight:700; font-size:0.9rem;">سعر الذهب الحالي (XAU/USD)</div>
                <div style="font-size: 2.5rem; font-weight: 900; color: #f8fafc; margin-top: 5px;">
                    ${current_price:.2f}
                </div>
            </div>
            """
        )

    with col_conf:
        render_html(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="color:#94a3b8; font-weight:700; font-size:0.9rem;">ثقة المحرك العصبي</div>
                <div style="font-size: 2.5rem; font-weight: 900; color: #38bdf8; margin-top: 5px;">
                    {raw_conf:.1f}%
                </div>
            </div>
            """
        )

    # تفاصيل الأهداف والوقف عند توفر صفقة
    if signal != "NEUTRAL":
        st.markdown("<h4 style='color:#38bdf8;'>🎯 تفاصيل الصفقة المقترحة</h4>", unsafe_allow_html=True)
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("نقطة الدخول (Entry)", f"${current_price:.2f}")
        t_col2.metric("وقف الخسارة (SL)", f"${sl_price:.2f}")
        t_col3.metric("جني الأرباح (TP)", f"${tp_price:.2f}")
        t_col4.metric("المخاطرة / العائد", f"1:{risk_reward:.1f}")

    # استجابة Groq AI
    if groq_result:
        st.markdown("---")
        st.markdown("<h4 style='color:#818cf8;'>🤖 التدقيق الثاني عبر Groq AI</h4>", unsafe_allow_html=True)
        g_col1, g_col2 = st.columns([1, 2])
        with g_col1:
            agreed = groq_result.get("agree", False)
            st.markdown(
                f"""
                <div class="glass-card" style="padding:15px; text-align:center;">
                    <div style="font-size:0.9rem; color:#94a3b8;">توافق Groq AI</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{'#10b981' if agreed else '#f43f5e'};">
                        {'موافق ✅' if agreed else 'غير موافق ❌'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with g_col2:
            st.info(f"**السبب التحليلي:** {groq_result.get('reason', 'لا يوجد تعليق')}")

with tab_tech:
    st.markdown("#### 📌 قراءة المؤشرات الفنية والمستويات المحورية")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("RSI (14)", f"{last_candle['rsi']:.1f}")
        st.metric("RSI Momentum", f"{last_candle['rsi_momentum']:.2f}")
    with m2:
        st.metric("MACD Hist", f"{last_candle['macd_hist']:.3f}")
        st.metric("Stochastic %K", f"{last_candle['stoch_k']:.1f}")
    with m3:
        st.metric("ATR (14)", f"${last_candle['atr']:.2f}")
        st.metric("نطاق Bollinger %", f"{last_candle['bb_pct']*100:.1f}%")
    with m4:
        st.metric("بعد المقاومة", f"{last_candle['dist_to_res']*100:.2f}%")
        st.metric("بعد الدعم", f"{last_candle['dist_to_sup']*100:.2f}%")

with tab_ai:
    st.markdown("#### 🔬 ميزات الشبكة العصبية المدخلة (MLP Feature Vector)")
    st.dataframe(pd.DataFrame([last_candle[FEATURES]]), use_container_width=True)

with tab_history:
    st.markdown("#### 📜 سجل العمليات وتاريخ الأداء")
    conn = get_db_connection()
    try:
        trades_df = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC LIMIT 50", conn)
        if not trades_df.empty:
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("لا توجد صفقات مسجلة في السجل حالياً.")
    finally:
        conn.close()
