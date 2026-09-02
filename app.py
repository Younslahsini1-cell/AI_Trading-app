"""
XAU/USD Deep AI Engine — v3 (Background Engine Edition)
=========================================================
ICT / Smart Money + Neural Network + Groq Second Opinion

[قاعدة الترند صديقك]:
- إذا كان الترند هابطاً: ندرس المؤشرات ونبحث عن فرص بيع فقط (SL/TP).
- إذا كان الترند صاعداً: ندرس المؤشرات ونبحث عن فرص شراء فقط (SL/TP).
- لا يوجد حظر إطلاقاً، فقط توجيه ذكي نحو اتجاه الترند.
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
# جلسة HTTP مشتركة (Connection Pooling + Retry) لتسريع الاتصال
# ============================================================

def _build_http_session():
    """
    جلسة requests واحدة يعاد استخدامها لكل الاتصالات الخارجية
    (Twelve Data, Yahoo, Groq, Ntfy) بدلاً من فتح اتصال TCP/TLS جديد
    في كل نداء. هذا يقلل زمن الاستجابة بشكل ملحوظ خصوصاً أن المحرك
    الخلفي يستدعي هذه الخدمات كل 45 ثانية.
    """
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
    adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry_strategy)
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
            st.markdown(html_content, unsafe_allow_html=True)
    except Exception:
        st.markdown(html_content, unsafe_allow_html=True)


# ============================================================
# CSS
# ============================================================

render_html(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900');

    html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; }
    .stApp { background-color: #07090e; color: #f3f4f6; }
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }

    .ai-level-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        padding: 30px; border-radius: 20px; text-align: center;
        border: 1px solid #3b82f6;
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.4);
        margin-bottom: 20px;
    }
    .ai-level-title { font-size: 1.2rem; color: #93c5fd; font-weight: 700; margin-bottom: 10px; }
    .ai-level-value { font-size: 4rem; font-weight: 900; color: #fbbf24; line-height: 1; }
    .ai-level-sub { font-size: 1rem; color: #94a3b8; margin-top: 10px; }

    .trade-status-card {
        background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
        border: 1px solid #334155; border-radius: 18px;
        padding: 22px; margin-bottom: 20px; text-align: center;
    }
    .trade-status-title { font-size: 1rem; color: #94a3b8; font-weight: 700; }
    .trade-status-value { font-size: 2.3rem; font-weight: 900; margin-top: 8px; }
    .trade-buy { color: #22c55e; }
    .trade-sell { color: #ef4444; }
    .trade-neutral { color: #94a3b8; }

    .engine-pulse {
        display: inline-block; width: 10px; height: 10px; border-radius: 50%;
        background: #22c55e; box-shadow: 0 0 10px #22c55e; margin-inline-end: 8px;
    }
</style>
"""
)


# ============================================================
# SECRETS (تم وضعها في الأعلى لمنع أي خطأ NameError)
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

FEATURES = ["atr", "ema_50", "ema_200", "rsi"]
ICT_SWING_LOOKBACK = 3
ICT_OB_DISPLACEMENT_MULT = 1.2


# ============================================================
# Database
# ============================================================

def get_db_connection():
    return sqlite3.connect(DB_FILE, timeout=20, check_same_thread=False)


def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, symbol TEXT, direction TEXT, entry REAL, sl REAL, tp REAL,
        win INTEGER, note TEXT, claude_conf REAL, claude_note TEXT,
        groq_conf REAL, groq_note TEXT, ai_conf_before_groq REAL, ai_conf_after_groq REAL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS active_trade (
        id INTEGER PRIMARY KEY, symbol TEXT, direction TEXT, entry REAL, sl REAL, tp REAL,
        time TEXT, features TEXT, ai_conf REAL, groq_conf REAL, groq_note TEXT, signal_bar_time TEXT
    )""")
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


def get_successful_trades_count():
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM trades WHERE win = 1")
        return int(c.fetchone()[0])
    finally:
        conn.close()


def get_total_trades_count():
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM trades")
        return int(c.fetchone()[0])
    finally:
        conn.close()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("⚙️ إعدادات الذكاء الاصطناعي")

# مخزن محلي في متصفح المستخدم — يبقى محفوظاً حتى بعد إعادة تشغيل (reboot)
# التطبيق على الخادم، لأن قاعدة البيانات المحلية (sqlite) تُمسح عند كل ريبوت
# على Streamlit Community Cloud بينما تخزين المتصفح لا يتأثر بذلك إطلاقاً.
localS = LocalStorage()

twelve_secret = get_secret_value("TWELVE_DATA_API_KEY", "")
if "twelve_key" not in st.session_state:
    stored_twelve_key = localS.getItem("twelve_key_ls") or load_setting("twelve_key", "")
    st.session_state["twelve_key"] = twelve_secret or stored_twelve_key
twelve_key = st.sidebar.text_input("مفتاح Twelve Data API", type="password", key="twelve_key")
if twelve_key:
    save_setting("twelve_key", twelve_key)
    localS.setItem("twelve_key_ls", twelve_key)

ntfy_channel = st.sidebar.text_input("قناة Ntfy للتنبيهات", value=load_setting("ntfy", "xau_deep_channel"))
save_setting("ntfy", ntfy_channel)

st.sidebar.markdown("---")
st.sidebar.header("🧠 الرأي الثاني (Groq)")
use_groq = st.sidebar.checkbox("تفعيل مراجعة Groq", value=(load_setting("use_groq", "1") == "1"))
save_setting("use_groq", "1" if use_groq else "0")

groq_secret = get_secret_value("GROQ_API_KEY", "")
if "groq_key" not in st.session_state:
    stored_groq_key = localS.getItem("groq_key_ls") or load_setting("groq_key", "")
    st.session_state["groq_key"] = groq_secret or stored_groq_key
groq_key = st.sidebar.text_input("مفتاح Groq API", type="password", key="groq_key")
if groq_key:
    save_setting("groq_key", groq_key)
    localS.setItem("groq_key_ls", groq_key)

groq_model = st.sidebar.text_input("اسم نموذج Groq", value=load_setting("groq_model", "openai/gpt-oss-120b"))
save_setting("groq_model", groq_model)

min_groq_conf = st.sidebar.slider("أدنى ثقة مطلوبة من Groq", 30, 95, 50, 1)
save_setting("min_groq_conf", min_groq_conf)

st.sidebar.markdown("---")
st.sidebar.header("🎯 إدارة المخاطر")
atr_mult = st.sidebar.slider("معامل الوقف ATR", 1.0, 3.0, 1.5, 0.1)
save_setting("atr_mult", atr_mult)
risk_reward = st.sidebar.slider("نسبة العائد R:R", 1.5, 4.0, 2.0, 0.5)
save_setting("risk_reward", risk_reward)
min_conf = st.sidebar.slider("أدنى ثقة نهائية مطلوبة لفتح صفقة", 50, 95, 65, 1)
save_setting("min_conf", min_conf)

if st.sidebar.button("🔄 إعادة تدريب النموذج من الصفر"):
    for file_path in (MODEL_FILE, SCALER_FILE, TRAINING_LOCK_FILE):
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
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

def send_alert(msg, title="🧠 Deep AI Alert"):
    channel_setting = load_setting("ntfy", "")
    if not channel_setting:
        return
    channel = channel_setting.strip().split("/")[-1]
    if not channel:
        return
    try:
        HTTP_SESSION.post(f"https://ntfy.sh/{channel}", data=msg.encode("utf-8"), headers={"Title": title, "Priority": "high"}, timeout=5)
    except Exception:
        pass


def send_trade_confirmation_alert(direction, entry, sl, tp, final_confidence, risk_reward_ratio, ai_conf, groq_conf=None):
    """
    إشعار Ntfy مخصص يُرسل فور تأكيد فتح الصفقة (بعد اجتياز كل شروط
    الترند + ICT + الخبرة + Groq). منفصل عن تنبيه الإشارة الأساسي.
    """
    groq_line = f"\nGroq Confidence: {groq_conf:.1f}%" if groq_conf is not None else ""
    msg = (
        f"✅ تم تأكيد الصفقة (Trade Confirmed)\n"
        f"الاتجاه: {direction}\n"
        f"الدخول: ${entry}\n"
        f"وقف الخسارة (SL): ${sl}\n"
        f"جني الأرباح (TP): ${tp}\n"
        f"نسبة R:R = 1:{risk_reward_ratio}\n"
        f"ثقة AI الخام: {ai_conf:.1f}%{groq_line}\n"
        f"الثقة النهائية: {final_confidence:.1f}%"
    )
    send_alert(msg, title="✅ Trade Confirmed — XAU/USD")


def fetch_twelve_series(api_key, symbol="XAU/USD", interval="1h", outputsize=150):
    if not api_key:
        return pd.DataFrame()
    try:
        params = {"symbol": symbol, "interval": interval, "outputsize": min(int(outputsize), 5000), "timezone": "UTC", "apikey": api_key}
        response = HTTP_SESSION.get("https://api.twelvedata.com/time_series", params=params, timeout=10)
        response.raise_for_status()
        result = response.json()
        if "values" not in result:
            APP_STATE_set("last_twelve_error", result.get("message", "استجابة غير متوقعة"))
            return pd.DataFrame()
        values = result["values"]
        if not values:
            return pd.DataFrame()
        df = pd.DataFrame(values)
        required = ["datetime", "open", "high", "low", "close"]
        if not all(col in df.columns for col in required):
            return pd.DataFrame()
        df = df[required].copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["datetime", "open", "high", "low", "close"], inplace=True)
        df.sort_values("datetime", inplace=True)
        df.drop_duplicates(subset=["datetime"], keep="last", inplace=True)
        df.reset_index(drop=True, inplace=True)
        APP_STATE_set("last_twelve_error", None)
        return df
    except Exception as exc:
        APP_STATE_set("last_twelve_error", f"تعذّر الاتصال: {exc}")
        return pd.DataFrame()


def keep_closed_candles(df, interval_hours=1):
    if df is None or df.empty or "datetime" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    now_utc = datetime.now(timezone.utc)
    candle_delta = timedelta(hours=interval_hours)
    mask = df["datetime"] + candle_delta <= pd.Timestamp(now_utc)
    closed = df.loc[mask].copy()
    return closed.reset_index(drop=True)


def fetch_training_data_twelve(api_key):
    return fetch_twelve_series(api_key, symbol="XAU/USD", interval="1h", outputsize=TRAINING_OUTPUT_SIZE)


def fetch_free_training_series(symbol="XAUUSD=X", interval="60m", range_="730d"):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        response = HTTP_SESSION.get(url, params={"interval": interval, "range": range_}, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
        payload = response.json()
        result_list = (payload.get("chart") or {}).get("result") or []
        if not result_list:
            return pd.DataFrame()
        result = result_list[0]
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        if not timestamps or not closes:
            return pd.DataFrame()
        df = pd.DataFrame({"datetime": pd.to_datetime(timestamps, unit="s", utc=True), "open": opens, "high": highs, "low": lows, "close": closes})
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["datetime", "open", "high", "low", "close"], inplace=True)
        df.sort_values("datetime", inplace=True)
        df.drop_duplicates(subset=["datetime"], keep="last", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_training_data_free():
    return fetch_free_training_series(symbol="XAUUSD=X", interval="60m", range_="730d")


# ============================================================
# Indicators & Model
# ============================================================

def apply_deep_indicators(df):
    if df is None or df.empty:
        return pd.DataFrame()
    if len(df) < 60:
        return pd.DataFrame()
    df = df.copy()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df["rsi"] = 100 - (100 / (1 + rs))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURES, inplace=True)
    return df.reset_index(drop=True)


def model_is_ready(model_obj, scaler_obj):
    if model_obj is None or scaler_obj is None:
        return False
    if not hasattr(scaler_obj, "mean_") or not hasattr(scaler_obj, "scale_"):
        return False
    if not hasattr(model_obj, "classes_"):
        return False
    try:
        if len(model_obj.classes_) < 2:
            return False
        if getattr(model_obj, "n_features_in_", len(FEATURES)) != len(FEATURES):
            return False
    except Exception:
        return False
    return True


# ============================================================
# Background Training
# ============================================================

def _background_train_and_save(api_key):
    try:
        df_train = fetch_training_data_free()
        if df_train.empty and api_key:
            df_train = fetch_training_data_twelve(api_key)
        df_train = keep_closed_candles(df_train, interval_hours=1)
        df_train = apply_deep_indicators(df_train)
        if df_train.empty or len(df_train) < 60:
            return
        future_close = df_train["close"].shift(-1)
        valid_mask = future_close.notna()
        train_df = df_train.loc[valid_mask].copy()
        if len(train_df) < 30:
            return
        X = train_df[FEATURES].astype(float).values
        y = np.where(train_df["close"].shift(-1) > train_df["close"], 1, 0)
        X = X[:-1]
        y = y[:-1]
        if len(X) < 30 or len(y) < 30:
            return
        if len(np.unique(y)) < 2:
            return
        new_scaler = StandardScaler()
        X_scaled = new_scaler.fit_transform(X)
        new_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam", alpha=0.0001, learning_rate_init=0.001, max_iter=2000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=40, random_state=42)
        new_model.fit(X_scaled, y)
        if not model_is_ready(new_model, new_scaler):
            return
        model_tmp = MODEL_FILE + ".tmp"
        scaler_tmp = SCALER_FILE + ".tmp"
        joblib.dump(new_model, model_tmp)
        joblib.dump(new_scaler, scaler_tmp)
        os.replace(model_tmp, MODEL_FILE)
        os.replace(scaler_tmp, SCALER_FILE)
        APP_STATE_set("last_train_time", datetime.now(timezone.utc).isoformat())
    except Exception:
        pass
    finally:
        if os.path.exists(TRAINING_LOCK_FILE):
            try:
                os.remove(TRAINING_LOCK_FILE)
            except OSError:
                pass


def clean_stale_training_lock():
    if not os.path.exists(TRAINING_LOCK_FILE):
        return
    try:
        age = time.time() - os.path.getmtime(TRAINING_LOCK_FILE)
        if age > TRAINING_LOCK_MAX_AGE:
            os.remove(TRAINING_LOCK_FILE)
    except Exception:
        pass


def maybe_spawn_training(api_key, force=False):
    if os.path.exists(TRAINING_LOCK_FILE):
        return
    needs_training = force or not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE))
    if not needs_training:
        try:
            age = time.time() - os.path.getmtime(MODEL_FILE)
            if age > RETRAIN_INTERVAL_SECONDS:
                needs_training = True
        except OSError:
            needs_training = True
    if not needs_training:
        return
    try:
        with open(TRAINING_LOCK_FILE, "x", encoding="utf-8") as file:
            file.write(datetime.now(timezone.utc).isoformat())
        thread = threading.Thread(target=_background_train_and_save, args=(api_key,), daemon=True, name="bg_trainer")
        thread.start()
    except FileExistsError:
        pass
    except Exception:
        pass


def load_current_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            loaded_model = joblib.load(MODEL_FILE)
            loaded_scaler = joblib.load(SCALER_FILE)
            if model_is_ready(loaded_model, loaded_scaler):
                return loaded_model, loaded_scaler
        except Exception:
            pass
    return None, None


clean_stale_training_lock()


# ============================================================
# Experience Layer & Groq
# ============================================================

def get_experience_adjustment(direction, ai_conf):
    total = get_total_trades_count()
    if total < 20:
        return {"available": False, "confidence": ai_conf, "win_rate": None, "sample": total}
    normalized_direction = "BUY" if "BUY" in str(direction) else "SELL"
    conn = get_db_connection()
    try:
        df = pd.read_sql("SELECT direction, win FROM trades WHERE direction LIKE ?", conn, params=(f"%{normalized_direction}%",))
    finally:
        conn.close()
    if len(df) < 10:
        return {"available": False, "confidence": ai_conf, "win_rate": None, "sample": len(df)}
    wins = float(df["win"].sum())
    n = len(df)
    smoothed_rate = (wins + 5.0) / (n + 10.0) * 100
    adjusted = ai_conf * 0.70 + smoothed_rate * 0.30
    return {"available": True, "confidence": round(float(adjusted), 1), "win_rate": round(smoothed_rate, 1), "sample": n}


def get_groq_review(direction, last_row, ai_conf, api_key, model_name):
    if not api_key:
        APP_STATE_set("last_groq_error", "لم يتم إدخال مفتاح Groq API.")
        return None
    try:
        prompt = (
            "أنت محلل فني مساعد لصفقة محتملة على XAU/USD.\n"
            f"الاتجاه المقترح: {direction}.\n"
            f"ثقة نموذج AI الخام: {ai_conf:.1f}%.\n"
            f"ATR={last_row['atr']:.2f}.\n"
            f"EMA50={last_row['ema_50']:.2f}.\n"
            f"EMA200={last_row['ema_200']:.2f}.\n"
            f"RSI={last_row['rsi']:.1f}.\n"
            f"السعر={last_row['close']:.2f}.\n"
            "\n"
            "راجع الاتجاه بشكل مستقل بناءً على البيانات المعطاة فقط. "
            "لا تفترض أن نموذج AI صحيح، لكن أيضًا لا تكن متشددًا "
            "بلا داعٍ — إن كانت المؤشرات معقولة الدعم للاتجاه فوافق.\n"
            "\n"
            'يجب أن يكون الرد JSON فقط بهذا الشكل: {"agree": true, "confidence": 0, "reason": "..."}'
        )
        response = HTTP_SESSION.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": model_name, "temperature": 0, "max_completion_tokens": 2000, "messages": [{"role": "system", "content": "أنت محلل فني متوازن. أعد JSON صالح فقط."}, {"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}, timeout=20)
        if not response.ok:
            APP_STATE_set("last_groq_error", f"HTTP {response.status_code} من Groq: {response.text[:400]}")
            return None
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            APP_STATE_set("last_groq_error", f"رد Groq بلا choices: {json.dumps(data)[:400]}")
            return None
        message = choices[0].get("message", {})
        text = message.get("content", "")
        if not text:
            APP_STATE_set("last_groq_error", "رد Groq وصل لكن بلا محتوى نصي.")
            return None
        cleaned = str(text).replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(cleaned)
        except Exception as parse_exc:
            APP_STATE_set("last_groq_error", f"تعذّر تحليل رد Groq كـ JSON: {parse_exc} — النص: {cleaned[:400]}")
            return None
        agree = bool(parsed.get("agree", False))
        confidence = float(parsed.get("confidence", 0))
        confidence = max(0, min(100, confidence))
        reason = str(parsed.get("reason", ""))
        APP_STATE_set("last_groq_error", None)
        return {"agree": agree, "confidence": confidence, "reason": reason}
    except Exception as exc:
        APP_STATE_set("last_groq_error", f"استثناء أثناء الاتصال بـ Groq: {exc}")
        return None


# ============================================================
# ICT Engine (كل الدوال كما هي)
# ============================================================

def find_swing_points(df, lookback=3):
    if df is None or df.empty:
        return [], []
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    n = len(df)
    swing_highs = []
    swing_lows = []
    for i in range(lookback, n - lookback):
        high_window = highs[i - lookback:i + lookback + 1]
        low_window = lows[i - lookback:i + lookback + 1]
        if highs[i] == np.max(high_window):
            swing_highs.append((i, float(highs[i])))
        if lows[i] == np.min(low_window):
            swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows


def analyze_market_structure(swing_highs, swing_lows):
    events = [(i, "high", p) for i, p in swing_highs] + [(i, "low", p) for i, p in swing_lows]
    events.sort(key=lambda x: x[0])
    structure_breaks = []
    trend = None
    last_high = None
    last_low = None
    for i, kind, price in events:
        if kind == "high":
            if last_high is not None and price > last_high:
                label = "BOS" if trend == "bullish" else "CHoCH"
                structure_breaks.append({"type": label, "direction": "BULLISH", "price": round(price, 2)})
                trend = "bullish"
            last_high = price
        else:
            if last_low is not None and price < last_low:
                label = "BOS" if trend == "bearish" else "CHoCH"
                structure_breaks.append({"type": label, "direction": "BEARISH", "price": round(price, 2)})
                trend = "bearish"
            last_low = price
    current_bias = structure_breaks[-1]["direction"] if structure_breaks else "NEUTRAL"
    return current_bias, structure_breaks[-8:]


def detect_order_blocks(df, lookback=40, displacement_atr_mult=1.2):
    if df is None or df.empty or "atr" not in df.columns or len(df) < 5:
        return None, None
    recent = df.iloc[-lookback:].reset_index(drop=True)
    bullish_ob = None
    bearish_ob = None
    for i in range(1, len(recent)):
        atr_value = float(recent["atr"].iloc[i])
        if not np.isfinite(atr_value) or atr_value <= 0:
            continue
        body = recent["close"].iloc[i] - recent["open"].iloc[i]
        is_displacement = abs(body) > displacement_atr_mult * atr_value
        previous = recent.iloc[i - 1]
        if is_displacement and body > 0 and previous["close"] < previous["open"]:
            bullish_ob = {"top": round(float(previous["open"]), 2), "bottom": round(float(previous["low"]), 2), "bias": "BULLISH", "strength": round(min((abs(body) / atr_value) * 100, 999), 1)}
        elif is_displacement and body < 0 and previous["close"] > previous["open"]:
            bearish_ob = {"top": round(float(previous["high"]), 2), "bottom": round(float(previous["open"]), 2), "bias": "BEARISH", "strength": round(min((abs(body) / atr_value) * 100, 999), 1)}
    return bullish_ob, bearish_ob


def detect_fvg(df, lookback=60):
    if df is None or df.empty or len(df) < 3:
        return None, None
    recent = df.iloc[-lookback:].reset_index(drop=True)
    bullish_fvg = None
    bearish_fvg = None
    for i in range(2, len(recent)):
        c1 = recent.iloc[i - 2]
        c3 = recent.iloc[i]
        if c1["high"] < c3["low"]:
            bullish_fvg = {"top": round(float(c3["low"]), 2), "bottom": round(float(c1["high"]), 2), "bias": "BULLISH"}
        if c1["low"] > c3["high"]:
            bearish_fvg = {"top": round(float(c1["low"]), 2), "bottom": round(float(c3["high"]), 2), "bias": "BEARISH"}
    return bullish_fvg, bearish_fvg


def detect_liquidity_and_manipulation(df, swing_highs, swing_lows):
    if not swing_highs or not swing_lows or df is None or df.empty:
        return None, None, False, ""
    recent_highs = swing_highs[-5:]
    recent_lows = swing_lows[-5:]
    bsl_price = max(p for _, p in recent_highs)
    ssl_price = min(p for _, p in recent_lows)
    last = df.iloc[-1]
    manipulation = False
    note = ""
    if last["high"] > bsl_price and last["close"] < bsl_price:
        manipulation = True
        note = "BSL swept — احتمال انعكاس."
    elif last["low"] < ssl_price and last["close"] > ssl_price:
        manipulation = True
        note = "SSL swept — احتمال انعكاس."
    return round(bsl_price, 2), round(ssl_price, 2), manipulation, note


def session_analyzer(df, session_len=24):
    if df is None or df.empty or len(df) < 5:
        return None
    window = df.iloc[-min(session_len, len(df)):]
    session_high = float(window["high"].max())
    session_low = float(window["low"].min())
    session_range = session_high - session_low
    net_change = float(window["close"].iloc[-1]) - float(window["open"].iloc[0])
    if net_change < 0:
        bias = "BEARISH"
    elif net_change > 0:
        bias = "BULLISH"
    else:
        bias = "NEUTRAL"
    main_score = round(min(abs(net_change) / (session_range + 1e-6) * 100, 100), 1)
    return {"bias": bias, "main_score": main_score, "range": round(session_range, 2), "high": round(session_high, 2), "low": round(session_low, 2)}


def compute_asian_session_levels(df):
    if df is None or df.empty or len(df) < 24:
        return None
    window = df.iloc[-24:]
    return {"asian_high": round(float(window["high"].max()), 2), "asian_low": round(float(window["low"].min()), 2)}


def compute_fibonacci_extension(swing_low, swing_high, direction):
    diff = swing_high - swing_low
    if diff <= 0:
        return {}
    ratios = [(1.0, "100% EXTENSION"), (1.27, "127% EXTENSION"), (1.618, "161.8% EXTENSION"), (2.0, "200% EXTENSION"), (2.618, "261.8% EXTENSION")]
    levels = {}
    for ratio, label in ratios:
        if direction == "BULLISH":
            levels[label] = round(swing_high + diff * (ratio - 1), 2)
        else:
            levels[label] = round(swing_low - diff * (ratio - 1), 2)
    levels["EQUILIBRIUM TARGET"] = round((swing_high + swing_low) / 2, 2)
    return levels


def compute_ote_zone(swing_low, swing_high, direction, current_price):
    diff = swing_high - swing_low
    if diff <= 0:
        return None
    if direction == "BULLISH":
        top = swing_high - diff * 0.618
        bottom = swing_high - diff * 0.79
    else:
        top = swing_low + diff * 0.79
        bottom = swing_low + diff * 0.618
    lo = min(top, bottom)
    hi = max(top, bottom)
    return {"top": round(hi, 2), "bottom": round(lo, 2), "inside": lo <= current_price <= hi, "direction": direction}


def compute_volatility_risk(df):
    if df is None or df.empty or "atr" not in df.columns:
        return "N/A", 0, 0.0
    atr_series = df["atr"].dropna()
    if atr_series.empty:
        return "N/A", 0, 0.0
    current_atr = float(atr_series.iloc[-1])
    percentile = (atr_series < current_atr).mean() * 100
    if percentile >= 90:
        label, score = "CRISIS", 90
    elif percentile >= 65:
        label, score = "HIGH", 70
    elif percentile >= 35:
        label, score = "MEDIUM", 50
    else:
        label, score = "LOW", 25
    return label, score, round(current_atr, 2)


def detect_recent_displacement(df, lookback=10, atr_mult=1.3):
    if df is None or df.empty:
        return []
    recent = df.iloc[-lookback:]
    results = []
    for _, row in recent.iterrows():
        atr_value = row.get("atr", np.nan)
        if pd.isna(atr_value) or atr_value <= 0:
            continue
        body = row["close"] - row["open"]
        candle_range = row["high"] - row["low"]
        body_pct = round(abs(body) / (candle_range + 1e-6) * 100, 1)
        if abs(body) > atr_mult * atr_value:
            results.append({"bias": "BULLISH" if body > 0 else "BEARISH", "body_pct": body_pct})
    return results[-3:]


def run_ict_engine(df_processed, swing_lookback=3, ob_mult=1.2):
    if df_processed is None or df_processed.empty or len(df_processed) < max(30, swing_lookback * 6):
        return None
    swing_highs, swing_lows = find_swing_points(df_processed, lookback=swing_lookback)
    bias, structure_breaks = analyze_market_structure(swing_highs, swing_lows)
    bull_ob, bear_ob = detect_order_blocks(df_processed, displacement_atr_mult=ob_mult)
    bull_fvg, bear_fvg = detect_fvg(df_processed)
    bsl, ssl, manipulation, manip_note = detect_liquidity_and_manipulation(df_processed, swing_highs, swing_lows)
    session_info = session_analyzer(df_processed)
    asian_levels = compute_asian_session_levels(df_processed)
    vol_label, vol_score, atr_value = compute_volatility_risk(df_processed)
    displacements = detect_recent_displacement(df_processed)
    current_price = round(float(df_processed["close"].iloc[-1]), 2)
    fib_levels = {}
    ote = None
    if swing_highs and swing_lows:
        last_low_idx, last_low_price = swing_lows[-1]
        last_high_idx, last_high_price = swing_highs[-1]
        leg_direction = "BULLISH" if last_low_idx > last_high_idx else "BEARISH"
        lo_price = min(last_low_price, last_high_price)
        hi_price = max(last_low_price, last_high_price)
        fib_levels = compute_fibonacci_extension(lo_price, hi_price, leg_direction)
        ote = compute_ote_zone(lo_price, hi_price, leg_direction, current_price)
    matching_breaks = [brk for brk in structure_breaks if brk["direction"] == bias]
    if bias != "NEUTRAL":
        structure_score = min(100, 25 + len(matching_breaks) * 25)
    else:
        structure_score = 50
    liquidity_score = 80 if manipulation else 40
    if bias == "BULLISH":
        order_block_score = bull_ob["strength"] if bull_ob else 30
        fvg_score = 70 if bull_fvg else 35
    elif bias == "BEARISH":
        order_block_score = bear_ob["strength"] if bear_ob else 30
        fvg_score = 70 if bear_fvg else 35
    else:
        order_block_score = 40
        fvg_score = 40
    order_block_score = min(100, order_block_score)
    confidence = round(np.mean([structure_score, liquidity_score, order_block_score, fvg_score]), 1)
    return {"bias": bias, "structure_breaks": structure_breaks, "bull_ob": bull_ob, "bear_ob": bear_ob, "bull_fvg": bull_fvg, "bear_fvg": bear_fvg, "bsl": bsl, "ssl": ssl, "manipulation": manipulation, "manip_note": manip_note, "session": session_info, "asian_levels": asian_levels, "vol_label": vol_label, "vol_score": vol_score, "atr": atr_value, "displacements": displacements, "fib_levels": fib_levels, "ote": ote, "current_price": current_price, "scores": {"structure": round(structure_score, 1), "liquidity": round(liquidity_score, 1), "order_block": round(order_block_score, 1), "fvg": round(fvg_score, 1)}, "confidence": confidence}


# ============================================================
# AI Scanner (قاعدة: الترند صديقك - لا حظر، توجيه ذكي)
# ============================================================

def ai_scanner(df_h1_processed, df_m15_processed, df_m5_processed, model, scaler, ict_data_m15, ict_data_m5, cfg):
    result = {
        "trade_exists": False, "direction": None, "ai_conf_before_groq": 0.0,
        "experience_conf": 0.0, "experience_available": False, "experience_win_rate": None,
        "experience_sample": 0, "ict_confidence": None, "ict_bias": None,
        "h1_trend": None, "m15_bias": None, "m5_bias": None,
        "groq_called": False, "groq_available": False, "groq_agree": None,
        "groq_conf": None, "groq_reason": "", "final_confidence": 0.0, "status": "",
    }

    twelve_key_local = cfg["twelve_key"]
    use_groq_local = cfg["use_groq"]
    groq_key_local = cfg["groq_key"]
    groq_model_local = cfg["groq_model"]
    min_groq_conf_local = cfg["min_groq_conf"]
    min_conf_local = cfg["min_conf"]
    atr_mult_local = cfg["atr_mult"]
    risk_reward_local = cfg["risk_reward"]

    if not twelve_key_local:
        result["status"] = "النظام متوقف: يرجى إدخال مفتاح Twelve Data API."
        return result["status"], result

    conn = get_db_connection()
    try:
        df_act = pd.read_sql("SELECT * FROM active_trade WHERE id = 1", conn)
    finally:
        conn.close()

    if not df_act.empty:
        active = df_act.iloc[0]
        result["trade_exists"] = True
        result["direction"] = active["direction"]
        result["ai_conf_before_groq"] = float(active.get("ai_conf", 0) or 0)
        groq_conf = active.get("groq_conf", None)
        if pd.notna(groq_conf):
            result["groq_conf"] = float(groq_conf)
        result["groq_reason"] = str(active.get("groq_note", "") or "")
        result["final_confidence"] = result["groq_conf"] if result["groq_conf"] is not None else result["ai_conf_before_groq"]
        result["status"] = "الذكاء الاصطناعي يدير صفقة نشطة حالياً."
        return result["status"], result

    if df_h1_processed is None or df_h1_processed.empty:
        result["status"] = "لا توجد صفقة: بيانات السوق غير كافية."
        return result["status"], result

    if not model_is_ready(model, scaler):
        result["status"] = "الشبكة العصبية قيد التهيئة والتدريب."
        return result["status"], result

    last = df_h1_processed.iloc[-1]
    signal_bar_time = str(last.get("datetime", ""))
    last_signal_key = load_setting("last_signal_key", "")

    try:
        feature_values = last[FEATURES].astype(float).values.reshape(1, -1)
        if not np.isfinite(feature_values).all():
            result["status"] = "لا توجد صفقة: بيانات المؤشرات غير صالحة."
            return result["status"], result
        x_input = scaler.transform(feature_values)
        probabilities = model.predict_proba(x_input)[0]
        classes = np.asarray(model.classes_)
        best_index = int(np.argmax(probabilities))
        pred = int(classes[best_index])
        ai_conf = float(probabilities[best_index] * 100)
    except Exception as exc:
        result["status"] = f"تعذر تنفيذ تنبؤ الشبكة العصبية: {exc}"
        return result["status"], result

    result["ai_conf_before_groq"] = ai_conf
    direction = "BUY 🟢" if pred == 1 else "SELL 🔴"
    result["direction"] = direction

    # ============================================================
    # تطبيق مبدأ "الترند صديقك" (Trend is your friend)
    # ============================================================

    h1_last = df_h1_processed.iloc[-1]
    h1_trend = "BULLISH" if h1_last['ema_50'] > h1_last['ema_200'] else "BEARISH"
    result['h1_trend'] = h1_trend

    # إذا كان الترند هابطاً، نتجاهل إشارة الشراء ونتجه للبيع
    if h1_trend == "BEARISH" and pred == 1:
        direction = "SELL 🔴"
        pred = 0
        result["direction"] = direction
        result["status"] = "ترند هابط: تم توجيه البحث نحو فرص البيع."

    # إذا كان الترند صاعداً، نتجاهل إشارة البيع ونتجه للشراء
    elif h1_trend == "BULLISH" and pred == 0:
        direction = "BUY 🟢"
        pred = 1
        result["direction"] = direction
        result["status"] = "ترند صاعد: تم توجيه البحث نحو فرص الشراء."

    m15_bias = ict_data_m15.get("bias") if ict_data_m15 else "NEUTRAL"
    result['m15_bias'] = m15_bias
    m5_bias = ict_data_m5.get("bias") if ict_data_m5 else "NEUTRAL"
    result['m5_bias'] = m5_bias

    confirmation_score = 1.0
    if m15_bias != "NEUTRAL" and m15_bias != h1_trend:
        confirmation_score -= 0.35
    if m5_bias != "NEUTRAL" and m5_bias != h1_trend:
        confirmation_score -= 0.35

    # ============================================================
    # نهاية قاعدة الترند
    # ============================================================

    experience = get_experience_adjustment(direction, ai_conf)
    result["experience_available"] = experience["available"]
    result["experience_conf"] = experience["confidence"]
    result["experience_win_rate"] = experience["win_rate"]
    result["experience_sample"] = experience["sample"]
    working_conf = experience["confidence"] if experience["available"] else ai_conf

    if ict_data_m15 is not None:
        ict_bias = ict_data_m15.get("bias")
        ict_conf = ict_data_m15.get("confidence", 50.0)
        result["ict_bias"] = ict_bias
        result["ict_confidence"] = ict_conf
        ai_direction_bias = "BULLISH" if pred == 1 else "BEARISH"
        if ict_bias == ai_direction_bias:
            ict_component = ict_conf
        elif ict_bias == "NEUTRAL":
            ict_component = 50.0
        else:
            ict_component = 100.0 - ict_conf
        working_conf = working_conf * 0.85 + ict_component * 0.15

    working_conf = working_conf * confirmation_score

    groq_result = None
    if use_groq_local:
        result["groq_called"] = True
        groq_result = get_groq_review(direction, last, working_conf, groq_key_local, groq_model_local)
        if groq_result is not None:
            result["groq_available"] = True
            result["groq_agree"] = groq_result["agree"]
            result["groq_conf"] = groq_result["confidence"]
            result["groq_reason"] = groq_result["reason"]
            if not groq_result["agree"] or groq_result["confidence"] < min_groq_conf_local:
                penalty_ratio = 0.35 if not groq_result["agree"] else 0.15
                blended = working_conf * (1 - penalty_ratio) + groq_result["confidence"] * penalty_ratio
                result["final_confidence"] = round(blended, 1)
            else:
                result["final_confidence"] = round((working_conf + groq_result["confidence"]) / 2, 1)
        else:
            result["groq_available"] = False
            result["final_confidence"] = round(working_conf * 0.92, 1)
    else:
        result["final_confidence"] = round(working_conf, 1)

    if result["final_confidence"] < min_conf_local:
        result["status"] = f"🟡 الاتجاه ({direction}) مطابق للترند، لكن لم تكتمل شروط التأكيد (الثقة {result['final_confidence']:.1f}% < {min_conf_local}%)"
        return result["status"], result

    if signal_bar_time and last_signal_key == signal_bar_time:
        result["status"] = "لا توجد صفقة جديدة: تمت معالجة هذه الشمعة سابقاً."
        return result["status"], result

    curr = round(float(last["close"]), 2)
    atr_value = float(last["atr"])
    if not np.isfinite(atr_value) or atr_value <= 0:
        result["status"] = "لا توجد صفقة: قيمة ATR غير صالحة."
        return result["status"], result
    sl_distance = round(atr_value * atr_mult_local, 2)
    tp_distance = round(sl_distance * risk_reward_local, 2)
    if pred == 1:
        sl_price = round(curr - sl_distance, 2)
        tp_price = round(curr + tp_distance, 2)
    else:
        sl_price = round(curr + sl_distance, 2)
        tp_price = round(curr - tp_distance, 2)

    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM active_trade")
        c.execute("""INSERT INTO active_trade (id, symbol, direction, entry, sl, tp, time, features, ai_conf, groq_conf, groq_note, signal_bar_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (1, "XAU/USD", direction, curr, sl_price, tp_price, datetime.now(timezone.utc).isoformat(), json.dumps({feature: float(last[feature]) for feature in FEATURES}), ai_conf, result["groq_conf"] if result["groq_conf"] is not None else None, result["groq_reason"], signal_bar_time))
        conn.commit()
    finally:
        conn.close()
    save_setting("last_signal_key", signal_bar_time)

    groq_line = f"\nGroq: {result['groq_conf']:.1f}%" if result["groq_available"] else ""
    send_alert((f"🧠 AI Trade Signal\nDirection: {direction}\nEntry: ${curr}\nSL: ${sl_price}\nTP: ${tp_price}\nAI Raw: {ai_conf:.1f}%\nExperience+ICT: {working_conf:.1f}%{groq_line}\nFinal Confidence: {result['final_confidence']:.1f}%"))

    # إشعار Ntfy مستقل يؤكد فتح الصفقة فعلياً بعد اجتياز كل الفلاتر
    send_trade_confirmation_alert(
        direction=direction,
        entry=curr,
        sl=sl_price,
        tp=tp_price,
        final_confidence=result["final_confidence"],
        risk_reward_ratio=risk_reward_local,
        ai_conf=ai_conf,
        groq_conf=result["groq_conf"] if result["groq_available"] else None,
    )

    result["trade_exists"] = True
    result["status"] = (f"🟢 تم إطلاق الإشارة ({direction}) مطابقة للترند — AI: {ai_conf:.1f}% — Final: {result['final_confidence']:.1f}%")

    return result["status"], result


# ============================================================
# محرك الخلفية (Background Engine)
# ============================================================

@st.cache_resource
def _get_shared_engine_state():
    return {"data": {"last_twelve_error": None, "last_groq_error": None, "last_train_time": None, "last_update_time": None, "ai_result": None, "scan_msg": "", "ict_confidence": None, "snapshot": None, "engine_running": False, "engine_error": None}, "lock": threading.Lock()}


_shared_engine_state = _get_shared_engine_state()
APP_STATE = _shared_engine_state["data"]
_APP_STATE_LOCK = _shared_engine_state["lock"]


def APP_STATE_set(key, value):
    with _APP_STATE_LOCK:
        APP_STATE[key] = value


def APP_STATE_get(key, default=None):
    with _APP_STATE_LOCK:
        return APP_STATE.get(key, default)


def _read_worker_config():
    return {"twelve_key": load_setting("twelve_key", ""), "use_groq": load_setting("use_groq", "1") == "1", "groq_key": load_setting("groq_key", ""), "groq_model": load_setting("groq_model", "openai/gpt-oss-120b"), "min_groq_conf": float(load_setting("min_groq_conf", "50") or 50), "min_conf": float(load_setting("min_conf", "65") or 65), "atr_mult": float(load_setting("atr_mult", "1.5") or 1.5), "risk_reward": float(load_setting("risk_reward", "2.0") or 2.0)}


def _monitor_active_trade(df_live_processed, model, scaler, cfg):
    conn = get_db_connection()
    try:
        active_df = pd.read_sql("SELECT * FROM active_trade WHERE id = 1", conn)
    finally:
        conn.close()
    if active_df.empty or df_live_processed is None or df_live_processed.empty:
        return
    trade_row = active_df.iloc[0]
    last_row = df_live_processed.iloc[-1]
    is_buy_trade = "BUY" in str(trade_row["direction"])
    signal_bar_time = str(trade_row.get("signal_bar_time", "") or "")
    current_bar_time = str(last_row.get("datetime", ""))
    can_monitor_trade = current_bar_time != signal_bar_time
    if can_monitor_trade and model_is_ready(model, scaler):
        try:
            x_current = scaler.transform(last_row[FEATURES].astype(float).values.reshape(1, -1))
            current_probs = model.predict_proba(x_current)[0]
            classes = np.asarray(model.classes_)
            current_index = int(np.argmax(current_probs))
            current_pred = int(classes[current_index])
            current_conf = float(current_probs[current_index] * 100)
            reversal_detected = False
            if is_buy_trade and current_pred == 0 and current_conf >= (cfg["min_conf"] - 5):
                reversal_detected = True
            elif not is_buy_trade and current_pred == 1 and current_conf >= (cfg["min_conf"] - 5):
                reversal_detected = True
            if reversal_detected:
                send_alert((f"⚠️ تنبيه من الشبكة العصبية: رصد انعكاس محتمل ضد الصفقة ({trade_row['direction']}) بقوة ({current_conf:.1f}%)."), "🚨 AI Reversal Warning")
        except Exception:
            pass
    if not can_monitor_trade:
        return
    high_price = float(last_row["high"])
    low_price = float(last_row["low"])
    sl_price = float(trade_row["sl"])
    tp_price = float(trade_row["tp"])
    hit_sl = False
    hit_tp = False
    if is_buy_trade:
        if low_price <= sl_price:
            hit_sl = True
        if high_price >= tp_price:
            hit_tp = True
    else:
        if high_price >= sl_price:
            hit_sl = True
        if low_price <= tp_price:
            hit_tp = True
    both_hit = hit_sl and hit_tp
    if both_hit:
        hit_tp = False
        hit_sl = True
    if not (hit_sl or hit_tp):
        return
    win_value = 1 if hit_tp else 0
    note_str = "SL and TP touched in same candle; conservative SL outcome." if both_hit else ("AI Target Reached" if hit_tp else "AI Stop Loss Hit")
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("""INSERT INTO trades (date, symbol, direction, entry, sl, tp, win, note, claude_conf, claude_note, groq_conf, groq_note, ai_conf_before_groq, ai_conf_after_groq) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (str(datetime.now(timezone.utc).date()), trade_row["symbol"], trade_row["direction"], float(trade_row["entry"]), float(trade_row["sl"]), float(trade_row["tp"]), win_value, note_str, None, None, float(trade_row.get("groq_conf")) if pd.notna(trade_row.get("groq_conf")) else None, str(trade_row.get("groq_note", "") or ""), float(trade_row.get("ai_conf", 0) or 0), float(trade_row.get("groq_conf")) if pd.notna(trade_row.get("groq_conf")) else float(trade_row.get("ai_conf", 0) or 0)))
        c.execute("DELETE FROM active_trade WHERE id = 1")
        conn.commit()
    finally:
        conn.close()
    if model_is_ready(model, scaler):
        try:
            feat_dict = json.loads(trade_row.get("features") or "{}")
            if feat_dict:
                x_vec = np.array([[float(feat_dict.get(f, np.nan)) for f in FEATURES]], dtype=float)
                if np.isfinite(x_vec).all():
                    x_scaled = scaler.transform(x_vec)
                    outcome_up = (is_buy_trade and win_value == 1) or ((not is_buy_trade) and win_value == 0)
                    label = np.array([1 if outcome_up else 0])
                    model.partial_fit(x_scaled, label, classes=np.asarray(model.classes_))
                    joblib.dump(model, MODEL_FILE)
                    APP_STATE_set("last_train_time", datetime.now(timezone.utc).isoformat())
        except Exception:
            pass
    send_alert(f"Closed {trade_row['symbol']} {trade_row['direction']} -> {note_str}", "🧠 AI Trade Settled")


def _maybe_send_heartbeat(ai_result, snapshot):
    """
    ينبض المحرك دورياً (كل HEARTBEAT_INTERVAL_SECONDS) عبر Ntfy حتى لو كان
    الموقع مغلقاً عند المستخدم. هذا يثبت أن العملية الخلفية لا تزال حية،
    وإذا توقفت النبضات فهذا يعني أن التطبيق دخل في وضع السكون على المنصة.
    """
    last_heartbeat_raw = load_setting("last_heartbeat_time", "")
    now = datetime.now(timezone.utc)
    should_send = True
    if last_heartbeat_raw:
        try:
            last_heartbeat = datetime.fromisoformat(last_heartbeat_raw)
            if (now - last_heartbeat).total_seconds() < HEARTBEAT_INTERVAL_SECONDS:
                should_send = False
        except Exception:
            should_send = True
    if not should_send:
        return
    price_txt = f"${snapshot['close']}" if snapshot and snapshot.get("close") else "—"
    trade_txt = ai_result.get("direction") if ai_result and ai_result.get("trade_exists") else "لا توجد صفقة نشطة حالياً"
    send_alert(
        f"❤️ المحرك يعمل بشكل طبيعي\nآخر سعر: {price_txt}\nالحالة: {trade_txt}\nالوقت: {now.strftime('%Y-%m-%d %H:%M UTC')}",
        title="❤️ Engine Heartbeat",
    )
    save_setting("last_heartbeat_time", now.isoformat())


def _engine_cycle():
    cfg = _read_worker_config()
    twelve_key_local = cfg["twelve_key"]
    maybe_spawn_training(twelve_key_local)
    if not twelve_key_local:
        APP_STATE_set("scan_msg", "النظام متوقف: يرجى إدخال مفتاح Twelve Data API.")
        return
    model, scaler = load_current_model()
    df_live_raw_h1 = fetch_twelve_series(twelve_key_local, symbol="XAU/USD", interval="1h", outputsize=LIVE_OUTPUT_SIZE)
    df_live_raw_m15 = fetch_twelve_series(twelve_key_local, symbol="XAU/USD", interval="15min", outputsize=LIVE_OUTPUT_SIZE)
    df_live_raw_m5 = fetch_twelve_series(twelve_key_local, symbol="XAU/USD", interval="5min", outputsize=LIVE_OUTPUT_SIZE)
    
    df_live_h1 = keep_closed_candles(df_live_raw_h1, interval_hours=1)
    df_live_m15 = keep_closed_candles(df_live_raw_m15, interval_hours=0.25)
    df_live_m5 = keep_closed_candles(df_live_raw_m5, interval_hours=5/60)
    
    df_h1_processed = apply_deep_indicators(df_live_h1)
    df_m15_processed = apply_deep_indicators(df_live_m15)
    df_m5_processed = apply_deep_indicators(df_live_m5)
    
    ict_data_m15 = run_ict_engine(df_m15_processed, swing_lookback=ICT_SWING_LOOKBACK, ob_mult=ICT_OB_DISPLACEMENT_MULT) if not df_m15_processed.empty else None
    ict_data_m5 = run_ict_engine(df_m5_processed, swing_lookback=ICT_SWING_LOOKBACK, ob_mult=ICT_OB_DISPLACEMENT_MULT) if not df_m5_processed.empty else None
    
    scan_msg, ai_result = ai_scanner(df_h1_processed, df_m15_processed, df_m5_processed, model, scaler, ict_data_m15, ict_data_m5, cfg)
    _monitor_active_trade(df_h1_processed, model, scaler, cfg)
    
    snapshot = None
    if not df_h1_processed.empty:
        last_snapshot = df_h1_processed.iloc[-1]
        snapshot = {"close": float(last_snapshot["close"]), "rsi": float(last_snapshot["rsi"]), "ema_50": float(last_snapshot["ema_50"]), "ema_200": float(last_snapshot["ema_200"]), "atr": float(last_snapshot["atr"]), "datetime": str(last_snapshot.get("datetime", ""))}

    with _APP_STATE_LOCK:
        APP_STATE["ai_result"] = ai_result
        APP_STATE["scan_msg"] = scan_msg
        APP_STATE["ict_confidence"] = ict_data_m15.get("confidence") if ict_data_m15 else None
        APP_STATE["snapshot"] = snapshot
        APP_STATE["last_update_time"] = datetime.now(timezone.utc).isoformat()
        APP_STATE["engine_error"] = None

    _maybe_send_heartbeat(ai_result, snapshot)


def background_worker_loop():
    APP_STATE_set("engine_running", True)
    while True:
        try:
            _engine_cycle()
        except Exception as exc:
            APP_STATE_set("engine_error", f"{exc}\n{traceback.format_exc()}")
        time.sleep(WORKER_LOOP_SECONDS)


def ensure_background_worker_started():
    for t in threading.enumerate():
        if t.name == "bg_worker_loop":
            return
    worker = threading.Thread(target=background_worker_loop, daemon=True, name="bg_worker_loop")
    worker.start()


ensure_background_worker_started()


# ============================================================
# UI
# ============================================================

st.title("🧠 نظام التداول العميق — XAU/USD")

success_count = get_successful_trades_count()
total_count = get_total_trades_count()

if not twelve_key:
    st.warning("⚠️ النظام نائم: أدخل مفتاح Twelve Data API في الشريط الجانبي.")

if os.path.exists(TRAINING_LOCK_FILE):
    st.info("🧠 النموذج يتدرب حالياً في الخلفية على البيانات التاريخية.")

last_update = APP_STATE_get("last_update_time")
if last_update:
    st.caption(f"🟢 المحرك الخلفي يعمل — آخر تحديث: {last_update}")
else:
    st.caption("🟡 المحرك الخلفي بدأ للتو، بانتظار أول دورة تحليل...")

with st.expander("🔧 حالة المحرك (تشخيص)"):
    diag_model, diag_scaler = load_current_model()
    model_ready_now = model_is_ready(diag_model, diag_scaler)
    d1, d2 = st.columns(2)
    d1.write("🔑 مفتاح Twelve Data: " + ("✅ موجود" if twelve_key else "❌ غير مُدخل"))
    d1.write("🧠 حالة النموذج: " + ("✅ مُدرَّب وجاهز" if model_ready_now else "⏳ غير جاهز بعد"))
    d1.write("🔒 قفل تدريب نشط الآن: " + ("نعم" if os.path.exists(TRAINING_LOCK_FILE) else "لا"))
    last_train_time = APP_STATE_get("last_train_time")
    d2.write(f"🕒 آخر تدريب ناجح: {last_train_time or 'لم يحدث بعد'}")
    d2.write(f"🔄 آخر دورة تحليل: {last_update or 'لم تبدأ بعد'}")
    d2.write(f"📶 عدد صفقات في السجل: {get_total_trades_count()}")

ai_result = APP_STATE_get("ai_result") or {}
scan_msg = APP_STATE_get("scan_msg") or ""

st.markdown("### 🎯 حالة الصفقة")
trade_exists = bool(ai_result.get("trade_exists", False))
if trade_exists:
    direction = ai_result.get("direction")
    if direction and "BUY" in str(direction): status_class, status_text = "trade-buy", "🟢 توجد صفقة"
    elif direction and "SELL" in str(direction): status_class, status_text = "trade-sell", "🔴 توجد صفقة"
    else: status_class, status_text = "trade-neutral", "🟡 توجد إشارة"
else:
    status_class, status_text = "trade-neutral", "⚪ لا توجد صفقة"
render_html(f'<div class="trade-status-card"><div class="trade-status-title">TRADE STATUS</div><div class="trade-status-value {status_class}">{status_text}</div></div>')

h1_trend = ai_result.get("h1_trend")
m15_bias = ai_result.get("m15_bias")
m5_bias = ai_result.get("m5_bias")
if h1_trend or m15_bias or m5_bias:
    display_text = ""
    if h1_trend: display_text += f"📊 H1 Trend: **{h1_trend}**  |  "
    if m15_bias: display_text += f"⚙️ M15 Setup: **{m15_bias}**  |  "
    if m5_bias: display_text += f"✅ M5 Confirm: **{m5_bias}**"
    st.info(display_text)

st.markdown("### 🧠 مستوى الثقة")
final_conf_raw = float(ai_result.get("final_confidence", 0) or 0)
raw_ai_conf = float(ai_result.get("ai_conf_before_groq", 0) or 0)

if final_conf_raw > 0:
    final_conf = final_conf_raw
    confidence_note = "كل المؤشرات (ICT, Experience, Groq) مدمجة في هذا الرقم"
elif ai_result.get("h1_trend"):
    final_conf = raw_ai_conf
    confidence_note = "ثقة مؤقتة بانتظار تأكيد M15/M5 للاتجاه"
else:
    final_conf = raw_ai_conf
    confidence_note = "بانتظار أول تنبؤ من الشبكة العصبية"

snapshot = APP_STATE_get("snapshot")
last_price_txt = ""
if snapshot and snapshot.get("close"):
    last_price_txt = f" | آخر سعر: ${snapshot.get('close')}"

render_html(
    f"""
<div class="ai-level-card">
    <div class="ai-level-title">AI CONFIDENCE LEVEL</div>
    <div class="ai-level-value">{final_conf:.1f}%</div>
    <div class="ai-level-sub">
        Trades: {total_count} | Wins: {success_count} |
        {confidence_note} {last_price_txt}
    </div>
</div>
"""
)

if ai_result.get("direction"):
    st.info(f"📌 الاتجاه المقترح (وفقاً للترند): **{ai_result['direction']}**")

if ai_result.get("groq_called"):
    if ai_result.get("groq_available"):
        groq_conf_val = ai_result.get("groq_conf")
        groq_conf_txt = f"{groq_conf_val:.1f}%" if groq_conf_val is not None else "—"
        if ai_result.get("groq_agree"): st.success(f"✅ Groq وافق على الإشارة — ثقة Groq: {groq_conf_txt}")
        else: st.warning(f"❌ Groq لم يوافق على الإشارة — ثقة Groq: {groq_conf_txt}")
        if ai_result.get("groq_reason"): st.caption(f"🧠 رأي Groq: {ai_result['groq_reason']}")
    else:
        st.warning("🟠 تم استدعاء Groq لكن لم تصل استجابة صالحة منه هذه الدورة.")
        groq_error_detail = APP_STATE_get("last_groq_error")
        if groq_error_detail: st.code(groq_error_detail)

if scan_msg:
    st.caption(f"🔍 {scan_msg}")

twelve_error = APP_STATE_get("last_twelve_error")
if twelve_error and twelve_key:
    st.error(f"⚠️ Twelve Data: {twelve_error}")

engine_error = APP_STATE_get("engine_error")
if engine_error:
    st.warning("⚠️ حدث خطأ داخلي في المحرك الخلفي، سيُعاد المحاولة تلقائيًا في الدورة القادمة.")

conn = get_db_connection()
try:
    df_active = pd.read_sql("SELECT * FROM active_trade WHERE id = 1", conn)
finally:
    conn.close()
if not df_active.empty and twelve_key:
    active_trade = df_active.iloc[0]
    st.warning(f"""🔒 **صفقة نشطة حالياً**\n\nالاتجاه: {active_trade['direction']}\nالدخول: ${active_trade['entry']}\nSL: ${active_trade['sl']}\nTP: ${active_trade['tp']}\nشمعة الإشارة: {active_trade.get('signal_bar_time', '')}""")

st.markdown("### 📊 سجل الصفقات")
conn = get_db_connection()
try:
    df_log = pd.read_sql("SELECT * FROM trades ORDER BY id DESC", conn)
finally:
    conn.close()
if not df_log.empty:
    win_rate = df_log["win"].sum() / len(df_log) * 100
    m1, m2, m3 = st.columns(3)
    m1.metric("إجمالي الصفقات", len(df_log))
    m2.metric("نسبة الربح", f"{win_rate:.1f}%")
    m3.metric("الصفقات الرابحة", int(df_log["win"].sum()))
    st.dataframe(df_log[["date", "direction", "entry", "sl", "tp", "win", "note", "ai_conf_before_groq", "groq_conf"]], use_container_width=True)
else:
    st.info("لا توجد صفقات مغلقة مسجلة حتى الآن.")

st_autorefresh(interval=20000, key="deep_ai_ui_refresh")
