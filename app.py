import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import json
import os
import sqlite3
import threading
import time
import traceback
import requests

from datetime import datetime, timezone, timedelta
from streamlit_autorefresh import st_autorefresh
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# إعدادات الصفحة
# ============================================================
st.set_page_config(page_title="XAU/USD Deep AI Engine", layout="wide", page_icon="🧠")

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
# CSS مخصص (يجمع بين التصميمين)
# ============================================================
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; }
    .stApp { background-color: #0a0e17; color: #f3f4f6; }
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }

    /* بطاقات النظام العميق */
    .ai-level-card { background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); padding: 30px; border-radius: 20px; text-align: center; border: 1px solid #3b82f6; box-shadow: 0 0 25px rgba(59, 130, 246, 0.4); margin-bottom: 20px; }
    .ai-level-title { font-size: 1.2rem; color: #93c5fd; font-weight: 700; margin-bottom: 10px; }
    .ai-level-value { font-size: 4rem; font-weight: 900; color: #fbbf24; line-height: 1; }
    .ai-level-sub { font-size: 1rem; color: #94a3b8; margin-top: 10px; }

    /* بطاقات الحالة */
    .trade-status-card { background: linear-gradient(135deg, #111827 0%, #0f172a 100%); border: 1px solid #334155; border-radius: 18px; padding: 22px; margin-bottom: 20px; text-align: center; }
    .trade-status-title { font-size: 1rem; color: #94a3b8; font-weight: 700; }
    .trade-status-value { font-size: 2.3rem; font-weight: 900; margin-top: 8px; }
    .trade-buy { color: #22c55e; }
    .trade-sell { color: #ef4444; }
    .trade-neutral { color: #94a3b8; }

    /* بطاقات الثقة */
    .confidence-card { background: #0f172a; border: 1px solid #1e293b; border-radius: 14px; padding: 18px; text-align: center; }
    .confidence-title { color: #94a3b8; font-size: 0.85rem; }
    .confidence-value { color: #fbbf24; font-size: 2rem; font-weight: 900; }

    /* بطاقات ICT */
    .ict-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 20px; border-radius: 16px; text-align: center; border: 1px solid #334155; margin-bottom: 14px; }
    .ict-title { font-size: 0.85rem; color: #93c5fd; font-weight: 700; letter-spacing: 1px; margin-bottom: 8px; text-transform: uppercase; }
    .ict-value { font-size: 1.8rem; font-weight: 900; color: #fbbf24; line-height: 1.1; }
    .ict-sub { font-size: 0.85rem; color: #94a3b8; margin-top: 6px; }
    .ict-bullish { color: #22c55e !important; }
    .ict-bearish { color: #ef4444 !important; }
    .ict-neutral { color: #94a3b8 !important; }
    .ict-row { background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; }

    /* بطاقات المحاكي اليدوي */
    .manual-trade-card { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 12px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); color: #1f2937; }
    .manual-trade-card h4 { color: #1f2937; }
    .manual-trade-card .price { font-size: 1.8rem; font-weight: 700; color: #2563eb; }
</style>
""")

# ============================================================
# الثوابت وقاعدة البيانات
# ============================================================
DB_FILE = "xau_deep_ai.db"
MODEL_FILE = "xau_deep_mlp_v2.pkl"
SCALER_FILE = "xau_deep_scaler_v2.pkl"
TRAINING_LOCK_FILE = "training.lock"
TRAINING_OUTPUT_SIZE = 5000
LIVE_OUTPUT_SIZE = 220
TRAINING_LOCK_MAX_AGE = 60 * 60
FEATURES = ["atr", "ema_50", "ema_200", "rsi"]

def get_db_connection():
    return sqlite3.connect(DB_FILE, timeout=20, check_same_thread=False)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, symbol TEXT, direction TEXT, entry REAL, sl REAL, tp REAL,
        win INTEGER, note TEXT, claude_conf REAL, claude_note TEXT, groq_conf REAL, groq_note TEXT,
        ai_conf_before_groq REAL, ai_conf_after_groq REAL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS active_trade (
        id INTEGER PRIMARY KEY, symbol TEXT, direction TEXT, entry REAL, sl REAL, tp REAL,
        time TEXT, features TEXT, ai_conf REAL, groq_conf REAL, groq_note TEXT, signal_bar_time TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS manual_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, symbol TEXT, type TEXT, amount REAL, price REAL
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
# إعدادات الشريط الجانبي
# ============================================================
st.sidebar.header("⚙️ إعدادات الذكاء الاصطناعي")
twelve_secret = st.secrets.get("TWELVE_DATA_API_KEY", "")
twelve_key = st.sidebar.text_input("مفتاح Twelve Data API", type="password", value=twelve_secret if twelve_secret else st.session_state.get("twelve_key", ""))
st.session_state["twelve_key"] = twelve_key
ntfy_channel = st.sidebar.text_input("قناة Ntfy للتنبيهات", value=load_setting("ntfy", "xau_deep_channel"))
save_setting("ntfy", ntfy_channel)

st.sidebar.markdown("---")
st.sidebar.header("🧠 الرأي الثاني (Groq)")
use_groq = st.sidebar.checkbox("تفعيل مراجعة Groq قبل فتح الصفقة", value=load_setting("use_groq", "1") == "1")
save_setting("use_groq", "1" if use_groq else "0")
groq_secret = st.secrets.get("GROQ_API_KEY", "")
groq_key = st.sidebar.text_input("مفتاح Groq API", type="password", value=groq_secret if groq_secret else st.session_state.get("groq_key", ""))
st.session_state["groq_key"] = groq_key
groq_model = st.sidebar.text_input("اسم نموذج Groq", value=load_setting("groq_model", "llama-3.3-70b-versatile"))
save_setting("groq_model", groq_model)
min_groq_conf = st.sidebar.slider("أدنى ثقة مطلوبة من Groq (%)", 40, 95, 60, 1)

st.sidebar.markdown("---")
st.sidebar.header("🎯 إدارة المخاطر")
atr_mult = st.sidebar.slider("معامل الوقف ATR", 1.0, 3.0, 1.5, 0.1)
risk_reward = st.sidebar.slider("نسبة العائد R:R", 1.5, 4.0, 2.0, 0.5)
min_conf = st.sidebar.slider("أدنى ثقة مطلوبة من الشبكة العصبية (%)", 60, 95, 75, 1)

st.sidebar.markdown("---")
st.sidebar.header("🧭 إعدادات لوحة ICT / Smart Money")
show_ict_tab = st.sidebar.checkbox("إظهار تبويب ICT / Smart Money", value=True)
swing_lookback = st.sidebar.slider("حساسية القمم/القيعان", 2, 8, 3, 1)
ob_displacement_mult = st.sidebar.slider("معامل قوة الاندفاع", 0.8, 2.5, 1.2, 0.1)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 إعادة تدريب النموذج من الصفر"):
    for file_path in (MODEL_FILE, SCALER_FILE, TRAINING_LOCK_FILE):
        if os.path.exists(file_path):
            try: os.remove(file_path)
            except OSError: pass
    try: st.cache_data.clear()
    except Exception: pass
    st.rerun()

# ============================================================
# دوال تنبيهات Ntfy
# ============================================================
def send_alert(msg, title="🧠 Deep AI Alert"):
    if not ntfy_channel: return
    channel = ntfy_channel.strip().split("/")[-1]
    if not channel: return
    try:
        requests.post(f"https://ntfy.sh/{channel}", data=msg.encode("utf-8"),
                      headers={"Title": title, "Priority": "high"}, timeout=5)
    except Exception: pass

# ============================================================
# جلب البيانات من Twelve Data
# ============================================================
def fetch_twelve_series(api_key, symbol="XAU/USD", interval="1h", outputsize=150):
    if not api_key: return pd.DataFrame()
    try:
        params = {"symbol": symbol, "interval": interval, "outputsize": min(int(outputsize), 5000), "timezone": "UTC", "apikey": api_key}
        response = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=10)
        response.raise_for_status()
        result = response.json()
        if "values" not in result:
            st.session_state["last_twelve_error"] = result.get("message", "استجابة غير متوقعة من Twelve Data.")
            return pd.DataFrame()
        values = result["values"]
        if not values: return pd.DataFrame()
        df = pd.DataFrame(values)
        required = ["datetime", "open", "high", "low", "close"]
        if not all(col in df.columns for col in required):
            st.session_state["last_twelve_error"] = "بيانات Twelve Data لا تحتوي الأعمدة المطلوبة."
            return pd.DataFrame()
        df = df[required].copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["datetime", "open", "high", "low", "close"], inplace=True)
        df.sort_values("datetime", inplace=True)
        df.drop_duplicates(subset=["datetime"], keep="last", inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.session_state["last_twelve_error"] = None
        return df
    except Exception as exc:
        st.session_state["last_twelve_error"] = f"تعذّر الاتصال بـ Twelve Data: {exc}"
        return pd.DataFrame()

def keep_closed_candles(df, interval_hours=1):
    if df is None or df.empty or "datetime" not in df.columns: return pd.DataFrame()
    df = df.copy()
    now_utc = datetime.now(timezone.utc)
    candle_delta = timedelta(hours=interval_hours)
    mask = (df["datetime"] + candle_delta <= pd.Timestamp(now_utc))
    closed = df.loc[mask].copy()
    return closed.reset_index(drop=True)

# ============================================================
# المؤشرات الفنية
# ============================================================
def apply_deep_indicators(df):
    if df is None or df.empty: return pd.DataFrame()
    if len(df) < 210: return pd.DataFrame()
    df = df.copy()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df["rsi"] = 100 - (100 / (1 + rs))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURES, inplace=True)
    return df.reset_index(drop=True)

# ============================================================
# فحص جاهزية النموذج
# ============================================================
def model_is_ready(model_obj, scaler_obj):
    if model_obj is None or scaler_obj is None: return False
    if not hasattr(scaler_obj, "mean_") or not hasattr(scaler_obj, "scale_"): return False
    if not hasattr(model_obj, "classes_"): return False
    try:
        if len(model_obj.classes_) < 2: return False
        if getattr(model_obj, "n_features_in_", len(FEATURES)) != len(FEATURES): return False
    except Exception:
        return False
    return True

# ============================================================
# تدريب النموذج في الخلفية
# ============================================================
def _background_train_and_save(api_key):
    try:
        df_train = fetch_twelve_series(api_key, symbol="XAU/USD", interval="1h", outputsize=TRAINING_OUTPUT_SIZE)
        df_train = keep_closed_candles(df_train, interval_hours=1)
        df_train = apply_deep_indicators(df_train)
        if df_train.empty or len(df_train) < 250: return
        future_close = df_train["close"].shift(-1)
        valid_mask = future_close.notna()
        train_df = df_train.loc[valid_mask].copy()
        if len(train_df) < 100: return
        X = train_df[FEATURES].astype(float).values
        y = np.where(train_df["close"].shift(-1) > train_df["close"], 1, 0)
        X = X[:-1]
        y = y[:-1]
        if len(X) < 100 or len(y) < 100: return
        if len(np.unique(y)) < 2: return
        new_scaler = StandardScaler()
        X_scaled = new_scaler.fit_transform(X)
        new_model = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.0001,
                                  learning_rate_init=0.001, max_iter=1000, early_stopping=True,
                                  validation_fraction=0.15, n_iter_no_change=30, random_state=42)
        new_model.fit(X_scaled, y)
        if not model_is_ready(new_model, new_scaler): return
        model_tmp = MODEL_FILE + ".tmp"
        scaler_tmp = SCALER_FILE + ".tmp"
        joblib.dump(new_model, model_tmp)
        joblib.dump(new_scaler, scaler_tmp)
        os.replace(model_tmp, MODEL_FILE)
        os.replace(scaler_tmp, SCALER_FILE)
    except Exception:
        pass
    finally:
        if os.path.exists(TRAINING_LOCK_FILE):
            try: os.remove(TRAINING_LOCK_FILE)
            except OSError: pass

def clean_stale_training_lock():
    if not os.path.exists(TRAINING_LOCK_FILE): return
    try:
        age = time.time() - os.path.getmtime(TRAINING_LOCK_FILE)
        if age > TRAINING_LOCK_MAX_AGE:
            os.remove(TRAINING_LOCK_FILE)
    except Exception:
        pass

clean_stale_training_lock()

def train_deep_model(api_key):
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            loaded_model = joblib.load(MODEL_FILE)
            loaded_scaler = joblib.load(SCALER_FILE)
            if model_is_ready(loaded_model, loaded_scaler):
                return loaded_model, loaded_scaler
        except Exception:
            pass
    if api_key and not os.path.exists(TRAINING_LOCK_FILE):
        try:
            with open(TRAINING_LOCK_FILE, "x", encoding="utf-8") as f:
                f.write(datetime.now(timezone.utc).isoformat())
            thread = threading.Thread(target=_background_train_and_save, args=(api_key,), daemon=True)
            thread.start()
        except FileExistsError:
            pass
        except Exception:
            pass
    return None, None

model, scaler = train_deep_model(twelve_key)

# ============================================================
# طبقة الخبرة (Experience Layer)
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
    smoothed_rate = ((wins + 5.0) / (n + 10.0)) * 100
    adjusted = ai_conf * 0.70 + smoothed_rate * 0.30
    return {"available": True, "confidence": round(float(adjusted), 1), "win_rate": round(smoothed_rate, 1), "sample": n}

# ============================================================
# مراجعة Groq (الرأي الثاني)
# ============================================================
def get_groq_review(direction, last_row, ai_conf, api_key, model_name):
    if not api_key: return None
    try:
        prompt = (f"أنت محلل فني مساعد لصفقة محتملة على XAU/USD.\n"
                  f"الاتجاه المقترح: {direction}.\n"
                  f"ثقة نموذج AI الخام: {ai_conf:.1f}%.\n"
                  f"ATR={last_row['atr']:.2f}.\n"
                  f"EMA50={last_row['ema_50']:.2f}.\n"
                  f"EMA200={last_row['ema_200']:.2f}.\n"
                  f"RSI={last_row['rsi']:.1f}.\n"
                  f"السعر={last_row['close']:.2f}.\n\n"
                  "راجع الاتجاه بشكل مستقل. لا تفترض أن نموذج AI صحيح. أعط رأياً محافظاً.\n"
                  "يجب أن يكون الرد JSON فقط بهذا الشكل:\n"
                  '{"agree": true, "confidence": 0, "reason": "..."}')
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model_name, "temperature": 0, "max_completion_tokens": 300,
                  "messages": [{"role": "system", "content": "أنت محلل فني محافظ. أعد JSON صالح فقط."},
                               {"role": "user", "content": prompt}],
                  "response_format": {"type": "json_object"}}, timeout=20)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices: return None
        message = choices[0].get("message", {})
        text = message.get("content", "")
        if not text: return None
        cleaned = str(text).replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        agree = bool(parsed.get("agree", False))
        confidence = float(parsed.get("confidence", 0))
        confidence = max(0, min(100, confidence))
        reason = str(parsed.get("reason", ""))
        return {"agree": agree, "confidence": confidence, "reason": reason}
    except Exception:
        return None

# ============================================================
# الماسح الضوئي للذكاء الاصطناعي
# ============================================================
def ai_scanner(df_live_processed, ict_data=None):
    result = {
        "trade_exists": False, "direction": None, "ai_conf_before_groq": 0.0,
        "experience_conf": 0.0, "experience_available": False, "experience_win_rate": None,
        "experience_sample": 0, "groq_called": False, "groq_available": False, "groq_agree": None,
        "groq_conf": None, "groq_reason": "", "final_confidence": 0.0, "status": "",
        "ict_used": False, "ict_bias": None, "ict_confidence": None, "ict_adjustment": 0.0
    }
    if not twelve_key:
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
    if df_live_processed is None or df_live_processed.empty:
        result["status"] = "لا توجد صفقة: بيانات السوق غير كافية."
        return result["status"], result
    if not model_is_ready(model, scaler):
        result["status"] = "الشبكة العصبية قيد التهيئة والتدريب."
        return result["status"], result
    last = df_live_processed.iloc[-1]
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
    if ai_conf < min_conf:
        result["status"] = f"لا توجد صفقة: ثقة AI ({ai_conf:.1f}%) أقل من المطلوب ({min_conf}%)."
        return result["status"], result
    direction = "BUY 🟢" if pred == 1 else "SELL 🔴"
    result["direction"] = direction
    experience = get_experience_adjustment(direction, ai_conf)
    result["experience_available"] = experience["available"]
    result["experience_conf"] = experience["confidence"]
    result["experience_win_rate"] = experience["win_rate"]
    result["experience_sample"] = experience["sample"]
    working_conf = experience["confidence"] if experience["available"] else ai_conf
    ict_adjustment = 0.0
    ict_bias = None
    ict_confidence = None
    ict_used = False
    if ict_data is not None:
        ict_bias = ict_data.get("bias", "NEUTRAL")
        ict_confidence = ict_data.get("confidence", 0)
        ict_used = True
        ai_direction = "BULLISH" if "BUY" in direction else "BEARISH"
        if ict_bias == ai_direction:
            ict_adjustment = min(5.0, (ict_confidence - 50) * 0.1)
            working_conf += ict_adjustment
        elif ict_bias != "NEUTRAL":
            ict_adjustment = -min(5.0, (ict_confidence - 50) * 0.1)
            working_conf += ict_adjustment
        working_conf = max(0, min(100, working_conf))
        result["ict_used"] = ict_used
        result["ict_bias"] = ict_bias
        result["ict_confidence"] = ict_confidence
        result["ict_adjustment"] = round(ict_adjustment, 2)
        if working_conf < min_conf:
            result["final_confidence"] = working_conf
            result["status"] = "🟡 إشارة AI موجودة، لكن ICT خفّض الثقة دون الحد الأدنى."
            return result["status"], result
    groq_result = None
    if use_groq:
        result["groq_called"] = True
        groq_result = get_groq_review(direction, last, working_conf, groq_key, groq_model)
        if groq_result is not None:
            result["groq_available"] = True
            result["groq_agree"] = groq_result["agree"]
            result["groq_conf"] = groq_result["confidence"]
            result["groq_reason"] = groq_result["reason"]
            if not groq_result["agree"] or groq_result["confidence"] < min_groq_conf:
                result["final_confidence"] = min(working_conf, groq_result["confidence"])
                result["status"] = "🟡 إشارة AI موجودة، لكن Groq لم يعتمد الصفقة."
                return result["status"], result
            result["final_confidence"] = round((working_conf + groq_result["confidence"]) / 2, 1)
        else:
            result["groq_available"] = False
            result["final_confidence"] = working_conf
            result["status"] = "🟠 AI أعطى إشارة، لكن Groq لم يستجب. تم منع الصفقة تحفظاً."
            return result["status"], result
    else:
        result["final_confidence"] = working_conf
    if signal_bar_time and last_signal_key == signal_bar_time:
        result["status"] = "لا توجد صفقة جديدة: تمت معالجة هذه الشمعة سابقاً."
        return result["status"], result
    curr = round(float(last["close"]), 2)
    atr_value = float(last["atr"])
    if not np.isfinite(atr_value) or atr_value <= 0:
        result["status"] = "لا توجد صفقة: قيمة ATR غير صالحة."
        return result["status"], result
    sl_distance = round(atr_value * atr_mult, 2)
    tp_distance = round(sl_distance * risk_reward, 2)
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
        c.execute("""INSERT INTO active_trade
            (id, symbol, direction, entry, sl, tp, time, features, ai_conf, groq_conf, groq_note, signal_bar_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (1, "XAU/USD", direction, curr, sl_price, tp_price,
             datetime.now(timezone.utc).isoformat(),
             json.dumps({feature: float(last[feature]) for feature in FEATURES}),
             ai_conf, (result["groq_conf"] if result["groq_conf"] is not None else None),
             result["groq_reason"], signal_bar_time))
        conn.commit()
    finally:
        conn.close()
    save_setting("last_signal_key", signal_bar_time)
    groq_line = ""
    if result["groq_available"]:
        groq_line = f"\nGroq: {result['groq_conf']:.1f}%"
    ict_line = ""
    if ict_used:
        ict_line = f"\nICT: {ict_bias} ({ict_confidence:.1f}%) تعديل: {ict_adjustment:+.1f}%"
    send_alert((f"🧠 AI Trade Signal\nDirection: {direction}\nEntry: ${curr}\nSL: ${sl_price}\nTP: ${tp_price}\n"
                f"AI Raw: {ai_conf:.1f}%\nExperience: {working_conf:.1f}%{ict_line}{groq_line}\n"
                f"Final Confidence: {result['final_confidence']:.1f}%"))
    result["trade_exists"] = True
    result["status"] = f"🟢 تم إطلاق الإشارة ({direction}) — AI: {ai_conf:.1f}% — Final: {result['final_confidence']:.1f}%"
    return result["status"], result

# (يتبع باقي دوال ICT وواجهة المستخدم...)
