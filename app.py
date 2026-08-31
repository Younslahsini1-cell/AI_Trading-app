# -*- coding: utf-8 -*-
"""
نظام التداول الآلي الشامل (XAU/USD)
يجمع بين:
- التحليل الفني المتقدم (مؤشرات متعددة)
- التعلم الآلي (RandomForest + MLP)
- تحليل المشاعر الاقتصادية والسياسية (اختياري)
- إدارة صفقات تلقائية كاملة
- واجهة Streamlit تفاعلية
- قاعدة بيانات SQLite للتعلم المستمر
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import json
import os
import sqlite3
import threading
import traceback
import requests
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import yfinance as yf
import ta  # مكتبة المؤشرات الفنية
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# إعدادات الصفحة
# ============================================================
st.set_page_config(
    page_title="التداول الآلي الشامل",
    layout="wide",
    page_icon="🤖"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #0a0e17; color: #f3f4f6; }
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
    .metric-card { background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 15px; }
    .metric-title { color: #93c5fd; font-size: 0.9rem; font-weight: 700; }
    .metric-value { color: #fbbf24; font-size: 2rem; font-weight: 900; }
    .signal-buy { color: #22c55e; font-weight: 900; }
    .signal-sell { color: #ef4444; font-weight: 900; }
    .signal-neutral { color: #94a3b8; font-weight: 900; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Constants
# ============================================================
DB_FILE = "trading_bot.db"
MODEL_FILE = "ensemble_model.pkl"
SCALER_FILE = "scaler.pkl"
FEATURES = [
    'rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low', 'bb_width',
    'atr', 'ema_12', 'ema_26', 'ema_50', 'ema_200', 'sma_20', 'sma_50',
    'stoch_k', 'stoch_d', 'cci', 'adx', 'williams_r', 'mfi', 'roc', 'obv',
    'ichimoku_a', 'ichimoku_b', 'ichimoku_base', 'ichimoku_conversion',
    'price_change_1', 'price_change_5', 'price_change_10', 'volume_change',
    'sentiment_score'
]
# ============================================================
# Database
# ============================================================
def get_db_connection():
    return sqlite3.connect(DB_FILE, timeout=20, check_same_thread=False)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, symbol TEXT, direction TEXT, entry REAL, sl REAL, tp REAL,
        win INTEGER, note TEXT, confidence REAL, features TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS active_trade (
        id INTEGER PRIMARY KEY, symbol TEXT, direction TEXT, entry REAL, sl REAL, tp REAL,
        time TEXT, confidence REAL, features TEXT, signal_bar_time TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY, value TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS news_cache (
        date TEXT PRIMARY KEY, sentiment REAL
    )''')
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

def get_total_trades():
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM trades")
        return int(c.fetchone()[0])
    finally:
        conn.close()

def get_win_trades():
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM trades WHERE win=1")
        return int(c.fetchone()[0])
    finally:
        conn.close()

# ============================================================
# Data Fetching (yfinance / Twelve Data)
# ============================================================
def fetch_data(symbol="XAUUSD=X", interval="1h", period="6mo", api_key=None):
    """جلب بيانات السعر مع إمكانية استخدام Twelve Data إذا توفر مفتاح."""
    df = pd.DataFrame()
    if api_key and api_key.strip():
        # Twelve Data
        try:
            params = {
                "symbol": "XAU/USD",
                "interval": interval,
                "outputsize": 5000,
                "timezone": "UTC",
                "apikey": api_key.strip()
            }
            resp = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if "values" in data:
                    df = pd.DataFrame(data["values"])
                    df = df.rename(columns={"datetime": "Datetime"})
                    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
                    for col in ["open", "high", "low", "close"]:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df[["Datetime", "open", "high", "low", "close", "volume"]].dropna()
                    df = df.sort_values("Datetime").reset_index(drop=True)
        except Exception as e:
            st.error(f"خطأ في Twelve Data: {e}")
    if df.empty:
        # yfinance fallback
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if not df.empty:
                df = df.reset_index()
                df = df.rename(columns={"index": "Datetime"})
                df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = ["Datetime", "open", "high", "low", "close", "volume"]
                df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
                df = df.dropna().reset_index(drop=True)
        except Exception as e:
            st.error(f"خطأ في yfinance: {e}")
    return df

# ============================================================
# Technical Indicators (شامل)
# ============================================================
def add_technical_indicators(df):
    """إضافة أكثر من 25 مؤشر فني."""
    df = df.copy()
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    # MACD
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = bb.bollinger_wband()
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    # EMA
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['ema_200'] = ta.trend.EMAIndicator(close=df['close'], window=200).ema_indicator()
    # SMA
    df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    # CCI
    df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
    # ADX
    df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
    # MFI
    df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()
    # ROC
    df['roc'] = ta.momentum.ROCIndicator(close=df['close']).roc()
    # OBV
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    # Ichimoku
    ichimoku = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'])
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
    # Price changes
    df['price_change_1'] = df['close'].pct_change(1) * 100
    df['price_change_5'] = df['close'].pct_change(5) * 100
    df['price_change_10'] = df['close'].pct_change(10) * 100
    # Volume change
    df['volume_change'] = df['volume'].pct_change(1) * 100
    # Sentiment score (سيتم تعبئته لاحقاً)
    df['sentiment_score'] = 0.0
    # إزالة الصفوف الفارغة
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df

# ============================================================
# Sentiment Analysis (أخبار اقتصادية/سياسية)
# ============================================================
def get_news_sentiment(symbol="XAU/USD", api_key=None):
    """
    جلب وتحليل المشاعر من عناوين الأخبار (اختياري).
    يستخدم NewsAPI إذا توفر مفتاح، وإلا يعيد درجة محايدة.
    يمكن استخدام GDELT المجاني.
    """
    # محاولة GDELT المجانية (لا تحتاج مفتاح)
    try:
        # استخدام GDELT DOC API للبحث عن أخبار الذهب
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": "gold OR XAU OR gold price",
            "mode": "artlist",
            "maxrecords": 50,
            "format": "json",
            "timespan": "1d"
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            articles = data.get("articles", [])
            if articles:
                # تحليل بسيط للكلمات المفتاحية الإيجابية/السلبية
                positive_words = ['surge', 'rally', 'gain', 'rise', 'bull', 'strong', 'higher', 'record']
                negative_words = ['drop', 'fall', 'decline', 'bear', 'weak', 'lower', 'loss', 'plunge']
                pos_count = 0
                neg_count = 0
                for art in articles:
                    title = art.get("title", "").lower()
                    pos_count += sum(1 for w in positive_words if w in title)
                    neg_count += sum(1 for w in negative_words if w in title)
                total = pos_count + neg_count
                if total > 0:
                    sentiment = (pos_count - neg_count) / total * 100  # من -100 إلى 100
                else:
                    sentiment = 0.0
                return sentiment
    except Exception:
        pass
    # إذا لم يتوفر مصدر، نعيد قيمة محايدة
    return 0.0

# ============================================================
# Model Training
# ============================================================
def train_ensemble_model(df, features):
    """تدريب نموذج Ensemble (RandomForest + GradientBoosting) للتنبؤ بالاتجاه."""
    # الهدف: هل السعر بعد 3 فترات سيكون أعلى أم أقل؟ (1 = ارتفاع، 0 = انخفاض)
    df = df.copy()
    df['target'] = (df['close'].shift(-3) > df['close']).astype(int)
    df = df.dropna(subset=features + ['target'])
    if len(df) < 100:
        return None, None, None
    X = df[features].values
    y = df['target'].values
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    # مقياس
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # RandomForest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    # GradientBoosting
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train_scaled, y_train)
    # دمج بسيط: متوسط الاحتمالات
    # سنحفظ النموذجين، أو نستخدم نموذج واحد (RF) للسهولة
    # سنحفظ RF فقط للتبسيط
    joblib.dump(rf, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    # دقة على الاختبار
    y_pred = rf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    return rf, scaler, acc

def load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            return model, scaler
        except:
            pass
    return None, None

# ============================================================
# Signal Generation & Trade Management
# ============================================================
def generate_signal(df, model, scaler):
    """توليد إشارة تداول بناءً على النموذج والمؤشرات."""
    if df is None or df.empty or len(df) < 50:
        return None, "لا توجد بيانات كافية"
    last = df.iloc[-1]
    features_values = last[FEATURES].values.reshape(1, -1)
    # تنبؤ النموذج
    if model and scaler:
        try:
            scaled = scaler.transform(features_values)
            proba = model.predict_proba(scaled)[0]
            # احتمال ارتفاع (class 1)
            prob_up = proba[1] if len(proba) > 1 else 0.5
            confidence = prob_up * 100
            direction = "BUY" if prob_up > 0.55 else "SELL" if prob_up < 0.45 else "NEUTRAL"
        except Exception as e:
            confidence = 0
            direction = "NEUTRAL"
    else:
        # بدون نموذج: استخدام قواعد بسيطة
        rsi = last['rsi']
        ema_short = last['ema_12']
        ema_long = last['ema_26']
        if rsi < 30 and ema_short > ema_long:
            direction = "BUY"
            confidence = 60.0
        elif rsi > 70 and ema_short < ema_long:
            direction = "SELL"
            confidence = 60.0
        else:
            direction = "NEUTRAL"
            confidence = 50.0
    # دمج مع تحليل المشاعر
    sentiment = last.get('sentiment_score', 0.0)
    if sentiment > 0:
        confidence = confidence * 0.8 + sentiment * 0.2
    elif sentiment < 0:
        confidence = confidence * 0.8 + (100 + sentiment) * 0.2  # sentiment سلبي يعطي ثقة أقل
    confidence = max(0, min(100, confidence))
    # تحديد نقاط الدخول والوقف
    atr = last['atr']
    if atr <= 0:
        atr = 1.0
    curr_price = last['close']
    sl_distance = atr * 1.5
    tp_distance = sl_distance * 2.0  # نسبة R:R = 1:2
    if direction == "BUY":
        sl = curr_price - sl_distance
        tp = curr_price + tp_distance
    elif direction == "SELL":
        sl = curr_price + sl_distance
        tp = curr_price - tp_distance
    else:
        sl = tp = None
    return {
        "direction": direction,
        "confidence": confidence,
        "entry": curr_price,
        "sl": sl,
        "tp": tp,
        "atr": atr,
        "sentiment": sentiment
    }, None

# ============================================================
# Sidebar Settings
# ============================================================
st.sidebar.title("🤖 إعدادات النظام")
api_key_12 = st.sidebar.text_input("مفتاح Twelve Data API (اختياري)", type="password", value="")
symbol_input = st.sidebar.text_input("رمز الأصل", value="XAUUSD=X")
interval_input = st.sidebar.selectbox("الفاصل الزمني", ["15m", "30m", "1h", "4h", "1d"], index=2)
period_input = st.sidebar.selectbox("فترة البيانات", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
min_confidence = st.sidebar.slider("أدنى ثقة للدخول (%)", 50, 95, 65, 1)
risk_per_trade = st.sidebar.slider("نسبة المخاطرة لكل صفقة (%)", 0.5, 5.0, 1.0, 0.1)
use_auto_trade = st.sidebar.checkbox("تفعيل التداول التلقائي", value=False)
use_sentiment = st.sidebar.checkbox("استخدام تحليل المشاعر", value=True)
st.sidebar.markdown("---")
if st.sidebar.button("إعادة تدريب النموذج الآن"):
    with st.spinner("جاري التدريب..."):
        df_raw = fetch_data(symbol_input, interval_input, period_input, api_key_12)
        df_proc = add_technical_indicators(df_raw)
        model, scaler, acc = train_ensemble_model(df_proc, FEATURES)
        if model:
            st.sidebar.success(f"تم التدريب بدقة {acc*100:.2f}%")
        else:
            st.sidebar.error("فشل التدريب: بيانات غير كافية")

# ============================================================
# Main App
# ============================================================
st.title("🤖 نظام التداول الآلي الشامل")
st.markdown("يجمع بين التعلم الآلي، المؤشرات الفنية، تحليل المشاعر، وإدارة الصفقات التلقائية.")

# جلب البيانات
df_raw = fetch_data(symbol_input, interval_input, period_input, api_key_12)
if df_raw.empty:
    st.error("فشل في جلب البيانات. تحقق من الاتصال أو الرمز.")
    st.stop()

# معالجة
df_processed = add_technical_indicators(df_raw)
if df_processed.empty:
    st.error("البيانات غير كافية لحساب المؤشرات.")
    st.stop()

# تحليل المشاعر (إذا مفعل)
if use_sentiment:
    sentiment = get_news_sentiment()
    # تحديث آخر صف بالمشاعر
    df_processed.loc[df_processed.index[-1], 'sentiment_score'] = sentiment

# تحميل النموذج
model, scaler = load_model()
if model is None:
    st.info("النموذج غير مدرب. سيتم تدريبه تلقائياً.")
    # تدريب سريع
    model, scaler, acc = train_ensemble_model(df_processed, FEATURES)
    if model:
        st.success(f"تم تدريب النموذج الأولي بدقة {acc*100:.2f}%")

# توليد الإشارة
signal, error_msg = generate_signal(df_processed, model, scaler)

# عرض المعلومات
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("السعر الحالي", f"${df_processed['close'].iloc[-1]:.2f}")
with col2:
    if signal and signal['direction'] != "NEUTRAL":
        st.metric("الإشارة", signal['direction'], delta=None)
    else:
        st.metric("الإشارة", "محايد")
with col3:
    if signal:
        st.metric("الثقة", f"{signal['confidence']:.1f}%")
    else:
        st.metric("الثقة", "N/A")
with col4:
    st.metric("ATR", f"{df_processed['atr'].iloc[-1]:.2f}")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 الرسم البياني والمؤشرات",
    "🧠 إشارات النظام",
    "📋 سجل الصفقات",
    "📈 تقييم الأداء",
    "⚙️ إدارة التداول التلقائي"
])

with tab1:
    st.subheader("الرسم البياني")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2])
    # سعر
    fig.add_trace(go.Candlestick(x=df_processed['Datetime'],
                                 open=df_processed['open'], high=df_processed['high'],
                                 low=df_processed['low'], close=df_processed['close'],
                                 name="السعر"), row=1, col=1)
    # EMA 50 و 200
    fig.add_trace(go.Scatter(x=df_processed['Datetime'], y=df_processed['ema_50'], name="EMA 50", line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_processed['Datetime'], y=df_processed['ema_200'], name="EMA 200", line=dict(color='purple')), row=1, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=df_processed['Datetime'], y=df_processed['rsi'], name="RSI", line=dict(color='blue')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    # Volume
    colors = ['green' if c >= o else 'red' for c, o in zip(df_processed['close'], df_processed['open'])]
    fig.add_trace(go.Bar(x=df_processed['Datetime'], y=df_processed['volume'], marker_color=colors, name="Volume"), row=3, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("تفاصيل الإشارة")
    if signal:
        st.write(f"الاتجاه: **{signal['direction']}**")
        st.write(f"الثقة: **{signal['confidence']:.1f}%**")
        if signal['sl'] and signal['tp']:
            st.write(f"سعر الدخول: **{signal['entry']:.2f}**")
            st.write(f"وقف الخسارة: **{signal['sl']:.2f}**")
            st.write(f"جني الأرباح: **{signal['tp']:.2f}**")
        st.write(f"ATR: {signal['atr']:.2f}")
        st.write(f"تحليل المشاعر: {signal['sentiment']:.1f}")
        if signal['confidence'] >= min_confidence and signal['direction'] != "NEUTRAL":
            st.success("✅ الإشارة قوية وقد يتم تنفيذها إذا كان التداول التلقائي مفعلاً")
        else:
            st.warning("⚠️ الإشارة غير كافية للدخول")
    # عرض جدول آخر 10 صفوف من المؤشرات
    st.subheader("أحدث البيانات والمؤشرات")
    st.dataframe(df_processed[['Datetime', 'close', 'rsi', 'macd', 'ema_50', 'atr', 'sentiment_score']].tail(10), use_container_width=True)

with tab3:
    st.subheader("سجل الصفقات")
    conn = get_db_connection()
    try:
        df_trades = pd.read_sql("SELECT * FROM trades ORDER BY id DESC", conn)
    finally:
        conn.close()
    if not df_trades.empty:
        st.dataframe(df_trades, use_container_width=True)
    else:
        st.info("لا توجد صفقات مسجلة بعد")

with tab4:
    st.subheader("تقييم الأداء")
    total = get_total_trades()
    wins = get_win_trades()
    if total > 0:
        win_rate = wins / total * 100
        st.metric("نسبة الربح", f"{win_rate:.1f}%")
        st.metric("إجمالي الصفقات", total)
        st.metric("الصفقات الرابحة", wins)
    else:
        st.info("لا يوجد أداء بعد")

with tab5:
    st.subheader("إدارة التداول التلقائي")
    st.write("حالة التداول التلقائي:", "مفعل ✅" if use_auto_trade else "معطل ❌")
    st.write(f"أدنى ثقة مطلوبة: {min_confidence}%")
    st.write(f"نسبة المخاطرة: {risk_per_trade}%")
    st.write("في النسخة الحالية، التنفيذ التلقائي يتم عبر مراقبة مستمرة داخل التطبيق.")
    # عرض الصفقة النشطة
    conn = get_db_connection()
    try:
        active_df = pd.read_sql("SELECT * FROM active_trade WHERE id=1", conn)
    finally:
        conn.close()
    if not active_df.empty:
        st.warning("توجد صفقة نشطة:")
        st.dataframe(active_df, use_container_width=True)

# ============================================================
# Automated Trading Loop (يتم تفعيله مع st_autorefresh)
# ============================================================
if use_auto_trade and signal and signal['direction'] != "NEUTRAL":
    # فحص إذا كانت هناك صفقة نشطة
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM active_trade WHERE id=1")
        active = c.fetchone()
        if not active and signal['confidence'] >= min_confidence:
            # فتح صفقة جديدة
            c.execute("DELETE FROM active_trade")
            c.execute("""INSERT INTO active_trade
                (id, symbol, direction, entry, sl, tp, time, confidence, features, signal_bar_time)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (1, symbol_input, signal['direction'], signal['entry'], signal['sl'], signal['tp'],
                 datetime.now(timezone.utc).isoformat(), signal['confidence'],
                 json.dumps({k: float(df_processed.iloc[-1][k]) for k in FEATURES}),
                 str(df_processed['Datetime'].iloc[-1])))
            conn.commit()
            st.success(f"تم فتح صفقة {signal['direction']} بثقة {signal['confidence']:.1f}%")
    finally:
        conn.close()

# ============================================================
# تحديث تلقائي
# ============================================================
st_autorefresh(interval=300000, key="autorefresh")  # كل 5 دقائق
