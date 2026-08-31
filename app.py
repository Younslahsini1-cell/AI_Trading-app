# -*- coding: utf-8 -*-
"""
النظام المتكامل للتداول الآلي على XAU/USD
يجمع بين:
- التحليل الفني المتقدم (أكثر من 40 مؤشر)
- التعلم الآلي (RandomForest, GradientBoosting, MLP)
- تحليل ICT / Smart Money
- تحليل المشاعر (GDELT)
- استشارة Groq (اختياري)
- إدارة صفقات تلقائية
- حفظ المفاتيح والإعدادات في قاعدة البيانات
- واجهة بسيطة تعرض الصفقات فقط
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
import warnings
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import yfinance as yf
import ta
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel, KeltnerChannel, UlcerIndex
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator, CCIIndicator, IchimokuIndicator, TRIXIndicator, AroonIndicator, VortexIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, TSIIndicator, UltimateOscillator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, ForceIndexIndicator, EaseOfMovementIndicator, VolumePriceTrendIndicator, MFIIndicator, AccDistIndexIndicator
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ============================================================
# إعدادات الصفحة
# ============================================================
st.set_page_config(page_title="نظام التداول الآلي", layout="wide", page_icon="📈")

# ============================================================
# ثوابت النظام
# ============================================================
DB_FILE = "trading_system.db"
MODEL_FILE = "ensemble_model.pkl"
SCALER_FILE = "scaler.pkl"
TRAINING_LOCK_FILE = "training.lock"
TRAINING_OUTPUT_SIZE = 8000
LIVE_OUTPUT_SIZE = 500
TRAINING_LOCK_MAX_AGE = 60 * 60  # ساعة

FEATURES = [
    # مؤشرات الزخم
    'rsi', 'stoch_k', 'stoch_d', 'williams_r', 'cci', 'adx', 'roc', 'mfi', 'tsi', 'uo',
    # مؤشرات الاتجاه
    'macd', 'macd_signal', 'macd_diff', 'ema_12', 'ema_26', 'ema_50', 'ema_200',
    'sma_20', 'sma_50', 'sma_200', 'ichimoku_a', 'ichimoku_b', 'ichimoku_base',
    'ichimoku_conversion', 'trix', 'aroon_up', 'aroon_down', 'vortex_pos', 'vortex_neg',
    # مؤشرات التقلب
    'atr', 'bb_high', 'bb_low', 'bb_width', 'donchian_high', 'donchian_low',
    'keltner_high', 'keltner_low', 'ulcer_index',
    # مؤشرات الحجم
    'obv', 'cmf', 'force_index', 'eom', 'vpt', 'adi',
    # تغيرات السعر والحجم
    'price_change_1', 'price_change_5', 'price_change_10', 'volume_change_1', 'volume_change_5',
    # تحليل المشاعر
    'sentiment_score',
    # ICT features
    'ict_bias_numeric', 'ict_liquidity_sweep', 'ict_ob_present', 'ict_fvg_present'
]

# ============================================================
# إعداد قاعدة البيانات
# ============================================================
def get_db_connection():
    return sqlite3.connect(DB_FILE, timeout=20, check_same_thread=False)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    # جدول الصفقات
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        symbol TEXT,
        direction TEXT,
        entry REAL,
        sl REAL,
        tp REAL,
        win INTEGER,
        note TEXT,
        confidence REAL,
        features TEXT,
        closed_date TEXT
    )''')
    # جدول الصفقة النشطة
    c.execute('''CREATE TABLE IF NOT EXISTS active_trade (
        id INTEGER PRIMARY KEY,
        symbol TEXT,
        direction TEXT,
        entry REAL,
        sl REAL,
        tp REAL,
        time TEXT,
        confidence REAL,
        features TEXT,
        signal_bar_time TEXT
    )''')
    # جدول الإعدادات
    c.execute('''CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )''')
    # جدول سجل التدريب
    c.execute('''CREATE TABLE IF NOT EXISTS training_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        accuracy REAL,
        description TEXT
    )''')
    # جدول المشاعر المخزنة
    c.execute('''CREATE TABLE IF NOT EXISTS sentiment_cache (
        date TEXT PRIMARY KEY,
        sentiment REAL
    )''')
    conn.commit()
    conn.close()

init_db()

# دوال الإعدادات
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

# ============================================================
# إرسال تنبيهات Ntfy
# ============================================================
def send_ntfy_alert(message, title="Trading Bot Alert", channel=""):
    if not channel:
        channel = load_setting("ntfy_channel", "")
    if not channel:
        return
    channel = channel.strip().split("/")[-1]
    if not channel:
        return
    try:
        requests.post(
            f"https://ntfy.sh/{channel}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": "high"},
            timeout=5,
        )
    except Exception as e:
        print(f"Ntfy error: {e}")

# ============================================================
# جلب البيانات (Twelve Data / yfinance)
# ============================================================
def fetch_data(symbol="XAUUSD=X", interval="1h", period="6mo", api_key=None):
    """جلب بيانات الأسعار مع تنسيق موحد."""
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
                    df["volume"] = 0
                    df = df[["Datetime", "open", "high", "low", "close", "volume"]].dropna()
                    df = df.sort_values("Datetime").reset_index(drop=True)
        except Exception as e:
            st.error(f"Twelve Data error: {e}")
    if df.empty:
        # yfinance fallback
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if not df.empty:
                df = df.reset_index()
                df = df.rename(columns={"index": "Datetime"})
                df = df.rename(columns={
                    "Datetime": "Datetime",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                })
                if 'volume' not in df.columns:
                    df['volume'] = 0
                df = df[["Datetime", "open", "high", "low", "close", "volume"]].copy()
                df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
                df = df.dropna().reset_index(drop=True)
        except Exception as e:
            st.error(f"yfinance error: {e}")
    return df

# ============================================================
# إزالة الشموع غير المكتملة
# ============================================================
def keep_closed_candles(df, interval_hours=1):
    if df is None or df.empty or "Datetime" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    now_utc = datetime.now(timezone.utc)
    candle_delta = timedelta(hours=interval_hours)
    mask = (df["Datetime"] + candle_delta <= pd.Timestamp(now_utc))
    closed = df.loc[mask].copy()
    return closed.reset_index(drop=True)

# ============================================================
# إضافة المؤشرات الفنية
# ============================================================
def add_all_technical_indicators(df):
    """إضافة مجموعة شاملة من المؤشرات الفنية."""
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            df[col] = 0

    # مؤشرات الزخم
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
    df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
    df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    df['roc'] = ta.momentum.ROCIndicator(close=df['close']).roc()
    df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()
    df['tsi'] = ta.momentum.TSIIndicator(close=df['close']).tsi()
    df['uo'] = ta.momentum.UltimateOscillator(high=df['high'], low=df['low'], close=df['close']).ultimate_oscillator()

    # مؤشرات الاتجاه
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['ema_200'] = ta.trend.EMAIndicator(close=df['close'], window=200).ema_indicator()
    df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['sma_200'] = ta.trend.SMAIndicator(close=df['close'], window=200).sma_indicator()
    ichimoku = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'])
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
    df['trix'] = ta.trend.TRIXIndicator(close=df['close']).trix()
    aroon = ta.trend.AroonIndicator(high=df['high'], low=df['low'])
    df['aroon_up'] = aroon.aroon_up()
    df['aroon_down'] = aroon.aroon_down()
    vortex = ta.trend.VortexIndicator(high=df['high'], low=df['low'], close=df['close'])
    df['vortex_pos'] = vortex.vortex_indicator_pos()
    df['vortex_neg'] = vortex.vortex_indicator_neg()

    # مؤشرات التقلب
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = bb.bollinger_wband()
    donchian = ta.volatility.DonchianChannel(high=df['high'], low=df['low'], close=df['close'])
    df['donchian_high'] = donchian.donchian_channel_hband()
    df['donchian_low'] = donchian.donchian_channel_lband()
    keltner = ta.volatility.KeltnerChannel(high=df['high'], low=df['low'], close=df['close'])
    df['keltner_high'] = keltner.keltner_channel_hband()
    df['keltner_low'] = keltner.keltner_channel_lband()
    df['ulcer_index'] = ta.volatility.UlcerIndex(close=df['close']).ulcer_index()

    # مؤشرات الحجم
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).chaikin_money_flow()
    df['force_index'] = ta.volume.ForceIndexIndicator(close=df['close'], volume=df['volume']).force_index()
    df['eom'] = ta.volume.EaseOfMovementIndicator(high=df['high'], low=df['low'], volume=df['volume']).ease_of_movement()
    df['vpt'] = ta.volume.VolumePriceTrendIndicator(close=df['close'], volume=df['volume']).volume_price_trend()
    df['adi'] = ta.volume.AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).acc_dist_index()

    # تغيرات السعر والحجم
    df['price_change_1'] = df['close'].pct_change(1) * 100
    df['price_change_5'] = df['close'].pct_change(5) * 100
    df['price_change_10'] = df['close'].pct_change(10) * 100
    df['volume_change_1'] = df['volume'].pct_change(1) * 100
    df['volume_change_5'] = df['volume'].pct_change(5) * 100

    # متغيرات ICT (ستُحسب لاحقًا)
    df['ict_bias_numeric'] = 0
    df['ict_liquidity_sweep'] = 0
    df['ict_ob_present'] = 0
    df['ict_fvg_present'] = 0
    df['sentiment_score'] = 0.0

    # تنظيف
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df

# ============================================================
# تحليل ICT / Smart Money
# ============================================================
def find_swing_points(df, lookback=3):
    if df is None or df.empty:
        return [], []
    highs = df['high'].astype(float).values
    lows = df['low'].astype(float).values
    n = len(df)
    swing_highs = []
    swing_lows = []
    for i in range(lookback, n - lookback):
        if highs[i] == np.max(highs[i-lookback:i+lookback+1]):
            swing_highs.append((i, float(highs[i])))
        if lows[i] == np.min(lows[i-lookback:i+lookback+1]):
            swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows

def analyze_market_structure(swing_highs, swing_lows):
    events = ([(i, 'high', p) for i, p in swing_highs] +
              [(i, 'low', p) for i, p in swing_lows])
    events.sort(key=lambda x: x[0])
    structure_breaks = []
    trend = None
    last_high = None
    last_low = None
    for i, kind, price in events:
        if kind == 'high':
            if last_high is not None and price > last_high:
                label = 'BOS' if trend == 'bullish' else 'CHoCH'
                structure_breaks.append({'type': label, 'direction': 'BULLISH', 'price': round(price, 2)})
                trend = 'bullish'
            last_high = price
        else:
            if last_low is not None and price < last_low:
                label = 'BOS' if trend == 'bearish' else 'CHoCH'
                structure_breaks.append({'type': label, 'direction': 'BEARISH', 'price': round(price, 2)})
                trend = 'bearish'
            last_low = price
    current_bias = structure_breaks[-1]['direction'] if structure_breaks else 'NEUTRAL'
    return current_bias, structure_breaks[-8:]

def detect_order_blocks(df, lookback=40, displacement_atr_mult=1.2):
    if df is None or df.empty or 'atr' not in df.columns or len(df) < 5:
        return None, None
    recent = df.iloc[-lookback:].reset_index(drop=True)
    bullish_ob = None
    bearish_ob = None
    for i in range(1, len(recent)):
        atr_val = float(recent['atr'].iloc[i])
        if not np.isfinite(atr_val) or atr_val <= 0:
            continue
        body = recent['close'].iloc[i] - recent['open'].iloc[i]
        is_disp = abs(body) > displacement_atr_mult * atr_val
        prev = recent.iloc[i-1]
        if is_disp and body > 0 and prev['close'] < prev['open']:
            bullish_ob = {'top': round(float(prev['open']), 2), 'bottom': round(float(prev['low']), 2)}
        elif is_disp and body < 0 and prev['close'] > prev['open']:
            bearish_ob = {'top': round(float(prev['high']), 2), 'bottom': round(float(prev['open']), 2)}
    return bullish_ob, bearish_ob

def detect_fair_value_gaps(df, lookback=60):
    if df is None or df.empty or len(df) < 3:
        return None, None
    recent = df.iloc[-lookback:].reset_index(drop=True)
    bullish_fvg = None
    bearish_fvg = None
    for i in range(2, len(recent)):
        c1 = recent.iloc[i-2]
        c3 = recent.iloc[i]
        if c1['high'] < c3['low']:
            bullish_fvg = {'top': round(float(c3['low']), 2), 'bottom': round(float(c1['high']), 2)}
        if c1['low'] > c3['high']:
            bearish_fvg = {'top': round(float(c1['low']), 2), 'bottom': round(float(c3['high']), 2)}
    return bullish_fvg, bearish_fvg

def detect_liquidity_sweep(df, swing_highs, swing_lows):
    if not swing_highs or not swing_lows or df is None or df.empty:
        return False, 0
    recent_highs = swing_highs[-5:]
    recent_lows = swing_lows[-5:]
    bsl = max(p for _, p in recent_highs)
    ssl = min(p for _, p in recent_lows)
    last = df.iloc[-1]
    if last['high'] > bsl and last['close'] < bsl:
        return True, 1
    elif last['low'] < ssl and last['close'] > ssl:
        return True, -1
    return False, 0

def compute_ict_features(df, swing_lookback=3):
    if df is None or df.empty or len(df) < max(30, swing_lookback*6):
        return df
    df = df.copy()
    swing_highs, swing_lows = find_swing_points(df, lookback=swing_lookback)
    bias, breaks = analyze_market_structure(swing_highs, swing_lows)
    bull_ob, bear_ob = detect_order_blocks(df)
    bull_fvg, bear_fvg = detect_fair_value_gaps(df)
    sweep, sweep_dir = detect_liquidity_sweep(df, swing_highs, swing_lows)

    bias_numeric = 1 if bias == 'BULLISH' else (-1 if bias == 'BEARISH' else 0)
    df['ict_bias_numeric'] = bias_numeric
    df['ict_liquidity_sweep'] = sweep_dir
    df['ict_ob_present'] = 1 if (bull_ob or bear_ob) else 0
    df['ict_fvg_present'] = 1 if (bull_fvg or bear_fvg) else 0
    return df

# ============================================================
# تحليل المشاعر
# ============================================================
def get_sentiment_from_gdelt():
    try:
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
                positive_words = ['surge', 'rally', 'gain', 'rise', 'bull', 'strong', 'higher', 'record', 'boost']
                negative_words = ['drop', 'fall', 'decline', 'bear', 'weak', 'lower', 'loss', 'plunge', 'crash', 'sell-off']
                pos_count = 0
                neg_count = 0
                for art in articles:
                    title = art.get("title", "").lower()
                    pos_count += sum(1 for w in positive_words if w in title)
                    neg_count += sum(1 for w in negative_words if w in title)
                total = pos_count + neg_count
                if total > 0:
                    sentiment = (pos_count - neg_count) / total * 100
                    return sentiment
    except Exception:
        pass
    return 0.0

def get_cached_sentiment():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT sentiment FROM sentiment_cache WHERE date=?", (today,))
        row = c.fetchone()
        if row:
            return float(row[0])
        else:
            sentiment = get_sentiment_from_gdelt()
            c.execute("INSERT OR REPLACE INTO sentiment_cache (date, sentiment) VALUES (?, ?)", (today, sentiment))
            conn.commit()
            return sentiment
    finally:
        conn.close()

# ============================================================
# استشارة Groq
# ============================================================
def get_groq_review(direction, last_row, ai_conf, api_key, model_name="llama-3.3-70b-versatile"):
    if not api_key:
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
            "راجع الاتجاه بشكل مستقل. لا تفترض أن نموذج AI صحيح. أعط رأياً محافظاً.\n"
            "\n"
            "يجب أن يكون الرد JSON فقط بهذا الشكل:\n"
            '{"agree": true, "confidence": 0, "reason": "..."}'
        )
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "temperature": 0,
                "max_completion_tokens": 300,
                "messages": [
                    {"role": "system", "content": "أنت محلل فني محافظ. أعد JSON صالح فقط."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        message = choices[0].get("message", {})
        text = message.get("content", "")
        if not text:
            return None
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
# تدريب النموذج
# ============================================================
def train_model(df, features):
    df = df.copy()
    df['target'] = (df['close'].shift(-3) > df['close']).astype(int)
    df = df.dropna(subset=features + ['target'])
    if len(df) < 200:
        return None, None, None
    X = df[features].values
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.0001,
                        max_iter=1000, early_stopping=True, validation_fraction=0.15, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('mlp', mlp)], voting='soft')
    ensemble.fit(X_train_scaled, y_train)
    joblib.dump(ensemble, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    y_pred = ensemble.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    return ensemble, scaler, acc

def load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            return model, scaler
        except:
            pass
    return None, None

def background_train(api_key, symbol, interval, period):
    if os.path.exists(TRAINING_LOCK_FILE):
        return
    try:
        with open(TRAINING_LOCK_FILE, 'w') as f:
            f.write(datetime.now(timezone.utc).isoformat())
        df = fetch_data(symbol, interval, period, api_key)
        df = keep_closed_candles(df)
        df = add_all_technical_indicators(df)
        df = compute_ict_features(df)
        sentiment = get_cached_sentiment()
        df['sentiment_score'] = sentiment
        model, scaler, acc = train_model(df, FEATURES)
        if model is not None:
            conn = get_db_connection()
            try:
                c = conn.cursor()
                c.execute("INSERT INTO training_log (timestamp, accuracy, description) VALUES (?, ?, ?)",
                          (datetime.now(timezone.utc).isoformat(), acc, f"Trained on {len(df)} samples"))
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        print(f"Background training error: {e}")
    finally:
        if os.path.exists(TRAINING_LOCK_FILE):
            try:
                os.remove(TRAINING_LOCK_FILE)
            except:
                pass

# ============================================================
# توليد الإشارة مع Groq
# ============================================================
def generate_signal(df, model, scaler, min_conf=65, use_groq=False, groq_api_key="", groq_model="llama-3.3-70b-versatile"):
    if df is None or df.empty or len(df) < 50:
        return None, "لا توجد بيانات كافية"
    last = df.iloc[-1]
    features_values = last[FEATURES].values.reshape(1, -1)
    if model and scaler:
        try:
            scaled = scaler.transform(features_values)
            proba = model.predict_proba(scaled)[0]
            prob_up = proba[1] if len(proba) > 1 else 0.5
            confidence = prob_up * 100
            direction = "BUY" if prob_up > 0.55 else "SELL" if prob_up < 0.45 else "NEUTRAL"
        except Exception as e:
            confidence = 0
            direction = "NEUTRAL"
    else:
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

    # دمج المشاعر
    sentiment = last.get('sentiment_score', 0.0)
    if sentiment > 0:
        confidence = confidence * 0.85 + sentiment * 0.15
    elif sentiment < 0:
        confidence = confidence * 0.85 + (100 + sentiment) * 0.15
    confidence = max(0, min(100, confidence))

    # استشارة Groq (اختياري)
    groq_result = None
    if use_groq and direction != "NEUTRAL" and groq_api_key:
        groq_result = get_groq_review(direction, last, confidence, groq_api_key, groq_model)
        if groq_result is not None:
            if not groq_result["agree"] or groq_result["confidence"] < 50:
                direction = "NEUTRAL"  # رفض الصفقة
                confidence = min(confidence, groq_result["confidence"])
            else:
                confidence = (confidence + groq_result["confidence"]) / 2

    atr = last['atr'] if last['atr'] > 0 else 1.0
    curr_price = last['close']
    sl_distance = atr * 1.5
    tp_distance = sl_distance * 2.0
    if direction == "BUY":
        sl = curr_price - sl_distance
        tp = curr_price + tp_distance
    elif direction == "SELL":
        sl = curr_price + sl_distance
        tp = curr_price - tp_distance
    else:
        sl = tp = None

    signal = {
        "direction": direction,
        "confidence": confidence,
        "entry": curr_price,
        "sl": sl,
        "tp": tp,
        "atr": atr,
        "sentiment": sentiment,
        "datetime": last['Datetime'],
        "groq_reason": groq_result["reason"] if groq_result else ""
    }
    return signal, None

# ============================================================
# إدارة الصفقات
# ============================================================
def open_trade(symbol, signal):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM active_trade")
        c.execute("""INSERT INTO active_trade
            (id, symbol, direction, entry, sl, tp, time, confidence, features, signal_bar_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (1, symbol, signal['direction'], signal['entry'], signal['sl'], signal['tp'],
             datetime.now(timezone.utc).isoformat(), signal['confidence'],
             json.dumps({k: float(signal[k]) for k in ['atr', 'sentiment']}),
             str(signal['datetime'])))
        conn.commit()
    finally:
        conn.close()

def close_trade(win, note=""):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM active_trade WHERE id=1")
        active = c.fetchone()
        if active:
            c.execute("""INSERT INTO trades
                (date, symbol, direction, entry, sl, tp, win, note, confidence, features, closed_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now(timezone.utc).isoformat(), active[1], active[2], active[3], active[4], active[5],
                 win, note, active[8], active[9], datetime.now(timezone.utc).isoformat()))
            c.execute("DELETE FROM active_trade WHERE id=1")
            conn.commit()
    finally:
        conn.close()

def check_active_trade(df):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM active_trade WHERE id=1")
        active = c.fetchone()
        if not active:
            return None
        direction = active[2]
        entry = float(active[3])
        sl = float(active[4])
        tp = float(active[5])
        signal_bar_time = str(active[9])
        if df is None or df.empty:
            return active
        last = df.iloc[-1]
        current_bar_time = str(last['Datetime'])
        if current_bar_time == signal_bar_time:
            return active
        high = float(last['high'])
        low = float(last['low'])
        hit_sl = False
        hit_tp = False
        if direction.startswith("BUY"):
            if low <= sl:
                hit_sl = True
            if high >= tp:
                hit_tp = True
        else:
            if high >= sl:
                hit_sl = True
            if low <= tp:
                hit_tp = True
        if hit_sl:
            close_trade(0, "Stop Loss Hit")
            send_ntfy_alert(f"🔴 صفقة {direction} أُغلقت على وقف الخسارة", "Trade Closed")
            return None
        elif hit_tp:
            close_trade(1, "Take Profit Hit")
            send_ntfy_alert(f"🟢 صفقة {direction} أُغلقت على الهدف", "Trade Closed")
            return None
        return active
    finally:
        conn.close()

# ============================================================
# الواجهة الرئيسية
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #0a0e17; color: #f3f4f6; }
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
</style>
""", unsafe_allow_html=True)

# حفظ الإعدادات من الجلسة السابقة
if 'twelve_key' not in st.session_state:
    st.session_state.twelve_key = load_setting("twelve_api_key", "")
if 'groq_key' not in st.session_state:
    st.session_state.groq_key = load_setting("groq_api_key", "")
if 'ntfy_channel' not in st.session_state:
    st.session_state.ntfy_channel = load_setting("ntfy_channel", "")

# الشريط الجانبي
st.sidebar.header("⚙️ الإعدادات")
api_key = st.sidebar.text_input("مفتاح Twelve Data API", type="password", value=st.session_state.twelve_key)
groq_api_key = st.sidebar.text_input("مفتاح Groq API (اختياري)", type="password", value=st.session_state.groq_key)
ntfy_channel = st.sidebar.text_input("قناة Ntfy", value=st.session_state.ntfy_channel)

# حفظ القيم في الجلسة وقاعدة البيانات
st.session_state.twelve_key = api_key
st.session_state.groq_key = groq_api_key
st.session_state.ntfy_channel = ntfy_channel
save_setting("twelve_api_key", api_key)
save_setting("groq_api_key", groq_api_key)
save_setting("ntfy_channel", ntfy_channel)

symbol = st.sidebar.text_input("رمز الأصل", value="XAUUSD=X")
interval = st.sidebar.selectbox("الإطار الزمني", ["15m", "30m", "1h", "4h", "1d"], index=2)
period = st.sidebar.selectbox("فترة البيانات للتدريب", ["1mo", "3mo", "6mo", "1y"], index=2)
min_confidence = st.sidebar.slider("أدنى ثقة للصفقة (%)", 50, 95, 65, 1)
auto_trade = st.sidebar.checkbox("تفعيل التداول التلقائي", value=False)
use_sentiment = st.sidebar.checkbox("استخدام تحليل المشاعر", value=True)
use_groq = st.sidebar.checkbox("استشارة Groq", value=False)
st.sidebar.markdown("---")
if st.sidebar.button("إعادة تدريب النموذج"):
    threading.Thread(target=background_train, args=(api_key, symbol, interval, period), daemon=True).start()
    st.sidebar.success("بدأ التدريب في الخلفية")

# جلب البيانات ومعالجتها
df_raw = fetch_data(symbol, interval, period, api_key)
if df_raw.empty:
    st.error("فشل في جلب البيانات. تحقق من الإعدادات.")
    st.stop()

df_closed = keep_closed_candles(df_raw)
df_processed = add_all_technical_indicators(df_closed)
df_processed = compute_ict_features(df_processed)
if use_sentiment:
    sentiment_val = get_cached_sentiment()
    df_processed['sentiment_score'] = sentiment_val
else:
    df_processed['sentiment_score'] = 0.0

# تحميل النموذج
model, scaler = load_model()
if model is None and api_key:
    st.info("النموذج غير موجود. سيتم التدريب في الخلفية...")
    threading.Thread(target=background_train, args=(api_key, symbol, interval, period), daemon=True).start()

# توليد الإشارة
signal, err = generate_signal(df_processed, model, scaler, min_confidence,
                              use_groq=use_groq, groq_api_key=groq_api_key)

# فحص الصفقة النشطة
active = check_active_trade(df_processed)

# التداول التلقائي
if auto_trade and not active and signal and signal['direction'] != "NEUTRAL" and signal['confidence'] >= min_confidence:
    open_trade(symbol, signal)
    send_ntfy_alert(f"🟢 صفقة جديدة: {signal['direction']} بثقة {signal['confidence']:.1f}%", "New Trade")
    st.rerun()

# عرض الواجهة
st.title("📈 نظام التداول الآلي — XAU/USD")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.markdown("### الإشارة الحالية")
    if signal:
        direction = signal['direction']
        if direction == "BUY":
            st.markdown(f"<h2 style='color:#22c55e'>شراء 🟢</h2>", unsafe_allow_html=True)
        elif direction == "SELL":
            st.markdown(f"<h2 style='color:#ef4444'>بيع 🔴</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:#94a3b8'>محايد ⚪</h2>", unsafe_allow_html=True)
        st.write(f"الثقة: **{signal['confidence']:.1f}%**")
        if signal['sl'] and signal['tp']:
            st.write(f"الدخول: **{signal['entry']:.2f}**")
            st.write(f"وقف الخسارة: **{signal['sl']:.2f}**")
            st.write(f"جني الأرباح: **{signal['tp']:.2f}**")
        if signal.get('groq_reason'):
            st.write(f"رأي Groq: {signal['groq_reason']}")
    else:
        st.info("لا توجد إشارة حالياً")
with col2:
    st.markdown("### الصفقة النشطة")
    if active:
        st.write(f"الاتجاه: **{active[2]}**")
        st.write(f"الدخول: **{active[3]}**")
        st.write(f"وقف الخسارة: **{active[4]}**")
        st.write(f"جني الأرباح: **{active[5]}**")
        st.write(f"الثقة: **{active[8]}**")
    else:
        st.info("لا توجد صفقة نشطة")
with col3:
    st.markdown("### الإحصائيات")
    total = get_total_trades()
    wins = get_win_trades()
    st.metric("عدد الصفقات", total)
    st.metric("الصفقات الرابحة", wins)
    if total > 0:
        st.metric("نسبة الربح", f"{wins/total*100:.1f}%")
    else:
        st.metric("نسبة الربح", "N/A")

# سجل الصفقات
st.markdown("---")
st.subheader("سجل الصفقات السابقة")
conn = get_db_connection()
try:
    df_trades = pd.read_sql("SELECT date, symbol, direction, entry, sl, tp, win, confidence FROM trades ORDER BY id DESC LIMIT 50", conn)
finally:
    conn.close()
if not df_trades.empty:
    def highlight_win(val):
        if val == 1:
            return 'background-color: #14532d; color: #bbf7d0'
        elif val == 0:
            return 'background-color: #7f1d1d; color: #fecaca'
        else:
            return ''
    st.dataframe(df_trades.style.applymap(highlight_win, subset=['win']), use_container_width=True)
else:
    st.info("لا توجد صفقات سابقة")

# تحديث تلقائي كل 5 دقائق
st_autorefresh(interval=300000, key="autorefresh")
