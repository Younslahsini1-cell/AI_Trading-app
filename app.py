import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
import requests
import joblib
import os
from datetime import datetime, timezone
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# ==========================================
# 1. إعدادات النظام وقاعدة البيانات الدائمة
# ==========================================
DB_FILE = "titan_engine.db"
MODEL_FILE = "titan_ai_model.pkl"
SCALER_FILE = "titan_scaler.pkl"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # جدول الإعدادات والمفاتيح
    c.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    # جدول الصفقات المغلقة (للتدريب)
    c.execute("""
        CREATE TABLE IF NOT EXISTS history_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, direction TEXT, entry_price REAL, exit_price REAL,
            pnl REAL, win INTEGER, features TEXT
        )
    """)
    # جدول الصفقة النشطة (صفقة واحدة فقط)
    c.execute("""
        CREATE TABLE IF NOT EXISTS active_trade (
            id INTEGER PRIMARY KEY, strategy TEXT, direction TEXT,
            entry_price REAL, sl REAL, tp REAL, features TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_setting(key, value):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def load_setting(key):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""

# ==========================================
# 2. مصادر البيانات (مفصولة كما طلبت)
# ==========================================
def fetch_training_data():
    """جلب بيانات التدريب من Yahoo Finance (مصدر منفصل للذكاء الاصطناعي)"""
    try:
        # GC=F هو رمز عقود الذهب الآجلة، يعكس XAUUSD
        df = yf.download("GC=F", period="2y", interval="1h", progress=False)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"خطأ في جلب بيانات التدريب: {e}")
        return pd.DataFrame()

def fetch_live_data(api_key):
    """جلب بيانات السوق الحية فقط من Twelve Data للصفقات"""
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=100&apikey={api_key}"
    try:
        res = requests.get(url).json()
        if 'values' in res:
            df = pd.DataFrame(res['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df.astype(float).iloc[::-1] # ترتيب زمني صحيح
            return df
    except Exception as e:
        st.error(f"خطأ في جلب بيانات Twelve Data: {e}")
    return pd.DataFrame()

# ==========================================
# 3. محرك الرياضيات والمؤشرات الفنية (Indicators)
# ==========================================
def apply_technical_analysis(df):
    if df.empty or len(df) < 50: return df
    
    df = df.copy()
    # 1. RSI
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    # 2. MACD
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    # 3. Bollinger Bands
    bb = BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    # 4. EMAs
    df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema_200'] = EMAIndicator(df['close'], window=200).ema_indicator()
    # 5. Stochastic
    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    # 6. ATR (لإدارة المخاطر)
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    df.dropna(inplace=True)
    return df

# ==========================================
# 4. الذكاء الاصطناعي (AI Training & Feedback Loop)
# ==========================================
FEATURES = ['rsi', 'macd', 'stoch_k', 'ema_50', 'ema_200', 'bb_high', 'bb_low']

def train_ai_model():
    df = fetch_training_data()
    df = apply_technical_analysis(df)
    if df.empty: return False

    # دمج البيانات التاريخية مع الصفقات الخاطئة والصحيحة من النظام ليتعلم من أخطائه
    conn = sqlite3.connect(DB_FILE)
    history = pd.read_sql("SELECT * FROM history_trades", conn)
    conn.close()

    # بناء بيانات التدريب (التوقع المستقبلي للسعر)
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    
    X = df[FEATURES].values
    y = df['target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # شبكة عصبية عميقة تتعلم سلوك الذهب
    model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return True

def ai_predict(features_array):
    if not os.path.exists(MODEL_FILE): return "NEUTRAL"
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    X_scaled = scaler.transform([features_array])
    pred = model.predict(X_scaled)[0]
    return "BUY" if pred == 1 else "SELL"

# ==========================================
# 5. محرك الـ 7 استراتيجيات والتحكم المركزي
# ==========================================
def evaluate_strategies(df, ai_signal):
    last = df.iloc[-1]
    strategies = {}

    # 1. استراتيجية تقاطع المتوسطات (Golden/Death Cross)
    if last['ema_50'] > last['ema_200']: strategies['EMA_Cross'] = "BUY"
    elif last['ema_50'] < last['ema_200']: strategies['EMA_Cross'] = "SELL"

    # 2. استراتيجية الانعكاس من البولينجر باند
    if last['close'] <= last['bb_low']: strategies['BB_Bounce'] = "BUY"
    elif last['close'] >= last['bb_high']: strategies['BB_Bounce'] = "SELL"

    # 3. استراتيجية التشبع (RSI)
    if last['rsi'] < 30: strategies['RSI_Extreme'] = "BUY"
    elif last['rsi'] > 70: strategies['RSI_Extreme'] = "SELL"

    # 4. استراتيجية MACD
    if last['macd'] > last['macd_signal']: strategies['MACD_Trend'] = "BUY"
    elif last['macd'] < last['macd_signal']: strategies['MACD_Trend'] = "SELL"

    # 5. الاستوكاستيك
    if last['stoch_k'] < 20: strategies['Stoch_Reversal'] = "BUY"
    elif last['stoch_k'] > 80: strategies['Stoch_Reversal'] = "SELL"

    # 6. التوافق المزدوج (زخم + اتجاه)
    if last['rsi'] < 40 and last['close'] > last['ema_200']: strategies['Momentum_Trend'] = "BUY"
    elif last['rsi'] > 60 and last['close'] < last['ema_200']: strategies['Momentum_Trend'] = "SELL"

    # 7. قرار الذكاء الاصطناعي (AI Titan)
    strategies['AI_Titan'] = ai_signal

    # نظام الإيقاف: إرجاع أول استراتيجية تعطي إشارة، وتجاهل الباقي
    for name, signal in strategies.items():
        if signal in ["BUY", "SELL"]:
            return name, signal
    return None, "NEUTRAL"

# ==========================================
# 6. واجهة المستخدم والتنفيذ (Streamlit UI)
# ==========================================
st.set_page_config(page_title="XAU/USD Titan AI Engine", layout="wide")
st.title("🤖 نظام التداول العملاق XAU/USD (Titan AI)")

# إدارة المفاتيح (تحفظ للأبد)
saved_key = load_setting("twelve_key")
api_key = st.sidebar.text_input("مفتاح Twelve Data:", value=saved_key, type="password")
if st.sidebar.button("حفظ الإعدادات للأبد"):
    save_setting("twelve_key", api_key)
    st.sidebar.success("تم الحفظ بنجاح في قاعدة البيانات!")

if st.sidebar.button("🧠 تدريب الذكاء الاصطناعي (Feed Model)"):
    with st.spinner("جاري جلب البيانات من Yahoo Finance وتدريب الشبكات العصبية..."):
        if train_ai_model():
            st.sidebar.success("تم التدريب بنجاح!")
        else:
            st.sidebar.error("فشل التدريب.")

if api_key:
    df_live = fetch_live_data(api_key)
    df_live = apply_technical_analysis(df_live)
    
    if not df_live.empty:
        current_price = df_live.iloc[-1]['close']
        current_atr = df_live.iloc[-1]['atr']
        features_vals = df_live.iloc[-1][FEATURES].values
        
        st.metric("سعر الذهب الحالي (XAU/USD)", f"${current_price:.2f}")

        # فحص وجود صفقة نشطة (Halt Logic)
        conn = sqlite3.connect(DB_FILE)
        active = pd.read_sql("SELECT * FROM active_trade", conn)
        
        if not active.empty:
            st.warning("⚠️ هناك صفقة نشطة حالياً. تم إيقاف جميع الاستراتيجيات الأخرى حتى تنتهي هذه الصفقة.")
            trade = active.iloc[0]
            st.write(f"**الاستراتيجية المفعلة:** {trade['strategy']} | **الاتجاه:** {trade['direction']}")
            st.write(f"الدخول: {trade['entry_price']} | SL: {trade['sl']} | TP: {trade['tp']}")
            
            # محاكاة إغلاق الصفقة (لأغراض التدريب والتعلم)
            if st.button("إغلاق الصفقة وتلقين الذكاء الاصطناعي (Feedback)"):
                # حساب الربح/الخسارة الوهمي للتوضيح
                pnl = current_price - trade['entry_price'] if trade['direction'] == "BUY" else trade['entry_price'] - current_price
                win = 1 if pnl > 0 else 0
                
                c = conn.cursor()
                c.execute("""
                    INSERT INTO history_trades (strategy, direction, entry_price, exit_price, pnl, win, features)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (trade['strategy'], trade['direction'], trade['entry_price'], current_price, pnl, win, trade['features']))
                c.execute("DELETE FROM active_trade")
                conn.commit()
                st.success("تم الإغلاق وتغذية النموذج بنتيجة الصفقة ليتعلم منها!")
                st.rerun()
        else:
            st.info("✅ جميع الاستراتيجيات تعمل الآن وتبحث عن فرص...")
            ai_sig = ai_predict(features_vals)
            triggered_strategy, final_signal = evaluate_strategies(df_live, ai_sig)
            
            if final_signal != "NEUTRAL":
                st.success(f"🚀 تم اكتشاف فرصة! الاستراتيجية: {triggered_strategy} | الإشارة: {final_signal}")
                
                if st.button("تنفيذ الصفقة وإيقاف باقي الاستراتيجيات"):
                    sl = current_price - (current_atr * 2) if final_signal == "BUY" else current_price + (current_atr * 2)
                    tp = current_price + (current_atr * 4) if final_signal == "BUY" else current_price - (current_atr * 4)
                    
                    c = conn.cursor()
                    c.execute("""
                        INSERT INTO active_trade (id, strategy, direction, entry_price, sl, tp, features)
                        VALUES (1, ?, ?, ?, ?, ?, ?)
                    """, (triggered_strategy, final_signal, current_price, sl, tp, str(list(features_vals))))
                    conn.commit()
                    st.rerun()
        conn.close()
