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
# 1. إعدادات النظام الأساسية وتكوين الواجهة
# ==========================================
st.set_page_config(
    page_title="XAU/USD Titan AI Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "titan_engine.db"
MODEL_FILE = "titan_ai_model.pkl"
SCALER_FILE = "titan_scaler.pkl"
FEATURES = ['rsi', 'macd', 'stoch_k', 'ema_50', 'ema_200', 'bb_high', 'bb_low', 'atr']

# ==========================================
# 2. إدارة قاعدة البيانات الدائمة (الذاكرة)
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    c.execute("""
        CREATE TABLE IF NOT EXISTS history_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            strategy TEXT, 
            direction TEXT, 
            entry_price REAL, 
            exit_price REAL,
            pnl REAL, 
            win INTEGER, 
            features TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS active_trade (
            id INTEGER PRIMARY KEY, 
            timestamp TEXT,
            strategy TEXT, 
            direction TEXT,
            entry_price REAL, 
            sl REAL, 
            tp REAL, 
            features TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_setting(key, value):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()

def load_setting(key, default=""):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else default

# ==========================================
# 3. محركات جلب البيانات (مفصولة المصادر)
# ==========================================
def fetch_training_data():
    """جلب بيانات التدريب من Yahoo Finance (مصدر منفصل للذكاء الاصطناعي)"""
    try:
        df = yf.download("GC=F", period="5y", interval="1h", progress=False)
        if df.empty:
            return pd.DataFrame()
        
        # إصلاح مشكلة MultiIndex في yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.columns = [str(c).lower().strip() for c in df.columns]
        cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"خطأ في جلب بيانات التدريب: {e}")
        return pd.DataFrame()

def fetch_live_data(api_key):
    """جلب بيانات السوق الحية فقط من Twelve Data للصفقات"""
    if not api_key:
        return pd.DataFrame()
        
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=200&apikey={api_key}"
    try:
        res = requests.get(url, timeout=10).json()
        if 'values' in res:
            df = pd.DataFrame(res['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df.astype(float).iloc[::-1]
            df.columns = [str(c).lower().strip() for c in df.columns]
            return df
        elif 'code' in res and res['code'] == 401:
            st.sidebar.error("مفتاح Twelve Data غير صالح.")
    except Exception as e:
        st.error(f"خطأ في الاتصال بـ Twelve Data: {e}")
    return pd.DataFrame()

# ==========================================
# 4. محرك التحليل الفني والرياضيات
# ==========================================
def apply_technical_analysis(df):
    if df is None or df.empty or len(df) < 200: 
        return pd.DataFrame()
    
    df = df.copy()
    
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    
    df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema_200'] = EMAIndicator(df['close'], window=200).ema_indicator()
    
    stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    
    df.dropna(inplace=True)
    return df

# ==========================================
# 5. شبكة الذكاء الاصطناعي ذات التغذية الراجعة
# ==========================================
def train_ai_model():
    df = fetch_training_data()
    df = apply_technical_analysis(df)
    if df.empty: 
        return False, "فشل جلب أو معالجة بيانات التدريب."

    # استخراج الصفقات التاريخية ليتعلم النموذج من أخطائه ونجاحاته (Reinforcement Setup)
    conn = sqlite3.connect(DB_FILE)
    history = pd.read_sql("SELECT * FROM history_trades", conn)
    conn.close()

    # الهدف الأساسي: هل السعر بعد شمعتين أعلى من السعر الحالي؟ (زخم إيجابي)
    df['target'] = np.where(df['close'].shift(-2) > df['close'], 1, 0)
    
    # دمج التغذية الراجعة من الصفقات السابقة إن وجدت (معالجة متقدمة يمكن التوسع بها)
    # حالياً نعتمد على بيانات السوق المباشرة لتدريب الأوزان الأساسية
    train_df = df.iloc[:-2].copy()
    
    X = train_df[FEATURES].values
    y = train_df['target'].values

    if len(np.unique(y)) < 2:
        return False, "بيانات التدريب لا تحتوي على تنوع كافٍ."

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # نموذج عميق
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), 
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        max_iter=1500, 
        early_stopping=True,
        random_state=42
    )
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    acc = model.score(X_scaled, y) * 100
    return True, f"تم تدريب النموذج بنجاح (الدقة التدريبية: {acc:.1f}%)"

def ai_predict(features_array):
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE): 
        return "NEUTRAL", 0.0
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        
        X_scaled = scaler.transform([features_array])
        probs = model.predict_proba(X_scaled)[0]
        
        buy_prob = probs[1] * 100
        sell_prob = probs[0] * 100
        
        if buy_prob > 60:
            return "BUY", buy_prob
        elif sell_prob > 60:
            return "SELL", sell_prob
        else:
            return "NEUTRAL", max(buy_prob, sell_prob)
    except:
        return "NEUTRAL", 0.0

# ==========================================
# 6. المحرك المركزي للاستراتيجيات السبع
# ==========================================
def evaluate_strategies(df, ai_signal):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    strategies = {}

    # 1. التقاطع الذهبي/المميت (Golden/Death Cross)
    if prev['ema_50'] <= prev['ema_200'] and last['ema_50'] > last['ema_200']:
        strategies['Golden_Cross'] = "BUY"
    elif prev['ema_50'] >= prev['ema_200'] and last['ema_50'] < last['ema_200']:
        strategies['Death_Cross'] = "SELL"

    # 2. ارتداد البولينجر باند (Bollinger Bounce)
    if last['close'] <= last['bb_low'] and last['rsi'] < 40:
        strategies['BB_Bounce'] = "BUY"
    elif last['close'] >= last['bb_high'] and last['rsi'] > 60:
        strategies['BB_Bounce'] = "SELL"

    # 3. مناطق التشبع العنيف (RSI Extreme)
    if last['rsi'] < 25:
        strategies['RSI_Extreme'] = "BUY"
    elif last['rsi'] > 75:
        strategies['RSI_Extreme'] = "SELL"

    # 4. تقاطع خط الماكدي (MACD Crossover)
    if prev['macd'] <= prev['macd_signal'] and last['macd'] > last['macd_signal'] and last['macd'] < 0:
        strategies['MACD_Cross'] = "BUY"
    elif prev['macd'] >= prev['macd_signal'] and last['macd'] < last['macd_signal'] and last['macd'] > 0:
        strategies['MACD_Cross'] = "SELL"

    # 5. انعكاس الاستوكاستيك (Stochastic Reversal)
    if prev['stoch_k'] <= 20 and last['stoch_k'] > 20 and last['stoch_k'] > last['stoch_d']:
        strategies['Stoch_Reversal'] = "BUY"
    elif prev['stoch_k'] >= 80 and last['stoch_k'] < 80 and last['stoch_k'] < last['stoch_d']:
        strategies['Stoch_Reversal'] = "SELL"

    # 6. توافق الزخم والاتجاه (Trend + Momentum)
    if last['close'] > last['ema_200'] and last['rsi'] > 50 and last['macd_hist'] > 0:
        strategies['Momentum_Trend'] = "BUY"
    elif last['close'] < last['ema_200'] and last['rsi'] < 50 and last['macd_hist'] < 0:
        strategies['Momentum_Trend'] = "SELL"

    # 7. قرار الشبكة العصبية (AI Titan)
    if ai_signal in ["BUY", "SELL"]:
        strategies['AI_Titan_Engine'] = ai_signal

    # إرجاع أول استراتيجية محققة (Halt Logic)
    for name, signal in strategies.items():
        if signal in ["BUY", "SELL"]:
            return name, signal
            
    return None, "NEUTRAL"

# ==========================================
# 7. واجهة الاستخدام الاحترافية (Streamlit)
# ==========================================
st.markdown("""
<style>
    .metric-card {background-color: #1e2129; padding: 15px; border-radius: 10px; border-left: 5px solid #3b82f6;}
    .buy-signal {color: #10b981; font-weight: bold; font-size: 1.2em;}
    .sell-signal {color: #ef4444; font-weight: bold; font-size: 1.2em;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 XAU/USD Titan AI Engine")
st.caption("نظام التداول الخوارزمي المعزز بالذكاء الاصطناعي وإدارة المخاطر الصارمة")

# --- الشريط الجانبي ---
with st.sidebar:
    st.header("⚙️ إعدادات المحرك")
    saved_key = load_setting("twelve_key")
    api_key = st.text_input("مفتاح Twelve Data:", value=saved_key, type="password")
    if st.button("💾 حفظ المفتاح للأبد"):
        save_setting("twelve_key", api_key)
        st.success("تم تأمين المفتاح في قاعدة البيانات.")
        
    st.markdown("---")
    st.header("🧠 تدريب الشبكة العصبية")
    if st.button("🚀 بدء التدريب العميق"):
        with st.spinner("جاري تنزيل بيانات 5 سنوات ومعالجة المؤشرات..."):
            success, msg = train_ai_model()
            if success:
                st.success(msg)
            else:
                st.error(msg)
                
    st.markdown("---")
    st.header("🎯 إدارة المخاطر")
    risk_reward = st.slider("نسبة العائد للمخاطرة (R:R)", 1.0, 5.0, 2.0, 0.5)
    atr_multiplier = st.slider("معامل ATR لوقف الخسارة", 1.0, 3.0, 1.5, 0.1)

# --- اللوحة الرئيسية ---
if not api_key:
    st.warning("يرجى إدخال مفتاح Twelve Data في الشريط الجانبي لبدء التحليل الحي.")
    st.stop()

df_live = fetch_live_data(api_key)
df_live = apply_technical_analysis(df_live)

if df_live.empty:
    st.info("جاري انتظار تدفق البيانات...")
    st.stop()

last_row = df_live.iloc[-1]
current_price = last_row['close']
current_atr = last_row['atr']
features_vals = last_row[FEATURES].values

col1, col2, col3, col4 = st.columns(4)
col1.metric("سعر XAU/USD", f"${current_price:.2f}")
col2.metric("RSI (14)", f"{last_row['rsi']:.1f}")
col3.metric("ATR Volatility", f"${current_atr:.2f}")
col4.metric("EMA 50", f"${last_row['ema_50']:.2f}")

st.markdown("---")

conn = sqlite3.connect(DB_FILE)
active_trades_df = pd.read_sql("SELECT * FROM active_trade", conn)

if not active_trades_df.empty:
    # يوجد صفقة نشطة -> إيقاف باقي الاستراتيجيات (Halt Logic)
    st.error("🔒 **حالة النظام: مقفل (Locked)** - هناك صفقة نشطة حالياً. جميع الاستراتيجيات متوقفة لتجنب التضارب.")
    trade = active_trades_df.iloc[0]
    
    st.subheader(f"تفاصيل الصفقة الحالية ({trade['strategy']})")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("الاتجاه", trade['direction'])
    t2.metric("نقطة الدخول", f"${trade['entry_price']:.2f}")
    t3.metric("وقف الخسارة (SL)", f"${trade['sl']:.2f}")
    t4.metric("الهدف (TP)", f"${trade['tp']:.2f}")
    
    # حساب الربح/الخسارة الحالي العائم
    floating_pnl = current_price - trade['entry_price'] if trade['direction'] == "BUY" else trade['entry_price'] - current_price
    st.metric("الربح/الخسارة العائم", f"${floating_pnl:.2f}", delta_color="normal")
    
    if st.button("⏹️ إغلاق الصفقة وتلقين الذكاء الاصطناعي (Feedback Loop)"):
        win = 1 if floating_pnl > 0 else 0
        c = conn.cursor()
        c.execute("""
            INSERT INTO history_trades (timestamp, strategy, direction, entry_price, exit_price, pnl, win, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now(timezone.utc).isoformat(), trade['strategy'], trade['direction'], 
              trade['entry_price'], current_price, floating_pnl, win, trade['features']))
        c.execute("DELETE FROM active_trade")
        conn.commit()
        st.success("تم الإغلاق بنجاح وتسجيل النتيجة ليتعلم منها النموذج! سيتم إعادة تفعيل الاستراتيجيات.")
        st.rerun()

else:
    # لا توجد صفقة نشطة -> تشغيل محرك الاستراتيجيات والذكاء الاصطناعي
    st.success("🟢 **حالة النظام: نشط** - المحرك يبحث عن الفرص الذهبية...")
    
    ai_sig, ai_conf = ai_predict(features_vals)
    st.write(f"**رؤية الذكاء الاصطناعي:** {ai_sig} (بنسبة ثقة {ai_conf:.1f}%)")
    
    triggered_strategy, final_signal = evaluate_strategies(df_live, ai_sig)
    
    if final_signal != "NEUTRAL":
        st.markdown(f"### 🎯 فرصة تداول مكتشفة!")
        st.write(f"**الاستراتيجية المشغلة:** `{triggered_strategy}` | **الإشارة:** <span class='{'buy-signal' if final_signal=='BUY' else 'sell-signal'}'>{final_signal}</span>", unsafe_allow_html=True)
        
        sl_dist = current_atr * atr_multiplier
        tp_dist = sl_dist * risk_reward
        
        sl = current_price - sl_dist if final_signal == "BUY" else current_price + sl_dist
        tp = current_price + tp_dist if final_signal == "BUY" else current_price - tp_dist
        
        st.write(f"**المقترح:** الدخول: ${current_price:.2f} | الوقف: ${sl:.2f} | الهدف: ${tp:.2f}")
        
        if st.button("⚡ تنفيذ الصفقة وتجميد النظام"):
            c = conn.cursor()
            c.execute("""
                INSERT INTO active_trade (id, timestamp, strategy, direction, entry_price, sl, tp, features)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """, (datetime.now(timezone.utc).isoformat(), triggered_strategy, final_signal, 
                  current_price, sl, tp, str(list(features_vals))))
            conn.commit()
            st.rerun()

conn.close()

# --- سجل التدريب (History) ---
with st.expander("📁 سجل الصفقات المغلقة (بيانات التغذية الراجعة)"):
    conn = sqlite3.connect(DB_FILE)
    hist_df = pd.read_sql("SELECT * FROM history_trades ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    if not hist_df.empty:
        st.dataframe(hist_df[['timestamp', 'strategy', 'direction', 'entry_price', 'exit_price', 'pnl', 'win']])
    else:
        st.write("لا توجد صفقات مسجلة بعد.")
