from datetime import datetime, timezone
import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="XAU/USD Deep AI Engine",
    layout="wide",
    page_icon="🧠",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; }
    .stApp { background-color: #07090e; color: #f3f4f6; }
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
    .ai-level-card { background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); padding: 30px; border-radius: 20px; text-align: center; border: 1px solid #3b82f6; box-shadow: 0 0 25px rgba(59, 130, 246, 0.4); margin-bottom: 20px;}
    .ai-level-title { font-size: 1.2rem; color: #93c5fd; font-weight: 700; margin-bottom: 10px; }
    .ai-level-value { font-size: 4rem; font-weight: 900; color: #fbbf24; line-height: 1; }
    .ai-level-sub { font-size: 1rem; color: #64748b; margin-top: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

DB_FILE = 'xau_deep_ai.db'
MODEL_FILE = 'xau_deep_mlp.pkl'
SCALER_FILE = 'xau_deep_scaler.pkl'


# --- قواعد البيانات وتخزين الخبرة ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, symbol TEXT, direction TEXT, entry REAL, sl REAL, tp REAL, win INTEGER, note TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS active_trade (id INTEGER PRIMARY KEY, symbol TEXT, direction TEXT, entry REAL, sl REAL, tp REAL, peak REAL, time TEXT)""")
    conn.commit()
    conn.close()

init_db()

def save_setting(key, val):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, str(val)))
    conn.commit()
    conn.close()

def load_setting(key, default=''):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT value FROM settings WHERE key = ?', (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else default

def get_successful_trades_count():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM trades WHERE win = 1')
    count = c.fetchone()[0]
    conn.close()
    return count

# --- القائمة الجانبية ---
st.sidebar.header('⚙️ إعدادات الذكاء الاصطناعي')
twelve_key = st.sidebar.text_input('مفتاح Twelve Data API', type='password', value=load_setting('twelve_key', ''))
save_setting('twelve_key', twelve_key)

ntfy_channel = st.sidebar.text_input('قناة Ntfy للتنبيهات', value=load_setting('ntfy', 'xau_deep_channel'))
save_setting('ntfy', ntfy_channel)

if not twelve_key:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('DELETE FROM active_trade')
    conn.commit()
    conn.close()

st.sidebar.markdown('---')
st.sidebar.header('🎯 إدارة المخاطر')
atr_mult = st.sidebar.slider('معامل الوقف ATR', 1.0, 3.0, 1.5, 0.1)
risk_reward = st.sidebar.slider('نسبة العائد (R:R)', 1.5, 4.0, 2.0, 0.5)
min_conf = st.sidebar.slider('أدنى ثقة مطلوبة (%)', 60, 95, 75, 1)

def send_alert(msg, title='🧠 Deep AI Alert'):
    if ntfy_channel:
        ch = ntfy_channel.strip().split('/')[-1]
        try:
            requests.post(
                f'https://ntfy.sh/{ch}',
                data=msg.encode('utf-8'),
                headers={'Title': title, 'Priority': 'high'},
                timeout=4,
            )
        except Exception:
            pass

# --- مصادر البيانات والتحليل المزدوج ---
# 1. التدريب الخلفي باستخدام YFinance (الذهب + الدولار)
@st.cache_data(ttl=86400) # تحديث بيانات التدريب يومياً فقط لتخفيف الضغط
def fetch_background_training_data():
    try:
        gold = yf.download("GC=F", period="1y", interval="1d", progress=False)
        dxy = yf.download("DX-Y.NYB", period="1y", interval="1d", progress=False)
        if gold.empty or dxy.empty:
            return pd.DataFrame()

        df = pd.DataFrame()
        df['close'] = gold['Close'].squeeze()
        df['high'] = gold['High'].squeeze()
        df['low'] = gold['Low'].squeeze()
        df['dxy_close'] = dxy['Close'].squeeze()
        
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

# 2. البيانات اللحظية من Twelve Data (الذهب فقط للتنفيذ الدقيق)
def fetch_live_data_twelve(limit=100):
    if not twelve_key:
        return pd.DataFrame()
    try:
        url = f'https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize={limit}&apikey={twelve_key}'
        res = requests.get(url, timeout=6).json()
        if 'values' in res:
            df = pd.DataFrame(res['values'])[['open', 'high', 'low', 'close']].astype(float)
            return df.iloc[::-1].reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()

def apply_deep_indicators(df, is_training=False):
    if df is None or df.empty or len(df) < 52:
        return pd.DataFrame()
    
    tr = pd.concat([df['high'] - df['low'], np.abs(df['high'] - df['close'].shift()), np.abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi'] = 100 - (100 / (1 + rs))

    # إضافة تأثير الدولار إذا كنا في مرحلة التدريب
    if is_training and 'dxy_close' in df.columns:
        df['dxy_roc'] = df['dxy_close'].pct_change(periods=14) * 100
    else:
        df['dxy_roc'] = 0.0 # قيمة محايدة أثناء التنفيذ الحي لعدم استهلاك API اضافي

    df.dropna(inplace=True)
    return df

# --- محرك التعلم العميق (Deep Learning Engine) ---
def train_deep_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)

    df_train = fetch_background_training_data()
    if df_train.empty:
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        scaler = StandardScaler()
        return model, scaler

    df_train = apply_deep_indicators(df_train, is_training=True)
    features = ['atr', 'ema_50', 'ema_200', 'rsi', 'dxy_roc']
    
    if df_train.empty or len(df_train) < 50:
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        scaler = StandardScaler()
        return model, scaler

    X = df_train[features].values[:-1]
    y = np.where(df_train['close'].shift(-1) > df_train['close'], 1, 0)[:-1]

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    
    # شبكة عصبية عميقة تتعلم العلاقات المعقدة بين الدولار والذهب والمؤشرات الفنية
    model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
    model.fit(X_sc, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return model, scaler

model, scaler = train_deep_model()

def execute_autonomous_scan():
    if not twelve_key:
        return 'النظام متوقف: يرجى إدخال مفتاح Twelve Data API.'

    conn = sqlite3.connect(DB_FILE)
    df_act = pd.read_sql('SELECT * FROM active_trade WHERE id = 1', conn)
    conn.close()

    if not df_act.empty:
        return 'الذكاء الاصطناعي يراقب الصفقة النشطة حالياً.'

    df_live = fetch_live_data_twelve(150)
    if df_live.empty or len(df_live) < 52:
        return 'لا توجد صفقة: جاري جمع بيانات السوق الحية.'

    df_proc = apply_deep_indicators(df_live, is_training=False)
    if df_proc.empty:
        return 'لا توجد صفقة: الشروط الفنية لم تكتمل بعد.'

    last = df_proc.iloc[-1]
    feat = ['atr', 'ema_50', 'ema_200', 'rsi', 'dxy_roc']

    try:
        x_in = scaler.transform(last[feat].values.reshape(1, -1))
        probs = model.predict_proba(x_in)[0]
        pred = np.argmax(probs)
        conf = probs[pred] * 100
    except Exception:
        return 'الشبكة العصبية قيد التهيئة.'

    curr = round(last['close'], 2)
    atr_v = last['atr']
    sl_d = round(atr_v * atr_mult, 2)
    tp_d = round(sl_d * risk_reward, 2)

    if conf >= min_conf:
        direction = 'BUY 🟢' if pred == 1 else 'SELL 🔴'
        sl_p = round(curr - sl_d, 2) if pred == 1 else round(curr + sl_d, 2)
        tp_p = round(curr + tp_d, 2) if pred == 1 else round(curr - tp_d, 2)

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('DELETE FROM active_trade')
        c.execute('INSERT INTO active_trade (id, symbol, direction, entry, sl, tp, peak, time) VALUES (1, ?, ?, ?, ?, ?, ?, ?)',
                  ('XAU/USD', direction, curr, sl_p, tp_p, curr, str(datetime.now(timezone.utc).strftime('%H:%M:%S'))))
        conn.commit()
        conn.close()

        send_alert(f'🧠 AI Trade Executed: {direction}\nEntry: ${curr}\nSL: ${sl_p}\nTP: ${tp_p}\nAI Confidence: {conf:.1f}%')
        return f'تم اتخاذ قرار ({direction}) بثقة {conf:.1f}% وتم إرسال التنبيه.'

    return f'لا توجد صفقة: ثقة الذكاء الاصطناعي الحالية ({conf:.1f}%) أقل من المطلوب ({min_conf}%).'


# --- واجهة المستخدم (التركيز على مستوى الذكاء الاصطناعي فقط) ---
st.title("🧠 نظام التداول العميق — XAU/USD")

# حساب مستوى الذكاء الاصطناعي (AI Level = عدد الصفقات الناجحة)
success_count = get_successful_trades_count()
ai_level = max(1, int(success_count * 1.5)) # معادلة بسيطة لتطور المستوى

st.markdown(f"""
<div class="ai-level-card">
    <div class="ai-level-title">AI EVOLUTION LEVEL</div>
    <div class="ai-level-value">Lvl. {ai_level}</div>
    <div class="ai-level-sub">Successful Trades: {success_count} | Deep Neural Network Active</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(['⚡ حالة الذكاء الاصطناعي', '📊 سجل الخبرات المكتسبة (الصفقات)'])

with tab1:
    if not twelve_key:
        st.warning('⚠️ النظام نائم: أدخل مفتاح Twelve Data API لإيقاظ الشبكة العصبية وربطها بالسوق.')

    with st.spinner('الشبكة العصبية تحلل البيانات...'):
        scan_msg = execute_autonomous_scan()

    conn = sqlite3.connect(DB_FILE)
    df_act = pd.read_sql('SELECT * FROM active_trade WHERE id = 1', conn)
    conn.close()

    if not df_act.empty and twelve_key:
        t = df_act.iloc[0]
        st.warning(f"🔒 **الشبكة العصبية تدير صفقة حالياً:** {t['direction']} | الدخول: ${t['entry']} | SL: ${t['sl']} | TP: ${t['tp']}")
    else:
        st.info(f'🔍 {scan_msg}')

with tab2:
    conn = sqlite3.connect(DB_FILE)
    df_log = pd.read_sql('SELECT * FROM trades ORDER BY id DESC', conn)
    conn.close()
    if not df_log.empty:
        st.dataframe(df_log, use_container_width=True)
    else:
        st.info('لا توجد صفقات مغلقة مسجلة حتى الآن. الشبكة العصبية بانتظار أول نجاح.')

# --- المراقبة التلقائية (كل 60 ثانية) ---
st_autorefresh(interval=60000, key='deep_ai_loop')

if twelve_key:
    conn = sqlite3.connect(DB_FILE)
    c_active = pd.read_sql('SELECT * FROM active_trade WHERE id = 1', conn)
    conn.close()

    if not c_active.empty:
        t_row = c_active.iloc[0]
        df_check = fetch_live_data_twelve(50)
        
        if not df_check.empty and len(df_check) >= 52:
            df_analyzed = apply_deep_indicators(df_check, is_training=False)
            
            if not df_analyzed.empty:
                last_row = df_analyzed.iloc[-1]
                feat_list = ['atr', 'ema_50', 'ema_200', 'rsi', 'dxy_roc']
                
                try:
                    x_current = scaler.transform(last_row[feat_list].values.reshape(1, -1))
                    current_probs = model.predict_proba(x_current)[0]
                    curr_pred = np.argmax(current_probs)
                    curr_conf = current_probs[curr_pred] * 100

                    is_buy_trade = 'BUY' in t_row['direction']
                    reversal_detected = False

                    if is_buy_trade and curr_pred == 0 and curr_conf >= (min_conf - 5):
                        reversal_detected = True
                    elif not is_buy_trade and curr_pred == 1 and curr_conf >= (min_conf - 5):
                        reversal_detected = True

                    if reversal_detected:
                        send_alert(f'⚠️ تنبيه من الشبكة العصبية: رصد انعكاس للسوق ضد الصفقة ({t_row["direction"]}) بقوة ({curr_conf:.1f}%).', '🚨 AI Reversal Warning')
                except Exception:
                    pass

                h, l = last_row['high'], last_row['low']
                hit_sl, hit_tp = False, False

                if is_buy_trade:
                    if l <= t_row['sl']: hit_sl = True
                    elif h >= t_row['tp']: hit_tp = True
                else:
                    if h >= t_row['sl']: hit_sl = True
                    elif l <= t_row['tp']: hit_tp = True

                if hit_sl or hit_tp:
                    win_val = 1 if hit_tp else 0
                    note_str = 'AI Target Reached (تم التعلم بنجاح)' if hit_tp else 'AI Stop Loss Hit (خطأ وتم الاستيعاب)'

                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute('INSERT INTO trades (date, symbol, direction, entry, sl, tp, win, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                              (str(datetime.now(timezone.utc).date()), t_row['symbol'], t_row['direction'], t_row['entry'], t_row['sl'], t_row['tp'], win_val, note_str))
                    c.execute('DELETE FROM active_trade')
                    conn.commit()
                    conn.close()

                    send_alert(f'Closed {t_row["symbol"]} {t_row["direction"]} -> {note_str}', '🧠 AI Trade Settled')
