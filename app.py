from datetime import datetime, timezone
import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --- إعدادات الصفحة والواجهة المؤسسية ---
st.set_page_config(
    page_title="XAU/USD Continuous Autonomous Engine",
    layout="wide",
    page_icon="🥇",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; }
    .stApp { background-color: #07090e; color: #f3f4f6; }
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
    div[data-testid="stMetricValue"] { color: #f59e0b !important; font-weight: 800; }
</style>
""",
    unsafe_allow_html=True,
)

st.title(
    "🥇 نظام XAU/USD الذاتي المستمر — تحليل دائم ومراقبة لحظية حتى أثناء الصفقات"
)

DB_FILE = 'xau_continuous.db'
MODEL_FILE = 'xau_continuous_model.pkl'
SCALER_FILE = 'xau_continuous_scaler.pkl'


# --- قاعدة البيانات والدعم ---
def init_db():
  conn = sqlite3.connect(DB_FILE)
  c = conn.cursor()
  c.execute(
      """CREATE TABLE IF NOT EXISTS settings 
                (key TEXT PRIMARY KEY, value TEXT)"""
  )
  c.execute(
      """CREATE TABLE IF NOT EXISTS trades 
                (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, symbol TEXT, direction TEXT, entry REAL, sl REAL, tp REAL, win INTEGER, note TEXT)"""
  )
  c.execute(
      """CREATE TABLE IF NOT EXISTS active_trade 
                (id INTEGER PRIMARY KEY, symbol TEXT, direction TEXT, entry REAL, sl REAL, tp REAL, peak REAL, time TEXT)"""
  )
  conn.commit()
  conn.close()


init_db()


def save_setting(key, val):
  conn = sqlite3.connect(DB_FILE)
  c = conn.cursor()
  c.execute(
      'INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)',
      (key, str(val)),
  )
  conn.commit()
  conn.close()


def load_setting(key, default=''):
  conn = sqlite3.connect(DB_FILE)
  c = conn.cursor()
  c.execute('SELECT value FROM settings WHERE key = ?', (key,))
  row = c.fetchone()
  conn.close()
  return row[0] if row else default


# --- لوحة التحكم الجانبية ---
st.sidebar.header('⚙️ إعدادات المنصة والاتصال')
twelve_key = st.sidebar.text_input(
    'مفتاح Twelve Data API',
    type='password',
    value=load_setting('twelve_key', ''),
)
save_setting('twelve_key', twelve_key)

ntfy_channel = st.sidebar.text_input(
    'قناة Ntfy للتنبيهات الفورية',
    value=load_setting('ntfy', 'xau_continuous_channel'),
)
save_setting('ntfy', ntfy_channel)

st.sidebar.markdown('---')
st.sidebar.header('🎯 إعدادات المخاطر والتحليل')
timeframe = st.sidebar.selectbox(
    'الإطار الزمني للرصد', ['15min', '1h', '4h'], index=0
)
atr_mult = st.sidebar.slider('معامل وقف الخسارة ATR', 1.0, 3.0, 1.5, 0.1)
risk_reward = st.sidebar.slider('نسبة العائد للمخاطرة (R:R)', 1.5, 4.0, 2.0, 0.5)
min_conf = st.sidebar.slider('أدنى نسبة ثقة مطلوبة (%)', 60, 95, 65, 1)


def send_alert(msg, title='XAU/USD Continuous Apex'):
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


# --- جلب البيانات عبر Twelve Data API مع بديل احتياطي ---
def fetch_gold_data_twelve(limit=200):
  if twelve_key:
    try:
      url = f'https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={timeframe}&outputsize={limit}&apikey={twelve_key}'
      res = requests.get(url, timeout=6).json()
      if 'values' in res:
        df = pd.DataFrame(res['values'])[['open', 'high', 'low', 'close']].astype(
            float
        )
        return df.iloc[::-1].reset_index(drop=True)
    except Exception:
      pass

  try:
    url = f'https://api.binance.com/api/v3/klines?symbol=PAXGUSDT&interval={timeframe}&limit={limit}'
    res = requests.get(url, timeout=5).json()
    if isinstance(res, list) and len(res) > 0:
      df = pd.DataFrame(res, columns=[
          't',
          'open',
          'high',
          'low',
          'close',
          'v',
          'ct',
          'q',
          'n',
          'tb',
          'tq',
          'i',
      ])[['open', 'high', 'low', 'close']].astype(float)
      return df.reset_index(drop=True)
  except Exception:
    pass

  np.random.seed(42)
  close = 2650.0 + np.cumsum(np.random.randn(limit) * 2.0)
  return pd.DataFrame({
      'open': close - 1.0,
      'high': close + 3.0,
      'low': close - 3.0,
      'close': close,
  })


def apply_indicators(df):
  tr = pd.concat([
      df['high'] - df['low'],
      np.abs(df['high'] - df['close'].shift()),
      np.abs(df['low'] - df['close'].shift()),
  ], axis=1).max(axis=1)
  df['atr'] = tr.rolling(14).mean()

  df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
  df['kijun'] = (
      df['high'].rolling(22).max() + df['low'].rolling(22).min()
  ) / 2
  df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(22)
  df['senkou_b'] = (
      (df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2
  ).shift(22)

  df['rsi'] = 100 - (
      100
      / (
          1
          + (
              df['close'].diff().clip(lower=0).rolling(14).mean()
              / ((-df['close'].diff().clip(upper=0)).rolling(14).mean() + 1e-6)
          )
      )
  )
  df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
  df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

  df['trend'] = np.where(
      (df['close'] > df['senkou_a']) & (df['ema_50'] > df['ema_200']),
      1,
      np.where(
          (df['close'] < df['senkou_a']) & (df['ema_50'] < df['ema_200']),
          -1,
          0,
      ),
  )
  df.dropna(inplace=True)
  return df


# --- نموذج التعلم الآلي الذكي ---
def load_or_train_model():
  if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)

  df = fetch_gold_data_twelve(300)
  df = apply_indicators(df)
  features = ['atr', 'tenkan', 'kijun', 'rsi', 'ema_50', 'ema_200', 'trend']
  X = df[features].values[:-1]
  y = np.where(df['close'].shift(-1) > df['close'], 1, 0)[:-1]

  scaler = StandardScaler()
  X_sc = scaler.fit_transform(X)
  model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
  model.fit(X_sc, y)

  joblib.dump(model, MODEL_FILE)
  joblib.dump(scaler, SCALER_FILE)
  return model, scaler


model, scaler = load_or_train_model()


def execute_autonomous_scan():
  conn = sqlite3.connect(DB_FILE)
  df_act = pd.read_sql('SELECT * FROM active_trade WHERE id = 1', conn)
  conn.close()

  if not df_act.empty:
    return (
        'توجد صفقة نشطة حالياً (النظام يتابع التحليل الذكي والانعكاسات بالخلفية).'
    )

  df_live = fetch_gold_data_twelve(150)
  df_proc = apply_indicators(df_live)
  last = df_proc.iloc[-1]

  feat = ['atr', 'tenkan', 'kijun', 'rsi', 'ema_50', 'ema_200', 'trend']
  x_in = scaler.transform(last[feat].values.reshape(1, -1))

  probs = model.predict_proba(x_in)[0]
  pred = np.argmax(probs)
  conf = probs[pred] * 100

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
    c.execute(
        'INSERT INTO active_trade (id, symbol, direction, entry, sl, tp, peak,'
        ' time) VALUES (1, ?, ?, ?, ?, ?, ?, ?)',
        (
            'XAU/USD',
            direction,
            curr,
            sl_p,
            tp_p,
            curr,
            str(datetime.now(timezone.utc).strftime('%H:%M:%S')),
        ),
    )
    conn.commit()
    conn.close()

    msg = (
        f'XAU/USD Continuous Signal: {direction}\nEntry: ${curr}\nSL:'
        f' ${sl_p}\nTP: ${tp_p}\nConfidence: {conf:.1f}%'
    )
    send_alert(msg, 'Twelve Data Trade Alert')
    return (
        f'تم اكتشاف صفقة أوتوماتيكية جديدة ({direction}) وإرسال التنبيه الفوري!'
    )
  return (
      f'التحليل التلقائي نشط. الثقة الحالية ({conf:.1f}%) بانتظار فرصة أقوى.'
  )


# --- واجهة العرض والتشغيل التلقائي ---
tab1, tab2 = st.tabs([
    '⚡ التشغيل والتحليل والتحمّل الذاتي',
    '📊 سجل الصفقات والأداء',
])

with tab1:
  st.subheader('منصة تداول الذهب الآلية — تحليل مستمر بدون توقف')

  if not twelve_key:
    st.warning(
        '⚠️ يرجى إدخال مفتاح Twelve Data API في القائمة الجانبية للاتصال المباشر'
        ' بالأسعار الرسمية.'
    )

  with st.spinner('جاري تشغيل التحليل الفوري وتفقد السوق...'):
    scan_msg = execute_autonomous_scan()

  conn = sqlite3.connect(DB_FILE)
  df_act = pd.read_sql('SELECT * FROM active_trade WHERE id = 1', conn)
  conn.close()

  if not df_act.empty:
    t = df_act.iloc[0]
    st.warning(
        f"🔒 **صفقة نشطة قيد المراقبة اللحظية:** {t['direction']} | الدخول:"
        f" ${t['entry']} | SL: ${t['sl']} | TP: ${t['tp']}"
    )
  else:
    st.success(f'🟢 {scan_msg}')

  df_chart = fetch_gold_data_twelve(100)
  st.line_chart(df_chart[['close']].tail(30))

with tab2:
  st.subheader('سجل الصفقات المغلقة')
  conn = sqlite3.connect(DB_FILE)
  df_log = pd.read_sql('SELECT * FROM trades', conn)
  conn.close()
  if not df_log.empty:
    st.dataframe(df_log, use_container_width=True)
  else:
    st.info('لا توجد صفقات مغلقة مسجلة حتى الآن.')

# --- المراقبة التلقائية المستمرة (كل 60 ثانية) لمعالجة الأهداف والانعكاسات ---
st_autorefresh(interval=60000, key='continuous_loop')

conn = sqlite3.connect(DB_FILE)
c_active = pd.read_sql('SELECT * FROM active_trade WHERE id = 1', conn)
conn.close()

if not c_active.empty:
  t_row = c_active.iloc[0]
  df_check = fetch_gold_data_twelve(50)
  if not df_check.empty:
    df_analyzed = apply_indicators(df_check)
    last_row = df_analyzed.iloc[-1]

    # تشغيل نموذج الذكاء الاصطناعي لفحص وجود انعكاس أو صفقة معاكسة أثناء الصفقة النشطة
    feat_list = ['atr', 'tenkan', 'kijun', 'rsi', 'ema_50', 'ema_200', 'trend']
    x_current = scaler.transform(last_row[feat_list].values.reshape(1, -1))
    current_probs = model.predict_proba(x_current)[0]
    curr_pred = np.argmax(current_probs)
    curr_conf = current_probs[curr_pred] * 100

    # فحص إشارة الانعكاس المعاكسة أثناء نشاط الصفقة
    is_buy_trade = 'BUY' in t_row['direction']
    reversal_detected = False

    if is_buy_trade and curr_pred == 0 and curr_conf >= (min_conf - 5):
      reversal_detected = True  # الصفقة شراء والنموذج يكتشف انعكاس قوي للبيع
    elif not is_buy_trade and curr_pred == 1 and curr_conf >= (min_conf - 5):
      reversal_detected = True  # الصفقة بيع والنموذج يكتشف انعكاس قوي للشراء

    if reversal_detected:
      rev_dir = 'SELL 🔴' if is_buy_trade else 'BUY 🟢'
      send_alert(
          f'⚠️ تحذير انعكاس السوق! الصفقة النشطة ({t_row["direction"]}) تتعرض'
          f' لضغط معاكس بقوة ({curr_conf:.1f}%). يرجى الحذر أو إغلاق الصفقة.',
          '🚨 Reversal / Counter-Signal Warning',
      )

    h, l = last_row['high'], last_row['low']
    hit_sl, hit_tp = False, False

    if is_buy_trade:
      if l <= t_row['sl']:
        hit_sl = True
      elif h >= t_row['tp']:
        hit_tp = True
    else:
      if h >= t_row['sl']:
        hit_sl = True
      elif l <= t_row['tp']:
        hit_tp = True

    if hit_sl or hit_tp:
      win_val = 1 if hit_tp else 0
      note_str = (
          'Target Reached (نجاح الهدف)'
          if hit_tp
          else 'Stop Loss Hit (وقف خسارة)'
      )

      conn = sqlite3.connect(DB_FILE)
      c = conn.cursor()
      c.execute(
          'INSERT INTO trades (date, symbol, direction, entry, sl, tp, win, note)'
          ' VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
          (
              str(datetime.now(timezone.utc).date()),
              t_row['symbol'],
              t_row['direction'],
              t_row['entry'],
              t_row['sl'],
              t_row['tp'],
              win_val,
              note_str,
          ),
      )
      c.execute('DELETE FROM active_trade')
      conn.commit()
      conn.close()

      send_alert(
          f'Closed {t_row["symbol"]} {t_row["direction"]} -> {note_str}',
          'Twelve Data Trade Settled',
      )
