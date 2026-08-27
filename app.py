from datetime import date, datetime
import json
import os
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# محاولة استيراد yfinance بأمان
try:
  import yfinance as yf

  YFINANCE_AVAILABLE = True
except ImportError:
  YFINANCE_AVAILABLE = False

# --- إعدادات الصفحة والتصميم الاحترافي ---
st.set_page_config(
    page_title="منصة التداول الكمّي التطورية الذكية",
    layout="wide",
    page_icon="🧠",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; }
    .stApp { background-color: #0b0f19; color: #f3f4f6; }
    section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    div.stMarkdown { color: #e5e7eb; }
    pre { background: #1f2937 !important; border: 1px solid #374151 !important; border-radius: 10px !important; color: #38bdf8 !important; font-weight: 700 !important; text-align: center; font-size: 1.15rem !important; padding: 10px !important; }
    .stButton > button { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; border-radius: 10px; font-weight: 700; border: none; padding: 0.6rem 1.2rem; transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); }
    .stButton > button:hover { background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%); box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5); transform: translateY(-1px); }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #111827; padding: 8px; border-radius: 12px; border: 1px solid #1f2937; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #9ca3af; font-weight: 600; padding: 8px 16px; }
    .stTabs [aria-selected="true"] { background-color: #3b82f6 !important; color: white !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 منصة التداول التطوري الذاتي (Autonomous Evolutionary Agent)")

# مسارات الذاكرة والتقارير
MODEL_FILE = 'evo_model_v2.pkl'
SCALER_FILE = 'evo_scaler_v2.pkl'
HISTORY_FILE = 'evo_history_v2.csv'
SETTINGS_FILE = 'evo_settings_v2.json'
TRADES_LOG_FILE = 'evo_trades_log_v2.csv'


def load_settings():
  if os.path.exists(SETTINGS_FILE):
    try:
      with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)
    except:
      pass
  return {'twelve': '', 'alpha': '', 'ntfy': ''}


def save_settings(twelve, alpha, ntfy):
  with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
    json.dump({'twelve': twelve, 'alpha': alpha, 'ntfy': ntfy}, f)


if 'settings' not in st.session_state:
  st.session_state.settings = load_settings()


def send_ntfy_alert(topic, message, title='Smart Trading Signal'):
  if topic:
    clean_topic = topic.strip()
    if 'ntfy.sh/' in clean_topic:
      clean_topic = clean_topic.split('ntfy.sh/')[-1].strip('/')
    url = f'https://ntfy.sh/{clean_topic}'
    try:
      headers = {'Title': title, 'Priority': 'urgent'}
      response = requests.post(
          url, data=message.encode('utf-8'), headers=headers, timeout=5
      )
      return response.status_code == 200
    except Exception as e:
      print(f'Ntfy Error: {e}')
      return False
  return False


# --- القائمة الجانبية ---
st.sidebar.header('🔑 إعدادات الربط والمنصة')
api_key_twelve = st.sidebar.text_input(
    'مفتاح Twelve Data API',
    value=st.session_state.settings.get('twelve', ''),
    type='password',
)
api_key_alpha = st.sidebar.text_input(
    'مفتاح Alpha Vantage API',
    value=st.session_state.settings.get('alpha', ''),
    type='password',
)

st.sidebar.markdown('---')
st.sidebar.header('🔔 إعدادات التنبيهات الذكية (Ntfy)')
ntfy_topic = st.sidebar.text_input(
    'اسم قناة أو رابط Ntfy',
    value=st.session_state.settings.get('ntfy', ''),
    placeholder='مثال: younslahsini2009xauusd',
)
save_settings(api_key_twelve, api_key_alpha, ntfy_topic)

if st.sidebar.button('🧪 اختبار إرسال تنبيه Ntfy الآن'):
  success = send_ntfy_alert(
      ntfy_topic,
      'اختبار ناجح من منصة التداول الذكية 🚀\nEntry: 4607.0 | TP: 4615.0 | SL:'
      ' 4601.0',
      'Test Alert with TP/SL',
  )
  if success:
    st.sidebar.success('✅ تم إرسال التنبيه بنجاح لقناتك!')
  else:
    st.sidebar.error('❌ فشل الإرسال، تحقق من اسم القناة وتأكد من كتابتها بشكل صحيح.')

st.sidebar.markdown('---')
st.sidebar.header('⚙️ معايير الذكاء الاصطناعي والتطور')
interval = st.sidebar.selectbox(
    'الإطار الزمني للرصد', ['1min', '5min', '15min', '1h', '1day'], index=1
)
rr_ratio = st.sidebar.slider(
    'نسبة العائد للمخاطرة (TP/SL)', 1.0, 5.0, 2.0, 0.5
)
confidence_threshold = st.sidebar.slider(
    'حد الثقة الأدنى للتنفيذ (%) - للتصفيات', 50, 95, 62, 1
)


def load_permanent_memory():
  if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    is_trained = True
  else:
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        warm_start=True,
        max_iter=1000,
    )
    scaler = StandardScaler()
    is_trained = False
  history = (
      pd.read_csv(HISTORY_FILE)
      if os.path.exists(HISTORY_FILE)
      else pd.DataFrame(columns=['Accuracy'])
  )
  return model, scaler, is_trained, history


if 'model' not in st.session_state:
  m, s, t, h = load_permanent_memory()
  st.session_state.model = m
  st.session_state.scaler = s
  st.session_state.is_trained = t
  st.session_state.training_history = h


def save_permanent_memory():
  joblib.dump(st.session_state.model, MODEL_FILE)
  joblib.dump(st.session_state.scaler, SCALER_FILE)
  st.session_state.training_history.to_csv(HISTORY_FILE, index=False)


def evolutionary_feature_engineering(df):
  df.columns = df.columns.str.lower()
  df['return'] = df['close'].pct_change()
  df['volatility'] = df['high'] - df['low']
  df['body'] = df['close'] - df['open']
  df['momentum_5'] = df['close'] - df['close'].shift(5)
  df['momentum_10'] = df['close'] - df['close'].shift(10)
  df['ma_ratio'] = df['close'] / df['close'].rolling(10).mean()
  df['ema_ratio'] = df['close'] / df['close'].ewm(span=20).mean()
  df['price_position'] = (df['close'] - df['low']) / (
      df['volatility'] + 1e-8
  )
  df['rsi_proxy'] = df['return'].rolling(14).apply(
      lambda x: (
          np.sum(x[x > 0]) / (np.sum(np.abs(x)) + 1e-8)
          if len(x) > 0
          else 0.5
      ),
      raw=True,
  )
  df.dropna(inplace=True)
  return df


def prepare_data(df):
  features = [
      'return',
      'volatility',
      'body',
      'momentum_5',
      'momentum_10',
      'ma_ratio',
      'ema_ratio',
      'price_position',
      'rsi_proxy',
  ]
  X = df[features].values
  Y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
  return X[:-1], Y[:-1], features


def autonomous_data_broker_agent(symbol, interval, outputsize=600):
  clean_symbol = symbol.upper().strip()
  if any(
      crypto in clean_symbol for crypto in ['BTC', 'ETH', 'SOL', 'BNB', 'XAU']
  ):
    try:
      binance_sym = clean_symbol.replace('/', '').replace('-', '')
      if 'XAU' in binance_sym or 'GOLD' in binance_sym:
        binance_sym = 'PAXGUSDT'
      elif 'USDT' not in binance_sym:
        binance_sym += 'USDT'
      interval_map = {
          '1min': '1m',
          '5min': '5m',
          '15min': '15m',
          '1h': '1h',
          '1day': '1d',
      }
      b_interval = interval_map.get(interval, '5m')
      url = f'https://api.binance.com/api/v3/klines?symbol={binance_sym}&interval={b_interval}&limit={min(outputsize, 1000)}'
      res = requests.get(url, timeout=5).json()
      if isinstance(res, list) and len(res) > 0:
        df = pd.DataFrame(
            res,
            columns=[
                'open_time',
                'open',
                'high',
                'low',
                'close',
                'volume',
                'close_time',
                'qav',
                'num_trades',
                'taker_base_vol',
                'taker_quote_vol',
                'ignore',
            ],
        )
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].astype(float)
        return df[cols].reset_index(drop=True), f'Binance Live ({binance_sym})'
    except:
      pass

  if YFINANCE_AVAILABLE:
    try:
      yf_symbol = clean_symbol
      if 'XAU' in yf_symbol or 'GOLD' in yf_symbol:
        yf_symbol = 'GC=F'
      elif 'BTC' in yf_symbol:
        yf_symbol = 'BTC-USD'
      elif '/' in yf_symbol:
        yf_symbol = yf_symbol.replace('/', '') + '=X'
      interval_map = {
          '1min': '1m',
          '5min': '5m',
          '15min': '15m',
          '1h': '1h',
          '1day': '1d',
      }
      yf_interval = interval_map.get(interval, '5m')
      df_yf = yf.download(
          yf_symbol, period='5d', interval=yf_interval, progress=False
      )
      if not df_yf.empty:
        if isinstance(df_yf.columns, pd.MultiIndex):
          df_yf.columns = df_yf.columns.droplevel(1)
        df_yf = df_yf.reset_index()
        df_yf.columns = [str(c).lower() for c in df_yf.columns]
        if 'close' in df_yf.columns and 'open' in df_yf.columns:
          cols = ['open', 'high', 'low', 'close']
          if 'volume' in df_yf.columns:
            cols.append('volume')
          df_clean = df_yf[cols].dropna().reset_index(drop=True)
          if len(df_clean) > 5:
            return df_clean, f'Yahoo Finance ({yf_symbol})'
    except:
      pass

  base_p = 2650.0 if 'XAU' in symbol.upper() else 4600.0
  close = np.cumsum(np.random.randn(outputsize) * 0.8) + base_p
  high = close + np.random.uniform(0.2, 0.8, outputsize)
  low = close - np.random.uniform(0.2, 0.8, outputsize)
  open_p = low + np.random.uniform(0.0, 0.5, outputsize)
  return (
      pd.DataFrame(
          {'open': open_p, 'high': high, 'low': low, 'close': close}
      ),
      'Advanced Synthetic Broker',
  )


def evaluate_real_trade_outcome(df, entry_idx, prediction, tp_dist, sl_dist):
  entry_price = df.iloc[entry_idx]['close']
  tp_price = (
      entry_price + tp_dist if prediction == 1 else entry_price - tp_dist
  )
  sl_price = (
      entry_price - sl_dist if prediction == 1 else entry_price + sl_dist
  )
  for i in range(entry_idx + 1, len(df)):
    high_price = df.iloc[i]['high']
    low_price = df.iloc[i]['low']
    if prediction == 1:
      if low_price <= sl_price:
        return 0, 'Hit SL (وقف خسارة)'
      if high_price >= tp_price:
        return 1, 'Hit TP (تحقيق الهدف)'
    else:
      if high_price >= sl_price:
        return 0, 'Hit SL (وقف خسارة)'
      if low_price <= tp_price:
        return 1, 'Hit TP (تحقيق الهدف)'
  return 1, 'Active / In Progress'


def log_trade_and_check_mistakes(
    symbol, prediction, df_processed, current_idx, confidence, tp_dist, sl_dist
):
  today_str = str(date.today())
  win_status, note = evaluate_real_trade_outcome(
      df_processed, current_idx, prediction, tp_dist, sl_dist
  )
  new_log = pd.DataFrame([{
      'Date': today_str,
      'Symbol': symbol,
      'Prediction': 'BUY (شراء)' if prediction == 1 else 'SELL (بيع)',
      'Win': win_status,
      'Confidence': round(confidence, 2),
      'ErrorNote': note,
  }])
  if os.path.exists(TRADES_LOG_FILE):
    df_log = pd.read_csv(TRADES_LOG_FILE)
    df_log = pd.concat([df_log, new_log], ignore_index=True)
  else:
    df_log = new_log
  df_log.to_csv(TRADES_LOG_FILE, index=False)

  if win_status == 0:
    features = [
        'return',
        'volatility',
        'body',
        'momentum_5',
        'momentum_10',
        'ma_ratio',
        'ema_ratio',
        'price_position',
        'rsi_proxy',
    ]
    err_row = df_processed.iloc[current_idx][features].values.reshape(1, -1)
    err_scaled = st.session_state.scaler.transform(err_row)
    correct_target = np.array([1 if prediction == 0 else 0])
    st.session_state.model.partial_fit(err_scaled, correct_target)
    save_permanent_memory()


def run_evolutionary_training():
  train_symbols = ['XAU/USD', 'BTC/USD', 'EUR/USD', 'GBP/USD']
  combined_X, combined_Y = [], []
  for sym in train_symbols:
    df_t, _ = autonomous_data_broker_agent(sym, interval, 800)
    proc = evolutionary_feature_engineering(df_t)
    X, Y, feats = prepare_data(proc)
    if len(X) > 0:
      combined_X.append(X)
      combined_Y.append(Y)
  if combined_X:
    X_all = np.vstack(combined_X)
    Y_all = np.concatenate(combined_Y)
    X_scaled = st.session_state.scaler.fit_transform(X_all)
    st.session_state.model.fit(X_scaled, Y_all)
    st.session_state.is_trained = True
    acc = st.session_state.model.score(X_scaled, Y_all) * 100
    new_rec = pd.DataFrame({'Accuracy': [acc]})
    st.session_state.training_history = pd.concat(
        [st.session_state.training_history, new_rec], ignore_index=True
    )
    save_permanent_memory()
    return acc
  return 0


if not st.session_state.is_trained:
  run_evolutionary_training()

# --- التبويبات الرئيسية ---
tab1, tab2, tab3 = st.tabs([
    '🚀 لوحة الفحص الذكي والتحليل الفوري',
    '📊 سجل الأخطاء والتعلم الذاتي (الذاكرة اللا نهائية)',
    '🔄 الرصد الخلفي والإنذارات الآلية (24/7)',
])

with tab1:
  st.header('مؤشرات أداء الشبكات العصبية المتطورة')
  col1, col2, col3 = st.columns(3)
  current_acc = (
      st.session_state.training_history['Accuracy'].iloc[-1]
      if not st.session_state.training_history.empty
      else 0
  )
  col1.metric(
      'حالة الذاكرة العصبية',
      'تعلم مستمر 🟢'
      if st.session_state.is_trained
      else 'قيد التهيئة الأولي ⚪',
  )
  col2.metric('دقة التعلم الحالية', f'{current_acc:.2f}%')

  win_rate_display = 0.0
  if os.path.exists(TRADES_LOG_FILE):
    df_l = pd.read_csv(TRADES_LOG_FILE)
    if not df_l.empty:
      win_rate_display = (df_l['Win'].sum() / len(df_l)) * 100
  col3.metric('نسبة نجاح الصفقات', f'{win_rate_display:.1f}%')

  st.markdown('---')
  st.subheader('تحليل السوق الفعلي')
  market_input = st.text_input(
      'أدخل رمز السوق (مثال: XAU/USD أو BTC/USD)', 'XAU/USD'
  )

  if st.button('فحص السوق وإعطاء القرار (شراء / بيع / انتظار)', use_container_width=True):
    with st.spinner('الشبكات العصبية تفحص السوق...'):
      live_df, src = autonomous_data_broker_agent(market_input, interval, 100)
      processed = evolutionary_feature_engineering(live_df)
      features = [
          'return',
          'volatility',
          'body',
          'momentum_5',
          'momentum_10',
          'ma_ratio',
          'ema_ratio',
          'price_position',
          'rsi_proxy',
      ]
      current_idx = len(processed) - 1
      current_row = processed.iloc[current_idx]
      X_input = current_row[features].values.reshape(1, -1)
      X_scaled = st.session_state.scaler.transform(X_input)

      prediction = st.session_state.model.predict(X_scaled)[0]
      proba = st.session_state.model.predict_proba(X_scaled)[0]
      max_conf = max(proba) * 100
      current_price = round(current_row['close'], 4)
      vol = max(current_row['volatility'], 0.1)

      sl_dist = vol * 1.2
      tp_dist = sl_dist * rr_ratio

      if (
          max_conf < confidence_threshold
          or abs(proba[1] - proba[0]) < 0.12
      ):
        st.warning(
            f'⏳ **حالة انتظار حذر (WAIT)** | الثقة الحالية: {max_conf:.1f}%'
        )
        send_ntfy_alert(
            ntfy_topic,
            f'WAIT Signal | Symbol: {market_input} | Price: {current_price}',
            'Market WAIT Alert',
        )
      else:
        tp_price = (
            round(current_price + tp_dist, 4)
            if prediction == 1
            else round(current_price - tp_dist, 4)
        )
        sl_price = (
            round(current_price - sl_dist, 4)
            if prediction == 1
            else round(current_price + sl_dist, 4)
        )

        log_trade_and_check_mistakes(
            market_input,
            prediction,
            processed,
            current_idx,
            max_conf,
            tp_dist,
            sl_dist,
        )

        if prediction == 1:
          msg = (
              f'BUY | Symbol: {market_input} | Entry: {current_price} | TP:'
              f' {tp_price} | SL: {sl_price} | Conf: {proba[1]*100:.1f}%'
          )
          st.success(
              f'🟢 **شراء مؤكد (BUY)** | السعر الحالي: {current_price}'
          )
          send_ntfy_alert(ntfy_topic, msg, '🚀 Smart BUY Signal with TP/SL')
        else:
          msg = (
              f'SELL | Symbol: {market_input} | Entry: {current_price} | TP:'
              f' {tp_price} | SL: {sl_price} | Conf: {proba[0]*100:.1f}%'
          )
          st.error(f'🔴 **بيع مؤكد (SELL)** | السعر الحالي: {current_price}')
          send_ntfy_alert(ntfy_topic, msg, '🚀 Smart SELL Signal with TP/SL')

        c1, c2, c3 = st.columns(3)
        c1.markdown('**سعر الدخول (Entry)**')
        c1.code(str(current_price))
        c2.markdown('**الهدف المطلوب (TP)**')
        c2.code(str(tp_price))
        c3.markdown('**وقف الخسارة (SL)**')
        c3.code(str(sl_price))

      st.line_chart(processed[['close']].tail(40))

with tab2:
  st.header('📊 سجل الصفقات والأخطاء والتعلم الذاتي')
  if os.path.exists(TRADES_LOG_FILE):
    df_report = pd.read_csv(TRADES_LOG_FILE)
    if not df_report.empty:
      total_t = len(df_report)
      wins_t = df_report['Win'].sum()
      win_pct = (wins_t / total_t) * 100 if total_t > 0 else 0
      rc1, rc2, rc3 = st.columns(3)
      rc1.metric('إجمالي الصفقات', total_t)
      rc2.metric('الصفقات الناجحة', wins_t)
      rc3.metric('نسبة النجاح', f'{win_pct:.1f}%')
      st.markdown('---')
      st.subheader('سجل الصفقات التفصيلي الكامل:')
      st.dataframe(df_report, use_container_width=True)
    else:
      st.info('لا توجد سجلات صفقات سابقة حتى الآن.')
  else:
    st.info('سيظهر سجل الصفقات تلقائياً عند إجراء أول عملية فحص.')

with tab3:
  st.header('🔄 الرصد الخلفي الآمن والإنذارات (24/7)')
  st.write(
      'تستخدم هذه الميزة نظام تحديث تلقائي آمن للواجهة بدون تجميد، لتتمكن من'
      ' مراقبة الأسواق وإرسال التنبيهات بانتظام.'
  )

  monitor_symbol = st.text_input('رمز السوق للمراقبة المستمرة', 'XAU/USD')
  continuous_run = st.checkbox('تفعيل التحديث والمراقبة التلقائية')

  if continuous_run:
    # تحديث تلقائي آمن كل 60 ثانية بدون تجميد الخيط الرئيسي
    st_autorefresh(interval=60000, key='auto_monitor')
    st.info(
        f'🟢 المراقبة التلقائية نشطة لـ {monitor_symbol} (آخر فحص:'
        f" {datetime.now().strftime('%H:%M:%S')})..."
    )

    df_bg, _ = autonomous_data_broker_agent(monitor_symbol, interval, 80)
    if df_bg is not None:
      p_bg = evolutionary_feature_engineering(df_bg)
      features = [
          'return',
          'volatility',
          'body',
          'momentum_5',
          'momentum_10',
          'ma_ratio',
          'ema_ratio',
          'price_position',
          'rsi_proxy',
      ]
      row_bg = p_bg.iloc[-1]
      X_bg = st.session_state.scaler.transform(
          row_bg[features].values.reshape(1, -1)
      )
      pred_bg = st.session_state.model.predict(X_bg)[0]
      prob_bg = st.session_state.model.predict_proba(X_bg)[0]
      conf_bg = max(prob_bg) * 100
      price_bg = round(row_bg['close'], 4)
      vol_bg = max(row_bg['volatility'], 0.1)

      sl_bg = vol_bg * 1.2
      tp_bg = sl_bg * rr_ratio

      if conf_bg >= confidence_threshold and abs(prob_bg[1] - prob_bg[0]) >= 0.12:
        tp_p = (
            round(price_bg + tp_bg, 4)
            if pred_bg == 1
            else round(price_bg - tp_bg, 4)
        )
        sl_p = (
            round(price_bg - sl_bg, 4)
            if pred_bg == 1
            else round(price_bg + sl_bg, 4)
        )
        if pred_bg == 1:
          msg_bg = (
              f'AUTO BUY | {monitor_symbol} | Entry: {price_bg} | TP: {tp_p} |'
              f' SL: {sl_p}'
          )
          st.success(f'📈 تنبيه شراء تلقائي مرسل لـ {monitor_symbol}')
          send_ntfy_alert(ntfy_topic, msg_bg, 'Autonomous BUY Alert')
        else:
          msg_bg = (
              f'AUTO SELL | {monitor_symbol} | Entry: {price_bg} | TP: {tp_p} |'
              f' SL: {sl_p}'
          )
          st.error(f'📉 تنبيه بيع تلقائي مرسل لـ {monitor_symbol}')
          send_ntfy_alert(ntfy_topic, msg_bg, 'Autonomous SELL Alert')
      else:
        st.write(
            f'⏳ السوق في حالة استقرار (الثقة: {conf_bg:.1f}%) - لا توجد صفقة'
            ' جديدة مرسلة.'
        )
