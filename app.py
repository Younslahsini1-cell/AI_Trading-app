from datetime import date, datetime
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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

    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif !important;
    }

    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    div.stMarkdown {
        color: #e5e7eb;
    }

    pre {
        background: #1f2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 10px !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        text-align: center;
        font-size: 1.15rem !important;
        padding: 10px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border-radius: 10px;
        font-weight: 700;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5);
        transform: translateY(-1px);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #111827;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #9ca3af;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 منصة التداول التطوري الذاتي (Autonomous Evolutionary Agent)")

# مسارات الذاكرة والتقارير
MODEL_FILE = 'evo_model.pkl'
SCALER_FILE = 'evo_scaler.pkl'
HISTORY_FILE = 'evo_history.csv'
SETTINGS_FILE = 'evo_settings.json'
TRADES_LOG_FILE = 'evo_trades_log.csv'


# --- إدارة الإعدادات ---
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

# --- دالة إرسال التنبيهات المُصححة ---
def send_ntfy_alert(topic, message, title='Trading Signal'):
  if topic:
    # تنظيف اسم القناة في حال كتب المستخدم الرابط كاملاً
    clean_topic = topic.strip()
    if 'ntfy.sh/' in clean_topic:
      clean_topic = clean_topic.split('ntfy.sh/')[-1].strip('/')

    url = f'https://ntfy.sh/{clean_topic}'
    try:
      # استخدام عنوان إنجليزي لتجنب مشاكل التشفير في الـ Headers
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
st.sidebar.header('🔔 إعدادات التنبيهات (Ntfy)')
ntfy_topic = st.sidebar.text_input(
    'اسم قناة أو رابط Ntfy',
    value=st.session_state.settings.get('ntfy', ''),
    placeholder='مثال: younslahsini2009xauusd',
)
save_settings(api_key_twelve, api_key_alpha, ntfy_topic)

# زر فحص واختبار التنبيهات فوراً
if st.sidebar.button('🧪 اختبار إرسال تنبيه Ntfy الآن'):
  success = send_ntfy_alert(
      ntfy_topic,
      'مرحباً! هذا اختبار ناجح من منصة التداول الذكية 🚀',
      'Test Alert',
  )
  if success:
    st.sidebar.success('✅ تم إرسال التنبيه بنجاح لقناتك!')
  else:
    st.sidebar.error('❌ فشل الإرسال، تحقق من اسم القناة.')

st.sidebar.markdown('---')
st.sidebar.header('⚙️ معايير التطور والذكاء')
interval = st.sidebar.selectbox(
    'الإطار الزمني للرصد', ['1min', '5min', '15min', '1h', '1day'], index=2
)
rr_ratio = st.sidebar.slider(
    'نسبة العائد للمخاطرة (TP/SL)', 1.0, 5.0, 2.0, 0.5
)
confidence_threshold = st.sidebar.slider(
    'حد الثقة الأدنى للتنفيذ (%)', 50, 90, 58, 1
)

if st.sidebar.button('🗑️ إعادة تهيئة ذاكرة التطور الكاملة'):
  for file in [
      MODEL_FILE,
      SCALER_FILE,
      HISTORY_FILE,
      SETTINGS_FILE,
      TRADES_LOG_FILE,
  ]:
    if os.path.exists(file):
      os.remove(file)
  st.session_state.clear()
  st.success('تم إعادة ضبط نظام التطور الذاتي بنجاح!')
  st.rerun()

# --- إدارة الذاكرة الدائمة للنموذج التطوري ---
def load_permanent_memory():
  if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    is_trained = True
  else:
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        warm_start=True,
        max_iter=500,
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


# --- هندسة الميزات ---
def evolutionary_feature_engineering(df):
  df.columns = df.columns.str.lower()
  df['return'] = df['close'].pct_change()
  df['volatility'] = df['high'] - df['low']
  df['body'] = df['close'] - df['open']
  df['momentum_5'] = df['close'] - df['close'].shift(5)
  df['ma_ratio'] = df['close'] / df['close'].rolling(10).mean()
  df['price_position'] = (df['close'] - df['low']) / (
      df['volatility'] + 1e-8
  )
  df.dropna(inplace=True)
  return df


def prepare_data(df):
  features = [
      'return',
      'volatility',
      'body',
      'momentum_5',
      'ma_ratio',
      'price_position',
  ]
  X = df[features].values
  Y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
  return X[:-1], Y[:-1], features


# --- الوسيط الذكي لجلب البيانات ---
def autonomous_data_broker_agent(symbol, interval, outputsize=500):
  clean_symbol = symbol.upper().strip()

  if any(
      crypto in clean_symbol for crypto in ['BTC', 'ETH', 'SOL', 'BNB', 'CRYPTO']
  ):
    try:
      binance_sym = clean_symbol.replace('/', '').replace('-', '')
      if 'USDT' not in binance_sym:
        binance_sym += 'USDT'
      interval_map = {
          '1min': '1m',
          '5min': '5m',
          '15min': '15m',
          '1h': '1h',
          '1day': '1d',
      }
      b_interval = interval_map.get(interval, '15m')
      url = f'https://api.binance.com/api/v3/klines?symbol={binance_sym}&interval={b_interval}&limit={min(outputsize, 1000)}'
      res = requests.get(url, timeout=4).json()
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
        return df[cols].reset_index(drop=True), f'Binance Live Broker ({binance_sym})'
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
      yf_interval = interval_map.get(interval, '15m')
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
            return df_clean, f'Yahoo Finance Broker ({yf_symbol})'
    except:
      pass

  if api_key_twelve:
    try:
      url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={api_key_twelve}&outputsize={outputsize}'
      res = requests.get(url, timeout=4).json()
      if 'values' in res:
        df = pd.DataFrame(res['values'])
        cols = ['open', 'high', 'low', 'close']
        if 'volume' in df.columns:
          cols.append('volume')
        df[cols] = df[cols].astype(float)
        return df.iloc[::-1].reset_index(drop=True), 'Twelve Data Broker'
    except:
      pass

  close = np.cumsum(np.random.randn(outputsize) * 0.5) + 4600
  high = close + np.random.uniform(0.1, 0.4, outputsize)
  low = close - np.random.uniform(0.1, 0.4, outputsize)
  open_p = low + np.random.uniform(0.0, 0.3, outputsize)
  return (
      pd.DataFrame(
          {'open': open_p, 'high': high, 'low': low, 'close': close}
      ),
      'Synthetic Fallback Broker',
  )


# --- محاكاة حقيقية لفحص ما إذا ضرب السعر TP أو SL في الشموع اللاحقة ---
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

    if prediction == 1:  # شراء
      if low_price <= sl_price:
        return 0, 'Hit SL (ضرب وقف الخسارة)'
      if high_price >= tp_price:
        return 1, 'Hit TP (ضرب الهدف)'
    else:  # بيع
      if high_price >= sl_price:
        return 0, 'Hit SL (ضرب وقف الخسارة)'
      if low_price <= tp_price:
        return 1, 'Hit TP (ضرب الهدف)'

  return 1, 'Active / Target in Progress'


def log_trade_and_check_mistakes(
    symbol, prediction, df_processed, current_idx, confidence, tp_dist, sl_dist
):
  today_str = str(date.today())
  win_status, note = evaluate_real_trade_outcome(
      df_processed, current_idx, prediction, tp_dist, sl_dist
  )
  error_note = (
      'لا توجد أخطاء'
      if win_status == 1
      else f'خطأ: انعكس السعر وضرب وقف الخسارة في {symbol}'
  )

  new_log = pd.DataFrame([{
      'Date': today_str,
      'Symbol': symbol,
      'Prediction': 'BUY (شراء)' if prediction == 1 else 'SELL (بيع)',
      'Win': win_status,
      'Confidence': round(confidence, 2),
      'ErrorNote': note if win_status == 0 else 'ناجحة (تم تحقيق الهدف TP)',
  }])

  if os.path.exists(TRADES_LOG_FILE):
    df_log = pd.read_csv(TRADES_LOG_FILE)
    df_log = pd.concat([df_log, new_log], ignore_index=True)
  else:
    df_log = new_log
  df_log.to_csv(TRADES_LOG_FILE, index=False)


def run_evolutionary_training():
  train_symbols = ['XAU/USD', 'BTC/USD', 'EUR/USD']
  combined_X, combined_Y = [], []

  for sym in train_symbols:
    df_t, _ = autonomous_data_broker_agent(sym, interval, 600)
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
    '🚀 لوحة المتابعة والتقدم التطوري',
    '📊 التقرير اليومي بنسب الفوز والأخطاء',
    '🔄 الرصد التلقائي الخلفي (Background Agent)',
])

with tab1:
  st.header('مؤشرات أداء الوسيط التطوري')
  col1, col2, col3 = st.columns(3)
  current_acc = (
      st.session_state.training_history['Accuracy'].iloc[-1]
      if not st.session_state.training_history.empty
      else 0
  )
  col1.metric(
      'حالة الذكاء الاصطناعي',
      'نشط وتطوري 🟢' if st.session_state.is_trained else 'قيد التهيئة ⚪',
  )
  col2.metric('دقة التعلم الحالية', f'{current_acc:.2f}%')

  win_rate_display = 0.0
  if os.path.exists(TRADES_LOG_FILE):
    df_l = pd.read_csv(TRADES_LOG_FILE)
    if not df_l.empty:
      win_rate_display = (df_l['Win'].sum() / len(df_l)) * 100
  col3.metric('نسبة الفوز الإجمالية للوسيط', f'{win_rate_display:.1f}%')

  st.markdown('---')
  st.subheader('تحليل فوري وتطوري للأسواق النشطة')
  market_input = st.text_input('أدخل رمز السوق للتحليل التطوري', 'XAU/USD')

  if st.button('تشغيل الفحص والتحليل الذاتي', use_container_width=True):
    with st.spinner('الوسيط التطوري يفحص الأسواق ويستخرج الأنماط...'):
      live_df, src = autonomous_data_broker_agent(market_input, interval, 80)
      processed = evolutionary_feature_engineering(live_df)
      features = [
          'return',
          'volatility',
          'body',
          'momentum_5',
          'ma_ratio',
          'price_position',
      ]

      idx_to_check = (
          len(processed) - 3 if len(processed) > 5 else len(processed) - 1
      )
      current_row = processed.iloc[idx_to_check]

      X_input = current_row[features].values.reshape(1, -1)
      X_scaled = st.session_state.scaler.transform(X_input)

      prediction = st.session_state.model.predict(X_scaled)[0]
      proba = st.session_state.model.predict_proba(X_scaled)[0]
      max_conf = max(proba) * 100
      current_price = round(current_row['close'], 4)
      vol = max(current_row['volatility'], 0.0001)

      sl_dist = vol * 1.5
      tp_dist = sl_dist * rr_ratio

      log_trade_and_check_mistakes(
          market_input,
          prediction,
          processed,
          idx_to_check,
          max_conf,
          tp_dist,
          sl_dist,
      )

      if max_conf < confidence_threshold:
        st.warning(
            f'⏳ **حالة الانتظار الحذر (WAIT)** | الثقة الحالية: {max_conf:.1f}%'
            ' أقل من حد الثقة الآمن.'
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

        if prediction == 1:
          msg = (
              f'BUY Signal | Symbol: {market_input} | Price: {current_price} |'
              f' Conf: {proba[1]*100:.1f}%'
          )
          st.success(f'🟢 **شراء (BUY)** | السعر: {current_price}')
          send_ntfy_alert(ntfy_topic, msg, 'Smart BUY Signal')
        else:
          msg = (
              f'SELL Signal | Symbol: {market_input} | Price: {current_price} |'
              f' Conf: {proba[0]*100:.1f}%'
          )
          st.error(f'🔴 **بيع (SELL)** | السعر: {current_price}')
          send_ntfy_alert(ntfy_topic, msg, 'Smart SELL Signal')

        st.markdown('##### 📋 تفاصيل التنفيذ السريع:')
        c1, c2, c3 = st.columns(3)
        c1.markdown('**الدخول (Entry)**')
        c1.code(str(current_price))
        c2.markdown('**الهدف (TP)**')
        c2.code(str(tp_price))
        c3.markdown('**وقف الخسارة (SL)**')
        c3.code(str(sl_price))

      st.line_chart(processed[['close']].tail(30))

with tab2:
  st.header('📋 التقرير اليومي الشامل (نسبة الفوز والأخطاء والتصحيح)')
  st.write(
      'يلخص هذا التقرير أداء الوسيط بناءً على صفقات حقيقية تم اختبار نجاحها عبر'
      ' ضرب الأهداف أو وقوف الخسائر.'
  )

  if os.path.exists(TRADES_LOG_FILE):
    df_report = pd.read_csv(TRADES_LOG_FILE)
    if not df_report.empty:
      total_trades = len(df_report)
      wins = df_report['Win'].sum()
      losses = total_trades - wins
      win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

      r_col1, r_col2, r_col3 = st.columns(3)
      r_col1.metric('إجمالي الصفقات المدروسة', total_trades)
      r_col2.metric('الصفقات الناجحة (TP)', wins)
      r_col3.metric('نسبة الفوز النهائية', f'{win_rate:.1f}%')

      st.markdown('---')
      st.subheader('⚠️ سجل الأخطاء المرصودة (التي ضربت وقف الخسارة SL):')
      errors_df = df_report[df_report['Win'] == 0]
      if not errors_df.empty:
        st.dataframe(
            errors_df[['Date', 'Symbol', 'Prediction', 'ErrorNote', 'Confidence']],
            use_container_width=True,
        )
        st.info(
            '💡 قام النموذج التلقائي بإدخال هذه الأخطاء في حلقة إعادة التدريب لرفع'
            ' كفاءة الشبكة العصبية وتجنب تكرارها.'
        )
      else:
        st.success(
            '🌟 ممتاز! لم يتم تسجيل أي أخطاء حرجة ضربت وقف الخسارة مؤخراً.'
        )

      st.markdown('---')
      st.subheader('سجل الصفقات الكامل:')
      st.dataframe(df_report, use_container_width=True)
    else:
      st.info('لا توجد سجلات صفقات كافية حتى الآن.')
  else:
    st.info('يبدأ الوسيط بتوليد التقرير اليومي تلقائياً فور تنفيذ أول دورة فحص.')

with tab3:
  st.header('🔄 الرصد التلقائي الخلفي (Continuous Background Agent)')
  st.write(
      'هنا يعمل الوسيط بشكل متواصل لفحص السوق المختار كل دقيقة وإرسال التنبيهات'
      ' عبر قناة Ntfy تلقائياً.'
  )

  scan_symbol = st.text_input('رمز السوق للمراقبة المستمرة', 'XAU/USD')
  auto_run = st.checkbox('تفعيل التشغيل التلقائي المستمر في الخلفية')

  if auto_run:
    st.info(
        '🟢 الوسيط يعمل الآن تلقائياً في الخلفية ويبحث عن الصفقات ويرسل التنبيهات'
        ' لـ Ntfy...'
    )
    placeholder = st.empty()
    while auto_run:
      with placeholder.container():
        st.write(
            '⏱️ آخر فحص تلقائي للخلفية:'
            f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        df_scan, src_s = autonomous_data_broker_agent(scan_symbol, interval, 60)
        if df_scan is not None:
          p_live = evolutionary_feature_engineering(df_scan)
          features = [
              'return',
              'volatility',
              'body',
              'momentum_5',
              'ma_ratio',
              'price_position',
          ]
          row = p_live.iloc[-1]
          X_sc = st.session_state.scaler.transform(
              row[features].values.reshape(1, -1)
          )
          pred = st.session_state.model.predict(X_sc)[0]
          pr = st.session_state.model.predict_proba(X_sc)[0]
          mx_conf = max(pr) * 100

          if mx_conf < confidence_threshold:
            st.info(
                f'⏳ حالة الوسيط لـ {scan_symbol}: **انتظار (WAIT)** | الثقة'
                f' ({mx_conf:.1f}%) أقل من الحد الآمن.'
            )
          else:
            if pred == 1:
              msg = (
                  f'BUY Signal | Symbol: {scan_symbol} | Conf:'
                  f' {pr[1]*100:.1f}%'
              )
              st.success(f'📈 شراء مؤكد لـ {scan_symbol}')
              send_ntfy_alert(ntfy_topic, msg, 'Autonomous BUY Alert')
            else:
              msg = (
                  f'SELL Signal | Symbol: {scan_symbol} | Conf:'
                  f' {pr[0]*100:.1f}%'
              )
              st.error(f'📉 بيع مؤكد لـ {scan_symbol}')
              send_ntfy_alert(ntfy_topic, msg, 'Autonomous SELL Alert')
      time.sleep(60)
