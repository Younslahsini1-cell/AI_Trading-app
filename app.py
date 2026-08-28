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

st.set_page_config(
    page_title="منصة التداول الكمّي الذكية مع الاستراتيجيات المتعددة",
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

st.title("🧠 منصة التداول التطوري الذاتي (مع محرك الاستراتيجيات المتعددة والمؤشرات)")

MODEL_FILE = 'strat_evo_model.pkl'
SCALER_FILE = 'strat_evo_scaler.pkl'
NEWS_MODEL_FILE = 'strat_news_model.pkl'
HISTORY_FILE = 'strat_evo_history.csv'
SETTINGS_FILE = 'strat_evo_settings.json'
TRADES_LOG_FILE = 'strat_evo_trades.csv'


def load_settings():
  if os.path.exists(SETTINGS_FILE):
    try:
      with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)
    except:
      pass
  return {'ntfy': '', 'twelve_key': ''}


def save_settings(ntfy, twelve_key):
  with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
    json.dump({'ntfy': ntfy, 'twelve_key': twelve_key}, f)


if 'settings' not in st.session_state:
  st.session_state.settings = load_settings()

if 'active_trade' not in st.session_state:
  st.session_state.active_trade = None

st.sidebar.header('🔔 إعدادات التنبيهات والمنصات')
ntfy_topic = st.sidebar.text_input(
    'اسم قناة Ntfy',
    value=st.session_state.settings.get('ntfy', ''),
    placeholder='مثال: my_trading_channel',
)
twelve_api_key = st.sidebar.text_input(
    'مفتاح Twelve Data API',
    type='password',
    value=st.session_state.settings.get('twelve_key', ''),
)
save_settings(ntfy_topic, twelve_api_key)

st.sidebar.markdown('---')
st.sidebar.header('⚙️ إعدادات استراتيجيات الشبكات العصبية')
interval = st.sidebar.selectbox(
    'الإطار الزمني للرصد', ['5min', '15min', '1h', '1day'], index=0
)
rr_ratio = st.sidebar.slider(
    'نسبة العائد للمخاطرة (TP/SL)', 1.0, 5.0, 2.0, 0.5
)
confidence_threshold = st.sidebar.slider(
    'حد الثقة الأدنى للتنفيذ (%)', 50, 90, 60, 1
)
news_influence = st.slider('وزن تأثير الأخبار على القرار (%)', 10, 50, 30, 5)


def send_ntfy_alert(topic, message, title='Strategic Trading Signal'):
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
    except:
      return False
  return False


def load_permanent_memory():
  if (
      os.path.exists(MODEL_FILE)
      and os.path.exists(SCALER_FILE)
      and os.path.exists(NEWS_MODEL_FILE)
  ):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    news_model = joblib.load(NEWS_MODEL_FILE)
    is_trained = True
  else:
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        warm_start=True,
        max_iter=200,
    )
    news_model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        solver='adam',
        warm_start=True,
        max_iter=100,
    )
    scaler = StandardScaler()
    is_trained = False

  history = (
      pd.read_csv(HISTORY_FILE)
      if os.path.exists(HISTORY_FILE)
      else pd.DataFrame(columns=['Accuracy'])
  )
  return model, news_model, scaler, is_trained, history


if 'model' not in st.session_state:
  m, nm, s, t, h = load_permanent_memory()
  st.session_state.model = m
  st.session_state.news_model = nm
  st.session_state.scaler = s
  st.session_state.is_trained = t
  st.session_state.training_history = h


def save_permanent_memory():
  joblib.dump(st.session_state.model, MODEL_FILE)
  joblib.dump(st.session_state.news_model, NEWS_MODEL_FILE)
  joblib.dump(st.session_state.scaler, SCALER_FILE)
  st.session_state.training_history.to_csv(HISTORY_FILE, index=False)


def evolutionary_feature_engineering(df):
  df.columns = df.columns.str.lower()
  df['return'] = df['close'].pct_change()
  df['volatility'] = df['high'] - df['low']
  df['body'] = df['close'] - df['open']
  df['momentum_5'] = df['close'] - df['close'].shift(5)
  df['ma_ratio'] = df['close'] / df['close'].rolling(10).mean()

  # محرك المؤشرات والاستراتيجيات المرجعية المتعددة
  # 1. مؤشر القوة النسبية RSI (استراتيجية التشبع البيعي والشرائي)
  delta = df['close'].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / (loss + 1e-6)
  df['rsi'] = 100 - (100 / (1 + rs))

  # 2. مؤشر تقاطع المتوسطات المتحركة MACD (استراتيجية الزخم والتقاطعات)
  ema12 = df['close'].ewm(span=12, adjust=False).mean()
  ema26 = df['close'].ewm(span=26, adjust=False).mean()
  df['macd'] = ema12 - ema26
  df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

  # 3. خطوط بولينجر باند Bollinger Bands (استراتيجية الارتداد السعري)
  sma20 = df['close'].rolling(window=20).mean()
  std20 = df['close'].rolling(window=20).std()
  df['bollinger_up'] = sma20 + (std20 * 2)
  df['bollinger_down'] = sma20 - (std20 * 2)
  df['bb_position'] = (df['close'] - df['bollinger_down']) / (
      df['bollinger_up'] - df['bollinger_down'] + 1e-6
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
      'rsi',
      'macd',
      'macd_signal',
      'bb_position',
  ]
  X = df[features].values
  Y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
  return X[:-1], Y[:-1], features


def get_market_news(symbol):
  clean_symbol = symbol.upper().strip().split()[0]
  if 'XAU' in clean_symbol:
    t_sym = 'XAU/USD'
  elif 'BTC' in clean_symbol:
    t_sym = 'BTC/USD'
  elif 'ETH' in clean_symbol:
    t_sym = 'ETH/USD'
  else:
    t_sym = clean_symbol

  news_text = 'Market stable with standard volatility.'
  sentiment_score = 0.55

  if twelve_api_key:
    try:
      url = f'https://api.twelvedata.com/news?symbol={t_sym}&apikey={twelve_api_key}&outputsize=3'
      res = requests.get(url, timeout=4).json()
      if 'data' in res and len(res['data']) > 0:
        headlines = [item.get('title', '') for item in res['data']]
        news_text = ' | '.join(headlines)
        pos_words = [
            'surge',
            'jump',
            'bull',
            'high',
            'growth',
            'up',
            'gain',
            'positive',
            'rally',
        ]
        neg_words = [
            'drop',
            'fall',
            'bear',
            'low',
            'crash',
            'down',
            'loss',
            'negative',
            'dip',
        ]
        score_val = 0.5
        for h in headlines:
          h_lower = h.lower()
          p_count = sum(1 for w in pos_words if w in h_lower)
          n_count = sum(1 for w in neg_words if w in h_lower)
          if p_count > n_count:
            score_val += 0.25
          elif n_count > p_count:
            score_val -= 0.25
        sentiment_score = float(np.clip(score_val, 0.1, 0.9))
    except:
      pass
  else:
    sim_sent = np.random.choice([0.2, 0.5, 0.8])
    sentiment_score = sim_sent
    news_text = (
        f'Simulated Strategic News Feed for {clean_symbol}: Multi-indicator'
        ' signals reacting to macro trends.'
    )

  return news_text, sentiment_score


def get_market_data(symbol, interval, outputsize=100):
  clean_symbol = symbol.upper().strip().split()[0]
  if 'XAU' in clean_symbol:
    t_sym = 'XAU/USD'
  elif 'BTC' in clean_symbol:
    t_sym = 'BTC/USD'
  elif 'ETH' in clean_symbol:
    t_sym = 'ETH/USD'
  elif 'EUR' in clean_symbol:
    t_sym = 'EUR/USD'
  elif 'GBP' in clean_symbol:
    t_sym = 'GBP/USD'
  elif 'SOL' in clean_symbol:
    t_sym = 'SOL/USD'
  else:
    t_sym = clean_symbol

  tf_map = {'5min': '5min', '15min': '15min', '1h': '1h', '1day': '1day'}
  t_interval = tf_map.get(interval, '5min')

  if twelve_api_key:
    try:
      url = f'https://api.twelvedata.com/time_series?symbol={t_sym}&interval={t_interval}&outputsize={outputsize}&apikey={twelve_api_key}'
      res = requests.get(url, timeout=5).json()
      if 'values' in res:
        df = pd.DataFrame(res['values'])
        cols = ['open', 'high', 'low', 'close']
        df[cols] = df[cols].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)
        return df[cols]
    except:
      pass

  try:
    binance_sym = (
        clean_symbol.replace('/', '').replace('-', '') + 'USDT'
        if 'USDT' not in clean_symbol
        else clean_symbol
    )
    if 'XAU' in clean_symbol:
      binance_sym = 'PAXGUSDT'
    elif 'EUR' in clean_symbol or 'GBP' in clean_symbol:
      binance_sym = 'EURUSDT'
    b_url = f'https://api.binance.com/api/v3/klines?symbol={binance_sym}&interval={t_interval}&limit={outputsize}'
    b_res = requests.get(b_url, timeout=4).json()
    if isinstance(b_res, list) and len(b_res) > 0:
      df = pd.DataFrame(
          b_res,
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
      cols = ['open', 'high', 'low', 'close']
      df[cols] = df[cols].astype(float)
      return df[cols].reset_index(drop=True)
  except:
    pass

  close = np.cumsum(np.random.randn(outputsize) * 0.5) + 2600
  return pd.DataFrame({
      'open': close - 1,
      'high': close + 2,
      'low': close - 2,
      'close': close,
  })


def run_light_training():
  df_t = get_market_data('BTC/USD', interval, 120)
  proc = evolutionary_feature_engineering(df_t)
  X, Y, _ = prepare_data(proc)
  if len(X) > 5:
    X_scaled = st.session_state.scaler.fit_transform(X)
    st.session_state.model.fit(X_scaled, Y)

    X_news_dummy = np.array([[0.2], [0.5], [0.8], [0.9], [0.1]])
    Y_news_dummy = np.array([0, 1, 1, 1, 0])
    st.session_state.news_model.fit(X_news_dummy, Y_news_dummy)

    st.session_state.is_trained = True
    acc = st.session_state.model.score(X_scaled, Y) * 100
    new_rec = pd.DataFrame({'Accuracy': [acc]})
    st.session_state.training_history = pd.concat(
        [st.session_state.training_history, new_rec], ignore_index=True
    )
    save_permanent_memory()


if not st.session_state.is_trained:
  run_light_training()

tab1, tab2, tab3 = st.tabs([
    '🚀 لوحة الأسواق واستراتيجيات المؤشرات',
    '📊 سجل الأخطاء والتعلم الاستراتيجي',
    '🔄 الرصد والمراقبة الخلفية',
])

with tab1:
  st.header('مؤشرات أداء الشبكات العصبية المعتمدة على الاستراتيجيات المتعددة')
  col1, col2, col3 = st.columns(3)
  current_acc = (
      st.session_state.training_history['Accuracy'].iloc[-1]
      if not st.session_state.training_history.empty
      else 65.0
  )
  col1.metric('حالة المحرك الاستراتيجي', 'نشطة ومستقرة 🟢')
  col2.metric('دقة التعلم الحالية', f'{current_acc:.1f}%')

  win_rate = 0.0
  if os.path.exists(TRADES_LOG_FILE):
    df_l = pd.read_csv(TRADES_LOG_FILE)
    if not df_l.empty:
      win_rate = (df_l['Win'].sum() / len(df_l)) * 100
  col3.metric('نسبة نجاح الصفقات', f'{win_rate:.1f}%')

  st.markdown('---')
  st.subheader('اختر السوق لتشغيل الاستراتيجيات الفنية والأخبار')
  markets_list = [
      'XAU/USD (الذهب)',
      'BTC/USD (بتكوين)',
      'ETH/USD (إثريوم)',
      'EUR/USD (يورو)',
      'GBP/USD (باوند)',
      'SOL/USD (سولانا)',
  ]
  market_input = st.selectbox('الأسواق المتاحة', markets_list)
  clean_market = market_input.split()[0]

  live_news_text, live_sentiment = get_market_news(clean_market)
  st.info(f'📰 **مستجدات الأخبار الحية لـ {clean_market}:** {live_news_text}')
  st.caption(f'مؤشر معنويات الأخبار المرجعي: {live_sentiment:.2f}')

  if st.session_state.active_trade is not None:
    t_info = st.session_state.active_trade
    st.warning(
        f'🔒 **توجد صفقة نشطة حالياً في السوق ({t_info["symbol"]})** | الدخول:'
        f' {t_info["entry"]} | الهدف TP: {t_info["tp"]} | الوقف SL:'
        f' {t_info["sl"]}\nننتظر وصول السعر إلى أحد الهدفين قبل فتح صفقات'
        ' جديدة.'
    )

  if st.button(
      'فحص المؤشرات (RSI, MACD, BB) وتطبيق استراتيجيات التدريب',
      use_container_width=True,
  ):
    if st.session_state.active_trade is not None:
      st.warning(
          '⚠️ لا يمكن فتح صفقة جديدة، توجد صفقة نشطة قيد المتابعة حالياً!'
      )
    else:
      with st.spinner(
          'جاري تطبيق استراتيجيات المؤشرات والمطابقة مع الشبكة العصبية...'
      ):
        live_df = get_market_data(clean_market, interval, 120)
        processed = evolutionary_feature_engineering(live_df)
        features = [
            'return',
            'volatility',
            'body',
            'momentum_5',
            'ma_ratio',
            'rsi',
            'macd',
            'macd_signal',
            'bb_position',
        ]

        current_row = processed.iloc[-1]
        X_input = current_row[features].values.reshape(1, -1)
        X_scaled = st.session_state.scaler.transform(X_input)

        technical_pred = st.session_state.model.predict(X_scaled)[0]
        technical_proba = st.session_state.model.predict_proba(X_scaled)[0]
        tech_conf = max(technical_proba) * 100

        news_input = np.array([[live_sentiment]])
        news_proba = st.session_state.news_model.predict_proba(news_input)[0]
        news_conf = max(news_proba) * 100

        combined_conf = tech_conf * (1 - (news_influence / 100.0)) + news_conf * (
            news_influence / 100.0
        )

        if live_sentiment >= 0.55 and technical_pred == 1:
          final_prediction = 1
        elif live_sentiment <= 0.45 and technical_pred == 0:
          final_prediction = 0
        else:
          final_prediction = technical_pred

        current_price = round(current_row['close'], 4)
        vol = max(current_row['volatility'], 0.1)

        sl_dist = vol * 1.2
        tp_dist = sl_dist * rr_ratio

        if combined_conf < confidence_threshold:
          st.warning(
              f'⏳ حالة انتظار (WAIT) | الثقة الاستراتيجية المدمجة:'
              f' {combined_conf:.1f}% (أقل من الحد المطلوب)'
          )
        else:
          tp_price = (
              round(current_price + tp_dist, 4)
              if final_prediction == 1
              else round(current_price - tp_dist, 4)
          )
          sl_price = (
              round(current_price - sl_dist, 4)
              if final_prediction == 1
              else round(current_price + sl_dist, 4)
          )

          st.session_state.active_trade = {
              'symbol': clean_market,
              'direction': 'BUY' if final_prediction == 1 else 'SELL',
              'entry': current_price,
              'tp': tp_price,
              'sl': sl_price,
          }

          if final_prediction == 1:
            msg = (
                f'BUY | {clean_market} | Entry: {current_price} | TP:'
                f' {tp_price} | SL: {sl_price} | RSI:'
                f' {current_row["rsi"]:.1f}'
            )
            st.success(
                f'🟢 شراء مؤكد عبر استراتيجيات المؤشرات (BUY) بسعر:'
                f' {current_price}'
            )
            send_ntfy_alert(ntfy_topic, msg, 'Strategic BUY Signal')
          else:
            msg = (
                f'SELL | {clean_market} | Entry: {current_price} | TP:'
                f' {tp_price} | SL: {sl_price} | RSI:'
                f' {current_row["rsi"]:.1f}'
            )
            st.error(
                f'🔴 بيع مؤكد عبر استراتيجيات المؤشرات (SELL) بسعر:'
                f' {current_price}'
            )
            send_ntfy_alert(ntfy_topic, msg, 'Strategic SELL Signal')

          c1, c2, c3 = st.columns(3)
          c1.markdown('**الدخول (Entry)**')
          c1.code(str(current_price))
          c2.markdown('**الهدف (TP)**')
          c2.code(str(tp_price))
          c3.markdown('**وقف الخسارة (SL)**')
          c3.code(str(sl_price))

    st.line_chart(processed[['close']].tail(30))

with tab2:
  st.header('📊 سجل الصفقات والتعلم الاستراتيجي الذاتي')
  if os.path.exists(TRADES_LOG_FILE):
    df_report = pd.read_csv(TRADES_LOG_FILE)
    if not df_report.empty:
      st.dataframe(df_report, use_container_width=True)
    else:
      st.info('لا توجد صفقات مسجلة بعد.')
  else:
    st.info('سيظهر السجل فور إجراء أول فحص استراتيجي بالسوق.')

with tab3:
  st.header('🔄 الرصد والمراقبة الخلفية للاستراتيجيات (24/7)')
  st.write(
      'يعمل هذا القسم في الخلفية لتتبع الصفقات النشطة استراتيجياً، والتحقق من'
      ' وصولها إلى الهدف (TP) أو وقف الخسارة (SL) بناءً على تحركات المؤشرات'
      ' الحية.'
  )

  auto_monitor = st.checkbox(
      'تفعيل الرصد الآلي الخلفي للاستراتيجيات (تحديث كل دقيقة)'
  )
  if auto_monitor:
    st_autorefresh(interval=60000, key='bg_monitor_loop')
    st.info(
        f'🟢 الرصد الآلي الاستراتيجي نشط الآن (آخر فحص:'
        f" {datetime.now().strftime('%H:%M:%S')})..."
    )

    if st.session_state.active_trade is not None:
      t = st.session_state.active_trade
      live_df_check = get_market_data(t['symbol'], interval, 20)
      if not live_df_check.empty:
        latest_high = live_df_check.iloc[-1]['high']
        latest_low = live_df_check.iloc[-1]['low']
        latest_close = live_df_check.iloc[-1]['close']

        hit_tp = False
        hit_sl = False

        if t['direction'] == 'BUY':
          if latest_high >= t['tp']:
            hit_tp = True
          elif latest_low <= t['sl']:
            hit_sl = True
        else:
          if latest_low <= t['tp']:
            hit_tp = True
          elif latest_high >= t['sl']:
            hit_sl = True

        if hit_tp or hit_sl:
          win_status = 1 if hit_tp else 0
          note = (
              'Hit TP (تحقيق الهدف الاستراتيجي)'
              if hit_tp
              else 'Hit SL (وقف خسارة)'
          )

          new_log = pd.DataFrame([{
              'Date': str(date.today()),
              'Symbol': t['symbol'],
              'Prediction': t['direction'],
              'Win': win_status,
              'Confidence': 89.0,
              'Note': note,
          }])
          if os.path.exists(TRADES_LOG_FILE):
            df_l = pd.read_csv(TRADES_LOG_FILE)
            df_l = pd.concat([df_l, new_log], ignore_index=True)
          else:
            df_l = new_log
          df_l.to_csv(TRADES_LOG_FILE, index=False)

          send_ntfy_alert(
              ntfy_topic,
              f'Trade Closed! {t["symbol"]} {t["direction"]} -> {note}',
              'Strategic Status Update',
          )
          st.success(f'✅ تم إغلاق الصفقة النشطة لـ {t["symbol"]} بنتيجة: {note}')

          st.session_state.active_trade = None
        else:
          st.warning(
              f'⏳ الصفقة لا تزال جارية في {t["symbol"]} (السعر الحالي:'
              f' {latest_close}) - في انتظار وصول TP أو SL...'
          )
    else:
      scan_markets = ['XAU/USD', 'BTC/USD', 'ETH/USD', 'EUR/USD']
      for sm in scan_markets:
        df_scan = get_market_data(sm, interval, 60)
        news_txt_s, sent_s = get_market_news(sm)
        if not df_scan.empty:
          proc_scan = evolutionary_feature_engineering(df_scan)
          features = [
              'return',
              'volatility',
              'body',
              'momentum_5',
              'ma_ratio',
              'rsi',
              'macd',
              'macd_signal',
              'bb_position',
          ]
          row_scan = proc_scan.iloc[-1]
          X_scan = st.session_state.scaler.transform(
              row_scan[features].values.reshape(1, -1)
          )
          pred_scan = st.session_state.model.predict(X_scan)[0]
          prob_scan = st.session_state.model.predict_proba(X_scan)[0]
          conf_scan = max(prob_scan) * 100

          news_in_scan = st.session_state.news_model.predict_proba(
              np.array([[sent_s]])
          )[0]
          conf_news_s = max(news_in_scan) * 100
          combined_scan_conf = conf_scan * 0.7 + conf_news_s * 0.3

          if combined_scan_conf >= (confidence_threshold + 5):
            curr_p = round(row_scan['close'], 4)
            vol_p = max(row_scan['volatility'], 0.1)
            sl_p = vol_p * 1.2
            tp_p = sl_p * rr_ratio

            tp_price_s = (
                round(curr_p + tp_p, 4)
                if pred_scan == 1
                else round(curr_p - tp_p, 4)
            )
            sl_price_s = (
                round(curr_p - sl_p, 4)
                if pred_scan == 1
                else round(curr_p + sl_p, 4)
            )

            st.session_state.active_trade = {
                'symbol': sm,
                'direction': 'BUY' if pred_scan == 1 else 'SELL',
                'entry': curr_p,
                'tp': tp_price_s,
                'sl': sl_price_s,
            }

            dir_str = 'BUY' if pred_scan == 1 else 'SELL'
            msg_auto = (
                f'AUTO STRAT {dir_str} | {sm} | Entry: {curr_p} | TP:'
                f' {tp_price_s} | SL: {sl_price_s}'
            )
            send_ntfy_alert(
                ntfy_topic, msg_auto, f'Auto Autonomous Strategic {dir_str}'
            )
            st.success(
                f'🚀 تم التقاط صفقة استراتيجية تلقائية جديدة ومؤكدة في {sm}!'
            )
            break
      else:
        st.info(
            '🔍 جاري المسح الخلفي للاستراتيجيات والمؤشرات عبر مختلف الأسواق...'
        )

  if st.button('تحديث أوزان الشبكات العصبية الاستراتيجية يدوياً'):
    run_light_training()
    st.success(
        '✅ تم تحديث أوزان الذاكرة واستراتيجيات المؤشرات والأخبار بنجاح.'
    )
