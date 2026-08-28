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

# --- إعدادات الصفحة والواجهة المؤسسية ---
st.set_page_config(
    page_title="XAU/USD Deep Autonomous AI",
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
    .ai-level-container { background: linear-gradient(90deg, #1e3a8a 0%, #0f172a 100%); padding: 20px; border-radius: 15px; text-align: center; border: 1px solid #3b82f6; box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
    .ai-level-text { font-size: 2.5rem; font-weight: 900; color: #fbbf24; margin: 0; }
</style>
""",
    unsafe_allow_html=True,
)

DB_FILE = 'xau_deep_ai.db'
MODEL_FILE = 'xau_deep_mlp.pkl'
SCALER_FILE = 'xau_deep_scaler.pkl'


# --- قواعد البيانات وتخزين مستوى الخبرة ---
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

# --- القائمة الجانبية (الإعدادات) ---
st.sidebar.header('⚙️ إعدادات الذكاء الاصطناعي')
twelve_key = st.sidebar.text_input('مفتاح Twelve Data API (للتنفيذ اللحظي)', type='password', value=load_setting('twelve_key', ''))
save_setting('twelve_key', twelve_key)

ntfy_channel = st.sidebar.text_input('قناة Ntfy للتنبيهات الفورية', value=load_setting('ntfy', 'xau_deep_channel'))
save_setting('ntfy', ntfy_channel)

if not twelve_key:
  conn = sqlite3.connect(DB_FILE)
  c = conn.cursor()
  c.execute('DELETE FROM active_trade')
  conn.commit()
  conn.close()

st.sidebar.markdown('---')
st.sidebar.header('🎯 إدارة المخاطر')
atr_mult = st.sidebar.slider('معامل وقف الخسارة ATR', 1.0, 3.0, 1.5, 0.1)
risk_reward = st.sidebar.slider('نسبة العائد (R:R)', 1.5, 4.0, 2.0, 0.5)
min_conf = st.sidebar.slider('أدنى ثقة مطلوبة (%)', 60, 95, 75, 1)


def send_alert(msg, title='🧠 XAU/USD Deep AI Alert'):
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

# --- مصادر البيانات والتحليل المزدوج (فني + أساسي) ---
# 1. المصدر الخلفي (YFinance) للتدريب الماكرو (الدولار والذهب)
@st.cache_data(ttl=3600)
def fetch_background_training_data():
    try:
        # جلب بيانات الذهب
        gold = yf.download("GC=F", period="60d", interval="1h", progress=False)
        # جلب بيانات مؤشر الدولار (التحليل الأساسي)
        dxy = yf.download("DX-Y.NYB", period="60d", interval="1h", progress=False)
        
        if gold.empty or dxy.empty:
            return pd.DataFrame()

        df = pd.DataFrame()
        df['close'] = gold['Close'].squeeze()
        df['high'] = gold['High'].squeeze()
        df['low'] = gold['Low'].squeeze()
        
        # مؤشر الدولار كمقوم أساسي
        df['dxy_close'] = dxy['Close'].squeeze()
        
        # ملء الفراغات الناتجة عن اختلاف أوقات التداول
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

# 2. المصدر الحي (Twelve Data) لتنفيذ الصفقات
def fetch_live_execution_data_twelve(limit=100):
  if not twelve_key:
    return pd.DataFrame()
  try:
    url_xau = f'https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize={limit}&apikey={twelve_key}'
    url_dxy = f'https://api.twelvedata.com/time_series?symbol=DXY&interval=1h&outputsize={limit}&apikey={twelve_key}'
    
    res_xau = requests.get(url_xau, timeout=6).json()
    res_dxy = requests.get(url_dxy, timeout=6).json()
    
    if 'values' in res_xau and 'values' in res_dxy:
        df_xau = pd.DataFrame(res_xau['values'])[::-1].reset_index(drop=True)
        df_dxy = pd.DataFrame(res_dxy['values'])[::-1].reset_index(drop=True)
        
        df = pd.DataFrame()
        df['close'] = df_xau['close'].astype(float)
        df['high'] = df_xau['high'].astype(float)
        df['low'] = df_xau['low'].astype(float)
        df['dxy_close'] = df_dxy['close'].astype(float)
        return df
  except Exception:
    pass
  return pd.DataFrame()

def apply_deep_indicators(df):
  if df is None or df.empty or len(df) < 52:
    return pd.DataFrame()
  
  # التحليل الفني للذهب
  tr = pd.concat([df['high'] - df['low'], np.abs(df['high'] - df['close'].shift()), np.abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
  df['atr'] = tr.rolling(14).mean()
  df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
  df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
  df['rsi'] = 100 - (100 / (1 + (df['close'].diff().clip(lower=0).rolling(14).mean() / ((-df['close'].diff().clip(upper=0)).rolling(14).mean() + 1e-6))))
  
  # التحليل الأساسي/الماكرو (قوة وتأثير الدولار)
  # حساب ارتباط الذهب بالدولار وتغيراته (علاقة عكسية عادة)
  df['dxy_roc'] = df['dxy_close'].pct_change(periods=14) * 100 # زخم الدولار
  df['macro_divergence'] = df['close'].pct_change() - df['dxy_close'].pct_change()

  df.dropna(inplace=True)
  return df


# --- محرك الشبكة العصبية العميقة للتدريب والتطور ---
def train_deep_neural_network():
  # بناء أو تحميل العقل الذكي
  if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
  else:
    # شبكةلتصميم هذا النظام الذي يدمج التحليلين الفني والأساسي ويفصل بين حركة الذهب والدولار قبل دراسة تأثيرهما المشترك، يجب بناء معمارية برمجية تعتمد على خوادمك الخاصة (Backend) لتشغيل الشبكات العصبية في الخلفية بشكل مستمر، حيث أن دوري كنموذج ذكاء اصطناعي هو تزويدك بالهيكلة والأكواد اللازمة لتدريب هذه النماذج وتشغيلها، وليس استضافة العمليات الحية.

إليك الخطة المعمارية والبرمجية لتنفيذ هذا النظام:

**1. معمارية الشبكات العصبية (الخلفية والتدريب)**
لتحقيق تحليل منفصل ثم دمج التأثير على زوج (XAU/USD)، تحتاج إلى بناء 3 نماذج ذكاء اصطناعي تعمل بالتوازي باستخدام مكتبات مثل TensorFlow أو PyTorch:
*   **الشبكة الأولى (محلل الدولار):** تتدرب حصرياً على بيانات مؤشر الدولار (DXY) والأخبار الاقتصادية الأمريكية.
*   **الشبكة الثانية (محلل الذهب):** تتدرب حصرياً على عقود الذهب (XAU)، معدلات التضخم العالمية، والطلب على الملاذ الآمن.
*   **الشبكة الثالثة (النموذج المدمج - Meta Model):** تأخذ المخرجات (التوقعات ونسب الثقة) من الشبكتين الأولى والثانية كـ "مدخلات جديدة"، وتربطها بالبيانات التاريخية لزوج XAU/USD لتوقع الاتجاه النهائي.

**2. دمج التحليل الفني والأساسي**
تتطلب الشبكات العصبية تحويل جميع البيانات إلى أرقام لتتمكن من معالجتها:
*   **التحليل الفني:** يتم جلب أسعار الافتتاح، الإغلاق، الحجم (OHLCV) والمؤشرات (RSI, MACD) عبر مفتاح **Twelve Data API**.
*   **التحليل الأساسي:** يتم جلب الأخبار الاقتصادية عبر واجهات برمجية أخرى (مثل Finnhub أو Alpha Vantage). يتم تمرير هذه الأخبار عبر نموذج تحليل المشاعر (Sentiment Analysis) مثل `FinBERT` لتحويل الخبر إلى درجة رقمية (مثلاً: +1 إيجابي جداً للدولار، -1 سلبي)، وتُدمج هذه الدرجات مع البيانات الفنية.

**3. واجهة المستخدم المبسطة (Frontend)**
بناءً على طلبك، ستكون الواجهة الأمامية خالية من التعقيدات والمخططات البيانية (Charts)، وتحتوي فقط على:
*   **مؤشر مستوى الذكاء (AI Evolution Level):** خوارزمية بسيطة تحسب المستوى بناءً على الصفقات الناجحة. 
    *   المعادلة المقترحة: `المستوى = (إجمالي الصفقات الناجحة × معدل الربح) / 100`.
    *   كلما زادت الصفقات الناجحة، يرتفع شريط التقدم (Progress Bar) ويعرض المستوى الحالي (مثلاً: Level 42).
*   **سجل الإشعارات:** قائمة بالقرارات التي اتخذها النظام مسبقاً لمتابعة الأداء.

**4. منطق الإشعارات وتتبع الصفقات**
يتم برمجة الخلفية (Backend) لتقييم السوق كل فترة زمنية (مثلاً كل 15 دقيقة):
*   إذا أعطى "النموذج المدمج" نسبة ثقة تتجاوز 80% في اتجاه معين لزوج XAU/USD، يقوم النظام بتسجيل الصفقة افتراضياً وتتبعها عبر بيانات Twelve Data.
*   يتم إرسال الإشعار فوراً إلى الواجهة عبر (WebSockets) أو إلى تطبيقات خارجية (مثل Telegram) متضمناً: نوع الصفقة (بيع/شراء)، والسبب باختصار (مثلاً: ضعف الدولار الأساسي مع تشبع بيعي فني للذهب).
*   عند إغلاق الصفقة الافتراضية، يتحقق النظام من النتيجة. إذا كانت ناجحة، يتم تحديث "مستوى تطور الذكاء" في قاعدة البيانات وعكسه فوراً على الواجهة.

**5. الهيكل البرمجي الأساسي (بايثون - للخوادم)**
إليك المخطط البرمجي لآلية جمع البيانات وتغذية النماذج في الخلفية:

```python
import requests
from transformers import pipeline
# افتراض وجود نماذجك المدربة مسبقاً
# from my_models import GoldModel, USDModel, CombinedModel 

TWELVE_DATA_API_KEY = "YOUR_TWELVE_KEY"
sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")

def get_technical_data(symbol):
    url = f"[https://api.twelvedata.com/time_series?symbol=](https://api.twelvedata.com/time_series?symbol=){symbol}&interval=1h&apikey={TWELVE_DATA_API_KEY}"
    response = requests.get(url).json()
    return response['values']

def get_fundamental_score(news_headlines):
    # تحويل الأخبار إلى درجات رقمية
    scores = sentiment_analyzer(news_headlines)
    # تجميع الدرجات لإنتاج رقم يمثل الحالة الأساسية
    return aggregate_scores(scores)

def analyze_market():
    # 1. جلب البيانات
    usd_tech = get_technical_data("DXY")
    gold_tech = get_technical_data("XAU/USD")
    
    usd_fund = get_fundamental_score(get_usd_news())
    gold_fund = get_fundamental_score(get_gold_news())
    
    # 2. التحليل الفردي
    usd_prediction = USDModel.predict(usd_tech, usd_fund)
    gold_prediction = GoldModel.predict(gold_tech, gold_fund)
    
    # 3. التحليل المدمج
    final_trade_signal = CombinedModel.predict(usd_prediction, gold_prediction, gold_tech)
    
    # 4. إرسال الإشعار وتحديث الواجهة إذا كانت الثقة عالية
    if final_trade_signal.confidence > 0.80:
        send_notification(final_trade_signal)
        track_trade_for_ai_level(final_trade_signal)
