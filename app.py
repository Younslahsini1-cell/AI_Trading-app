"""
XAU/USD Deep AI Engine — نسخة مُصلَّحة + مطوَّرة (ICT / Smart Money Edition)
=====================================================
هذا الملف نسخة معدّلة من التطبيق الأصلي. كل تعديل جوهري مُعلَّم بتعليق
يبدأ بـ "# FIX:" حتى يسهل عليك مقارنته بالنسخة القديمة، وكل إضافة جديدة
من هذه الجولة مُعلَّمة بـ "# NEW:".

أهم التغييرات الوظيفية من الجولة الأولى (لا تزال موجودة كاملة):
 1) توحيد مصدر واختيار الفريم الزمني للتدريب والتنفيذ الحي (Twelve Data
    بالساعة لكليهما) بدل تدريب يومي (yfinance) وتنفيذ بالساعة.
 2) حذف ميزة dxy_roc المزيّفة (كانت=0 دائماً وقت التشغيل) بدل الإبقاء
    عليها ميزة وهمية.
 3) تعلّم حقيقي من نتائج الصفقات المغلقة عبر partial_fit بدل نموذج مجمّد.
 4) استدعاء واحد فقط لبيانات Twelve Data الحيّة لكل دورة تحديث بدل اثنين.
 5) إضافة طبقة "رأي ثانٍ" من Groq API (مجاني، اختياري) قبل فتح أي صفقة.
 6) اسم ملفات النموذج تغيّر (v2) عمداً حتى لا يتم تحميل نموذج قديم
    غير متوافق أبعاد الميزات معه.

أهم الإضافات الجديدة في هذه الجولة (بناءً على الفيديو المرفق):
 7) محرك تحليل كامل بطريقة ICT / Smart Money Concepts فوق نفس بيانات
    الشموع (قراءة فقط، لا يتحكم في قرار الشراء/البيع الآلي):
    - Market Structure: كسور الهيكل (BOS/CHoCH) واتجاه هيكلي حالي.
    - Order Blocks (آخر Order Block صاعد/هابط + قوته).
    - Fair Value Gaps (فجوات القيمة العادلة الصاعدة/الهابطة).
    - Liquidity & Manipulation: مستويات BSL/SSL واكتشاف اصطياد السيولة.
    - Session Analyzer: نطاق/قمة/قاع الجلسة الحالية + الجلسة الآسيوية.
    - Fibonacci Extensions + منطقة الدخول المثلى (OTE).
    - Score Breakdown (Structure/Liquidity/OrderBlock/FVG) + Confidence.
    - Recent Displacements (حركات اندفاعية قوية حديثة).
 8) تبويب واجهة جديد "🧭 لوحة ICT / Smart Money" بخمسة أقسام فرعية
    تعرض كل ما سبق، دون حذف أي شيء من التبويبين القديمين.
 9) إعدادات جانبية جديدة للتحكم في حساسية محرك ICT (Swing Lookback،
    معامل الاندفاع Displacement).
"""

from datetime import datetime, timezone
import json
import os
import sqlite3
import threading

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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
    .claude-note { background:#111827; border:1px solid #374151; border-radius:12px; padding:14px; margin-top:10px; }

    /* NEW: بطاقات لوحة ICT / Smart Money — بنفس روح بطاقات الفيديو المرفق */
    .ict-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 20px; border-radius: 16px;
                text-align: center; border: 1px solid #334155; margin-bottom: 14px; height: 100%; }
    .ict-title { font-size: 0.85rem; color: #93c5fd; font-weight: 700; letter-spacing: 1px; margin-bottom: 8px; text-transform: uppercase; }
    .ict-value { font-size: 1.8rem; font-weight: 900; color: #fbbf24; line-height: 1.1; }
    .ict-sub { font-size: 0.85rem; color: #64748b; margin-top: 6px; }
    .ict-bullish { color: #22c55e !important; }
    .ict-bearish { color: #ef4444 !important; }
    .ict-neutral { color: #94a3b8 !important; }
    .ict-row { background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:10px 14px; margin-bottom:8px; }
    .ict-badge-yes { background:#7f1d1d; color:#fecaca; padding:4px 12px; border-radius:999px; font-weight:700; }
    .ict-badge-no { background:#14532d; color:#bbf7d0; padding:4px 12px; border-radius:999px; font-weight:700; }
</style>
""",
    unsafe_allow_html=True,
)

DB_FILE = 'xau_deep_ai.db'
# FIX: أسماء ملفات جديدة (v2) عمداً — مجموعة الميزات تغيّرت (4 بدل 5،
# وحُذفت dxy_roc)، فلو أُبقي على الاسم القديم سيحاول الكود تحميل نموذج
# مدرَّب على شكل مدخلات مختلف وسيفشل أو يعطي نتائج خاطئة بصمت.
MODEL_FILE = 'xau_deep_mlp_v2.pkl'
SCALER_FILE = 'xau_deep_scaler_v2.pkl'
# FIX: قفل ملفي بسيط يمنع بدء أكثر من عملية تدريب خلفية واحدة في نفس
# الوقت (لو فتح أكثر من زائر/بينغ الصفحة أثناء التدريب الأول)
TRAINING_LOCK_FILE = 'training.lock'
FEATURES = ['atr', 'ema_50', 'ema_200', 'rsi']  # FIX: dxy_roc حُذفت نهائياً


# --- قواعد البيانات وتخزين الخبرة ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, symbol TEXT, direction TEXT,
                    entry REAL, sl REAL, tp REAL, win INTEGER, note TEXT,
                    claude_conf REAL, claude_note TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS active_trade (
                    id INTEGER PRIMARY KEY, symbol TEXT, direction TEXT, entry REAL, sl REAL, tp REAL,
                    time TEXT, features TEXT)""")
    conn.commit()

    # FIX: ترحيل آمن لقواعد بيانات قديمة كانت منشأة بالمخطط السابق
    # (يضيف الأعمدة الجديدة إن لم تكن موجودة، ويتجاهل الخطأ إن كانت موجودة)
    for stmt in [
        "ALTER TABLE trades ADD COLUMN claude_conf REAL",
        "ALTER TABLE trades ADD COLUMN claude_note TEXT",
        "ALTER TABLE active_trade ADD COLUMN features TEXT",
    ]:
        try:
            c.execute(stmt)
            conn.commit()
        except sqlite3.OperationalError:
            pass
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

st.sidebar.markdown('---')
st.sidebar.header('🧠 الرأي الثاني (Groq — مجاني)')
# FIX: تحويل طبقة "الرأي الثاني" من Anthropic API (مدفوع، لا يوجد له
# باقة مجانية) إلى Groq (مجاني بالكامل، بدون بطاقة ائتمان)
use_claude = st.sidebar.checkbox('تفعيل مراجعة Groq قبل فتح الصفقة', value=(load_setting('use_claude', '1') == '1'))
save_setting('use_claude', '1' if use_claude else '0')

groq_key = st.sidebar.text_input('مفتاح Groq API', type='password', value=load_setting('groq_key', ''))
save_setting('groq_key', groq_key)

groq_model = st.sidebar.text_input('اسم النموذج', value=load_setting('groq_model', 'llama-3.3-70b-versatile'))
save_setting('groq_model', groq_model)
st.sidebar.caption('راجع console.groq.com/docs/models للتأكد من اسم النموذج المتاح حالياً مجاناً.')

min_claude_conf = st.sidebar.slider('أدنى ثقة مطلوبة من Groq (%)', 40, 95, 60, 1)

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
min_conf = st.sidebar.slider('أدنى ثقة مطلوبة من الشبكة العصبية (%)', 60, 95, 75, 1)

# NEW: إعدادات جانبية مخصّصة لمحرك ICT / Smart Money — لا تؤثر إطلاقاً
# على منطق الشبكة العصبية أو فتح/إغلاق الصفقات الآلية أعلاه، فهي فقط
# تتحكم في حساسية اللوحة التحليلية الجديدة (تبويب ICT).
st.sidebar.markdown('---')
st.sidebar.header('🧭 إعدادات لوحة ICT / Smart Money')
show_ict_tab = st.sidebar.checkbox('إظهار تبويب ICT / Smart Money', value=True)
swing_lookback = st.sidebar.slider('حساسية القمم/القيعان (Swing Lookback)', 2, 8, 3, 1)
ob_displacement_mult = st.sidebar.slider('معامل قوة الاندفاع (Order Block)', 0.8, 2.5, 1.2, 0.1)
st.sidebar.caption('لوحة ICT تحليلية للقراءة فقط (Read-Only) ولا تفتح أو تغلق أي صفقة، وهي ليست نصيحة استثمارية.')

st.sidebar.markdown('---')
# FIX: تطوير جديد — إعادة تدريب يدوية. سابقاً كان الملف المخزَّن يُحمَّل
# للأبد بلا وسيلة لإجبار تدريب جديد على بيانات سوق أحدث سوى حذف الملفات
# يدوياً من السيرفر. الآن زر واحد يفعل ذلك ويعيد تشغيل الصفحة.
if st.sidebar.button('🔄 إعادة تدريب النموذج من الصفر'):
    for f in (MODEL_FILE, SCALER_FILE, TRAINING_LOCK_FILE):
        if os.path.exists(f):
            os.remove(f)
    st.cache_data.clear()
    st.rerun()

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

# --- مصدر بيانات واحد فقط: Twelve Data ---
# FIX: أُلغي استخدام yfinance بالكامل. التدريب والتنفيذ الحي يستخدمان الآن
# نفس المصدر ونفس الفريم الزمني (ساعة) بدل تدريب يومي وتنفيذ بالساعة،
# وهو السبب الأصلي وراء كون تنبؤات النموذج بلا معنى إحصائياً.
def fetch_twelve_series(api_key, symbol='XAU/USD', interval='1h', outputsize=150):
    if not api_key:
        return pd.DataFrame()
    try:
        url = (f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}'
               f'&outputsize={outputsize}&apikey={api_key}')
        res = requests.get(url, timeout=8).json()
        if 'values' in res:
            st.session_state['last_twelve_error'] = None
            df = pd.DataFrame(res['values'])[['open', 'high', 'low', 'close']].astype(float)
            return df.iloc[::-1].reset_index(drop=True)
        # FIX: تشخيص أوضح — Twelve Data يرجع {"status":"error","message":...}
        # عند مفتاح خاطئ أو تجاوز حد الطلبات؛ كانت هذه الرسالة تُبتلع بصمت
        # وتظهر للمستخدم كـ"بيانات غير كافية" بلا تفسير حقيقي.
        st.session_state['last_twelve_error'] = res.get('message', 'استجابة غير متوقعة من Twelve Data.')
    except Exception as e:
        st.session_state['last_twelve_error'] = f'تعذّر الاتصال بـ Twelve Data: {e}'
    return pd.DataFrame()

# بيانات التدريب: سحبة واحدة كبيرة (بالساعة) تُحدَّث مرة كل 24 ساعة فقط
@st.cache_data(ttl=86400)
def fetch_training_data_twelve(api_key):
    return fetch_twelve_series(api_key, symbol='XAU/USD', interval='1h', outputsize=5000)
    # ملاحظة: لو باقتك المجانية في Twelve Data لا تدعم outputsize=5000
    # أو تستهلك رصيد الطلبات بسرعة، قلّل الرقم (مثلاً 2000).

def apply_deep_indicators(df):
    # FIX: دالة واحدة موحّدة تُستخدم للتدريب وللتنفيذ الحي معاً (بدل نسختين
    # بمعامل is_training كانتا تنتجان ميزات بمقياسين مختلفين تماماً)
    if df is None or df.empty or len(df) < 210:
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

    df.dropna(inplace=True)
    return df

# --- محرك التعلم (Neural Net) ---
def _background_train_and_save(api_key):
    """
    FIX: تطوير جديد. سابقاً كان أول تحميل للصفحة (بلا نموذج محفوظ) يُجبر
    المستخدم/الزائر على انتظار جلب 5000 شمعة وتدريب MLPClassifier كاملاً
    قبل أن تُعرض الصفحة إطلاقاً. لو كانت خدمة خارجية (مثل UptimeRobot)
    تفتح الموقع كل 5 دقائق، فقد تنتهي مهلة الطلب (timeout) قبل اكتمال
    التدريب وتفشل الزيارة بالكامل. الآن يعمل التدريب في Thread منفصل في
    الخلفية، وتُعرض الصفحة فوراً بنموذج مؤقت ريثما يجهز النموذج الحقيقي.
    """
    try:
        df_train = fetch_training_data_twelve(api_key)
        df_train = apply_deep_indicators(df_train)
        if not df_train.empty and len(df_train) >= 100:
            X = df_train[FEATURES].values[:-1]
            y = np.where(df_train['close'].shift(-1) > df_train['close'], 1, 0)[:-1]
            new_scaler = StandardScaler()
            X_sc = new_scaler.fit_transform(X)
            new_model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
            new_model.fit(X_sc, y)
            joblib.dump(new_model, MODEL_FILE)
            joblib.dump(new_scaler, SCALER_FILE)
    except Exception:
        pass
    finally:
        if os.path.exists(TRAINING_LOCK_FILE):
            try:
                os.remove(TRAINING_LOCK_FILE)
            except Exception:
                pass

def train_deep_model(api_key):
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)

    if api_key and not os.path.exists(TRAINING_LOCK_FILE):
        open(TRAINING_LOCK_FILE, 'w').close()
        threading.Thread(target=_background_train_and_save, args=(api_key,), daemon=True).start()

    # نموذج مؤقت غير مدرَّب — execute_autonomous_scan يتعامل مع فشل
    # predict_proba عليه بهدوء (رسالة "قيد التهيئة") إلى أن يجهز النموذج
    # الحقيقي المحفوظ من الخيط الخلفي، فتُحمَّل تلقائياً بالزيارة التالية.
    return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42), StandardScaler()

model, scaler = train_deep_model(twelve_key)

# --- الرأي الثاني: Groq API (مجاني، متوافق مع تنسيق OpenAI) ---
def get_claude_review(direction, last_row, mlp_conf, api_key, model_name):
    """
    FIX: طبقة "الرأي الثاني" أصبحت تستدعي Groq بدل Anthropic (Groq مجاني
    بالكامل بدون بطاقة ائتمان). تُستدعى فقط عندما تجاوزت ثقة الشبكة
    العصبية الحد الأدنى. عند أي فشل (مفتاح ناقص، شبكة، تحليل JSON) تُرجع
    None ويستمر القرار بالاعتماد على الشبكة العصبية وحدها.
    """
    if not api_key:
        return None
    try:
        prompt = (
            "أنت محلل فني مساعد لصفقة محتملة على XAU/USD. "
            f"الاتجاه المقترح من نموذج آخر: {direction}. ثقة ذلك النموذج: {mlp_conf:.1f}%. "
            f"القيم الحالية: ATR={last_row['atr']:.2f}, EMA50={last_row['ema_50']:.2f}, "
            f"EMA200={last_row['ema_200']:.2f}, RSI={last_row['rsi']:.1f}, السعر={last_row['close']:.2f}. "
            "أجب فقط بصيغة JSON بدون أي نص إضافي وبدون Markdown، بهذا الشكل بالضبط: "
            '{"agree": true, "confidence": 0, "reason": "..."}'
        )
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json={
                "model": model_name,
                "max_completion_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            timeout=15,
        )
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        cleaned = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        return {
            "agree": bool(parsed.get("agree")),
            "confidence": float(parsed.get("confidence", 0)),
            "reason": str(parsed.get("reason", "")),
        }
    except Exception:
        return None

def execute_autonomous_scan(df_live_processed):
    # FIX: لم تعد هذه الدالة تجلب البيانات بنفسها — تستقبلها جاهزة من
    # الاستدعاء الوحيد في أسفل الملف، فلا يوجد استدعاء API مكرر لكل دورة.
    if not twelve_key:
        return 'النظام متوقف: يرجى إدخال مفتاح Twelve Data API.', None

    conn = sqlite3.connect(DB_FILE)
    df_act = pd.read_sql('SELECT * FROM active_trade WHERE id = 1', conn)
    conn.close()
    if not df_act.empty:
        return 'الذكاء الاصطناعي يراقب الصفقة النشطة حالياً.', None

    if df_live_processed.empty:
        return 'لا توجد صفقة: بيانات السوق غير كافية بعد.', None

    last = df_live_processed.iloc[-1]

    try:
        x_in = scaler.transform(last[FEATURES].values.reshape(1, -1))
        probs = model.predict_proba(x_in)[0]
        pred = np.argmax(probs)
        conf = probs[pred] * 100
    except Exception:
        return 'الشبكة العصبية قيد التهيئة (بيانات تدريب غير كافية بعد).', None

    curr = round(last['close'], 2)
    atr_v = last['atr']
    sl_d = round(atr_v * atr_mult, 2)
    tp_d = round(sl_d * risk_reward, 2)

    if conf < min_conf:
        return f'لا توجد صفقة: ثقة الشبكة العصبية الحالية ({conf:.1f}%) أقل من المطلوب ({min_conf}%).', None

    direction = 'BUY 🟢' if pred == 1 else 'SELL 🔴'

    claude_result = None
    if use_claude:
        claude_result = get_claude_review(direction, last, conf, groq_key, groq_model)
        if claude_result is not None:
            if not claude_result['agree'] or claude_result['confidence'] < min_claude_conf:
                note = (f"الشبكة العصبية اقترحت {direction} بثقة {conf:.1f}%، لكن مراجعة Groq لم توافق "
                        f"(ثقته: {claude_result['confidence']:.0f}%) — تم تجاهل الإشارة تحفظاً.")
                return note, claude_result
        # FIX: تمييز واضح بين "Groq رفض الإشارة" و"Groq لم يستجب أصلاً" —
        # سابقاً كانت الحالتان تُعاملان بنفس الطريقة (فتح الصفقة بصمت دون
        # أي إشارة لعدم توفر المراجعة)، مما يصعّب تشخيص مشاكل المفتاح.

    sl_p = round(curr - sl_d, 2) if pred == 1 else round(curr + sl_d, 2)
    tp_p = round(curr + tp_d, 2) if pred == 1 else round(curr - tp_d, 2)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('DELETE FROM active_trade')
    c.execute('''INSERT INTO active_trade (id, symbol, direction, entry, sl, tp, time, features)
                 VALUES (1, ?, ?, ?, ?, ?, ?, ?)''',
              ('XAU/USD', direction, curr, sl_p, tp_p,
               str(datetime.now(timezone.utc).strftime('%H:%M:%S')),
               json.dumps(last[FEATURES].to_dict())))
    conn.commit()
    conn.close()

    claude_line = ''
    if claude_result is not None:
        claude_line = f"\nمراجعة Groq: موافق ({claude_result['confidence']:.0f}%) — {claude_result['reason']}"
    elif use_claude:
        claude_line = "\nملاحظة: تعذّر الحصول على مراجعة Groq (مفتاح غير صالح أو تعذّر الاتصال) — تم فتح الصفقة بالاعتماد على الشبكة العصبية وحدها."
    send_alert(f'🧠 AI Trade Executed: {direction}\nEntry: ${curr}\nSL: ${sl_p}\nTP: ${tp_p}\nAI Confidence: {conf:.1f}%{claude_line}')
    return f'تم اتخاذ قرار ({direction}) بثقة {conf:.1f}% وتم إرسال التنبيه.', claude_result


# ============================================================
# NEW: محرك ICT / Smart Money Concepts
# ============================================================
# كل الدوال هنا "قراءة فقط" (read-only) على شموع df_live_processed —
# لا تكتب في قاعدة البيانات ولا تؤثر بأي شكل على قرار الشراء/البيع
# الآلي في execute_autonomous_scan أعلاه. الهدف منها عرض لوحة تحليلية
# شبيهة بما يظهر في تطبيقات الـICT الاحترافية (مثل الفيديو المرفق):
# Market Structure, Order Blocks, FVG, Liquidity, Sessions, Fibonacci,
# OTE, و Score Breakdown / Confidence.

def find_swing_points(df, lookback=3):
    """يحدد القمم/القيعان الهيكلية (Fractal Swings) — قمة إن كانت أعلى
    من كل الشموع المحيطة بها ضمن نافذة lookback على كل جانب، وبالمثل
    للقاع."""
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    swing_highs, swing_lows = [], []
    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback:i + lookback + 1]
        window_l = lows[i - lookback:i + lookback + 1]
        if highs[i] == window_h.max():
            swing_highs.append((i, float(highs[i])))
        if lows[i] == window_l.min():
            swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows


def analyze_market_structure(swing_highs, swing_lows):
    """يبني تتابع كسور الهيكل (BOS = استمرار الاتجاه، CHoCH = تغيّر
    الاتجاه) من تسلسل القمم/القيعان، ويحدد الاتجاه الهيكلي الحالي."""
    events = [(i, 'high', p) for i, p in swing_highs] + [(i, 'low', p) for i, p in swing_lows]
    events.sort(key=lambda x: x[0])

    structure_breaks = []
    trend = None
    last_high, last_low = None, None
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
    """يبحث عن آخر Order Block صاعد/هابط: آخر شمعة معاكسة قبل حركة
    اندفاعية (displacement) قوية بعكس اتجاهها الأصلي، وهو التعريف
    القياسي لـOrder Block في منهجية ICT."""
    if df.empty or 'atr' not in df.columns or len(df) < 5:
        return None, None
    recent = df.iloc[-lookback:].reset_index(drop=True)
    bullish_ob, bearish_ob = None, None
    for i in range(1, len(recent)):
        atr_v = recent['atr'].iloc[i]
        if pd.isna(atr_v) or atr_v <= 0:
            continue
        body = recent['close'].iloc[i] - recent['open'].iloc[i]
        is_disp = abs(body) > displacement_atr_mult * atr_v
        prev = recent.iloc[i - 1]
        if is_disp and body > 0 and prev['close'] < prev['open']:
            bullish_ob = {'top': round(prev['open'], 2), 'bottom': round(prev['low'], 2),
                          'bias': 'BULLISH', 'strength': round(min(abs(body) / atr_v * 100, 999), 1)}
        elif is_disp and body < 0 and prev['close'] > prev['open']:
            bearish_ob = {'top': round(prev['high'], 2), 'bottom': round(prev['open'], 2),
                          'bias': 'BEARISH', 'strength': round(min(abs(body) / atr_v * 100, 999), 1)}
    return bullish_ob, bearish_ob


def detect_fvg(df, lookback=60):
    """يبحث عن آخر Fair Value Gap صاعد/هابط بنمط الثلاث شمعات القياسي
    (فجوة بين قمة الشمعة الأولى وقاع الثالثة، أو العكس)."""
    if df.empty or len(df) < 3:
        return None, None
    recent = df.iloc[-lookback:].reset_index(drop=True)
    bullish_fvg, bearish_fvg = None, None
    for i in range(2, len(recent)):
        c1, c3 = recent.iloc[i - 2], recent.iloc[i]
        if c1['high'] < c3['low']:
            bullish_fvg = {'top': round(c3['low'], 2), 'bottom': round(c1['high'], 2), 'bias': 'BULLISH'}
        if c1['low'] > c3['high']:
            bearish_fvg = {'top': round(c1['low'], 2), 'bottom': round(c3['high'], 2), 'bias': 'BEARISH'}
    return bullish_fvg, bearish_fvg


def detect_liquidity_and_manipulation(df, swing_highs, swing_lows):
    """يحدد مستويات السيولة (BSL من تجمّع آخر القمم، SSL من تجمّع آخر
    القيعان) ويكتشف اصطياد سيولة (Manipulation/Liquidity Sweep): كسر
    مستوى ثم إغلاق عائد إلى الداخل — إشارة انعكاس محتمل شائعة في ICT."""
    if not swing_highs or not swing_lows or df.empty:
        return None, None, False, ''

    bsl_price = max(p for _, p in swing_highs[-5:])
    ssl_price = min(p for _, p in swing_lows[-5:])
    last = df.iloc[-1]

    manipulation, note = False, ''
    if last['high'] > bsl_price and last['close'] < bsl_price:
        manipulation, note = True, 'BSL swept, expecting reversal'
    elif last['low'] < ssl_price and last['close'] > ssl_price:
        manipulation, note = True, 'SSL swept, expecting reversal'

    return round(bsl_price, 2), round(ssl_price, 2), manipulation, note


def session_analyzer(df, session_len=24):
    """يحلل الجلسة الحالية تقريبياً (آخر session_len شمعة بالساعة): قمة
    وقاع ونطاق الجلسة، إضافة إلى "Main Score" يقيس مدى اتساق الحركة مع
    اتجاه معيّن (صاعد/هابط) داخل هذا النطاق."""
    if df.empty or len(df) < 5:
        return None
    window = df.iloc[-session_len:]
    session_high = window['high'].max()
    session_low = window['low'].min()
    session_range = session_high - session_low
    net_change = window['close'].iloc[-1] - window['open'].iloc[0]
    bias = 'BEARISH' if net_change < 0 else 'BULLISH'
    main_score = round(min(abs(net_change) / (session_range + 1e-6) * 100, 100), 1)
    return {'bias': bias, 'main_score': main_score, 'range': round(session_range, 2),
            'high': round(session_high, 2), 'low': round(session_low, 2)}


def compute_asian_session_levels(df):
    """تقريب لقمة/قاع آخر جلسة آسيوية مكتملة، بأخذ نافذة الـ24 ساعة
    الأخيرة من بيانات الشموع بالساعة."""
    if df.empty or len(df) < 24:
        return None
    window = df.iloc[-24:]
    return {'asian_high': round(window['high'].max(), 2), 'asian_low': round(window['low'].min(), 2)}


def compute_fibonacci_extension(swing_low, swing_high, direction):
    """مستويات امتداد فيبوناتشي القياسية في ICT فوق آخر ساق سعرية
    (Leg) محددة بين قاع وقمة هيكليين حديثين."""
    diff = swing_high - swing_low
    ratios = [(1.0, '100% EXTENSION'), (1.27, '127% EXTENSION'), (1.618, '161.8% EXTENSION'),
              (2.0, '200% EXTENSION'), (2.618, '261.8% EXTENSION')]
    levels = {}
    for r, label in ratios:
        if direction == 'BULLISH':
            levels[label] = round(swing_high + diff * (r - 1), 2)
        else:
            levels[label] = round(swing_low - diff * (r - 1), 2)
    levels['EQUILIBRIUM TARGET'] = round((swing_high + swing_low) / 2, 2)
    return levels


def compute_ote_zone(swing_low, swing_high, direction, current_price):
    """منطقة الدخول المثلى (Optimal Trade Entry) بين تصحيح 61.8% و79%
    لآخر ساق سعرية — أحد أهم مفاهيم ICT لتوقيت الدخول."""
    diff = swing_high - swing_low
    if direction == 'BULLISH':
        top = swing_high - diff * 0.618
        bottom = swing_high - diff * 0.79
    else:
        top = swing_low + diff * 0.79
        bottom = swing_low + diff * 0.618
    lo, hi = min(top, bottom), max(top, bottom)
    return {'top': round(hi, 2), 'bottom': round(lo, 2), 'inside': lo <= current_price <= hi, 'direction': direction}


def compute_volatility_risk(df):
    """يصنّف التذبذب الحالي (ATR) نسبة إلى تاريخه الحديث ضمن أربع فئات
    (LOW/MEDIUM/HIGH/CRISIS) مطابقة لتسميات لوحة الفيديو المرفق."""
    atr_series = df['atr'].dropna()
    if atr_series.empty:
        return 'N/A', 0, 0.0
    current_atr = atr_series.iloc[-1]
    percentile = (atr_series < current_atr).mean() * 100
    if percentile >= 90:
        label, score = 'CRISIS', 90
    elif percentile >= 65:
        label, score = 'HIGH', 70
    elif percentile >= 35:
        label, score = 'MEDIUM', 50
    else:
        label, score = 'LOW', 25
    return label, score, round(float(current_atr), 2)


def detect_recent_displacement(df, lookback=10, atr_mult=1.3):
    """يرصد آخر حركات اندفاعية قوية (Displacement) — شموع جسمها أكبر من
    معامل × ATR — وهي علامة "نية مؤسساتية" في تحليل ICT."""
    if df.empty:
        return []
    recent = df.iloc[-lookback:]
    results = []
    for _, row in recent.iterrows():
        atr_v = row.get('atr', np.nan)
        if pd.isna(atr_v) or atr_v <= 0:
            continue
        body = row['close'] - row['open']
        candle_range = row['high'] - row['low']
        body_pct = round(abs(body) / (candle_range + 1e-6) * 100, 1)
        if abs(body) > atr_mult * atr_v:
            results.append({'bias': 'BULLISH' if body > 0 else 'BEARISH', 'body_pct': body_pct})
    return results[-3:]


def run_ict_engine(df_processed, swing_lookback=3, ob_mult=1.2):
    """المنسّق الرئيسي: يستدعي كل دوال التحليل أعلاه مرة واحدة لكل دورة
    تحديث ويعيد قاموساً واحداً تستهلكه واجهة تبويب ICT / Smart Money
    بالكامل. لا يفتح ولا يغلق أي صفقة — تحليلي فقط."""
    if df_processed is None or df_processed.empty or len(df_processed) < max(30, swing_lookback * 6):
        return None

    swing_highs, swing_lows = find_swing_points(df_processed, lookback=swing_lookback)
    bias, structure_breaks = analyze_market_structure(swing_highs, swing_lows)
    bull_ob, bear_ob = detect_order_blocks(df_processed, displacement_atr_mult=ob_mult)
    bull_fvg, bear_fvg = detect_fvg(df_processed)
    bsl, ssl, manipulation, manip_note = detect_liquidity_and_manipulation(df_processed, swing_highs, swing_lows)
    session_info = session_analyzer(df_processed)
    asian_levels = compute_asian_session_levels(df_processed)
    vol_label, vol_score, atr_v = compute_volatility_risk(df_processed)
    displacements = detect_recent_displacement(df_processed)
    current_price = round(float(df_processed['close'].iloc[-1]), 2)

    fib_levels, ote = {}, None
    if swing_highs and swing_lows:
        last_low_idx, last_low_price = swing_lows[-1]
        last_high_idx, last_high_price = swing_highs[-1]
        leg_direction = 'BULLISH' if last_low_idx > last_high_idx else 'BEARISH'
        lo_p, hi_p = min(last_low_price, last_high_price), max(last_low_price, last_high_price)
        fib_levels = compute_fibonacci_extension(lo_p, hi_p, leg_direction)
        ote = compute_ote_zone(lo_p, hi_p, leg_direction, current_price)

    # NEW: Score Breakdown — تفصيل الثقة إلى أربعة مكوّنات مطابقة لتسميات
    # الفيديو (Structure / Liquidity / Order Block / FVG Score)
    matching_breaks = [b for b in structure_breaks if b['direction'] == bias]
    structure_score = min(100, 25 + len(matching_breaks) * 25) if bias != 'NEUTRAL' else 50
    liquidity_score = 80 if manipulation else 40
    if bias == 'BULLISH':
        order_block_score = bull_ob['strength'] if bull_ob else 30
        fvg_score = 70 if bull_fvg else 35
    elif bias == 'BEARISH':
        order_block_score = bear_ob['strength'] if bear_ob else 30
        fvg_score = 70 if bear_fvg else 35
    else:
        order_block_score, fvg_score = 40, 40
    order_block_score = min(100, order_block_score)
    confidence = round(np.mean([structure_score, liquidity_score, order_block_score, fvg_score]), 1)

    return {
        'bias': bias,
        'structure_breaks': structure_breaks,
        'bull_ob': bull_ob, 'bear_ob': bear_ob,
        'bull_fvg': bull_fvg, 'bear_fvg': bear_fvg,
        'bsl': bsl, 'ssl': ssl,
        'manipulation': manipulation, 'manip_note': manip_note,
        'session': session_info,
        'asian_levels': asian_levels,
        'vol_label': vol_label, 'vol_score': vol_score, 'atr': atr_v,
        'displacements': displacements,
        'fib_levels': fib_levels,
        'ote': ote,
        'current_price': current_price,
        'scores': {
            'structure': round(structure_score, 1),
            'liquidity': round(liquidity_score, 1),
            'order_block': round(order_block_score, 1),
            'fvg': round(fvg_score, 1),
        },
        'confidence': confidence,
    }


# --- واجهة المستخدم ---
st.title("🧠 نظام التداول العميق — XAU/USD")

success_count = get_successful_trades_count()
ai_level = max(1, int(success_count * 1.5))

st.markdown(f"""
<div class="ai-level-card">
    <div class="ai-level-title">AI EVOLUTION LEVEL</div>
    <div class="ai-level-value">Lvl. {ai_level}</div>
    <div class="ai-level-sub">Successful Trades: {success_count} | Deep Neural Network Active</div>
</div>
""", unsafe_allow_html=True)

# FIX: جلب واحد فقط لبيانات السوق الحيّة لكل دورة تحديث، يُعاد استخدامه
# لكل من فتح الصفقات ومراقبة الصفقة النشطة (بدل استدعاءين منفصلين)
df_live_raw = fetch_twelve_series(twelve_key, symbol='XAU/USD', interval='1h', outputsize=220) if twelve_key else pd.DataFrame()
df_live_processed = apply_deep_indicators(df_live_raw)

# NEW: تشغيل محرك ICT مرة واحدة لكل دورة تحديث (نفس الشموع المستخدمة في
# الشبكة العصبية، بدون أي استدعاء API إضافي)
ict_data = run_ict_engine(df_live_processed, swing_lookback=swing_lookback, ob_mult=ob_displacement_mult) if not df_live_processed.empty else None

if show_ict_tab:
    tab1, tab2, tab3 = st.tabs(['⚡ حالة الذكاء الاصطناعي', '📊 سجل الخبرات المكتسبة (الصفقات)', '🧭 لوحة ICT / Smart Money'])
else:
    tab1, tab2 = st.tabs(['⚡ حالة الذكاء الاصطناعي', '📊 سجل الخبرات المكتسبة (الصفقات)'])
    tab3 = None

with tab1:
    if not twelve_key:
        st.warning('⚠️ النظام نائم: أدخل مفتاح Twelve Data API لإيقاظ الشبكة العصبية وربطها بالسوق.')

    # FIX: تطوير جديد — إشعار واضح أن التدريب الأول يجري في الخلفية،
    # بدل شاشة صامتة لا تفسّر لماذا لا توجد صفقات بعد.
    if os.path.exists(TRAINING_LOCK_FILE):
        st.info('🧠 النموذج يتدرب حالياً في الخلفية على بيانات تاريخية (أول مرة فقط) — قد يستغرق ذلك دقيقة أو دقيقتين، وستُعرض القرارات تلقائياً بمجرد الجاهزية.')

    with st.spinner('الشبكة العصبية تحلل البيانات...'):
        scan_msg, claude_info = execute_autonomous_scan(df_live_processed)

    conn = sqlite3.connect(DB_FILE)
    df_act = pd.read_sql('SELECT * FROM active_trade WHERE id = 1', conn)
    conn.close()

    if not df_act.empty and twelve_key:
        t = df_act.iloc[0]
        st.warning(f"🔒 **الشبكة العصبية تدير صفقة حالياً:** {t['direction']} | الدخول: ${t['entry']} | SL: ${t['sl']} | TP: ${t['tp']}")
    else:
        st.info(f'🔍 {scan_msg}')

    # FIX: تطوير جديد — إظهار رسالة الخطأ الفعلية من Twelve Data (مفتاح
    # خاطئ، تجاوز حد الطلبات...) بدل رسالة عامة غامضة فقط.
    twelve_err = st.session_state.get('last_twelve_error')
    if twelve_err and twelve_key:
        st.error(f'⚠️ Twelve Data: {twelve_err}')

    if claude_info is not None:
        agree_txt = '✅ موافق' if claude_info['agree'] else '❌ غير موافق'
        st.markdown(f"""<div class="claude-note">
            <b>🧠 رأي Groq:</b> {agree_txt} — ثقة {claude_info['confidence']:.0f}%<br>{claude_info['reason']}
        </div>""", unsafe_allow_html=True)

    # FIX: تطوير جديد — لوحة شفافية تُظهر قيم المؤشرات اللحظية بدل صندوق
    # أسود بلا تفاصيل، تساعد على فهم/تشخيص قرارات النموذج.
    if not df_live_processed.empty:
        last_snapshot = df_live_processed.iloc[-1]
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric('السعر', f"{last_snapshot['close']:.2f}")
        s2.metric('RSI', f"{last_snapshot['rsi']:.1f}")
        s3.metric('EMA50', f"{last_snapshot['ema_50']:.2f}")
        s4.metric('EMA200', f"{last_snapshot['ema_200']:.2f}")
        s5.metric('ATR', f"{last_snapshot['atr']:.2f}")

with tab2:
    conn = sqlite3.connect(DB_FILE)
    df_log = pd.read_sql('SELECT * FROM trades ORDER BY id DESC', conn)
    conn.close()
    if not df_log.empty:
        # FIX: تطوير جديد — نسبة الربح % إلى جانب العدد الخام، مقياس أوضح
        # لأداء النموذج الفعلي من مجرد "عدد الصفقات الناجحة".
        win_rate = (df_log['win'].sum() / len(df_log)) * 100
        m1, m2 = st.columns(2)
        m1.metric('إجمالي الصفقات', len(df_log))
        m2.metric('نسبة الربح', f"{win_rate:.1f}%")
        st.dataframe(df_log, use_container_width=True)
    else:
        st.info('لا توجد صفقات مغلقة مسجلة حتى الآن. الشبكة العصبية بانتظار أول نجاح.')

# NEW: تبويب لوحة ICT / Smart Money Concepts — يعرض بالكامل التحليل الذي
# يحسبه run_ict_engine أعلاه، بنفس روح البطاقات والأقسام الظاهرة في
# الفيديو المرفق (Volatility & Risk، Market Structure، Smart Money
# Concepts، Fibonacci & OTE، Trade Setup / Signals).
if tab3 is not None:
    with tab3:
        if not twelve_key:
            st.warning('⚠️ أدخل مفتاح Twelve Data API لتفعيل لوحة ICT / Smart Money.')
        elif ict_data is None:
            st.info('بيانات غير كافية بعد لعرض تحليل ICT / Smart Money (يحتاج على الأقل 30 شمعة مكتملة).')
        else:
            st.caption('🧭 لوحة تحليلية للقراءة فقط (Read-Only) — لا تتحكم في فتح/إغلاق الصفقات الآلية، وليست نصيحة استثمارية.')

            sub1, sub2, sub3, sub4, sub5 = st.tabs([
                '📉 Volatility & Risk', '🏗️ Market Structure', '🏦 Smart Money Concepts',
                '📐 Fibonacci & OTE', '🎯 Trade Setup / Signals',
            ])

            # --- 1) Volatility & Risk ---
            with sub1:
                risk_label = 'HIGH' if ict_data['vol_score'] >= 70 else 'MEDIUM' if ict_data['vol_score'] >= 50 else 'LOW'
                v1, v2 = st.columns(2)
                v1.markdown(f"""<div class="ict-card">
                    <div class="ict-title">Volatility</div>
                    <div class="ict-value">{ict_data['vol_label']}</div>
                    <div class="ict-sub">ATR: {ict_data['atr']}</div></div>""", unsafe_allow_html=True)
                v2.markdown(f"""<div class="ict-card">
                    <div class="ict-title">Risk Level</div>
                    <div class="ict-value">{risk_label}</div>
                    <div class="ict-sub">Score: {ict_data['vol_score']}</div></div>""", unsafe_allow_html=True)

                if ict_data['session']:
                    st.markdown('#### 🕒 Session Analyzer')
                    s = ict_data['session']
                    bias_class = 'ict-bullish' if s['bias'] == 'BULLISH' else 'ict-bearish'
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.markdown(f"<div class='ict-row'><b>BIAS</b><br><span class='{bias_class}'>{s['bias']}</span></div>", unsafe_allow_html=True)
                    sc2.markdown(f"<div class='ict-row'><b>MAIN SCORE</b><br>{s['main_score']}</div>", unsafe_allow_html=True)
                    sc3.markdown(f"<div class='ict-row'><b>SESSION RANGE</b><br>{s['range']}</div>", unsafe_allow_html=True)
                    sc4.markdown(f"<div class='ict-row'><b>SESSION HIGH/LOW</b><br>{s['high']} / {s['low']}</div>", unsafe_allow_html=True)

                if ict_data['asian_levels']:
                    st.markdown('#### 🌏 Asian Session (آخر ~24 ساعة)')
                    ah, al = st.columns(2)
                    ah.metric('Asian High', ict_data['asian_levels']['asian_high'])
                    al.metric('Asian Low', ict_data['asian_levels']['asian_low'])

            # --- 2) Market Structure ---
            with sub2:
                bias_class = 'ict-bullish' if ict_data['bias'] == 'BULLISH' else ('ict-bearish' if ict_data['bias'] == 'BEARISH' else 'ict-neutral')
                st.markdown(f"### الاتجاه الهيكلي الحالي: <span class='{bias_class}'>{ict_data['bias']}</span>", unsafe_allow_html=True)

                st.markdown('#### 📊 Score Breakdown')
                b1, b2, b3, b4 = st.columns(4)
                b1.metric('Structure Score', ict_data['scores']['structure'])
                b2.metric('Liquidity Score', ict_data['scores']['liquidity'])
                b3.metric('Order Block Score', ict_data['scores']['order_block'])
                b4.metric('FVG Score', ict_data['scores']['fvg'])

                st.markdown('#### 🧱 Structure Breaks (BOS / CHoCH)')
                if ict_data['structure_breaks']:
                    for brk in reversed(ict_data['structure_breaks']):
                        css = 'ict-bullish' if brk['direction'] == 'BULLISH' else 'ict-bearish'
                        st.markdown(f"<div class='ict-row'><b>{brk['type']}</b> — <span class='{css}'>{brk['direction']}</span> @ {brk['price']}</div>", unsafe_allow_html=True)
                else:
                    st.caption('لا توجد كسور هيكل واضحة ضمن النافذة الحالية.')

                if ict_data['displacements']:
                    st.markdown('#### ⚡ Recent Displacements')
                    for d in ict_data['displacements']:
                        css = 'ict-bullish' if d['bias'] == 'BULLISH' else 'ict-bearish'
                        st.markdown(f"<div class='ict-row'><span class='{css}'>{d['bias']}</span> — Body: {d['body_pct']}%</div>", unsafe_allow_html=True)

            # --- 3) Smart Money Concepts ---
            with sub3:
                st.markdown('#### 📦 Order Blocks')
                oc1, oc2 = st.columns(2)
                if ict_data['bull_ob']:
                    oc1.success(f"Bullish OB\nTop: {ict_data['bull_ob']['top']} | Bottom: {ict_data['bull_ob']['bottom']}\nStrength: {ict_data['bull_ob']['strength']}%")
                else:
                    oc1.caption('لا يوجد Bullish Order Block حديث.')
                if ict_data['bear_ob']:
                    oc2.error(f"Bearish OB\nTop: {ict_data['bear_ob']['top']} | Bottom: {ict_data['bear_ob']['bottom']}\nStrength: {ict_data['bear_ob']['strength']}%")
                else:
                    oc2.caption('لا يوجد Bearish Order Block حديث.')

                st.markdown('#### 🌀 Fair Value Gaps (FVG)')
                fc1, fc2 = st.columns(2)
                if ict_data['bull_fvg']:
                    fc1.success(f"Bullish FVG: {ict_data['bull_fvg']['bottom']} → {ict_data['bull_fvg']['top']}")
                else:
                    fc1.caption('لا توجد فجوة سعرية صاعدة حديثة.')
                if ict_data['bear_fvg']:
                    fc2.error(f"Bearish FVG: {ict_data['bear_fvg']['bottom']} → {ict_data['bear_fvg']['top']}")
                else:
                    fc2.caption('لا توجد فجوة سعرية هابطة حديثة.')

                st.markdown('#### 💧 Liquidity & Manipulation')
                lc1, lc2 = st.columns(2)
                lc1.metric('BSL (Buy-side Liquidity)', ict_data['bsl'])
                lc2.metric('SSL (Sell-side Liquidity)', ict_data['ssl'])
                if ict_data['manipulation']:
                    st.markdown(f"<span class='ict-badge-yes'>MANIPULATION: YES</span> &nbsp; {ict_data['manip_note']}", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='ict-badge-no'>MANIPULATION: NO</span> &nbsp; لا يوجد اصطياد سيولة واضح حالياً.", unsafe_allow_html=True)

            # --- 4) Fibonacci & OTE ---
            with sub4:
                if ict_data['ote']:
                    ote = ict_data['ote']
                    st.markdown(f"#### 🎯 Optimal Trade Entry (OTE) — {ote['direction']}")
                    st.write(f"المنطقة: **{ote['bottom']} → {ote['top']}**")
                    if ote['inside']:
                        st.success('✅ السعر الحالي داخل منطقة OTE')
                    else:
                        st.info('⚪ Outside OTE — السعر الحالي خارج منطقة الدخول المثلى')
                if ict_data['fib_levels']:
                    st.markdown('#### 📐 Fibonacci Extensions')
                    for label, val in ict_data['fib_levels'].items():
                        st.markdown(f"<div class='ict-row'><b>{label}</b>: {val}</div>", unsafe_allow_html=True)

            # --- 5) Trade Setup / Signals ---
            with sub5:
                bias_txt = ict_data['bias']
                color_class = 'ict-bullish' if bias_txt == 'BULLISH' else ('ict-bearish' if bias_txt == 'BEARISH' else 'ict-neutral')
                st.markdown(f"""<div class="ict-card">
                    <div class="ict-title">Bias</div>
                    <div class="ict-value {color_class}">{bias_txt}</div>
                    <div class="ict-sub">Confidence: {ict_data['confidence']}%</div>
                </div>""", unsafe_allow_html=True)

                st.markdown('#### 📡 Signals')
                if ict_data['manipulation']:
                    st.write(f"- {ict_data['manip_note']}")
                if ict_data['ote'] and ict_data['ote']['inside']:
                    st.write(f"- السعر داخل منطقة OTE باتجاه {ict_data['ote']['direction']}")
                if bias_txt == 'BULLISH' and ict_data['bull_ob']:
                    st.write("- Order Block صاعد نشط يدعم الاتجاه الهيكلي الحالي.")
                if bias_txt == 'BEARISH' and ict_data['bear_ob']:
                    st.write("- Order Block هابط نشط يدعم الاتجاه الهيكلي الحالي.")
                if not ict_data['manipulation'] and not (ict_data['ote'] and ict_data['ote']['inside']):
                    st.caption('لا توجد إشارات إضافية بارزة في هذه اللحظة.')

                st.caption('هذا القسم تحليلي بالكامل ولا يفتح أي صفقة تلقائياً — القرار الآلي الوحيد يبقى في تبويب "⚡ حالة الذكاء الاصطناعي" (الشبكة العصبية + مراجعة Groq الاختيارية).')

# --- المراقبة التلقائية (كل 60 ثانية) ---
st_autorefresh(interval=60000, key='deep_ai_loop')

if twelve_key:
    conn = sqlite3.connect(DB_FILE)
    c_active = pd.read_sql('SELECT * FROM active_trade WHERE id = 1', conn)
    conn.close()

    if not c_active.empty and not df_live_processed.empty:
        t_row = c_active.iloc[0]
        last_row = df_live_processed.iloc[-1]
        # FIX (خطأ حقيقي): is_buy_trade كانت تُحسب داخل try الخاصة بفحص
        # الانعكاس فقط، بينما تُستخدم لاحقاً خارجها لفحص SL/TP. لو فشل
        # predict_proba لأي سبب، كان التطبيق يتوقف بخطأ NameError بدل أن
        # يتجاهل الخطأ بهدوء. حُسبت الآن قبل أي try، فهي متاحة دائماً.
        is_buy_trade = 'BUY' in t_row['direction']

        try:
            x_current = scaler.transform(last_row[FEATURES].values.reshape(1, -1))
            current_probs = model.predict_proba(x_current)[0]
            curr_pred = np.argmax(current_probs)
            curr_conf = current_probs[curr_pred] * 100

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

            # FIX: تعلّم فعلي — إعادة تغذية النموذج بنتيجة الصفقة الحقيقية
            # (partial_fit) بدل نموذج مجمّد لا يتأثر أبداً بالنتائج الفعلية
            try:
                stored_features = t_row.get('features')
                if stored_features:
                    feat_dict = json.loads(stored_features)
                    x_replay = scaler.transform(np.array([[feat_dict[f] for f in FEATURES]]))
                    model.partial_fit(x_replay, [win_val])
                    joblib.dump(model, MODEL_FILE)
            except Exception:
                pass

            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''INSERT INTO trades (date, symbol, direction, entry, sl, tp, win, note)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (str(datetime.now(timezone.utc).date()), t_row['symbol'], t_row['direction'],
                       t_row['entry'], t_row['sl'], t_row['tp'], win_val, note_str))
            c.execute('DELETE FROM active_trade')
            conn.commit()
            conn.close()

            send_alert(f'Closed {t_row["symbol"]} {t_row["direction"]} -> {note_str}', '🧠 AI Trade Settled')
