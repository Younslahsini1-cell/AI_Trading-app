"""
XAU/USD Deep AI Engine — نسخة مُصلَّحة + مطوَّرة
=====================================================
هذا الملف نسخة معدّلة من التطبيق الأصلي. كل تعديل جوهري مُعلَّم بتعليق
يبدأ بـ "# FIX:" حتى يسهل عليك مقارنته بالنسخة القديمة.

أهم التغييرات الوظيفية (ملخّص، وموجود أيضا في ردّي في المحادثة):
 1) توحيد مصدر واختيار الفريم الزمني للتدريب والتنفيذ الحي (Twelve Data
    بالساعة لكليهما) بدل تدريب يومي (yfinance) وتنفيذ بالساعة.
 2) حذف ميزة dxy_roc المزيّفة (كانت=0 دائماً وقت التشغيل) بدل الإبقاء
    عليها ميزة وهمية.
 3) تعلّم حقيقي من نتائج الصفقات المغلقة عبر partial_fit بدل نموذج مجمّد.
 4) استدعاء واحد فقط لبيانات Twelve Data الحيّة لكل دورة تحديث بدل اثنين.
 5) إضافة طبقة "رأي ثان" من Claude API (اختياري) قبل فتح أي صفقة.
 6) اسم ملفات النموذج تغيّر (v2) عمداً حتى لا يتم تحميل نموذج قديم
    غير متوافق أبعاد الميزات معه.
"""

from datetime import datetime, timezone
import json
import os
import sqlite3

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
st.sidebar.header('🧠 الرأي الثاني (Claude API)')
# FIX: طبقة جديدة كاملة — استدعاء اختياري لـ Claude كمراجع قبل فتح أي صفقة
use_claude = st.sidebar.checkbox('تفعيل مراجعة Claude قبل فتح الصفقة', value=(load_setting('use_claude', '1') == '1'))
save_setting('use_claude', '1' if use_claude else '0')

anthropic_key = st.sidebar.text_input('مفتاح Anthropic API', type='password', value=load_setting('anthropic_key', ''))
save_setting('anthropic_key', anthropic_key)

anthropic_model = st.sidebar.text_input('اسم النموذج', value=load_setting('anthropic_model', 'claude-sonnet-5'))
save_setting('anthropic_model', anthropic_model)
st.sidebar.caption('راجع docs.claude.com للتأكد من اسم النموذج المتاح حالياً في حسابك.')

min_claude_conf = st.sidebar.slider('أدنى ثقة مطلوبة من Claude (%)', 40, 95, 60, 1)

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
            df = pd.DataFrame(res['values'])[['open', 'high', 'low', 'close']].astype(float)
            return df.iloc[::-1].reset_index(drop=True)
    except Exception:
        pass
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
def train_deep_model(api_key):
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)

    df_train = fetch_training_data_twelve(api_key)
    df_train = apply_deep_indicators(df_train)

    if df_train.empty or len(df_train) < 100:
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        scaler = StandardScaler()
        return model, scaler

    X = df_train[FEATURES].values[:-1]
    y = np.where(df_train['close'].shift(-1) > df_train['close'], 1, 0)[:-1]

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
    model.fit(X_sc, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return model, scaler

model, scaler = train_deep_model(twelve_key)

# --- الرأي الثاني: Claude API ---
def get_claude_review(direction, last_row, mlp_conf, api_key, model_name):
    """
    FIX: طبقة جديدة. تُستدعى فقط عندما تجاوزت ثقة الشبكة العصبية الحد
    الأدنى، حتى لا نستهلك استدعاءات Claude API بلا داعٍ. عند أي فشل
    (مفتاح ناقص، شبكة، تحليل JSON) ترجع None ويستمر القرار بالاعتماد
    على الشبكة العصبية وحدها بدل تعطيل التطبيق بالكامل.
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
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model_name,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []))
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
        claude_result = get_claude_review(direction, last, conf, anthropic_key, anthropic_model)
        if claude_result is not None:
            if not claude_result['agree'] or claude_result['confidence'] < min_claude_conf:
                note = (f"الشبكة العصبية اقترحت {direction} بثقة {conf:.1f}%، لكن مراجعة Claude لم توافق "
                        f"(ثقته: {claude_result['confidence']:.0f}%) — تم تجاهل الإشارة تحفظاً.")
                return note, claude_result

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
        claude_line = f"\nمراجعة Claude: موافق ({claude_result['confidence']:.0f}%) — {claude_result['reason']}"
    send_alert(f'🧠 AI Trade Executed: {direction}\nEntry: ${curr}\nSL: ${sl_p}\nTP: ${tp_p}\nAI Confidence: {conf:.1f}%{claude_line}')
    return f'تم اتخاذ قرار ({direction}) بثقة {conf:.1f}% وتم إرسال التنبيه.', claude_result


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

tab1, tab2 = st.tabs(['⚡ حالة الذكاء الاصطناعي', '📊 سجل الخبرات المكتسبة (الصفقات)'])

with tab1:
    if not twelve_key:
        st.warning('⚠️ النظام نائم: أدخل مفتاح Twelve Data API لإيقاظ الشبكة العصبية وربطها بالسوق.')

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

    if claude_info is not None:
        agree_txt = '✅ موافق' if claude_info['agree'] else '❌ غير موافق'
        st.markdown(f"""<div class="claude-note">
            <b>🧠 رأي Claude:</b> {agree_txt} — ثقة {claude_info['confidence']:.0f}%<br>{claude_info['reason']}
        </div>""", unsafe_allow_html=True)

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

    if not c_active.empty and not df_live_processed.empty:
        t_row = c_active.iloc[0]
        last_row = df_live_processed.iloc[-1]

        try:
            x_current = scaler.transform(last_row[FEATURES].values.reshape(1, -1))
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
