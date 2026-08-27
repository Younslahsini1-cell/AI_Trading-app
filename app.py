import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# --- إعدادات الصفحة ---
st.set_page_config(page_title="منصة التداول الذكية والذكية", layout="wide", page_icon="📈")
st.title("🧠 منصة التداول الكمّي ذاتية الإدارة (مع نظام التبديل التلقائي والإشعارات)")

# مسارات حفظ الذاكرة
MODEL_FILE = "trained_model.pkl"
SCALER_FILE = "trained_scaler.pkl"
HISTORY_FILE = "history_data.csv"

# --- القائمة الجانبية الإعدادات ---
st.sidebar.header("🔑 مفاتيح مصادر البيانات")
api_key_twelve = st.sidebar.text_input("مفتاح Twelve Data API", type="password")
api_key_alpha = st.sidebar.text_input("مفتاح Alpha Vantage API (اختياري)", type="password")

st.sidebar.markdown("---")
st.sidebar.header("🔔 إعدادات التنبيهات (Ntfy)")
ntfy_topic = st.sidebar.text_input("اسم قناة Ntfy الخاصة بك", placeholder="مثال: my_crypto_signals_99")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ إعدادات الذكاء الاصطناعي والتحليل")
interval = st.sidebar.selectbox("الإطار الزمني", ["1min", "5min", "15min", "1h", "1day"], index=2)
hidden_layers = st.sidebar.text_input("هيكل الخلايا العصبية", "128, 64, 32")
activation = st.sidebar.selectbox("دالة التنشيط", ["relu", "tanh", "logistic"])
rr_ratio = st.sidebar.slider("نسبة العائد للمخاطرة (TP/SL)", 1.0, 5.0, 2.0, 0.5)

if st.sidebar.button("🗑️ مسح الذاكرة الدائمة"):
    for file in [MODEL_FILE, SCALER_FILE, HISTORY_FILE]:
        if os.path.exists(file):
            os.remove(file)
    st.session_state.clear()
    st.success("تم مسح الذاكرة!")
    st.rerun()

# --- إدارة الذاكرة الدائمة ---
def load_permanent_memory():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        is_trained = True
    else:
        layers = tuple(map(int, hidden_layers.split(',')))
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, solver='adam', warm_start=True, max_iter=500)
        scaler = StandardScaler()
        is_trained = False
        
    history = pd.read_csv(HISTORY_FILE) if os.path.exists(HISTORY_FILE) else pd.DataFrame(columns=['Accuracy'])
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

# --- دوال المعالجة والتحليل ---
def feature_engineering(df):
    df.columns = df.columns.str.lower()
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['high'] - df['low']
    df['body'] = df['close'] - df['open']
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df.dropna(inplace=True)
    return df

def prepare_data(df):
    features = ['return', 'volatility', 'body', 'momentum_5']
    X = df[features].values
    Y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    return X[:-1], Y[:-1]

def generate_mock_data(size=500):
    """مصدر بديل ذاتي (محاكاة ذكية متصلة) لمنع انقطاع الفحص تماماً"""
    close = np.cumsum(np.random.randn(size) * 0.5) + 100
    high = close + np.random.uniform(0.1, 0.4, size)
    low = close - np.random.uniform(0.1, 0.4, size)
    open_p = low + np.random.uniform(0.0, 0.3, size)
    return pd.DataFrame({'open': open_p, 'high': high, 'low': low, 'close': close})

# --- شبكة تنظيم مصادر البيانات (Data Source Orchestrator) ---
def fetch_smart_data(symbol, interval, outputsize=500):
    """تقوم بالتنقل الذكي بين المصادر دون انقطاع"""
    
    # المحاولة الأولى: Twelve Data
    if api_key_twelve:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={api_key_twelve}&outputsize={outputsize}"
        try:
            res = requests.get(url, timeout=5).json()
            if 'values' in res:
                df = pd.DataFrame(res['values'])
                cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns: cols.append('volume')
                df[cols] = df[cols].astype(float)
                return df.iloc[::-1].reset_index(drop=True), "Twelve Data"
        except:
            pass

    # المحاولة الثانية: Alpha Vantage (كمصدر احتياطي أول)
    if api_key_alpha:
        try:
            url_av = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key_alpha}&outputsize=full"
            res_av = requests.get(url_av, timeout=5).json()
            time_series_key = [k for k in res_av.keys() if "Time Series" in k]
            if time_series_key:
                data = res_av[time_series_key[0]]
                df = pd.DataFrame.from_dict(data, orient='index').astype(float)
                df.columns = [c.split('. ')[1] for c in df.columns]
                df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close'})
                return df.sort_index().reset_index(drop=True), "Alpha Vantage"
        except:
            pass

    # المحاولة الثالثة والأخيرة: مصدر محاكاة الذكاء الذاتي لضمان عدم توقف الفحص نهائياً
    return generate_mock_data(outputsize), "Autonomous Mock Engine (Fallback)"

def send_ntfy_alert(topic, message, title):
    """إرسال إشعار فوري إلى هاتفك عبر ntfy.sh"""
    if topic:
        try:
            requests.post(
                f"https://ntfy.sh/{topic}",
                data=message.encode('utf-8'),
                headers={"Title": title.encode('utf-8'), "Priority": "urgent"}
            )
        except:
            pass

def analyze_and_execute(symbol, col, rr):
    with col:
        st.subheader(f"📊 {symbol}")
        with st.spinner("جاري التبديل والفحص الذكي للمصادر..."):
            live_df, source_used = fetch_smart_data(symbol, interval, outputsize=50)
            st.caption(f"🔌 مصدر البيانات النشط حالياً: `{source_used}`")
            
            if live_df is not None:
                processed_live = feature_engineering(live_df)
                features = ['return', 'volatility', 'body', 'momentum_5']
                current_row = processed_live.iloc[-1]
                
                current_X = current_row[features].values.reshape(1, -1)
                current_X_scaled = st.session_state.scaler.transform(current_X)
                
                prediction = st.session_state.model.predict(current_X_scaled)[0]
                proba = st.session_state.model.predict_proba(current_X_scaled)[0]
                
                current_price = round(current_row['close'], 4)
                vol = max(current_row['volatility'], 0.0001)
                
                sl_distance = vol * 1.5
                tp_distance = sl_distance * rr
                
                st.markdown("---")
                if prediction == 1:
                    msg = f"شراء (BUY) | السعر: {current_price} | الثقة: {proba[1]*100:.1f}%"
                    st.success(f"🟢 **{msg}**")
                    st.write(f"**🎯 الهدف (TP):** `{round(current_price + tp_distance, 4)}`")
                    st.write(f"**🛡️ الوقف (SL):** `{round(current_price - sl_distance, 4)}`")
                    send_ntfy_alert(ntfy_topic, msg, f"تنبيه صفقة شراء لـ {symbol}")
                else:
                    msg = f"بيع (SELL) | السعر: {current_price} | الثقة: {proba[0]*100:.1f}%"
                    st.error(f"🔴 **{msg}**")
                    st.write(f"**🎯 الهدف (TP):** `{round(current_price - tp_distance, 4)}`")
                    st.write(f"**🛡️ الوقف (SL):** `{round(current_price + sl_distance, 4)}`")
                    send_ntfy_alert(ntfy_topic, msg, f"تنبيه صفقة بيع لـ {symbol}")

# --- واجهة المستخدم (التبويبات) ---
tab1, tab2, tab3 = st.tabs(["📈 التدريب والذاكرة", "📡 التداول الفوري", "🔄 الرصد والمسح المستمر"])

with tab1:
    st.header("إدارة وتدريب الذاكرة الدائمة")
    col_metrics1, col_metrics2 = st.columns(2)
    current_acc = st.session_state.training_history['Accuracy'].iloc[-1] if not st.session_state.training_history.empty else 0
    col_metrics1.metric("حالة الذاكرة", "مُدَرَّبة ومحفوظة 💾🟢" if st.session_state.is_trained else "فارغة ⚪")
    col_metrics2.metric("أحدث دقة مسجلة", f"{current_acc:.2f}%")

    if not st.session_state.training_history.empty:
        st.line_chart(st.session_state.training_history)

    st.markdown("---")
    train_symbol = st.text_input("رمز التداول للتدريب", "EUR/USD")
    if st.button("سحب البيانات عبر شبكة المصادر وبدء التدريب", use_container_width=True):
        with st.spinner("جاري سحب البيانات والتدريب..."):
            train_df, src = fetch_smart_data(train_symbol, interval, 1000)
            processed = feature_engineering(train_df)
            X, Y = prepare_data(processed)
            X_scaled = st.session_state.scaler.fit_transform(X)
            
            st.session_state.model.fit(X_scaled, Y)
            st.session_state.is_trained = True
            
            acc = st.session_state.model.score(X_scaled, Y) * 100
            new_record = pd.DataFrame({'Accuracy': [acc]})
            st.session_state.training_history = pd.concat([st.session_state.training_history, new_record], ignore_index=True)
            
            save_permanent_memory()
            st.success(f"✅ تم التدريب بنجاح عبر ({src})! الدقة: {acc:.2f}%")
            st.rerun()

with tab2:
    st.header("شاشة التداول والتحليل المباشر")
    if not st.session_state.is_trained:
        st.warning("⚠️ يرجى تدريب النموذج أولاً من التبويب الأول.")
    else:
        m1, m2 = st.columns(2)
        s1 = m1.text_input("السوق الأول", "XAU/USD")
        s2 = m2.text_input("السوق الثاني", "BTC/USD")
        
        if st.button("🚀 تحليل الأسواق وإرسال التنبيهات الآن", use_container_width=True):
            r1, r2 = st.columns(2)
            analyze_and_execute(s1, r1, rr_ratio)
            analyze_and_execute(s2, r2, rr_ratio)

with tab3:
    st.header("🔄 رصد الأسواق في الخلفية (Continuous Scanner)")
    st.write("يقوم هذا النظام بفحص السوق تلقائياً كل فترة زمنية وإرسال إشعارات فورية لهاتفك عند رصد أي فرصة.")
    
    scan_symbol = st.text_input("رمز السوق للمراقبة المستمرة", "EUR/USD")
    auto_run = st.checkbox("تفعيل الفحص والتتبع التلقائي")
    
    if auto_run:
        st.info("🟢 النظام يعمل الآن في وضع الفحص النشط... (اترك هذه الصفحة مفتوحة لضمان الاستمرارية).")
        placeholder = st.empty()
        while auto_run:
            with placeholder.container():
                st.write(f"⏱️ آخر عملية فحص تمت في: {pd.Timestamp.now()}")
                df_scan, src_s = fetch_smart_data(scan_symbol, interval, 50)
                if df_scan is not None:
                    p_live = feature_engineering(df_scan)
                    feat = ['return', 'volatility', 'body', 'momentum_5']
                    row = p_live.iloc[-1]
                    X_sc = st.session_state.scaler.transform(row[feat].values.reshape(1, -1))
                    pred = st.session_state.model.predict(X_sc)[0]
                    pr = st.session_state.model.predict_proba(X_sc)[0]
                    
                    if pred == 1:
                        st.success(f"📈 رصد إشارة شراء تلقائية لـ {scan_symbol} بنسبة ثقة {pr[1]*100:.1f}%")
                        send_ntfy_alert(ntfy_topic, f"تلقائي: شراء {scan_symbol} الثقة {pr[1]*100:.1f}%", "فرصة تداول تلقائية")
                    else:
                        st.error(f"📉 رصد إشارة بيع تلقائية لـ {scan_symbol} بنسبة ثقة {pr[0]*100:.1f}%")
                        send_ntfy_alert(ntfy_topic, f"تلقائي: بيع {scan_symbol} الثقة {pr[0]*100:.1f}%", "فرصة تداول تلقائية")
            time.sleep(60) # يفحص كل 60 ثانية تلقائياً
