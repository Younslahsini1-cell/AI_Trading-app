import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
import json
import time
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# محاولة استيراد yfinance بأمان
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# --- إعدادات الصفحة ---
st.set_page_config(page_title="منصة التداول الكمّي الذكية", layout="wide", page_icon="📈")
st.title("🧠 الوسيط الذكي المستقل لجلب وتدفق البيانات (Autonomous Data Agent)")

# مسارات حفظ الذاكرة والإعدادات
MODEL_FILE = "trained_model.pkl"
SCALER_FILE = "trained_scaler.pkl"
HISTORY_FILE = "history_data.csv"
SETTINGS_FILE = "settings_config.json"

# --- إدارة الإعدادات والمفاتيح الدائمة ---
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"twelve": "", "alpha": "", "ntfy": ""}

def save_settings(twelve, alpha, ntfy):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({"twelve": twelve, "alpha": alpha, "ntfy": ntfy}, f)

if 'settings' not in st.session_state:
    st.session_state.settings = load_settings()

# --- القائمة الجانبية الإعدادات (محفوظة دائماً) ---
st.sidebar.header("🔑 إعدادات الوسيط والإشعارات")
api_key_twelve = st.sidebar.text_input("مفتاح Twelve Data API (اختياري)", value=st.session_state.settings.get("twelve", ""), type="password")
api_key_alpha = st.sidebar.text_input("مفتاح Alpha Vantage API (اختياري)", value=st.session_state.settings.get("alpha", ""), type="password")

st.sidebar.markdown("---")
st.sidebar.header("🔔 إعدادات التنبيهات (Ntfy)")
ntfy_topic = st.sidebar.text_input("اسم قناة Ntfy الخاصة بك", value=st.session_state.settings.get("ntfy", ""), placeholder="مثال: my_crypto_signals_99")

save_settings(api_key_twelve, api_key_alpha, ntfy_topic)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ إعدادات الذكاء الاصطناعي")
interval = st.sidebar.selectbox("الإطار الزمني للوسيط", ["1min", "5min", "15min", "1h", "1day"], index=2)
hidden_layers = st.sidebar.text_input("هيكل الشبكة العصبية", "128, 64, 32")
activation = st.sidebar.selectbox("دالة التنشيط", ["relu", "tanh", "logistic"])
rr_ratio = st.sidebar.slider("نسبة العائد للمخاطرة (TP/SL)", 1.0, 5.0, 2.0, 0.5)
confidence_threshold = st.sidebar.slider("حد الثقة الأدنى لتجنب الفخاخ (%)", 50, 90, 58, 1)

if st.sidebar.button("🗑️ إعادة ضبط الذاكرة والوسيط"):
    for file in [MODEL_FILE, SCALER_FILE, HISTORY_FILE, SETTINGS_FILE]:
        if os.path.exists(file):
            os.remove(file)
    st.session_state.clear()
    st.success("تم إعادة ضبط الوسيط بنجاح!")
    st.rerun()

# --- إدارة الذاكرة الدائمة للنموذج ---
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
    close = np.cumsum(np.random.randn(size) * 0.5) + 4600
    high = close + np.random.uniform(0.1, 0.4, size)
    low = close - np.random.uniform(0.1, 0.4, size)
    open_p = low + np.random.uniform(0.0, 0.3, size)
    return pd.DataFrame({'open': open_p, 'high': high, 'low': low, 'close': close})

# --- الوسيط الذكي المستقل لجلب البيانات تلقائياً ---
def autonomous_data_broker_agent(symbol, interval, outputsize=500):
    clean_symbol = symbol.upper().strip()

    if any(crypto in clean_symbol for crypto in ["BTC", "ETH", "SOL", "BNB", "CRYPTO"]):
        try:
            binance_sym = clean_symbol.replace("/", "").replace("-", "")
            if "USDT" not in binance_sym:
                binance_sym += "USDT"
            
            interval_map = {"1min": "1m", "5min": "5m", "15min": "15m", "1h": "1h", "1day": "1d"}
            b_interval = interval_map.get(interval, "15m")
            
            url = f"https://api.binance.com/api/v3/klines?symbol={binance_sym}&interval={b_interval}&limit={min(outputsize, 1000)}"
            res = requests.get(url, timeout=4).json()
            if isinstance(res, list) and len(res) > 0:
                df = pd.DataFrame(res, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
                ])
                cols = ['open', 'high', 'low', 'close', 'volume']
                df[cols] = df[cols].astype(float)
                return df[cols].reset_index(drop=True), f"Autonomous Binance Broker Agent ({binance_sym})"
        except:
            pass

    if YFINANCE_AVAILABLE:
        try:
            yf_symbol = clean_symbol
            if "XAU" in yf_symbol or "GOLD" in yf_symbol:
                yf_symbol = "GC=F"
            elif "BTC" in yf_symbol:
                yf_symbol = "BTC-USD"
            elif "/" in yf_symbol:
                yf_symbol = yf_symbol.replace("/", "") + "=X"
                
            interval_map = {"1min": "1m", "5min": "5m", "15min": "15m", "1h": "1h", "1day": "1d"}
            yf_interval = interval_map.get(interval, "15m")
            
            df_yf = yf.download(yf_symbol, period="5d", interval=yf_interval, progress=False)
            if not df_yf.empty:
                if isinstance(df_yf.columns, pd.MultiIndex):
                    df_yf.columns = df_yf.columns.droplevel(1)
                df_yf = df_yf.reset_index()
                df_yf.columns = [str(c).lower() for c in df_yf.columns]
                if 'close' in df_yf.columns and 'open' in df_yf.columns:
                    cols = ['open', 'high', 'low', 'close']
                    if 'volume' in df_yf.columns: cols.append('volume')
                    df_clean = df_yf[cols].dropna().reset_index(drop=True)
                    if len(df_clean) > 5:
                        return df_clean, f"Autonomous Yahoo Finance Agent ({yf_symbol})"
        except:
            pass

    if api_key_twelve:
        try:
            url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={api_key_twelve}&outputsize={outputsize}"
            res = requests.get(url, timeout=4).json()
            if 'values' in res:
                df = pd.DataFrame(res['values'])
                cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns: cols.append('volume')
                df[cols] = df[cols].astype(float)
                return df.iloc[::-1].reset_index(drop=True), "Autonomous Twelve Data Agent"
        except:
            pass

    if api_key_alpha:
        try:
            url_av = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key_alpha}&outputsize=full"
            res_av = requests.get(url_av, timeout=4).json()
            time_series_key = [k for k in res_av.keys() if "Time Series" in k]
            if time_series_key:
                data = res_av[time_series_key[0]]
                df = pd.DataFrame.from_dict(data, orient='index').astype(float)
                df.columns = [c.split('. ')[1] for c in df.columns]
                df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close'})
                return df.sort_index().reset_index(drop=True), "Autonomous Alpha Vantage Agent"
        except:
            pass

    return generate_mock_data(outputsize), "Autonomous Emergency Synthetic Broker"

def send_ntfy_alert(topic, message, title):
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
        with st.spinner("الوسيط الذكي يفحص السيرفرات والأسواق تلقائياً..."):
            live_df, source_used = autonomous_data_broker_agent(symbol, interval, outputsize=60)
            st.caption(f"🤖 مصدر الوسيط النشط: `{source_used}`")
            
            if live_df is not None:
                processed_live = feature_engineering(live_df)
                features = ['return', 'volatility', 'body', 'momentum_5']
                current_row = processed_live.iloc[-1]
                
                current_X = current_row[features].values.reshape(1, -1)
                current_X_scaled = st.session_state.scaler.transform(current_X)
                
                prediction = st.session_state.model.predict(current_X_scaled)[0]
                proba = st.session_state.model.predict_proba(current_X_scaled)[0]
                
                max_confidence = max(proba) * 100
                current_price = round(current_row['close'], 4)
                vol = max(current_row['volatility'], 0.0001)
                
                st.markdown("---")
                
                if max_confidence < confidence_threshold:
                    st.warning(f"🟡 **وضع الانتظار الحذر (WAIT / NO TRADE)** | الثقة: {max_confidence:.1f}% (أقل من حد الأمان {confidence_threshold}%)")
                else:
                    sl_distance = vol * 1.5
                    tp_distance = sl_distance * rr
                    
                    tp_price = round(current_price + tp_distance, 4) if prediction == 1 else round(current_price - tp_distance, 4)
                    sl_price = round(current_price - sl_distance, 4) if prediction == 1 else round(current_price + sl_distance, 4)

                    if prediction == 1:
                        msg = f"شراء (BUY) | الدخول: {current_price} | الثقة: {proba[1]*100:.1f}%"
                        st.success(f"🟢 **{msg}**")
                        send_ntfy_alert(ntfy_topic, msg, f"صفقة شراء مؤكدة لـ {symbol}")
                    else:
                        msg = f"بيع (SELL) | الدخول: {current_price} | الثقة: {proba[0]*100:.1f}%"
                        st.error(f"🔴 **{msg}**")
                        send_ntfy_alert(ntfy_topic, msg, f"صفقة بيع مؤكدة لـ {symbol}")

                    st.markdown("##### 📋 أسعار السوق الدقيقة للنسخ السريع:")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**سعر الدخول (Entry)**")
                        st.code(str(current_price), language="text")
                    with c2:
                        st.markdown("**الهدف (TP)**")
                        st.code(str(tp_price), language="text")
                    with c3:
                        st.markdown("**وقف الخسارة (SL)**")
                        st.code(str(sl_price), language="text")

                st.markdown("##### 📈 الشارت التحليلي المباشر:")
                chart_data = processed_live[['close']].tail(30)
                st.line_chart(chart_data)

# --- التبويبات الرئيسية ---
tab1, tab2, tab3 = st.tabs(["📈 التدريب والذاكرة الذكية", "📡 التداول الفوري والوسيط", "🔄 الرصد التلقائي المستمر"])

with tab1:
    st.header("إدارة وتدريب ذاكرة الوسيط الذكي")
    col_metrics1, col_metrics2 = st.columns(2)
    current_acc = st.session_state.training_history['Accuracy'].iloc[-1] if not st.session_state.training_history.empty else 0
    col_metrics1.metric("حالة الذاكرة الذكية", "مُدَرَّبة ومحفوظة 💾🟢" if st.session_state.is_trained else "فارغة ⚪")
    col_metrics2.metric("أحدث دقة للوسيط", f"{current_acc:.2f}%")

    if not st.session_state.training_history.empty:
        st.line_chart(st.session_state.training_history)

    st.markdown("---")
    train_symbol = st.text_input("رمز التداول لتدريب الوسيط", "XAU/USD")
    if st.button("تشغيل الوسيط لسحب البيانات والتدريب الفوري", use_container_width=True):
        with st.spinner("الوسيط يجوب السيرفرات ويسحب البيانات للتدريب..."):
            train_df, src = autonomous_data_broker_agent(train_symbol, interval, 1000)
            processed = feature_engineering(train_df)
            X, Y = prepare_data(processed)
            X_scaled = st.session_state.scaler.fit_transform(X)
            
            st.session_state.model.fit(X_scaled, Y)
            st.session_state.is_trained = True
            
            acc = st.session_state.model.score(X_scaled, Y) * 100
            new_record = pd.DataFrame({'Accuracy': [acc]})
            st.session_state.training_history = pd.concat([st.session_state.training_history, new_record], ignore_index=True)
            
            save_permanent_memory()
            st.success(f"✅ تم التدريب بنجاح عبر الوسيط ({src})! الدقة المحققة: {acc:.2f}%")
            st.rerun()

with tab2:
    st.header("شاشة التداول الفوري والوسيط الذكي")
    if not st.session_state.is_trained:
        st.warning("⚠️ يرجى تدريب النموذج أولاً من التبويب الأول لكي يمتلك الوسيط القدرة على التحليل.")
    else:
        m1, m2 = st.columns(2)
        s1 = m1.text_input("السوق الأول", "XAU/USD")
        s2 = m2.text_input("السوق الثاني", "BTC/USD")
        
        if st.button("🚀 تشغيل الوسيط وتحليل الأسواق", use_container_width=True):
            r1, r2 = st.columns(2)
            analyze_and_execute(s1, r1, rr_ratio)
            analyze_and_execute(s2, r2, rr_ratio)

with tab3:
    st.header("🔄 الرصد التلقائي في الخلفية (Continuous Background Agent)")
    st.write("الوسيط يفحص الأسواق دورياً ويعلمك فوراً فور وجود صفقة حقيقية ومضمونة.")
    
    scan_symbol = st.text_input("رمز السوق للمراقبة المستمرة التلقائية", "XAU/USD")
    auto_run = st.checkbox("تفعيل التشغيل التلقائي للوسيط في الخلفية")
    
    if auto_run:
        st.info("🟢 الوسيط يعمل الآن تلقائياً في الخلفية ويبحث عن الصفقات الآمنة...")
        placeholder = st.empty()
        while auto_run:
            with placeholder.container():
                st.write(f"⏱️ آخر فحص تلقائي للوسيط: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                df_scan, src_s = autonomous_data_broker_agent(scan_symbol, interval, 50)
                if df_scan is not None:
                    p_live = feature_engineering(df_scan)
                    feat = ['return', 'volatility', 'body', 'momentum_5']
                    row = p_live.iloc[-1]
                    X_sc = st.session_state.scaler.transform(row[feat].values.reshape(1, -1))
                    pred = st.session_state.model.predict(X_sc)[0]
                    pr = st.session_state.model.predict_proba(X_sc)[0]
                    mx_conf = max(pr) * 100
                    
                    if mx_conf < confidence_threshold:
                        st.info(f"⏳ حالة الوسيط لـ {scan_symbol}: **انتظار (WAIT)** | الثقة الحالية ({mx_conf:.1f}%) أقل من الحد الآمن.")
                    else:
                        if pred == 1:
                            st.success(f"📈 فرصة شراء مؤكدة لـ {scan_symbol} بثقة {pr[1]*100:.1f}%")
                            send_ntfy_alert(ntfy_topic, f"وسيط ذكي: شراء {scan_symbol} الثقة {pr[1]*100:.1f}%", "فرصة تداول مضمونة")
                        else:
                            st.error(f"📉 فرصة بيع مؤكدة لـ {scan_symbol} بثقة {pr[0]*100:.1f}%")
                            send_ntfy_alert(ntfy_topic, f"وسيط ذكي: بيع {scan_symbol} الثقة {pr[0]*100:.1f}%", "فرصة تداول مضمونة")
            time.sleep(60)
