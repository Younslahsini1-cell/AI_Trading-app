import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# --- إعدادات الصفحة ---
st.set_page_config(page_title="منصة التداول الذكية", layout="wide", page_icon="📈")
st.title("🧠 منصة التداول الكمّي المتطورة")

# مسارات حفظ الذاكرة
MODEL_FILE = "trained_model.pkl"
SCALER_FILE = "trained_scaler.pkl"
HISTORY_FILE = "history_data.csv"

# --- القائمة الجانبية ---
st.sidebar.header("🔑 إعدادات الاتصال")
api_key = st.sidebar.text_input("مفتاح Twelve Data API", type="password")
interval = st.sidebar.selectbox("الإطار الزمني", ["1min", "5min", "15min", "1h", "1day"], index=2)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ خيارات الذكاء الاصطناعي")
hidden_layers = st.sidebar.text_input("هيكل الخلايا العصبية", "128, 64, 32")
activation = st.sidebar.selectbox("دالة التنشيط", ["relu", "tanh", "logistic"])

st.sidebar.markdown("---")
st.sidebar.header("⚖️ إدارة المخاطر")
rr_ratio = st.sidebar.slider("نسبة العائد للمخاطرة (TP/SL)", 1.0, 5.0, 2.0, 0.5)

if st.sidebar.button("🗑️ مسح الذاكرة الدائمة"):
    for file in [MODEL_FILE, SCALER_FILE, HISTORY_FILE]:
        if os.path.exists(file):
            os.remove(file)
    st.session_state.clear()
    st.success("تم مسح الذاكرة!")
    st.rerun()

# --- إدارة الذاكرة ---
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
    """توليد بيانات وهمية حية للاختبار في حال عدم وجود API"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=size, freq='15min')
    close = np.cumsum(np.random.randn(size)) + 100
    high = close + np.random.uniform(0.1, 0.5, size)
    low = close - np.random.uniform(0.1, 0.5, size)
    open_p = low + np.random.uniform(0.0, 0.4, size)
    return pd.DataFrame({'open': open_p, 'high': high, 'low': low, 'close': close})

def fetch_twelve_data(symbol, api_key, interval, outputsize=500):
    if not api_key:
        st.warning("⚠️ لم يتم إدخال API Key، تم استخدام بيانات محاكاة للاختبار.")
        return generate_mock_data(outputsize)
        
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={api_key}&outputsize={outputsize}"
    try:
        response = requests.get(url).json()
        if 'values' in response:
            df = pd.DataFrame(response['values'])
            
            # التحقق من الأعمدة الموجودة وتفادي الخطأ إذا كان "volume" غير موجود
            cols = ['open', 'high', 'low', 'close']
            if 'volume' in df.columns:
                cols.append('volume')
                
            df[cols] = df[cols].astype(float)
            df = df.iloc[::-1].reset_index(drop=True)
            return df
        else:
            st.error(f"خطأ من API: {response.get('message', 'رمز غير صحيح أو تجاوزت الحد المسموح')}")
            return None
    except Exception as e:
        st.error(f"فشل الاتصال: {e}")
        return None

def analyze_market(symbol, api_key, interval, col, rr):
    with col:
        st.subheader(f"📊 {symbol}")
        with st.spinner("جاري التحليل..."):
            live_df = fetch_twelve_data(symbol, api_key, interval, outputsize=50)
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
                    st.success(f"🟢 **شراء (BUY)** | الثقة: {proba[1]*100:.1f}%")
                    st.write(f"**سعر الدخول:** `{current_price}`")
                    st.write(f"**🎯 الهدف (TP):** `{round(current_price + tp_distance, 4)}`")
                    st.write(f"**🛡️ الوقف (SL):** `{round(current_price - sl_distance, 4)}`")
                else:
                    st.error(f"🔴 **بيع (SELL)** | الثقة: {proba[0]*100:.1f}%")
                    st.write(f"**سعر الدخول:** `{current_price}`")
                    st.write(f"**🎯 الهدف (TP):** `{round(current_price - tp_distance, 4)}`")
                    st.write(f"**🛡️ الوقف (SL):** `{round(current_price + sl_distance, 4)}`")

                st.session_state.model.partial_fit(current_X_scaled, [prediction])
                save_permanent_memory()

# --- التبويبات الرئيسية ---
tab1, tab2 = st.tabs(["📈 التدريب والذاكرة", "📡 التداول المباشر"])

with tab1:
    st.header("أداء النموذج")
    col_metrics1, col_metrics2 = st.columns(2)
    current_acc = st.session_state.training_history['Accuracy'].iloc[-1] if not st.session_state.training_history.empty else 0
    col_metrics1.metric("حالة الذاكرة", "جاهز ومُدَرَّب 🟢" if st.session_state.is_trained else "فارغ ⚪")
    col_metrics2.metric("أحدث دقة", f"{current_acc:.2f}%")

    if not st.session_state.training_history.empty:
        st.line_chart(st.session_state.training_history)

    st.markdown("---")
    st.subheader("إضافة بيانات جديدة")
    train_symbol = st.text_input("رمز التداول", "EUR/USD")
    data_source = st.radio("المصدر:", ["API تلقائي", "رفع ملف CSV"], horizontal=True)
    
    train_df = None
    if data_source == "API تلقائي":
        out_size = st.slider("عدد الشموع", 100, 5000, 1000, 100)
        if st.button("سحب البيانات وبدء التدريب", use_container_width=True):
            with st.spinner("جاري التجهيز..."):
                train_df = fetch_twelve_data(train_symbol, api_key, interval, out_size)
    else:
        uploaded_file = st.file_uploader("ارفع ملف CSV", type="csv")
        if uploaded_file:
            train_df = pd.read_csv(uploaded_file)

    if train_df is not None:
        with st.spinner("جاري معالجة البيانات وتحديث الذاكرة..."):
            processed = feature_engineering(train_df)
            X, Y = prepare_data(processed)
            X_scaled = st.session_state.scaler.fit_transform(X)
            
            st.session_state.model.fit(X_scaled, Y)
            st.session_state.is_trained = True
            
            acc = st.session_state.model.score(X_scaled, Y) * 100
            new_record = pd.DataFrame({'Accuracy': [acc]})
            st.session_state.training_history = pd.concat([st.session_state.training_history, new_record], ignore_index=True)
            
            save_permanent_memory()
            st.success(f"✅ تم التدريب بنجاح! الدقة المحققة: {acc:.2f}%")
            st.rerun()

with tab2:
    st.header("مراقبة الأسواق الحية")
    if not st.session_state.is_trained:
        st.warning("⚠️ قم بتدريب النموذج في التبويب الأول أولاً.")
    else:
        m_col1, m_col2 = st.columns(2)
        market1 = m_col1.text_input("السوق الأول", "XAU/USD")
        market2 = m_col2.text_input("السوق الثاني", "BTC/USD")
        
        if st.button("🚀 تحليل الأسواق واستخراج الإشارات", use_container_width=True):
            r_col1, r_col2 = st.columns(2)
            analyze_market(market1, api_key, interval, r_col1, rr_ratio)
            analyze_market(market2, api_key, interval, r_col2, rr_ratio)
