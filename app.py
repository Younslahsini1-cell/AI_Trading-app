import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# --- إعدادات الصفحة ---
st.set_page_config(page_title="منصة التداول الكمّي الذكية", layout="wide")
st.title("🧠 منصة تدريب الذكاء الاصطناعي للتداول")

# --- تهيئة متغيرات الجلسة ---
if 'model' not in st.session_state:
    st.session_state.model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', warm_start=True)
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False

# --- تعريف الدوال أولاً قبل استدعائها ---
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

def fetch_twelve_data(symbol, api_key, interval="15min", outputsize=500):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={api_key}&outputsize={outputsize}"
    response = requests.get(url).json()
    if 'values' in response:
        df = pd.DataFrame(response['values'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    else:
        st.error(f"خطأ في API: {response.get('message', 'تأكد من الرمز ومفتاح API')}")
        return None

# --- الواجهة الجانبية ---
st.sidebar.header("⚙️ الإعدادات العامة")
api_key = st.sidebar.text_input("Twelve Data API Key", type="password")
symbol = st.sidebar.text_input("رمز التداول", "EUR/USD")
interval = st.sidebar.selectbox("الإطار الزمني", ["1min", "5min", "15min", "1h", "1day"], index=2)

# --- التبويبات الرئيسية ---
tab1, tab2 = st.tabs(["📂 تدريب النموذج", "📡 التداول الحي والتوقعات"])

with tab1:
    st.header("تدريب نموذج الذكاء الاصطناعي")
    data_source = st.radio("اختر مصدر بيانات التدريب:", ["سحب تلقائي عبر API", "سحب يدوي (رفع ملف CSV)"], horizontal=True)
    train_df = None
    
    if data_source == "سحب تلقائي عبر API":
        output_size = st.slider("عدد الشموع التاريخية للتدريب", min_value=100, max_value=5000, value=500, step=100)
        if st.button("سحب البيانات تلقائياً وتدريب النموذج"):
            if not api_key:
                st.error("يرجى إدخال مفتاح Twelve Data API في القائمة الجانبية أولاً.")
            else:
                with st.spinner("جاري سحب البيانات..."):
                    train_df = fetch_twelve_data(symbol, api_key, interval, output_size)
    else:
        uploaded_file = st.file_uploader("اختر ملف CSV", type="csv")
        if uploaded_file is not None:
            train_df = pd.read_csv(uploaded_file)
            st.dataframe(train_df.head(3))
            if st.button("بدء تدريب النموذج على الملف المرفوع"):
                pass

    if train_df is not None:
        with st.spinner("جاري تدريب الخلايا العصبية..."):
            processed_data = feature_engineering(train_df)
            X, Y = prepare_data(processed_data)
            X_scaled = st.session_state.scaler.fit_transform(X)
            st.session_state.model.fit(X_scaled, Y)
            st.session_state.is_trained = True
            acc = st.session_state.model.score(X_scaled, Y) * 100
            st.success(f"✅ تم التدريب بنجاح! دقة النموذج: {acc:.2f}%")

with tab2:
    st.header("استخراج إشارات التداول الحية")
    if not st.session_state.is_trained:
        st.warning("⚠️ يجب تدريب النموذج أولاً من التبويب الأول.")
    else:
        if st.button("تحليل السوق الآن"):
            if not api_key:
                st.error("أدخل مفتاح API في القائمة الجانبية.")
            else:
                with st.spinner("جاري قراءة السوق..."):
                    live_df = fetch_twelve_data(symbol, api_key, interval, outputsize=50)
                    if live_df is not None:
                        processed_live = feature_engineering(live_df)
                        features = ['return', 'volatility', 'body', 'momentum_5']
                        current_X = processed_live[features].iloc[-1].values.reshape(1, -1)
                        current_X_scaled = st.session_state.scaler.transform(current_X)
                        
                        prediction = st.session_state.model.predict(current_X_scaled)[0]
                        proba = st.session_state.model.predict_proba(current_X_scaled)[0]
                        
                        st.markdown("---")
                        if prediction == 1:
                            st.success(f"🟢 **صفقة شراء (BUY)** | نسبة الثقة: {proba[1]*100:.1f}%")
                        else:
                            st.error(f"🔴 **صفقة بيع (SELL)** | نسبة الثقة: {proba[0]*100:.1f}%")
                        
                        st.session_state.model.partial_fit(current_X_scaled, [prediction])
