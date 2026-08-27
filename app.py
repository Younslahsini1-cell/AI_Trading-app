import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# --- إعدادات الصفحة ---
st.set_page_config(page_title="منصة التداول الذكية", layout="wide", page_icon="📈")
st.title("🧠 منصة التداول الكمّي المتطورة")

# --- القائمة الجانبية (الإعدادات المركزية) ---
st.sidebar.header("🔑 إعدادات الاتصال")
api_key = st.sidebar.text_input("مفتاح Twelve Data API", type="password", help="أدخل مفتاحك هنا للاتصال بالسوق لايف")
interval = st.sidebar.selectbox("الإطار الزمني للشموع", ["1min", "5min", "15min", "1h", "1day"], index=2)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ خيارات الذكاء الاصطناعي")
hidden_layers = st.sidebar.text_input("هيكل الخلايا العصبية", "128, 64, 32", help="أرقام مفصولة بفاصلة تحدد عمق الشبكة")
activation = st.sidebar.selectbox("دالة التنشيط (Activation)", ["relu", "tanh", "logistic"], help="الطريقة الرياضية التي يحلل بها النموذج الأنماط")

st.sidebar.markdown("---")
st.sidebar.header("⚖️ إدارة المخاطر")
rr_ratio = st.sidebar.slider("نسبة العائد للمخاطرة (TP/SL)", min_value=1.0, max_value=5.0, value=2.0, step=0.5, help="مثلاً: 2.0 تعني أن هدف الربح ضعف مسافة وقف الخسارة")

if st.sidebar.button("🗑️ إعادة ضبط ومسح ذاكرة النموذج"):
    st.session_state.clear()
    st.rerun()

# --- تهيئة متغيرات الجلسة والنموذج ---
def init_model():
    # تحويل النص المدخل في القائمة الجانبية إلى أرقام للشبكة العصبية
    layers = tuple(map(int, hidden_layers.split(',')))
    return MLPClassifier(hidden_layer_sizes=layers, activation=activation, solver='adam', warm_start=True, max_iter=500)

if 'model' not in st.session_state:
    st.session_state.model = init_model()
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = pd.DataFrame(columns=['Accuracy'])
if 'total_samples' not in st.session_state:
    st.session_state.total_samples = 0

# --- تعريف دوال التحليل ---
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

def fetch_twelve_data(symbol, api_key, interval, outputsize=500):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={api_key}&outputsize={outputsize}"
    response = requests.get(url).json()
    if 'values' in response:
        df = pd.DataFrame(response['values'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    return None

def analyze_market(symbol, api_key, interval, col, rr):
    with col:
        st.subheader(f"📊 {symbol}")
        with st.spinner("جاري قراءة السوق..."):
            live_df = fetch_twelve_data(symbol, api_key, interval, outputsize=50)
            if live_df is not None:
                processed_live = feature_engineering(live_df)
                features = ['return', 'volatility', 'body', 'momentum_5']
                current_row = processed_live.iloc[-1]
                
                current_X = current_row[features].values.reshape(1, -1)
                current_X_scaled = st.session_state.scaler.transform(current_X)
                
                prediction = st.session_state.model.predict(current_X_scaled)[0]
                proba = st.session_state.model.predict_proba(current_X_scaled)[0]
                
                current_price = current_row['close']
                vol = max(current_row['volatility'], 0.0001)
                
                # حساب الهدف والوقف بناءً على شريط التمرير (Slider) الجانبي
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

                # التعلم المستمر أثناء المراقبة الحية
                st.session_state.model.partial_fit(current_X_scaled, [prediction])
                st.session_state.total_samples += 1

# --- التبويبات الرئيسية ---
tab1, tab2 = st.tabs(["📈 التدريب وتطور النموذج", "📡 شاشة التداول المباشر"])

with tab1:
    st.header("مؤشرات أداء الذكاء الاصطناعي")
    
    # بطاقات الإحصائيات العلوية
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    current_acc = st.session_state.training_history['Accuracy'].iloc[-1] if not st.session_state.training_history.empty else 0
    col_metrics1.metric("حالة النموذج", "مُدَرَّب وجاهز 🟢" if st.session_state.is_trained else "يحتاج تدريب 🔴")
    col_metrics2.metric("حجم الذاكرة (عدد الشموع المدروسة)", f"{st.session_state.total_samples} شمعة")
    col_metrics3.metric("أحدث دقة مسجلة", f"{current_acc:.2f}%")

    # الرسم البياني لتطور الدقة
    if not st.session_state.training_history.empty:
        st.write("**مسار تطور دقة النموذج عبر عمليات التدريب المستمرة:**")
        st.line_chart(st.session_state.training_history)

    st.markdown("---")
    st.subheader("إضافة بيانات جديدة للنموذج")
    train_symbol = st.text_input("رمز التداول للتدريب (مثال: EUR/USD)", "EUR/USD")
    data_source = st.radio("آلية السحب:", ["API (تلقائي)", "رفع ملف CSV (يدوي)"], horizontal=True)
    
    train_df = None
    if data_source == "API (تلقائي)":
        out_size = st.slider("عدد الشموع التاريخية المراد سحبها", 100, 5000, 1000, 100)
        if st.button("سحب البيانات وبدء التدريب", use_container_width=True):
            if not api_key: 
                st.error("⚠️ يرجى إدخال مفتاح Twelve Data API في القائمة الجانبية أولاً.")
            else:
                with st.spinner("جاري سحب البيانات..."):
                    train_df = fetch_twelve_data(train_symbol, api_key, interval, out_size)
    else:
        uploaded_file = st.file_uploader("ارفع ملف CSV", type="csv")
        if uploaded_file:
            train_df = pd.read_csv(uploaded_file)

    if train_df is not None:
        with st.spinner("تجري الآن عملية المعالجة وتحديث الأوزان العصبية..."):
            processed = feature_engineering(train_df)
            X, Y = prepare_data(processed)
            X_scaled = st.session_state.scaler.fit_transform(X)
            
            st.session_state.model.fit(X_scaled, Y)
            st.session_state.is_trained = True
            st.session_state.total_samples += len(Y)
            
            # تسجيل الدقة في التاريخ للرسم البياني
            acc = st.session_state.model.score(X_scaled, Y) * 100
            new_record = pd.DataFrame({'Accuracy': [acc]})
            st.session_state.training_history = pd.concat([st.session_state.training_history, new_record], ignore_index=True)
            
            st.success(f"✅ تم التدريب وتحديث ذاكرة النموذج بنجاح! الدقة المحققة: {acc:.2f}%")
            st.rerun()

with tab2:
    st.header("مراقبة الأسواق الحية")
    if not st.session_state.is_trained:
        st.warning("⚠️ النموذج حالياً لا يملك بيانات ليتخذ قراراً. عد للتبويب الأول وقم بتدريبه.")
    else:
        st.info("💡 ملاحظة: يتم حساب أهداف الصفقات (TP) والوقف (SL) تلقائياً بناءً على شريط 'إدارة المخاطر' في القائمة الجانبية.")
        m_col1, m_col2 = st.columns(2)
        market1 = m_col1.text_input("السوق الأول (المعدن)", "XAU/USD")
        market2 = m_col2.text_input("السوق الثاني (سوق الزخم)", "BTC/USD")
        
        if st.button("🚀 تحليل الأسواق واستخراج الإشارات الآن", use_container_width=True):
            if not api_key: 
                st.error("⚠️ أدخل مفتاح API في القائمة الجانبية لكي يتمكن الذكاء الاصطناعي من رؤية السوق المباشر.")
            else:
                r_col1, r_col2 = st.columns(2)
                # نمرر قيمة شريط التمرير (rr_ratio) لحساب الأهداف بدقة
                analyze_market(market1, api_key, interval, r_col1, rr_ratio)
                analyze_market(market2, api_key, interval, r_col2, rr_ratio)
