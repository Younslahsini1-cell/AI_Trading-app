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

# --- تعريف الدوال ---
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
        st.error(f"خطأ في جلب بيانات {symbol}: تأكد من الرمز ومفتاح API")
        return None

def analyze_market(symbol, api_key, interval, col):
    """دالة مخصصة لتحليل سوق معين وعرض نتائجه داخل عمود محدد"""
    with col:
        st.subheader(f"📊 تحليل سوق {symbol}")
        
        with st.spinner(f"جاري قراءة تحركات {symbol}..."):
            live_df = fetch_twelve_data(symbol, api_key, interval, outputsize=50)
            
            if live_df is not None:
                processed_live = feature_engineering(live_df)
                features = ['return', 'volatility', 'body', 'momentum_5']
                current_row = processed_live.iloc[-1]
                
                current_X = current_row[features].values.reshape(1, -1)
                current_X_scaled = st.session_state.scaler.transform(current_X)
                
                prediction = st.session_state.model.predict(current_X_scaled)[0]
                proba = st.session_state.model.predict_proba(current_X_scaled)[0]
                
                # استخراج بيانات الشمعة الحالية لحساب TP و SL
                current_price = current_row['close']
                vol = max(current_row['volatility'], 0.0001) # منع القسمة على صفر
                momentum = current_row['momentum_5']
                body = current_row['body']
                
                # إدارة المخاطر (نسبة العائد للمخاطرة 1:2)
                sl_distance = vol * 1.5
                tp_distance = vol * 3.0
                
                st.markdown("---")
                if prediction == 1:
                    st.success(f"🟢 **صفقة شراء (BUY)** | نسبة الثقة: {proba[1]*100:.1f}%")
                    st.write(f"**سعر الدخول:** `{current_price}`")
                    st.write(f"**🎯 الهدف (TP):** `{round(current_price + tp_distance, 4)}`")
                    st.write(f"**🛡️ وقف الخسارة (SL):** `{round(current_price - sl_distance, 4)}`")
                    
                    st.markdown("**📝 أسباب الدخول:**")
                    st.caption(f"- **الزخم (Momentum):** {'يدعم الصعود بقوة' if momentum > 0 else 'ضعيف حالياً لكن الذكاء الاصطناعي يتوقع ارتداداً صاعداً بناءً على تاريخ السعر.'}")
                    st.caption(f"- **هيكل الشمعة:** {'شمعة شرائية (إغلاق أعلى من الفتح)' if body > 0 else 'تجميع سيولة قبل الانطلاق.'}")
                    st.caption(f"- **التقلبات (Volatility):** السعر يتحرك في نطاق {round(vol, 4)} نقطة، مما يسمح بوضع وقف خسارة آمن.")
                    
                else:
                    st.error(f"🔴 **صفقة بيع (SELL)** | نسبة الثقة: {proba[0]*100:.1f}%")
                    st.write(f"**سعر الدخول:** `{current_price}`")
                    st.write(f"**🎯 الهدف (TP):** `{round(current_price - tp_distance, 4)}`")
                    st.write(f"**🛡️ وقف الخسارة (SL):** `{round(current_price + sl_distance, 4)}`")
                    
                    st.markdown("**📝 أسباب الدخول:**")
                    st.caption(f"- **الزخم (Momentum):** {'يدعم الهبوط بقوة' if momentum < 0 else 'الذكاء الاصطناعي رصد تشبعاً شرائياً ويتوقع انعكاساً هابطاً.'}")
                    st.caption(f"- **هيكل الشمعة:** {'شمعة بيعية (إغلاق أقل من الفتح)' if body < 0 else 'تصريف سيولة تمهيداً للهبوط.'}")
                    st.caption(f"- **التقلبات (Volatility):** السعر يتحرك في نطاق {round(vol, 4)} نقطة، وتم تحديد الأهداف بناءً على ذلك.")

                # التعلم المستمر
                st.session_state.model.partial_fit(current_X_scaled, [prediction])

# --- الواجهة الجانبية ---
st.sidebar.header("⚙️ الإعدادات العامة")
api_key = st.sidebar.text_input("Twelve Data API Key", type="password")
interval = st.sidebar.selectbox("الإطار الزمني", ["1min", "5min", "15min", "1h", "1day"], index=2)

# --- التبويبات الرئيسية ---
tab1, tab2 = st.tabs(["📂 تدريب النموذج", "📡 التداول الحي (مراقبة سوقين)"])

with tab1:
    st.header("تدريب نموذج الذكاء الاصطناعي")
    st.info("يجب عليك تدريب النموذج هنا على بيانات أي زوج أولاً لكي يتمكن من تحليل الأسواق في التبويب الثاني.")
    
    # دمجنا التدريب هنا لتبسيط الواجهة وتركيزها على التدريب اليدوي/التلقائي
    symbol_train = st.text_input("رمز التداول للتدريب (مثال: EUR/USD)", "EUR/USD")
    data_source = st.radio("اختر مصدر بيانات التدريب:", ["سحب تلقائي عبر API", "سحب يدوي (رفع ملف CSV)"], horizontal=True)
    train_df = None
    
    if data_source == "سحب تلقائي عبر API":
        output_size = st.slider("عدد الشموع التاريخية للتدريب", min_value=100, max_value=5000, value=500, step=100)
        if st.button("سحب البيانات تلقائياً وتدريب النموذج"):
            if not api_key:
                st.error("يرجى إدخال مفتاح Twelve Data API في القائمة الجانبية أولاً.")
            else:
                with st.spinner("جاري سحب البيانات..."):
                    train_df = fetch_twelve_data(symbol_train, api_key, interval, output_size)
    else:
        uploaded_file = st.file_uploader("اختر ملف CSV", type="csv")
        if uploaded_file is not None:
            train_df = pd.read_csv(uploaded_file)
            st.dataframe(train_df.head(3))

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
    st.header("شاشة المراقبة المزدوجة (Live Trading)")
    
    if not st.session_state.is_trained:
        st.warning("⚠️ لا يمكن استخراج إشارات. قم بتدريب النموذج في التبويب الأول أولاً.")
    else:
        st.write("قم بإدخال رمزي السوقين اللذين ترغب في مراقبتهما معاً:")
        
        # تقسيم الشاشة لعمودين لإدخال الأسواق
        input_col1, input_col2 = st.columns(2)
        market1 = input_col1.text_input("السوق الأول (المعدن الأصفر)", "XAU/USD")
        market2 = input_col2.text_input("السوق الثاني (سوق عالي الزخم)", "BTC/USD")
        
        if st.button("🚀 قراءة الأسواق واستخراج الصفقات الآن", use_container_width=True):
            if not api_key:
                st.error("أدخل مفتاح API في القائمة الجانبية.")
            else:
                # تقسيم الشاشة لعمودين لعرض النتائج جنباً إلى جنب
                res_col1, res_col2 = st.columns(2)
                
                # تشغيل التحليل للسوقين
                analyze_market(market1, api_key, interval, res_col1)
                analyze_market(market2, api_key, interval, res_col2)
