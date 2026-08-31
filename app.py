import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# إعداد الصفحة
st.set_page_config(page_title="أوراكل تريد", page_icon="⚡", layout="wide")

# تنسيق CSS بسيط
st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {background-color: #1d4ed8;}
    .trade-history {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ========== بيانات أولية ==========
if 'prices' not in st.session_state:
    # توليد 60 نقطة سعرية
    np.random.seed(42)
    st.session_state.prices = list(65000 + np.cumsum(np.random.randn(60) * 100))
    st.session_state.labels = [datetime.datetime.now() - datetime.timedelta(minutes=i) for i in range(60, 0, -1)]

if 'balance' not in st.session_state:
    st.session_state.balance = 128450.0

if 'trades' not in st.session_state:
    st.session_state.trades = []

# ========== شريط جانبي ==========
st.sidebar.title("⚡ أوراكل تريد")
st.sidebar.markdown("---")
page = st.sidebar.radio("التنقل", ["الرئيسية", "الرسم البياني", "التداول", "الذكاء الاصطناعي"])

# ========== الصفحة الرئيسية ==========
if page == "الرئيسية":
    st.title("تداول بذكاء مع قوة الذكاء الاصطناعي")
    st.markdown("منصة تداول متطورة تستخدم أحدث خوارزميات التعلم الآلي لتحليل الأسواق وتقديم توصيات دقيقة في الوقت الفعلي.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("سعر BTC/USDT", f"${st.session_state.prices[-1]:,.2f}", delta="+2.34%")
    with col2:
        st.metric("رصيد المحفظة", f"${st.session_state.balance:,.2f}")
    
    # بطاقات إحصائية
    c1, c2, c3 = st.columns(3)
    c1.metric("أرباح اليوم", "+$3,245.80")
    c2.metric("صفقات ناجحة", "87.5%")
    c3.metric("مؤشر الثقة", "92%")

# ========== الرسم البياني ==========
elif page == "الرسم البياني":
    st.header("📊 الرسم البياني المباشر")
    
    # إضافة نقطة جديدة كل فترة (يمكن استخدام st_autorefresh)
    if st.button("تحديث البيانات"):
        new_price = st.session_state.prices[-1] + np.random.randn() * 200
        st.session_state.prices.append(new_price)
        st.session_state.labels.append(datetime.datetime.now())
    
    # إنشاء DataFrame
    df = pd.DataFrame({'time': st.session_state.labels, 'price': st.session_state.prices})
    
    # رسم باستخدام Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['price'], mode='lines', name='BTC/USDT',
                             line=dict(color='#2563eb', width=2)))
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20),
                      paper_bgcolor='white', plot_bgcolor='#fafafa',
                      xaxis_title='الوقت', yaxis_title='السعر (USDT)')
    st.plotly_chart(fig, use_container_width=True)

# ========== التداول ==========
elif page == "التداول":
    st.header("💰 محاكي التداول")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        amount = st.number_input("الكمية (USDT)", min_value=1.0, value=100.0, step=10.0)
    with col2:
        asset = st.selectbox("الأصل", ["BTC", "ETH", "SOL", "BNB"])
    
    buy_col, sell_col = st.columns(2)
    with buy_col:
        if st.button("شراء ▲", use_container_width=True):
            st.session_state.balance -= amount
            st.session_state.trades.append({
                'time': datetime.datetime.now().strftime('%H:%M:%S'),
                'type': 'شراء',
                'asset': asset,
                'amount': amount
            })
            st.success(f"تم شراء {asset} بقيمة ${amount:,.2f}")
    with sell_col:
        if st.button("بيع ▼", use_container_width=True):
            st.session_state.balance += amount
            st.session_state.trades.append({
                'time': datetime.datetime.now().strftime('%H:%M:%S'),
                'type': 'بيع',
                'asset': asset,
                'amount': amount
            })
            st.success(f"تم بيع {asset} بقيمة ${amount:,.2f}")
    
    st.markdown("---")
    st.subheader("سجل الصفقات")
    if st.session_state.trades:
        trades_df = pd.DataFrame(st.session_state.trades)
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("لا توجد صفقات بعد")

# ========== الذكاء الاصطناعي ==========
elif page == "الذكاء الاصطناعي":
    st.header("🧠 توصيات الذكاء الاصطناعي")
    
    # توليد توصية عشوائية
    recs = [
        ("اتجاه صاعد قوي - فرصة شراء", "نموذج الاختراق الصاعد مع حجم تداول مرتفع.", 78),
        ("تصحيح مؤقت - انتظر التأكيد", "مؤشر RSI في منطقة تشبع شرائي.", 65),
        ("اتجاه هابط - تجنب الشراء", "كسر مستويات الدعم الرئيسية.", 82),
        ("تذبذب عرضي - تداول بنطاق", "السوق في حالة توازن.", 55),
    ]
    rec = random.choice(recs)
    
    st.markdown(f"**{rec[0]}**")
    st.write(rec[1])
    st.progress(rec[2] / 100)
    st.caption(f"مستوى الثقة: {rec[2]}%")
    
    # مؤشرات فنية
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RSI", "68.5")
    col2.metric("MACD", "إيجابي")
    col3.metric("المتوسطات", "صاعدة")
    col4.metric("التقلب", "متوسط")
