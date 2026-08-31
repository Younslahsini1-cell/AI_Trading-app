"""
==========================================================================
PATCH — دمج ICT فعلياً في قرار الصفقة + تدريب أذكى وأدق
==========================================================================

المشكلة في النسخة الحالية:
---------------------------
1) لوحة ICT (Market Structure / OB / FVG / Liquidity / OTE) هي عرض فقط.
   الشبكة العصبية (MLP) تقرر الصفقة بمعزل تام عنها، فتفوت تأكيد ICT
   وقد تدخل عكس السيولة/الهيكل.
2) الميزات (FEATURES) أربعة فقط: atr, ema_50, ema_200, rsi — لا تحتوي
   أي معلومة عن البنية أو السيولة أو الاندفاع، فالنموذج "أعمى" عن ICT.
3) التدريب يحدث مرة واحدة فقط عند غياب الملفات، ولا يُعاد دورياً على
   بيانات أحدث => النموذج يتجمّد.
4) نموذج واحد (MLP) بلا مقارنة/تصويت => لا مقاومة لعدم استقرار الشبكة.

الحل هنا (طبّق الأجزاء التالية داخل الملف الأصلي، بنفس أسماء الدوال):
--------------------------------------------------------------------
A) دالة compute_ict_features_row: تحسب ميزات ICT رقمية لكل صف تاريخياً
   (بشكل vectorized-friendly عبر rolling window) لتُستخدم في التدريب
   وليس فقط في العرض الحي.
B) FEATURES موسّعة لتشمل هذه الميزات.
C) ai_scanner معدّلة: تحسب ict_data على نفس الشموك المستخدمة في القرار،
   وتشترط توافق اتجاه AI مع bias الخاص بـ ICT، وتدمج ثقة ICT (confidence)
   مع ثقة الشبكة بدل تجاهلها.
D) تدريب دوري (كل N ساعة) بدل مرة واحدة، + Ensemble من نموذجين
   (MLP + HistGradientBoosting) بالتصويت على الاحتمال، لتقليل الضجيج
   وزيادة الاستقرار قبل الحدث لا بعده.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# ==========================================================================
# A) ميزات ICT رقمية — قابلة للحساب على كل صف تاريخي وليس فقط آخر شمعة
# ==========================================================================
#
# فكرة الحساب: بدل تشغيل run_ict_engine() الكامل (وهو مصمم لآخر شمعة فقط)
# على كل صف في 5000 شمعة (بطيء جداً)، نحسب نسخة مبسّطة ومتجهة (vectorized)
# تُعطي نفس المعنى الجوهري لكل صف باستخدام نافذة متحركة محدودة.

ICT_FEATURES = [
    "struct_bias",        # 1 = صاعد، -1 = هابط، 0 = محايد (آخر BOS/CHoCH)
    "liquidity_sweep",     # 1 = حدث اصطياد سيولة (BSL/SSL) في آخر شمعة
    "displacement_score",  # قوة آخر شمعة اندفاع نسبة لـ ATR
    "fvg_bias",             # 1 = فجوة صاعدة قريبة، -1 = هابطة، 0 = لا شيء
    "dist_to_ob_atr",       # المسافة بين السعر وأقرب Order Block / ATR
    "session_momentum",     # net_change / range لآخر 24 شمعة (اتجاه الجلسة)
]

FEATURES_FULL = [
    "atr",
    "ema_50",
    "ema_200",
    "rsi",
] + ICT_FEATURES


def compute_ict_features_bulk(df, swing_lookback=3, ob_mult=1.2,
                               struct_window=80, session_window=24):
    """
    يحسب ميزات ICT رقمية لكل صف في df (بعد apply_deep_indicators)
    بشكل قابل للاستخدام في تدريب النموذج، وليس فقط لآخر شمعة.

    df يجب أن يحتوي: open, high, low, close, atr (من apply_deep_indicators)
    """

    n = len(df)
    if n < struct_window + 5:
        return pd.DataFrame()

    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values

    struct_bias = np.zeros(n)
    liquidity_sweep = np.zeros(n)
    displacement_score = np.zeros(n)
    fvg_bias = np.zeros(n)
    dist_to_ob_atr = np.zeros(n)
    session_momentum = np.zeros(n)

    for i in range(struct_window, n):

        window_hi = high[i - struct_window:i + 1]
        window_lo = low[i - struct_window:i + 1]

        # --- Structure bias: هل آخر قمة/قاع كسرت السابق؟
        mid = struct_window // 2
        recent_high = np.max(window_hi[mid:])
        prior_high = np.max(window_hi[:mid])
        recent_low = np.min(window_lo[mid:])
        prior_low = np.min(window_lo[:mid])

        if recent_high > prior_high and recent_low >= prior_low:
            struct_bias[i] = 1.0
        elif recent_low < prior_low and recent_high <= prior_high:
            struct_bias[i] = -1.0
        else:
            struct_bias[i] = 0.0

        # --- Liquidity sweep: اختراق أعلى/أدنى نافذة ثم إغلاق بالعكس
        bsl = np.max(window_hi[:-1])
        ssl = np.min(window_lo[:-1])

        if high[i] > bsl and close[i] < bsl:
            liquidity_sweep[i] = 1.0
        elif low[i] < ssl and close[i] > ssl:
            liquidity_sweep[i] = 1.0
        else:
            liquidity_sweep[i] = 0.0

        # --- Displacement score
        a = atr[i]
        if np.isfinite(a) and a > 0:
            body = close[i] - open_[i]
            displacement_score[i] = min(abs(body) / a, 5.0)

        # --- FVG bias (آخر 3 شموع)
        if i >= 2:
            c1_high, c1_low = high[i - 2], low[i - 2]
            c3_high, c3_low = high[i], low[i]
            if c1_high < c3_low:
                fvg_bias[i] = 1.0
            elif c1_low > c3_high:
                fvg_bias[i] = -1.0

        # --- Distance to nearest displacement candle (proxy لـ OB) بوحدات ATR
        if np.isfinite(a) and a > 0:
            lookback_slice = np.arange(max(0, i - 40), i)
            if len(lookback_slice) > 0:
                bodies = np.abs(close[lookback_slice] - open_[lookback_slice])
                strong_idx = lookback_slice[bodies > ob_mult * a] if a > 0 else []
                if len(strong_idx) > 0:
                    nearest = strong_idx[-1]
                    ob_price = close[nearest]
                    dist_to_ob_atr[i] = (close[i] - ob_price) / a
                else:
                    dist_to_ob_atr[i] = 0.0

        # --- Session momentum
        s_slice_close = close[max(0, i - session_window):i + 1]
        s_slice_high = high[max(0, i - session_window):i + 1]
        s_slice_low = low[max(0, i - session_window):i + 1]
        rng = np.max(s_slice_high) - np.min(s_slice_low)
        if rng > 0:
            session_momentum[i] = (s_slice_close[-1] - s_slice_close[0]) / rng

    out = pd.DataFrame({
        "struct_bias": struct_bias,
        "liquidity_sweep": liquidity_sweep,
        "displacement_score": displacement_score,
        "fvg_bias": fvg_bias,
        "dist_to_ob_atr": dist_to_ob_atr,
        "session_momentum": session_momentum,
    })

    return out


def build_training_dataframe(df_indicators):
    """
    يدمج المؤشرات الأساسية + ميزات ICT في جدول تدريب واحد جاهز.
    استبدل استدعاء apply_deep_indicators + FEATURES القديمة بهذه الدالة
    في _background_train_and_save و في ai_scanner.
    """

    ict_feats = compute_ict_features_bulk(df_indicators)

    if ict_feats.empty:
        return pd.DataFrame()

    merged = df_indicators.iloc[-len(ict_feats):].reset_index(drop=True)
    merged = pd.concat([merged, ict_feats], axis=1)

    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.dropna(subset=FEATURES_FULL, inplace=True)

    return merged.reset_index(drop=True)


# ==========================================================================
# D) تدريب دوري + Ensemble (MLP + HistGradientBoosting)
# ==========================================================================

RETRAIN_INTERVAL_HOURS = 6   # أعد التدريب كل 6 ساعات بدل مرة واحدة فقط
MODEL_META_FILE = "xau_deep_model_meta.json"


def should_retrain(model_file, meta_getter):
    """
    meta_getter: دالة تُرجع آخر وقت تدريب مخزّن (مثلاً من settings في SQLite
    عبر load_setting('last_train_time', '')).
    """
    import os as _os

    if not _os.path.exists(model_file):
        return True

    last_train_str = meta_getter()
    if not last_train_str:
        return True

    try:
        last_train = datetime.fromisoformat(last_train_str)
    except Exception:
        return True

    age_hours = (
        datetime.now(timezone.utc) - last_train
    ).total_seconds() / 3600.0

    return age_hours >= RETRAIN_INTERVAL_HOURS


def train_ensemble(X, y):
    """
    يدرّب نموذجين مختلفين (MLP + HistGradientBoosting) ويعيدهما مع scaler.
    التصويت لاحقاً بمتوسط الاحتمالات (soft voting) يقلل ضجيج نموذج واحد
    ويرفع صرامة الإشارة قبل حدوثها فعلياً.
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate_init=0.001,
        max_iter=1500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        random_state=42,
    )
    mlp.fit(X_scaled, y)

    hgb = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=0.1,
        random_state=42,
    )
    hgb.fit(X, y)  # HGB لا يحتاج تقييس

    return mlp, hgb, scaler


def ensemble_predict_proba(mlp, hgb, scaler, feature_row):
    """
    feature_row: 2D array (1, n_features) بنفس ترتيب FEATURES_FULL
    """
    x_scaled = scaler.transform(feature_row)

    p_mlp = mlp.predict_proba(x_scaled)[0]
    p_hgb = hgb.predict_proba(feature_row)[0]

    classes = mlp.classes_  # يفترض نفس الترتيب في كلا النموذجين لأن y ثنائي 0/1
    avg_proba = (p_mlp + p_hgb) / 2.0

    best_idx = int(np.argmax(avg_proba))
    pred = int(classes[best_idx])
    conf = float(avg_proba[best_idx] * 100)

    # درجة اتفاق النموذجين — عدم الاتفاق = ثقة أقل فعلياً
    agreement = 1.0 - abs(p_mlp[best_idx] - p_hgb[best_idx])
    conf = conf * (0.7 + 0.3 * agreement)  # عقوبة خفيفة عند الخلاف

    return pred, conf


# ==========================================================================
# C) دمج ICT في قرار الدخول داخل ai_scanner
# ==========================================================================
#
# ضع هذا المنطق داخل ai_scanner() بعد حساب (pred, ai_conf) من الشبكة،
# وقبل استدعاء get_experience_adjustment، بدلاً من تجاهل ict_data تماماً.

ICT_GATE_SNIPPET = '''
    # --- بعد الحصول على pred, ai_conf من النموذج ---

    direction_from_ai = "BUY" if pred == 1 else "SELL"

    ict_snapshot = run_ict_engine(
        df_live_processed,
        swing_lookback=swing_lookback,
        ob_mult=ob_displacement_mult,
    )

    if ict_snapshot is None:
        result["status"] = "لا توجد صفقة: تحليل ICT غير كافٍ للتأكيد."
        return result["status"], result

    ict_bias = ict_snapshot["bias"]  # BULLISH / BEARISH / NEUTRAL

    ict_direction_matches = (
        (direction_from_ai == "BUY" and ict_bias == "BULLISH") or
        (direction_from_ai == "SELL" and ict_bias == "BEARISH")
    )

    if not ict_direction_matches:
        result["status"] = (
            f"🚫 لا توجد صفقة: AI يقترح {direction_from_ai} لكن "
            f"هيكل ICT الحالي {ict_bias} — تم رفض الإشارة لعدم توافق السيولة/البنية."
        )
        return result["status"], result

    # منطقة تأكيد إضافية: تفضيل الدخول عند OB متوافق أو داخل OTE
    confluence_bonus = 0.0
    if ict_snapshot["manipulation"]:
        confluence_bonus += 5.0
    if ict_snapshot["ote"] and ict_snapshot["ote"]["inside"]:
        confluence_bonus += 5.0
    if direction_from_ai == "BUY" and ict_snapshot["bull_ob"]:
        confluence_bonus += 5.0
    if direction_from_ai == "SELL" and ict_snapshot["bear_ob"]:
        confluence_bonus += 5.0

    # دمج ثقة ICT الرقمية (confidence) مع ثقة الشبكة، بدل تجاهل ICT كلياً
    ai_conf = (ai_conf * 0.6) + (ict_snapshot["confidence"] * 0.4) + confluence_bonus
    ai_conf = min(ai_conf, 100.0)
'''
