"""
سكربت إيقاظ التطبيق على Streamlit Community Cloud.
يفتح جلسة متصفح حقيقية (WebSocket) على رابط التطبيق، وهذا هو الشيء
الوحيد الذي يُعيد عداد الـ 12 ساعة إلى الصفر — طلب HTTP عادي (GET)
لا يكفي لأن Streamlit يحتاج جلسة متصفح فعلية لاعتباره "زيارة".

يُشغَّل هذا السكربت تلقائياً كل 6 ساعات عبر GitHub Actions
(انظر keep_alive_workflow.yml).
"""

import os
import sys
import time

from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("STREAMLIT_APP_URL", "").strip()


def main():
    if not APP_URL:
        print("خطأ: لم يتم تعريف STREAMLIT_APP_URL في GitHub Secrets.")
        sys.exit(1)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print(f"زيارة التطبيق: {APP_URL}")
        try:
            page.goto(APP_URL, wait_until="networkidle", timeout=60000)
        except Exception as exc:
            print(f"تحذير أثناء التحميل الأول: {exc}")
        time.sleep(4)

        # إذا كان التطبيق نائماً، تظهر صفحة توقظه بزر "Yes, get this app back up!"
        try:
            wake_button = page.get_by_text("get this app back up", exact=False)
            if wake_button.is_visible(timeout=5000):
                print("التطبيق كان نائماً — جاري إيقاظه الآن...")
                wake_button.click()
                time.sleep(25)
                print("تم إرسال أمر الإيقاظ.")
            else:
                print("التطبيق كان نشطاً بالفعل — تمت زيارته لإبقائه مستيقظاً.")
        except Exception:
            print("التطبيق كان نشطاً بالفعل (لم يظهر زر الإيقاظ).")

        browser.close()


if __name__ == "__main__":
    main()
