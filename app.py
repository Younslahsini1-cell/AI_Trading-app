from flask import Flask

app = Flask(__name__)

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>أوراكل تريد | تداول بالذكاء الاصطناعي</title>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #111827;
            --glass-bg: rgba(17, 25, 40, 0.7);
            --glass-border: rgba(255, 255, 255, 0.1);
            --neon-blue: #00d4ff;
            --neon-purple: #a855f7;
            --neon-green: #10b981;
            --neon-red: #ef4444;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --shadow-glow: 0 0 30px rgba(0, 212, 255, 0.3);
            --radius: 20px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Cairo', sans-serif;
        }
        body {
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
            background-image: 
                radial-gradient(ellipse at 20% 20%, rgba(168, 85, 247, 0.15) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 80%, rgba(0, 212, 255, 0.1) 0%, transparent 60%);
        }
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: 0;
        }
        .glow-orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(80px);
            pointer-events: none;
            z-index: 0;
            animation: float 8s infinite ease-in-out;
        }
        .glow-orb.one {
            width: 300px;
            height: 300px;
            background: rgba(168, 85, 247, 0.4);
            top: -100px;
            right: -100px;
        }
        .glow-orb.two {
            width: 250px;
            height: 250px;
            background: rgba(0, 212, 255, 0.3);
            bottom: -80px;
            left: -80px;
            animation-delay: -4s;
        }
        @keyframes float {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(30px, -30px) scale(1.1); }
            66% { transform: translate(-20px, 20px) scale(0.9); }
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 20px;
            position: relative;
            z-index: 1;
        }
        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            position: relative;
            z-index: 10;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.8rem;
            font-weight: 900;
            background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -1px;
        }
        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            -webkit-text-fill-color: white;
            color: white;
            box-shadow: var(--shadow-glow);
            animation: pulse-glow 2s infinite;
        }
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.4); }
            50% { box-shadow: 0 0 40px rgba(0, 212, 255, 0.8); }
        }
        .nav-links {
            display: flex;
            gap: 30px;
            align-items: center;
        }
        .nav-links a {
            color: var(--text-secondary);
            text-decoration: none;
            font-weight: 500;
            transition: var(--transition);
            position: relative;
            font-size: 1rem;
        }
        .nav-links a:hover {
            color: var(--neon-blue);
        }
        .nav-links a::after {
            content: '';
            position: absolute;
            bottom: -5px;
            right: 0;
            width: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--neon-blue), var(--neon-purple));
            transition: var(--transition);
        }
        .nav-links a:hover::after {
            width: 100%;
        }
        .btn {
            padding: 10px 24px;
            border-radius: 30px;
            border: none;
            cursor: pointer;
            font-weight: 700;
            font-size: 1rem;
            transition: var(--transition);
            text-decoration: none;
            display: inline-block;
            font-family: 'Cairo', sans-serif;
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple));
            color: white;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
        }
        .btn-primary:hover {
            box-shadow: 0 0 40px rgba(0, 212, 255, 0.8);
            transform: translateY(-2px);
        }
        .hero {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            align-items: center;
            min-height: 80vh;
            padding: 40px 0;
        }
        .hero-content h1 {
            font-size: 3.5rem;
            font-weight: 900;
            line-height: 1.2;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #ffffff 0%, var(--neon-blue) 50%, var(--neon-purple) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-content p {
            font-size: 1.2rem;
            color: var(--text-secondary);
            margin-bottom: 30px;
            line-height: 1.8;
        }
        .hero-buttons {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .btn-secondary {
            background: transparent;
            border: 2px solid var(--neon-blue);
            color: var(--neon-blue);
        }
        .btn-secondary:hover {
            background: rgba(0, 212, 255, 0.1);
            box-shadow: var(--shadow-glow);
            transform: translateY(-2px);
        }
        .hero-visual {
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .hero-card {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(20px);
            border-radius: var(--radius);
            padding: 30px;
            width: 100%;
            max-width: 500px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            animation: float-card 5s infinite ease-in-out;
        }
        @keyframes float-card {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-15px); }
        }
        .hero-card .price-display {
            font-size: 2.5rem;
            font-weight: 900;
            color: var(--neon-blue);
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
            margin: 10px 0;
        }
        .hero-card .change-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--neon-green);
            font-weight: 700;
        }
        .dashboard {
            margin-top: 80px;
        }
        .section-title {
            font-size: 2.5rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 50px;
            background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 25px;
            margin-bottom: 50px;
        }
        .stat-card {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(20px);
            border-radius: var(--radius);
            padding: 25px;
            transition: var(--transition);
            position: relative;
            overflow: hidden;
        }
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--neon-blue), var(--neon-purple));
            opacity: 0;
            transition: var(--transition);
        }
        .stat-card:hover::before {
            opacity: 1;
        }
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(0, 212, 255, 0.2);
        }
        .stat-card .label {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 10px;
        }
        .stat-card .value {
            font-size: 2rem;
            font-weight: 900;
            color: var(--text-primary);
        }
        .stat-card .value.positive {
            color: var(--neon-green);
            text-shadow: 0 0 15px rgba(16, 185, 129, 0.5);
        }
        .stat-card .value.negative {
            color: var(--neon-red);
            text-shadow: 0 0 15px rgba(239, 68, 68, 0.5);
        }
        .chart-container {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(20px);
            border-radius: var(--radius);
            padding: 30px;
            margin-bottom: 50px;
            position: relative;
        }
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 15px;
        }
        .chart-header h3 {
            font-size: 1.5rem;
            font-weight: 700;
        }
        .timeframe-buttons {
            display: flex;
            gap: 10px;
        }
        .timeframe-btn {
            padding: 8px 16px;
            border-radius: 20px;
            border: 1px solid var(--glass-border);
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            transition: var(--transition);
            font-weight: 600;
            font-size: 0.9rem;
        }
        .timeframe-btn.active {
            background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple));
            color: white;
            border-color: transparent;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
        }
        .timeframe-btn:hover:not(.active) {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
        }
        #priceChart {
            width: 100% !important;
            height: 400px !important;
        }
        .trading-simulator {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 50px;
        }
        .trade-panel, .ai-panel {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(20px);
            border-radius: var(--radius);
            padding: 30px;
        }
        .trade-panel h3, .ai-panel h3 {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .trade-form {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .trade-input-group {
            display: flex;
            gap: 10px;
        }
        .trade-input {
            flex: 1;
            padding: 12px 16px;
            border-radius: 12px;
            border: 1px solid var(--glass-border);
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
            font-weight: 600;
            font-size: 1rem;
            transition: var(--transition);
            font-family: 'Cairo', sans-serif;
        }
        .trade-input:focus {
            outline: none;
            border-color: var(--neon-blue);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
        }
        .trade-actions {
            display: flex;
            gap: 10px;
        }
        .btn-buy {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            flex: 1;
        }
        .btn-buy:hover {
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.5);
            transform: translateY(-2px);
        }
        .btn-sell {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
            flex: 1;
        }
        .btn-sell:hover {
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.5);
            transform: translateY(-2px);
        }
        .trade-history {
            margin-top: 20px;
            max-height: 200px;
            overflow-y: auto;
            padding-right: 10px;
        }
        .trade-history::-webkit-scrollbar {
            width: 5px;
        }
        .trade-history::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }
        .trade-history::-webkit-scrollbar-thumb {
            background: var(--neon-blue);
            border-radius: 10px;
        }
        .trade-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.9rem;
        }
        .trade-item.buy {
            color: var(--neon-green);
        }
        .trade-item.sell {
            color: var(--neon-red);
        }
        .ai-recommendation {
            background: rgba(0, 212, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        .ai-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple));
            color: white;
            margin-bottom: 10px;
        }
        .confidence-meter {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 15px;
        }
        .confidence-bar {
            flex: 1;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--neon-blue), var(--neon-purple));
            border-radius: 10px;
            transition: width 1s ease;
        }
        .market-analysis {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .analysis-item {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            transition: var(--transition);
        }
        .analysis-item:hover {
            background: rgba(255, 255, 255, 0.08);
        }
        .analysis-item .indicator-name {
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-bottom: 5px;
        }
        .analysis-item .indicator-value {
            font-size: 1.2rem;
            font-weight: 700;
        }
        .analysis-item .indicator-value.up {
            color: var(--neon-green);
        }
        .analysis-item .indicator-value.down {
            color: var(--neon-red);
        }
        .analysis-item .indicator-value.neutral {
            color: var(--text-secondary);
        }
        footer {
            text-align: center;
            padding: 40px 0;
            color: var(--text-secondary);
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            margin-top: 80px;
        }
        footer a {
            color: var(--neon-blue);
            text-decoration: none;
        }
        @media (max-width: 768px) {
            .hero {
                grid-template-columns: 1fr;
                text-align: center;
                min-height: auto;
                padding: 40px 0;
            }
            .hero-content h1 {
                font-size: 2.5rem;
            }
            .hero-buttons {
                justify-content: center;
            }
            .trading-simulator {
                grid-template-columns: 1fr;
            }
            .nav-links {
                display: none;
            }
            .chart-header {
                flex-direction: column;
                align-items: flex-start;
            }
            .timeframe-buttons {
                width: 100%;
                justify-content: space-between;
            }
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .fade-in-up {
            animation: fadeInUp 0.8s ease forwards;
        }
        .delay-1 { animation-delay: 0.2s; }
        .delay-2 { animation-delay: 0.4s; }
        .delay-3 { animation-delay: 0.6s; }
    </style>
</head>
<body>
    <div class="glow-orb one"></div>
    <div class="glow-orb two"></div>

    <div class="container">
        <nav class="fade-in-up">
            <div class="logo">
                <div class="logo-icon">⚡</div>
                أوراكل تريد
            </div>
            <div class="nav-links">
                <a href="#dashboard">لوحة التحكم</a>
                <a href="#chart">الرسوم البيانية</a>
                <a href="#trading">التداول</a>
                <a href="#ai">الذكاء الاصطناعي</a>
            </div>
            <a href="#" class="btn btn-primary">ابدأ الآن</a>
        </nav>

        <section class="hero fade-in-up delay-1">
            <div class="hero-content">
                <h1>تداول بذكاء<br>مع قوة الذكاء الاصطناعي</h1>
                <p>منصة تداول متطورة تستخدم أحدث خوارزميات التعلم الآلي لتحليل الأسواق وتقديم توصيات دقيقة في الوقت الفعلي.</p>
                <div class="hero-buttons">
                    <a href="#" class="btn btn-primary">جرب مجانًا</a>
                    <a href="#" class="btn btn-secondary">شاهد العرض</a>
                </div>
            </div>
            <div class="hero-visual">
                <div class="hero-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: var(--text-secondary)">BTC/USDT</span>
                        <span class="ai-badge">AI نشط</span>
                    </div>
                    <div class="price-display" id="heroPrice">$65,432.10</div>
                    <div class="change-indicator">
                        <span>▲</span>
                        <span id="heroChange">+2.34%</span>
                    </div>
                    <canvas id="miniChart" height="120"></canvas>
                </div>
            </div>
        </section>

        <section class="dashboard fade-in-up delay-2" id="dashboard">
            <h2 class="section-title">لوحة التحكم الرئيسية</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="label">رصيد المحفظة</div>
                    <div class="value">$128,450.00</div>
                </div>
                <div class="stat-card">
                    <div class="label">أرباح اليوم</div>
                    <div class="value positive">+$3,245.80</div>
                </div>
                <div class="stat-card">
                    <div class="label">صفقات ناجحة</div>
                    <div class="value positive">87.5%</div>
                </div>
                <div class="stat-card">
                    <div class="label">إجمالي الأصول</div>
                    <div class="value">$245,780.00</div>
                </div>
                <div class="stat-card">
                    <div class="label">توصيات الذكاء الاصطناعي</div>
                    <div class="value positive">+18.2%</div>
                </div>
                <div class="stat-card">
                    <div class="label">مؤشر الثقة</div>
                    <div class="value">92%</div>
                </div>
            </div>
        </section>

        <section class="chart-container fade-in-up" id="chart">
            <div class="chart-header">
                <h3>📊 الرسم البياني المباشر</h3>
                <div class="timeframe-buttons">
                    <button class="timeframe-btn active" data-timeframe="1m">1 دقيقة</button>
                    <button class="timeframe-btn" data-timeframe="5m">5 دقائق</button>
                    <button class="timeframe-btn" data-timeframe="1h">ساعة</button>
                    <button class="timeframe-btn" data-timeframe="1d">يوم</button>
                </div>
            </div>
            <canvas id="priceChart"></canvas>
        </section>

        <section class="trading-simulator fade-in-up delay-3" id="trading">
            <div class="trade-panel" id="ai">
                <h3>💰 محاكي التداول</h3>
                <div class="trade-form">
                    <div class="trade-input-group">
                        <input type="number" class="trade-input" id="tradeAmount" placeholder="الكمية (USDT)" value="100">
                        <input type="text" class="trade-input" id="tradeAsset" placeholder="الأصل (مثال: BTC)" value="BTC">
                    </div>
                    <div class="trade-actions">
                        <button class="btn btn-buy" onclick="executeTrade('buy')">شراء ▲</button>
                        <button class="btn btn-sell" onclick="executeTrade('sell')">بيع ▼</button>
                    </div>
                </div>
                <div style="margin-top: 20px; color: var(--text-secondary); font-size: 0.9rem;">
                    السعر الحالي: <span id="currentTradePrice" style="color: var(--neon-blue); font-weight: 700;">$65,432.10</span>
                </div>
                <div class="trade-history" id="tradeHistory">
                    <div style="text-align: center; color: var(--text-secondary); padding: 20px;">لا توجد صفقات بعد</div>
                </div>
            </div>

            <div class="ai-panel">
                <h3>🧠 توصيات الذكاء الاصطناعي</h3>
                <div class="ai-recommendation" id="aiRecommendation">
                    <span class="ai-badge">تحليل فوري</span>
                    <div style="font-weight: 700; font-size: 1.2rem; margin-bottom: 10px;" id="recommendationText">
                        اتجاه صاعد قوي - فرصة شراء
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem;" id="recommendationDetail">
                        نموذج الاختراق الصاعد مع حجم تداول مرتفع. احتمالية استمرار الاتجاه 78%.
                    </div>
                    <div class="confidence-meter">
                        <span style="font-size: 0.8rem; color: var(--text-secondary);">الثقة:</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="confidenceFill" style="width: 78%"></div>
                        </div>
                        <span style="font-size: 0.8rem; font-weight: 700;" id="confidencePercent">78%</span>
                    </div>
                </div>
                <div class="market-analysis">
                    <div class="analysis-item">
                        <div class="indicator-name">RSI</div>
                        <div class="indicator-value up" id="rsiValue">68.5</div>
                    </div>
                    <div class="analysis-item">
                        <div class="indicator-name">MACD</div>
                        <div class="indicator-value up" id="macdValue">إيجابي</div>
                    </div>
                    <div class="analysis-item">
                        <div class="indicator-name">المتوسطات</div>
                        <div class="indicator-value up" id="maValue">صاعدة</div>
                    </div>
                    <div class="analysis-item">
                        <div class="indicator-name">التقلب</div>
                        <div class="indicator-value neutral" id="volatilityValue">متوسط</div>
                    </div>
                </div>
            </div>
        </section>

        <footer class="fade-in-up">
            <p>© 2024 أوراكل تريد - جميع الحقوق محفوظة | مبني بتقنية <a href="#">الذكاء الاصطناعي</a></p>
        </footer>
    </div>

    <script>
        const chartData = {
            labels: [],
            prices: [],
            volumes: []
        };

        function generateInitialData() {
            const now = new Date();
            for (let i = 59; i >= 0; i--) {
                const time = new Date(now - i * 60000);
                chartData.labels.push(time.toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' }));
                chartData.prices.push(65000 + Math.random() * 2000);
                chartData.volumes.push(Math.random() * 100);
            }
        }

        generateInitialData();

        const ctx = document.getElementById('priceChart').getContext('2d');
        const priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.labels,
                datasets: [{
                    label: 'السعر (USDT)',
                    data: chartData.prices,
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    borderWidth: 3,
                    pointRadius: 0,
                    pointHoverRadius: 8,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: '#f3f4f6',
                            font: { family: 'Cairo', weight: '700' }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(17, 24, 39, 0.9)',
                        titleColor: '#f3f4f6',
                        bodyColor: '#9ca3af',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', font: { family: 'Cairo' } }
                    },
                    y: {
                        display: true,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', font: { family: 'Cairo' } }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });

        const miniCtx = document.getElementById('miniChart').getContext('2d');
        const miniChart = new Chart(miniCtx, {
            type: 'line',
            data: {
                labels: chartData.labels.slice(-20),
                datasets: [{
                    data: chartData.prices.slice(-20),
                    borderColor: '#a855f7',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { display: false }
                }
            }
        });

        function updateData() {
            const now = new Date();
            const time = now.toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' });
            const lastPrice = chartData.prices[chartData.prices.length - 1];
            const change = (Math.random() - 0.5) * 300;
            const newPrice = Math.max(62000, Math.min(68000, lastPrice + change));

            chartData.labels.push(time);
            chartData.prices.push(newPrice);

            if (chartData.labels.length > 100) {
                chartData.labels.shift();
                chartData.prices.shift();
            }

            priceChart.update();
            
            miniChart.data.labels = chartData.labels.slice(-20);
            miniChart.data.datasets[0].data = chartData.prices.slice(-20);
            miniChart.update();

            document.getElementById('heroPrice').textContent = '$' + newPrice.toFixed(2).replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
            document.getElementById('currentTradePrice').textContent = '$' + newPrice.toFixed(2).replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');

            const prevPrice = chartData.prices[chartData.prices.length - 2] || newPrice;
            const changePercent = ((newPrice - prevPrice) / prevPrice) * 100;
            const changeElement = document.getElementById('heroChange');
            changeElement.textContent = (changePercent >= 0 ? '+' : '') + changePercent.toFixed(2) + '%';
            changeElement.style.color = changePercent >= 0 ? '#10b981' : '#ef4444';
        }

        setInterval(updateData, 3000);

        let tradeCount = 0;
        let walletBalance = 128450.00;

        function executeTrade(type) {
            const amount = parseFloat(document.getElementById('tradeAmount').value);
            const asset = document.getElementById('tradeAsset').value || 'BTC';
            const currentPrice = chartData.prices[chartData.prices.length - 1];

            if (!amount || amount <= 0) {
                alert('الرجاء إدخال كمية صحيحة');
                return;
            }

            tradeCount++;
            const now = new Date();
            const timeStr = now.toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

            const historyDiv = document.getElementById('tradeHistory');
            if (tradeCount === 1) {
                historyDiv.innerHTML = '';
            }

            const tradeItem = document.createElement('div');
            tradeItem.className = `trade-item ${type}`;
            tradeItem.innerHTML = `
                <span>${type === 'buy' ? 'شراء' : 'بيع'} ${asset} - ${timeStr}</span>
                <span>${type === 'buy' ? '+' : '-'}$${amount.toFixed(2)}</span>
            `;
            historyDiv.prepend(tradeItem);

            if (type === 'buy') {
                walletBalance -= amount;
            } else {
                walletBalance += amount;
            }

            const balanceElements = document.querySelectorAll('.stat-card .value');
            if (balanceElements.length > 0) {
                balanceElements[0].textContent = '$' + walletBalance.toFixed(2).replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
            }

            updateAIRecommendation();
        }

        function updateAIRecommendation() {
            const recommendations = [
                { text: 'اتجاه صاعد قوي - فرصة شراء', detail: 'نموذج الاختراق الصاعد مع حجم تداول مرتفع. احتمالية استمرار الاتجاه 78%.', confidence: 78 },
                { text: 'تصحيح مؤقت - انتظر التأكيد', detail: 'مؤشر RSI في منطقة تشبع شرائي. توقع تراجع طفيف قبل استئناف الصعود.', confidence: 65 },
                { text: 'اتجاه هابط - تجنب الشراء', detail: 'كسر مستويات الدعم الرئيسية. ضغط بيعي متزايد من المؤسسات.', confidence: 82 },
                { text: 'تذبذب عرضي - تداول بنطاق', detail: 'السوق في حالة توازن. انتظر اختراق واضح للنطاق السعري.', confidence: 55 },
                { text: 'فرصة ذهبية - شراء قوي', detail: 'تشكل نموذج قاع مزدوج مع تباين إيجابي في مؤشر MACD.', confidence: 88 }
            ];

            const rec = recommendations[Math.floor(Math.random() * recommendations.length)];
            document.getElementById('recommendationText').textContent = rec.text;
            document.getElementById('recommendationDetail').textContent = rec.detail;
            document.getElementById('confidenceFill').style.width = rec.confidence + '%';
            document.getElementById('confidencePercent').textContent = rec.confidence + '%';

            const rsi = (Math.random() * 60 + 20).toFixed(1);
            document.getElementById('rsiValue').textContent = rsi;
            document.getElementById('rsiValue').className = 'indicator-value ' + (rsi > 70 ? 'down' : rsi < 30 ? 'up' : 'neutral');

            const macdPositive = Math.random() > 0.4;
            document.getElementById('macdValue').textContent = macdPositive ? 'إيجابي' : 'سلبي';
            document.getElementById('macdValue').className = 'indicator-value ' + (macdPositive ? 'up' : 'down');

            const maUp = Math.random() > 0.35;
            document.getElementById('maValue').textContent = maUp ? 'صاعدة' : 'هابطة';
            document.getElementById('maValue').className = 'indicator-value ' + (maUp ? 'up' : 'down');

            const volatilityOptions = ['منخفض', 'متوسط', 'مرتفع'];
            const vol = volatilityOptions[Math.floor(Math.random() * 3)];
            document.getElementById('volatilityValue').textContent = vol;
            document.getElementById('volatilityValue').className = 'indicator-value ' + (vol === 'منخفض' ? 'up' : vol === 'متوسط' ? 'neutral' : 'down');
        }

        updateAIRecommendation();

        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.timeframe-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
            });
        });

        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_CONTENT

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
