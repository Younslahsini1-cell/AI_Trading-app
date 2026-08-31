import streamlit as st
import streamlit.components.v1 as components

# ========== كود HTML مع تصميم مبسط ==========
HTML = """
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
            --bg-primary: #f5f7fa;
            --bg-card: #ffffff;
            --border-color: #e0e0e0;
            --primary-blue: #2563eb;
            --primary-blue-hover: #1d4ed8;
            --green: #16a34a;
            --red: #dc2626;
            --text-primary: #1f2937;
            --text-secondary: #6b7280;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.1);
            --radius: 12px;
            --transition: all 0.2s ease;
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
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }

        /* ===== شريط التنقل ===== */
        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 40px;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .logo-icon {
            width: 36px;
            height: 36px;
            background: var(--primary-blue);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            color: white;
        }

        .nav-links {
            display: flex;
            gap: 25px;
            align-items: center;
        }

        .nav-links a {
            color: var(--text-secondary);
            text-decoration: none;
            font-weight: 500;
            transition: var(--transition);
        }

        .nav-links a:hover {
            color: var(--primary-blue);
        }

        .btn {
            padding: 8px 20px;
            border-radius: 25px;
            border: none;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.95rem;
            transition: var(--transition);
            text-decoration: none;
            display: inline-block;
            font-family: 'Cairo', sans-serif;
        }

        .btn-primary {
            background: var(--primary-blue);
            color: white;
        }
        .btn-primary:hover {
            background: var(--primary-blue-hover);
            transform: translateY(-1px);
        }

        /* ===== Hero ===== */
        .hero {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            align-items: center;
            padding: 40px 0;
            margin-bottom: 40px;
        }

        .hero-content h1 {
            font-size: 2.8rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 15px;
            color: var(--text-primary);
        }

        .hero-content p {
            font-size: 1.1rem;
            color: var(--text-secondary);
            margin-bottom: 25px;
        }

        .hero-buttons {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .btn-secondary {
            background: transparent;
            border: 2px solid var(--primary-blue);
            color: var(--primary-blue);
        }
        .btn-secondary:hover {
            background: rgba(37, 99, 235, 0.05);
            transform: translateY(-1px);
        }

        .hero-visual {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .hero-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 25px;
            width: 100%;
            max-width: 450px;
            box-shadow: var(--shadow-md);
        }

        .hero-card .price-display {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--primary-blue);
            margin: 10px 0;
        }

        .hero-card .change-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            color: var(--green);
            font-weight: 600;
            font-size: 0.95rem;
        }

        /* ===== لوحة التحكم ===== */
        .dashboard {
            margin: 60px 0;
        }

        .section-title {
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 30px;
            color: var(--text-primary);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 20px;
            box-shadow: var(--shadow-sm);
            transition: var(--transition);
        }

        .stat-card:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }

        .stat-card .label {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 8px;
        }

        .stat-card .value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .stat-card .value.positive {
            color: var(--green);
        }

        .stat-card .value.negative {
            color: var(--red);
        }

        /* ===== الرسم البياني ===== */
        .chart-container {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 25px;
            margin: 40px 0;
            box-shadow: var(--shadow-sm);
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
            font-size: 1.4rem;
            font-weight: 600;
        }

        .timeframe-buttons {
            display: flex;
            gap: 8px;
        }

        .timeframe-btn {
            padding: 6px 14px;
            border-radius: 15px;
            border: 1px solid var(--border-color);
            background: white;
            color: var(--text-secondary);
            cursor: pointer;
            transition: var(--transition);
            font-weight: 500;
            font-size: 0.85rem;
        }

        .timeframe-btn.active {
            background: var(--primary-blue);
            color: white;
            border-color: var(--primary-blue);
        }

        .timeframe-btn:hover:not(.active) {
            background: #f0f0f0;
        }

        #priceChart {
            width: 100% !important;
            height: 380px !important;
        }

        /* ===== محاكي التداول ===== */
        .trading-simulator {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin: 40px 0;
        }

        .trade-panel, .ai-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 25px;
            box-shadow: var(--shadow-sm);
        }

        .trade-panel h3, .ai-panel h3 {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 20px;
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
            padding: 10px 14px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background: #fafafa;
            color: var(--text-primary);
            font-weight: 500;
            font-size: 0.95rem;
            transition: var(--transition);
            font-family: 'Cairo', sans-serif;
        }

        .trade-input:focus {
            outline: none;
            border-color: var(--primary-blue);
            background: white;
        }

        .trade-actions {
            display: flex;
            gap: 10px;
        }

        .btn-buy {
            background: var(--green);
            color: white;
            flex: 1;
        }
        .btn-buy:hover {
            background: #15803d;
            transform: translateY(-1px);
        }

        .btn-sell {
            background: var(--red);
            color: white;
            flex: 1;
        }
        .btn-sell:hover {
            background: #b91c1c;
            transform: translateY(-1px);
        }

        .trade-history {
            margin-top: 20px;
            max-height: 200px;
            overflow-y: auto;
            padding-right: 5px;
        }

        .trade-history::-webkit-scrollbar {
            width: 5px;
        }
        .trade-history::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        .trade-history::-webkit-scrollbar-thumb {
            background: #ccc;
            border-radius: 10px;
        }

        .trade-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #f0f0f0;
            font-size: 0.9rem;
        }
        .trade-item.buy { color: var(--green); }
        .trade-item.sell { color: var(--red); }

        /* ===== توصيات الذكاء الاصطناعي ===== */
        .ai-recommendation {
            background: #f8fafc;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }

        .ai-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            background: var(--primary-blue);
            color: white;
            margin-bottom: 10px;
        }

        .confidence-meter {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 12px;
        }

        .confidence-bar {
            flex: 1;
            height: 6px;
            background: #e5e7eb;
            border-radius: 10px;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            background: var(--primary-blue);
            border-radius: 10px;
            transition: width 0.5s ease;
        }

        .market-analysis {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
            gap: 12px;
            margin-top: 20px;
        }

        .analysis-item {
            background: #f9fafb;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }

        .analysis-item .indicator-name {
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-bottom: 4px;
        }

        .analysis-item .indicator-value {
            font-size: 1.1rem;
            font-weight: 600;
        }
        .analysis-item .indicator-value.up { color: var(--green); }
        .analysis-item .indicator-value.down { color: var(--red); }
        .analysis-item .indicator-value.neutral { color: var(--text-secondary); }

        /* ===== تذييل ===== */
        footer {
            text-align: center;
            padding: 30px 0;
            color: var(--text-secondary);
            border-top: 1px solid var(--border-color);
            margin-top: 60px;
            font-size: 0.9rem;
        }
        footer a {
            color: var(--primary-blue);
            text-decoration: none;
        }

        /* ===== استجابة ===== */
        @media (max-width: 768px) {
            .hero {
                grid-template-columns: 1fr;
                text-align: center;
            }
            .hero-content h1 { font-size: 2.2rem; }
            .hero-buttons { justify-content: center; }
            .trading-simulator { grid-template-columns: 1fr; }
            .nav-links { display: none; }
            .chart-header { flex-direction: column; align-items: flex-start; }
            .timeframe-buttons { width: 100%; justify-content: space-between; }
            .stats-grid { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- شريط التنقل -->
        <nav>
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

        <!-- Hero -->
        <section class="hero">
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

        <!-- لوحة التحكم -->
        <section class="dashboard" id="dashboard">
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

        <!-- الرسم البياني -->
        <section class="chart-container" id="chart">
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

        <!-- التداول والذكاء الاصطناعي -->
        <section class="trading-simulator" id="trading">
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
                    السعر الحالي: <span id="currentTradePrice" style="color: var(--primary-blue); font-weight: 700;">$65,432.10</span>
                </div>
                <div class="trade-history" id="tradeHistory">
                    <div style="text-align: center; color: var(--text-secondary); padding: 20px;">لا توجد صفقات بعد</div>
                </div>
            </div>

            <div class="ai-panel">
                <h3>🧠 توصيات الذكاء الاصطناعي</h3>
                <div class="ai-recommendation" id="aiRecommendation">
                    <span class="ai-badge">تحليل فوري</span>
                    <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;" id="recommendationText">
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

        <footer>
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
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 6,
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
                            color: '#1f2937',
                            font: { family: 'Cairo', weight: '600' }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(255,255,255,0.95)',
                        titleColor: '#1f2937',
                        bodyColor: '#6b7280',
                        borderColor: '#e0e0e0',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: { color: '#f0f0f0' },
                        ticks: { color: '#6b7280', font: { family: 'Cairo' } }
                    },
                    y: {
                        display: true,
                        grid: { color: '#f0f0f0' },
                        ticks: { color: '#6b7280', font: { family: 'Cairo' } }
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
                    borderColor: '#2563eb',
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
            changeElement.style.color = changePercent >= 0 ? '#16a34a' : '#dc2626';
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

# ========== إعداد صفحة Streamlit ==========
st.set_page_config(
    page_title="أوراكل تريد | تداول بالذكاء الاصطناعي",
    page_icon="⚡",
    layout="wide",
)

components.html(HTML, height=1200, scrolling=True)
