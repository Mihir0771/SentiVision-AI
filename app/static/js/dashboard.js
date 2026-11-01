/**
 * SentiVision AI — Dashboard Analytics & Charts
 */

(function () {
    'use strict';

    let sentimentChart = null;
    let timelineChart = null;
    let emotionRadarChart = null;

    // Palette tokens
    const COLORS = {
        positive: '#00c853',
        negative: '#ff1744',
        neutral: '#ffc400',
        accentStart: '#667eea',
        accentEnd: '#764ba2',
        borderDark: 'rgba(255, 255, 255, 0.1)',
        borderLight: 'rgba(0, 0, 0, 0.1)',
        textDark: '#9fa8da',
        textLight: '#4a5568',
    };

    function isDarkTheme() {
        return (document.documentElement.getAttribute('data-bs-theme') || 'dark') === 'dark';
    }

    function initStatCounters() {
        const counters = document.querySelectorAll('.stat-number[data-target]');
        counters.forEach((counter) => {
            const target = parseFloat(counter.getAttribute('data-target')) || 0;
            const isPercent = counter.getAttribute('data-percent') === 'true';
            let current = 0;
            const duration = 1200;
            const stepTime = 20;
            const increment = target / (duration / stepTime);

            const timer = setInterval(() => {
                current += increment;
                if ((increment >= 0 && current >= target) || (increment < 0 && current <= target)) {
                    current = target;
                    clearInterval(timer);
                }
                counter.textContent = isPercent
                    ? current.toFixed(1) + '%'
                    : Math.floor(current).toLocaleString();
            }, stepTime);
        });
    }

    function initSentimentDoughnut() {
        const ctx = document.getElementById('sentimentDoughnut');
        if (!ctx) return;

        const pos = parseInt(ctx.getAttribute('data-positive') || '0', 10);
        const neg = parseInt(ctx.getAttribute('data-negative') || '0', 10);
        const neu = parseInt(ctx.getAttribute('data-neutral') || '0', 10);

        const total = pos + neg + neu;
        const dataValues = total > 0 ? [pos, neg, neu] : [1, 1, 1];
        const bgColors = total > 0
            ? [COLORS.positive, COLORS.negative, COLORS.neutral]
            : ['rgba(0,200,83,0.3)', 'rgba(255,23,68,0.3)', 'rgba(255,196,0,0.3)'];

        if (sentimentChart) sentimentChart.destroy();

        sentimentChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    data: dataValues,
                    backgroundColor: bgColors,
                    borderWidth: 2,
                    borderColor: isDarkTheme() ? '#0a0e27' : '#ffffff',
                    hoverOffset: 6,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: isDarkTheme() ? COLORS.textDark : COLORS.textLight,
                            padding: 18,
                            font: { family: 'inherit', size: 13 },
                        },
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                if (total === 0) return `${context.label}: 0 (No data)`;
                                const val = context.raw || 0;
                                const pct = ((val / total) * 100).toFixed(1);
                                return `${context.label}: ${val} (${pct}%)`;
                            },
                        },
                    },
                },
            },
        });
    }

    function initTimelineChart() {
        const ctx = document.getElementById('timelineChart');
        if (!ctx) return;

        const rawData = ctx.getAttribute('data-timeline');
        let timeline = { labels: [], positive: [], negative: [], neutral: [] };
        if (rawData) {
            try {
                timeline = JSON.parse(rawData);
            } catch (e) {
                console.warn('Could not parse timeline data');
            }
        }

        // Demo sample if timeline empty
        if (!timeline.labels || timeline.labels.length === 0) {
            timeline.labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
            timeline.positive = [12, 19, 15, 25, 22, 30, 28];
            timeline.negative = [5, 3, 7, 4, 8, 2, 5];
            timeline.neutral = [8, 11, 10, 14, 12, 9, 13];
        }

        if (timelineChart) timelineChart.destroy();

        const textColor = isDarkTheme() ? COLORS.textDark : COLORS.textLight;
        const gridColor = isDarkTheme() ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';

        timelineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeline.labels,
                datasets: [
                    {
                        label: 'Positive',
                        data: timeline.positive,
                        borderColor: COLORS.positive,
                        backgroundColor: 'rgba(0, 200, 83, 0.1)',
                        tension: 0.35,
                        fill: true,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                    },
                    {
                        label: 'Negative',
                        data: timeline.negative,
                        borderColor: COLORS.negative,
                        backgroundColor: 'rgba(255, 23, 68, 0.1)',
                        tension: 0.35,
                        fill: true,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                    },
                    {
                        label: 'Neutral',
                        data: timeline.neutral,
                        borderColor: COLORS.neutral,
                        backgroundColor: 'rgba(255, 196, 0, 0.1)',
                        tension: 0.35,
                        fill: true,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: gridColor },
                        ticks: { color: textColor },
                    },
                    y: {
                        beginAtZero: true,
                        grid: { color: gridColor },
                        ticks: { color: textColor, precision: 0 },
                    },
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: textColor, font: { size: 12 } },
                    },
                },
            },
        });
    }

    function initRadarChart() {
        const ctx = document.getElementById('emotionRadarChart');
        if (!ctx) return;

        const rawData = ctx.getAttribute('data-emotions');
        let emotions = { happy: 35, surprise: 20, neutral: 25, sad: 8, angry: 5, fear: 4, disgust: 3 };
        if (rawData) {
            try {
                emotions = JSON.parse(rawData);
            } catch (e) {}
        }

        if (emotionRadarChart) emotionRadarChart.destroy();

        const textColor = isDarkTheme() ? COLORS.textDark : COLORS.textLight;

        emotionRadarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Happy', 'Surprise', 'Neutral', 'Sad', 'Angry', 'Fear', 'Disgust'],
                datasets: [{
                    label: 'Emotion Distribution %',
                    data: [
                        emotions.happy || 0,
                        emotions.surprise || 0,
                        emotions.neutral || 0,
                        emotions.sad || 0,
                        emotions.angry || 0,
                        emotions.fear || 0,
                        emotions.disgust || 0,
                    ],
                    backgroundColor: 'rgba(102, 126, 234, 0.25)',
                    borderColor: '#667eea',
                    pointBackgroundColor: '#764ba2',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#764ba2',
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: isDarkTheme() ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)' },
                        grid: { color: isDarkTheme() ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)' },
                        pointLabels: { color: textColor, font: { size: 12 } },
                        ticks: { display: false, beginAtZero: true },
                    },
                },
                plugins: {
                    legend: { display: false },
                },
            },
        });
    }

    window.updateChartsTheme = function () {
        initSentimentDoughnut();
        initTimelineChart();
        initRadarChart();
    };

    document.addEventListener('DOMContentLoaded', () => {
        initStatCounters();
        initSentimentDoughnut();
        initTimelineChart();
        initRadarChart();
    });
})();
