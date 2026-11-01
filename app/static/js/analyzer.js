/**
 * SentiVision AI — Interactive Text Sentiment Analyzer Module
 */

(function () {
    'use strict';

    let ensembleBarChart = null;

    // Presets for quick testing
    const SAMPLE_TEXTS = {
        positive: "I absolutely love this product! The build quality is top-notch, battery life easily lasts two full days, and customer support was lightning fast and exceptionally helpful.",
        negative: "Terrible customer experience. The device arrived damaged, constantly crashed upon startup, and technical support refused to issue a refund. Complete waste of money.",
        neutral: "The package was delivered on Tuesday afternoon. It contains standard documentation, a charging cable, and the user manual in three languages.",
        social: "Finally tried the new update on my workstation! Super smooth rendering, clean UI animations, but export times are slightly higher than expected. Overall decent upgrade! #tech #review",
    };

    function initTextCounters() {
        const textarea = document.getElementById('analysisInput');
        const charCount = document.getElementById('charCount');
        const wordCount = document.getElementById('wordCount');
        if (!textarea) return;

        function updateCounts() {
            const val = textarea.value;
            if (charCount) charCount.textContent = val.length.toLocaleString();
            if (wordCount) {
                const words = val.trim() ? val.trim().split(/\s+/).length : 0;
                wordCount.textContent = words.toLocaleString();
            }
        }

        textarea.addEventListener('input', updateCounts);
        updateCounts();

        // Sample buttons
        document.querySelectorAll('.sample-btn').forEach((btn) => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const type = btn.getAttribute('data-sample');
                if (SAMPLE_TEXTS[type]) {
                    textarea.value = SAMPLE_TEXTS[type];
                    updateCounts();
                    textarea.focus();
                }
            });
        });
    }

    function initFormSubmit() {
        const form = document.getElementById('textAnalysisForm');
        const textarea = document.getElementById('analysisInput');
        const btnSubmit = document.getElementById('btnSubmitAnalysis');

        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = (textarea.value || '').trim();
            if (!text) {
                window.showToast('Please enter text to analyze.', 'warning');
                textarea.focus();
                return;
            }

            // Gather selected models
            const models = [];
            if (document.getElementById('modelTransformer')?.checked) models.push('transformer');
            if (document.getElementById('modelVader')?.checked) models.push('vader');
            if (document.getElementById('modelTextBlob')?.checked) models.push('textblob');

            setLoading(true);

            try {
                const res = await window.apiCall('/analyze/text', 'POST', {
                    text: text,
                    models: models.length > 0 ? models : null,
                });

                if (res.success && res.results) {
                    renderResults(res);
                    const resultsSection = document.getElementById('resultsSection');
                    if (resultsSection) {
                        resultsSection.classList.remove('d-none');
                        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                    window.showToast('Analysis completed successfully!', 'success');
                } else {
                    throw new Error(res.error || 'Failed to complete analysis.');
                }
            } catch (err) {
                window.showToast(err.message || 'An error occurred during analysis.', 'error');
            } finally {
                setLoading(false);
            }
        });

        // URL Analysis Form
        const urlForm = document.getElementById('urlAnalysisForm');
        const urlInput = document.getElementById('urlInput');
        if (urlForm) {
            urlForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const url = (urlInput?.value || '').trim();
                if (!url) {
                    window.showToast('Please enter a valid URL.', 'warning');
                    return;
                }

                setLoading(true, 'Scraping and analyzing URL content…');
                try {
                    const res = await window.apiCall('/analyze/url', 'POST', { url });
                    if (res.success && res.data) {
                        renderUrlResults(res.data);
                        const resultsSection = document.getElementById('resultsSection');
                        if (resultsSection) {
                            resultsSection.classList.remove('d-none');
                            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }
                        window.showToast('Webpage sentiment analyzed!', 'success');
                    } else {
                        throw new Error(res.error || 'Failed to scrape URL.');
                    }
                } catch (err) {
                    window.showToast(err.message || 'URL scraping failed.', 'error');
                } finally {
                    setLoading(false);
                }
            });
        }

        function setLoading(isLoading, msg = 'Running multi-model NLP analysis…') {
            if (btnSubmit) {
                btnSubmit.disabled = isLoading;
                btnSubmit.innerHTML = isLoading
                    ? '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Analyzing…'
                    : '<i class="fas fa-magic me-2"></i>Analyze Sentiment';
            }
            if (isLoading) {
                window.showLoading(msg);
            } else {
                window.hideLoading();
            }
        }
    }

    function renderResults(data) {
        const results = data.results;
        const sentiment = (results.overall_sentiment || 'neutral').toLowerCase();
        const confidence = results.overall_confidence || 0.0;
        const confidencePct = Math.round(confidence * 100);

        // Overall Sentiment Badge
        const sentimentBadge = document.getElementById('overallSentimentBadge');
        if (sentimentBadge) {
            sentimentBadge.className = `badge p-3 fs-5 text-uppercase sentiment-badge-${sentiment}`;
            const iconMap = {
                positive: 'fa-smile-beam',
                negative: 'fa-frown',
                neutral: 'fa-meh',
            };
            sentimentBadge.innerHTML = `<i class="fas ${iconMap[sentiment] || 'fa-brain'} me-2"></i>${sentiment} (${confidencePct}%)`;
        }

        // Confidence Meter
        const confBar = document.getElementById('confidenceMeterBar');
        const confText = document.getElementById('confidenceMeterText');
        if (confBar) {
            confBar.style.width = `${confidencePct}%`;
            confBar.className = `progress-bar bg-${sentiment === 'positive' ? 'success' : sentiment === 'negative' ? 'danger' : 'warning'}`;
        }
        if (confText) confText.textContent = `${confidencePct}%`;

        // Model Cards
        const models = results.models || {};
        renderModelCard('roberta', models.transformer);
        renderModelCard('vader', models.vader);
        renderModelCard('textblob', models.textblob);

        // Ensemble Scores Chart
        renderEnsembleChart(results.ensemble_scores || {});

        // Keywords
        const keywordsContainer = document.getElementById('keywordsContainer');
        if (keywordsContainer) {
            const keywords = data.keywords || [];
            if (keywords.length > 0) {
                keywordsContainer.innerHTML = keywords
                    .map((k) => `<span class="badge bg-secondary-subtle text-light border me-2 mb-2 p-2"><i class="fas fa-tag me-1 text-accent"></i>${k.keyword || k}</span>`)
                    .join('');
            } else {
                keywordsContainer.innerHTML = '<span class="text-muted fst-italic">No significant keywords extracted.</span>';
            }
        }

        // Entities
        const entitiesContainer = document.getElementById('entitiesContainer');
        if (entitiesContainer) {
            const entities = data.entities || [];
            if (entities.length > 0) {
                entitiesContainer.innerHTML = entities
                    .map((e) => `<span class="badge bg-dark border border-secondary me-2 mb-2 p-2"><strong class="text-accent">${e.type}:</strong> ${e.text}</span>`)
                    .join('');
            } else {
                entitiesContainer.innerHTML = '<span class="text-muted fst-italic">No named entities detected.</span>';
            }
        }

        // Text Statistics
        const stats = data.statistics || {};
        setTextContent('statWordCount', stats.word_count || 0);
        setTextContent('statSentenceCount', stats.sentence_count || 0);
        setTextContent('statReadingTime', stats.reading_time || '< 1 min');
        setTextContent('statDetectedLang', (data.language?.language || 'en').toUpperCase());

        // Update record link if saved
        const viewReportBtn = document.getElementById('btnViewFullReport');
        if (viewReportBtn && data.id) {
            viewReportBtn.href = `/analyze/result/${data.id}`;
            viewReportBtn.classList.remove('d-none');
        }
    }

    function renderModelCard(modelKey, modelData) {
        const el = document.getElementById(`modelCard_${modelKey}`);
        if (!el) return;

        if (!modelData || modelData.error) {
            el.innerHTML = '<div class="text-muted small">Model output unavailable.</div>';
            return;
        }

        let label = modelData.label || 'neutral';
        let detail = '';

        if (modelKey === 'roberta') {
            const conf = Math.round((modelData.confidence || 0) * 100);
            detail = `Confidence: <strong>${conf}%</strong>`;
        } else if (modelKey === 'vader') {
            detail = `Compound: <strong>${(modelData.compound || 0).toFixed(2)}</strong> (pos: ${modelData.pos}, neg: ${modelData.neg})`;
        } else if (modelKey === 'textblob') {
            detail = `Polarity: <strong>${(modelData.polarity || 0).toFixed(2)}</strong>, Subjectivity: <strong>${(modelData.subjectivity || 0).toFixed(2)}</strong>`;
        }

        const badgeClass = label === 'positive' ? 'text-success' : label === 'negative' ? 'text-danger' : 'text-warning';

        el.innerHTML = `
            <div class="d-flex justify-content-between align-items-center mb-1">
                <span class="fw-bold ${badgeClass} text-uppercase">${label}</span>
            </div>
            <div class="small text-muted">${detail}</div>
        `;
    }

    function renderEnsembleChart(scores) {
        const ctx = document.getElementById('ensembleScoresChart');
        if (!ctx) return;

        const pos = Math.round((scores.positive || 0) * 100);
        const neg = Math.round((scores.negative || 0) * 100);
        const neu = Math.round((scores.neutral || 0) * 100);

        if (ensembleBarChart) ensembleBarChart.destroy();

        ensembleBarChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    label: 'Score Probability (%)',
                    data: [pos, neg, neu],
                    backgroundColor: ['#00c853', '#ff1744', '#ffc400'],
                    borderRadius: 6,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { callback: (val) => val + '%' },
                    },
                },
                plugins: {
                    legend: { display: false },
                },
            },
        });
    }

    function renderUrlResults(urlData) {
        // Adapt URL scrape result into standard result view
        const adapted = {
            results: {
                overall_sentiment: urlData.overall_sentiment,
                overall_confidence: urlData.overall_confidence,
                models: {
                    transformer: { label: urlData.overall_sentiment, confidence: urlData.overall_confidence },
                },
                ensemble_scores: urlData.sentiment_distribution,
            },
            keywords: [],
            entities: [],
            statistics: {
                word_count: urlData.extracted_count,
                sentence_count: urlData.extracted_count,
                reading_time: 'Multiple items',
            },
            language: { language: 'en' },
            id: urlData.id,
        };
        renderResults(adapted);
    }

    function setTextContent(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    document.addEventListener('DOMContentLoaded', () => {
        initTextCounters();
        initFormSubmit();
    });
})();
