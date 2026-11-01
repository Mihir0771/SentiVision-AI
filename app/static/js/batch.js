/**
 * SentiVision AI — Batch Tabular File Processing Module
 */

(function () {
    'use strict';

    let selectedFile = null;
    let batchChart = null;
    let batchDataRows = [];
    let currentPage = 1;
    const rowsPerPage = 10;

    function initDropZone() {
        const dropZone = document.getElementById('batchDropZone');
        const fileInput = document.getElementById('batchFileInput');
        const selectBtn = document.getElementById('btnSelectBatchFile');

        if (!dropZone || !fileInput) return;

        if (selectBtn) selectBtn.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-active');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-active');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-active');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleFile(fileInput.files[0]);
            }
        });
    }

    async function handleFile(file) {
        selectedFile = file;
        const fileInfo = document.getElementById('selectedFileInfo');
        const fileNameEl = document.getElementById('batchFileName');
        const fileSizeEl = document.getElementById('batchFileSize');

        if (fileNameEl) fileNameEl.textContent = file.name;
        if (fileSizeEl) fileSizeEl.textContent = formatBytes(file.size);
        if (fileInfo) fileInfo.classList.remove('d-none');

        // Request preview from backend to populate columns
        window.showLoading('Reading file schema and preview rows…');
        const formData = new FormData();
        formData.append('file', file);
        formData.append('preview', 'true');

        try {
            const res = await window.apiCall('/batch/upload', 'POST', formData, true);
            if (res.success && res.preview) {
                populateColumnSelector(res.columns, res.detected_column);
                renderPreviewTable(res.sample_rows, res.columns);
                const configSection = document.getElementById('batchConfigSection');
                if (configSection) configSection.classList.remove('d-none');
                window.showToast('File read successfully. Select text column to proceed.', 'info');
            } else {
                throw new Error(res.error || 'Failed to inspect file.');
            }
        } catch (err) {
            window.showToast(err.message || 'Error inspecting file.', 'error');
        } finally {
            window.hideLoading();
        }
    }

    function populateColumnSelector(columns, detectedColumn) {
        const select = document.getElementById('textColumnSelect');
        if (!select) return;

        select.innerHTML = '';
        columns.forEach((col) => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            if (col === detectedColumn) opt.selected = true;
            select.appendChild(opt);
        });
    }

    function renderPreviewTable(rows, columns) {
        const container = document.getElementById('previewTableContainer');
        if (!container || !rows || rows.length === 0) return;

        let html = '<div class="table-responsive"><table class="table table-sm table-hover text-nowrap mb-0">';
        html += '<thead class="table-dark"><tr>';
        columns.forEach((col) => (html += `<th>${col}</th>`));
        html += '</tr></thead><tbody>';

        rows.forEach((r) => {
            html += '<tr>';
            columns.forEach((col) => {
                const val = String(r[col] || '');
                const displayVal = val.length > 50 ? val.substring(0, 47) + '…' : val;
                html += `<td>${displayVal}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table></div>';
        container.innerHTML = html;
    }

    function initProcessAction() {
        const btnProcess = document.getElementById('btnStartBatchProcessing');
        if (!btnProcess) return;

        btnProcess.addEventListener('click', async () => {
            if (!selectedFile) {
                window.showToast('Please select a file first.', 'warning');
                return;
            }

            const select = document.getElementById('textColumnSelect');
            const selectedCol = select?.value || '';

            btnProcess.disabled = true;
            btnProcess.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing Reviews…';
            window.showLoading('Analyzing batch reviews with multi-model NLP…');

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('text_column', selectedCol);

            try {
                const res = await window.apiCall('/batch/upload', 'POST', formData, true);
                if (res.success && res.data) {
                    renderBatchResults(res.data);
                    const resultsSection = document.getElementById('batchResultsSection');
                    if (resultsSection) {
                        resultsSection.classList.remove('d-none');
                        resultsSection.scrollIntoView({ behavior: 'smooth' });
                    }
                    window.showToast('Batch processing complete!', 'success');
                } else {
                    throw new Error(res.error || 'Failed to process batch.');
                }
            } catch (err) {
                window.showToast(err.message || 'Error processing batch file.', 'error');
            } finally {
                btnProcess.disabled = false;
                btnProcess.innerHTML = '<i class="fas fa-play me-2"></i>Run Sentiment Analysis';
                window.hideLoading();
            }
        });
    }

    function renderBatchResults(data) {
        const summary = data.summary || {};
        const items = data.results || [];
        batchDataRows = items;
        currentPage = 1;

        // Stat cards
        setElementText('batchTotalItems', data.total_items || items.length);
        setElementText('batchPositivePct', `${summary.positive_pct || 0}%`);
        setElementText('batchNegativePct', `${summary.negative_pct || 0}%`);
        setElementText('batchNeutralPct', `${summary.neutral_pct || 0}%`);
        setElementText('batchAvgConfidence', `${Math.round((summary.avg_confidence || 0) * 100)}%`);

        // Export links
        const exportCsv = document.getElementById('btnExportBatchCsv');
        const exportJson = document.getElementById('btnExportBatchJson');
        if (exportCsv && data.id) exportCsv.href = `/batch/export/${data.id}/csv`;
        if (exportJson && data.id) exportJson.href = `/batch/export/${data.id}/json`;

        // Render Chart
        renderBatchPie(summary);

        // Render Results Table
        renderTablePage();
        initTableFilter();
    }

    function renderBatchPie(summary) {
        const ctx = document.getElementById('batchDistributionChart');
        if (!ctx) return;

        if (batchChart) batchChart.destroy();

        batchChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    data: [summary.positive_count || 0, summary.negative_count || 0, summary.neutral_count || 0],
                    backgroundColor: ['#00c853', '#ff1744', '#ffc400'],
                    borderWidth: 0,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' },
                },
            },
        });
    }

    function renderTablePage() {
        const tbody = document.getElementById('batchTableBody');
        const pagination = document.getElementById('batchPagination');
        if (!tbody) return;

        const start = (currentPage - 1) * rowsPerPage;
        const pageItems = batchDataRows.slice(start, start + rowsPerPage);

        tbody.innerHTML = '';
        pageItems.forEach((item, idx) => {
            const tr = document.createElement('tr');
            const sentiment = (item.sentiment || 'neutral').toLowerCase();
            const badgeClass = sentiment === 'positive' ? 'bg-success' : sentiment === 'negative' ? 'bg-danger' : 'bg-warning text-dark';
            const conf = Math.round((item.confidence || 0) * 100);

            tr.innerHTML = `
                <td class="text-muted">${start + idx + 1}</td>
                <td><span class="text-truncate d-inline-block" style="max-width: 420px;" title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</span></td>
                <td><span class="badge ${badgeClass} text-uppercase">${sentiment}</span></td>
                <td><span class="fw-semibold">${conf}%</span></td>
            `;
            tbody.appendChild(tr);
        });

        // Pagination
        if (pagination) {
            const totalPages = Math.ceil(batchDataRows.length / rowsPerPage) || 1;
            pagination.innerHTML = `
                <li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
                    <button class="page-link" id="prevPageBtn">&laquo; Prev</button>
                </li>
                <li class="page-item disabled">
                    <span class="page-link">Page ${currentPage} of ${totalPages}</span>
                </li>
                <li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
                    <button class="page-link" id="nextPageBtn">Next &raquo;</button>
                </li>
            `;

            document.getElementById('prevPageBtn')?.addEventListener('click', () => {
                if (currentPage > 1) {
                    currentPage--;
                    renderTablePage();
                }
            });

            document.getElementById('nextPageBtn')?.addEventListener('click', () => {
                if (currentPage < totalPages) {
                    currentPage++;
                    renderTablePage();
                }
            });
        }
    }

    function initTableFilter() {
        const searchInput = document.getElementById('batchSearchFilter');
        const sentimentSelect = document.getElementById('batchSentimentFilter');

        function applyFilter() {
            const query = (searchInput?.value || '').toLowerCase();
            const filterSent = (sentimentSelect?.value || 'all').toLowerCase();

            batchDataRows = (window.cachedBatchRows || batchDataRows).filter((r) => {
                const matchesText = !query || (r.text || '').toLowerCase().includes(query);
                const matchesSent = filterSent === 'all' || (r.sentiment || '').toLowerCase() === filterSent;
                return matchesText && matchesSent;
            });
            currentPage = 1;
            renderTablePage();
        }

        if (searchInput) searchInput.addEventListener('input', applyFilter);
        if (sentimentSelect) sentimentSelect.addEventListener('change', applyFilter);
    }

    function setElementText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    document.addEventListener('DOMContentLoaded', () => {
        initDropZone();
        initProcessAction();
    });
})();
