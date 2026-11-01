/**
 * SentiVision AI — Common Utilities & Global Interactions
 */

(function () {
    'use strict';

    // -------------------------------------------------------------------------
    // Theme Management (Dark / Light)
    // -------------------------------------------------------------------------
    function initTheme() {
        const themeToggle = document.getElementById('themeToggle');
        const themeIcon = document.getElementById('themeIcon');
        const html = document.documentElement;

        const savedTheme = localStorage.getItem('sentivision_theme') || 'dark';
        applyTheme(savedTheme);

        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                const current = html.getAttribute('data-bs-theme') || 'dark';
                const next = current === 'dark' ? 'light' : 'dark';
                applyTheme(next);
            });
        }

        function applyTheme(theme) {
            html.setAttribute('data-bs-theme', theme);
            localStorage.setItem('sentivision_theme', theme);
            if (themeIcon) {
                if (theme === 'dark') {
                    themeIcon.classList.remove('fa-sun');
                    themeIcon.classList.add('fa-moon');
                } else {
                    themeIcon.classList.remove('fa-moon');
                    themeIcon.classList.add('fa-sun');
                }
            }
            // Trigger chart re-render if function exists
            if (window.updateChartsTheme) {
                window.updateChartsTheme(theme);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Toast Notification System
    // -------------------------------------------------------------------------
    window.showToast = function (message, type = 'info', title = '') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const id = 'toast-' + Math.random().toString(36).substring(2, 9);
        const iconMap = {
            success: 'fa-check-circle text-success',
            error: 'fa-exclamation-circle text-danger',
            warning: 'fa-exclamation-triangle text-warning',
            info: 'fa-info-circle text-info',
        };
        const icon = iconMap[type] || iconMap.info;

        const toastEl = document.createElement('div');
        toastEl.className = 'toast align-items-center border-0 shadow-lg mb-2';
        toastEl.setAttribute('role', 'alert');
        toastEl.setAttribute('aria-live', 'assertive');
        toastEl.setAttribute('aria-atomic', 'true');
        toastEl.id = id;

        toastEl.innerHTML = `
            <div class="d-flex">
                <div class="toast-body d-flex align-items-center gap-2">
                    <i class="fas ${icon} fs-5"></i>
                    <div>
                        ${title ? `<strong class="d-block">${title}</strong>` : ''}
                        <span>${message}</span>
                    </div>
                </div>
                <button type="button" class="btn-close me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;

        container.appendChild(toastEl);
        if (window.bootstrap && window.bootstrap.Toast) {
            const bsToast = new window.bootstrap.Toast(toastEl, { delay: 4500 });
            bsToast.show();
            toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
        } else {
            setTimeout(() => toastEl.remove(), 4500);
        }
    };

    // -------------------------------------------------------------------------
    // Loading Overlay
    // -------------------------------------------------------------------------
    window.showLoading = function (msg = 'Processing…') {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            const text = overlay.querySelector('p');
            if (text) text.textContent = msg;
            overlay.classList.remove('d-none');
        }
    };

    window.hideLoading = function () {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('d-none');
        }
    };

    // -------------------------------------------------------------------------
    // Universal AJAX helper
    // -------------------------------------------------------------------------
    window.apiCall = async function (url, method = 'GET', data = null, isFormData = false) {
        const options = {
            method: method.toUpperCase(),
            headers: {},
        };

        if (data && !isFormData) {
            options.headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(data);
        } else if (data && isFormData) {
            options.body = data;
        }

        try {
            const response = await fetch(url, options);
            const contentType = response.headers.get('content-type') || '';

            if (contentType.includes('application/json')) {
                const json = await response.json();
                if (!response.ok) {
                    throw new Error(json.error || json.message || `Request failed (${response.status})`);
                }
                return json;
            } else {
                if (!response.ok) {
                    throw new Error(`Request failed with status ${response.status}`);
                }
                return await response.text();
            }
        } catch (err) {
            console.error('API Error:', err);
            throw err;
        }
    };

    // -------------------------------------------------------------------------
    // Copy to Clipboard Utility
    // -------------------------------------------------------------------------
    window.copyToClipboard = function (text, successMsg = 'Copied to clipboard!') {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(() => {
                window.showToast(successMsg, 'success');
            }).catch(() => {
                fallbackCopy(text, successMsg);
            });
        } else {
            fallbackCopy(text, successMsg);
        }
    };

    function fallbackCopy(text, successMsg) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
            document.execCommand('copy');
            window.showToast(successMsg, 'success');
        } catch (e) {
            window.showToast('Could not copy to clipboard', 'error');
        }
        document.body.removeChild(textarea);
    }

    // -------------------------------------------------------------------------
    // Auto Dismiss Flash Alerts
    // -------------------------------------------------------------------------
    function initFlashDismissal() {
        setTimeout(() => {
            const alerts = document.querySelectorAll('.flash-alert');
            alerts.forEach((alert) => {
                if (window.bootstrap && window.bootstrap.Alert) {
                    const bsAlert = new window.bootstrap.Alert(alert);
                    bsAlert.close();
                } else {
                    alert.style.display = 'none';
                }
            });
        }, 5000);
    }

    // Initialize on DOM load
    document.addEventListener('DOMContentLoaded', () => {
        initTheme();
        initFlashDismissal();
    });
})();
