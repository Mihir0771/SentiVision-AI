/**
 * SentiVision AI — Live Camera & Facial Emotion Recognition Module
 * Supports physical webcam streams and virtual face simulation for zero-friction testing.
 */

(function () {
    'use strict';

    let videoEl = null;
    let canvasEl = null;
    let overlayCanvas = null;
    let streamInstance = null;
    let isStreaming = false;
    let isVirtualMode = false;
    let isMirrored = true; // Mirror feed by default like standard webcams
    let streamInterval = null;
    let virtualSimInterval = null;
    let socket = null;
    let emotionTimelineChart = null;
    let emotionTimelineData = { labels: [], happy: [], neutral: [], sad: [], surprise: [], angry: [] };

    // Smoothed emotion scores to avoid flickering UI
    const smoothedEmotions = {
        happy: 0,
        neutral: 0,
        surprise: 0,
        sad: 0,
        angry: 0,
        fear: 0,
        disgust: 0,
    };

    const EMOJI_MAP = {
        happy: '😊',
        sad: '😢',
        angry: '😠',
        surprise: '😮',
        fear: '😨',
        disgust: '🤢',
        neutral: '😐',
    };

    const EMOTION_COLORS = {
        happy: '#00c853',
        surprise: '#29b6f6',
        neutral: '#ffc400',
        sad: '#7e57c2',
        angry: '#ff1744',
        fear: '#ff9100',
        disgust: '#8d6e63',
    };

    // Virtual simulator expression presets
    const VIRTUAL_EXPRESSIONS = [
        { emotion: 'happy', label: 'HAPPY', conf: 0.94, emotions: { happy: 0.92, neutral: 0.05, surprise: 0.03, sad: 0.0, angry: 0.0, fear: 0.0, disgust: 0.0 } },
        { emotion: 'surprise', label: 'SURPRISED', conf: 0.88, emotions: { surprise: 0.86, happy: 0.08, neutral: 0.04, sad: 0.01, angry: 0.01, fear: 0.0, disgust: 0.0 } },
        { emotion: 'neutral', label: 'NEUTRAL', conf: 0.91, emotions: { neutral: 0.88, happy: 0.06, surprise: 0.04, sad: 0.02, angry: 0.0, fear: 0.0, disgust: 0.0 } },
        { emotion: 'happy', label: 'EXCITED', conf: 0.96, emotions: { happy: 0.95, surprise: 0.03, neutral: 0.02, sad: 0.0, angry: 0.0, fear: 0.0, disgust: 0.0 } },
    ];
    let virtualExpressionIdx = 0;

    function checkSecurityContext() {
        const isLocalhost = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
        const isHttps = location.protocol === 'https:';

        // If accessed via LAN IP over plain HTTP, modern browsers block getUserMedia
        if (!isLocalhost && !isHttps) {
            const notice = document.getElementById('cameraSecurityNotice');
            if (notice) {
                notice.classList.remove('d-none');
                const port = location.port ? `:${location.port}` : '';
                const link = document.getElementById('localHostLink');
                if (link) {
                    link.href = `http://localhost${port}/camera/`;
                    link.textContent = `http://localhost${port}/camera/`;
                }
            }
        }
    }

    function initSocket() {
        if (typeof io !== 'undefined') {
            try {
                socket = io('/camera', {
                    transports: ['websocket', 'polling'],
                    reconnection: true,
                    reconnectionAttempts: 10,
                    reconnectionDelay: 1000,
                });

                socket.on('connect', () => {
                    const statusBadge = document.getElementById('cameraStatusBadge');
                    if (statusBadge) {
                        statusBadge.className = 'badge bg-success';
                        statusBadge.textContent = 'Socket Connected';
                    }
                });

                socket.on('emotion_result', (data) => {
                    if (safetyTimeout) clearTimeout(safetyTimeout);
                    isProcessingFrame = false;
                    handleEmotionResult(data);
                });

                socket.on('emotion_error', (err) => {
                    if (safetyTimeout) clearTimeout(safetyTimeout);
                    isProcessingFrame = false;
                    console.warn('Backend emotion frame warning:', err);
                });

                socket.on('disconnect', () => {
                    const statusBadge = document.getElementById('cameraStatusBadge');
                    if (statusBadge) {
                        statusBadge.className = 'badge bg-secondary';
                        statusBadge.textContent = 'Socket Standby';
                    }
                });
            } catch (e) {
                console.warn('Socket.IO init fallback active:', e);
            }
        }
    }

    async function startCamera() {
        // Stop any active virtual mode first
        if (isVirtualMode) stopVirtualDemo();

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            window.showToast(
                'Camera API restricted by browser on this URL. Starting Virtual Face Simulator instead!',
                'warning',
                'Webcam Notice'
            );
            startVirtualDemo();
            return;
        }

        try {
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user',
                },
                audio: false,
            };

            streamInstance = await navigator.mediaDevices.getUserMedia(constraints);
            videoEl.srcObject = streamInstance;
            videoEl.style.transform = isMirrored ? 'scaleX(-1)' : 'none';
            await videoEl.play();

            isStreaming = true;
            isVirtualMode = false;
            setFeedBadge('Live Webcam', 'bg-success');
            toggleControls(true);
            window.showToast('Camera active. Tracking face and emotions in real-time…', 'success');

            startFrameLoop();
        } catch (err) {
            console.warn('Webcam stream error:', err);
            let userMsg = 'Webcam not available. Starting Virtual Face Simulator for testing!';
            if (err.name === 'NotAllowedError') {
                userMsg = 'Webcam permission denied in browser settings. Starting Virtual Simulator!';
            }
            window.showToast(userMsg, 'info', 'Simulator Active');
            startVirtualDemo();
        }
    }

    function stopCamera() {
        if (streamInstance) {
            streamInstance.getTracks().forEach((track) => track.stop());
            streamInstance = null;
        }
        if (videoEl) videoEl.srcObject = null;
        if (streamInterval) {
            clearInterval(streamInterval);
            streamInterval = null;
        }
        if (virtualSimInterval) {
            clearInterval(virtualSimInterval);
            virtualSimInterval = null;
        }
        if (safetyTimeout) {
            clearTimeout(safetyTimeout);
            safetyTimeout = null;
        }
        isStreaming = false;
        isVirtualMode = false;
        isProcessingFrame = false;
        setFeedBadge('Standby', 'bg-dark-subtle');
        toggleControls(false);
        clearOverlay();
        window.showToast('Feed stopped.', 'info');
    }

    function toggleMirror() {
        isMirrored = !isMirrored;
        if (videoEl) {
            videoEl.style.transform = isMirrored ? 'scaleX(-1)' : 'none';
        }
        const mirrorText = document.getElementById('mirrorStatusText');
        if (mirrorText) {
            mirrorText.textContent = isMirrored ? 'Mirror: On' : 'Mirror: Off';
        }
    }

    // -------------------------------------------------------------------------
    // Virtual Face Simulator
    // -------------------------------------------------------------------------
    function startVirtualDemo() {
        stopCamera();
        isVirtualMode = true;
        isStreaming = true;

        toggleControls(true);
        setFeedBadge('Virtual Simulator', 'bg-primary');
        window.showToast('Virtual Face Simulator active. Simulating multi-emotion expressions!', 'success');

        overlayCanvas.style.top = '0px';
        overlayCanvas.style.left = '0px';
        overlayCanvas.style.width = '100%';
        overlayCanvas.style.height = '100%';
        overlayCanvas.width = 640;
        overlayCanvas.height = 480;

        const ctx = overlayCanvas.getContext('2d');
        let frameCounter = 0;

        virtualSimInterval = setInterval(() => {
            if (!isVirtualMode) return;
            frameCounter++;

            // Switch expression every ~16 frames (every 4 seconds)
            if (frameCounter % 16 === 0) {
                virtualExpressionIdx = (virtualExpressionIdx + 1) % VIRTUAL_EXPRESSIONS.length;
            }

            const current = VIRTUAL_EXPRESSIONS[virtualExpressionIdx];
            const dominant = current.emotion;

            // Draw animated simulated face on overlay canvas
            drawVirtualAvatar(ctx, dominant, frameCounter);

            // Mock face detection bounding box
            const faces = [{
                box: { x: 190, y: 80, w: 260, h: 290 },
                dominant_emotion: dominant,
                confidence: current.conf,
                emotions: current.emotions,
            }];

            handleEmotionResult({
                faces_detected: 1,
                faces: faces,
            });

        }, 250); // 4 FPS
    }

    function stopVirtualDemo() {
        if (virtualSimInterval) {
            clearInterval(virtualSimInterval);
            virtualSimInterval = null;
        }
        isVirtualMode = false;
    }

    function drawVirtualAvatar(ctx, emotion, tick) {
        ctx.clearRect(0, 0, 640, 480);

        // Dark background for simulated camera feed
        ctx.fillStyle = '#0d1124';
        ctx.fillRect(0, 0, 640, 480);

        // Head
        ctx.fillStyle = '#ffdfba';
        ctx.beginPath();
        ctx.ellipse(320, 220, 110, 130, 0, 0, Math.PI * 2);
        ctx.fill();

        // Eyes
        ctx.fillStyle = '#2c3e50';
        const eyeOffset = Math.sin(tick * 0.2) * 2;
        ctx.beginPath();
        ctx.arc(280, 195 + eyeOffset, emotion === 'surprise' ? 14 : 10, 0, Math.PI * 2);
        ctx.arc(360, 195 + eyeOffset, emotion === 'surprise' ? 14 : 10, 0, Math.PI * 2);
        ctx.fill();

        // Mouth (changes shape based on emotion)
        ctx.strokeStyle = '#c0392b';
        ctx.lineWidth = 6;
        ctx.beginPath();
        if (emotion === 'happy') {
            ctx.arc(320, 250, 45, 0.1 * Math.PI, 0.9 * Math.PI);
        } else if (emotion === 'surprise') {
            ctx.ellipse(320, 265, 18, 28, 0, 0, Math.PI * 2);
        } else if (emotion === 'sad') {
            ctx.arc(320, 285, 40, 1.1 * Math.PI, 1.9 * Math.PI);
        } else {
            // Neutral
            ctx.moveTo(290, 265);
            ctx.lineTo(350, 265);
        }
        ctx.stroke();

        // Draw bounding box
        const color = EMOTION_COLORS[emotion] || '#667eea';
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(190, 80, 260, 290);

        // Header tag
        ctx.fillStyle = color;
        ctx.fillRect(190, 52, 260, 28);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText(`SIMULATED: ${EMOJI_MAP[emotion] || ''} ${emotion.toUpperCase()}`, 200, 71);
    }

    // -------------------------------------------------------------------------
    // Frame Streaming & Bounding Box Logic
    // -------------------------------------------------------------------------
    let isProcessingFrame = false;
    let safetyTimeout = null;

    function startFrameLoop() {
        if (streamInterval) clearInterval(streamInterval);

        const ctx = canvasEl.getContext('2d');
        const targetFPS = 6; // 6 FPS for responsive real-time inference

        streamInterval = setInterval(() => {
            if (!isStreaming || isVirtualMode || !videoEl || videoEl.paused || videoEl.ended || isProcessingFrame) return;

            const vw = videoEl.videoWidth;
            const vh = videoEl.videoHeight;
            if (!vw || !vh) return;

            isProcessingFrame = true;

            // Safety timeout to prevent loop stalling if an event is dropped
            if (safetyTimeout) clearTimeout(safetyTimeout);
            safetyTimeout = setTimeout(() => {
                isProcessingFrame = false;
            }, 500);

            // Downsample to 480p width for ultra-fast deep-learning inference
            const targetW = 480;
            const targetH = Math.round(targetW * (vh / vw));

            canvasEl.width = targetW;
            canvasEl.height = targetH;
            ctx.drawImage(videoEl, 0, 0, targetW, targetH);

            const b64Frame = canvasEl.toDataURL('image/jpeg', 0.65);

            if (socket && socket.connected) {
                socket.emit('process_frame', { frame: b64Frame });
            } else {
                sendSnapshotHttp(b64Frame, false).finally(() => {
                    if (safetyTimeout) clearTimeout(safetyTimeout);
                    isProcessingFrame = false;
                });
            }
        }, 1000 / targetFPS);
    }

    async function sendSnapshotHttp(base64Image, notify = true) {
        try {
            const res = await window.apiCall('/camera/snapshot', 'POST', {
                image: base64Image,
                save: notify, // Only persist to DB when user manually requests snapshot
            });
            if (res.success && res.data) {
                handleEmotionResult(res.data);
                if (notify) {
                    addSnapshotToGallery(base64Image, res.data);
                    window.showToast('Snapshot captured and analyzed!', 'success');
                }
            }
        } catch (e) {
            console.error('Snapshot HTTP error:', e);
        }
    }

    function getVideoRenderedRect(video) {
        const videoRatio = (video.videoWidth || 640) / (video.videoHeight || 480);
        const elementWidth = video.clientWidth || 640;
        const elementHeight = video.clientHeight || 480;
        const elementRatio = elementWidth / elementHeight;

        let renderedWidth, renderedHeight, offsetX, offsetY;
        if (elementRatio > videoRatio) {
            renderedHeight = elementHeight;
            renderedWidth = elementHeight * videoRatio;
            offsetX = (elementWidth - renderedWidth) / 2;
            offsetY = 0;
        } else {
            renderedWidth = elementWidth;
            renderedHeight = elementWidth / videoRatio;
            offsetX = 0;
            offsetY = (elementHeight - renderedHeight) / 2;
        }
        return { x: offsetX, y: offsetY, w: renderedWidth, h: renderedHeight };
    }

    function handleEmotionResult(data) {
        const faces = data.faces || [];
        const detected = data.faces_detected !== undefined ? data.faces_detected : faces.length;

        // Draw bounding box overlay for real camera
        if (!isVirtualMode) {
            drawFacesOverlay(faces, data);
        }

        const dominantLabelEl = document.getElementById('dominantEmotionLabel');
        const dominantEmojiEl = document.getElementById('dominantEmotionEmoji');
        const dominantConfEl = document.getElementById('dominantEmotionConfidence');
        const facesCountEl = document.getElementById('facesDetectedCount');

        if (facesCountEl) facesCountEl.textContent = detected;

        if (faces.length > 0) {
            const primaryFace = faces[0];
            const dominant = primaryFace.dominant_emotion || 'neutral';
            const emotions = primaryFace.emotions || {};
            const conf = Math.round((primaryFace.confidence || (emotions[dominant] || 0)) * 100);

            if (dominantLabelEl) {
                dominantLabelEl.textContent = dominant.toUpperCase();
                dominantLabelEl.style.color = EMOTION_COLORS[dominant] || '#fff';
            }
            if (dominantEmojiEl) {
                dominantEmojiEl.textContent = EMOJI_MAP[dominant] || '😐';
            }
            if (dominantConfEl) {
                dominantConfEl.textContent = `${conf}% Confidence`;
            }

            // Exponential smoothing for fluid progress bar transitions
            const alpha = 0.7;
            for (const em of ['happy', 'neutral', 'surprise', 'sad', 'angry', 'fear', 'disgust']) {
                const target = emotions[em] || 0;
                smoothedEmotions[em] = (alpha * target) + ((1 - alpha) * (smoothedEmotions[em] || 0));
                updateEmotionBar(em, smoothedEmotions[em]);
            }

            // Update Rolling Timeline Chart
            pushTimelineData(emotions);
        } else {
            if (dominantLabelEl) dominantLabelEl.textContent = 'NO FACE DETECTED';
            if (dominantEmojiEl) dominantEmojiEl.textContent = '👤';
            if (dominantConfEl) dominantConfEl.textContent = 'Position face within camera view';
            clearOverlay();
        }
    }

    function updateEmotionBar(emotion, score) {
        const pct = Math.min(100, Math.max(0, Math.round(score * 100)));
        const bar = document.getElementById(`bar_${emotion}`);
        const text = document.getElementById(`pct_${emotion}`);
        if (bar) bar.style.width = `${pct}%`;
        if (text) text.textContent = `${pct}%`;
    }

    function drawFacesOverlay(faces, data) {
        if (!overlayCanvas || !videoEl) return;
        const ctx = overlayCanvas.getContext('2d');
        const rect = getVideoRenderedRect(videoEl);

        // Align canvas size and position with rendered video rectangle
        overlayCanvas.style.top = `${rect.y}px`;
        overlayCanvas.style.left = `${rect.x}px`;
        overlayCanvas.style.width = `${rect.w}px`;
        overlayCanvas.style.height = `${rect.h}px`;
        overlayCanvas.width = rect.w;
        overlayCanvas.height = rect.h;

        ctx.clearRect(0, 0, rect.w, rect.h);

        const sentW = (data && data.image_width) || (canvasEl ? canvasEl.width : 480) || 480;
        const sentH = (data && data.image_height) || (canvasEl ? canvasEl.height : 360) || 360;
        const scaleX = rect.w / sentW;
        const scaleY = rect.h / sentH;

        faces.forEach((f) => {
            const box = f.box || {};
            let rawX = (box.x || 0) * scaleX;
            const rawY = (box.y || 0) * scaleY;
            const rawW = (box.w || 0) * scaleX;
            const rawH = (box.h || 0) * scaleY;

            // Invert X coordinate if webcam is mirrored
            if (isMirrored) {
                rawX = rect.w - rawX - rawW;
            }

            const dominant = f.dominant_emotion || 'neutral';
            const conf = Math.round((f.confidence || 0) * 100);
            const color = EMOTION_COLORS[dominant] || '#667eea';

            // Draw bounding rect with subtle neon glow
            ctx.shadowColor = color;
            ctx.shadowBlur = 10;
            ctx.strokeStyle = color;
            ctx.lineWidth = 3.5;
            ctx.strokeRect(rawX, rawY, rawW, rawH);
            ctx.shadowBlur = 0;

            // Header background pill
            const headerH = 28;
            const headerW = Math.max(rawW, 140);
            const pillY = Math.max(0, rawY - headerH);
            ctx.fillStyle = color;
            ctx.fillRect(rawX, pillY, headerW, headerH);

            // Header text
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 13px system-ui, -apple-system, sans-serif';
            const labelText = `${EMOJI_MAP[dominant] || ''} ${dominant.toUpperCase()} ${conf}%`;
            ctx.fillText(labelText, rawX + 8, pillY + 19);
        });
    }

    function clearOverlay() {
        if (overlayCanvas) {
            const ctx = overlayCanvas.getContext('2d');
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        }
    }

    function pushTimelineData(emotions) {
        if (!emotionTimelineChart) return;

        const now = new Date();
        const timeStr = `${now.getHours()}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

        if (emotionTimelineData.labels.length > 20) {
            emotionTimelineData.labels.shift();
            emotionTimelineData.happy.shift();
            emotionTimelineData.neutral.shift();
            emotionTimelineData.sad.shift();
            emotionTimelineData.surprise.shift();
            emotionTimelineData.angry.shift();
        }

        emotionTimelineData.labels.push(timeStr);
        emotionTimelineData.happy.push(Math.round((emotions.happy || 0) * 100));
        emotionTimelineData.neutral.push(Math.round((emotions.neutral || 0) * 100));
        emotionTimelineData.sad.push(Math.round((emotions.sad || 0) * 100));
        emotionTimelineData.surprise.push(Math.round((emotions.surprise || 0) * 100));
        emotionTimelineData.angry.push(Math.round((emotions.angry || 0) * 100));

        emotionTimelineChart.update('none');
    }

    function initEmotionTimelineChart() {
        const ctx = document.getElementById('cameraEmotionTimeline');
        if (!ctx) return;

        emotionTimelineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: emotionTimelineData.labels,
                datasets: [
                    { label: 'Happy', data: emotionTimelineData.happy, borderColor: EMOTION_COLORS.happy, tension: 0.3, pointRadius: 0 },
                    { label: 'Neutral', data: emotionTimelineData.neutral, borderColor: EMOTION_COLORS.neutral, tension: 0.3, pointRadius: 0 },
                    { label: 'Surprise', data: emotionTimelineData.surprise, borderColor: EMOTION_COLORS.surprise, tension: 0.3, pointRadius: 0 },
                    { label: 'Sad', data: emotionTimelineData.sad, borderColor: EMOTION_COLORS.sad, tension: 0.3, pointRadius: 0 },
                    { label: 'Angry', data: emotionTimelineData.angry, borderColor: EMOTION_COLORS.angry, tension: 0.3, pointRadius: 0 },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { display: false },
                    y: { min: 0, max: 100, ticks: { display: false }, grid: { display: false } },
                },
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } },
                },
            },
        });
    }

    function addSnapshotToGallery(imgSrc, data) {
        const gallery = document.getElementById('snapshotGallery');
        if (!gallery) return;

        const faces = data.faces || [];
        const dominant = faces.length > 0 ? faces[0].dominant_emotion : 'neutral';
        const emoji = EMOJI_MAP[dominant] || '😐';

        const col = document.createElement('div');
        col.className = 'col-6 col-md-4 mb-3';
        col.innerHTML = `
            <div class="card h-100 border-0 shadow-sm overflow-hidden">
                <img src="${imgSrc}" class="card-img-top object-fit-cover" style="height: 120px;" alt="Snapshot">
                <div class="card-body p-2 text-center">
                    <span class="badge text-bg-dark">${emoji} ${dominant.toUpperCase()}</span>
                    <div class="small text-muted mt-1">${new Date().toLocaleTimeString()}</div>
                </div>
            </div>
        `;
        gallery.prepend(col);
    }

    function setFeedBadge(text, bgClass) {
        const badge = document.getElementById('feedModeBadge');
        if (badge) {
            badge.className = `badge ${bgClass} text-light me-2`;
            badge.textContent = text;
        }
    }

    function toggleControls(streaming) {
        const btnStart = document.getElementById('btnStartCamera');
        const btnStop = document.getElementById('btnStopCamera');
        const btnSnap = document.getElementById('btnTakeSnapshot');
        const placeholder = document.getElementById('cameraPlaceholder');

        if (btnStart) btnStart.classList.toggle('d-none', streaming);
        if (btnStop) btnStop.classList.toggle('d-none', !streaming);
        if (btnSnap) btnSnap.disabled = !streaming;
        if (placeholder) placeholder.classList.toggle('d-none', streaming);
        if (videoEl && !isVirtualMode) videoEl.classList.toggle('d-none', !streaming);
        if (overlayCanvas) overlayCanvas.classList.toggle('d-none', !streaming);
    }

    function initStaticImageUpload() {
        const dropZone = document.getElementById('photoDropZone');
        const fileInput = document.getElementById('photoFileInput');

        if (!dropZone || !fileInput) return;

        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-accent');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('border-accent');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-accent');
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handlePhotoUpload(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handlePhotoUpload(fileInput.files[0]);
            }
        });

        async function handlePhotoUpload(file) {
            window.showLoading('Analyzing facial emotions in photo…');
            const formData = new FormData();
            formData.append('image', file);

            try {
                const res = await window.apiCall('/camera/upload-image', 'POST', formData, true);
                if (res.success && res.data) {
                    handleEmotionResult(res.data);
                    const reader = new FileReader();
                    reader.onload = (e) => addSnapshotToGallery(e.target.result, res.data);
                    reader.readAsDataURL(file);
                    window.showToast('Facial expression detection complete!', 'success');
                } else {
                    throw new Error(res.error || 'Failed to detect facial emotions.');
                }
            } catch (err) {
                window.showToast(err.message || 'Image analysis error.', 'error');
            } finally {
                window.hideLoading();
            }
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        videoEl = document.getElementById('webcamVideo');
        canvasEl = document.getElementById('captureCanvas');
        overlayCanvas = document.getElementById('overlayCanvas');

        const btnStart = document.getElementById('btnStartCamera');
        const btnStop = document.getElementById('btnStopCamera');
        const btnSnap = document.getElementById('btnTakeSnapshot');
        const btnMirror = document.getElementById('btnToggleMirror');
        const btnVirtual = document.getElementById('btnStartVirtualDemo');
        const btnVirtualAlert = document.getElementById('btnLaunchDemoFromAlert');
        const btnToggleFeed = document.getElementById('btnToggleVirtualFeed');

        if (btnStart) btnStart.addEventListener('click', startCamera);
        if (btnStop) btnStop.addEventListener('click', stopCamera);
        if (btnMirror) btnMirror.addEventListener('click', toggleMirror);
        if (btnVirtual) btnVirtual.addEventListener('click', startVirtualDemo);
        if (btnVirtualAlert) btnVirtualAlert.addEventListener('click', startVirtualDemo);
        if (btnToggleFeed) {
            btnToggleFeed.addEventListener('click', () => {
                if (isVirtualMode) {
                    virtualExpressionIdx = (virtualExpressionIdx + 1) % VIRTUAL_EXPRESSIONS.length;
                    window.showToast(`Switched virtual emotion: ${VIRTUAL_EXPRESSIONS[virtualExpressionIdx].label}`, 'info');
                } else {
                    startVirtualDemo();
                }
            });
        }

        if (btnSnap) {
            btnSnap.addEventListener('click', () => {
                if (!isStreaming) return;

                if (isVirtualMode && overlayCanvas) {
                    const b64 = overlayCanvas.toDataURL('image/jpeg', 0.85);
                    const cur = VIRTUAL_EXPRESSIONS[virtualExpressionIdx];
                    addSnapshotToGallery(b64, {
                        faces: [{ dominant_emotion: cur.emotion, confidence: cur.conf }],
                    });
                    window.showToast('Virtual snapshot captured!', 'success');
                    return;
                }

                const ctx = canvasEl.getContext('2d');
                canvasEl.width = videoEl.videoWidth || 640;
                canvasEl.height = videoEl.videoHeight || 480;
                ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
                const b64 = canvasEl.toDataURL('image/jpeg', 0.85);
                sendSnapshotHttp(b64, true);
            });
        }

        checkSecurityContext();
        initSocket();
        initEmotionTimelineChart();
        initStaticImageUpload();
    });
})();
