/**
 * Real-Time Age and Gender Detection System - Frontend Controller
 */

let isStreamPlaying = true;
let statusPollingInterval = null;
let currentTab = "live";
let uploadedImageDetections = null;
const VIDEO_FEED_URL = "/video_feed";

// ==============================================================================
// Initialization & Polling
// ==============================================================================

document.addEventListener("DOMContentLoaded", () => {
    console.log("[INIT] Age & Gender Detection UI initialized.");
    startStatusPolling();
    setupDropzone();
});

function startStatusPolling() {
    fetchSystemStatus();
    statusPollingInterval = setInterval(fetchSystemStatus, 1000);
}

async function fetchSystemStatus() {
    try {
        const response = await fetch("/api/status");
        if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
        const data = await response.json();
        updateSystemHUD(data);
    } catch (err) {
        console.warn("[STATUS] Polling failed:", err);
        updateStatusBadge(false, "Offline / Connecting...");
    }
}

function updateSystemHUD(data) {
    const fpsDisplay = document.getElementById("fpsDisplay");
    const faceCountNumber = document.getElementById("faceCountNumber");

    if (fpsDisplay) {
        fpsDisplay.textContent = data.fps !== undefined ? data.fps : "--";
    }

    if (!data.models_loaded) {
        updateStatusBadge(false, "Models Missing / Error");
    } else if (data.camera_active) {
        updateStatusBadge(true, "Live Stream Active");
    } else {
        updateStatusBadge(false, "Webcam Idle / Disconnected");
    }

    // Only update sidebar from live webcam stream if currently viewing the live feed tab
    if (currentTab === "live") {
        if (faceCountNumber) {
            faceCountNumber.textContent = data.faces_detected || 0;
        }
        updateDetectionsList(data.detections || []);
    }
}

function updateStatusBadge(isActive, text) {
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");

    if (statusDot && statusText) {
        statusDot.className = `status-dot ${isActive ? "active" : "error"}`;
        statusText.textContent = text;
    }
}

function updateDetectionsList(detections) {
    const container = document.getElementById("detectionsList");
    if (!container) return;

    if (!detections || detections.length === 0) {
        container.innerHTML = `
            <div class="empty-state" id="emptyDetections">
                <div class="empty-icon">👤</div>
                <p>No faces currently detected in frame</p>
                <small>${currentTab === "live" ? "Position yourself in front of the camera" : "Upload an image with a visible human face"}</small>
            </div>
        `;
        return;
    }

    let html = "";
    detections.forEach((det, idx) => {
        const genderClass = (det.gender || "").toLowerCase() === "female" ? "female" : "male";
        html += `
            <div class="detection-item ${genderClass}">
                <div class="detection-header">
                    <span class="person-badge">Face #${idx + 1}</span>
                    <span class="chip ${genderClass === "female" ? "gender-female" : "gender-male"}">${det.gender}</span>
                </div>
                <div class="detection-details">
                    <div class="det-field">
                        <span class="det-label">Age Range</span>
                        <span class="det-value age">${det.age}</span>
                    </div>
                    <div class="det-field">
                        <span class="det-label">Confidence</span>
                        <span class="det-value">${det.gender_confidence || 95}%</span>
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// ==============================================================================
// Video Stream Controls
// ==============================================================================

function toggleStream() {
    const videoEl = document.getElementById("videoFeed");
    const toggleText = document.getElementById("toggleText");
    const toggleIcon = document.getElementById("toggleIcon");

    if (isStreamPlaying) {
        videoEl.src = "";
        isStreamPlaying = false;
        toggleText.textContent = "Resume Stream";
        toggleIcon.textContent = "▶";
    } else {
        videoEl.src = `${VIDEO_FEED_URL}?t=${new Date().getTime()}`;
        isStreamPlaying = true;
        toggleText.textContent = "Pause Stream";
        toggleIcon.textContent = "⏸";
    }
}

function handleStreamError() {
    console.warn("[STREAM] Video stream error event fired.");
}

async function reconnectCamera() {
    const overlay = document.getElementById("streamOverlayMsg");
    const overlayText = document.getElementById("overlayMsgText");
    if (overlay) {
        overlayText.textContent = "Reconnecting webcam feed...";
        overlay.style.display = "flex";
    }

    try {
        const resp = await fetch("/api/camera/reconnect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: 0 })
        });
        const result = await resp.json();
        console.log("[CAMERA] Reconnected:", result);
    } catch (err) {
        console.error("[CAMERA] Reconnect error:", err);
    } finally {
        setTimeout(() => {
            const videoEl = document.getElementById("videoFeed");
            if (videoEl) {
                videoEl.src = `${VIDEO_FEED_URL}?t=${new Date().getTime()}`;
            }
            if (overlay) overlay.style.display = "none";
        }, 1200);
    }
}

function toggleFullscreen() {
    const elem = document.getElementById("feedWrapper");
    if (!document.fullscreenElement) {
        if (elem.requestFullscreen) {
            elem.requestFullscreen();
        } else if (elem.webkitRequestFullscreen) {
            elem.webkitRequestFullscreen();
        }
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        }
    }
}

function captureSnapshot() {
    const videoEl = document.getElementById("videoFeed");
    if (!videoEl || !isStreamPlaying) {
        alert("Please resume the live stream to capture a snapshot.");
        return;
    }

    try {
        const canvas = document.getElementById("snapshotCanvas");
        canvas.width = videoEl.naturalWidth || 640;
        canvas.height = videoEl.naturalHeight || 480;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

        const dataUrl = canvas.toDataURL("image/jpeg", 0.95);
        const link = document.createElement("a");
        link.download = `age_gender_snapshot_${Date.now()}.jpg`;
        link.href = dataUrl;
        link.click();
    } catch (err) {
        console.error("[SNAPSHOT] Failed to save frame:", err);
        window.open(videoEl.src, "_blank");
    }
}

// ==============================================================================
// View Tab Switching
// ==============================================================================

function switchTab(tab) {
    currentTab = tab;
    const liveContainer = document.getElementById("liveStreamContainer");
    const uploadContainer = document.getElementById("imageUploadContainer");
    const tabLive = document.getElementById("tabLiveStream");
    const tabUpload = document.getElementById("tabImageUpload");
    const faceCountNumber = document.getElementById("faceCountNumber");

    if (tab === "live") {
        liveContainer.style.display = "flex";
        uploadContainer.style.display = "none";
        tabLive.classList.add("active");
        tabUpload.classList.remove("active");
        fetchSystemStatus();
    } else {
        liveContainer.style.display = "none";
        uploadContainer.style.display = "flex";
        tabLive.classList.remove("active");
        tabUpload.classList.add("active");

        // If an image was previously analyzed, restore its detections in sidebar
        if (uploadedImageDetections) {
            updateDetectionsList(uploadedImageDetections);
            if (faceCountNumber) faceCountNumber.textContent = uploadedImageDetections.length;
        } else {
            updateDetectionsList([]);
            if (faceCountNumber) faceCountNumber.textContent = 0;
        }
    }
}

// ==============================================================================
// Drag & Drop / Image Upload Mode
// ==============================================================================

function setupDropzone() {
    const dropzone = document.getElementById("dropzone");
    if (!dropzone) return;

    ["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add("dragover"), false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove("dragover"), false);
    });

    dropzone.addEventListener("drop", handleDrop, false);
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
        processUploadedImage(files[0]);
    }
}

function handleFileSelected(event) {
    const file = event.target.files[0];
    if (file) {
        processUploadedImage(file);
    }
}

async function processUploadedImage(file) {
    const formData = new FormData();
    formData.append("image", file);

    const promptEl = document.getElementById("dropzonePrompt");
    const resultWrapper = document.getElementById("uploadResultWrapper");
    const resultImg = document.getElementById("uploadedResultImage");
    const uploadToolbar = document.getElementById("uploadToolbar");
    const downloadBtn = document.getElementById("btnDownloadAnalyzed");

    if (promptEl) promptEl.innerHTML = `<div class="overlay-spinner"></div><p>Analyzing face age and gender...</p>`;

    try {
        const response = await fetch("/api/analyze_upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        if (!data.success) {
            alert(`Analysis Error: ${data.error || "Failed to analyze image"}`);
            resetUploadPrompt();
            return;
        }

        if (promptEl) promptEl.style.display = "none";
        if (resultWrapper) resultWrapper.style.display = "flex";
        if (resultImg) resultImg.src = data.annotated_image;
        if (uploadToolbar) uploadToolbar.style.display = "flex";
        if (downloadBtn) downloadBtn.href = data.annotated_image;

        uploadedImageDetections = data.detections || [];
        updateDetectionsList(uploadedImageDetections);
        
        const faceCountNumber = document.getElementById("faceCountNumber");
        if (faceCountNumber) faceCountNumber.textContent = data.faces_detected || 0;

    } catch (err) {
        console.error("[UPLOAD] Analysis request failed:", err);
        alert("Failed to communicate with server.");
        resetUploadPrompt();
    }
}

function resetUploadPrompt() {
    const promptEl = document.getElementById("dropzonePrompt");
    const resultWrapper = document.getElementById("uploadResultWrapper");
    const uploadToolbar = document.getElementById("uploadToolbar");

    if (promptEl) {
        promptEl.style.display = "flex";
        promptEl.innerHTML = `
            <div class="upload-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
            </div>
            <h3>Click or Drag & Drop an Image to Analyze</h3>
            <p>Supports JPG, PNG, WEBP files</p>
        `;
    }
    if (resultWrapper) resultWrapper.style.display = "none";
    if (uploadToolbar) uploadToolbar.style.display = "none";
    uploadedImageDetections = null;
}
