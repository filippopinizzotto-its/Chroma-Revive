const sliderHandle = document.getElementById('slider-handle');
const sliderContainer = document.getElementById('slider-container');
const beforeImg = document.getElementById('before-img');
const afterImg = document.getElementById('after-img');
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const loadingOverlay = document.getElementById('loading-overlay');
const actionsPanel = document.getElementById('actions-panel');
const downloadBtn = document.getElementById('download-btn');
const toastContainer = document.getElementById('toast-container');

let isSliding = false;
let selectedModelId = 'coco30k';

// Se il frontend non è servito dal backend FastAPI, usa il server locale di default.
const DEFAULT_API_URL = 'http://127.0.0.1:8000';
const API_URL = (() => {
    const origin = window.location.origin;
    const port = window.location.port;

    if (!origin || origin === 'null' || origin.startsWith('file://')) {
        return DEFAULT_API_URL;
    }

    if (port === '8000') {
        return origin;
    }

    return DEFAULT_API_URL;
})();

document.querySelectorAll('.model-option').forEach(option => {
    option.addEventListener('click', (e) => {
        e.stopPropagation();
        document.querySelectorAll('.model-option').forEach(opt => opt.classList.remove('active'));
        option.classList.add('active');
        selectedModelId = option.dataset.model;
    });
});

document.querySelectorAll('.example-thumb').forEach(thumb => {
    thumb.addEventListener('click', async () => {
        const imageUrl = thumb.dataset.url;
        try {
            const response = await fetch(imageUrl);
            const blob = await response.blob();
            const file = new File([blob], "example.jpg", { type: "image/jpeg" });
            handleFile(file);
            showToast("Esempio caricato con successo!", "success");
        } catch (error) {
            showToast("Errore nel caricamento dell'esempio.", "error");
        }
    });
});

function moveSlider(e) {
    if (!isSliding) return;
    const rect = sliderContainer.getBoundingClientRect();
    const x = (e.pageX || (e.touches ? e.touches[0].pageX : 0)) - rect.left;
    let position = (x / rect.width) * 100;
    position = Math.max(0, Math.min(100, position));
    updateSliderPosition(position);
}

sliderHandle.addEventListener('mousedown', () => {
    isSliding = true;
    sliderHandle.classList.add('active');
});
window.addEventListener('mouseup', () => {
    isSliding = false;
    sliderHandle.classList.remove('active');
});
window.addEventListener('mousemove', moveSlider);
sliderHandle.addEventListener('touchstart', () => {
    isSliding = true;
    sliderHandle.classList.add('active');
});
window.addEventListener('touchend', () => {
    isSliding = false;
    sliderHandle.classList.remove('active');
});
window.addEventListener('touchmove', moveSlider);

window.addEventListener('keydown', (e) => {
    if (document.activeElement !== sliderHandle) return;

    let currentLeft = parseFloat(sliderHandle.style.left) || 50;
    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        updateSliderPosition(Math.max(0, currentLeft - 2));
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        updateSliderPosition(Math.min(100, currentLeft + 2));
    }
});

function updateSliderPosition(position) {
    sliderHandle.style.left = `${position}%`;
    sliderHandle.setAttribute('aria-valuenow', Math.round(position));
    beforeImg.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
}

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        beforeImg.src = e.target.result;
        afterImg.src = e.target.result;
        
        sliderHandle.style.left = '50%';
        beforeImg.style.clipPath = 'inset(0 50% 0 0)';
        
        colorizeImage(file);
    };
    reader.readAsDataURL(file);
}

async function colorizeImage(file) {
    loadingOverlay.classList.add('active');
    actionsPanel.style.display = 'none';
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/colorize?model_id=${selectedModelId}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Errore server');

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        afterImg.src = url;
        actionsPanel.style.display = 'flex';
        
        downloadBtn.onclick = () => {
            const a = document.createElement('a');
            a.href = url;
            a.download = `chroma-revive-${selectedModelId}.png`;
            a.click();
        };

    } catch (error) {
        showToast(error.message || "Errore durante la colorizzazione. Verifica che il server sia attivo.", "error");
    } finally {
        loadingOverlay.classList.remove('active');
    }
}

function resetApp() {
    beforeImg.src = 'placeholder.png';
    afterImg.src = 'placeholder.png';
    sliderHandle.style.left = '50%';
    beforeImg.style.clipPath = 'inset(0 50% 0 0)';
    actionsPanel.style.display = 'none';
    fileInput.value = '';
    showToast("Pronto per una nuova foto!", "info");
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    let icon = '';
    if (type === 'error') icon = '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>';
    else if (type === 'success') icon = '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>';
    else icon = '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';

    toast.innerHTML = `${icon}<span>${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-out');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
