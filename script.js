const sliderHandle = document.getElementById('slider-handle');
const sliderContainer = document.getElementById('slider-container');
const beforeImg = document.getElementById('before-img');
const afterImg = document.getElementById('after-img');
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const loadingOverlay = document.getElementById('loading-overlay');
const actionsPanel = document.getElementById('actions-panel');
const downloadBtn = document.getElementById('download-btn');

let isSliding = false;
let selectedModelId = 'coco30k';

// Rileva automaticamente se sei su localhost o su Render
const API_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "https://chroma-revive.onrender.com";

document.querySelectorAll('.model-option').forEach(option => {
    option.addEventListener('click', (e) => {
        e.stopPropagation();
        document.querySelectorAll('.model-option').forEach(opt => opt.classList.remove('active'));
        option.classList.add('active');
        selectedModelId = option.dataset.model;
    });
});

function moveSlider(e) {
    if (!isSliding) return;
    const rect = sliderContainer.getBoundingClientRect();
    const x = (e.pageX || (e.touches ? e.touches[0].pageX : 0)) - rect.left;
    let position = (x / rect.width) * 100;
    position = Math.max(0, Math.min(100, position));
    sliderHandle.style.left = `${position}%`;
    beforeImg.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
}

sliderHandle.addEventListener('mousedown', () => isSliding = true);
window.addEventListener('mouseup', () => isSliding = false);
window.addEventListener('mousemove', moveSlider);
sliderHandle.addEventListener('touchstart', () => isSliding = true);
window.addEventListener('touchend', () => isSliding = false);
window.addEventListener('touchmove', moveSlider);

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
        alert("Errore durante la colorizzazione. Verifica che il server sia attivo.");
    } finally {
        loadingOverlay.classList.remove('active');
    }
}
