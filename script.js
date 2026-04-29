const container = document.getElementById('slider-container');
const beforeImg = document.getElementById('before-img');
const handle = document.getElementById('slider-handle');
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const afterImg = document.getElementById('after-img');
const loadingOverlay = document.getElementById('loading-overlay');
const actionsPanel = document.getElementById('actions-panel');
const downloadBtn = document.getElementById('download-btn');

let isDragging = false;
let colorizedBlobUrl = null;

// Slider Logic
const moveSlider = (e) => {
    if (!isDragging && e.type !== 'mousemove' && e.type !== 'touchmove') return;
    
    const rect = container.getBoundingClientRect();
    let x = (e.pageX || (e.touches ? e.touches[0].pageX : 0)) - rect.left;
    
    if (x < 0) x = 0;
    if (x > rect.width) x = rect.width;
    
    const percent = (x / rect.width) * 100;
    beforeImg.style.clipPath = `inset(0 ${100 - percent}% 0 0)`;
    handle.style.left = `${percent}%`;
};

container.addEventListener('mousedown', () => isDragging = true);
window.addEventListener('mouseup', () => isDragging = false);
window.addEventListener('mousemove', moveSlider);

container.addEventListener('touchstart', () => isDragging = true);
window.addEventListener('touchend', () => isDragging = false);
window.addEventListener('touchmove', moveSlider);

// File Upload Logic
const handleFile = async (file) => {
    if (file && file.type.startsWith('image/')) {
        // 1. Anteprima locale immediata
        const reader = new FileReader();
        reader.onload = (e) => {
            const src = e.target.result;
            beforeImg.src = src;
            afterImg.src = src;
            
            // Reset slider
            beforeImg.style.clipPath = 'inset(0 50% 0 0)';
            handle.style.left = '50%';
        };
        reader.readAsDataURL(file);

        // 2. Mostra caricamento
        loadingOverlay.classList.add('active');
        actionsPanel.style.display = 'none';

        // 3. Chiamata API
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/colorize', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Errore server');

            const blob = await response.blob();
            if (colorizedBlobUrl) URL.revokeObjectURL(colorizedBlobUrl);
            colorizedBlobUrl = URL.createObjectURL(blob);
            
            // 4. Mostra risultato
            afterImg.src = colorizedBlobUrl;
            actionsPanel.style.display = 'flex';
            
        } catch (error) {
            console.error(error);
            alert("Il server non risponde. Assicurati che main.py sia attivo.");
        } finally {
            loadingOverlay.classList.remove('active');
        }
    }
};

fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFile(e.dataTransfer.files[0]);
});

// Download Logic
downloadBtn.addEventListener('click', () => {
    if (colorizedBlobUrl) {
        const link = document.createElement('a');
        link.href = colorizedBlobUrl;
        link.download = 'chroma-revive-result.png';
        link.click();
    }
});
