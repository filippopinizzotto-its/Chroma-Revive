const container = document.getElementById('slider-container');
const beforeImg = document.getElementById('before-img');
const handle = document.getElementById('slider-handle');
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const afterImg = document.getElementById('after-img');

let isDragging = false;

// Slider Logic
const moveSlider = (e) => {
    if (!isDragging && e.type !== 'mousemove' && e.type !== 'touchmove') return;
    
    const rect = container.getBoundingClientRect();
    let x = (e.pageX || e.touches[0].pageX) - rect.left;
    
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
const handleFile = (file) => {
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const src = e.target.result;
            afterImg.src = src;
            beforeImg.src = src;
            
            // Reset slider to middle
            beforeImg.style.clipPath = 'inset(0 50% 0 0)';
            handle.style.left = '50%';
        };
        reader.readAsDataURL(file);
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
