const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('previewImg');
const predictBtn = document.getElementById('predictBtn');
const resultSection = document.getElementById('resultSection');
const predLabel = document.getElementById('predLabel');
const predConfidence = document.getElementById('predConfidence');
const probBars = document.getElementById('probBars');
const pieCanvas = document.getElementById('pieChart');
const pieLegend = document.getElementById('pieLegend');

let selectedFile = null;

const classColors = {
    brain_glioma: '#f87171',
    brain_menin: '#fbbf24',
    brain_tumor: '#c084fc',
    healthy: '#34d399'
};

dropZone.addEventListener('click', () => fileInput.click());

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
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) return;
    selectedFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    preview.style.display = 'block';
    predictBtn.disabled = false;
    resultSection.style.display = 'none';
}

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    predictBtn.disabled = true;
    predictBtn.textContent = 'Predicting...';
    dropZone.classList.add('loading');

    const form = new FormData();
    form.append('image', selectedFile);

    try {
        const res = await fetch('/api/predict', { method: 'POST', body: form });
        const data = await res.json();
        if (res.ok) showResult(data);
        else alert(data.error || 'Prediction failed');
    } catch (e) {
        alert('Network error: ' + e.message);
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict';
        dropZone.classList.remove('loading');
    }
});

function drawPieChart(probabilities) {
    const ctx = pieCanvas.getContext('2d');
    const cx = pieCanvas.width / 2;
    const cy = pieCanvas.height / 2;
    const radius = Math.min(cx, cy) - 8;

    ctx.clearRect(0, 0, pieCanvas.width, pieCanvas.height);

    const entries = Object.entries(probabilities).filter(([, v]) => v > 0);
    const total = entries.reduce((sum, [, v]) => sum + v, 0);
    if (total === 0) return;

    let startAngle = -Math.PI / 2;

    for (const [cls, value] of entries) {
        const sliceAngle = (value / total) * 2 * Math.PI;
        const color = classColors[cls] || '#6366f1';

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, radius, startAngle, startAngle + sliceAngle);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();

        if (sliceAngle > 0.3) {
            const midAngle = startAngle + sliceAngle / 2;
            const labelR = radius * 0.65;
            const lx = cx + Math.cos(midAngle) * labelR;
            const ly = cy + Math.sin(midAngle) * labelR;
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 12px system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(value.toFixed(1) + '%', lx, ly);
        }

        startAngle += sliceAngle;
    }

    ctx.beginPath();
    ctx.arc(cx, cy, radius * 0.38, 0, 2 * Math.PI);
    ctx.fillStyle = '#111827';
    ctx.fill();

    const predText = predLabel.textContent;
    ctx.fillStyle = '#f9fafb';
    ctx.font = 'bold 13px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(predText, cx, cy - 6);
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillStyle = '#9ca3af';
    ctx.fillText(predConfidence.textContent, cx, cy + 10);

    pieLegend.innerHTML = '';
    for (const [cls, value] of entries) {
        const color = classColors[cls] || '#6366f1';
        pieLegend.innerHTML += `
            <div class="legend-item">
                <span class="legend-dot" style="background:${color}"></span>
                <span class="legend-text">${cls.replace(/_/g, ' ')}</span>
                <span class="legend-val">${value.toFixed(1)}%</span>
            </div>`;
    }
}

function showResult(data) {
    resultSection.style.display = 'block';
    predLabel.textContent = data.prediction.replace(/_/g, ' ');
    predLabel.style.color = classColors[data.prediction] || '#6366f1';
    predConfidence.textContent = data.confidence + '%';
    predConfidence.style.color = classColors[data.prediction] || '#6366f1';

    probBars.innerHTML = '';
    const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    for (const [cls, prob] of sorted) {
        const color = classColors[cls] || '#6366f1';
        probBars.innerHTML += `
            <div class="prob-bar-row">
                <span class="prob-bar-label">${cls.replace(/_/g, ' ')}</span>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width:${prob}%;background:${color}">${prob}%</div>
                </div>
            </div>`;
    }

    drawPieChart(data.probabilities);
}
