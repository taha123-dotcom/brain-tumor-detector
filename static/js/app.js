const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('previewImg');
const predictBtn = document.getElementById('predictBtn');
const resultSection = document.getElementById('resultSection');
const predLabel = document.getElementById('predLabel');
const predConfidence = document.getElementById('predConfidence');
const probBars = document.getElementById('probBars');
const statsGrid = document.getElementById('statsGrid');
const historyTable = document.getElementById('historyTable');
const pieCanvas = document.getElementById('pieChart');
const pieLegend = document.getElementById('pieLegend');

let selectedFile = null;

const classColors = {
    brain_glioma: '#ef4444',
    brain_menin: '#f59e0b',
    brain_tumor: '#a855f7',
    healthy: '#22c55e'
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
        loadHistory();
        loadStats();
    }
});

function drawPieChart(probabilities) {
    const ctx = pieCanvas.getContext('2d');
    const cx = pieCanvas.width / 2;
    const cy = pieCanvas.height / 2;
    const radius = Math.min(cx, cy) - 10;

    ctx.clearRect(0, 0, pieCanvas.width, pieCanvas.height);

    const entries = Object.entries(probabilities).filter(([, v]) => v > 0);
    const total = entries.reduce((sum, [, v]) => sum + v, 0);

    if (total === 0) return;

    let startAngle = -Math.PI / 2;

    for (const [cls, value] of entries) {
        const sliceAngle = (value / total) * 2 * Math.PI;
        const color = classColors[cls] || '#3b82f6';

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
            ctx.font = 'bold 13px system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(value.toFixed(1) + '%', lx, ly);
        }

        startAngle += sliceAngle;
    }

    ctx.beginPath();
    ctx.arc(cx, cy, radius * 0.4, 0, 2 * Math.PI);
    ctx.fillStyle = '#1e293b';
    ctx.fill();

    const predText = document.getElementById('predLabel').textContent;
    ctx.fillStyle = '#f1f5f9';
    ctx.font = 'bold 14px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(predText, cx, cy - 8);
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(document.getElementById('predConfidence').textContent, cx, cy + 10);

    pieLegend.innerHTML = '';
    for (const [cls, value] of entries) {
        const color = classColors[cls] || '#3b82f6';
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
    predLabel.style.color = classColors[data.prediction] || '#3b82f6';
    predConfidence.textContent = data.confidence + '%';
    predConfidence.style.color = classColors[data.prediction] || '#3b82f6';

    probBars.innerHTML = '';
    const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    for (const [cls, prob] of sorted) {
        const color = classColors[cls] || '#3b82f6';
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

async function loadHistory() {
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        if (!data.length) {
            historyTable.innerHTML = '<p style="color:var(--text-dim)">No predictions yet.</p>';
            return;
        }
        let html = `<table class="history-table">
            <thead><tr><th>File</th><th>Prediction</th><th>Confidence</th><th>Time</th></tr></thead>
            <tbody>`;
        for (const r of data) {
            const t = new Date(r.created_at).toLocaleString();
            html += `<tr>
                <td>${r.filename}</td>
                <td><span class="pred-tag ${r.prediction}">${r.prediction.replace(/_/g, ' ')}</span></td>
                <td>${r.confidence}%</td>
                <td>${t}</td>
            </tr>`;
        }
        html += '</tbody></table>';
        historyTable.innerHTML = html;
    } catch (e) {
        historyTable.innerHTML = '<p style="color:var(--text-dim)">Could not load history.</p>';
    }
}

async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        statsGrid.innerHTML = `
            <div class="stat-card"><div class="stat-value">${data.total_predictions}</div><div class="stat-label">Total Predictions</div></div>
            <div class="stat-card"><div class="stat-value">${data.model_accuracy}%</div><div class="stat-label">Model Accuracy</div></div>
            <div class="stat-card"><div class="stat-value">${data.classes.length}</div><div class="stat-label">Classes</div></div>`;
    } catch (e) {}
}

loadHistory();
loadStats();
