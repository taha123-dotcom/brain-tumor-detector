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
    previewImg.src = URL.createObjectURL(file);
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
        if (res.ok) {
            showResult(data);
        } else {
            alert(data.error || 'Prediction failed');
        }
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

function showResult(data) {
    resultSection.style.display = 'block';
    const color = classColors[data.prediction] || '#3b82f6';
    predLabel.textContent = data.prediction.replace(/_/g, ' ');
    predLabel.style.color = color;
    predConfidence.textContent = data.confidence + '%';
    predConfidence.style.color = color;

    probBars.innerHTML = '';
    const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    for (const [cls, prob] of sorted) {
        const c = classColors[cls] || '#3b82f6';
        probBars.innerHTML += `
            <div class="prob-bar-row">
                <span class="prob-bar-label">${cls.replace(/_/g, ' ')}</span>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width:${prob}%;background:${c}">${prob}%</div>
                </div>
            </div>`;
    }
}

async function loadHistory() {
    try {
        const res = await fetch('/api/history');
        if (!res.ok) throw new Error('HTTP ' + res.status);
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
        historyTable.innerHTML = '<p style="color:var(--text-dim)">No prediction history available.</p>';
    }
}

async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        statsGrid.innerHTML = `
            <div class="stat-card"><div class="stat-value">${data.total_predictions}</div><div class="stat-label">Total Predictions</div></div>
            <div class="stat-card"><div class="stat-value">${data.model_accuracy}%</div><div class="stat-label">Model Accuracy</div></div>
            <div class="stat-card"><div class="stat-value">${data.classes.length}</div><div class="stat-label">Classes</div></div>`;
    } catch (e) {
        statsGrid.innerHTML = '';
    }
}

loadHistory();
loadStats();
