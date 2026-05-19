/**
 * app.js — CameraForensics frontend logic
 * Handles: drag-and-drop, file preview, predict API call,
 *          results rendering, vote-bar charts, training panel.
 */

'use strict';

/* ── Element refs ──────────────────────────────────────────────────────────── */
const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const previewWrap    = document.getElementById('preview-wrap');
const previewImg     = document.getElementById('preview-img');
const clearBtn       = document.getElementById('clear-btn');
const predictBtn     = document.getElementById('predict-btn');
const resultsCard    = document.getElementById('results-card');
const healthBadge    = document.getElementById('health-badge');
const helpBtn        = document.getElementById('help-btn');
const helpModal      = document.getElementById('help-modal');
const helpCloseBtn   = document.getElementById('help-close');
const overlay        = document.getElementById('global-overlay');
const toastContainer = document.getElementById('toast-container');

// Result fields
const resBrand       = document.getElementById('res-brand');
const resModel       = document.getElementById('res-model');
const resPatches     = document.getElementById('res-patches');
const brandConfBar   = document.getElementById('brand-conf-bar');
const brandConfPct   = document.getElementById('brand-conf-pct');
const modelConfBar   = document.getElementById('model-conf-bar');
const modelConfPct   = document.getElementById('model-conf-pct');
const brandVoteChart = document.getElementById('brand-votes-chart');
const modelVoteChart = document.getElementById('model-votes-chart');
const modelVoteSec   = document.getElementById('model-votes-section');

// Training panel
const trainLevel     = document.getElementById('train-level');
const brandField     = document.getElementById('brand-field');
const trainBrand     = document.getElementById('train-brand');
const trainEpochs    = document.getElementById('train-epochs');
const trainBtn       = document.getElementById('train-btn');
const trainStatusEl  = document.getElementById('train-status');
const statusDot      = document.getElementById('status-dot');
const statusText     = document.getElementById('status-text');
const jobIdBadge     = document.getElementById('job-id-badge');
const trainLog       = document.getElementById('train-log');

/* ── State ─────────────────────────────────────────────────────────────────── */
let selectedFile    = null;
let pollInterval    = null;
let currentJobId    = null;

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Health check                                                                */
/* ═══════════════════════════════════════════════════════════════════════════ */
async function checkHealth() {
  try {
    const res  = await fetch('/api/health');
    const data = await res.json();
    if (data.status === 'ok') {
      const gpuLabel = data.gpu ? `GPU: ${data.gpu_name}` : 'CPU only';
      const brands   = data.brands.length ? data.brands.join(', ') : 'No weights loaded';
      healthBadge.textContent = `${gpuLabel} · ${data.brands.length} brands`;
      healthBadge.classList.remove('badge--loading', 'badge--error');
      healthBadge.classList.add('badge--ok');
      healthBadge.title = `Brands: ${brands} | Models: ${data.num_models}`;
    }
  } catch {
    healthBadge.textContent = 'Server offline';
    healthBadge.classList.remove('badge--loading', 'badge--ok');
    healthBadge.classList.add('badge--error');
  }
}
checkHealth();

/* Help modal handlers */
function toggleHelp(open) {
  const show = typeof open === 'boolean' ? open : helpModal.classList.contains('hidden');
  if (show) {
    helpModal.classList.remove('hidden');
    helpModal.querySelector('.help-inner').focus();
  } else {
    helpModal.classList.add('hidden');
  }
}
helpBtn?.addEventListener('click', () => toggleHelp(true));
helpCloseBtn?.addEventListener('click', () => toggleHelp(false));
document.addEventListener('keydown', e => {
  if ((e.key === 'h' || e.key === 'H') && document.activeElement.tagName !== 'INPUT') toggleHelp(true);
  if (e.key === 'Escape') toggleHelp(false);
});

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Drag-and-drop & file selection                                              */
/* ═══════════════════════════════════════════════════════════════════════════ */
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

clearBtn.addEventListener('click', clearFile);

function setFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Please select an image file.', 'error');
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewWrap.classList.remove('hidden');
  dropZone.classList.add('hidden');
  predictBtn.disabled = false;
  resultsCard.classList.add('hidden');
}

function clearFile() {
  selectedFile = null;
  previewImg.src = '';
  previewWrap.classList.add('hidden');
  dropZone.classList.remove('hidden');
  predictBtn.disabled = true;
  resultsCard.classList.add('hidden');
  fileInput.value = '';
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Prediction                                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */
predictBtn.addEventListener('click', runPrediction);

async function runPrediction() {
  if (!selectedFile) return;

  setLoading(predictBtn, true);
  resultsCard.classList.add('hidden');

  try {
    const form = new FormData();
    form.append('file', selectedFile);

    const res  = await fetch('/api/predict', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Prediction failed');
    }
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setLoading(predictBtn, false);
  }
}

function renderResults(d) {
  /* Scalar fields */
  resBrand.textContent   = d.brand  || '—';
  resModel.textContent   = d.model  || '—';
  resPatches.textContent = d.num_patches ?? '—';

  /* Confidence bars (animate after next frame) */
  const bPct = Math.round((d.brand_confidence ?? 0) * 100);
  const mPct = Math.round((d.model_confidence ?? 0) * 100);
  requestAnimationFrame(() => {
    brandConfBar.style.width = `${bPct}%`;
    brandConfPct.textContent = `${bPct}%`;
    modelConfBar.style.width = `${mPct}%`;
    modelConfPct.textContent = `${mPct}%`;
  });

  /* Vote distribution charts */
  renderVoteChart(brandVoteChart, d.brand_votes, d.brand, 'primary');

  if (d.model_votes && Object.keys(d.model_votes).length) {
    renderVoteChart(modelVoteChart, d.model_votes, d.model, 'secondary');
    modelVoteSec.classList.remove('hidden');
  } else {
    modelVoteSec.classList.add('hidden');
  }

  resultsCard.classList.remove('hidden');
  resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderVoteChart(container, votes, winner, type) {
  container.innerHTML = '';
  const total   = Object.values(votes).reduce((a, b) => a + b, 0) || 1;
  const sorted  = Object.entries(votes).sort((a, b) => b[1] - a[1]);
  const maxVote = sorted[0]?.[1] || 1;

  sorted.forEach(([name, count]) => {
    const pct      = (count / maxVote * 100).toFixed(1);
    const isWinner = name === winner;
    const grad     = type === 'primary'
      ? 'linear-gradient(90deg,#6366f1,#06b6d4)'
      : 'linear-gradient(90deg,#8b5cf6,#ec4899)';

    const row = document.createElement('div');
    row.className = 'vote-row';
    row.innerHTML = `
      <span class="vote-name" title="${name}">${isWinner ? '★ ' : ''}${name}</span>
      <div class="vote-bar-wrap">
        <div class="vote-bar-fill" style="width:0%;background:${grad}"
             data-target="${pct}"></div>
      </div>
      <span class="vote-count">${count}</span>`;
    container.appendChild(row);
  });

  /* Animate bars after DOM insertion */
  requestAnimationFrame(() => {
    container.querySelectorAll('.vote-bar-fill').forEach(el => {
      el.style.width = el.dataset.target + '%';
    });
  });
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Training panel                                                              */
/* ═══════════════════════════════════════════════════════════════════════════ */
trainLevel.addEventListener('change', () => {
  brandField.style.display = trainLevel.value === 'model' ? 'flex' : 'none';
});

trainBtn.addEventListener('click', startTraining);

async function startTraining() {
  const level  = trainLevel.value;
  const brand  = trainBrand.value.trim() || null;
  const epochs = parseInt(trainEpochs.value, 10) || 100;

  if (level === 'model' && !brand) {
    showToast('Enter a brand name for model-level training.', 'error');
    return;
  }

  setLoading(trainBtn, true);
  trainBtn.disabled = true;

  try {
    const res  = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ level, brand, max_epochs: epochs }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to start training');

    currentJobId = data.job_id;
    jobIdBadge.textContent = `Job: ${currentJobId}`;
    trainStatusEl.classList.remove('hidden');
    updateStatusUI('running', 'Training in progress…');
    startPolling(currentJobId);
  } catch (err) {
    showToast(err.message, 'error');
    setLoading(trainBtn, false);
    trainBtn.disabled = false;
  }
}

function startPolling(jobId) {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(() => pollStatus(jobId), 2500);
}

async function pollStatus(jobId) {
  try {
    const res  = await fetch(`/api/train/status/${jobId}`);
    const data = await res.json();

    /* Update log */
    if (data.log_tail?.length) {
      trainLog.textContent = data.log_tail.join('\n');
      trainLog.scrollTop   = trainLog.scrollHeight;
    }

    if (data.status === 'done') {
      clearInterval(pollInterval);
      updateStatusUI('done', 'Training complete.');
      setLoading(trainBtn, false);
      trainBtn.disabled = false;
      checkHealth();                       // refresh badge
    } else if (data.status === 'error') {
      clearInterval(pollInterval);
      updateStatusUI('error', 'Training failed — see log.');
      setLoading(trainBtn, false);
      trainBtn.disabled = false;
    }
  } catch { /* network blip — keep polling */ }
}

function updateStatusUI(state, message) {
  statusDot.className  = `status-dot ${state}`;
  statusText.textContent = message;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Helpers                                                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */
function setLoading(btn, loading) {
  btn.classList.toggle('loading', loading);
  btn.disabled = loading;
  // Show global overlay when long-running primary actions run
  if (btn === predictBtn || btn === trainBtn) {
    overlay.classList.toggle('hidden', !loading);
  }
}

function showToast(message, type = 'info', timeout = 4000) {
  if (!toastContainer) {
    // fallback to old behavior
    const t = document.createElement('div'); t.textContent = message; document.body.appendChild(t);
    setTimeout(() => t.remove(), timeout); return;
  }
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.setAttribute('role', 'status');
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => toast.remove(), timeout);
}
