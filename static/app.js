const el = (id) => document.getElementById(id);
const root = document.querySelector('.container');
const ENV_COUNT = parseInt(root?.getAttribute('data-env-count') || '0', 10);
const transcriptEl = el('transcript');
const graphImg = el('graphImg');
const hypsEl = el('hyps');
const metricsEl = el('metrics');
const statusEl = el('status');
const envSelect = el('envSelect');
const startBtn = el('startBtn');
const stepBtn = el('stepBtn');

let running = false;

function renderHistory(history) {
  transcriptEl.innerHTML = '';
  history.forEach(r => {
    const item = document.createElement('div');
    item.className = 'msg';
    const tgt = r.target || 'all';
    item.innerHTML = `
      <div class="meta">[${r.turn}] ${r.speaker} → ${tgt} · tone: ${r.tone} · emotion: ${r.emotion}</div>
      <div class="text">${escapeHtml(r.reply)}</div>
    `;
    transcriptEl.appendChild(item);
  });
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function renderHyps(hyps) {
  hypsEl.innerHTML = '';
  (hyps || []).forEach(h => {
    const li = document.createElement('li');
    li.textContent = h;
    hypsEl.appendChild(li);
  });
}

function renderMetrics(md) {
  if (!metricsEl) return;
  if (!md) { metricsEl.innerHTML = ''; return; }

  const lines = md.split(/\r?\n/);
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^#\s+/.test(line)) {
      out.push('<h3>' + escapeHtml(line.replace(/^#\s+/, '')) + '</h3>');
      i++; continue;
    }
    if (/^##\s+/.test(line)) {
      out.push('<h4>' + escapeHtml(line.replace(/^##\s+/, '')) + '</h4>');
      i++; continue;
    }
    if (/^\|.*\|$/.test(line)) {
      // parse markdown table block
      const tableLines = [];
      while (i < lines.length && /^\|.*\|$/.test(lines[i])) {
        tableLines.push(lines[i]); i++;
      }
      if (tableLines.length >= 2) {
        const header = tableLines[0];
        const rows = tableLines.slice(2); // skip separator row
        const headers = header.split('|').slice(1, -1).map(s => s.trim());
        out.push('<div class="tbl-wrap"><table class="metrics-table"><thead><tr>' + headers.map(h => '<th>' + escapeHtml(h) + '</th>').join('') + '</tr></thead><tbody>');
        rows.forEach(r => {
          const cells = r.split('|').slice(1, -1).map(s => s.trim());
          if (cells.length === headers.length) {
            out.push('<tr>' + cells.map(c => '<td>' + escapeHtml(c) + '</td>').join('') + '</tr>');
          }
        });
        out.push('</tbody></table></div>');
        continue;
      }
    }
    if (line.trim().length === 0) {
      out.push('<div class="spacer"></div>');
      i++; continue;
    }
    out.push('<p>' + escapeHtml(line) + '</p>');
    i++;
  }
  metricsEl.innerHTML = out.join('\n');
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

async function api(path, method='GET', body) {
  const resp = await fetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  return resp.json();
}

startBtn.addEventListener('click', async () => {
  const env_index = parseInt(envSelect.value || '0', 10);
  startBtn.disabled = true;
  statusEl.textContent = 'Initializing...';
  try {
    const data = await api('/api/start', 'POST', { env_index });
    if (!data.ok) throw new Error(data.error || 'failed');
    running = true;
    stepBtn.disabled = false;
    statusEl.textContent = 'Session started. Click “Next message”.';
    renderHistory([]);
    renderHyps([]);
    graphImg.src = '';
    renderMetrics('');
  } catch (e) {
    console.error(e);
    statusEl.textContent = 'Start error';
    startBtn.disabled = false;
  }
});

stepBtn.addEventListener('click', async () => {
  if (!running) return;
  stepBtn.disabled = true;
  statusEl.textContent = 'Generating message...';
  try {
    const data = await api('/api/step', 'POST', {});
    if (!data.ok) throw new Error(data.error || 'failed');
    renderHistory(data.history || []);
    if (data.image_url) {
      // cache bust
      graphImg.src = data.image_url + '?t=' + Date.now();
    }
    if (data.hypotheses) {
      renderHyps(data.hypotheses);
    }
    if (data.metrics_md) {
      renderMetrics(data.metrics_md);
    }
    statusEl.textContent = `Turn: ${data.turn}`;
  } catch (e) {
    console.error(e);
    statusEl.textContent = 'Step error';
  } finally {
    stepBtn.disabled = false;
  }
});


