// ==================== EXPERIMENT PARTICIPANT VIEW ====================

const el = (id) => document.getElementById(id);

// URL params
const params = new URLSearchParams(window.location.search);
let myName = params.get('name') ? decodeURIComponent(params.get('name')) : '';

// DOM refs
const joinScreen = el('joinScreen');
const nameInput = el('nameInput');
const joinBtn = el('joinBtn');
const joinError = el('joinError');
const expMain = el('expMain');
const expEnv = el('expEnv');
const myNameBadge = el('myNameBadge');
const expTurn = el('expTurn');
const turnBanner = el('turnBanner');
const turnText = el('turnText');
const participantsBar = el('participantsBar');
const inputArea = el('inputArea');
const targetSelect = el('targetSelect');
const messageInput = el('messageInput');
const toneSelect = el('toneSelect');
const sendBtn = el('sendBtn');
const transcriptEl = el('transcript');
const graphImg = el('graphImg');
const graphPlaceholder = el('graphPlaceholder');
const statusEl = el('status');

// State
let joined = false;
let agentsData = [];
let lastPendingHuman = null;
let isAdvancing = false;
let pollInterval = null;

// Pre-fill name from URL
if (myName) {
    nameInput.value = myName;
}

// ==================== CHECK EXPERIMENT STATUS ON LOAD ====================

(async function checkExperimentOnLoad() {
    try {
        const res = await fetch('/api/experiment/status');
        const data = await res.json();
        if (!data.ok || !data.active || !data.multi_user) {
            joinError.textContent = 'Experiment is not active yet. Ask the moderator to start the experiment first.';
            joinError.style.display = 'block';
            joinBtn.disabled = true;
            joinBtn.textContent = 'Waiting for experiment...';
            // Keep checking every 3 seconds
            const checkInterval = setInterval(async () => {
                try {
                    const r = await fetch('/api/experiment/status');
                    const d = await r.json();
                    if (d.ok && d.active && d.multi_user) {
                        clearInterval(checkInterval);
                        joinError.style.display = 'none';
                        joinBtn.disabled = false;
                        joinBtn.textContent = 'Join Experiment';
                        // Auto-join if name was in URL
                        if (myName) doJoin();
                    }
                } catch(e) {}
            }, 3000);
        } else if (myName) {
            // Experiment is active and name is in URL — auto-join
            doJoin();
        }
    } catch(e) {
        // Server not reachable
        joinError.textContent = 'Cannot connect to server. Check the URL.';
        joinError.style.display = 'block';
    }
})();

// ==================== JOIN ====================

joinBtn.addEventListener('click', doJoin);
nameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') doJoin();
});

async function doJoin() {
    myName = nameInput.value.trim();
    if (!myName) return;

    joinBtn.disabled = true;
    joinBtn.textContent = 'Joining...';
    joinError.style.display = 'none';

    try {
        const res = await fetch('/api/experiment/join', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: myName }),
        });
        const data = await res.json();
        if (!data.ok) {
            joinError.textContent = data.error;
            joinError.style.display = 'block';
            return;
        }

        joined = true;
        agentsData = data.agents_data || [];
        joinScreen.style.display = 'none';
        expMain.style.display = 'block';
        expEnv.textContent = data.env || '';
        myNameBadge.textContent = myName;

        populateTargets();
        renderAll(data);
        startPolling();
        statusEl.textContent = data.pending_human
            ? `Waiting for ${data.pending_human}...`
            : 'Joined. Wait for the moderator to start the experiment.';
    } catch (e) {
        joinError.textContent = 'Connection error: ' + e.message;
        joinError.style.display = 'block';
    } finally {
        joinBtn.disabled = false;
        joinBtn.textContent = 'Join Experiment';
    }
}

// ==================== TARGET DROPDOWN ====================

function populateTargets() {
    targetSelect.innerHTML = '';
    agentsData.filter(a => a.name !== myName && !a.is_observer).forEach(a => {
        const opt = document.createElement('option');
        opt.value = a.name;
        opt.textContent = `${a.name} (${a.nature})`;
        targetSelect.appendChild(opt);
    });
}

// ==================== POLLING ====================

function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/experiment/status?caller=${encodeURIComponent(myName)}`);
            const data = await res.json();
            if (data.ok && data.active) {
                if (data.agents_data) agentsData = data.agents_data;
                renderAll(data);
            }
        } catch (e) {
            statusEl.textContent = 'Connection lost...';
        }
    }, 2000);
}

// ==================== RENDER ====================

function renderAll(data) {
    renderTranscript(data.history || []);
    renderTurnBanner(data.pending_human);
    renderGraph(data.image_url);
    renderParticipants(data.agents_data || agentsData, data.connected_humans || {});
    expTurn.textContent = `Turn: ${data.turn || 0}`;
}

function renderTurnBanner(pendingHuman) {
    const wasMyTurn = lastPendingHuman === myName;
    const isMyTurn = pendingHuman === myName;

    if (isMyTurn) {
        turnBanner.className = 'exp-turn-banner my-turn';
        turnText.textContent = 'YOUR TURN — Write your message below!';
        inputArea.classList.remove('disabled');
        sendBtn.disabled = false;
        if (!wasMyTurn) {
            playNotification();
            messageInput.focus();
        }
    } else if (pendingHuman) {
        turnBanner.className = 'exp-turn-banner waiting';
        turnText.textContent = `Waiting for ${pendingHuman}...`;
        inputArea.classList.add('disabled');
        sendBtn.disabled = true;
    } else {
        turnBanner.className = 'exp-turn-banner processing';
        turnText.textContent = 'Processing...';
        inputArea.classList.add('disabled');
        sendBtn.disabled = true;
    }

    lastPendingHuman = pendingHuman;
}

function renderTranscript(history) {
    transcriptEl.innerHTML = '';
    const reversed = [...history].reverse();
    reversed.forEach(r => {
        const item = document.createElement('div');
        const isMe = r.speaker === myName;
        item.className = 'msg' + (isMe ? ' msg-user' : '');

        const agent = agentsData.find(a => a.name === r.speaker);
        const color = agent ? agent.color : '#6b7a94';
        const toneColor = r.tone === 'positive' ? 'var(--tone-positive)'
                        : r.tone === 'negative' ? 'var(--tone-negative)'
                        : 'var(--tone-neutral)';
        item.style.borderLeftColor = color;

        const youTag = isMe ? ' <span class="you-tag">(You)</span>' : '';
        const humanTag = r.is_human ? ' <span style="opacity:0.5;font-size:11px;">[human]</span>' : '';
        item.innerHTML = `
            <div class="meta">
                <span class="speaker" style="color:${color}">${escapeHtml(r.speaker)}</span>${youTag}${humanTag}
                <span class="arrow">\u2192</span>
                <span class="target">${escapeHtml(r.target || 'all')}</span>
                <span class="tone-dot" style="background:${toneColor}"></span>
                <span class="tone-label">${escapeHtml(r.emotion || '')}</span>
            </div>
            <div class="text">${escapeHtml(r.reply)}</div>
        `;
        transcriptEl.appendChild(item);
    });
    transcriptEl.scrollTop = 0;
}

function renderGraph(imageUrl) {
    if (imageUrl) {
        graphImg.src = imageUrl + '?t=' + Date.now();
        graphImg.style.display = 'block';
        if (graphPlaceholder) graphPlaceholder.style.display = 'none';
    }
}

function renderParticipants(agents, connectedHumans) {
    participantsBar.innerHTML = '';
    (agents || []).forEach(a => {
        if (a.is_observer) return;
        const badge = document.createElement('div');
        badge.className = 'agent-badge';
        const isHuman = a.is_human;
        const isOnline = connectedHumans && connectedHumans[a.name] && connectedHumans[a.name].online;
        const dotColor = isHuman
            ? (isOnline ? 'var(--tone-positive)' : 'var(--tone-negative)')
            : 'var(--accent)';
        const typeLabel = isHuman ? (isOnline ? 'online' : 'offline') : 'AI';
        badge.innerHTML = `
            <span class="exp-participant-dot" style="background:${dotColor}"></span>
            <span class="agent-badge-name">${escapeHtml(a.name)}</span>
            <span class="agent-badge-nature">(${escapeHtml(a.nature)}) ${typeLabel}</span>
        `;
        if (a.name === myName) badge.style.border = '1px solid var(--accent)';
        participantsBar.appendChild(badge);
    });
}

// ==================== SEND MESSAGE ====================

sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    const reply = messageInput.value.trim();
    if (!reply || sendBtn.disabled) return;

    sendBtn.disabled = true;
    sendBtn.textContent = 'Sending...';

    try {
        const res = await fetch('/api/experiment/human_message', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                speaker: myName,
                reply: reply,
                target: targetSelect.value,
                tone: toneSelect.value,
            }),
        });
        const data = await res.json();
        if (!data.ok) {
            statusEl.textContent = 'Error: ' + data.error;
            return;
        }

        messageInput.value = '';
        renderAll(data);

        // Auto-advance: trigger agent turns until next human
        await advanceUntilHumanTurn();
    } catch (e) {
        statusEl.textContent = 'Send error: ' + e.message;
    } finally {
        sendBtn.textContent = 'Send';
        // sendBtn.disabled is managed by renderTurnBanner via polling
    }
}

// ==================== AUTO-ADVANCE ====================

async function advanceUntilHumanTurn() {
    if (isAdvancing) return;
    isAdvancing = true;

    try {
        for (let i = 0; i < 20; i++) {
            await new Promise(r => setTimeout(r, 800));

            const res = await fetch('/api/experiment/step', { method: 'POST' });
            const data = await res.json();
            renderAll(data);

            if (data.waiting_for) break;
            if (!data.ok) break;
        }
    } catch (e) {
        console.error('Auto-advance error:', e);
    } finally {
        isAdvancing = false;
    }
}

// ==================== NOTIFICATION SOUND ====================

function playNotification() {
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = 800;
        gain.gain.value = 0.3;
        osc.start();
        osc.stop(ctx.currentTime + 0.15);
    } catch (e) { /* ignore audio errors */ }
}

// ==================== UTILS ====================

function escapeHtml(s) {
    return String(s || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
