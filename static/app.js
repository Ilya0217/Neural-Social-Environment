// ==================== DOM ELEMENTS ====================
const el = (id) => document.getElementById(id);

// Wizard elements
const setupWizard = el('setupWizard');
const step1 = el('step1');
const step2 = el('step2');
const agentCountSlider = el('agentCountSlider');
const sliderValue = el('sliderValue');
const wizardEnvSelect = el('wizardEnvSelect');
const toStep2Btn = el('toStep2Btn');
const backToStep1Btn = el('backToStep1Btn');
const prevAgentBtn = el('prevAgentBtn');
const nextAgentBtn = el('nextAgentBtn');
const startDialogueBtn = el('startDialogueBtn');
const currentAgentNum = el('currentAgentNum');
const totalAgentNum = el('totalAgentNum');
const agentConfigTitle = el('agentConfigTitle');
const agentCards = el('agentCards');

// Agent form fields
const agentNameInput = el('agentName');
const agentNatureSelect = el('agentNature');
const agentColorInput = el('agentColor');
// Basic info
const qBackground = el('q_background');
const qMotivation = el('q_motivation');
// Social behavior patterns
const qSocialStyle = el('q_social_style');
const qConflictApproach = el('q_conflict_approach');
const qDecisionStyle = el('q_decision_style');
const qTrustLevel = el('q_trust_level');
const qCooperationStyle = el('q_cooperation_style');
const qEmotionalOpenness = el('q_emotional_openness');
const qLeadershipTendency = el('q_leadership_tendency');
const qCriticismReaction = el('q_criticism_reaction');
const qGroupDynamics = el('q_group_dynamics');
// Communication style
const qSpeechFlaws = el('q_speech_flaws');
const qFavoriteTopics = el('q_favorite_topics');
const qStressReaction = el('q_stress_reaction');
const qSignaturePhrase = el('q_signature_phrase');

// Main app elements
const mainApp = el('mainApp');
const transcriptEl = el('transcript');
const typingEl = el('typing');
const graphImg = el('graphImg');
const hypsEl = el('hyps');
const metricsEl = el('metrics');
const statusEl = el('status');
const stepBtn = el('stepBtn');
const newSessionBtn = el('newSessionBtn');
const agentBadgesEl = el('agentBadges');

// Scientific analysis elements
const scientificReportEl = el('scientificReport');
const scientificHypothesesEl = el('scientificHypotheses');
const downloadValidationBtn = el('downloadValidation');

// Tab elements
const tabBtns = document.querySelectorAll('.tab-btn');
const dialogueTab = el('dialogueTab');
const scientificTab = el('scientificTab');

// User participation elements
const participateBtn = el('participateBtn');
const userInputModal = el('userInputModal');
const userTargetSelect = el('userTargetSelect');
const userReplyInput = el('userReplyInput');
const userToneSelect = el('userToneSelect');
const userSendBtn = el('userSendBtn');
const userCancelBtn = el('userCancelBtn');
const joinAsParticipantCheckbox = el('joinAsParticipant');

// ==================== STATE ====================
let agentCount = 3;
let currentAgentIndex = 0;
let agents = [];
let running = false;
let userParticipating = false;

// Extended default agent configurations for up to 20 agents
const DEFAULT_AGENT_NAMES = [
  'Alex', 'Jordan', 'Sam', 'Riley', 'Casey',
  'Morgan', 'Taylor', 'Quinn', 'Avery', 'Parker',
  'Skyler', 'Drew', 'Blake', 'Charlie', 'Emery',
  'Finley', 'Harper', 'Jamie', 'Kendall', 'Logan'
];

const DEFAULT_COLORS = [
  '#5b8cff', '#ef4444', '#10b981', '#f59e0b', '#a855f7',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
  '#14b8a6', '#eab308', '#e11d48', '#8b5cf6', '#22c55e',
  '#0ea5e9', '#d946ef', '#fb923c', '#4ade80', '#f43f5e'
];

// Default roles for agent initialization
const DEFAULT_ROLES = [
  'Explorer', 'Critic', 'Facilitator', 'Analyst', 'Visionary',
  'Scientist', 'Manager', 'Designer', 'Developer', 'Researcher',
  'Strategist', 'Innovator', 'Mentor', 'Coordinator', 'Specialist',
  'Architect', 'Planner', 'Advisor', 'Engineer', 'Consultant'
];

// ==================== TAB LOGIC ====================

tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const tabName = btn.dataset.tab;
    
    // Update button states
    tabBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    
    // Update tab visibility
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    const targetTab = document.getElementById(tabName + 'Tab');
    if (targetTab) {
      targetTab.classList.add('active');
      targetTab.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// ==================== SLIDER LOGIC ====================

function updateSliderBackground() {
  const min = parseInt(agentCountSlider.min);
  const max = parseInt(agentCountSlider.max);
  const val = parseInt(agentCountSlider.value);
  const percentage = ((val - min) / (max - min)) * 100;
  agentCountSlider.style.background = `linear-gradient(to right, var(--accent) 0%, var(--accent) ${percentage}%, var(--border) ${percentage}%, var(--border) 100%)`;
}

agentCountSlider.addEventListener('input', () => {
  agentCount = parseInt(agentCountSlider.value, 10);
  sliderValue.textContent = agentCount;
  updateSliderBackground();
});

// Initialize slider
updateSliderBackground();

// ==================== WIZARD LOGIC ====================

// Go to step 2
toStep2Btn.addEventListener('click', () => {
  // Initialize agents array with defaults
  agents = [];
  for (let i = 0; i < agentCount; i++) {
    agents.push({
      name: DEFAULT_AGENT_NAMES[i] || `Agent${i + 1}`,
      nature: DEFAULT_ROLES[i % DEFAULT_ROLES.length],
      color: DEFAULT_COLORS[i] || DEFAULT_COLORS[i % DEFAULT_COLORS.length],
      questionnaire: {}
    });
  }
  
  currentAgentIndex = 0;
  totalAgentNum.textContent = agentCount;
  
  step1.classList.remove('active');
  step2.classList.add('active');
  
  updateAgentForm();
  renderAgentCards();
});

// Back to step 1
backToStep1Btn.addEventListener('click', () => {
  step2.classList.remove('active');
  step1.classList.add('active');
});

// Previous agent
prevAgentBtn.addEventListener('click', () => {
  saveCurrentAgent();
  if (currentAgentIndex > 0) {
    currentAgentIndex--;
    updateAgentForm();
    renderAgentCards();
  }
});

// Save current agent and go to next
nextAgentBtn.addEventListener('click', () => {
  saveCurrentAgent();
  
  if (currentAgentIndex < agentCount - 1) {
    currentAgentIndex++;
    updateAgentForm();
    renderAgentCards();
  }
});

// Start dialogue
startDialogueBtn.addEventListener('click', async () => {
  saveCurrentAgent();
  
  // Validate all agents have names
  for (let i = 0; i < agents.length; i++) {
    if (!agents[i].name.trim()) {
      alert(`Please enter a name for Agent ${i + 1}`);
      currentAgentIndex = i;
      updateAgentForm();
      renderAgentCards();
      return;
    }
  }
  
  // Check for duplicate names
  const names = agents.map(a => a.name.toLowerCase().trim());
  const uniqueNames = new Set(names);
  if (uniqueNames.size !== names.length) {
    alert('Each agent must have a unique name');
    return;
  }
  
  await startSession();
});

// New session button
newSessionBtn.addEventListener('click', () => {
  running = false;
  userParticipating = false;
  stepBtn.disabled = true;
  if (participateBtn) participateBtn.style.display = 'none';
  mainApp.style.display = 'none';
  setupWizard.style.display = 'flex';
  step2.classList.remove('active');
  step1.classList.add('active');
});

function saveCurrentAgent() {
  agents[currentAgentIndex] = {
    name: agentNameInput.value.trim() || agents[currentAgentIndex].name,
    nature: agentNatureSelect.value,
    color: agentColorInput.value,
    questionnaire: {
      // Basic info
      background: qBackground.value.trim(),
      motivation: qMotivation.value.trim(),
      // Social behavior patterns
      social_style: qSocialStyle.value.trim(),
      conflict_approach: qConflictApproach.value.trim(),
      decision_style: qDecisionStyle.value.trim(),
      trust_level: qTrustLevel.value.trim(),
      cooperation_style: qCooperationStyle.value.trim(),
      emotional_openness: qEmotionalOpenness.value.trim(),
      leadership_tendency: qLeadershipTendency.value.trim(),
      criticism_reaction: qCriticismReaction.value.trim(),
      group_dynamics: qGroupDynamics.value.trim(),
      // Communication style
      speech_flaws: qSpeechFlaws.value.trim(),
      favorite_topics: qFavoriteTopics.value.trim(),
      stress_reaction: qStressReaction.value.trim(),
      signature_phrase: qSignaturePhrase.value.trim()
    }
  };
  renderAgentCards();
}

function updateAgentForm() {
  const agent = agents[currentAgentIndex];
  
  currentAgentNum.textContent = currentAgentIndex + 1;
  agentConfigTitle.textContent = `Configure Agent ${currentAgentIndex + 1}`;
  
  agentNameInput.value = agent.name;
  agentNatureSelect.value = agent.nature;
  agentColorInput.value = agent.color;
  
  const q = agent.questionnaire || {};
  qBackground.value = q.background || '';
  qMotivation.value = q.motivation || '';
  qSocialStyle.value = q.social_style || '';
  qConflictApproach.value = q.conflict_approach || '';
  qDecisionStyle.value = q.decision_style || '';
  qTrustLevel.value = q.trust_level || '';
  qCooperationStyle.value = q.cooperation_style || '';
  qEmotionalOpenness.value = q.emotional_openness || '';
  qLeadershipTendency.value = q.leadership_tendency || '';
  qCriticismReaction.value = q.criticism_reaction || '';
  qGroupDynamics.value = q.group_dynamics || '';
  qSpeechFlaws.value = q.speech_flaws || '';
  qFavoriteTopics.value = q.favorite_topics || '';
  qStressReaction.value = q.stress_reaction || '';
  qSignaturePhrase.value = q.signature_phrase || '';
  
  // Update button visibility
  // Show/hide prev button
  if (currentAgentIndex === 0) {
    prevAgentBtn.style.display = 'none';
  } else {
    prevAgentBtn.style.display = 'inline-flex';
  }
  
  // Show/hide next/start buttons
  if (currentAgentIndex === agentCount - 1) {
    nextAgentBtn.style.display = 'none';
    startDialogueBtn.style.display = 'inline-flex';
  } else {
    nextAgentBtn.style.display = 'inline-flex';
    startDialogueBtn.style.display = 'none';
  }
  
  // Scroll form to top
  const scrollContainer = document.querySelector('.wizard-content-scroll');
  if (scrollContainer) {
    scrollContainer.scrollTop = 0;
  }
}

function renderAgentCards() {
  agentCards.innerHTML = '';
  
  agents.forEach((agent, idx) => {
    const card = document.createElement('div');
    card.className = 'agent-card' + (idx === currentAgentIndex ? ' active' : '');
    card.innerHTML = `
      <div class="agent-card-dot" style="background: ${agent.color}"></div>
      <div>
        <div class="agent-card-name">${agent.name || `Agent ${idx + 1}`}</div>
        <div class="agent-card-nature">${agent.nature || 'No role'}</div>
      </div>
    `;
    card.addEventListener('click', () => {
      saveCurrentAgent();
      currentAgentIndex = idx;
      updateAgentForm();
      renderAgentCards();
    });
    agentCards.appendChild(card);
  });
}

// ==================== MAIN APP LOGIC ====================

async function api(path, method = 'GET', body) {
  const resp = await fetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  return resp.json();
}

async function startSession() {
  const envIndex = parseInt(wizardEnvSelect.value || '0', 10);
  
  statusEl.textContent = 'Initializing session...';
  
  // Show loading overlay
  const loadingEl = document.createElement('div');
  loadingEl.className = 'loading-overlay';
  loadingEl.innerHTML = `
    <div class="loading-spinner"></div>
    <div class="loading-text">Generating agent profiles via LLM</div>
    <div class="loading-agent" id="loadingAgent">Initializing<span class="loading-dots"><span></span><span></span><span></span></span></div>
  `;
  document.body.appendChild(loadingEl);
  
  // Animate agent names in loading
  const agentNames = agents.map(a => `${a.name} (${a.nature})`);
  let loadIdx = 0;
  const loadingInterval = setInterval(() => {
    const el = document.getElementById('loadingAgent');
    if (el && loadIdx < agentNames.length) {
      el.innerHTML = `${agentNames[loadIdx]}<span class="loading-dots"><span></span><span></span><span></span></span>`;
      loadIdx++;
    } else if (el) {
      el.innerHTML = `Finalizing<span class="loading-dots"><span></span><span></span><span></span></span>`;
    }
  }, 2500);
  
  try {
    const joinChecked = joinAsParticipantCheckbox ? joinAsParticipantCheckbox.checked : false;
    const data = await api('/api/start', 'POST', {
      env_index: envIndex,
      join_as_participant: joinChecked,
      agents: agents.map(a => ({
        name: a.name,
        nature: a.nature,
        color: a.color,
        questionnaire: a.questionnaire
      }))
    });
    
    clearInterval(loadingInterval);
    loadingEl.remove();
    
    if (!data.ok) throw new Error(data.error || 'Failed to start session');
    
    running = true;
    stepBtn.disabled = false;
    
    // Hide wizard, show main app
    setupWizard.style.display = 'none';
    mainApp.style.display = 'block';
    
    // Render agent badges
    renderAgentBadges();
    
    // Clear previous data
    renderHistory([]);
    renderHyps([]);
    graphImg.src = '';
    graphImg.style.display = 'none';
    const gp = document.getElementById('graphPlaceholder');
    if (gp) gp.style.display = 'flex';
    renderMetrics('');
    renderScientificReport('');
    renderScientificHypotheses([]);
    
    // Hide validation button on new session
    if (downloadValidationBtn) {
      downloadValidationBtn.style.display = 'none';
    }
    
    // Switch to dialogue tab
    tabBtns.forEach(b => b.classList.remove('active'));
    document.querySelector('.tab-btn[data-tab="dialogue"]').classList.add('active');
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    dialogueTab.classList.add('active');
    
    // Use agents_data if available (includes big_five)
    if (data.agents_data) {
      agents = data.agents_data;
      renderAgentBadges();
    }
    
    // Set user participation state
    userParticipating = !!data.user_participating;
    if (participateBtn) {
      participateBtn.style.display = userParticipating ? 'inline-flex' : 'none';
    }

    statusEl.textContent = 'Session started. Click "Next message".';
  } catch (e) {
    clearInterval(loadingInterval);
    loadingEl.remove();
    console.error(e);
    statusEl.textContent = 'Start error: ' + e.message;
  }
}

function renderAgentBadges() {
  agentBadgesEl.innerHTML = '';
  agents.forEach(agent => {
    const badge = document.createElement('div');
    const isObs = agent.is_observer;
    badge.className = 'agent-badge' + (isObs ? ' agent-badge-observer' : '');
    const b5 = agent.big_five;
    const b5Tooltip = b5 
      ? `O:${b5.openness} C:${b5.conscientiousness} E:${b5.extraversion} A:${b5.agreeableness} N:${b5.neuroticism}`
      : '';
    const b5Html = b5
      ? `<span class="agent-badge-b5" title="Big Five (OCEAN)">
          <span class="b5-bar" style="--val:${b5.openness}" title="Openness ${b5.openness}">O</span>
          <span class="b5-bar" style="--val:${b5.conscientiousness}" title="Conscientiousness ${b5.conscientiousness}">C</span>
          <span class="b5-bar" style="--val:${b5.extraversion}" title="Extraversion ${b5.extraversion}">E</span>
          <span class="b5-bar" style="--val:${b5.agreeableness}" title="Agreeableness ${b5.agreeableness}">A</span>
          <span class="b5-bar" style="--val:${b5.neuroticism}" title="Neuroticism ${b5.neuroticism}">N</span>
        </span>`
      : '';
    const obsIcon = isObs ? '🔭 ' : '';
    badge.innerHTML = `
      <div class="agent-badge-dot" style="background:${agent.color}${isObs ? ';opacity:.5' : ''}"></div>
      <span class="agent-badge-name">${obsIcon}${agent.name}</span>
      <span class="agent-badge-nature">(${agent.nature})</span>
      ${b5Html}
    `;
    if (b5Tooltip) badge.title = `Big Five: ${b5Tooltip}`;
    if (isObs) badge.title = 'Non-participating observer agent (methodological triangulation)';
    agentBadgesEl.appendChild(badge);
  });
}

function renderHistory(history) {
  transcriptEl.innerHTML = '';
  const reversedHistory = [...history].reverse();
  reversedHistory.forEach(r => {
    const item = document.createElement('div');
    const isUser = r.speaker === 'User' || r.is_human;
    item.className = 'msg' + (isUser ? ' msg-user' : '');
    const tgt = r.target || 'all';

    // Find agent color
    const agent = agents.find(a => a.name === r.speaker);
    const color = agent ? agent.color : '#6b7a94';

    const toneColor = r.tone === 'positive' ? 'var(--tone-positive)'
                   : r.tone === 'negative' ? 'var(--tone-negative)'
                   : 'var(--tone-neutral)';
    item.style.borderLeftColor = color;
    const youTag = isUser ? ' <span class="you-tag">(You)</span>' : '';
    item.innerHTML = `
      <div class="meta">
        <span class="speaker" style="color:${color}">${r.speaker}</span>${youTag}
        <span class="arrow">→</span>
        <span class="target">${tgt}</span>
        <span class="tone-dot" style="background:${toneColor}"></span>
        <span class="tone-label">${r.emotion}</span>
      </div>
      <div class="text">${escapeHtml(r.reply)}</div>
    `;
    transcriptEl.appendChild(item);
  });
  transcriptEl.scrollTop = 0;
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
      const tableLines = [];
      while (i < lines.length && /^\|.*\|$/.test(lines[i])) {
        tableLines.push(lines[i]); i++;
      }
      if (tableLines.length >= 2) {
        const header = tableLines[0];
        const rows = tableLines.slice(2);
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

// ==================== SCIENTIFIC ANALYSIS RENDERING ====================

function renderScientificReport(reportMd) {
  if (!scientificReportEl) return;
  if (!reportMd) {
    scientificReportEl.innerHTML = '<p class="placeholder">Scientific analysis will appear after 5 dialogue turns.</p>';
    return;
  }
  
  // Convert markdown to HTML (simple conversion)
  const html = convertMarkdownToHtml(reportMd);
  scientificReportEl.innerHTML = html;
}

function renderScientificHypotheses(hypotheses) {
  if (!scientificHypothesesEl) return;
  if (!hypotheses || hypotheses.length === 0) {
    scientificHypothesesEl.innerHTML = '<p class="placeholder">Scientific hypotheses will appear after analysis.</p>';
    return;
  }
  
  scientificHypothesesEl.innerHTML = '';
  
  hypotheses.forEach(hyp => {
    const card = document.createElement('div');
    card.className = 'sci-hyp-card';
    card.innerHTML = `
      <div class="sci-hyp-header">
        <span class="sci-hyp-category">${escapeHtml(hyp.category || 'General')}</span>
        <span class="sci-hyp-framework">${escapeHtml(hyp.framework || '')}</span>
      </div>
      <div class="sci-hyp-finding">${escapeHtml(hyp.finding || '')}</div>
      <div class="sci-hyp-recommendation">${escapeHtml(hyp.recommendation || '')}</div>
      <div class="sci-hyp-reference">${escapeHtml(hyp.reference || '')}</div>
    `;
    scientificHypothesesEl.appendChild(card);
  });
}

function convertMarkdownToHtml(md) {
  const lines = md.split(/\r?\n/);
  const out = [];
  let i = 0;
  let inList = false;
  
  while (i < lines.length) {
    const line = lines[i];
    
    // Headers
    if (/^###\s+/.test(line)) {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push('<h3>' + escapeHtml(line.replace(/^###\s+/, '')) + '</h3>');
      i++; continue;
    }
    if (/^##\s+/.test(line)) {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push('<h2>' + escapeHtml(line.replace(/^##\s+/, '')) + '</h2>');
      i++; continue;
    }
    if (/^#\s+/.test(line)) {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push('<h1>' + escapeHtml(line.replace(/^#\s+/, '')) + '</h1>');
      i++; continue;
    }
    
    // Tables
    if (/^\|.*\|$/.test(line)) {
      if (inList) { out.push('</ul>'); inList = false; }
      const tableLines = [];
      while (i < lines.length && /^\|.*\|$/.test(lines[i])) {
        tableLines.push(lines[i]); i++;
      }
      if (tableLines.length >= 2) {
        const header = tableLines[0];
        const rows = tableLines.slice(2);
        const headers = header.split('|').slice(1, -1).map(s => s.trim());
        out.push('<div class="tbl-wrap"><table class="metrics-table"><thead><tr>' + 
          headers.map(h => '<th>' + escapeHtml(h) + '</th>').join('') + 
          '</tr></thead><tbody>');
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
    
    // List items
    if (/^[-*]\s+/.test(line)) {
      if (!inList) { out.push('<ul>'); inList = true; }
      // Handle bold text
      let content = line.replace(/^[-*]\s+/, '');
      content = content.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      out.push('<li>' + escapeHtml(content).replace(/&lt;strong&gt;/g, '<strong>').replace(/&lt;\/strong&gt;/g, '</strong>') + '</li>');
      i++; continue;
    }
    
    // Horizontal rule
    if (/^---+$/.test(line)) {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push('<hr>');
      i++; continue;
    }
    
    // Empty line
    if (line.trim().length === 0) {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push('<div class="spacer"></div>');
      i++; continue;
    }
    
    // Regular paragraph with bold handling
    if (inList) { out.push('</ul>'); inList = false; }
    let content = line;
    content = content.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    out.push('<p>' + escapeHtml(content).replace(/&lt;strong&gt;/g, '<strong>').replace(/&lt;\/strong&gt;/g, '</strong>') + '</p>');
    i++;
  }
  
  if (inList) out.push('</ul>');
  return out.join('\n');
}

// Step button handler
stepBtn.addEventListener('click', async () => {
  if (!running) return;
  stepBtn.disabled = true;
  statusEl.textContent = 'Generating message...';
  
  try {
    typingEl && (typingEl.style.display = 'flex');
    const data = await api('/api/step', 'POST', {});
    
    if (!data.ok) throw new Error(data.error || 'failed');
    
    renderHistory(data.history || []);
    
    if (data.image_url) {
      graphImg.src = data.image_url + '?t=' + Date.now();
      graphImg.style.display = 'block';
      const placeholder = document.getElementById('graphPlaceholder');
      if (placeholder) placeholder.style.display = 'none';
    }
    if (data.hypotheses) {
      renderHyps(data.hypotheses);
    }
    if (data.metrics_md) {
      renderMetrics(data.metrics_md);
    }
    
    // Render scientific analysis
    if (data.scientific_report) {
      renderScientificReport(data.scientific_report);
    }
    if (data.scientific_hypotheses) {
      renderScientificHypotheses(data.scientific_hypotheses);
    }
    
    statusEl.textContent = `Turn: ${data.turn}`;

    // Notify if agent addressed User
    if (data.addressed_user && userParticipating && participateBtn) {
      participateBtn.classList.add('btn-participate-pulse');
      statusEl.textContent = `${data.record.speaker} addressed you — click Participate to respond`;
    } else if (participateBtn) {
      participateBtn.classList.remove('btn-participate-pulse');
    }

    // Check for validation button visibility
    checkValidationButton(data);
  } catch (e) {
    console.error(e);
    statusEl.textContent = 'Step error: ' + e.message;
  } finally {
    typingEl && (typingEl.style.display = 'none');
    stepBtn.disabled = false;
  }
});

// Helper function to check and show validation button
function checkValidationButton(data) {
  if (downloadValidationBtn) {
    if (data && (data.validation_report || data.has_validation_report)) {
      downloadValidationBtn.style.display = 'inline-block';
    } else {
      downloadValidationBtn.style.display = 'none';
    }
  }
}

// ==================== ADVANCED ANALYTICS ====================
const loadAdvancedBtn = document.getElementById('loadAdvancedBtn');
if (loadAdvancedBtn) {
  loadAdvancedBtn.addEventListener('click', async () => {
    loadAdvancedBtn.disabled = true;
    loadAdvancedBtn.textContent = 'Loading...';
    try {
      const res = await fetch('/api/advanced');
      const data = await res.json();
      const reportDiv = document.getElementById('advancedReport');
      const metricsDiv = document.getElementById('advancedMetrics');
      if (data.ok && data.report) {
        reportDiv.innerHTML = renderMarkdown(data.report);
        const d = data.data || {};
        let html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:1rem;">';
        html += `<div class="stat-card"><div class="stat-value">${(d.contagion_rate*100||0).toFixed(0)}%</div><div class="stat-label">Contagion Rate</div></div>`;
        html += `<div class="stat-card"><div class="stat-value">${(d.group_lsm||0).toFixed(2)}</div><div class="stat-label">Group LSM</div></div>`;
        html += `<div class="stat-card"><div class="stat-value">${(d.discourse_coherence||0).toFixed(2)}</div><div class="stat-label">Coherence</div></div>`;
        html += `<div class="stat-card"><div class="stat-value">${((d.thread_continuity||0)*100).toFixed(0)}%</div><div class="stat-label">Thread Continuity</div></div>`;
        html += '</div>';
        if (d.most_contagious) html += `<p style="padding:0 1rem;font-size:12px;color:var(--text-dim)">Most contagious: <strong>${d.most_contagious}</strong></p>`;
        if (d.most_susceptible) html += `<p style="padding:0 1rem;font-size:12px;color:var(--text-dim)">Most susceptible: <strong>${d.most_susceptible}</strong></p>`;
        if (d.topic_drift_points && d.topic_drift_points.length) html += `<p style="padding:0 1rem;font-size:12px;color:var(--tone-negative)">Topic drift at turns: ${d.topic_drift_points.join(', ')}</p>`;
        metricsDiv.innerHTML = html;
      } else {
        reportDiv.innerHTML = `<p style="color:var(--text-secondary);padding:2rem;">${data.message || 'No data'}</p>`;
      }
    } catch(e) { console.error(e); }
    loadAdvancedBtn.disabled = false;
    loadAdvancedBtn.textContent = 'Load Analysis';
  });
}

// ==================== OBSERVER AGENT ====================
const runObserverBtn = document.getElementById('runObserverBtn');
if (runObserverBtn) {
  runObserverBtn.addEventListener('click', async () => {
    runObserverBtn.disabled = true;
    runObserverBtn.textContent = 'Analyzing...';
    try {
      const res = await fetch('/api/observer', { method: 'POST' });
      const data = await res.json();
      const reportDiv = document.getElementById('observerReport');
      const summaryDiv = document.getElementById('triangulationSummary');
      if (data.ok && data.report) {
        reportDiv.innerHTML = renderMarkdown(data.report);
        // Render triangulation summary
        const s = data.triangulation_summary || {};
        let html = '<div style="padding: 1rem;">';
        html += `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem;">`;
        html += `<div class="stat-card"><div class="stat-value">${(data.convergence_score * 100).toFixed(0)}%</div><div class="stat-label">Convergence</div></div>`;
        html += `<div class="stat-card"><div class="stat-value">${(data.agreements || []).length}</div><div class="stat-label">Agreements</div></div>`;
        html += `<div class="stat-card"><div class="stat-value">${(data.divergences || []).length}</div><div class="stat-label">Divergences</div></div>`;
        html += `</div>`;
        if (data.novel_insights && data.novel_insights.length) {
          html += '<h3>Novel Insights</h3><ul>';
          data.novel_insights.forEach(i => html += `<li>${i}</li>`);
          html += '</ul>';
        }
        if (s.observations > 1) {
          html += `<h3>Longitudinal Trend</h3>`;
          html += `<p>Observations: ${s.observations} | Avg convergence: ${(s.avg_convergence * 100).toFixed(0)}%</p>`;
          html += `<p>Range: ${(s.min_convergence * 100).toFixed(0)}% – ${(s.max_convergence * 100).toFixed(0)}%</p>`;
        }
        html += '</div>';
        summaryDiv.innerHTML = html;
      } else {
        reportDiv.innerHTML = `<p style="color:var(--text-secondary);padding:2rem;">${data.error || 'No data'}</p>`;
      }
    } catch(e) {
      console.error(e);
    }
    runObserverBtn.disabled = false;
    runObserverBtn.textContent = 'Run Observer Analysis';
  });
}

// ==================== USER PARTICIPATION ====================

if (participateBtn) {
  participateBtn.addEventListener('click', () => {
    // Заполнить список агентов (все кроме User и Observer)
    userTargetSelect.innerHTML = '';
    agents.filter(a => a.name !== 'User' && !a.is_observer).forEach(a => {
      const opt = document.createElement('option');
      opt.value = a.name;
      opt.textContent = a.name;
      userTargetSelect.appendChild(opt);
    });
    userReplyInput.value = '';
    userToneSelect.value = 'auto';
    userInputModal.style.display = 'flex';
    userReplyInput.focus();
    participateBtn.classList.remove('btn-participate-pulse');
  });
}

if (userCancelBtn) {
  userCancelBtn.addEventListener('click', () => {
    userInputModal.style.display = 'none';
  });
}

if (userSendBtn) {
  userSendBtn.addEventListener('click', async () => {
    const reply = userReplyInput.value.trim();
    if (!reply) return;

    const target = userTargetSelect.value;
    const tone = userToneSelect.value;

    userSendBtn.disabled = true;
    userSendBtn.textContent = 'Sending...';

    try {
      const data = await api('/api/user_message', 'POST', { reply, target, tone });
      if (!data.ok) throw new Error(data.error || 'Failed');

      userInputModal.style.display = 'none';
      renderHistory(data.history || []);
      statusEl.textContent = `Turn: ${data.turn}`;

      if (data.image_url) {
        graphImg.src = data.image_url + '?t=' + Date.now();
        graphImg.style.display = 'block';
        const placeholder = document.getElementById('graphPlaceholder');
        if (placeholder) placeholder.style.display = 'none';
      }
      if (data.hypotheses) renderHyps(data.hypotheses);
      if (data.metrics_md) renderMetrics(data.metrics_md);
      if (data.scientific_report) renderScientificReport(data.scientific_report);
      if (data.scientific_hypotheses) renderScientificHypotheses(data.scientific_hypotheses);
      checkValidationButton(data);
    } catch (e) {
      console.error(e);
      statusEl.textContent = 'Error: ' + e.message;
    } finally {
      userSendBtn.disabled = false;
      userSendBtn.textContent = 'Send';
    }
  });
}

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && userInputModal && userInputModal.style.display === 'flex') {
    userInputModal.style.display = 'none';
  }
});

// Simple markdown to HTML renderer
function renderMarkdown(md) {
  if (!md) return '';
  return md
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
    .replace(/^---$/gm, '<hr>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>')
    .replace(/^(.+)$/gm, (m) => m.startsWith('<') ? m : `<p>${m}</p>`);
}
