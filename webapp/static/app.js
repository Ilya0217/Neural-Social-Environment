// ==================== DOM ELEMENTS ====================
const el = (id) => document.getElementById(id);

// Wizard elements
const setupWizard = el('setupWizard');
const step0 = el('step0');
const step1 = el('step1');
const step2 = el('step2');
const providerCards = document.querySelectorAll('.provider-card');
const apiTokenInput = el('apiTokenInput');
const toStep1Btn = el('toStep1Btn');
const backToStep0Btn = el('backToStep0Btn');
const providerError = el('providerError');
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
const observerHypothesisChecksEl = el('observerHypothesisChecks');
const triangulationSummaryEl = el('triangulationSummary');
const observerReportEl = el('observerReport');

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

// ==================== LANGUAGE TOGGLE ====================
// Persist выбор языка между сессиями (работает и в визарде, и в шапке)
window.dialogLanguage = localStorage.getItem('dialogLanguage') || 'ru';
function applyLangButtons() {
  document.querySelectorAll('#langToggle .lang-btn, #langToggleWizard .lang-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.lang === window.dialogLanguage);
  });
}
document.addEventListener('DOMContentLoaded', () => {
  applyLangButtons();
  document.querySelectorAll('#langToggle .lang-btn, #langToggleWizard .lang-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      window.dialogLanguage = btn.dataset.lang;
      localStorage.setItem('dialogLanguage', window.dialogLanguage);
      applyLangButtons();
      const hint = window.dialogLanguage === 'ru'
        ? 'Язык переключён на русский — применится при следующей «Новой сессии».'
        : 'Language switched to English — will apply at the next "New Session".';
      const status = document.getElementById('status');
      if (status) {
        const prev = status.textContent;
        status.textContent = hint;
        setTimeout(() => { if (status.textContent === hint) status.textContent = prev || 'Готово'; }, 4000);
      }
    });
  });
});

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

// --- Step 0: provider + token selection ---
let selectedProvider = localStorage.getItem('selectedProvider') || '';
let apiToken = '';  // Хранится только в памяти, в localStorage не пишем (безопасность).

function showProviderError(msg) {
  if (!providerError) return;
  if (!msg) {
    providerError.style.display = 'none';
    providerError.textContent = '';
  } else {
    providerError.style.display = 'block';
    providerError.textContent = msg;
  }
}

function refreshStep0Continue() {
  const ok = !!selectedProvider && !!apiTokenInput.value.trim();
  if (toStep1Btn) toStep1Btn.disabled = !ok;
  if (ok) showProviderError('');
}

providerCards.forEach(card => {
  card.addEventListener('click', () => {
    providerCards.forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    selectedProvider = card.dataset.provider || '';
    localStorage.setItem('selectedProvider', selectedProvider);
    refreshStep0Continue();
  });
  if (selectedProvider && card.dataset.provider === selectedProvider) {
    card.classList.add('selected');
  }
});

if (apiTokenInput) {
  apiTokenInput.addEventListener('input', refreshStep0Continue);
}

if (toStep1Btn) {
  toStep1Btn.addEventListener('click', () => {
    if (!selectedProvider) {
      showProviderError('Выберите провайдера (ChatGPT или DeepSeek).');
      return;
    }
    apiToken = (apiTokenInput.value || '').trim();
    if (!apiToken) {
      showProviderError('Введите API-токен.');
      return;
    }
    showProviderError('');
    step0.classList.remove('active');
    step1.classList.add('active');
  });
}

if (backToStep0Btn) {
  backToStep0Btn.addEventListener('click', () => {
    step1.classList.remove('active');
    step0.classList.add('active');
  });
}

// Инициализация: если в localStorage остался провайдер — кнопка пока всё равно
// заблокирована, пока пользователь не введёт токен (его в storage не держим).
refreshStep0Continue();

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
  stepBtn.textContent = 'Следующий ход';
  if (participateBtn) participateBtn.style.display = 'none';
  // Сразу чистим старый диалог из DOM, чтобы он не мелькнул при следующем входе.
  renderHistory([]);
  renderHyps([]);
  statusEl.textContent = '';
  mainApp.style.display = 'none';
  setupWizard.style.display = 'flex';
  step2.classList.remove('active');
  step1.classList.remove('active');
  step0.classList.add('active');
  // На повторном входе всегда требуем заново ввести токен.
  if (apiTokenInput) apiTokenInput.value = '';
  apiToken = '';
  refreshStep0Continue();
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
  agentConfigTitle.textContent = `Настройка агента ${currentAgentIndex + 1}`;

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

  statusEl.textContent = 'Запускаю сессию…';

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
      el.innerHTML = `Финализирую<span class="loading-dots"><span></span><span></span><span></span></span>`;
    }
  }, 2500);

  try {
    const joinChecked = joinAsParticipantCheckbox ? joinAsParticipantCheckbox.checked : false;
    const data = await api('/api/start', 'POST', {
      env_index: envIndex,
      join_as_participant: joinChecked,
      language: (window.dialogLanguage || 'ru'),
      provider: selectedProvider || undefined,
      api_key: apiToken || undefined,
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
    renderMetricsCards(null, 0);
    renderVerdicts(null, 0);
    const dlSci = document.getElementById('downloadScientificReport');
    if (dlSci) dlSci.style.display = 'none';
    renderScientificReport('');
    renderScientificHypotheses([]);
    renderObserverAnalysis({ hypothesis_registry: data.hypothesis_registry || [] });
    renderTriangulationSummary({});
    if (observerReportEl) {
      observerReportEl.innerHTML = '<p style="color: var(--text-secondary); padding: 2rem; text-align: center;">Observer checkpoints will appear automatically after 5 dialogue turns.</p>';
    }

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

    statusEl.textContent = 'Сессия запущена. Нажмите «Следующий ход».';
  } catch (e) {
    clearInterval(loadingInterval);
    loadingEl.remove();
    console.error(e);
    statusEl.textContent = 'Ошибка запуска: ' + e.message;
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

// ==================== ПРОВЕРКА НАУЧНЫХ ГИПОТЕЗ H1–H5 ====================
// Те же 5 гипотез, что в backend `_build_verdicts_for_report`.
//   H1. Big Five extraversion → длина реплики (Soto & John 2017; Park et al. 2015)
//   H2. Стадии группового развития (Wheelan 2016; Bonebright 2010)
//   H3. Эффективность многоуровневой валидации (Beauchamp & Childress 2019)
//   H4. Эффективность CoT (Wei et al. 2022)
//   H5. Turn-taking ↔ человеческие корпуса (Levinson & Torreira 2015; Dunbar et al. 2015)

function renderVerdicts(m, turnNo) {
  const grid = document.getElementById('verdictsGrid');
  const turnLabel = document.getElementById('verdictsTurn');
  if (!grid) return;
  if (!m || !m.messages_window) {
    grid.innerHTML = '<div class="metrics-empty">Анализ гипотез появится после 5 ходов диалога</div>';
    if (turnLabel) turnLabel.textContent = '';
    return;
  }
  if (turnLabel) turnLabel.textContent = `· обновлено на ходу ${turnNo}`;

  const verdicts = [];
  const nAgents = Math.max(1, parseInt(m.n_dialogue_agents || Object.keys(m.per_agent || {}).length || 1));

  // ============ H1. Extraversion → reply length ============
  const pairs = [];
  for (const [name, info] of Object.entries(m.per_agent || {})) {
    const bf = info.big_five || null;
    if (bf && info.avg_words != null && info.msgs > 0 && bf.extraversion != null) {
      pairs.push({ name, ext: bf.extraversion, words: info.avg_words });
    }
  }
  let v1;
  if (pairs.length >= 3) {
    // Spearman через ранги
    const sortedExt = [...pairs].sort((a,b)=>a.ext-b.ext);
    const sortedWrd = [...pairs].sort((a,b)=>a.words-b.words);
    const extR = {}, wrdR = {};
    sortedExt.forEach((p,i)=>{ extR[p.name]=i; });
    sortedWrd.forEach((p,i)=>{ wrdR[p.name]=i; });
    const n = pairs.length;
    const d2 = pairs.reduce((s,p)=> s + Math.pow(extR[p.name]-wrdR[p.name], 2), 0);
    const rho = 1 - (6*d2)/(n*(n*n-1));
    const ranks = [...pairs].sort((a,b)=>b.ext-a.ext)
        .map(p=>`${p.name}: E=${p.ext.toFixed(2)}/words=${p.words.toFixed(1)}`).join('; ');
    if (rho >= 0.5)       v1 = { status:'confirmed', claim:'Экстраверсия положительно связана с длиной реплики.',
                                 evidence: `Spearman ρ = ${rho.toFixed(3)} (n=${n}). ${ranks}.` };
    else if (rho > 0)     v1 = { status:'partial',   claim:'Слабый положительный ранговый тренд.',
                                 evidence: `Spearman ρ = ${rho.toFixed(3)} (n=${n}). ${ranks}.` };
    else                  v1 = { status:'refuted',   claim:'Связи не наблюдается или обратная.',
                                 evidence: `Spearman ρ = ${rho.toFixed(3)} (n=${n}). ${ranks}.` };
  } else if (pairs.length === 2) {
    const [a,b] = pairs;
    const sameDir = (a.ext > b.ext) === (a.words > b.words);
    v1 = sameDir
      ? { status:'partial', claim:'Только 2 точки — направление совпадает с прогнозом.',
          evidence: `${a.name}: E=${a.ext.toFixed(2)}/words=${a.words.toFixed(1)}; ${b.name}: E=${b.ext.toFixed(2)}/words=${b.words.toFixed(1)}.` }
      : { status:'refuted', claim:'Только 2 точки — направление противоречит прогнозу.',
          evidence: `${a.name}: E=${a.ext.toFixed(2)}/words=${a.words.toFixed(1)}; ${b.name}: E=${b.ext.toFixed(2)}/words=${b.words.toFixed(1)}.` };
  } else {
    v1 = { status:'pending', claim:'Недостаточно говорящих агентов с OCEAN для проверки H1.',
           evidence: `Найдено ${pairs.length} активных агента(ов) с extraversion.` };
  }
  verdicts.push({
    theory: 'H1. Экстраверсия → длина реплики',
    framework: 'Big Five (BFI-2); лингвистические корреляты',
    ref: 'Soto & John (2017) — BFI-2; Park et al. (2015) — корпус 65 тыс. пользователей',
    ...v1,
  });

  // ============ H2. Стадии группового развития ============
  let stage, v2_status;
  if (m.actionability_rate >= 0.3 && m.reciprocity >= 0.5)             { stage='Performing'; v2_status='confirmed'; }
  else if (m.negative_frac >= 0.3)                                      { stage='Storming';   v2_status='confirmed'; }
  else if (m.question_rate >= 0.3 && m.actionability_rate < 0.2 && m.reciprocity < 0.4)
                                                                        { stage='Forming';    v2_status='confirmed'; }
  else if (m.reciprocity >= 0.4 && m.actionability_rate < 0.3 && m.negative_frac < 0.2)
                                                                        { stage='Norming';    v2_status='confirmed'; }
  else                                                                  { stage='переходное состояние'; v2_status='partial'; }
  verdicts.push({
    theory: 'H2. Воспроизведение стадий группового развития',
    framework: 'Интегрированная модель развития малой группы',
    ref: 'Wheelan (2016); Bonebright (2010)',
    status: v2_status,
    claim: `Текущая стадия: ${stage}.`,
    evidence: `actionability = ${Math.round(m.actionability_rate*100)}%, reciprocity = ${Math.round(m.reciprocity*100)}%, negative = ${Math.round(m.negative_frac*100)}%, questions = ${Math.round(m.question_rate*100)}%.`,
  });

  // ============ H3. Эффективность многоуровневой валидации ============
  const valIssues = m.total_validation_issues || 0;
  const totalMsgs = m.messages_total || 0;
  let v3;
  if (totalMsgs === 0) {
    v3 = { status:'pending', claim:'Нет реплик для оценки.', evidence:'—' };
  } else {
    const rate = valIssues / totalMsgs;
    if (rate < 0.15)
      v3 = { status:'confirmed', claim:'Многоуровневая валидация эффективно фильтрует акты речи.',
             evidence:`${valIssues}/${totalMsgs} реплик помечены валидаторами (${(rate*100).toFixed(1)}%, ниже порога 15%).` };
    else if (rate < 0.35)
      v3 = { status:'partial', claim:'Валидация работает, но процент срабатываний высок.',
             evidence:`${valIssues}/${totalMsgs} реплик помечены валидаторами (${(rate*100).toFixed(1)}%).` };
    else
      v3 = { status:'refuted', claim:'Слишком много замечаний — валидаторы строги или модель отвечает плохо.',
             evidence:`${valIssues}/${totalMsgs} реплик помечены валидаторами (${(rate*100).toFixed(1)}%, выше порога 35%).` };
  }
  verdicts.push({
    theory: 'H3. Эффективность многоуровневой валидации',
    framework: 'Четыре биоэтических принципа как критерии приемлемости речевых актов',
    ref: 'Beauchamp & Childress (2019) — Principles of Biomedical Ethics, 8th ed.',
    ...v3,
  });

  // ============ H4. CoT ============
  let v4;
  if (!m.cot_enabled) {
    v4 = { status:'pending',
           claim:'CoT-промптинг в данной сессии не включён — гипотеза не тестируется.',
           evidence:'Для проверки H4 запустите эксперимент h6_cot.yaml через experiments/run_experiments.py.' };
  } else if (m.actionability_rate >= 0.40)
    v4 = { status:'confirmed', claim:'CoT существенно повышает долю конкретных шагов.',
           evidence:`actionability = ${Math.round(m.actionability_rate*100)}% (≥40% — выше baseline ~20%).` };
  else if (m.actionability_rate >= 0.25)
    v4 = { status:'partial', claim:'CoT даёт умеренное улучшение.',
           evidence:`actionability = ${Math.round(m.actionability_rate*100)}% (выше baseline, но < 40%).` };
  else
    v4 = { status:'refuted', claim:'CoT не привёл к ожидаемому росту actionability.',
           evidence:`actionability = ${Math.round(m.actionability_rate*100)}%.` };
  verdicts.push({
    theory: 'H4. Эффективность Chain-of-Thought-промптинга',
    framework: 'CoT улучшает многошаговые рассуждения LLM',
    ref: 'Wei et al. (2022) — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models',
    ...v4,
  });

  // ============ H5. Turn-taking ↔ человеческие корпуса ============
  const expectedShare = 1.0 / nAgents;
  const threshold = Math.min(0.55, expectedShare * 1.5 + 0.05);
  let v5;
  if (m.top_speaker_share <= threshold && m.targeting_rate >= 0.8)
    v5 = { status:'confirmed',
           claim:'Turn-taking соответствует человеческой норме: равномерно и адресно.',
           evidence:`top speaker ${m.top_speaker} = ${Math.round(m.top_speaker_share*100)}% (порог 1.5/N = ${Math.round(threshold*100)}%); targeting = ${Math.round(m.targeting_rate*100)}%.` };
  else if (m.top_speaker_share > expectedShare * 2)
    v5 = { status:'refuted',
           claim:`Сильное доминирование «${m.top_speaker}» отклоняет от человеческой структуры.`,
           evidence:`${Math.round(m.top_speaker_share*100)}% реплик от одного агента при ожидаемых ${Math.round(expectedShare*100)}% (1/N).` };
  else
    v5 = { status:'partial',
           claim:'Turn-taking близок к норме, но с перекосом.',
           evidence:`top speaker ${m.top_speaker} = ${Math.round(m.top_speaker_share*100)}%; targeting = ${Math.round(m.targeting_rate*100)}%; ожидание 1/N = ${Math.round(expectedShare*100)}%.` };
  verdicts.push({
    theory: 'H5. Соответствие turn-taking человеческим корпусам',
    framework: 'Структура распределения реплик в реальных малых группах',
    ref: 'Levinson & Torreira (2015); Dunbar et al. (2015)',
    ...v5,
  });

  // Render
  const badge = { confirmed: 'ПОДТВЕРЖДЕНА', partial: 'ЧАСТИЧНО', refuted: 'ОТКЛОНЕНА', pending: 'НЕОПРЕДЕЛЕНО' };
  grid.innerHTML = verdicts.map(v => `
    <div class="verdict-card ${v.status}">
      <div class="verdict-head">
        <span class="verdict-theory">${escapeHtml(v.theory)} · ${escapeHtml(v.framework)}</span>
        <span class="verdict-badge ${v.status}">${badge[v.status]}</span>
      </div>
      <div class="verdict-claim">${escapeHtml(v.claim)}</div>
      <div class="verdict-evidence">${escapeHtml(v.evidence)}</div>
      <div class="verdict-ref">${escapeHtml(v.ref)}</div>
    </div>
  `).join('');

  // Покажем кнопку скачивания отчёта, когда verdicts готовы
  const dlBtn = document.getElementById('downloadScientificReport');
  if (dlBtn) dlBtn.style.display = 'inline-flex';
}

// ==================== METRICS CARDS (Russian, clean view) ====================
// Обновляются каждые 5 ходов через should_plot() в DialogueManager.
// Источник данных: data.metrics_summary в ответе /api/step.

function renderMetricsCards(m, turnNo) {
  const grid = document.getElementById('metricsCards');
  const agentsBox = document.getElementById('metricsAgents');
  const turnLabel = document.getElementById('metricsTurn');
  if (!grid) return;

  if (!m || !m.messages_window) {
    grid.innerHTML = '<div class="metrics-empty">Метрики появятся после 5 ходов диалога</div>';
    if (agentsBox) agentsBox.innerHTML = '';
    if (turnLabel) turnLabel.textContent = '';
    return;
  }

  if (turnLabel) {
    turnLabel.textContent = `· обновлено на ходу ${turnNo} (окно ${m.messages_window} реплик)`;
  }

  // Каждая метрика: {label, value, unit, hint, kind, fillPct}
  // kind: 'positive' | 'negative' | 'warning' | 'neutral'
  const cards = [];

  // Средний тон (-1 .. +1)
  const toneKind = m.avg_tone > 0.2 ? 'positive' : (m.avg_tone < -0.2 ? 'negative' : 'neutral');
  cards.push({
    label: 'Средний тон',
    value: (m.avg_tone >= 0 ? '+' : '') + m.avg_tone.toFixed(2),
    hint: m.avg_tone > 0.2 ? 'Позитивная атмосфера' : (m.avg_tone < -0.2 ? 'Напряжение в группе' : 'Нейтрально'),
    kind: toneKind,
    fillPct: ((m.avg_tone + 1) / 2) * 100,
  });

  // Доли тонов
  cards.push({
    label: 'Позитивных реплик',
    value: Math.round(m.positive_frac * 100),
    unit: '%',
    kind: m.positive_frac > 0.4 ? 'positive' : 'neutral',
    fillPct: m.positive_frac * 100,
  });
  cards.push({
    label: 'Негативных реплик',
    value: Math.round(m.negative_frac * 100),
    unit: '%',
    kind: m.negative_frac > 0.15 ? 'negative' : 'neutral',
    fillPct: m.negative_frac * 100,
  });

  // Адресность
  cards.push({
    label: 'Адресных обращений',
    value: Math.round(m.targeting_rate * 100),
    unit: '%',
    hint: m.targeting_rate < 0.7 ? 'Слабая адресация' : 'Хорошая адресация',
    kind: m.targeting_rate < 0.7 ? 'warning' : 'positive',
    fillPct: m.targeting_rate * 100,
  });

  // Взаимность
  cards.push({
    label: 'Взаимность связей',
    value: Math.round(m.reciprocity * 100),
    unit: '%',
    hint: 'Доля рёбер A↔B (Borgatti, SNA)',
    kind: m.reciprocity > 0.5 ? 'positive' : 'neutral',
    fillPct: m.reciprocity * 100,
  });

  // Разнообразие эмоций
  cards.push({
    label: 'Разнообразие эмоций',
    value: m.emotion_entropy.toFixed(2),
    hint: `${m.emotion_diversity} уникальных`,
    kind: m.emotion_entropy > 2.0 ? 'positive' : (m.emotion_entropy < 1.0 ? 'warning' : 'neutral'),
    fillPct: Math.min(100, (m.emotion_entropy / 3.5) * 100),
  });

  // Длина реплики
  cards.push({
    label: 'Длина реплики',
    value: m.avg_reply_words.toFixed(1),
    unit: 'слов',
    kind: 'neutral',
  });

  // Вопросы
  cards.push({
    label: 'Доля вопросов',
    value: Math.round(m.question_rate * 100),
    unit: '%',
    hint: m.question_rate < 0.15 ? 'Мало исследования' : 'Хорошо исследуют',
    kind: m.question_rate < 0.15 ? 'warning' : 'positive',
    fillPct: m.question_rate * 100,
  });

  // Конкретика / actionability
  cards.push({
    label: 'Конкретные шаги',
    value: Math.round(m.actionability_rate * 100),
    unit: '%',
    hint: 'plan, deadline, owner, task',
    kind: m.actionability_rate > 0.3 ? 'positive' : 'neutral',
    fillPct: m.actionability_rate * 100,
  });

  // Доминирование
  if (m.top_speaker && m.top_speaker_share > 0) {
    const dominant = m.top_speaker_share > 0.45;
    cards.push({
      label: 'Доминирование',
      value: Math.round(m.top_speaker_share * 100),
      unit: '%',
      hint: `${m.top_speaker}${dominant ? ' (перевес)' : ''}`,
      kind: dominant ? 'warning' : 'neutral',
      fillPct: m.top_speaker_share * 100,
    });
  }

  // Задержка ответа
  if (m.addressing_delay > 0) {
    cards.push({
      label: 'Задержка ответа',
      value: m.addressing_delay.toFixed(1),
      unit: 'ходов',
      kind: m.addressing_delay > 2.5 ? 'warning' : 'neutral',
    });
  }

  grid.innerHTML = cards.map(c => {
    const fillBar = c.fillPct !== undefined
      ? `<div class="metric-bar"><div class="metric-bar-fill" style="width:${Math.max(0, Math.min(100, c.fillPct))}%"></div></div>`
      : '';
    return `
      <div class="metric-card is-${c.kind || 'neutral'}">
        <div class="metric-label">${escapeHtml(c.label)}</div>
        <div class="metric-value">${escapeHtml(String(c.value))}${c.unit ? `<span class="metric-unit">${escapeHtml(c.unit)}</span>` : ''}</div>
        ${c.hint ? `<div class="metric-hint">${escapeHtml(c.hint)}</div>` : ''}
        ${fillBar}
      </div>
    `;
  }).join('');

  // По-агентная таблица
  if (agentsBox && m.per_agent && Object.keys(m.per_agent).length) {
    const rows = Object.entries(m.per_agent)
      .sort((a, b) => b[1].msgs - a[1].msgs)
      .map(([name, info]) => {
        const tone = info.avg_tone > 0.2 ? 'positive' : (info.avg_tone < -0.2 ? 'negative' : 'neutral');
        const toneSign = info.avg_tone >= 0 ? '+' : '';
        return `<tr>
          <td class="agent-name">${escapeHtml(name)}</td>
          <td class="num">${info.msgs}</td>
          <td class="num">${info.avg_words.toFixed(1)}</td>
          <td><span class="tone-pill ${tone}">${toneSign}${info.avg_tone.toFixed(2)}</span></td>
          <td>${escapeHtml(info.last_emotion || '—')}</td>
        </tr>`;
      }).join('');
    agentsBox.innerHTML = `
      <div class="metrics-section-title">По агентам</div>
      <table class="metrics-agents-table">
        <thead>
          <tr><th>Агент</th><th class="num">Реплик</th><th class="num">Ср. слов</th><th>Ср. тон</th><th>Эмоция</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>`;
  }
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

  hypotheses.forEach((hyp, idx) => {
    const card = document.createElement('div');
    card.className = 'sci-hyp-card';

    const br = hyp.brief_report;
    let briefBlock = '';
    if (br && (br.method || br.interpretation || (br.metrics && Object.keys(br.metrics).length))) {
      const reportId = `sci-hyp-brief-${idx}-${Math.random().toString(36).slice(2, 7)}`;
      const metricsRows = (br.metrics && Object.keys(br.metrics).length)
        ? Object.entries(br.metrics).map(([k, v]) =>
            `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(String(v))}</td></tr>`
          ).join('')
        : '';
      briefBlock = `
        <button type="button" class="sci-hyp-toggle" aria-expanded="false" aria-controls="${reportId}">
          <span class="toggle-icon">▸</span>
          <span class="toggle-label">Краткий отчёт</span>
        </button>
        <div id="${reportId}" class="sci-hyp-brief" hidden>
          ${br.method ? `
            <div class="brief-section">
              <div class="brief-section-title">Метод</div>
              <div class="brief-section-body">${escapeHtml(br.method)}</div>
            </div>` : ''}
          ${metricsRows ? `
            <div class="brief-section">
              <div class="brief-section-title">Метрики</div>
              <table class="brief-metrics-table"><tbody>${metricsRows}</tbody></table>
            </div>` : ''}
          ${br.interpretation ? `
            <div class="brief-section">
              <div class="brief-section-title">Интерпретация</div>
              <div class="brief-section-body">${escapeHtml(br.interpretation)}</div>
            </div>` : ''}
        </div>
      `;
    }

    card.innerHTML = `
      <div class="sci-hyp-header">
        <span class="sci-hyp-category">${escapeHtml(hyp.category || 'General')}</span>
        <span class="sci-hyp-framework">${escapeHtml(hyp.framework || '')}</span>
      </div>
      <div class="sci-hyp-finding">${escapeHtml(hyp.finding || '')}</div>
      <div class="sci-hyp-recommendation">${escapeHtml(hyp.recommendation || '')}</div>
      ${briefBlock}
      <div class="sci-hyp-reference">${escapeHtml(hyp.reference || '')}</div>
    `;

    const toggleBtn = card.querySelector('.sci-hyp-toggle');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        const target = card.querySelector(`#${toggleBtn.getAttribute('aria-controls')}`);
        if (!target) return;
        const isOpen = !target.hasAttribute('hidden');
        if (isOpen) {
          target.setAttribute('hidden', '');
          toggleBtn.setAttribute('aria-expanded', 'false');
          toggleBtn.querySelector('.toggle-icon').textContent = '▸';
        } else {
          target.removeAttribute('hidden');
          toggleBtn.setAttribute('aria-expanded', 'true');
          toggleBtn.querySelector('.toggle-icon').textContent = '▾';
        }
      });
    }

    scientificHypothesesEl.appendChild(card);
  });
}

function renderObserverHypothesisChecks(checks, summary = '') {
  if (!observerHypothesisChecksEl) return;
  if (!checks || checks.length === 0) {
    observerHypothesisChecksEl.innerHTML = '<p style="color: var(--text-secondary); padding: 2rem; text-align: center;">The theory-backed hypothesis list will appear here when the dialogue starts.</p>';
    return;
  }

  const statusMeta = {
    confirmed: { label: 'Confirmed', bg: 'rgba(16,185,129,0.16)', color: '#34d399' },
    partial: { label: 'Partial', bg: 'rgba(245,158,11,0.16)', color: '#fbbf24' },
    not_confirmed: { label: 'Not confirmed', bg: 'rgba(239,68,68,0.16)', color: '#f87171' },
    insufficient_data: { label: 'Insufficient data', bg: 'rgba(148,163,184,0.16)', color: '#cbd5e1' },
  };

  const groups = {
    confirmed: [],
    partial: [],
    not_confirmed: [],
    insufficient_data: [],
    pending: [],
  };
  checks.forEach((check) => {
    const key = groups[check.status] ? check.status : 'pending';
    groups[key].push(check);
  });

  const renderGroup = (title, items) => {
    if (!items.length) return '';
    let section = `<div style="margin-bottom:1.1rem;"><h3 style="margin:0 0 0.8rem 0;">${title}</h3>`;
    items.forEach((check) => {
      const meta = statusMeta[check.status] || { label: 'Pending', bg: 'rgba(99,102,241,0.16)', color: '#a5b4fc' };
      section += `
        <div style="border:1px solid var(--border);border-radius:16px;padding:1rem;margin-bottom:0.9rem;background:rgba(255,255,255,0.02);">
          <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;margin-bottom:0.45rem;">
            <div>
              <div style="font-weight:700;">${escapeHtml(check.theory || 'Theory')}</div>
              <div style="font-size:12px;color:var(--text-dim);">${escapeHtml(check.citation || '')}</div>
            </div>
            <span style="white-space:nowrap;padding:0.3rem 0.7rem;border-radius:999px;background:${meta.bg};color:${meta.color};font-size:12px;font-weight:700;">${meta.label}</span>
          </div>
          <div style="font-size:13px;line-height:1.45;margin-bottom:0.45rem;"><strong>Hypothesis:</strong> ${escapeHtml(check.hypothesis || '')}</div>
          <div style="font-size:13px;line-height:1.45;color:var(--text-secondary);"><strong>Evidence:</strong> ${escapeHtml(check.evidence || (check.status === 'pending' ? 'Not checked yet. First observer checkpoint runs after turn 5.' : 'No evidence provided.'))}</div>
        </div>
      `;
    });
    section += '</div>';
    return section;
  };

  let html = '<div style="padding:1rem;">';
  if (summary) {
    html += `<div style="margin-bottom:1rem;padding:0.9rem 1rem;border:1px solid var(--border);border-radius:14px;background:rgba(255,255,255,0.02);">${escapeHtml(summary)}</div>`;
  }
  html += renderGroup('Confirmed', groups.confirmed);
  html += renderGroup('Partially Confirmed', groups.partial);
  html += renderGroup('Not Confirmed', groups.not_confirmed);
  html += renderGroup('Insufficient Data', groups.insufficient_data);
  html += renderGroup('Pending Check', groups.pending);
  html += '</div>';
  observerHypothesisChecksEl.innerHTML = html;
}

function renderTriangulationSummary(summary) {
  if (!triangulationSummaryEl) return;
  if (!summary || Object.keys(summary).length === 0) {
    triangulationSummaryEl.innerHTML = '<p style="color: var(--text-secondary); padding: 2rem; text-align: center;">No observations yet.</p>';
    return;
  }

  let html = '<div style="padding: 1rem;">';
  html += `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem;">`;
  html += `<div class="stat-card"><div class="stat-value">${((summary.avg_convergence || 0) * 100).toFixed(0)}%</div><div class="stat-label">Avg convergence</div></div>`;
  html += `<div class="stat-card"><div class="stat-value">${summary.total_agreements || 0}</div><div class="stat-label">Agreements</div></div>`;
  html += `<div class="stat-card"><div class="stat-value">${summary.total_divergences || 0}</div><div class="stat-label">Divergences</div></div>`;
  html += `</div>`;
  html += `<p>Observations: ${summary.observations || 0}</p>`;
  if ((summary.observations || 0) > 1) {
    html += `<p>Range: ${((summary.min_convergence || 0) * 100).toFixed(0)}% – ${((summary.max_convergence || 0) * 100).toFixed(0)}%</p>`;
  }
  if (summary.total_novel_insights) {
    html += `<p>Novel insights captured: ${summary.total_novel_insights}</p>`;
  }
  html += '</div>';
  triangulationSummaryEl.innerHTML = html;
}

function renderObserverAnalysis(data) {
  if (observerReportEl && data && data.observer_report) {
    observerReportEl.innerHTML = renderMarkdown(data.observer_report);
  }
  if (observerReportEl && data && data.report) {
    observerReportEl.innerHTML = renderMarkdown(data.report);
  }
  renderObserverHypothesisChecks(
    (data && (data.hypothesis_registry || data.observer_hypothesis_checks || data.hypothesis_checks)) || [],
    (data && (data.observer_hypothesis_summary || data.hypothesis_summary)) || ''
  );
  renderTriangulationSummary((data && data.triangulation_summary) || {});
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
  statusEl.textContent = 'Генерирую реплику…';

  try {
    typingEl && (typingEl.style.display = 'flex');

    // In experiment mode, use experiment endpoint and loop until human turn
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
    if (data.metrics_summary) {
      renderMetricsCards(data.metrics_summary, data.turn);
      renderVerdicts(data.metrics_summary, data.turn);
    }

    // Render scientific analysis
    if (data.scientific_report) {
      renderScientificReport(data.scientific_report);
    }
    if (data.scientific_hypotheses) {
      renderScientificHypotheses(data.scientific_hypotheses);
    }
    renderObserverAnalysis(data);

    statusEl.textContent = `Ход: ${data.turn}`;

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
    statusEl.textContent = 'Ошибка хода: ' + e.message;
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
    loadAdvancedBtn.textContent = 'Загрузка…';
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
    loadAdvancedBtn.textContent = 'Загрузить анализ';
  });
}

// ==================== OBSERVER AGENT ====================
const runObserverBtn = document.getElementById('runObserverBtn');
if (runObserverBtn) {
  runObserverBtn.addEventListener('click', async () => {
    runObserverBtn.disabled = true;
    runObserverBtn.textContent = 'Анализирую…';
    try {
      const res = await fetch('/api/observer', { method: 'POST' });
      const data = await res.json();
      if (data.ok && data.report) {
        renderObserverAnalysis(data);
      } else {
        if (observerReportEl) {
          observerReportEl.innerHTML = `<p style="color:var(--text-secondary);padding:2rem;">${data.error || 'No data'}</p>`;
        }
      }
    } catch(e) {
      console.error(e);
    }
    runObserverBtn.disabled = false;
    runObserverBtn.textContent = 'Запустить анализ наблюдателя';
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
    userSendBtn.textContent = 'Отправка…';

    try {
      const data = await api('/api/user_message', 'POST', { reply, target, tone });
      if (!data.ok) throw new Error(data.error || 'Failed');

      userInputModal.style.display = 'none';
      renderHistory(data.history || []);
      statusEl.textContent = `Ход: ${data.turn}`;

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
      renderObserverAnalysis(data);
      checkValidationButton(data);
    } catch (e) {
      console.error(e);
      statusEl.textContent = 'Ошибка: ' + e.message;
    } finally {
      userSendBtn.disabled = false;
      userSendBtn.textContent = 'Отправить';
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
