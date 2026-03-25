# Prompt for Claude Opus 4.6 — Bug Fixes + User Participation in Agent Dialogue Simulator

Ты работаешь над проектом **Agent Dialogue Simulator** — многоагентная симуляция диалогов с веб-интерфейсом (Flask + vanilla JS). Проект расположен в `/Users/ilya/Mephi/agent_dialogue_sim`.

---

## СТРУКТУРА ПРОЕКТА (ключевые файлы)

```
agent_dialogue_sim/
├── dialogue_manager.py   — ядро: DialogueManager, step(), model_turn(), build_messages(), выбор спикера, mood-система
├── web_app.py            — Flask-приложение, AppState, API-эндпоинты (/api/start, /api/step, /api/state и др.)
├── agents.py             — Agent dataclass (name, nature, color, is_human, persona, big_five), BigFiveProfile
├── prompts.py            — AgentTurn (Pydantic), BASE_SYSTEM_PROMPTS, STRUCTURE_INSTRUCTION, SESSION_GOAL, HUMAN_STYLE
├── config.py             — MODEL, TEMPERATURE, MAX_TOKENS, DIALOGUE_PHASES, MOOD_*, USER_AGENT конфиг
├── human_io.py           — HumanIO класс (CLI-адаптер для ввода пользователя)
├── validators.py         — ValidationPipeline (8 валидаторов), ValidationLevel
├── analytics.py          — compute_metrics, hypotheses_from_metrics
├── observer_agent.py     — ObserverAgent (LLM-наблюдатель, методологическая триангуляция)
├── scientific_analytics.py — научный анализ, group stage detection, network density
├── advanced_analytics.py — emotional contagion, LSM, discourse coherence
├── visualize.py          — NetworkX-графы взаимодействий (PNG)
├── profiles.py           — initialize_agents, initialize_agents_from_config
├── cache_manager.py      — LRU-кэш с TTL (НЕ интегрирован)
├── templates/index.html  — HTML (wizard настройки + main dashboard)
├── static/app.js         — фронтенд логика (wizard, step-by-step UI, рендеринг)
└── static/style.css      — стили (dark theme, CSS-переменные)
```

---

## ТЕКУЩАЯ АРХИТЕКТУРА ДИАЛОГА

1. Пользователь через wizard задаёт количество агентов (2–20), окружение, настраивает каждого агента (имя, роль, цвет, анкета из 14 вопросов).
2. По нажатию **"Next message"** → `POST /api/step` → `DialogueManager.step()` выбирает следующего спикера (round-robin + приоритет адресату + анти-ping-pong), генерирует ответ через OpenAI API (3-уровневый fallback: function calling → JSON response_format → natural language extraction).
3. Каждый ход возвращает `AgentTurn`: `{reply, tone, emotion, target}`.
4. Каждые 5 ходов строятся графы, метрики, научные гипотезы, валидация гипотез.
5. В веб-версии **пользователь НЕ участвует** в диалоге (`human_io=None` в web_app.py строка 103). Участие пользователя реализовано ТОЛЬКО для CLI-версии (main.py).

---

## ЗАДАНИЕ: ИСПРАВЬ ВСЕ БАГИ + ДОБАВЬ УЧАСТИЕ ПОЛЬЗОВАТЕЛЯ В ДИАЛОГЕ

---

### ЧАСТЬ 1: ИСПРАВЛЕНИЕ БАГОВ

---

#### Баг 1 — Валидация не блокирует CRITICAL ошибки

**Файл:** `dialogue_manager.py`, метод `step()`, строки 534–578.

**Проблема:** Переменная `is_valid` вычисляется на строке 550:
```python
is_valid, validation_results = self.validation_pipeline.validate_turn(turn_data, context)
```
Но она **НИКОГДА не используется** для блокировки сообщения — `record` всегда добавляется в `self.history` (строка 578) независимо от результата валидации.

**Что нужно сделать:**
- Если `is_valid == False` (есть CRITICAL ошибки), **перегенерируй** ответ агента (максимум 2 повторные попытки через `model_turn()`).
- Если после 2 попыток всё ещё CRITICAL — используй safe fallback: `AgentTurn(reply="...", tone="neutral", emotion="thoughtful", target=allowed_targets[0])` с нейтральным текстом.
- Логируй каждую попытку перегенерации в stderr.

---

#### Баг 2 — eval() для парсинга JSON в human_io.py

**Файл:** `human_io.py`, строка 40.

**Проблема:** Используется `eval(m.group(0))` для парсинга ответа модели. Это небезопасно — `eval()` выполняет произвольный Python-код.

**Что нужно сделать:**
- Заменить `eval(m.group(0))` на `json.loads(m.group(0))`.
- Добавить `import json` в начало файла.
- Обернуть в try/except `json.JSONDecodeError` с fallback на `{"tone": "neutral", "emotion": "neutral"}`.

---

#### Баг 3 — Поля анкеты не восстанавливаются при переключении агентов

**Файл:** `static/app.js`, функция `updateAgentForm()`, строки 251–291.

**Проблема:** При переключении между агентами (кнопки Prev/Next или клик по карточке) восстанавливаются только 6 из 14 полей анкеты:
- ✅ background, motivation, speech_flaws, favorite_topics, stress_reaction, signature_phrase
- ❌ social_style, conflict_approach, decision_style, trust_level, cooperation_style, emotional_openness, leadership_tendency, criticism_reaction, group_dynamics

**Что нужно сделать:**
Добавить в `updateAgentForm()` после строки 267 восстановление недостающих полей:
```javascript
qSocialStyle.value = q.social_style || '';
qConflictApproach.value = q.conflict_approach || '';
qDecisionStyle.value = q.decision_style || '';
qTrustLevel.value = q.trust_level || '';
qCooperationStyle.value = q.cooperation_style || '';
qEmotionalOpenness.value = q.emotional_openness || '';
qLeadershipTendency.value = q.leadership_tendency || '';
qCriticismReaction.value = q.criticism_reaction || '';
qGroupDynamics.value = q.group_dynamics || '';
```

---

#### Баг 4 — Cache Manager не интегрирован

**Файл:** `cache_manager.py` — существует (8500+ строк), но **нигде не импортируется и не используется**.

**Что нужно сделать:**
- Если кэширование не критично для текущей версии — оставь файл как есть, но добавь комментарий в начало файла: `# NOTE: This module is not yet integrated into the main pipeline. See dialogue_manager.py.`
- НЕ удаляй файл — он может пригодиться позже.

---

#### Баг 5 — Observer Agent без retry-логики для rate limit

**Файл:** `observer_agent.py`, метод где вызывается `self.client.chat.completions.create()`.

**Проблема:** В `dialogue_manager.py` есть полноценная retry-логика с exponential backoff для `RateLimitError` (функция `make_api_call()`). В `observer_agent.py` — просто `try/except` без retry.

**Что нужно сделать:**
- Добавь retry-логику аналогичную `make_api_call()` из dialogue_manager.py:
  - max_retries = 3
  - exponential backoff: 3s, 6s, 9s
  - Импортируй `from openai import RateLimitError`
  - При исчерпании попыток — возвращай fallback-отчёт вместо crash.

---

### ЧАСТЬ 2: НОВЫЙ ФУНКЦИОНАЛ — УЧАСТИЕ ПОЛЬЗОВАТЕЛЯ В WEB-ДИАЛОГЕ

Сейчас в web-версии пользователь только наблюдает. Нужно дать ему возможность **активно участвовать в диалоге**: после каждого хода AI-агента пользователь может нажать кнопку "Participate", выбрать к кому обращается, написать реплику и отправить её в диалог. Агенты тоже должны иметь возможность обращаться к пользователю.

---

#### 2.1 Инициализация пользователя как участника диалога

**index.html (wizard, step 1):**
- Добавь чекбокс **"Join as participant"** (id="joinAsParticipant") в step1, после выбора окружения. По умолчанию **включён**.
- Рядом с чекбоксом — краткое пояснение: "You will appear as 'User' in the dialogue. Agents can address you."

**app.js:**
- При нажатии "Start Dialogue" (`startDialogueBtn.click`) — читай состояние чекбокса.
- Передавай `join_as_participant: true/false` в body запроса `POST /api/start`.
- Сохраняй в глобальную переменную `let userParticipating = false;`.

**web_app.py (`api_start`):**
- Читай `data.get("join_as_participant", False)`.
- Если `True`:
  - Создай агента User из конфига `USER_AGENT` (config.py): `Agent(name="User", nature="human", color="#405686", is_human=True)`.
  - Добавь его в список agents **первым** (чтобы он был виден всем).
  - В `agents_meta` добавь User с `is_human: True`.
- В `AppState` добавь поле `self.user_participating: bool = False`.
- `DialogueManager` создавай как обычно — User будет в списке agents, но `human_io=None` (ввод через API, не CLI).

**dialogue_manager.py (`build_messages`):**
- Если среди agents есть agent с `is_human=True`, добавь в system prompt блок:
```
"User is a real human participant in this conversation. They type their own messages.
You can address User naturally — ask them questions, react to what they said, include them.
Treat User like any other person in the group."
```

---

#### 2.2 Кнопка "Participate" и модальное окно ввода

**index.html:**
- Рядом с кнопкой `#stepBtn` ("Next message") добавь:
```html
<button id="participateBtn" class="btn btn-participate" style="display: none;">✋ Participate</button>
```
- Добавь модальное окно для ввода реплики пользователя:
```html
<div id="userInputModal" class="modal-overlay" style="display: none;">
  <div class="modal-card">
    <h3>Your turn to speak</h3>
    <div class="form-group">
      <label>Address to:</label>
      <select id="userTargetSelect" class="wizard-select"></select>
    </div>
    <div class="form-group">
      <label>Your message:</label>
      <textarea id="userReplyInput" class="wizard-input" rows="3" placeholder="Type your message..."></textarea>
    </div>
    <div class="form-group">
      <label>Tone:</label>
      <select id="userToneSelect" class="wizard-select">
        <option value="auto">Auto-detect</option>
        <option value="positive">Positive</option>
        <option value="neutral" selected>Neutral</option>
        <option value="negative">Negative</option>
      </select>
    </div>
    <div class="modal-actions">
      <button id="userSendBtn" class="btn btn-primary">Send</button>
      <button id="userCancelBtn" class="btn">Cancel</button>
    </div>
  </div>
</div>
```

**static/style.css — стили модального окна:**
```css
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.modal-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2rem;
  width: 480px;
  max-width: 90vw;
}
.modal-card h3 {
  margin: 0 0 1.5rem;
  color: var(--text-primary);
}
.modal-actions {
  display: flex;
  gap: 0.75rem;
  justify-content: flex-end;
  margin-top: 1.5rem;
}
.btn-participate {
  background: var(--accent);
  color: white;
  border: none;
}
.btn-participate:hover {
  filter: brightness(1.1);
}
```

**app.js — логика кнопки Participate:**

```javascript
const participateBtn = el('participateBtn');
const userInputModal = el('userInputModal');
const userTargetSelect = el('userTargetSelect');
const userReplyInput = el('userReplyInput');
const userToneSelect = el('userToneSelect');
const userSendBtn = el('userSendBtn');
const userCancelBtn = el('userCancelBtn');

// Показать кнопку Participate если пользователь участвует
// (устанавливается после startSession)
let userParticipating = false;

// При нажатии Participate — открыть модал
participateBtn.addEventListener('click', () => {
  // Заполнить список агентов (все кроме User)
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
});

// Cancel
userCancelBtn.addEventListener('click', () => {
  userInputModal.style.display = 'none';
});

// Send
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
```

- После `startSession()` успешно завершился: если `userParticipating === true`, показать `participateBtn`.
- В `renderHistory()` — для сообщений где `speaker === 'User'` добавь CSS-класс `msg-user` и пометку "(You)" рядом с именем.

---

#### 2.3 Бэкенд — эндпоинт /api/user_message и метод inject_user_turn

**dialogue_manager.py — новый метод `inject_user_turn()`:**

```python
def inject_user_turn(self, reply: str, target: str, tone: str = "neutral", emotion: str = "neutral") -> Dict[str, Any]:
    """Inject a user's message into the dialogue history."""
    self._decay_moods()
    self.turn_no += 1

    target = normalize_target(target)
    self._adjust_mood("User", tone, target)

    record = {
        "turn": self.turn_no,
        "ts_utc": datetime.utcnow().isoformat(),
        "env_context": self.env_context,
        "speaker": "User",
        "speaker_nature": "human",
        "reply": reply,
        "tone": tone,
        "emotion": emotion,
        "target": target,
        "is_human": True,
        "validation_issues": 0,
    }

    # Валидация
    if self.enable_validation and self.validation_pipeline:
        turn_data = {"speaker": "User", "target": target, "reply": reply, "tone": tone, "emotion": emotion}
        allowed_targets = [a.name for a in self.agents if a.name != "User"]
        context = {"history": self.history, "allowed_targets": allowed_targets, "env_context": self.env_context}
        is_valid, validation_results = self.validation_pipeline.validate_turn(turn_data, context)
        record["validation_issues"] = len(validation_results) if validation_results else 0

    if target:
        self.edges_window.append({"src": "User", "dst": target, "tone": tone})
        if len(self.edges_window) > 10:
            self.edges_window = self.edges_window[-10:]

    self.history.append(record)
    return record
```

**web_app.py — новый эндпоинт:**

```python
@app.route("/api/user_message", methods=["POST"])
def api_user_message():
    if not STATE.dm or not STATE.logger:
        return jsonify({"ok": False, "error": "Session not initialized"}), 400

    if not STATE.user_participating:
        return jsonify({"ok": False, "error": "User is not a participant"}), 400

    data = request.get_json(silent=True) or {}
    reply = data.get("reply", "").strip()
    target = data.get("target", "")
    tone = data.get("tone", "auto")

    if not reply:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    # Auto-classify tone if needed
    if tone == "auto":
        tone = "neutral"  # Простой fallback; можно заменить на LLM-классификацию

    emotion = "neutral"  # Можно расширить автоклассификацией позже

    record = STATE.dm.inject_user_turn(reply=reply, target=target, tone=tone, emotion=emotion)

    # Логирование
    STATE.logger.write_jsonl(record)
    STATE.logger.write_markdown(
        turn_no=record["turn"],
        speaker=record["speaker"],
        target=record.get("target"),
        tone=record["tone"],
        emotion=record["emotion"],
        text=record["reply"],
    )

    # Метрики и графы (аналогично api_step)
    image_url = None
    metrics_md = None
    hyps = []

    if STATE.dm.should_plot():
        # ... (скопировать логику построения графов и метрик из api_step)
        pass

    return jsonify({
        "ok": True,
        "record": record,
        "turn": STATE.dm.turn_no,
        "history": STATE.dm.history[-20:],
        "image_url": image_url,
        "hypotheses": hyps or STATE.last_hyps,
        "metrics_md": metrics_md,
        "scientific_hypotheses": STATE.last_scientific_hyps,
        "scientific_report": STATE.last_scientific_report if STATE.dm.should_plot() else None,
        "validation_report": STATE.last_validation_report if STATE.dm.should_plot() else None,
        "has_validation_report": bool(STATE.last_validation_report),
    })
```

**ВАЖНО:** Вынеси общую логику построения графов/метрик из `api_step` в отдельный метод `_compute_and_store_metrics(record)` в `AppState`, чтобы не дублировать код между `api_step` и `api_user_message`.

---

#### 2.4 Пропуск human-агента в автоматическом step()

**dialogue_manager.py, метод `step()`:**

После определения `speaker = self.agents[idx]` (строка 505), добавь проверку:

```python
# Если выбранный спикер — human, пропускаем его (он пишет через UI)
if getattr(speaker, "is_human", False):
    # Выбираем следующего НЕ-human агента
    for offset in range(1, len(self.agents)):
        candidate_idx = (idx + offset) % len(self.agents)
        if not getattr(self.agents[candidate_idx], "is_human", False):
            idx = candidate_idx
            speaker = self.agents[idx]
            break
```

Это гарантирует что `step()` никогда не заблокируется ожидая ввод от human.

---

#### 2.5 Уведомление когда агент обращается к User

**web_app.py (`api_step`):**
- В ответе добавь поле `"addressed_user": record.get("target") == "User"`.

**app.js (stepBtn click handler):**
- Если `data.record.target === 'User'` и `userParticipating`:
  - Подсвети кнопку "Participate" (добавь CSS-класс `btn-participate-pulse`).
  - Покажи уведомление в statusEl: `"${data.record.speaker} addressed you — click Participate to respond"`.

**style.css — пульсация:**
```css
.btn-participate-pulse {
  animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
  50% { box-shadow: 0 0 0 8px rgba(99, 102, 241, 0); }
}
```

---

#### 2.6 Стилизация сообщений пользователя в транскрипте

**app.js (`renderHistory`):**
- Для сообщений с `r.speaker === 'User'` или `r.is_human === true`:
  - Добавь CSS-класс `msg-user` к элементу `.msg`.
  - В `.meta` добавь "(You)" после имени спикера.

**style.css:**
```css
.msg-user {
  border-left-color: #405686 !important;
  background: rgba(64, 86, 134, 0.08);
}
.msg-user .speaker::after {
  content: " (You)";
  font-size: 0.75em;
  opacity: 0.6;
}
```

---

## ВАЖНЫЕ ОГРАНИЧЕНИЯ

1. **НЕ ломай существующий функционал** — всё что работает сейчас (wizard, step-by-step, графы, метрики, observer, advanced analytics, валидация, экспорт) должно продолжать работать.
2. **НЕ меняй Pydantic модель AgentTurn** — она используется повсюду.
3. **НЕ меняй формат history records** — добавляй новые поля (`is_human`) только как опциональные.
4. **Сохраняй стиль кода** проекта — dataclasses, типизация, комментарии на русском/английском.
5. **Используй существующую CSS-систему переменных** (`var(--accent)`, `var(--bg-card)`, `var(--border)`, `var(--text-primary)` и т.д.).
6. **Не добавляй новые зависимости** — всё решается существующим стеком (Flask, OpenAI, Pydantic).
7. **Читай каждый файл перед редактированием.** Не угадывай содержимое — используй Read tool.

---

## ПОРЯДОК ВЫПОЛНЕНИЯ

1. **Баг 1** — Валидация CRITICAL в dialogue_manager.py → перегенерация
2. **Баг 2** — eval() → json.loads() в human_io.py
3. **Баг 3** — Восстановление полей анкеты в app.js
4. **Баг 4** — Комментарий в cache_manager.py
5. **Баг 5** — Retry-логика в observer_agent.py
6. **Бэкенд пользователя** — inject_user_turn() в dialogue_manager.py + user_participating в AppState + /api/user_message в web_app.py + рефакторинг метрик в общий метод
7. **Фронтенд пользователя** — чекбокс в wizard + кнопка Participate + модальное окно + логика отправки в app.js
8. **Интеграция** — пропуск human в step() + инструкция агентам про User в build_messages() + уведомления
9. **Стилизация** — msg-user, modal-overlay, btn-participate-pulse в style.css
10. **Проверка** — запусти `python -m agent_dialogue_sim.web_app` и проверь полный цикл: создание сессии с участием пользователя → несколько ходов агентов → участие пользователя → агенты обращаются к пользователю → графы/метрики учитывают пользователя
