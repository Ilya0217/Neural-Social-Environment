# PROMPT v2: Multi-User Experiment Mode (10 Humans + 5 AI Agents)

**Model**: Claude Opus 4.6 (1M context)
**Task**: Implement multi-user experiment functionality for the Agent Dialogue Simulator
**Working Directory**: `/Users/ilya/Mephi/agent_dialogue_sim/`

---

## CONTEXT

This is an existing Python dialogue simulation system built with Flask. Currently it supports **1 human** ("User") + N AI agents. The system uses OpenAI-compatible API for agent responses, has a web UI with setup wizard, real-time transcript, interaction graphs, and scientific analytics.

### Current Architecture (DO NOT break existing functionality):

**Backend (Python):**
- `web_app.py` — Flask REST API, `AppState` singleton, endpoints `/api/start`, `/api/step`, `/api/user_message`, `/api/state`
- `dialogue_manager.py` — `DialogueManager` dataclass: turn-based orchestration, speaker selection, mood dynamics, validation pipeline. Key methods: `step()` returns `Tuple[Dict, Optional[Dict]]` (record, edge). `inject_user_turn()` returns `Dict`. `build_messages()` constructs LLM prompt. `model_turn()` calls OpenAI API.
- `agents.py` — `Agent` dataclass with fields: `name, nature, color, is_human=False, persona="", big_five=None`. `BigFiveProfile` dataclass with OCEAN traits.
- `human_io.py` — CLI-only `HumanIO` class (NOT used in web mode, `human_io=None` in web)
- `config.py` — Constants: `MODEL`, `TEMPERATURE`, `MAX_TOKENS`, `DEFAULT_AGENTS` (list of 3 dicts), `USER_AGENT` (single dict), `DIALOGUE_PHASES`, `MOOD_DECAY`, `MOOD_DELTA`, `TURNS_BETWEEN_PLOTS=5`, `VIZ_EDGE_WINDOW=12`
- `profiles.py` — `initialize_agents(interactive)` and `initialize_agents_from_config(agents_config, client)`. `QUESTION_BANK` list of questionnaire fields. `_format_persona()`, `_ensure_answers()`.
- `prompts.py` — `AgentTurn` pydantic model (`reply, tone, emotion, target`), `BASE_SYSTEM_PROMPTS` dict (explorer/critic/facilitator), `build_system_prompt(nature, persona)`, `STRUCTURE_INSTRUCTION`, `SESSION_GOAL`, `HUMAN_STYLE`
- `validators.py` — `ValidationPipeline` with grammatical/logical/ethical checks
- `analytics.py`, `scientific_analytics.py`, `advanced_analytics.py` — Metrics computation
- `observer_agent.py` — Non-participating observer for triangulation
- `visualize.py` — NetworkX interaction graph generation (PNG)
- `io_logger.py` — `IOLogger` with `write_jsonl(record)` and `write_markdown(turn_no, speaker, target, tone, emotion, text)`

**Frontend:**
- `templates/index.html` — Single-page HTML: setup wizard (step1=agent count/env, step2=per-agent config with questionnaire) + main dialogue interface (transcript, graph, hypotheses, metrics, tabs for scientific/advanced/observer)
- `static/app.js` — Vanilla JS: wizard logic, `api(path, method, body)` helper, `startSession()`, `renderHistory()`, `renderAgentBadges()`, step button handler, user participation modal. State vars: `agents[]`, `running`, `userParticipating`.
- `static/style.css` — Dark theme CSS with variables: `--bg, --surface, --accent, --text, --text-dim, --border, --tone-positive, --tone-negative, --tone-neutral`

### Key Current Behavior That MUST Be Preserved:
1. `AppState.reset(env_index, agents_config, join_as_participant)` creates AI agents via `initialize_agents_from_config()` or `initialize_agents()`, optionally appends 1 "User" agent with `is_human=True`
2. `DialogueManager.step()` returns `(record: Dict, edge: Optional[Dict])`. Speaker selection: checks inactive agents → addressee priority → ping-pong avoidance → round-robin. If speaker is human AND `self.human_io is None` → skips human, picks next AI agent (lines 517-523).
3. Human messages via `POST /api/user_message` → `dm.inject_user_turn(reply, target, tone, emotion)` which hardcodes `speaker="User"` (line 649)
4. Frontend "Participate" button opens modal for the single User

---

## GOAL

Extend the system to support **experiments with N real human participants** (e.g., 10 humans + 5 AI agents = 15 total). Each human connects via their own browser tab, identifies by name, sees the shared conversation, and takes turns writing messages when it's their turn.

### Requirements:

1. **Multi-human support**: Any number of agents can have `is_human=True`, each with a unique name
2. **Turn-based queue with human priority**: Configurable ratio (e.g., 2 human turns per 1 agent turn). Agents must NOT dominate.
3. **Per-user web interface**: Each human opens `/experiment?name=ИмяУчастника`, sees personalized view with "YOUR TURN" / "Waiting for X"
4. **Sequential turns**: Only one person speaks at a time. System WAITS for the specific human to submit.
5. **Address selection**: Humans choose who they're addressing from all participants
6. **Real-time updates**: All participants see new messages via polling (2s interval)
7. **Backward compatibility**: All existing single-user and pure-agent modes unchanged

---

## IMPLEMENTATION PLAN

### PHASE 1: Backend Core — Config & DialogueManager

#### 1.1 Edit `config.py`

Add at the end, after `MOOD_GUIDANCE`:

```python
# === Multi-user experiment settings ===
HUMAN_TO_AGENT_RATIO = 2        # N human turns before 1 agent turn
HUMAN_TURN_TIMEOUT = 300        # seconds to wait for human input before auto-skip
```

Do NOT add `MULTI_USER_MODE` here — it's a runtime state, not a config constant.

#### 1.2 Edit `dialogue_manager.py` — The Most Critical File

**IMPORTANT**: The existing `step()` method returns `Tuple[Dict, Optional[Dict]]`. This contract MUST be preserved for backward compatibility. In multi-user mode, when a human is selected, `step()` must return a special record dict with a `"waiting_for"` key AND still conform to the tuple return type.

**Step 1: Add imports at the top of the file (after existing imports):**

```python
import time as _time  # for pending_human timestamps
from .config import HUMAN_TO_AGENT_RATIO, HUMAN_TURN_TIMEOUT
```

**Step 2: Add new dataclass fields after existing ones:**

```python
@dataclass
class DialogueManager:
    # ... ALL existing fields unchanged ...

    # Multi-user experiment fields
    multi_user_mode: bool = False              # True when >1 human participant
    pending_human: Optional[str] = None        # Name of human whose turn it is
    pending_human_since: Optional[float] = None  # time.time() when they were prompted
    _human_turn_count: int = 0                 # consecutive human turns in current cycle
    _human_to_agent_ratio: int = 2             # loaded from config
```

**Step 3: Update `__post_init__`:**

Add to the END of existing `__post_init__`:
```python
    # Multi-user setup
    humans = [a for a in self.agents if getattr(a, "is_human", False)]
    self.multi_user_mode = len(humans) > 1
    self._human_to_agent_ratio = HUMAN_TO_AGENT_RATIO
```

**Step 4: Add new helper methods BEFORE `step()`:**

```python
def _pick_next_human(self, humans: list) -> 'Agent':
    """Pick the next human to speak, prioritizing:
    1. Addressed human (if last message targeted a human)
    2. Least-recently-spoken human
    """
    human_names = {h.name for h in humans}

    # Priority 1: If last message addressed a human, give them priority
    if self.history:
        last_target = self.history[-1].get("target")
        if last_target and last_target in human_names:
            # Check this human hasn't JUST spoken (avoid ping-pong)
            last_speaker = self.history[-1].get("speaker")
            if last_target != last_speaker:
                return next(h for h in humans if h.name == last_target)

    # Priority 2: Least-recently-spoken human
    last_spoke = {h.name: -1 for h in humans}
    for i, record in enumerate(self.history):
        if record["speaker"] in human_names:
            last_spoke[record["speaker"]] = i

    sorted_humans = sorted(humans, key=lambda h: last_spoke[h.name])
    return sorted_humans[0]

def _pick_agent_for_multi_user(self) -> int:
    """Pick an AI agent index using existing logic, excluding humans."""
    ai_indices = [i for i, a in enumerate(self.agents) if not getattr(a, "is_human", False)]
    if not ai_indices:
        return 0  # fallback

    # Use existing inactive-agent and ping-pong avoidance logic
    recent_window = len(self.agents)
    recent_speakers = set()
    if len(self.history) >= recent_window:
        recent_speakers = {h["speaker"] for h in self.history[-recent_window:]}

    # Find AI agents that haven't spoken recently
    ai_names = {self.agents[i].name for i in ai_indices}
    inactive_ai = ai_names - recent_speakers

    if inactive_ai and len(self.history) >= recent_window:
        speaker_counts = {}
        for h in self.history[-recent_window:]:
            speaker_counts[h["speaker"]] = speaker_counts.get(h["speaker"], 0) + 1
        inactive_with_counts = [(name, speaker_counts.get(name, 0)) for name in inactive_ai]
        inactive_with_counts.sort(key=lambda x: (x[1], hash(x[0])))
        selected_name = inactive_with_counts[0][0]
        return next(i for i, a in enumerate(self.agents) if a.name == selected_name)

    # Ping-pong avoidance
    if self.history and self.history[-1].get("target"):
        last_tgt = self.history[-1]["target"]
        last_speaker = self.history[-1]["speaker"]
        if last_tgt in ai_names:
            if len(self.history) >= 2:
                prev = self.history[-2]
                if prev.get("speaker") == last_tgt and prev.get("target") == last_speaker:
                    ping_pong_pair = {last_speaker, last_tgt}
                    other = [i for i in ai_indices if self.agents[i].name not in ping_pong_pair]
                    if other:
                        return other[0]
            return next((i for i in ai_indices if self.agents[i].name == last_tgt), ai_indices[0])

    # Round-robin among AI agents
    ai_cycle = self.turn_no % len(ai_indices)
    return ai_indices[ai_cycle]

def submit_human_turn(self, speaker_name: str, reply: str, target: str,
                      tone: str = "neutral", emotion: str = "neutral") -> Dict[str, Any]:
    """Submit a turn from a specific human participant (multi-user mode).

    Returns a record dict, same format as step() records.
    Raises ValueError if it's not this human's turn.
    """
    if self.pending_human and self.pending_human != speaker_name:
        raise ValueError(f"Сейчас ход {self.pending_human}, а не {speaker_name}")

    if not self.pending_human:
        raise ValueError(f"Сейчас не ожидается ход от человека")

    self._decay_moods()
    self.turn_no += 1

    target = normalize_target(target)
    if speaker_name not in self.agent_moods:
        self.agent_moods[speaker_name] = 0.0
    self._adjust_mood(speaker_name, tone, target)

    # Find speaker agent
    speaker_agent = next((a for a in self.agents if a.name == speaker_name), None)
    speaker_nature = speaker_agent.nature if speaker_agent else "human"

    record = {
        "turn": self.turn_no,
        "ts_utc": datetime.utcnow().isoformat(),
        "env_context": self.env_context,
        "speaker": speaker_name,
        "speaker_nature": speaker_nature,
        "reply": reply,
        "tone": tone,
        "emotion": emotion,
        "target": target,
        "is_human": True,
        "validation_issues": 0,
    }

    # Validation (same as inject_user_turn)
    if self.enable_validation and self.validation_pipeline:
        turn_data = {"speaker": speaker_name, "target": target, "reply": reply, "tone": tone, "emotion": emotion}
        allowed_targets = [a.name for a in self.agents if a.name != speaker_name]
        context = {"history": self.history, "allowed_targets": allowed_targets, "env_context": self.env_context}
        is_valid, validation_results = self.validation_pipeline.validate_turn(turn_data, context)
        record["validation_issues"] = len(validation_results) if validation_results else 0

    if target:
        self.edges_window.append({"src": speaker_name, "dst": target, "tone": tone})
        if len(self.edges_window) > 10:
            self.edges_window = self.edges_window[-10:]

    self.history.append(record)

    # Clear pending state and update counter
    self._human_turn_count += 1
    self.pending_human = None
    self.pending_human_since = None

    return record
```

**Step 5: Modify `step()` method.**

The key insight: `step()` must STILL return `(record_dict, edge_or_none)`. For the "waiting for human" case, we return a synthetic record with `"waiting_for"` key. The web layer checks for this key.

Replace the ENTIRE `step()` method body (lines 455-633 in current file). Here is the complete new method:

```python
def step(self):
    self._decay_moods()

    # ===== MULTI-USER MODE =====
    if self.multi_user_mode:
        # Check pending human timeout
        if self.pending_human:
            if self.pending_human_since and (_time.time() - self.pending_human_since) > HUMAN_TURN_TIMEOUT:
                # Timeout — skip this human
                import sys
                print(f"[MultiUser] {self.pending_human} timed out, skipping", file=sys.stderr)
                self.pending_human = None
                self.pending_human_since = None
                self._human_turn_count += 1  # count as if they spoke (so ratio advances)
            else:
                # Still waiting — return wait signal
                return {"waiting_for": self.pending_human, "turn": self.turn_no}, None

        humans = [a for a in self.agents if getattr(a, "is_human", False)]

        # Decide: human turn or agent turn?
        if self._human_turn_count < self._human_to_agent_ratio and humans:
            # Human's turn — pick one and wait
            speaker = self._pick_next_human(humans)
            self.pending_human = speaker.name
            self.pending_human_since = _time.time()
            return {"waiting_for": speaker.name, "turn": self.turn_no}, None
        else:
            # Agent's turn — reset counter, pick AI agent
            self._human_turn_count = 0
            idx = self._pick_agent_for_multi_user()
            speaker = self.agents[idx]

            # Allowed targets: everyone except speaker
            allowed_targets = [a.name for a in self.agents if a.name != speaker.name] or [a.name for a in self.agents]

            messages = self.build_messages(idx)
            try:
                parsed = self.model_turn(messages, allowed_targets, agent=speaker)
            except RuntimeError as e:
                import sys
                print(f"[Warning] Failed to generate response for {speaker.name}: {e}", file=sys.stderr)
                parsed = AgentTurn(
                    reply="I'm here, but having trouble responding right now...",
                    tone="neutral", emotion="neutral",
                    target=allowed_targets[0] if allowed_targets else None,
                )

            # Record turn (same as original logic)
            self.turn_no += 1
            target = normalize_target(parsed.target)
            self._adjust_mood(speaker.name, parsed.tone, target)

            # Validation
            turn_data = {
                "speaker": speaker.name, "target": target,
                "reply": parsed.reply, "tone": parsed.tone, "emotion": parsed.emotion,
            }
            validation_results = []
            is_valid = True
            if self.enable_validation and self.validation_pipeline:
                context = {"history": self.history, "allowed_targets": allowed_targets, "env_context": self.env_context}
                is_valid, validation_results = self.validation_pipeline.validate_turn(turn_data, context)
                if validation_results:
                    import sys
                    formatted = self.validation_pipeline.format_results(validation_results)
                    print(f"\n[Validation Turn {self.turn_no}]", file=sys.stderr)
                    print(formatted, file=sys.stderr)

                # CRITICAL regeneration
                if not is_valid:
                    for retry in range(2):
                        import sys
                        print(f"[Validation] Regenerating (attempt {retry+1}/2)...", file=sys.stderr)
                        try:
                            messages = self.build_messages(idx)
                            parsed = self.model_turn(messages, allowed_targets, agent=speaker)
                            target = normalize_target(parsed.target)
                            turn_data = {"speaker": speaker.name, "target": target, "reply": parsed.reply, "tone": parsed.tone, "emotion": parsed.emotion}
                            is_valid, validation_results = self.validation_pipeline.validate_turn(turn_data, context)
                            if is_valid:
                                break
                        except Exception as e:
                            print(f"[Validation] Retry {retry+1} failed: {e}", file=sys.stderr)
                    if not is_valid:
                        parsed = AgentTurn(reply="Hmm, let me think about that for a moment...", tone="neutral", emotion="thoughtful", target=allowed_targets[0] if allowed_targets else None)
                        target = normalize_target(parsed.target)

            record = {
                "turn": self.turn_no,
                "ts_utc": datetime.utcnow().isoformat(),
                "env_context": self.env_context,
                "speaker": speaker.name,
                "speaker_nature": speaker.nature,
                "reply": parsed.reply,
                "tone": parsed.tone,
                "emotion": parsed.emotion,
                "target": target,
                "validation_issues": len(validation_results) if validation_results else 0,
            }

            if target:
                self.edges_window.append({"src": speaker.name, "dst": target, "tone": parsed.tone})
                if len(self.edges_window) > 10:
                    self.edges_window = self.edges_window[-10:]
            self.history.append(record)
            return record, ({"src": speaker.name, "dst": target, "tone": parsed.tone} if target else None)

    # ===== ORIGINAL SINGLE-USER / NO-HUMAN MODE =====
    # ... KEEP THE ENTIRE EXISTING step() BODY HERE UNCHANGED ...
    # (everything from the inactive-agent detection through to the final return)
```

**CRITICAL**: Do NOT delete or modify the existing step() logic. Wrap it in the `else` branch after the multi-user block. The existing code handles: inactive agent detection, ping-pong avoidance, addressee priority, human skip (single user), model_turn call, validation, record creation. All of this stays for backward compatibility.

**Step 6: Modify `build_messages()` — update human_block (line ~140-147):**

Replace:
```python
human_block = ""
if any(getattr(a, "is_human", False) for a in self.agents):
    human_block = (
        "\nUser is a real human participant in this conversation. They type their own messages.\n"
        "You can address User naturally — ask them questions, react to what they said, include them.\n"
        "Treat User like any other person in the group.\n"
    )
```

With:
```python
human_block = ""
human_names = [a.name for a in self.agents if getattr(a, "is_human", False)]
if human_names:
    if len(human_names) == 1 and human_names[0] == "User":
        # Single-user mode — original message
        human_block = (
            "\nUser is a real human participant in this conversation. They type their own messages.\n"
            "You can address User naturally — ask them questions, react to what they said, include them.\n"
            "Treat User like any other person in the group.\n"
        )
    else:
        # Multi-user mode — list all humans
        names_str = ", ".join(human_names)
        human_block = (
            f"\nThe following participants are real humans (not AI): {names_str}.\n"
            "They type their own messages. You can address any of them naturally — "
            "ask them questions, react to what they said, include them.\n"
            "Treat every human like any other person in the group. Be natural.\n"
        )
```

---

### PHASE 2: Backend — Web Endpoints

#### 2.1 Edit `web_app.py`

**Step 1: Add import at top:**

```python
import time  # for experiment tracking
```

**Step 2: Extend `AppState.__init__`:**

Add these lines at the end of `__init__`:
```python
        self.human_participants: Dict[str, Dict[str, Any]] = {}
        self.multi_user_mode: bool = False
```

**Step 3: Extend `AppState.reset()` signature and body.**

Change the method signature to:
```python
def reset(self, env_index: int, agents_config=None, join_as_participant=False, human_participants=None):
```

After the line where agents are initialized (line ~101), before `self.user_participating = join_as_participant`, add:

```python
        # Multi-user experiment mode
        self.human_participants = {}
        self.multi_user_mode = False

        if human_participants and len(human_participants) > 0:
            self.multi_user_mode = True
            self.user_participating = True
            for hp in human_participants:
                hp_name = hp["name"]
                hp_color = hp.get("color", "#405686")
                human_agent = Agent(
                    name=hp_name,
                    nature="human",
                    color=hp_color,
                    is_human=True,
                )
                agents.append(human_agent)
                self.human_participants[hp_name] = {"connected": False, "last_seen": None}
        elif join_as_participant:
            # Original single-user mode
            self.user_participating = True
            from .config import USER_AGENT
            user_agent = Agent(
                name=USER_AGENT["name"],
                nature=USER_AGENT["nature"],
                color=USER_AGENT["color"],
                is_human=True,
            )
            agents.insert(0, user_agent)
        else:
            self.user_participating = False
```

Remove the existing `join_as_participant` block (lines 104-113) since it's now in the code above.

**Step 4: Modify `api_step()` to handle waiting_for signal:**

In the `api_step` function, after `record, _ = STATE.dm.step()`, add a check:

```python
    record, _ = STATE.dm.step()

    # Multi-user: step() may return a "waiting" signal instead of a real record
    if "waiting_for" in record:
        return jsonify({
            "ok": True,
            "waiting_for": record["waiting_for"],
            "turn": STATE.dm.turn_no,
            "history": STATE.dm.history[-20:],
        })

    # Normal flow continues...
    STATE.logger.write_jsonl(record)
    # ... rest unchanged ...
```

**Step 5: Add new experiment endpoints (add BEFORE `create_app()`):**

```python
# ==================== MULTI-USER EXPERIMENT ENDPOINTS ====================

@app.route("/experiment")
def experiment_page():
    """Serve the experiment participant view."""
    return render_template("experiment.html")


@app.route("/api/experiment/start", methods=["POST"])
def api_experiment_start():
    """Start a multi-user experiment session.

    Body: {
        env_index: int,
        agents: [{name, nature, color, questionnaire?}, ...],  // AI agents
        humans: [{name, color?}, ...],  // Human participants
    }
    """
    data = request.get_json(silent=True) or {}
    env_index = int(data.get("env_index", 0))
    agents_config = data.get("agents", [])
    human_participants = data.get("humans", [])

    if not human_participants:
        return jsonify({"ok": False, "error": "Need at least 1 human participant"}), 400

    # Validate names
    all_names = [a.get("name", "") for a in agents_config] + [h.get("name", "") for h in human_participants]
    if any(not n.strip() for n in all_names):
        return jsonify({"ok": False, "error": "All participants must have names"}), 400
    if len(set(n.lower() for n in all_names)) != len(all_names):
        return jsonify({"ok": False, "error": "All participant names must be unique"}), 400
    if "Observer" in all_names:
        return jsonify({"ok": False, "error": "'Observer' is reserved"}), 400

    try:
        STATE.reset(env_index, agents_config, human_participants=human_participants)
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    # Generate join links
    join_links = {}
    for hp in human_participants:
        name = hp["name"]
        # URL-encode the name for safe use in query params
        from urllib.parse import quote
        encoded_name = quote(name, safe='')
        join_links[name] = f"/experiment?name={encoded_name}"

    return jsonify({
        "ok": True,
        "env": STATE.env_context,
        "agents": list(STATE.agents_meta.keys()),
        "agents_data": [{"name": n, **m} for n, m in STATE.agents_meta.items()],
        "humans": list(STATE.human_participants.keys()),
        "join_links": join_links,
        "multi_user": True,
    })


@app.route("/api/experiment/join", methods=["POST"])
def api_experiment_join():
    """Human participant joins the experiment. Body: {name: str}"""
    if not STATE.dm or not STATE.multi_user_mode:
        return jsonify({"ok": False, "error": "No multi-user experiment active"}), 400

    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()

    if name not in STATE.human_participants:
        return jsonify({"ok": False, "error": f"Участник '{name}' не найден в эксперименте"}), 400

    STATE.human_participants[name]["connected"] = True
    STATE.human_participants[name]["last_seen"] = time.time()

    return jsonify({
        "ok": True,
        "name": name,
        "agents_data": [{"name": n, **m} for n, m in STATE.agents_meta.items()],
        "env": STATE.env_context,
        "history": STATE.dm.history[-50:],
        "turn": STATE.dm.turn_no,
        "pending_human": STATE.dm.pending_human,
    })


@app.route("/api/experiment/step", methods=["POST"])
def api_experiment_step():
    """Advance experiment by one turn. Called by moderator/auto-advance.

    If it's a human's turn, returns {waiting_for: name}.
    If it's an agent's turn, processes it and returns the record.
    """
    if not STATE.dm:
        return jsonify({"ok": False, "error": "No session"}), 400

    # If already waiting for a human, just report status
    if STATE.dm.pending_human:
        return jsonify({
            "ok": True,
            "waiting_for": STATE.dm.pending_human,
            "turn": STATE.dm.turn_no,
            "history": STATE.dm.history[-50:],
        })

    record, edge = STATE.dm.step()

    # Check if step() returned waiting signal
    if "waiting_for" in record:
        return jsonify({
            "ok": True,
            "waiting_for": record["waiting_for"],
            "turn": STATE.dm.turn_no,
            "history": STATE.dm.history[-50:],
        })

    # Agent turn completed — log it
    STATE.logger.write_jsonl(record)
    STATE.logger.write_markdown(
        turn_no=record["turn"], speaker=record["speaker"],
        target=record.get("target"), tone=record["tone"],
        emotion=record["emotion"], text=record["reply"],
    )

    image_url, metrics_md, hyps = STATE.compute_metrics_if_needed(record)

    return jsonify({
        "ok": True,
        "record": record,
        "turn": STATE.dm.turn_no,
        "history": STATE.dm.history[-50:],
        "image_url": image_url,
        "hypotheses": hyps or STATE.last_hyps,
        "metrics_md": metrics_md,
        "pending_human": STATE.dm.pending_human,
    })


@app.route("/api/experiment/human_message", methods=["POST"])
def api_experiment_human_message():
    """Human submits their message. Body: {speaker, reply, target, tone}"""
    if not STATE.dm:
        return jsonify({"ok": False, "error": "No session"}), 400

    data = request.get_json(silent=True) or {}
    speaker = data.get("speaker", "").strip()
    reply = data.get("reply", "").strip()
    target = data.get("target", "")
    tone = data.get("tone", "neutral")

    if not speaker or not reply:
        return jsonify({"ok": False, "error": "speaker and reply are required"}), 400

    if tone == "auto":
        tone = "neutral"

    try:
        record = STATE.dm.submit_human_turn(
            speaker_name=speaker, reply=reply,
            target=target, tone=tone, emotion="neutral",
        )
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    STATE.logger.write_jsonl(record)
    STATE.logger.write_markdown(
        turn_no=record["turn"], speaker=record["speaker"],
        target=record.get("target"), tone=record["tone"],
        emotion=record["emotion"], text=record["reply"],
    )

    image_url, metrics_md, hyps = STATE.compute_metrics_if_needed(record)

    return jsonify({
        "ok": True,
        "record": record,
        "turn": STATE.dm.turn_no,
        "history": STATE.dm.history[-50:],
        "image_url": image_url,
        "hypotheses": hyps or STATE.last_hyps,
        "metrics_md": metrics_md,
        "pending_human": STATE.dm.pending_human,
    })


@app.route("/api/experiment/status", methods=["GET"])
def api_experiment_status():
    """Polling endpoint: returns current experiment state."""
    if not STATE.dm:
        return jsonify({"ok": True, "active": False})

    latest_img = None
    if OUT_DIR.exists():
        cand = sorted(OUT_DIR.glob("graph_turn_*.png"))
        latest_img = cand[-1].name if cand else None

    # Update last_seen for polling humans
    caller = request.args.get("caller", "")
    if caller and caller in STATE.human_participants:
        STATE.human_participants[caller]["last_seen"] = time.time()

    return jsonify({
        "ok": True,
        "active": True,
        "multi_user": STATE.multi_user_mode,
        "turn": STATE.dm.turn_no,
        "history": STATE.dm.history[-50:],
        "pending_human": STATE.dm.pending_human,
        "image_url": f"/outputs/{latest_img}" if latest_img else None,
        "hypotheses": STATE.last_hyps,
        "agents_data": [{"name": n, **m} for n, m in STATE.agents_meta.items()],
        "connected_humans": {
            name: {
                "connected": info["connected"],
                "online": info["last_seen"] and (time.time() - info["last_seen"]) < 30,
            }
            for name, info in STATE.human_participants.items()
        },
    })
```

---

### PHASE 3: Frontend — Participant View

#### 3.1 Create `templates/experiment.html`

This is a NEW file. The participant sees:
- Join screen (if not yet joined)
- Transcript + turn indicator + input form + participant list + graph

**IMPORTANT design decisions:**
- Reuse `static/style.css` from existing app (same dark theme)
- The JS is in a separate file `static/experiment.js`
- Auto-advance: ONLY the participant whose turn it was calls `advanceUntilHumanTurn()` after submitting. This prevents duplicate agent turns from multiple clients polling simultaneously.
- Each participant polls `/api/experiment/status?caller=MyName` every 2 seconds.
- When polling detects `pending_human === null` AND there are no waiting-for signals, ONE client (the one who just submitted) triggers auto-advance.

```html
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment — Participant View</title>
    <link rel="stylesheet" href="/static/style.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* Experiment-specific styles — on top of existing style.css */
        .exp-join { ... }
        .exp-turn-banner { ... }
        .exp-turn-banner.my-turn { background: var(--tone-positive); ... }
        .exp-turn-banner.waiting { background: var(--surface); ... }
        .exp-input-area { ... }
        .exp-input-area.disabled { opacity: 0.4; pointer-events: none; }
        .exp-participants { ... }
        .exp-participant-dot { width: 8px; height: 8px; border-radius: 50%; }
        .exp-participant-dot.online { background: var(--tone-positive); }
        .exp-participant-dot.offline { background: var(--tone-negative); }
        .exp-participant-dot.is-ai { background: var(--accent); }
    </style>
</head>
<body>
    <!-- JOIN SCREEN -->
    <div id="joinScreen" class="wizard-overlay">
        <div class="wizard-container" style="max-width: 480px;">
            <div class="wizard-header">
                <h1>Dialogue Experiment</h1>
                <p class="wizard-subtitle">Enter your participant name to join</p>
            </div>
            <div class="wizard-content">
                <div class="form-group">
                    <label>Your name:</label>
                    <input type="text" id="nameInput" class="wizard-input" placeholder="e.g., Иван">
                </div>
                <p id="joinError" style="color: var(--tone-negative); display: none;"></p>
            </div>
            <div class="wizard-actions">
                <button id="joinBtn" class="wizard-btn wizard-btn-primary">Join Experiment</button>
            </div>
        </div>
    </div>

    <!-- MAIN EXPERIMENT VIEW -->
    <div id="expMain" style="display: none;">
        <header class="header">
            <div>
                <h1>Dialogue Experiment</h1>
                <p class="subtitle" id="expEnv"></p>
            </div>
            <div class="controls">
                <span id="myNameBadge" class="agent-badge" style="font-weight: 600;"></span>
                <span id="expTurn" style="color: var(--text-dim);">Turn: 0</span>
            </div>
        </header>

        <!-- Turn indicator banner -->
        <div id="turnBanner" class="exp-turn-banner waiting">
            <span id="turnText">Connecting...</span>
        </div>

        <!-- Participants bar -->
        <div id="participantsBar" class="agent-badges"></div>

        <!-- Main grid: transcript + input + graph -->
        <main class="grid" style="grid-template-columns: 1fr 1fr; grid-template-rows: auto 1fr;">
            <!-- Input area (top-left, only when it's my turn) -->
            <section id="inputArea" class="panel exp-input-area disabled" style="grid-column: 1;">
                <h2>Your message</h2>
                <div class="form-group">
                    <label>Address to:</label>
                    <select id="targetSelect" class="wizard-select"></select>
                </div>
                <div class="form-group">
                    <label>Message:</label>
                    <textarea id="messageInput" class="wizard-input" rows="3" placeholder="Type your message..."></textarea>
                </div>
                <div class="form-group" style="display: flex; gap: 12px; align-items: center;">
                    <select id="toneSelect" class="wizard-select" style="width: auto;">
                        <option value="auto">Auto tone</option>
                        <option value="positive">Positive</option>
                        <option value="neutral" selected>Neutral</option>
                        <option value="negative">Negative</option>
                    </select>
                    <button id="sendBtn" class="btn btn-primary" disabled>Send</button>
                </div>
            </section>

            <!-- Graph (top-right) -->
            <section class="panel graph" style="grid-column: 2; grid-row: 1 / 3;">
                <h2>Interaction graph</h2>
                <div class="img-wrap">
                    <div id="graphPlaceholder" class="graph-placeholder">
                        <span>Graph appears after 5 messages</span>
                    </div>
                    <img id="graphImg" src="" alt="Graph" style="display: none;" />
                </div>
            </section>

            <!-- Transcript (bottom-left) -->
            <section class="panel transcript" style="grid-column: 1;">
                <h2>Transcript</h2>
                <div id="transcript" class="scroll"></div>
            </section>
        </main>

        <footer class="footer">
            <span id="status">Ready</span>
        </footer>
    </div>

    <script src="/static/experiment.js"></script>
</body>
</html>
```

Write the complete template with all the inline styles needed. Use existing CSS classes where possible.

#### 3.2 Create `static/experiment.js`

**Complete implementation. Key behaviors:**

1. On load: read `name` from URL query param, pre-fill input, show join screen
2. On join: POST to `/api/experiment/join`, hide join screen, show main view, start polling
3. Polling: every 2 seconds, GET `/api/experiment/status?caller={myName}`
4. Render: update transcript, turn banner, graph, participants
5. Turn banner:
   - `pending_human === myName` → "YOUR TURN!" (green), enable input, play notification sound, focus textarea
   - `pending_human === someoneElse` → "Waiting for [Name]..." (gray), disable input
   - `pending_human === null` → "Processing..." (blue)
6. On send: POST to `/api/experiment/human_message`, then call `advanceUntilHumanTurn()`
7. `advanceUntilHumanTurn()`: loop calling `/api/experiment/step` with 800ms delays until `waiting_for` appears. This ensures agent turns happen between human turns. **ONLY the sender calls this** — other clients just see updates via polling.
8. Notification: when it becomes my turn, play a short beep using `AudioContext` API (no external files needed):

```javascript
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
    } catch(e) {}
}
```

9. **Double-submit protection**: After clicking Send, disable button immediately, re-enable after response.
10. **Transcript rendering**: Reuse the same format as `renderHistory` in app.js. Messages from `myName` get a special "you" badge.
11. **Escape HTML**: Must escape all user text to prevent XSS.

```javascript
// === experiment.js — Full Implementation ===

const el = (id) => document.getElementById(id);

// URL params
const params = new URLSearchParams(window.location.search);
let myName = params.get('name') || '';

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
let isAdvancing = false;  // lock to prevent multiple auto-advance loops
let pollInterval = null;

// Pre-fill name from URL
if (myName) {
    nameInput.value = decodeURIComponent(myName);
    myName = decodeURIComponent(myName);
}

// Join
joinBtn.addEventListener('click', async () => {
    myName = nameInput.value.trim();
    if (!myName) return;

    joinBtn.disabled = true;
    joinBtn.textContent = 'Joining...';
    joinError.style.display = 'none';

    try {
        const res = await fetch('/api/experiment/join', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
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
    } catch(e) {
        joinError.textContent = 'Connection error: ' + e.message;
        joinError.style.display = 'block';
    } finally {
        joinBtn.disabled = false;
        joinBtn.textContent = 'Join Experiment';
    }
});

// Populate target dropdown
function populateTargets() {
    targetSelect.innerHTML = '';
    agentsData.filter(a => a.name !== myName && !a.is_observer).forEach(a => {
        const opt = document.createElement('option');
        opt.value = a.name;
        opt.textContent = `${a.name} (${a.nature})`;
        targetSelect.appendChild(opt);
    });
}

// Polling
function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/experiment/status?caller=${encodeURIComponent(myName)}`);
            const data = await res.json();
            if (data.ok && data.active) {
                if (data.agents_data) agentsData = data.agents_data;
                renderAll(data);
            }
        } catch(e) {
            statusEl.textContent = 'Connection lost...';
        }
    }, 2000);
}

// Render everything
function renderAll(data) {
    renderTranscript(data.history || []);
    renderTurnBanner(data.pending_human);
    renderGraph(data.image_url);
    renderParticipants(data.agents_data || agentsData, data.connected_humans || {});
    expTurn.textContent = `Turn: ${data.turn || 0}`;
}

// Turn banner + input control
function renderTurnBanner(pendingHuman) {
    const wasMyTurn = lastPendingHuman === myName;
    const isMyTurn = pendingHuman === myName;

    if (isMyTurn) {
        turnBanner.className = 'exp-turn-banner my-turn';
        turnText.textContent = 'YOUR TURN — Write your message below!';
        inputArea.classList.remove('disabled');
        sendBtn.disabled = false;
        // Notify only on transition
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
        turnBanner.className = 'exp-turn-banner waiting';
        turnText.textContent = 'Processing...';
        inputArea.classList.add('disabled');
        sendBtn.disabled = true;
    }

    lastPendingHuman = pendingHuman;
}

// Transcript
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
                <span class="arrow">→</span>
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

// Graph
function renderGraph(imageUrl) {
    if (imageUrl) {
        graphImg.src = imageUrl + '?t=' + Date.now();
        graphImg.style.display = 'block';
        if (graphPlaceholder) graphPlaceholder.style.display = 'none';
    }
}

// Participants sidebar
function renderParticipants(agents, connectedHumans) {
    participantsBar.innerHTML = '';
    (agents || []).forEach(a => {
        if (a.is_observer) return;
        const badge = document.createElement('div');
        badge.className = 'agent-badge';
        const isHuman = a.is_human;
        const isOnline = connectedHumans && connectedHumans[a.name] && connectedHumans[a.name].online;
        const dotClass = isHuman ? (isOnline ? 'online' : 'offline') : 'is-ai';
        const typeLabel = isHuman ? (isOnline ? 'online' : 'offline') : 'AI';
        badge.innerHTML = `
            <span class="exp-participant-dot ${dotClass}" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${
                isHuman ? (isOnline ? 'var(--tone-positive)' : 'var(--tone-negative)') : 'var(--accent)'
            }"></span>
            <span class="agent-badge-name">${escapeHtml(a.name)}</span>
            <span class="agent-badge-nature">(${escapeHtml(a.nature)}) ${typeLabel}</span>
        `;
        if (a.name === myName) badge.style.border = '1px solid var(--accent)';
        participantsBar.appendChild(badge);
    });
}

// Send message
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
            headers: {'Content-Type': 'application/json'},
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

        // Auto-advance: this client triggers agent turns until next human
        await advanceUntilHumanTurn();

    } catch(e) {
        statusEl.textContent = 'Send error: ' + e.message;
    } finally {
        sendBtn.textContent = 'Send';
        // sendBtn.disabled is managed by renderTurnBanner
    }
}

// Auto-advance: call step() until a human's turn comes up
// ONLY called by the client who just submitted their message
async function advanceUntilHumanTurn() {
    if (isAdvancing) return;  // prevent concurrent loops
    isAdvancing = true;

    try {
        for (let i = 0; i < 20; i++) {  // safety limit: max 20 iterations
            await new Promise(r => setTimeout(r, 800));  // delay for visual pacing

            const res = await fetch('/api/experiment/step', { method: 'POST' });
            const data = await res.json();
            renderAll(data);

            if (data.waiting_for) {
                // Human's turn came up — stop
                break;
            }
            if (!data.ok) break;
        }
    } catch(e) {
        console.error('Auto-advance error:', e);
    } finally {
        isAdvancing = false;
    }
}

// Audio notification
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
    } catch(e) {}
}

// Escape HTML
function escapeHtml(s) {
    return String(s || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
```

---

### PHASE 4: Moderator UI Additions

#### 4.1 Edit `templates/index.html`

Add experiment mode entry point. In step1 div, after the "Join as participant" checkbox group, add:

```html
<div style="margin-top: 32px; padding-top: 24px; border-top: 1px solid var(--border); text-align: center;">
    <p style="color: var(--text-dim); font-size: 13px; margin-bottom: 12px;">— or —</p>
    <button id="experimentModeBtn" class="wizard-btn wizard-btn-primary" style="background: #7c3aed;">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="9" cy="7" r="3"/><circle cx="17" cy="11" r="3"/>
            <path d="M3 21v-2a4 4 0 014-4h4"/><path d="M21 21v-2a4 4 0 00-3-3.87"/>
        </svg>
        Multi-User Experiment
    </button>
    <p class="wizard-hint" style="margin-top: 8px;">10 humans + 5 agents with turn management</p>
</div>
```

Add a new wizard step (step3) for experiment setup — configuring human participants:

```html
<!-- Step 3: Experiment Setup (Human Participants) -->
<div id="step3" class="wizard-step">
    <div class="wizard-header compact">
        <h2>Multi-User Experiment Setup</h2>
        <p class="wizard-subtitle">Configure human participants and AI agents</p>
    </div>
    <div class="wizard-content-scroll">
        <div class="wizard-content">
            <label class="wizard-label">Environment</label>
            <select id="expEnvSelect" class="wizard-select">
                {% for e in envs %}
                    <option value="{{ loop.index0 }}">{{ e }}</option>
                {% endfor %}
            </select>

            <label class="wizard-label" style="margin-top: 20px;">Human Participants</label>
            <div id="humanList"></div>
            <button id="addHumanBtn" class="btn" style="margin-top: 8px;">+ Add Participant</button>

            <label class="wizard-label" style="margin-top: 20px;">AI Agents (count)</label>
            <div class="slider-container">
                <input type="range" id="expAgentSlider" class="agent-slider" min="1" max="10" value="5">
                <div class="slider-value"><span id="expAgentCount">5</span> agents</div>
            </div>

            <div id="expAgentList" style="margin-top: 12px;"></div>
        </div>
    </div>
    <div class="wizard-actions">
        <button id="backFromExp" class="wizard-btn wizard-btn-secondary">Back</button>
        <button id="startExpBtn" class="wizard-btn wizard-btn-success">Start Experiment</button>
    </div>
    <!-- Join links (shown after start) -->
    <div id="joinLinksPanel" style="display: none; margin-top: 20px;">
        <h3>Join Links (share with participants):</h3>
        <div id="joinLinksList"></div>
    </div>
</div>
```

#### 4.2 Edit `static/app.js`

Add experiment mode JS at the end of the file:

```javascript
// ==================== EXPERIMENT MODE ====================

const experimentModeBtn = el('experimentModeBtn');
const step3 = el('step3');

if (experimentModeBtn && step3) {
    const humanList = el('humanList');
    const addHumanBtn = el('addHumanBtn');
    const expAgentSlider = el('expAgentSlider');
    const expAgentCount = el('expAgentCount');
    const startExpBtn = el('startExpBtn');
    const backFromExp = el('backFromExp');
    const joinLinksPanel = el('joinLinksPanel');
    const joinLinksList = el('joinLinksList');

    let expHumans = [];
    const HUMAN_COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#2980b9','#27ae60','#c0392b','#8e44ad','#16a085'];

    // Switch to experiment wizard
    experimentModeBtn.addEventListener('click', () => {
        step1.classList.remove('active');
        step3.classList.add('active');
        // Init with 10 default humans
        expHumans = [];
        for (let i = 0; i < 10; i++) {
            expHumans.push({ name: `Participant${i+1}`, color: HUMAN_COLORS[i % HUMAN_COLORS.length] });
        }
        renderHumanList();
    });

    backFromExp.addEventListener('click', () => {
        step3.classList.remove('active');
        step1.classList.add('active');
    });

    addHumanBtn.addEventListener('click', () => {
        const idx = expHumans.length;
        expHumans.push({ name: `Participant${idx+1}`, color: HUMAN_COLORS[idx % HUMAN_COLORS.length] });
        renderHumanList();
    });

    function renderHumanList() {
        humanList.innerHTML = '';
        expHumans.forEach((h, i) => {
            const row = document.createElement('div');
            row.style.cssText = 'display:flex;gap:8px;align-items:center;margin-bottom:6px;';
            row.innerHTML = `
                <input type="color" value="${h.color}" style="width:32px;height:32px;" onchange="this._parent=${i}">
                <input type="text" value="${h.name}" class="wizard-input" style="flex:1;" placeholder="Name">
                <button class="btn" style="padding:4px 8px;font-size:12px;">✕</button>
            `;
            const colorInput = row.querySelector('input[type=color]');
            const nameInput = row.querySelector('input[type=text]');
            const removeBtn = row.querySelector('button');
            colorInput.addEventListener('change', () => { expHumans[i].color = colorInput.value; });
            nameInput.addEventListener('change', () => { expHumans[i].name = nameInput.value.trim(); });
            removeBtn.addEventListener('click', () => { expHumans.splice(i, 1); renderHumanList(); });
            humanList.appendChild(row);
        });
    }

    expAgentSlider.addEventListener('input', () => {
        expAgentCount.textContent = expAgentSlider.value;
    });

    startExpBtn.addEventListener('click', async () => {
        // Validate
        if (expHumans.length === 0) { alert('Add at least 1 human'); return; }
        if (expHumans.some(h => !h.name)) { alert('All humans must have names'); return; }

        // Build AI agent configs (simple defaults)
        const numAgents = parseInt(expAgentSlider.value);
        const aiAgents = [];
        for (let i = 0; i < numAgents; i++) {
            aiAgents.push({
                name: DEFAULT_AGENT_NAMES[i] || `Agent${i+1}`,
                nature: DEFAULT_ROLES[i % DEFAULT_ROLES.length],
                color: DEFAULT_COLORS[i] || DEFAULT_COLORS[i % DEFAULT_COLORS.length],
            });
        }

        startExpBtn.disabled = true;
        startExpBtn.textContent = 'Starting...';

        try {
            const data = await api('/api/experiment/start', 'POST', {
                env_index: parseInt(el('expEnvSelect').value),
                agents: aiAgents,
                humans: expHumans,
            });

            if (!data.ok) throw new Error(data.error);

            // Show join links
            joinLinksPanel.style.display = 'block';
            joinLinksList.innerHTML = '';
            const baseUrl = window.location.origin;
            for (const [name, link] of Object.entries(data.join_links || {})) {
                const div = document.createElement('div');
                div.style.cssText = 'margin:6px 0;padding:8px 12px;background:var(--surface);border-radius:8px;font-family:JetBrains Mono,monospace;font-size:13px;';
                const fullLink = baseUrl + link;
                div.innerHTML = `<strong>${escapeHtml(name)}:</strong> <a href="${fullLink}" target="_blank" style="color:var(--accent);">${escapeHtml(fullLink)}</a>`;
                joinLinksList.appendChild(div);
            }

            // Also show moderator dashboard
            agents = data.agents_data || [];
            running = true;
            setupWizard.style.display = 'none';
            mainApp.style.display = 'block';
            stepBtn.disabled = false;
            renderAgentBadges();
            renderHistory([]);
            statusEl.textContent = 'Experiment started. Share join links with participants.';

        } catch(e) {
            alert('Error: ' + e.message);
        } finally {
            startExpBtn.disabled = false;
            startExpBtn.textContent = 'Start Experiment';
        }
    });
}
```

---

## CRITICAL RULES & CONSTRAINTS

### 1. Return Type Consistency

`DialogueManager.step()` ALWAYS returns `Tuple[Dict, Optional[Dict]]`.
- Normal agent turn: `(record_dict, edge_dict_or_none)`
- Waiting for human: `({"waiting_for": "Name", "turn": N}, None)`

The web layer (`api_step`, `api_experiment_step`) checks `"waiting_for" in record` to distinguish.

### 2. Race Condition Prevention

**Problem**: If 10 clients all poll and see `pending_human === null`, multiple could try to call `/api/experiment/step` simultaneously, causing duplicate agent turns.

**Solution**: Only ONE client — the one who just submitted their human message — calls `advanceUntilHumanTurn()`. Other clients just observe via polling. This is enforced by:
- The `isAdvancing` lock in experiment.js
- The fact that `/api/experiment/step` is idempotent when `pending_human` is set (returns the same waiting status)
- Agent turns in `step()` are synchronous (Flask's default single-threaded mode)

**IMPORTANT**: Flask in debug mode is single-threaded. In production, use `app.run(threaded=False)` or add a threading lock in `AppState` for the step operation:

```python
import threading

class AppState:
    def __init__(self):
        # ... existing ...
        self._step_lock = threading.Lock()
```

And in `api_experiment_step`:
```python
    with STATE._step_lock:
        record, edge = STATE.dm.step()
```

### 3. Backward Compatibility Checklist

- `DialogueManager(multi_user_mode=False)` by default → all existing behavior unchanged
- `__post_init__` sets `multi_user_mode = len(humans) > 1` automatically
- Original `step()` code lives in the `else` branch, completely untouched
- Original endpoints `/api/start`, `/api/step`, `/api/user_message`, `/api/state` unchanged
- New endpoints all live under `/api/experiment/*` namespace
- `inject_user_turn()` method unchanged (still used by `/api/user_message`)

### 4. URL Encoding for Cyrillic Names

Human names can be in Russian (Иван, Мария, etc.). The join links use `urllib.parse.quote()` for URL-safe encoding. The `experiment.js` uses `decodeURIComponent()` to decode the name from the URL.

### 5. No External Dependencies

- No npm, no webpack, no socket.io
- Plain Flask + vanilla JS
- Polling at 2s interval is sufficient for 15 participants
- AudioContext API for notification sound (no audio files)

---

## FILE CHANGES SUMMARY

| File | Action | What Changes |
|------|--------|-------------|
| `config.py` | EDIT | Add 2 constants: `HUMAN_TO_AGENT_RATIO`, `HUMAN_TURN_TIMEOUT` |
| `dialogue_manager.py` | EDIT | Add 5 fields, 4 methods (`_pick_next_human`, `_pick_agent_for_multi_user`, `submit_human_turn`, modified `step()`), update `build_messages()` human_block |
| `web_app.py` | EDIT | Extend `AppState`, add `_step_lock`, extend `reset()`, modify `api_step()`, add 6 new endpoints under `/api/experiment/*` |
| `templates/experiment.html` | CREATE | Full participant view HTML |
| `static/experiment.js` | CREATE | Full participant frontend JS (~250 lines) |
| `templates/index.html` | EDIT | Add experiment mode button + step3 wizard for experiment config |
| `static/app.js` | EDIT | Add experiment mode JS at end (~100 lines) |

**Files NOT changed** (verify these still work): `agents.py`, `profiles.py`, `prompts.py`, `human_io.py`, `validators.py`, `analytics.py`, `scientific_analytics.py`, `advanced_analytics.py`, `observer_agent.py`, `visualize.py`, `io_logger.py`, `static/style.css`

---

## EXECUTION ORDER

1. `config.py` — 2 lines added
2. `dialogue_manager.py` — core multi-user logic (most critical, test carefully)
3. `web_app.py` — new endpoints + AppState changes
4. `templates/experiment.html` — new file
5. `static/experiment.js` — new file
6. `templates/index.html` — add experiment wizard step
7. `static/app.js` — add experiment mode JS

After each file, mentally verify it doesn't break existing imports/behavior.

---

## TESTING PLAN

After implementation, test in this order:

1. **Regression**: Start normal session via wizard with 3 agents, no humans → dialogue works
2. **Regression**: Start with "Join as participant" checked → single User mode works
3. **Experiment setup**: Click "Multi-User Experiment" → configure 3 humans + 2 agents → start
4. **Join links**: Verify join links appear with correct URL-encoded names
5. **Participant join**: Open `/experiment?name=Participant1` in new tab → join succeeds
6. **Turn management**: Click "Next message" on moderator → should show "Waiting for [Human]"
7. **Human message**: In participant tab, type message and send → message appears in transcript
8. **Auto-advance**: After human sends, agent responds automatically, then next human is prompted
9. **Ratio enforcement**: Verify pattern: Human → Human → Agent → Human → Human → Agent
10. **Addressee priority**: If Agent addresses Human3, Human3 should get next turn
11. **Multiple browsers**: Open 3 browser tabs as 3 different humans → all see updates
12. **Timeout**: Wait 5+ minutes without responding → human is skipped

Start implementing now. Read each file before editing it. Do NOT refactor unrelated code.
