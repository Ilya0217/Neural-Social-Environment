# Отрывки кода для презентации: РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ

## 1. 📝 Промпты инициализации агентов

**Название для презентации:** "Системные промпты для ролевого моделирования"

**Файл:** `prompts.py` (строки 13-79)

**Описание:** Базовые промпты, определяющие личность и стиль общения агентов

**Ключевой код:**
```python
BASE_SYSTEM_PROMPTS = {
    "explorer": """You are a real person named Explorer in a real meeting/discussion...
    - You're naturally curious and ask questions...
    - Use natural speech patterns: "I'm wondering if...", "That's a good point"...
    - Address others with informal "you" (ты)...
    """,
    
    "critic": """You are a real person named Critic...
    - You naturally spot risks and gaps...
    - Use natural, thoughtful language: "I'm concerned that..."...
    """,
    
    "facilitator": """You are a real person named Facilitator...
    - You naturally help people connect and move forward...
    - Use warm, inclusive language: "I think we're all saying..."...
    """
}
```

**Что показать:**
- Структура промптов (личность + стиль общения)
- Использование неформального "ты" для естественности
- Интеграция с Chain-of-Thought (см. `advanced_prompts.py`)

---

## 2. 🔗 API и веб-интерфейс

**Название для презентации:** "RESTful API для управления симуляцией"

**Файл:** `web_app.py` (строки 127-232)

**Описание:** Flask API endpoints для управления диалогом и получения результатов

**Ключевой код:**
```python
@app.route("/api/start", methods=["POST"])
def api_start():
    """Инициализация сессии диалога"""
    # Создание DialogueManager, инициализация агентов
    # Возврат метаданных агентов

@app.route("/api/step", methods=["POST"])
def api_step():
    """Выполнение одного шага диалога"""
    record, _ = STATE.dm.step()
    # Генерация графа каждые 5 шагов
    # Вычисление метрик и гипотез
    # Валидация гипотез
    return jsonify({
        "ok": True,
        "record": record,
        "turn": STATE.dm.turn_no,
        "history": STATE.dm.history[-20:],
        "image_url": image_url,  # Граф взаимодействий
        "metrics_md": metrics_md,
        "scientific_hypotheses": STATE.last_scientific_hyps,
        "validation_report": STATE.last_validation_report
    })
```

**Что показать:**
- REST API endpoints (`/api/start`, `/api/step`, `/api/state`)
- JSON-ответы с метриками, гипотезами, графами
- Интеграция с фронтендом (React/vanilla JS)

---

## 3. 📊 Реализация графа взаимодействий

**Название для презентации:** "Визуализация социальной сети агентов (NetworkX + Matplotlib)"

**Файл:** `visualize.py` (строки 108-297)

**Описание:** Построение направленного графа с цветовой кодировкой тона и метриками

**Ключевой код:**
```python
def draw_interactions_pro(
    agents_meta: Dict[str, Dict[str, str]],
    history: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    window: int = 12,
):
    """Визуализация:
    - Nodes: size ~ #messages, fill from agent color, outline by last emotion
    - Edges: color = avg tone (-1..1), width ~ frequency within window
    """
    # Создание направленного графа
    G = nx.DiGraph()
    for a in agents:
        G.add_node(a)
    pos = nx.spring_layout(G, seed=seed, k=1.2)
    
    # Узлы: размер зависит от количества сообщений
    sizes = [base_node_size + 80 * math.sqrt(max(talks, 1))]
    # Цвет узла = цвет агента, обводка = эмоция
    
    # Рёбра: цвет = тон (зелёный=positive, красный=negative)
    # Ширина = частота взаимодействий
    for rec in history[-window:]:
        src = rec["speaker"]
        dst = rec.get("target")
        tone = rec.get("tone", "neutral")
        col = tone_to_color(TONE_SCORE.get(tone, 0.0))
        # Рисуем дугу с цветом тона
```

**Что показать:**
- Использование NetworkX для построения графа
- Визуальное кодирование: размер узла = активность, цвет ребра = тон
- Панель метрик справа от графа

---

## 4. 💾 Экспорт диалога в различные форматы

**Название для презентации:** "Многформатный экспорт данных для анализа"

**Файл:** `export_manager.py` (строки 19-311)

**Описание:** Экспорт диалога в CSV, Excel, JSON, Markdown с метриками

**Ключевой код:**
```python
class ExportManager:
    def export_to_csv(self, history, filename):
        """Экспорт в CSV для статистического анализа"""
        fieldnames = [
            "turn", "timestamp", "speaker", "speaker_nature",
            "target", "reply", "tone", "emotion", "word_count",
            "validation_issues", "env_context"
        ]
        # Запись в CSV с кодировкой UTF-8
    
    def export_to_excel(self, history, metrics, filename):
        """Экспорт в Excel с двумя листами"""
        # Лист 1: Диалог с форматированием
        # Лист 2: Метрики (таблицы, графики)
        # Автоматическая подгонка ширины колонок
    
    def export_to_json(self, history, metrics, filename):
        """Полная структурированная информация"""
        return {
            "metadata": {...},
            "dialogue": history,
            "metrics": metrics,
            "scientific_analysis": {...}
        }
    
    def export_research_report(self, history, metrics, filename):
        """Research-ready Markdown отчёт"""
        # Executive summary
        # Детальные метрики
        # Per-agent анализ
        # Sample excerpts
```

**Что показать:**
- Множество форматов (CSV, Excel, JSON, Markdown)
- Структурированные данные для анализа
- Research-ready отчёты

---

## 5. ✅ Валидация гипотез с примерами из диалога

**Название для презентации:** "Автоматическая проверка гипотез с доказательствами"

**Файл:** `hypothesis_validator.py` (строки 196-214, 245-252)

**Описание:** Проверка научных гипотез на основе метрик и примеров из диалога

**Ключевой код:**
```python
def _validate_network_structure(self, hyp_id, hypothesis, current, previous, history):
    """Проверка гипотез о структуре сети"""
    # Извлечение метрик
    centrality = scientific.centrality
    max_central = max(centrality.items(), key=lambda x: x[1].get("total_centrality", 0))
    evidence.append(f"📊 Метрика: Максимальная центральность: {max_val:.0%}")
    
    # Примеры из диалога
    if history:
        central_messages = [r for r in history[-15:] if r.get("speaker") == central_agent]
        if len(central_messages) >= 3:
            evidence.append(f"💬 Примеры активности центрального агента:")
            for r in central_messages[:3]:
                evidence.append(
                    f"  • Ход {r.get('turn')}: {central_agent} → {target}: "
                    f"\"{r.get('reply', '')[:70]}...\""
                )
    
    return HypothesisValidation(
        hypothesis_id=hyp_id,
        hypothesis_text=finding,
        status="confirmed" if confidence > 0.6 else "partial",
        confidence=confidence,
        evidence=evidence  # Метрики + примеры из диалога
    )
```

**Что показать:**
- Комбинация количественных метрик и качественных примеров
- Автоматическое извлечение цитат из диалога
- Оценка уверенности в подтверждении гипотезы

---

## 6. 🧠 Научный анализ на основе теорий

**Название для презентации:** "Интеграция научных фреймворков для анализа диалога"

**Файл:** `scientific_analytics.py` (строки 779-841)

**Описание:** Применение современных научных теорий для анализа социальных взаимодействий

**Ключевой код:**
```python
def compute_scientific_analysis(history, agents, window_size=20):
    """
    Интеграция научных фреймворков:
    - Dimensional Emotion Model (Russell & Barrett, 1999)
    - ISO 24617-2 Dialogue Act Taxonomy (Bunt et al., 2017)
    - Social Network Analysis (Borgatti et al., 2009)
    - Social Capital Theory (Burt, 2004)
    - Integrated Model of Group Development (Wheelan, 2009)
    """
    # Анализ эмоций
    emotion_dist = compute_emotion_distribution(emotions)
    
    # Анализ диалоговых актов
    dialogue_acts = compute_dialogue_act_profile(recent)
    
    # Анализ социальной сети
    centrality = compute_degree_centrality(history, agents)
    density = compute_network_density(history, agents)
    
    # Социальный капитал
    social_capital = compute_social_capital_indices(history, agents)
    
    # Стадия развития группы
    group_stage = detect_group_development_stage(history)
    
    return ScientificAnalysisResult(
        emotion_distribution=emotion_dist,
        dialogue_act_profile=dialogue_acts,
        centrality=centrality,
        network_density=density,
        social_capital=social_capital,
        group_stage=group_stage
    )
```

**Что показать:**
- Интеграция 6+ научных фреймворков
- Вычисление метрик на основе теорий
- Генерация гипотез с литературными ссылками

---

## 7. 🔄 Управление диалогом и выбор следующего говорящего

**Название для презентации:** "Алгоритм выбора говорящего с гарантией участия всех агентов"

**Файл:** `dialogue_manager.py` (строки 471-540)

**Описание:** Логика выбора следующего говорящего с предотвращением доминирования

**Ключевой код:**
```python
def step(self):
    """Выполнение одного шага диалога"""
    # Определяем агентов, которые не говорили в последних N ходах
    recent_window = len(self.agents)
    recent_speakers = {h["speaker"] for h in self.history[-recent_window:]}
    inactive_agents = all_agent_names - recent_speakers
    
    # Если есть неактивные агенты, выбираем одного из них
    if inactive_agents and len(self.history) >= recent_window:
        # Выбираем агента с наименьшим количеством сообщений
        # ...
    
    # Построение промпта с контекстом
    messages = self.build_messages(idx)
    
    # Генерация ответа через LLM API
    parsed = self.model_turn(messages, allowed_targets, agent=speaker)
    
    # Валидация и запись в историю
    record = {
        "turn": self.turn_no,
        "speaker": speaker.name,
        "reply": parsed.reply,
        "tone": parsed.tone,
        "emotion": parsed.emotion,
        "target": target
    }
    self.history.append(record)
```

**Что показать:**
- Алгоритм предотвращения доминирования
- Построение контекстного промпта
- Интеграция с LLM API

---

## Рекомендации по использованию в презентации:

1. **Промпты инициализации** — показать структуру и примеры
2. **API интерфейс** — показать JSON-ответы и endpoints
3. **Граф взаимодействий** — показать визуализацию с объяснением кодирования
4. **Экспорт диалога** — показать примеры экспортированных файлов
5. **Валидация гипотез** — показать отчёт с метриками и примерами
6. **Научный анализ** — показать интеграцию теорий
7. **Управление диалогом** — показать алгоритм выбора говорящего

**Формат для презентации:**
- Код: синтаксис-подсветка, краткие комментарии
- Описание: что делает, зачем нужно
- Результат: примеры выходных данных

