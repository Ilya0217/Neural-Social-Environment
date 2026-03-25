# Последние обновления проекта

## 2026-03-18

### 1. Обновление научных теорий (post-2010)

Все 8 научных фреймворков заменены на современные (2010–2025):

| # | Было | Стало |
|---|---|---|
| 1 | Wheelan (2009) — Group Development | **Shuffler et al. (2018) + Mathieu et al. (2017)** — Team Temporal Dynamics |
| 2 | Borgatti et al. (2009) — SNA | **Contractor et al. (2012)** — Multidimensional Networks in Teams |
| 3 | Burt (2004) — Structural Holes | **Reagans et al. (2016) + Leenders et al. (2016)** — Network Dynamics & Team Performance |
| 4 | Bunt et al. (2017) — ISO 24617-2 | **Bunt et al. (2020)** — ISO 24617-2 (обновлённая ревизия) |
| 5 | Clark (1996) — Speech Acts | **Jurafsky & Martin (2024)** — Computational Dialogue Analysis |
| 6 | Herring (2004) — CMC Discourse | **Tausczik & Pennebaker (2010) + Gonzales et al. (2010)** — LIWC & Language Style Matching |
| 7 | Russell & Barrett (1999) — Circumplex | **Cowen & Keltner (2017) + Demszky et al. (2020)** — 27 Emotions & GoEmotions |
| 8 | VADER / Hutto (2014) — Sentiment | **Mohammad et al. (2018)** — SemEval Affect in Tweets |

**Затронутые файлы:** `scientific_analytics.py`, `analytics.py`, `hypothesis_validator.py`

---

### 2. Observer Agent — LLM-наблюдатель с методологической триангуляцией

**Новый файл:** `observer_agent.py`

Не участвует в диалоге. Периодически анализирует контекст через LLM и сравнивает свои качественные выводы с алгоритмическими метриками.

Генерирует:
- Group dynamics, leadership pattern, coalition analysis
- Communication barriers, emotional undercurrent
- Triangulation: agreements / divergences / novel insights / convergence score

Научная основа: методологическая триангуляция (Denzin, 2012; Creswell & Plano Clark, 2017).

**API endpoints (в `web_app.py`):**
- `POST /api/observer` — запустить анализ
- `GET /api/observer/history` — все отчёты + summary

---

### 3. Контролируемые A/B эксперименты со статистическим анализом

**Новый файл:** `experiments/ab_experiments.py`

Фреймворк для проведения контролируемых экспериментов с автоматическим сбором метрик и статистическим анализом.

**Предустановленные эксперименты:**
- `facilitator_effect` — с/без фасилитатора
- `group_size` — 3 vs 4 агента
- `environment_effect` — университет vs стартап

**Статистические тесты:**
- Welch's t-test (параметрический)
- Mann-Whitney U (непараметрический)
- Cohen's d (размер эффекта)
- 95% confidence intervals

**Запуск:**
```bash
python -m agent_dialogue_sim.experiments.ab_experiments \
    --experiment facilitator_effect --runs 10 --turns 15
```

Генерирует `report.md` с таблицами p-values и effect sizes + `results.json` с сырыми данными.

---

### 4. Big Five личностные профили агентов (OCEAN)

**Изменённый файл:** `agents.py`

Каждый агент теперь имеет профиль Big Five (Goldberg, 1990):
- Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism (0.0–1.0)

**Предустановленные профили:**
- Explorer: O=0.9, C=0.4, E=0.7, A=0.6, N=0.3 (креативный, открытый)
- Critic: O=0.5, C=0.8, E=0.4, A=0.3, N=0.5 (организованный, прямой)
- Facilitator: O=0.6, C=0.7, E=0.8, A=0.9, N=0.2 (общительный, кооперативный)

Профиль автоматически встраивается в system prompt и влияет на поведение агента.

Кастомные профили:
```python
from agent_dialogue_sim.agents import Agent, BigFiveProfile
agent = Agent(
    name="Rebel", nature="explorer", color="#FF0000",
    big_five=BigFiveProfile(openness=0.9, agreeableness=0.1, neuroticism=0.8)
)
```

---

### 5. Поддержка OpenRouter

**Изменённые файлы:** `config.py`, `main.py`, `web_app.py`, `experiments/run_experiments.py`

Добавлена переменная `OPENAI_BASE_URL` — позволяет использовать OpenRouter или другие OpenAI-совместимые провайдеры.

```env
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

---

### 6. Порт изменён на 8080

`web_app.py` теперь запускается на порту **8080** вместо 5000 (конфликт с AirPlay на macOS).

---

### 7. Радикальная переработка реалистичности диалогов

**Изменённые файлы:** `prompts.py`, `dialogue_manager.py`, `config.py`, `validators.py`, `visualize.py`

**Проблема:** агенты звучали шаблонно — каждый ответ начинался с "I think...", "That's a great point!". Промпты повторяли "be real, be natural" 15+ раз, что парадоксально делало ответы ещё более роботизированными.

**Решение:**

**`prompts.py`** — полностью переписан:
- Каждый агент — **конкретный человек с биографией и привычками**:
  - Alex (26, data analyst, DJ) — перескакивает между мыслями, говорит "oh wait", trailing "like..."
  - Jordan (34, project manager, шахматист) — сухой юмор, начинает с "Look," или "Here's the thing", пауза перед ответом
  - Sam (29, HR) — соединяет идеи людей, использует имена, смеётся "haha", иногда overshares
- `HUMAN_STYLE` — список **антипаттернов** (чего НЕ делать):
  - "DON'T start every reply with 'I think...'"
  - "DON'T summarize what others said"
  - "DON'T be balanced every time — real people have lopsided opinions"
  - "DO interrupt your own thoughts", "DO use filler words", "DO be blunt sometimes"
- `SESSION_GOAL` — сокращён с 20 строк до 4

**`dialogue_manager.py`** — `build_messages()` упрощён:
- Системный промпт сокращён с ~2000 до ~500 слов
- Убраны все повторяющиеся инструкции
- User prompt: одна строка вместо абзаца

**`config.py`:**
- `TEMPERATURE` 0.85 → **0.95** (более непредсказуемые ответы)
- `MAX_TOKENS` 600 → **300** (короче = естественнее)

**`validators.py`:**
- Порог "слишком длинный ответ" увеличен с 50 до 80 слов

**`visualize.py`:**
- Matplotlib backend переключён на `Agg` (исправлен краш графиков в Flask-потоке)

---

### 8. Визуальные обновления UI

**Изменённые файлы:** `templates/index.html`, `static/app.js`, `static/style.css`, `web_app.py`

**Новая вкладка "Observer Agent":**
- Третья вкладка в интерфейсе: 💬 Dialogue / 🔬 Scientific Analysis / 🔭 Observer Agent
- Кнопка "Run Observer Analysis" — запускает LLM-наблюдателя через `POST /api/observer`
- Отображает полный отчёт: group dynamics, leadership, coalitions, communication barriers
- Панель Triangulation Summary с карточками метрик: Convergence %, Agreements, Divergences
- Блок Novel Insights — инсайты, которые алгоритмические метрики не ловят

**Big Five мини-бары в бейджах агентов:**
- Рядом с именем каждого агента отображаются 5 цветных полосок (O C E A N)
- Высота заполнения полоски = значение трейта (0.0–1.0)
- Tooltip при наведении показывает числовые значения каждого трейта
- API `/api/start` теперь возвращает `agents_data` с Big Five профилями

**Обновлённый список теорий:**
- На вкладке Scientific Analysis обновлён HTML-список теорий на актуальные (2010–2025)

**Исправления:**
- Табы теперь корректно переключаются между всеми тремя вкладками
- При новой сессии все вкладки сбрасываются

---

### 9. Big Five для всех агентов + Observer как +1

**Изменённые файлы:** `agents.py`, `web_app.py`, `static/app.js`, `static/style.css`

**Big Five для любого количества агентов:**
- Добавлены пресеты для 8 ролей: explorer, critic, facilitator, scientist, manager, mediator, innovator, analyst
- Для любых других ролей Big Five **генерируется детерминистически** на основе хеша названия роли (md5 → 5 значений 0.2–0.9)
- Гарантия: одна и та же роль всегда получит одинаковый профиль

**Observer Agent автоматически +1:**
- При старте сессии Observer автоматически добавляется в `agents_meta` как +1 к выбранным агентам
- В UI отображается с пунктирной рамкой, иконкой 🔭 и подписью "(observer)"
- Не участвует в диалоге — только анализирует
- Выбираешь 3 агента → в UI видно 3 + Observer = 4 бейджа

---

### 10. Apple Glass Design

**Полностью переписан:** `static/style.css`

- Чистый чёрный фон (#000) + полупрозрачные стеклянные панели (`backdrop-filter: blur(40px) saturate(180%)`)
- Системные цвета Apple: #0A84FF (синий), #30D158 (зелёный), #FF453A (красный), #BF5AF2 (фиолетовый)
- Белый слайдер-thumb, тонкие inset-тени, iOS spring-curve анимации
- SF Pro / system-ui шрифт, letter-spacing на заголовках
- Минимальные бордеры `rgba(255,255,255,.08)`, hover-эффекты через opacity

---

### 11. LLM-инициализация агентов: Big Five + персона генерируются нейросетью

**Изменённые файлы:** `profiles.py`, `web_app.py`

Раньше Big Five и персона создавались из шаблонов. Теперь при старте сессии LLM анализирует конфигурацию каждого агента (роль + анкета) и генерирует:

**1. Big Five профиль (`_generate_big_five_via_llm`)**
- LLM получает роль, бэкграунд, social style, conflict approach и т.д.
- Генерирует психологически консистентные OCEAN-скоры (0.1–1.0)
- Пример: "избегает конфликтов" → высокий Agreeableness, низкий Neuroticism

**2. Персона-описание (`_generate_persona_via_llm`)**
- LLM создаёт 6-8 строк уникального описания: backstory, speech quirks, triggers, signature phrases
- Каждый агент получает уникальную персону вместо шаблонной

**Fallback:** если LLM недоступен — используется хеш-генерация Big Five + шаблонная персона.

**Процесс при запуске:**
1. Пользователь настраивает агентов в wizard (имя, роль, анкета)
2. При нажатии "Start Dialogue" → LLM генерирует Big Five + персону для каждого агента
3. Результат отображается в бейджах (OCEAN бары) и влияет на поведение агента в диалоге

---

### 12. Emotional Contagion Tracker

**Новый файл:** `advanced_analytics.py`

Отслеживает распространение эмоций по сети агентов на основе теории эмоционального заражения (Barsade, 2002; Hatfield et al., 1994).

Метрики:
- **Contagion rate** — % взаимодействий с распространением тона
- **Average lag** — среднее кол-во ходов для заражения
- **Infection matrix** — попарная вероятность заражения (A→B)
- **Most contagious / most susceptible agent** — кто заражает и кто заражается
- **Tone trajectories** — траектория тона каждого агента по ходам

---

### 13. Language Style Matching (LSM) — реальный расчёт

**Файл:** `advanced_analytics.py`

Реализован полноценный расчёт LSM по методологии Gonzales et al. (2010) и LIWC (Tausczik & Pennebaker, 2010):

- 8 категорий function words: pronouns, articles, prepositions, auxiliaries, negations, conjunctions, quantifiers, fillers
- **Pairwise LSM** — матрица стилистического совпадения для каждой пары агентов
- **Group LSM** — общий показатель (>0.8 = высокая сплочённость)
- **Per-category scores** — какие категории слов совпадают больше/меньше
- **Agent profiles** — лингвистический профиль каждого агента

Формула: `LSM = 1 - |rate_a - rate_b| / (rate_a + rate_b + ε)`

---

### 14. Bad Apple эксперимент (Felps et al., 2006)

**Файл:** `experiments/ab_experiments.py` — добавлен эксперимент `bad_apple`

Предустановленный контролируемый эксперимент:
- **Control:** 3 сбалансированных агента (Explorer + Critic + Facilitator)
- **Treatment:** те же 3 + токсичный агент "Viper" (Neuroticism=0.95, Agreeableness=0.1)
  - Использует сарказм, отвергает идеи, говорит "that won't work", "waste of time"
  - Не предлагает альтернатив — только критикует

Запуск:
```bash
python -m agent_dialogue_sim.experiments.ab_experiments \
    --experiment bad_apple --runs 10 --turns 15
```

Измеряет: avg_tone, reciprocity, question_rate, emotion_entropy + статистика (t-test, Cohen's d).

---

### 15. Discourse Coherence Analysis (Graesser et al., 2014)

**Файл:** `advanced_analytics.py`

Измеряет семантическую связность диалога:
- **Overall coherence** — средний Jaccard similarity между соседними репликами
- **Thread continuity** — % реплик, использующих слова из сообщения адресата
- **Topic drift points** — ходы, где когерентность резко падает (mean - 1.5σ)
- **Per-agent coherence** — насколько каждый агент связан с предыдущим контекстом

Позволяет количественно определить: "говорят ли агенты по делу или мимо друг друга?"

---

### 16. Вкладка Advanced Analytics в UI

**Изменённые файлы:** `templates/index.html`, `static/app.js`, `web_app.py`

- Новая вкладка 📈 Advanced Analytics (между Scientific и Observer)
- Кнопка "Load Analysis" → вызывает `GET /api/advanced`
- Карточки метрик: Contagion Rate, Group LSM, Coherence, Thread Continuity
- Полный markdown-отчёт с матрицами заражения, LSM-таблицами, bar-визуализацией
