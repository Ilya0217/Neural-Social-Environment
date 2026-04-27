# Отчёт о состоянии научной работы — 26 апреля 2026

**Тема:** Разработка комплекса средств для использования больших языковых моделей в социальной сфере.

**Объект:** Многоагентная диалоговая система на базе LLM, имитирующая социальное взаимодействие в малых группах с автоматической проверкой статистических гипотез.

**Главный исследовательский вопрос (RQ-main):**
> Способна ли многоагентная LLM-система при персонологической параметризации (Big Five) и многоуровневой валидации воспроизводить статистически значимые социально-психологические закономерности малых групп с эффектами размера ≥ medium (Cohen's d ≥ 0.5 / η² ≥ 0.06)?

---

## 1. Теоретическая основа

В работе используются 5 теорий социальной психологии и компьютерной лингвистики, на базе которых сформулированы 5 формальных статистических гипотез.

### 1.1 Теория личности — Big Five / OCEAN
- **Источник:** Goldberg (1990), Costa & McCrae (1992); связь с речевым поведением — Mehl, Gosling & Pennebaker (2006).
- **Содержание:** пятифакторная модель личности (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism). Каждый параметр ∈ [0, 1].
- **Эмпирический факт:** экстраверты в речи продуцируют больше слов и более длинные реплики; добросовестные — больше task-ориентированных актов.
- **Связана с:** **гипотезой H1**.

### 1.2 Интегрированная модель развития группы — Wheelan (2009)
- **Источник:** Wheelan, S. A. (2009). Group Size, Group Development, and Group Productivity. *Small Group Research*, 40(2), 247–262.
- **Содержание:** 4 стадии развития малой группы — **forming → storming → norming → performing**. Стадии диагностируются по соотношению категорий диалоговых актов (доля socio-emotional vs task; доля negative socio для storming; доля agreements для norming).
- **Связана с:** **гипотезой H3**.

### 1.3 Этика биомедицинских принципов — Beauchamp & Childress (2019)
- **Источник:** Beauchamp, T. L., & Childress, J. F. (2019). *Principles of Biomedical Ethics* (8th ed.).
- **Содержание:** 4 моральных принципа — fairness, autonomy, inclusivity, non-maleficence. Применяются для классификации речевых нарушений.
- **Связана с:** **гипотезой H5** (эффективность многоуровневой валидации).

### 1.4 Chain-of-Thought-промптинг — Wei et al. (2022)
- **Источник:** Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*.
- **Содержание:** структурирование промпта в виде шагов рассуждения улучшает качество ответов LLM на задачах, требующих многошагового вывода. В диалоговом контексте — должно повышать долю actionable-предложений.
- **Связана с:** **гипотезой H6**.

### 1.5 Computer-Mediated Discourse Analysis — Herring (2004)
- **Источник:** Herring, S. C. (2004). Computer-mediated discourse analysis. *Designing for Virtual Communities in the Service of Learning*.
- **Дополнение:** Dunbar et al. (1995) — структура свободно формирующихся малых групп.
- **Содержание:** структура турн-тейкинга в реальных малых группах характеризуется коэффициентом Джини в диапазоне [0.2, 0.45]. Если LLM-симуляция воспроизводит человекоподобный паттерн, её распределение Джини должно быть статистически эквивалентным человеческому корпусу.
- **Связана с:** **гипотезой H7**.

### 1.6 Вспомогательные таксономии (используются для разметки)
- **ISO 24617-2** (Bunt et al., 2020) — таксономия диалоговых актов. Используется для классификации каждой реплики по 12 категориям (positive socio, negative socio, task questions, task answers).
- **Cowen & Keltner (2017)** — высокоразмерная модель эмоций (27 категорий).
- **Cohen (1988)** — пороги размеров эффектов (small/medium/large).

---

## 2. Гипотезы (математически строгая формулировка)

Все гипотезы — в форме нулевой H₀ и альтернативной H₁ с указанием статистического критерия, уровня значимости α и минимального размера эффекта.

### Сводная таблица

| ID | RQ | Тест | α | Effect ≥ | План N |
|---|---|---|---|---|---|
| **H1** | Big Five влияет на длину реплики | ANOVA + Tukey HSD | 0.05 | η² ≥ 0.06 | 480 диалогов (160/группа × 3) |
| **H3** | Группа проходит стадии Wheelan | χ² goodness-of-fit + Cox–Stuart | 0.01 | w ≥ 0.3 + положительный тренд | 30 диалогов × 60 ходов |
| **H5** | Валидация снижает токсичность | z-test двух пропорций (1-side) | 0.01 | Δp ≥ 0.05 | 60 диалогов (30/группа) |
| **H6** | CoT повышает actionability | Welch t-test (1-side) | 0.05 | d ≥ 0.5 | 110 диалогов (55/группа) |
| **H7** | LLM-Gini ≈ human-Gini | TOST (Mann–Whitney) | 0.05 | \|Δmedian\| ≤ 0.1 | 70 LLM + 70 human |

### H1. Влияние экстраверсии на длину реплики

- **H₀:** μ_low = μ_mid = μ_high (средняя длина реплики не зависит от уровня экстраверсии).
- **H₁:** ∃ i, j: μ_i ≠ μ_j.
- **IV:** Big Five `extraversion` ∈ {low=0.2, mid=0.5, high=0.8}; остальные OCEAN зафиксированы на 0.5.
- **DV:** средняя длина реплики агента в словах за диалог (`avg_reply_length`).
- **Решающее правило:** reject H₀ если **p < 0.05 И η² ≥ 0.06**.

### H3. Воспроизведение стадий Wheelan

- **H₀:** последовательность стадий случайна (равномерная марковская цепь, каждый переход p = 1/4).
- **H₁:** наблюдаемая матрица переходов P_obs ≠ P_uniform.
- **Дополнительно:** тест Кокса–Стюарта проверяет монотонный тренд forming → performing.
- **Решающее правило:** reject H₀ если **χ² p < 0.01 И w ≥ 0.3 И Cox–Stuart p < 0.05**.

### H5. Эффективность валидации

- **H₀:** p_A = p_B (доля токсичных реплик одинакова с/без валидации).
- **H₁:** p_A < p_B (валидация уменьшает долю).
- **A/B:** treatment = `enable_validation=True`, control = `False`.
- **Inter-rater reliability:** для разметки токсичности требуется κ Коэна ≥ 0.7 (LLM-судья + ручная подвыборка ≥ 100 реплик).
- **Решающее правило:** reject H₀ если **p < 0.01 И \|Δp\| ≥ 0.05**.

### H6. Эффективность Chain-of-Thought

- **H₀:** μ_A = μ_B (actionability_rate одинаков с CoT и без).
- **H₁:** μ_A > μ_B (CoT улучшает).
- **A/B:** treatment = `enable_cot=True`, control = базовый промпт.
- **DV:** `actionability_rate` диалога — доля реплик, содержащих action-маркеры (модальные глаголы should/could/let's, императивы, конкретные шаги). Реализована в `analytics.py:actionability_rate`.
- **Решающее правило:** reject H₀ если **p < 0.05 И d ≥ 0.5**.

### H7. Эквивалентность турн-тейкинга человеческому

- **H₀_TOST:** медианная разность Gini-LLM vs Gini-human лежит вне [-0.1, +0.1] — распределения практически отличаются.
- **H₁_TOST:** \|Δmedian\| < 0.1 — распределения эквивалентны.
- **Источник эталона:** AMI Meeting Corpus (Carletta et al., 2005), 100 часов записанных совещаний по 4 человека с разметкой говорящих.
- **Решающее правило:** TOST — оба односторонних теста должны дать p < 0.05 для подтверждения эквивалентности.

---

## 3. План выборок и обоснование (Power Analysis)

Размеры выборок определены формальным power-анализом с использованием `power_analysis.py`. Целевая мощность **1−β = 0.80**, ожидаемый эффект — **medium** (d=0.5, η²=0.06, w=0.3).

| Гипотеза | Тест | Plan N | Required N | Статус |
|---|---|---|---|---|
| H1 | ANOVA, k=3 | 160/группа | 158/группа | ✅ |
| H3 | χ² (4×4 матрица) | ≈300 transitions | 283 | ✅ |
| H5 | z-test 2 prop (1-side) | 900 реплик/группа | 180 | ✅ |
| H6 | Welch t (1-side) | 55/группа | 51 | ✅ |
| H7 | TOST (MWU, ARE-coorr) | 70/группа | 66 | ✅ |

**Sensitivity (для t-test, α=0.05, power=0.80):**

| Cohen d | Required N per group |
|---|---|
| 0.2 (small) | 310 |
| 0.3 | 139 |
| 0.4 | 78 |
| **0.5 (medium)** | **51** |
| 0.6 | 36 |
| 0.7 | 26 |
| 0.8 (large) | 21 |
| 1.0 | 14 |

### Поправка на множественные сравнения

Тестируется семейство из 5 гипотез. Применяется **Holm–Bonferroni**:
- p-values сортируются по возрастанию: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍₅₎.
- H_(i) отвергается, если p₍ᵢ₎ < α / (K − i + 1), K = 5.
- Реализация: `statistical_tests.holm_bonferroni()`.

---

## 4. Конфигурация агентов

### 4.1 Базовая модель агента

```python
@dataclass
class Agent:
    name: str
    nature: str      # роль: explorer / critic / facilitator / analyst / innovator
    color: str       # цвет для визуализации
    persona: str = ""
    big_five: Optional[BigFiveProfile] = None
```

`BigFiveProfile` содержит 5 параметров OCEAN ∈ [0, 1]; преобразуется в текстовое описание через `to_prompt_description()` и инжектируется в системный промпт.

### 4.2 Состав группы

- **Размер:** 3 агента по умолчанию (соответствует диапазону "малой группы").
- **Роли:** `explorer`, `critic`, `facilitator` — три комплементарные роли с разными OCEAN-преcetами.
- **Гендерная нейтральность:** все имена (Alex, Jordan, Sam) — gender-neutral.

### 4.3 OCEAN-пресеты по умолчанию

| Роль | O | C | E | A | N |
|---|---|---|---|---|---|
| explorer | 0.9 | 0.4 | 0.7 | 0.6 | 0.3 |
| critic   | 0.5 | 0.8 | 0.4 | 0.3 | 0.5 |
| facilitator | 0.6 | 0.7 | 0.8 | 0.9 | 0.2 |
| analyst | 0.6 | 0.9 | 0.3 | 0.4 | 0.5 |
| innovator | 0.95 | 0.3 | 0.8 | 0.5 | 0.4 |

### 4.4 Манипулируемые конфигурации (для гипотез)

| Гипотеза | Override |
|---|---|
| H1 (low/mid/high extraversion) | OCEAN: всем агентам экстраверсия = {0.2, 0.5, 0.8}; остальные = 0.5 |
| H3 (стадии Wheelan) | OCEAN рандомизированы с фиксированным seed |
| H5 (validation on/off) | DialogueManager.enable_validation = {True, False} |
| H6 (CoT on/off) | DialogueManager.enable_cot = {True, False} |
| H7 (turn-taking) | OCEAN рандомизированы; темы из единого набора 5 шт. |

### 4.5 Системный промпт агента

Собирается в `agents.py:Agent.system_prompt` через `build_system_prompt(nature, persona)` с инъекцией:
- базовой роли (explorer/critic/facilitator)
- персоны (возраст, профессия, бэкграунд)
- блока OCEAN с числовыми значениями + текстовым описанием
- инструкции "embody don't announce"

При активации CoT-режима через `DialogueManager.enable_cot=True` добавляется блок:

```
REASONING PROTOCOL (think step by step, internally, BEFORE you reply):
1. Identify the most important point or open question in the latest message.
2. Decide what is missing: a concrete next step, a constraint, an example,
   a risk, a counter-argument, or an assignment of responsibility.
3. Compose a reply that adds exactly that missing piece — do not just acknowledge.
Do not output the reasoning steps themselves — only the final reply.
```

### 4.6 Параметры LLM (фиксированы)

| Параметр | Значение |
|---|---|
| Модель | `openai/gpt-4o-mini` через OpenRouter |
| Temperature | 0.55 (после смены с 0.7 — для большей когерентности) |
| Max tokens | 220 |
| Структурированный вывод | OpenAI function calling (tool=`submit_agent_turn`); fallback на JSON mode |
| Random seed | базовый seed эксперимента + индекс диалога |

---

## 5. Архитектура программного комплекса

### 5.1 Ключевые модули

| Модуль | Назначение | Строк |
|---|---|---|
| `agents.py` | Agent + BigFiveProfile + 8 OCEAN-пресетов | 159 |
| `prompts.py` | Базовые системные промпты ролей | ~140 |
| `profiles.py` | Опросник из 14 вопросов для генерации persona | ~340 |
| `dialogue_manager.py` | Оркестратор диалога: выбор говорящего, сборка контекста, вызов LLM, валидация, mood-decay | ~1100 |
| `validators.py` | 8 валидаторов: грамматика, логика, токсичность, моральные схемы | ~390 |
| `analytics.py` | Базовые метрики (тон, энтропия, reciprocity, actionability_rate) | ~340 |
| `scientific_analytics.py` | 8 фреймворков: SNA, Wheelan, ISO 24617-2, Cowen-Keltner, LSM, Herring, Mathieu | ~1100 |
| `hypothesis_validator.py` | Эвристическая проверка гипотез (заменяется на p-value pipeline) | ~950 |
| **`statistical_tests.py`** *(новый)* | Унифицированный API мат-стат тестов | 380 |
| **`power_analysis.py`** *(новый)* | Расчёт минимального N для каждой гипотезы | 220 |
| **`experiment_runner.py`** *(новый)* | Пакетный запуск через YAML; mock+real диспатчер; интеграция стат-тестов | 470 |
| `web_app.py` | Flask UI + API | ~570 |
| `main.py` | CLI | ~120 |

### 5.2 Pipeline эксперимента

```
YAML config
    ↓
ExperimentConfig.from_yaml()
    ↓
ExperimentRunner.run()
    ↓
для каждой "arm" (треатмент-группы):
    ThreadPoolExecutor(parallelism=6):
        для каждого диалога с фикс. seed:
            real_dispatcher() →
                OpenAI client (OpenRouter)
                Agent ← OCEAN override из arm
                DialogueManager(enable_cot=..., enable_validation=...)
                for turn in range(n_turns):
                    dm.step() → LLM call → validation → history.append
                analytics.compute_metrics(history)
                return DialogueResult dict
        save arm_<name>.jsonl
    ↓
extract metric per dialogue → list[float] на arm
    ↓
statistical_tests.<test>(groups, **params) → TestResult
    ↓
save summary.json (config, metrics, p-value, effect, decision)
```

### 5.3 Юнит-тесты

- `test_statistical_tests.py` — **25/25 PASS**: для каждой стат-функции — кейс TRUE_H1 (синтетика с эффектом, должны reject H₀) и кейс NULL_H0 (без эффекта, должны fail_to_reject).
- `test_experiment_runner.py` — **10/10 PASS**: загрузка YAML, mock-диспатчер, детерминизм seeds, корректность сохранения arm_*.jsonl и summary.json.

**Итого: 35/35 unit-тестов проходят.**

---

## 6. Запущенные эксперименты

### 6.1 Smoke-test интеграции (валидация pipeline на реальной LLM)

| Параметр | Значение |
|---|---|
| Конфиг | `experiments/configs/smoke_test.yaml` |
| Гипотеза | H6 (CoT vs baseline) — но N=2 заведомо мало |
| Цель | проверить, что real_dispatcher работает end-to-end |
| Размер | 2 arm × 2 диалога × 4 хода = **16 LLM-вызовов** |
| Время | 19.5 секунд |
| Стоимость | ~$0.002 |
| Результат | ✅ pipeline работает: метрики собраны, тест применён, файлы сохранены |
| Найденные баги | 1) ключи метрик были вложены в `summary.all` (исправлено); 2) `ScientificAnalysisResult` не сериализовался в JSON (добавлен `_json_default` fallback) |

**Конкретные значения метрик в smoke-test:**
| seed | arm | reply_len | actionability | reciprocity |
|---|---|---|---|---|
| 99000 | cot_on | 32.0 | 0.000 | 0.500 |
| 99001 | cot_on | 32.5 | 0.250 | 0.500 |
| 99100 | baseline | 34.5 | 0.000 | 0.500 |
| 99101 | baseline | 33.2 | 0.000 | 0.500 |

Среднее actionability: cot_on = 0.125, baseline = 0.000. Направление эффекта совпадает с гипотезой H₁, но N=2 на arm не позволяет принять решение.

### 6.2 Эксперимент H6 — основной прогон (запущен)

| Параметр | Значение |
|---|---|
| Конфиг | `experiments/configs/h6_cot.yaml` |
| Гипотеза | H6: CoT улучшает actionability_rate |
| arm A (treatment) | `enable_cot=True`, 55 диалогов, seeds 6000–6054 |
| arm B (control) | `enable_cot=False`, 55 диалогов, seeds 6100–6154 |
| Длина диалога | 20 ходов |
| Группа | 3 агента: Explorer / Critic / Facilitator с базовыми OCEAN-пресетами |
| Тест | Welch t-test (односторонний `greater`), α = 0.05, d_threshold = 0.5 |
| Метрика | `actionability_rate` (доля реплик с action-маркерами) |
| Параллелизм | 6 потоков |
| Ожидаемая длительность | ~10 минут |
| Ожидаемая стоимость | $0.5–0.7 |

### Результаты прогона H6 (2026-04-26)

⚠ **Прогон ЗАВЕРШИЛСЯ ТЕХНИЧЕСКИ, НО НАУЧНО НЕ ВАЛИДЕН.**

#### Что произошло
Через несколько минут после старта OpenRouter API начал возвращать **HTTP 402 "Insufficient credits"** — баланс ключа исчерпался по ходу прогона. Pipeline продолжил работу, но `DialogueManager` подменял ответы fallback-строкой `"I'm here, but I'm having trouble responding right now..."` и возвращал её как реплику агента.

#### Распределение реальных vs fallback-реплик

| arm | real replies | fallback replies | % real |
|---|---|---|---|
| cot_on | 467 | 633 | 42.5% |
| baseline | **0** | 1100 | **0.0%** |
| **Итого** | 467 | 1733 | 21.2% |

| arm | диалоги ≥1 реальной реплики | всего диалогов |
|---|---|---|
| cot_on | 24 | 55 |
| baseline | **0** | 55 |

К моменту запуска arm `baseline` все кредиты OpenRouter были исчерпаны → ни одного реального LLM-ответа в контрольной группе.

#### Технический результат теста (артефактный)

| Параметр | Значение |
|---|---|
| Welch t-test statistic | 4.39 |
| p-value | 2.65 × 10⁻⁵ |
| Cohen's d | 0.84 |
| Decision | REJECT_H0 |
| Длительность | 449.5 секунд |

**Этот результат необходимо отвергнуть.** Тест обнаружил различие между "смесью настоящих и заглушечных реплик" в treatment vs "только заглушечные реплики" в control. Это **не CoT-эффект**, а артефакт fallback-механизма + обрыва кредитов на середине прогона.

#### План восстановления

1. **Пополнить баланс OpenRouter** — оценочная стоимость на полный прогон H6 при `gpt-4o-mini`: ~$1 (с запасом).
2. **Параллельно — отключить или сделать видимым fallback-механизм** в `DialogueManager`: при текущей реализации he silently produces nonsensical metric data при API-сбое. Варианты:
   - Жёстко падать (raise exception) при исчерпанных credits — диалог помечается как `"error": "..."` и не попадает в выборку (мягко обрабатывается уже в `experiment_runner.py:_run_arm`).
   - Сохранять флаг `had_fallback: bool` в результате и фильтровать в `_extract_metric()` (методологически чище).
3. **Re-run H6** после пополнения с проверкой `0 fallbacks` на короткой выборке (3–5 диалогов).
4. **Smoke-test перед каждым большим прогоном** должен включать проверку *qualifying credits*: вызывать `/api/v1/auth/key` или иной эндпоинт OpenRouter для получения текущего баланса до запуска серии.

#### Что мы всё-таки узнали (методологический урок)

- Pipeline работает — exit code 0, файлы сохранены, тест применён, summary.json валиден.
- Баг `DialogueManager`: тихая подмена реплики вместо явной ошибки — критическая проблема для научной воспроизводимости. Будет исправлена до следующего прогона.
- Power-анализ был нужен (powerful test обнаружил даже артефактную разницу с p=2.6e-5).

### 6.3 Будущие эксперименты

| Гипотеза | Объём | Оценка стоимости | Готовность |
|---|---|---|---|
| H1 | 480 диалогов × 20 ходов | ~$2 | конфиг готов: `h1_extraversion.yaml` |
| H5 | 60 диалогов × 30 ходов | ~$0.5 | конфиг готов: `h5_validation.yaml`; нужна разметка токсичности через LLM-судью |
| H3 | 30 диалогов × 60 ходов | ~$1 | требует классификатора стадий Wheelan на окнах |
| H7 | 70 LLM + 70 human (AMI) | ~$1 + парсинг AMI | требует подключения корпуса AMI Meeting |

---

## 7. Решающие правила для финального вывода

В терминологии научного руководителя:

1. **Если p < α И размер эффекта ≥ minimum** → данные **не противоречат H₁** → гипотеза принимается как теория для класса систем "LLM-многоагентные диалоги".
2. **Если p ≥ α** → нет оснований отвергнуть H₀; **H₁ не подтверждена** (но и не отвергнута окончательно — это важно для интерпретации).
3. **Если p < α, но эффект мал** → описывается отдельно: "статистически значимо, но практически малый эффект".

**Сценарии комплексного исхода всех 5 гипотез:**

| Подтверждено | Интерпретация для главного RQ |
|---|---|
| 5/5 | Сильное свидетельство в пользу способности LLM воспроизводить социально-психологические закономерности |
| 3–4/5 | Умеренное свидетельство; необходимо обсуждение специфики отвергнутых |
| 1–2/5 | Слабое свидетельство; вывод — система воспроизводит лишь отдельные аспекты |
| 0/5 | Гипотеза о валидности LLM-симуляции в текущей реализации **не подтверждается**; формулируются ограничения и перспективы |

---

## 8. Этическая и репродуцируемая база

- **Preregistration:** все гипотезы зафиксированы в `preregistration.md` ДО начала экспериментов; журнал изменений ведётся явно.
- **Random seeds:** для каждого диалога — фиксированный детерминированный seed (`seed_base + arm.seed_offset + dialogue_index`).
- **Открытые данные:** все артефакты эксперимента — YAML config, arm_*.jsonl, summary.json — сохраняются в `experiments/results/<exp_id>__<timestamp>/`.
- **Inter-rater reliability:** для H5 — обязательная вторичная разметка ≥ 100 реплик с κ Коэна ≥ 0.7.
- **Поправка на множественность:** Holm–Bonferroni применяется к семейству 5 гипотез.

---

## 9. Что осталось на следующих этапах

### Этап 3 (после H6)

1. Прогоны H1, H5, H3, H7 в указанной последовательности.
2. Замена эвристического `confidence` в `hypothesis_validator.py` на честный p-value через `statistical_tests.py`.
3. Реализация классификатора стадий Wheelan на основе ISO 24617-2 для H3.
4. Подключение корпуса AMI для H7 (парсер + расчёт Gini для human-baseline).
5. Реализация LLM-судьи для разметки токсичности (H5) с κ ≥ 0.7.

### Этап 4

1. Финальный отчёт: таблица "гипотеза | n | критерий | статистика | p_raw | p_holm | эффект | вывод".
2. Глава "Результаты и обсуждение" в дипломе.
3. Глава "Ограничения" с честным обсуждением методологических компромиссов.
4. Защита: презентация по структуре RQ → 5 гипотез → таблица результатов → выводы.

---

## 10. Сводка ссылок

1. Beauchamp, T. L., & Childress, J. F. (2019). *Principles of Biomedical Ethics* (8th ed.). Oxford University Press.
2. Bunt, H., Petukhova, V., Traum, D., & Alexandersson, J. (2020). Dialogue Act Annotation with the ISO 24617-2 Standard.
3. Carletta, J., et al. (2005). The AMI Meeting Corpus.
4. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
5. Costa, P. T., & McCrae, R. R. (1992). *NEO PI-R Professional Manual*. PAR.
6. Cowen, A. S., & Keltner, D. (2017). Self-report captures 27 distinct categories of emotion. *PNAS*, 114(38), E7900–E7909.
7. Dunbar, R. I. M., Duncan, N. D. C., & Nettle, D. (1995). Size and structure of freely forming conversational groups. *Human Nature*, 6, 67–78.
8. Goldberg, L. R. (1990). An alternative description of personality. *Journal of Personality and Social Psychology*, 59(6), 1216–1229.
9. Herring, S. C. (2004). Computer-mediated discourse analysis. *Designing for Virtual Communities in the Service of Learning*.
10. Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33, 159–174.
11. Mehl, M. R., Gosling, S. D., & Pennebaker, J. W. (2006). Personality in its natural habitat. *Journal of Personality and Social Psychology*, 90(5), 862–877.
12. Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*.
13. Wheelan, S. A. (2009). Group Size, Group Development, and Group Productivity. *Small Group Research*, 40(2), 247–262.

---

*Документ автоматически сгенерирован на основании состояния репозитория от 2026-04-26.*
*Раздел 6.2 будет дополнен фактическими результатами H6 после завершения прогона.*
