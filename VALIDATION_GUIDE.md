# 🛡️ Validation System Guide

## Обзор

Система валидации в `agent_dialogue_sim` обеспечивает многоуровневую проверку сообщений агентов на **грамматическую корректность**, **логическую согласованность** и **моральные/этические стандарты**.

## Архитектура

### Три уровня валидации

```
┌─────────────────────────────────────────────┐
│         VALIDATION PIPELINE                  │
├─────────────────────────────────────────────┤
│                                              │
│  1️⃣  ГРАММАТИЧЕСКАЯ ВАЛИДАЦИЯ               │
│     • GrammaticalValidator                   │
│     • LanguageValidator                      │
│                                              │
│  2️⃣  ЛОГИЧЕСКАЯ ВАЛИДАЦИЯ                   │
│     • LogicalConsistencyValidator            │
│     • ToneEmotionValidator                   │
│                                              │
│  3️⃣  МОРАЛЬНАЯ/ЭТИЧЕСКАЯ ВАЛИДАЦИЯ          │
│     • ToxicityValidator                      │
│     • RespectValidator                       │
│     • ProfessionalBoundariesValidator        │
│     • MoralSchemaValidator                   │
│                                              │
└─────────────────────────────────────────────┘
```

## Уровни серьезности (Severity Levels)

| Уровень | Значок | Описание | Блокирует сообщение? |
|---------|--------|----------|----------------------|
| **INFO** | ℹ️ | Информационное замечание | ❌ Нет |
| **WARNING** | ⚠️ | Предупреждение (нежелательно) | ❌ Нет |
| **ERROR** | ❌ | Ошибка (серьезное нарушение) | ❌ Нет |
| **CRITICAL** | 🚨 | Критическая ошибка | ✅ **Да** |

**Только CRITICAL блокирует выполнение** — остальные уровни логируются, но не останавливают диалог.

---

## 1️⃣ Грамматическая валидация

### `GrammaticalValidator`

Проверяет базовое качество текста:

| Проверка | Условие | Уровень | Пример |
|----------|---------|---------|--------|
| **Пустое сообщение** | `reply == ""` | CRITICAL | `""` |
| **Слишком длинное** | `> 50 слов` | WARNING | 60-словное сообщение |
| **Слишком короткое** | `< 2 слов` | WARNING | `"Ok"` |
| **Нет пунктуации** | Не заканчивается на `.!?` | INFO | `"Great idea"` |
| **Без заглавной** | Первая буква не uppercase | INFO | `"hello there"` |
| **Избыток пунктуации** | `!!!`, `???` (3+) | WARNING | `"Really???"` |
| **Все заглавные (CAPS)** | Весь текст uppercase | WARNING | `"THIS IS SHOUTING"` |

### `LanguageValidator`

Обеспечивает использование **только английского языка**:

```python
# ❌ CRITICAL: Кириллица запрещена
"Привет, как дела?" → CRITICAL error

# ✅ OK
"Hello, how are you?" → PASS
```

---

## 2️⃣ Логическая валидация

### `LogicalConsistencyValidator`

Проверяет логику диалога:

| Проверка | Описание | Уровень |
|----------|----------|---------|
| **Self-targeting** | Агент обращается к самому себе (`speaker == target`) | CRITICAL |
| **Invalid target** | Target не из списка `allowed_targets` | CRITICAL |
| **Missing target** | Target = null/None (broadcasts запрещены) | ERROR |
| **Repetitive targeting** | Один и тот же target 3+ раза подряд | WARNING |
| **Context relevance** | Нет пересечения ключевых слов с последними 3 сообщениями | INFO |

**Пример:**

```python
turn_data = {
    "speaker": "Alice",
    "target": "Alice",  # ❌ CRITICAL: self-targeting!
    "reply": "I think we should proceed."
}
```

### `ToneEmotionValidator`

Проверяет согласованность тона и эмоции:

| Tone | Compatible Emotions | Incompatible Emotions |
|------|---------------------|------------------------|
| `positive` | joyful, confident, excited, happy, calm, grateful | angry, sad, frustrated, anxious |
| `negative` | angry, sad, frustrated, anxious, skeptical, disappointed | joyful, happy, excited |
| `neutral` | neutral, curious, thoughtful, focused | — |

**Пример:**

```python
# ⚠️ WARNING: Misalignment
{
    "tone": "positive",
    "emotion": "angry"  # Несовместимы!
}
```

---

## 3️⃣ Моральная/этическая валидация

### `ToxicityValidator`

Детектирует токсичный язык (оскорбления, ругательства):

**Паттерны:**
- `stupid`, `idiot`, `dumb`, `moron`
- `hate you`, `despise you`
- `shut up`, `fuck`, `shit`
- `useless`, `worthless`, `pathetic`

```python
# 🚨 CRITICAL
"You're such an idiot." → Blocked

# ✅ OK
"I respectfully disagree." → Pass
```

### `RespectValidator`

Проверяет уважительность:

**Паттерны неуважения:**
- `you always fail`, `you never listen`
- `I told you so`
- `whatever` (пренебрежение)
- `I don't care`

```python
# ❌ ERROR
"Whatever, I don't care." → Disrespectful

# ✅ OK
"I understand your point." → Pass
```

### `ProfessionalBoundariesValidator`

Проверяет соблюдение профессиональных границ (в контексте университета/стартапа/исследований):

**Неприемлемо:**
- Романтические темы: `love`, `dating`, `kiss`
- Личные темы: `personal life`, `private matter`

```python
# ⚠️ WARNING (в контексте "University: ...")
"Let's talk about your personal life." → Crosses boundaries

# ✅ OK
"Let's discuss the project timeline." → Pass
```

### `MoralSchemaValidator`

Проверяет **моральные принципы** (основа вашей работы!):

| Принцип | Паттерн | Уровень |
|---------|---------|---------|
| **Exclusion** (исключение) | `exclude`, `ignore`, `leave out` | ERROR |
| **Scapegoating** (козёл отпущения) | `blame X for everything` | ERROR |
| **Coercion** (принуждение) | `force you`, `must do`, `make you` | ERROR |
| **Gatekeeping** (монополия знаний) | `only I know`, `only I can` | ERROR |

**Примеры:**

```python
# ❌ ERROR: Exclusion
"Let's exclude Bob from the discussion." → Violates fairness

# ❌ ERROR: Coercion
"You must do this my way." → Violates autonomy

# ✅ OK
"Let's hear everyone's input." → Promotes inclusivity
```

---

## Использование в коде

### Вариант 1: Через `validate_agent_turn` (простой)

```python
from validators import validate_agent_turn

turn_data = {
    "speaker": "Alice",
    "target": "Bob",
    "reply": "Great idea! Let's implement it.",
    "tone": "positive",
    "emotion": "confident",
}

history = [...]  # Previous turns
allowed_targets = ["Alice", "Bob", "Charlie"]
env_context = "University: AI coursework project"

is_valid, results, formatted_msg = validate_agent_turn(
    turn_data, history, allowed_targets, env_context
)

if not is_valid:
    print(f"❌ Validation failed:\n{formatted_msg}")
else:
    print(f"✅ Valid ({len(results)} non-blocking issues)")
```

### Вариант 2: Через `ValidationPipeline` (кастомизация)

```python
from validators import ValidationPipeline, GrammaticalValidator, ToxicityValidator

# Используем только грамматику и токсичность
pipeline = ValidationPipeline(validators=[
    GrammaticalValidator(),
    ToxicityValidator(),
])

turn_data = {...}
context = {"history": [...], "allowed_targets": [...], "env_context": "..."}

is_valid, results = pipeline.validate_turn(turn_data, context)

for r in results:
    print(f"[{r.level.value.upper()}] {r.message}")
    if r.suggestion:
        print(f"  💡 {r.suggestion}")
```

### Вариант 3: Автоматически в `DialogueManager`

Валидация **автоматически включена** при использовании `DialogueManager`:

```python
from dialogue_manager import DialogueManager

dm = DialogueManager(
    client=client,
    agents=agents,
    env_context=env_context,
    enable_validation=True  # Включена по умолчанию
)

# Валидация выполняется автоматически при dm.step()
record, edge = dm.step()

# Результаты выводятся в stderr и сохраняются в record
print(record["validation_issues"])  # Количество найденных проблем
```

**Отключить валидацию:**

```python
dm = DialogueManager(
    ...,
    enable_validation=False
)
```

---

## Тестирование

Запустите тесты:

```bash
python -m agent_dialogue_sim.test_validators
```

**Вывод:**

```
🧪 VALIDATION SYSTEM TEST SUITE

============================================================
TEST 1: Grammatical Validation
============================================================

✅ Good message: 'That's a great idea! Let's discuss it further.'
   Issues: 0

⚠️  Long message (60 words)
   Issues: ['Reply is too long (60 words)']

...

============================================================
✅ All tests completed!
============================================================
```

---

## Расширение системы

### Добавление нового валидатора

```python
from validators import BaseValidator, ValidationResult, ValidationLevel

class CustomValidator(BaseValidator):
    def __init__(self):
        super().__init__("CustomValidator")
    
    def validate(self, turn_data, context):
        results = []
        reply = turn_data.get("reply", "")
        
        # Ваша логика проверки
        if "forbidden_word" in reply.lower():
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                validator_name=self.name,
                message="Forbidden word detected",
                field="reply",
                suggestion="Remove the forbidden word"
            ))
        
        return results

# Добавляем в pipeline
pipeline = ValidationPipeline(validators=[
    CustomValidator(),
    # ... другие валидаторы
])
```

### Интеграция ML-модели для токсичности

Замените `ToxicityValidator` на ML-based:

```python
from transformers import pipeline as hf_pipeline

class MLToxicityValidator(BaseValidator):
    def __init__(self):
        super().__init__("MLToxicity")
        self.model = hf_pipeline("text-classification", 
                                  model="unitary/toxic-bert")
    
    def validate(self, turn_data, context):
        reply = turn_data.get("reply", "")
        result = self.model(reply)[0]
        
        if result["label"] == "toxic" and result["score"] > 0.7:
            return [ValidationResult(
                level=ValidationLevel.CRITICAL,
                validator_name=self.name,
                message=f"Toxic content detected (confidence: {result['score']:.2f})",
                field="reply"
            )]
        return []
```

---

## Метрики валидации

После прогона диалога можно собрать статистику:

```python
from collections import Counter

# Подсчёт ошибок по типам
all_issues = []
for turn in history:
    all_issues.extend(turn.get("validation_results", []))

by_validator = Counter(r.validator_name for r in all_issues)
by_level = Counter(r.level.value for r in all_issues)

print(f"Issues by validator: {dict(by_validator)}")
print(f"Issues by level: {dict(by_level)}")
```

**Пример вывода:**

```
Issues by validator: {
    'Grammatical': 12,
    'LogicalConsistency': 5,
    'Toxicity': 2,
    'MoralSchema': 1
}
Issues by level: {
    'info': 8,
    'warning': 10,
    'error': 2,
    'critical': 0
}
```

---

## Связь с дипломной работой

### Как описать в дипломе:

**"Моральные схемы в системе валидации":**

> В работе реализована система многоуровневой валидации диалогов агентов, включающая **моральные схемы** (moral schemas) на основе принципов:
> 
> 1. **Справедливость** (Fairness) — предотвращение исключения участников
> 2. **Уважение автономии** (Autonomy) — запрет принуждения
> 3. **Инклюзивность** (Inclusivity) — поощрение участия всех агентов
> 4. **Ненанесение вреда** (Non-maleficence) — детекция токсичности и оскорблений
> 
> Валидаторы работают на основе **эвристик** (keyword patterns) и могут быть расширены ML-моделями (например, Perspective API, Toxic-BERT).

### Формализм:

**Моральная схема** определяется как кортеж:

\[
M = (P, R, V)
\]

где:
- \( P \) — набор **принципов** (fairness, respect, inclusivity, ...)
- \( R \) — набор **правил** (паттернов), проверяющих соблюдение принципов
- \( V \) — функция **оценки серьезности** нарушения: \( V: \text{violation} \to \{\text{info, warning, error, critical}\} \)

**Пример правила:**

\[
r_{\text{exclusion}}: \text{if } \text{reply} \text{ matches } \texttt{/(exclude|ignore) \w+/} \Rightarrow \text{violation}(P_{\text{inclusivity}}, \text{ERROR})
\]

---

## FAQ

**Q: Как отключить конкретный валидатор?**

A: Создайте кастомный pipeline без него:

```python
from validators import ValidationPipeline, GrammaticalValidator, LogicalConsistencyValidator

pipeline = ValidationPipeline(validators=[
    GrammaticalValidator(),
    LogicalConsistencyValidator(),
    # ToxicityValidator НЕ добавлен
])

dm.validation_pipeline = pipeline
```

**Q: Можно ли блокировать не только CRITICAL, но и ERROR?**

A: Да, измените `ValidationResult.is_blocking`:

```python
@property
def is_blocking(self) -> bool:
    return self.level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR]
```

**Q: Как добавить валидацию через внешний API (например, Perspective API)?**

A: Создайте асинхронный валидатор:

```python
import requests

class PerspectiveValidator(BaseValidator):
    def __init__(self, api_key):
        super().__init__("Perspective")
        self.api_key = api_key
    
    def validate(self, turn_data, context):
        reply = turn_data["reply"]
        response = requests.post(
            "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze",
            params={"key": self.api_key},
            json={
                "comment": {"text": reply},
                "requestedAttributes": {"TOXICITY": {}}
            }
        )
        score = response.json()["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        
        if score > 0.7:
            return [ValidationResult(
                level=ValidationLevel.CRITICAL,
                validator_name=self.name,
                message=f"High toxicity score: {score:.2f}",
                field="reply"
            )]
        return []
```

---

## Заключение

Система валидации делает ваш проект:

✅ **Более надёжным** — предотвращает некорректные сообщения  
✅ **Этичным** — соблюдает моральные стандарты  
✅ **Научно обоснованным** — формальные правила + возможность ML-интеграции  
✅ **Расширяемым** — легко добавлять новые валидаторы  

**Для диплома:** это отличная **теоретическая база** (моральные схемы, формализм) + **практическая реализация** (код, тесты, метрики).

