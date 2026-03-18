# 🛡️ Система Валидации — Краткая Справка

## Что добавлено в проект

В вашу дипломную работу добавлена **многоуровневая система валидации диалогов агентов** с проверкой не только грамматики, но и **логической согласованности** и **моральных/этических норм**.

---

## Три уровня валидации

### 1️⃣ Грамматическая валидация

Проверяет качество текста:

- ✅ Пустое сообщение → **БЛОКИРУЕТ** (критическая ошибка)
- ✅ Слишком длинное/короткое → предупреждение
- ✅ Нет пунктуации/заглавных букв → информация
- ✅ CAPS LOCK (крик) → предупреждение
- ✅ Кириллица (не английский) → **БЛОКИРУЕТ**

**Пример:**
```python
"Привет" → 🚨 БЛОКИРУЕТСЯ (кириллица)
"Hello!" → ✅ OK
```

---

### 2️⃣ Логическая валидация

Проверяет согласованность диалога:

- ✅ **Self-targeting** (агент обращается к себе) → **БЛОКИРУЕТ**
- ✅ Неправильный адресат → **БЛОКИРУЕТ**
- ✅ Отсутствие адресата (null) → ошибка
- ✅ Повтор одного адресата 3+ раза подряд → предупреждение
- ✅ Несоответствие тона и эмоции:
  - `tone: positive` + `emotion: angry` → предупреждение

**Пример:**
```python
speaker: "Alice", target: "Alice" → 🚨 БЛОКИРУЕТСЯ
tone: "positive", emotion: "sad" → ⚠️ Предупреждение
```

---

### 3️⃣ Моральная/этическая валидация

**Самое важное для диплома!** Проверяет соблюдение моральных норм:

#### A) Токсичность (`ToxicityValidator`)
Блокирует оскорбления:
- `stupid`, `idiot`, `dumb`, `moron`
- `hate you`, `shut up`
- `useless`, `pathetic`

```python
"You're an idiot" → 🚨 БЛОКИРУЕТСЯ
"I disagree" → ✅ OK
```

#### B) Уважение (`RespectValidator`)
Блокирует неуважительные фразы:
- `you always fail`, `you never listen`
- `I told you so`
- `whatever`, `I don't care`

```python
"Whatever, I don't care" → ❌ Ошибка
"I understand your point" → ✅ OK
```

#### C) Профессиональные границы (`ProfessionalBoundariesValidator`)
В контексте университета/стартапа блокирует:
- Романтические темы (`love`, `dating`)
- Личные вопросы (`personal life`)

```python
"Let's talk about your personal life" → ⚠️ Предупреждение (в контексте университета)
"Let's discuss the project" → ✅ OK
```

#### D) **Моральные схемы** (`MoralSchemaValidator`) ⭐

**Главное для диплома!** Проверяет принципы:

| Принцип | Что запрещено | Пример нарушения |
|---------|---------------|------------------|
| **Справедливость** (Fairness) | Исключение участников | "Let's exclude Bob" |
| **Автономия** (Autonomy) | Принуждение | "You must do this" |
| **Инклюзивность** (Inclusivity) | Игнорирование | "Let's ignore Alice" |
| **Ненанесение вреда** (Non-maleficence) | Обвинения всех проблем на одного | "Blame X for everything" |

```python
"Let's exclude Bob from this" → ❌ Нарушение справедливости
"Let's hear everyone's input" → ✅ Поощряет инклюзивность
```

---

## Как работает

### Автоматическая интеграция

Валидация **уже встроена** в `DialogueManager`:

```python
dm = DialogueManager(
    client=client,
    agents=agents,
    env_context=env_context,
    enable_validation=True  # Включена по умолчанию
)

# При каждом dm.step() автоматически проверяется сообщение
record, edge = dm.step()

# Результаты валидации:
print(record["validation_issues"])  # Количество проблем
```

**Логи валидации** выводятся в `stderr`:

```
[Validation Turn 5]
Validation results:
🚨 [Toxicity] Potentially toxic language detected: '\b(stupid|idiot)\b'
   → Suggestion: Rephrase respectfully
❌ [MoralSchema] Potential violation of moral principle: exclusion
   → Suggestion: Promote fairness, inclusivity, and collaboration

Summary: 1 critical, 1 errors, 0 warnings, 0 info
```

### Ручное использование

```python
from validators import validate_agent_turn

turn_data = {
    "speaker": "Alice",
    "target": "Bob",
    "reply": "Great idea!",
    "tone": "positive",
    "emotion": "joyful",
}

is_valid, results, formatted = validate_agent_turn(
    turn_data,
    history=[...],
    allowed_targets=["Alice", "Bob", "Charlie"],
    env_context="University: AI project"
)

if not is_valid:
    print("❌ Сообщение заблокировано:")
    print(formatted)
```

---

## Тестирование

Запустите тесты:

```bash
python -m agent_dialogue_sim.test_validators
```

Посмотрите примеры:

```bash
python validation_examples.py
```

---

## Для дипломной работы

### Как описать в дипломе

**"Система валидации на основе моральных схем"**

> В работе реализована **многоуровневая система валидации диалогов агентов**, включающая проверку:
> 
> 1. **Грамматической корректности** — качество текста, язык
> 2. **Логической согласованности** — правильность адресации, соответствие тона и эмоций
> 3. **Моральных/этических норм** — на основе **моральных схем** (moral schemas)
>
> Моральные схемы основаны на принципах:
> - **Справедливость** (fairness) — предотвращение исключения участников
> - **Уважение автономии** (autonomy) — запрет принуждения
> - **Инклюзивность** (inclusivity) — поощрение участия всех агентов  
> - **Ненанесение вреда** (non-maleficence) — детекция токсичности
>
> Система работает на основе **эвристических правил** (паттернов) и может быть расширена ML-моделями (Perspective API, Toxic-BERT).

### Формализм

**Моральная схема** определяется как кортеж:

\[
M = (P, R, V)
\]

где:
- \( P = \{p_1, p_2, ..., p_n\} \) — набор **принципов** (fairness, respect, inclusivity, autonomy)
- \( R = \{r_1, r_2, ..., r_m\} \) — набор **правил** (регулярных выражений), проверяющих соблюдение принципов
- \( V: \text{violation} \to L \) — функция **оценки серьезности**, где \( L = \{\text{info, warning, error, critical}\} \)

**Пример правила:**

\[
r_{\text{exclusion}}: \text{if } \text{reply} \text{ matches } \texttt{/(exclude|ignore|leave out)/} \Rightarrow \text{violation}(p_{\text{inclusivity}}, \text{ERROR})
\]

---

## Структура кода

### Главные файлы

- **`validators.py`** — все валидаторы (8 классов + pipeline)
- **`dialogue_manager.py`** — интеграция валидации в диалог
- **`test_validators.py`** — тесты и демо
- **`validation_examples.py`** — быстрые примеры
- **`VALIDATION_GUIDE.md`** — полная документация (на английском)

### Архитектура

```
ValidationPipeline
├── GrammaticalValidator
│   ├── проверка длины
│   ├── пунктуация
│   └── CAPS
├── LanguageValidator (English only)
├── LogicalConsistencyValidator
│   ├── self-targeting
│   ├── invalid target
│   └── repetitive targeting
├── ToneEmotionValidator
├── ToxicityValidator
├── RespectValidator
├── ProfessionalBoundariesValidator
└── MoralSchemaValidator ⭐
    ├── Fairness (исключение)
    ├── Autonomy (принуждение)
    ├── Inclusivity (игнорирование)
    └── Non-maleficence (обвинения)
```

---

## Расширение

### Добавить свой валидатор

```python
from validators import BaseValidator, ValidationResult, ValidationLevel

class MyValidator(BaseValidator):
    def __init__(self):
        super().__init__("MyValidator")
    
    def validate(self, turn_data, context):
        results = []
        reply = turn_data["reply"]
        
        if "запрещенное_слово" in reply:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                validator_name=self.name,
                message="Найдено запрещенное слово",
                field="reply",
                suggestion="Удалите запрещенное слово"
            ))
        
        return results
```

Добавьте в pipeline:

```python
from validators import ValidationPipeline
pipeline = ValidationPipeline(validators=[
    MyValidator(),
    # ... другие
])
```

### Интеграция ML-модели

Замените `ToxicityValidator` на ML:

```python
from transformers import pipeline as hf_pipeline

class MLToxicityValidator(BaseValidator):
    def __init__(self):
        super().__init__("MLToxicity")
        self.model = hf_pipeline("text-classification", 
                                  model="unitary/toxic-bert")
    
    def validate(self, turn_data, context):
        reply = turn_data["reply"]
        result = self.model(reply)[0]
        
        if result["label"] == "toxic" and result["score"] > 0.7:
            return [ValidationResult(
                level=ValidationLevel.CRITICAL,
                validator_name=self.name,
                message=f"Токсичность: {result['score']:.2%}",
                field="reply"
            )]
        return []
```

---

## Метрики

Можно собрать статистику валидации:

```python
from collections import Counter

# Подсчёт ошибок
all_issues = []
for turn in history:
    all_issues.extend(turn.get("validation_results", []))

by_level = Counter(r.level.value for r in all_issues)
by_validator = Counter(r.validator_name for r in all_issues)

print(f"По уровням: {dict(by_level)}")
print(f"По валидаторам: {dict(by_validator)}")
```

**Вывод:**
```
По уровням: {'info': 8, 'warning': 10, 'error': 2, 'critical': 0}
По валидаторам: {'Grammatical': 12, 'Toxicity': 2, 'MoralSchema': 1}
```

---

## Преимущества для диплома

✅ **Теоретическая база** — моральные схемы, формализм, принципы этики  
✅ **Практическая реализация** — рабочий код с тестами  
✅ **Расширяемость** — легко добавить ML-модели (Perspective API, BERT)  
✅ **Метрики** — можно анализировать качество диалогов количественно  
✅ **Научная новизна** — применение моральных схем к multi-agent системам  

---

## Быстрый старт

1. **Запустите тесты:**
   ```bash
   python -m agent_dialogue_sim.test_validators
   ```

2. **Посмотрите примеры:**
   ```bash
   python validation_examples.py
   ```

3. **Запустите симуляцию с валидацией:**
   ```bash
   python -m agent_dialogue_sim.main
   ```
   
   Валидация работает автоматически, результаты в stderr.

4. **Прочитайте полную документацию:**
   ```bash
   cat VALIDATION_GUIDE.md
   ```

---

## FAQ

**В: Как отключить валидацию?**

О: При создании `DialogueManager`:
```python
dm = DialogueManager(..., enable_validation=False)
```

**В: Как добавить свои запретные слова?**

О: Отредактируйте `ToxicityValidator` в `validators.py`:
```python
self.toxic_patterns = [
    r'\b(моё_слово|другое_слово)\b',
    # ...
]
```

**В: Можно ли использовать внешний API для проверки токсичности?**

О: Да! Смотрите раздел "Расширение" в `VALIDATION_GUIDE.md`.

**В: Где логи валидации?**

О: В `stderr` (терминал). Также в каждом `record["validation_issues"]`.

---

## Итог

Вы получили **полноценную систему валидации** с:

- 8 валидаторов (грамматика, логика, мораль)
- Моральными схемами (fairness, autonomy, inclusivity, non-maleficence)
- Тестами и примерами
- Интеграцией в DialogueManager
- Расширяемой архитектурой (легко добавить ML)
- Документацией на русском и английском

**Это солидная теоретическая и практическая база для дипломной работы!** 🎓

