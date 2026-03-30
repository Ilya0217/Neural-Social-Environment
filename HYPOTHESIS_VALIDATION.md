# Проверка гипотез на основе теорий о социальных отношениях

## Описание

Модуль `hypothesis_validator.py` реализует систему проверки гипотез, выдвинутых на основе научных теорий о социальных отношениях. Гипотезы проверяются путём сравнения метрик диалога в разные моменты времени и оценки статистической значимости изменений.

## Основные возможности

### 1. Валидация гипотез по категориям

Модуль проверяет гипотезы из следующих категорий:

- **Group Development** (Развитие группы) - проверка стадий развития группы по модели Wheelan (2009)
- **Network Structure** (Структура сети) - проверка централизации и плотности сети
- **Social Capital** (Социальный капитал) - проверка сплочённости и сетевого статуса участников
- **Emotional Climate** (Эмоциональный климат) - проверка распределения эмоций
- **Dialogue Structure** (Структура диалога) - проверка диалоговых актов
- **Turn-Taking** (Чередование реплик) - проверка неравенства в распределении реплик

### 2. Методы проверки

Для каждой гипотезы модуль:

1. **Извлекает релевантные метрики** из научного анализа
2. **Сравнивает с предыдущими метриками** для выявления изменений
3. **Оценивает уверенность** (confidence) в подтверждении гипотезы
4. **Формирует доказательства** на основе фактических данных
5. **Определяет статус**: confirmed, partial, rejected, insufficient_data

### 3. Статусы проверки

- **✅ confirmed** - гипотеза подтверждена (confidence > 0.6)
- **⚠️ partial** - частичное подтверждение (confidence 0.3-0.6)
- **❌ rejected** - гипотеза отклонена (confidence < 0.3)
- **❓ insufficient_data** - недостаточно данных для проверки

## Использование

### Базовое использование

```python
from hypothesis_validator import HypothesisValidator
from analytics import compute_metrics, generate_scientific_hypotheses

# Создать валидатор
validator = HypothesisValidator()

# Получить метрики
metrics = compute_metrics(history, agents, window_size=10)
scientific = metrics.get("scientific")

# Сгенерировать гипотезы
hypotheses = generate_scientific_hypotheses(scientific, agents)

# Проверить гипотезы
validations = validator.validate_all_hypotheses(
    hypotheses,
    metrics,
    previous_metrics=None  # опционально
)

# Получить отчёт
report = validator.render_validation_report(validations, turn=10)
print(report)
```

### Интеграция в analytics.py

Модуль автоматически интегрирован в `analytics.py`:

```python
from analytics import validate_hypotheses, get_hypothesis_validation_report

# Проверить гипотезы
validations = validate_hypotheses(hypotheses, current_metrics, previous_metrics)

# Получить отчёт
report = get_hypothesis_validation_report(hypotheses, metrics, previous_metrics, turn)
```

### В веб-интерфейсе

Проверка гипотез автоматически выполняется каждые 5 ходов и сохраняется в:
- `outputs/validation_turn_XXXX.md` - Markdown отчёт
- Доступен через API: `validation_report` в ответе `/api/step`

## Примеры проверки

### Пример 1: Развитие группы

**Гипотеза:** "Группа находится на стадии формирования (forming)"

**Проверка:**
- Извлекается стадия из `scientific.group_stage`
- Проверяется соответствие метрик стадии (низкая плотность сети для forming)
- Сравнивается с предыдущей стадией для выявления переходов

**Результат:**
```
✅ Подтверждено (уверенность: 75%)
Доказательства:
- Текущая стадия: forming (уверенность: 70%)
- Низкая плотность сети (35%) соответствует стадии формирования
```

### Пример 2: Структура сети

**Гипотеза:** "Агент 'Explorer' занимает центральную позицию в сети"

**Проверка:**
- Вычисляется центральность всех агентов
- Находится агент с максимальной центральностью
- Проверяется, превышает ли центральность порог 0.7

**Результат:**
```
✅ Подтверждено (уверенность: 80%)
Доказательства:
- Максимальная центральность: 78%
- Высокая централизация подтверждает гипотезу
```

### Пример 3: Эмоциональный климат

**Гипотеза:** "Преобладание негативных эмоций"

**Проверка:**
- Извлекается распределение эмоций
- Вычисляется доля негативных эмоций (anger, fear, sadness)
- Сравнивается с предыдущими метриками для выявления трендов

**Результат:**
```
✅ Подтверждено (уверенность: 80%)
Доказательства:
- Доминирующая эмоция: concern
- Доля негативных эмоций: 45%
- Преобладание негативных эмоций подтверждает гипотезу
- Изменение негативных эмоций: +12% (ухудшение)
```

## Структура данных

### HypothesisValidation

```python
@dataclass
class HypothesisValidation:
    hypothesis_id: str              # Уникальный ID гипотезы
    hypothesis_text: str           # Текст гипотезы
    framework: str                  # Научный фреймворк
    status: str                     # Статус проверки
    confidence: float               # Уверенность (0.0-1.0)
    evidence: List[str]             # Список доказательств
    metrics_before: Optional[Dict]  # Метрики до
    metrics_after: Optional[Dict]   # Метрики после
    change_magnitude: Optional[float] # Величина изменения
```

## Научные фреймворки

Модуль поддерживает проверку гипотез на основе:

1. **Integrated Model of Group Development** (Wheelan, 2009)
2. **Social Network Analysis** (Borgatti et al., 2009)
3. **Social Capital Theory** (Burt, 2004; Gittell, 2002)
4. **Dimensional Emotion Model** (Russell & Barrett, 1999)
5. **ISO 24617-2 Dialogue Act Taxonomy** (Bunt et al., 2017)
6. **Computer-Mediated Discourse Analysis** (Herring, 2004)

## Отчёты

Отчёты генерируются в формате Markdown и включают:

1. **Сводку** - общую статистику по проверке
2. **Детали проверки** - для каждой гипотезы:
   - Статус и уверенность
   - Список доказательств
   - Сравнение метрик (если доступно)

## Интеграция

Модуль интегрирован в:
- `analytics.py` - функции `validate_hypotheses()` и `get_hypothesis_validation_report()`
- `web_app.py` - автоматическая проверка каждые 5 ходов
- Сохранение отчётов в `outputs/validation_turn_XXXX.md`

## Расширение

Для добавления новых типов проверки:

1. Добавьте метод `_validate_<category>()` в класс `HypothesisValidator`
2. Добавьте обработку категории в метод `validate_hypothesis()`
3. Используйте метрики из `scientific` для проверки

Пример:
```python
def _validate_my_category(self, hyp_id, hypothesis, current, previous):
    # Ваша логика проверки
    evidence = []
    confidence = 0.5
    # ...
    return HypothesisValidation(...)
```

