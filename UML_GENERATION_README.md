# Генерация UML диаграмм

Два скрипта для создания UML диаграмм проекта:

## 🚀 Быстрый старт (рекомендуется)

### Вариант 1: Упрощённый (без зависимостей)

```bash
python generate_uml_simple.py
```

Создаст файл `uml_diagrams/agent_dialogue_sim.puml`, который можно визуализировать:

1. **Онлайн** (самый простой способ):
   - Откройте http://www.plantuml.com/plantuml/uml/
   - Скопируйте содержимое файла `uml_diagrams/agent_dialogue_sim.puml`
   - Вставьте в редактор на сайте
   - Нажмите "Submit" для генерации изображения

2. **VS Code**:
   - Установите расширение "PlantUML" (jebbs.plantuml)
   - Откройте файл `.puml`
   - Нажмите `Alt+D` для предпросмотра

3. **CLI** (требует установки PlantUML):
   ```bash
   # Установка PlantUML (требует Java)
   # Windows: choco install plantuml
   # macOS: brew install plantuml
   # Linux: sudo apt-get install plantuml
   
   plantuml uml_diagrams/agent_dialogue_sim.puml
   ```

### Вариант 2: Автоматический анализ кода

```bash
# Установка зависимостей
pip install pylint

# Запуск
python generate_uml.py
```

Выберите формат:
- `dot` - Graphviz DOT (всегда доступен)
- `png` - PNG изображение (требует Graphviz)
- `svg` - SVG векторное (требует Graphviz)
- `pdf` - PDF документ (требует Graphviz)

**Требования для PNG/SVG/PDF:**
1. Установите Graphviz: https://graphviz.org/download/
2. Установите Python пакет: `pip install graphviz`

## 📊 Что включено в диаграмму

### Основные компоненты:
- **Agent** - представление виртуального участника
- **DialogueManager** - центральный оркестратор диалога
- **AgentTurn** - структурированный ответ агента

### Система валидации:
- **ValidationPipeline** - оркестратор валидации
- **BaseValidator** - базовый класс (Strategy Pattern)
- 8 типов валидаторов (Grammatical, Language, Logical, Tone, Toxicity, Respect, Professional, Moral)

### Система промптов:
- **PromptBuilder** - построение системных промптов
- Chain-of-Thought техники
- Context-aware адаптация

### Аналитика:
- **MetricsCalculator** - расчёт метрик
- **ScientificAnalyzer** - научный анализ
- **ScientificAnalysisResult** - результаты анализа

### Производительность:
- **CacheManager** / **SemanticCacheManager** - кэширование
- **PerformanceMonitor** - мониторинг API вызовов

### Экспорт и логирование:
- **ExportManager** - экспорт в CSV, Excel, JSON, Markdown
- **IOLogger** - логирование диалогов

### Интерфейсы:
- **FlaskApp** / **AppState** - веб-интерфейс
- **GraphVisualizer** - визуализация графов

## 🔧 Решение проблем

### Ошибка "pylint not found"
```bash
pip install pylint
```

### Ошибка "graphviz not found"
1. Установите системный Graphviz: https://graphviz.org/download/
2. Установите Python пакет: `pip install graphviz`

### PlantUML не рендерится
- Используйте онлайн версию: http://www.plantuml.com/plantuml/uml/
- Или установите PlantUML CLI (требует Java)

## 📝 Форматы вывода

- **`.puml`** - PlantUML исходный код (текстовый, легко редактировать)
- **`.dot`** - Graphviz DOT формат
- **`.png`** - растровое изображение
- **`.svg`** - векторное изображение (лучше для документов)
- **`.pdf`** - PDF документ

## 🎨 Кастомизация

Отредактируйте файл `uml_diagrams/agent_dialogue_sim.puml` для изменения:
- Цветов схемы
- Расположения компонентов
- Добавления новых классов
- Изменения связей

Синтаксис PlantUML: https://plantuml.com/class-diagram

