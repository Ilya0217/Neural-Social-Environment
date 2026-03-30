#!/usr/bin/env python3
"""
Генератор UML диаграмм для проекта agent_dialogue_sim.

Использует pyreverse для автоматического анализа Python кода и создания
диаграмм классов и пакетов.
"""

import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Проверка наличия необходимых зависимостей."""
    try:
        import pylint
    except ImportError:
        print("❌ Ошибка: pylint не установлен")
        print("Установите: pip install pylint")
        return False
    
    try:
        import graphviz
    except ImportError:
        print("⚠️  Предупреждение: graphviz не установлен")
        print("Для PNG/SVG вывода установите: pip install graphviz")
        print("Также установите системный пакет Graphviz: https://graphviz.org/download/")
        return True  # Продолжаем, но без graphviz
    
    return True

def generate_uml(output_format='png', output_dir='uml_diagrams'):
    """
    Генерация UML диаграмм проекта.
    
    Args:
        output_format: Формат вывода ('png', 'svg', 'pdf', 'dot')
        output_dir: Директория для сохранения диаграмм
    """
    project_root = Path(__file__).parent
    output_path = project_root / output_dir
    output_path.mkdir(exist_ok=True)
    
    # Имя модуля для анализа
    module_name = "agent_dialogue_sim"
    
    print(f"🔍 Анализ проекта: {module_name}")
    print(f"📁 Выходная директория: {output_path}")
    
    # Команда pyreverse для генерации диаграммы классов
    cmd_classes = [
        'pyreverse',
        '-o', output_format,  # Формат вывода
        '-p', 'AgentDialogueSim',  # Префикс для имён файлов
        '--project', module_name,
        '-d', str(output_path),  # Директория вывода
        str(project_root / module_name),  # Путь к модулю
    ]
    
    # Команда для генерации диаграммы пакетов
    cmd_packages = [
        'pyreverse',
        '-o', output_format,
        '-p', 'AgentDialogueSimPackages',
        '--project', module_name,
        '-AS',  # Только пакеты (packages only)
        '-d', str(output_path),
        str(project_root / module_name),
    ]
    
    results = []
    
    # Генерация диаграммы классов
    print("\n📊 Генерация диаграммы классов...")
    try:
        result = subprocess.run(
            cmd_classes,
            capture_output=True,
            text=True,
            cwd=project_root
        )
        if result.returncode == 0:
            print("✅ Диаграмма классов создана")
            results.append("classes")
        else:
            print(f"⚠️  Предупреждение: {result.stderr}")
            if result.stdout:
                print(result.stdout)
    except Exception as e:
        print(f"❌ Ошибка при генерации диаграммы классов: {e}")
    
    # Генерация диаграммы пакетов
    print("\n📦 Генерация диаграммы пакетов...")
    try:
        result = subprocess.run(
            cmd_packages,
            capture_output=True,
            text=True,
            cwd=project_root
        )
        if result.returncode == 0:
            print("✅ Диаграмма пакетов создана")
            results.append("packages")
        else:
            print(f"⚠️  Предупреждение: {result.stderr}")
            if result.stdout:
                print(result.stdout)
    except Exception as e:
        print(f"❌ Ошибка при генерации диаграммы пакетов: {e}")
    
    # Вывод информации о созданных файлах
    print(f"\n📄 Созданные файлы в {output_path}:")
    for file in sorted(output_path.glob(f"*{output_format}")):
        size_kb = file.stat().st_size / 1024
        print(f"  - {file.name} ({size_kb:.1f} KB)")
    
    # Если создан DOT файл, можно конвертировать в другие форматы
    dot_files = list(output_path.glob("*.dot"))
    if dot_files and output_format != 'dot':
        print(f"\n💡 Для конвертации DOT в другие форматы используйте Graphviz:")
        for dot_file in dot_files:
            print(f"  dot -T{output_format} {dot_file} -o {dot_file.stem}.{output_format}")
    
    return results

def generate_plantuml_diagram():
    """Генерация PlantUML диаграммы (альтернативный метод)."""
    project_root = Path(__file__).parent
    output_path = project_root / "uml_diagrams" / "diagram.puml"
    output_path.parent.mkdir(exist_ok=True)
    
    # Создаём упрощённую PlantUML диаграмму основных классов
    puml_content = """@startuml AgentDialogueSim
!theme plain
skinparam classAttributeIconSize 0

package "agent_dialogue_sim" {
    
    class Agent {
        +str name
        +str nature
        +str color
        +bool is_human
        +str persona
        +system_prompt() : str
    }
    
    class DialogueManager {
        +OpenAI client
        +List[Agent] agents
        +str env_context
        +List[Dict] history
        +int turn_no
        +bool enable_validation
        +ValidationPipeline validation_pipeline
        +Dict[str, float] agent_moods
        +next_speaker_idx() : int
        +build_messages(agent_idx) : List[Dict]
        +model_turn(messages, allowed_targets) : AgentTurn
        +step() : Tuple[Dict, Dict]
        +current_phase() : Dict
        +_decay_moods()
        +_adjust_mood(speaker, tone, target)
    }
    
    class AgentTurn {
        +str reply
        +str tone
        +str emotion
        +Optional[str] target
    }
    
    abstract class BaseValidator {
        +str name
        +validate(turn_data, context) : List[ValidationResult]
    }
    
    class GrammaticalValidator extends BaseValidator
    class LanguageValidator extends BaseValidator
    class LogicalConsistencyValidator extends BaseValidator
    class ToneEmotionValidator extends BaseValidator
    class ToxicityValidator extends BaseValidator
    class RespectValidator extends BaseValidator
    class ProfessionalBoundariesValidator extends BaseValidator
    class MoralSchemaValidator extends BaseValidator
    
    class ValidationPipeline {
        +List[BaseValidator] validators
        +validate_turn(turn_data, context) : Tuple[bool, List[ValidationResult]]
        +format_results(results) : str
    }
    
    class ValidationResult {
        +ValidationLevel level
        +str validator_name
        +str message
        +Optional[str] field
        +Optional[str] suggestion
        +is_blocking() : bool
    }
    
    enum ValidationLevel {
        INFO
        WARNING
        ERROR
        CRITICAL
    }
    
    class CacheManager {
        +int max_size
        +int ttl_seconds
        +get(context) : Optional[Any]
        +set(context, value)
        +get_stats() : Dict
    }
    
    class SemanticCacheManager extends CacheManager {
        +float similarity_threshold
    }
    
    class PerformanceMonitor {
        +log_call(metrics)
        +get_stats() : PerformanceStats
        +generate_report() : str
    }
    
    class ExportManager {
        +export_to_csv(history, filename) : Path
        +export_to_excel(history, metrics, filename) : Path
        +export_to_json(history, metrics, filename) : Path
        +export_research_report(history, metrics, filename) : Path
    }
    
    class IOLogger {
        +write_jsonl(record)
        +write_markdown(turn_no, speaker, target, tone, emotion, text)
        +write_metrics_block(title, content)
    }
    
    class HumanIO {
        +get_user_turn(speaker, allowed_targets, history) : AgentTurn
    }
    
    class AppState {
        +Optional[OpenAI] client
        +Optional[DialogueManager] dm
        +Optional[IOLogger] logger
        +Dict agents_meta
        +str env_context
        +reset(env_index, agents_config)
    }
    
    class ScientificAnalysisResult {
        +Dict group_stage
        +float network_density
        +float clustering_coefficient
        +Dict social_capital
        +Dict dialogue_act_profile
        +str dominant_emotion
        +Dict emotion_distribution
        +Dict turn_taking
        +Dict centrality
        +Dict betweenness
    }
    
    ' Связи
    DialogueManager "1" *-- "many" Agent : manages
    DialogueManager "1" --> "1" ValidationPipeline : uses
    DialogueManager "1" --> "1" HumanIO : optional
    DialogueManager "1" --> "many" AgentTurn : generates
    
    ValidationPipeline "1" *-- "many" BaseValidator : contains
    BaseValidator <|-- GrammaticalValidator
    BaseValidator <|-- LanguageValidator
    BaseValidator <|-- LogicalConsistencyValidator
    BaseValidator <|-- ToneEmotionValidator
    BaseValidator <|-- ToxicityValidator
    BaseValidator <|-- RespectValidator
    BaseValidator <|-- ProfessionalBoundariesValidator
    BaseValidator <|-- MoralSchemaValidator
    
    ValidationPipeline "1" --> "many" ValidationResult : produces
    ValidationResult "1" --> "1" ValidationLevel : has
    
    Agent "1" --> "1" AgentTurn : generates
    AgentTurn ..> Agent : references
    
    CacheManager <|-- SemanticCacheManager
    
    AppState "1" --> "1" DialogueManager : contains
    AppState "1" --> "1" IOLogger : contains
    
    DialogueManager ..> ScientificAnalysisResult : analyzes to
    
    note right of DialogueManager
        Центральный оркестратор
        многоагентного взаимодействия
    end note
    
    note right of ValidationPipeline
        Многоуровневая система
        проверки качества сообщений
    end note
    
    note right of BaseValidator
        Strategy Pattern:
        Плагинная архитектура
        валидаторов
    end note

}
@enduml
"""
    
    output_path.write_text(puml_content, encoding='utf-8')
    print(f"✅ PlantUML диаграмма создана: {output_path}")
    print(f"💡 Для визуализации установите PlantUML и выполните:")
    print(f"   plantuml {output_path}")
    print(f"   или используйте онлайн: http://www.plantuml.com/plantuml/uml/")
    
    return output_path

def main():
    """Главная функция."""
    print("=" * 60)
    print("🎨 Генератор UML диаграмм для agent_dialogue_sim")
    print("=" * 60)
    
    # Проверка зависимостей
    if not check_dependencies():
        print("\n❌ Установите необходимые зависимости и попробуйте снова")
        return 1
    
    # Выбор формата
    print("\n📋 Доступные форматы:")
    print("  1. png - PNG изображение (требует Graphviz)")
    print("  2. svg - SVG векторное изображение (требует Graphviz)")
    print("  3. pdf - PDF документ (требует Graphviz)")
    print("  4. dot - Graphviz DOT формат (всегда доступен)")
    print("  5. puml - PlantUML диаграмма (альтернативный метод)")
    
    choice = input("\nВыберите формат [1-5] (по умолчанию: 4): ").strip() or "4"
    
    format_map = {
        "1": "png",
        "2": "svg",
        "3": "pdf",
        "4": "dot",
        "5": "puml"
    }
    
    output_format = format_map.get(choice, "dot")
    
    if output_format == "puml":
        # Генерация PlantUML диаграммы
        generate_plantuml_diagram()
    else:
        # Генерация через pyreverse
        results = generate_uml(output_format=output_format)
        
        if not results:
            print("\n⚠️  Не удалось создать диаграммы через pyreverse")
            print("💡 Попробуйте альтернативный метод PlantUML (выберите опцию 5)")
    
    print("\n" + "=" * 60)
    print("✅ Готово!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

