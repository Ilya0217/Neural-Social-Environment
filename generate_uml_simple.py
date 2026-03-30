#!/usr/bin/env python3
"""
Упрощённый генератор UML диаграмм (не требует дополнительных зависимостей).

Создаёт PlantUML файл, который можно визуализировать онлайн или через PlantUML CLI.
"""

from pathlib import Path

def generate_plantuml_diagram(compact=True):
    """Генерация PlantUML диаграммы проекта.
    
    Args:
        compact: Если True, создаёт упрощённую версию для научной работы
    """
    project_root = Path(__file__).parent
    if compact:
        output_path = project_root / "uml_diagrams" / "agent_dialogue_sim_final.puml"
    else:
        output_path = project_root / "uml_diagrams" / "agent_dialogue_sim.puml"
    output_path.parent.mkdir(exist_ok=True)
    
    if compact:
        # Упрощённая версия для научной работы
        puml_content = """@startuml AgentDialogueSim
!theme plain
skinparam classAttributeIconSize 0
skinparam linetype ortho
skinparam packageStyle rectangle
skinparam shadowing false
skinparam roundcorner 3
skinparam defaultFontSize 9
skinparam classFontSize 9
skinparam packageFontSize 10

title Архитектура системы симуляции многоагентных диалогов

package "Ядро системы" {
    class Agent {
        +name
        +nature
        +system_prompt()
    }
    
    class DialogueManager {
        +agents
        +history
        +step()
        +build_messages()
        +model_turn()
    }
    
    class AgentTurn {
        +reply
        +tone
        +emotion
        +target
    }
}

package "Валидация" {
    abstract class BaseValidator {
        +validate()
    }
    
    class ValidationPipeline {
        +validators
        +validate_turn()
    }
    
    BaseValidator <|-- GrammaticalValidator
    BaseValidator <|-- LanguageValidator
    BaseValidator <|-- LogicalValidator
    BaseValidator <|-- EthicalValidator
}

package "Аналитика" {
    class MetricsCalculator {
        +compute_metrics()
        +hypotheses_from_metrics()
    }
    
    class ScientificAnalyzer {
        +compute_scientific_analysis()
    }
}

package "Интерфейсы" {
    class WebApp {
        +api_start()
        +api_step()
        +api_state()
    }
    
    class IOLogger {
        +write_jsonl()
        +write_markdown()
    }
}

' Основные связи
DialogueManager "1" *-- "many" Agent : управляет
DialogueManager "1" --> "many" AgentTurn : генерирует
DialogueManager "1" --> "1" ValidationPipeline : использует
DialogueManager ..> MetricsCalculator : анализирует
DialogueManager ..> ScientificAnalyzer : анализирует
DialogueManager ..> IOLogger : логирует

ValidationPipeline "1" *-- "many" BaseValidator : содержит

WebApp "1" --> "1" DialogueManager : использует
WebApp "1" --> "1" IOLogger : использует

note right of DialogueManager
    **Центральный оркестратор**
    Управление диалогом,
    генерация ответов через LLM API,
    координация агентов
end note

note right of ValidationPipeline
    **Strategy Pattern**
    Многоуровневая валидация:
    грамматическая, логическая,
    этическая
end note

@enduml
"""
    else:
        # Полная версия
        puml_content = """@startuml AgentDialogueSim
!theme plain
skinparam classAttributeIconSize 0
skinparam linetype ortho

title Диаграмма классов системы Agent Dialogue Simulator

package "Core Components" {
    
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
        +List[Dict] phase_plan
        --
        +next_speaker_idx() : int
        +build_messages(agent_idx) : List[Dict]
        +model_turn(messages, allowed_targets) : AgentTurn
        +step() : Tuple[Dict, Dict]
        +current_phase() : Dict
        +should_plot() : bool
        -_decay_moods()
        -_adjust_mood(speaker, tone, target)
        -_phase_block() : str
        -_mood_guidance(agent_name) : str
    }
    
    class AgentTurn {
        +str reply
        +Literal["positive","neutral","negative"] tone
        +str emotion
        +Optional[str] target
    }
    
    class HumanIO {
        +OpenAI client
        +bool classify_with_model
        +get_user_turn(speaker, allowed_targets, history) : AgentTurn
    }
}

package "Validation System" {
    
    enum ValidationLevel {
        INFO
        WARNING
        ERROR
        CRITICAL
    }
    
    class ValidationResult {
        +ValidationLevel level
        +str validator_name
        +str message
        +Optional[str] field
        +Optional[str] suggestion
        +is_blocking() : bool
    }
    
    abstract class BaseValidator {
        +str name
        +validate(turn_data, context) : List[ValidationResult]
    }
    
    class GrammaticalValidator extends BaseValidator {
        +validate(turn_data, context) : List[ValidationResult]
    }
    
    class LanguageValidator extends BaseValidator {
        +validate(turn_data, context) : List[ValidationResult]
    }
    
    class LogicalConsistencyValidator extends BaseValidator {
        +validate(turn_data, context) : List[ValidationResult]
    }
    
    class ToneEmotionValidator extends BaseValidator {
        +validate(turn_data, context) : List[ValidationResult]
    }
    
    class ToxicityValidator extends BaseValidator {
        +validate(turn_data, context) : List[ValidationResult]
    }
    
    class RespectValidator extends BaseValidator {
        +validate(turn_data, context) : List[ValidationResult]
    }
    
    class ProfessionalBoundariesValidator extends BaseValidator {
        +validate(turn_data, context) : List[ValidationResult]
    }
    
    class MoralSchemaValidator extends BaseValidator {
        +validate(turn_data, context) : List[ValidationResult]
    }
    
    class ValidationPipeline {
        +List[BaseValidator] validators
        +validate_turn(turn_data, context) : Tuple[bool, List[ValidationResult]]
        +get_summary(results) : Dict
        +format_results(results) : str
    }
}

package "Prompt System" {
    
    class PromptBuilder {
        <<utility>>
        +build_system_prompt(nature, persona) : str
        +get_advanced_prompt(role, context_type, include_cot, include_examples) : str
        +detect_context_type(history) : str
    }
    
    note right of PromptBuilder
        Chain-of-Thought prompting
        Context-aware adaptation
        Few-shot examples
    end note
}

package "Analytics" {
    
    class MetricsCalculator {
        <<utility>>
        +compute_metrics(history, agents, window_size) : Dict
        +hypotheses_from_metrics(metrics, user_name) : List[str]
        +render_markdown_report(metrics, turn, title) : str
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
    
    class ScientificAnalyzer {
        <<utility>>
        +compute_scientific_analysis(history, agents, window_size) : ScientificAnalysisResult
        +generate_scientific_hypotheses(analysis, agents) : List[Dict]
        +render_scientific_report(analysis, hypotheses, turn) : str
    }
}

package "Performance & Optimization" {
    
    class CacheManager {
        +int max_size
        +int ttl_seconds
        +bool enable_persistence
        +get(context) : Optional[Any]
        +set(context, value)
        +get_stats() : Dict
        +clear()
    }
    
    class SemanticCacheManager extends CacheManager {
        +float similarity_threshold
        +_compute_similarity(ctx1, ctx2) : float
    }
    
    class PerformanceMonitor {
        +Path log_path
        +log_call(prompt_tokens, completion_tokens, latency_ms, success)
        +get_stats() : PerformanceStats
        +generate_report() : str
    }
    
    class APICallMetrics {
        +int prompt_tokens
        +int completion_tokens
        +int total_tokens
        +float latency_ms
        +bool success
        +float cost
        +datetime timestamp
    }
    
    class PerformanceStats {
        +int total_calls
        +int successful_calls
        +int failed_calls
        +float success_rate
        +int total_prompt_tokens
        +int total_completion_tokens
        +int total_tokens
        +float total_cost
        +float avg_latency_ms
        +float min_latency_ms
        +float max_latency_ms
    }
}

package "Export & Logging" {
    
    class ExportManager {
        +Path output_dir
        +export_to_csv(history, filename) : Path
        +export_to_excel(history, metrics, filename) : Path
        +export_to_json(history, metrics, filename) : Path
        +export_research_report(history, metrics, filename) : Path
        +export_all_formats(history, metrics, base_name) : Dict
    }
    
    class IOLogger {
        +Path jsonl_path
        +Path md_path
        +str env_context
        +write_jsonl(record)
        +write_markdown(turn_no, speaker, target, tone, emotion, text)
        +write_metrics_block(title, content)
    }
}

package "Visualization" {
    
    class GraphVisualizer {
        <<utility>>
        +draw_interactions_pro(agents_meta, history, out_path, title, window, seed) : None
        +tone_to_color(avg_score) : str
        +compute_stats(agents_meta, history, window) : Dict
    }
}

package "Web Interface" {
    
    class AppState {
        +Optional[OpenAI] client
        +Optional[DialogueManager] dm
        +Optional[IOLogger] logger
        +Dict agents_meta
        +str env_context
        +Optional[Dict] last_metrics
        +List[str] last_hyps
        +List[Dict] last_scientific_hyps
        +str last_scientific_report
        +reset(env_index, agents_config)
    }
    
    class FlaskApp {
        +Flask app
        +AppState state
        +index() : Response
        +api_start() : JSON
        +api_step() : JSON
        +api_state() : JSON
        +api_scientific() : JSON
        +serve_outputs(filename) : Response
        +download_log() : Response
    }
}

package "Configuration" {
    
    class Config {
        <<configuration>>
        +str MODEL
        +float TEMPERATURE
        +int MAX_TOKENS
        +List[Dict] DEFAULT_AGENTS
        +List[Dict] DIALOGUE_PHASES
        +float MOOD_DECAY
        +Dict MOOD_DELTA
        +List[Tuple] MOOD_GUIDANCE
    }
}

' Основные связи
DialogueManager "1" *-- "many" Agent : manages
DialogueManager "1" --> "1" ValidationPipeline : uses
DialogueManager "1" --> "0..1" HumanIO : optional
DialogueManager "1" --> "many" AgentTurn : generates
DialogueManager ..> PromptBuilder : uses
DialogueManager ..> MetricsCalculator : uses
DialogueManager ..> ScientificAnalyzer : uses

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
HumanIO "1" --> "1" AgentTurn : generates

DialogueManager ..> CacheManager : optional
DialogueManager ..> PerformanceMonitor : optional
DialogueManager ..> IOLogger : uses
DialogueManager ..> GraphVisualizer : uses
DialogueManager ..> ExportManager : uses

ScientificAnalyzer "1" --> "1" ScientificAnalysisResult : produces

CacheManager <|-- SemanticCacheManager

PerformanceMonitor "1" --> "many" APICallMetrics : logs
PerformanceMonitor "1" --> "1" PerformanceStats : aggregates

AppState "1" --> "1" DialogueManager : contains
AppState "1" --> "1" IOLogger : contains
FlaskApp "1" --> "1" AppState : uses

DialogueManager ..> Config : reads

' Заметки
note right of DialogueManager
    **Центральный оркестратор**
    - Управление фазами диалога
    - Выбор следующего говорящего
    - Сборка промптов
    - Генерация ответов через API
    - Управление настроением агентов
end note

note right of ValidationPipeline
    **Многоуровневая валидация**
    - Грамматическая
    - Логическая
    - Этическая
    Strategy Pattern
end note

note right of BaseValidator
    **Плагинная архитектура**
    Легко добавлять новые
    типы валидаторов
end note

note bottom of PromptBuilder
    **Chain-of-Thought**
    Context-aware adaptation
    Few-shot learning
end note

note right of ScientificAnalyzer
    **Научные фреймворки:**
    - Social Network Analysis
    - Group Development (Wheelan)
    - Dialogue Acts (ISO 24617-2)
    - Dimensional Emotion Model
end note

@enduml
"""
    
    output_path.write_text(puml_content, encoding='utf-8')
    print(f"✅ PlantUML диаграмма создана: {output_path}")
    print(f"\n📊 Для визуализации:")
    print(f"   1. Онлайн: http://www.plantuml.com/plantuml/uml/")
    print(f"      Скопируйте содержимое файла на сайт")
    print(f"   2. CLI: plantuml {output_path}")
    print(f"      (требует установки PlantUML)")
    print(f"   3. VS Code: установите расширение 'PlantUML'")
    print(f"\n📁 Файл: {output_path.absolute()}")
    
    return output_path

if __name__ == "__main__":
    print("=" * 60)
    print("🎨 Генератор UML диаграмм")
    print("=" * 60)
    
    choice = input("\nВыберите версию:\n  1. Упрощённая (для научной работы) [по умолчанию]\n  2. Полная\nВаш выбор [1-2]: ").strip() or "1"
    
    compact = choice == "1"
    version = "упрощённая" if compact else "полная"
    
    print(f"\n📊 Генерация {version} версии...")
    generate_plantuml_diagram(compact=compact)
    print("\n✅ Готово!")

