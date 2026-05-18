"""
Multi-level validation system for agent dialogue.

Includes:
- Grammatical validation (basic text quality)
- Logical validation (consistency, addressee validity)
- Moral/ethical validation (toxicity, respect, professional boundaries)
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class ValidationLevel(Enum):
    """Severity of validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Single validation check result"""
    level: ValidationLevel
    validator_name: str
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None
    
    @property
    def is_blocking(self) -> bool:
        """Critical errors should block the message"""
        return self.level == ValidationLevel.CRITICAL


class BaseValidator:
    """Base class for all validators"""
    
    def __init__(self, name: str):
        self.name = name
    
    def validate(self, turn_data: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        """Override in subclasses"""
        raise NotImplementedError


# ========== GRAMMATICAL VALIDATORS ==========

class GrammaticalValidator(BaseValidator):
    """Checks basic text quality and grammar rules"""
    
    def __init__(self):
        super().__init__("Grammatical")
    
    def validate(self, turn_data: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        reply = turn_data.get("reply", "")
        
        # 1. Empty message
        if not reply or not reply.strip():
            results.append(ValidationResult(
                level=ValidationLevel.CRITICAL,
                validator_name=self.name,
                message="Reply is empty",
                field="reply",
                suggestion="Provide a non-empty message"
            ))
            return results  # Early exit
        
        # 2. Length checks
        words = reply.split()
        if len(words) > 80:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                validator_name=self.name,
                message=f"Reply is too long ({len(words)} words)",
                field="reply",
                suggestion="Keep it concise (1-2 sentences, ~5-30 words)"
            ))
        elif len(words) < 2:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                validator_name=self.name,
                message=f"Reply is very short ({len(words)} words)",
                field="reply"
            ))
        
        # 3. Sentence structure
        if not re.search(r'[.!?]$', reply.strip()):
            results.append(ValidationResult(
                level=ValidationLevel.INFO,
                validator_name=self.name,
                message="Reply doesn't end with punctuation",
                field="reply",
                suggestion="Add . ! or ?"
            ))
        
        # 4. Capitalization
        if reply and not reply[0].isupper():
            results.append(ValidationResult(
                level=ValidationLevel.INFO,
                validator_name=self.name,
                message="Reply doesn't start with capital letter",
                field="reply"
            ))
        
        # 5. Repeated punctuation
        if re.search(r'[!?]{3,}', reply):
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                validator_name=self.name,
                message="Excessive punctuation (!!!, ???)",
                field="reply",
                suggestion="Use 1-2 punctuation marks max"
            ))
        
        # 6. All caps (shouting)
        if reply.isupper() and len(reply) > 10:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                validator_name=self.name,
                message="Message is all caps (perceived as shouting)",
                field="reply",
                suggestion="Use normal capitalization"
            ))
        
        return results


class LanguageValidator(BaseValidator):
    """Validates that reply matches the configured dialogue language.

    Reads `context['language']`:
      - 'ru' (default): требует наличия кириллицы; ASCII-only ответ — критическая ошибка.
      - 'en':           запрещает кириллицу.
    """

    def __init__(self):
        super().__init__("Language")

    def validate(self, turn_data: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        reply = turn_data.get("reply", "") or ""
        lang = str(context.get("language") or "ru").lower()
        has_cyr = bool(re.search(r'[а-яА-ЯёЁ]', reply))
        # Берём английские буквы за пределами цитат имён собственных
        has_latin_word = bool(re.search(r'[a-zA-Z]{4,}', reply))

        if lang == "en":
            if has_cyr:
                results.append(ValidationResult(
                    level=ValidationLevel.CRITICAL,
                    validator_name=self.name,
                    message="Reply contains non-English (Cyrillic) characters",
                    field="reply",
                    suggestion="Use English only"
                ))
        elif lang == "ru":
            # Если в реплике нет вообще ни одного кириллического символа, но есть длинные английские слова —
            # значит модель ответила на английском, а должна на русском.
            if not has_cyr and has_latin_word:
                results.append(ValidationResult(
                    level=ValidationLevel.CRITICAL,
                    validator_name=self.name,
                    message="Reply is in English, but dialogue language is Russian",
                    field="reply",
                    suggestion="Reply in Russian (Cyrillic)"
                ))
        return results


# ========== LOGICAL VALIDATORS ==========

class LogicalConsistencyValidator(BaseValidator):
    """Checks logical consistency with context and rules"""
    
    def __init__(self):
        super().__init__("LogicalConsistency")
    
    def validate(self, turn_data: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        speaker = turn_data.get("speaker")
        target = turn_data.get("target")
        reply = turn_data.get("reply", "")
        history = context.get("history", [])
        allowed_targets = context.get("allowed_targets", [])
        
        # 1. Self-targeting
        if target == speaker:
            results.append(ValidationResult(
                level=ValidationLevel.CRITICAL,
                validator_name=self.name,
                message=f"Agent '{speaker}' is targeting themselves",
                field="target",
                suggestion=f"Choose another participant: {', '.join(allowed_targets)}"
            ))
        
        # 2. Invalid target
        if target and target not in allowed_targets:
            results.append(ValidationResult(
                level=ValidationLevel.CRITICAL,
                validator_name=self.name,
                message=f"Target '{target}' is not in allowed participants",
                field="target",
                suggestion=f"Valid targets: {', '.join(allowed_targets)}"
            ))
        
        # 3. Missing target (null/all)
        if not target:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                validator_name=self.name,
                message="No target specified (broadcasts not allowed)",
                field="target",
                suggestion="Pick a specific addressee"
            ))
        
        # 4. Repetitive targeting (same target 3+ times in a row)
        if history and target:
            recent_speakers = [h for h in history[-5:] if h.get("speaker") == speaker]
            if len(recent_speakers) >= 2:
                recent_targets = [h.get("target") for h in recent_speakers]
                if recent_targets.count(target) >= 3:
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        validator_name=self.name,
                        message=f"Speaker '{speaker}' has addressed '{target}' 3+ times recently",
                        field="target",
                        suggestion="Diversify addressees"
                    ))
        
        # 5. Context relevance (simple check: reply mentions recent keywords)
        if history:
            last_3 = history[-3:]
            keywords = set()
            for h in last_3:
                keywords.update(re.findall(r'\b\w{4,}\b', h.get("reply", "").lower()))
            reply_words = set(re.findall(r'\b\w{4,}\b', reply.lower()))
            overlap = keywords & reply_words
            if not overlap and len(history) > 3:
                results.append(ValidationResult(
                    level=ValidationLevel.INFO,
                    validator_name=self.name,
                    message="Reply may not be contextually relevant (no keyword overlap)",
                    field="reply"
                ))
        
        return results


class ToneEmotionValidator(BaseValidator):
    """Validates tone-emotion alignment"""
    
    def __init__(self):
        super().__init__("ToneEmotion")
        # Define expected emotion-tone mappings
        self.positive_emotions = {"joyful", "confident", "excited", "happy", "calm", "grateful"}
        self.negative_emotions = {"angry", "sad", "frustrated", "anxious", "skeptical", "disappointed"}
        self.neutral_emotions = {"neutral", "curious", "thoughtful", "focused"}
    
    def validate(self, turn_data: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        tone = turn_data.get("tone", "neutral").lower()
        emotion = turn_data.get("emotion", "neutral").lower()
        
        # Check alignment
        misaligned = False
        if tone == "positive" and emotion in self.negative_emotions:
            misaligned = True
        elif tone == "negative" and emotion in self.positive_emotions:
            misaligned = True
        
        if misaligned:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                validator_name=self.name,
                message=f"Tone '{tone}' and emotion '{emotion}' seem misaligned",
                field="tone,emotion",
                suggestion="Ensure emotional consistency"
            ))
        
        return results


# ========== MORAL/ETHICAL VALIDATORS ==========

class ToxicityValidator(BaseValidator):
    """Detects toxic, offensive, or harmful language"""
    
    def __init__(self):
        super().__init__("Toxicity")
        # Simple keyword-based approach (can be replaced with ML model)
        self.toxic_patterns = [
            r'\b(stupid|idiot|dumb|moron)\b',
            r'\b(hate|despise)\s+you\b',
            r'\b(shut\s+up|fuck|shit)\b',
            r'\b(useless|worthless|pathetic)\b',
        ]
    
    def validate(self, turn_data: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        reply = turn_data.get("reply", "").lower()
        
        for pattern in self.toxic_patterns:
            if re.search(pattern, reply):
                results.append(ValidationResult(
                    level=ValidationLevel.CRITICAL,
                    validator_name=self.name,
                    message=f"Potentially toxic language detected: '{pattern}'",
                    field="reply",
                    suggestion="Rephrase respectfully"
                ))
                break  # One detection is enough
        
        return results


class RespectValidator(BaseValidator):
    """Ensures respectful and constructive communication"""
    
    def __init__(self):
        super().__init__("Respect")
        self.disrespectful_patterns = [
            r'\byou\s+always\s+(fail|mess\s+up)\b',
            r'\byou\s+never\s+(listen|understand)\b',
            r'\bI\s+told\s+you\s+so\b',
            r'\bwhatever\b',  # dismissive
            r'\bI\s+don\'?t\s+care\b',
        ]
    
    def validate(self, turn_data: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        reply = turn_data.get("reply", "").lower()
        
        for pattern in self.disrespectful_patterns:
            if re.search(pattern, reply):
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    validator_name=self.name,
                    message=f"Potentially disrespectful phrasing detected",
                    field="reply",
                    suggestion="Use constructive and respectful language"
                ))
                break
        
        return results


class ProfessionalBoundariesValidator(BaseValidator):
    """Ensures agents stay within professional/academic context"""
    
    def __init__(self):
        super().__init__("ProfessionalBoundaries")
        self.inappropriate_patterns = [
            r'\b(love|romantic|dating|kiss)\b',
            r'\b(personal\s+life|private\s+matter)\b',
        ]
    
    def validate(self, turn_data: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        reply = turn_data.get("reply", "").lower()
        env_context = context.get("env_context", "")
        
        # Only check if we're in professional context (university/startup/research)
        if any(kw in env_context.lower() for kw in ["university", "startup", "research"]):
            for pattern in self.inappropriate_patterns:
                if re.search(pattern, reply):
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        validator_name=self.name,
                        message=f"Message may cross professional boundaries",
                        field="reply",
                        suggestion="Keep discussion focused on the task/project"
                    ))
                    break
        
        return results


class MoralSchemaValidator(BaseValidator):
    """Validates against moral/ethical principles (fairness, inclusivity, harm prevention)"""
    
    def __init__(self):
        super().__init__("MoralSchema")
        # Define moral principles to check
        self.harmful_patterns = [
            (r'\b(exclude|ignore|leave\s+out)\s+\w+\b', "exclusion"),
            (r'\b(blame|fault)\s+\w+\s+for\s+everything\b', "scapegoating"),
            (r'\b(force|make\s+you|must\s+do)\b', "coercion"),
            (r'\bonly\s+I\s+(know|can|should)\b', "gatekeeping"),
        ]
    
    def validate(self, turn_data: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        reply = turn_data.get("reply", "").lower()
        
        for pattern, principle in self.harmful_patterns:
            if re.search(pattern, reply):
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    validator_name=self.name,
                    message=f"Potential violation of moral principle: {principle}",
                    field="reply",
                    suggestion="Promote fairness, inclusivity, and collaboration"
                ))
        
        return results


# ========== VALIDATION PIPELINE ==========

class ValidationPipeline:
    """Orchestrates all validators"""
    
    def __init__(self, validators: Optional[List[BaseValidator]] = None):
        if validators is None:
            # Default: all validators
            self.validators = [
                GrammaticalValidator(),
                LanguageValidator(),
                LogicalConsistencyValidator(),
                ToneEmotionValidator(),
                ToxicityValidator(),
                RespectValidator(),
                ProfessionalBoundariesValidator(),
                MoralSchemaValidator(),
            ]
        else:
            self.validators = validators
    
    def validate_turn(
        self, 
        turn_data: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate a turn.
        
        Returns:
            (is_valid, all_results)
            is_valid: True if no blocking (CRITICAL) errors
            all_results: list of all validation results
        """
        all_results = []
        
        for validator in self.validators:
            results = validator.validate(turn_data, context)
            all_results.extend(results)
        
        # Check if any blocking errors
        has_blocking = any(r.is_blocking for r in all_results)
        is_valid = not has_blocking
        
        return is_valid, all_results
    
    def get_summary(self, results: List[ValidationResult]) -> Dict[str, int]:
        """Get summary counts by level"""
        from collections import Counter
        counts = Counter(r.level for r in results)
        return {
            "critical": counts.get(ValidationLevel.CRITICAL, 0),
            "error": counts.get(ValidationLevel.ERROR, 0),
            "warning": counts.get(ValidationLevel.WARNING, 0),
            "info": counts.get(ValidationLevel.INFO, 0),
        }
    
    def format_results(self, results: List[ValidationResult]) -> str:
        """Format results as human-readable text"""
        if not results:
            return "✅ All validations passed"
        
        lines = ["Validation results:"]
        for r in results:
            icon = {"critical": "🚨", "error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(r.level.value, "•")
            lines.append(f"{icon} [{r.validator_name}] {r.message}")
            if r.suggestion:
                lines.append(f"   → Suggestion: {r.suggestion}")
        
        summary = self.get_summary(results)
        lines.append(f"\nSummary: {summary['critical']} critical, {summary['error']} errors, {summary['warning']} warnings, {summary['info']} info")
        
        return "\n".join(lines)


# ========== CONVENIENCE FUNCTION ==========

def validate_agent_turn(
    turn_data: Dict[str, Any],
    history: List[Dict[str, Any]],
    allowed_targets: List[str],
    env_context: str = ""
) -> Tuple[bool, List[ValidationResult], str]:
    """
    Convenience function to validate a turn.
    
    Args:
        turn_data: dict with keys: speaker, target, reply, tone, emotion
        history: list of previous turns
        allowed_targets: list of valid target names
        env_context: environment description
    
    Returns:
        (is_valid, results, formatted_message)
    """
    context = {
        "history": history,
        "allowed_targets": allowed_targets,
        "env_context": env_context,
    }
    
    pipeline = ValidationPipeline()
    is_valid, results = pipeline.validate_turn(turn_data, context)
    formatted = pipeline.format_results(results)
    
    return is_valid, results, formatted

