"""
Quick examples of validation system usage.
"""

# ========== EXAMPLE 1: Basic validation check ==========

from validators import validate_agent_turn

# Setup
history = [
    {"speaker": "Alice", "target": "Bob", "reply": "What do you think?"},
]
allowed_targets = ["Alice", "Bob", "Charlie"]
env_context = "University: Planning an AI project"

# Turn to validate
turn_data = {
    "speaker": "Bob",
    "target": "Alice",
    "reply": "That's a great idea! Let's proceed.",
    "tone": "positive",
    "emotion": "confident",
}

# Validate
is_valid, results, formatted = validate_agent_turn(
    turn_data, history, allowed_targets, env_context
)

print("Example 1: Valid message")
print(f"Is valid: {is_valid}")
print(f"Issues found: {len(results)}")
if results:
    print(formatted)
print()

# ========== EXAMPLE 2: Invalid message (toxic) ==========

turn_data_toxic = {
    "speaker": "Charlie",
    "target": "Bob",
    "reply": "That's a stupid suggestion.",
    "tone": "negative",
    "emotion": "angry",
}

is_valid, results, formatted = validate_agent_turn(
    turn_data_toxic, history, allowed_targets, env_context
)

print("Example 2: Toxic message")
print(f"Is valid: {is_valid}")
print(formatted)
print()

# ========== EXAMPLE 3: Logical error (self-targeting) ==========

turn_data_self = {
    "speaker": "Alice",
    "target": "Alice",  # ❌ Self-targeting!
    "reply": "I'll handle this myself.",
    "tone": "neutral",
    "emotion": "focused",
}

is_valid, results, formatted = validate_agent_turn(
    turn_data_self, history, allowed_targets, env_context
)

print("Example 3: Self-targeting")
print(f"Is valid: {is_valid}")
print(formatted)
print()

# ========== EXAMPLE 4: Moral violation (exclusion) ==========

turn_data_exclusion = {
    "speaker": "Bob",
    "target": "Charlie",
    "reply": "Let's exclude Alice from this discussion.",
    "tone": "neutral",
    "emotion": "focused",
}

is_valid, results, formatted = validate_agent_turn(
    turn_data_exclusion, history, allowed_targets, env_context
)

print("Example 4: Moral violation (exclusion)")
print(f"Is valid: {is_valid}")
print(formatted)
print()

# ========== EXAMPLE 5: Custom validator pipeline ==========

from validators import (
    ValidationPipeline,
    GrammaticalValidator,
    LogicalConsistencyValidator,
    MoralSchemaValidator,
)

# Create custom pipeline (only specific validators)
custom_pipeline = ValidationPipeline(validators=[
    GrammaticalValidator(),
    LogicalConsistencyValidator(),
    MoralSchemaValidator(),
])

context = {
    "history": history,
    "allowed_targets": allowed_targets,
    "env_context": env_context,
}

turn_data = {
    "speaker": "Bob",
    "target": "Charlie",
    "reply": "great idea",  # ℹ️ No capitalization, no punctuation
    "tone": "positive",
    "emotion": "happy",
}

is_valid, results = custom_pipeline.validate_turn(turn_data, context)

print("Example 5: Custom pipeline (grammar + logic + moral only)")
print(f"Is valid: {is_valid}")
print(f"Issues: {len(results)}")
for r in results:
    print(f"  [{r.level.value}] {r.message}")
print()

# ========== EXAMPLE 6: Validation statistics ==========

from collections import Counter

# Simulate multiple turns
test_turns = [
    {"reply": "Great!", "speaker": "A", "target": "B", "tone": "positive", "emotion": "joyful"},
    {"reply": "STOP SHOUTING", "speaker": "B", "target": "A", "tone": "negative", "emotion": "angry"},
    {"reply": "you're an idiot", "speaker": "C", "target": "A", "tone": "negative", "emotion": "angry"},
    {"reply": "Let's exclude Bob", "speaker": "A", "target": "C", "tone": "neutral", "emotion": "focused"},
]

all_results = []
for turn in test_turns:
    _, results, _ = validate_agent_turn(turn, [], ["A", "B", "C"], "Test")
    all_results.extend(results)

print("Example 6: Validation statistics across multiple turns")
print(f"Total issues: {len(all_results)}")
by_level = Counter(r.level.value for r in all_results)
print(f"By level: {dict(by_level)}")
by_validator = Counter(r.validator_name for r in all_results)
print(f"By validator: {dict(by_validator)}")

