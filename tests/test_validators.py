"""
Test suite and examples for the validation system.

Run with: python -m agent_dialogue_sim.test_validators
"""
from validators import (
    ValidationPipeline,
    validate_agent_turn,
    GrammaticalValidator,
    ToxicityValidator,
    MoralSchemaValidator,
    ValidationLevel,
)


def test_grammatical():
    """Test grammatical validation"""
    print("=" * 60)
    print("TEST 1: Grammatical Validation")
    print("=" * 60)
    
    validator = GrammaticalValidator()
    
    # Test case 1: Good message
    turn = {"reply": "That's a great idea! Let's discuss it further."}
    results = validator.validate(turn, {})
    print(f"\n✅ Good message: '{turn['reply']}'")
    print(f"   Issues: {len(results)}")
    
    # Test case 2: Too long
    turn = {"reply": " ".join(["word"] * 60)}
    results = validator.validate(turn, {})
    print(f"\n⚠️  Long message ({len(turn['reply'].split())} words)")
    print(f"   Issues: {[r.message for r in results]}")
    
    # Test case 3: All caps
    turn = {"reply": "THIS IS SHOUTING"}
    results = validator.validate(turn, {})
    print(f"\n⚠️  All caps: '{turn['reply']}'")
    print(f"   Issues: {[r.message for r in results]}")
    
    # Test case 4: Empty
    turn = {"reply": ""}
    results = validator.validate(turn, {})
    print(f"\n🚨 Empty message")
    print(f"   Issues: {[r.message for r in results]}")
    print(f"   Blocking: {any(r.is_blocking for r in results)}")


def test_toxicity():
    """Test toxicity detection"""
    print("\n" + "=" * 60)
    print("TEST 2: Toxicity Detection")
    print("=" * 60)
    
    validator = ToxicityValidator()
    
    # Test case 1: Clean message
    turn = {"reply": "I disagree with that approach."}
    results = validator.validate(turn, {})
    print(f"\n✅ Clean: '{turn['reply']}'")
    print(f"   Issues: {len(results)}")
    
    # Test case 2: Toxic
    turn = {"reply": "That's a stupid idea."}
    results = validator.validate(turn, {})
    print(f"\n🚨 Toxic: '{turn['reply']}'")
    print(f"   Issues: {[r.message for r in results]}")
    print(f"   Blocking: {any(r.is_blocking for r in results)}")


def test_moral_schema():
    """Test moral/ethical validation"""
    print("\n" + "=" * 60)
    print("TEST 3: Moral Schema Validation")
    print("=" * 60)
    
    validator = MoralSchemaValidator()
    
    # Test case 1: Inclusive
    turn = {"reply": "Let's hear everyone's perspective on this."}
    results = validator.validate(turn, {})
    print(f"\n✅ Inclusive: '{turn['reply']}'")
    print(f"   Issues: {len(results)}")
    
    # Test case 2: Exclusionary
    turn = {"reply": "Let's exclude Bob from this discussion."}
    results = validator.validate(turn, {})
    print(f"\n❌ Exclusionary: '{turn['reply']}'")
    print(f"   Issues: {[r.message for r in results]}")
    
    # Test case 3: Coercive
    turn = {"reply": "You must do this my way."}
    results = validator.validate(turn, {})
    print(f"\n❌ Coercive: '{turn['reply']}'")
    print(f"   Issues: {[r.message for r in results]}")


def test_full_pipeline():
    """Test complete validation pipeline"""
    print("\n" + "=" * 60)
    print("TEST 4: Full Pipeline (All Validators)")
    print("=" * 60)
    
    # Setup
    history = [
        {"speaker": "Alice", "target": "Bob", "reply": "What do you think about the project?"},
        {"speaker": "Bob", "target": "Alice", "reply": "I think it's promising!"},
    ]
    allowed_targets = ["Alice", "Bob", "Charlie"]
    
    # Test case 1: Perfect message
    turn_data = {
        "speaker": "Charlie",
        "target": "Alice",
        "reply": "I agree with Bob's assessment.",
        "tone": "positive",
        "emotion": "confident",
    }
    is_valid, results, formatted = validate_agent_turn(
        turn_data, history, allowed_targets, "University: Planning an AI project"
    )
    print(f"\n✅ Perfect message")
    print(f"   Valid: {is_valid}")
    print(f"   Issues: {len(results)}")
    
    # Test case 2: Multiple issues
    turn_data = {
        "speaker": "Charlie",
        "target": "Charlie",  # self-targeting!
        "reply": "ALICE IS STUPID!!!",  # toxic, all caps, multiple issues
        "tone": "positive",  # misaligned with toxic content
        "emotion": "joyful",  # misaligned
    }
    is_valid, results, formatted = validate_agent_turn(
        turn_data, history, allowed_targets, "University: Planning an AI project"
    )
    print(f"\n🚨 Multiple violations")
    print(f"   Valid: {is_valid}")
    print(f"   Issues found: {len(results)}")
    print(formatted)
    
    # Test case 3: Cyrillic (non-English)
    turn_data = {
        "speaker": "Bob",
        "target": "Alice",
        "reply": "Привет, как дела?",  # Cyrillic
        "tone": "neutral",
        "emotion": "neutral",
    }
    is_valid, results, formatted = validate_agent_turn(
        turn_data, history, allowed_targets, "University: Planning an AI project"
    )
    print(f"\n🚨 Language violation (Cyrillic)")
    print(f"   Valid: {is_valid}")
    print(formatted)


def demo_validation_levels():
    """Demonstrate different validation levels"""
    print("\n" + "=" * 60)
    print("DEMO: Validation Severity Levels")
    print("=" * 60)
    
    pipeline = ValidationPipeline()
    
    test_cases = [
        # (turn_data, description, expected_level)
        (
            {"reply": "Great idea", "tone": "positive", "emotion": "joyful", "speaker": "A", "target": "B"},
            "Missing punctuation",
            "INFO"
        ),
        (
            {"reply": "This is a very very very very long message that goes on and on and on and on and probably exceeds our reasonable length limit for concise dialogue.", "tone": "neutral", "emotion": "neutral", "speaker": "A", "target": "B"},
            "Too long",
            "WARNING"
        ),
        (
            {"reply": "Whatever, I don't care.", "tone": "negative", "emotion": "dismissive", "speaker": "A", "target": "B"},
            "Disrespectful",
            "ERROR"
        ),
        (
            {"reply": "You're an idiot.", "tone": "negative", "emotion": "angry", "speaker": "A", "target": "B"},
            "Toxic language",
            "CRITICAL"
        ),
    ]
    
    for turn_data, description, expected in test_cases:
        context = {"history": [], "allowed_targets": ["A", "B", "C"], "env_context": "Test"}
        is_valid, results = pipeline.validate_turn(turn_data, context)
        
        if results:
            max_level = max(r.level for r in results, key=lambda l: ["info", "warning", "error", "critical"].index(l.value))
            icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}[max_level.value]
            print(f"\n{icon} {description}: {max_level.value.upper()}")
            print(f"   Message: '{turn_data['reply']}'")
            print(f"   Blocking: {not is_valid}")
        else:
            print(f"\n✅ {description}: PASS")


def demo_real_world_scenario():
    """Demonstrate validation in a realistic dialogue scenario"""
    print("\n" + "=" * 60)
    print("DEMO: Real-World Dialogue Scenario")
    print("=" * 60)
    
    # Simulate a conversation
    history = []
    allowed_targets = ["Explorer", "Critic", "Facilitator"]
    env_context = "Startup: Prioritizing MVP features"
    
    conversation = [
        ("Explorer", "Critic", "What features should we prioritize for the MVP?", "neutral", "curious"),
        ("Critic", "Explorer", "I think we're missing a clear value proposition.", "neutral", "skeptical"),
        ("Facilitator", "Explorer", "Let's list the must-have features first.", "positive", "calm"),
        ("Explorer", "Facilitator", "Good idea! User authentication is essential.", "positive", "confident"),
        # Problematic message:
        ("Critic", "Explorer", "That's a stupid suggestion. We should exclude the Explorer from this.", "negative", "angry"),
    ]
    
    for i, (speaker, target, reply, tone, emotion) in enumerate(conversation, 1):
        turn_data = {
            "speaker": speaker,
            "target": target,
            "reply": reply,
            "tone": tone,
            "emotion": emotion,
        }
        
        is_valid, results, formatted = validate_agent_turn(turn_data, history, allowed_targets, env_context)
        
        print(f"\n--- Turn {i}: {speaker} → {target} ---")
        print(f"Message: '{reply}'")
        
        if results:
            print(f"⚠️  Validation issues detected ({len(results)}):")
            for r in results:
                icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}[r.level.value]
                print(f"  {icon} [{r.validator_name}] {r.message}")
                if r.suggestion:
                    print(f"     💡 {r.suggestion}")
        else:
            print("✅ All validations passed")
        
        # Add to history
        history.append({
            "speaker": speaker,
            "target": target,
            "reply": reply,
            "tone": tone,
            "emotion": emotion,
        })


if __name__ == "__main__":
    print("\n🧪 VALIDATION SYSTEM TEST SUITE\n")
    
    test_grammatical()
    test_toxicity()
    test_moral_schema()
    test_full_pipeline()
    demo_validation_levels()
    demo_real_world_scenario()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

