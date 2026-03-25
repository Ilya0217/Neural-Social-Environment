from dataclasses import dataclass, field
from typing import Dict, Optional
from .prompts import build_system_prompt


@dataclass
class BigFiveProfile:
    """
    Big Five personality traits (OCEAN model).
    Each trait is a float from 0.0 (very low) to 1.0 (very high).

    Reference: Goldberg, L. R. (1990). An alternative description of personality.
    Journal of Personality and Social Psychology, 59(6), 1216-1229.
    See also: Woolley, A. W., et al. (2010). Evidence for a collective intelligence factor.
    Science, 330(6004), 686-688.
    """
    openness: float = 0.5         # Curiosity, creativity, openness to experience
    conscientiousness: float = 0.5  # Organization, dependability, self-discipline
    extraversion: float = 0.5     # Sociability, assertiveness, positive emotions
    agreeableness: float = 0.5    # Cooperation, trust, altruism
    neuroticism: float = 0.5      # Emotional instability, anxiety, moodiness

    def to_prompt_description(self) -> str:
        """Convert Big Five scores to natural language personality description."""
        traits = []

        # Openness
        if self.openness > 0.7:
            traits.append("highly creative and open to new ideas, loves exploring unconventional approaches")
        elif self.openness < 0.3:
            traits.append("practical and conventional, prefers tried-and-true methods")

        # Conscientiousness
        if self.conscientiousness > 0.7:
            traits.append("very organized and detail-oriented, follows through on commitments")
        elif self.conscientiousness < 0.3:
            traits.append("spontaneous and flexible, sometimes disorganized but adaptable")

        # Extraversion
        if self.extraversion > 0.7:
            traits.append("outgoing and talkative, energized by group discussions, speaks up readily")
        elif self.extraversion < 0.3:
            traits.append("reserved and thoughtful, prefers to listen before speaking, speaks concisely")

        # Agreeableness
        if self.agreeableness > 0.7:
            traits.append("very cooperative and supportive, avoids conflict, seeks harmony")
        elif self.agreeableness < 0.3:
            traits.append("direct and challenging, not afraid to disagree, prioritizes truth over harmony")

        # Neuroticism
        if self.neuroticism > 0.7:
            traits.append("emotionally reactive, tends to worry, sensitive to criticism and stress")
        elif self.neuroticism < 0.3:
            traits.append("emotionally stable and calm under pressure, rarely gets upset")

        if not traits:
            return "balanced personality with moderate traits across all dimensions"

        return "; ".join(traits)

    def to_dict(self) -> Dict[str, float]:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }


# Predefined Big Five profiles for common agent roles
BIG_FIVE_PRESETS: Dict[str, BigFiveProfile] = {
    "explorer": BigFiveProfile(
        openness=0.9, conscientiousness=0.4, extraversion=0.7,
        agreeableness=0.6, neuroticism=0.3,
    ),
    "critic": BigFiveProfile(
        openness=0.5, conscientiousness=0.8, extraversion=0.4,
        agreeableness=0.3, neuroticism=0.5,
    ),
    "facilitator": BigFiveProfile(
        openness=0.6, conscientiousness=0.7, extraversion=0.8,
        agreeableness=0.9, neuroticism=0.2,
    ),
    "scientist": BigFiveProfile(
        openness=0.85, conscientiousness=0.9, extraversion=0.3,
        agreeableness=0.5, neuroticism=0.4,
    ),
    "manager": BigFiveProfile(
        openness=0.5, conscientiousness=0.85, extraversion=0.7,
        agreeableness=0.6, neuroticism=0.35,
    ),
    "mediator": BigFiveProfile(
        openness=0.7, conscientiousness=0.6, extraversion=0.6,
        agreeableness=0.95, neuroticism=0.2,
    ),
    "innovator": BigFiveProfile(
        openness=0.95, conscientiousness=0.3, extraversion=0.8,
        agreeableness=0.5, neuroticism=0.4,
    ),
    "analyst": BigFiveProfile(
        openness=0.6, conscientiousness=0.9, extraversion=0.3,
        agreeableness=0.4, neuroticism=0.5,
    ),
}


def generate_random_big_five(seed_str: str = "") -> BigFiveProfile:
    """Generate a pseudo-random but deterministic Big Five profile based on role/name string.
    This ensures the same role always gets the same profile, but different roles differ."""
    import hashlib
    h = hashlib.md5(seed_str.lower().encode()).hexdigest()
    # Extract 5 values from hash, map to 0.2-0.9 range (avoid extremes)
    vals = [int(h[i*2:i*2+2], 16) / 255 * 0.7 + 0.2 for i in range(5)]
    return BigFiveProfile(
        openness=round(vals[0], 2),
        conscientiousness=round(vals[1], 2),
        extraversion=round(vals[2], 2),
        agreeableness=round(vals[3], 2),
        neuroticism=round(vals[4], 2),
    )


@dataclass
class Agent:
    name: str
    nature: str
    color: str
    is_human: bool = False
    persona: str = ""
    big_five: Optional[BigFiveProfile] = None

    def __post_init__(self):
        # Auto-assign Big Five profile from presets or generate deterministically
        if self.big_five is None:
            preset = BIG_FIVE_PRESETS.get(self.nature.lower())
            self.big_five = preset if preset else generate_random_big_five(self.nature)

    @property
    def system_prompt(self) -> str:
        # Append Big Five personality description to persona
        persona = self.persona or ""
        if self.big_five:
            big_five_desc = self.big_five.to_prompt_description()
            persona = (
                f"{persona}\n\n"
                f"=== Personality Profile (Big Five / OCEAN) ===\n"
                f"Openness: {self.big_five.openness:.1f} | "
                f"Conscientiousness: {self.big_five.conscientiousness:.1f} | "
                f"Extraversion: {self.big_five.extraversion:.1f} | "
                f"Agreeableness: {self.big_five.agreeableness:.1f} | "
                f"Neuroticism: {self.big_five.neuroticism:.1f}\n"
                f"Your personality: {big_five_desc}\n"
                f"Let these traits influence HOW you speak and react naturally — "
                f"don't announce them, just embody them.\n"
            ).strip()
        return build_system_prompt(self.nature, persona)
