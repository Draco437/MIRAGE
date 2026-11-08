"""
adaptation_rules.py - turns an engagement label into scene settings.

This is plain, readable logic, not machine learning: a therapist can read it,
question it and change it. The ML model decides what state the child seems to
be in; these rules decide what to do about it.

Design choices for sensory-sensitive children:
  * Calm down quickly. Stress reacts almost immediately.
  * Make things harder slowly. Difficulty only rises after a sustained
    engaged stretch and a long enough gap since the last change.
  * Never flap. A cooldown stops settings from changing every few seconds.

Pure Python: no Django, no database.
"""
from dataclasses import dataclass

# What Unity applies. npc_prompt is a one-shot nudge, not a persistent setting.
DEFAULT_SETTINGS = {
    "light_intensity": 0.8,
    "volume": 0.6,
    "npc_count": 2,
    "difficulty": 2,
    "npc_prompt": None,
}

# (min, max) for each numeric setting. Also used to validate therapist overrides.
LIMITS = {
    "light_intensity": (0.3, 1.0),
    "volume": (0.1, 1.0),
    "npc_count": (0, 3),
    "difficulty": (1, 3),
}

NPC_PROMPTS = ("call_name", "wave")

STRESS_CONFIDENCE_MIN = 0.5
STRESS_COOLDOWN_S = 8       # calming may repeat fairly quickly
NUDGE_COOLDOWN_S = 15       # distraction nudges
INCREASE_COOLDOWN_S = 60    # difficulty may rise at most once a minute
DISTRACTED_WINDOWS_FOR_NUDGE = 2
DISTRACTED_WINDOWS_FOR_EASIER = 3
ENGAGED_WINDOWS_FOR_HARDER = 3


@dataclass
class Decision:
    settings: dict
    changed: bool
    reason: str


def clamp_settings(settings):
    """Return a copy with every value inside LIMITS and the right type."""
    out = dict(settings)
    for key, (low, high) in LIMITS.items():
        if key in out:
            value = min(max(out[key], low), high)
            out[key] = int(round(value)) if isinstance(low, int) else round(float(value), 2)
    return out


def merge_settings(base, partial):
    """Apply a partial update (e.g. a therapist override) on top of base."""
    merged = dict(base)
    merged.update(partial)
    return clamp_settings(merged)


def _last_n_all(labels, n, wanted):
    return len(labels) >= n and all(lab == wanted for lab in labels[-n:])


def decide(label, confidence, recent_labels, prev_settings, seconds_since_last_change):
    """
    label:                      newest engagement label
    recent_labels:              last few labels, oldest first, newest included
    prev_settings:              settings currently active in the scene
    seconds_since_last_change:  time since any settings change (auto or therapist)
    """
    settings = clamp_settings({**DEFAULT_SETTINGS, **prev_settings})
    settings["npc_prompt"] = None
    unchanged = Decision(settings, False, "no change needed")

    if label == "stressed" and confidence >= STRESS_CONFIDENCE_MIN:
        if seconds_since_last_change < STRESS_COOLDOWN_S:
            return Decision(settings, False, "stress detected; waiting for cooldown")
        new = dict(settings)
        new["light_intensity"] -= 0.2
        new["volume"] -= 0.2
        new["npc_count"] -= 1
        new["difficulty"] -= 1
        new = clamp_settings(new)
        if new == settings:
            return Decision(settings, False, "stress detected; already at calmest settings")
        return Decision(new, True, "stress detected: lowered light, volume, NPCs and difficulty")

    if label == "distracted":
        if not _last_n_all(recent_labels, DISTRACTED_WINDOWS_FOR_NUDGE, "distracted"):
            return unchanged
        if seconds_since_last_change < NUDGE_COOLDOWN_S:
            return Decision(settings, False, "distraction detected; waiting for cooldown")
        new = dict(settings)
        new["npc_prompt"] = "call_name"
        reason = "sustained distraction: NPC calls the child's name"
        if _last_n_all(recent_labels, DISTRACTED_WINDOWS_FOR_EASIER, "distracted"):
            new["difficulty"] -= 1
            reason += " and the task gets easier"
        return Decision(clamp_settings(new), True, reason)

    if label == "engaged":
        if not _last_n_all(recent_labels, ENGAGED_WINDOWS_FOR_HARDER, "engaged"):
            return unchanged
        if seconds_since_last_change < INCREASE_COOLDOWN_S:
            return unchanged
        if settings["difficulty"] >= LIMITS["difficulty"][1]:
            return unchanged
        new = dict(settings)
        new["difficulty"] += 1
        return Decision(clamp_settings(new), True, "sustained engagement: difficulty raised one step")

    return unchanged