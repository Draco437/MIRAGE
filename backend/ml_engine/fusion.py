"""
fusion.py - combines the four sensor signals into one decision.

No dataset contains all four kinds of data for the same child, so there is no way
to TRAIN one model on all of them together. Instead each sensor model is trained
on its own dataset, and this file combines their outputs with explicit, readable
rules. Every decision lists the evidence behind it, so a therapist can see why.

Cases, checked in this order:
  1. STRESSED     face distress is strong, OR distress and body agitation agree.
                  Agitation on its own is never read as stress (children fidget for many reasons).
  2. UNCERTAIN    some distress but nothing backing it up: do not act on a guess.
  3. DISTRACTED   attention is low and the child is not responding.
  4. ENGAGED      attending, or responding.
  5. UNCERTAIN    anything else, or fewer than two trustworthy signals.

"uncertain" makes adaptation_rules.decide() change nothing, which is the safe default.

Pure Python: no Django, no database.
"""
from dataclasses import dataclass, field

from .signals import SIGNAL_NAMES

MIN_SIGNAL_CONFIDENCE = 0.4   # a signal below this is ignored
MIN_SIGNALS = 2               # need at least this many usable signals to decide anything

# How much each signal counts toward "coverage" (how complete the picture is).
WEIGHTS = {"attention": 0.35, "distress": 0.30, "arousal": 0.15, "responsiveness": 0.20}

STRESS_WEIGHTS = {"distress": 0.65, "arousal": 0.35}
STRESS_THRESHOLD = 0.6        # combined stress score needed
STRESS_ALONE = 0.8            # distress this strong is enough without a second sign
SOME_DISTRESS = 0.5
LOW_ATTENTION = 0.3
LOW_RESPONSIVENESS = 0.3
ENGAGED_THRESHOLD = 0.5


@dataclass
class FusionResult:
    label: str                       # engaged | distracted | stressed | uncertain
    confidence: float                # 0..1
    used: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    evidence: list = field(default_factory=list)


def _describe(signal):
    return f"{signal.name} {signal.value:.2f} ({signal.source}, confidence {signal.confidence:.2f})"


def fuse(signals):
    """signals: {name: Signal}. Missing sensors are simply absent from the dict."""
    usable = {
        name: sig
        for name, sig in signals.items()
        if sig is not None and sig.confidence >= MIN_SIGNAL_CONFIDENCE
    }
    used = [n for n in SIGNAL_NAMES if n in usable]
    missing = [n for n in SIGNAL_NAMES if n not in usable]

    if len(usable) < MIN_SIGNALS:
        return FusionResult(
            "uncertain", 0.0, used, missing,
            [f"only {len(usable)} trustworthy signal(s); at least {MIN_SIGNALS} are needed"],
        )

    coverage = sum(WEIGHTS[n] for n in used)      # 0..1, how complete the picture is
    scale = 0.5 + 0.5 * coverage                  # fewer sensors -> less confident

    # 1. Stress
    stress_parts = {n: usable[n] for n in STRESS_WEIGHTS if n in usable}
    if stress_parts:
        total_weight = sum(STRESS_WEIGHTS[n] for n in stress_parts)
        stress = sum(STRESS_WEIGHTS[n] * s.value for n, s in stress_parts.items()) / total_weight
        distress = stress_parts.get("distress")
        strong_alone = distress is not None and distress.value >= STRESS_ALONE
        if stress >= STRESS_THRESHOLD and (len(stress_parts) >= 2 or strong_alone):
            return FusionResult(
                "stressed", round(stress * scale, 3), used, missing,
                [_describe(s) for s in stress_parts.values()] + [f"combined stress score {stress:.2f}"],
            )

    # 2. Some distress that nothing backs up: do not guess.
    distress = usable.get("distress")
    if distress is not None and distress.value >= SOME_DISTRESS:
        return FusionResult(
            "uncertain", 0.0, used, missing,
            [_describe(distress), "some distress, but not confirmed by a second signal"],
        )

    attention = usable.get("attention")
    responsiveness = usable.get("responsiveness")

    # 3. Distracted
    if attention is not None and attention.value < LOW_ATTENTION and (
        responsiveness is None or responsiveness.value < LOW_RESPONSIVENESS
    ):
        evidence = [_describe(attention)]
        if responsiveness is not None:
            evidence.append(_describe(responsiveness))
        return FusionResult("distracted", round((1.0 - attention.value) * scale, 3), used, missing, evidence)

    # 4. Engaged
    best = max(
        (s for s in (attention, responsiveness) if s is not None),
        key=lambda s: s.value,
        default=None,
    )
    if best is not None and best.value >= ENGAGED_THRESHOLD:
        return FusionResult("engaged", round(best.value * scale, 3), used, missing, [_describe(best)])

    # 5. Anything else
    return FusionResult(
        "uncertain", 0.0, used, missing,
        ["signals are in the middle or disagree; no confident call"],
    )