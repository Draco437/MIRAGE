"""
Tests for how the four sensor signals are combined.
Run from backend/:   python -m unittest ml_engine.test_fusion -v
"""
import unittest

from ml_engine.fusion import fuse
from ml_engine.signals import Signal


def sig(name, value, confidence=0.9):
    return Signal(name, value, confidence, "test")


def bundle(**values):
    return {name: sig(name, v) for name, v in values.items()}


class SignalTests(unittest.TestCase):
    def test_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            Signal("attention", 1.5, 0.9, "x")
        with self.assertRaises(ValueError):
            Signal("attention", 0.5, -0.1, "x")
        with self.assertRaises(ValueError):
            Signal("mood", 0.5, 0.9, "x")


class StressTests(unittest.TestCase):
    def test_distress_and_agitation_together_mean_stressed(self):
        r = fuse(bundle(distress=0.7, arousal=0.8, attention=0.6))
        self.assertEqual(r.label, "stressed")
        self.assertGreater(r.confidence, 0.4)

    def test_agitation_alone_is_never_stress(self):
        r = fuse(bundle(arousal=0.95, attention=0.7))
        self.assertEqual(r.label, "engaged")

    def test_very_strong_distress_alone_is_enough(self):
        r = fuse(bundle(distress=0.9, attention=0.6))
        self.assertEqual(r.label, "stressed")

    def test_moderate_distress_without_backup_is_uncertain(self):
        r = fuse(bundle(distress=0.65, attention=0.8))
        self.assertEqual(r.label, "uncertain")

    def test_calm_face_with_high_agitation_is_not_stressed(self):
        r = fuse(bundle(distress=0.1, arousal=0.9, attention=0.7))
        self.assertNotEqual(r.label, "stressed")


class AttentionTests(unittest.TestCase):
    def test_low_attention_and_no_response_is_distracted(self):
        r = fuse(bundle(attention=0.1, arousal=0.2))
        self.assertEqual(r.label, "distracted")

    def test_looking_away_but_responding_is_engaged(self):
        r = fuse(bundle(attention=0.1, responsiveness=0.8))
        self.assertEqual(r.label, "engaged")

    def test_attending_is_engaged(self):
        r = fuse(bundle(attention=0.85, distress=0.1))
        self.assertEqual(r.label, "engaged")

    def test_middle_values_are_uncertain(self):
        r = fuse(bundle(attention=0.4, arousal=0.4))
        self.assertEqual(r.label, "uncertain")


class SafetyTests(unittest.TestCase):
    def test_one_signal_is_not_enough(self):
        r = fuse(bundle(attention=0.9))
        self.assertEqual(r.label, "uncertain")
        self.assertEqual(r.confidence, 0.0)

    def test_untrustworthy_signals_are_ignored(self):
        signals = {"attention": sig("attention", 0.9), "distress": sig("distress", 0.95, confidence=0.2)}
        r = fuse(signals)
        self.assertEqual(r.label, "uncertain")          # only one trustworthy signal left
        self.assertIn("distress", r.missing)

    def test_more_sensors_means_more_confidence(self):
        few = fuse(bundle(attention=0.8, arousal=0.2))
        many = fuse(bundle(attention=0.8, arousal=0.2, distress=0.1, responsiveness=0.7))
        self.assertEqual(few.label, "engaged")
        self.assertEqual(many.label, "engaged")
        self.assertGreater(many.confidence, few.confidence)

    def test_every_decision_explains_itself(self):
        r = fuse(bundle(distress=0.7, arousal=0.8))
        self.assertTrue(r.evidence)
        self.assertEqual(set(r.used), {"distress", "arousal"})
        self.assertEqual(set(r.missing), {"attention", "responsiveness"})

    def test_uncertain_leaves_the_scene_alone(self):
        from ml_engine import adaptation_rules as rules

        d = rules.decide("uncertain", 0.0, ["uncertain"], rules.DEFAULT_SETTINGS, 100)
        self.assertFalse(d.changed)


if __name__ == "__main__":
    unittest.main()