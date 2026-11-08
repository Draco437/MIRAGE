"""
Tests for the pure ML logic (no database, no network needed).
Run from backend/:   python -m unittest ml_engine.tests -v
"""
import unittest
from datetime import datetime, timedelta, timezone

from ml_engine import adaptation_rules as rules
from ml_engine import engagement_model
from ml_engine.features import FEATURE_NAMES, build_features, to_vector

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def gaze(targets):
    return [{"target": t, "received_at": T0 + timedelta(seconds=i / 10)} for i, t in enumerate(targets)]


class FeatureTests(unittest.TestCase):
    def test_rates_and_shifts(self):
        feats = build_features(gaze(["eyes", "eyes", "face", "away"]), [], [], [], window_seconds=10)
        self.assertAlmostEqual(feats["eye_contact_rate"], 0.5)
        self.assertAlmostEqual(feats["face_rate"], 0.75)
        self.assertAlmostEqual(feats["gaze_shift_rate"], 0.2)  # eyes->face, face->away

    def test_missing_sources_use_defaults_and_flags(self):
        feats = build_features(gaze(["eyes"] * 10), [], [], [], window_seconds=10)
        self.assertEqual(feats["has_emotion"], 0.0)
        self.assertEqual(feats["has_response"], 0.0)
        self.assertAlmostEqual(feats["p_happy"], 0.25)
        self.assertEqual(set(feats), set(FEATURE_NAMES))
        self.assertEqual(len(to_vector(feats)), len(FEATURE_NAMES))

    def test_emotion_probabilities_are_averaged(self):
        emotions = [
            {"probs": {"happy": 1.0, "neutral": 0.0, "sad": 0.0, "distressed": 0.0}},
            {"probs": {"happy": 0.0, "neutral": 0.0, "sad": 0.0, "distressed": 1.0}},
        ]
        feats = build_features(gaze(["eyes"] * 10), [], [], emotions, window_seconds=10)
        self.assertAlmostEqual(feats["p_happy"], 0.5)
        self.assertAlmostEqual(feats["p_distressed"], 0.5)
        self.assertEqual(feats["has_emotion"], 1.0)


class RuleTests(unittest.TestCase):
    def test_stress_calms_the_scene(self):
        d = rules.decide("stressed", 0.9, ["stressed"], rules.DEFAULT_SETTINGS, 100)
        self.assertTrue(d.changed)
        self.assertLess(d.settings["light_intensity"], rules.DEFAULT_SETTINGS["light_intensity"])
        self.assertLess(d.settings["volume"], rules.DEFAULT_SETTINGS["volume"])
        self.assertEqual(d.settings["npc_count"], 1)
        self.assertEqual(d.settings["difficulty"], 1)

    def test_stress_stops_at_the_floor(self):
        floor = {"light_intensity": 0.3, "volume": 0.1, "npc_count": 0, "difficulty": 1}
        d = rules.decide("stressed", 0.9, ["stressed"], floor, 100)
        self.assertFalse(d.changed)

    def test_low_confidence_stress_is_ignored(self):
        d = rules.decide("stressed", 0.3, ["stressed"], rules.DEFAULT_SETTINGS, 100)
        self.assertFalse(d.changed)

    def test_cooldown_blocks_repeated_stress_changes(self):
        d = rules.decide("stressed", 0.9, ["stressed"], rules.DEFAULT_SETTINGS, 2)
        self.assertFalse(d.changed)

    def test_one_distracted_window_does_nothing_two_nudge(self):
        one = rules.decide("distracted", 0.8, ["engaged", "distracted"], rules.DEFAULT_SETTINGS, 100)
        two = rules.decide("distracted", 0.8, ["distracted", "distracted"], rules.DEFAULT_SETTINGS, 100)
        self.assertFalse(one.changed)
        self.assertTrue(two.changed)
        self.assertEqual(two.settings["npc_prompt"], "call_name")
        self.assertEqual(two.settings["difficulty"], 2)  # not easier yet

    def test_three_distracted_windows_also_make_it_easier(self):
        d = rules.decide("distracted", 0.8, ["distracted"] * 3, rules.DEFAULT_SETTINGS, 100)
        self.assertEqual(d.settings["difficulty"], 1)

    def test_difficulty_rises_slowly_and_caps(self):
        engaged = ["engaged"] * 3
        too_soon = rules.decide("engaged", 0.9, engaged, rules.DEFAULT_SETTINGS, 30)
        ok = rules.decide("engaged", 0.9, engaged, rules.DEFAULT_SETTINGS, 120)
        capped = rules.decide("engaged", 0.9, engaged, {**rules.DEFAULT_SETTINGS, "difficulty": 3}, 120)
        self.assertFalse(too_soon.changed)
        self.assertEqual(ok.settings["difficulty"], 3)
        self.assertFalse(capped.changed)

    def test_override_values_are_clamped(self):
        merged = rules.merge_settings(rules.DEFAULT_SETTINGS, {"volume": 5.0, "npc_count": -2})
        self.assertEqual(merged["volume"], 1.0)
        self.assertEqual(merged["npc_count"], 0)

    def test_prompt_is_one_shot(self):
        prev = {**rules.DEFAULT_SETTINGS, "npc_prompt": "call_name"}
        d = rules.decide("engaged", 0.9, ["engaged"], prev, 100)
        self.assertIsNone(d.settings["npc_prompt"])


class HeuristicTests(unittest.TestCase):
    def feats(self, **over):
        base = build_features(gaze(["eyes"] * 10), [], [], [], window_seconds=10)
        base.update(over)
        return base

    def test_looks_at_face_is_engaged(self):
        r = engagement_model.heuristic_predict(self.feats(face_rate=0.9, eye_contact_rate=0.6))
        self.assertEqual(r["label"], "engaged")

    def test_looking_away_is_distracted(self):
        r = engagement_model.heuristic_predict(self.feats(face_rate=0.1, eye_contact_rate=0.0))
        self.assertEqual(r["label"], "distracted")

    def test_distressed_face_and_fidgeting_is_stressed(self):
        r = engagement_model.heuristic_predict(
            self.feats(face_rate=0.7, eye_contact_rate=0.3, p_distressed=0.85, has_emotion=1.0,
                       hand_jitter_mean=0.3)
        )
        self.assertEqual(r["label"], "stressed")
        self.assertEqual(r["source"], "heuristic")


if __name__ == "__main__":
    unittest.main()