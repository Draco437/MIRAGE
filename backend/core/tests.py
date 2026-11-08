"""
Tests that need no database or running services.
Run from backend/:   python manage.py test core
"""
from django.test import SimpleTestCase

from . import auth
from .serializers import (
    FramePacketSerializer,
    GazePacketSerializer,
    SettingsSerializer,
)


class PacketValidationTests(SimpleTestCase):
    def test_valid_gaze_packet(self):
        s = GazePacketSerializer(data={"type": "gaze_data", "t": 1.5, "target": "eyes"})
        self.assertTrue(s.is_valid(), s.errors)

    def test_unknown_gaze_target_is_rejected(self):
        s = GazePacketSerializer(data={"t": 1.5, "target": "ceiling"})
        self.assertFalse(s.is_valid())

    def test_oversized_frame_is_rejected(self):
        s = FramePacketSerializer(data={"t": 1.0, "image": "A" * 500_000})
        self.assertFalse(s.is_valid())

    def test_override_values_must_be_in_range(self):
        self.assertTrue(SettingsSerializer(data={"volume": 0.5, "difficulty": 2}).is_valid())
        self.assertFalse(SettingsSerializer(data={"volume": 5}).is_valid())
        self.assertFalse(SettingsSerializer(data={"npc_prompt": "dance"}).is_valid())


class TokenTests(SimpleTestCase):
    def test_unity_token_round_trip(self):
        token = auth.make_unity_token("a" * 24, 7)
        result = auth.authenticate_token(token)
        self.assertEqual(result, {"kind": "unity", "user_id": 7, "session_id": "a" * 24})

    def test_garbage_and_empty_tokens_are_rejected(self):
        self.assertIsNone(auth.authenticate_token(""))
        self.assertIsNone(auth.authenticate_token("not-a-token"))
        self.assertIsNone(auth.authenticate_token(None))

    def test_tampered_unity_token_is_rejected(self):
        token = auth.make_unity_token("a" * 24, 7)
        self.assertIsNone(auth.authenticate_token(token[:-2] + "xx"))