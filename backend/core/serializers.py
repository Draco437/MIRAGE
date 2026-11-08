"""
serializers.py - validates everything that enters the backend.

Two groups:
  * REST request bodies (register, child, session)
  * WebSocket packets from Unity and the dashboard

Nothing reaches MongoDB or the ML code without passing through one of these,
so a buggy or malicious client can't store malformed data.
"""
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password as run_password_validators
from rest_framework import serializers

from ml_engine.adaptation_rules import LIMITS, NPC_PROMPTS
from ml_engine.features import GAZE_TARGETS

User = get_user_model()

SCENARIOS = ("greeting", "turn_taking")
SESSION_EVENTS = ("scene_loaded", "prompt_shown", "prompt_done", "pause", "resume")
OBJECT_ID = r"^[0-9a-f]{24}$"


# --- REST ---------------------------------------------------------------
class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True, min_length=8, max_length=128)

    def validate_username(self, value):
        if User.objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError("That username is already taken.")
        return value

    def validate_password(self, value):
        run_password_validators(value)
        return value

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data["username"], password=validated_data["password"]
        )


class ChildSerializer(serializers.Serializer):
    # A nickname or code is enough; avoid storing a real full name.
    name = serializers.CharField(max_length=80)
    age = serializers.IntegerField(min_value=2, max_value=18)
    notes = serializers.CharField(max_length=1000, required=False, allow_blank=True, default="")


class SessionCreateSerializer(serializers.Serializer):
    child_id = serializers.RegexField(OBJECT_ID)
    scenario = serializers.ChoiceField(choices=SCENARIOS, default="greeting")


class SettingsSerializer(serializers.Serializer):
    """Scene settings. Used for therapist overrides; every field is optional."""

    light_intensity = serializers.FloatField(
        required=False, min_value=LIMITS["light_intensity"][0], max_value=LIMITS["light_intensity"][1]
    )
    volume = serializers.FloatField(
        required=False, min_value=LIMITS["volume"][0], max_value=LIMITS["volume"][1]
    )
    npc_count = serializers.IntegerField(
        required=False, min_value=LIMITS["npc_count"][0], max_value=LIMITS["npc_count"][1]
    )
    difficulty = serializers.IntegerField(
        required=False, min_value=LIMITS["difficulty"][0], max_value=LIMITS["difficulty"][1]
    )
    npc_prompt = serializers.ChoiceField(choices=NPC_PROMPTS, required=False, allow_null=True)


# --- WebSocket packets from Unity ---------------------------------------
class GazePacketSerializer(serializers.Serializer):
    t = serializers.FloatField(min_value=0)  # Unity's Time.time in seconds
    target = serializers.ChoiceField(choices=GAZE_TARGETS)


class MotorPacketSerializer(serializers.Serializer):
    t = serializers.FloatField(min_value=0)
    hand_speed = serializers.FloatField(min_value=0)  # m/s
    jitter = serializers.FloatField(min_value=0)      # std-dev of hand velocity, m/s


class ResponsePacketSerializer(serializers.Serializer):
    t = serializers.FloatField(min_value=0)
    prompt_id = serializers.CharField(max_length=64)
    response_time_ms = serializers.FloatField(min_value=0, max_value=120_000)
    correct = serializers.BooleanField()


class FramePacketSerializer(serializers.Serializer):
    t = serializers.FloatField(min_value=0)
    image = serializers.CharField(trim_whitespace=False)  # base64 JPEG

    def validate_image(self, value):
        if len(value) > settings.MIRAGE["MAX_FRAME_BASE64_CHARS"]:
            raise serializers.ValidationError("Frame is too large; send a smaller JPEG.")
        return value


class SessionEventPacketSerializer(serializers.Serializer):
    t = serializers.FloatField(min_value=0)
    event = serializers.ChoiceField(choices=SESSION_EVENTS)
    detail = serializers.CharField(max_length=200, required=False, allow_blank=True, default="")


# Maps the "type" field of an incoming packet to its validator.
PACKET_SERIALIZERS = {
    "gaze_data": GazePacketSerializer,
    "motor_data": MotorPacketSerializer,
    "response_event": ResponsePacketSerializer,
    "camera_frame": FramePacketSerializer,
    "session_event": SessionEventPacketSerializer,
}