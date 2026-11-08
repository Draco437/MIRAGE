"""
auth.py - authenticates WebSocket connections.

Browsers and Unity can't send an Authorization header when opening a
WebSocket, so the token travels in the URL:  ws://host/ws/...?token=<token>

Two kinds of token are accepted:
  * "jwt"   - the therapist's normal login access token (used by the dashboard)
  * "unity" - a signed token made when a session is created. It is valid for
              that one session only, so the VR headset never needs the
              therapist's password or login token.

The middleware puts the result in scope["auth"], which consumers check.
No database access happens here.
"""
from urllib.parse import parse_qs

from django.conf import settings
from django.core import signing
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import AccessToken

UNITY_TOKEN_SALT = "mirage.unity-session"


def make_unity_token(session_id, user_id):
    return signing.dumps({"sid": session_id, "uid": user_id}, salt=UNITY_TOKEN_SALT)


def _read_unity_token(token):
    try:
        return signing.loads(
            token, salt=UNITY_TOKEN_SALT, max_age=settings.MIRAGE["UNITY_TOKEN_MAX_AGE"]
        )
    except signing.BadSignature:  # also covers expired tokens
        return None


def authenticate_token(token):
    """Return {"kind", "user_id", "session_id"} or None if the token is not valid."""
    if not token:
        return None
    try:
        access = AccessToken(token)  # checks signature, expiry and that it is an access token
        return {"kind": "jwt", "user_id": int(access["user_id"]), "session_id": None}
    except TokenError:
        pass
    except (KeyError, ValueError):
        return None
    data = _read_unity_token(token)
    if data:
        return {"kind": "unity", "user_id": data["uid"], "session_id": data["sid"]}
    return None


class TokenAuthMiddleware:
    """ASGI middleware: reads ?token= and stores the result in scope["auth"]."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        scope = dict(scope)
        query = parse_qs(scope.get("query_string", b"").decode())
        token = (query.get("token") or [None])[0]
        scope["auth"] = authenticate_token(token)
        return await self.inner(scope, receive, send)