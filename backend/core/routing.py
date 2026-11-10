"""WebSocket URL table (the WebSocket equivalent of urls.py)."""
from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    # The VR client (Unity) streams sensor data here and receives scene changes.
    re_path(r"^ws/session/(?P<session_id>[0-9a-f]{24})/$", consumers.UnityConsumer.as_asgi()),
    # The therapist dashboard watches live results and can override settings.
    re_path(r"^ws/dashboard/(?P<session_id>[0-9a-f]{24})/$", consumers.DashboardConsumer.as_asgi()),
]