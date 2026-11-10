"""
ASGI entry point (served by Daphne:  daphne -p 8000 main.asgi:application).

HTTP requests go to normal Django. WebSocket connections pass through
TokenAuthMiddleware, then the URL router in core/routing.py.

AllowedHostsOriginValidator is intentionally not used: Unity's WebSocket
client sends no Origin header, so it would reject every Unity connection.
Authentication is the signed token in the URL instead (see core/auth.py).
"""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
# Must run before importing anything that touches Django models/settings.
django_asgi_app = get_asgi_application()

from channels.routing import ProtocolTypeRouter, URLRouter  # noqa: E402

from core.auth import TokenAuthMiddleware  # noqa: E402
from core.routing import websocket_urlpatterns  # noqa: E402

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": TokenAuthMiddleware(URLRouter(websocket_urlpatterns)),
    }
)