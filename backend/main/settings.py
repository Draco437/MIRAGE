"""
Mirage settings. Secrets and per-machine values come from backend/.env
(see .env.example), so nothing sensitive is committed.
"""
import os
import ssl
from datetime import timedelta
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def env_list(name: str, default: str) -> list:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


# --- Core ---------------------------------------------------------------
DEBUG = env_bool("DJANGO_DEBUG", True)
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
if not DEBUG and SECRET_KEY.startswith("dev-only"):
    raise ImproperlyConfigured("Set DJANGO_SECRET_KEY when DJANGO_DEBUG is off.")

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")

INSTALLED_APPS = [
    "daphne",  # first: makes `runserver` speak ASGI/WebSocket too
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # third-party
    "channels",
    "rest_framework",
    "corsheaders",
    # Mirage
    "core",
    "ml_engine",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",  # must be before CommonMiddleware
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "main.urls"
WSGI_APPLICATION = "main.wsgi.application"
ASGI_APPLICATION = "main.asgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# --- Databases ----------------------------------------------------------
# SQLite holds only login accounts. All Mirage data is in MongoDB (see db.py).
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"  # everything is stored in UTC; the dashboard formats for the viewer
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- REST API -----------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": ("rest_framework.permissions.IsAuthenticated",),
    "DEFAULT_RENDERER_CLASSES": ("rest_framework.renderers.JSONRenderer",),
    # Slows down password guessing on login/register
    "DEFAULT_THROTTLE_CLASSES": ("rest_framework.throttling.AnonRateThrottle",),
    "DEFAULT_THROTTLE_RATES": {"anon": "30/min"},
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=60),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
}

CORS_ALLOWED_ORIGINS = env_list("CORS_ALLOWED_ORIGINS", "http://localhost:3000")

# --- Channels (WebSocket) and Celery ------------------------------------
# REDIS_URL works with a local Redis (redis://localhost:6379/0) and with Render
# Key Value. From outside Render, use its EXTERNAL url, which starts with
# rediss:// (TLS) - see .env.example.
REDIS_URL = os.environ.get("REDIS_URL", "").strip()
REDIS_USES_TLS = REDIS_URL.startswith("rediss://")
REDIS_SSL_VERIFY = env_bool("REDIS_SSL_VERIFY", True)  # leave on; see .env.example

if REDIS_URL:
    CHANNEL_LAYERS = {
        "default": {
            "BACKEND": "channels_redis.core.RedisChannelLayer",
            "CONFIG": {"hosts": [REDIS_URL]},
        }
    }
else:
    # Single-process only: a separate Celery worker could not reach the consumers.
    CHANNEL_LAYERS = {"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}

CELERY_BROKER_URL = REDIS_URL or "memory://"
if REDIS_USES_TLS:
    # Celery must be told explicitly that the broker connection is TLS-secured.
    CELERY_BROKER_USE_SSL = {
        "ssl_cert_reqs": ssl.CERT_REQUIRED if REDIS_SSL_VERIFY else ssl.CERT_NONE
    }
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True
# Without Redis there is no worker process, so run tasks inline.
CELERY_TASK_ALWAYS_EAGER = env_bool("CELERY_EAGER", not REDIS_URL)
CELERY_TASK_SERIALIZER = "json"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_IGNORE_RESULT = True
CELERY_TASK_TIME_LIMIT = 30  # seconds; a stuck inference must not block the queue
if os.name == "nt":
    CELERY_WORKER_POOL = "solo"  # Celery's default pool does not work on Windows

# --- Mirage tuning knobs ------------------------------------------------
MIRAGE = {
    "WINDOW_SECONDS": 10,               # how much recent data each analysis looks at
    "ANALYSIS_INTERVAL_SECONDS": 5,     # how often a session triggers an analysis
    "FRAME_MIN_INTERVAL_SECONDS": 1.0,  # extra camera frames inside this gap are dropped
    "MAX_FRAME_BASE64_CHARS": 400_000,  # about 300 KB of JPEG
    "UNITY_TOKEN_MAX_AGE": 4 * 60 * 60, # seconds a session token stays valid
}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}