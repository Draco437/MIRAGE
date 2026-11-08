import logging

from django.apps import AppConfig

log = logging.getLogger(__name__)


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        # Create MongoDB indexes at startup. A down database must not stop
        # Django from booting (migrate, tests, ...), so only warn.
        try:
            import db

            db.ensure_indexes()
        except Exception as exc:
            log.warning("Could not create MongoDB indexes (is MongoDB running?): %s", exc)