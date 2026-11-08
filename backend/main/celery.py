"""Celery app definition. Workers start with:  celery -A main worker -l info"""
import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")

app = Celery("main")
# Reads every CELERY_* value from settings.py
app.config_from_object("django.conf:settings", namespace="CELERY")
# Finds tasks.py in each installed app (here: ml_engine/tasks.py)
app.autodiscover_tasks()