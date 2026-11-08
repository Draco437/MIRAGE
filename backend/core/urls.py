"""REST routes, all mounted under /api/ (see main/urls.py)."""
from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from . import views

urlpatterns = [
    path("health/", views.HealthView.as_view()),
    # Auth
    path("auth/register/", views.RegisterView.as_view()),
    path("auth/login/", TokenObtainPairView.as_view()),
    path("auth/refresh/", TokenRefreshView.as_view()),
    # Children
    path("children/", views.ChildListCreateView.as_view()),
    path("children/<str:child_id>/", views.ChildDetailView.as_view()),
    path("children/<str:child_id>/progress/", views.ChildProgressView.as_view()),
    # Sessions
    path("sessions/", views.SessionListCreateView.as_view()),
    path("sessions/<str:session_id>/", views.SessionDetailView.as_view()),
    path("sessions/<str:session_id>/analytics/", views.SessionAnalyticsView.as_view()),
]