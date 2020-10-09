from django.contrib import admin
from django.urls import path, include

from django.conf.urls.static import static
from . import views

from django.conf import settings

urlpatterns = [
    path('', views.index , name = 'index'),
    path('upload', views.uploadImage, name = 'uploadImage'),

    # Adding a new URL
    # path('detect/', views.runMLCode, name = 'runMLCode'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

