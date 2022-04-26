from django.urls import path
from . import views
from . import orc

urlpatterns = [
    path('', views.index, name='index'),
    path('orc', orc.index, name='index'),
]
