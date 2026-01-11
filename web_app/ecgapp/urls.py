from django.urls import path
from . import views
from ecgapp import views

urlpatterns = [
    path('', views.upload_image, name='upload'),
    path('export_pdf/', views.export_pdf, name='export_pdf'),  # <--- Add this line
]