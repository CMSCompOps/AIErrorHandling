#from django.contrib import admin
from django.urls import path
from AIErrorHandling.webapp.webapp import views as views

print("url is loaded")
urlpatterns = [
    path('predict/', views.predict),
    path('test/' , views.predict2 )
]
