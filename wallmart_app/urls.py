from django.urls import path
from .views import model_results_view, predict

urlpatterns = [
    path('', model_results_view, name='results'),  # Main view for displaying all model results
    path('predict/',predict,name='predict')
]
