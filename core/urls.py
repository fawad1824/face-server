from django.urls import path
from . import views


urlpatterns = [
    path('view_image/<face_id>/', views.view_image),
    path('face_registration/', views.face_registration),
    path('face_match/', views.face_match),
    path('id_image/', views.id_image),
    path('id_face/', views.id_face),
    path('delete/<face_id>/', views.delete_entry),
    path('delete_image/<image_id>/', views.delete_image),
    path('get_image/', views.get_image),
    path('ent_ext/', views.entities_extraction),
    path('face_compare/', views.FaceCompare),
]