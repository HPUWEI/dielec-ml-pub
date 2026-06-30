from . import views, viewsRecords
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('login/', auth_views.LoginView.as_view(template_name='cifapp/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='record_list'), name='logout'),
    path('register/', viewsRecords.register, name='register'),
    path('workspace/', viewsRecords.my_workspace, name='my_workspace'),#工作台
    path('workspace/download/<int:dataset_id>/', viewsRecords.download_dataset, name='download_dataset'),
    path('workspace/delete/<int:dataset_id>/', viewsRecords.delete_dataset, name='delete_dataset'),
    path('delete/<int:record_id>/', viewsRecords.delete_record, name='delete_record'),
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),  # 根路径直接访问上传页面
    path('calculate/', views.calculate, name='calculate'),
    path('home/', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('record_list/', viewsRecords.record_list, name='record_list'),
    path('upload_file/', viewsRecords.upload_file, name='upload_file'),
    path('download/<int:record_id>/<str:file_type>/', viewsRecords.download_file, name='download_file'),
    path('illustration/', views.illustration, name='illustration'),
    path('ML/', views.ML, name='ML'),
    path("predict/", views.predict, name="predict"),
    path('upload/', views.upload_excel, name='upload_excel'),
    path('process/', views.process_excel, name='process_excel'),
    path('feature_engineering/', views.feature_engineering, name='feature_engineering'),
    path("ai_interpretation/",views.ai_interpretation,name="ai_interpretation"),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)