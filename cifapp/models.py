from django.db import models
from django.contrib.auth.models import User

class UserDataset(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    dataset_name = models.CharField(max_length=255)  # 数据库名称
    note = models.TextField(blank=True)               # 备注
    data_file = models.FileField(upload_to='user_datasets/')  # excel/csv
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

# Create your models here.
class Record(models.Model):
    
    owner = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True
    )
    is_public = models.BooleanField(default=True)
    
    structure_file = models.FileField(upload_to='sturestrus/')
    original_structure_name = models.CharField(max_length=255)  # 存储原始结构文件名
    features_file = models.FileField(upload_to='features/')
    original_features_name = models.CharField(max_length=255)  # 存储原始特征文件名
    performance_k = models.CharField(max_length=255)
    performance_tan_omega = models.CharField(max_length=255)
    performance_tck = models.CharField(max_length=255)
    note = models.TextField(blank=True)
    reference = models.CharField(max_length=255)
    uploader = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)




