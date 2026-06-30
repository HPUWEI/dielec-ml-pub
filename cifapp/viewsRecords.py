import os
from django.core.paginator import Paginator
from django.http import JsonResponse, HttpResponseNotFound, FileResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.utils.crypto import get_random_string
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from cifapp.models import Record
from cifapp.Status_codes import StatusCodes
import json
from django.utils import timezone
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from cifapp.models import Record, UserDataset

@login_required(login_url='/login/')
def my_workspace(request):
    upload_success = False
    if request.method == 'POST':
        dataset_name = request.POST.get('dataset_name', '').strip()
        note = request.POST.get('note', '').strip()
        data_file = request.FILES.get('data_file')
        # 临时调试：无论什么情况都写日志
        import logging
        logger = logging.getLogger('cifapp')
        logger.error("=== POST received ===")
        logger.error("dataset_name: %s", repr(dataset_name))
        logger.error("data_file: %s", repr(data_file))
        logger.error("POST keys: %s", list(request.POST.keys()))
        logger.error("FILES keys: %s", list(request.FILES.keys()))
        if dataset_name and data_file:
            try:
                dataset = UserDataset(
                    owner=request.user,
                    dataset_name=dataset_name,
                    note=note,
                    data_file=data_file,
                    original_filename=data_file.name,
                )
                dataset.save()
                upload_success = True
                logger.error("=== SAVED OK, id=%s ===", dataset.id)
            except Exception as e:
                logger.error("=== SAVE ERROR: %s ===", str(e))
        else:
            logger.error("=== CONDITION FAILED ===")
    datasets = UserDataset.objects.filter(owner=request.user).order_by('-uploaded_at')
    return render(request, 'cifapp/my_workspace.html', {
        'datasets': datasets,
        'upload_success': upload_success,
    })


@login_required(login_url='/login/')
def download_dataset(request, dataset_id):
    ds = get_object_or_404(UserDataset, pk=dataset_id, owner=request.user)
    response = FileResponse(ds.data_file, content_type='application/octet-stream')
    response['Content-Disposition'] = f'attachment; filename="{ds.original_filename}"'
    return response

@login_required(login_url='/login/')
def delete_dataset(request, dataset_id):
    ds = get_object_or_404(UserDataset, pk=dataset_id, owner=request.user)
    ds.delete()
    return redirect('my_workspace')


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('my_workspace')
    else:
        form = UserCreationForm()
    return render(request, 'cifapp/register.html', {'form': form})

    
@login_required(login_url='/login/')
def delete_record(request, record_id):
    record = get_object_or_404(Record, pk=record_id, owner=request.user)
    record.delete()
    return redirect('my_workspace')

def record_list(request):
    # 获取所有记录
    records = Record.objects.all().order_by('original_structure_name')

    # 获取搜索关键词
    search_query = request.GET.get('search', '').strip()

    # 如果有搜索关键词，则过滤记录
    if search_query:
        records = records.filter(
            original_structure_name__icontains=search_query
        )

    # 分页
    paginator = Paginator(records, 25)  # 每页显示25条记录
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    # 获取总页数
    total_pages = paginator.num_pages

    # 获取当前页码
    current_page = page_obj.number

    # 计算显示的页码范围
    page_range = []
    if total_pages <= 5:
        page_range = range(1, total_pages + 1)
    else:
        if current_page <= 3:
            page_range = range(1, 6)
        elif current_page >= total_pages - 2:
            page_range = range(total_pages - 4, total_pages + 1)
        else:
            page_range = range(current_page - 2, current_page + 3)

    context = {
        'records': page_obj,
        'total_pages': total_pages,
        'page_range': page_range,
        'search_query': search_query  # 将搜索关键词传递到模板
    }

    return render(request, 'cifapp/record_list.html', context)


def upload_file(request):
    if request.method == 'POST':
            structure_file = request.FILES.get('Structure')
            features_file = request.FILES.get('Features')
            performance_k = request.POST.get('Performance[k]')
            performance_tan_omega = request.POST.get('Performance[tanò]')
            performance_tck = request.POST.get('Performance[TCK]')
            note = request.POST.get('Note')
            reference = request.POST.get('Reference')
            uploader = request.POST.get('Uploader')

            original_structure_name = "None"
            original_features_name = "None"
            # 获取原始文件名
            if structure_file is not None:
                original_structure_name = structure_file.name
            if features_file is not None:
                original_features_name = features_file.name

            # 生成随机序列并修改文件名
            random_string = get_random_string(length=8)
            if structure_file is not None:
                structure_file.name = f"{os.path.splitext(original_structure_name)[0]}_{random_string}{os.path.splitext(original_structure_name)[1]}"
            if features_file is not None:
                features_file.name = f"{os.path.splitext(original_features_name)[0]}_{random_string}{os.path.splitext(original_features_name)[1]}"

            record = Record(
                structure_file=structure_file,
                features_file=features_file,
                original_structure_name=original_structure_name,  # 存储原始文件名
                original_features_name=original_features_name,
                performance_k=performance_k,
                performance_tan_omega=performance_tan_omega,
                performance_tck=performance_tck,
                note=note,
                reference=reference,
                uploader=uploader,
                uploaded_at=timezone.now()
            )

            if request.user.is_authenticated:
                record.owner = request.user
            record.save()
            return redirect('record_list')
            # 处理错误，例如重定向回表单页面并显示错误消息
    else:
        return render(request, 'cifapp/record_upload.html')


def download_file(request, record_id, file_type):
    record = get_object_or_404(Record, pk=record_id)
    if file_type == 'structure':
        file = record.structure_file
    elif file_type == 'features':
        file = record.features_file
    else:
        return HttpResponseNotFound("File not found")

    response = FileResponse(file, content_type='application/octet-stream')
    response['Content-Disposition'] = f'attachment; filename="{file.name}"'
    return response



