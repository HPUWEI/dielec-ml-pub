from django import forms

class UploadCIFForm(forms.Form):
    cif_file = forms.FileField()