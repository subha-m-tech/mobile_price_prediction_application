from django import forms
from .models import *


class mppForm(forms.ModelForm):
    class Meta():
        model=mppModel
        fields=['ppi','cpu_core','internal_mem','ram','cpu_freq', 'rearcam', 'thickness','battery']
