from django.db import models

# Create your models here.
class mppModel(models.Model):

    ppi=models.IntegerField()
    cpu_core=models.IntegerField()
    internal_mem=models.FloatField()
    ram=models.FloatField()
    cpu_freq=models.FloatField()
    rearcam = models.FloatField()
    thickness = models.FloatField()
    battery = models.IntegerField()
