from django.db import models

# Create your models here.
class Cell(models.Model):
    name = models.CharField(max_length=255, default='inputimage.png')
    pic = models.ImageField(upload_to = "pictures");

class Result(models.Model):
    cell_condition = models.CharField(max_length=100)

