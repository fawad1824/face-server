# Generated by Django 5.0 on 2024-03-17 16:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0003_imageembedding_delete_faceimage'),
    ]

    operations = [
        migrations.RenameField(
            model_name='imageembedding',
            old_name='embedding',
            new_name='embeddings',
        ),
        migrations.AlterField(
            model_name='imageembedding',
            name='name',
            field=models.CharField(max_length=100),
        ),
    ]
