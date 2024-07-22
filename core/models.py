from django.db import models
import string, random

def generate_unique_face_id():
    while True:
        face_id = ''.join(random.choices(string.digits, k=10))
        if not FaceRegistration.objects.filter(face_id=face_id).exists():
            return face_id
        
    
class FaceRegistration(models.Model):
    face_id = models.CharField(max_length=10, unique=True, editable=False)
    face_embedding = models.JSONField()

    def save(self, *args, **kwargs):
        if not self.face_id:
            self.face_id = generate_unique_face_id()
        super().save(*args, **kwargs)

    def __str__(self):
        return self.face_id

