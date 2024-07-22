from rest_framework import serializers
from .models import FaceRegistration
import re

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()

class RegisterFaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceRegistration
        fields = ['face_id', 'face_embedding']


class ImageSerializer(serializers.Serializer):
    face_id = serializers.CharField(max_length=10)

    def validate_face_id(self, value):
        if not re.match(r'^\d{10}$', value):
            raise serializers.ValidationError("Face ID must be a 10-digit number.")
        return value

class IdSerializer(serializers.Serializer):
    id = serializers.CharField(max_length=255)
    folder_name = serializers.CharField(max_length=255, required=False)

    def validate_id(self, value):
        pattern = r'^\d{8}_\d{6}$'
        if not re.match(pattern, value):
            raise serializers.ValidationError("ID format must be YYYYMMDD_HHMMSS.")
        return value

class ImageSerializertwo(serializers.Serializer):
    image1 = serializers.ImageField()
    image2 = serializers.ImageField()

class ImageFolderSerializer(serializers.Serializer):
    image = serializers.ImageField()
    folder_name = serializers.CharField(max_length=255, required=False)


class TextInputSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=2000)