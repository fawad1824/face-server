from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
import cv2
import numpy as np
from PIL import Image

def handle_error(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR):
    return Response({'message': message, 'match': None, 'score': None,
                     'image_url': None, 'status': False}, status=status_code)

def save_image(image, path, resize=False):
    with default_storage.open(path, 'wb') as destination:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if resize:
            image = cv2.resize(image, (150, 200))
        _, buffer = cv2.imencode(".jpg", image)
        destination.write(buffer.tobytes())

def convert2array(image):
    input_image = Image.open(image)
    input_image = input_image.convert('RGB')
    image_array = np.array(input_image)
    return image_array

