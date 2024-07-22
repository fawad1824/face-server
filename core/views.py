"""
This module provides APIs for face and ID card processing, including face registration,
face matching, image retrieval, entry deletion, and ID number extraction from images.
"""

import numpy as np
import os
import logging
from datetime import datetime
from ultralytics import YOLO
import easyocr
import spacy
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageUploadSerializer, ImageSerializer, ImageFolderSerializer, TextInputSerializer, ImageSerializertwo
from .modules.preprocessing import face_extraction, face_detection, expand_face, face_detection_2
from .id_extraction import extract_card, extract_id, extract_text
from .face_recogniton import identification, registration, face_comp, face_proccessing, find_cosine_distance
from .utils import handle_error, save_image, convert2array
from .permission import HasAccessCode
from .models import FaceRegistration
from PIL import Image
import threading
from threading import Thread
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models
DETECTION_MODEL_PATH = os.path.join(settings.BASE_DIR, "core", "model_files", "best.tflite")
NER_MODEL_PATH = os.path.join(settings.BASE_DIR, "core", "model_files", "output_ner_ZIM_L", "model-best")
detection_model = YOLO(DETECTION_MODEL_PATH, task='detect')
nlp1 = spacy.load(NER_MODEL_PATH)
reader = easyocr.Reader(['en'])
@api_view(['POST'])
@permission_classes([HasAccessCode])
def face_registration(request):
    logger.info("Face registration request received")
    serializer = ImageUploadSerializer(data=request.data)
    if serializer.is_valid():
        image = serializer.validated_data['image']
        result_data = {}  # Shared dictionary to store results
        error_data = None  # Variable to store any error that might occur

        def process_face_registration():
            nonlocal error_data  # Allow modification of the outer variable
            try:
                face, _ = face_detection(image)
                if face is None:
                    error_data = 'No Face Detected.'
                    return

                ext = face_extraction(face)
                if ext is None:
                    error_data = 'Face extraction failed.'
                    return

                message, match, score, stat = registration(ext)
                url = request.build_absolute_uri(f"/media/faces/{match}.jpg")
                result_data.update({
                    'message': message,
                    'match': match,
                    'score': score,
                    'url': url,
                    'status': stat
                })

                if stat == 1:
                    face_save_path = os.path.join(settings.MEDIA_ROOT, 'faces', f"{match}.jpg")
                    save_image(face, face_save_path, resize=True)

            except Exception as e:
                logger.error(f"Error during face registration: {str(e)}")
                error_data = f'Error processing image: {str(e)}'

        # Create a thread for processing
        processing_thread = Thread(target=process_face_registration)
        processing_thread.start()
        processing_thread.join()  # Wait for the thread to finish

        # Check if any error occurred during processing
        if error_data:
            logger.warning(f"Face registration error: {error_data}")
            return handle_error(error_data, status.HTTP_400_BAD_REQUEST)

        logger.info("Face registration successful")
        return Response(result_data, status=status.HTTP_200_OK)

    logger.warning("Invalid form submission for face registration")
    return handle_error('Invalid form submission.', status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([HasAccessCode])
def face_match(request):
    logger.info("Face match request received")
    serializer = ImageUploadSerializer(data=request.data)
    if serializer.is_valid():
        image = serializer.validated_data['image']
        result_data = {}
        error_data = None

        def process_face_matching():
            nonlocal error_data
            try:
                face, _ = face_detection(image)
                if face is None:
                    error_data = 'No Face Detected.'
                    return

                ext = face_extraction(face)
                message, match, score, stat = identification(ext)

                result_data.update({
                    'message': message,
                    'match': match,
                    'score': score,
                    'status': stat
                })

            except Exception as e:
                logger.error(f"Error during face match: {str(e)}")
                error_data = f'Error processing image: {str(e)}'

        # Create a thread for processing
        processing_thread = Thread(target=process_face_matching)
        processing_thread.start()
        processing_thread.join()

        # Check if any error occurred during processing
        if error_data:
            logger.warning(f"Face match error: {error_data}")
            return handle_error(error_data, status.HTTP_400_BAD_REQUEST)

        logger.info("Face match successful")
        return Response(result_data, status=status.HTTP_200_OK)

    logger.warning("Invalid form submission for face match")
    return handle_error('Invalid form submission.', status=status.HTTP_400_BAD_REQUEST)
   
@api_view(['GET'])
@permission_classes([HasAccessCode])
def view_image(request, face_id):
    logger.info(f"View image request received for face_id: {face_id}")
    serializer = ImageSerializer(data={'face_id': face_id})
    if not serializer.is_valid():
        logger.warning(f"Invalid face_id for view_image: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    image_path = os.path.join(settings.MEDIA_ROOT, 'faces', f"{face_id}.jpg")
    if os.path.exists(image_path):
        try:
            url = request.build_absolute_uri(f"/media/faces/{face_id}.jpg")
            data = {
                'status': True,
                'url': url,
                'error': None
            }
            logger.info(f"Image found for face_id: {face_id}")
            return Response(data, status=status.HTTP_200_OK)
        except Exception as error:
            logger.error(f"Error opening image file for face_id: {face_id}, error: {str(error)}")
            return Response({'error': f"Error opening image file: {str(error)}"},
                            status=status.HTTP_400_BAD_REQUEST)
    else:
        logger.warning(f"Image not found for face_id: {face_id}")
        return Response({'error': 'Image not found'}, status=status.HTTP_404_NOT_FOUND)


@api_view(['POST'])
@permission_classes([HasAccessCode])
def delete_entry(request, face_id):
    logger.info(f"Delete entry request received for face_id: {face_id}")
    serializer = ImageSerializer(data={'face_id': face_id})
    if not serializer.is_valid():
        logger.warning(f"Invalid face_id for delete_entry: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    image_path = os.path.join(settings.MEDIA_ROOT, 'faces', f"{face_id}.jpg")
    if os.path.exists(image_path):
        try:
            entry = FaceRegistration.objects.get(face_id=face_id)
            if entry is None:
                logger.warning(f"No matching database entry found for face_id: {face_id}")
                return Response({'error': 'No matching database entry found.'},
                                status=status.HTTP_404_NOT_FOUND)
            entry.delete()
            os.remove(image_path)
            logger.info(f"Image and entry deleted successfully for face_id: {face_id}")
            return Response({'status': True,
                             'message': 'Image and entry deleted successfully.'},
                            status=status.HTTP_200_OK)
        except Exception as error:
            # logger.error(f"Error deleting image or entry for face_id: {face_id}, error: {str(error)}")
            return Response({'error': f"Error deleting image or entry: {str(error)}"},
                            status=status.HTTP_400_BAD_REQUEST)
    else:
        logger.warning(f"Image not found for face_id: {face_id}")
        return Response({'error': 'Image not found'},
                        status=status.HTTP_404_NOT_FOUND)


@api_view(['POST'])
@permission_classes([HasAccessCode])
def id_image(request):
    logger.info("ID image processing request received")
    serializer = ImageUploadSerializer(data=request.data)
    if serializer.is_valid():
        image = serializer.validated_data['image']
        try:
            detected_card = extract_card(image, detection_model)
            if detected_card is None:
                logger.warning("No ID card detected in image")
                return Response({'status': False, 'ID': None, 'card_url': None,
                                 'error': 'No Card Detected'},
                                 status=status.HTTP_200_OK)

            card_info = extract_id(detected_card, reader, nlp1)
            detected_card = np.array(detected_card)
            if not card_info:
                logger.warning("ID number not extracted from card")
                return Response({'status': False, 'ID': None, 'card_url': None,
                                 'error': 'ID number not extracted, try again'},
                                 status=status.HTTP_200_OK)
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                img_filename = f"id_{timestamp}.jpg"
                img_save_path = os.path.join(settings.MEDIA_ROOT, 'cards', img_filename)
                card_image = Image.fromarray(detected_card.astype(np.uint8))
                card_image.save(img_save_path)
                url = request.build_absolute_uri(f"/media/cards/{img_filename}")

                logger.info("ID card and number extracted successfully")
                return Response({'status': True, 'ID': card_info[0], 'card_url': url, 'error': None},
                                status=status.HTTP_200_OK)
            except Exception as e:
                logger.error(f"Error saving image for ID card: {str(e)}")
                return handle_error(f'Error saving image: {str(e)}', status.HTTP_400_BAD_REQUEST)
        except Exception as error:
            logger.error(f"Error processing image for ID card extraction: {str(error)}")
            return handle_error(f'Error processing image: {str(error)}', status.HTTP_400_BAD_REQUEST)

    logger.warning("Invalid form submission for ID image processing")
    return handle_error('Invalid form submission.', status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([HasAccessCode])
def id_face(request):
    serializer = ImageUploadSerializer(data=request.data)
    if serializer.is_valid():
        image = serializer.validated_data['image']
        try:
            # image = cv2.resize(image, (800,800))
            detected_card = extract_card(image, detection_model)
            if detected_card is None:
                return Response({'status': False, 'ID': None, 'card_url': None,
                                 'error': 'No Card Detected'},
                                 status=status.HTTP_200_OK)       
            _, coord = face_detection(image)
            if not coord:
                coord=None
            card_info = extract_id(detected_card, reader, nlp1)
            detected_card=np.array(detected_card)
            if not card_info:
                return Response({'status': False, 'face_coord': None, 'card_url': None,
                                 'error': 'Card information not extracted, try again'},
                                 status=status.HTTP_200_OK)
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_save_path = os.path.join(settings.MEDIA_ROOT, 'id_card', f"{timestamp}.jpg")
                save_image(detected_card, image_save_path)
                url = request.build_absolute_uri(f"/media/id_card/{timestamp}.jpg")
                data = {'status': True, 'face_coord':coord, 'card_url': url, 'error': None}
                data.update(card_info)
                return Response(data, status=status.HTTP_200_OK)
            except Exception as error:
                return Response({'status': False, 'face_coord': None, 'card_url': None,
                                 'error': f"Error saving card image: {str(error)}"},
                                 status=status.HTTP_400_BAD_REQUEST)
        except Exception as error:
            return Response({'status': False, 'ID': None, 'card_url': None,
                             'error': f"Error processing image: {str(error)}"}, 
                             status=status.HTTP_400_BAD_REQUEST)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([HasAccessCode])
def FaceCompare(request):
    print(request.data)
    serializer = ImageSerializertwo(data=request.data)

    if serializer.is_valid():
        image1 = serializer.validated_data['image1']
        image2 = serializer.validated_data['image2']

        result_data = {}
        error_data = []

        def process_image(image, key):
            try:
                input_image = Image.open(image)
                input_image = input_image.convert('RGB')
                image_array = np.array(input_image)  # Ensure thread safety
                print(f"Image Array for {key}: {image_array.shape}")

                face = face_detection_2(image_array)  # This should be thread-safe
                if face is None:
                    raise ValueError(f'No Face Detected in {key}.')
                if isinstance(face, tuple):
                    face = np.array(face)  # Convert tuple to numpy array if necessary
                face = face_proccessing(face)  # This should be thread-safe
                if face is None:
                    raise ValueError(f'Face processing failed for {key}.')

                face = np.copy(face)  # Ensure that face is copied if it's a NumPy array
                result_data[key] = face
                print(f"Processed Face for {key}: {face.shape}")

            except (ValueError, OSError) as e:
                error_data.append(f'Error processing {key}: {str(e)}')

        # Create and start a single thread for image processing
        def process_images():
            # Process each image sequentially within the thread
            process_image(image1, 'face1')
            process_image(image2, 'face2')

        thread = threading.Thread(target=process_images)
        thread.start()
        thread.join()  # Wait for the thread to complete

        if error_data:
            return Response({'message': ' '.join(error_data), 'score': None}, status=status.HTTP_400_BAD_REQUEST)

        face1 = result_data.get('face1')
        face2 = result_data.get('face2')

        if face1 is None:
            return Response({'message': 'No Face Detected in image1', 'score': None}, status=status.HTTP_400_BAD_REQUEST)
        if face2 is None:
            return Response({'message': 'No Face Detected in image2', 'score': None}, status=status.HTTP_400_BAD_REQUEST)

        try:
            cosine_similarity = find_cosine_distance(face1 / 255.0, face2 / 255.0)
        except TypeError as e:
            return Response({'message': str(e), 'score': None}, status=status.HTTP_400_BAD_REQUEST)

        score = (1 - cosine_similarity) * 100
        result, score = face_comp(face1, face2)

        return Response({
            'message': 'Face Matched' if result else 'Not Matched',
            'score': round(score, 2) if result else None
        }, status=status.HTTP_200_OK)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ExtractThread(Thread):
    def __init__(self, text, nlp1):
        Thread.__init__(self)
        self.text = text
        self.nlp1 = nlp1
        self.result = None
    def run(self):
        self.result = extract_text(self.text, self.nlp1)

@api_view(['POST'])
@permission_classes([HasAccessCode])
def entities_extraction(request):
    serializer = TextInputSerializer(data=request.data)
    if serializer.is_valid():
        input_text = serializer.validated_data['text']
        try:
            id_number = extract_text(input_text, nlp1)
            if not id_number:
                logger.warning("No ID number detected in text input")
                return Response({'status': False, 'ID': None, 'error': 'No ID Number Detected'},
                                status=status.HTTP_200_OK)

            logger.info("ID number extracted successfully from text input")
            return Response({'status': True, 'ID': id_number, 'error': None},
                            status=status.HTTP_200_OK)
        except Exception as error:
            logger.error(f"Error extracting ID number from text input: {str(error)}")
            return handle_error(f'Error processing text: {str(error)}', status.HTTP_400_BAD_REQUEST)

    logger.warning("Invalid form submission for text input")
    return handle_error('Invalid form submission.', status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([HasAccessCode])
def get_image(request):
    serializer = ImageFolderSerializer(data=request.data)
    if serializer.is_valid():
        try:
            image = serializer.validated_data['image']
            folder_name = serializer.validated_data.get('folder_name', 'images')
            # image = cv2.resize(image, (800,800)) 
            face, coord = face_detection(image)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_directory = os.path.join(settings.MEDIA_ROOT, folder_name)
            face_save_path = os.path.join(face_directory, f"{timestamp}.jpg")

            # Ensure the directory exists
            os.makedirs(face_directory, exist_ok=True)

            url = request.build_absolute_uri(f"/media/{folder_name}/{timestamp}.jpg")
            if face is None:
                image = convert2array(image)
                save_image(image, face_save_path)        
                return Response({'status': 'Original Image Saved', 'ID': timestamp,
                                 'URL': str(url), 'error': None},
                                    status=status.HTTP_200_OK)

            expanded_face = expand_face(image, coord)
            save_image(expanded_face, face_save_path , resize=True)
            return Response({'status': 'Face Image Saved', 'ID': timestamp, 'url': url,
                                    'error': None},
                                    status=status.HTTP_200_OK)

        except Exception as error:
            return Response({'status': False, 'ID': None, 'url': None,
                                    'error': str(error)},
                                    status=status.HTTP_400_BAD_REQUEST)
    else:
        return Response({'status': False, 'ID': None, 'url': None,
                                'error': serializer.errors},
                                status=status.HTTP_400_BAD_REQUEST)
    

@api_view(['POST'])
@permission_classes([HasAccessCode])
def delete_image(request, image_id):
    folder_name = request.data.get('folder_name', 'images')
    
    # Construct the image path
    image_path = os.path.join(settings.MEDIA_ROOT, folder_name, f"{image_id}.jpg")
    
    if not os.path.exists(image_path):
        # If the image is not found in the specified folder, check the default 'images' folder
        image_path = os.path.join(settings.MEDIA_ROOT, 'images', f"{image_id}.jpg")

    if os.path.exists(image_path):
        try:
            os.remove(image_path)
            return Response({'status': True, 'message': 'Image deleted successfully.'},
                            status=status.HTTP_200_OK)
        except Exception as error:
            return Response({'error': f"Error deleting image: {str(error)}"},
                            status=status.HTTP_400_BAD_REQUEST)
    else:
        return Response({'error': 'Image not found'},
                        status=status.HTTP_404_NOT_FOUND)

