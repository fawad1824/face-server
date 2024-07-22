import cv2 as cv
import numpy as np
from PIL import Image
import mediapipe as mp
from typing import Union
from core.modules.yunet import YuNetClient

detector = YuNetClient()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def face_detection_2(image):
    """
    Detects a face in an image using deepface and returns the detected face.

    Args:
        image (InMemoryUploadedFile): The uploaded image file.

    Returns:
        Optional[np.ndarray]: Detected face or None if no face is found.
    """
    
    try:
        # with torch.no_grad():
            
            # image_array = torch.cuda.empty(image.shape, dtype=torch.float32)
            # image_array.copy_(image)
            # image_array=cv.resize(image_array, 1000, 1000)

            facial_areas = detector.detect_faces(image)
            if len(facial_areas) == 0 or len(facial_areas) > 1:
                return None, None

            expand_percentage = 0
            for facial_area in facial_areas:
                x = facial_area.x
                y = facial_area.y
                w = facial_area.w
                h = facial_area.h
                left_eye = facial_area.left_eye
                right_eye = facial_area.right_eye

                if expand_percentage > 0:
                    expanded_w = w + int(w * expand_percentage / 100)
                    expanded_h = h + int(h * expand_percentage / 100)
                    x = max(0, x - int((expanded_w - w) / 2))
                    y = max(0, y - int((expanded_h - h) / 2))
                    w = min(image.shape[1] - x, expanded_w)
                    h = min(image.shape[0] - y, expanded_h)

                detected_face = image[int(y):int(y + h), int(x):int(x + w)]
                coord = [x, y, w, h]
                align_image, _ = align_face(detected_face, left_eye, right_eye)

                return align_image
    except:
        return None
    
def align_face(
    img: np.ndarray,
    left_eye: Union[list, tuple],
    right_eye: Union[list, tuple],
):
    """
    Align a given image horizantally with respect to their left and right eye locations
    Args:
        img (np.ndarray): pre-loaded image with detected face
        left_eye (list or tuple): coordinates of left eye with respect to the person itself
        right_eye(list or tuple): coordinates of right eye with respect to the person itself
    Returns:
        img (np.ndarray): aligned facial image
    """
    # if eye could not be detected for the given image, return image itself
    if left_eye is None or right_eye is None:
        return img, 0

    # sometimes unexpectedly detected images come with nil dimensions
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img, 0

    angle = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))
    img = np.array(Image.fromarray(img).rotate(angle))
    return img, angle



def facenet_normalization(img: np.ndarray) -> np.ndarray:
    """
    Normalize the image using mean and standard deviation.

    Args:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: Normalized image array.
    """
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    return img
import torch

def face_detection(image):
    """
    Detects a face in an image using deepface and returns the detected face.

    Args:
        image (InMemoryUploadedFile): The uploaded image file.

    Returns:
        Optional[np.ndarray]: Detected face or None if no face is found.
    """
    
    try:
        with torch.no_grad():
            
            input_image = Image.open(image)
            # input_image  = input_image .resize(800,800)
            input_image = input_image.convert('RGB')
            image_array = np.array(input_image)
            # image_array=cv.resize(image_array, 1000, 1000)

            facial_areas = detector.detect_faces(image_array)
            if len(facial_areas) == 0 or len(facial_areas) > 1:
                return None, None

            expand_percentage = 0
            for facial_area in facial_areas:
                x = facial_area.x
                y = facial_area.y
                w = facial_area.w
                h = facial_area.h
                left_eye = facial_area.left_eye
                right_eye = facial_area.right_eye

                if expand_percentage > 0:
                    expanded_w = w + int(w * expand_percentage / 100)
                    expanded_h = h + int(h * expand_percentage / 100)
                    x = max(0, x - int((expanded_w - w) / 2))
                    y = max(0, y - int((expanded_h - h) / 2))
                    w = min(image_array.shape[1] - x, expanded_w)
                    h = min(image_array.shape[0] - y, expanded_h)

                detected_face = image_array[int(y):int(y + h), int(x):int(x + w)]
                coord = [x, y, w, h]
                align_image, _ = align_face(detected_face, left_eye, right_eye)

                return align_image, coord
    except:
        return None, None

def face_extraction(image):
    """
    Extracts face landmarks from an image using Mediapipe and 
    returns the extracted face and the face region.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        Optional[np.ndarray]: Extracted face or None if no face is detected.
    """
    # Convert the image to a numpy array if it's a PIL Image
    roi_1 = np.array(image)
    image_1 = cv.resize(roi_1, (roi_1.shape[1], roi_1.shape[0]))

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image_1)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks_np = np.zeros((468, 2), dtype=np.int32)
            for i, landmark in enumerate(face_landmarks.landmark):
                landmarks_np[i] = (int(landmark.x * image_1.shape[1]),
                                   int(landmark.y * image_1.shape[0]))
            mask = np.zeros((image_1.shape[0], image_1.shape[1]), dtype=np.uint8)
            hull = cv.convexHull(landmarks_np)
            cv.fillConvexPoly(mask, hull, 255)
            face_extracted = cv.bitwise_and(image_1, image_1, mask=mask)
            return face_extracted
        return None


def find_cosine_distance(source_representation: np.ndarray,
                         test_representation: np.ndarray):
    """
    Calculates the cosine distance between two face embeddings.

    Args:
        source_representation (np.ndarray): Source face embedding.
        test_representation (np.ndarray): Test face embedding.

    Returns:
        float: Cosine distance between the two embeddings.
    """
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def expand_face(image, coord):
    image = Image.open(image)
    image = image.convert('RGB')
    image = np.array(image)
    img_height, img_width = image.shape[:2]

    # Extract face coordinates
    x, y, w, h = coord

    # Calculate new coordinates with 20% padding
    padding = 0.2
    new_x = max(0, int(x - w * padding))
    new_y = max(0, int(y - h * padding))
    new_w = min(img_width, int(w + w * 2 * padding))
    new_h = min(img_height, int(h + h * 2 * padding))

    # Extract the expanded face region
    expanded_face = image[new_y:new_y + new_h, new_x:new_x + new_w]
    return expanded_face
