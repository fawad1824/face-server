from typing import Union, Tuple, List, Optional
import numpy as np
from .models import FaceRegistration
# from .modules.preprocessing import facenet_normalization
from .modules.face_embedding import get_embedding

def face_proccessing(face):
    # normalized_face = facenet_normalization(face)
    embedding = get_embedding(face)
    embedding = embedding.tolist()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def face_comp(face_1, face_2):
    cosine_similarity = find_cosine_distance(face_1, face_2)
    score = (1 - cosine_similarity) * 100
    print("score",score)
    if score > 45.0:
        return True, score
    return False, score


def search_person(embedding: np.ndarray, database_entries: List[dict],
                  threshold: float = 45.0) -> Tuple[Optional[str], Optional[float]]:
    """
    Searches for the best match of a given embedding in the database entries.

    Args:
        embedding (np.ndarray): The embedding of the face to search for.
        database_entries (List[dict]): List of database entries with 'face_id' and 'face_embedding'.
        threshold (float): The score threshold for considering a match. Defaults to 65.0.

    Returns:
        Tuple[Optional[str], Optional[float]]: The best match face ID and the similarity score.
    """
    highest_similarity = -1
    best_match = None

    embedding = embedding / np.linalg.norm(embedding)
    for entry in database_entries:
        stored_embedding = np.array(entry['face_embedding'])
        stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
        cosine_similarity = find_cosine_distance(embedding, stored_embedding)
        score = (1 - cosine_similarity) * 100
        if score > highest_similarity:
            highest_similarity = score
            best_match = entry['face_id']

    if highest_similarity > threshold:
        return best_match, round(highest_similarity, 2)

    return None, None

def identification(face: np.ndarray):
    """
    Identifies a face and returns the identification result.

    Args:
        face (np.ndarray): The face image to identify.
        mode (str): The mode of operation, either 'regist' for registration or 
        'recog' for recognition. Defaults to 'regist'.

    Returns:
        Tuple[str, Optional[str], Optional[float], bool]: Message, match ID, score, and status.
    """
    try:
        # normalized_face = facenet_normalization(face)
        embedding = get_embedding(face)
        embedding = embedding.tolist()

        if FaceRegistration.objects.exists():
            database_entries = FaceRegistration.objects.values('face_id', 'face_embedding')

            for entry in database_entries:
                if isinstance(entry['face_embedding'], list):
                    entry['face_embedding'] = np.array(entry['face_embedding'])
            match, score = search_person(np.array(embedding), database_entries)
            if match:
                return 'Face Matched!', match, score, True
            return 'Not Matched', None, None, False
        return 'No Registered Faces in the Database', None, None, False
    except Exception as e:
        return f'Error processing image: {str(e)}', None, None, False
    
def registration(face):
    """
    Identifies a face and returns the identification result.

    Args:
        face (np.ndarray): The face image to identify.
        mode (str): The mode of operation, either 'regist' for registration or
        'recog' for recognition. Defaults to 'regist'.

    Returns:
        Tuple[str, Optional[str], Optional[float], bool]: Message, match ID, score, and status.
    """
    try:
        # normalized_face = facenet_normalization(face)
        embedding = get_embedding(face)
        embedding = embedding.tolist()

        if FaceRegistration.objects.exists():
            database_entries = FaceRegistration.objects.values('face_id', 'face_embedding')

            for entry in database_entries:
                if isinstance(entry['face_embedding'], list):
                    entry['face_embedding'] = np.array(entry['face_embedding'])

            match, score = search_person(np.array(embedding), database_entries)
            if match:
                return 'Face Already Present!', match, score, False

            new_entry = FaceRegistration(face_embedding=embedding)
            new_entry.save()
            return 'Registered Successfully!', new_entry.face_id, None, True

        new_entry = FaceRegistration(face_embedding=embedding)
        new_entry.save()
        return 'Registered Successfully!', new_entry.face_id, None, True

    except Exception as e:
        return f'Error processing image: {str(e)}', None, None, False


def find_cosine_distance(source_representation: Union[np.ndarray, list],
                         test_representation: Union[np.ndarray, list]) -> float:
    """
    Calculates the cosine distance between two embeddings.

    Args:
        source_representation (Union[np.ndarray, list]): The source embedding.
        test_representation (Union[np.ndarray, list]): The test embedding.

    Returns:
        float: The cosine distance between the source and test embeddings.
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)
    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    a = np.dot(source_representation, test_representation)
    b = np.linalg.norm(source_representation)
    c = np.linalg.norm(test_representation)

    return 1 - (a / (b * c))
