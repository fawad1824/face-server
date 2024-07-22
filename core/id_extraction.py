"""
This module provides functions for extracting card images and IDs from input images using 
deep learning and OCR techniques. The module includes functions to extract a card image 
from a detected bounding box and to extract an ID from an image using Optical Character 
Recognition (OCR).
"""

import re
import math
from PIL import Image
import numpy as np
import torch

def extract_card(image, model):
    """
    Extracts a card image from the input image using a specified model.

    Parameters:
    image (str): Path to the input image.
    model (YOLO): The YOLO model used for detecting the card in the image.

    Returns:
    PIL.Image or None: The cropped card image if a single bounding box is detected, 
                       None otherwise.
    """
    # if torch.cuda.is_available():
    #     # device = 'cuda'
    #     print("GPU is available. Using GPU.")
    # else:
    #     # device = 'cpu'
    #     print("GPU is not available. Using CPU.")
    # model.to(device)
    # input_tensor = torch.tensor(np.array(image)).to(device)
    input_image = Image.open(image)
    results = model(input_image)
    for result in results:
        boxes = np.array(result.boxes.xyxy)
        if len(boxes) == 1:
            # Ceil the box coordinates and convert to integers
            rounded_numbers = [math.ceil(number) for number in boxes[0]]
            rounded_numbers = list(map(int, rounded_numbers))
            # Crop the image using the bounding box 
            cropped_image = input_image.crop((rounded_numbers[0], rounded_numbers[1],
                                              rounded_numbers[2], rounded_numbers[3]))
            return cropped_image
        return None
import spacy

def extract_id(image, reader, nlpl):
    """
    Extracts the ID from the given image using OCR.

    Parameters:
    image (np.ndarray): The image to process, as a NumPy array.
    reader (easyocr.Reader): The OCR reader used for detecting text in the image.

    Returns:
    str: The detected ID, or None if no valid ID is found.
    """
    result = reader.readtext(np.array(image), detail=0)
    if result:
        result = ' '.join(result)
        doc = nlpl(result)
        entities_dict = {}
        exclude_keys = ["EXTRA", "ORIGIN TITLE"]
        for ent in doc.ents:
            if ent.label_ not in exclude_keys:
                entities_dict[ent.label_] = ent.text if ent.text else ""
        return entities_dict
    return None

def extract_text(text, nlpl):
    """
    Extracts the ID from the given image using OCR.

    Parameters:
    image (np.ndarray): The image to process, as a NumPy array.
    reader (easyocr.Reader): The OCR reader used for detecting text in the image.

    Returns:
    str: The detected ID, or None if no valid ID is found.
    """

    doc = nlpl(text)
    entities_dict = {}
    exclude_keys = ["EXTRA", "ORIGIN TITLE"]
    for ent in doc.ents:
            if ent.label_ not in exclude_keys:
                entities_dict[ent.label_] = ent.text if ent.text else ""
    return entities_dict
