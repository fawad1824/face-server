import onnxruntime as ort
import numpy as np
from PIL import Image
session = ort.InferenceSession("core/model_files/arcface.onnx")

def get_embedding(image_array):
    # Convert to PIL image
    image = Image.fromarray(image_array.astype(np.uint8))
    resized_image = image.resize((112, 112))  # Resize to (112, 112)

    # Convert back to numpy array and normalize
    resized_image_array = np.array(resized_image).astype(np.float32) / 255.0

    # Rearrange dimensions: (112, 112, 3) -> (3, 112, 112)
    rearranged_image_array = np.transpose(resized_image_array, (2, 0, 1))

    # Add batch dimension: (3, 112, 112) -> (1, 3, 112, 112)
    final_image_array = np.expand_dims(rearranged_image_array, axis=0)

    # Get the input name for the model
    input_name = session.get_inputs()[0].name

    # Run the model on the input data
    outputs = session.run(None, {input_name: final_image_array})

    # The outputs variable now contains the results of the inference
    return outputs[0][0]