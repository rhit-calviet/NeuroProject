import os
import pydicom
import numpy as np
from PIL import Image

def convert_dicom_to_jpg(dicom_path, output_folder):
    """
    Reads a DICOM file, extracts a representative axial slice, converts it
    to a JPG image, and saves it.

    This function handles both single-frame (2D) and multi-frame (3D)
    DICOM files. For 3D files, it selects the middle slice.

    Args:
        dicom_path (str): The full path to the input DICOM file.
        output_folder (str): The folder where the output JPG will be saved.

    Returns:
        str: The full path to the newly created JPG file.
        
    Raises:
        Exception: If the DICOM file cannot be read or processed.
    """
    try:
        # Read the DICOM file using pydicom
        dcm = pydicom.dcmread(dicom_path)

        # Access the pixel data from the DICOM file
        pixel_array = dcm.pixel_array

        # If the DICOM contains multiple frames (a 3D volume), select the middle slice.
        # Otherwise, use the single frame.
        if pixel_array.ndim == 3:
            slice_index = pixel_array.shape[0] // 2
            image_frame = pixel_array[slice_index, :, :]
        else:
            image_frame = pixel_array

        # Normalize the pixel values to a 0-255 grayscale range.
        # This is a standard step to make the image viewable.
        image_frame = image_frame.astype(float)
        image_frame = (np.maximum(image_frame, 0) / image_frame.max()) * 255.0
        image_frame = np.uint8(image_frame)

        # Create a Pillow Image object from the numpy array
        img = Image.fromarray(image_frame)
        
        # The analysis models expect a 3-channel RGB image, so convert if necessary.
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Construct a new filename for the output JPG.
        # e.g., 'scan.dcm' -> 'scan.dcm.jpg'
        base_filename = os.path.basename(dicom_path)
        new_filename = f"{base_filename}.jpg"
        output_path = os.path.join(output_folder, new_filename)

        # Save the converted image as a JPG file
        img.save(output_path, 'JPEG')

        return output_path

    except Exception as e:
        print(f"Error converting DICOM file {dicom_path}: {e}")
        # Re-raise the exception to be handled by the Flask app
        raise