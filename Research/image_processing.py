"""
=========================================================
image_processing.py

Image Processing Module

Functions:
    • Image -> Bitstream
    • Bitstream -> Image

Author : Ojaswi Chand
=========================================================
"""

import numpy as np
from PIL import Image
import os


#########################################################
# Load Image
#########################################################

def load_image(image_path):
    """
    Load an image and convert it to grayscale.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path)

    image = image.convert("L")

    return image


#########################################################
# Image -> Bitstream
#########################################################

def image_to_bits(image_path):
    """
    Convert grayscale image into binary bitstream.

    Returns
    -------
    bits : ndarray
    image_shape : tuple
    """

    image = load_image(image_path)

    image_array = np.array(image, dtype=np.uint8)

    image_shape = image_array.shape

    flat_pixels = image_array.flatten()

    bits = np.unpackbits(flat_pixels)

    return bits.astype(np.uint8), image_shape


#########################################################
# Bitstream -> Image
#########################################################

def bits_to_image(bits, image_shape):
    """
    Convert bitstream back into image.
    """

    bits = np.asarray(bits, dtype=np.uint8)

    total_pixels = image_shape[0] * image_shape[1]

    required_bits = total_pixels * 8

    #####################################################
    # Remove Extra Padding Bits
    #####################################################

    bits = bits[:required_bits]

    #####################################################
    # Add Missing Bits (if required)
    #####################################################

    if len(bits) < required_bits:

        padding = required_bits - len(bits)

        bits = np.concatenate([
            bits,
            np.zeros(padding, dtype=np.uint8)
        ])

    #####################################################

    pixels = np.packbits(bits)

    image = pixels.reshape(image_shape)

    return image


#########################################################
# Save Image
#########################################################

def save_image(image_array, output_path):
    """
    Save reconstructed image.
    """

    image = Image.fromarray(image_array.astype(np.uint8))

    image.save(output_path)


#########################################################
# Image Information
#########################################################

def image_info(image_path):
    """
    Print image information.
    """

    image = load_image(image_path)

    arr = np.array(image)

    print("\nImage Information")
    print("--------------------------")
    print("Width  :", arr.shape[1])
    print("Height :", arr.shape[0])
    print("Pixels :", arr.size)
    print("Bits   :", arr.size * 8)
    print("--------------------------")


#########################################################
# Compare Images
#########################################################

def image_difference(original, reconstructed):
    """
    Absolute pixel difference.
    """

    original = np.asarray(original, dtype=np.int16)

    reconstructed = np.asarray(
        reconstructed,
        dtype=np.int16
    )

    return np.abs(original - reconstructed)


#########################################################
# Test
#########################################################

if __name__ == "__main__":

    IMAGE = "images/image.png"

    bits, shape = image_to_bits(IMAGE)

    print("Image Shape :", shape)

    print("Total Bits :", len(bits))

    reconstructed = bits_to_image(bits, shape)

    save_image(
        reconstructed,
        "images/test_reconstructed.png"
    )

    image_info(IMAGE)

    print("\nTest Completed Successfully.")