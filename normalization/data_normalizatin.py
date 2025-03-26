import numpy as np
import cv2


def normalize_image(image):
    """
    Normalize an image using the average color balance and standard deviation of the pixels.

    Args:
        image (numpy.ndarray): Input image in COCO format (H x W x C).

    Returns:
        numpy.ndarray: Normalized image (float32).
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")

    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True)
    std[std == 0] = 1

    normalized_image = (image - mean) / std
    return normalized_image


def save_normalized_image(normalized_image, output_path):
    """
    Rescale and save a normalized image as a JPG.

    Args:
        normalized_image (numpy.ndarray): Normalized image.
        output_path (str): Path to save the image (e.g., 'output.jpg').
    """
    rescaled = normalized_image - normalized_image.min()
    rescaled /= rescaled.max()
    rescaled *= 255.0

    output_image = rescaled.astype(np.uint8)

    cv2.imwrite(output_path, output_image)


if __name__ == "__main__":
    image_path = "./graphene_image.png"
    image = cv2.imread(image_path)

    if image is not None:
        normalized_image = normalize_image(image)
        save_normalized_image(normalized_image, "normalized_graphene.jpg")
        print("Image normalized and saved as JPG.")
    else:
        print("Failed to load the image.")
