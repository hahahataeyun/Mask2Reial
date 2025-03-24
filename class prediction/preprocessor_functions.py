import os

import cv2
import numpy as np


def get_contrasts_from_dir(image_directory):
    contrasts = []

    background_color = None

    for filename in os.listdir(image_directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_directory, filename)
            image = cv2.imread(image_path)
            
            if background_color is None:
                background_color = calculate_background_color(image)
            
            contrast_image = (image - background_color) / background_color
            contrasts.append(contrast_image)

    return contrasts


# def remove_vignette(
#     image,
#     flatfield,
#     max_background_value: int = 241,
# ):
#     """Removes the Vignette from the Image

#     Args:
#         image (NxMx3 Array): The Image with the Vignette
#         flatfield (NxMx3 Array): the Flat Field in RGB
#         max_background_value (int): the maximum value of the background

#     Returns:
#         (NxMx3 Array): The Image without the Vignette
#     """
#     image_no_vigentte = image / flatfield * cv2.mean(flatfield)[:-1]
#     image_no_vigentte[image_no_vigentte > max_background_value] = max_background_value
#     return np.asarray(image_no_vigentte, dtype=np.uint8)


def calculate_background_color(img, radius=5):
    masks = []

    for i in range(3):
        img_channel = img[:, :, i]
        mask = cv2.inRange(img_channel, 20, 230)
        hist = cv2.calcHist([img_channel], [0], mask, [256], [0, 256])
        hist_mode = np.argmax(hist)
        thresholded_image = cv2.inRange(
            img_channel, int(hist_mode - radius), int(hist_mode + radius)
        )
        background_mask_channel = cv2.erode(
            thresholded_image, np.ones((3, 3)), iterations=3
        )
        masks.append(background_mask_channel)

    final_mask = cv2.bitwise_and(masks[0], masks[1])
    final_mask = cv2.bitwise_and(final_mask, masks[2])

    return cv2.mean(img, mask=final_mask)[:3]


def instance_masking(projected_images, maskformer_output):
    # maskformer_output has a form of list[dict]
    # each dict has the results for one image. The dict contains the following keys:
    # * "sem_seg":
    #     A Tensor that represents the
    #     per-pixel segmentation prediced by the head.
    #     The prediction has shape KxHxW that represents the logits of
    #     each class for each pixel.
    # * "panoptic_seg":
    #     A tuple that represent panoptic output
    #     panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
    #     segments_info (list[dict]): Describe each segment in `panoptic_seg`.
    #         Each dict contains keys "id", "category_id", "isthing".

    masked_images = []

    # pred_mask_list is a list of mask predictions(KxHxW) for each image
    pred_masks_list = [output['sem_seg'] for output in maskformer_output]


    for image, pred_mask in zip(projected_images, pred_masks_list):
        # Get the most probable class for each pixel    
        most_probable_class = np.argmax(pred_mask, axis=0)
        # most_probable_class has shape (H, W) and contains the index of the most probable class for each pixel

        # Expand dimensions to match the image shape
        most_probable_class_expanded = np.expand_dims(most_probable_class, axis=2)
        # Adds a new dimension (axis=2), transforming the shape from (H, W) to (H, W, 1)

        # Concatenate the image with the most probable class
        concatenated_image = np.concatenate((image, most_probable_class_expanded), axis=2)
        # Concatenate along the channel axis (axis=2) to create a new image with shape (H, W, 4)

        masked_images.append(concatenated_image)

    return masked_images




