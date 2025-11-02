import numpy as np
import cv2

images = []
max_width = 0
total_height = 0


def AddImageToList(img):
    """Adds an image to the global list for visual debugging output."""
    global max_width, total_height
    images.append(img)
    max_width = max(max_width, img.shape[1])
    total_height += img.shape[0]


def CreateCombinedImage():
    """Vertically stacks all debug images into a single output image."""
    final_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    current_y = 0
    for image in images:
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        final_image[current_y:image.shape[0] + current_y, :image.shape[1], :] = image
        current_y += image.shape[0]
    return final_image
