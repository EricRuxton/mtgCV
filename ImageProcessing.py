import base64
import cv2
import numpy as np


def IsTextLight(img):
    """Detects whether text is lighter than its background."""
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    mask = cv2.bitwise_not(dilated)
    text_pixels = gray[mask == 0]
    background_pixels = gray[mask != 0]

    if len(text_pixels) and len(background_pixels):
        return np.mean(text_pixels) > np.mean(background_pixels)

    return False


def RemoveHorizontalLines(img):
    """Removes horizontal black bars or lines from a binary image."""
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cleaned = cv2.bitwise_not(cv2.subtract(img, detected_lines))
    return cleaned


def Base64ToOpencv(base64String):
    """Converts a base64-encoded string into an OpenCV image."""
    image_bytes = base64.b64decode(base64String)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def GetContours(img, contourThreshold=None, minArea=2000, draw=False, filter=0):
    """Finds significant external contours, optionally drawing them."""
    if contourThreshold is None:
        contourThreshold = [100, 100]
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, contourThreshold[0], contourThreshold[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThresh = cv2.erode(imgDial, kernel, iterations=2)

    contours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = [
        [len(approx), area, approx, cv2.boundingRect(approx), i]
        for i in contours
        if (area := cv2.contourArea(i)) > minArea
        for param in [cv2.arcLength(i, True)]
        for approx in [cv2.approxPolyDP(i, 0.02 * param, True)]
        if filter == 0 or len(approx) == filter
    ]

    finalContours.sort(key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
    return img, finalContours


def Highpass(img, sigma):
    """Applies a high-pass filter to emphasize edges."""
    return img - cv2.GaussianBlur(img, (0, 0), sigma) + 127
    # return img - cv2.GaussianBlur(img, (5, 5, sigma) + 127


def ThresholdImage(img, invert=False):
    """Applies binary thresholding to an image."""
    white = 255
    threshold = 100

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(img, threshold, white, thresh_type)
    return binary


def IncreaseContrast(finalImage):
    """Enhances local contrast using CLAHE."""
    lab = cv2.cvtColor(finalImage, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def GetMinAndMaxFromPoints(points):
    """Extracts min/max X,Y bounds from a set of contour points."""
    minX = min(point[0][0] for point in points)
    minY = min(point[0][1] for point in points)
    maxX = max(point[0][0] for point in points)
    # maxY = max(point[0][1] for point in points)  # Unused

    cardWidth = maxX - minX
    widthMultiplicationConst = 0.08
    xOffset = int(cardWidth * widthMultiplicationConst)

    # topOffset = 22
    # heightMultiplicationConst = 0.11
    topOffset = int(cardWidth * 0.075)
    # print("topOffset: " + str(topOffset))
    heightMultiplicationConst = 0.08
    cardBotOffsetPx = int(cardWidth * heightMultiplicationConst + topOffset)

    return minX + xOffset, minY + topOffset, maxX - xOffset, minY + cardBotOffsetPx


def CropToCard(img, points):
    """Crops the image to the detected card region."""
    minX, minY, maxX, maxY = GetMinAndMaxFromPoints(points)
    height, width, _ = img.shape
    padding = -5

    minX, minY = max(minX - padding, 0), max(minY - padding, 0)
    maxX, maxY = min(maxX + padding, width), min(maxY + padding, height)

    return img[minY:maxY, minX:maxX]


def SmoothBinaryImage(binary_img):
    """
    Removes small noise from a binary (black/white) image
    using morphological opening and closing.
    """
    kernel = np.ones((2, 2), np.uint8)

    # Remove small white specks (opening)
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill small black holes inside text (closing)
    smoothed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    return smoothed


def FindTextRightBoundary(binary_img, density_threshold_ratio=0.05, gap_tolerance=10):
    """
    Detects the rightmost column that contains text (excluding logos/whitespace).
    Returns the x-coordinate to crop at.

    density_threshold_ratio: ratio of max pixel density considered 'text'
    gap_tolerance: number of consecutive blank columns before we assume text ended
    """
    # Ensure text is white on black
    white_pixels = np.sum(binary_img == 255)
    black_pixels = np.sum(binary_img == 0)
    if black_pixels < white_pixels:
        binary_img = cv2.bitwise_not(binary_img)

    h, w = binary_img.shape
    col_sums = np.sum(binary_img == 255, axis=0)
    max_val = np.max(col_sums)
    threshold = max_val * density_threshold_ratio

    # Find columns with text
    text_cols = np.where(col_sums > threshold)[0]
    if len(text_cols) == 0:
        return w  # fallback: no text detected, return full width

    # Scan from the rightmost text column and detect a sustained drop (whitespace/symbol region)
    consecutive_blank = 0
    for x in range(0, w):
        if col_sums[x] <= threshold:
            consecutive_blank += 1
            if consecutive_blank >= gap_tolerance:
                return x - gap_tolerance
        else:
            consecutive_blank = 0

    return w  # no clear drop found, assume full width
