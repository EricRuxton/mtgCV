import numpy as np
import cv2
import ImageCombiner
import requests

def CropToSetSymbol(boxImg, minX, minY, cardWidth):
    symbolEstimate = boxImg[int(minY +
                                cardWidth * 0.81):int(minY + cardWidth * 0.93), int(minX +
                                                                                    cardWidth * 0.88):int(
        minX + cardWidth * 0.98)]
    ImageCombiner.AddImageToList(symbolEstimate)

    symbol = TightenCropToBox(symbolEstimate)

    return symbol


def TightenCropToBox(crop, blackRatioThresh=0.9):
    # 1. Convert to HSV for color thresholding (same as before)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    # --- estimate background color from center patch ---
    y1, y2 = int(h * 0.25), int(h * 0.75)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    center_patch = hsv[y1:y2, x1:x2]
    mean_hsv = center_patch.mean(axis=(0, 1))
    h_mean, s_mean, v_mean = mean_hsv

    h_tol, s_tol, v_tol = 7, 30, 30  # tune as needed

    lower = np.array([max(h_mean - h_tol, 0),
                      max(s_mean - s_tol, 0),
                      max(v_mean - v_tol, 0)], dtype=np.uint8)
    upper = np.array([min(h_mean + h_tol, 179),
                      min(s_mean + s_tol, 255),
                      min(v_mean + v_tol, 255)], dtype=np.uint8)

    # 2. Mask: white (255) = background color region, black (0) = other stuff
    mask = cv2.inRange(hsv, lower, upper)

    # Optional clean-up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Compute per-row and per-column black ratios
    #    black = 0, white = 255
    row_black_ratio = (mask == 0).sum(axis=1) / float(w)
    col_black_ratio = (mask == 0).sum(axis=0) / float(h)

    # 4. Find vertical bounds: first and last rows that are NOT "mostly black"
    #    i.e., black_ratio < blackRatioThresh
    y_top = None
    for y in range(h):
        if row_black_ratio[y] < blackRatioThresh:
            y_top = y
            break

    y_bottom = None
    for y in range(h - 1, -1, -1):
        if row_black_ratio[y] < blackRatioThresh:
            y_bottom = y
            break

    # 5. Find horizontal bounds: first and last columns that are NOT "mostly black"
    x_left = 0
    for x in range(w):
        if col_black_ratio[x] < blackRatioThresh:
            x_left = x
            break

    x_right = w
    for x in range(w - 1, -1, -1):
        if col_black_ratio[x] < blackRatioThresh:
            x_right = x
            break

    # If we didn't find any non-mostly-black region, bail out
    if None in (y_top, y_bottom):

        # TODO if here it's probably a white card, look for horizontal lines to crop to

        return crop

    # Safety: ensure indices are valid
    if y_bottom < y_top or x_right < x_left:
        return crop

    # 6. Crop using these bounds
    tightened = crop[y_top:y_bottom + 1, x_left:x_right + 1]
    ImageCombiner.AddImageToList(tightened)
    return tightened


def GuessSet(setSymbol, cardName):

    apiCall = "https://api.scryfall.com/cards/search?q=!%22" + cardName.replace(" ", "+") + "%22&unique=prints"
    r = requests.get(apiCall).json()

    if "data" not in r:
        return "unknown"

    if "set_name" not in r["data"][0]:
        return "unknown"

    return r["data"][0]["set_name"]

