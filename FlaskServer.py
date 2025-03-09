import base64
import re

from flask import Flask, render_template, request
import cv2
import numpy as np
import math

import tensorflow as tf
import tensorflow_hub as hub

from pytesseract import pytesseract

app = Flask(__name__)

# inches / pixel
scale = 0.02450502473611342250789329380908
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ReadCard', methods=['POST'])
def ReadCard():
    if request.method == 'POST':
        img64 = request.form["base64Img"]

        readText = ReadTitleFromCard(img64)
        return f"Read Text: {readText}"
    else:
        return render_template('index.html')


def ReadTitleFromCard(base64String):
    img = base64_to_opencv(base64String)
    # img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

    boxImg, conts = GetContours(img, contourThreshold=[100, 100])

    if len(conts) == 0:
        return "No card detected"

    biggest = conts[0][2]

    croppedImg = CropToCard(boxImg, biggest)

    '''
    preprocessedImg = PreprocessImage(croppedImg)

    model = hub.load(SAVED_MODEL_PATH)

    fake_image = model(preprocessedImg)
    hiresImage = tf.squeeze(fake_image)

    finalImage = tf.clip_by_value(hiresImage, 0, 255)
    finalImage = tf.cast(finalImage, tf.uint8).numpy()
    '''
    # finalImage = IncreaseContrast(finalImage)
    finalImage = IncreaseContrast(croppedImg)

    finalImage = cv2.cvtColor(finalImage, cv2.COLOR_BGR2GRAY)

    finalImage = ThresholdImage(finalImage)

    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract
    text = pytesseract.image_to_string(finalImage)

    return clean_string(text[:-1])


def clean_string(text):
    return re.sub(r'[^a-zA-Z0-9 ]+', '', text).strip()


def base64_to_opencv(base64String):
    # Decode base64 string to bytes
    image_bytes = base64.b64decode(base64String)

    # Convert bytes to a NumPy array
    np_arr = np.frombuffer(image_bytes, np.uint8)

    # Decode the NumPy array into an OpenCV image
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return image


def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (5, 5), sigma) + 127


def GetContours(img, contourThreshold=None, minArea=2000, draw=False, filter=0):
    if contourThreshold is None:
        contourThreshold = [100, 100]
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, contourThreshold[0], contourThreshold[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThresh = cv2.erode(imgDial, kernel, iterations=2)

    contours, hiearachy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            param = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * param, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:  # this will never work since we're expecting it to have errors -> extra corners
                if len(approx) == filter:
                    finalContours.append([len(approx), area, approx, bbox, i])

            else:
                finalContours.append([len(approx), area, approx, bbox, i])

    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
    return img, finalContours


def ThresholdImage(img):
    # Set pixels above the threshold to white (255)
    img[img > 70] = 255

    return img


def CalcSizePx(points, heightPointMinDistPx=50):
    # Sort list so bottom first
    sortedPoints = sorted(points.tolist(), key=lambda x: x[0][1], reverse=True)

    # Calc the two dimensions using them
    length = scale * calcHypotenuse(sortedPoints[0][0], sortedPoints[1][0])
    width = scale * calcHypotenuse(sortedPoints[0][0], sortedPoints[2][0])

    return True, (
        (length, sortedPoints[0][0], sortedPoints[1][0]),
        (width, sortedPoints[0][0], sortedPoints[2][0]))


def calcHypotenuse(pointA, pointB):
    return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)


def CropToCard(img, points):
    minX, minY, maxX, maxY = GetMinAndMaxFromPoints(points)
    height, width, _ = img.shape
    # padding = int(width / 20)
    padding = 0

    minX = max(minX - padding, 0)
    minY = max(minY - padding, 0)
    maxX = min(maxX + padding, width)
    maxY = min(maxY + padding, height)

    crop_img = img[minY:maxY, minX:maxX]

    return crop_img


def IncreaseContrast(finalImage):
    # converting to LAB color space
    lab = cv2.cvtColor(finalImage, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    # Applying CLAHE to L-channel

    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img


def GetMinAndMaxFromPoints(points):
    minX = 10000
    minY = 10000
    maxX = 0
    maxY = 0

    for point in points:
        minX = min(minX, point[0][0])
        minY = min(minY, point[0][1])
        maxX = max(maxX, point[0][0])
        maxY = max(maxY, point[0][1])

    # TODO: calculate how much to add to min values based on card size in px
    return minX, minY + 10, maxX, minY + 50
    # return minX, minY + 10, maxX, minY + 100



def PreprocessImage(img):
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if img.shape[-1] == 4:
        img = img[..., :-1]
    hr_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
    img = tf.image.crop_to_bounding_box(img, 0, 0, hr_size[0], hr_size[1])
    img = tf.cast(img, tf.float32)
    return tf.expand_dims(img, 0)


if __name__ == '__main__':
    app.run(debug=True)
