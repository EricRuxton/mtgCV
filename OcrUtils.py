import ImageProcessing as iProc
import SetSymbolProcessing as sProc
import ImageCombiner
import cv2
from pytesseract import pytesseract
import pandas as pd
import re

pytesseractConfig = r'--oem 3 --psm 7'


def ReadTitleFromCard(base64String):
    """Main OCR entry: decode, process, and read card title text."""
    img = iProc.Base64ToOpencv(base64String)
    boxImg, conts = iProc.GetContours(img, contourThreshold=[100, 100])
    if not conts:
        return "No card detected", img

    biggest = conts[0][2]
    finalImage, minXTitle, minYTitle = CropToTitle(biggest, boxImg)
    if finalImage is None:
        return "No title detected", boxImg
    text = pytesseract.image_to_string(finalImage, config=pytesseractConfig, lang='eng')
    firstLine = text.splitlines()[0] if text.strip() else ""

    cleanedText = CleanString(firstLine)

    minX, minY, maxX, maxY = iProc.GetMinAndMaxFromPoints(biggest)
    cardWidth = maxX - minX
    setSymbol = sProc.CropToSetSymbol(boxImg, minXTitle, minYTitle, cardWidth)

    setName = sProc.GuessSet(setSymbol, cleanedText)

    return cleanedText, setName, ImageCombiner.CreateCombinedImage()



def CropToTitle(biggest, boxImg):
    """Filter image to determine region of the title then return boxImg cropped to just that region"""
    minX, minY, maxX, maxY = iProc.GetMinAndMaxFromPoints(biggest)

    height, width, _ = boxImg.shape
    padding = -5

    minX, minY = max(minX - padding, 0), max(minY - padding, 0)
    maxX, maxY = min(maxX + padding, width), min(maxY + padding, height)

    croppedAroundTitle = GetCroppedImage(boxImg, maxX, maxY, minX, minY)

    greyCropped = cv2.cvtColor(croppedAroundTitle, cv2.COLOR_BGR2GRAY)
    _, binaryCropped = cv2.threshold(greyCropped, 127, 255, cv2.THRESH_BINARY)

    boxes, ocr_data = DetectTextRegions(binaryCropped)

    minHeight = 10
    validBoxList = []
    lowestBox = 0
    for (x, y, w, h) in boxes:
        if h < minHeight:
            continue
        validBoxList.append((w, y))
        lowestBox = max(lowestBox, (y + h))

    if not validBoxList:
        return None

    sortedValidBoxList = sorted(validBoxList)

    heightPadding = int(width * 0.003)
    minYTitle = minY + max(0, (sortedValidBoxList[0][1] - heightPadding))

    croppedTighter = GetCroppedImage(boxImg, maxX, (minY + lowestBox + heightPadding), minX, minYTitle)

    ImageCombiner.AddImageToList(croppedTighter)

    return croppedTighter, minX, minYTitle


def GetCroppedImage(boxImg, maxX, maxY, minX, minY):
    croppedImg = boxImg[minY:maxY, minX:maxX]
    contrastedImg = iProc.IncreaseContrast(croppedImg)
    greyImage = cv2.cvtColor(contrastedImg, cv2.COLOR_BGR2GRAY)
    highpassImage = iProc.Highpass(greyImage, 3)
    if iProc.IsTextLight(highpassImage):
        highpassImage = cv2.bitwise_not(highpassImage)
    binaryImg = iProc.ThresholdImage(highpassImage, invert=True)
    cutoff_x = iProc.FindTextRightBoundary(binaryImg)
    croppedAroundTitle = boxImg[minY:maxY, minX:(minX + cutoff_x)]
    return croppedAroundTitle


def DetectTextContours(binary_img, min_area=50):
    """
    Detects text-like regions by finding contours in a binary image.
    Returns bounding boxes sorted from left to right.
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > min_area:  # ignore tiny noise
            boxes.append((x, y, w, h))

    # Sort by horizontal position
    boxes.sort(key=lambda b: b[0])
    return boxes


def DetectTextRegions(img):
    """
    Uses pytesseract to detect text bounding boxes and confidence scores.
    Returns a list of bounding boxes [(x, y, w, h), ...]
    """
    # Convert to grayscale if needed
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Run OCR with detailed output
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DATAFRAME)

    # Filter out low-confidence or empty entries
    data = data.dropna(subset=['text'])
    data = data[data.conf > 80]  # confidence threshold (adjust as needed)

    boxes = []
    for _, row in data.iterrows():
        x, y, w, h = int(row['left']), int(row['top']), int(row['width']), int(row['height'])
        boxes.append((x, y, w, h))

    return boxes, data


def CleanString(text):
    """Removes non-alphanumeric characters from OCR output."""
    return re.sub(r'[^a-zA-Z0-9 ]+', '', text).strip()
