import ImageProcessing as iProc
import ImageCombiner
import cv2
from pytesseract import pytesseract
import re

pytesseractConfig = r'--oem 3 --psm 7'


def ReadTitleFromCard(base64String):
    """Main OCR entry: decode, process, and read card title text."""
    img = iProc.Base64ToOpencv(base64String)
    boxImg, conts = iProc.GetContours(img, contourThreshold=[100, 100])
    if not conts:
        return "No card detected", img

    biggest = conts[0][2]
    finalImage = CropToTitle(biggest, boxImg)

    text = pytesseract.image_to_string(finalImage, config=pytesseractConfig, lang='eng')
    firstLine = text.splitlines()[0] if text.strip() else ""
    cleanedText = CleanString(firstLine)

    return cleanedText, ImageCombiner.CreateCombinedImage()


def CropToTitle(biggest, boxImg):
    """Filter image to determine region of the title then return boxImg cropped to just that region"""
    minX, minY, maxX, maxY = iProc.GetMinAndMaxFromPoints(biggest)

    height, width, _ = boxImg.shape
    padding = -5

    minX, minY = max(minX - padding, 0), max(minY - padding, 0)
    maxX, maxY = min(maxX + padding, width), min(maxY + padding, height)

    croppedImg = boxImg[minY:maxY, minX:maxX]

    contrastedImg = iProc.IncreaseContrast(croppedImg)

    greyImage = cv2.cvtColor(contrastedImg, cv2.COLOR_BGR2GRAY)

    highpassImage = iProc.Highpass(greyImage, 3)

    if iProc.IsTextLight(highpassImage):
        highpassImage = cv2.bitwise_not(highpassImage)

    binaryImg = iProc.ThresholdImage(highpassImage, invert=True)
    cutoff_x = iProc.FindTextRightBoundary(binaryImg)
    return boxImg[minY:maxY, minX:(minX + cutoff_x)]


def CleanString(text):
    """Removes non-alphanumeric characters from OCR output."""
    return re.sub(r'[^a-zA-Z0-9 ]+', '', text).strip()
