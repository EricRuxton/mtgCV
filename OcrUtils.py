import ImageProcessing as iProc
import ImageCombiner
import cv2
from pytesseract import pytesseract
import re


def ReadTitleFromCard(base64String):
    """Main OCR entry: decode, process, and read card title text."""
    img = iProc.Base64ToOpencv(base64String)
    boxImg, conts = iProc.GetContours(img, contourThreshold=[100, 100])
    if not conts:
        return "No card detected", img

    biggest = conts[0][2]
    finalImage = PrepareForOpticalCharacterRecognition(biggest, boxImg)
    ImageCombiner.AddImageToList(finalImage)

    # TODO detect whitespace between words

    text = pytesseract.image_to_string(finalImage)
    firstLine = text.splitlines()[0] if text.strip() else ""
    cleanedText = CleanString(firstLine)

    return cleanedText, ImageCombiner.CreateCombinedImage()


def PrepareForOpticalCharacterRecognition(biggest, boxImg):
    """Preprocessing chain for OCR: crop, enhance, high-pass, threshold, and clean."""
    croppedImg = iProc.CropToCard(boxImg, biggest)
    ImageCombiner.AddImageToList(croppedImg)

    contrastedImg = iProc.IncreaseContrast(croppedImg)
    ImageCombiner.AddImageToList(contrastedImg)

    greyImage = cv2.cvtColor(contrastedImg, cv2.COLOR_BGR2GRAY)
    ImageCombiner.AddImageToList(greyImage)

    numCascades = 1
    highpassImage = greyImage
    for _ in range(numCascades):
        highpassImage = iProc.Highpass(highpassImage, 3)
        ImageCombiner.AddImageToList(highpassImage)

    if iProc.IsTextLight(highpassImage):
        highpassImage = cv2.bitwise_not(highpassImage)

    binaryImg = iProc.ThresholdImage(highpassImage, invert=True)
    noLineImg = iProc.RemoveHorizontalLines(binaryImg)

    return noLineImg


def CleanString(text):
    """Removes non-alphanumeric characters from OCR output."""
    return re.sub(r'[^a-zA-Z0-9 ]+', '', text).strip()
