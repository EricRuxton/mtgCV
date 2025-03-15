import math
from threading import Thread
from tkinter import *
import cv2
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt


from PIL import Image
from pytesseract import pytesseract

# webcam = False
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
path = "10.jpg"
cap = cv2.VideoCapture(0)

# inches / pixel
scale = 0.02450502473611342250789329380908

cap.set(10, 160)
cap.set(3, 1200)
cap.set(4, 1600)

master = Tk()
master.geometry('560x380')
master.title("CardChecker")

Label(master, text='Config options', font=("Arial", 12)).place(x=5, y=0)
useWebcam = BooleanVar()
useWebcam.set(True)
webcamCheckbox = Checkbutton(master, text="Use Webcam", variable=useWebcam)
webcamCheckbox.place(x=10, y=25)

useGreyscale = BooleanVar()
useGreyscale.set(True)
greyscaleCheckbox = Checkbutton(master, text="Use Greyscale", variable=useGreyscale)
greyscaleCheckbox.place(x=10, y=45)

useBlur = BooleanVar()
useBlur.set(False)
blurCheckbox = Checkbutton(master, text="Highpass filter", variable=useBlur)
blurCheckbox.place(x=150, y=45)

useMask = BooleanVar()
useMask.set(False)
maskCheckbox = Checkbutton(master, text="Use Mask", variable=useMask)
maskCheckbox.place(x=10, y=65)

increaseContrast = BooleanVar()
increaseContrast.set(True)
contrastCheckbox = Checkbutton(master, text="Increase Contrast", variable=increaseContrast)
contrastCheckbox.place(x=150, y=65)

isolateTitle = BooleanVar()
isolateTitle.set(True)
isolateTitleCheckbox = Checkbutton(master, text="Isolate Title", variable=isolateTitle)
isolateTitleCheckbox.place(x=10, y=85)

useEdge = BooleanVar()
useEdge.set(False)
useEdgeCheckbox = Checkbutton(master, text="Use canny filter", variable=useEdge)
useEdgeCheckbox.place(x=150, y=85)

useUpscale = BooleanVar()
useUpscale.set(False)
useEdgeCheckbox = Checkbutton(master, text="Use upscale", variable=useUpscale)
useEdgeCheckbox.place(x=150, y=105)

Label(master, text='Clip lim', font=("Arial", 12)).place(x=280, y=45)

Label(master, text='y', font=("Arial", 12)).place(x=465, y=70)
Label(master, text='Tile grid size x', font=("Arial", 12)).place(x=345, y=70)


clipLimitEntry = Entry(master, font=("Arial", 12), width=5)
clipLimitEntry.insert(END, "2.0")
clipLimitEntry.place(x=285, y=65)

Label(master, text='Tile grid size x', font=("Arial", 12)).place(x=345, y=45)
xCoordEntry = Entry(master, font=("Arial", 12), width=2)
xCoordEntry.insert(END, "8")
xCoordEntry.place(x=433, y=65)

Label(master, text='y', font=("Arial", 12)).place(x=465, y=45)
yCoordEntry = Entry(master, font=("Arial", 12), width=2)
yCoordEntry.insert(END, "8")
yCoordEntry.place(x=463, y=65)

sigmaEntry = Entry(master, font=("Arial", 12), width=5)
sigmaEntry.insert(END, "3")
sigmaEntry.place(x=285, y=90)

kSizeEntryOne = Entry(master, font=("Arial", 12), width=2)
kSizeEntryOne.insert(END, "0")
kSizeEntryOne.place(x=433, y=90)

kSizeEntryTwo = Entry(master, font=("Arial", 12), width=2)
kSizeEntryTwo.insert(END, "0")
kSizeEntryTwo.place(x=463, y=90)

thingToAdd = Entry(master, font=("Arial", 12), width=4)
thingToAdd.insert(END, "127")
thingToAdd.place(x=500, y=90)

useThreshold = BooleanVar()
useThreshold.set(False)
useThresholdCheckbox = Checkbutton(master, text="Use upscale", variable=useThreshold)
useThresholdCheckbox.place(x=150, y=105)

Label(master, text='Threshold', font=("Arial", 12)).place(x=415, y=10)
thresholdEntry = Entry(master, font=("Arial", 12), width=4)
thresholdEntry.insert(END, "127")
thresholdEntry.place(x=500, y=10)

'''
rotateImage = BooleanVar()
rotateImage.set(True)
rotateCheckbox = Checkbutton(master, text="Rotate Image", variable=rotateImage)
rotateCheckbox.place(x=10, y=85)
'''

Label(master, text='Display options', font=("Arial", 12)).place(x=5, y=105)

showInitialPicture = BooleanVar()
showInitialPicture.set(True)
initPicCheckbox = Checkbutton(master, text="Show initial picture", variable=showInitialPicture)
initPicCheckbox.place(x=10, y=125)

showCanny = BooleanVar()
showCanny.set(False)
cannyCheckbox = Checkbutton(master, text="Show canny", variable=showCanny)
cannyCheckbox.place(x=10, y=145)

showCardPoints = BooleanVar()
showCardPoints.set(False)
contourCheckbox = Checkbutton(master, text="Show card points", variable=showCardPoints)
contourCheckbox.place(x=10, y=165)

showCroppedPicture = BooleanVar()
showCroppedPicture.set(False)
croppedPictureCheckbox = Checkbutton(master, text="Show cropped picture", variable=showCroppedPicture)
croppedPictureCheckbox.place(x=10, y=185)

showCroppedPicture = BooleanVar()
showCroppedPicture.set(False)
croppedPictureCheckbox = Checkbutton(master, text="Show cropped picture", variable=showCroppedPicture)
croppedPictureCheckbox.place(x=10, y=185)

showContrastComparison = BooleanVar()
showContrastComparison.set(True)
croppedPictureCheckbox = Checkbutton(master, text="Show contrast comparison", variable=showContrastComparison)
croppedPictureCheckbox.place(x=180, y=125)

showFinalImage = BooleanVar()
showFinalImage.set(True)
finalImageCheckbox = Checkbutton(master, text="Show final image", variable=showFinalImage)
finalImageCheckbox.place(x=180, y=145)

Label(master, text='File name', font=("Arial", 12)).place(x=150, y=5)
fileNameEntry = Entry(master, font=("Arial", 12))
fileNameEntry.insert(END, "card.jpg")
fileNameEntry.place(x=150, y=27)
'''
Label(master, text='Mask filters', font=("Arial", 12)).place(x=150, y=50)
maskFilterEntry = Entry(master, font=("Arial", 12))
maskFilterEntry.place(x=150, y=77)
'''
Label(master, text='Card text', font=("Arial", 12)).place(x=5, y=220)
cardTextEntry = Text(master, font=("Arial", 12), width=50, height=7)
cardTextEntry.place(x=5, y=247)


def Initialize():
    thread = Thread(target=CameraLoop)
    thread.start()

    master.mainloop()


def CameraLoop():
    while True:
        if useWebcam.get():
            success, img = cap.read()
        else:
            img = cv2.imread(fileNameEntry.get())
            img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

        # boxImg = GetBox(img, showCanny=False)
        if showInitialPicture.get():
            cv2.imshow("image", img)
            cv2.waitKey()

        boxImg, conts = GetContours(img, contourThreshold=[100, 100])

        if len(conts) == 0:
            continue
            # return

        biggest = conts[0][2]

        if showCardPoints.get():
            # cv2.drawContours(img, conts[0][4], -1, (0, 0, 255), 3)
            DrawPoints(img, biggest)

        '''
        angle = GetRotationAngle(biggest)
        rotatedImg = RotateImage(boxImg, angle)
        
        cv2.imshow("rotated image", rotatedImg)
        cv2.waitKey()
        '''

        croppedImg = CropToCard(boxImg, biggest)

        if showCroppedPicture.get():
            cv2.imshow("cropped image", croppedImg)
            cv2.waitKey()

        finalImage = croppedImg
        
        if useUpscale.get():
            preprocessedImg = PreprocessImage(croppedImg)
            model = hub.load(SAVED_MODEL_PATH)

            fake_image = model(preprocessedImg)
            hiresImage = tf.squeeze(fake_image)

            finalImage = tf.clip_by_value(hiresImage, 0, 255)
            finalImage = tf.cast(finalImage, tf.uint8).numpy()

        if increaseContrast.get():
            finalImage = IncreaseContrast(finalImage)
        if useGreyscale.get():
            finalImage = cv2.cvtColor(finalImage, cv2.COLOR_BGR2GRAY)
        if useEdge.get():
            finalImage = cv2.Canny(finalImage, 100, 100)
            '''
            ddepth = cv2.CV_16S
            grad_x = cv2.Scharr(finalImage, ddepth, 1, 0)
            grad_y = cv2.Scharr(finalImage, ddepth, 0, 1)

            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            finalImage = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            '''
        if useBlur.get():
            finalImage = highpass(finalImage, 3)

        if useMask.get():
            finalImage = ThresholdImage(finalImage)

        path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pytesseract.tesseract_cmd = path_to_tesseract
        text = pytesseract.image_to_string(finalImage)

        # Displaying the extracted text
        # print(text[:-1])
        cardTextEntry.delete(1.0, END)
        cardTextEntry.insert(END, text[:-1])
        if showFinalImage.get():
            cv2.imshow("Threshold image", finalImage)
            cv2.waitKey()

        '''
        calcSuccess, measurements = CalcSizePx(biggest)
    
        if not calcSuccess:
            # draw error text on image
            errorImg = cv2.putText(img, "Error, please make",
                                   (10, 200),
                                   cv2.FONT_HERSHEY_PLAIN,
                                   7,
                                   (0,0,255),
                                   5,
                                   2)
            errorImg = cv2.putText(errorImg, "sure box view is",
                                   (10, 300),
                                   cv2.FONT_HERSHEY_PLAIN,
                                   7,
                                   (0,0,255),
                                   5,
                                   2)
            errorImg = cv2.putText(errorImg, "isometric",
                                   (10, 400),
                                   cv2.FONT_HERSHEY_PLAIN,
                                   7,
                                   (0,0,255),
                                   5,
                                   2)
            errorImg = cv2.resize(errorImg, (0, 0), None, 0.5, 0.5)
            cv2.imshow('Error', errorImg)
            cv2.waitKey()
            continue
    
        print(measurements)
        measuredImg = cv2.arrowedLine(img, measurements[0][1], measurements[0][2],
                                      (0, 0, 255), 10)
    
        measuredImg = cv2.arrowedLine(measuredImg, measurements[1][1], measurements[1][2],
                                      (0, 255, 0), 10)
        
        measuredImg = cv2.arrowedLine(measuredImg, measurements[2][2], measurements[2][1],
        
        measuredImg = cv2.putText(measuredImg, f"L:{round(measurements[0][0],2)}",
                               (10, 75),
                               cv2.FONT_HERSHEY_PLAIN,
                               6,
                               (0, 0, 255),
                               5,
                               2)
        measuredImg = cv2.putText(measuredImg, f"W:{round(measurements[1][0],2)}",
                               (10, 150),
                               cv2.FONT_HERSHEY_PLAIN,
                               6,
                               (0, 255, 0),
                               5,
                               2)
        
        measuredImg = cv2.putText(measuredImg, f"H:{round(measurements[2][0],2)}",
                               (10, 225),
                               cv2.FONT_HERSHEY_PLAIN,
                               6,
                               (255, 0, 0),
                               5,
                               2)
        
        measuredImg = cv2.resize(measuredImg, (0, 0), None, 0.5, 0.5)
        cv2.imshow('Measured', measuredImg)
        cv2.waitKey()
        '''


def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (int(kSizeEntryOne.get()), int(kSizeEntryTwo.get())),
                                  float(sigmaEntry.get())) + int(thingToAdd.get())
    # return img - cv2.GaussianBlur(img, (5, 5), 1) + 127 - BAD


def GetContours(img, contourThreshold=None, minArea=2000, draw=False, filter=0):
    if contourThreshold is None:
        contourThreshold = [100, 100]
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, contourThreshold[0], contourThreshold[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThresh = cv2.erode(imgDial, kernel, iterations=2)
    if showCanny.get(): cv2.imshow('Canny', imgThresh)

    contours, hiearachy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            param = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * param, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:  # this will never work since we're expecting it to have errors -> extra corners
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox, i])

            else:
                finalCountours.append([len(approx), area, approx, bbox, i])

    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalCountours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
    return img, finalCountours


def ThresholdImage(img):
    # Set pixels above the threshold to white (255)
    img[img > int(thresholdEntry.get())] = 255

    return img


def MaskImage(img):
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask = img
    filters = [
        ([70, 70, 70], [255, 255, 255])
    ]
    for (lower, upper) in filters:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(img, lower, upper)
        '''
        output = output | mask
        # cv2.imshow('Canny', output)
        # cv2.waitKey()
    # show the images
    # cv2.imshow("images", np.hstack([image, output]))
    realOutput = cv2.bitwise_and(img, img, mask=output)

    imgCanny = cv2.Canny(realOutput, 100, 100)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(realOutput, kernel, iterations=3)
    # imgThresh = cv2.erode(imgDial, kernel, iterations=2)
    '''
    return mask


def GetBox(img):
    imgBlur = cv2.GaussianBlur(img, (5, 5), 1)
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    filters = [
        ([0, 0, 0], [50, 50, 50])
        # ([220, 220, 220], [255, 255, 255]),
    ]
    for (lower, upper) in filters:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(imgBlur, lower, upper)
        output = output | mask
        # cv2.imshow('Canny', output)
        # cv2.waitKey()
    # show the images
    # cv2.imshow("images", np.hstack([image, output]))
    realOutput = cv2.bitwise_not(imgBlur, imgBlur, mask=output)
    if showCanny: cv2.imshow('Canny', realOutput)
    return realOutput


def GetRotationAngle(points):
    sortedPoints = sorted(points.tolist(), key=lambda x: x[0][1])

    minPoint = sortedPoints[0][0]
    maxPoint = sortedPoints[1][0]
    if sortedPoints[0][0][0] < sortedPoints[1][0][0]:
        maxPoint = sortedPoints[0][0]
        minPoint = sortedPoints[1][0]

    adjacent = maxPoint[0] - minPoint[0]
    opposite = maxPoint[1] - minPoint[1]

    angle = math.atan(opposite / adjacent)

    return angle


def RotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def CalcSizePx(points, heightPointMinDistPx=50):
    # Sort list so bottom first
    sortedPoints = sorted(points.tolist(), key=lambda x: x[0][1], reverse=True)
    # Verify that other 2 points are in different directions
    '''
    if sortedPoints[0][0] > sortedPoints[1][0]:
        if sortedPoints[0][0] < sortedPoints[2][0]:
            pass
        else:
            return False, None
    else:
        if sortedPoints[0][0] > sortedPoints[2][0]:
            pass
        else:
            return False, None
    '''

    # Calc the two dimensions using them
    length = scale * calcHypotenuse(sortedPoints[0][0], sortedPoints[1][0])
    witdth = scale * calcHypotenuse(sortedPoints[0][0], sortedPoints[2][0])

    '''
    # Find the closest point to the one of other two on the x-axis
    minDist = sys.maxsize
    closestPointSide = -1
    closestPointIndex = -1

    for x in range(3, len(sortedPoints)):
        distance1 = abs(sortedPoints[1][0][0] - sortedPoints[x][0][0])
        distance2 = abs(sortedPoints[2][0][0] - sortedPoints[x][0][0])

        if distance1 < minDist:
            yDist = abs(sortedPoints[1][0][1] - sortedPoints[x][0][1])
            if yDist > heightPointMinDistPx:
                closestPointSide = 1
                closestPointIndex = x
                minDist = distance1

        if distance2 < minDist:
            yDist = abs(sortedPoints[2][0][1] - sortedPoints[x][0][1])
            if yDist > heightPointMinDistPx:
                closestPointSide = 2
                closestPointIndex = x
                minDist = distance2

    # Get height from that
    if closestPointIndex == -1:
        return False, None

    height = scale * calcHypotenuse(sortedPoints[closestPointIndex][0], sortedPoints[closestPointSide][0])
'''
    return True, (
        (length, sortedPoints[0][0], sortedPoints[1][0]),
        (witdth, sortedPoints[0][0], sortedPoints[2][0]))
    # (height, sortedPoints[closestPointIndex][0], sortedPoints[closestPointSide][0]))


def calcHypotenuse(pointA, pointB):
    return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)


def DrawPoints(img, points):
    for point in points:
        center = (point.tolist()[0][0], point.tolist()[0][1])
        img = cv2.circle(img, center, 5, (255, 0, 0), 5)
    # img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow('Points', img)


def CropToCard(img, points):
    minX, minY, maxX, maxY = GetMinAndMaxFromPoints(points)
    height, width, _ = img.shape
    # padding = int(width / 20)
    padding = 0

    minX = max(minX - padding, 0)
    minY = max(minY - padding, 0)
    maxX = min(maxX + padding, width)
    maxY = min(maxY + padding, height)

    # img = cv2.circle(img, (minX,minY), 5, (255, 0, 0), 5)
    # img = cv2.circle(img, (maxX,maxY), 5, (255, 0, 0), 5)

    crop_img = img[minY:maxY, minX:maxX]

    # crop_img = cv2.resize(crop_img, (0, 0), None, 0.5, 0.5)
    return crop_img


def IncreaseContrast(finalImage):
    # converting to LAB color space
    lab = cv2.cvtColor(finalImage, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    # Applying CLAHE to L-channel

    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=float(clipLimitEntry.get()), tileGridSize=(int(xCoordEntry.get()),
                                                                                 int(yCoordEntry.get())))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    result = np.hstack((finalImage, enhanced_img))
    if showContrastComparison.get():
        cv2.imshow('Contrast Comparison', result)
        cv2.waitKey()

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

    if isolateTitle.get():
        return minX, minY + 10, maxX, minY + 50
        # return minX, minY + 10, maxX, minY + 100

    return minX, minY, maxX, maxY


def PreprocessImage(img):
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if img.shape[-1] == 4:
        img = img[..., :-1]
    hr_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
    img = tf.image.crop_to_bounding_box(img, 0, 0, hr_size[0], hr_size[1])
    img = tf.cast(img, tf.float32)
    return tf.expand_dims(img, 0)


if __name__ == "__main__":
    Initialize()
