import cv2
import numpy as np
from keras.models import load_model

from search import WordSearch
from utils import *


# load model
saved_model_path = 'model/letters_classifier.keras'
model = load_model(saved_model_path)

# input files
puzzle = 'assets/puzzle1.png'
puzzleWords = 'assets/puzzle1words.txt'

# input grid dimensions
nRows, nCols = (10, 10)

# image dimensions
hInputImg, wInputImg = (500, 500)
hDatasetImg, wDatasetImg = (28, 28)

# thresholds for contour's width to height ratio
contourMin = 0.25
contourMax = 1.00


def predictionPreprocessing(img):
    img = cv2.bitwise_not(img)
    img = img / 255
    return img


def getContours(imgPath, inputImgH, inputImgW):
    '''Read the image and return the found contours, resized image & gray scale image'''

    img = cv2.imread(imgPath)
    img = cv2.resize(img, (inputImgH, inputImgW))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU |
                           cv2.THRESH_BINARY_INV)[1]
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilation = cv2.dilate(thresh, rect_kernel, iterations=1)
    contours = cv2.findContours(
        dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    return contours, img, gray


def getCoordsAndLabels(contours, grayImg, datasetImgH, datasetImgW, trainedModel):
    '''Loop through the contours, recognize the characters and return an array of co-ordinates and class labels of the characters'''

    # [(x, y, 'P')] -> array
    coordsAndLabels = []
    alphabet_mapping = getAlphabetMapping()

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        if contourMin <= (w/h) <= contourMax:

            letter = grayImg[y:y+h, x:x+w]
            letter = cv2.resize(letter, (datasetImgH, datasetImgW))
            letter = predictionPreprocessing(letter)
            letter = letter.reshape(1, datasetImgH, datasetImgW, 1)

            # prediction
            prediction = trainedModel.predict(letter)
            # confidence = np.amax(prediction)

            classId = np.argmax(prediction, axis=1)[0]
            className = alphabet_mapping[classId]

            coordsAndLabels.append((x, y, className))

    return coordsAndLabels


def normalizeYAndSort(coordsLabels):
    '''Normalize (make all y co-ordinate values of the row the same) the y co-ordinates to a same value for the entire row and sort by Y first then by X'''

    y_for_row = coordsLabels[0][1]
    for i, cnt in enumerate(coordsLabels):
        x, y, className = cnt

        # if beginning of row (col 1) then change 'y_for_row' to y value at that position
        if i % nCols == 0:
            y_for_row = y

        # else make y as the set 'y_for_row' value
        else:
            y = y_for_row

        # replace the tuple in the array
        coordsLabels[i] = (x, y, className)

    # return sorted by Y first then by X
    return sorted(
        coordsLabels, key=lambda k: [k[1], k[0]])


def getGrid(coordsLabels, cols):
    '''Create a grid with only labels as per required grid dimensions'''
    grid = []
    for i in range(0, len(coordsLabels), cols):
        row = list(map(lambda x: x[2], coordsLabels[i:i+cols]))
        grid.append(row)

    return grid


if __name__ == '__main__':

    contours, img, grayImg = getContours(puzzle, hInputImg, wInputImg)

    coordsAndLabels = getCoordsAndLabels(
        contours, grayImg, hDatasetImg, wDatasetImg, model)
    coordsAndLabels = normalizeYAndSort(coordsAndLabels)

    grid = getGrid(coordsAndLabels, nCols)
    words = readWordsFromFile(puzzleWords)

    WS = WordSearch(words, grid)
    foundWords, summary = WS.findAll()

    print(foundWords)
    print(summary)
