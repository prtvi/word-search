import cv2
import numpy as np
from keras.models import load_model

from search import WordSearch
from utils import *


# load model
saved_model_path = 'model/letters_classifier.keras'
MODEL = load_model(saved_model_path)

# input files
inputFile = 'assets/10x10'

INPUT_IMG = f'{inputFile}.png'
INPUT_WORDS = f'{inputFile}.txt'

# input grid dimensions
N_ROWS, N_COLUMNS = (int(inputFile[inputFile.index(
    '/')+1:inputFile.index('x')]), int(inputFile[inputFile.index('x')+1:]))

# image dimensions (to which input image will be resized to)
H_IMG, W_IMG = (500, 500)

# dataset image dimensions
H_DATASET_IMG, W_DATASET_IMG = (28, 28)

# thresholds for contour's width to height ratio
contourMin = 0.25
contourMax = 1.00


def predictionPreprocessing(img):
    img = cv2.bitwise_not(img)
    img = img / 255
    return img


def getContours(imgPath, resizeHeight, resizeWidth):
    '''Read the image and return the found contours, resized image & gray scale image'''

    img = cv2.imread(imgPath)
    img = cv2.resize(img, (resizeHeight, resizeWidth))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU |
                           cv2.THRESH_BINARY_INV)[1]
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilation = cv2.dilate(thresh, rect_kernel, iterations=1)
    contours = cv2.findContours(
        dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    return contours, img, gray


def getCoordsAndLabels(contours, grayImg, datasetImgH, datasetImgW, trainedModel):
    '''Loop through the contours, recognize the characters and return an array of image co-ordinates and class labels of the characters'''

    # (image co-ordinates)
    # [(x, y, 'P')] -> array
    coordsAndLabels = []
    alphabet_mapping = getAlphabetMapping()

    for cnt in contours:
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


def normalizeYAndSort(coordsLabels, cols):
    '''Normalize (make all y co-ordinate values of the row the same) the y co-ordinates to a same value for the entire row and sort by Y first then by X'''

    y_for_row = coordsLabels[0][1]
    for i, cnt in enumerate(coordsLabels):
        x, y, className = cnt

        # if beginning of row (col 1) then change 'y_for_row' to y value at that position
        if i % cols == 0:
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
    '''Create a grid with only labels (characters) as per required grid dimensions'''
    # loop by iterating at 'cols' per iteration, append 'cols' number of characters as rows into grid

    grid = []
    for i in range(0, len(coordsLabels), cols):
        row = list(map(lambda x: x[2], coordsLabels[i:i+cols]))
        grid.append(row)

    return grid


def generateImageFromGrid(grid, outputImgH, outputImgW, gridCols, gridRows):
    '''Generates new image with the grid letters'''

    # init empty image
    output = np.zeros((outputImgH + 50, outputImgW + 100, 3), np.uint8)

    # get the number of columns and rows that need to fit inside the image as per grid dimensions
    outputCols = outputImgW // gridCols
    outputRows = outputImgH // gridRows

    # the width and height of each character (as a box) in the grid
    # (or x & y offsets)
    charW = outputCols
    charH = outputRows

    # set an offset equal to outputCols
    offset = outputCols

    # set originStartX & originStartY to offset
    originStartX = offset
    originStartY = offset

    # loop through each column
    for cols in grid:
        # looping through each row item in the column
        for row in cols:

            # write the character to image
            cv2.putText(
                output, row, (originStartX, originStartY), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            # increase originStartX by offset to write next character in the current row
            originStartX += charW

        # reset originStartX to offset on finishing a row 
        originStartX = offset

        # increase originStartY by offset on finishing a row 
        originStartY += offset

    return output, (charH, charW), offset


def markFoundWords(foundWords, outputImg, charH, charW, offset):
    '''Marks the word in the grid image'''

    for foundWord in foundWords:
        # get start & end co-ordinates in grid
        word, start, end = foundWord

        # multiply the co-ordinate with char(w/h) and add offset
        xStart = start[0] * charW + offset
        yStart = start[1] * charH + offset

        xEnd = end[0] * charW + offset
        yEnd = end[1] * charH + offset

        # draw an arrow line from start to end
        cv2.arrowedLine(outputImg, (xStart, yStart),
                        (xEnd, yEnd), (0, 255, 0), 1)

    # return the output image after marking
    return outputImg


if __name__ == '__main__':

    # get the contours to extract characters
    contours, img, grayImg = getContours(INPUT_IMG, H_IMG, W_IMG)

    # get the co-ordinates and corresponding labels for recognized characters
    coordsAndLabels = getCoordsAndLabels(
        contours, grayImg, H_DATASET_IMG, W_DATASET_IMG, MODEL)

    # normalize and sort first by Y and then sort by X
    coordsAndLabels = normalizeYAndSort(coordsAndLabels, N_COLUMNS)

    # form a grid with characters
    grid = getGrid(coordsAndLabels, N_COLUMNS)

    # read words to be found from file
    words = readWordsFromFile(INPUT_WORDS)

    # create a WordSearch object and find all words
    WS = WordSearch(words, grid)
    foundWords, summary = WS.findAll()

    print(summary)

    # generating new grid image
    outputImg, (charH, charW), offset = generateImageFromGrid(
        grid, H_IMG, W_IMG, N_COLUMNS, N_ROWS)

    # overlaying solution over it
    outputImg = markFoundWords(foundWords, outputImg, charH, charW, offset)

    cv2.imshow('output', outputImg)
    cv2.imwrite('assets/output.png', outputImg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
