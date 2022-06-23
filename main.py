import cv2
import numpy as np
from keras.models import load_model

saved_model_path = 'model/letters_classifier.keras'
model = load_model(saved_model_path)

puzzle = 'assets/puzzle1.png'
hImg, wImg = (500, 500)

nRows, nCols = (14, 14)

# prediction confidence
confidence_lt = 0.25
confidence_ht = 1.00

img = cv2.imread(puzzle)
img = cv2.resize(img, (hImg, wImg))
cv2.imshow('raw image', img)

result = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU |
                       cv2.THRESH_BINARY_INV)[1]
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
dilation = cv2.dilate(thresh, rect_kernel, iterations=1)
contours, hierarchy = cv2.findContours(
    dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


def getAlphabetMapping():
    return {num: chr(char) for (num, char) in zip(list(range(0, 26)), list(range(65, 65+26)))}


def predictionPreprocessing(img):
    img = cv2.bitwise_not(img)
    img = img / 255
    return img


am = getAlphabetMapping()

# eligible_contours = []

for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)

    if confidence_lt <= (w/h) <= confidence_ht:

        letter = gray[y:y+h, x:x+w]
        letter = cv2.resize(letter, (28, 28))
        letter = predictionPreprocessing(letter)
        letter = letter.reshape(1, 28, 28, 1)

        # prediction
        prediction = model.predict(letter)
        classId = np.argmax(prediction, axis=1)[0]
        confidence = np.amax(prediction)
        className = am[classId]

        if confidence > 0.5:
            result = cv2.rectangle(
                result, (x, y), (x + w, y + h), (0, 255, 0), 1)

        else:
            result = cv2.rectangle(
                result, (x, y), (x + w, y + h), (0, 0, 255), 1)
            print(confidence, className)

        # result = cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(result, className, (x, y),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

    # eligible_contours.append((x, y))


# eligible_contours = sorted(eligible_contours)
# print(eligible_contours)


cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
