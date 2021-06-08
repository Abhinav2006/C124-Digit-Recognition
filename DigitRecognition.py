import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

X, y = fetch_openml('mnist_784', version = 1, return_X_y = True)
print(pd.Series(y).value_counts())
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nclasses = len(classes)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

xtrain = xtrain/255.0
xtest = xtest/255.0
lr = LogisticRegression(solver = 'saga', multi_class = 'multinomial')
lr.fit(xtrain, ytrain)

y_pred = lr.predict(xtest)
accuracy = accuracy_score(ytest, y_pred)
print(accuracy)

#if(not os.environ.get('PYTHONHTTPSVERIFY','')and getattr(ssl, '_create_unverified_context', None)):
#ssl._create_default_https_context = ssl._create_unverified_context

cap = cv2.VideoCapture(1)
while(True):
    try:
        ret, frame  = cap.read()
        height, width = gray.shape()
        upperLeft = (int(width/2 - 56), int(height/2 - 56))
        bottomRight = (int(width/2 + 56), int(height/2 +56))
        cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 0), 2)
        roi = gray[upperLeft[1]: bottomRight[1], upperLeft[0]: bottomRight[0]]
        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert('L')
        image_bw_resize = image_bw.resize((28, 28), Image.ANTIALIAS)
        image_bw_resize_inverted = PIL.ImageOps.invert(image_bw_resize)
        pixelfilter = 20
        minpixel = np.percentile(image_bw_resize_inverted, pixelfilter)
        image_bw_resize_inverted_scaled = np.clip(image_bw_resize_inverted - minpixel, 0, 255)
        max_pixel = np.max(image_bw_resize_inverted)
        image_bw_resize_inverted_scaled = np.asarray(image_bw_resize_inverted_scaled)/max_pixel
        test_sample = np.array(image_bw_resize_inverted_scaled).reshape(1784)
        test_predict = lr.predict(test_sample)
        print("Predicted number is", test_predict)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1)&0XFF == ord('q'):
            break
    except Exception as e:
        pass
cv2.destroyAllWindows()
cap.release()