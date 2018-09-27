import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Dropout
from keras.optimizers import Adam
import cv2
model = Sequential()
model = load_model('my_model.h5')


X_test = []
y_test = []
fp = open('little/test.txt', "r")

line = fp.readline()
while line:
    lab = np.zeros(15,dtype = np.float32)
    mid = line.find(".jpg")+4
    lab[int(line[mid:len(line)])] = 1
    img = cv2.imread(line[0:mid])
    img = cv2.resize(img,dsize=(160,120))
    X_test.append(img)
    y_test.append(lab)
    line = fp.readline()
X_test = np.asarray(X_test,dtype=np.float32)
y_test = np.asarray(y_test)

X_test = X_test.reshape(-1, 3,160, 120)/255.
y_test = y_test.reshape(-1, 15)

loss, accuracy = model.evaluate(X_test, y_test)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)