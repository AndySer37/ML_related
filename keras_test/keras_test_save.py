
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Dropout
from keras.optimizers import Adam
import cv2

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


X_train = []
y_train = []
fp = open('little/train.txt', "r")
line = fp.readline()
while line:
    lab = np.zeros(15,dtype = np.float32)
    mid = line.find(".jpg")+4
    lab[int(line[mid:len(line)])] = 1
    img = cv2.imread(line[0:mid])
    img = cv2.resize(img,dsize=(160,120))
    X_train.append(img)
    y_train.append(lab)
    line = fp.readline()
X_train = np.asarray(X_train,dtype=np.float32)
y_train = np.asarray(y_train)

# data pre-processing
X_train = X_train.reshape(-1, 3,160, 120)/255.
X_test = X_test.reshape(-1, 3,160, 120)/255.
y_train = y_train.reshape(-1, 15)
y_test = y_test.reshape(-1, 15)

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    batch_input_shape=(None, 3, 160, 120),
    filters=32,
    kernel_size=4,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
))

# Conv layer 2 output shape (32, 14, 14)
model.add(Convolution2D(32, 4, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

model.add(Convolution2D(32, 4, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

model.add(Convolution2D(32, 4, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
# Pooling layer 2 (max pooling) output shape (64, 7, 7)


# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(300))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
# Fully connected layer 2 to shape (15) for 15 classes
model.add(Dense(15))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=5*(1e-5))

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=10, batch_size=100,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)
model.save('my_model.h5')
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)