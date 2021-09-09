# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class sequntial:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        #First layer
        model.add(Conv2D(filters=8, # output feature maps
                             kernel_size=(3,3), # matrix size for feature detector
                             input_shape=(20, 20, 3), # input image shape, 3 is for rgb coloured image with 128*128 px
                             kernel_initializer='he_uniform', # weights distriution
                             activation='relu')) # activation function
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=16,kernel_size=(3,3),kernel_initializer='he_uniform',activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_uniform',activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='he_uniform',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        #Fifth layer
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu')) 
       
        #Output layer 
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

EPOCHS = 30
INIT_LR = 0.0001
BS = 10
# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images("Dataset")))
random.seed(42)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = img_to_array(image)
    image = image[:,:,:3]
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    if label=='Flood':
       labels.append(0) 
    elif label=='NoFlood':
       labels.append(1)
 
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print(np.shape(data))
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.3, random_state=42)
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = sequntial.build(width=20, height=20, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the network
print("[INFO] training network...")
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)

print("[INFO] serializing network...")
model.save("CNN Patch.h5")

# list all data in history
import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# image folder
folder_path = '/content/Test1/'
# path to model
model_path = '/content/CNN Patch.h5'
# dimensions of images
img_width, img_height = 20, 20

# load the trained model
model = load_model(model_path)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# load all images into a list
images = []
for img in os.listdir(folder_path):
    img = os.path.join(folder_path, img)
    img = image.load_img(img, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

# stack up images list to pass for prediction
images = np.vstack(images)

classes = model.predict(images, batch_size=10)
y_classes = pred_test.argmax(axis=-1)
print(y_classes)
if y_classes==0:
   print('FLOOD')
elif y_classes==1:
   print('NOFLOOD')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/content/Test/10.png', target_size = (20, 20))
#print(training_set.class_indices)
print(result)
if result[0][0] == 0:
    print('flood')
else:
    print('noflood')