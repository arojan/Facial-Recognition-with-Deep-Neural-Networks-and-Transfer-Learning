# -*- coding: utf-8 -*-
import cv2
import pickle
import os.path
import numpy as np
from helpers import resize_to_fit
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, ZeroPadding2D, Convolution2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras import backend as K
from sklearn.metrics import log_loss
from keras.utils import np_utils
from keras import regularizers
import keras


def load_data(img_rows, img_cols,):

    IMAGES_FOLDER = "dataset"
    MODEL_LABELS_FILENAME = "model_labels.dat"
    # Resize trainging
    data = []
    labels = []

    # loop over the input images
    for image_file in paths.list_images(IMAGES_FOLDER):
        image = cv2.imread(image_file)

        # # Resize the letter so it fits in a 80x80 pixel box
        image = resize_to_fit(image, img_rows, img_cols)

        # Grab the name of the letter based on the folder it was in
        label = image_file.split(os.path.sep)[-2]

        # Add the letter image and it's label to our training data
        data.append(image)
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1] (this improves training)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)


    # Split the training data into separate train and test sets
    (X_train, X_valid, Y_train, Y_valid) = train_test_split(data, labels, test_size=0.25, random_state=0)

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_valid = lb.transform(Y_valid)

    # Save the label. Useful when predicting
    with open(MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    return X_train, Y_train, X_valid, Y_valid



def vgg_face(weights_path, img_rows, img_cols, channel=3, num_classes=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))
    model.add(Convolution2D(64, (3, 3), activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', trainable=False))
    model.add(Dropout(0.25))
    model.add(Dense(4096, activation='relu', trainable=False))
    model.add(Dropout(0.25))
    model.add(Dense(2622, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
        activity_regularizer=regularizers.l1(0.01)))

    # Loads ImageNet pre-trained data
    model.load_weights(weights_path)

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ## Return the model
    return model




if __name__ == '__main__':

    # Loading the pre-trained weights
    weights_path = 'rcmalli_vggface_tf_vgg16.h5'

    img_rows, img_cols = 224, 224
    channel = 3
    num_classes = 7
    batch_size = 24
    epochs = 420

    # Naming the weights of the model
    MODEL_FILENAME = "model.hdf5"

    # Load data
    X_train, Y_train, X_valid, Y_valid = load_data(img_rows, img_cols)

    # Load our model
    model = vgg_face(weights_path, img_rows, img_cols, channel, num_classes )

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    # Save the learnt weights
    model.save(MODEL_FILENAME)
    # Make predictions
    score = model.evaluate(X_valid, Y_valid, batch_size=24)
    print('SCORE:', score)


# # References:
#         François Chollet's [https://github.com/fchollet]
#         Greg Chu's article on 'How to use transfer learning and fine-tuning in Keras and Tensorflow to build an image recognition system and classify (almost) any object'
#         Deep Face Recognition, from Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman | BMVC 2015
#         [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
#         The VGG16 and VGG19 weights are ported from the ones released by VGG at Oxford under the Creative Commons Attribution License.
#         https://www.robots.ox.ac.uk/%7Evgg/research/very_deep/
#
# # Most of the portions of the above code have referred from these resources or at least, an understanding concepts were taken.
