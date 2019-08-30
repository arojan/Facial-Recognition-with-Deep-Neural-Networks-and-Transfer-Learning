# -*- coding: utf-8 -*-
import cv2
import pickle
import os.path
import numpy as np
from helpers import resize_to_fit
from imutils import paths
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, ZeroPadding2D, Convolution2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras import backend as K
from sklearn.metrics import log_loss
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras import regularizers
import keras
# from ann_visualizer.visualize import ann_viz




def load_data(img_rows, img_cols,):

    IMAGES_FOLDER = "dataset"
    MODEL_LABELS_FILENAME = "models/model_labels.dat"
    # Resize trainging 
    data = []
    labels = []

    # loop over the input images

    for image_file in paths.list_images(IMAGES_FOLDER):
        # Load the image and convert it to grayscale
        image = cv2.imread(image_file)
        #image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # Resize the letter so it fits in a 80x80 pixel box
        image = resize_to_fit(image, img_rows, img_cols)

        # # Add a third channel dimension to the image to make Keras happy
        # image = np.expand_dims(image, axis=2)

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


    with open(MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)
        print("Saved the labels")


    # # Transform targets to keras compatible format
    # Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    # Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid




def vgg_face(weights_path, img_rows, img_cols, channel=3, num_classes=None):
    print('initialized . .. ')
    # img = Input(shape=(img_rows, img_cols, channel))
    # print('input done')

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
    # model.add(Dense(64, input_dim=64,
    #             kernel_regularizer=regularizers.l2(0.01),
    #             activity_regularizer=regularizers.l1(0.01)))

    # Loads ImageNet pre-trained data
    model.load_weights(weights_path)


    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    #model.params.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Learning rate is changed to   0.001
    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # for layer in model.layers[:-3]:
    #     layer.trainable=False

    # ann_viz(model, title="")

    return model




if __name__ == '__main__':

    #weights_path = 'vgg-face-keras-fc.h5'
    # weights_path = 'rcmalli_vggface_tf_notop_vgg16.h5'
    weights_path = 'models/rcmalli_vggface_tf_vgg16.h5'
    #weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'

    img_rows, img_cols = 224, 224
    channel = 3
    num_classes = 4
    batch_size = 40
    epochs = 300

    # IMAGES_FOLDER = "data"
    MODEL_FILENAME = "models/model.hdf5"

    # Load data
    print('loading .. .. . ')
    X_train, Y_train, X_valid, Y_valid = load_data(img_rows, img_cols)
    print('loaded.........')

    # Load our model
    model = vgg_face(weights_path, img_rows, img_cols, channel, num_classes )

    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              # callbacks=[tensorboard]
              )

    model.save(MODEL_FILENAME)
    # Make predictions
    score = model.evaluate(X_valid, Y_valid, batch_size=32)
    print('SCORE:', score)
    # predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    # score = log_loss(Y_valid, predictions_valid)
# 



