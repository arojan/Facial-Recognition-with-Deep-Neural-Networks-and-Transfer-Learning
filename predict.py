from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import os


MODEL_FILENAME = "models/model.hdf5"
MODEL_LABELS_FILENAME = "models/model_labels.dat"


# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

image = cv2.imread('test/test.jpg')

predictions = []
letter_image = resize_to_fit(image, 224, 224)

# Turn the single image into a 4d list of images to make Keras happy
# letter_image = np.expand_dims(letter_image, axis=2)
letter_image = np.expand_dims(letter_image, axis=0)

# Ask the neural network to make a prediction
prediction = model.predict(letter_image)
print(prediction)

# Convert the one-hot-encoded prediction back to a normal letter
letter = lb.inverse_transform(prediction)
print(letter)