#!/usr/bin/#!/usr/bin/env python3

# Importing the libraries that will be used later
import warnings

# Disabling any minor warmings and logs
warnings.filterwarnings("ignore", category=FutureWarning)

from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import os
import sys



# Defining the color codes and value, later to be used
class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


# The learnt model containing all the information
MODEL_FILENAME = "model.hdf5"

# Defining the saved labels filename
MODEL_LABELS_FILENAME = "model_labels.dat"

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
	lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

# Later to be used for running the program in loop
continuee = 'y'

# Using the while loop to run the program
while (continuee=='y' or continuee=='Y'):
	print("#\n#\n#INITIALIZING...\n")
	print("Welcome to the Facial Recognition System. Let me show you the way around!\n")
	print("Firstly, this system is capable of recognizing faces from videos and image \nfiles. The program is indeed ready for recognize faces from the CCTV Cameras \ntoo. But due to the unavailability of resource, the program has not been tested \non CCTV Cameras yet.")

	print("\nOptions:")
	print("To Predict Through Image Files, Select >>  [\'image\']")
	print("To Predict Through Video, Select >>  [\'video\']\n\n")

	# Taking input from user on either chosing to predict from the image or vidoe
	prediction_option = input(bcolors.BOLD + "Enter the option: " + bcolors.ENDC)

	if prediction_option=="image":

		# Notifying that the choice has been saved and is success
		print(bcolors.OKGREEN + "Success!" + bcolors.ENDC)
		print("Please Wait...")

		# Again, allowing the user to choose to predict from a single file or from the directory containing multiple files
		print("\nThis option allows you to predict multiple faces at once from the entire directory.")
		print("Or You can opt out and only predict a single image file\n")
		print("To predict from the directory: Enter '1'")
		print("To predict a single image file, Enter '2'\n")
		decision = int(input(bcolors.BOLD + "Enter your choice: " + bcolors.ENDC))

		if decision==1:
			# Taking input from the user on the name of directory
			IMAGES_FOLDER = input("Please input the name of the directory containing the image files: ")

			# Taking input from the user on ACCURACY_METRIC
			ACCURACY_METRIC = float(input("Please input the value of accuracy metric you wish to set[ standard -- 0.70 ]: "))
			print("\n\nPredicting image files from the directory >>",os.path.dirname(IMAGES_FOLDER))

			# loop over the image paths
			for image_file in paths.list_images(IMAGES_FOLDER):

				# Reading the image file and converting into numpy array
				image = cv2.imread(image_file)

				# Resizing the image dimensions
				predictions = []
				letter_image = resize_to_fit(image, 224, 224)

				# Expanding the image dimension with zero axis
				letter_image = np.expand_dims(letter_image, axis=0)

				# This allows to change and view of the accuracy of the prediction in percentage or in floating point
				letter_image = np.array(letter_image, dtype="float") / 255.0

				# Ask the neural network to make a prediction
				prediction = model.predict(letter_image)

				# Converting the prediction value of tuple into list and then saving the maximum value
				predictions = np.array(prediction).tolist()
				top_value = max(predictions)
				top_value = max(top_value)

				# Convert the one-hot-encoded prediction back to a normal letter
				letter = lb.inverse_transform(prediction)
				print("File Name >>",os.path.basename(image_file))

				# Printing the prediction of the face in the terminal
				if top_value > ACCURACY_METRIC:
					print("Prediction >>>   %s    [%.2f percentage sure]" % (letter,top_value*100))
				else:
					print("Status >>>   ['Unconfirmed']")
					print(bcolors.FAIL + "<<< POSSIBLE SUSPECT >>>  -%s-    [%.2f percentage sure]" % (letter,top_value*100) + bcolors.ENDC)
				print("\n")



		elif decision==2:
			image_file = input(bcolors.BOLD + "Please input the name of the image file you wish to predict: " + bcolors.ENDC)
			ACCURACY_METRIC = float(input("Please input the value of accuracy metric you wish to set[ standard -- 0.70 ]: "))
			image = cv2.imread(image_file)

			predictions = []

			# Resizing the image dimensions
			letter_image = resize_to_fit(image, 224, 224)

			# Expanding the image dimension with zero axis
			letter_image = np.expand_dims(letter_image, axis=0)

			# This allows to change and view of the accuracy of the prediction in percentage or in floating point
			letter_image = np.array(letter_image, dtype="float") / 255.0

			# Ask the neural network to make a prediction
			prediction = model.predict(letter_image)

			# Converting the prediction value of tuple into list and then saving the maximum value
			predictions = np.array(prediction).tolist()
			top_value = max(predictions)
			top_value = max(top_value)

			# Convert the one-hot-encoded prediction back to a normal letter
			letter = lb.inverse_transform(prediction)
			print("\n\nPredicting the image with the file name >>",os.path.basename(image_file))

			# Printing the prediction of the face in the terminal
			if top_value > ACCURACY_METRIC:
				print("Prediction >>>   %s              [%.2f percentage sure]" % (letter,top_value*100))
			else:
				print("Status >>>   ['Unconfirmed']")
				print(bcolors.FAIL + "<<< POSSIBLE SUSPECT >>>  -%s-              [%.2f percentage sure]" % (letter,top_value*100) + bcolors.ENDC)
			print("\n")

		else:
			print(bcolors.FAIL + "Sorry, you entered the wrong choice. Please try again!" + bcolors.ENDC)


	# Predicting from the video
	elif prediction_option=="video":
		print(bcolors.OKGREEN + "Success!" + bcolors.ENDC)
		print("\nPredicting through video\n")
		print("This option allows to predict faces from the real-time video processing.\n")
		print("To predict through webcam, Enter '1'")
		print("To predict through live cctv camera, Enter '2'")

		# Taking input from user on either to predict from the webcam or CCTV camera
		decision = int(input(bcolors.BOLD + "\nDo you wish to predict through webcam or cctv camera: " + bcolors.ENDC))

		if decision==1:
			video = 0

		# Though cctv camera is not ready to use and not available either, showing the basic concept of how its done
		elif decision==2:
			print("\nTo predict faces from the cctv camera live feed, you will need to enter: ")
			print("* Ip address")
			print("* Port number")
			print("* Admin username")
			print("* Admin password")
			print("#FORMAT: http://<admin>:<password>@<ipaddress>:<portnumber>")
			print("#EXAMPLE: http://admin:password@192.168.1.1:8080")
			video = input("Please input the ip camera's link: ")

		else:
			print(bcolors.FAIL + "\nSorry, you entered the wrong choice. Please try again!" + bcolors.ENDC)

		# Using the haarcascade for determining the faces landmark in the image and then only trying to predict those faces
		haar_file = 'haarcascade_frontalface_default.xml'
		faceCascade = cv2.CascadeClassifier(haar_file)

		video_capture = cv2.VideoCapture(video)
		# Above defined value of video as '0' is used to use the webcam
		# Capturing the video

		# Defining the font size, scales and colors to show as the prediction label on the webcam's video screening window
		font                   = cv2.FONT_HERSHEY_PLAIN
		fontScale              = 1.5
		fontColor              = (0, 102, 0)
		lineType               = 1

		# As earlier, defining the accuracy metric to alter the admininstrator in wrong prediction
		ACCURACY_METRIC = float(input("Please input the value of accuracy metric you wish to set[ standard -- 0.99 ]: "))

		while True:
			# Capture frame-by-frame
			ret, frame = video_capture.read()
			# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			image = faceCascade.detectMultiScale(frame,1.8,5)

			predictions = []

			# Drawing the rectangular line on all the detected face landmarks
			for (x,y,w,h) in image:
				cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
				face = frame[y:y + h, x:x + w]
				letter_image = cv2.resize(face, (224, 224))
				letter_image = np.array(letter_image, dtype="float") / 255.0

				# Turn the single image into a 4d list of images to make Keras happy
				# letter_image = np.expand_dims(letter_image, axis=2)
				letter_image = np.expand_dims(letter_image, axis=0)

				# Ask the neural network to make a prediction
				prediction = model.predict(letter_image)

				predictions = np.array(prediction).tolist()
				top_value = max(predictions)
				top_value = max(top_value)

				# Convert the one-hot-encoded prediction back to a normal letter
				letter = lb.inverse_transform(prediction)

				if top_value > ACCURACY_METRIC:
					cv2.putText(frame,str(letter),
						(x-10,y-10),
						font,
						fontScale,
						fontColor,
						lineType)
				else:
					cv2.putText(frame,"unknown",
						(x-10,y-10),
						font,
						fontScale,
						fontColor,
						lineType)

				# Display the resulting frame
				cv2.imshow('Video', frame)

			# Functionality for quiting the webcam program
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		# Closing the webcam window and stop capturing
		# When everything is done, release the capture
		video_capture.release()
		cv2.destroyAllWindows()



	else:
		# Notifying that the user made a wrong choice
		print(bcolors.BOLD + bcolors.FAIL + "Sorry, you entered the wrong choice. Please try again!" + bcolors.ENDC)

	# Allowing the user to choose to either continue the predcition program or quit the program
	continuee = input(bcolors.BOLD + bcolors.WARNING + "\nDo you wish to continue predicting [Y/n]: " + bcolors.ENDC)

print("#The program has terminated...")
