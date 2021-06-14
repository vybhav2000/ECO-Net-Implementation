
import numpy as np
import argparse
import imutils
import sys
import cv2

#Parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained human activity recognition model")
ap.add_argument("-c", "--classes", required=True,
	help="path to class labels file")
ap.add_argument("-i", "--input", type=str, default="",
	help=" path to video file")
args = vars(ap.parse_args())

#Loading labels and specifying sample size of frames as it is 3d cnn
CLASSES = open(args["classes"]).read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

# load the pre-trained model on kinetics 400
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(args["model"])

#initializing video instance
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

#read throught the frames
while True:
	frames = []

	for i in range(0, SAMPLE_DURATION):
		(grabbed, frame) = vs.read()

		# if the frame was not grabbed then we've reached the end of
		# the video stream so exit the script
		if not grabbed:
			print("[INFO] no frame read from stream - exiting")
			sys.exit(0)

		# frame resizing
		frame = imutils.resize(frame, width=400)
		frames.append(frame)

	blob = cv2.dnn.blobFromImages(frames, 1.0,
		(SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
		swapRB=True, crop=True)
	#making image into test format 4d excluding batch dimension
	blob = np.transpose(blob, (1, 0, 2, 3))
	blob = np.expand_dims(blob, axis=0)

	# pass the blob through the network to obtain our human activity
	# recognition predictions
	net.setInput(blob)
	outputs = net.forward()
	label = CLASSES[np.argmax(outputs)]

	# loop over our frames
	for frame in frames:
		# draw the predicted activity on the frame
		cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
		cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
			0.8, (255, 255, 255), 2)
		cv2.imshow("Activity Recognition", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break