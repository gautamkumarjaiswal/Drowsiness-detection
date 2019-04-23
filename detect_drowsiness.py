# HOW TO USE
# python detect_drowsiness.py --landmark-identifier shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --landmark-identifier shape_predictor_68_face_landmarks.dat --alarm alarm.wav

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from collections import OrderedDict
import numpy as np
import cv2

# activating the alarm
def sound_alarm(path):
	playsound.playsound(path)

def cal_E_A_R(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	dist_a = dist.euclidean(eye[1], eye[5])
	dist_b = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	dist_c = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	e_a_r = (dist_a + dist_b) / (2.0 * dist_c)

	# return the eye aspect ratio
	return e_a_r

	
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--landmark-identifier", 
	required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", 
	type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", 
	type=int, default=0,
	help="index of webcam on system")

args = vars(ap.parse_args())

print("printing all the input arguments")
print("landmark-identifier ",args["landmark_identifier"])
print("alarm ",args["alarm"])
print("webcam ", args["webcam"]+1)


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
EYE_AR_CONSEC_FRAMES_blink = 20

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["landmark_identifier"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()

	# if the colors list is None, initialize it with a unique
	# color for each facial landmark region
	if colors is None:
		colors = [(19, 199, 109), 
			(79, 76, 240), 
			(230, 159, 23),
			(168, 100, 168), 
			(158, 163, 32),
			(163, 38, 32), 
			(180, 42, 220)]

	# loop over the facial landmark regions individually
	for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
		# grab the (x, y)-coordinates associated with the
		# face landmark
		(j, k) = FACIAL_LANDMARKS_IDXS[name]
		pts = shape[j:k]

		# check if are supposed to draw the jawline
		if name == "jaw":
			# since the jawline is a non-enclosed facial region,
			# just draw lines between the (x, y)-coordinates
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)

		# otherwise, compute the convex hull of the facial
		# landmark coordinates points and display it
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)

	# apply the transparent overlay
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

	# return the output image
	return output

# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=700)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	cv2.putText(frame, "Press 'q' to exit", (600, 550),
					cv2.FONT_HERSHEY_SIMPLEX, 
					0.7, (0, 255, 255), 2)
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			# clone the original image so we can draw on it, then
			# display the name of the face part on the image
			#clone = frame.copy()
			#cv2.putText(frame, name, (30, 50), 
			#	cv2.FONT_HERSHEY_SIMPLEX,
			#	0.7, (0, 0, 255), 2)
	 
			# loop over the subset of facial landmarks, drawing the
			# specific face part
			for (x, y) in shape[i:j]:
				cv2.circle(frame, (x, y),
				1, (0, 0, 255), -1)
			# extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = frame[y:y + h, x:x + w]
			roi = imutils.resize(roi, width=600, 
				inter=cv2.INTER_CUBIC)


		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		left_EAR = cal_E_A_R(leftEye)
		right_EAR = cal_E_A_R(rightEye)

		# average the eye aspect ratio together for both eyes
		e_a_r = (left_EAR + right_EAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, 
			(0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, 
			(0, 255, 0), 1)
	
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if e_a_r < EYE_AR_THRESH:
			COUNTER += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7,
					(0, 0, 255), 2)
			
			elif COUNTER >= EYE_AR_CONSEC_FRAMES_blink:
				cv2.putText(frame, "YOU JUST BLINKED", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
					(0, 255, 0), 2)
					
		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False
			cv2.putText(frame, "THE DRIVER IS AWAKE", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (0, 255, 0), 2)

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "Eye Aspect Ratio: {:.2f}".format(e_a_r), 
				(400, 30),
				cv2.FONT_HERSHEY_TRIPLEX, 
				0.7, (0, 255, 0), 2)
	
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()