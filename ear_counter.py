#import shit
from __future__ import print_function
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils import face_utils
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from itertools import cycle
import argparse
import imutils
import time
import dlib
import cv2
import csv
import numpy as np

#input shit
vidfile = input("File name with extension: ")
ssid = input("Subject ID: ")

#adjust gamma to make features more visible, especially on dark skin
def adjust_gamma(video, gamma=3):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
	return cv2.LUT(video, table)

#find video midpoint and pull 5 minutes around it
cap = cv2.VideoCapture(vidfile) 
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
vid_dur = frame_count/fps
mid_dur = vid_dur/2
startpt = mid_dur-10 #change when troubleshooting
endpt = mid_dur+10 #change when troubleshooting

#trim video to 5 min
ffmpeg_extract_subclip(vidfile, startpt, endpt, targetname=ssid + "_trimmed_vid.wmv")

#create empty EAR list
earlist = []

#tell computer what EAR is
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear
 
# construct the argument parse and parse the arguments - you shouldn't need to change this
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, default="shapes.dat")
ap.add_argument("-v", "--video", required=False, default=ssid + "_trimmed_vid.wmv")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] Starting loop through video...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
time.sleep(1.0)

# loop over frames from the video stream
while True:
    
    # try to grab the frame from the threaded video file stream, resize it, and convert it to grayscale
    try:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
    
    # if it doesn't find a frame (ie video end is reached), break
    except:
        break
        
    # loop over various values of gamma
    for gamma in np.arange(0.0, 3.5, 0.5):
        gamma = gamma if gamma > 0 else 0.1
        adjusted = adjust_gamma(frame, gamma=gamma)
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then convert the facial landmark (x, y) coordinates to a numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the coordinates to compute the EAR for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the EAR together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        
    # add to list and display on screen so you know something is happening
    earlist.append(ear)
    print(ear)
    
# print list to csv
with open(ssid + "_timeseries.csv", "w") as output:
    writer = csv.writer(output)
    writer.writerow(earlist)

# do a bit of cleanup        
cv2.destroyAllWindows()
vs.stop()