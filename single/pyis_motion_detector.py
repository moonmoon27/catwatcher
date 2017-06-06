# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import json

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-v", "--video", help="path to the video file")
#ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-c", "--conf", required=True, help="path to the JSON conf file")
args = vars(ap.parse_args())
# if the video argument is None, then we are reading from webcam
#if args.get("video", None) is None:
#	camera = cv2.VideoCapture(0)
#	time.sleep(0.25)
# otherwise, we are reading from a video file
#else:
#	camera = cv2.VideoCapture(args["video"])
conf = json.load(open(args["conf"]))

# A initialize camera
# initialize the first frame in the video stream
#firstFrame = None
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

print("[INFO] warming up ...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0


# loop over the frames of the video
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        
        #while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	#(grabbed, frame) = camera.read()
        frame = f.array
        timestamp = datetime.datetime.now()
        text = "Unoccupied"

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	#if not grabbed:
	#	break

	# resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
        if avg is None:
                print("[INFO] starting background model...")
                avg = gray.copy().astype("float")
                rawCapture.truncate(0)
                
	#if firstFrame is None:
	#	firstFrame = gray
                continue
	

	# compute the absolute difference between the current frame and
	# first frame
	#frameDelta = cv2.absdiff(firstFrame, gray)
	#thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        #acumulate the weighted average between the current frame and
	#previous frames, then compute the difference between the current
	#frame and running average
        cv2.accumulateWeighted(gray, avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

        #threshold the delta image, dilate the thresholded image to fill
	#in holes, then find contours on thresholded image
        thresh = cv2.threshold(frameDelta, conf["delta_thresh"],255, cv2.THRESH_BINARY)[1]
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# loop over the contours
        for c in cnts:
		# if the contour is too small, ignore it
                if cv2.contourArea(c) < conf["min_area"]:
                    continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),1)
                text = ("Occupied: center (%s;%s)" % ((x+w/2.),(y+h/2.)))   

	# draw the text and timestamp on the frame
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

	# show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
        if key == ord("q"):
                break
        #necessary to empty buffer
        rawCapture.truncate(0)
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()


        
