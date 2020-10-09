#!/usr/bin/env python3
# USAGE
# python3 yolo.py --yolo yolo-coco

# import the necessary packages
import argparse
import time
import numpy as np
import rospy
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os
import math

def init ():
	# load the COCO class labels our YOLO model was trained on
	labelsPath = 'yolo-coco/coco.names' 
	LABELS = open(labelsPath).read().strip().split("\n")

	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = 'yolo-coco/yolov3.weights'
	configPath =  'yolo-coco/yolov3.cfg'

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] setting preferable backend and target to CUDA...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	#uncomment when opencv-GPU is installed

	# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	return net,COLORS,LABELS



def yoloo(img,net,COLORS,LABELS,conf=0.5,threshold=0.3):
	
		# load our input image and grab its spatial dimensions

	#------------------#
	image = img #cv2.imread(args["image"])
	(H, W) = image.shape[:2]

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > conf:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# Getting the angle
				## Under develompent
				angle_rad = math.atan2( (y + int(height)) , (x + int(width/2)) ) # For radians
				angle_deg = angle_rad * (180 / math.pi) # For degress

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height), [angle_rad, angle_deg]])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf,
		threshold)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2)

			# Add circle beneath detected object
			cv2.circle(image, ((x + int(w/2)), (y + int(h))), 5, color = (175, 12, 159), thickness = -1)
			cv2.circle(image, ((x + int(w/2)), (y + int(h))), 6, color = (1, 0, 0), thickness = 2)


	# show the output image
	i=1
	while i >0:
		i=i-1
		cv2.imshow('YOLO',image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break  
	# cv2.imshow("Image", image)
	print(boxes)
	bridge = CvBridge()
	res = bridge.cv2_to_imgmsg(image, encoding="passthrough")
	# cv2.waitKey(0)
	return res
#-------------------------------------------------------------------

def callback(data):
    bridge1 = CvBridge()
    pub = rospy.Publisher('yoloo',Image,queue_size=10 )
    cv_image = bridge1.imgmsg_to_cv2(data, desired_encoding='passthrough')
    im_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    cv2.imshow('img',im_rgb)
    # net,COLORS,LABELS=init()
    res = yoloo(im_rgb,net,COLORS,LABELS)
    pub.publish(res)

	

def start():
	global net,COLORS,LABELS 
	rospy.init_node('warped',anonymous=True)
	rospy.Subscriber('/cameraC/image_raw', Image,callback) 
	net,COLORS,LABELS=init()
	rospy.spin()

if __name__ == '__main__':

     try:
        start()

     except rospy.ROSInterruptException:
        pass
