#!/usr/bin/env python3
import torch
import os, sys, rospy
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
sys.path.remove('/home/amr/catkin_ws/devel/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/home/amr/catkin_ws/devel/lib/python2.7/dist-packages') # in order to import cv2 under python3
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import numpy as np
from collections import OrderedDict
from models import get_model
from hardnet import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from numpy import savetxt, loadtxt

#path of the pretrained model (change it to fit your directory)
device,model = init_model("/home/amr/workspace/src/caric-segmentation/pretrained/hardnet70_cityscapes_model.pkl")
seg_pub = rospy.Publisher("/segment_pub",Image,queue_size = 10)
def callback(data):
	#using cv_bridge to deal with image inside ros
    bridge1 = CvBridge()
    cv_image = bridge1.imgmsg_to_cv2(data, "bgr8")
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    img_resized,decoded = process_img(img,[420,680],device,model.cuda())
    M = loadtxt('/home/amr/workspace/src/segmentation/BEV.csv', delimiter=',')####BEV
    wraped = cv2.warpPerspective(img_resized, M, (640,420),       flags=cv2.INTER_LINEAR)

    decoded = decoded.astype('float32')*255
    decoded = np.uint8(decoded)
    ros_image = bridge1.cv2_to_imgmsg(decoded, encoding = 'bgr8')
    seg_pub.publish(ros_image)



def listener():

    rospy.init_node('segment', anonymous=False)
#insert the camera topic here
    rospy.Subscriber('cameraC/image_raw', Image, callback)


    rospy.spin()


if __name__ == '__main__':
    listener()
