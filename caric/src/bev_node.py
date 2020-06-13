#!/usr/bin/env python3
import torch
import os
import sys
sys.path.remove('/home/amr/catkin_ws/devel/lib/python2.7/dist-packages') # in order to import cv2 under python3
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/home/amr/catkin_ws/devel/lib/python2.7/dist-packages') # in order to import cv2 under python3
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import numpy as np
from collections import OrderedDict
from models import get_model
from hardnet import *
import rospy
import time
from numpy import savetxt, loadtxt
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo

#path of the pretrained model (change it to fit your directory)

bev_pub = rospy.Publisher("/bev_pub",Image,queue_size = 10)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--debug', help='use raw camera images or segmentation images', required = True)
args = parser.parse_args()

def callback(data):
    bridge1 = CvBridge()

    cv_image = bridge1.imgmsg_to_cv2(data, "rgb8")
    tic = time.time()
    M = loadtxt('/home/amr/workspace/src/segmentation/BEV.csv', delimiter=',')####BEV
    wraped = cv2.warpPerspective(cv_image, M, (620,480),       flags=cv2.INTER_LINEAR)####BEV
    toc = time.time()
    print ("Estimated BEV inference time in [s] : {0}".format(toc-tic))
    #bridge2 = CvBridge()
    #cv2.imshow("image2d",img_resized)
    #cv2.imshow("image", decoded)
    #print(type(decoded))
    cv2.imwrite("/home/amr/workspace/src/segmentation/wraped.jpg",wraped)
    #scan_list = [[1.0,0.17], [15.0,0.000001], [10.0,-0.51], [2.0,0.6],[22.0,1.2], [6.7,-0.17],
              #            [5.0,0.001], [10.0,-0.5], [3.5,0.75],[2.5,-1.0], [31.0,0.001],[0.7, -0.5]]
    ros_image = bridge1.cv2_to_imgmsg(wraped, encoding = 'bgr8')
    #rospy.loginfo(decoded)
    bev_pub.publish(ros_image)
    


def listener():
    if args.debug == 'true':
        sub_topic= '/cameraC/image_raw'
    else:
        sub_topic= '/segment_pub'

    rospy.init_node('bev', anonymous=False)
#insert the camera topic here
    rospy.Subscriber(sub_topic, Image, callback)


    rospy.spin()


if __name__ == '__main__':
    listener()



