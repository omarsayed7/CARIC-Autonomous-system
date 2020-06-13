#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image, CameraInfo
import std_srvs.srv
from cv_bridge import CvBridge, CvBridgeError

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
sys.path.remove('/home/amr/catkin_ws/devel/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/home/amr/catkin_ws/devel/lib/python2.7/dist-packages') # in order to import cv2 under python3
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import torch
import math, argpars, os, sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
from collections import OrderedDict

# Global Variables regarding inverse perspective mapping
IMG_SIZE = (680,420)
M = loadtxt('/home/amr/workspace/src/segmentation/BEV.csv', delimiter=',')
pixel_per_meter_y= 97.85714285714286
pixel_per_meter_x= 243.5897435897436
DEAD_ZONE = 0.7

# scan converter publisher node
goal_pub = rospy.Publisher('/ourscan', LaserScan, queue_size=10)
scan_msg = LaserScan()


def callback(data):
	#put the gpx points here(points and msg type
    scan_msg.header.frame_id= "laser_scanner2"
    scan_msg.header.stamp= rospy.Time.now()
    scan_msg.angle_min= -0.785398
    scan_msg.angle_max= 0.785398
    scan_msg.angle_increment= 0.0021816616
    scan_msg.time_increment = 0.0
    scan_msg.range_min= 0.800000011921
    scan_msg.range_max = 30.0
    ranges = [float("inf")]*720

    bridge1 = CvBridge()
    cv_image = bridge1.imgmsg_to_cv2(data, "bgr8")
    sample_image = cv_image

    lower_limit = np.array([50,50,50])
    upper_limit = np.array([200, 200, 200])
    mask = cv2.inRange(sample_image, lower_limit, upper_limit)

    contours, hierarchy = cv2.findContours(mask*200,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours_warped = sample_image.copy()
    for contour in contours:
    	 cv2.drawContours(contours_warped, contour, -1, (0, 255, 0), 3)

    original_center = np.array([[[IMG_SIZE[0]/2,IMG_SIZE[1]]]],dtype=np.float32)
    warped_center = cv2.perspectiveTransform(original_center, M)[0][0]
    warped_center[1] = warped_center[1] + DEAD_ZONE # Assume 50cm of view deadzone (value to be found with propper extrinsic calibration)

    scan_distances = []
    scan_angles = []
    for contour in contours:
        for point in contour:
            distance = math.sqrt(((point[0][0]-warped_center[0])/pixel_per_meter_x)**2 + ((point[0][1]-warped_center[1])/pixel_per_meter_y)**2)
            angle = math.atan2((point[0][0] - warped_center[0])/pixel_per_meter_x, (warped_center[1]-point[0][1])/pixel_per_meter_y)
            scan_distances.append(distance)
            scan_angles.append(angle)
            cv2.circle(contours_warped,(point[0][0],point[0][1]), 2, (0,0,255), -1)

    def takeSecond(elem):
        return elem[1]
    scan_array = np.float32(([scan_distances, scan_angles])).T
    scan_list = list(scan_array)
    scan_list.sort(key=takeSecond)
    scan_array = np.array(scan_list)
    #sort in terms of distance to catch less distance first
    for scan in scan_list:
        scan = list(scan)
        distance = scan[0]
        theta = scan[1]
        if distance > scan_msg.range_max or distance < scan_msg.range_min:
            continue
        else:
           #check the angle range
            if (theta > scan_msg.angle_max) or (theta< scan_msg.angle_min):
                continue
            else:
            #valid point in terms of distance and angle
                diff = scan_msg.angle_max-theta
                idx = int(round(diff / scan_msg.angle_increment))
                if ranges[idx] != math.inf:
                     idx = idx + 1
                else :
                     idx = idx
                ranges[idx] = distance

    scan_msg.ranges = ranges
    #rospy.loginfo(ranges.index(15))
    clear_ogm = rospy.ServiceProxy('/move_base/clear_costmaps', std_srvs.srv.Empty)
    clear_ogm()
    goal_pub.publish(scan_msg)
    #clear_ogm()
    # spin() simply keeps python from exiting until this node is stopped

def start():
    rospy.init_node('laser_pub', anonymous=False)
    rospy.Subscriber("/bev_pub", Image,callback)
    rospy.spin()

if __name__ == '__main__':
    start()
