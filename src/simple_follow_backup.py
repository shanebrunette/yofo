#!/usr/bin/env python
# just some really basic code to get started following someone
# just run this as an individual node using rosrun comp3431_project simple_follow.py
# you also need to be running vision.py

from __future__ import division
import rospy
import sys
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from myvis.msg import Object
from myvis.msg import Objects
from myvis.msg import Detection
from myvis.msg import Detections
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time
import cv2


MID_POINT_THRESHOLD = 100 #update this with the actual threshold
ROLO_CONFIDENCE_THRESHOLD = 0.7


# TO DO
# need to ensure that the camera is pointing straight ahead relative to the robot - have a look at the transforms for this
#
#

class Follower:

	def __init__(self):
		# rospy.Subscriber("Objects", Objects, self.callback_objects)
		
		self.bridge = CvBridge()
		# TODO 

		self.pub_move = rospy.Publisher("/hsrb/command_velocity", Twist, queue_size=10)

		self.bgr_image = np.zeros((480,640,3), np.uint8)
		self.depth_image = np.zeros((480,640,1), np.uint8)

		self.human_target = None # {obj, img, depth} # TODO add face data
		self.prev_targets = []

		self.sub_yolo = rospy.Subscriber("/yolo2_node/detections", Detections, self.callback_main)
		self.sub_bgr_image = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color", Image, self.callback_image, queue_size=1, buff_size=480*640*8)
		self.sub_depth_image = rospy.Subscriber("/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw", Image, self.callback_depth, queue_size=1, buff_size=480*640*8)



	
	def isMyHuman(self, obj):
		current_image = self.bgr_image
		if obj.confidence < 0.7:
			return False
		else:
			pass

	def get_target(self, humans):
		if not self.human_target:
			 self.init_target(humans)
		return self.human_target

	def init_target(self, humans):
		# TODO assuming only one human in frame
		# TODO pick closest human based on depth
		# receives 
		time.sleep(1)
		# TODO make it pick closest most center human
		threshold = 2
		min_diff = 400 #the actual max value is 360 
		min_key = ""
		for key in humans:
			x = abs(360 - humans[key]['obj'].x)
			if x < min_diff and humans[key]['depth'] < threshold:
				min_diff = x
				min_key = key
		self.human_target = humans[min_key]


	def find_human_target(self, humans):
		'''
		param humans = dictionairy => index: {obj, img, depth}
		'''
		if humans:
			target = self.get_target(humans) # will set self.human_target
			print(len(humans),' humans found')
			#this will return only the humans that have a probability over some threshold
			humans = self.get_depth_probs(humans, target)
			humans = self.color_hist_probabilities(humans, target)
			print(len(humans),' humans to be compared')
			if len(humans) == 0:
				return False
			elif len(humans) == 1:
				#if only one human passes the thresholds, they must be our_human
				k, v = humans.items()[0]
				our_human = v
			else: 
				our_human = self.max_probs(humans)
				
			if(len(self.prev_targets) <= 20 ):
				self.prev_targets.append(our_human)
			return our_human
			

		else:
			return False


	def max_probs(self, humans):
		# pos_probs, img_probs
		max_prob = 0
		max_index = 0
		for i in humans:
			humans[i]['prob'] = humans[i]['pos_probs'] + humans[i]['img_probs']
			print('*** {} ***'.format(i))
			print('x position', humans[i]['obj'].x)
			print('y position', humans[i]['obj'].y)
			print('depth position', humans[i]['depth'])
			print('img_probs', humans[i]['img_probs'])
			print('pos_probs', humans[i]['pos_probs'])
			print('prob', humans[i]['prob'])
			print('**********')
			if humans[i]['prob'] > max_prob:
				max_prob = humans[i]['prob']
				max_index = i
		return humans[max_index]

	def get_depth_probs(self, humans, target):
		w_pos = 1
		wd = 2
		wx = 2
		probs = []
		for i in humans:
			x = abs(humans[i]['obj'].x - target['obj'].x)
			y = abs(humans[i]['obj'].y - target['obj'].y)
			depth = abs(humans[i]['depth'] - target['depth'])
			prob = (w_pos * ((wx * x) + y)) + (wd * depth)
			humans[i]["pos_probs"] = prob
			probs.append(prob)
		total = sum(probs)
		print(total)
		print(humans[i]['pos_probs'])
		for i in humans:
			if humans[i]["pos_probs"] == total:
				print('***** same total ******** ')
				humans[i]["pos_probs"] = 1
			else:
				humans[i]["pos_probs"] = 1 - (humans[i]["pos_probs"]/total)
		return humans


	def color_hist_probabilities(self, humans, target):
		method = cv2.cv.CV_COMP_CORREL
		#histograms = self.color_histogram(humans, target)
		#use those to get a probability distribution
		results = {}
		humans_copy = {}
		distribution = {}
		total = 0
		corr_threshold = 0.4
		for key in humans:
			hist = humans[key]["hist"]
			correlation = self.compare_hist_to_targets(hist,method)
			print ('for human:', key, ', correlation = ', correlation)
			if correlation < corr_threshold:
				print('*******DROPPING', correlation)

				continue
			humans_copy[key] = humans[key]
			results[key] = correlation
			total += correlation
			#humans[key]["hist"] = hist

		for key in results:
			humans_copy[key]["img_probs"] = results[key]/total

		return humans_copy

	def compare_hist_to_targets(self, hist, method):
		total = 0
		for i in range(len(self.prev_targets)):
			target = self.prev_targets[i]
			total += cv2.compareHist(target["hist"], hist, method)
		if len(self.prev_targets):
			return total/len(self.prev_targets)
		else:
			return 1

	def color_histogram(self, humans, target):
		#return a probability distribution [all the probabilities add up to 1]
		hists = {}
		#get colour histograms of all the images
		for i in humans:
			image = humans[i]["img"]
			hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
			hist = cv2.normalize(hist).flatten()
			hists[i] = hist
		return hists

	


	def get_associated_data(self, human_objects, image, depth):
		# input list of rolo data
		# returns dictionairy - index: {obj: rolo data, img: rgb_crop, depth: depth_crop}
		humans = {}
		for i in range(len(human_objects)):
			obj = human_objects[i]
			x = int(obj.x) # object center
			y = int(obj.y) # object ceter
			h = int(obj.height)
			w = int(obj.width)
			# crop_image = image[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
			crop_image = image[int(y-h/4):int(y+h/4), int(x-w/4):int(x+w/4)]
			crop_depth = depth[int(y-h/4):int(y+h/4), int(x-w/4):int(x+w/4)] # divide by four to try and remove noise
			depth_dist = self.calculate_depth(crop_depth)
			hist_dict = ({"0": {'img': crop_image}})
			color_histogram = self.color_histogram(hist_dict, None)
			hist = color_histogram["0"]
			humans[i] = {
				'obj': obj,
				'img': crop_image,
				'depth': depth_dist,
				'hist': hist
			}
		return humans



	def calculate_depth(self, depth_image):
		depth_image.setflags(write=1)
		deptharray = np.asarray(depth_image)
		nana = np.isnan(deptharray)
		deptharray[nana] = 0
		depth = np.nanmean(deptharray)
		return depth # TODO test depth 



	def extract_human_detections(self, objects, image, depth):
		human_objects = []
		for i in range(0, len(objects)):
			obj = objects[i]
			if obj.class_name == "person" and obj.confidence > ROLO_CONFIDENCE_THRESHOLD:
				human_objects.append(obj)
		humans = self.get_associated_data(human_objects, image, depth) # dictionairy index: {obj, img, depth}
		return humans



	def turn_human(self, human):
		# requires object.x as input where obj is our human target
		img = human['img']
		# cv2.namedWindow("crop")
		# cv2.imshow('crop', img)
		# time.sleep(3)
		print('move target', human['obj'].x)
		# return
		middle = 360
		max_rotation = 0.7 #this is for turning velocity
		twist = Twist()	
		if human['depth'] > 1.5:
			threshold = 30
			twist.linear.x = min(0.5, 0.15 * human['depth']) #  TODO test if this works ok
		else:
			threshold = 60
		full_angle = 360 - threshold #the full width of possible angles
		if human['obj'].x > middle + threshold:
			# rotate left to correct
			print("rotate right")
			angle_right = human['obj'].x - middle - threshold
			rotation_amount = max_rotation * angle_right/full_angle
			print(rotation_amount)
			twist.angular.z = -rotation_amount # http://wiki.ros.org/turtlesim/Tutorials/Rotating%20Left%20and%20Right
			#twist.linear.x = 0.2
		elif human['obj'].x < middle - threshold:
			print("rotate left")
			angle_left = middle - threshold - human['obj'].x
			rotation_amount = max_rotation * angle_left/full_angle
			print(rotation_amount)
			twist.angular.z = rotation_amount
		self.pub_move.publish(twist)

	def callback_image(self, data):
		try:
			img = self.bridge.imgmsg_to_cv2(data, "bgr8")
			self.bgr_image = img
		except CvBridgeError as e:
			print(e)

	def callback_depth(self, data):
		try:
			img = self.bridge.imgmsg_to_cv2(data)
			self.depth_image = img
		except CvBridgeError as e:
			print(e)	

	def callback_main(self, data): 
		print('****************************************************************')
		start = time.time()
		image = self.bgr_image # called to make sure up to date
		depth = self.depth_image # called to make sure up to date

		objects = data.detections
		humans = self.extract_human_detections(objects, image, depth)
		our_human = self.find_human_target(humans)
		if our_human:
			self.human_target = our_human # update out human
			self.turn_human(our_human) # TODO uncomment to move
			print(time.time()-start)
		else:
			# TODO code to get the human to tell you where it is
			# maybe have a counter where if this happens enough then ask
			print(time.time()-start)
			return


	
rospy.init_node('follower', anonymous=True)
follower = Follower()


rospy.spin()


# class_id: 0
# class_name: person
# confidence: 0.492961704731
# x: 380.4921875
# y: 318.443206787
# height: 152
# width: 53
# class_id: 0
# class_name: person
# confidence: 0.761228561401
# x: 85.8810501099
# y: 259.162200928
# height: 504
# width: 175
# class_id: 0
# class_name: person
# confidence: 0.876062333584
# x: 324.470458984
# y: 267.020233154
# height: 470
# width: 150



# threshold = 300
# probs = []
# for i in humans:
# 	x = abs(humans[i]['obj'].x - target['obj'].x)
# 	y = abs(humans[i]['obj'].y - target['obj'].y)
# 	depth = abs(humans[i]['depth'] - target['depth'])
# 	if x > threshold:
# 		print('$$$$$$$$$$$$$', x)
# 		humans[i]["pos_probs"] = 0
# 	else:
# 		prob = (w_pos * ((wx * x) + y)) + (wd * depth)
# 		humans[i]["pos_probs"] = prob
# 		probs.append(prob)
# total = sum(probs)
# print(total)
# print(humans[i]['pos_probs'])
# copy_humans = {}
# for i in humans:
# 	if humans[i]["pos_probs"] != 0:
# 		copy_humans[i] = humans[i]
# 		if humans[i]["pos_probs"] == total:
# 			print('***** same total ******** ')
# 			copy_humans[i]["pos_probs"] = 1
# 		else:
# 			copy_humans[i]["pos_probs"] = 1 - (humans[i]["pos_probs"]/total)
# 	else:
# 		print('$$$$$$$$$$$$$$$$$ dropping based on pos')