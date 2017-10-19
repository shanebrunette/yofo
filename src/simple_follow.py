#!/usr/bin/env python
# just some really basic code to get started following someone
# just run this as an individual node using rosrun comp3431_project simple_follow.py

from __future__ import division
import rospy
import sys
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from myvis.msg import Object
from myvis.msg import Objects
from myvis.msg import Detection
from myvis.msg import Detections
from tmc_msgs.msg import Voice
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time
import cv2
import tf
import math
from human import Human


MID_POINT_THRESHOLD = 100 #update this with the actual threshold
ROLO_CONFIDENCE_THRESHOLD = 0.7

# HEAD_TOPIC = '/hsrb/head_trajectory_controller/command'
HEAD_CAMERA_TF_TOPIC = "/head_rgbd_sensor_link"
BODY_TF_TOPIC = "/base_link"

# TO DO
# need to ensure that the camera is pointing straight ahead relative to the robot - have a look at the transforms for this
#
#

class Follower:

	def __init__(self):
		# rospy.Subscriber("Objects", Objects, self.callback_objects)
		
		self.bridge = CvBridge()
		
		self.rec_depth = False
		self.rec_image = False
		# TODO 

		self.pub_body_move = rospy.Publisher("/hsrb/command_velocity", Twist, queue_size=10)
		self.pub_head_move = rospy.Publisher("hsrb/head_trajectory_controller/command", JointTrajectory, queue_size=10)

		self.pub_voice = rospy.Publisher('talk_request', Voice, queue_size = 10)

		self.bgr_image = np.zeros((480,640,3), np.uint8)
		self.depth_image = np.zeros((480,640,1), np.uint8)

		self.human_target = None # {obj, img, depth} # TODO add face data
		self.prev_targets = []
		self.tf_listener = tf.TransformListener()

		self.sub_yolo = rospy.Subscriber("/yolo2_node/detections", Detections, self.callback_main)
		self.sub_bgr_image = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color", Image, self.callback_image, queue_size=1, buff_size=480*640*8)
		self.sub_depth_image = rospy.Subscriber("/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw", Image, self.callback_depth, queue_size=1, buff_size=480*640*8)
		self.tf_head = rospy.Subscriber(HEAD_CAMERA_TF_TOPIC, JointTrajectoryControllerState)
		self.tf_body = rospy.Subscriber(BODY_TF_TOPIC, JointTrajectoryControllerState)


	def init_target(self, humans, depth_threshold=6):
		"""
		init_target(self, humans, depth_threshold=2.5)
		param humans is list of Human objects
		param depth_threshold used for max distance from robot
		updates self.human_target with most center human within depth_threshold
		returns self.human_target with Human or None
		"""
		time.sleep(1)
		min_diff = 400 #the actual max value is 360 
		print([human.depth for human in humans])
		potential_humans = [human for human in humans if human.depth < depth_threshold]
		if not potential_humans:
			self.human_target = None
		else:
			distance_center = [abs(360 - human.x) for human in potential_humans]
			min_index = distance_center.index(min(distance_center))
			self.human_target = potential_humans[min_index]
		return self.human_target


	def find_human_target(self, humans):
		"""
		param humans is list of Human objects.
		compares each Human to target Human
		by calculating position and color histogram similarity probabilities
		returns single Human object based on max probability likelihood
		or None if no Human object is found
		"""

		def _update_humans_with_probs(humans):
			if humans:
				target = self.human_target # will set self.human_target
				humans = self.get_depth_probs(humans, target)
				humans = self.get_img_probs(humans, target)
				print('depth, img probs', [(human.img_probs, human.pos_probs) for human in humans])
				humans = self.discard_unlikely_probs(humans)
			return humans

		def _get_our_human(humans):
			if not humans:
				return None
			elif len(humans) == 1:
				our_human = humans[0]
			else: 
				our_human = self.max_probs(humans)
			return our_human

		print('before clean', len(humans))
		humans = _update_humans_with_probs(humans)
		print('after clean', len(humans))
		our_human = _get_our_human(humans)
		return our_human

	def discard_unlikely_probs(self, humans, threshold=0):
		"""
		param list of Human objects
		discards any human where any prob <= threshold
		return list of Human objects
		"""
		# TODO test different threshold
		return [human for human in humans if (human.img_probs * human.pos_probs) > threshold]

	def max_probs(self, humans):
		"""
		param humans is list of Human objects
		weights image and position probabilities when potential match
		returns maximum likelihood Human
		"""
		def _get_prob(human):
			return human.pos_probs + human.img_probs

		probs = [_get_prob(human) for human in humans]
		max_index = probs.index(max(probs))
		return humans[max_index]


	def get_depth_probs(self, humans, target):
		"""
		param humans is list of Human objects
		param target is Human object to be matched against
		Human.pos_probs is normalized against probability distribution
		if any position probability is below threshold critera determined
		by Human.get_position_prob it is set to 0
		returns list of Human objects with updated position probability
		"""
		humans = [human.get_position_prob(target) for human in humans]
		total = sum([human.pos_probs for human in humans])
		humans = [human.normalize_position_prob(total) for human in humans]
		return humans


	def get_img_probs(self, humans, target):
		humans = [human.get_image_prob(target, self.prev_targets) for human in humans]
		total = sum([human.img_probs for human in humans])
		humans = [human.normalize_img_prob(total) for human in humans]
		return humans


	def extract_humans(self, objects, rgb_image, depth_image):
		"""
		param human_objects is yolo object with class name 'people'
		param rgb_image of entire camera view
		param depth_image of entire camera depth view
		returns list of Human objects with each Human object containing:
			x, y, h, w, depth, img, hist
			see human.py Human class for more details on Human object
		"""
		human_objects = [
						obj for obj in objects 
						if obj.class_name == "person" 
						and obj.confidence > ROLO_CONFIDENCE_THRESHOLD]
		return [Human(obj, rgb_image, depth_image) for obj in human_objects]


	def turn_human9(self, human):
		"""
		param human is Human object that is our_human target
		publishes Twist message to turn towards human
		"""
		img = human.img
		middle = 360
		max_rotation = 0.7 #this is for turning velocity
		twist = Twist()

		if human.depth > 1.5:
			threshold = 30
			twist.linear.x = min(0.8, 0.15 * human.depth) #  TODO test if this works ok
		else:
			threshold = 60
		
		full_angle = 360 - threshold #the full width of possible angles
		if human.x > middle + threshold:
			# rotate left to correct
			angle_right = human.x - middle - threshold
			rotation_amount = max_rotation * angle_right/full_angle
			twist.angular.z = -rotation_amount # http://wiki.ros.org/turtlesim/Tutorials/Rotating%20Left%20and%20Right
			#twist.linear.x = 0.2
		elif human.x < middle - threshold:
			angle_left = middle - threshold - human.x
			rotation_amount = max_rotation * angle_left/full_angle
			twist.angular.z = rotation_amount

		self.pub_body_move.publish(twist)

	def turn_human(self, human):
		"""
		param human is Human object that is our_human target
		publishes Twist message to turn towards human
		"""

		try:
			(trans,rot) = self.tf_listener.lookupTransform(HEAD_CAMERA_TF_TOPIC, BODY_TF_TOPIC, rospy.Time(0))
  		except Exception as e:
			print("error when getting transform. No movement made")
			print(e)
			return

		middle = 360
		body_threshold = 0.5 # this needs to be tested - how quickly it moves?
		max_rotation = 0.7 #this is for body turning velocity

		img = human.img
		body_twist = Twist()
		head_twist = JointTrajectory()
		points = JointTrajectoryPoint()
		head_twist.joint_names = ["head_pan_joint", "head_joint_tilt"]

		# this needs to be tested - it should in theory calculate the angle between the body and head and tell the body to rotate that far
		#body_angular = 4 * math.atan2(trans[1], trans[0])
		tf_x = trans[0]
		tf_y = trans[1]
		
		acute_angle = math.degrees(math.atan2(math.fabs(trans[1]),math.fabs(trans[0])))
		if tf_x < 0:
			if tf_y < 0:
				body_angle = 270 - acute_angle
			else:
				body_angle = 270 + acute_angle
		else : 
			if tf_y < 0: 
				body_angle = 90 + acute_angle
			else: 	
				body_angle = 90 - acute_angle
		
		if body_angle < 30 or body_angle > 330:
			body_twist.angular.z = 0
		else: 
			body_twist.angular.z = body_angle/360

	
		points.time_from_start = rospy.Time(2)
		
		if human.depth > 1.8 and math.fabs(body_twist.angular.z) < 0.2 :
			threshold = 30
			body_twist.linear.x = min(0.5, 0.15 * human.depth) # TODO
		else:
			threshold = 60
		
		#TODO alter this to be in the Joint Trajectory message type
		full_angle = 360
		#full_angle = 360 - threshold #the full width of possible angles
		if human.x > middle + threshold:
			print('@@@@@@@@@@@left')
			# rotate left to correct

			angle_right = human.x - middle - threshold
			rotation_amount = max_rotation * angle_right/full_angle
			print(angle_right)
			print(rotation_amount)
			#this needs to be put into Joint tra
			points.positions = [-rotation_amount, 0] #https://docs.hsr.io/manual_en/development/ros_controller_head.html
			#twist.linear.x = 0.2
		elif human.x < middle - threshold:
			print('@@@@@@@right')
			angle_left = middle - threshold - human.x
			rotation_amount = max_rotation * angle_left/full_angle
			points.positions = [rotation_amount, 0]
		
		points.velocities = [0,0]
		head_twist.points = [points]

		self.pub_head_move.publish(head_twist)
		# self.pub_body_move.publish(body_twist)


	def callback_image(self, data):
		self.rec_image = True
		try:
			img = self.bridge.imgmsg_to_cv2(data, "bgr8")
			self.bgr_image = img
		except CvBridgeError as e:
			print(e)

	def found_human(self):
		self.pub_voice.publish(False, False, 1, 'i see you')
		self.pub_voice.publish(False, False, 1, 'you must die')

	def following_human(self):
		self.pub_voice.publish(False, False, 1, 'kill!')

	def get_human(self):
		if self.human_target:
			self.pub_voice.publish(False, False, 1, 'hey! come back here.')
		else:
			self.pub_voice.publish(False, False, 1, 'come stand infront of me')

	

	def callback_depth(self, data):
		self.rec_depth = True
		try:
			img = self.bridge.imgmsg_to_cv2(data)
			self.depth_image = img
		except CvBridgeError as e:
			print(e)	

	
	def callback_main(self, data): 
		print('**')
		if not self.rec_depth or not self.rec_image:
			print('not')
			return
		history_threshold = 10
		image = self.bgr_image 
		depth = self.depth_image
		objects = data.detections
		humans = self.extract_humans(objects, image, depth)
		if not humans:
			print('zero humans detected')
			self.get_human()
			return

		if not self.human_target:
			print('init target')
			self.found_human()
			our_human = self.init_target(humans)
		else:
			print('finding human')
			our_human = self.find_human_target(humans)

		if our_human:
			if len(self.prev_targets) <= history_threshold:
				print('storing history of human')
				self.prev_targets.append(our_human)
			self.human_target = our_human # update out human
			print('our human z, x', our_human.depth, our_human.x)
			self.following_human()
			self.turn_human(our_human) # TODO uncomment to move
		else:
			print('no human')
			self.get_human()
			# TODO code to get the human to tell you where it is
			# TODO have a counter where if this happens enough then ask
			# for now just have the head swivel and try to find them
			head_twist = JointTrajectory()
			head_twist.joint_names = ["head_pan_joint", "head_joint_tilt"]
			points = JointTrajectoryPoint()
			#how quickly it should move
			points.time_from_start = rospy.Time(1)
			points.positions = [0,-0.2]
			points.velocities = [0,0]
			head_twist.points = [points]
			self.pub_head_move.publish(head_twist)
			return


if __name__ == '__main__':
	rospy.init_node('follower', anonymous=True)
	follower = Follower()
	print('starting')
	while not rospy.is_shutdown():
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
