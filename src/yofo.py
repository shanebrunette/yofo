#!/usr/bin/env python

"""
Human Following Code for use with the Toyota Human Support Robot
University of New South Wales
@Shane Brunette 	shanebrunette@gmail.com 	 github:shanebrunette
@Alison McCann  	alison.r.mccann@gmail.com 	 github:A-McCann
Supervisor @Dr Claude Sammut
"""

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
from myvis.msg import Detections
from tmc_msgs.msg import Voice
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time
import cv2
import tf
import math
from human import Human






class Follower:
	def __init__(self):
		"""
		Before initialization start stand_tall to get robot in position.
		Following object publishes twist message to base and head controllers
		On initialization follows most centred human in field of vision 
		and under within depth threshold.
		Advanced setting rotates head when human is almost or out of field of vision
		Face detection setting require human to be within 2-3 metres of robot vision
		And standing at eye level on initialization of robot.
		On initialization the robot will wait for ~5 frames before looking for human
		Make sure your kill switch is handy!
		"""
		self.face_detection_setting = int(sys.argv[1])
		self.advanced_head_turning_setting = int(sys.argv[2])

		self.bridge = CvBridge()
		self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
		self.recognizer = cv2.createLBPHFaceRecognizer(threshold=5.0)
		self.head_turn = 0.3
		self.rolo_confidence_threshold = 0.7

		self.human_target = None
		self.prev_targets = []
		self.rec_depth = False
		self.rec_image = False
		self.bgr_image = np.zeros((480,640,3), np.uint8)
		self.depth_image = np.zeros((480,640,1), np.uint8)
		self.head_position = 0
		self.turn_right = False
		self.counter = -1
		self.extreme = False
		self.turn_direction = None
		self.head_position = 0
		self.history_threshold = 10

		self.pub_body_move = rospy.Publisher("/hsrb/command_velocity", Twist, queue_size=10)
		self.pub_head_move = rospy.Publisher("hsrb/head_trajectory_controller/command", JointTrajectory, queue_size=10)
		self.pub_voice = rospy.Publisher('talk_request', Voice, queue_size = 10)

		self.sub_yolo = rospy.Subscriber("/yolo2_node/detections", Detections, self.callback_main)
		self.sub_bgr_image = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color", Image, self.callback_image, queue_size=1, buff_size=480*640*8)
		self.sub_depth_image = rospy.Subscriber("/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw", Image, self.callback_depth, queue_size=1, buff_size=480*640*8)
		

	def init_target(self, humans, depth_threshold=3):
		"""
		Used to initialize HSR with human target
		Checks for potential humans by checking if the human is under depth threshold
		Calculates distance from center for each human
		Chooses the human that is most centred
		Also initiates face training if face_detection_setting is set to true

		init_target(self, humans, depth_threshold=2.5)
		param humans is list of Human objects
		param depth_threshold used for max distance from robot
		updates self.human_target with most center human within depth_threshold
		returns self.human_target with Human or False
		"""
		time.sleep(1)
		middle = 320 # middle of frame of view
		potential_humans = [human for human in humans if human.depth < depth_threshold]
		if not potential_humans:
			self.human_target = None
			return False
		else:
			distance_center = [abs(middle - human.x) for human in potential_humans]
			min_index = distance_center.index(min(distance_center))
			human_target = potential_humans[min_index]
			if self.face_detection_setting:
				target_face = self.train_face(human_target)
				if not target_face:
					return False 
			self.human_target = human_target
			self.found_human()
		return self.human_target


	def train_face(self, human):
		"""
		Looks face in human object.
		Trains fisherfeace model declared under self.recognizer on detected human face.
		Sets the face target to self.recognizer if face is detected
		param human is human object
		returns True if face was trained correctly
		otherwise returns False
		"""
		face = human.face
		if not face.any():
			return False
		img_numpy = np.array(face,'uint8')
		faces = self.detector.detectMultiScale(img_numpy)
		if len(faces):
			(x,y,w,h) = faces[0]
			cropped_face = img_numpy[y:y+h,x:x+w]
			self.recognizer.train([cropped_face], np.array([1]))
			self.face_target = self.recognizer
			if not self.face_target:
				return False
			return True
		else:
			return False


	def find_human_target(self, humans):
		"""
		param humans is list of Human objects.
		compares each Human to target Human
		by calculating position and color histogram similarity probabilities
		returns single Human object based on max probability likelihood
		or None if no Human object is found
		"""

		def _update_humans_with_probs(humans):
			"""
			updates list of human objects with correct probabilities
			discards any highly unlikely probabilities
			param humans is list of human objects
			updates and returns humans list
			"""
			if humans:
				target = self.human_target
				humans = self.get_depth_probs(humans, target)
				humans = self.get_img_probs(humans, target)
				if self.face_detection_setting:
					humans = self.get_face_matches(humans, self.face_target)
				humans = self.discard_unlikely_probs(humans)
			return humans

		def _get_our_human(humans):
			"""
			chooses our human from human list by picking human
			with max prob
			param humans is list of humans
			returns single human object our_human
			"""
			if not humans:
				return None
			elif len(humans) == 1:
				our_human = humans[0]
			else: 
				our_human = self.max_probs(humans)
			return our_human

		humans = _update_humans_with_probs(humans)
		our_human = _get_our_human(humans)
		return our_human

	def get_face_matches(self, humans, face_target):
		"""
		param humans list of Human objects
		param face_target is trained fisherfaces model
		checks each human against face model to see if face is recognized
		returns humans list with update self.face_match
		"""
		return [human.check_face(face_target) for human in humans]

	def discard_unlikely_probs(self, humans, threshold=0):
		"""
		param humans list of Human objects
		discards any human where any prob <= threshold
		threshold defaults to 0
		return list of Human objects
		"""
		return [human for human in humans if (human.img_probs * human.pos_probs) > threshold]

	def max_probs(self, humans):
		"""
		maximizes probabilities by summing probs of each human
		param humans is list of Human objects
		weights image and position probabilities when potential match
		returns maximum likelihood Human
		"""
		def _get_prob(human):
			return human.pos_probs + human.img_probs + human.face_target

		probs = [_get_prob(human) for human in humans]
		max_index = probs.index(max(probs))
		return humans[max_index]


	def get_depth_probs(self, humans, target):
		"""
		param humans is list of Human objects
		param target is Human object to be matched against
		Human.pos_probs is normalized against probability distribution
		returns list of Human objects with updated position probability
		"""
		humans = [human.get_position_prob(target) for human in humans]
		total = sum([human.pos_probs for human in humans])
		humans = [human.normalize_position_prob(total) for human in humans]
		return humans


	def get_img_probs(self, humans, target):
		"""
		gets color histogram probabilities based on shirt of human
		param humans is list og Human objects
		param target is Human object to be matched against
		"""
		humans = [human.get_image_prob(target, self.prev_targets) for human in humans]
		total = sum([human.img_probs for human in humans])
		humans = [human.normalize_img_prob(total) for human in humans]
		return humans


	def extract_humans(self, objects, rgb_image, depth_image):
		"""
		extracts human objects from yolo published data stream
		if confidence above certain threshold
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
						and obj.confidence > self.rolo_confidence_threshold]
		return [Human(obj, rgb_image, depth_image) for obj in human_objects]


	def turn_human(self, human):
		"""
		simple turn human function which turns to face target human
		moves base only
		param human is Human object that is our_human target
		publishes Twist message to turn towards human
		linear.x is forward motion
		angular.z is rotation motion
		for more information on rotation see
		http://wiki.ros.org/turtlesim/Tutorials/Rotating%20Left%20and%20Right 
		"""
		middle = 320
		max_rotation = 0.7 #this is for turning velocity
		max_forward = 0.8
		human_depth_weight = 0.15
		twist = Twist()

		if human.depth > 1.5:
			threshold = 30
			twist.linear.x = min(max_forward, human_depth_weight * human.depth) 
		else:
			threshold = 60
		
		if human.x > middle + threshold:
			# rotate right from robot perspective
			displacement_right = human.x - middle - threshold
			rotation_amount = max_rotation * displacement_right/middle
			twist.angular.z = -rotation_amount 
		elif human.x < middle - threshold:
			# rotate left
			displacement_left = middle - threshold - human.x
			rotation_amount = max_rotation * displacement_left/middle
			twist.angular.z = rotation_amount

		self.pub_body_move.publish(twist)

	def adv_turn_human(self, human):
		"""
		param human is Human object that is our_human target
		publishes Twist message to turn base towards human
		also publishes Twist message to turn head if human is at extreme threshold
		updates turn_direction as pos, neg so that scan knows which way to turn if
		target human leaves field of vision
		see basic turn_human function for more information
		"""
		middle = 320
		maximum_threshold = 640 # number of pixels in image
		body_threshold = 0.5
		max_rotation = 0.5 # 0.8 this is for body turning velocity 
		body_rotation = 0
		target_pos = 0
		extreme_threshold = 100

		body_twist = Twist()
		traj = JointTrajectory()
		traj.joint_names = ["head_pan_joint", "head_tilt_joint"]
		p = JointTrajectoryPoint()
		p.positions = [0, 0]
		p.velocities = [0, 0]
		body_twist.angular.z = 0	
		threshold = 30	
		
		if abs(self.head_position) <= (self.head_turn * 1.5):
			if human.depth > 2:
				body_twist.linear.x = min(0.8, 0.5 * human.depth) # TODO
			else:
				threshold = 60

		if human.x < middle:
			self.turn_direction = 'pos'
		else:
			self.turn_direction = 'neg'

		if human.x > middle + threshold or self.head_position >= self.head_turn:
			# rotate right from robot perspective
			angle_right = human.x - middle - threshold
			body_rotation = -(max_rotation * angle_right/middle)

		elif human.x < middle - threshold or self.head_position <= -self.head_turn:
			# rotate left
			angle_left = middle - threshold - human.x
			body_rotation = max_rotation * angle_left/middle

		if human.x < extreme_threshold:
			body_rotation * 1.5
			self.extreme = 'pos'
			if self.head_position < abs(self.head_turn):
				self.head_position = self.head_turn
		elif human.x >  maximum_threshold - extreme_threshold:
			body_rotation * 1.5
			self.extreme = 'neg'
			if self.head_position < abs(self.head_turn):
				self.head_position = -self.head_turn
		else:
			self.head_position = self.head_position - (self.head_position / 4)
			self.extreme = False

		if abs(self.head_position) >= 0.9:
			body_rotation * 1.5
		elif abs(self.head_position) >= 1.2:
			body_rotation * 2

		body_twist.angular.z = body_rotation
		p.positions = [self.head_position, 0]
		p.time_from_start = rospy.Time(2)
		traj.points = [p]

		self.pub_head_move.publish(traj)
		self.pub_body_move.publish(body_twist)



	def scan_room(self, max_head_rotation=1.2):
		"""
		Command used for scanning the room by twisting head.
		If more then five frames seen without human
		scans the room up to max_head_rotation 
		head turning only available on advanced setting.
		Every frame over 15 where human is not detected will initiate
		self.get_human voice command request human to return.
		"""
		self.counter += 1
		if self.counter > 15:
			self.get_human()
			self.counter = 0

		if self.advanced_head_turning_setting:
			if abs(self.head_position) < max_head_rotation and (
					self.extreme or (self.human_target and self.counter > 5)):
				if self.turn_direction == 'pos':
					self.head_position += self.head_turn
				else:
					self.head_position -= self.head_turn
			else:
				if self.head_position > max_head_rotation:
					self.turn_direction = 'neg'
				elif self.head_position < -max_head_rotation:
					self.turn_direction = 'pos'
			self.twist_head(self.head_position)
		return
		

	def twist_head(self, pos):
		"""
		twist head helper function to move head on x axis based on pos
		publishes to head topic
		"""
		traj = JointTrajectory()
		traj.joint_names = ["head_pan_joint", "head_tilt_joint"]
		p = JointTrajectoryPoint()
		p.positions = [pos,0]
		p.velocities = [0,0]
		p.time_from_start = rospy.Time(3)
		traj.points = [p]
		self.pub_head_move.publish(traj)
		return


	def found_human(self):
		"""
		use when you have found the human
		publishes to voice topic
		"""
		self.pub_voice.publish(False, False, 1, 'i see you, now following')

	def following_human(self):
		"""
		used when you are following the human
		publishes to voice topic
		"""
		self.pub_voice.publish(False, False, 1, 'bleep!')

	def get_human(self):
		"""
		used on start if no human target or when human target is lost
		publishes to voice topic
		"""
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


	def callback_image(self, data):
		self.rec_image = True
		try:
			img = self.bridge.imgmsg_to_cv2(data, "bgr8")
			self.bgr_image = img
		except CvBridgeError as e:
			print(e)

	
	def callback_main(self, data): 
		"""
		main function
		if not depth or image started it will return
		on initiation will call get_human()
		if no humans found will call snan_room()
		after fifth frame will start calling init_target
		this gives robot time to 'warm up'
		once human target found it will call find_human_target
		if our target human is found it will turn to face human
		additionally it will store target human in history
		if history threshold has not been reached
		otherwise if no target human is found it will call scan_room
		"""
		if not self.rec_depth or not self.rec_image:
			return

		if self.counter == -1:
			self.counter += 1
			self.get_human()

		
		image = self.bgr_image 
		depth = self.depth_image
		objects = data.detections
		humans = self.extract_humans(objects, image, depth)
		if not humans:
			self.scan_room()
			return

		if not self.human_target:
			if self.counter < 5:
				self.counter += 1
				return
			else:
				our_human = self.init_target(humans)
				self.counter = 0
		else:
			our_human = self.find_human_target(humans)

		if our_human:
			self.counter = 0
			if len(self.prev_targets) <= self.history_threshold:
				self.prev_targets.append(our_human)
			self.human_target = our_human
			if self.advanced_head_turning_setting:
				self.adv_turn_human(our_human)
			else:
				self.turn_human(our_human)
		else:
			self.scan_room()
			return


if __name__ == '__main__':
	rospy.init_node('follower', anonymous=True)
	follower = Follower()
	while not rospy.is_shutdown():
		rospy.spin()

