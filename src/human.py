#!/usr/bin/env python

"""
Human class object called by yofo.
Stores information about individual humans to be used when 
deciding if human object is human targe.t
Information includes
x, y, h, w based on detections from yolo
depth
shirt rgb crop
face rgb crop
color histogram
pos_probs probability that it is human target based on position
img_probs probability that it is human target based on shirt color
face_target boolean faces matches human target face
"""

from __future__ import division
import uuid
import cv2
import numpy as np


class Human:
	def __init__(self, yolo_data, rgb_image, depth_image):
		"""
		human class object stores information about human
		and related probabilities of being the target
		"""
		self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
		self.id = uuid.uuid4()
		self.x = int(yolo_data.x) # object center
		self.y = int(yolo_data.y) # object ceter
		self.h = int(yolo_data.height)
		self.w = int(yolo_data.width)
		self.depth = self.get_depth(depth_image)
		self.shirt = self.get_shirt(rgb_image)
		self.face = self.get_head(rgb_image)
		self.hist = self.get_color_histogram(self.shirt)
		self.pos_probs = None
		self.img_probs = None
		self.face_target = 0



	def get_depth(self, depth_image):
		"""
		gets depth based on depth image and self.x,y,h,w
		returns depth
		"""
		x, y, h, w = self.x, self.y, self.h, self.w
		crop_depth = depth_image[int(y-h/4):int(y+h/4), int(x-w/4):int(x+w/4)] # divide by four to try and remove noise
		crop_depth.setflags(write=1)
		np_cropped_depth = np.asarray(crop_depth)
		depth = np.nanmean(np_cropped_depth)
		return depth


	def get_head(self, rgb_image):
		"""
		x, y are center of image
		h, w refer to bounding box
		param rgb_image is rgb image
		gets estimated head location of imaage
		returns cropped image
		"""
		x, y, h, w = self.x, self.y, self.h, self.w
		gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
		crop_image = gray_image[int(y-h/2):int(y), int(x-w/3):int(x+w/3)]
		# print(crop_image.shape)
		if not crop_image.shape[0]:
			return []
		# cv2.imshow('img', crop_image)
		# if cv2.waitKey(1) & 0xFF == ord('q'):
		#         cv2.destroyAllWindows()
		return crop_image

	def check_face(self, target):
		"""
		checks self human object face against target fisherfaces model face
		param target is fisher faces model with identity 1 being target face
		updates self.face_match to 1 if match, else 0
		optional: could be udated to negative value if found face but wasnt a match
		returns self
		"""
		faces = self.detector.detectMultiScale(self.face, 1.3,5)
		if not len(faces):
			self.face_match = 0
			return self
		(x,y,w,h) = faces[0]
		match = target.predict(self.face[y:y+h,x:x+w])
		if match == 1:
			self.face_match = 1
		else:
			self.face_match = 0 # negative if found face but wasnt match
		return self


	def get_shirt(self, rgb_image):
		"""
		x, y are center of image
		h, w refer to bounding box
		"""
		x, y, h, w = self.x, self.y, self.h, self.w
		crop_image = rgb_image[int(y-h/6):int(y), int(x-w/4):int(x+w/4)]
		# cv2.imshow('img', crop_image)
		# if cv2.waitKey(1) & 0xFF == ord('q'):
		#         cv2.destroyAllWindows()
		return crop_image


	def get_color_histogram(self, image):
		"""
		helper function to get color histogram from given image using opencv
		param image is RGB image
		returns histogram
		"""
		hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		hist = cv2.normalize(hist).flatten()
		return hist


	def get_position_prob(self, target, w_position=1, w_depth=2, w_x=2):
		"""
		get_position_prob(self, target, w_position=1, w_depth=2, w_x=2)
		param target is a Human object
		param w_position is weight for x and y distance		
		param w_depth is weight for depth distance
		param w_x is weight for x distance compared to y
		weights were optimized based on empirical testing with HSR robot
		calculates probability of self being target from x, y, and depth distance
		updates self.pos_probs
		returns self
		"""
		x = abs(self.x - target.x)
		y = abs(self.y - target.y)
		depth = abs(self.depth - target.depth)
		self.pos_probs = (w_position * ((w_x * x) + y)) + (w_depth * depth)
		return self

	

	def normalize_position_prob(self, total):
		"""
		param total is float
		normalizes position probability based on total probability distribution
		where max is more likely to match target
		updates self.pos_probs
		returns self
		"""
		def _normalize(prob, total):
			if not prob:
				prob = 0
			elif prob == total:
				prob = 1
			else:
				prob = 1 - (prob / total)
			return prob

		self.pos_probs = _normalize(self.pos_probs, total)
		return self

	def normalize_img_prob(self, total):
		"""
		param total is float
		normalizes img probability based on total probability distribution
		where max is more likely to match target
		updates self.color_probs
		returns self
		"""
		def _normalize(prob, total):
			if not prob:
				prob = 0
			elif prob == total:
				prob = 1
			else:
				prob /= total
			return prob

		self.img_probs = _normalize(self.img_probs, total)
		return self


	def get_image_prob(
		self, 
		target, 
		target_history, 
		method=cv2.cv.CV_COMP_CORREL, 
		correlation_threshold=0.1):
		"""
		param target is human object
		param target_history is list of human objects captured on init of target
		param method is cv2 correlation algorithm or similar
		correlation threshold is minimum amount of correlation before penalizing
		updates img_probs of self human object
		returns self
		"""
		def compare_hist_to_targets(new_hist, target_history, method):

			total = 0
			for previous_human in target_history:
				old_hist = previous_human.hist
				total += cv2.compareHist(old_hist, new_hist, method)
			if len(target_history):
				total /= len(target_history)
			else:
				total = 1
			return total

		self.img_probs = compare_hist_to_targets(self.hist, target_history, method)
		if self.img_probs < correlation_threshold:
			self.img_probs * 0.5
		return self


