#!/usr/bin/env python

import sys
import time
import controller_manager_msgs.srv
import rospy
import trajectory_msgs.msg


DEG_90 = 1.5708
GRIPPER_OPEN_ANGLE = 1.239
GRIPPER_CLOSE_ANGLE = -0.105
ARM_LOWERING_SPEED = float(sys.argv[1])
HAND_CAMERA_TOPIC = '/hsrb/hand_camera/image_raw'
HEAD_RGB_TOPIC = '/hsrb/head_rgbd_sensor/rgb/image_rect_color'
HEAD_DEPT_TOPIC = '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'


class Arm_Controller:
	def __init__(self):
		rospy.init_node('arm_test')

		# initialize ROS publisher
		self.arm_pub = rospy.Publisher('/hsrb/arm_trajectory_controller/command',
							  trajectory_msgs.msg.JointTrajectory, queue_size=10)
		self.gripper_pub = rospy.Publisher('/hsrb/gripper_controller/command',
    							trajectory_msgs.msg.JointTrajectory, queue_size=10)
		# wait to establish connection between the controller

		print(0)

		while self.arm_pub.get_num_connections() == 0:
			rospy.sleep(0.1)

		print(1)

		# make sure the controller is running
		rospy.wait_for_service('/hsrb/controller_manager/list_controllers')
		list_controllers = (
			rospy.ServiceProxy('/hsrb/controller_manager/list_controllers',
							   controller_manager_msgs.srv.ListControllers))

		print(2)
		
		# Check that both the arm trajectory and gripper controllers are running
		arm_running = False
		gripper_running = False
		while arm_running is False and gripper_running is False:
			rospy.sleep(0.1)
			for c in list_controllers().controller:
				if c.name == 'arm_trajectory_controller' and c.state == 'running':
					arm_running = True
				elif c.name == 'gripper_controller' and c.state == 'running':
					gripper_running = True

		print(3)

		# Arm joints
		self.al_joint = 0
		self.af_joint = 0
		self.ar_joint = DEG_90
		self.wf_joint = DEG_90
		self.wr_joint = 0
		self.reset_default_pos()
		# Gripper joints
		self.hm_joint = 0

	# Reset the arm to the default position
	def reset_default_pos(self):
		self.move_arm(0, 0, DEG_90, DEG_90, 0)

	# Turns the arm roll and wrist flex joint to face the hand north of the robot
	def face_hand_forward(self):
		self.move_arm(self.al_joint, self.af_joint, 0, -DEG_90, self.wr_joint)

	# Lower the arm and adjust the wrist flex joint so that the hand is always 90 degrees
	def lower_arm_sync_wrist(self):
		while self.af_joint > -DEG_90:
			temp_af_joint = max(self.af_joint-ARM_LOWERING_SPEED, -DEG_90)
			temp_wf_joint = -(DEG_90-abs(temp_af_joint))
			self.move_arm(self.al_joint, temp_af_joint, self.ar_joint, temp_wf_joint, self.wr_joint)
			time.sleep(3)

	# Move the arm by pusblishing a joint trajectory message with the relevant arm joints
	def move_arm(self, al_joint, af_joint, ar_joint, wf_joint, wr_joint):
		# fill ROS message
		traj = trajectory_msgs.msg.JointTrajectory()
		traj.joint_names = ["arm_lift_joint", "arm_flex_joint",
							"arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
		p = trajectory_msgs.msg.JointTrajectoryPoint()
		p.positions = [al_joint, af_joint, ar_joint, wf_joint, wr_joint]
		p.velocities = [0, 0, 0, 0, 0]
		p.time_from_start = rospy.Time(3)
		traj.points = [p]
		# Update joint angles
		self.al_joint = al_joint
		self.af_joint = af_joint
		self.ar_joint = ar_joint
		self.wf_joint = wf_joint
		self.wr_joint = wr_joint
		# publish ROS message
		self.arm_pub.publish(traj)

	# Move the gripper by publishing a joint trajectory message with the hand motor joint
	def move_gripper(self, hm_joint):
		traj = trajectory_msgs.msg.JointTrajectory()
		traj.joint_names = ["hand_motor_joint"]
		p = trajectory_msgs.msg.JointTrajectoryPoint()
		p.positions = [hm_joint]
		p.velocities = [0]
		p.effort = [0.1]
		p.time_from_start = rospy.Time(3)
		# Update joint
		self.hm_joint = hm_joint
		traj.points = [p]
		# publish ROS message
		self.gripper_pub.publish(traj)


	def open_gripper(self):
		self.move_gripper(GRIPPER_OPEN_ANGLE)

	def close_gripper(self):
		self.move_gripper(GRIPPER_CLOSE_ANGLE)

def main(args):
	print("Start")
	ac = Arm_Controller()
	print("Arm controller initialised")
	ac.face_hand_forward()
	print("Arm facing forward published")
	#time.sleep(3)
	#ac.lower_arm_sync_wrist()
	#ac.open_gripper()
	#time.sleep(3)
	ac.close_gripper()
	sys.exit(0)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
	main(sys.argv)
